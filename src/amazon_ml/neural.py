"""CUDA multilingual cross-encoder with structured matching features.

Training and model selection never use audit/confirmation labels. The neural gate
is fixed before training; all reference truth counts remain in the F0.5 metric.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .data import fingerprint, log, write_json
from .metrics import tune_threshold
from .train import metric_report, subset_arrays

MODEL_ID = "intfloat/multilingual-e5-large"
REVISION = "main"

def download(destination):
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL_ID, revision=REVISION, local_dir=destination,
                             allow_patterns=["config.json", "model.safetensors", "tokenizer*", "sentencepiece.bpe.model", "special_tokens_map.json", "README.md", "pytorch_model.bin"])
    write_json(Path(destination)/"origin.json", {"model_id": MODEL_ID, "revision": REVISION, "license": "Apache-2.0"})
    log(f"Downloaded licensed model to {path}")


def tokenize(data, model_path, length):
    data = Path(data)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    prepared = json.loads((data/"prepared.json").read_text(encoding="utf-8"))
    n = prepared["pairs"]
    config = {"length": length, "model": fingerprint([Path(model_path)/"config.json", Path(model_path)/"tokenizer.json"]),
              "pairs": fingerprint([data/"pairs.jsonl"])}
    cache = data/f"tokens_{length}"
    cache.mkdir(exist_ok=True)
    marker = cache/"complete.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != config:
            raise ValueError("Token cache changed")
        return cache
    ids = np.lib.format.open_memmap(cache/"ids.npy", mode="w+", dtype=np.int32, shape=(n, length))
    masks = np.lib.format.open_memmap(cache/"mask.npy", mode="w+", dtype=np.uint8, shape=(n, length))
    types = np.lib.format.open_memmap(cache/"types.npy", mode="w+", dtype=np.uint8, shape=(n, length))
    left, right, offset = [], [], 0
    def flush():
        nonlocal offset
        if not left:
            return
        encoded = tokenizer(left, right, padding="max_length", truncation="longest_first", max_length=length, return_tensors="np")
        end = offset+len(left)
        ids[offset:end] = encoded["input_ids"]
        masks[offset:end] = encoded["attention_mask"]
        types[offset:end] = encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"]))
        offset = end
        left.clear(); right.clear()
    with (data/"pairs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            left.append(row["left"]); right.append(row["right"])
            if len(left) == 1024:
                flush()
        flush()
    ids.flush(); masks.flush(); types.flush()
    if offset != n:
        raise ValueError("Token count mismatch")
    write_json(marker, config)
    return cache


class Matcher(nn.Module):
    def __init__(self, model_path, mean, std):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True,
                                                 trust_remote_code=False, use_safetensors=True,
                                                 attn_implementation="sdpa", low_cpu_mem_usage=True,
                                                 device_map={"": "cuda"})
        hidden = self.encoder.config.hidden_size
        self.register_buffer("feature_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(std, dtype=torch.float32))
        self.numeric = nn.Sequential(nn.Linear(len(mean), 64), nn.GELU(), nn.LayerNorm(64))
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(hidden*2+64, 256), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(256, 1))

    def forward(self, ids, mask, types, numeric):
        hidden = self.encoder(input_ids=ids, attention_mask=mask, token_type_ids=types).last_hidden_state
        valid = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden*valid).sum(1)/valid.sum(1).clamp_min(1)
        structured = self.numeric(((numeric-self.feature_mean)/self.feature_std).clamp(-10, 10))
        return self.head(torch.cat([hidden[:, 0], pooled, structured], dim=1)).squeeze(-1)


class Batches:
    def __init__(self, data, tokens, device):
        self.ids = np.load(tokens/"ids.npy", mmap_mode="r")
        self.mask = np.load(tokens/"mask.npy", mmap_mode="r")
        self.types = np.load(tokens/"types.npy", mmap_mode="r")
        self.numeric = np.load(data/"numeric.npy", mmap_mode="r")
        self.device = device

    def get(self, indices):
        # Dynamic trimming reduces compute while preserving every cached token.
        length = int(self.mask[indices].sum(1).max())
        length = min(self.ids.shape[1], max(8, math.ceil(length/8)*8))
        return (torch.as_tensor(self.ids[indices, :length].astype(np.int64), device=self.device),
                torch.as_tensor(self.mask[indices, :length].astype(np.int64), device=self.device),
                torch.as_tensor(self.types[indices, :length].astype(np.int64), device=self.device),
                torch.as_tensor(self.numeric[indices], device=self.device))


@torch.inference_mode()
def probabilities(model, batches, indices, size, dtype):
    model.eval()
    out = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), size):
        chosen = indices[start:start+size]
        with torch.autocast("cuda", dtype=dtype):
            values = model(*batches.get(chosen))
        out[start:start+len(chosen)] = values.float().sigmoid().cpu().numpy()
    return out


def train(data, output, model_path, epochs=4, batch_size=16, accumulation=2, length=192, seed=427, benchmark_only=False):
    data, output, model_path = map(Path, (data, output, model_path))
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this experiment must run on a GPU")
    torch.set_num_threads(2)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    write_json(output/"status.json", {"state": "tokenizing", "gpu": torch.cuda.get_device_name(0)})
    tokens = tokenize(data, model_path, length)
    metadata = json.loads((data/"entities.json").read_text(encoding="utf-8"))
    g, y, teacher = [np.load(data/(name+".npy")) for name in ("groups", "labels", "teacher")]
    split = np.array([m["split"] for m in metadata])
    fit_ids, tune_ids = np.flatnonzero(split[g] == "fit"), np.flatnonzero(split[g] == "tune")
    batches = Batches(data, tokens, "cuda")
    fx = batches.numeric[fit_ids]
    mean, std = fx.mean(0), np.maximum(fx.std(0), .01)
    del fx
    model = Matcher(model_path, mean, std).cuda()
    count = sum(p.numel() for p in model.parameters())
    if count > 8_000_000_000:
        raise ValueError("Competition model size limit exceeded")
    encoder = list(model.encoder.parameters())
    head = list(model.numeric.parameters())+list(model.head.parameters())
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": 2e-5}, {"params": head, "lr": 3e-4}], weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    truth = np.array([max(1, m["truth_count"]) for m in metadata])
    weights = 1/np.sqrt(truth[g[fit_ids]])
    weights /= weights.mean()
    all_weights = np.ones(len(g), dtype=np.float32)
    all_weights[fit_ids] = weights
    steps_epoch = math.ceil(len(fit_ids)/batch_size/accumulation)
    total_steps = steps_epoch*epochs
    def rate(step):
        warmup = max(1, int(total_steps*.06))
        return (step+1)/warmup if step < warmup else max(.05, (total_steps-step)/max(1, total_steps-warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
              "model_id": MODEL_ID, "revision": REVISION, "license": "Apache-2.0", "parameters": count,
              "fit_pairs": len(fit_ids), "tune_pairs": len(tune_ids), "epochs": [],
              "settings": {"epochs": epochs, "batch_size": batch_size, "accumulation": accumulation, "length": length, "seed": seed},
              "data": fingerprint([data/"prepared.json"])}
    prepared = json.loads((data/"prepared.json").read_text(encoding="utf-8"))
    report["gate"] = prepared["config"]["gate"]
    teacher_dir = Path("artifacts/experiment_v3/model")
    teacher_manifest = json.loads((teacher_dir/"manifest.json").read_text(encoding="utf-8"))
    report["teacher_model"] = {"manifest": teacher_manifest,
        "sha256": [hashlib.sha256((teacher_dir/f).read_bytes()).hexdigest() for f in teacher_manifest["model_files"]]}
    rng = np.random.default_rng(seed)
    best_score = -1.
    started = time.monotonic()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(fit_ids)
        optimizer.zero_grad(set_to_none=True)
        loss_sum, seen = 0., 0
        epoch_start = time.monotonic()
        for step, start in enumerate(range(0, len(order), batch_size)):
            ids = order[start:start+batch_size]
            labels = torch.as_tensor(y[ids], device="cuda", dtype=torch.float32)
            weight = torch.as_tensor(all_weights[ids], device="cuda")
            with torch.autocast("cuda", dtype=dtype):
                logits = model(*batches.get(ids))
                loss = (nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")*weight).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite neural training loss")
            scaler.scale(loss/accumulation).backward()
            if (step+1) % accumulation == 0 or start+batch_size >= len(order):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True); scheduler.step()
            loss_sum += float(loss.detach())*len(ids)
            seen += len(ids)
            if (step+1) % 50 == 0:
                torch.cuda.synchronize()
                speed = seen/max(.001, time.monotonic()-epoch_start)
                state = {"state": "training", "epoch": epoch+1, "epochs": epochs, "pairs_seen": seen,
                         "pairs_per_epoch": len(order), "pairs_per_second": speed, "loss": loss_sum/seen,
                         "gpu": report["gpu"], "peak_gpu_GiB": torch.cuda.max_memory_allocated()/2**30}
                write_json(output/"status.json", state)
                log(f"GPU epoch {epoch+1}/{epochs}: {seen:,}/{len(order):,}; loss={loss_sum/seen:.4f}; {speed:.1f} pairs/s")
                if benchmark_only:
                    write_json(output/"benchmark.json", state)
                    write_json(output/"status.json", {**state, "state": "benchmark_complete"})
                    return
        write_json(output/"status.json", {"state": "tuning", "epoch": epoch+1})
        p = teacher.copy()
        neural_tune = probabilities(model, batches, tune_ids, batch_size*2, dtype)
        choices = []
        tune_entities = np.flatnonzero(split == "tune")
        for alpha in (0., .15, .3, .5, .7, 1.):
            p[tune_ids] = (1-alpha)*teacher[tune_ids]+alpha*neural_tune
            threshold, score = tune_threshold(*subset_arrays(tune_entities, g, y, p, metadata))
            choices.append({"alpha": alpha, "threshold": threshold, "tune_macro_f0_5": score})
        selected = max(choices, key=lambda x: (x["tune_macro_f0_5"], -x["alpha"]))
        report["epochs"].append({"epoch": epoch+1, "loss": loss_sum/seen, "seconds": time.monotonic()-epoch_start, "choices": choices, "selected": selected})
        log(f"Epoch {epoch+1}: best tune={selected['tune_macro_f0_5']:.6f}, neural weight={selected['alpha']}")
        if selected["tune_macro_f0_5"] > best_score:
            best_score = selected["tune_macro_f0_5"]
            report["selected"] = {"epoch": epoch+1, **selected}
            checkpoint = output/"best.pt"
            torch.save(model.state_dict(), checkpoint.with_suffix(".tmp"))
            checkpoint.with_suffix(".tmp").replace(checkpoint)
        write_json(output/"report.json", report)
    # Freeze the epoch, blend and threshold before inspecting either audit set.
    model.load_state_dict(torch.load(output/"best.pt", map_location="cuda", weights_only=True))
    selected = report["selected"]
    final = teacher.copy()
    raw_neural = np.full(len(teacher), np.nan, dtype=np.float32)
    report["metrics"] = {}
    for s in ("tune", "audit", "confirm"):
        ids = np.flatnonzero(split[g] == s)
        neural_p = probabilities(model, batches, ids, batch_size*2, dtype)
        raw_neural[ids] = neural_p
        final[ids] = (1-selected["alpha"])*teacher[ids]+selected["alpha"]*neural_p
        report["metrics"][s] = metric_report(np.flatnonzero(split == s), g, y, final, metadata, selected["threshold"])
    report["elapsed_seconds"] = time.monotonic()-started
    report["peak_gpu_GiB"] = torch.cuda.max_memory_allocated()/2**30
    report["note"] = "No audit/confirmation labels used in fitting, epoch selection, blend or threshold. Confirmation businesses are disjoint from training. Local score is not leaderboard score."
    np.save(output/"probabilities.npy", final)
    np.save(output/"neural_probabilities.npy", raw_neural)
    write_json(output/"report.json", report)
    write_json(output/"status.json", {"state": "complete", "selected": selected, "metrics": report["metrics"]})
    log(f"GPU experiment complete: {report['metrics']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model", default="artifacts/pretrained/multilingual_minilm")
    parser.add_argument("--data", default="artifacts/neural_data_v1")
    parser.add_argument("--output", default="artifacts/neural_v1")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--length", type=int, default=192)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.download:
            download(args.model)
        else:
            train(args.data, args.output, args.model, args.epochs, args.batch_size, args.accumulation, args.length, benchmark_only=args.benchmark_only)
    except Exception as exc:
        write_json(Path(args.output)/"status.json", {"state": "failed", "error": repr(exc)})
        raise
