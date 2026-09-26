"""Reuse completed candidate chunks with new tree/neural models.

Four CPU workers compute features; one coordinator owns the GPU. No GPU model
is duplicated across CPU workers, and every original candidate stays in the audit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3

import lightgbm as lgb
import numpy as np

from .blocking import rank_candidates
from .data import fingerprint, log, rows, write_json
from .enhanced import FEATURE_NAMES, SCHEMA, feature_matrix
from .models import load_models
from .predict import merge
from .text import Record, VERSION


def build_raw(data, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(data)/"test"/f"test_source{i}.tsv" for i in (1, 2, 3)]
    signature = {"inputs": fingerprint(paths), "normalization_version": VERSION}
    db = sqlite3.connect(destination)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-65536")
    db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT, country TEXT, name TEXT, core TEXT, address TEXT, raw_name TEXT, raw_address TEXT, raw_country TEXT)")
    meta = dict(db.execute("SELECT key,value FROM meta"))
    if "config" in meta and json.loads(meta["config"]) != signature:
        raise ValueError("Raw index input changed")
    if meta.get("complete") == "yes":
        db.close()
        return
    db.execute("INSERT OR REPLACE INTO meta VALUES ('config',?)", (json.dumps(signature),))
    db.commit()
    done = int(meta.get("rows", 0))
    batch, count = [], 0
    def commit():
        db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?)", batch)
        db.execute("INSERT OR REPLACE INTO meta VALUES ('rows',?)", (str(count),))
        db.commit()
        batch.clear()
    for path in paths:
        for row in rows(path):
            count += 1
            if count <= done:
                continue
            r = Record.from_row(row)
            batch.append((r.id, r.country, r.name, r.core, r.address, row["business_name"], row["business_address"], row["country"]))
            if len(batch) == 10000:
                commit()
                if count % 250000 == 0:
                    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    log(f"Raw/normalized lookup: {count:,} records")
    if batch:
        commit()
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS records_id ON records(id)")
    db.execute("INSERT OR REPLACE INTO meta VALUES ('complete','yes')")
    db.commit()
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    log(f"Raw/normalized lookup complete: {count:,} records")


_DB = None
_MODELS = None
_MANIFEST = None


def initialize(raw_index, tree_model):
    global _DB, _MODELS, _MANIFEST
    _DB = sqlite3.connect(f"file:{Path(raw_index).resolve().as_posix()}?mode=ro", uri=True)
    _DB.execute("PRAGMA cache_size=-32768")
    _MANIFEST = json.loads((Path(tree_model)/"manifest.json").read_text(encoding="utf-8"))
    if _MANIFEST.get("feature_schema") != SCHEMA or _MANIFEST["feature_names"] != FEATURE_NAMES:
        raise ValueError("Rescoring expects the enhanced tree schema")
    _MODELS = load_models(tree_model, _MANIFEST)


def get_records(ids):
    result = {}
    ids = list(ids)
    for start in range(0, len(ids), 500):
        chunk = ids[start:start+500]
        for row in _DB.execute("SELECT * FROM records WHERE id IN ("+",".join("?" for _ in chunk)+")", chunk):
            result[row[0]] = (Record(*row[:5]), f"name: {row[5]}; address: {row[6]}; country: {row[7]}")
    if len(result) != len(set(ids)):
        raise ValueError("Missing raw test records")
    return result


def score_chunk(path, gate):
    with Path(path).open(encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]
    ids = {sid for sid, _, _ in cases}
    ids.update(t for _, _, candidates in cases for t in candidates)
    records = get_records(ids)
    matrices, boundaries, ranked_ids = [], [0], []
    for sid, _, candidates in cases:
        ref = records[sid][0]
        ranked = rank_candidates(ref, [records[t][0] for t in candidates], len(candidates))
        matrices.append(feature_matrix(ref, ranked))
        ranked_ids.append([r.id for r, _ in ranked])
        boundaries.append(boundaries[-1]+len(ranked))
    x = np.concatenate(matrices) if matrices else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    p = np.mean([m.predict(x, num_threads=1) for m in _MODELS], axis=0) if len(x) else np.empty(0)
    take = np.flatnonzero(p >= gate)
    text_pairs = []
    for i, (sid, _, _) in enumerate(cases):
        for j, tid in enumerate(ranked_ids[i], start=boundaries[i]):
            if p[j] >= gate:
                text_pairs.append((records[sid][1], records[tid][1]))
    return [c[0] for c in cases], ranked_ids, boundaries, p, take, x[take], text_pairs


def run(data, raw_index, source_output, output, tree_model, neural_model=None, workers=4, gpu_batch=32):
    source_output, output, tree_model = map(Path, (source_output, output, tree_model))
    output.mkdir(parents=True, exist_ok=True)
    # The baseline is only read; partially completed full runs cannot be mistaken
    # for a complete new submission.
    roots = sorted((source_output/"chunks").glob("shard-*"))
    if not roots or any(not (r/"complete.json").exists() for r in roots):
        raise ValueError("Wait for all original prediction shards to finish before full rescoring")
    completions = [json.loads((r/"complete.json").read_text(encoding="utf-8")) for r in roots]
    if any(c["partial"] for c in completions):
        raise ValueError("Cannot create a full submission from preview candidates")
    configs = [json.loads((r/"config.json").read_text(encoding="utf-8")) for r in roots]
    manifest = json.loads((tree_model/"manifest.json").read_text(encoding="utf-8"))
    for c in configs:
        if any(c["manifest"][k] != manifest[k] for k in ("top_k", "block_cap", "normalization_version")):
            raise ValueError("Candidate cache is incompatible with the selected retrieval settings")
    db = sqlite3.connect(f"file:{Path(raw_index).resolve().as_posix()}?mode=ro", uri=True)
    meta = dict(db.execute("SELECT key,value FROM meta"))
    db.close()
    expected = {"inputs": fingerprint([Path(data)/"test"/f"test_source{i}.tsv" for i in (1, 2, 3)]), "normalization_version": VERSION}
    if meta.get("complete") != "yes" or json.loads(meta["config"]) != expected:
        raise ValueError("Raw index is incomplete or from different data")
    alpha, threshold, gate = 0., manifest["threshold"], 2.
    neural_signature = None
    if neural_model:
        import torch
        from transformers import AutoTokenizer
        from .neural import Matcher
        neural_model = Path(neural_model)
        report = json.loads((neural_model/"report.json").read_text(encoding="utf-8"))
        if "metrics" not in report:
            raise ValueError("Neural training/validation has not completed")
        alpha, threshold = report["selected"]["alpha"], report["selected"]["threshold"]
        gate = report["gate"]
        teacher_signature = {"manifest": manifest,
            "sha256": [hashlib.sha256((tree_model/f).read_bytes()).hexdigest() for f in manifest["model_files"]]}
        if report["teacher_model"] != teacher_signature:
            raise ValueError("Neural experiment used a different tree model")
        neural_signature = {"report": report, "weights_sha256": hashlib.sha256((neural_model/"best.pt").read_bytes()).hexdigest()}
        if alpha > 0:
            model_path = Path("artifacts/pretrained/multilingual_minilm")
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = Matcher(model_path, np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES))).cuda()
            model.load_state_dict(torch.load(neural_model/"best.pt", map_location="cuda", weights_only=True))
            model.eval()
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    signature = {"source": configs, "tree_manifest": manifest,
                 "tree_sha256": [hashlib.sha256((tree_model/f).read_bytes()).hexdigest() for f in manifest["model_files"]],
                 "neural": neural_signature, "raw_inputs": expected, "shards": 1, "shard": 0, "limit": None}
    chunks = output/"chunks"/"shard-000"
    chunks.mkdir(parents=True, exist_ok=True)
    state = chunks/"config.json"
    if state.exists() and json.loads(state.read_text(encoding="utf-8")) != signature:
        raise ValueError("Rescore settings changed; use a new output directory")
    write_json(state, signature)
    tasks = [r/f"{i:07d}.jsonl" for r, c in zip(roots, completions) for i in range(c["batches"])]
    pending = [(i, p) for i, p in enumerate(tasks) if not (chunks/f"{i:07d}.jsonl").exists()]
    write_json(output/"run_status.json", {"state": "rescoring", "cpu_workers": workers, "gpu": alpha > 0, "chunks": len(tasks), "remaining_chunks": len(pending)})
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(str(raw_index), str(tree_model))) as pool:
        completed_chunks = len(tasks)-len(pending)
        active = []
        iterator = iter(pending)
        for _ in range(workers):
            if (task := next(iterator, None)) is not None:
                active.append((task[0], pool.submit(score_chunk, str(task[1]), gate)))
        while active:
            number, future = active.pop(0)
            sids, ids, bounds, p, selected, numeric, pairs = future.result()
            if alpha > 0 and len(selected):
                neural_p = np.empty(len(selected), dtype=np.float32)
                with torch.inference_mode():
                    for start in range(0, len(selected), gpu_batch):
                        batch = pairs[start:start+gpu_batch]
                        encoded = tokenizer([a for a, b in batch], [b for a, b in batch], padding=True,
                                            truncation="longest_first", max_length=report["settings"]["length"], return_tensors="pt")
                        token_ids = encoded["input_ids"].cuda()
                        mask = encoded["attention_mask"].cuda()
                        types = encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])).cuda()
                        with torch.autocast("cuda", dtype=dtype):
                            logits = model(token_ids, mask, types, torch.as_tensor(numeric[start:start+len(batch)], device="cuda"))
                        neural_p[start:start+len(batch)] = logits.float().sigmoid().cpu().numpy()
                p[selected] = (1-alpha)*p[selected]+alpha*neural_p
            path = chunks/f"{number:07d}.jsonl"
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as f:
                for i, sid in enumerate(sids):
                    matches = sorted(t for t, prob in zip(ids[i], p[bounds[i]:bounds[i+1]]) if prob >= threshold)
                    f.write(json.dumps([sid, matches, sorted(ids[i])])+"\n")
            temporary.replace(path)
            completed_chunks += 1
            write_json(output/"run_status.json", {"state": "rescoring", "cpu_workers": workers, "gpu": alpha > 0,
                       "completed_chunks": completed_chunks, "total_chunks": len(tasks)})
            log(f"Rescored chunk {number+1:,}/{len(tasks):,}")
            if (task := next(iterator, None)) is not None:
                active.append((task[0], pool.submit(score_chunk, str(task[1]), gate)))
    write_json(chunks/"complete.json", {"rows": sum(c["rows"] for c in completions), "batches": len(tasks), "partial": False})
    merge(output, 1)
    from .validate import validate
    write_json(output/"run_status.json", {"state": "validating"})
    validate(data, output)
    write_json(output/"run_status.json", {"state": "complete", "cpu_workers": workers, "gpu": alpha > 0})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data set/student_resource/dataset")
    p.add_argument("--raw-index", default="artifacts/test_raw.sqlite")
    p.add_argument("--build-raw", action="store_true")
    p.add_argument("--source-output", default="output")
    p.add_argument("--output", default="output_improved")
    p.add_argument("--tree-model", default="artifacts/experiment_v3/model")
    p.add_argument("--neural-model")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu-batch", type=int, default=32)
    args = p.parse_args()
    if args.build_raw:
        build_raw(args.data, args.raw_index)
    else:
        run(args.data, args.raw_index, args.source_output, args.output, args.tree_model, args.neural_model, args.workers, args.gpu_batch)
