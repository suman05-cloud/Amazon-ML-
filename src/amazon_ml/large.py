"""Scale to 32,000 additional labeled businesses, then train CUDA XGBoost."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np

from .blocking import DiskIndex
from .data import fingerprint, log, sample_references, selected_truth, write_json
from .enhanced import FEATURE_NAMES, SCHEMA, feature_matrix
from .text import Record, VERSION
from .train import metric_report, subset_arrays
from .metrics import tune_threshold


def wait_for_index(index, output):
    while True:
        complete, count = False, 0
        if Path(index).exists():
            try:
                db = sqlite3.connect(f"file:{Path(index).resolve().as_posix()}?mode=ro", uri=True, timeout=1)
                meta = dict(db.execute("SELECT key,value FROM meta"))
                complete, count = meta.get("complete") == "yes", int(meta.get("rows", 0))
                db.close()
            except sqlite3.OperationalError:
                pass
        if complete:
            return
        write_json(Path(output)/"status.json", {"state": "waiting_for_training_index", "indexed_targets": count})
        time.sleep(20)


def prepare(data, index, output, additional=32000, seed=77391):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source, base = Path("artifacts/experiment_v2"), Path("artifacts/experiment_v3")
    paths = [Path(data)/"train"/f"train_source{k}.tsv" for k in (1, 2, 3)]+[Path(data)/"train"/"train_ground_truth.tsv"]
    config = {"additional": additional, "seed": seed, "inputs": fingerprint(paths), "schema": SCHEMA,
              "base": fingerprint([source/"entities.json", source/"labels.npy", source/"groups.npy", base/"features.npy"]),
              "top_k": 80, "cap": 400}
    marker = output/"prepared.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8"))["config"] != config:
            raise ValueError("Large experiment configuration changed")
        return
    saved_config = output/"config.json"
    if saved_config.exists() and json.loads(saved_config.read_text(encoding="utf-8")) != config:
        raise ValueError("Cannot resume a different experiment")
    write_json(saved_config, config)
    selected_path = output/"new_references.json"
    if not selected_path.exists():
        excluded_ids, excluded_names, excluded_targets = set(), set(), set()
        for folder in (Path("artifacts/experiment"), source, Path("artifacts/confirmation_v3")):
            with (folder/"cases.jsonl").open(encoding="utf-8") as f:
                for line in f:
                    case = json.loads(line)
                    truth = set(case["truth"])
                    excluded_ids.add(case["reference"]["id"])
                    excluded_targets.update(truth)
                    for r in [case["reference"]]+[c for c in case["candidates"] if c["id"] in truth]:
                        excluded_names.add((r["country"], " ".join(sorted(r["core"].split()))))
        refs = sample_references(paths[0], additional*2, seed)
        refs = [r for r in refs if r.id not in excluded_ids and (r.country, " ".join(sorted(r.core.split()))) not in excluded_names]
        truth = selected_truth(paths[3], refs)
        refs = [r for r in refs if not truth[r.id] & excluded_targets][:additional]
        if len(refs) != additional:
            raise ValueError("Insufficient disjoint additional references")
        write_json(selected_path, [{"record": asdict(r), "truth": sorted(truth[r.id])} for r in refs])
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    original_meta = json.loads((source/"entities.json").read_text(encoding="utf-8"))
    nbase = len(original_meta)
    base_y = np.load(source/"labels.npy")
    base_groups = np.load(source/"groups.npy")
    base_x = np.load(base/"features.npy", mmap_mode="r")
    capacity = len(base_y)+additional*80
    checkpoint_path = output/"checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        x = np.load(output/"features.npy", mmap_mode="r+")
        y = np.load(output/"labels.npy", mmap_mode="r+")
        groups = np.load(output/"groups.npy", mmap_mode="r+")
        metadata = json.loads((output/"entities.json").read_text(encoding="utf-8"))[:nbase+checkpoint["references"]]
    else:
        x = np.lib.format.open_memmap(output/"features.npy", mode="w+", dtype=np.float32, shape=(capacity, len(FEATURE_NAMES)))
        y = np.lib.format.open_memmap(output/"labels.npy", mode="w+", dtype=np.uint8, shape=(capacity,))
        groups = np.lib.format.open_memmap(output/"groups.npy", mode="w+", dtype=np.int32, shape=(capacity,))
        x[:len(base_y)], y[:len(base_y)], groups[:len(base_y)] = base_x, base_y, base_groups
        metadata = original_meta
        checkpoint = {"references": 0, "pairs": len(base_y), "case_bytes": 0}
        write_json(output/"entities.json", metadata)
        write_json(checkpoint_path, checkpoint)
    del base_x, base_y, base_groups
    disk = DiskIndex(index, block_cap=400)
    if json.loads(disk.meta("inputs")) != fingerprint(paths[1:3]):
        raise ValueError("Training index points to different target files")
    case_path = output/"cases_extra.jsonl"
    offset = checkpoint["pairs"]
    with case_path.open("a+b") as f:
        f.truncate(checkpoint["case_bytes"])
        f.seek(0, 2)
        for i in range(checkpoint["references"], len(selected)):
            r = Record(**selected[i]["record"])
            truth = set(selected[i]["truth"])
            ranked = disk.query(r, 80)
            n = len(ranked)
            x[offset:offset+n] = feature_matrix(r, ranked)
            y[offset:offset+n] = [int(c.id in truth) for c, _ in ranked]
            groups[offset:offset+n] = nbase+i
            offset += n
            metadata.append({"id": r.id, "country": r.country, "split": "fit", "truth_count": len(truth), "candidate_count": n})
            f.write((json.dumps({"reference": asdict(r), "truth": sorted(truth), "candidates": [asdict(c) for c, _ in ranked]}, ensure_ascii=False)+"\n").encode("utf-8"))
            if (i+1) % 500 == 0 or i+1 == len(selected):
                x.flush(); y.flush(); groups.flush(); f.flush()
                write_json(output/"entities.json", metadata)
                write_json(checkpoint_path, {"references": i+1, "pairs": offset, "case_bytes": f.tell()})
                write_json(output/"status.json", {"state": "preparing_features", "new_references": i+1, "total_new_references": additional, "pairs": offset})
                log(f"Large training cache: {i+1:,}/{additional:,} new references; {offset:,} pairs")
    disk.close()
    write_json(marker, {"config": config, "pairs": offset, "references": len(metadata), "original_references": nbase,
                        "note": "Additional references all fit-only, excluding prior reference IDs, normalized name groups and truth target IDs."})


def fit(output, rounds=1600):
    import xgboost as xgb
    output = Path(output)
    prepared = json.loads((output/"prepared.json").read_text(encoding="utf-8"))
    metadata = json.loads((output/"entities.json").read_text(encoding="utf-8"))
    n = prepared["pairs"]
    x, y, g = [np.load(output/(p+".npy"), mmap_mode="r")[:n] for p in ("features", "labels", "groups")]
    split = np.array([r["split"] for r in metadata])
    fm, tm = split[g] == "fit", split[g] == "tune"
    write_json(output/"status.json", {"state": "building_gpu_matrices", "fit_pairs": int(fm.sum())})
    dfit = xgb.QuantileDMatrix(x[fm], label=y[fm], feature_names=FEATURE_NAMES, nthread=2, max_bin=256)
    dtune = xgb.QuantileDMatrix(x[tm], label=y[tm], feature_names=FEATURE_NAMES, nthread=2, max_bin=256, ref=dfit)
    model_dir = output/"model"
    model_dir.mkdir(exist_ok=True)
    old = np.load("artifacts/experiment_v3/probabilities.npy")
    teacher = np.zeros(n)
    teacher[:len(old)] = old
    from .models import load_models
    teacher_dir = Path("artifacts/experiment_v3/model")
    teacher_manifest = json.loads((teacher_dir/"manifest.json").read_text(encoding="utf-8"))
    teacher_models = load_models(teacher_dir, teacher_manifest)
    for start in range(len(old), n, 20000):
        block = x[start:start+20000]
        teacher[start:start+len(block)] = np.mean([m.predict(block, num_threads=2) for m in teacher_models], axis=0)
    tune_ids = np.flatnonzero(split == "tune")
    result = {"prepared": prepared, "trials": [], "xgboost": xgb.__version__, "device": "cuda", "selection": "tune only"}
    all_predictions = []
    for depth in (6, 8):
        write_json(output/"status.json", {"state": "gpu_training", "depth": depth, "fit_pairs": int(fm.sum())})
        params = {"objective": "binary:logistic", "eval_metric": "logloss", "device": "cuda", "tree_method": "hist",
                  "eta": .04, "max_depth": depth, "min_child_weight": 10., "lambda": 8., "alpha": .1,
                  "subsample": .9, "colsample_bytree": .9, "seed": 42, "nthread": 2, "max_bin": 256}
        log(f"CUDA XGBoost depth {depth}: {int(fm.sum()):,} fitting pairs")
        class Progress(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                if epoch % 50 == 0:
                    write_json(output/"status.json", {"state": "gpu_training", "depth": depth, "round": epoch+1,
                               "max_rounds": rounds, "tune_logloss": evals_log["tune"]["logloss"][-1]})
                return False
        model = xgb.train(params, dfit, num_boost_round=rounds, evals=[(dtune, "tune")], early_stopping_rounds=100,
                          callbacks=[Progress()], verbose_eval=100)
        saved_config = json.loads(model.save_config())
        if not saved_config["learner"]["generic_param"]["device"].startswith("cuda"):
            raise RuntimeError("XGBoost did not use CUDA")
        model = model[:model.best_iteration+1]
        model.save_model(model_dir/f"depth{depth}.ubj")
        prediction = np.empty(n, dtype=np.float32)
        for start in range(0, n, 20000):
            block = xgb.DMatrix(x[start:start+20000], feature_names=FEATURE_NAMES, nthread=2)
            prediction[start:start+len(x[start:start+20000])] = model.predict(block)
        all_predictions.append(prediction)
        np.save(output/f"probabilities_depth{depth}.npy", prediction)
        for alpha in (.5, 1.):
            p = alpha*prediction+(1-alpha)*teacher
            threshold, score = tune_threshold(*subset_arrays(tune_ids, g, y, p, metadata))
            result["trials"].append({"depths": [depth], "alpha": alpha, "threshold": threshold, "tune_macro_f0_5": score,
                                      "trees": model.num_boosted_rounds(), "parameters": params})
        write_json(output/"report.json", result)
    for alpha in (.5, 1.):
        p = alpha*np.mean(all_predictions, axis=0)+(1-alpha)*teacher
        threshold, score = tune_threshold(*subset_arrays(tune_ids, g, y, p, metadata))
        result["trials"].append({"depths": [6, 8], "alpha": alpha, "threshold": threshold, "tune_macro_f0_5": score})
    # Baseline is an explicit candidate; a larger GPU model need not win.
    t, score = tune_threshold(*subset_arrays(tune_ids, g, y, teacher, metadata))
    result["trials"].append({"depths": [], "alpha": 0., "threshold": t, "tune_macro_f0_5": score})
    selected = max(result["trials"], key=lambda v: v["tune_macro_f0_5"])
    p = teacher.copy()
    if selected["depths"]:
        p = selected["alpha"]*np.mean([all_predictions[(6, 8).index(d)] for d in selected["depths"]], axis=0)+(1-selected["alpha"])*teacher
    result["selected"] = selected
    result["metrics"] = {s: metric_report(np.flatnonzero(split == s), g, y, p, metadata, selected["threshold"]) for s in ("tune", "audit")}
    files, types, weights = [], [], []
    for d in selected["depths"]:
        files.append(f"depth{d}.ubj"); types.append("xgboost"); weights.append(selected["alpha"]/len(selected["depths"]))
    if selected["alpha"] < 1:
        old_dir = Path("artifacts/experiment_v3/model")
        old_manifest = json.loads((old_dir/"manifest.json").read_text(encoding="utf-8"))
        for name in old_manifest["model_files"]:
            shutil.copyfile(old_dir/name, model_dir/name)
            files.append(name); types.append("lightgbm"); weights.append((1-selected["alpha"])/len(old_manifest["model_files"]))
    write_json(model_dir/"manifest.json", {"model_files": files, "model_types": types, "model_weights": weights,
               "threshold": selected["threshold"], "feature_names": FEATURE_NAMES, "feature_schema": SCHEMA,
               "normalization_version": VERSION, "top_k": 80, "block_cap": 400,
               "training": "Expanded fit-only references; GPU XGBoost; model/blend/threshold selected on original tune"})
    np.save(output/"probabilities.npy", p)
    write_json(output/"report.json", result)
    write_json(output/"status.json", {"state": "complete", "selected": selected, "metrics": result["metrics"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data set/student_resource/dataset")
    parser.add_argument("--index", default="artifacts/train.sqlite")
    parser.add_argument("--output", default="artifacts/large_v1")
    parser.add_argument("--additional", type=int, default=32000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--wait-for-neural", action="store_true")
    args = parser.parse_args()
    try:
        wait_for_index(args.index, args.output)
        prepare(args.data, args.index, args.output, args.additional)
        if not args.prepare_only:
            if args.wait_for_neural:
                state_path = Path("artifacts/neural_v1/status.json")
                while state_path.exists() and json.loads(state_path.read_text(encoding="utf-8")).get("state") not in ("complete", "failed"):
                    write_json(Path(args.output)/"status.json", {"state": "waiting_for_neural_gpu"})
                    time.sleep(20)
            fit(args.output)
    except Exception as exc:
        write_json(Path(args.output)/"status.json", {"state": "failed", "error": repr(exc)})
        raise
