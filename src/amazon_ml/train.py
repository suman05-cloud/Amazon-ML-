"""Full-pool candidate evaluation and independent fit/tune/audit partitions."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np

from .blocking import FocusedIndex
from .data import fingerprint, log, partition, sample_references, selected_truth, write_json
from .metrics import evaluate, tune_threshold
from .text import FEATURE_NAMES, VERSION, features


def prepare(data_dir, output, sample_size=8000, seed=2027, block_cap=400, top_k=80):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base = Path(data_dir) / "train"
    inputs = [base / f"train_source{i}.tsv" for i in (1, 2, 3)] + [base / "train_ground_truth.tsv"]
    config = {"inputs": fingerprint(inputs), "sample_size": sample_size, "seed": seed,
              "block_cap": block_cap, "top_k": top_k, "version": VERSION, "feature_names": FEATURE_NAMES}
    if (output / "prepared.json").exists():
        existing = json.loads((output / "prepared.json").read_text(encoding="utf-8"))
        if existing["config"] != config:
            raise ValueError("Experiment cache configuration changed; use a new work directory")
        log("Reusing complete feature cache")
        return
    started = time.monotonic()
    log(f"Sampling {sample_size:,} reference businesses")
    refs = sample_references(inputs[0], sample_size, seed)
    truth = selected_truth(inputs[-1], refs)
    index = FocusedIndex(refs, block_cap)
    index.build(inputs[1:3])
    log("Full-pool scan complete; constructing candidate features")
    X = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype=np.float32,
                                 shape=(len(refs) * top_k, len(FEATURE_NAMES)))
    y = np.zeros(len(refs) * top_k, dtype=np.uint8)
    groups = np.zeros(len(y), dtype=np.int32)
    metadata = []
    offset = 0
    with (output / "cases.jsonl").open("w", encoding="utf-8") as cases:
        for i, r in enumerate(refs):
            candidates = index.query(r, top_k)
            best = candidates[0][1] if candidates else 0.
            for target, score in candidates:
                X[offset] = features(r, target, score, best - score, len(candidates))
                y[offset] = target.id in truth[r.id]
                groups[offset] = i
                offset += 1
            metadata.append({"id": r.id, "country": r.country, "split": partition(r, seed),
                             "truth_count": len(truth[r.id]), "candidate_count": len(candidates)})
            cases.write(json.dumps({"reference": asdict(r), "truth": sorted(truth[r.id]),
                                    "candidates": [asdict(c) for c, _ in candidates]}, ensure_ascii=False) + "\n")
    X.flush()
    np.save(output / "labels.npy", y[:offset])
    np.save(output / "groups.npy", groups[:offset])
    write_json(output / "entities.json", metadata)
    total_links = sum(len(t) for t in truth.values())
    report = {"config": config, "reference_count": len(refs), "scanned_targets": index.scanned,
              "pairs": offset, "true_links": total_links, "candidate_recall": float(y[:offset].sum() / max(1, total_links)),
              "reduction_ratio": 1 - offset / max(1, len(refs) * index.scanned),
              "seconds": time.monotonic() - started,
              "split_counts": dict(Counter(r["split"] for r in metadata)),
              "country_counts": dict(Counter(r["country"] for r in metadata))}
    write_json(output / "prepared.json", report)
    log(f"Candidate recall {report['candidate_recall']:.4%}; {offset:,} model pairs")


def subset_arrays(entity_indices, groups, y, probs, metadata):
    lookup = np.full(len(metadata), -1, dtype=np.int32)
    lookup[entity_indices] = np.arange(len(entity_indices))
    keep = lookup[groups] >= 0
    return (np.array([metadata[i]["truth_count"] for i in entity_indices]),
            lookup[groups[keep]], y[keep], probs[keep])


def metric_report(indices, groups, y, probabilities, metadata, threshold):
    arrays = subset_arrays(indices, groups, y, probabilities, metadata)
    result = evaluate(*arrays, threshold)
    result["candidate_recall"] = float(arrays[2].sum() / max(1, arrays[0].sum()))
    result["blocking_macro_ceiling"] = evaluate(arrays[0], arrays[1], arrays[2], arrays[2], .5)["macro_f0_5"]
    return result


def fit(output, threads=4, seeds=(42, 2026), rounds=800, transfer=True):
    output = Path(output)
    prepared = json.loads((output / "prepared.json").read_text(encoding="utf-8"))
    if prepared["config"]["version"] != VERSION or prepared["config"]["feature_names"] != FEATURE_NAMES:
        raise ValueError("Cached features were produced by another normalization/retrieval version; prepare a new work directory")
    metadata = json.loads((output / "entities.json").read_text(encoding="utf-8"))
    n = prepared["pairs"]
    X = np.load(output / "features.npy", mmap_mode="r")[:n]
    y = np.load(output / "labels.npy")
    groups = np.load(output / "groups.npy")
    split = np.array([m["split"] for m in metadata])
    country = np.array([m["country"] for m in metadata])
    masks = {s: split[groups] == s for s in ("fit", "tune", "audit")}
    if any(np.unique(y[masks[s]]).size < 2 for s in ("fit", "tune")):
        raise ValueError("Need positive and negative candidate pairs in fit and tune; increase the sample")
    model_dir = output / "model"
    model_dir.mkdir(exist_ok=True)
    probabilities = np.zeros(n, dtype=np.float64)
    model_files, best_iterations = [], []
    params = dict(objective="binary", learning_rate=.045, num_leaves=31,
                  min_child_samples=80, reg_lambda=5., reg_alpha=.1,
                  colsample_bytree=.9, subsample=.9, subsample_freq=1,
                  n_estimators=rounds, n_jobs=threads, verbosity=-1,
                  deterministic=True, force_col_wise=True)
    importance = np.zeros(len(FEATURE_NAMES))
    for seed in seeds:
        log(f"Training LightGBM seed {seed}; {int(masks['fit'].sum()):,} fit pairs")
        model = lgb.LGBMClassifier(**params, random_state=seed)
        model.fit(X[masks["fit"]], y[masks["fit"]],
                  eval_set=[(X[masks["tune"]], y[masks["tune"]])], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(60, verbose=False)], feature_name=FEATURE_NAMES)
        path = model_dir / f"lightgbm_{seed}.txt"
        model.booster_.save_model(str(path))
        model_files.append(path.name)
        best_iterations.append(model.best_iteration_)
        probabilities += model.booster_.predict(X, num_threads=threads) / len(seeds)
        importance += model.booster_.feature_importance(importance_type="gain") / len(seeds)
    tune_ids = np.flatnonzero(split == "tune")
    threshold, _ = tune_threshold(*subset_arrays(tune_ids, groups, y, probabilities, metadata))
    report = {"threshold": threshold, "best_iterations": best_iterations, "prepared": prepared,
              "training": {"seeds": list(seeds), "rounds": rounds, "threads": threads, "parameters": params},
              "metrics": {}, "feature_importance": dict(sorted(zip(FEATURE_NAMES, importance.tolist()), key=lambda x: -x[1]))}
    for s in ("tune", "audit"):
        report["metrics"][s] = metric_report(np.flatnonzero(split == s), groups, y, probabilities, metadata, threshold)
        for c in sorted(set(country)):
            report["metrics"][f"{s}/{c}"] = metric_report(np.flatnonzero((split == s) & (country == c)), groups, y, probabilities, metadata, threshold)
    if transfer:
        report["country_transfer"] = {}
        for held_country in sorted(set(country)):
            # Fit and tune exclusively on the other country; evaluate held-country audit.
            fm = masks["fit"] & (country[groups] != held_country)
            tm = masks["tune"] & (country[groups] != held_country)
            if np.unique(y[fm]).size < 2 or np.unique(y[tm]).size < 2:
                continue
            log(f"Unseen-country stress test: holding out {held_country}")
            model = lgb.LGBMClassifier(**params, random_state=seeds[0])
            model.fit(X[fm], y[fm], eval_set=[(X[tm], y[tm])], callbacks=[lgb.early_stopping(60, verbose=False)])
            p = model.booster_.predict(X, num_threads=threads)
            t, _ = tune_threshold(*subset_arrays(np.flatnonzero((split == "tune") & (country != held_country)), groups, y, p, metadata))
            report["country_transfer"][held_country] = {"threshold": t, **metric_report(np.flatnonzero((split == "audit") & (country == held_country)), groups, y, p, metadata, t)}
    # Audit predictions are never fed back into fitting or threshold selection.
    np.save(output / "probabilities.npy", probabilities)
    write_json(output / "report.json", report)
    write_json(model_dir / "manifest.json", {"model_files": model_files, "threshold": threshold,
               "feature_names": FEATURE_NAMES, "normalization_version": VERSION,
               "top_k": prepared["config"]["top_k"], "block_cap": prepared["config"]["block_cap"],
               "threads": threads, "training": "fit partition only; tune for early stopping and threshold; audit untouched"})
    export_errors(output, metadata, groups, y, probabilities, threshold)
    log(f"AUDIT macro F0.5 = {report['metrics']['audit']['macro_f0_5']:.6f}; threshold = {threshold:.3f}")
    return report


def export_errors(output, metadata, groups, y, probabilities, threshold):
    starts = np.searchsorted(groups, np.arange(len(metadata) + 1))
    with (output / "errors.jsonl").open("w", encoding="utf-8") as out, (output / "cases.jsonl").open(encoding="utf-8") as inp:
        for i, line in enumerate(inp):
            if metadata[i]["split"] != "audit":
                continue
            case = json.loads(line)
            truth = set(case["truth"])
            p = probabilities[starts[i]:starts[i + 1]]
            predictions = {c["id"] for c, value in zip(case["candidates"], p) if value >= threshold}
            if predictions != truth:
                case["false_positives"] = sorted(predictions - truth)
                case["false_negatives"] = sorted(truth - predictions)
                case["blocked_out"] = sorted(truth - {c["id"] for c in case["candidates"]})
                case["probabilities"] = p.tolist()
                out.write(json.dumps(case, ensure_ascii=False) + "\n")
