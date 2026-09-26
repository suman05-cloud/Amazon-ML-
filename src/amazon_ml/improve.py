"""Non-destructive experiments using existing full-pool candidates.

Run: python -m amazon_ml.improve --source artifacts/experiment_v2 --output artifacts/experiment_v3
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import lightgbm as lgb
import numpy as np

from .data import fingerprint, log, write_json
from .enhanced import FEATURE_NAMES, SCHEMA, feature_matrix
from .metrics import tune_threshold
from .text import Record, VERSION
from .train import metric_report, subset_arrays


def prepare(source, output):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Experimental output must be separate from the baseline")
    output.mkdir(parents=True, exist_ok=True)
    inputs = [source / p for p in ("prepared.json", "cases.jsonl", "features.npy", "entities.json", "labels.npy", "groups.npy")]
    signature = {"schema": SCHEMA, "features": FEATURE_NAMES, "inputs": fingerprint(inputs)}
    marker = output / "enhanced_prepared.json"
    if marker.exists():
        report = json.loads(marker.read_text(encoding="utf-8"))
        if report["signature"] != signature:
            raise ValueError("Changed cache/schema; choose a fresh experiment directory")
        return
    prepared = json.loads((source / "prepared.json").read_text(encoding="utf-8"))
    n = prepared["pairs"]
    base = np.load(source / "features.npy", mmap_mode="r")[:n]
    X = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype=np.float32, shape=(n, len(FEATURE_NAMES)))
    offset = 0
    started = time.monotonic()
    with (source / "cases.jsonl").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            case = json.loads(line)
            ref = Record(**case["reference"])
            targets = [(Record(**c), 0.) for c in case["candidates"]]
            count = len(targets)
            X[offset:offset+count] = feature_matrix(ref, targets, base[offset:offset+count])
            offset += count
            if (i+1) % 500 == 0:
                log(f"Enhanced features: {i+1:,} references; {offset:,} pairs")
    if offset != n:
        raise ValueError("Candidate/base-feature alignment mismatch")
    X.flush()
    for name in ("labels.npy", "groups.npy", "entities.json"):
        shutil.copyfile(source / name, output / name)
    write_json(marker, {"signature": signature, "prepared": prepared, "seconds": time.monotonic()-started})


def run(source, output, threads=2, rounds=1600):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Experimental output must be separate from the baseline")
    write_json(output / "status.json", {"state": "preparing", "source": str(source)})
    prepare(source, output)
    cache = json.loads((output / "enhanced_prepared.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "entities.json").read_text(encoding="utf-8"))
    split = np.array([m["split"] for m in metadata])
    country = np.array([m["country"] for m in metadata])
    groups = np.load(output / "groups.npy")
    y = np.load(output / "labels.npy")
    X = np.load(output / "features.npy", mmap_mode="r")
    fm, tm = split[groups] == "fit", split[groups] == "tune"
    tune_ids = np.flatnonzero(split == "tune")
    # Materialize each split only once; preserve RAM for the live prediction run.
    xf, xt = X[fm], X[tm]
    baseline = np.load(source / "probabilities.npy")
    old_report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    result = {"source": str(source), "schema": SCHEMA, "features": len(FEATURE_NAMES),
              "baseline_tune": old_report["metrics"]["tune"], "trials": [],
              "evaluation_note": "Old audit is a development diagnostic after error inspection. A new holdout is required to confirm gains. Selection uses tune only."}
    variants = [("rich31", 31, False), ("macro31", 31, True), ("macro63", 63, True)]
    model_dir = output / "model"
    model_dir.mkdir(exist_ok=True)
    all_probs = []
    for j, (name, leaves, macro_weight) in enumerate(variants):
        write_json(output / "status.json", {"state": "training", "trial": name, "trial_index": j+1, "trials": len(variants)})
        params = dict(objective="binary", learning_rate=.035, num_leaves=leaves,
                      min_child_samples=60, reg_lambda=6., reg_alpha=.1,
                      colsample_bytree=.9, subsample=.9, subsample_freq=1,
                      n_estimators=rounds, n_jobs=threads, verbosity=-1,
                      deterministic=True, force_col_wise=True, random_state=42)
        weights = None
        if macro_weight:
            # Equalizing references approximately prevents large clusters from
            # dominating a metric that gives each reference equal weight.
            counts = np.array([max(1, m["truth_count"]) for m in metadata])
            weights = 1 / np.sqrt(counts[groups[fm]])
            weights /= weights.mean()
        log(f"Training {name}: {len(xf):,} fit pairs, {len(FEATURE_NAMES)} features, {threads} threads")
        model = lgb.LGBMClassifier(**params)
        model.fit(xf, y[fm], sample_weight=weights, eval_set=[(xt, y[tm])],
                  callbacks=[lgb.early_stopping(100, verbose=False)], feature_name=FEATURE_NAMES)
        model.booster_.save_model(str(model_dir / f"{name}.txt"))
        p = model.booster_.predict(X, num_threads=threads)
        all_probs.append(p)
        threshold, score = tune_threshold(*subset_arrays(tune_ids, groups, y, p, metadata))
        trial = {"name": name, "model_files": [f"{name}.txt"], "threshold": threshold,
                 "tune_macro_f0_5": score, "best_iteration": model.best_iteration_,
                 "parameters": params, "weighting": "inverse_sqrt_reference_truth_count" if macro_weight else "none"}
        result["trials"].append(trial)
        np.save(output / f"probabilities_{name}.npy", p)
        write_json(output / "experiment_report.json", result)
        log(f"{name}: tune F0.5={score:.6f}, threshold={threshold:.3f}, trees={model.best_iteration_}")
        del model
    # A small, predetermined ensemble menu; no search using audit scores.
    for indices in ((0, 1), (1, 2), (0, 1, 2)):
        p = np.mean([all_probs[i] for i in indices], axis=0)
        threshold, score = tune_threshold(*subset_arrays(tune_ids, groups, y, p, metadata))
        result["trials"].append({"name": "+".join(variants[i][0] for i in indices),
                                 "model_files": [variants[i][0]+".txt" for i in indices],
                                 "threshold": threshold, "tune_macro_f0_5": score})
    best = max(result["trials"], key=lambda t: t["tune_macro_f0_5"])
    selected = [i for i, v in enumerate(variants) if v[0]+".txt" in best["model_files"]]
    p = np.mean([all_probs[i] for i in selected], axis=0)
    result["selected"] = best
    result["metrics"] = {}
    for s in ("tune", "audit"):
        ids = np.flatnonzero(split == s)
        result["metrics"][s] = metric_report(ids, groups, y, p, metadata, best["threshold"])
        for c in sorted(set(country)):
            result["metrics"][f"{s}/{c}"] = metric_report(np.flatnonzero((split == s) & (country == c)), groups, y, p, metadata, best["threshold"])
    # Paired entity bootstrap measures sample uncertainty, not tuning bias.
    ids = np.flatnonzero(split == "audit")
    truth, g, labels, ps = subset_arrays(ids, groups, y, p, metadata)
    _, _, _, pb = subset_arrays(ids, groups, y, baseline, metadata)
    def per_entity(prob, threshold):
        take = prob >= threshold
        pred = np.bincount(g[take], minlength=len(ids))
        tp = np.bincount(g[take & (labels == 1)], minlength=len(ids))
        den = .25*truth+pred
        return np.divide(1.25*tp, den, out=np.ones(len(ids)), where=den != 0)
    delta = per_entity(ps, best["threshold"])-per_entity(pb, old_report["threshold"])
    rng = np.random.default_rng(491)
    boot = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)]
    result["development_audit_delta"] = {"mean": float(delta.mean()), "paired_bootstrap_95_percent": np.quantile(boot, [.025, .975]).tolist()}
    np.save(output / "probabilities.npy", p)
    prepared = cache["prepared"]
    write_json(model_dir / "manifest.json", {"model_files": best["model_files"], "threshold": best["threshold"],
               "feature_names": FEATURE_NAMES, "feature_schema": SCHEMA, "normalization_version": VERSION,
               "top_k": prepared["config"]["top_k"], "block_cap": prepared["config"]["block_cap"], "threads": threads,
               "training": "fit partition only; tune selects models and threshold; original audit is development diagnostic"})
    write_json(output / "experiment_report.json", result)
    write_json(output / "status.json", {"state": "complete", "selected": best,
               "development_audit": result["metrics"]["audit"], "fresh_holdout_confirmed": False})
    log(f"Selected {best['name']}; tune={best['tune_macro_f0_5']:.6f}; development audit={result['metrics']['audit']['macro_f0_5']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="artifacts/experiment_v2")
    parser.add_argument("--output", default="artifacts/experiment_v3")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=1600)
    args = parser.parse_args()
    try:
        run(args.source, args.output, args.threads, args.rounds)
    except Exception as exc:
        write_json(Path(args.output) / "status.json", {"state": "failed", "error": repr(exc)})
        raise
