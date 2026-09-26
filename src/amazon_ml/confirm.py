"""One-shot comparison on new reference businesses, using frozen models.

No labels from this holdout are used in training, model selection or thresholds.
Candidate retrieval still scans the complete training target pool.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .blocking import FocusedIndex
from .data import fingerprint, log, sample_references, selected_truth, write_json
from .enhanced import FEATURE_NAMES as ENHANCED_NAMES, SCHEMA, feature_matrix
from .metrics import evaluate
from .models import load_models
from .text import Record, FEATURE_NAMES, VERSION, features


def name_key(r):
    return r.country, " ".join(sorted(r.core.split()))


def load_model(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    schema = manifest.get("feature_schema", "baseline")
    if schema not in ("baseline", SCHEMA):
        raise ValueError("Unknown feature schema")
    expected = FEATURE_NAMES if schema == "baseline" else ENHANCED_NAMES
    if manifest["feature_names"] != expected or manifest["normalization_version"] != VERSION:
        raise ValueError("Incompatible model")
    models = load_models(path, manifest)
    hashes = {f: hashlib.sha256((path / f).read_bytes()).hexdigest() for f in manifest["model_files"]}
    return manifest, models, hashes


def run(data, output, baseline, candidate, exclude, sample_size=1500, seed=99173, threads=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bm, baseline_models, bh = load_model(baseline)
    cm, candidate_models, ch = load_model(candidate)
    if any(bm[k] != cm[k] for k in ("top_k", "block_cap")):
        raise ValueError("This paired comparison requires identical retrieval settings")
    base = Path(data) / "train"
    inputs = [base / f"train_source{i}.tsv" for i in (1, 2, 3)] + [base / "train_ground_truth.tsv"]
    lock = {"baseline_manifest": bm, "candidate_manifest": cm, "baseline_sha256": bh,
            "candidate_sha256": ch, "sample_size": sample_size, "seed": seed,
            "excluded_cases": fingerprint(exclude), "data": fingerprint(inputs)}
    lock_path = output / "frozen_config.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise ValueError("Frozen confirmation configuration changed; do not reuse this holdout")
    if (output / "report.json").exists():
        log("Confirmation already complete; no rerun needed")
        return
    write_json(lock_path, lock)
    write_json(output / "status.json", {"state": "sampling", "seed": seed})
    excluded_ids, excluded_names, excluded_targets = set(), set(), set()
    for path in exclude:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                case = json.loads(line)
                r = Record(**case["reference"])
                excluded_ids.add(r.id)
                excluded_names.add(name_key(r))
                truth = set(case["truth"])
                excluded_targets.update(truth)
                for c in case["candidates"]:
                    if c["id"] in truth:
                        excluded_names.add(name_key(Record(**c)))
    sampled = sample_references(inputs[0], sample_size * 2, seed)
    sampled = [r for r in sampled if r.id not in excluded_ids and name_key(r) not in excluded_names]
    truth = selected_truth(inputs[3], sampled)
    refs = [r for r in sampled if not (truth[r.id] & excluded_targets)][:sample_size]
    if len(refs) != sample_size:
        raise ValueError("Not enough disjoint holdout references")
    write_json(output / "holdout_ids.json", [r.id for r in refs])
    index = FocusedIndex(refs, bm["block_cap"])
    write_json(output / "status.json", {"state": "scanning_targets", "references": len(refs)})
    log(f"Fresh holdout: {len(refs):,} references disjoint from previous reference IDs, name groups and truth target IDs")
    index.build(inputs[1:3])
    groups, labels, oldp, newp = [], [], [], []
    write_json(output / "status.json", {"state": "scoring", "scanned_targets": index.scanned})
    with (output / "cases.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(refs):
            ranked = index.query(r, bm["top_k"])
            best = ranked[0][1] if ranked else 0.
            x = np.asarray([features(r, t, s, best-s, len(ranked)) for t, s in ranked], dtype=np.float32)
            if ranked:
                xe = feature_matrix(r, ranked, x)
                oldp.extend(np.mean([m.predict(x, num_threads=threads) for m in baseline_models], axis=0))
                newx = x if cm.get("feature_schema", "baseline") == "baseline" else xe
                newp.extend(np.mean([m.predict(newx, num_threads=threads) for m in candidate_models], axis=0))
                groups.extend([i]*len(ranked))
                labels.extend(int(t.id in truth[r.id]) for t, _ in ranked)
            f.write(json.dumps({"reference": asdict(r), "truth": sorted(truth[r.id]),
                                "candidates": [asdict(t) for t, _ in ranked]}, ensure_ascii=False)+"\n")
            if (i+1) % 100 == 0:
                log(f"Confirmed {i+1:,}/{len(refs):,} references")
    g, y = np.array(groups, dtype=int), np.array(labels, dtype=np.uint8)
    p0, p1 = np.array(oldp), np.array(newp)
    tc = np.array([len(truth[r.id]) for r in refs])
    countries = np.array([r.country for r in refs])
    report = {"references": len(refs), "scanned_targets": index.scanned,
              "candidate_recall": float(y.sum()/max(1, tc.sum())),
              "baseline": evaluate(tc, g, y, p0, bm["threshold"]),
              "enhanced": evaluate(tc, g, y, p1, cm["threshold"]),
              "thresholds": {"baseline": bm["threshold"], "enhanced": cm["threshold"]},
              "country_metrics": {}, "note": "Frozen thresholds/models; disjoint new businesses; no France labels available. This is local validation, not leaderboard score."}
    for c in sorted(set(countries)):
        ids = np.flatnonzero(countries == c)
        mapping = np.full(len(refs), -1, dtype=int)
        mapping[ids] = np.arange(len(ids))
        mask = mapping[g] >= 0
        report["country_metrics"][c] = {
            "baseline": evaluate(tc[ids], mapping[g[mask]], y[mask], p0[mask], bm["threshold"]),
            "enhanced": evaluate(tc[ids], mapping[g[mask]], y[mask], p1[mask], cm["threshold"])}
    def per_entity(p, threshold):
        take = p >= threshold
        n = np.bincount(g[take], minlength=len(refs))
        tp = np.bincount(g[take & (y == 1)], minlength=len(refs))
        den = .25*tc+n
        return np.divide(1.25*tp, den, out=np.ones(len(refs)), where=den != 0)
    delta = per_entity(p1, cm["threshold"])-per_entity(p0, bm["threshold"])
    rng = np.random.default_rng(918)
    boot = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)]
    report["paired_improvement"] = {"mean": float(delta.mean()), "bootstrap_95_percent": np.quantile(boot, [.025, .975]).tolist()}
    np.savez(output / "predictions.npz", truth_counts=tc, groups=g, labels=y, baseline=p0, enhanced=p1)
    write_json(output / "report.json", report)
    write_json(output / "status.json", {"state": "complete", "baseline_f0_5": report["baseline"]["macro_f0_5"],
               "enhanced_f0_5": report["enhanced"]["macro_f0_5"], "paired_improvement": report["paired_improvement"]})
    log(f"FRESH HOLDOUT baseline={report['baseline']['macro_f0_5']:.6f}, enhanced={report['enhanced']['macro_f0_5']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data set/student_resource/dataset")
    parser.add_argument("--output", default="artifacts/confirmation_v3")
    parser.add_argument("--baseline", default="artifacts/experiment_v2/model")
    parser.add_argument("--candidate", default="artifacts/experiment_v3/model")
    parser.add_argument("--exclude", nargs="+", default=["artifacts/experiment/cases.jsonl", "artifacts/experiment_v2/cases.jsonl"])
    parser.add_argument("--sample-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=99173)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        run(args.data, args.output, args.baseline, args.candidate, args.exclude, args.sample_size, args.seed, args.threads)
    except Exception as exc:
        write_json(Path(args.output) / "status.json", {"state": "failed", "error": repr(exc)})
        raise
