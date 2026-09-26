"""Batched inference, restartable chunks, exact candidate audit trail."""
import csv
import hashlib
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np

from .blocking import DiskIndex
from .data import fingerprint, log, records, write_json
from .models import load_models
from .text import FEATURE_NAMES, VERSION, features


def predict(data_dir, index_path, model_dir, output_dir, batch_size=500, shard=0, shards=1, limit=None, threads=4):
    output = Path(output_dir)
    if not 0 <= shard < shards or batch_size < 1:
        raise ValueError("Invalid shard or batch size")
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    schema = manifest.get("feature_schema", "baseline")
    enhanced_matrix = None
    expected_names = FEATURE_NAMES
    if schema != "baseline":
        from .enhanced import SCHEMA, FEATURE_NAMES as enhanced_names, feature_matrix
        if schema != SCHEMA:
            raise ValueError(f"Unknown feature schema: {schema}")
        expected_names, enhanced_matrix = enhanced_names, feature_matrix
    if manifest["feature_names"] != expected_names or manifest["normalization_version"] != VERSION:
        raise ValueError("Model and feature code versions do not match")
    models = load_models(model_dir, manifest)
    index = DiskIndex(index_path, manifest["block_cap"])
    source = Path(data_dir) / "test" / "test_source1.tsv"
    target_paths = [Path(data_dir) / "test" / f"test_source{i}.tsv" for i in (2, 3)]
    if json.loads(index.meta("inputs")) != fingerprint(target_paths):
        raise ValueError("Index was built from different target files")
    chunks = output / "chunks" / f"shard-{shard:03d}"
    chunks.mkdir(parents=True, exist_ok=True)
    signature = {"source": fingerprint([source]), "index_inputs": index.meta("inputs"), "manifest": manifest,
                 "model_hashes": [hashlib.sha256((model_dir / p).read_bytes()).hexdigest() for p in manifest["model_files"]],
                 "batch_size": batch_size, "shard": shard, "shards": shards, "limit": limit}
    state = chunks / "config.json"
    if state.exists() and json.loads(state.read_text(encoding="utf-8")) != signature:
        raise ValueError("Inference settings changed; choose a fresh output directory")
    write_json(state, signature)
    batch, count, batch_number = [], 0, 0
    started = time.monotonic()
    def flush():
        nonlocal batch_number
        path = chunks / f"{batch_number:07d}.jsonl"
        batch_number += 1
        if path.exists():
            batch.clear()
            return
        candidates, matrix, boundaries = [], [], [0]
        for r in batch:
            ranked = index.query(r, manifest["top_k"])
            candidates.append([t.id for t, _ in ranked])
            best = ranked[0][1] if ranked else 0.
            if enhanced_matrix is None:
                matrix.extend(features(r, t, s, best - s, len(ranked)) for t, s in ranked)
            else:
                matrix.extend(enhanced_matrix(r, ranked))
            boundaries.append(len(matrix))
        probabilities = np.zeros(len(matrix))
        if matrix:
            X = np.asarray(matrix, dtype=np.float32)
            for model in models:
                probabilities += model.predict(X, num_threads=threads) / len(models)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for i, r in enumerate(batch):
                selected = sorted(t for t, p in zip(candidates[i], probabilities[boundaries[i]:boundaries[i + 1]]) if p >= manifest["threshold"])
                f.write(json.dumps([r.id, selected, sorted(candidates[i])]) + "\n")
        tmp.replace(path)
        batch.clear()
        log(f"Shard {shard}: {count:,} references processed in {time.monotonic() - started:.0f}s")
    try:
        for i, r in enumerate(records(source)):
            if i % shards != shard:
                continue
            if limit is not None and count >= limit:
                break
            batch.append(r)
            count += 1
            if len(batch) == batch_size:
                flush()
        if batch:
            flush()
        write_json(chunks / "complete.json", {"rows": count, "batches": batch_number, "partial": limit is not None})
    finally:
        index.close()
    if shards == 1:
        merge(output, shards)


def merge(output_dir, shards=1):
    output = Path(output_dir)
    configurations = []
    completion = []
    for shard in range(shards):
        path = output / "chunks" / f"shard-{shard:03d}"
        configurations.append(json.loads((path / "config.json").read_text(encoding="utf-8")))
        completion.append(json.loads((path / "complete.json").read_text(encoding="utf-8")))
    base = {k: v for k, v in configurations[0].items() if k != "shard"}
    for c in configurations:
        if {k: v for k, v in c.items() if k != "shard"} != base or c["shards"] != shards:
            raise ValueError("Cannot merge shards with different model/input settings")
    partial = any(c["partial"] for c in completion)
    if not partial:
        (output / "validation.json").unlink(missing_ok=True)
    suffix = ".preview.tsv" if partial else ".tsv"
    matching = output / ("matching_results" + suffix)
    candidate = output / ("candidate_pairs" + suffix)
    with matching.with_suffix(".tmp").open("w", encoding="utf-8", newline="") as mf, candidate.with_suffix(".tmp").open("w", encoding="utf-8", newline="") as cf:
        mw, cw = csv.writer(mf, delimiter="\t", lineterminator="\n"), csv.writer(cf, delimiter="\t", lineterminator="\n")
        mw.writerow(["source1_entity_id", "matched_entity_ids"])
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])
        for shard in range(shards):
            root = output / "chunks" / f"shard-{shard:03d}"
            for i in range(completion[shard]["batches"]):
                with (root / f"{i:07d}.jsonl").open(encoding="utf-8") as f:
                    for line in f:
                        sid, matches, candidates = json.loads(line)
                        mw.writerow([sid, ",".join(matches)])
                        cw.writerow([sid, ",".join(candidates)])
    matching.with_suffix(".tmp").replace(matching)
    candidate.with_suffix(".tmp").replace(candidate)
    write_json(output / "prediction_summary.json", {"rows": sum(c["rows"] for c in completion), "partial": partial, "shards": shards})
    log(f"Wrote {matching} and {candidate}")
