"""Compare broader retrieval on tuning businesses only, against all targets."""
from dataclasses import asdict
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .blocking import FocusedIndex, rank_candidates
from .data import log, write_json
from .enhanced import feature_matrix
from .metrics import evaluate, tune_threshold
from .text import Record, blocking_keys


class PackedIndex(FocusedIndex):
    """Compact retained text during the full-pool scan; identical block logic."""
    def add(self, record):
        super().add(record)
        if record.id in self.targets:
            self.targets[record.id] = "\t".join((record.country, record.name, record.core, record.address))

    def query_cap(self, reference, cap, top_k):
        ids = set()
        for key in blocking_keys(reference):
            posting = self.postings.get(key)
            if posting and len(posting) <= cap:
                ids.update(posting)
        return rank_candidates(reference, (Record(rid, *self.targets[rid].split("\t")) for rid in ids), top_k)


def run(data="data set/student_resource/dataset", output="artifacts/retrieval_study_v1"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path("artifacts/experiment_v2")
    meta = json.loads((source/"entities.json").read_text(encoding="utf-8"))
    refs, truths = [], []
    with (source/"cases.jsonl").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if meta[i]["split"] == "tune":
                case = json.loads(line)
                refs.append(Record(**case["reference"]))
                truths.append(set(case["truth"]))
    write_json(output/"status.json", {"state": "scanning", "references": len(refs), "maximum_block_cap": 800})
    index = PackedIndex(refs, 800)
    index.build([Path(data)/"train"/f"train_source{k}.tsv" for k in (2, 3)])
    model_dir = Path("artifacts/experiment_v3/model")
    manifest = json.loads((model_dir/"manifest.json").read_text(encoding="utf-8"))
    models = [lgb.Booster(model_file=str(model_dir/p)) for p in manifest["model_files"]]
    results = []
    truth_counts = np.array([len(t) for t in truths])
    for cap, top_k in ((400, 80), (400, 160), (800, 160), (800, 240)):
        write_json(output/"status.json", {"state": "evaluating", "block_cap": cap, "top_k": top_k})
        labels, groups, probabilities = [], [], []
        with (output/f"cases_{cap}_{top_k}.jsonl").open("w", encoding="utf-8") as f:
            for i, (r, truth) in enumerate(zip(refs, truths)):
                ranked = index.query_cap(r, cap, top_k)
                x = feature_matrix(r, ranked)
                if ranked:
                    probabilities.extend(np.mean([m.predict(x, num_threads=1) for m in models], axis=0))
                    labels.extend(int(c.id in truth) for c, _ in ranked)
                    groups.extend([i]*len(ranked))
                f.write(json.dumps({"reference": asdict(r), "truth": sorted(truth), "candidates": [asdict(c) for c, _ in ranked]}, ensure_ascii=False)+"\n")
        y, g, p = np.array(labels, dtype=np.uint8), np.array(groups, dtype=int), np.array(probabilities)
        threshold, score = tune_threshold(truth_counts, g, y, p)
        result = {"cap": cap, "top_k": top_k, "pairs": len(y), "candidate_recall": float(y.sum()/truth_counts.sum()),
                  "oracle_macro_f0_5": evaluate(truth_counts, g, y, y, .5)["macro_f0_5"],
                  "threshold": threshold, "metrics": evaluate(truth_counts, g, y, p, threshold)}
        results.append(result)
        np.savez(output/f"predictions_{cap}_{top_k}.npz", labels=y, groups=g, probabilities=p, truth_counts=truth_counts)
        write_json(output/"report.json", {"tuning_only": True, "scanned_targets": index.scanned, "results": results})
        log(f"Retrieval cap={cap}, top={top_k}: recall={result['candidate_recall']:.5f}, oracle={result['oracle_macro_f0_5']:.5f}, tune F0.5={score:.5f}")
    write_json(output/"status.json", {"state": "complete", "results": results})


if __name__ == "__main__":
    run()
