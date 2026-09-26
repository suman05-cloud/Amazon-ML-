import csv
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from amazon_ml.enhanced import FEATURE_NAMES
from amazon_ml.neural_data import prepare
from amazon_ml.text import Record


class NeuralDataTests(unittest.TestCase):
    def test_eval_gate_does_not_use_labels_and_raw_unicode_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, teacher, confirm = (root/x for x in ("source", "teacher", "confirm"))
            for p in (source, teacher, confirm, root/"data"/"train"):
                p.mkdir(parents=True)
            raw = {}
            cases, entities = [], []
            for i, split in enumerate(("fit", "tune", "audit", "confirm")):
                for prefix in ("S1", "S2"):
                    rid = f"{prefix}-{i}"
                    raw[rid] = {"entity_id": rid, "business_name": "राम मार्केटिंग", "business_address": "१२ रोड", "country": "India"}
                a, b = (Record.from_row(raw[f"{s}-{i}"]) for s in ("S1", "S2"))
                case = {"reference": asdict(a), "truth": [b.id], "candidates": [asdict(b)]}
                if split == "confirm":
                    (confirm/"cases.jsonl").write_text(json.dumps(case)+"\n", encoding="utf-8")
                else:
                    cases.append(case)
                    entities.append({"id": a.id, "country": "india", "split": split, "truth_count": 1})
            (source/"cases.jsonl").write_text("".join(json.dumps(c)+"\n" for c in cases), encoding="utf-8")
            (source/"entities.json").write_text(json.dumps(entities), encoding="utf-8")
            np.save(teacher/"probabilities.npy", np.array([0., 0., .9]))
            np.save(teacher/"features.npy", np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32))
            np.savez(confirm/"predictions.npz", enhanced=np.array([0.]))
            for k in (1, 2, 3):
                with (root/"data"/"train"/f"train_source{k}.tsv").open("w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["entity_id", "business_name", "business_address", "country"], delimiter="\t")
                    w.writeheader()
                    w.writerows(v for key, v in raw.items() if key.startswith(f"S{k}-"))
            prepare(root/"data", root/"output", source, teacher, confirm)
            output = root/"output"
            self.assertEqual(np.load(output/"groups.npy").tolist(), [0, 2])
            info = json.loads((output/"entities.json").read_text(encoding="utf-8"))
            self.assertEqual([r["truth_count"] for r in info], [1, 1, 1, 1])
            text = (output/"pairs.jsonl").read_text(encoding="utf-8")
            self.assertIn("राम", text)
            self.assertEqual(len(text.splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
