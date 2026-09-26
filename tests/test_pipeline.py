import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from amazon_ml.blocking import DiskIndex, FocusedIndex, rank_candidates
from amazon_ml.data import partition
from amazon_ml.metrics import evaluate, tune_threshold
from amazon_ml.text import FEATURE_NAMES, Record, cheap_score, features, normalize


def rec(i, name, address, country="France"):
    return Record.from_row(dict(entity_id=i, business_name=name, business_address=address, country=country))


class MetricTests(unittest.TestCase):
    def test_official_example(self):
        result = evaluate([2], [0, 0, 0], [1, 0, 1], [1, 1, 1], .5)
        self.assertAlmostEqual(result["macro_f0_5"], 5 / 7)

    def test_singletons_and_unretrieved_truth_are_counted(self):
        # Scores are 1 (correct empty), 0 (false merge), 5/6 (one of two), 0 (missed).
        result = evaluate([0, 0, 2, 1], [1, 2], [0, 1], [1, 1], .5)
        self.assertAlmostEqual(result["macro_f0_5"], (1 + 5 / 6) / 4)
        self.assertEqual(result["false_negatives"], 2)

    def test_no_candidates(self):
        self.assertEqual(evaluate([0, 1], [], [], [], .5)["macro_f0_5"], .5)

    def test_macro_not_micro_threshold(self):
        t, score = tune_threshold([0, 1], [0, 1], [0, 1], [.6, .9])
        self.assertGreater(t, .6)
        self.assertLessEqual(t, .9)
        self.assertEqual(score, 1.)


class RetrievalTests(unittest.TestCase):
    def test_batched_ranker_matches_scalar_formula(self):
        a = rec("S1-a", "Green Solutions Ltd", "18 Market Street")
        choices = [rec("S2-a", "Solutions Green", "18 Market St"),
                   rec("S2-b", "Green Solution", ""), rec("S3-a", "Blue Shop", "18 Market Street")]
        for ref in (a, rec("S1-b", "Green", "")):
            expected = sorted([(r.id, cheap_score(ref, r)) for r in choices], key=lambda x: (x[1], x[0]), reverse=True)
            actual = [(r.id, s) for r, s in rank_candidates(ref, choices, 3)]
            self.assertEqual([r for r, _ in expected], [r for r, _ in actual])
            np.testing.assert_allclose([s for _, s in expected], [s for _, s in actual], rtol=0, atol=1e-14)

    def test_unseen_country_and_missing_address(self):
        a = rec("S1-a", "École Amicale SARL", "")
        b = rec("S2-b", "Ecole Amicale", "")
        vector = features(a, b)
        self.assertEqual(len(vector), len(FEATURE_NAMES))
        self.assertTrue(np.isfinite(vector).all())
        self.assertEqual(vector[17], 0, "two missing addresses are not positive evidence")
        self.assertEqual(normalize("École"), "ecole")
        self.assertNotEqual(normalize("राम मार्केटिंग"), "")
        self.assertEqual(partition(a, 42), partition(b, 42))

    def test_focused_index_equals_full_disk_index_and_resume(self):
        refs = [rec("S1-a", "Amicale", "10 Unique Road"), rec("S1-b", "New Shop", "88 Zebra Lane")]
        raw = [dict(entity_id=f"S2-{i}", business_name="Amicale", business_address=f"{10+i} Unique Road", country="France") for i in range(5)]
        raw += [dict(entity_id="S3-a", business_name="New Shop", business_address="88 Zebra Lane", country="France"),
                dict(entity_id="S3-b", business_name="Amicale", business_address="10 Unique Road", country="India")]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "targets.tsv"
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(raw[0]), delimiter="\t")
                w.writeheader(); w.writerows(raw)
            for cap in (1, 3, 10):
                focused = FocusedIndex(refs, cap)
                for row in raw:
                    focused.add(Record.from_row(row))
                db = DiskIndex(Path(tmp) / f"index{cap}.sqlite", cap, writable=True)
                db.build([p]); db.build([p]); db.close()
                db = DiskIndex(Path(tmp) / f"index{cap}.sqlite", cap)
                for r in refs:
                    self.assertEqual([(x.id, s) for x, s in focused.query(r, 40)], [(x.id, s) for x, s in db.query(r, 40)])
                    self.assertTrue(all(x.country == r.country for x, _ in db.query(r, 40)))
                db.close()


class EndToEndTests(unittest.TestCase):
    def test_train_predict_resume_shards_and_validate(self):
        from amazon_ml.train import prepare, fit
        from amazon_ml.predict import predict, merge
        from amazon_ml.validate import validate
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("train", "test"):
                folder = root / split
                folder.mkdir()
                for source in (1, 2, 3):
                    with (folder / f"{split}_source{source}.tsv").open("w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f, delimiter="\t")
                        w.writerow(["entity_id", "business_name", "business_address", "country"])
                        for i in range(80):
                            w.writerow([f"S{source}-{i}", f"Example{i} Services Ltd", f"{i+1} Cedar Road", "France" if split == "test" else "US"])
                if split == "train":
                    with (folder / "train_ground_truth.tsv").open("w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f, delimiter="\t")
                        w.writerow(["source1_entity_id", "matched_entity_ids"])
                        for i in range(80):
                            w.writerow([f"S1-{i}", f"S2-{i},S3-{i}"])
            work = root / "work"
            prepare(root, work, sample_size=80, top_k=8)
            fit(work, threads=1, seeds=(42,), rounds=30, transfer=False)
            index = root / "test.sqlite"
            db = DiskIndex(index, writable=True)
            db.build([root / "test" / f"test_source{i}.tsv" for i in (2, 3)])
            db.close()
            out = root / "output"
            predict(root, index, work / "model", out, batch_size=19, threads=1)
            before = (out / "matching_results.tsv").read_bytes()
            predict(root, index, work / "model", out, batch_size=19, threads=1)
            self.assertEqual(before, (out / "matching_results.tsv").read_bytes())
            self.assertEqual(validate(root, out), 80)
            candidates_path = out / "candidate_pairs.tsv"
            with candidates_path.open(encoding="utf-8", newline="") as f:
                candidate_rows = list(csv.reader(f, delimiter="\t"))
            candidate_rows[1][1] += ",S2-does-not-exist"
            with candidates_path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f, delimiter="\t").writerows(candidate_rows)
            with self.assertRaisesRegex(ValueError, "Unknown target"):
                validate(root, out)
            self.assertFalse((out / "validation.json").exists(), "A failed check must not retain an earlier PASS")
            for shard in (0, 1):
                predict(root, index, work / "model", root / "sharded", batch_size=13, threads=1, shard=shard, shards=2)
            merge(root / "sharded", 2)
            self.assertEqual(validate(root, root / "sharded"), 80)
            self.assertEqual(sorted(before.splitlines()[1:]), sorted((root / "sharded" / "matching_results.tsv").read_bytes().splitlines()[1:]))
            predict(root, index, work / "model", root / "preview", limit=3, threads=1)
            self.assertFalse((root / "preview" / "matching_results.tsv").exists())
            self.assertTrue((root / "preview" / "matching_results.preview.tsv").exists())
            from amazon_ml.runner import finish
            import json
            finish(root, index, work / "model", root / "automatic", workers=2, threads=1, batch_size=20)
            status = json.loads((root / "automatic" / "run_status.json").read_text())
            self.assertEqual(status["state"], "complete")
            self.assertTrue((root / "automatic" / "validation.json").exists())
            # The enhanced schema must be usable end-to-end without changing the
            # baseline model or rebuilding the retrieval index.
            from amazon_ml.improve import run as improve
            improved = root / "improved"
            improve(work, improved, threads=1, rounds=20)
            enhanced_out = root / "enhanced_output"
            predict(root, index, improved / "model", enhanced_out, batch_size=19, threads=1)
            enhanced_before = (enhanced_out / "matching_results.tsv").read_bytes()
            predict(root, index, improved / "model", enhanced_out, batch_size=19, threads=1)
            self.assertEqual(enhanced_before, (enhanced_out / "matching_results.tsv").read_bytes())
            self.assertEqual(validate(root, enhanced_out), 80)
            from amazon_ml.rescore import build_raw, run as rescore
            raw_index = root / "raw.sqlite"
            build_raw(root, raw_index)
            build_raw(root, raw_index)
            rescored = root / "rescored"
            rescore(root, raw_index, out, rescored, improved / "model", workers=2)
            self.assertEqual(sorted(enhanced_before.splitlines()[1:]),
                             sorted((rescored / "matching_results.tsv").read_bytes().splitlines()[1:]))
            rescore(root, raw_index, out, rescored, improved / "model", workers=2)
            self.assertEqual(validate(root, rescored), 80)


if __name__ == "__main__":
    unittest.main()
