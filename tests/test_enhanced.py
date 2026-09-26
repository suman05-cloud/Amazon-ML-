"""Regression checks for numeric conflicts and training/inference parity."""
import unittest

import numpy as np

from amazon_ml.enhanced import FEATURE_NAMES, EXTRA_NAMES, extra_features, feature_matrix
from amazon_ml.text import Record, cheap_score, features


def rec(i, name, address):
    return Record.from_row(dict(entity_id=i, business_name=name, business_address=address, country="India"))


class EnhancedTests(unittest.TestCase):
    def test_packed_retrieval_preserves_global_block_caps(self):
        from amazon_ml.blocking import FocusedIndex
        from amazon_ml.retrieval_study import PackedIndex
        ref = rec("r", "Common Market", "18 Main Road")
        targets = [rec(str(i), "Common Market", f"{i+18} Main Road") for i in range(7)]
        packed = PackedIndex([ref], 8)
        for t in targets:
            packed.add(t)
        for cap in (1, 3, 8):
            original = FocusedIndex([ref], cap)
            for t in targets:
                original.add(t)
            a = [(r.id, s) for r, s in original.query(ref, 8)]
            b = [(r.id, s) for r, s in packed.query_cap(ref, cap, 8)]
            self.assertEqual(a, b)

    def test_embedded_number_conflict(self):
        a = rec("a", "Example", "1221ansal sf1873a4f")
        b = rec("b", "Example", "1222ansal sf1873a4f")
        f = dict(zip(EXTRA_NAMES, extra_features(a, b)))
        self.assertEqual(f["digits_first_conflict"], 1)
        self.assertLess(f["digits_jaccard"], 1)
        self.assertEqual(f["address_alpha_sorted"], 1)

    def test_cache_and_live_parity_and_candidate_order(self):
        a = rec("a", "Alpha Marketing", "18 Main Road")
        targets = [rec("b", "Alpha Marketing", "18 Main Road"),
                   rec("c", "Alfa Marketing", "18 Main Road"),
                   rec("d", "Alpha Marketing", "19 Main Road")]
        ranked = sorted([(r, cheap_score(a, r)) for r in targets], key=lambda p: -p[1])
        base = np.asarray([features(a, r, s, ranked[0][1]-s, len(ranked)) for r, s in ranked], dtype=np.float32)
        live = feature_matrix(a, ranked)
        cached = feature_matrix(a, [(r, 0.) for r, _ in ranked], base)
        np.testing.assert_array_equal(live, cached)
        shuffled = feature_matrix(a, list(reversed(ranked)), base[::-1])
        np.testing.assert_array_equal(live, shuffled[::-1])
        self.assertEqual(live.shape, (3, len(FEATURE_NAMES)))

    def test_missing_is_not_matching_evidence(self):
        a = rec("a", "", "")
        b = rec("b", "", "")
        f = dict(zip(EXTRA_NAMES, extra_features(a, b)))
        self.assertEqual(f["address_alpha_sorted"], 0)
        self.assertEqual(f["digits_first_equal"], 0)
        self.assertTrue(np.isfinite(feature_matrix(a, [(b, 0.)])).all())
        self.assertEqual(feature_matrix(a, []).shape, (0, len(FEATURE_NAMES)))


if __name__ == "__main__":
    unittest.main()
