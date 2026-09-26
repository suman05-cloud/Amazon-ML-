"""Versioned, label-free entity features. Baseline normalization stays unchanged."""
from functools import lru_cache
import re

import numpy as np
from rapidfuzz import fuzz

from .text import FEATURE_NAMES as BASE_NAMES, BORING_ADDRESS, features

SCHEMA = "enhanced_v1"
EXTRA_NAMES = [
    "digits_jaccard", "digits_left_coverage", "digits_right_coverage", "digits_disjoint",
    "digits_first_equal", "digits_last_equal", "digits_sequence_ratio", "digits_count_difference",
    "digits_first_edit", "digits_first_conflict", "digits_left_missing", "digits_right_missing",
    "alnum_jaccard", "alnum_conflict", "address_alpha_sorted", "address_alpha_set",
    "address_content_sorted", "address_content_set", "address_compact_ratio",
    "address_prefix_ratio", "address_suffix_ratio", "address_last_word_equal",
    "address_last_two_sorted", "name_compact_ratio", "name_compact_partial",
    "name_consonant_ratio", "name_consonant_sorted", "name_acronym_ratio",
    "name_acronym_expansion", "name_first_word_ratio", "name_last_word_ratio",
    "name_tokens_left", "name_tokens_right", "address_tokens_left", "address_tokens_right",
    "name_fuzzy_left_mean", "name_fuzzy_right_mean", "name_fuzzy_left_min",
    "name_fuzzy_right_min", "name_fuzzy_left_coverage", "name_fuzzy_right_coverage",
    "name_fuzzy_left_weighted", "name_fuzzy_right_weighted",
    "name_bigram_dice", "name_trigram_dice", "address_bigram_dice", "address_trigram_dice",
    "address_numeric_tokens_ratio", "address_leading_unit_ratio", "name_digit_conflict",
    "name_digit_equal", "name_partial_sorted", "address_partial_sorted",
]
CONTEXT_NAMES = [f"context_{field}_{metric}" for field in ("name", "address", "joint")
                 for metric in ("gap", "fraction_better", "near_best", "max")]
FEATURE_NAMES = BASE_NAMES + EXTRA_NAMES + CONTEXT_NAMES


def ratio(a, b):
    return fuzz.ratio(a, b) / 100 if a and b else 0.


def dice(a, b):
    return 2 * len(a & b) / max(1, len(a) + len(b))


@lru_cache(maxsize=8192)
def parts(core, address):
    words, aw = tuple(core.split()), tuple(address.split())
    # Unlike word-boundary numbers, this includes 1221ansal and sf1873a4f.
    nums = tuple(str(int(x)) for x in re.findall(r"\d+", address))
    alnum = frozenset(x for x in aw if re.search(r"\d", x) and re.search("[a-z]", x))
    alpha = " ".join(re.findall("[a-z]+", address))
    content = " ".join(x for x in aw if x not in BORING_ADDRESS)
    compact = core.replace(" ", "")
    consonants = re.sub("[aeiou]", "", core)
    acronym = "".join(x[0] for x in words)
    def grams(s, n):
        return frozenset(s[i:i+n] for i in range(max(0, len(s)-n+1)))
    return (words, aw, nums, frozenset(nums), alnum, alpha, content, compact,
            consonants, acronym, grams(compact, 2), grams(compact, 3),
            grams(address, 2), grams(address, 3),
            " ".join(x for x in aw if re.search(r"\d", x)),
            frozenset(re.findall(r"\d+", core)))


def fuzzy_words(a, b):
    if not a or not b:
        return [0.] * 8
    # Native ratio is cheap; word assignment is deliberately not label-derived.
    scores = [[fuzz.ratio(x, y) / 100 for y in b] for x in a]
    left = [max(x) for x in scores]
    right = [max(x) for x in zip(*scores)]
    return [sum(left)/len(left), sum(right)/len(right), min(left), min(right),
            sum(x >= .8 for x in left)/len(left), sum(x >= .8 for x in right)/len(right),
            sum(s*len(w) for s, w in zip(left, a))/sum(map(len, a)),
            sum(s*len(w) for s, w in zip(right, b))/sum(map(len, b))]


def extra_features(a, b):
    ap, bp = parts(a.core, a.address), parts(b.core, b.address)
    an, bn, ax, bx = ap[2], bp[2], ap[3], bp[3]
    common = len(ax & bx)
    aw, bw = ap[1], bp[1]
    nc = [common/max(1, len(ax | bx)), common/max(1, len(ax)), common/max(1, len(bx)),
          float(bool(ax and bx and not common)), float(bool(an and bn and an[0] == bn[0])),
          float(bool(an and bn and an[-1] == bn[-1])), ratio(" ".join(an), " ".join(bn)),
          abs(len(an)-len(bn)), ratio(an[0], bn[0]) if an and bn else 0.,
          float(bool(an and bn and an[0] != bn[0])), float(not an), float(not bn),
          len(ap[4] & bp[4])/max(1, len(ap[4] | bp[4])), float(bool(ap[4] and bp[4] and not ap[4] & bp[4]))]
    def sorted_ratio(x, y):
        return fuzz.token_sort_ratio(x, y)/100 if x and y else 0.
    def set_ratio(x, y):
        return fuzz.token_set_ratio(x, y)/100 if x and y else 0.
    nc += [sorted_ratio(ap[5], bp[5]), set_ratio(ap[5], bp[5]),
           sorted_ratio(ap[6], bp[6]), set_ratio(ap[6], bp[6]),
           ratio(a.address.replace(" ", ""), b.address.replace(" ", "")),
           ratio(" ".join(aw[:3]), " ".join(bw[:3])), ratio(" ".join(aw[-3:]), " ".join(bw[-3:])),
           float(bool(aw and bw and aw[-1] == bw[-1])), sorted_ratio(" ".join(aw[-2:]), " ".join(bw[-2:])),
           ratio(ap[7], bp[7]), fuzz.partial_ratio(ap[7], bp[7])/100 if ap[7] and bp[7] else 0.,
           ratio(ap[8], bp[8]), sorted_ratio(ap[8], bp[8]), ratio(ap[9], bp[9]),
           max(ratio(ap[9], bp[7]), ratio(bp[9], ap[7])),
           ratio(ap[0][0], bp[0][0]) if ap[0] and bp[0] else 0.,
           ratio(ap[0][-1], bp[0][-1]) if ap[0] and bp[0] else 0.,
           len(ap[0]), len(bp[0]), len(aw), len(bw)]
    nc += fuzzy_words(ap[0], bp[0])
    nc += [dice(ap[i], bp[i]) for i in (10, 11, 12, 13)]
    nc += [ratio(ap[14], bp[14]), ratio(aw[0], bw[0]) if aw and bw else 0.,
           float(bool(ap[15] and bp[15] and not ap[15] & bp[15])), float(bool(ap[15] & bp[15])),
           fuzz.partial_ratio(" ".join(sorted(ap[0])), " ".join(sorted(bp[0])))/100 if ap[0] and bp[0] else 0.,
           fuzz.partial_ratio(" ".join(sorted(aw)), " ".join(sorted(bw)))/100 if aw and bw else 0.]
    return nc


def feature_matrix(reference, ranked, base=None):
    """Same function in cached training and live inference; no IDs or labels."""
    if not ranked:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    if base is None:
        best = ranked[0][1]
        base = np.asarray([features(reference, t, s, best-s, len(ranked)) for t, s in ranked], dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    extra = np.asarray([extra_features(reference, t) for t, _ in ranked], dtype=np.float32)
    name = np.maximum(base[:, 6], base[:, 7])
    addr = base[:, 13]
    context = []
    for score in (name, addr, name * addr):
        best = score.max()
        # Equal-score duplicate target records must not outrank one another.
        better = (len(score)-np.searchsorted(np.sort(score), score, side="right"))/len(score)
        context += [best-score, better, np.full(len(score), np.mean(score >= best-.03)), np.full(len(score), best)]
    return np.column_stack([base, extra, *context]).astype(np.float32, copy=False)
