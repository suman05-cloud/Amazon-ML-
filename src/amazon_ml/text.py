"""Country-independent text features; IDs are never model features."""

from dataclasses import dataclass
import hashlib
from itertools import combinations
import re

from anyascii import anyascii
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

VERSION = 2
LEGAL = set("inc incorporated llc llp ltd limited corp corporation co company pvt private plc sarl sas sasu sa sci eurl gmbh the and de du des la le les et".split())
ALIASES = dict(zip(
    "road street avenue boulevard drive lane court place highway apartment suite floor building sector nagar extension route saint".split(),
    "rd st ave blvd dr ln ct pl hwy apt ste fl bldg sec ngr ext rte st".split(),
))
BORING_ADDRESS = set("rd st ave blvd dr ln ct pl hwy apt ste fl bldg sec ngr ext rte near opposite opp behind beside post village district dist india us france".split())
TOKEN = re.compile(r"[a-z0-9]+")


def normalize(value: str) -> str:
    return " ".join(TOKEN.findall(anyascii(value).lower().replace("&", " and ")))


@dataclass(slots=True)
class Record:
    id: str
    country: str
    name: str
    core: str
    address: str

    @classmethod
    def from_row(cls, row):
        name = normalize(row["business_name"])
        core = " ".join(t for t in name.split() if t not in LEGAL)
        address = " ".join(ALIASES.get(t, t) for t in normalize(row["business_address"]).split())
        address = re.sub(r"\b0+(\d+)\b", lambda m: str(int(m[0])), address)
        return cls(row["entity_id"], normalize(row["country"]), name, core or name, address)


def key_hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "little", signed=True)


def blocking_keys(r: Record) -> set[int]:
    """Independent name/address views; oversized blocks are suppressed globally."""
    keys = set()
    def add(kind, value):
        if value:
            keys.add(key_hash(r.country + "|" + kind + "|" + value))
    words = sorted({w for w in r.core.split() if len(w) >= 2 and not w.isdigit()}, key=lambda w: (-len(w), w))[:4]
    address_words = sorted({w for w in r.address.split() if len(w) >= 4 and w not in BORING_ADDRESS and not w.isdigit()}, key=lambda w: (-len(w), w))[:3]
    numbers = re.findall(r"\b\d+[a-z]?\b", r.address)
    add("name", " ".join(sorted(set(r.core.split()))))
    add("compact", r.core.replace(" ", ""))
    add("address", " ".join(sorted(r.address.split())))
    for word in words:
        add("word", word)
        add("prefix", word[:4])
        consonants = re.sub("[aeiou]", "", word)
        if len(consonants) >= 3:
            add("sound", consonants)
    for word in address_words:
        add("addrword", word)
        for number in numbers[:1]:
            add("numberword", number + " " + word[:6])
    for number in numbers:
        if len(number) in (5, 6) and number.isdigit():
            add("postal", number)
    # Generic business words and repeated addresses overflow single-field blocks.
    # Conjunctions retain discriminatory evidence instead of raising a global cap.
    for a, b in combinations(sorted(words), 2):
        add("namepair", a + " " + b)
    for word in words[:3]:
        for aw in address_words:
            add("nameplace", word[:4] + " " + aw[:5])
        for number in sorted(set(numbers))[:3]:
            add("namenumber", word[:4] + " " + number)
    for a, b in combinations(sorted(address_words), 2):
        add("addresspair", a[:6] + " " + b[:6])
    # Short names have no spare token when a character is corrupt. Symmetric
    # deletion signatures can meet after one substitution, anchored to an address.
    compact = r.core.replace(" ", "")
    if 3 <= len(compact) <= 8:
        for number in sorted(set(numbers))[:2]:
            for i in range(len(compact)):
                add("shortname", compact[:i] + compact[i + 1:] + " " + number)
    return keys


def overlap(a, b):
    x, y = set(a.split()), set(b.split())
    common = len(x & y)
    return [common / max(1, len(x | y)), common / max(1, min(len(x), len(y))), common / max(1, len(x)), common / max(1, len(y))]


def similarities(a, b):
    if not a or not b:
        return [0.] * 6
    return [fuzz.ratio(a, b) / 100, fuzz.token_sort_ratio(a, b) / 100,
            fuzz.token_set_ratio(a, b) / 100, fuzz.partial_ratio(a, b) / 100,
            JaroWinkler.normalized_similarity(a, b), float(a == b)]


FEATURE_NAMES = [f"{field}_{metric}" for field in ("name", "core", "address")
                 for metric in ("ratio", "sorted", "set", "partial", "jaro", "exact")]
FEATURE_NAMES += [f"{field}_{metric}" for field in ("core", "address")
                  for metric in ("jaccard", "containment", "left_overlap", "right_overlap")]
FEATURE_NAMES += ["number_jaccard", "number_conflict", "first_number_equal", "postal_equal", "postal_conflict",
                  "left_address_missing", "right_address_missing", "name_length_ratio", "address_length_ratio",
                  "name_address_product", "name_address_min", "retrieval_score", "retrieval_gap", "candidate_count"]


def cheap_score(a: Record, b: Record) -> float:
    name = max(fuzz.ratio(a.core, b.core), fuzz.token_sort_ratio(a.core, b.core)) / 100
    addr = fuzz.token_sort_ratio(a.address, b.address) / 100 if a.address and b.address else 0
    return max(.62 * name + .38 * addr,
               (.92 if not a.address or not b.address else .70) * name,
               .82 * addr + .18 * name)


def features(a: Record, b: Record, score=0., gap=0., count=0):
    out = similarities(a.name, b.name) + similarities(a.core, b.core) + similarities(a.address, b.address)
    out += overlap(a.core, b.core) + overlap(a.address, b.address)
    an, bn = re.findall(r"\b\d+[a-z]?\b", a.address), re.findall(r"\b\d+[a-z]?\b", b.address)
    ax, bx = set(an), set(bn)
    ap, bp = {x for x in ax if x.isdigit() and len(x) in (5, 6)}, {x for x in bx if x.isdigit() and len(x) in (5, 6)}
    ns, ads = max(out[6:9]), max(out[12:15])
    out += [len(ax & bx) / max(1, len(ax | bx)), float(bool(ax and bx and not ax & bx)),
            float(bool(an and bn and an[0] == bn[0])), float(bool(ap & bp)), float(bool(ap and bp and not ap & bp)),
            float(not a.address), float(not b.address), min(len(a.core), len(b.core)) / max(1, len(a.core), len(b.core)),
            min(len(a.address), len(b.address)) / max(1, len(a.address), len(b.address)),
            ns * ads, min(ns, ads), score, gap, np.log1p(count)]
    return out
