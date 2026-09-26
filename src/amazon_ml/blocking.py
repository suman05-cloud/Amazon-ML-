"""Bounded memory retrieval over the complete target corpus."""

from collections import OrderedDict
import heapq
import json
from pathlib import Path
import sqlite3

import numpy as np
from rapidfuzz import fuzz, process

from .data import fingerprint, log, records
from .text import VERSION, Record, blocking_keys, cheap_score


def rank_candidates(reference, targets, top_k):
    targets = list(targets)
    if not targets:
        return []
    # RapidFuzz caches query-side tokenization in its native batch scorer. Float64
    # preserves the scalar formula and deterministic ID tie-breaking.
    names = [r.core for r in targets]
    addresses = [r.address for r in targets]
    ratio = process.cdist([reference.core], names, scorer=fuzz.ratio, dtype=np.float64)[0]
    sorted_ratio = process.cdist([reference.core], names, scorer=fuzz.token_sort_ratio, dtype=np.float64)[0]
    name = np.maximum(ratio, sorted_ratio) / 100
    both_addresses = np.array([bool(reference.address and r.address) for r in targets])
    addr = process.cdist([reference.address], addresses, scorer=fuzz.token_sort_ratio, dtype=np.float64)[0] / 100
    addr[~both_addresses] = 0.
    scores = np.maximum.reduce([.62 * name + .38 * addr,
                                np.where(both_addresses, .70, .92) * name,
                                .82 * addr + .18 * name])
    ranked = heapq.nlargest(top_k, ((float(s), r.id, r) for s, r in zip(scores, targets)), key=lambda x: (x[0], x[1]))
    return [(r, score) for score, _, r in ranked]


class FocusedIndex:
    """Full-pool block results for a bounded set of reference queries.

    Blocks exceeding the cap are disabled globally, never truncated. Ref-count
    eviction frees targets no longer reachable. This class never reads labels.
    """
    def __init__(self, refs, block_cap):
        self.cap = block_cap
        self.postings = {k: [] for r in refs for k in blocking_keys(r)}
        self.targets = {}
        self.refcounts = {}
        self.scanned = 0

    def add(self, r):
        keys = blocking_keys(r) & self.postings.keys()
        live = []
        for k in keys:
            posting = self.postings[k]
            if posting is None:
                continue
            if len(posting) == self.cap:
                for rid in posting:
                    self.refcounts[rid] -= 1
                    if self.refcounts[rid] == 0:
                        del self.refcounts[rid]
                        del self.targets[rid]
                self.postings[k] = None
            else:
                live.append(k)
        if live:
            self.targets[r.id] = r
            self.refcounts[r.id] = self.refcounts.get(r.id, 0) + len(live)
            for k in live:
                self.postings[k].append(r.id)
        self.scanned += 1

    def query(self, r, top_k):
        ids = set()
        for k in blocking_keys(r):
            ids.update(self.postings.get(k) or ())
        return rank_candidates(r, (self.targets[i] for i in ids), top_k)

    def build(self, paths):
        for path in paths:
            for r in records(path):
                self.add(r)
                if self.scanned % 250000 == 0:
                    log(f"Scanned {self.scanned:,} targets; retaining {len(self.targets):,}")


class DiskIndex:
    """SQLite disk index. Committed input batches resume after interruption."""
    def __init__(self, path, block_cap=400, writable=False):
        self.path = Path(path)
        if not writable and not self.path.exists():
            raise FileNotFoundError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.execute("PRAGMA cache_size=-65536")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("PRAGMA threads=4")
        self.cap = block_cap
        self.cache = OrderedDict()
        self.record_cache = OrderedDict()
        if writable:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript("CREATE TABLE IF NOT EXISTS records (rid INTEGER PRIMARY KEY, id TEXT, country TEXT, name TEXT, core TEXT, address TEXT); CREATE TABLE IF NOT EXISTS blocks (key INTEGER, rid INTEGER); CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);")
        else:
            if self.meta("complete") != "yes":
                raise ValueError("Index is incomplete; resume build-index first")
            if self.meta("version") != str(VERSION):
                raise ValueError("Normalization changed; build a new index")

    def meta(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def build(self, paths):
        signature = json.dumps(fingerprint(paths), sort_keys=True)
        if self.meta("inputs") not in (None, signature) or self.meta("version") not in (None, str(VERSION)):
            raise ValueError("Index inputs or normalization changed; choose a new index path")
        if self.meta("complete") == "yes":
            log("Complete index already exists")
            return
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('inputs',?)", (signature,))
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('version',?)", (str(VERSION),))
        self.db.commit()
        completed = int(self.meta("rows") or 0)
        from .data import rows
        count = 0
        pending_records, pending_blocks = [], []
        def flush():
            with self.db:
                self.db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)", pending_records)
                self.db.executemany("INSERT INTO blocks VALUES (?,?)", pending_blocks)
                self.db.execute("INSERT OR REPLACE INTO meta VALUES ('rows',?)", (str(count),))
            pending_records.clear()
            pending_blocks.clear()
        for path in paths:
            for row in rows(path):
                count += 1
                if count <= completed:
                    continue
                r = Record.from_row(row)
                pending_records.append((count, r.id, r.country, r.name, r.core, r.address))
                pending_blocks.extend((k, count) for k in blocking_keys(r))
                if len(pending_records) >= 10000:
                    flush()
                    if count % 100000 == 0:
                        log(f"Indexed {count:,} target records")
        if pending_records:
            flush()
        log("Creating disk lookup index; allow additional time and disk space")
        self.db.execute("CREATE INDEX IF NOT EXISTS blocks_lookup ON blocks(key,rid)")
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('complete','yes')")
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        log(f"Disk index complete: {count:,} targets")

    def posting(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        ids = [r[0] for r in self.db.execute("SELECT rid FROM blocks WHERE key=? LIMIT ?", (key, self.cap + 1))]
        if len(ids) > self.cap:
            ids = []
        self.cache[key] = ids
        if len(self.cache) > 20000:
            self.cache.popitem(last=False)
        return ids

    def query(self, r, top_k):
        ids = set()
        for key in blocking_keys(r):
            ids.update(self.posting(key))
        targets = []
        missing = []
        for rid in sorted(ids):
            if rid in self.record_cache:
                targets.append(self.record_cache[rid])
                self.record_cache.move_to_end(rid)
            else:
                missing.append(rid)
        for start in range(0, len(missing), 500):
            chunk = missing[start:start + 500]
            for row in self.db.execute("SELECT rid,id,country,name,core,address FROM records WHERE rid IN (" + ",".join("?" for _ in chunk) + ")", chunk):
                record = Record(*row[1:])
                targets.append(record)
                self.record_cache[row[0]] = record
                if len(self.record_cache) > 20000:
                    self.record_cache.popitem(last=False)
        return rank_candidates(r, targets, top_k)

    def close(self):
        self.db.close()
