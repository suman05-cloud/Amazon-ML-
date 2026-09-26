import csv
import hashlib
import json
from pathlib import Path
import random
import time

from .text import Record


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def records(path):
    for row in rows(path):
        yield Record.from_row(row)


def fingerprint(paths):
    return [{"path": str(Path(p).resolve()), "size": Path(p).stat().st_size,
             "mtime_ns": Path(p).stat().st_mtime_ns} for p in paths]


def sample_references(path, n, seed):
    rng = random.Random(seed)
    sample = []
    for i, row in enumerate(rows(path)):
        if i < n:
            sample.append(row)
        else:
            j = rng.randrange(i + 1)
            if j < n:
                sample[j] = row
    return [Record.from_row(r) for r in sample]


def selected_truth(path, refs):
    ids = {r.id for r in refs}
    truth, owner = {}, {}
    for row in rows(path):
        sid = row["source1_entity_id"]
        if sid in ids:
            links = set(filter(None, row["matched_entity_ids"].split(",")))
            truth[sid] = links
            for target in links:
                if target in owner and owner[target] != sid:
                    raise ValueError("A target has multiple reference owners; revise group splitting")
                owner[target] = sid
    if set(truth) != ids:
        raise ValueError("Missing training ground truth")
    return truth


def partition(record, seed):
    group = f"{seed}|{record.country}|{' '.join(sorted(record.core.split()))}"
    bucket = int.from_bytes(hashlib.blake2b(group.encode(), digest_size=8).digest(), "little") % 10
    return "fit" if bucket < 6 else "tune" if bucket < 8 else "audit"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
