"""Streaming strict validation without keeping all candidate lists in RAM."""
import csv
from pathlib import Path
import sqlite3
import tempfile

from .data import fingerprint, log, rows, write_json


def validate(data_dir, output_dir):
    output, test = Path(output_dir), Path(data_dir) / "test"
    (output / "validation.json").unlink(missing_ok=True)
    # Temporary on-disk ID tables keep validation memory independent of corpus size.
    with tempfile.TemporaryDirectory(prefix="validate-", dir=output) as temp:
        db = sqlite3.connect(str(Path(temp) / "ids.sqlite"))
        try:
            db.execute("PRAGMA cache_size=-32768")
            db.executescript("CREATE TABLE required (id TEXT PRIMARY KEY, seen INTEGER DEFAULT 0) WITHOUT ROWID; CREATE TABLE targets (id TEXT PRIMARY KEY) WITHOUT ROWID;")
            for source in (1, 2, 3):
                table = "required" if source == 1 else "targets"
                batch = []
                for row in rows(test / f"test_source{source}.tsv"):
                    batch.append((row["entity_id"],))
                    if len(batch) == 50000:
                        db.executemany(f"INSERT INTO {table}(id) VALUES (?)", batch)
                        db.commit(); batch.clear()
                if batch:
                    db.executemany(f"INSERT INTO {table}(id) VALUES (?)", batch)
                    db.commit()
            log("Validation ID index ready")
            with (output / "matching_results.tsv").open(encoding="utf-8", newline="") as mf, (output / "candidate_pairs.tsv").open(encoding="utf-8", newline="") as cf:
                matching, candidates = csv.reader(mf, delimiter="\t"), csv.reader(cf, delimiter="\t")
                if next(matching) != ["source1_entity_id", "matched_entity_ids"] or next(candidates) != ["source1_entity_id", "candidate_entity_ids"]:
                    raise ValueError("Invalid output headers")
                count = 0
                for m in matching:
                    c = next(candidates, None)
                    if len(m) != 2 or c is None or len(c) != 2 or c[0] != m[0]:
                        raise ValueError("Outputs must have identical row order and two columns")
                    if db.execute("UPDATE required SET seen=1 WHERE id=? AND seen=0", (m[0],)).rowcount != 1:
                        raise ValueError(f"Duplicate or unknown reference: {m[0]}")
                    mids = m[1].split(",") if m[1] else []
                    cids = c[1].split(",") if c[1] else []
                    if len(set(mids)) != len(mids) or len(set(cids)) != len(cids):
                        raise ValueError(f"Duplicate ID in row: {m[0]}")
                    if not set(mids) <= set(cids):
                        raise ValueError(f"Matches outside candidate set: {m[0]}")
                    for begin in range(0, len(cids), 500):
                        part = cids[begin:begin + 500]
                        if any(not i.startswith(("S2-", "S3-")) for i in part):
                            raise ValueError("Invalid target ID prefix")
                        found = db.execute("SELECT COUNT(*) FROM targets WHERE id IN (" + ",".join("?" for _ in part) + ")", part).fetchone()[0]
                        if found != len(part):
                            raise ValueError(f"Unknown target in row: {m[0]}")
                    count += 1
                    if count % 50000 == 0:
                        db.commit()
                        log(f"Validated {count:,} reference rows")
                if next(candidates, None) is not None:
                    raise ValueError("Extra candidate rows")
            missing = db.execute("SELECT COUNT(*) FROM required WHERE seen=0").fetchone()[0]
            if missing:
                raise ValueError(f"Missing {missing} reference rows")
            log(f"PASS: {count:,} rows; all IDs valid; every match is a scored candidate")
            write_json(output / "validation.json", {"status": "PASS", "rows": count, "checked_target_ids": True,
                       "checked_candidate_subset": True, "files": fingerprint([output / "matching_results.tsv", output / "candidate_pairs.tsv"])})
            return count
        finally:
            db.close()
