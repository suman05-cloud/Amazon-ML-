from collections import Counter
from pathlib import Path

from .data import log, rows, write_json


def profile(data_dir, destination):
    report = {}
    for path in sorted(Path(data_dir).rglob("*.tsv")):
        n = 0
        countries, missing, matches = Counter(), Counter(), Counter()
        for row in rows(path):
            n += 1
            if "country" in row:
                countries[row["country"]] += 1
            if "matched_entity_ids" in row:
                matches[len(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else 0] += 1
            missing.update(k for k, v in row.items() if not v)
        result = {"rows": n, "bytes": path.stat().st_size, "countries": dict(countries),
                  "missing": dict(missing), "match_count_distribution": dict(matches)}
        report[path.name] = result
        log(f"{path.name}: {n:,} rows")
    write_json(destination, report)
    return report
