"""Build raw multilingual pairs and hard negatives from supplied data only."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np

from .data import fingerprint, log, rows, write_json
from .enhanced import FEATURE_NAMES, feature_matrix
from .text import Record, cheap_score


def prepare(data, output, source="artifacts/experiment_v2", teacher="artifacts/experiment_v3", confirmation="artifacts/confirmation_v3", seed=4127, gate=.0001):
    output, source, teacher, confirmation = map(Path, (output, source, teacher, confirmation))
    output.mkdir(parents=True, exist_ok=True)
    config = {"version": 1, "seed": seed, "gate": gate, "feature_names": FEATURE_NAMES,
              "inputs": fingerprint([source/"cases.jsonl", source/"entities.json", teacher/"probabilities.npy", teacher/"features.npy",
                                      confirmation/"cases.jsonl", confirmation/"predictions.npz"])}
    marker = output/"prepared.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8"))["config"] != config:
            raise ValueError("Neural data configuration changed")
        return
    write_json(output/"status.json", {"state": "selecting_pairs"})
    metadata = json.loads((source/"entities.json").read_text(encoding="utf-8"))
    p = np.load(teacher/"probabilities.npy")
    base_x = np.load(teacher/"features.npy", mmap_mode="r")
    confirm_p = np.load(confirmation/"predictions.npz")["enhanced"]
    selected, feature_rows, entities, wanted = [], [], [], set()
    rng = np.random.default_rng(seed)
    for case_file, probs, is_confirm in ((source/"cases.jsonl", p, False), (confirmation/"cases.jsonl", confirm_p, True)):
        offset = 0
        with case_file.open(encoding="utf-8") as inp:
            for local_i, line in enumerate(inp):
                case = json.loads(line)
                ref = Record(**case["reference"])
                candidates = case["candidates"]
                truth = set(case["truth"])
                split = "confirm" if is_confirm else metadata[local_i]["split"]
                group = len(entities)
                entities.append({"id": ref.id, "country": ref.country, "split": split, "truth_count": len(truth)})
                values = probs[offset:offset+len(candidates)]
                labels = np.array([c["id"] in truth for c in candidates])
                if split == "fit":
                    positive = np.flatnonzero(labels)
                    negative = np.flatnonzero(~labels)
                    # Hard negatives are actual retrieved businesses, not random
                    # names from a reduced target pool. Easy negatives are mixed in.
                    hard = negative[np.argsort(-values[negative], kind="stable")[:8]]
                    rest = np.setdiff1d(negative, hard)
                    easy = rng.choice(rest, min(2, len(rest)), replace=False)
                    take = np.sort(np.concatenate([positive, hard, easy]))
                else:
                    # Evaluation gate uses teacher probabilities only, never labels.
                    take = np.flatnonzero(values >= gate)
                if is_confirm:
                    ranked = [(Record(**c), cheap_score(ref, Record(**c))) for c in candidates]
                    matrix = feature_matrix(ref, ranked)
                else:
                    matrix = base_x[offset:offset+len(candidates)]
                for j in take:
                    cid = candidates[j]["id"]
                    selected.append((ref.id, cid, group, int(labels[j]), float(values[j])))
                    feature_rows.append(matrix[j].copy())
                    wanted.update((ref.id, cid))
                offset += len(candidates)
        if offset != len(probs):
            raise ValueError("Case/probability alignment failed")
    write_json(output/"entities.json", entities)
    np.save(output/"numeric.npy", np.asarray(feature_rows, dtype=np.float32))
    del feature_rows, base_x
    log(f"Selected {len(selected):,} neural pairs; retrieving {len(wanted):,} original-language records")
    db = sqlite3.connect(output/"raw.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS raw (id TEXT PRIMARY KEY, name TEXT, address TEXT, country TEXT)")
    found = {r[0] for r in db.execute("SELECT id FROM raw")}
    wanted.difference_update(found)
    del found
    write_json(output/"status.json", {"state": "recovering_raw_text", "pairs": len(selected), "wanted_records": len(wanted)})
    for k in (1, 2, 3):
        batch = []
        for row in rows(Path(data)/"train"/f"train_source{k}.tsv"):
            if row["entity_id"] in wanted:
                batch.append((row["entity_id"], row["business_name"], row["business_address"], row["country"]))
                wanted.remove(row["entity_id"])
                if len(batch) >= 5000:
                    db.executemany("INSERT OR REPLACE INTO raw VALUES (?,?,?,?)", batch)
                    db.commit()
                    batch.clear()
            if not wanted:
                break
        db.executemany("INSERT OR REPLACE INTO raw VALUES (?,?,?,?)", batch)
        db.commit()
        log(f"Recovered source {k}; {len(wanted):,} records remaining")
    if wanted:
        raise ValueError(f"Missing {len(wanted)} original records")
    def text_for(rid):
        name, address, country = db.execute("SELECT name,address,country FROM raw WHERE id=?", (rid,)).fetchone()
        return f"name: {name}; address: {address}; country: {country}"
    with (output/"pairs.jsonl").open("w", encoding="utf-8") as out:
        for sid, tid, group, label, probability in selected:
            out.write(json.dumps({"left": text_for(sid), "right": text_for(tid), "group": group,
                                  "label": label, "teacher": probability}, ensure_ascii=False)+"\n")
    db.close()
    group_array = np.array([r[2] for r in selected], dtype=np.int32)
    np.save(output/"groups.npy", group_array)
    np.save(output/"labels.npy", np.array([r[3] for r in selected], dtype=np.uint8))
    np.save(output/"teacher.npy", np.array([r[4] for r in selected], dtype=np.float32))
    counts = {s: int(sum(entities[g]["split"] == s for g in group_array)) for s in ("fit", "tune", "audit", "confirm")}
    write_json(marker, {"config": config, "pairs": len(selected), "split_pair_counts": counts,
                        "note": "All original labels outside the neural gate remain in truth counts. Gate-excluded pairs get no match; no positive label forces an evaluation pair through the gate."})
    write_json(output/"status.json", {"state": "prepared", "pairs": len(selected), "split_pair_counts": counts})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data set/student_resource/dataset")
    p.add_argument("--output", default="artifacts/neural_data_v1")
    args = p.parse_args()
    prepare(args.data, args.output)
