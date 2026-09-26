"""Build the challenge write-up from measured results, not hand-entered scores."""
import json
from pathlib import Path


def document(work, destination, team="[Your Team Name]", members="[List all team members]"):
    work = Path(work)
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    prepared, metrics = report["prepared"], report["metrics"]
    cfg = prepared["config"]
    settings = report.get("training", {"rounds": 800, "threads": 4, "seeds": [42, 2026]})
    seed_args = " ".join(str(s) for s in settings["seeds"])
    table = "\n".join(f"| {name} | {m['entities']:,} | {m['macro_f0_5']:.6f} | {m['micro_precision']:.4%} | {m['micro_recall']:.4%} | {m['candidate_recall']:.4%} |" for name, m in metrics.items())
    transfer = "\n".join(f"- Hold out **{name}**: macro F0.5 **{m['macro_f0_5']:.6f}**, precision {m['micro_precision']:.4%}, recall {m['micro_recall']:.4%}." for name, m in report.get("country_transfer", {}).items())
    audit = metrics["audit"]
    text = f"""# ML Challenge 2026: Business Entity Resolution

**Team Name:** {team}  
**Team Members:** {members}  
**Submission Date:** Fill with actual upload date.

## 1. Executive Summary

The solution retrieves candidates through complementary name and address blocking
rules, then scores them with an ensemble of two supervised LightGBM classifiers.
The decision threshold maximizes the exact entity-macro F0.5 on a tuning partition,
including correct empty predictions for singletons. The measured audit score is
**{audit['macro_f0_5']:.6f}**; it is not a leaderboard or hidden-test score.

## 2. Methodology

### 2.1 Problem Analysis

Training contains 2,206,821 reference businesses and 10,320,219 target records.
The 1,732,544 test references include 259,452 France businesses; training only
contains US and India. Missing addresses, script changes, punctuation, legal
suffixes, duplicated words, typos, repeated addresses and generic names complicate
matching. Training has 123,247 singleton references.

### 2.2 Solution Strategy

**Approach:** Blocking + supervised pair classifier.

The reference sample contains **{prepared['reference_count']:,}** entities, selected
uniformly with seed **{cfg['seed']}**. Their candidates are retrieved against the
**entire {prepared['scanned_targets']:,}-record training target pool**, not a reduced
distractor sample. Labels do not participate in retrieval and missing true links
are never forced into the candidate list.

Businesses with the same normalized core name within a country share a deterministic
split: approximately 60% fit, 20% tune, 20% audit. Tune controls early stopping and
thresholds; audit is excluded from both. Different aliases can remain in different
groups; this is a limitation. Reference IDs and target IDs are never model features.

## 3. Candidate Generation

- Country-aware normalized name, compact name, token, prefix and consonant keys.
- Normalized address, address-word, house-number/address-word and postal-like keys.
- Combined name/location and name/number keys, name/address token pairs, and
  address-anchored deletion signatures for short names (retrieval version 2).
- Blocks above **{cfg['block_cap']}** target records are suppressed in full rather
  than truncated to arbitrary first records.
- A fuzzy name/address ranker retains at most **{cfg['top_k']}** candidates per
  reference, then the final matching model scores every retained candidate.
- Measured model candidate pairs: **{prepared['pairs']:,}** across the sampled
  references; this count is not the full-test candidate count.
- Overall sampled candidate-link recall: **{prepared['candidate_recall']:.4%}**.
- Reduction ratio against all sampled-reference/target comparisons:
  **{prepared['reduction_ratio']:.8%}**.

The reference-focused in-memory implementation is tested for equivalence with the
full SQLite implementation. Country labels are open strings and France is not
filtered out. The country blocking assumption and top-k truncation can lose links;
the reported metric includes those misses.

## 4. Matching Model

**Model:** Two LightGBM gradient-boosted tree classifiers trained from scratch on
the supplied data. Model/source license: MIT. No pretrained language model, external
business registry, geocoding service, or outside identity data is used.

**Features:** 40 numeric similarities and indicators: full/core name and address
character ratios, token-sort/set/partial scores, Jaro-Winkler similarity, exact
agreement, token overlap/containment, numeric and postal-like agreement/conflicts,
missing fields, relative lengths, name/address interactions, and retrieval strength.
Unicode transliteration uses a generic local text library.

**Training:** 31 leaves, learning rate 0.045, minimum 80 examples per leaf,
L1/L2 regularization, feature/row subsampling, deterministic CPU settings. The two
seeds use best iteration counts **{report['best_iterations']}**. Inference averages
their probabilities. No class-weighted accuracy objective is substituted for F0.5.

**Threshold:** **{report['threshold']:.3f}**, selected from a fixed grid using tune
entity-macro F0.5. It allows multiple matches or an empty set. The saved models are
the exact evaluated models; they are not silently refit on the audit data.

## 5. Results and Error Analysis

| Partition | References | Macro F0.5 | Micro precision | Micro recall | Candidate recall |
|---|---:|---:|---:|---:|---:|
{table}

Micro precision and recall are diagnostic quantities, not the leaderboard metric.
Audit includes **{audit['false_positives']:,}** false-positive links and
**{audit['false_negatives']:,}** false-negative links, including blocking misses.
The audit blocking-only macro score ceiling is **{audit['blocking_macro_ceiling']:.6f}**.
Individual audit errors, their text, probabilities, and blocked-out link IDs are
exported to `errors.jsonl` for review.

### Unseen-country stress tests

Each test fits and tunes only on other countries and evaluates the held country's
audit references. These are transfer diagnostics, not France performance estimates.

{transfer}

There are no France training labels or hidden-test labels. Repeated development
against audit errors would invalidate the audit as a final independent estimate;
future changes need a fresh confirmation sample.

## 6. Conclusion

This is a measured, reproducible entity-resolution pipeline with full-pool retrieval,
precision-sensitive thresholding, explicit singleton handling and restartable full
inference. Remaining risks include native-script name corruption, common names,
missing addresses, ambiguous repeated addresses and country shift. A winning rank
is not inferred from a local validation score.

## Appendix: Code and Reproduction

The final ZIP layout is `output/` (both TSVs),
`code/business_entity_resolution/` (source, README and pinned requirements), and
this document. See the README for Windows and Linux/AWS installation and commands.
Use `prepare`, `train`, `build-index`, `predict`, `validate`, then `package`.
The preparation settings needed to reproduce this experiment are:

```text
python -m amazon_ml prepare --data PATH_TO_DATASET --work artifacts/experiment_v2 --sample-size {cfg['sample_size']} --seed {cfg['seed']} --block-cap {cfg['block_cap']} --top-k {cfg['top_k']}
python -m amazon_ml train --work artifacts/experiment_v2 --threads {settings['threads']} --rounds {settings['rounds']} --seeds {seed_args}
```

`candidate_pairs.tsv` contains exactly the final pairs scored by the ML models.
Full outputs require a completed test run and successful strict validation; a
limited `.preview.tsv` run is not a submission. Team details and the actual upload
date must be completed before the final package is submitted.
"""
    Path(destination).write_text(text, encoding="utf-8")
