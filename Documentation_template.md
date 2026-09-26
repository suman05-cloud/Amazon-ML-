# ML Challenge 2026: Business Entity Resolution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** Fill with actual upload date.

## 1. Executive Summary

The solution retrieves candidates through complementary name and address blocking
rules, then scores them with an ensemble of two supervised LightGBM classifiers.
The decision threshold maximizes the exact entity-macro F0.5 on a tuning partition,
including correct empty predictions for singletons. The measured audit score is
**0.931151**; it is not a leaderboard or hidden-test score.

## 2. Methodology

### 2.1 Problem Analysis

Training contains 2,206,821 reference businesses and 10,320,219 target records.
The 1,732,544 test references include 259,452 France businesses; training only
contains US and India. Missing addresses, script changes, punctuation, legal
suffixes, duplicated words, typos, repeated addresses and generic names complicate
matching. Training has 123,247 singleton references.

### 2.2 Solution Strategy

**Approach:** Blocking + supervised pair classifier.

The reference sample contains **8,000** entities, selected
uniformly with seed **2027**. Their candidates are retrieved against the
**entire 10,320,219-record training target pool**, not a reduced
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
- Blocks above **400** target records are suppressed in full rather
  than truncated to arbitrary first records.
- A fuzzy name/address ranker retains at most **80** candidates per
  reference, then the final matching model scores every retained candidate.
- Measured model candidate pairs: **632,411** across the sampled
  references; this count is not the full-test candidate count.
- Overall sampled candidate-link recall: **95.8632%**.
- Reduction ratio against all sampled-reference/target comparisons:
  **99.99923401%**.

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
seeds use best iteration counts **[798, 786]**. Inference averages
their probabilities. No class-weighted accuracy objective is substituted for F0.5.

**Threshold:** **0.530**, selected from a fixed grid using tune
entity-macro F0.5. It allows multiple matches or an empty set. The saved models are
the exact evaluated models; they are not silently refit on the audit data.

## 5. Results and Error Analysis

| Partition | References | Macro F0.5 | Micro precision | Micro recall | Candidate recall |
|---|---:|---:|---:|---:|---:|
| tune | 1,553 | 0.930710 | 97.0975% | 86.2389% | 95.6788% |
| tune/india | 617 | 0.900148 | 95.2140% | 80.8317% | 91.0612% |
| tune/us | 936 | 0.950855 | 98.2078% | 89.6667% | 98.6061% |
| audit | 1,603 | 0.931151 | 96.5339% | 87.4887% | 96.2448% |
| audit/india | 588 | 0.893701 | 93.7464% | 82.8600% | 91.8357% |
| audit/us | 1,015 | 0.952847 | 98.0165% | 90.0477% | 98.6824% |

Micro precision and recall are diagnostic quantities, not the leaderboard metric.
Audit includes **174** false-positive links and
**693** false-negative links, including blocking misses.
The audit blocking-only macro score ceiling is **0.985417**.
Individual audit errors, their text, probabilities, and blocked-out link IDs are
exported to `errors.jsonl` for review.

### Unseen-country stress tests

Each test fits and tunes only on other countries and evaluates the held country's
audit references. These are transfer diagnostics, not France performance estimates.

- Hold out **india**: macro F0.5 **0.843870**, precision 90.5131%, recall 76.9270%.
- Hold out **us**: macro F0.5 **0.908610**, precision 95.2040%, recall 85.7023%.

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
python -m amazon_ml prepare --data PATH_TO_DATASET --work artifacts/experiment_v2 --sample-size 8000 --seed 2027 --block-cap 400 --top-k 80
python -m amazon_ml train --work artifacts/experiment_v2 --threads 4 --rounds 800 --seeds 42 2026
```

`candidate_pairs.tsv` contains exactly the final pairs scored by the ML models.
Full outputs require a completed test run and successful strict validation; a
limited `.preview.tsv` run is not a submission. Team details and the actual upload
date must be completed before the final package is submitted.
