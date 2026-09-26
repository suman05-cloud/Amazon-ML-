# Amazon ML Challenge: business entity resolution

Match every Source 1 business to zero or more Source 2/3 records. The objective is
**entity-macro F0.5**, including singleton entities. It is not classification accuracy
or pair-micro F0.5. All training examples and business information come from the
provided TSVs. The pipeline never calls a business lookup, geocoding, or matching API.

## Step-by-step explanation

1. **Understand the labels.** Each Source 1 row is a separate matching decision.
   Source 1 is deduplicated; Source 2/3 may have several matching records. Predicting
   an empty set is valid. For a nonempty case the exact score simplifies to
   `1.25 * true_positive_count / (0.25 * true_match_count + predicted_match_count)`.
   If truth and prediction are both empty, the score is 1. Otherwise a wrong empty
   prediction scores 0. The final score averages these values over references.
2. **Normalize without discarding evidence.** Transliterate Unicode, fold case and
   accents, standardize punctuation and address abbreviations. Keep both full and
   legal-suffix-stripped names. Preserve numbers. Missing addresses supply zero
   similarity, rather than an artificial perfect match between empty strings.
3. **Find candidates through independent routes.** Use normalized names, compact
   names, distinctive name words, prefixes, consonant signatures, normalized
   addresses, address words, number-plus-address-word keys, postal-like numbers, name/location and name/number conjunctions, token pairs,
   and address-anchored deletion signatures for short names.
   Each key includes the actual country string; no country list is hard-coded.
   Suppress an entire block if it has more than 400 target records. Rerank the union
   using inexpensive fuzzy name/address evidence and retain 80 candidates.
4. **Measure retrieval before the classifier.** Uniformly sample 8,000 references,
   but scan **all 10,320,219 training targets** for their blocks. A reference-focused
   index retains only relevant blocks; it is equivalent to querying the disk index
   over the same full corpus. True matches are **not injected into the candidates**.
   Blocking misses remain false negatives in the reported score.
5. **Train on actual hard candidates.** Forty features capture character similarity,
   token overlap, name/address agreement, missingness, number agreement/conflicts,
   and retrieval strength. Two LightGBM models with different random seeds average
   their probabilities. IDs and country category labels are not model features.
6. **Tune separately from evaluation.** Normalized business-name groups within each
   country deterministically split approximately 60% fit / 20% tune / 20% audit.
   Tune supplies early stopping and the macro-F0.5 threshold. Audit supplies the
   held-out score. The saved models are the models evaluated; there is no silent
   all-data refit with an unverified threshold. Country-transfer tests train and
   tune on one country and evaluate the other country's audit businesses.
7. **Run the full test set in batches.** A disk-backed SQLite index controls memory.
   Prediction checkpoints are atomic JSONL chunks containing the exact scored
   candidate IDs and final matched IDs. Empty rows and unseen countries are retained.
8. **Validate and package.** Strict validation checks every reference, every target
   ID, duplicates, headers, and the matches-subset-of-candidates relationship. Then
   package the code, methodology, and the two TSVs in the required ZIP structure.

## Dataset inspected

| File | Rows |
|---|---:|
| Training Source 1 | 2,206,821 |
| Training Source 2 | 5,034,616 |
| Training Source 3 | 5,285,603 |
| Test Source 1 | 1,732,544 |
| Test Source 2 | 4,887,273 |
| Test Source 3 | 5,082,316 |

Training includes US and India; the test set includes 259,452 France references.
There are 123,247 training singletons. Missing target addresses occur in every
split. Names include Devanagari and accented Latin scripts. The data needs explicit
`sep="\t"` / `delimiter="\t"`; it is not comma-separated.

## Installation

Python 3.12 is the tested environment. From the project root:

```powershell
uv --cache-dir .uv-cache sync --frozen
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Without uv, including on Linux/AWS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

`requirements.txt` pins the tested libraries and `uv.lock` pins the full uv
environment. Windows paths below may be replaced with `python` in an activated
Linux virtual environment. No pretrained weights or GPU installation are required.

## Reproduce the experiment

### Measured results from this workspace

| Experiment | Sampled references | Candidate recall (all sampled links) | Audit macro F0.5 |
|---|---:|---:|---:|
| Initial blocking, 40 candidates | 8,000 | 83.60% | 0.87188 |
| Revised blocking, 80 candidates | 8,000 | 95.86% | **0.93115** |

These experiments use different seeded reference samples; the comparison is not
a paired statistical test. Each searches the complete training target pool.
The selected revised model was evaluated on 1,603 audit references: micro link
precision **96.53%**, micro link recall **87.49%**, and singleton accuracy **92.78%**.
Audit macro F0.5 was 0.89370 for India and 0.95285 for US. These are local validation
results, not test/leaderboard scores. France has no labeled validation examples.

A 2,000-round ceiling was also tried with early stopping. It stopped at 802/786
iterations and did not improve the tuning macro F0.5, so the 800-round-ceiling
model at `artifacts/experiment_v2/model` remains selected. The model choice is
based on tune scores, not choosing the larger audit score.

The full test index has been built locally (13.80 GiB). A 1,000-reference test
preview covered 390 US, 482 India and 128 France references. The batched ranker
and bounded record cache produced byte-identical candidate and matching files;
the optimized preview took approximately 33 seconds, with operating-system cache
warmth potentially contributing. See `artifacts/runtime_report.json` for details.

A full two-worker run was launched on September 25, 2026. Its current state is in
`output/run_status.json`; inspect it rather than assuming the output is complete.
Do not launch a duplicate while it is running. If interrupted, resume with
`python -m amazon_ml finish --workers 2 --threads 2` from this project environment.
Only a completed run and successful `output/validation.json` make the TSVs ready
for upload. No external service upload or cloud provisioning is part of the run.

### Commands

```powershell
.\.venv\Scripts\python.exe -m amazon_ml prepare --sample-size 8000 --work artifacts/experiment_v2
.\.venv\Scripts\python.exe -m amazon_ml train --work artifacts/experiment_v2 --threads 4
```

Pass `--data /path/to/dataset` when the data is stored elsewhere. The data directory
contains `train/` and `test/`, not `student_resource/` itself. The preparation cache
includes input-file fingerprints and normalization settings; use a new `--work`
directory after changing these. Incomplete preparation scans restart; complete
preparation caches are reusable. `train` replaces the models/report in its work
directory, so use another directory when preserving experiment comparisons.

Useful experiment outputs:

- `artifacts/experiment_v2/prepared.json`: candidate recall, full-pool scan count, timing.
- `artifacts/experiment_v2/report.json`: tune/audit and country-transfer metrics,
  blocking ceilings, threshold, feature importance.
- `artifacts/experiment_v2/errors.jsonl`: audit false merges, missed links, and blocked-out IDs.
- `artifacts/experiment_v2/cases.jsonl`: reference and retrieved candidate text for analysis.
- `artifacts/experiment_v2/model/`: saved LightGBM weights and inference manifest.

The default 8,000 references are an initial supervised sample, not all training
references. The **target retrieval pool is full size**. Increase to 30,000 or more
in a new work directory after confirming runtime and memory; do not describe the
initial model as trained on 2.2 million labeled references.

## Full prediction and submission

```powershell
.\.venv\Scripts\python.exe -m amazon_ml build-index --index artifacts/test.sqlite
.\.venv\Scripts\python.exe -m amazon_ml predict --index artifacts/test.sqlite --model artifacts/experiment_v2/model --output output
.\.venv\Scripts\python.exe -m amazon_ml validate --output output
```

The default data path is `data set/student_resource/dataset`. On Windows,
`./run_local.ps1` executes preparation, training, indexing, full prediction and
validation sequentially and stops on an error. It does not upload anything.

Both `build-index` and `predict` resume when rerun with the same settings and paths.
Do not change source files midway. If normalization changes, use a new index path.
The index uses disk-backed records/postings and a 64 MiB SQLite cache. Allow
substantial SSD headroom for the records, lookup index, temporary index sorting,
candidate outputs, and the ZIP. Full-scale throughput must be benchmarked on the
chosen machine; the small test suite is not a runtime benchmark.

To measure inference before the full run:

```powershell
.\.venv\Scripts\python.exe -m amazon_ml predict --limit 1000 --output artifacts/preview
```

Limited runs produce **`.preview.tsv`** files, which are intentionally not named
like submissions. Use a fresh output directory for the complete run. The full
output must contain all 1,732,544 Source 1 rows. Runtime logs show processed rows.

The supplied validator can also run on a sufficiently large-memory machine:

```powershell
.\.venv\Scripts\python.exe "data set/student_resource/utils/validate_submission.py" --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir "data set/student_resource/dataset/test" --check-ids
```

The project's `validate` command performs strict checks using disk tables; it
does not retain every candidate list in RAM. Validation proves format and ID
correctness, **not a hidden-test score**.

Fill in the team details in `Documentation_template.md`, then:

```powershell
.\.venv\Scripts\python.exe -m amazon_ml package --destination team_submission.zip
```

Only `output/matching_results.tsv` is the leaderboard upload. The ZIP includes
both output files, `code/business_entity_resolution/`, and the methodology.

To regenerate the methodology from measured results and supply team details:

```powershell
.\.venv\Scripts\python.exe -m amazon_ml document --team "Your actual team name" --members "Your team members"
```

Use `predict --validate` to run full inference and strict validation in one command.
A completed validation writes `output/validation.json` with `status: PASS`.
The `profile` command can regenerate dataset row counts and missing-value statistics.

## Local GPU and AWS

The inspected laptop has an RTX 4050 with 6 GiB VRAM and approximately 15 GiB usable
system RAM. This pipeline's expensive stages are text retrieval and candidate
construction. CUDA does not automatically accelerate SQLite or Python text
processing. LightGBM's compact tabular model is trained on CPU; a large language
model is not inherently better for noisy business identity matching.

For the September 27 deadline, first obtain a valid full prediction, then improve
the measured bottleneck. If retrieval recall is low, a larger classifier cannot
repair it. If retrieval is high but matching is weak, use the exported candidate
text to investigate a multilingual neural reranker. Any proposed pretrained model
must satisfy the challenge's MIT/Apache-2.0 license and <=8B parameter requirement,
and its benefit must be measured on fixed validation businesses. No neural model
has been trained or claimed to improve this solution merely because a GPU exists.

The same source runs on Linux/AWS. Copy the data and project, install the pinned
dependencies, and use `--data`. More system RAM helps larger reference samples;
local SSD capacity and random-read speed matter for full indexing/prediction.
No cloud resources are provisioned automatically. Check your account's actual
credit eligibility and instance/storage charges before launching resources.

On a larger machine, prediction can use independent processes:

New `finish` runs default to four workers, as requested. The command starts the
workers, joins their results and validates them. Use a fresh output directory
when changing worker count; the already-running two-worker job must resume with
`--workers 2` and its existing output directory.

```powershell
.\.venv\Scripts\python.exe -m amazon_ml finish --workers 4 --threads 2 --output output_next
```

Progress is recorded in `output/run_status.json` and `output/logs/shard-000.log`
(and one log per additional worker). Repeating the command resumes completed
chunks. Keep the same worker count, batch size, model and output directory.
For manual control or separate terminals:

```bash
python -m amazon_ml predict --data /data/dataset --index /data/test.sqlite --model artifacts/experiment_v2/model --output output --shard 0 --shards 2 --threads 2
python -m amazon_ml predict --data /data/dataset --index /data/test.sqlite --model artifacts/experiment_v2/model --output output --shard 1 --shards 2 --threads 2
python -m amazon_ml merge --output output --shards 2
python -m amazon_ml validate --data /data/dataset --output output
```

Run the two predict commands in separate terminals/processes. Both must use the
same model, index, batch size, output directory and shard count. Merge verifies
the configuration and refuses unfinished shards. Output row order is immaterial
to the challenge; each reference is still present exactly once.

## Validation limits and iteration discipline

- The audit set is an initial held-out estimate. Repeatedly inspecting its errors
  turns it into development data; use a fresh seed/reference sample for the final
  confirmation after tuning the blocking rules or architecture.
- Equal normalized names are kept together, but different aliases of related
  businesses may still occur across splits. No ID features are used.
- The country-transfer score is a stress test, not an estimate of France accuracy.
  There are no France training labels. Only the organizer can score test predictions.
- Blocking assumes the supplied country labels are consistent across true matches.
  Oversized-block suppression and top-k truncation can miss true links, and these
  losses are included in the measured recall and F0.5.
- Postal-like numeric tokens are heuristic features, not externally geocoded facts.
- The classifier does not impose one match per source; many-to-one reference lists
  are allowed as required. Independent pair decisions can still produce ambiguous
  links across references; inspect false positives before adding unvalidated rules.

## Source layout and licenses

The September 25 enhanced-tree and CUDA experiments are documented in
[GPU_README.md](GPU_README.md). The enhanced tree model improved a new, disjoint
1,500-reference confirmation from 0.924841 to 0.949372 F0.5. The ongoing original
prediction still uses the baseline model; experimental training does not change it.

`text.py`: normalization/features; `blocking.py`: focused and disk retrieval;
`train.py`: preparation, fitting, thresholds and errors; `metrics.py`: exact scorer;
`predict.py`: checkpointed inference and shard merge; `validate.py`: strict checks;
`package.py`: final ZIP; `cli.py`: commands. Tests cover metric edge cases, retrieval
equivalence, unseen-country handling, full miniature training/inference, restart,
sharding and preview naming.

The project and its trained tree models are MIT licensed. LightGBM is
[MIT licensed](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE).
Its [official parameter documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html)
describes the deterministic training controls used here. String features use
[RapidFuzz's documented similarity functions](https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html).
Library documentation was consulted for implementation; no external business data
was used to resolve any entity.
