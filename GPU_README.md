# GPU accuracy experiments

The original two-worker prediction continues to use `artifacts/experiment_v2/model`.
Do not change its worker count, model or output directory while resuming it.

## Measured improvement

The new 105-feature tree ensemble is saved in `artifacts/experiment_v3/model`.
It uses the original full-pool candidates, embedded address-number conflicts,
fuzzy word matching, abbreviations and candidate comparisons.

| Evaluation | Baseline F0.5 | Enhanced F0.5 |
|---|---:|---:|
| Tuning, 1,553 references | 0.930710 | 0.949760 |
| Original audit, now development data | 0.931151 | 0.955144 |
| New disjoint confirmation, 1,500 references | 0.924841 | 0.949372 |

The confirmation improvement is 0.024531, with a paired bootstrap 95% interval
of [0.017928, 0.030757]. This measures local sample uncertainty, not country shift.
None of these values are hidden-test or leaderboard scores. France has no training labels.

## Neural experiment

The CUDA experiment starts from
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revision
`a3ded7f4d0d725c6fc251504622d65b2516a3aa1`, under Apache-2.0.
Its original-language pair input is combined with the 105 structured features.
All business records and labels come from the supplied challenge dataset.
No external business directory or geocoding data is used.

The prepared dataset has 64,298 fitting pairs, 20,247 tuning pairs, 20,893 audit
pairs and 19,115 confirmation pairs. Fit pairs include all retrieved positives
and a mixture of actual hard and easy negatives. Evaluation pairs use a fixed
tree probability gate of 0.0001, chosen without consulting their labels. Missing
links still count in the official per-reference F0.5 denominator.

Four epochs use mixed precision on CUDA. The tuning split selects epoch,
neural/tree blend and threshold. Audit and confirmation are scored only after
those settings are frozen. A neural model is not assumed to improve accuracy;
the tree-only option is included in the blend comparison.

The environment is separate so installation cannot alter the running predictor:

```powershell
uv --cache-dir .uv-cache venv --python .venv\Scripts\python.exe .venv_gpu
uv --cache-dir .uv-cache pip install --python .venv_gpu\Scripts\python.exe torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv --cache-dir .uv-cache pip install --python .venv_gpu\Scripts\python.exe -r requirements.txt -r requirements-gpu.txt -e .
.venv_gpu\Scripts\python.exe -m amazon_ml.neural --download
.venv\Scripts\python.exe -m amazon_ml.neural_data
.venv_gpu\Scripts\python.exe -m amazon_ml.neural --benchmark-only --output artifacts/neural_benchmark
.venv_gpu\Scripts\python.exe -u -m amazon_ml.neural --output artifacts/neural_v1
```

Read `artifacts/neural_v1/status.json` for actual epoch, throughput, loss and peak
allocated VRAM. `report.json` records tuning results and, upon completion, the
frozen-model audit and confirmation scores. A preparation or download process
does not mean training has started. GPU usage is directly visible in `nvidia-smi`.

The `rescore` command reuses the original completed candidate chunks. It preserves
the fixed gate, raw text, numeric features, selected blend, threshold and exact
candidate audit trail. It refuses incomplete baseline runs and changed models.
Four CPU workers prepare features; one process owns the GPU. Full rescoring is
only appropriate after training and validation support model promotion.

```powershell
.venv\Scripts\python.exe -m amazon_ml.rescore --build-raw
.venv_gpu\Scripts\python.exe -m amazon_ml.rescore --neural-model artifacts/neural_v1 --workers 4 --output output_neural
```

Omit `--neural-model` and use the normal environment for enhanced-tree-only
rescoring. The command merges and strictly validates the final two TSVs.

## Reproducing the tree improvement

```powershell
.venv\Scripts\python.exe -m amazon_ml.improve
.venv\Scripts\python.exe -m amazon_ml.confirm
```

To run the enhanced tree model in a future full run, use a separate directory:

```powershell
.venv\Scripts\python.exe -m amazon_ml finish --workers 4 --threads 2 --model artifacts/experiment_v3/model --output output_v3
```

Do not start another full prediction just to monitor the existing run.

AWS has not been provisioned. The user's screenshot shows $100 of Free-plan
credits. GPU eligibility, quota, region, an explicit spending limit and any
account-plan upgrade must be resolved before cloud resources are launched.
