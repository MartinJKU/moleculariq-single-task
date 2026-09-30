# Single-task GRPO on MolecularIQ

Trains `Qwen/Qwen2.5-0.5B-Instruct` with TRL's GRPO on one
[MolecularIQ](https://arxiv.org/abs/2601.15279) task family at a time
(counting, atom-index attribution, constrained generation) and scores the base
model and the three trained models on the whole official benchmark (5,111
items, 3 attempts each) with the official `moleculariq-eval` harness.

- Training questions are generated offline from the MolecularIQ training pool,
  verified with the official scorer and frozen as a content-hashed dataset.
- The reward is the official score plus credit per requested property.
- The benchmark is test-only: never used for training, tuning or checkpoint selection.

Follow-ups: a supervised warm-up before GRPO and the 1.5B model are in
[extension/](extension/README.md); every earlier recipe and run is in
[history/](history/README.md).

## Results

Benchmark average accuracy (%), paired change vs the base model with 95%
bootstrap interval over items:

| Model | All items | Counting | Index | Generation |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct (base) | 2.01 | 2.94 | 0.77 | 2.31 |
| GRPO, counting | 2.42 (+0.41 [+0.08, +0.74]) | **4.33** | 0.25 | 2.63 |
| GRPO, index attribution | 1.52 (−0.49 [−0.77, −0.21]) | 2.20 | **0.29** | 2.10 |
| GRPO, constrained generation | **4.43** (+2.42 [+1.95, +2.90]) | 4.24 | 0.19 | **9.34** |

- **Generation is learned**: on questions none of the default molecules
  answers (`CC(O)C`, `CCO`, …), 0.27% → 6.12%. The repertoire is narrow:
  mostly chains of `CC(O)` units.
- **Counting improves modestly**, largely through per-property priors.
- **Index attribution does not improve** with GRPO alone.
- A supervised warm-up on worked solutions changes that for counting (7.54%)
  and index (3.89%); at 1.5B the counting model reaches 5.25% on all items
  (see [extension/](extension/README.md)).

## Running it

```bash
bash scripts/00_setup_env.sh          # environment at the locked versions (network)
python scripts/01_prefetch_assets.py  # model, training pool, benchmark into the HF cache (network)
bash scripts/02_build_dataset.sh      # frozen training set miq-train (CPU)
bash scripts/10_train.sh              # count, index, constraint_generation (GPU)
bash scripts/20_evaluate.sh           # benchmark: base model + 3 models (GPU)
bash scripts/25_dev_evaluate.sh       # dev-split breakdowns (GPU)
bash scripts/30_report.sh             # results/report/: analysis.txt, figures (CPU)
```

`scripts/run_all.sh` runs stages 02–30 and skips finished ones;
`scripts/status.sh` shows what is running or done. Heavy outputs go to
`$MIQ_DATA`, `$MIQ_RUNS`, `$MIQ_RESULTS` (default: `data/`, `runs/`,
`results/`). Step by step on one GPU: [docs/runbook.md](docs/runbook.md).

Checks without a GPU: `pytest -q` (186 tests) and `python scripts/check_vendor.py`.

## Data

| What | Source | Pinned by |
|---|---|---|
| Policy | `Qwen/Qwen2.5-0.5B-Instruct` | revision in each run's provenance |
| Training molecules | `ml-jku/moleculariq-trainPool` (1.27M) | revision in `configs/dataset.yaml` |
| Benchmark (test only) | `ml-jku/moleculariq-v0.0` via `moleculariq-eval` | harness commit `425ecaa` |
| Training set `miq-train` | built by `scripts/02_build_dataset.sh` | content hash in `configs/dataset.yaml` |

A rebuild reproduces the training set's content hash exactly. During
development it was called `miq-train-v003` (and the warm-up set `miq-sft-v001`);
the reported models, manifests and `history/` use those names. Only the stored
name differs, which is why the hash differs.

## Layout

```
configs/      dataset, the three experiments, benchmark runs, report
src/miqgrpo/  generation + build_dataset (data), rewards + train_grpo (training),
              evaluate + dev_eval (evaluation), report + analysis + breakdown (report)
scripts/      the pipeline in running order (+ slurm/)
tests/        rewards, prompts, data, benchmark isolation, reproducibility
extension/    supervised warm-up and 1.5B study
history/      earlier recipes, runs and reports
docs/         official semantics, runbook
```

## Limitations

- One training seed per model: intervals cover item sampling, not training variance.
- Only the first, pre-registered recipe is a clean held-out result; the final
  recipe was designed after earlier benchmark breakdowns had been seen (all
  settings chosen on the dev split).
- The generation model meets constraints with a narrow family of molecules.
- Single-task models lose accuracy on the other task families.

## Upstream

[moleculariq-core](https://github.com/ml-jku/moleculariq-core) `a1b8963`
(question generation, verifier) and
[moleculariq-eval](https://github.com/ml-jku/moleculariq-eval) `425ecaa`
(benchmark harness; its answer extraction is vendored in `src/miqgrpo/vendor/`).
