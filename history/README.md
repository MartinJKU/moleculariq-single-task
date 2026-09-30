# Experiment history

Every recipe, run and report before the final pipeline, kept for the write-up.
`git checkout pre-cleanup` runs any config below exactly as reported, with the
code it was run with.

- The final pipeline is recipe v003: the root models `count`, `index` and
  `constraint_generation` are the runs `grpo-*-r006`, not retrained. Its
  dataset `miq-train-v003` is `miq-train` in the root (same data; only the
  stored name differs).
- Each final model was benchmarked once and no checkpoint was chosen by a
  score. Only the main study (v001) is a clean held-out result; later recipes
  were designed after the previous benchmark breakdown had been seen.

| Folder | Contents |
|---|---|
| `configs/experiments/`, `configs/experiments/pilots/` | the 14 runs `grpo-*-r001..r006` and the 300-step pilots |
| `configs/preprocessing/` | the dataset versions `miq-train-v001..v003` |
| `configs/evaluation.yaml`, `configs/report.yaml` | every benchmark run and report part, as they were |
| `reports/{main-study,shaping-ablation,recipe-v002,recipe-v003}/` | `analysis.txt` (all numbers with intervals) and figures |
| `reports/dev/` | dev-split breakdowns of base models, pilots and runs |
| `diff_experiments.py` | the settings in which two experiment configs differ |

## Stages

Benchmark average accuracy (%) on all items and on the model's own task, for
the count / index / generation model. Base model: 2.01 all items;
2.94 / 0.77 / 2.31 own task (v001 was evaluated on the HF backend, base 2.00).

| Stage | Runs | Data | What changed | All items | Own task |
|---|---|---|---|---|---|
| main study (pre-registered) | `r001` | v001 | official score + 0.1 × answer shape (+ 0.05 × valid SMILES); lr 1e-6; 8 answers per question; 500 steps | 2.35 / 2.10 / 2.88 | 3.69 / 0.19 / 4.03 |
| shaping ablation | `r002` | v001 | answer-shape bonus off | – | 3.74 / 2.85 / 3.93 |
| recipe v002 | `r003–r005` | v002 | default answers filtered out of the data; per-property credit; lr 3e-6 / 1e-5; 16 answers; 600 steps | 2.20 / 1.84 / 2.79 | 3.78 / 0.19 / 4.58 |
| **recipe v003** (final) | `r006` | v003 | 3× data; credit for a correct "none", penalty for a wrong one; 1,000–3,000 steps | 2.42 / 1.52 / **4.43** | 4.33 / 0.29 / **9.34** |

- **Main study:** every own-task change came from the few questions a constant
  answer solves; the generation model answered `CC(O)C` (the prompt's example)
  in 88% of attempts. GRPO reinforced the only answers that were ever rewarded.
- **Shaping ablation:** removing the shape bonus only changed which default
  the index model collapsed to (short lists → `[]`).
- **v002:** with defaults filtered out and per-property credit, generation
  started to meet constraints (0.62% → 1.86% on questions a constant answer
  does not solve).
- **v003:** scale helped generation most (4.58% → 9.34% own task).

## What helped and what did not

| Change | Effect | Kept |
|---|---|---|
| Per-property partial credit | index groups with a learning signal ~2% → ~85%; generation on questions no default molecule solves 0.27% → 1.75% (v002) → 6.12% (v003) | yes |
| Rejecting constraints a default molecule meets; capping all-"none" questions | default answers of the generation model 88% → below 0.5% | yes |
| 3× data, 2.5–5× steps (v003) | generation dev 15.9% → 25.9%; counting plateaued after ~250 steps | yes |
| Credit for a correct "none", penalty 0.5 for a wrong one (counting) | all-zero dev questions 7% → 30%; "0" answers right 75% of the time (base 34%) | yes |
| Learning rate 3e-6 instead of 1e-5 | at 1e-5 the count policy became deterministic by step 200 | yes |
| 16 instead of 8 answers per question; vLLM rollouts | more groups with a better answer; much faster rollouts | yes |
| Answer-shape bonus (0.1) | collapse to defaults with or without it | removed |
| Valid-SMILES bonus (0.05) | the generation model collapsed onto `CC(O)C` | removed |
| Quarter credit for "none" without a penalty | count policy drifted to all-zero answers (run stopped at step 100) | replaced |
| "None" penalty for index (0.5 pilot, 0.25) | neither taught when a list is empty | 0.25 kept |
| Learning rate 1e-6 | too slow (counting dev 6.4% → 4.7% by step 150) | no |

## Runs

| Run | Data | Notes | Benchmark run |
|---|---|---|---|
| `grpo-{count,index,constraint}-r001` | v001 | main study | `miq-eval-*-r001` (HF) |
| `grpo-{count,index,constraint}-r002` | v001 | without the shape bonus | `miq-eval-*_noformat-r002` (HF) |
| `grpo-count-r003` | v002 | quarter credit for "none"; stopped at step 100 | – |
| `grpo-count-r004`, `-r005` | v002 | lr 1e-5 / 3e-6 | `miq-eval-count-r004`, `-r005` |
| `grpo-index-r003`, `grpo-constraint-r003` | v002 | lr 3e-6 (index resumed from step 500 after an operator mistake) | `miq-eval-index-r003`, `miq-eval-constraint_generation-r003` |
| `grpo-{count,index,constraint}-r006` | v003 | **final models** | `miq-eval-{count,index,constraint_generation}` |

Pilots (300 steps, dev split only) chose the learning rates (3e-6; 1e-5 for
the v002 count run) and the "none" penalties; configs and dev results are in
`configs/experiments/pilots/` and `reports/dev/`.
