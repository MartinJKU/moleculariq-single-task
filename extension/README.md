# Extension: supervised warm-up and a larger model

Two questions on top of the root pipeline (GRPO alone on Qwen2.5-0.5B-Instruct):

1. Does a **supervised warm-up on worked solutions** (SFT) before the same GRPO
   recipe help? The GRPO-only models answer in ~22 tokens without working
   anything out.
2. Does a **larger model** (Qwen2.5-1.5B-Instruct) help?

All decisions are made on the dev split; each final model is benchmarked once.
**Decision rule, fixed before any result:** a 1.5B model gets the warm-up if the
0.5B warm-up model beats the 0.5B GRPO-only model on the dev split (fully
correct answers on questions a constant answer does not solve; atom overlap
for index), otherwise GRPO alone. Outcome: warm-up for counting and index, GRPO
alone for generation.

## Worked solutions

`miqgrpo.sft_data` writes 16,000 per family from the training splits of
`miq-train`; each is kept only if the official scorer accepts it.

```
Atoms: C(0), C(1), O(2), c(3), c(4), ...
halogen_atom_count: Cl(7), Br(12) -> 2
<answer>{"halogen_atom_count": 2}</answer>
```

The atom notation is the official system prompt's; the atoms per property come
from `moleculariq-core`'s official atom lists. For generation: the constraints,
the molecule the question was built from, and a check line.

## Running it

After the root pipeline (dataset, three models, dev results), from the repository root:

```bash
python scripts/01_prefetch_assets.py --model Qwen/Qwen2.5-1.5B-Instruct --skip-benchmark --out assets_manifest_1.5b.json
python -m miqgrpo.sft_data build --config extension/configs/sft_data.yaml
python -m miqgrpo.train_sft train --config extension/configs/sft/<family>-<size>.yaml
python -m miqgrpo.train_grpo train --config extension/configs/grpo/<run>.yaml
python -m miqgrpo.dev_eval --model $MIQ_RUNS/<run>/final --artifact miq-train --split <family>_dev \
    --samples 8 --max-tokens 4096 --out $MIQ_RESULTS/dev/miq-train/<run>.json
python -m miqgrpo.evaluate all --config extension/configs/evaluation.yaml --only <labels>
python -m miqgrpo.report --spec extension/configs/report-1.5b.yaml --out $MIQ_RESULTS/report-extension/1.5b
```

Runs: `<family>-0.5b-sft` (SFT then GRPO), `count-1.5b`, `index-1.5b` (SFT then
GRPO), `constraint_generation-1.5b-grpo-only`. The 1.5B generation run was
stopped at step 1,250 of 3,000 because of a deadline; its model is the last
checkpoint saved before the stop (chosen by time, never by a score; see
`TRUNCATED.md` in its run directory).

## Results

Benchmark average accuracy (%), paired change vs the model's own base model
with 95% bootstrap interval:

| Model | All items | Counting | Index | Generation |
|---|---|---|---|---|
| 0.5B base | 2.01 | 2.94 | 0.77 | 2.31 |
| 0.5B, SFT + GRPO, counting | 2.72 (+0.71 [+0.29, +1.13]) | **7.54** | 0.15 | 0.04 |
| 0.5B, SFT + GRPO, index | 1.38 (−0.63 [−1.01, −0.25]) | 0.15 | **3.89** | 0.00 |
| 0.5B, SFT + GRPO, generation | 2.41 (+0.40 [+0.01, +0.78]) | 2.70 | 0.38 | **4.31** |
| 1.5B base | 2.46 | 3.70 | 0.13 | 3.61 |
| 1.5B, SFT + GRPO, counting | **5.25** (+2.79 [+2.30, +3.29]) | **10.30** | 2.57 | 2.44 |
| 1.5B, SFT + GRPO, index | 3.69 (+1.23 [+0.71, +1.75]) | 2.48 | **6.90** | 1.53 |
| 1.5B, GRPO, generation (1,250 steps)¹ | 4.32 (+1.86 [+1.49, +2.26]) | 4.22 | 0.02 | **9.19** |

¹ The harness's scoring of this model was killed after ~1.5 h without a result
(its memory grew past 90 GB), so the model was re-evaluated with the same components (official prompt, sampling, extraction
and scorer; a 2-minute limit per answer, never reached). This scorer
reproduces the harness item for item on two finished runs (0 mismatches in
5,111 items each). Against the 0.5B generation model after 3,000 steps:
−0.15 pp [−1.23, +0.95] on generation, i.e. level.

- **Warm-up, 0.5B:** counting nearly doubles over GRPO alone (4.33% → 7.54%),
  and exact atom lists become learnable (index 0.29% → 3.89%). Generation gets
  worse: the warm-up leaves a low-entropy policy that collapses onto one
  molecule (51.5% of its answers).
- **Off-task, 0.5B:** a warmed-up model applies its procedure to every
  question, so its other task families drop to near zero.
- **1.5B:** the warm-up recipes gain about 3 pp more on their own task than at
  0.5B and lose far less off-task; the counting model is the best overall model
  of the project. The generation model (stopped at 1,250 steps) is level with
  the 0.5B one on the benchmark despite its dev-split lead.

Dev split (every dev question, 8 answers, %): counting 5.6 (base) → 11.3 (GRPO)
→ 19.3 (0.5B SFT + GRPO) → 23.8 (1.5B); index exact lists 0.1 → 0.6 → 10.8 →
14.8; generation 1.1 → 25.9 (GRPO) / 20.7 (SFT + GRPO) → 33.1 (1.5B, after 1,250 steps).
Full numbers: `$MIQ_RESULTS/report-extension/{0.5b-sft,1.5b}/analysis.txt`.
