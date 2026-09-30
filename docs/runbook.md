# Runbook: one GPU

End to end on one GPU, the way the reported numbers were produced: one A100
80GB PCIe and a data volume for the large artifacts (here `/data`).

| Stage | Hardware | Time |
|---|---|---|
| dataset | CPU (64 workers) | ~30 min |
| training, 3 models | GPU | 2–3 h each |
| benchmark, 4 models | GPU | ~20 min each |
| dev-split breakdowns | GPU | a few minutes per model |
| report | CPU | minutes |

About 8–10 GPU-hours in total.

---

## 0. Machine

- Linux with Python 3.12 and CUDA 13 (the reference setup); the exact
  package versions are in `requirements-lock.txt`.
- GPU: 1 × A100 80GB PCIe.
- Disk: ≥ 100 GB for data, checkpoints and results.

Point the pipeline at the data volume:

```bash
cat >> ~/.bashrc <<'EOF'
export MIQ_ROOT=/data/miq
export MIQ_DATA=$MIQ_ROOT/data
export MIQ_RUNS=$MIQ_ROOT/runs
export MIQ_RESULTS=$MIQ_ROOT/results
export HF_HOME=$MIQ_ROOT/hf
EOF
source ~/.bashrc
mkdir -p "$MIQ_DATA" "$MIQ_RUNS" "$MIQ_RESULTS" "$HF_HOME"
```

The variables must be *exported*: the pipeline scripts run as child processes
and do not see plain shell variables.

---

## 1. Install (needs network)

```bash
cd /data
git clone https://github.com/MartinJKU/moleculariq-grpo.git && cd moleculariq-grpo
bash scripts/00_setup_env.sh
source .venv/bin/activate
pytest -q
```

The setup script clones `moleculariq-core` and `moleculariq-eval` at pinned
commits into `third_party/` and installs every package at the version in
`requirements-lock.txt`. `moleculariq-eval` *is* `lm_eval`: do not also install
upstream `lm-evaluation-harness`, or whichever wins on `sys.path` decides how
the benchmark is scored.

---

## 2. Stage assets (needs network)

```bash
python scripts/01_prefetch_assets.py
```

Downloads the policy, the training pool and the official benchmark into
`$HF_HOME`, then re-resolves all three with `HF_HUB_OFFLINE=1` and fails if any
cannot be loaded offline. Every later stage runs with the Hub switched off.

---

## 3. Dataset (CPU)

```bash
bash scripts/02_build_dataset.sh
```

Builds `miq-train` (`configs/dataset.yaml`) from the pinned training-pool
revision, then verifies it in a fresh process. The build prints the dataset's
content hash and checks it against the one pinned in the config:

```
      content hash : f146ee7cb496794186dbf57a3183f339157b2299840f2ee48ef83c23392d68df
      examples     : 96000
```

If it does not match, stop: training refuses to start on any other data.

---

## 4. Training (GPU)

```bash
bash scripts/10_train.sh                # count, index, constraint_generation
bash scripts/10_train.sh count          # one model
```

Each model runs a preflight first (batch arithmetic, dataset hash, rendered
prompt, one real rollout), then trains (1,000–3,000 optimizer steps, see its
config). Finished models are skipped on a re-run. To follow one, including its
dev-split evaluations:

```bash
bash scripts/watch_run.sh count
```

Two runs fit on one A100 80GB at the same time (each reserves 10% of the GPU
for vLLM). Give each its own `MASTER_PORT` and start the second only once the
first is training, or vLLM's memory check at startup can fail:

```bash
MASTER_PORT=29501 bash scripts/10_train.sh constraint_generation   # in one tmux window
MASTER_PORT=29502 bash scripts/10_train.sh count                   # in another, a few minutes later
```

After an interruption, resume the model explicitly:

```bash
python -m miqgrpo.train_grpo train --config configs/experiments/count.yaml --resume
```

---

## 5. Benchmark (GPU)

```bash
bash scripts/20_evaluate.sh
bash scripts/20_evaluate.sh --only base,count     # a subset, by label
```

Runs the whole official benchmark (5,111 items, 3 samples each) for the base
model and the three trained models in `configs/evaluation.yaml`, on the
harness's vLLM backend with the settings recorded there. Each run writes
`$MIQ_RESULTS/moleculariq/<run_id>/`: the manifest, the harness's own results
file, per-item logs and `output.log`. Finished runs are skipped.

An interrupted run leaves a directory without `summary.json`. The script refuses
to touch it; delete it by hand and re-run.

Then the dev-split breakdowns the report compares against (held-out training
questions, not the benchmark):

```bash
bash scripts/25_dev_evaluate.sh
```

---

## 6. Report

```bash
bash scripts/30_report.sh
ls $MIQ_RESULTS/report
```

To take everything home:

```bash
bash scripts/40_bundle_results.sh
```

prints the archive path and an `ssh ... "cat <archive>" > local.tgz` command
that also works where scp is not available.

`bash scripts/status.sh` shows at any point what is running, finished or
interrupted.

---

## Troubleshooting

**`dataset artifact ... does not match`**: the artifact on disk is not the
reference data. Rebuild it with `scripts/02_build_dataset.sh`. Do not re-pin the
hash in the configs, because that would silently change what the experiment IDs
mean.

**`run directory ... already exists and is not empty`**: deliberate. Resume with
`--resume`, or use a new experiment ID.

**`missing checkpoints (train them first, or check $MIQ_RUNS)`**: `MIQ_RUNS` is
unset or not exported in this shell. Run `source ~/.bashrc`.

**CUDA OOM during training**: lower `per_device_train_batch_size` and raise
`gradient_accumulation_steps` by the same factor, which keeps the generation
batch unchanged. That is a new experiment ID, because it changes the run.
