"""Build the report from configs/report.yaml.

    python -m miqgrpo.report --spec configs/report.yaml

Writes results/report/: analysis.txt, summary.csv and the figures,
comparing the untrained base model with the three single-task models on the
official benchmark, on the held-out dev split and in training.

Headline metrics are the official harness's own numbers; breakdowns and
intervals are computed from its per-item records (see miqgrpo.analysis).
Training metrics and benchmark accuracy are never drawn on the same axes: they
are different quantities measured on different data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from .analysis import (
    METRIC,
    TASK_LABELS,
    EvalRun,
    accuracy_by,
    bootstrap_ci,
    load_eval_run,
    load_training_metrics,
    paired_differences,
    response_lengths,
    series,
)
from .breakdown import (
    CONSTANT_SMILES,
    Scored,
    generated_smiles,
    is_trivial,
    paired_split,
    property_changes,
    score_samples,
    solved_by_any,
    split_accuracy,
    units,
)
from .paths import REPORT_ROOT, RESULTS_ROOT, dataset_dir, ensure_dirs
from .plots import SERIES, delta_matrix, grouped_bar_panels, grouped_bars, line_panels

__all__ = ["build_report", "main"]

HEADLINE = (("pass_at_1", "pass@1"), ("pass_at_3", "pass@3"), ("avg_accuracy", "avg accuracy"))
TASKS = tuple(TASK_LABELS)
BASE_NAME = "Qwen2.5-0.5B-Instruct (base)"

CONSTANT_EXPLAINED = [
    "A constant answer that ignores the molecule -- 0 for every count, [] for",
    f"every index list, and the system prompt's example molecule ({CONSTANT_SMILES})",
    "for generation -- is already correct on some questions. Accuracy on those",
    "questions says little about chemistry; accuracy on all other questions does.",
    "'Non-trivial properties' scores each requested property (or constraint)",
    "separately and keeps only those the constant answer gets wrong.",
]

# Dev-split metrics logged during training (eval_ prefix), in figure order.
DEV_PANELS = (
    ("eval_correctness/other_questions", "dev: fully correct, other questions (%)"),
    ("eval_partial/nontrivial_credit", "dev: per-property credit (%)"),
)

# Training-rollout metrics, one figure row each.
TRAINING_PANELS = (
    ("partial/nontrivial_credit", "per-property credit"),
    ("correctness/other_questions", "fully correct, other questions"),
    ("shortcut/constant_answer_share", "constant answers"),
    ("completions/mean_length", "completion length (tokens)"),
)
DIAGNOSTIC_PANELS = (
    ("entropy", "policy entropy"),
    ("frac_reward_zero_std", "groups without learning signal"),
    ("none/precision", 'precision of "none" answers'),
    ("grad_norm", "gradient norm"),
)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _pp(mean: float, low: float, high: float) -> str:
    verdict = "significant" if (low > 0 or high < 0) else "not significant"
    return f"{mean * 100:+6.2f} pp  [{low * 100:+6.2f}, {high * 100:+6.2f}]  {verdict}"


def _write(path: Path, lines: list[str]) -> None:
    ensure_dirs(path.parent)
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dirs(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames += [key for key in row if key not in fieldnames]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def _sortable(key: str) -> tuple[int, float | str]:
    try:
        return (0, float(key.split("-")[0].replace("+", "")))
    except ValueError:
        return (1, key)


def _percent(points: tuple[list[int], list[float]]) -> tuple[list[int], list[float]]:
    steps, values = points
    return steps, [v * 100 for v in values]


def _paired_vs(first: EvalRun, second: EvalRun) -> dict[str, tuple[float, float, float]]:
    """Paired first - second with CIs, overall and for each task type."""
    overall, by_task = paired_differences(first.samples, second.samples)
    out = {"overall": bootstrap_ci(overall)}
    for task in TASKS:
        out[task] = bootstrap_ci(by_task.get(task, []))
    return out


def _constant_score(scored: list[Scored], task: str) -> float:
    """Official accuracy of the constant answer: 1 on the questions it solves."""
    questions = [s.question for s in scored if s.question.task == task]
    return sum(is_trivial(q) for q in questions) / len(questions) if questions else 0.0


def _split_lines(names: list[str], scored: list[list[Scored]], task: str) -> list[str]:
    """Table of accuracy on constant-solvable vs other questions for one task type."""
    splits = [split_accuracy(s, task) for s in scored]
    n_trivial, n_other = splits[0]["trivial"][1], splits[0]["other"][1]
    lines = [
        f"{TASK_LABELS[task]}  (the constant answer solves {n_trivial} of {n_trivial + n_other} questions)",
        f"  {'':36s} {'constant-solvable':>18s} {'other questions':>16s} "
        f"{'non-trivial properties':>23s} {'constant answers':>17s}",
    ]
    for name, split in zip(names, splits):
        lines.append(
            f"  {name:36s} {_pct(split['trivial'][0]):>18s} {_pct(split['other'][0]):>16s} "
            f"{_pct(split['other_units'][0]):>23s} {split['constant_share'][0] * 100:16.0f}%"
        )
    return lines


def _typical_from_dataset(artifact: str, split: str) -> dict[str, Any]:
    """Most common true value per property in a frozen training split."""
    try:
        from datasets import load_from_disk

        from .dev_eval import typical_values

        return typical_values(load_from_disk(str(dataset_dir(artifact) / "dataset"))[split])
    except (FileNotFoundError, KeyError, ValueError):
        return {}


def _default_molecules(artifact: str) -> list[str]:
    """The generation default molecules a frozen dataset was filtered against."""
    try:
        config = yaml.safe_load((dataset_dir(artifact) / "preprocessing_config.yaml").read_text())
    except FileNotFoundError:
        return []
    tasks = ((config or {}).get("generation") or {}).get("tasks") or {}
    return list((tasks.get("constraint_generation") or {}).get("default_molecules") or [])


def _typical_split(scored: list[Scored], typical: dict[str, Any]) -> dict[str, tuple[float, int]]:
    """Per-property accuracy on non-trivial count properties, typical vs atypical."""
    buckets: dict[str, list[float]] = {"typical": [], "atypical": []}
    for item in scored:
        if item.question.task != "count":
            continue
        target = item.question.target_dict
        for unit in units(item.question):
            if unit.trivial or unit.key not in typical:
                continue
            bucket = "typical" if target[unit.key] == typical[unit.key] else "atypical"
            buckets[bucket].append(item.unit_accuracy[unit.key])
    return {k: (sum(v) / len(v) if v else float("nan"), len(v)) for k, v in buckets.items()}


def _atypical_diffs(
    first: list[Scored], second: list[Scored], typical: dict[str, Any]
) -> list[float]:
    """Per question: mean accuracy change over its atypical count properties.

    Averaged within a question so that the bootstrap resamples questions, the
    unit that is independent.
    """
    other = {s.key: s for s in second}
    diffs = []
    for item in first:
        match = other.get(item.key)
        if item.question.task != "count" or match is None:
            continue
        target = item.question.target_dict
        keys = [u.key for u in units(item.question)
                if not u.trivial and u.key in typical and target[u.key] != typical[u.key]]
        if keys:
            diffs.append(sum(item.unit_accuracy[k] - match.unit_accuracy[k] for k in keys) / len(keys))
    return diffs


def _generation_lines(
    rows: list[tuple[str, EvalRun, list[Scored]]], base_scored: list[Scored], molecules: list[str]
) -> list[str]:
    """Generation accuracy split by default-molecule solvability, and answer variety."""
    solvable = {
        item.key for item in base_scored
        if item.question.task == "generation" and solved_by_any(item.question, molecules)
    }
    n_generation = sum(1 for item in base_scored if item.question.task == "generation")
    base_by_key = {item.key: item for item in base_scored}
    lines = [
        "",
        f"Constrained generation, split by whether one of the {len(molecules)} default molecules",
        "the training data was filtered against meets every constraint",
        f"({', '.join(molecules)}):",
        f"  {'':36s} {'default solves':>15s} {'no default solves':>18s} "
        f"{'distinct answers':>17s} {'most common answer':>27s}",
    ]
    diffs: dict[str, list[float]] = {}
    for display, run, scored in rows:
        generation = [item for item in scored if item.question.task == "generation"]
        with_default = [item.accuracy for item in generation if item.key in solvable]
        without = [item.accuracy for item in generation if item.key not in solvable]
        answers = [
            smiles
            for record in run.samples
            if record["doc"].get("task_type") == "generation"
            for smiles in generated_smiles(record.get("extracted_answers") or [])
        ]
        counts: dict[str | None, int] = {}
        for smiles in answers:
            counts[smiles] = counts.get(smiles, 0) + 1
        top, top_n = max(counts.items(), key=lambda kv: kv[1]) if counts else (None, 0)

        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else float("nan")

        lines.append(
            f"  {display:36s} {_pct(mean(with_default)):>15s} {_pct(mean(without)):>18s}"
            f" {len(counts):>7d} / {len(answers):<7d} {str(top)[:18]:>18s} "
            f"{top_n / max(len(answers), 1) * 100:5.1f}%"
        )
        if scored is not base_scored:
            diffs[display] = [
                item.accuracy - base_by_key[item.key].accuracy
                for item in generation
                if item.key not in solvable and item.key in base_by_key
            ]
    lines.append(f"  (n = {len(solvable)} / {n_generation - len(solvable)} questions)")
    lines.append("  Paired change vs the base model on questions no default molecule solves:")
    for display, values in diffs.items():
        lines.append(f"    {display:34s} {_pp(*bootstrap_ci(values))}")
    return lines


def _load_dev(dataset: str, name: str) -> dict[str, Any] | None:
    path = RESULTS_ROOT / "dev" / dataset / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _dev_value(result: dict[str, Any] | None, *keys: str) -> float | None:
    node: Any = result
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, (int, float)) else None


def _dev_metrics(task: str) -> list[tuple[str, tuple[str, ...]]]:
    """What the dev breakdown shows per task type: label and path in the dev JSON."""
    metrics = [("other questions", ("questions", "other", "mean"))]
    if task == "count":
        metrics += [
            ("typical values", ("properties", "typical", "mean")),
            ("atypical values", ("properties", "atypical", "mean")),
            ('all-"0" questions', ("questions", "trivial", "mean")),
        ]
    elif task == "index":
        metrics += [
            ("atom overlap", ("properties", "nontrivial_credit", "mean")),
            ('all-"[]" questions', ("questions", "trivial", "mean")),
        ]
    return metrics


def _dev_lines(spec: dict[str, Any], families: list[str], metrics: list[list[dict[str, Any]]],
               out: Path) -> list[str]:
    lines = ["", "", "HELD-OUT DEV SPLIT (training distribution, never the benchmark)", ""]
    lines.append("During training, first and last evaluation (400 questions, 4 answers each):")
    for model, records in zip(spec["models"], metrics):
        evals = [r for r in records if "eval_reward" in r]
        if not evals:
            continue
        first, last = evals[0], evals[-1]

        def cell(record: dict[str, Any], key: str) -> str:
            value = record.get(key)
            return _pct(value) if isinstance(value, (int, float)) else "-"

        lines.append(
            f"  {model['display']:36s} step {first.get('step', 0):>4} -> {last.get('step', 0):>4}   "
            f"fully correct {cell(first, 'eval_correctness/other_questions')} -> "
            f"{cell(last, 'eval_correctness/other_questions')}   "
            f"per-property {cell(first, 'eval_partial/nontrivial_credit')} -> "
            f"{cell(last, 'eval_partial/nontrivial_credit')}"
        )
    if any(metrics):
        line_panels(
            [(family if row == 0 else "", ylabel)
             for row, (_, ylabel) in enumerate(DEV_PANELS) for family in families],
            [("GRPO", SERIES[1],
              [_percent(series(records, key)) for key, _ in DEV_PANELS for records in metrics])],
            out / "dev_learning_curves.png",
            "Held-out dev questions during training (step 0 = base model)",
            columns=len(families),
            smooth=False,
        )

    rows, panels, before_values, after_values = [], [], [], []
    for model, family in zip(spec["models"], families):
        names = model.get("dev") or {}
        before = _load_dev(spec["dataset"], names.get("base", ""))
        after = _load_dev(spec["dataset"], names.get("model", ""))
        if before is None or after is None:
            continue
        chosen = _dev_metrics(model["task_type"])
        panels.append((family, [label for label, _ in chosen]))
        before_values.append([(_dev_value(before, *keys) or 0) * 100 for _, keys in chosen])
        after_values.append([(_dev_value(after, *keys) or 0) * 100 for _, keys in chosen])
        rows += [(family, label, _dev_value(before, *keys), _dev_value(after, *keys)) for label, keys in chosen]
    if not rows:
        return lines + ["", "  (no dev-split results yet: run scripts/25_dev_evaluate.sh)"]
    lines += ["", "Base model vs final model, every dev question, 8 answers each:",
              "  'typical'/'atypical': properties whose true value is / is not the",
              "  property's most common value in the training split.",
              f"  {'':36s} {'':20s} {'base':>8s} {'final':>8s}"]
    for family, label, before, after in rows:
        lines.append(f"  {family:36s} {label:20s} {_pct(before or 0):>8s} {_pct(after or 0):>8s}")
    grouped_bar_panels(
        panels,
        [("base model", SERIES[0], before_values), ("GRPO", SERIES[1], after_values)],
        out / "dev_breakdown.png",
        "Held-out dev questions: base model vs final model",
        "accuracy (%)",
        value_format="{:.1f}",
        columns=2,
    )
    return lines


def build_report(spec: dict[str, Any], out: Path) -> None:
    """Write the report for one spec (see configs/report.yaml) into out.

    spec names the base model's evaluation run (baseline), the frozen
    training set (dataset) and, per trained model, its evaluation run,
    training run and dev-split results (models). Produces
    analysis.txt (every number with its bootstrap interval),
    summary.csv and the figures listed in the README.
    """
    base = load_eval_run(spec["baseline"])
    models = [(model, load_eval_run(model["evaluation"])) for model in spec["models"]]
    if len(models) + 1 > len(SERIES):
        raise ValueError(f"{len(models) + 1} series but only {len(SERIES)} colours")
    colors = [SERIES[0]] + list(SERIES[1 : 1 + len(models)])
    base_name = spec.get("baseline_display", BASE_NAME)
    names = [base_name] + [model["display"] for model, _ in models]
    families = [model["display"].split(", ", 1)[-1] for model, _ in models]
    runs = [base] + [run for _, run in models]
    per_task = [accuracy_by(run.samples) for run in runs]
    deltas = [_paired_vs(run, base) for _, run in models]
    scored = [score_samples(run.samples) for run in runs]
    constant = {t: _constant_score(scored[0], t) for t in TASKS}

    lines = [
        "MolecularIQ: base model vs single-task GRPO models",
        "Whole official benchmark, 5,111 items, 3 sampled attempts each.",
        "",
        "Official headline metrics (from the harness):",
    ]
    for name, run in zip(names, runs):
        bits = "  ".join(f"{label}={run.headline.get(key, 0) * 100:5.2f}%" for key, label in HEADLINE)
        lines.append(f"  {name:36s} {bits}")
    lines += ["", f"{METRIC} by task type:",
              "  " + " " * 36 + "".join(f"{TASK_LABELS[t]:>24s}" for t in TASKS)]
    for name, accuracy in zip(names, per_task):
        lines.append(f"  {name:36s}" + "".join(f"{_pct(accuracy.get(t, 0)):>24s}" for t in TASKS))
    lines.append(f"  {'constant answer (reference)':36s}" + "".join(f"{_pct(constant[t]):>24s}" for t in TASKS))

    lines += ["", "Paired change vs the base model, 95% bootstrap CI over items:"]
    for (model, _), delta in zip(models, deltas):
        lines.append(f"  {model['display']}  (trained on {TASK_LABELS[model['task_type']]})")
        lines.append(f"    {'headline (all items)':24s} {_pp(*delta['overall'])}")
        for task in TASKS:
            own = "  <- own task" if task == model["task_type"] else ""
            lines.append(f"    {TASK_LABELS[task]:24s} {_pp(*delta[task])}{own}")

    lines += ["", "", "WHERE THE ACCURACY COMES FROM", ""] + CONSTANT_EXPLAINED
    for task in TASKS:
        lines += [""] + _split_lines(names, scored, task)
        lines.append("  Paired change vs the base model on the other questions:")
        for (model, _), model_scored in zip(models, scored[1:]):
            delta = bootstrap_ci(paired_split(model_scored, scored[0], task, trivial=False))
            lines.append(f"    {model['display']:34s} {_pp(*delta)}")

    count_models = [(m, sc) for (m, _), sc in zip(models, scored[1:]) if m["task_type"] == "count"]
    typical = _typical_from_dataset(spec["dataset"], "count")
    if typical and count_models:
        lines += [
            "",
            "Counting, per property (true value > 0), split by whether the true value is",
            f"the property's most common value in the training split ({spec['dataset']});",
            "a per-property default can only get 'typical' right:",
            f"  {'':36s} {'typical':>10s} {'atypical':>10s}",
        ]
        for display, model_scored in [(base_name, scored[0])] + [(m["display"], sc) for m, sc in count_models]:
            split = _typical_split(model_scored, typical)
            lines.append(
                f"  {display:36s} {_pct(split['typical'][0]):>10s} {_pct(split['atypical'][0]):>10s}"
                f"   (n = {split['typical'][1]} / {split['atypical'][1]})"
            )
        lines.append("  Paired change vs the base model on atypical values, 95% CI over questions:")
        for model, model_scored in count_models:
            lines.append(f"    {model['display']:34s} "
                         f"{_pp(*bootstrap_ci(_atypical_diffs(model_scored, scored[0], typical)))}")

    molecules = _default_molecules(spec["dataset"])
    generation_models = [(m, run, sc) for (m, run), sc in zip(models, scored[1:])
                         if m["task_type"] == "generation"]
    if molecules and generation_models:
        rows = [(base_name, base, scored[0])] + [(m["display"], run, sc) for m, run, sc in generation_models]
        lines += _generation_lines(rows, scored[0], molecules)

    properties = [property_changes(model_scored, scored[0], "count") for model_scored in scored[1:]]
    labels = sorted(set.intersection(*(set(p) for p in properties)),
                    key=lambda label: -properties[0][label][0])
    property_cells = [[bootstrap_ci(p[label][1]) for p in properties] for label in labels]
    lines += [
        "",
        "Counting, per property: change vs the base model on non-trivial properties",
        "(true count > 0), percentage points, 95% bootstrap CI over questions,",
        "* = interval excludes zero. Properties asked at least 40 times.",
        f"  {'property':36s} {'n':>4s} {'base':>7s}"
        + "".join(f"{model['label'][:14]:>16s}" for model, _ in models),
    ]
    for label, cells in zip(labels, property_cells):
        base_acc, diffs = properties[0][label]
        lines.append(
            f"  {label:36s} {len(diffs):4d} {_pct(base_acc):>7s}"
            + "".join(f"{mean * 100:+13.1f}{' *' if (low > 0 or high < 0) else '  '} "
                      for mean, low, high in cells)
        )
    lines.append(f"  With {len(labels)} properties x {len(models)} models, a few cells are expected "
                 "to exclude zero by chance alone.")

    lines += ["", "Response length per attempt (generation cap: 28,672 tokens):"]
    for name, run in zip(names, runs):
        stats = response_lengths(run.samples)
        if stats:
            lines.append(f"  {name:36s} median {stats['median']:6d}  p99 {stats['p99']:6d}  "
                         f"max {stats['max']:7d} chars")

    metrics = [load_training_metrics(model["training"]) for model, _ in models]
    lines += _dev_lines(spec, families, metrics, out)
    lines += [
        "",
        "One training seed per model: the intervals above cover item sampling,",
        "not variance across training runs.",
    ]
    _write(out / "analysis.txt", lines)

    table = []
    for index, (name, run) in enumerate(zip(names, runs)):
        row: dict[str, Any] = {"model": name, "evaluation_run_id": run.run_id}
        row.update({label: round(run.headline.get(key, 0), 5) for key, label in HEADLINE})
        row.update({f"{METRIC}:{t}": round(per_task[index].get(t, 0), 5) for t in TASKS})
        if index:
            for scope, (mean, low, high) in deltas[index - 1].items():
                row[f"delta:{scope}"] = round(mean, 5)
                row[f"delta:{scope}:ci"] = f"[{low:.5f}, {high:.5f}]"
        table.append(row)
    _write_csv(out / "summary.csv", table)

    grouped_bars(
        [label for _, label in HEADLINE],
        [(name, color, [run.headline.get(key, 0) * 100 for key, _ in HEADLINE])
         for name, color, run in zip(names, colors, runs)],
        out / "benchmark_overall.png",
        "Official MolecularIQ benchmark, whole test split (5,111 items)",
        "score (%)",
    )
    grouped_bars(
        [TASK_LABELS[t] for t in TASKS],
        [(name, color, [accuracy.get(t, 0) * 100 for t in TASKS])
         for name, color, accuracy in zip(names, colors, per_task)],
        out / "benchmark_by_task_type.png",
        "Benchmark accuracy by task type",
        f"{METRIC.replace('_', ' ')} (%)",
        reference=("constant answer: 0 / [] / the prompt's example molecule",
                   [constant[t] * 100 for t in TASKS]),
    )
    delta_matrix(
        [f"trained on\n{TASK_LABELS[model['task_type']].lower()}" for model, _ in models],
        [TASK_LABELS[t] for t in TASKS],
        [[delta[t] for t in TASKS] for delta in deltas],
        out / "transfer_matrix.png",
        f"Change in {METRIC.replace('_', ' ')} vs the base model",
    )
    grouped_bar_panels(
        [("Questions a constant answer solves", [TASK_LABELS[t] for t in TASKS]),
         ("All other questions", [TASK_LABELS[t] for t in TASKS])],
        [
            (name, color, [
                [split_accuracy(model_scored, t)["trivial"][0] * 100 for t in TASKS],
                [split_accuracy(model_scored, t)["other"][0] * 100 for t in TASKS],
            ])
            for name, color, model_scored in zip(names, colors, scored)
        ],
        out / "where_accuracy_comes_from.png",
        "Benchmark accuracy, split by whether a constant answer is already correct",
        f"{METRIC.replace('_', ' ')} (%)",
        value_format="{:.1f}",
    )
    if labels:
        delta_matrix(
            [f"{label}  (n={len(properties[0][label][1])}, base {properties[0][label][0] * 100:.1f}%)"
             for label in labels],
            [model["display"].split(", ", 1)[-1] for model, _ in models],
            property_cells,
            out / "counting_by_property.png",
            "Counting: change vs the base model, per property, non-trivial counts only",
            compact=True,
            caption="percentage points  ·  * 95% bootstrap interval over questions excludes zero",
        )
    for field, title, filename in (
        ("complexity_bin", "Accuracy by molecular complexity (Bertz)", "benchmark_by_complexity.png"),
        ("multi_task_load", "Accuracy by number of properties asked", "benchmark_by_multitask_load.png"),
    ):
        groups = [accuracy_by(run.samples, by=field) for run in runs]
        keys = sorted({k for g in groups for k in g}, key=_sortable)
        grouped_bars(
            keys,
            [(name, color, [g.get(k, 0) * 100 for k in keys]) for name, color, g in zip(names, colors, groups)],
            out / filename,
            title,
            f"{METRIC.replace('_', ' ')} (%)",
        )
    if any(metrics):
        for panels, filename, title in (
            (TRAINING_PANELS, "training_curves.png", "Training rollouts"),
            (DIAGNOSTIC_PANELS, "training_diagnostics.png", "Training diagnostics: exploration and signal"),
        ):
            line_panels(
                [(family if row == 0 else "", ylabel)
                 for row, (_, ylabel) in enumerate(panels) for family in families],
                [("GRPO", SERIES[1],
                  [series([r for r in records if "reward" in r], key)
                   for key, _ in panels for records in metrics])],
                out / filename,
                title,
                columns=len(families),
            )


def main(argv: list[str] | None = None) -> None:
    """Command line: --spec <yaml> --out <dir>."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/report.yaml"))
    parser.add_argument("--out", type=Path, default=REPORT_ROOT)
    args = parser.parse_args(argv)
    build_report(yaml.safe_load(args.spec.read_text()), args.out)


if __name__ == "__main__":
    main()
