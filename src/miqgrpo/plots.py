"""Plotting primitives for the report.

Every function here takes exactly the data it draws and the path it writes.
Nothing discovers runs on its own; miqgrpo.report decides what goes into
which figure from configs/report.yaml.

Colours come from a categorical palette validated for colour-vision
deficiencies on a light surface (adjacent-pair CVD delta E >= 9). Several slots
sit below 3:1 contrast against the surface, so every bar carries a direct value
label and every figure is accompanied by a table in the report's summary.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .paths import ensure_dirs

__all__ = ["SERIES", "delta_matrix", "grouped_bar_panels", "grouped_bars", "line_panels"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8980"
GRID = "#e6e5e1"

# Categorical slots, in fixed order. Assign by identity, never by rank.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cc7")

# Diverging scale for changes relative to the base model.
_BELOW, _NEUTRAL, _ABOVE = "#2a78d6", "#f0efec", "#e34948"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "font.size": 10,
        }
    )


def _clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="x", visible=False)


def _save(fig, path: Path) -> Path:
    ensure_dirs(path.parent)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _draw_bars(ax, groups, series, value_format: str) -> None:
    group_width = 0.8
    bar_width = group_width / len(series)
    for offset, (name, color, values) in enumerate(series):
        xs = [i - group_width / 2 + bar_width * (offset + 0.5) for i in range(len(groups))]
        bars = ax.bar(xs, values, width=bar_width * 0.88, color=color, label=name)
        for bar, value in zip(bars, values):
            ax.annotate(
                value_format.format(value),
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK_SECONDARY,
            )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    _clean(ax)


def grouped_bars(
    groups: Sequence[str],
    series: Sequence[tuple[str, str, Sequence[float]]],
    path: Path,
    title: str,
    ylabel: str,
    value_format: str = "{:.2f}",
    reference: tuple[str, Sequence[float]] | None = None,
) -> Path:
    """Bars grouped by category; series is [(name, colour, values), ...].

    reference is (label, value per group), drawn as a dashed line across
    each group -- for a baseline that is not one of the compared models.
    """
    _style()
    fig, ax = plt.subplots(
        figsize=(max(8.0, 2.4 * len(groups) + 2.5), 4.4), layout="constrained"
    )
    _draw_bars(ax, groups, series, value_format)
    if reference is not None:
        label, values = reference
        for i, value in enumerate(values):
            ax.plot([i - 0.45, i + 0.45], [value, value], color=INK, linewidth=1.4,
                    linestyle=(0, (4, 3)), label=label if i == 0 else None)
            ax.annotate(value_format.format(value), xy=(i + 0.45, value), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Legend below the axes, where no bar can cover it.
    entries = len(series) + (reference is not None)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=entries if entries <= 4 else 3)
    return _save(fig, path)


def grouped_bar_panels(
    panels: Sequence[tuple[str, Sequence[str]]],
    series: Sequence[tuple[str, str, Sequence[Sequence[float]]]],
    path: Path,
    title: str,
    ylabel: str,
    value_format: str = "{:.2f}",
    columns: int | None = None,
) -> Path:
    """Grouped bar charts side by side (wrapping after columns), independent y axes.

    panels is [(panel title, groups)]; series is [(name, colour,
    [values for each panel])]. Independent axes because the panels usually
    differ in scale by an order of magnitude, which is the point of splitting.
    """
    _style()
    columns = columns or len(panels)
    rows = -(-len(panels) // columns)
    width = columns * max(max(4.0, 1.9 * len(groups) + 1.2) for _, groups in panels)
    fig, axes = plt.subplots(rows, columns, figsize=(width, 4.6 * rows), layout="constrained", squeeze=False)
    for index in range(len(panels), rows * columns):
        axes[index // columns][index % columns].set_visible(False)
    for index, (panel_title, groups) in enumerate(panels):
        ax = axes[index // columns][index % columns]
        _draw_bars(ax, groups, [(name, color, values[index]) for name, color, values in series],
                   value_format)
        ax.set_title(panel_title)
        if index % columns == 0:
            ax.set_ylabel(ylabel)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color, _ in series]
    fig.legend(handles, [name for name, _, _ in series], loc="outside lower center",
               ncol=min(len(series), 4))
    fig.suptitle(title, fontsize=13, color=INK)
    return _save(fig, path)


def delta_matrix(
    rows: Sequence[str],
    columns: Sequence[str],
    cells: Sequence[Sequence[tuple[float, float, float]]],
    path: Path,
    title: str,
    compact: bool = False,
    caption: str | None = None,
) -> Path:
    """Change vs a reference in percentage points, with a 95% interval per cell.

    cells[r][c] is (mean, low, high) as fractions. Cells whose interval
    excludes zero carry a *, so significance is readable without colour.
    compact drops the printed interval and shrinks rows, for long tables.
    """
    _style()
    values = [[mean * 100 for mean, _, _ in row] for row in cells]
    span = max(1.0, max(abs(v) for row in values for v in row))
    cmap = LinearSegmentedColormap.from_list("delta", [_BELOW, _NEUTRAL, _ABOVE])
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

    row_height, extra = (0.34, 1.6) if compact else (1.15, 2.4)
    fig, ax = plt.subplots(figsize=(2.3 * len(columns) + 3.4, row_height * len(rows) + extra))
    ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for r, row in enumerate(cells):
        for c, (mean, low, high) in enumerate(row):
            value = mean * 100
            color = INK if abs(value) < span * 0.55 else SURFACE
            star = " *" if (low > 0 or high < 0) else ""
            if compact:
                ax.text(c, r, f"{value:+.1f}{star}", ha="center", va="center",
                        fontsize=8.5, color=color, fontweight="bold")
                continue
            ax.text(c, r - 0.12, f"{value:+.2f}{star}", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")
            ax.text(c, r + 0.2, f"[{low * 100:+.2f}, {high * 100:+.2f}]", ha="center",
                    va="center", fontsize=8, color=color)
    # 2px surface gap between adjacent cells.
    ax.set_xticks([x - 0.5 for x in range(1, len(columns))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title)
    ax.text(
        0.0, -0.9 / (row_height * len(rows) + extra),
        caption or (
            "percentage points vs the base model  ·  95% bootstrap interval over items  ·  "
            "* interval excludes zero  ·  red above, blue below"
        ),
        transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="top",
    )
    return _save(fig, path)


def _rolling_mean(values: Sequence[float], window: int = 9) -> list[float]:
    if len(values) < window:
        return list(values)
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - window // 2), min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def line_panels(
    panels: Sequence[tuple[str, str]],
    lines: Sequence[tuple[str, str, Sequence[tuple[Sequence[int], Sequence[float]]]]],
    path: Path,
    title: str,
    columns: int | None = None,
    smooth: bool = True,
) -> Path:
    """Small multiples of training curves.

    panels is [(title, ylabel)] (an empty title leaves the panel
    untitled); lines is [(legend name, colour, [(steps, values) for each
    panel])]. The raw series stays visible behind a rolling mean -- smoothing
    a noisy RL curve without showing the noise would overstate how clean it is.
    smooth=False draws the points as measured, for sparse series that are
    already averages (dev evaluations), where a rolling mean would pull later
    points into the first one.
    """
    _style()
    columns = columns or len(panels)
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.2 * rows),
                             squeeze=False, layout="constrained")
    for index, (panel_title, ylabel) in enumerate(panels):
        ax = axes[index // columns][index % columns]
        drawn = False
        for _, color, data in lines:
            steps, values = data[index]
            if not values:
                continue
            drawn = True
            if not smooth:
                ax.plot(steps, values, color=color, marker="o", markersize=3,
                        solid_capstyle="round")
                continue
            ax.plot(steps, values, color=color, linewidth=1.0, alpha=0.25)
            ax.plot(steps, _rolling_mean(values), color=color, solid_capstyle="round")
        if not drawn:
            # No such metric for this panel: say so instead of drawing empty axes.
            ax.text(0.5, 0.5, "not applicable", transform=ax.transAxes, ha="center",
                    va="center", color=INK_MUTED)
            ax.set_xticks([])
            ax.set_yticks([])
        if panel_title:
            ax.set_title(panel_title)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(ylabel)
        _clean(ax)
    for index in range(len(panels), rows * columns):
        axes[index // columns][index % columns].set_visible(False)
    handles = [plt.Line2D([], [], color=color, lw=2, label=name) for name, color, _ in lines]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(title, fontsize=13, color=INK)
    return _save(fig, path)
