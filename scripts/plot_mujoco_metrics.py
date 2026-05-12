#!/usr/bin/env python3
"""Plot publication-style MuJoCo training metrics.

The learner writes one row per completed episode.  For paper figures, directly
plotting every raw episode makes dense vertical spikes and mixes several
physical units.  This script bins episodes by learner step, reports robust
summary curves, and exports both PNG and vector PDF figures.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    print(
        "matplotlib is required to plot metrics. "
        "Activate the training environment or install it with "
        "`python -m pip install matplotlib`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc
import numpy as np


COLORS = {
    "return": "#1f77b4",
    "success": "#2ca02c",
    "xy": "#d62728",
    "z": "#9467bd",
    "rot": "#8c564b",
    "intervention": "#ff7f0e",
}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _as_float_array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            values.append(np.nan)
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def _training_axis(rows: list[dict[str, str]]) -> tuple[np.ndarray, str]:
    learner_step = _as_float_array(rows, "learner_step")
    if np.isfinite(learner_step).any():
        return learner_step / 1e6, r"Learner steps ($\times 10^6$)"
    return np.arange(1, len(rows) + 1, dtype=np.float64), "Episode"


def _binned_summary(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int,
) -> dict[str, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        empty = np.asarray([], dtype=np.float64)
        return {
            "x": empty,
            "mean": empty,
            "median": empty,
            "q25": empty,
            "q75": empty,
            "stderr": empty,
            "count": empty,
        }

    bins = max(1, min(int(bins), x.size))
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if np.isclose(x_min, x_max):
        values = y[np.isfinite(y)]
        std = float(np.nanstd(values)) if values.size > 1 else 0.0
        count = float(values.size)
        stderr = std / np.sqrt(max(count, 1.0))
        return {
            "x": np.asarray([x_min], dtype=np.float64),
            "mean": np.asarray([float(np.nanmean(values))], dtype=np.float64),
            "median": np.asarray([float(np.nanmedian(values))], dtype=np.float64),
            "q25": np.asarray([float(np.nanpercentile(values, 25))], dtype=np.float64),
            "q75": np.asarray([float(np.nanpercentile(values, 75))], dtype=np.float64),
            "stderr": np.asarray([stderr], dtype=np.float64),
            "count": np.asarray([count], dtype=np.float64),
        }

    edges = np.linspace(x_min, x_max, bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    indices = np.searchsorted(edges, x, side="right") - 1
    indices = np.clip(indices, 0, bins - 1)

    mean = np.full(bins, np.nan, dtype=np.float64)
    median = np.full(bins, np.nan, dtype=np.float64)
    q25 = np.full(bins, np.nan, dtype=np.float64)
    q75 = np.full(bins, np.nan, dtype=np.float64)
    stderr = np.full(bins, np.nan, dtype=np.float64)
    count = np.zeros(bins, dtype=np.float64)

    for bin_index in range(bins):
        values = y[indices == bin_index]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        count[bin_index] = float(values.size)
        mean[bin_index] = float(np.nanmean(values))
        median[bin_index] = float(np.nanmedian(values))
        q25[bin_index] = float(np.nanpercentile(values, 25))
        q75[bin_index] = float(np.nanpercentile(values, 75))
        std = float(np.nanstd(values))
        stderr[bin_index] = std / np.sqrt(max(float(values.size), 1.0))

    valid = count > 0
    return {
        "x": centers[valid],
        "mean": mean[valid],
        "median": median[valid],
        "q25": q25[valid],
        "q75": q75[valid],
        "stderr": stderr[valid],
        "count": count[valid],
    }


def _style_axis(ax, *, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_quantile_curve(
    ax,
    *,
    summary: dict[str, np.ndarray],
    label: str,
    color: str,
    center: str = "median",
    band: str = "iqr",
) -> None:
    x = summary["x"]
    if x.size == 0:
        return
    y = summary[center]
    ax.plot(x, y, color=color, label=label)
    if band == "iqr":
        ax.fill_between(
            x,
            summary["q25"],
            summary["q75"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    elif band == "stderr":
        lo = y - summary["stderr"]
        hi = y + summary["stderr"]
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0)


def _save_figure(
    fig,
    *,
    output_stem: Path,
    png_dpi: int,
    save_pdf: bool,
) -> list[Path]:
    outputs = []
    png_path = output_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=png_dpi)
    outputs.append(png_path)
    if save_pdf:
        pdf_path = output_stem.with_suffix(".pdf")
        fig.savefig(pdf_path)
        outputs.append(pdf_path)
    plt.close(fig)
    for output in outputs:
        print(f"Saved {output}")
    return outputs


def _save_return_curve(
    *,
    metrics_dir: Path,
    x_label: str,
    step_x: np.ndarray,
    returns: np.ndarray,
    bins: int,
    png_dpi: int,
    save_pdf: bool,
) -> list[Path]:
    summary = _binned_summary(step_x, returns, bins=bins)
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    _plot_quantile_curve(
        ax,
        summary=summary,
        label="Episode return",
        color=COLORS["return"],
    )
    _style_axis(ax, xlabel=x_label, ylabel="Return")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.35)
    return _save_figure(
        fig,
        output_stem=metrics_dir / "reward_curve",
        png_dpi=png_dpi,
        save_pdf=save_pdf,
    )


def _save_success_curve(
    *,
    metrics_dir: Path,
    x_label: str,
    step_x: np.ndarray,
    success: np.ndarray,
    bins: int,
    png_dpi: int,
    save_pdf: bool,
) -> list[Path]:
    summary = _binned_summary(step_x, success, bins=bins)
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    _plot_quantile_curve(
        ax,
        summary=summary,
        label="Success rate",
        color=COLORS["success"],
        center="mean",
        band="stderr",
    )
    ax.set_ylim(-0.03, 1.03)
    _style_axis(ax, xlabel=x_label, ylabel="Success rate")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.35)
    return _save_figure(
        fig,
        output_stem=metrics_dir / "success_curve",
        png_dpi=png_dpi,
        save_pdf=save_pdf,
    )


def _save_training_performance(
    *,
    metrics_dir: Path,
    x_label: str,
    step_x: np.ndarray,
    returns: np.ndarray,
    success: np.ndarray,
    intervention_steps: np.ndarray,
    bins: int,
    png_dpi: int,
    save_pdf: bool,
) -> list[Path]:
    return_summary = _binned_summary(step_x, returns, bins=bins)
    success_summary = _binned_summary(step_x, success, bins=bins)
    intervention_summary = _binned_summary(step_x, intervention_steps, bins=bins)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), sharex=True)
    _plot_quantile_curve(
        axes[0],
        summary=return_summary,
        label="Return",
        color=COLORS["return"],
    )
    _style_axis(axes[0], xlabel=x_label, ylabel="Return")
    axes[0].legend(frameon=False, loc="best")

    _plot_quantile_curve(
        axes[1],
        summary=success_summary,
        label="Success rate",
        color=COLORS["success"],
        center="mean",
        band="stderr",
    )
    axes[1].set_ylim(-0.03, 1.03)
    _style_axis(axes[1], xlabel=x_label, ylabel="Success rate")
    axes[1].legend(frameon=False, loc="lower right")

    _plot_quantile_curve(
        axes[2],
        summary=intervention_summary,
        label="Intervention steps",
        color=COLORS["intervention"],
        center="mean",
        band="stderr",
    )
    _style_axis(axes[2], xlabel=x_label, ylabel="Intervention steps")
    axes[2].legend(frameon=False, loc="best")

    fig.tight_layout(pad=0.35, w_pad=1.0)
    return _save_figure(
        fig,
        output_stem=metrics_dir / "training_performance",
        png_dpi=png_dpi,
        save_pdf=save_pdf,
    )


def _save_error_curves(
    *,
    metrics_dir: Path,
    x_label: str,
    step_x: np.ndarray,
    xy_error: np.ndarray,
    z_gap: np.ndarray,
    rot_error: np.ndarray,
    bins: int,
    png_dpi: int,
    save_pdf: bool,
) -> list[Path]:
    series = [
        ("XY error", xy_error * 1000.0, "XY error (mm)", 2.0, COLORS["xy"]),
        ("Z gap", z_gap * 1000.0, "Z gap (mm)", 2.0, COLORS["z"]),
        (
            "Rotation error",
            np.degrees(rot_error),
            "Rot. error (deg)",
            np.degrees(0.1),
            COLORS["rot"],
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), sharex=True)
    for ax, (label, values, ylabel, threshold, color) in zip(axes, series):
        summary = _binned_summary(step_x, values, bins=bins)
        _plot_quantile_curve(
            ax,
            summary=summary,
            label=label,
            color=color,
        )
        ax.axhline(
            threshold,
            color="0.25",
            linestyle="--",
            linewidth=0.9,
            alpha=0.75,
            label="Success threshold",
        )
        _style_axis(ax, xlabel=x_label, ylabel=ylabel)
        ax.legend(frameon=False, loc="best")

    fig.tight_layout(pad=0.35, w_pad=1.0)
    return _save_figure(
        fig,
        output_stem=metrics_dir / "error_curve",
        png_dpi=png_dpi,
        save_pdf=save_pdf,
    )


def _save_intervention_curves(
    *,
    metrics_dir: Path,
    x_label: str,
    step_x: np.ndarray,
    intervention_steps: np.ndarray,
    bins: int,
    png_dpi: int,
    save_pdf: bool,
) -> list[Path]:
    if not np.isfinite(intervention_steps).any():
        return []

    intervention_summary = _binned_summary(step_x, intervention_steps, bins=bins)

    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    _plot_quantile_curve(
        ax,
        summary=intervention_summary,
        label="Intervention steps",
        color=COLORS["intervention"],
        center="mean",
        band="stderr",
    )
    _style_axis(ax, xlabel=x_label, ylabel="Steps / episode")
    ax.legend(frameon=False, loc="best")

    fig.tight_layout(pad=0.35)
    return _save_figure(
        fig,
        output_stem=metrics_dir / "intervention_metrics",
        png_dpi=png_dpi,
        save_pdf=save_pdf,
    )


def plot_metrics(
    metrics_dir: str | Path,
    window: int = 20,
    *,
    bins: int = 300,
    png_dpi: int = 600,
    save_pdf: bool = True,
) -> list[Path]:
    """Generate publication-style plots from ``episode_metrics.csv``.

    ``window`` is kept for backward compatibility with older callers.  The
    current plotting path uses learner-step binning instead of episode rolling
    averages because the training logs can contain more than 100k episodes.
    """
    del window
    _configure_matplotlib()

    metrics_dir = Path(metrics_dir).expanduser()
    csv_path = metrics_dir / "episode_metrics.csv"
    if not csv_path.exists():
        print(f"No episode metrics CSV found: {csv_path}")
        return []

    rows = _load_rows(csv_path)
    if not rows:
        print(f"Episode metrics CSV is empty: {csv_path}")
        return []

    metrics_dir.mkdir(parents=True, exist_ok=True)
    step_x, x_label = _training_axis(rows)
    bins = max(10, int(bins))

    returns = _as_float_array(rows, "episode_return")
    success = _as_float_array(rows, "success")
    xy_error = _as_float_array(rows, "xy_error")
    z_gap = _as_float_array(rows, "z_gap")
    rot_error = _as_float_array(rows, "rot_error")
    intervention_steps = _as_float_array(rows, "intervention_steps")

    outputs: list[Path] = []
    outputs.extend(
        _save_return_curve(
            metrics_dir=metrics_dir,
            x_label=x_label,
            step_x=step_x,
            returns=returns,
            bins=bins,
            png_dpi=png_dpi,
            save_pdf=save_pdf,
        )
    )
    outputs.extend(
        _save_success_curve(
            metrics_dir=metrics_dir,
            x_label=x_label,
            step_x=step_x,
            success=success,
            bins=bins,
            png_dpi=png_dpi,
            save_pdf=save_pdf,
        )
    )
    outputs.extend(
        _save_error_curves(
            metrics_dir=metrics_dir,
            x_label=x_label,
            step_x=step_x,
            xy_error=xy_error,
            z_gap=z_gap,
            rot_error=rot_error,
            bins=bins,
            png_dpi=png_dpi,
            save_pdf=save_pdf,
        )
    )
    outputs.extend(
        _save_training_performance(
            metrics_dir=metrics_dir,
            x_label=x_label,
            step_x=step_x,
            returns=returns,
            success=success,
            intervention_steps=intervention_steps,
            bins=bins,
            png_dpi=png_dpi,
            save_pdf=save_pdf,
        )
    )
    outputs.extend(
        _save_intervention_curves(
            metrics_dir=metrics_dir,
            x_label=x_label,
            step_x=step_x,
            intervention_steps=intervention_steps,
            bins=bins,
            png_dpi=png_dpi,
            save_pdf=save_pdf,
        )
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot local MuJoCo insertion training metrics."
    )
    parser.add_argument(
        "--metrics_dir",
        required=True,
        help="Directory containing episode_metrics.csv.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Deprecated rolling window argument kept for compatibility.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=300,
        help="Number of learner-step bins used for publication curves.",
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=600,
        help="DPI for PNG exports.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Disable vector PDF exports.",
    )
    args = parser.parse_args()
    plot_metrics(
        metrics_dir=args.metrics_dir,
        window=args.window,
        bins=args.bins,
        png_dpi=args.png_dpi,
        save_pdf=not args.no_pdf,
    )


if __name__ == "__main__":
    main()
