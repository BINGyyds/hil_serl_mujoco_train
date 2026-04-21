#!/usr/bin/env python3
"""Plot local MuJoCo training metrics recorded by train_rlpd_mujoco.py."""

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


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    rolled = np.empty_like(values, dtype=np.float64)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        segment = values[start : idx + 1]
        finite = segment[np.isfinite(segment)]
        rolled[idx] = np.nan if len(finite) == 0 else float(np.mean(finite))
    return rolled


def _save_line_plot(
    *,
    x: np.ndarray,
    series: Iterable[tuple[str, np.ndarray]],
    title: str,
    ylabel: str,
    output_path: Path,
    window: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, values in series:
        if not np.isfinite(values).any():
            continue
        ax.plot(x, values, alpha=0.25, linewidth=1.0, label=f"{label} raw")
        ax.plot(
            x,
            _rolling_mean(values, window),
            linewidth=2.0,
            label=f"{label} rolling({window})",
        )

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_metrics(metrics_dir: str | Path, window: int = 20) -> list[Path]:
    metrics_dir = Path(metrics_dir).expanduser()
    csv_path = metrics_dir / "episode_metrics.csv"
    if not csv_path.exists():
        print(f"No episode metrics CSV found: {csv_path}")
        return []

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print(f"Episode metrics CSV is empty: {csv_path}")
        return []

    metrics_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(rows) + 1, dtype=np.int64)
    outputs = [
        metrics_dir / "reward_curve.png",
        metrics_dir / "success_curve.png",
        metrics_dir / "error_curve.png",
    ]

    returns = _as_float_array(rows, "episode_return")
    _save_line_plot(
        x=x,
        series=(("return", returns),),
        title="Episode Return",
        ylabel="Return",
        output_path=outputs[0],
        window=window,
    )

    success = _as_float_array(rows, "success")
    _save_line_plot(
        x=x,
        series=(("success_rate", success),),
        title="Episode Success Rate",
        ylabel="Success",
        output_path=outputs[1],
        window=window,
    )

    _save_line_plot(
        x=x,
        series=(
            ("xy_error", _as_float_array(rows, "xy_error")),
            ("z_gap", _as_float_array(rows, "z_gap")),
            ("rot_error", _as_float_array(rows, "rot_error")),
        ),
        title="Insertion Error Metrics",
        ylabel="Error",
        output_path=outputs[2],
        window=window,
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
        help="Rolling average window in episodes.",
    )
    args = parser.parse_args()
    plot_metrics(metrics_dir=args.metrics_dir, window=args.window)


if __name__ == "__main__":
    main()
