#!/usr/bin/env python3
# coding: utf-8
"""Create 63-point-only visualizations for 63-record prediction results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
(SCRIPT_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROWS = 7
COLS = 9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", type=Path, default=SCRIPT_DIR)
    p.add_argument(
        "--result-dir",
        type=Path,
        action="append",
        default=None,
        help="Specific result directory to visualize. May be provided more than once.",
    )
    p.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "visualizations_63")
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Write 63-point plots into each result directory instead of the shared output directory.",
    )
    return p.parse_args()


def prediction_csv(result_dir: Path) -> Path | None:
    candidates = (
        "test_point_predictions.csv",
        "test_20260429_point_predictions.csv",
        "predict_20260429_point_predictions.csv",
        "test_20260429_point_predictions_calibrated.csv",
    )
    for name in candidates:
        path = result_dir / name
        if path.exists():
            return path
    return None


def dataset_file_prefix(df: pd.DataFrame) -> str:
    datasets = sorted(str(v) for v in df["dataset"].dropna().unique()) if "dataset" in df.columns else []
    if len(datasets) == 1:
        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in datasets[0])
        return f"test_{safe_name}"
    return "test"


def load_anchor_ids(result_dir: Path, df: pd.DataFrame) -> set[int]:
    if "is_calibration" in df.columns:
        return set(df.loc[df["is_calibration"].astype(bool), "record_id"].astype(int))
    metrics_path = result_dir / "metrics.json"
    if not metrics_path.exists():
        return set()
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    config = metrics.get("config", {})
    ids = set()
    for key in config.get("x_anchor_keys_used") or []:
        try:
            ids.add(int(str(key).split(":")[-1]))
        except ValueError:
            pass
    return ids


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    err_x = df["pred_x"].to_numpy(dtype=float) - df["x"].to_numpy(dtype=float)
    err_y = df["pred_y"].to_numpy(dtype=float) - df["y"].to_numpy(dtype=float)
    dist = np.sqrt(err_x * err_x + err_y * err_y)
    return {
        "mean_error": float(dist.mean()),
        "median_error": float(np.median(dist)),
        "rmse_x": float(np.sqrt(np.mean(err_x * err_x))),
        "rmse_y": float(np.sqrt(np.mean(err_y * err_y))),
        "max_error": float(dist.max()),
    }


def plot_points(
    df: pd.DataFrame,
    anchor_ids: set[int],
    result_dir: Path,
    out_dir: Path,
    file_prefix: str,
    extra_names: tuple[str, ...] = (),
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(df["x"], df["y"], s=52, color="#222222", label="true")
    ax.scatter(df["pred_x"], df["pred_y"], s=58, marker="x", color="#d95f02", label="predicted")
    for _, row in df.iterrows():
        color = "#7570b3" if int(row["record_id"]) in anchor_ids else "#888888"
        ax.plot([row["x"], row["pred_x"]], [row["y"], row["pred_y"]], color=color, alpha=0.55, linewidth=1.0)
        ax.text(row["x"] + 0.05, row["y"] + 0.05, str(int(row["record_id"])), fontsize=7)
    if anchor_ids:
        anchors = df[df["record_id"].astype(int).isin(anchor_ids)]
        ax.scatter(
            anchors["x"],
            anchors["y"],
            s=120,
            facecolors="none",
            edgecolors="#1b9e77",
            linewidths=1.6,
            label="anchor",
        )
    ax.set_xlim(-0.7, COLS - 0.3)
    ax.set_ylim(-0.7, ROWS - 0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(result_dir.name)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output_paths = [out_dir / f"{file_prefix}_points_63.png", *(out_dir / name for name in extra_names)]
    for path in output_paths:
        fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmap(
    df: pd.DataFrame,
    result_dir: Path,
    out_dir: Path,
    file_prefix: str,
    extra_names: tuple[str, ...] = (),
) -> None:
    heat = np.full((ROWS, COLS), np.nan)
    for _, row in df.iterrows():
        y = int(round(float(row["y"])))
        x = int(round(float(row["x"])))
        heat[y, x] = float(row["err_dist"]) if "err_dist" in df.columns else float(
            np.hypot(row["pred_x"] - row["x"], row["pred_y"] - row["y"])
        )
    fig, ax = plt.subplots(figsize=(9, 5.8))
    im = ax.imshow(heat, origin="lower", cmap="magma", vmin=0.0)
    fig.colorbar(im, ax=ax, label="Euclidean error")
    for y in range(ROWS):
        for x in range(COLS):
            if not np.isnan(heat[y, x]):
                ax.text(x, y, f"{heat[y, x]:.1f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{result_dir.name}: 63-point error heatmap")
    fig.tight_layout()
    output_paths = [out_dir / f"{file_prefix}_heatmap_63.png", *(out_dir / name for name in extra_names)]
    for path in output_paths:
        fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_comparison(summary: list[dict[str, float | str]], out_dir: Path) -> None:
    if not summary:
        return
    names = [str(row["name"]).replace("results_", "") for row in summary]
    mean = [float(row["mean_error"]) for row in summary]
    rmse_x = [float(row["rmse_x"]) for row in summary]
    rmse_y = [float(row["rmse_y"]) for row in summary]
    y_pos = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.45 * len(summary) + 1.5)))
    ax.barh(y_pos - 0.22, mean, height=0.2, label="mean")
    ax.barh(y_pos, rmse_x, height=0.2, label="RMSE x")
    ax.barh(y_pos + 0.22, rmse_y, height=0.2, label="RMSE y")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("grid units")
    ax.set_title("63-point prediction metrics")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_metrics_63.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.in_place:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    if args.result_dir:
        result_dirs = [p.resolve() for p in args.result_dir]
    else:
        result_dirs = sorted(p for p in args.root_dir.glob("results_*") if p.is_dir())
    for result_dir in result_dirs:
        csv_path = prediction_csv(result_dir)
        if csv_path is None:
            continue
        df = pd.read_csv(csv_path)
        df = df[df["record_id"].astype(int).between(1, 63)].copy()
        if df.empty:
            continue
        if "err_dist" not in df.columns:
            df["err_dist"] = np.hypot(df["pred_x"] - df["x"], df["pred_y"] - df["y"])
        anchor_ids = load_anchor_ids(result_dir, df)
        out_dir = result_dir if args.in_place else args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        file_prefix = dataset_file_prefix(df) if args.in_place else result_dir.name
        points_extra: tuple[str, ...] = ()
        heatmap_extra: tuple[str, ...] = ()
        if args.in_place:
            # Replace the broad 18x9 test plots written by the base training script
            # with 7x9 plots that show only the 63-record region.
            points_extra = ("test_point_predictions.png",)
            heatmap_extra = ("test_error_heatmap.png",)
            if (result_dir / "predict_20260429_point_predictions.csv").exists():
                points_extra = points_extra + ("predict_20260429_point_predictions.png",)
                heatmap_extra = heatmap_extra + ("predict_20260429_error_heatmap.png",)
        plot_points(df, anchor_ids, result_dir, out_dir, file_prefix, points_extra)
        plot_heatmap(df, result_dir, out_dir, file_prefix, heatmap_extra)
        row = {"name": result_dir.name}
        row.update(compute_metrics(df))
        summary.append(row)
    if args.in_place:
        for row in summary:
            result_dir = next(p for p in result_dirs if p.name == row["name"])
            pd.DataFrame([row]).to_csv(result_dir / "comparison_metrics_63.csv", index=False, encoding="utf-8-sig")
        print("Saved 63-point visualizations into result directories.")
    else:
        pd.DataFrame(summary).to_csv(args.output_dir / "comparison_metrics_63.csv", index=False, encoding="utf-8-sig")
        plot_comparison(summary, args.output_dir)
        print(f"Saved 63-point visualizations to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
