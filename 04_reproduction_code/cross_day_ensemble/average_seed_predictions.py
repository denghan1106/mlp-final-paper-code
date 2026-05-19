#!/usr/bin/env python3
# coding: utf-8
"""Average point predictions from multiple result folders."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
BASE_SCRIPT = (
    ROOT
    / "mlp"
    / "20260428_20260429_mlp_localization"
    / "mlp_localization_20260428_20260429_sklearn_mlp_style.py"
)


def load_base_module():
    spec = importlib.util.spec_from_file_location("base_mlp_localization", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base script from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--result-dir", type=Path, nargs="+", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = load_base_module()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    source_dirs = []
    for result_dir in args.result_dir:
        pred_path = result_dir / "test_point_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        df = pd.read_csv(pred_path).sort_values(["dataset", "record_id"]).reset_index(drop=True)
        frames.append(df)
        source_dirs.append(str(result_dir.resolve()))

    key_cols = ["dataset", "record_id", "x", "y"]
    reference = frames[0][key_cols].copy()
    for df, result_dir in zip(frames[1:], args.result_dir[1:]):
        if not reference.equals(df[key_cols].reset_index(drop=True)):
            raise ValueError(f"Prediction keys do not match: {result_dir}")

    out = reference.copy()
    out["pred_x"] = sum(df["pred_x"].to_numpy(dtype=float) for df in frames) / len(frames)
    out["pred_y"] = sum(df["pred_y"].to_numpy(dtype=float) for df in frames) / len(frames)
    out["err_x"] = out["pred_x"] - out["x"]
    out["err_y"] = out["pred_y"] - out["y"]
    out["err_dist"] = np.hypot(out["err_x"], out["err_y"])
    out["n_models"] = len(frames)

    metrics = base.compute_metrics(out[["x", "y"]].to_numpy(), out[["pred_x", "pred_y"]].to_numpy())
    out.to_csv(args.output_dir / "test_point_predictions.csv", index=False, encoding="utf-8-sig")
    datasets = sorted(str(v) for v in out["dataset"].dropna().unique())
    if len(datasets) == 1:
        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in datasets[0])
        out.to_csv(args.output_dir / f"test_{safe_name}_point_predictions.csv", index=False, encoding="utf-8-sig")
        if safe_name == "20260429":
            out.to_csv(args.output_dir / "test_20260429_point_predictions.csv", index=False, encoding="utf-8-sig")

    result = {
        "config": {
            "algorithm": "point_prediction_ensemble_mean",
            "n_models": len(frames),
            "source_dirs": source_dirs,
        },
        "test_point_metrics": metrics,
        "test_point_metrics_by_dataset": {
            dataset: base.compute_metrics(sub[["x", "y"]].to_numpy(), sub[["pred_x", "pred_y"]].to_numpy())
            for dataset, sub in out.groupby("dataset")
        },
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    import visualize_63_results

    visualize_63_results.main_args = None
    old_argv = __import__("sys").argv
    try:
        __import__("sys").argv = [
            "visualize_63_results.py",
            "--result-dir",
            str(args.output_dir),
            "--in-place",
        ]
        visualize_63_results.main()
    finally:
        __import__("sys").argv = old_argv


if __name__ == "__main__":
    main()
