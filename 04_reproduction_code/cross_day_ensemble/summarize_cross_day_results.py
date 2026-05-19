#!/usr/bin/env python3
# coding: utf-8
"""Print a compact metric table for this experiment directory."""

from __future__ import annotations

import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def fmt(metrics: dict | None) -> str:
    if not metrics:
        return "n/a"
    return (
        f"mean={metrics['mean_euclidean_error']:.3f}, "
        f"median={metrics['median_euclidean_error']:.3f}, "
        f"rmse_x={metrics['rmse_x']:.3f}, "
        f"rmse_y={metrics['rmse_y']:.3f}, "
        f"max={metrics['max_euclidean_error']:.3f}"
    )


def main() -> None:
    for path in sorted(SCRIPT_DIR.glob("results_*/metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("config", {})
        print(path.parent.name)
        if "calibrated_non_anchor_metrics" in data:
            print(f"  2d calibration: {cfg.get('method')} {cfg.get('calibration_record_ids')}")
            print(f"  raw non-anchor: {fmt(data.get('raw_non_anchor_metrics'))}")
            print(f"  calibrated non-anchor: {fmt(data.get('calibrated_non_anchor_metrics'))}")
            print(f"  calibrated all: {fmt(data.get('calibrated_all_metrics'))}")
        else:
            print(f"  anchor: {cfg.get('x_anchor_correction')} {cfg.get('x_anchor_keys_used')}")
            metrics = data.get("test_point_metrics") or data.get("predict_point_metrics")
            label = "all/eval"
            if data.get("predict_point_metrics") and not data.get("test_point_metrics"):
                label = "predict"
            print(f"  {label}: {fmt(metrics)}")
            print(f"  non-anchor: {fmt(data.get('test_point_metrics_non_anchor'))}")


if __name__ == "__main__":
    main()
