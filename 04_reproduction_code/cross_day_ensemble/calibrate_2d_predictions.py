#!/usr/bin/env python3
# coding: utf-8
"""Post-calibrate 20260429 point predictions with labeled anchors."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
BASE_SCRIPT = (
    ROOT
    / "mlp"
    / "20260428_20260429_mlp_localization"
    / "mlp_localization_20260428_20260429_sklearn_mlp_style.py"
)
DEFAULT_ANCHORS = (1, 9, 32, 55, 63)


def load_base_module():
    spec = importlib.util.spec_from_file_location("base_mlp_localization", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base script from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--calibration-record-ids", type=int, nargs="+", default=DEFAULT_ANCHORS)
    p.add_argument("--method", choices=("offset", "affine", "ridge_poly2"), default="affine")
    p.add_argument("--clip-x-max", type=float, default=8.0)
    p.add_argument("--clip-y-max", type=float, default=6.0)
    return p.parse_args()


def fit_calibrator(method: str, x_cal: np.ndarray, y_cal: np.ndarray):
    if method == "offset":
        offset = y_cal.mean(axis=0) - x_cal.mean(axis=0)
        return lambda x: x + offset, {"offset": offset.tolist()}
    if method == "affine":
        model = LinearRegression()
        model.fit(x_cal, y_cal)
        return model.predict, {"coef": model.coef_.tolist(), "intercept": model.intercept_.tolist()}
    if method == "ridge_poly2":
        model = Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1e-2)),
            ]
        )
        model.fit(x_cal, y_cal)
        return model.predict, {"degree": 2, "alpha": 1e-2}
    raise ValueError(method)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.source_dir.with_name(f"{args.source_dir.name}_cal2d_{args.method}5")
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = args.source_dir / "test_20260429_point_predictions.csv"
    if not pred_path.exists():
        pred_path = args.source_dir / "test_point_predictions.csv"
    point = pd.read_csv(pred_path)
    cal_ids = set(args.calibration_record_ids)
    point["is_calibration"] = point["record_id"].astype(int).isin(cal_ids)
    found = set(point.loc[point["is_calibration"], "record_id"].astype(int))
    if found != cal_ids:
        raise ValueError(f"Missing calibration record ids: {sorted(cal_ids - found)}")

    cal = point.loc[point["is_calibration"]].copy()
    test = point.loc[~point["is_calibration"]].copy()
    predict, fit_info = fit_calibrator(
        args.method,
        cal[["pred_x", "pred_y"]].to_numpy(dtype=float),
        cal[["x", "y"]].to_numpy(dtype=float),
    )
    pred = predict(point[["pred_x", "pred_y"]].to_numpy(dtype=float))
    pred[:, 0] = np.clip(pred[:, 0], 0.0, args.clip_x_max)
    pred[:, 1] = np.clip(pred[:, 1], 0.0, args.clip_y_max)

    out = point.copy()
    out["raw_pred_x"] = out["pred_x"]
    out["raw_pred_y"] = out["pred_y"]
    out["pred_x"] = pred[:, 0]
    out["pred_y"] = pred[:, 1]
    out["err_x"] = out["pred_x"] - out["x"]
    out["err_y"] = out["pred_y"] - out["y"]
    out["err_dist"] = np.sqrt(out["err_x"] ** 2 + out["err_y"] ** 2)

    cal_out = out.loc[out["is_calibration"]].reset_index(drop=True)
    test_out = out.loc[~out["is_calibration"]].reset_index(drop=True)
    out.to_csv(output_dir / "test_20260429_point_predictions_calibrated.csv", index=False, encoding="utf-8-sig")
    cal_out.to_csv(output_dir / "anchor_points.csv", index=False, encoding="utf-8-sig")
    test_out.to_csv(output_dir / "non_anchor_points.csv", index=False, encoding="utf-8-sig")

    result = {
        "config": {
            "source_dir": str(args.source_dir.resolve()),
            "method": args.method,
            "calibration_record_ids": list(args.calibration_record_ids),
            "fit_info": fit_info,
            "n_calibration_points": int(len(cal_out)),
            "n_non_anchor_points": int(len(test_out)),
        },
        "raw_non_anchor_metrics": base.compute_metrics(
            test[["x", "y"]].to_numpy(dtype=float),
            test[["pred_x", "pred_y"]].to_numpy(dtype=float),
        ),
        "calibrated_non_anchor_metrics": base.compute_metrics(
            test_out[["x", "y"]].to_numpy(dtype=float),
            test_out[["pred_x", "pred_y"]].to_numpy(dtype=float),
        ),
        "calibrated_anchor_metrics": base.compute_metrics(
            cal_out[["x", "y"]].to_numpy(dtype=float),
            cal_out[["pred_x", "pred_y"]].to_numpy(dtype=float),
        ),
        "calibrated_all_metrics": base.compute_metrics(
            out[["x", "y"]].to_numpy(dtype=float),
            out[["pred_x", "pred_y"]].to_numpy(dtype=float),
        ),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
