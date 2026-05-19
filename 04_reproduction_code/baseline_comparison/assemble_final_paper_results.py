#!/usr/bin/env python3
# coding: utf-8
"""Assemble the final requested 20260430_true_inputs baseline comparisons.

Final scope:
1. Single-sensor vs dual-sensor improved MLP on the same three 162-point
   validation splits used by the reference folders.
2. Dual-sensor model baselines on those same validation splits.
3. Dual-sensor model baselines for 20260430_true_inputs -> 20260429 only,
   keeping the same 29 target calibration records and evaluating the remaining
   34 records.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
BASE_SCRIPT = ROOT / "mlp" / "20260428_20260429_mlp_localization" / "mlp_localization_20260428_20260429_sklearn_mlp_style.py"
SAME_162_DIR = SCRIPT_DIR / "final_20260430_baselines" / "same_162"
OUTPUT_ROOT = SCRIPT_DIR / "final_20260430_baselines" / "requested_only_20260514"

LOCALIZATION_DIR = ROOT / "mlp" / "20260428_20260429_mlp_localization"
ANALYSIS_DIR = ROOT / "mlp" / "20260430_mlp_analysis"
CROSS_DIR = ROOT / "mlp" / "20260430true162_predict_20260429_anchor"

TRAIN_0430_DIR = ROOT / "mlp" / "20260430_true_inputs"
TARGET_0429_DIR = ROOT / "mlp" / "20260429"

CALIBRATION_IDS_0429 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    18,
    19,
    27,
    28,
    32,
    36,
    37,
    45,
    46,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
)


@dataclass(frozen=True)
class SameSplitReference:
    name: str
    ref_dir: Path


SAME_SPLITS = (
    SameSplitReference("validation_inner6", LOCALIZATION_DIR / "results_sklearn_mlp_style_inner_6pts_good"),
    SameSplitReference("validation_split1", ANALYSIS_DIR / "results_162_w120_blend30_anchor_scale_seed1_good"),
    SameSplitReference("validation_split7", ANALYSIS_DIR / "results_162_w120_blend30_anchor_auto5_seed7_good"),
)

CROSS_IMPROVED_REF = CROSS_DIR / "results_joint_record_median_calib29_w2_alpha0001_seed1_10_ensemble_good"
SAME_METHODS = ("plain_mlp", "grnn", "extra_trees")
CROSS_METHODS = ("plain_mlp", "grnn", "extra_trees")


def load_base_module():
    spec = importlib.util.spec_from_file_location("base_mlp_localization", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base script from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


class GRNNRegressor:
    """Gaussian-kernel Nadaraya-Watson/GRNN regressor."""

    def __init__(
        self,
        bandwidth: float | None = None,
        bandwidth_scale: float = 1.5,
        bandwidth_subset_size: int = 1200,
        chunk_size: int = 512,
        random_state: int = 42,
    ) -> None:
        self.bandwidth = bandwidth
        self.bandwidth_scale = bandwidth_scale
        self.bandwidth_subset_size = bandwidth_subset_size
        self.chunk_size = chunk_size
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "GRNNRegressor":
        self.scaler_ = StandardScaler()
        self.x_train_ = self.scaler_.fit_transform(np.asarray(x, dtype=float))
        self.y_train_ = np.asarray(y, dtype=float)
        self.sample_weight_ = np.ones(len(self.x_train_), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.bandwidth_ = float(self.bandwidth) if self.bandwidth is not None else self._estimate_bandwidth()
        if not np.isfinite(self.bandwidth_) or self.bandwidth_ <= 0:
            raise ValueError(f"Bad GRNN bandwidth: {self.bandwidth_}")
        return self

    def _estimate_bandwidth(self) -> float:
        n = len(self.x_train_)
        if n <= 2:
            return 1.0
        rng = np.random.default_rng(self.random_state)
        take = min(n, int(self.bandwidth_subset_size))
        sample = self.x_train_[rng.choice(n, size=take, replace=False)]
        k = min(6, len(sample))
        nn = NearestNeighbors(n_neighbors=k).fit(sample)
        distances, _ = nn.kneighbors(sample)
        positive = distances[:, -1][distances[:, -1] > 1e-9]
        return 1.0 if len(positive) == 0 else float(np.median(positive) * self.bandwidth_scale)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler_.transform(np.asarray(x, dtype=float))
        train_norm = np.sum(self.x_train_ * self.x_train_, axis=1)
        out = np.empty((len(x_scaled), self.y_train_.shape[1]), dtype=float)
        sigma2 = self.bandwidth_ * self.bandwidth_
        for start in range(0, len(x_scaled), self.chunk_size):
            chunk = x_scaled[start : start + self.chunk_size]
            d2 = np.sum(chunk * chunk, axis=1, keepdims=True) + train_norm[None, :] - 2.0 * chunk @ self.x_train_.T
            logits = -0.5 * np.maximum(d2, 0.0) / sigma2
            logits -= logits.max(axis=1, keepdims=True)
            weights = np.exp(logits) * self.sample_weight_[None, :]
            out[start : start + len(chunk)] = weights @ self.y_train_ / np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--plain-alpha", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--grnn-bandwidth", type=float, default=None)
    parser.add_argument("--grnn-bandwidth-scale", type=float, default=1.5)
    parser.add_argument("--extra-trees-n-estimators", type=int, default=300)
    parser.add_argument("--extra-trees-min-samples-leaf", type=int, default=2)
    return parser.parse_args()


def load_metrics(result_dir: Path) -> dict:
    with (result_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_row(group: str, split: str, model: str, result_dir: Path, note: str = "") -> dict:
    metrics = load_metrics(result_dir)
    point = metrics["test_point_metrics"]
    test_points = metrics["config"].get("test_points")
    if test_points is None and (result_dir / "test_point_predictions.csv").exists():
        test_points = int(pd.read_csv(result_dir / "test_point_predictions.csv").shape[0])
    return {
        "comparison_group": group,
        "split": split,
        "model": model,
        "mean_euclidean_error": point["mean_euclidean_error"],
        "median_euclidean_error": point["median_euclidean_error"],
        "rmse_x": point["rmse_x"],
        "rmse_y": point["rmse_y"],
        "max_euclidean_error": point["max_euclidean_error"],
        "test_points": test_points,
        "result_dir": str(result_dir),
        "note": note,
    }


def copy_result_dir(src: Path, dst: Path) -> None:
    if not (src / "metrics.json").exists():
        raise FileNotFoundError(f"Missing metrics.json in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def write_summary(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    metric_cols = ["mean_euclidean_error", "median_euclidean_error", "rmse_x", "rmse_y", "max_euclidean_error"]
    display = df.copy()
    for col in metric_cols:
        display[col] = display[col].map(lambda v: f"{float(v):.6f}")
    output_path.with_suffix(".md").write_text(display.to_markdown(index=False), encoding="utf-8")


def validate_same_split_keys() -> None:
    for item in SAME_SPLITS:
        ref_keys = load_metrics(item.ref_dir)["config"]["test_keys"]
        for method in ("sensor1_current_improved_mlp", *SAME_METHODS):
            result_dir = SAME_162_DIR / item.name / method
            keys = load_metrics(result_dir)["config"]["test_keys"]
            if list(keys) != list(ref_keys):
                raise ValueError(f"{item.name}/{method} does not match reference test_keys")


def organize_same_split_results(output_root: Path) -> list[dict]:
    validate_same_split_keys()
    rows: list[dict] = []
    sensor_group = output_root / "sensor_vs_dual_improved_same_split"
    model_group = output_root / "dual_sensor_models_same_split"

    for item in SAME_SPLITS:
        dual_dst = sensor_group / item.name / "dual_sensor_improved_mlp"
        single_dst = sensor_group / item.name / "single_sensor_improved_mlp"
        copy_result_dir(item.ref_dir, dual_dst)
        copy_result_dir(SAME_162_DIR / item.name / "sensor1_current_improved_mlp", single_dst)
        rows.append(metric_row("sensor_vs_dual_improved_same_split", item.name, "dual_sensor_improved_mlp", dual_dst, "reference improved MLP"))
        rows.append(metric_row("sensor_vs_dual_improved_same_split", item.name, "single_sensor_improved_mlp", single_dst, "right-side four-channel sensor only"))

        improved_dst = model_group / item.name / "improved_mlp"
        copy_result_dir(item.ref_dir, improved_dst)
        rows.append(metric_row("dual_sensor_models_same_split", item.name, "improved_mlp", improved_dst, "reference improved MLP"))
        for method in SAME_METHODS:
            dst = model_group / item.name / method
            copy_result_dir(SAME_162_DIR / item.name / method, dst)
            rows.append(metric_row("dual_sensor_models_same_split", item.name, method, dst))

    return rows


def cross_base_args(rows: int, data_dir: Path, output_dir: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=[data_dir],
        output_dir=output_dir or Path("."),
        rows=rows,
        cols=9,
        grid_step=1.0,
        left_sensor_x=3.0,
        left_sensor_y=-2.0,
        right_sensor_x=5.0,
        right_sensor_y=-2.0,
        reference_x=4.0,
        reference_y=-2.0,
        target_mode="center_relative",
        row_offsets=(),
        split_mode="keys",
        split_unit="dataset_record",
        test_size=0.0,
        max_edge_test_fraction=0.0,
        split_random_state=42,
        randomize_split=False,
        constrained_max_edge_test_fraction=0.25,
        constrained_max_attempts=10000,
        exclude_corner_test=False,
        test_keys=[],
        window_size=None,
        stride=None,
        signal_normalization="record_median",
        fundamental_freq=15.0,
        dataset_fundamental_freqs=(),
        harmonic_orders=(1, 2, 3, 4),
        max_iter=1000,
        model_random_state=42,
        hidden_layers=(128, 128, 64),
        alpha=1e-3,
        separate_x_model=False,
        x_model_type="window_mlp",
        x_blend_classifier_weight=0.0,
        augment_x_mirror=False,
        x_hidden_layers=(128, 128, 64),
        x_alpha=1e-4,
        x_model_random_state=42,
        x_ensemble_seeds=(),
        x_stretch_calibration="none",
        x_stretch_y_bins=3,
        x_anchor_record_ids=(),
        x_anchor_auto_count=0,
        x_anchor_correction="none",
        x_anchor_exclude_from_metrics=False,
        edge_sample_weight=1.0,
        near_edge_sample_weight=1.0,
        no_early_stopping=False,
        training_unit="window",
        max_windows_per_point=120,
        feature_set="all",
    )


def target_mask_by_ids(frame: pd.DataFrame, ids: tuple[int, ...], invert: bool = False) -> pd.Series:
    mask = frame["record_id"].astype(int).isin(set(ids))
    return ~mask if invert else mask


def make_weight_vector(frame: pd.DataFrame, calibration_dataset: str, calibration_weight: float) -> np.ndarray:
    weights = np.ones(len(frame), dtype=float)
    weights[frame["dataset"].astype(str).eq(calibration_dataset).to_numpy()] = float(calibration_weight)
    return weights


def metric_dict(point_df: pd.DataFrame) -> dict[str, float]:
    return base.compute_metrics(point_df[["x", "y"]].to_numpy(), point_df[["pred_x", "pred_y"]].to_numpy())


def save_cross_result(
    out_dir: Path,
    method: str,
    train_point: pd.DataFrame,
    test_point: pd.DataFrame,
    config: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_point.to_csv(out_dir / "train_point_predictions.csv", index=False, encoding="utf-8-sig")
    test_point.to_csv(out_dir / "test_point_predictions.csv", index=False, encoding="utf-8-sig")
    test_point.to_csv(out_dir / "test_20260429_point_predictions.csv", index=False, encoding="utf-8-sig")
    base.save_plots(train_point, out_dir, "train", f"{method}: train true vs predicted point mean", 18, 9)
    base.save_plots(test_point, out_dir, "test", f"{method}: 20260429 true vs predicted point mean", 7, 9)
    metrics = {
        "config": config,
        "train_point_metrics": metric_dict(train_point),
        "test_point_metrics": metric_dict(test_point),
        "test_point_metrics_by_dataset": {"20260429": metric_dict(test_point)},
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def run_cross_method(method: str, out_dir: Path, args: argparse.Namespace) -> None:
    train_args = cross_base_args(18, TRAIN_0430_DIR, out_dir)
    target_args = cross_base_args(7, TARGET_0429_DIR, out_dir)
    train_features, train_meta = base.build_features(train_args)
    target_features, target_meta = base.build_features(target_args)
    cal_features = target_features.loc[target_mask_by_ids(target_features, CALIBRATION_IDS_0429)].reset_index(drop=True)
    test_features = target_features.loc[target_mask_by_ids(target_features, CALIBRATION_IDS_0429, invert=True)].reset_index(drop=True)
    expected_test_ids = [rid for rid in range(1, 64) if rid not in set(CALIBRATION_IDS_0429)]
    actual_test_ids = sorted(test_features["record_id"].astype(int).unique().tolist())
    if actual_test_ids != expected_test_ids:
        raise ValueError(f"Unexpected 20260429 test ids: {actual_test_ids}")

    feature_cols = base.feature_columns(train_features, "all")
    config = {
        "method": method,
        "data_dirs": [str(TRAIN_0430_DIR.resolve()), str(TARGET_0429_DIR.resolve())],
        "train_dataset": "20260430_true_inputs",
        "predict_dataset": "20260429",
        "rows": 18,
        "predict_rows": 7,
        "cols": 9,
        "target_mode": "center_relative",
        "feature_set": "all",
        "feature_count": len(feature_cols),
        "harmonic_orders": [1, 2, 3, 4],
        "signal_normalization": "record_median",
        "max_windows_per_point": 120,
        "calibration_dataset": "20260429",
        "calibration_record_ids": list(CALIBRATION_IDS_0429),
        "calibration_sample_weight": 2.0,
        "test_keys": [f"20260429:{rid}" for rid in expected_test_ids],
        "train_points": int(train_features[["dataset", "record_id"]].drop_duplicates().shape[0] + len(CALIBRATION_IDS_0429)),
        "test_points": len(expected_test_ids),
        "dataset_feature_settings": {**train_meta, **target_meta},
    }

    if method == "plain_mlp":
        train_window = base.sample_windows_per_point(train_features, 120, args.random_state)
        cal_window = base.sample_windows_per_point(cal_features, 120, args.random_state)
        test_window = base.sample_windows_per_point(test_features, 120, args.random_state)
        fit_frame = pd.concat([train_window, cal_window], ignore_index=True)
        x_train = fit_frame[feature_cols].to_numpy(dtype=float)
        y_train, _ = base.make_targets(fit_frame, train_args)
        x_test = test_window[feature_cols].to_numpy(dtype=float)
        sample_weight = make_weight_vector(fit_frame, TARGET_0429_DIR.name, 2.0)
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 128, 64),
                        activation="relu",
                        solver="adam",
                        alpha=args.plain_alpha,
                        batch_size=256,
                        learning_rate_init=1e-3,
                        max_iter=args.max_iter,
                        early_stopping=True,
                        n_iter_no_change=15,
                        random_state=args.random_state,
                        verbose=False,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train, mlp__sample_weight=sample_weight)
        train_pred = base.pred_to_abs(model.predict(x_train), train_args)
        test_pred = base.pred_to_abs(model.predict(x_test), target_args)
        train_point = base.point_predictions(fit_frame, train_pred)
        test_point = base.point_predictions(test_window, test_pred)
        config.update({"training_unit": "window", "alpha": args.plain_alpha, "random_state": args.random_state})
    else:
        train_point_features = base.aggregate_point_mean_features(train_features, feature_cols)
        cal_point_features = base.aggregate_point_mean_features(cal_features, feature_cols)
        test_point_features = base.aggregate_point_mean_features(test_features, feature_cols)
        fit_frame = pd.concat([train_point_features, cal_point_features], ignore_index=True)
        x_train = fit_frame[feature_cols].to_numpy(dtype=float)
        y_train, _ = base.make_targets(fit_frame, train_args)
        x_test = test_point_features[feature_cols].to_numpy(dtype=float)
        sample_weight = make_weight_vector(fit_frame, TARGET_0429_DIR.name, 2.0)
        if method == "grnn":
            model = GRNNRegressor(
                bandwidth=args.grnn_bandwidth,
                bandwidth_scale=args.grnn_bandwidth_scale,
                random_state=args.random_state,
            )
            model.fit(x_train, y_train, sample_weight=sample_weight)
            config.update({"training_unit": "point_mean", "bandwidth": model.bandwidth_, "bandwidth_scale": args.grnn_bandwidth_scale})
        elif method == "extra_trees":
            model = ExtraTreesRegressor(
                n_estimators=args.extra_trees_n_estimators,
                random_state=args.random_state,
                min_samples_leaf=args.extra_trees_min_samples_leaf,
                max_features=0.8,
                n_jobs=-1,
            )
            model.fit(x_train, y_train, sample_weight=sample_weight)
            config.update(
                {
                    "training_unit": "point_mean",
                    "n_estimators": args.extra_trees_n_estimators,
                    "min_samples_leaf": args.extra_trees_min_samples_leaf,
                    "max_features": 0.8,
                }
            )
        else:
            raise ValueError(f"Unsupported cross method: {method}")
        train_pred = base.pred_to_abs(model.predict(x_train), train_args)
        test_pred = base.pred_to_abs(model.predict(x_test), target_args)
        train_point = base.point_predictions(fit_frame, train_pred)
        test_point = base.point_predictions(test_point_features, test_pred)

    save_cross_result(out_dir, method, train_point, test_point, config)


def organize_cross_results(output_root: Path, args: argparse.Namespace) -> list[dict]:
    group = output_root / "dual_sensor_models_cross_20260429" / "cross_20260429"
    improved_dst = group / "improved_mlp"
    copy_result_dir(CROSS_IMPROVED_REF, improved_dst)
    rows = [metric_row("dual_sensor_models_cross_20260429", "cross_20260429", "improved_mlp", improved_dst, "existing calibrated ensemble reference")]

    ref_test = pd.read_csv(CROSS_IMPROVED_REF / "test_point_predictions.csv")
    ref_ids = sorted(ref_test["record_id"].astype(int).tolist())
    expected_ids = [rid for rid in range(1, 64) if rid not in set(CALIBRATION_IDS_0429)]
    if ref_ids != expected_ids:
        raise ValueError(f"Reference cross result does not match expected 34 non-calibration ids: {ref_ids}")

    for method in CROSS_METHODS:
        out_dir = group / method
        run_cross_method(method, out_dir, args)
        pred_ids = sorted(pd.read_csv(out_dir / "test_point_predictions.csv")["record_id"].astype(int).tolist())
        if pred_ids != ref_ids:
            raise ValueError(f"{method} cross ids do not match reference ids")
        rows.append(metric_row("dual_sensor_models_cross_20260429", "cross_20260429", method, out_dir, "29 calibration records, weight=2"))
    return rows


def write_manifest(output_root: Path, rows: list[dict]) -> None:
    manifest = {
        "scope": [
            "Single-sensor vs dual-sensor improved MLP on the same three 20260430_true_inputs 162-point validation splits.",
            "Dual-sensor improved/plain MLP/GRNN/ExtraTrees on the same validation splits.",
            "Dual-sensor improved/plain MLP/GRNN/ExtraTrees for 20260430_true_inputs -> 20260429 only, with 29 calibration records and 34 evaluated records.",
        ],
        "same_split_references": {item.name: str(item.ref_dir) for item in SAME_SPLITS},
        "cross_reference": str(CROSS_IMPROVED_REF),
        "calibration_record_ids_20260429": list(CALIBRATION_IDS_0429),
        "summary_rows": len(rows),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    same_rows = organize_same_split_results(args.output_root)
    cross_rows = organize_cross_results(args.output_root, args)
    all_rows = same_rows + cross_rows

    write_summary([r for r in all_rows if r["comparison_group"] == "sensor_vs_dual_improved_same_split"], args.output_root / "sensor_vs_dual_improved_same_split_summary")
    write_summary([r for r in all_rows if r["comparison_group"] == "dual_sensor_models_same_split"], args.output_root / "dual_sensor_models_same_split_summary")
    write_summary([r for r in all_rows if r["comparison_group"] == "dual_sensor_models_cross_20260429"], args.output_root / "dual_sensor_models_cross_20260429_summary")
    write_summary(all_rows, args.output_root / "all_final_baselines_summary")
    write_manifest(args.output_root, all_rows)
    print(f"Saved final requested baselines to: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
