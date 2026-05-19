#!/usr/bin/env python3
# coding: utf-8
"""Create the requested 20260430 baseline comparison layout.

This script organizes two same-split comparisons and runs the missing cross-test
comparison:

1. Single-sensor vs dual-sensor improved MLP on the same 20260430_true_inputs
   validation test points.
2. Dual-sensor model comparison on the same 20260430_true_inputs validation
   test points, plus cross-test predictions from all 162 20260430 points to
   external 63-point test datasets.

Cross-test protocol: train on all 162 points in mlp/20260430_true_inputs and
predict all 63 records in each target dataset, without target calibration
points.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
BASE_SCRIPT = (
    ROOT
    / "mlp"
    / "20260428_20260429_mlp_localization"
    / "mlp_localization_20260428_20260429_sklearn_mlp_style.py"
)
ANALYSIS_DIR = ROOT / "mlp" / "20260430_mlp_analysis"
SAME_SPLIT_SOURCE = SCRIPT_DIR / "20260430_true_inputs_results"


@dataclass(frozen=True)
class SameSplitRef:
    name: str
    improved_dir: Path


@dataclass(frozen=True)
class CrossTarget:
    name: str
    data_dir: Path
    rows: int = 7
    cols: int = 9
    dataset_freqs: tuple[str, ...] = ()


SAME_SPLITS = (
    SameSplitRef("validation_split1", ANALYSIS_DIR / "results_162_w120_blend30_anchor_scale_seed1_good"),
    SameSplitRef("validation_split7", ANALYSIS_DIR / "results_162_w120_blend30_anchor_auto5_seed7_good"),
    SameSplitRef("validation_split10", ANALYSIS_DIR / "results_162_w120_blend30_anchor_auto5_seed7"),
)

CROSS_TARGETS = (
    CrossTarget("cross_20260429", ROOT / "mlp" / "20260429"),
    CrossTarget("cross_20260428", ROOT / "mlp" / "20260428"),
    CrossTarget("cross_20260429_2", ROOT / "mlp" / "20260429_2", dataset_freqs=("20260429_2:12",)),
)


def load_base_module():
    spec = importlib.util.spec_from_file_location("base_mlp_localization", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base script from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


class GRNNRegressor:
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
        self.sample_weight_ = (
            np.ones(len(self.x_train_), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        self.bandwidth_ = float(self.bandwidth) if self.bandwidth is not None else self._estimate_bandwidth()
        if not np.isfinite(self.bandwidth_) or self.bandwidth_ <= 0:
            raise ValueError(f"Bad bandwidth: {self.bandwidth_}")
        return self

    def _estimate_bandwidth(self) -> float:
        n = len(self.x_train_)
        if n <= 2:
            return 1.0
        rng = np.random.default_rng(self.random_state)
        sample = self.x_train_[rng.choice(n, size=min(n, self.bandwidth_subset_size), replace=False)]
        k = min(6, len(sample))
        distances, _ = NearestNeighbors(n_neighbors=k).fit(sample).kneighbors(sample)
        ref = distances[:, -1] if k > 1 else distances[:, 0]
        positive = ref[ref > 1e-9]
        return 1.0 if len(positive) == 0 else float(np.median(positive) * self.bandwidth_scale)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler_.transform(np.asarray(x, dtype=float))
        train_norm = np.sum(self.x_train_ * self.x_train_, axis=1)
        out = np.empty((len(x_scaled), self.y_train_.shape[1]), dtype=float)
        sigma2 = self.bandwidth_ * self.bandwidth_
        for start in range(0, len(x_scaled), self.chunk_size):
            chunk = x_scaled[start : start + self.chunk_size]
            d2 = (
                np.sum(chunk * chunk, axis=1, keepdims=True)
                + train_norm[None, :]
                - 2.0 * chunk @ self.x_train_.T
            )
            logits = -0.5 * np.maximum(d2, 0.0) / sigma2
            logits -= logits.max(axis=1, keepdims=True)
            weights = np.exp(logits) * self.sample_weight_[None, :]
            out[start : start + len(chunk)] = weights @ self.y_train_ / np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "requested_20260430_baselines")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--plain-alpha", type=float, default=1e-3)
    p.add_argument("--grnn-bandwidth", type=float, default=None)
    p.add_argument("--grnn-bandwidth-scale", type=float, default=1.5)
    p.add_argument("--extra-trees-n-estimators", type=int, default=300)
    p.add_argument("--extra-trees-min-samples-leaf", type=int, default=2)
    p.add_argument("--skip-existing-cross", action="store_true")
    return p.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def metric_row(group: str, split_or_target: str, method: str, result_dir: Path, metrics: dict) -> dict:
    test = metrics["test_point_metrics"]
    train = metrics.get("train_point_metrics", {})
    non_anchor = metrics.get("test_point_metrics_non_anchor") or {}
    return {
        "group": group,
        "split_or_target": split_or_target,
        "method": method,
        "train_points": metrics.get("config", {}).get("train_points"),
        "test_points": metrics.get("config", {}).get("test_points"),
        "train_mean_error": train.get("mean_euclidean_error"),
        "test_mean_error": test["mean_euclidean_error"],
        "test_median_error": test["median_euclidean_error"],
        "test_rmse_x": test["rmse_x"],
        "test_rmse_y": test["rmse_y"],
        "test_max_error": test["max_euclidean_error"],
        "test_non_anchor_mean_error": non_anchor.get("mean_euclidean_error"),
        "result_dir": str(result_dir.resolve()),
    }


def copy_result(src: Path, dst: Path, *, method: str, group: str, split_name: str) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    for filename in (
        "train_point_predictions.csv",
        "test_point_predictions.csv",
        "train_20260430_true_inputs_point_predictions.csv",
        "test_20260430_true_inputs_point_predictions.csv",
    ):
        source_file = src / filename
        if source_file.exists():
            shutil.copy2(source_file, dst / filename)
    metrics = read_json(src / "metrics.json")
    metrics.setdefault("config", {})
    metrics["config"] = {
        **metrics["config"],
        "requested_group": group,
        "requested_split": split_name,
        "requested_method": method,
    }
    write_json(dst / "metrics.json", metrics)
    return metrics


def organize_same_split_outputs(output_dir: Path) -> tuple[list[dict], list[dict]]:
    sensor_rows = []
    model_rows = []
    sensor_root = output_dir / "sensor_vs_dual_improved_same_split"
    model_root = output_dir / "dual_sensor_models_same_split"
    for ref in SAME_SPLITS:
        dual_metrics = copy_result(
            ref.improved_dir,
            sensor_root / ref.name / "dual_sensor_improved_mlp",
            method="dual_sensor_improved_mlp",
            group="sensor_vs_dual_improved_same_split",
            split_name=ref.name,
        )
        sensor_rows.append(
            metric_row(
                "sensor_vs_dual_improved_same_split",
                ref.name,
                "dual_sensor_improved_mlp",
                sensor_root / ref.name / "dual_sensor_improved_mlp",
                dual_metrics,
            )
        )
        same_sensor_src = SAME_SPLIT_SOURCE / ref.name / "sensor1_current_improved_mlp"
        single_metrics = copy_result(
            same_sensor_src,
            sensor_root / ref.name / "single_sensor_improved_mlp",
            method="single_sensor_improved_mlp",
            group="sensor_vs_dual_improved_same_split",
            split_name=ref.name,
        )
        sensor_rows.append(
            metric_row(
                "sensor_vs_dual_improved_same_split",
                ref.name,
                "single_sensor_improved_mlp",
                sensor_root / ref.name / "single_sensor_improved_mlp",
                single_metrics,
            )
        )

        dual_model_metrics = copy_result(
            ref.improved_dir,
            model_root / ref.name / "improved_mlp",
            method="improved_mlp",
            group="dual_sensor_models_same_split",
            split_name=ref.name,
        )
        model_rows.append(
            metric_row(
                "dual_sensor_models_same_split",
                ref.name,
                "improved_mlp",
                model_root / ref.name / "improved_mlp",
                dual_model_metrics,
            )
        )
        for method in ("plain_mlp", "grnn", "extra_trees"):
            src = SAME_SPLIT_SOURCE / ref.name / method
            metrics = copy_result(
                src,
                model_root / ref.name / method,
                method=method,
                group="dual_sensor_models_same_split",
                split_name=ref.name,
            )
            model_rows.append(
                metric_row(
                    "dual_sensor_models_same_split",
                    ref.name,
                    method,
                    model_root / ref.name / method,
                    metrics,
                )
            )
    return sensor_rows, model_rows


def make_base_args(
    *,
    data_dir: Path,
    output_dir: Path,
    rows: int,
    cols: int,
    signal_normalization: str,
    dataset_fundamental_freqs: tuple[str, ...] = (),
) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=[data_dir],
        output_dir=output_dir,
        rows=rows,
        cols=cols,
        grid_step=1.0,
        left_sensor_x=3.0,
        left_sensor_y=-2.0,
        right_sensor_x=5.0,
        right_sensor_y=-2.0,
        reference_x=4.0,
        reference_y=-2.0,
        target_mode="center_relative",
        split_mode="none",
        split_unit="dataset_record",
        test_size=0.0,
        max_edge_test_fraction=0.05,
        split_random_state=1,
        randomize_split=False,
        constrained_max_edge_test_fraction=0.25,
        constrained_max_attempts=10000,
        exclude_corner_test=False,
        test_keys=[],
        window_size=None,
        stride=None,
        signal_normalization=signal_normalization,
        fundamental_freq=15.0,
        dataset_fundamental_freqs=dataset_fundamental_freqs,
        harmonic_orders=(1, 2, 3, 4),
        max_iter=1000,
        model_random_state=42,
        hidden_layers=(128, 128, 64),
        alpha=1e-3,
        separate_x_model=False,
        x_model_type="window_mlp",
        x_hidden_layers=(128, 128, 64),
        x_alpha=1e-4,
        x_model_random_state=42,
        x_ensemble_seeds=(1, 2, 3, 4, 5, 17, 23, 42, 77, 99),
        x_stretch_calibration="none",
        x_stretch_y_bins=3,
        edge_sample_weight=3.0,
        near_edge_sample_weight=1.5,
        no_early_stopping=False,
        feature_set="all_physics",
        training_unit="window",
        max_windows_per_point=80,
        row_offsets=(),
    )


def fit_pipeline(model: Pipeline, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
    fit_params = inspect.signature(model.named_steps["mlp"].fit).parameters
    if sample_weight is not None and "sample_weight" in fit_params:
        model.fit(x, y, mlp__sample_weight=sample_weight)
    else:
        model.fit(x, y)


def regressor(alpha: float, max_iter: int, random_state: int, n_iter_no_change: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(128, 128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=max_iter,
                    early_stopping=True,
                    n_iter_no_change=n_iter_no_change,
                    random_state=random_state,
                ),
            ),
        ]
    )


def classifier(alpha: float, max_iter: int, random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=max_iter,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=random_state,
                ),
            ),
        ]
    )


def point_metrics(point: pd.DataFrame) -> dict[str, float]:
    return base.compute_metrics(point[["x", "y"]].to_numpy(dtype=float), point[["pred_x", "pred_y"]].to_numpy(dtype=float))


def save_cross_result(
    output_dir: Path,
    *,
    config: dict,
    train_point: pd.DataFrame,
    test_point: pd.DataFrame,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_point.to_csv(output_dir / "train_point_predictions.csv", index=False, encoding="utf-8-sig")
    test_point.to_csv(output_dir / "test_point_predictions.csv", index=False, encoding="utf-8-sig")
    dataset = str(test_point["dataset"].iloc[0])
    test_point.to_csv(output_dir / f"test_{dataset}_point_predictions.csv", index=False, encoding="utf-8-sig")
    result = {
        "config": config,
        "train_point_metrics": point_metrics(train_point),
        "test_point_metrics": point_metrics(test_point),
        "test_point_metrics_by_dataset": {dataset: point_metrics(test_point)},
    }
    write_json(output_dir / "metrics.json", result)
    return result


def train_improved_cross(
    train_features: pd.DataFrame,
    cols: list[str],
    train_args: argparse.Namespace,
    args: argparse.Namespace,
) -> dict:
    train_df = base.sample_windows_per_point(train_features, train_args.max_windows_per_point, train_args.model_random_state)
    x_train = train_df[cols].to_numpy(dtype=float)
    y_train = train_df["y"].to_numpy(dtype=float)
    sample_weight = base.training_sample_weights(train_df, train_args)

    y_model = regressor(1e-3, args.max_iter, 42, 15)
    fit_pipeline(y_model, x_train, y_train, sample_weight)
    train_y_pred = np.clip(y_model.predict(x_train), 0.0, train_args.rows - 1)

    train_proba = np.zeros((len(train_df), train_args.cols), dtype=float)
    classifiers = []
    labels = np.rint(train_df["x"].to_numpy(dtype=float)).astype(int)
    for seed in train_args.x_ensemble_seeds:
        clf = classifier(1e-4, args.max_iter, seed)
        fit_pipeline(clf, x_train, labels, sample_weight)
        train_proba += base.align_classifier_proba(clf.named_steps["mlp"], clf.predict_proba(x_train), train_args.cols)
        classifiers.append(clf)
    train_proba /= len(train_args.x_ensemble_seeds)
    gamma, gamma_metrics = base.fit_x_probability_temperature(train_df, train_proba, train_y_pred, train_args)
    train_pred = np.column_stack([base.x_proba_to_continuous(train_proba, gamma), train_y_pred])
    return {
        "train_df": train_df,
        "cols": cols,
        "y_model": y_model,
        "classifiers": classifiers,
        "gamma": gamma,
        "gamma_metrics": gamma_metrics,
        "train_point": base.point_predictions(train_df, train_pred),
    }


def predict_improved_cross(model: dict, target_features: pd.DataFrame, target_args: argparse.Namespace) -> pd.DataFrame:
    target_df = base.sample_windows_per_point(target_features, target_args.max_windows_per_point, target_args.model_random_state)
    x_target = target_df[model["cols"]].to_numpy(dtype=float)
    y_pred = np.clip(model["y_model"].predict(x_target), 0.0, target_args.rows - 1)
    proba = np.zeros((len(target_df), target_args.cols), dtype=float)
    for clf in model["classifiers"]:
        proba += base.align_classifier_proba(clf.named_steps["mlp"], clf.predict_proba(x_target), target_args.cols)
    proba /= len(model["classifiers"])
    pred = np.column_stack([base.x_proba_to_continuous(proba, model["gamma"]), y_pred])
    return base.point_predictions(target_df, pred)


def train_plain_cross(train_features: pd.DataFrame, cols: list[str], train_args: argparse.Namespace, args: argparse.Namespace) -> dict:
    train_df = base.sample_windows_per_point(train_features, train_args.max_windows_per_point, args.random_state)
    x_train = train_df[cols].to_numpy(dtype=float)
    y_train, _ = base.make_targets(train_df, train_args)
    model = regressor(args.plain_alpha, args.max_iter, args.random_state, 20)
    fit_pipeline(model, x_train, y_train, None)
    return {
        "train_df": train_df,
        "cols": cols,
        "model": model,
        "train_point": base.point_predictions(train_df, base.pred_to_abs(model.predict(x_train), train_args)),
    }


def predict_plain_cross(model: dict, target_features: pd.DataFrame, target_args: argparse.Namespace) -> pd.DataFrame:
    target_df = base.sample_windows_per_point(target_features, target_args.max_windows_per_point, target_args.model_random_state)
    pred = base.pred_to_abs(model["model"].predict(target_df[model["cols"]].to_numpy(dtype=float)), target_args)
    return base.point_predictions(target_df, pred)


def train_point_model_cross(
    train_features: pd.DataFrame,
    cols: list[str],
    train_args: argparse.Namespace,
    args: argparse.Namespace,
    method: str,
) -> dict:
    train_df = base.aggregate_point_mean_features(train_features, cols)
    x_train = train_df[cols].to_numpy(dtype=float)
    y_train, _ = base.make_targets(train_df, train_args)
    if method == "grnn":
        model = GRNNRegressor(
            bandwidth=args.grnn_bandwidth,
            bandwidth_scale=args.grnn_bandwidth_scale,
            random_state=args.random_state,
        ).fit(x_train, y_train)
        extra = {"bandwidth": model.bandwidth_, "bandwidth_scale": args.grnn_bandwidth_scale}
    elif method == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=args.extra_trees_n_estimators,
            min_samples_leaf=args.extra_trees_min_samples_leaf,
            max_features=0.7,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        extra = {
            "n_estimators": args.extra_trees_n_estimators,
            "min_samples_leaf": args.extra_trees_min_samples_leaf,
            "max_features": 0.7,
        }
    else:
        raise ValueError(method)
    return {
        "train_df": train_df,
        "cols": cols,
        "model": model,
        "extra": extra,
        "train_point": base.point_predictions(train_df, base.pred_to_abs(model.predict(x_train), train_args)),
    }


def predict_point_model_cross(model: dict, target_features: pd.DataFrame, target_args: argparse.Namespace) -> pd.DataFrame:
    target_df = base.aggregate_point_mean_features(target_features, model["cols"])
    pred = base.pred_to_abs(model["model"].predict(target_df[model["cols"]].to_numpy(dtype=float)), target_args)
    return base.point_predictions(target_df, pred)


def run_cross_outputs(output_dir: Path, args: argparse.Namespace) -> list[dict]:
    cross_root = output_dir / "dual_sensor_models_cross_dataset"
    train_args = make_base_args(
        data_dir=ROOT / "mlp" / "20260430_true_inputs",
        output_dir=cross_root,
        rows=18,
        cols=9,
        signal_normalization="record_median",
    )
    print("\nBuilding cross-train features from all 162 20260430_true_inputs points...")
    train_features, train_meta = base.build_features(train_args)
    cols = base.feature_columns(train_features, train_args.feature_set)

    print("Training cross improved MLP once for all targets...")
    improved = train_improved_cross(train_features, cols, train_args, args)
    print("Training cross plain MLP once for all targets...")
    plain = train_plain_cross(train_features, cols, train_args, args)
    print("Training cross GRNN once for all targets...")
    grnn = train_point_model_cross(train_features, cols, train_args, args, "grnn")
    print("Training cross ExtraTrees once for all targets...")
    extra = train_point_model_cross(train_features, cols, train_args, args, "extra_trees")
    models = {
        "improved_mlp": (improved, predict_improved_cross, {"algorithm": "x_column_probability_continuous"}),
        "plain_mlp": (plain, predict_plain_cross, {"algorithm": "plain_joint_mlp"}),
        "grnn": (grnn, predict_point_model_cross, {"algorithm": "grnn", **grnn["extra"]}),
        "extra_trees": (extra, predict_point_model_cross, {"algorithm": "extra_trees", **extra["extra"]}),
    }

    rows = []
    for target in CROSS_TARGETS:
        target_args = make_base_args(
            data_dir=target.data_dir,
            output_dir=cross_root / target.name,
            rows=target.rows,
            cols=target.cols,
            signal_normalization="record_median",
            dataset_fundamental_freqs=target.dataset_freqs,
        )
        print(f"Building target features for {target.name}...")
        target_features, target_meta = base.build_features(target_args)
        for method, (model, predict_fn, extra_cfg) in models.items():
            result_dir = cross_root / target.name / method
            if args.skip_existing_cross and (result_dir / "metrics.json").exists():
                metrics = read_json(result_dir / "metrics.json")
            else:
                test_point = predict_fn(model, target_features, target_args)
                config = {
                    "requested_group": "dual_sensor_models_cross_dataset",
                    "method": method,
                    "cross_protocol": "train_all_162_20260430_true_inputs_predict_all_target_63_no_target_calibration",
                    "train_data_dir": str(train_args.data_dir[0].resolve()),
                    "target_data_dir": str(target.data_dir.resolve()),
                    "target": target.name,
                    "train_points": 162,
                    "test_points": int(test_point[["dataset", "record_id"]].drop_duplicates().shape[0]),
                    "feature_set": train_args.feature_set,
                    "feature_count": len(cols),
                    "signal_normalization": train_args.signal_normalization,
                    "harmonic_orders": list(train_args.harmonic_orders),
                    "max_windows_per_point": train_args.max_windows_per_point,
                    "edge_sample_weight": train_args.edge_sample_weight if method == "improved_mlp" else 1.0,
                    "near_edge_sample_weight": train_args.near_edge_sample_weight if method == "improved_mlp" else 1.0,
                    "train_dataset_feature_settings": train_meta,
                    "target_dataset_feature_settings": target_meta,
                    **extra_cfg,
                }
                if method == "improved_mlp":
                    config["x_ensemble_seeds"] = list(train_args.x_ensemble_seeds)
                    config["x_probability_temperature_gamma"] = model["gamma"]
                    config["x_probability_temperature_train_metrics"] = model["gamma_metrics"]
                metrics = save_cross_result(
                    result_dir,
                    config=config,
                    train_point=model["train_point"],
                    test_point=test_point,
                )
            rows.append(metric_row("dual_sensor_models_cross_dataset", target.name, method, result_dir, metrics))
            print(f"  {target.name}/{method}: mean={rows[-1]['test_mean_error']:.4f}")
    return rows


def write_summary_markdown(path: Path, title: str, rows: list[dict]) -> None:
    lines = [f"# {title}", "", "| Split/target | Method | Train mean | Test mean | RMSE x | RMSE y | Max error |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        train_mean = row["train_mean_error"]
        lines.append(
            "| {split} | {method} | {train:.3f} | {mean:.3f} | {rmsex:.3f} | {rmsey:.3f} | {maxerr:.3f} |".format(
                split=row["split_or_target"],
                method=row["method"],
                train=float("nan") if train_mean is None else train_mean,
                mean=row["test_mean_error"],
                rmsex=row["test_rmse_x"],
                rmsey=row["test_rmse_y"],
                maxerr=row["test_max_error"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplconfig"))
    (args.output_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)

    print("Organizing same-split outputs...")
    sensor_rows, same_model_rows = organize_same_split_outputs(args.output_dir)
    cross_rows = run_cross_outputs(args.output_dir, args)

    sensor_df = pd.DataFrame(sensor_rows)
    same_model_df = pd.DataFrame(same_model_rows)
    cross_df = pd.DataFrame(cross_rows)
    all_df = pd.concat([sensor_df, same_model_df, cross_df], ignore_index=True)

    sensor_df.to_csv(args.output_dir / "sensor_vs_dual_improved_same_split_summary.csv", index=False, encoding="utf-8-sig")
    same_model_df.to_csv(args.output_dir / "dual_sensor_models_same_split_summary.csv", index=False, encoding="utf-8-sig")
    cross_df.to_csv(args.output_dir / "dual_sensor_models_cross_dataset_summary.csv", index=False, encoding="utf-8-sig")
    all_df.to_csv(args.output_dir / "all_requested_baselines_summary.csv", index=False, encoding="utf-8-sig")

    write_summary_markdown(args.output_dir / "sensor_vs_dual_improved_same_split_summary.md", "Sensor vs Dual Improved MLP", sensor_rows)
    write_summary_markdown(args.output_dir / "dual_sensor_models_same_split_summary.md", "Dual Sensor Models Same Split", same_model_rows)
    write_summary_markdown(args.output_dir / "dual_sensor_models_cross_dataset_summary.md", "Dual Sensor Models Cross Dataset", cross_rows)

    print(f"\nSaved requested baseline outputs to {args.output_dir.resolve()}")
    print("\nSensor vs dual improved MLP:")
    print(sensor_df[["split_or_target", "method", "test_mean_error", "test_rmse_x", "test_rmse_y"]].to_string(index=False))
    print("\nDual-sensor model comparison, same 162-point splits:")
    print(same_model_df[["split_or_target", "method", "test_mean_error", "test_rmse_x", "test_rmse_y"]].to_string(index=False))
    print("\nDual-sensor model comparison, cross datasets:")
    print(cross_df[["split_or_target", "method", "test_mean_error", "test_rmse_x", "test_rmse_y"]].to_string(index=False))


if __name__ == "__main__":
    main()
