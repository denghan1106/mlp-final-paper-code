#!/usr/bin/env python3
# coding: utf-8
"""Run localization baselines in a separate baseline folder.

The default run covers:
- sensor1-only current MLP recipe, using the four real right-side channels.
- full-feature plain MLP, GRNN, and ExtraTrees baselines.
- three target validations: 20260429, 20260428, and 20260429_2.
"""

from __future__ import annotations

import argparse
import inspect
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
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

DEFAULT_CAL_IDS = (
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
Y_ANCHOR_CAL_IDS = tuple(sorted(set(DEFAULT_CAL_IDS) | {34, 39, 40, 41, 42, 43}))


@dataclass(frozen=True)
class ValidationSpec:
    name: str
    predict_data_dir: Path
    calibration_ids: tuple[int, ...]
    dataset_fundamental_freqs: tuple[str, ...] = ()
    predict_record_count: int = 63

    @property
    def dataset(self) -> str:
        return self.predict_data_dir.name


def load_base_module():
    spec = importlib.util.spec_from_file_location("base_mlp_localization", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base script from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


class GRNNRegressor:
    """General regression neural network via Gaussian-kernel Nadaraya-Watson regression."""

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
        if self.bandwidth is None:
            self.bandwidth_ = self._estimate_bandwidth(self.x_train_)
        else:
            self.bandwidth_ = float(self.bandwidth)
        if not np.isfinite(self.bandwidth_) or self.bandwidth_ <= 0:
            raise ValueError(f"Bad GRNN bandwidth: {self.bandwidth_}")
        return self

    def _estimate_bandwidth(self, x: np.ndarray) -> float:
        rng = np.random.default_rng(self.random_state)
        n = len(x)
        if n <= 2:
            return 1.0
        take = min(n, int(self.bandwidth_subset_size))
        idx = rng.choice(n, size=take, replace=False)
        sample = x[idx]
        k = min(6, len(sample))
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(sample)
        distances, _ = nn.kneighbors(sample)
        ref = distances[:, -1] if k > 1 else distances[:, 0]
        positive = ref[ref > 1e-9]
        if len(positive) == 0:
            return 1.0
        return float(np.median(positive) * self.bandwidth_scale)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler_.transform(np.asarray(x, dtype=float))
        x_train_sq = np.sum(self.x_train_ * self.x_train_, axis=1)
        out = np.empty((len(x_scaled), self.y_train_.shape[1]), dtype=float)
        denom_floor = 1e-300
        sigma2 = self.bandwidth_ * self.bandwidth_
        for start in range(0, len(x_scaled), self.chunk_size):
            chunk = x_scaled[start : start + self.chunk_size]
            d2 = (
                np.sum(chunk * chunk, axis=1, keepdims=True)
                + x_train_sq[None, :]
                - 2.0 * chunk @ self.x_train_.T
            )
            logits = -0.5 * np.maximum(d2, 0.0) / sigma2
            logits -= logits.max(axis=1, keepdims=True)
            weights = np.exp(logits) * self.sample_weight_[None, :]
            denom = weights.sum(axis=1, keepdims=True)
            denom = np.maximum(denom, denom_floor)
            out[start : start + len(chunk)] = weights @ self.y_train_ / denom
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    p.add_argument("--train-data-dir", type=Path, default=ROOT / "mlp" / "20260430_true_inputs")
    p.add_argument(
        "--validations",
        nargs="+",
        choices=("20260429", "20260428", "20260429_2"),
        default=("20260429", "20260428", "20260429_2"),
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=("plain_mlp", "grnn", "extra_trees"),
        default=("plain_mlp", "grnn", "extra_trees"),
        help="Full-feature non-improved baseline methods.",
    )
    p.add_argument(
        "--skip-sensor1-current",
        action="store_true",
        help="Skip the four-channel sensor1-only run using the current MLP recipe.",
    )
    p.add_argument("--train-rows", type=int, default=18)
    p.add_argument("--predict-rows", type=int, default=7)
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--feature-set", default="all")
    p.add_argument("--signal-normalization", choices=("none", "record_median", "dataset_median"), default="record_median")
    p.add_argument("--fundamental-freq", type=float, default=15.0)
    p.add_argument("--max-windows-per-point", type=int, default=120)
    p.add_argument("--model-random-state", type=int, default=42)
    p.add_argument("--calibration-sample-weight", type=float, default=2.0)
    p.add_argument("--current-seeds", type=int, nargs="+", default=tuple(range(1, 11)))
    p.add_argument("--current-hidden-layers", type=int, nargs="+", default=(128, 128, 64))
    p.add_argument("--plain-hidden-layers", type=int, nargs="+", default=(100,))
    p.add_argument("--mlp-max-iter", type=int, default=1000)
    p.add_argument("--current-alpha", type=float, default=1e-4)
    p.add_argument("--plain-alpha", type=float, default=1e-4)
    p.add_argument("--extra-trees-n-estimators", type=int, default=300)
    p.add_argument("--extra-trees-min-samples-leaf", type=int, default=2)
    p.add_argument("--grnn-bandwidth", type=float, default=None)
    p.add_argument("--grnn-bandwidth-scale", type=float, default=1.5)
    p.add_argument("--grnn-bandwidth-subset-size", type=int, default=1200)
    p.add_argument("--grnn-chunk-size", type=int, default=512)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def validation_specs(args: argparse.Namespace) -> dict[str, ValidationSpec]:
    return {
        "20260429": ValidationSpec(
            name="validation_20260429",
            predict_data_dir=ROOT / "mlp" / "20260429",
            calibration_ids=DEFAULT_CAL_IDS,
        ),
        "20260428": ValidationSpec(
            name="validation_20260428",
            predict_data_dir=ROOT / "mlp" / "20260428",
            calibration_ids=DEFAULT_CAL_IDS,
        ),
        "20260429_2": ValidationSpec(
            name="validation_20260429_2",
            predict_data_dir=ROOT / "mlp" / "20260429_2",
            calibration_ids=Y_ANCHOR_CAL_IDS,
            dataset_fundamental_freqs=("20260429_2:12",),
        ),
    }


def make_base_args(args: argparse.Namespace, spec: ValidationSpec, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=[args.train_data_dir, spec.predict_data_dir],
        output_dir=output_dir,
        rows=args.train_rows,
        cols=args.cols,
        grid_step=1.0,
        left_sensor_x=3.0,
        left_sensor_y=-2.0,
        right_sensor_x=5.0,
        right_sensor_y=-2.0,
        reference_x=4.0,
        reference_y=-2.0,
        target_mode="center_relative",
        split_mode="keys",
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
        signal_normalization=args.signal_normalization,
        fundamental_freq=args.fundamental_freq,
        dataset_fundamental_freqs=spec.dataset_fundamental_freqs,
        harmonic_orders=tuple(base.DEFAULT_HARMONIC_ORDERS),
        max_iter=args.mlp_max_iter,
        model_random_state=args.model_random_state,
        hidden_layers=tuple(args.current_hidden_layers),
        alpha=args.current_alpha,
        separate_x_model=False,
        x_hidden_layers=tuple(args.current_hidden_layers),
        x_alpha=args.current_alpha,
        x_model_random_state=args.model_random_state,
        x_ensemble_seeds=tuple(args.current_seeds),
        x_stretch_calibration="none",
        x_stretch_y_bins=3,
        edge_sample_weight=1.0,
        near_edge_sample_weight=1.0,
        no_early_stopping=False,
        feature_set=args.feature_set,
        training_unit="window",
        max_windows_per_point=args.max_windows_per_point,
        row_offsets=(),
    )


def test_keys_for_validation(spec: ValidationSpec) -> list[str]:
    calibration = set(spec.calibration_ids)
    return [
        f"{spec.dataset}:{rid}"
        for rid in range(1, spec.predict_record_count + 1)
        if rid not in calibration
    ]


def selected_feature_columns(features: pd.DataFrame, feature_set: str, feature_scope: str) -> list[str]:
    cols = base.feature_columns(features, feature_set)
    if feature_scope == "full":
        return cols
    if feature_scope != "sensor1":
        raise ValueError(f"Unsupported feature_scope={feature_scope!r}")
    prefixes = tuple(f"{ch}_" for ch in base.SENSOR1_CHANNELS) + (
        "right_23_25_",
        "right_24_26_",
        "sensor1_",
    )
    selected = [c for c in cols if c.startswith(prefixes)]
    if not selected:
        raise ValueError(f"No sensor1 feature columns selected from feature_set={feature_set!r}")
    return selected


def sample_or_aggregate(
    frame: pd.DataFrame,
    cols: list[str],
    training_unit: str,
    max_windows_per_point: int | None,
    random_state: int,
) -> pd.DataFrame:
    if training_unit == "window":
        return base.sample_windows_per_point(frame, max_windows_per_point, random_state)
    if training_unit == "point_mean":
        return base.aggregate_point_mean_features(frame, cols)
    raise ValueError(f"Unsupported training_unit={training_unit!r}")


def sample_weights(frame: pd.DataFrame, spec: ValidationSpec, base_args: argparse.Namespace, calibration_weight: float) -> np.ndarray:
    weights = base.training_sample_weights(frame, base_args)
    if weights is None:
        weights = np.ones(len(frame), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).copy()
    weights[frame["dataset"].eq(spec.dataset).to_numpy()] *= float(calibration_weight)
    return weights


def fit_mlp_pipeline(model: Pipeline, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
    fit_params = inspect.signature(model.named_steps["mlp"].fit).parameters
    if "sample_weight" in fit_params:
        model.fit(x, y, mlp__sample_weight=weights)
    else:
        model.fit(x, y)


def plain_mlp_model(args: argparse.Namespace, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=tuple(args.plain_hidden_layers),
                    activation="relu",
                    solver="adam",
                    alpha=args.plain_alpha,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=args.mlp_max_iter,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=seed,
                    verbose=False,
                ),
            ),
        ]
    )


def current_mlp_model(args: argparse.Namespace, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=tuple(args.current_hidden_layers),
                    activation="relu",
                    solver="adam",
                    alpha=args.current_alpha,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=args.mlp_max_iter,
                    early_stopping=True,
                    n_iter_no_change=15,
                    random_state=seed,
                    verbose=False,
                ),
            ),
        ]
    )


def point_metrics(point: pd.DataFrame) -> dict[str, float]:
    return base.compute_metrics(point[["x", "y"]].to_numpy(dtype=float), point[["pred_x", "pred_y"]].to_numpy(dtype=float))


def metrics_by_dataset(point: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {str(dataset): point_metrics(sub) for dataset, sub in point.groupby("dataset")}


def average_point_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No frames to average")
    key_cols = ["dataset", "record_id", "x", "y"]
    reference = frames[0][key_cols].sort_values(["dataset", "record_id"]).reset_index(drop=True)
    pred_x = np.zeros(len(reference), dtype=float)
    pred_y = np.zeros(len(reference), dtype=float)
    for frame in frames:
        frame = frame.sort_values(["dataset", "record_id"]).reset_index(drop=True)
        if not reference.equals(frame[key_cols]):
            raise ValueError("Point prediction keys do not match across ensemble members")
        pred_x += frame["pred_x"].to_numpy(dtype=float)
        pred_y += frame["pred_y"].to_numpy(dtype=float)
    out = reference.copy()
    out["pred_x"] = pred_x / len(frames)
    out["pred_y"] = pred_y / len(frames)
    out["n_models"] = len(frames)
    out["err_x"] = out["pred_x"] - out["x"]
    out["err_y"] = out["pred_y"] - out["y"]
    out["err_dist"] = np.hypot(out["err_x"], out["err_y"])
    return out


def save_point_outputs(point: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    point.to_csv(output_dir / f"{prefix}_point_predictions.csv", index=False, encoding="utf-8-sig")
    for dataset, sub in point.groupby("dataset"):
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(dataset))
        sub.to_csv(output_dir / f"{prefix}_{safe}_point_predictions.csv", index=False, encoding="utf-8-sig")


def write_result(
    *,
    output_dir: Path,
    config: dict,
    train_point: pd.DataFrame,
    test_point: pd.DataFrame,
    seed_metrics: list[dict] | None = None,
) -> dict:
    save_point_outputs(train_point, output_dir, "train")
    save_point_outputs(test_point, output_dir, "test")
    result = {
        "config": config,
        "train_point_metrics": point_metrics(train_point),
        "test_point_metrics": point_metrics(test_point),
        "train_point_metrics_by_dataset": metrics_by_dataset(train_point),
        "test_point_metrics_by_dataset": metrics_by_dataset(test_point),
    }
    if seed_metrics is not None:
        result["seed_metrics"] = seed_metrics
        pd.DataFrame(seed_metrics).to_csv(output_dir / "seed_metrics.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def summary_row(result: dict, output_dir: Path) -> dict:
    cfg = result["config"]
    train = result["train_point_metrics"]
    test = result["test_point_metrics"]
    return {
        "validation": cfg["validation"],
        "dataset": cfg["predict_dataset"],
        "experiment": cfg["experiment"],
        "method": cfg["method"],
        "feature_scope": cfg["feature_scope"],
        "training_unit": cfg["training_unit"],
        "feature_count": cfg["feature_count"],
        "train_points": cfg["train_points"],
        "test_points": cfg["test_points"],
        "calibration_points": cfg["calibration_points"],
        "train_mean_error": train["mean_euclidean_error"],
        "train_rmse_x": train["rmse_x"],
        "train_rmse_y": train["rmse_y"],
        "test_mean_error": test["mean_euclidean_error"],
        "test_median_error": test["median_euclidean_error"],
        "test_rmse_x": test["rmse_x"],
        "test_rmse_y": test["rmse_y"],
        "test_max_error": test["max_euclidean_error"],
        "result_dir": str(output_dir.resolve()),
    }


def make_config(
    *,
    args: argparse.Namespace,
    spec: ValidationSpec,
    base_args: argparse.Namespace,
    dataset_meta: dict,
    experiment: str,
    method: str,
    feature_scope: str,
    training_unit: str,
    feature_count: int,
    train_point_count: int,
    test_point_count: int,
    extra: dict | None = None,
) -> dict:
    config = {
        "validation": spec.name,
        "predict_dataset": spec.dataset,
        "experiment": experiment,
        "method": method,
        "feature_scope": feature_scope,
        "training_unit": training_unit,
        "train_data_dir": str(args.train_data_dir.resolve()),
        "predict_data_dir": str(spec.predict_data_dir.resolve()),
        "rows": args.train_rows,
        "predict_rows": args.predict_rows,
        "cols": args.cols,
        "feature_set": args.feature_set,
        "feature_count": feature_count,
        "signal_normalization": args.signal_normalization,
        "fundamental_freq": args.fundamental_freq,
        "dataset_fundamental_freqs": list(spec.dataset_fundamental_freqs),
        "dataset_feature_settings": dataset_meta,
        "max_windows_per_point": args.max_windows_per_point,
        "target_mode": base_args.target_mode,
        "calibration_record_ids": list(spec.calibration_ids),
        "calibration_points": len(spec.calibration_ids),
        "calibration_sample_weight": args.calibration_sample_weight,
        "test_keys": test_keys_for_validation(spec),
        "train_points": train_point_count,
        "test_points": test_point_count,
        "sensor1_channels": list(base.SENSOR1_CHANNELS),
        "note": "sensor1 is the right-side four-channel sensor: right_input23/right_input24/right_input25/right_input26.",
    }
    if extra:
        config.update(extra)
    return config


def run_current_mlp_ensemble(
    *,
    args: argparse.Namespace,
    spec: ValidationSpec,
    base_args: argparse.Namespace,
    dataset_meta: dict,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    cols: list[str],
    output_dir: Path,
    feature_scope: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    seed_rows: list[dict] = []
    for seed in args.current_seeds:
        train_df = sample_or_aggregate(train_raw, cols, "window", args.max_windows_per_point, seed)
        test_df = sample_or_aggregate(test_raw, cols, "window", args.max_windows_per_point, seed)
        x_train = train_df[cols].to_numpy(dtype=float)
        x_test = test_df[cols].to_numpy(dtype=float)
        y_train_rel, _ = base.make_targets(train_df, base_args)
        weights = sample_weights(train_df, spec, base_args, args.calibration_sample_weight)
        model = current_mlp_model(args, seed)
        fit_mlp_pipeline(model, x_train, y_train_rel, weights)
        train_pred = base.pred_to_abs(model.predict(x_train), base_args)
        test_pred = base.pred_to_abs(model.predict(x_test), base_args)
        train_point = base.point_predictions(train_df, train_pred)
        test_point = base.point_predictions(test_df, test_pred)
        train_frames.append(train_point)
        test_frames.append(test_point)
        seed_rows.append(
            {
                "seed": seed,
                "n_iter": int(model.named_steps["mlp"].n_iter_),
                "train_mean_error": point_metrics(train_point)["mean_euclidean_error"],
                "test_mean_error": point_metrics(test_point)["mean_euclidean_error"],
                "test_rmse_x": point_metrics(test_point)["rmse_x"],
                "test_rmse_y": point_metrics(test_point)["rmse_y"],
            }
        )
        print(f"    seed {seed}: test_mean={seed_rows[-1]['test_mean_error']:.4f}")

    train_ensemble = average_point_frames(train_frames)
    test_ensemble = average_point_frames(test_frames)
    config = make_config(
        args=args,
        spec=spec,
        base_args=base_args,
        dataset_meta=dataset_meta,
        experiment=f"{feature_scope}_current_mlp_ensemble",
        method="current_mlp_ensemble",
        feature_scope=feature_scope,
        training_unit="window",
        feature_count=len(cols),
        train_point_count=len(train_ensemble),
        test_point_count=len(test_ensemble),
        extra={
            "current_seeds": list(args.current_seeds),
            "hidden_layers": list(args.current_hidden_layers),
            "alpha": args.current_alpha,
        },
    )
    return write_result(
        output_dir=output_dir,
        config=config,
        train_point=train_ensemble,
        test_point=test_ensemble,
        seed_metrics=seed_rows,
    )


def run_single_model(
    *,
    args: argparse.Namespace,
    spec: ValidationSpec,
    base_args: argparse.Namespace,
    dataset_meta: dict,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    cols: list[str],
    output_dir: Path,
    method: str,
    feature_scope: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    training_unit = "window" if method == "plain_mlp" else "point_mean"
    train_df = sample_or_aggregate(train_raw, cols, training_unit, args.max_windows_per_point, args.model_random_state)
    test_df = sample_or_aggregate(test_raw, cols, training_unit, args.max_windows_per_point, args.model_random_state)
    x_train = train_df[cols].to_numpy(dtype=float)
    x_test = test_df[cols].to_numpy(dtype=float)
    y_train_rel, _ = base.make_targets(train_df, base_args)
    weights = sample_weights(train_df, spec, base_args, args.calibration_sample_weight)

    extra: dict = {"random_state": args.model_random_state}
    if method == "plain_mlp":
        model = plain_mlp_model(args, args.model_random_state)
        fit_mlp_pipeline(model, x_train, y_train_rel, weights)
        extra.update(
            {
                "hidden_layers": list(args.plain_hidden_layers),
                "alpha": args.plain_alpha,
                "n_iter": int(model.named_steps["mlp"].n_iter_),
            }
        )
        pred_train_rel = model.predict(x_train)
        pred_test_rel = model.predict(x_test)
    elif method == "grnn":
        model = GRNNRegressor(
            bandwidth=args.grnn_bandwidth,
            bandwidth_scale=args.grnn_bandwidth_scale,
            bandwidth_subset_size=args.grnn_bandwidth_subset_size,
            chunk_size=args.grnn_chunk_size,
            random_state=args.model_random_state,
        ).fit(x_train, y_train_rel, weights)
        extra.update(
            {
                "bandwidth": model.bandwidth_,
                "bandwidth_scale": args.grnn_bandwidth_scale,
                "bandwidth_subset_size": args.grnn_bandwidth_subset_size,
            }
        )
        pred_train_rel = model.predict(x_train)
        pred_test_rel = model.predict(x_test)
    elif method == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=args.extra_trees_n_estimators,
            min_samples_leaf=args.extra_trees_min_samples_leaf,
            max_features=0.7,
            random_state=args.model_random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train_rel, sample_weight=weights)
        extra.update(
            {
                "n_estimators": args.extra_trees_n_estimators,
                "min_samples_leaf": args.extra_trees_min_samples_leaf,
                "max_features": 0.7,
            }
        )
        pred_train_rel = model.predict(x_train)
        pred_test_rel = model.predict(x_test)
    else:
        raise ValueError(f"Unsupported method={method!r}")

    train_point = base.point_predictions(train_df, base.pred_to_abs(pred_train_rel, base_args))
    test_point = base.point_predictions(test_df, base.pred_to_abs(pred_test_rel, base_args))
    config = make_config(
        args=args,
        spec=spec,
        base_args=base_args,
        dataset_meta=dataset_meta,
        experiment=f"{feature_scope}_{method}",
        method=method,
        feature_scope=feature_scope,
        training_unit=training_unit,
        feature_count=len(cols),
        train_point_count=len(train_point),
        test_point_count=len(test_point),
        extra=extra,
    )
    return write_result(output_dir=output_dir, config=config, train_point=train_point, test_point=test_point)


def load_existing_result(path: Path) -> dict:
    with (path / "metrics.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplconfig"))
    (args.output_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)

    specs = validation_specs(args)
    summary_rows: list[dict] = []
    for validation_key in args.validations:
        spec = specs[validation_key]
        validation_dir = args.output_dir / spec.name
        validation_dir.mkdir(parents=True, exist_ok=True)
        base_args = make_base_args(args, spec, validation_dir)
        print(f"\nBuilding features for {spec.name} ({spec.dataset})...")
        features, dataset_meta = base.build_features(base_args)
        test_keys = test_keys_for_validation(spec)
        train_raw, test_raw = base.split_by_keys(features, test_keys)
        print(
            f"  train points={base.count_dataset_points(train_raw)}, "
            f"test points={base.count_dataset_points(test_raw)}, "
            f"calibration={len(spec.calibration_ids)}"
        )

        experiments: list[tuple[str, str, str, Callable[..., dict]]] = []
        if not args.skip_sensor1_current:
            experiments.append(
                (
                    "sensor1_current_mlp_ensemble",
                    "current_mlp_ensemble",
                    "sensor1",
                    run_current_mlp_ensemble,
                )
            )
        for method in args.methods:
            experiments.append((f"full_{method}", method, "full", run_single_model))

        for experiment_name, method, feature_scope, runner in experiments:
            result_dir = validation_dir / experiment_name
            if args.skip_existing and (result_dir / "metrics.json").exists():
                print(f"  skip existing {result_dir}")
                result = load_existing_result(result_dir)
                summary_rows.append(summary_row(result, result_dir))
                continue
            cols = selected_feature_columns(features, args.feature_set, feature_scope)
            print(f"  running {experiment_name}: method={method}, features={len(cols)}")
            if runner is run_current_mlp_ensemble:
                result = runner(
                    args=args,
                    spec=spec,
                    base_args=base_args,
                    dataset_meta=dataset_meta,
                    train_raw=train_raw,
                    test_raw=test_raw,
                    cols=cols,
                    output_dir=result_dir,
                    feature_scope=feature_scope,
                )
            else:
                result = runner(
                    args=args,
                    spec=spec,
                    base_args=base_args,
                    dataset_meta=dataset_meta,
                    train_raw=train_raw,
                    test_raw=test_raw,
                    cols=cols,
                    output_dir=result_dir,
                    method=method,
                    feature_scope=feature_scope,
                )
            row = summary_row(result, result_dir)
            summary_rows.append(row)
            print(
                f"    done: test_mean={row['test_mean_error']:.4f}, "
                f"rmse_x={row['test_rmse_x']:.4f}, rmse_y={row['test_rmse_y']:.4f}"
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved baseline summary to {summary_path.resolve()}")
    if not summary.empty:
        display_cols = [
            "validation",
            "experiment",
            "test_mean_error",
            "test_rmse_x",
            "test_rmse_y",
            "test_max_error",
        ]
        print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
