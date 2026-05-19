#!/usr/bin/env python3
# coding: utf-8
"""Baselines for the 20260430_true_inputs analysis splits.

This script intentionally mirrors the fixed validation splits already present
in mlp/20260430_mlp_analysis, then compares:
- the current improved MLP recipe using only the right-side four-channel sensor
- a plain joint MLP
- GRNN
- ExtraTrees
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
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
LOCALIZATION_DIR = ROOT / "mlp" / "20260428_20260429_mlp_localization"


@dataclass(frozen=True)
class ReferenceRun:
    name: str
    result_dir: Path


REFERENCE_RUNS = (
    ReferenceRun("validation_inner6", LOCALIZATION_DIR / "results_sklearn_mlp_style_inner_6pts_good"),
    ReferenceRun("validation_split1", ANALYSIS_DIR / "results_162_w120_blend30_anchor_scale_seed1_good"),
    ReferenceRun("validation_split7", ANALYSIS_DIR / "results_162_w120_blend30_anchor_auto5_seed7_good"),
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
    """Gaussian-kernel GRNN / Nadaraya-Watson regressor."""

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
        ref = distances[:, -1] if k > 1 else distances[:, 0]
        positive = ref[ref > 1e-9]
        if len(positive) == 0:
            return 1.0
        return float(np.median(positive) * self.bandwidth_scale)

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
    p.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "20260430_true_inputs_results")
    p.add_argument(
        "--references",
        nargs="+",
        choices=[run.name for run in REFERENCE_RUNS],
        default=[run.name for run in REFERENCE_RUNS],
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=("plain_mlp", "grnn", "extra_trees"),
        default=("plain_mlp", "grnn", "extra_trees"),
    )
    p.add_argument("--skip-sensor1-current", action="store_true")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--plain-alpha", type=float, default=1e-3)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--grnn-bandwidth", type=float, default=None)
    p.add_argument("--grnn-bandwidth-scale", type=float, default=1.5)
    p.add_argument("--extra-trees-n-estimators", type=int, default=300)
    p.add_argument("--extra-trees-min-samples-leaf", type=int, default=2)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def load_metrics(path: Path) -> dict:
    with (path / "metrics.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_base_args(cfg: dict, output_dir: Path, *, unweighted: bool = False) -> argparse.Namespace:
    dataset_freqs = tuple(f"{k}:{v}" for k, v in (cfg.get("dataset_fundamental_freqs") or {}).items())
    x_model_type = cfg.get("x_model_type") or "window_mlp"
    x_blend_classifier_weight = cfg.get("x_blend_classifier_weight")
    x_blend_classifier_weight = 0.0 if x_blend_classifier_weight is None else float(x_blend_classifier_weight)
    x_anchor_auto_count = cfg.get("x_anchor_auto_count")
    x_anchor_auto_count = 0 if x_anchor_auto_count is None else int(x_anchor_auto_count)
    x_anchor_correction = cfg.get("x_anchor_correction") or "none"
    x_anchor_record_ids = cfg.get("x_anchor_record_ids") or ()
    return argparse.Namespace(
        data_dir=[Path(p) for p in cfg["data_dirs"]],
        output_dir=output_dir,
        rows=int(cfg["rows"]),
        cols=int(cfg["cols"]),
        grid_step=float(cfg["grid_step"]),
        left_sensor_x=float(cfg["left_sensor_xy"][0]),
        left_sensor_y=float(cfg["left_sensor_xy"][1]),
        right_sensor_x=float(cfg["right_sensor_xy"][0]),
        right_sensor_y=float(cfg["right_sensor_xy"][1]),
        reference_x=float(cfg["reference_xy"][0]),
        reference_y=float(cfg["reference_xy"][1]),
        target_mode=cfg["target_mode"],
        split_mode="keys",
        split_unit=cfg.get("split_unit", "dataset_record"),
        test_size=float(cfg.get("test_size", 0.14)),
        max_edge_test_fraction=float(cfg.get("max_edge_test_fraction", 0.05)),
        split_random_state=int(cfg["split_random_state"]),
        randomize_split=False,
        constrained_max_edge_test_fraction=float(cfg.get("constrained_max_edge_test_fraction", 0.25)),
        constrained_max_attempts=10000,
        exclude_corner_test=False,
        test_keys=list(cfg["test_keys"]),
        window_size=None,
        stride=None,
        signal_normalization=cfg["signal_normalization"],
        fundamental_freq=float(cfg["default_fundamental_freq"]),
        dataset_fundamental_freqs=dataset_freqs,
        harmonic_orders=tuple(int(v) for v in cfg["harmonic_orders"]),
        max_iter=int(cfg.get("max_iter", 1000)),
        model_random_state=int(cfg.get("x_model_random_state", 42)),
        hidden_layers=tuple(int(v) for v in cfg["hidden_layers"]),
        alpha=float(cfg["alpha"]),
        separate_x_model=bool(cfg.get("separate_x_model", False)),
        x_model_type=x_model_type,
        x_blend_classifier_weight=x_blend_classifier_weight,
        augment_x_mirror=bool(cfg.get("augment_x_mirror", False)),
        x_hidden_layers=tuple(int(v) for v in cfg.get("x_hidden_layers", cfg["hidden_layers"])),
        x_alpha=float(cfg.get("x_alpha", 1e-4)),
        x_model_random_state=int(cfg.get("x_model_random_state", 42)),
        x_ensemble_seeds=tuple(int(v) for v in cfg.get("x_ensemble_seeds", ())),
        x_stretch_calibration=cfg.get("x_stretch_calibration", "none"),
        x_stretch_y_bins=int(cfg.get("x_stretch_y_bins", 3)),
        x_anchor_record_ids=tuple(str(v) for v in x_anchor_record_ids),
        x_anchor_auto_count=x_anchor_auto_count,
        x_anchor_correction=x_anchor_correction,
        x_anchor_exclude_from_metrics=bool(cfg.get("x_anchor_exclude_from_metrics", False)),
        edge_sample_weight=1.0 if unweighted else float(cfg.get("edge_sample_weight", 1.0)),
        near_edge_sample_weight=1.0 if unweighted else float(cfg.get("near_edge_sample_weight", 1.0)),
        no_early_stopping=not bool(cfg.get("early_stopping", True)),
        feature_set=cfg["feature_set"],
        training_unit=cfg.get("training_unit", "window"),
        max_windows_per_point=cfg.get("max_windows_per_point"),
        row_offsets=(),
    )


def selected_columns(features: pd.DataFrame, feature_set: str, scope: str) -> list[str]:
    cols = base.feature_columns(features, feature_set)
    if scope == "full":
        return cols
    if scope != "sensor1":
        raise ValueError(f"Unsupported feature scope: {scope}")
    prefixes = tuple(f"{ch}_" for ch in base.SENSOR1_CHANNELS) + ("right_23_25_", "right_24_26_", "sensor1_")
    selected = [col for col in cols if col.startswith(prefixes)]
    if not selected:
        raise ValueError("No sensor1 columns selected")
    return selected


def fit_pipeline(model: Pipeline, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
    fit_params = inspect.signature(model.named_steps["mlp"].fit).parameters
    if sample_weight is not None and "sample_weight" in fit_params:
        model.fit(x, y, mlp__sample_weight=sample_weight)
    else:
        model.fit(x, y)


def mlp_regressor(hidden_layers: tuple[int, ...], alpha: float, max_iter: int, random_state: int, n_iter_no_change: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=max_iter,
                    early_stopping=True,
                    n_iter_no_change=n_iter_no_change,
                    random_state=random_state,
                    verbose=False,
                ),
            ),
        ]
    )


def mlp_classifier(hidden_layers: tuple[int, ...], alpha: float, max_iter: int, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=max_iter,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=random_state,
                    verbose=False,
                ),
            ),
        ]
    )


def point_metrics(point: pd.DataFrame) -> dict[str, float]:
    return base.compute_metrics(
        point[["x", "y"]].to_numpy(dtype=float),
        point[["pred_x", "pred_y"]].to_numpy(dtype=float),
    )


def by_dataset(point: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {str(dataset): point_metrics(sub) for dataset, sub in point.groupby("dataset")}


def save_points(point: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    point.to_csv(output_dir / f"{prefix}_point_predictions.csv", index=False, encoding="utf-8-sig")
    for dataset, sub in point.groupby("dataset"):
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(dataset))
        sub.to_csv(output_dir / f"{prefix}_{safe}_point_predictions.csv", index=False, encoding="utf-8-sig")


def write_metrics(
    output_dir: Path,
    *,
    config: dict,
    train_point: pd.DataFrame,
    test_point: pd.DataFrame,
    test_non_anchor: pd.DataFrame | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_points(train_point, output_dir, "train")
    save_points(test_point, output_dir, "test")
    base.save_plots(train_point, output_dir, "train", "train: true vs predicted point mean", int(config["rows"]), int(config["cols"]))
    base.save_plots(test_point, output_dir, "test", "test: true vs predicted point mean", int(config["rows"]), int(config["cols"]))
    result = {
        "config": config,
        "train_point_metrics": point_metrics(train_point),
        "test_point_metrics": point_metrics(test_point),
        "test_point_metrics_non_anchor": None if test_non_anchor is None or test_non_anchor.empty else point_metrics(test_non_anchor),
        "train_point_metrics_by_dataset": by_dataset(train_point),
        "test_point_metrics_by_dataset": by_dataset(test_point),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def train_test_frames(features: pd.DataFrame, base_args: argparse.Namespace, cols: list[str], unit: str, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw, test_raw = base.split_by_keys(features, base_args.test_keys)
    if unit == "window":
        train_df = base.sample_windows_per_point(train_raw, base_args.max_windows_per_point, random_state)
        test_df = base.sample_windows_per_point(test_raw, base_args.max_windows_per_point, random_state)
        return train_df, test_df
    if unit == "point_mean":
        return base.aggregate_point_mean_features(train_raw, cols), base.aggregate_point_mean_features(test_raw, cols)
    raise ValueError(f"Unsupported training unit: {unit}")


def run_sensor1_current(
    *,
    output_dir: Path,
    reference_name: str,
    reference_cfg: dict,
    features: pd.DataFrame,
    dataset_meta: dict,
    args: argparse.Namespace,
) -> dict:
    base_args = make_base_args(reference_cfg, output_dir, unweighted=False)
    cols = selected_columns(features, base_args.feature_set, "sensor1")
    train_df, test_df = train_test_frames(features, base_args, cols, "window", base_args.model_random_state)
    x_train = train_df[cols].to_numpy(dtype=float)
    x_test = test_df[cols].to_numpy(dtype=float)
    y_train_rel, y_train_abs = base.make_targets(train_df, base_args)
    y_test_rel, y_test_abs = base.make_targets(test_df, base_args)
    sample_weight = base.training_sample_weights(train_df, base_args)

    print(f"    joint y/current MLP, sensor1 columns={len(cols)}")
    y_model = mlp_regressor(base_args.hidden_layers, base_args.alpha, args.max_iter, base_args.model_random_state, 15)
    fit_pipeline(y_model, x_train, y_train_rel, sample_weight)
    train_pred_abs = base.pred_to_abs(y_model.predict(x_train), base_args)
    test_pred_abs = base.pred_to_abs(y_model.predict(x_test), base_args)

    gamma = None
    gamma_metrics = None
    blend_weight = 0.0
    seeds = list(base_args.x_ensemble_seeds) if base_args.x_ensemble_seeds else [base_args.x_model_random_state]
    if base_args.separate_x_model:
        train_x_preds = []
        test_x_preds = []
        print(f"    separate x MLP ensemble, seeds={seeds}")
        for seed in seeds:
            x_model = mlp_regressor(base_args.x_hidden_layers, base_args.x_alpha, args.max_iter, seed, 20)
            fit_pipeline(x_model, x_train, train_df["x"].to_numpy(dtype=float), sample_weight)
            train_x_preds.append(np.clip(x_model.predict(x_train), 0.0, base_args.cols - 1))
            test_x_preds.append(np.clip(x_model.predict(x_test), 0.0, base_args.cols - 1))
        train_pred_abs[:, 0] = np.mean(np.vstack(train_x_preds), axis=0)
        test_pred_abs[:, 0] = np.mean(np.vstack(test_x_preds), axis=0)

        blend_weight = float(np.clip(base_args.x_blend_classifier_weight, 0.0, 1.0))
        if base_args.x_model_type == "window_mlp_classifier_blend" and blend_weight > 0:
            print(f"    x classifier blend, weight={blend_weight:g}")
            train_proba = np.zeros((len(train_df), base_args.cols), dtype=float)
            test_proba = np.zeros((len(test_df), base_args.cols), dtype=float)
            labels = np.rint(train_df["x"].to_numpy(dtype=float)).astype(int)
            mlp_train_x = train_pred_abs[:, 0].copy()
            mlp_test_x = test_pred_abs[:, 0].copy()
            for seed in seeds:
                clf = mlp_classifier(base_args.x_hidden_layers, base_args.x_alpha, args.max_iter, seed)
                fit_pipeline(clf, x_train, labels, sample_weight)
                train_proba += base.align_classifier_proba(clf.named_steps["mlp"], clf.predict_proba(x_train), base_args.cols)
                test_proba += base.align_classifier_proba(clf.named_steps["mlp"], clf.predict_proba(x_test), base_args.cols)
            train_proba /= len(seeds)
            test_proba /= len(seeds)
            gamma, gamma_metrics = base.fit_x_probability_temperature(train_df, train_proba, train_pred_abs[:, 1], base_args)
            train_pred_abs[:, 0] = (1.0 - blend_weight) * mlp_train_x + blend_weight * base.x_proba_to_continuous(train_proba, gamma)
            test_pred_abs[:, 0] = (1.0 - blend_weight) * mlp_test_x + blend_weight * base.x_proba_to_continuous(test_proba, gamma)

    x_stretch_fit = {"mode": "none", "global_scale": 1.0, "bins": []}
    if base_args.x_stretch_calibration != "none":
        x_stretch_fit = base.x_stretch_calibration_from_train(base.point_predictions(train_df, train_pred_abs), base_args)
        train_pred_abs = base.apply_x_stretch(train_pred_abs, x_stretch_fit, base_args)
        test_pred_abs = base.apply_x_stretch(test_pred_abs, x_stretch_fit, base_args)

    correction_fit = {"method": "none", "anchors": []}
    anchor_items = list(base_args.x_anchor_record_ids)
    if base_args.x_anchor_correction != "none":
        uncorrected_test_point = base.point_predictions(test_df, test_pred_abs)
        if reference_cfg.get("x_anchor_keys_used"):
            anchor_items = list(reference_cfg["x_anchor_keys_used"])
        elif base_args.x_anchor_auto_count > 0:
            anchor_items = base.auto_anchor_items(uncorrected_test_point, base_args.x_anchor_auto_count, base_args)
        correction_fit = base.fit_x_anchor_correction(uncorrected_test_point, anchor_items, base_args.x_anchor_correction, base_args)
        test_pred_abs = base.apply_x_anchor_correction(test_pred_abs, correction_fit, base_args)

    train_point = base.point_predictions(train_df, train_pred_abs)
    test_point = base.point_predictions(test_df, test_pred_abs)
    anchor_mask, anchor_keys_used = base.anchor_point_mask(test_point, anchor_items)
    non_anchor = test_point.loc[~anchor_mask].reset_index(drop=True)
    config = {
        "reference": reference_name,
        "method": "sensor1_current_improved_mlp",
        "rows": base_args.rows,
        "cols": base_args.cols,
        "feature_scope": "sensor1_right_four_channels",
        "feature_count": len(cols),
        "sensor1_channels": list(base.SENSOR1_CHANNELS),
        "data_dirs": reference_cfg["data_dirs"],
        "split_random_state": reference_cfg["split_random_state"],
        "test_keys": reference_cfg["test_keys"],
        "train_points": base.count_dataset_points(train_df),
        "test_points": base.count_dataset_points(test_df),
        "dataset_feature_settings": dataset_meta,
        "hidden_layers": list(base_args.hidden_layers),
        "alpha": base_args.alpha,
        "x_model_type": base_args.x_model_type,
        "x_blend_classifier_weight": blend_weight,
        "x_ensemble_seeds": seeds,
        "x_probability_temperature_gamma": gamma,
        "x_probability_temperature_train_metrics": gamma_metrics,
        "x_stretch_calibration": base_args.x_stretch_calibration,
        "x_stretch_calibration_fit": x_stretch_fit,
        "x_anchor_correction": base_args.x_anchor_correction,
        "x_anchor_auto_count": base_args.x_anchor_auto_count,
        "x_anchor_record_ids": list(base_args.x_anchor_record_ids),
        "x_anchor_keys_used": anchor_keys_used,
        "x_anchor_correction_fit": correction_fit,
        "edge_sample_weight": base_args.edge_sample_weight,
        "near_edge_sample_weight": base_args.near_edge_sample_weight,
    }
    return write_metrics(output_dir, config=config, train_point=train_point, test_point=test_point, test_non_anchor=non_anchor)


def run_plain_mlp(
    output_dir: Path,
    reference_name: str,
    reference_cfg: dict,
    features: pd.DataFrame,
    dataset_meta: dict,
    args: argparse.Namespace,
) -> dict:
    base_args = make_base_args(reference_cfg, output_dir, unweighted=True)
    cols = selected_columns(features, base_args.feature_set, "full")
    train_df, test_df = train_test_frames(features, base_args, cols, "window", args.random_state)
    x_train = train_df[cols].to_numpy(dtype=float)
    x_test = test_df[cols].to_numpy(dtype=float)
    y_train_rel, _ = base.make_targets(train_df, base_args)
    model = mlp_regressor(base_args.hidden_layers, args.plain_alpha, args.max_iter, args.random_state, 20)
    fit_pipeline(model, x_train, y_train_rel, None)
    train_point = base.point_predictions(train_df, base.pred_to_abs(model.predict(x_train), base_args))
    test_point = base.point_predictions(test_df, base.pred_to_abs(model.predict(x_test), base_args))
    config = {
        "reference": reference_name,
        "method": "plain_mlp",
        "rows": base_args.rows,
        "cols": base_args.cols,
        "feature_scope": "full",
        "feature_count": len(cols),
        "data_dirs": reference_cfg["data_dirs"],
        "split_random_state": reference_cfg["split_random_state"],
        "test_keys": reference_cfg["test_keys"],
        "train_points": base.count_dataset_points(train_df),
        "test_points": base.count_dataset_points(test_df),
        "dataset_feature_settings": dataset_meta,
        "hidden_layers": list(base_args.hidden_layers),
        "alpha": args.plain_alpha,
        "edge_sample_weight": 1.0,
        "near_edge_sample_weight": 1.0,
        "note": "plain joint MLP: no separate x model, no x stretch, no anchor correction, no edge weighting.",
    }
    return write_metrics(output_dir, config=config, train_point=train_point, test_point=test_point)


def run_point_mean_baseline(
    output_dir: Path,
    reference_name: str,
    reference_cfg: dict,
    features: pd.DataFrame,
    dataset_meta: dict,
    args: argparse.Namespace,
    method: str,
) -> dict:
    base_args = make_base_args(reference_cfg, output_dir, unweighted=True)
    cols = selected_columns(features, base_args.feature_set, "full")
    train_df, test_df = train_test_frames(features, base_args, cols, "point_mean", args.random_state)
    x_train = train_df[cols].to_numpy(dtype=float)
    x_test = test_df[cols].to_numpy(dtype=float)
    y_train_rel, _ = base.make_targets(train_df, base_args)
    extra: dict = {}
    if method == "grnn":
        model = GRNNRegressor(
            bandwidth=args.grnn_bandwidth,
            bandwidth_scale=args.grnn_bandwidth_scale,
            random_state=args.random_state,
        ).fit(x_train, y_train_rel)
        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)
        extra = {"bandwidth": model.bandwidth_, "bandwidth_scale": args.grnn_bandwidth_scale}
    elif method == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=args.extra_trees_n_estimators,
            min_samples_leaf=args.extra_trees_min_samples_leaf,
            max_features=0.7,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train_rel)
        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)
        extra = {
            "n_estimators": args.extra_trees_n_estimators,
            "min_samples_leaf": args.extra_trees_min_samples_leaf,
            "max_features": 0.7,
        }
    else:
        raise ValueError(f"Unsupported method: {method}")
    train_point = base.point_predictions(train_df, base.pred_to_abs(pred_train, base_args))
    test_point = base.point_predictions(test_df, base.pred_to_abs(pred_test, base_args))
    config = {
        "reference": reference_name,
        "method": method,
        "rows": base_args.rows,
        "cols": base_args.cols,
        "feature_scope": "full",
        "training_unit": "point_mean",
        "feature_count": len(cols),
        "data_dirs": reference_cfg["data_dirs"],
        "split_random_state": reference_cfg["split_random_state"],
        "test_keys": reference_cfg["test_keys"],
        "train_points": base.count_dataset_points(train_df),
        "test_points": base.count_dataset_points(test_df),
        "dataset_feature_settings": dataset_meta,
        **extra,
    }
    return write_metrics(output_dir, config=config, train_point=train_point, test_point=test_point)


def summary_row(result: dict, result_dir: Path, reference_metrics: dict) -> dict:
    cfg = result["config"]
    m = result["test_point_metrics"]
    train = result["train_point_metrics"]
    non_anchor = result.get("test_point_metrics_non_anchor") or {}
    return {
        "reference": cfg["reference"],
        "split_random_state": cfg["split_random_state"],
        "method": cfg["method"],
        "feature_scope": cfg.get("feature_scope"),
        "feature_count": cfg.get("feature_count"),
        "train_points": cfg.get("train_points"),
        "test_points": cfg.get("test_points"),
        "reference_improved_mean_error": reference_metrics.get("mean_euclidean_error"),
        "train_mean_error": train["mean_euclidean_error"],
        "test_mean_error": m["mean_euclidean_error"],
        "test_median_error": m["median_euclidean_error"],
        "test_rmse_x": m["rmse_x"],
        "test_rmse_y": m["rmse_y"],
        "test_max_error": m["max_euclidean_error"],
        "test_non_anchor_mean_error": non_anchor.get("mean_euclidean_error"),
        "result_dir": str(result_dir.resolve()),
    }


def existing_result(path: Path) -> dict:
    return load_metrics(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplconfig"))
    (args.output_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
    reference_map = {run.name: run for run in REFERENCE_RUNS}
    rows = []
    for reference_name in args.references:
        ref = reference_map[reference_name]
        ref_metrics = load_metrics(ref.result_dir)
        ref_cfg = ref_metrics["config"]
        validation_dir = args.output_dir / reference_name
        validation_dir.mkdir(parents=True, exist_ok=True)
        feature_args = make_base_args(ref_cfg, validation_dir, unweighted=False)
        print(f"\nBuilding features for {reference_name} from {ref.result_dir.name}...")
        features, dataset_meta = base.build_features(feature_args)
        print(f"  exact test keys: {len(ref_cfg['test_keys'])}; split_random_state={ref_cfg['split_random_state']}")

        jobs: list[tuple[str, str]] = []
        if not args.skip_sensor1_current:
            jobs.append(("sensor1_current_improved_mlp", "sensor1_current"))
        for method in args.methods:
            jobs.append((method, method))

        for folder_name, method in jobs:
            result_dir = validation_dir / folder_name
            if args.skip_existing and (result_dir / "metrics.json").exists():
                result = existing_result(result_dir)
                rows.append(summary_row(result, result_dir, ref_metrics["test_point_metrics"]))
                print(f"  skip existing {reference_name}/{folder_name}")
                continue
            print(f"  running {reference_name}/{folder_name}...")
            if method == "sensor1_current":
                result = run_sensor1_current(
                    output_dir=result_dir,
                    reference_name=reference_name,
                    reference_cfg=ref_cfg,
                    features=features,
                    dataset_meta=dataset_meta,
                    args=args,
                )
            elif method == "plain_mlp":
                result = run_plain_mlp(result_dir, reference_name, ref_cfg, features, dataset_meta, args)
            else:
                result = run_point_mean_baseline(result_dir, reference_name, ref_cfg, features, dataset_meta, args, method)
            row = summary_row(result, result_dir, ref_metrics["test_point_metrics"])
            rows.append(row)
            print(
                f"    done: test_mean={row['test_mean_error']:.4f}, "
                f"rmse_x={row['test_rmse_x']:.4f}, rmse_y={row['test_rmse_y']:.4f}"
            )

    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved summary to {summary_path.resolve()}")
    if not summary.empty:
        cols = [
            "reference",
            "method",
            "reference_improved_mean_error",
            "test_mean_error",
            "test_rmse_x",
            "test_rmse_y",
            "test_max_error",
        ]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
