#!/usr/bin/env python3
# coding: utf-8
"""MLP.py-style sklearn localization using both 20260428 and 20260429 datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import secrets
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_sklearn_mlp_style_inner_6pts"

RAW_CHANNELS = {
    "left_input27": 9,
    "left_input28": 11,
    "left_input29": 13,
    "right_input23": 1,
    "right_input24": 3,
    "right_input25": 5,
    "right_input26": 7,
}
CHANNELS = [
    "left_input27",
    "left_input28",
    "left_input29",
    "left_theory30",
    "right_input23",
    "right_input24",
    "right_input25",
    "right_input26",
]
PHYSICS_PAIRS = {
    "left_27_29": ("left_input27", "left_input29"),
    "left_28_30": ("left_input28", "left_theory30"),
    "right_23_25": ("right_input23", "right_input25"),
    "right_24_26": ("right_input24", "right_input26"),
}
SENSOR1_CHANNELS = ("right_input23", "right_input24", "right_input25", "right_input26")
SENSOR2_CHANNELS = ("left_input27", "left_input28", "left_input29")
SENSOR_PHYSICS_PREFIXES = ("sensor1_", "sensor2_", "sensor12_")
MIRROR_PREFIX_PAIRS = (
    ("left_input27", "right_input23"),
    ("left_input28", "right_input24"),
    ("left_input29", "right_input25"),
    ("left_theory30", "right_input26"),
    ("left_27_29", "right_23_25"),
    ("left_28_30", "right_24_26"),
    ("sensor1", "sensor2"),
)
DEFAULT_HARMONIC_ORDERS = (1, 2, 3, 4)
DEFAULT_TEST_KEYS = (
    "20260428:12",
    "20260428:32",
    "20260428:52",
    "20260429:21",
    "20260429:43",
    "20260429:47",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", action="append", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--rows", type=int, default=7)
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--grid-step", type=float, default=1.0)
    p.add_argument("--left-sensor-x", type=float, default=3.0)
    p.add_argument("--left-sensor-y", type=float, default=-2.0)
    p.add_argument("--right-sensor-x", type=float, default=5.0)
    p.add_argument("--right-sensor-y", type=float, default=-2.0)
    p.add_argument("--reference-x", type=float, default=4.0)
    p.add_argument("--reference-y", type=float, default=-2.0)
    p.add_argument(
        "--target-mode",
        choices=("dual_relative", "center_relative", "absolute"),
        default="dual_relative",
        help="dual_relative is the earlier 4-output mode; center_relative matches MLP.py logic with one 2-output target.",
    )
    p.add_argument(
        "--split-mode",
        choices=("keys", "random_point", "stratified_edge", "limited_edge", "constrained_random"),
        default="keys",
    )
    p.add_argument(
        "--split-unit",
        choices=("dataset_record", "record_id"),
        default="dataset_record",
        help=(
            "dataset_record splits each dataset:record independently. "
            "record_id holds out the same physical point across all datasets."
        ),
    )
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--max-edge-test-fraction", type=float, default=0.05)
    p.add_argument("--split-random-state", type=int, default=1)
    p.add_argument(
        "--randomize-split",
        action="store_true",
        help="Use a fresh random split seed on each run. The actual seed is saved in metrics.json.",
    )
    p.add_argument(
        "--constrained-max-edge-test-fraction",
        type=float,
        default=0.25,
        help="For constrained_random, cap boundary test points to this fraction of the test set.",
    )
    p.add_argument(
        "--constrained-max-attempts",
        type=int,
        default=10000,
        help="Maximum random attempts for constrained_random split.",
    )
    p.add_argument(
        "--exclude-corner-test",
        action="store_true",
        help="Prevent the four corner points from being selected as test points. Corners remain available for training.",
    )
    p.add_argument(
        "--test-keys",
        nargs="+",
        default=list(DEFAULT_TEST_KEYS),
        help="Explicit dataset:record_id keys used as the test set, e.g. 20260429:32.",
    )
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument(
        "--signal-normalization",
        choices=("none", "record_median", "dataset_median"),
        default="none",
        help="record_median removes each record's DC level; dataset_median removes the dataset/day channel baseline.",
    )
    p.add_argument("--fundamental-freq", type=float, default=15.0)
    p.add_argument(
        "--dataset-fundamental-freqs",
        nargs="*",
        default=(),
        help="Optional dataset vibration frequencies, e.g. 20260429_2:12.",
    )
    p.add_argument(
        "--harmonic-orders",
        type=int,
        nargs="*",
        default=list(DEFAULT_HARMONIC_ORDERS),
        help="Harmonic indices used for FFT features, e.g. 1 2 3 4.",
    )
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--model-random-state", type=int, default=42)
    p.add_argument("--hidden-layers", type=int, nargs="+", default=(128, 128, 64))
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument(
        "--separate-x-model",
        action="store_true",
        help="Train a dedicated x-coordinate MLP and use the joint model for y. This can reduce x shrinkage.",
    )
    p.add_argument(
        "--x-model-type",
        choices=("window_mlp", "point_histgbr", "column_classifier", "window_mlp_classifier_blend"),
        default="window_mlp",
        help=(
            "Dedicated x model used with --separate-x-model. window_mlp is the earlier per-window MLP ensemble; "
            "point_histgbr trains x on per-point mean features plus the point mean predicted y; "
            "column_classifier classifies the 9 x columns per window and converts probabilities to continuous x; "
            "window_mlp_classifier_blend averages window_mlp and column_classifier x predictions."
        ),
    )
    p.add_argument(
        "--x-blend-classifier-weight",
        type=float,
        default=0.1,
        help="For --x-model-type window_mlp_classifier_blend, weight assigned to the column-classifier x prediction.",
    )
    p.add_argument(
        "--augment-x-mirror",
        action="store_true",
        help=(
            "Augment only the dedicated x model by mirroring samples left-right: swap left/right feature columns "
            "and train with mirrored x = cols - 1 - x."
        ),
    )
    p.add_argument("--x-hidden-layers", type=int, nargs="+", default=(128, 128, 64))
    p.add_argument("--x-alpha", type=float, default=1e-4)
    p.add_argument("--x-model-random-state", type=int, default=42)
    p.add_argument(
        "--x-ensemble-seeds",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Train one dedicated x MLP per seed and average their x predictions. "
            "If omitted, the single --x-model-random-state model is used."
        ),
    )
    p.add_argument(
        "--x-stretch-calibration",
        choices=("none", "train_x_mse", "train_mean_dist", "train_y_bins_mean_dist", "train_pred_y_bins_mean_dist"),
        default="none",
        help=(
            "Post-hoc x stretch around the grid center, fit on training point predictions only. "
            "The *_y_bins_* modes fit separate scales by y band."
        ),
    )
    p.add_argument(
        "--x-stretch-y-bins",
        type=int,
        default=3,
        help="Number of y bands used by y-binned x stretch calibration.",
    )
    p.add_argument(
        "--x-anchor-record-ids",
        nargs="*",
        default=(),
        help=(
            "Known anchor point record ids used to fit same-day x correction after prediction. "
            "Accepts record ids like 19 or dataset:record_id keys."
        ),
    )
    p.add_argument(
        "--x-anchor-auto-count",
        type=int,
        default=0,
        help=(
            "Automatically choose this many anchor points from the predicted test points. "
            "For 5 anchors, chooses points closest to left-bottom, right-bottom, center, left-top, and right-top."
        ),
    )
    p.add_argument(
        "--x-anchor-correction",
        choices=("none", "bias", "scale_bias", "y_bias_linear", "xy_affine_residual"),
        default="none",
        help=(
            "Post-prediction x correction fitted on --x-anchor-record-ids. "
            "bias adds mean x error; scale_bias fits true_x=a*pred_x+b; "
            "y_bias_linear fits x residual from pred_y; xy_affine_residual fits x residual from pred_x,pred_y."
        ),
    )
    p.add_argument(
        "--x-anchor-exclude-from-metrics",
        action="store_true",
        help="Also report test metrics excluding anchor points used for x correction.",
    )
    p.add_argument(
        "--edge-sample-weight",
        type=float,
        default=1.0,
        help="Training sample weight for boundary points. Values >1 reduce inward edge shrinkage.",
    )
    p.add_argument(
        "--near-edge-sample-weight",
        type=float,
        default=1.0,
        help="Training sample weight for points one grid step from the boundary.",
    )
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument(
        "--feature-set",
        choices=(
            "all",
            "no_phase",
            "ac_magnitude",
            "all_physics",
            "all_physics_sensor12",
            "variation_physics",
            "variation_physics_sensor12",
        ),
        default="all",
        help=(
            "all uses MLP.py features; all_physics also adds within-sensor differential features; "
            "all_physics_sensor12 also adds cross-sensor aggregate features; "
            "variation_* keeps only std/rms/harmonic magnitude features to reduce cross-day DC drift."
        ),
    )
    p.add_argument(
        "--training-unit",
        choices=("window", "point_mean"),
        default="window",
        help="window matches MLP.py; point_mean averages each record's window features before fitting.",
    )
    p.add_argument("--max-windows-per-point", type=int, default=None)
    p.add_argument(
        "--row-offsets",
        nargs="*",
        default=(),
        help="Optional dataset row-label offsets, e.g. 20260429_2:1 maps raw row 1 to label row 2.",
    )
    return p.parse_args()


def record_id_from_path(path: Path) -> int:
    m = re.search(r"(?:记录|Record)\s*(\d+)", path.stem, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse record id from filename: {path.name}")
    return int(m.group(1))


def record_id_to_xy(record_id: int, rows: int, cols: int, grid_step: float) -> tuple[float, float]:
    idx = record_id - 1
    row = idx // cols
    col_in_row = idx % cols
    col = col_in_row if row % 2 == 0 else cols - 1 - col_in_row
    return col * grid_step, row * grid_step


def parse_row_offsets(items: tuple[str, ...] | list[str]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Bad --row-offsets item {item!r}; expected dataset:rows")
        dataset, raw_offset = item.split(":", 1)
        offsets[dataset] = int(raw_offset)
    return offsets


def parse_float_mapping(items: tuple[str, ...] | list[str]) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Bad mapping item {item!r}; expected dataset:value")
        dataset, raw_value = item.split(":", 1)
        mapping[dataset] = float(raw_value)
    return mapping


def resolved_harmonic_orders(args: argparse.Namespace) -> tuple[int, ...]:
    raw_orders = getattr(args, "harmonic_orders", DEFAULT_HARMONIC_ORDERS)
    if raw_orders is None:
        raw_orders = DEFAULT_HARMONIC_ORDERS
    orders = sorted({int(v) for v in raw_orders if int(v) >= 1})
    if not orders:
        raise ValueError("At least one harmonic order >= 1 is required.")
    return tuple(orders)


def read_interval(path: Path) -> float:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        row = next(reader)
    return float(row[1])


def normalize_record_signal(df: pd.DataFrame, mode: str, baseline: dict[str, float] | None = None) -> pd.DataFrame:
    if mode == "none":
        return df
    if mode == "record_median":
        df = df.copy()
        for ch in CHANNELS:
            df[ch] = df[ch] - float(df[ch].median())
        return df
    if mode == "dataset_median":
        if baseline is None:
            raise ValueError("dataset_median normalization requires a per-channel baseline")
        df = df.copy()
        for ch in CHANNELS:
            df[ch] = df[ch] - baseline[ch]
        return df
    raise ValueError(f"Unsupported signal_normalization={mode!r}")


def load_record(
    path: Path,
    dataset: str,
    signal_normalization: str = "none",
    baseline: dict[str, float] | None = None,
) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, skiprows=4, encoding="utf-8-sig")
    data = {"t": pd.to_numeric(raw.iloc[:, 0], errors="coerce")}
    for name, col_idx in RAW_CHANNELS.items():
        data[name] = pd.to_numeric(raw.iloc[:, col_idx], errors="coerce")
    df = pd.DataFrame(data).dropna().sort_values("t").reset_index(drop=True)

    if dataset == "20260428":
        # Input 29 was polluted in 20260428. Replace it with theoretical 29.
        base27 = float(df["left_input27"].median())
        df["left_input29"] = base27 - (df["left_input27"] - base27)

    # Input 30 is missing in both datasets. Complete it from input 28.
    base28 = float(df["left_input28"].median())
    df["left_theory30"] = base28 - (df["left_input28"] - base28)
    return normalize_record_signal(df, signal_normalization, baseline)


def dataset_channel_baseline(files: list[Path], dataset: str) -> dict[str, float]:
    record_medians: dict[str, list[float]] = {ch: [] for ch in CHANNELS}
    for path in files:
        df = load_record(path, dataset, "none")
        for ch in CHANNELS:
            record_medians[ch].append(float(df[ch].median()))
    return {ch: float(np.median(values)) for ch, values in record_medians.items()}


def extract_features(
    df: pd.DataFrame,
    *,
    dataset: str,
    record_id: int,
    x: float,
    y: float,
    window_size: int,
    stride: int,
    fs: float,
    fundamental_freq: float,
    harmonic_orders: tuple[int, ...],
) -> list[dict[str, float | int | str]]:
    rows = []
    harmonics = [(f"h{i}", fundamental_freq * i) for i in harmonic_orders]
    bins = {name: int(round(freq * window_size / fs)) for name, freq in harmonics}
    start = 0
    while start + window_size <= len(df):
        window = df.iloc[start : start + window_size]
        row: dict[str, float | int | str] = {
            "dataset": dataset,
            "record_id": record_id,
            "x": x,
            "y": y,
            "t_start": float(window["t"].iloc[0]),
            "t_end": float(window["t"].iloc[-1]),
        }
        for ch in CHANNELS:
            values = window[ch].to_numpy(dtype=float)
            row[f"{ch}_mean"] = float(values.mean())
            row[f"{ch}_std"] = float(values.std(ddof=0))
            row[f"{ch}_min"] = float(values.min())
            row[f"{ch}_max"] = float(values.max())
            row[f"{ch}_rms"] = float(np.sqrt(np.mean(values * values)))
            fft_vals = np.fft.rfft(values)
            mag = np.abs(fft_vals) / len(values)
            phase = np.angle(fft_vals)
            for name, _freq in harmonics:
                k = bins[name]
                row[f"{ch}_{name}_mag"] = float(mag[k])
                row[f"{ch}_{name}_phase"] = float(phase[k])
        for pair_name, (a, b) in PHYSICS_PAIRS.items():
            va = window[a].to_numpy(dtype=float)
            vb = window[b].to_numpy(dtype=float)
            diff = va - vb
            summ = va + vb
            norm_diff = diff / (np.abs(va) + np.abs(vb) + 1e-9)
            for suffix, values in (
                ("diff", diff),
                ("sum", summ),
                ("normdiff", norm_diff),
            ):
                row[f"{pair_name}_{suffix}_mean"] = float(values.mean())
                row[f"{pair_name}_{suffix}_std"] = float(values.std(ddof=0))
                row[f"{pair_name}_{suffix}_rms"] = float(np.sqrt(np.mean(values * values)))
            fft_vals = np.fft.rfft(diff)
            mag = np.abs(fft_vals) / len(diff)
            for name, _freq in harmonics:
                row[f"{pair_name}_diff_{name}_mag"] = float(mag[bins[name]])
        sensor1 = window[list(SENSOR1_CHANNELS)].to_numpy(dtype=float)
        sensor2 = window[list(SENSOR2_CHANNELS)].to_numpy(dtype=float)
        s1_mean = sensor1.mean(axis=1)
        s2_mean = sensor2.mean(axis=1)
        s1_rms = np.sqrt(np.mean(sensor1 * sensor1, axis=1))
        s2_rms = np.sqrt(np.mean(sensor2 * sensor2, axis=1))
        for prefix, values in (
            ("sensor1_mean_signal", s1_mean),
            ("sensor2_mean_signal", s2_mean),
            ("sensor1_rms_signal", s1_rms),
            ("sensor2_rms_signal", s2_rms),
            ("sensor12_mean_diff", s1_mean - s2_mean),
            ("sensor12_mean_sum", s1_mean + s2_mean),
            ("sensor12_mean_normdiff", (s1_mean - s2_mean) / (np.abs(s1_mean) + np.abs(s2_mean) + 1e-9)),
            ("sensor12_rms_diff", s1_rms - s2_rms),
            ("sensor12_rms_sum", s1_rms + s2_rms),
            ("sensor12_rms_normdiff", (s1_rms - s2_rms) / (s1_rms + s2_rms + 1e-9)),
        ):
            row[f"{prefix}_mean"] = float(values.mean())
            row[f"{prefix}_std"] = float(values.std(ddof=0))
            row[f"{prefix}_rms"] = float(np.sqrt(np.mean(values * values)))
        for prefix, values in (
            ("sensor12_mean_diff", s1_mean - s2_mean),
            ("sensor12_rms_diff", s1_rms - s2_rms),
        ):
            fft_vals = np.fft.rfft(values)
            mag = np.abs(fft_vals) / len(values)
            for name, _freq in harmonics:
                row[f"{prefix}_{name}_mag"] = float(mag[bins[name]])
        rows.append(row)
        start += stride
    return rows


def build_features(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    data_dirs = args.data_dir or [ROOT / "20260428", ROOT / "20260429"]
    row_offsets = parse_row_offsets(args.row_offsets)
    dataset_fundamentals = parse_float_mapping(args.dataset_fundamental_freqs)
    harmonic_orders = resolved_harmonic_orders(args)
    rows = []
    dataset_meta = {}
    for data_dir in data_dirs:
        data_dir = data_dir.resolve()
        dataset = data_dir.name
        files = sorted(data_dir.glob("*.csv"), key=record_id_from_path)
        if len(files) != args.rows * args.cols:
            print(f"Warning: {data_dir} has {len(files)} CSV files.")
        baseline = dataset_channel_baseline(files, dataset) if args.signal_normalization == "dataset_median" else None
        fs = 1.0 / read_interval(files[0])
        fundamental_freq = dataset_fundamentals.get(dataset, args.fundamental_freq)
        window_size = args.window_size or max(16, int(round(fs / fundamental_freq)))
        stride = args.stride or max(1, window_size // 4)
        dataset_meta[dataset] = {
            "fs": fs,
            "fundamental_freq": fundamental_freq,
            "harmonic_orders": list(harmonic_orders),
            "harmonic_freqs": [fundamental_freq * i for i in harmonic_orders],
            "window_size": window_size,
            "stride": stride,
        }
        for path in files:
            rid = record_id_from_path(path)
            label_rid = rid + row_offsets.get(dataset, 0) * args.cols
            if label_rid < 1 or label_rid > args.rows * args.cols:
                print(
                    f"Skipping {dataset}:{rid}; row offset maps it to label record {label_rid}, "
                    f"outside 1..{args.rows * args.cols}."
                )
                continue
            x, y = record_id_to_xy(label_rid, args.rows, args.cols, args.grid_step)
            rows.extend(
                extract_features(
                    load_record(path, dataset, args.signal_normalization, baseline),
                    dataset=dataset,
                    record_id=rid,
                    x=x,
                    y=y,
                    window_size=window_size,
                    stride=stride,
                    fs=fs,
                    fundamental_freq=fundamental_freq,
                    harmonic_orders=harmonic_orders,
                )
            )
    return pd.DataFrame(rows), dataset_meta


def feature_columns(df: pd.DataFrame, feature_set: str = "all") -> list[str]:
    excluded = {"dataset", "record_id", "x", "y", "t_start", "t_end"}
    cols = [c for c in df.columns if c not in excluded]
    physics_prefixes = tuple(f"{name}_" for name in PHYSICS_PAIRS)
    all_physics_prefixes = physics_prefixes + SENSOR_PHYSICS_PREFIXES
    within_physics_cols = [c for c in cols if c.startswith(physics_prefixes)]
    sensor12_physics_cols = [c for c in cols if c.startswith(SENSOR_PHYSICS_PREFIXES)]
    base_cols = [c for c in cols if not c.startswith(all_physics_prefixes)]
    if feature_set == "all":
        return base_cols
    if feature_set == "all_physics":
        return base_cols + within_physics_cols
    if feature_set == "all_physics_sensor12":
        return base_cols + within_physics_cols + sensor12_physics_cols
    if feature_set == "no_phase":
        return [c for c in base_cols if not c.endswith("_phase")]
    if feature_set == "ac_magnitude":
        return [c for c in base_cols if c.endswith("_std") or c.endswith("_mag")]
    if feature_set == "variation_physics":
        return [
            c
            for c in base_cols + within_physics_cols
            if c.endswith("_std") or c.endswith("_rms") or c.endswith("_mag")
        ]
    if feature_set == "variation_physics_sensor12":
        return [
            c
            for c in base_cols + within_physics_cols + sensor12_physics_cols
            if c.endswith("_std") or c.endswith("_rms") or c.endswith("_mag")
        ]
    raise ValueError(f"Unsupported feature_set={feature_set!r}")


def count_dataset_points(df: pd.DataFrame) -> int:
    return int(df[["dataset", "record_id"]].drop_duplicates().shape[0])


def add_point_key(points: pd.DataFrame) -> pd.DataFrame:
    points = points.copy()
    points["key"] = points["dataset"].astype(str) + ":" + points["record_id"].astype(str)
    return points


def candidate_split_points(df: pd.DataFrame, split_unit: str) -> pd.DataFrame:
    if split_unit == "dataset_record":
        return add_point_key(df[["dataset", "record_id", "x", "y"]].drop_duplicates())
    if split_unit == "record_id":
        points = df[["record_id", "x", "y"]].drop_duplicates().copy()
        if points["record_id"].duplicated().any():
            duplicates = points.loc[points["record_id"].duplicated(), "record_id"].tolist()
            raise ValueError(f"record_id split requires each record_id to map to one coordinate, duplicates: {duplicates}")
        points["dataset"] = "all"
        points["key"] = points["record_id"].astype(str)
        return points[["dataset", "record_id", "x", "y", "key"]]
    raise ValueError(f"Unsupported split_unit={split_unit!r}")


def split_mask(df: pd.DataFrame, test_keys: list[str], split_unit: str) -> pd.Series:
    test_set = set(test_keys)
    if split_unit == "dataset_record":
        keys = df["dataset"].astype(str) + ":" + df["record_id"].astype(str)
    elif split_unit == "record_id":
        keys = df["record_id"].astype(str)
    else:
        raise ValueError(f"Unsupported split_unit={split_unit!r}")
    return keys.isin(test_set)


def is_corner_point(x: float, y: float, rows: int, cols: int) -> bool:
    x_i = int(round(x))
    y_i = int(round(y))
    return (
        (x_i == 0 and y_i == 0)
        or (x_i == cols - 1 and y_i == 0)
        or (x_i == 0 and y_i == rows - 1)
        or (x_i == cols - 1 and y_i == rows - 1)
    )


def get_corner_keys(df: pd.DataFrame, rows: int, cols: int) -> list[str]:
    points = add_point_key(df[["dataset", "record_id", "x", "y"]].drop_duplicates())
    is_corner = points.apply(
        lambda r: is_corner_point(float(r["x"]), float(r["y"]), rows, cols),
        axis=1,
    )
    return sorted(points.loc[is_corner, "key"].tolist())


def validate_no_corner_test(test_keys: list[str], df: pd.DataFrame, rows: int, cols: int) -> None:
    corner_keys = set(get_corner_keys(df, rows, cols))
    bad = sorted(set(test_keys) & corner_keys)
    if bad:
        raise ValueError(f"Corner points were selected as test keys even though they should be excluded: {bad}")


def split_by_keys(df: pd.DataFrame, test_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_set = set(test_keys)
    keys = df["dataset"].astype(str) + ":" + df["record_id"].astype(str)
    train_df = df[~keys.isin(test_set)].reset_index(drop=True)
    test_df = df[keys.isin(test_set)].reset_index(drop=True)
    if test_df.empty:
        raise ValueError(f"No rows matched --test-keys={test_keys}")
    return train_df, test_df


def split_random_points(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    rows: int,
    cols: int,
    exclude_corner_test: bool = False,
    split_unit: str = "dataset_record",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    points = candidate_split_points(df, split_unit)

    if exclude_corner_test:
        is_corner = points.apply(
            lambda r: is_corner_point(float(r["x"]), float(r["y"]), rows, cols),
            axis=1,
        )
        candidate_points = points.loc[~is_corner].copy()
    else:
        candidate_points = points

    point_keys = candidate_points["key"].tolist()

    train_keys, test_keys = train_test_split(
        point_keys,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    if exclude_corner_test:
        bad = candidate_points[candidate_points["key"].isin(test_keys)].apply(
            lambda r: is_corner_point(float(r["x"]), float(r["y"]), rows, cols),
            axis=1,
        )
        if bool(bad.any()):
            raise ValueError(f"Corner points were selected as test keys even though they should be excluded: {test_keys}")

    mask = split_mask(df, test_keys, split_unit)
    train_df = df[~mask].reset_index(drop=True)
    test_df = df[mask].reset_index(drop=True)
    return train_df, test_df, sorted(test_keys)


def edge_distance(x: float, y: float, rows: int, cols: int) -> int:
    return int(round(min(x, (cols - 1) - x, y, (rows - 1) - y)))


def has_three_consecutive_points(points: pd.DataFrame) -> bool:
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    for _dataset, sub in points.groupby("dataset"):
        coords = {(int(round(r["x"])), int(round(r["y"]))) for _, r in sub.iterrows()}
        for x, y in coords:
            for dx, dy in directions:
                if (x + dx, y + dy) in coords and (x + 2 * dx, y + 2 * dy) in coords:
                    return True
    return False


def split_constrained_random_points(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    rows: int,
    cols: int,
    *,
    split_unit: str = "dataset_record",
    max_edge_fraction: float = 0.25,
    max_attempts: int = 10000,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    points = candidate_split_points(df, split_unit)
    is_corner = points.apply(lambda r: is_corner_point(float(r["x"]), float(r["y"]), rows, cols), axis=1)
    candidate_points = points.loc[~is_corner].copy().reset_index(drop=True)
    if candidate_points.empty:
        raise ValueError("No candidate test points after excluding corners")

    n_test = int(np.ceil(len(candidate_points) * test_size))
    n_test = max(1, min(n_test, len(candidate_points)))
    max_edge = int(np.floor(n_test * max_edge_fraction))
    if max_edge_fraction > 0 and max_edge == 0:
        max_edge = 1

    candidate_points["edge_dist"] = candidate_points.apply(
        lambda r: edge_distance(float(r["x"]), float(r["y"]), rows, cols),
        axis=1,
    )
    rng = np.random.default_rng(random_state)

    for _attempt in range(max_attempts):
        chosen_idx = rng.choice(candidate_points.index.to_numpy(), size=n_test, replace=False)
        chosen = candidate_points.loc[chosen_idx].copy()
        edge_count = int((chosen["edge_dist"] == 0).sum())
        if edge_count > max_edge:
            continue
        if has_three_consecutive_points(chosen):
            continue

        test_keys = sorted(chosen["key"].tolist())
        mask = split_mask(df, test_keys, split_unit)
        return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True), test_keys

    raise ValueError(
        "Could not find a constrained random test split. "
        f"Try increasing --constrained-max-edge-test-fraction or --constrained-max-attempts. "
        f"n_test={n_test}, max_edge={max_edge}, candidates={len(candidate_points)}"
    )


def training_sample_weights(frame: pd.DataFrame, args: argparse.Namespace) -> np.ndarray | None:
    if args.edge_sample_weight == 1.0 and args.near_edge_sample_weight == 1.0:
        return None
    weights = np.ones(len(frame), dtype=float)
    dists = frame.apply(lambda r: edge_distance(float(r["x"]), float(r["y"]), args.rows, args.cols), axis=1)
    weights[dists.to_numpy() == 0] = args.edge_sample_weight
    weights[dists.to_numpy() == 1] = args.near_edge_sample_weight
    return weights


def mirror_feature_name(name: str) -> str | None:
    for left, right in MIRROR_PREFIX_PAIRS:
        left_prefix = left + "_"
        right_prefix = right + "_"
        if name.startswith(left_prefix):
            return right + name[len(left) :]
        if name.startswith(right_prefix):
            return left + name[len(right) :]
    return None


def mirror_x_training_arrays(
    frame: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mirrored = frame.copy()
    for col in feature_cols:
        mirror_col = mirror_feature_name(col)
        if mirror_col is not None and mirror_col in frame.columns:
            mirrored[col] = frame[mirror_col].to_numpy()
        elif col.startswith("sensor12_") and ("_diff" in col or "_normdiff" in col):
            mirrored[col] = -frame[col].to_numpy(dtype=float)
    mirrored["x"] = (args.cols - 1) - mirrored["x"].to_numpy(dtype=float)

    x_original = frame[feature_cols].to_numpy(dtype=float)
    x_mirrored = mirrored[feature_cols].to_numpy(dtype=float)
    y_original = frame["x"].to_numpy(dtype=float)
    y_mirrored = mirrored["x"].to_numpy(dtype=float)
    x_aug = np.vstack([x_original, x_mirrored])
    y_aug = np.concatenate([y_original, y_mirrored])
    if sample_weight is None:
        return x_aug, y_aug, None
    return x_aug, y_aug, np.concatenate([sample_weight, sample_weight])


def split_stratified_edge_points(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    rows: int,
    cols: int,
    exclude_corner_test: bool = False,
    split_unit: str = "dataset_record",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    points = candidate_split_points(df, split_unit)
    points["edge_dist"] = points.apply(lambda r: edge_distance(float(r["x"]), float(r["y"]), rows, cols), axis=1)
    points["stratum"] = points["dataset"].astype(str) + ":edge" + points["edge_dist"].astype(str)

    if exclude_corner_test:
        is_corner = points.apply(
            lambda r: is_corner_point(float(r["x"]), float(r["y"]), rows, cols),
            axis=1,
        )
        candidate_points = points.loc[~is_corner].copy()
    else:
        candidate_points = points

    train_points, test_points = train_test_split(
        candidate_points,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=candidate_points["stratum"],
    )
    test_keys = sorted(test_points["key"].tolist())

    if exclude_corner_test:
        bad = candidate_points[candidate_points["key"].isin(test_keys)].apply(
            lambda r: is_corner_point(float(r["x"]), float(r["y"]), rows, cols),
            axis=1,
        )
        if bool(bad.any()):
            raise ValueError(f"Corner points were selected as test keys even though they should be excluded: {test_keys}")

    mask = split_mask(df, test_keys, split_unit)
    train_df = df[~mask].reset_index(drop=True)
    test_df = df[mask].reset_index(drop=True)
    return train_df, test_df, test_keys


def split_limited_edge_points(
    df: pd.DataFrame,
    test_size: float,
    max_edge_fraction: float,
    random_state: int,
    rows: int,
    cols: int,
    exclude_corner_test: bool = False,
    split_unit: str = "dataset_record",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(random_state)
    points = candidate_split_points(df, split_unit)
    points["edge_dist"] = points.apply(lambda r: edge_distance(float(r["x"]), float(r["y"]), rows, cols), axis=1)

    if exclude_corner_test:
        is_corner = points.apply(
            lambda r: is_corner_point(float(r["x"]), float(r["y"]), rows, cols),
            axis=1,
        )
        candidate_points = points.loc[~is_corner].copy()
    else:
        candidate_points = points

    n_total = len(candidate_points)
    n_test = int(np.ceil(n_total * test_size))
    max_edge = int(np.floor(n_test * max_edge_fraction))

    dataset_counts = candidate_points["dataset"].value_counts().sort_index()
    raw_targets = dataset_counts / n_total * n_test
    per_dataset = np.floor(raw_targets).astype(int)
    remainder = n_test - int(per_dataset.sum())
    if remainder:
        frac_order = (raw_targets - np.floor(raw_targets)).sort_values(ascending=False)
        for dataset in frac_order.index[:remainder]:
            per_dataset[dataset] += 1

    selected_keys: list[str] = []
    selected_edge = 0
    for dataset, target in per_dataset.items():
        sub = candidate_points[candidate_points["dataset"] == dataset]
        inner = sub[sub["edge_dist"] > 0]
        edge = sub[sub["edge_dist"] == 0]
        edge_quota = max(0, max_edge - selected_edge)
        take_edge = min(edge_quota, target, len(edge))
        if take_edge:
            chosen_edge = edge.sample(n=take_edge, random_state=int(rng.integers(0, 2**31 - 1)))
            selected_keys.extend(chosen_edge["key"].tolist())
            selected_edge += take_edge
        take_inner = target - take_edge
        if take_inner > len(inner):
            raise ValueError(f"Not enough non-edge points in {dataset}: need {take_inner}, have {len(inner)}")
        chosen_inner = inner.sample(n=take_inner, random_state=int(rng.integers(0, 2**31 - 1)))
        selected_keys.extend(chosen_inner["key"].tolist())

    selected_keys = sorted(selected_keys)

    if exclude_corner_test:
        bad = candidate_points[candidate_points["key"].isin(selected_keys)].apply(
            lambda r: is_corner_point(float(r["x"]), float(r["y"]), rows, cols),
            axis=1,
        )
        if bool(bad.any()):
            raise ValueError(f"Corner points were selected as test keys even though they should be excluded: {selected_keys}")

    mask = split_mask(df, selected_keys, split_unit)
    train_df = df[~mask].reset_index(drop=True)
    test_df = df[mask].reset_index(drop=True)
    return train_df, test_df, selected_keys


def make_targets(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    y_abs = frame[["x", "y"]].to_numpy(dtype=float)
    if args.target_mode == "dual_relative":
        left = np.array([args.left_sensor_x, args.left_sensor_y])
        right = np.array([args.right_sensor_x, args.right_sensor_y])
        return np.column_stack((y_abs - left, y_abs - right)), y_abs
    if args.target_mode == "center_relative":
        ref = np.array([args.reference_x, args.reference_y])
        return y_abs - ref, y_abs
    if args.target_mode == "absolute":
        return y_abs, y_abs
    raise ValueError(f"Unsupported target_mode={args.target_mode!r}")


def pred_to_abs(pred: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.target_mode == "dual_relative":
        left = np.array([args.left_sensor_x, args.left_sensor_y])
        right = np.array([args.right_sensor_x, args.right_sensor_y])
        return 0.5 * ((pred[:, 0:2] + left) + (pred[:, 2:4] + right))
    if args.target_mode == "center_relative":
        ref = np.array([args.reference_x, args.reference_y])
        return pred + ref
    if args.target_mode == "absolute":
        return pred
    raise ValueError(f"Unsupported target_mode={args.target_mode!r}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    dist = np.sqrt(np.sum(err * err, axis=1))
    return {
        "rmse_x": float(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))),
        "rmse_y": float(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))),
        "rmse_all": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mean_euclidean_error": float(dist.mean()),
        "median_euclidean_error": float(np.median(dist)),
        "max_euclidean_error": float(dist.max()),
        "r2_x": float(r2_score(y_true[:, 0], y_pred[:, 0])),
        "r2_y": float(r2_score(y_true[:, 1], y_pred[:, 1])),
        "r2_all": float(r2_score(y_true, y_pred)),
    }


def align_classifier_proba(clf: MLPClassifier, raw_proba: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(raw_proba), n_classes), dtype=float)
    for src_idx, cls in enumerate(clf.classes_):
        out[:, int(cls)] = raw_proba[:, src_idx]
    return out


def x_proba_to_continuous(proba: np.ndarray, gamma: float) -> np.ndarray:
    sharpened = np.power(np.clip(proba, 1e-12, 1.0), gamma)
    sharpened = sharpened / sharpened.sum(axis=1, keepdims=True)
    return sharpened @ np.arange(proba.shape[1], dtype=float)


def fit_x_probability_temperature(
    train_df: pd.DataFrame,
    train_proba: np.ndarray,
    train_y_pred: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    best_gamma = 1.0
    best_metrics: dict[str, float] | None = None
    best_score = float("inf")
    for gamma in np.arange(0.5, 12.0001, 0.25):
        pred_abs = np.column_stack([x_proba_to_continuous(train_proba, float(gamma)), train_y_pred])
        point = point_predictions(train_df, pred_abs)
        metrics = compute_metrics(point[["x", "y"]].to_numpy(), point[["pred_x", "pred_y"]].to_numpy())
        score = metrics["mean_euclidean_error"]
        if score < best_score:
            best_score = score
            best_gamma = float(gamma)
            best_metrics = metrics
    assert best_metrics is not None
    return best_gamma, best_metrics


def point_predictions(frame: pd.DataFrame, pred_abs: np.ndarray) -> pd.DataFrame:
    tmp = frame[["dataset", "record_id", "x", "y"]].copy()
    tmp["pred_x"] = pred_abs[:, 0]
    tmp["pred_y"] = pred_abs[:, 1]
    grouped = (
        tmp.groupby(["dataset", "record_id"], as_index=False)
        .agg(
            x=("x", "first"),
            y=("y", "first"),
            pred_x=("pred_x", "mean"),
            pred_y=("pred_y", "mean"),
            n_windows=("pred_x", "size"),
        )
        .sort_values(["dataset", "record_id"])
        .reset_index(drop=True)
    )
    grouped["err_x"] = grouped["pred_x"] - grouped["x"]
    grouped["err_y"] = grouped["pred_y"] - grouped["y"]
    grouped["err_dist"] = np.sqrt(grouped["err_x"] ** 2 + grouped["err_y"] ** 2)
    return grouped


def point_key(dataset: object, record_id: object) -> str:
    return f"{dataset}:{int(record_id)}"


def anchor_point_mask(point_df: pd.DataFrame, anchor_items: tuple[str, ...] | list[str]) -> tuple[pd.Series, list[str]]:
    if not anchor_items:
        return pd.Series(False, index=point_df.index), []
    wanted_keys = {str(item) for item in anchor_items if ":" in str(item)}
    wanted_ids = {int(str(item)) for item in anchor_items if ":" not in str(item)}
    keys = point_df.apply(lambda r: point_key(r["dataset"], r["record_id"]), axis=1)
    mask = keys.isin(wanted_keys) | point_df["record_id"].astype(int).isin(wanted_ids)
    used = keys.loc[mask].tolist()
    missing = sorted(wanted_keys - set(used))
    used_ids = {int(k.split(":", 1)[1]) for k in used}
    missing.extend(str(v) for v in sorted(wanted_ids - used_ids))
    if missing:
        raise ValueError(f"Anchor record ids were not found in predicted test points: {missing}")
    return mask, used


def auto_anchor_items(point_df: pd.DataFrame, count: int, args: argparse.Namespace) -> list[str]:
    count = int(count)
    if count <= 0:
        return []
    if count > len(point_df):
        raise ValueError(f"--x-anchor-auto-count={count} exceeds available test points ({len(point_df)})")

    points = point_df.reset_index(drop=True).copy()
    chosen: list[int] = []
    if count == 5:
        targets = np.array(
            [
                [0.0, 0.0],
                [args.cols - 1.0, 0.0],
                [(args.cols - 1.0) * 0.5, (args.rows - 1.0) * 0.5],
                [0.0, args.rows - 1.0],
                [args.cols - 1.0, args.rows - 1.0],
            ],
            dtype=float,
        )
        coords = points[["x", "y"]].to_numpy(dtype=float)
        for target in targets:
            dist = np.sum((coords - target) ** 2, axis=1)
            if chosen:
                dist[chosen] = np.inf
            chosen.append(int(np.argmin(dist)))
    else:
        coords = points[["x", "y"]].to_numpy(dtype=float)
        center = np.array([(args.cols - 1.0) * 0.5, (args.rows - 1.0) * 0.5])
        chosen.append(int(np.argmin(np.sum((coords - center) ** 2, axis=1))))
        while len(chosen) < count:
            dist_to_chosen = np.min([np.sum((coords - coords[idx]) ** 2, axis=1) for idx in chosen], axis=0)
            dist_to_chosen[chosen] = -1.0
            chosen.append(int(np.argmax(dist_to_chosen)))

    return [point_key(points.loc[idx, "dataset"], points.loc[idx, "record_id"]) for idx in chosen]


def fit_x_anchor_correction(point_df: pd.DataFrame, anchor_items: tuple[str, ...] | list[str], method: str, args: argparse.Namespace) -> dict:
    if method == "none":
        return {"method": "none", "anchors": []}
    anchor_mask, used = anchor_point_mask(point_df, anchor_items)
    if not bool(anchor_mask.any()):
        raise ValueError("--x-anchor-correction requires at least one matching --x-anchor-record-ids item")

    anchors = point_df.loc[anchor_mask].copy()
    residual = anchors["x"].to_numpy(dtype=float) - anchors["pred_x"].to_numpy(dtype=float)
    correction: dict = {
        "method": method,
        "anchors": used,
        "n_anchors": int(len(anchors)),
    }
    if method == "bias":
        correction["delta"] = float(residual.mean())
        return correction
    if method == "scale_bias":
        if len(anchors) < 2 or float(anchors["pred_x"].std(ddof=0)) <= 1e-9:
            correction.update({"method": "bias", "fallback_from": method, "delta": float(residual.mean())})
            return correction
        slope, intercept = np.polyfit(anchors["pred_x"].to_numpy(dtype=float), anchors["x"].to_numpy(dtype=float), 1)
        correction.update({"slope": float(slope), "intercept": float(intercept)})
        return correction
    if method == "y_bias_linear":
        if len(anchors) < 2 or float(anchors["pred_y"].std(ddof=0)) <= 1e-9:
            correction.update({"method": "bias", "fallback_from": method, "delta": float(residual.mean())})
            return correction
        slope, intercept = np.polyfit(anchors["pred_y"].to_numpy(dtype=float), residual, 1)
        correction.update({"slope": float(slope), "intercept": float(intercept), "clip_delta": 1.2})
        return correction
    if method == "xy_affine_residual":
        if len(anchors) < 3:
            correction.update({"method": "bias", "fallback_from": method, "delta": float(residual.mean())})
            return correction
        x_mat = np.column_stack(
            [
                anchors["pred_x"].to_numpy(dtype=float),
                anchors["pred_y"].to_numpy(dtype=float),
                np.ones(len(anchors), dtype=float),
            ]
        )
        penalty = np.diag([1.0, 1.0, 0.0])
        alpha = 1.0
        coef = np.linalg.solve(x_mat.T @ x_mat + alpha * penalty, x_mat.T @ residual)
        correction.update({"coef_pred_x": float(coef[0]), "coef_pred_y": float(coef[1]), "intercept": float(coef[2]), "alpha": alpha, "clip_delta": 1.5})
        return correction
    raise ValueError(f"Unsupported x anchor correction method: {method}")


def apply_x_anchor_correction(pred_abs: np.ndarray, correction: dict, args: argparse.Namespace) -> np.ndarray:
    method = correction.get("method", "none")
    pred_abs = pred_abs.copy()
    if method == "none":
        return pred_abs
    pred_x = pred_abs[:, 0]
    pred_y = pred_abs[:, 1]
    if method == "bias":
        corrected = pred_x + float(correction["delta"])
    elif method == "scale_bias":
        corrected = float(correction["slope"]) * pred_x + float(correction["intercept"])
    elif method == "y_bias_linear":
        delta = float(correction["slope"]) * pred_y + float(correction["intercept"])
        delta = np.clip(delta, -float(correction.get("clip_delta", 1.2)), float(correction.get("clip_delta", 1.2)))
        corrected = pred_x + delta
    elif method == "xy_affine_residual":
        delta = (
            float(correction["coef_pred_x"]) * pred_x
            + float(correction["coef_pred_y"]) * pred_y
            + float(correction["intercept"])
        )
        delta = np.clip(delta, -float(correction.get("clip_delta", 1.5)), float(correction.get("clip_delta", 1.5)))
        corrected = pred_x + delta
    else:
        raise ValueError(f"Unsupported fitted x anchor correction method: {method}")
    pred_abs[:, 0] = np.clip(corrected, 0.0, args.cols - 1)
    return pred_abs


def optimize_x_stretch_scale(train_point: pd.DataFrame, args: argparse.Namespace, mode: str) -> float:
    center_x = (args.cols - 1) * 0.5
    z = train_point["pred_x"].to_numpy(dtype=float) - center_x
    target = train_point["x"].to_numpy(dtype=float) - center_x
    denom = float(np.dot(z, z))
    if denom <= 1e-12:
        return 1.0
    if mode == "train_x_mse":
        return float(np.dot(z, target) / denom)
    if mode.endswith("mean_dist"):
        best_scale = 1.0
        best_metric = float("inf")
        for scale in np.arange(0.8, 1.5001, 0.005):
            px = np.clip(center_x + z * scale, 0.0, args.cols - 1)
            ex = px - train_point["x"].to_numpy(dtype=float)
            ey = train_point["err_y"].to_numpy(dtype=float)
            metric = float(np.sqrt(ex * ex + ey * ey).mean())
            if metric < best_metric:
                best_metric = metric
                best_scale = float(scale)
        return best_scale
    return 1.0


def x_stretch_calibration_from_train(train_point: pd.DataFrame, args: argparse.Namespace) -> dict:
    mode = args.x_stretch_calibration
    if mode == "none":
        return {"mode": "none", "global_scale": 1.0, "bins": []}
    if mode in ("train_x_mse", "train_mean_dist"):
        return {"mode": mode, "global_scale": optimize_x_stretch_scale(train_point, args, mode), "bins": []}

    base_mode = "train_mean_dist"
    global_scale = optimize_x_stretch_scale(train_point, args, base_mode)
    bin_source_col = "pred_y" if mode == "train_pred_y_bins_mean_dist" else "y"
    y_values = train_point[bin_source_col].to_numpy(dtype=float)
    n_bins = max(1, int(args.x_stretch_y_bins))
    edges = np.quantile(y_values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return {"mode": mode, "global_scale": global_scale, "bin_source": bin_source_col, "bins": []}

    bins = []
    for idx in range(len(edges) - 1):
        lower = float(edges[idx])
        upper = float(edges[idx + 1])
        if idx == len(edges) - 2:
            mask = (y_values >= lower) & (y_values <= upper)
        else:
            mask = (y_values >= lower) & (y_values < upper)
        sub = train_point.loc[mask]
        scale = global_scale if len(sub) < 4 else optimize_x_stretch_scale(sub, args, base_mode)
        bins.append({"lower": lower, "upper": upper, "scale": float(scale), "n_points": int(len(sub))})
    return {"mode": mode, "global_scale": global_scale, "bin_source": bin_source_col, "bins": bins}


def apply_x_stretch(pred_abs: np.ndarray, calibration: dict, args: argparse.Namespace) -> np.ndarray:
    pred_abs = pred_abs.copy()
    center_x = (args.cols - 1) * 0.5
    if calibration.get("mode") == "none":
        return pred_abs
    scales = np.full(len(pred_abs), float(calibration.get("global_scale", 1.0)))
    bins = calibration.get("bins") or []
    if bins:
        y_values = pred_abs[:, 1]
        for idx, item in enumerate(bins):
            lower = float(item["lower"])
            upper = float(item["upper"])
            if idx == len(bins) - 1:
                mask = (y_values >= lower) & (y_values <= upper)
            else:
                mask = (y_values >= lower) & (y_values < upper)
            scales[mask] = float(item["scale"])
    pred_abs[:, 0] = np.clip(center_x + (pred_abs[:, 0] - center_x) * scales, 0.0, args.cols - 1)
    return pred_abs


def aggregate_point_mean_features(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    meta = frame.groupby(["dataset", "record_id"], as_index=False).agg(
        x=("x", "first"),
        y=("y", "first"),
        t_start=("t_start", "min"),
        t_end=("t_end", "max"),
    )
    feats = frame.groupby(["dataset", "record_id"], as_index=False)[cols].mean()
    return meta.merge(feats, on=["dataset", "record_id"], how="inner")


def aggregate_point_mean_features_with_pred_y(frame: pd.DataFrame, cols: list[str], pred_y: np.ndarray) -> pd.DataFrame:
    tmp = frame[["dataset", "record_id", "x", "y"]].copy()
    tmp["pred_y_point_mean"] = np.asarray(pred_y, dtype=float)
    meta = tmp.groupby(["dataset", "record_id"], as_index=False).agg(
        x=("x", "first"),
        y=("y", "first"),
        pred_y_point_mean=("pred_y_point_mean", "mean"),
    )
    feats = frame.groupby(["dataset", "record_id"], as_index=False)[cols].mean()
    return meta.merge(feats, on=["dataset", "record_id"], how="inner")


def point_predictions_to_window_values(frame: pd.DataFrame, point_frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    pred = point_frame[["dataset", "record_id"]].copy()
    pred["value"] = np.asarray(values, dtype=float)
    merged = frame[["dataset", "record_id"]].merge(pred, on=["dataset", "record_id"], how="left", sort=False)
    if merged["value"].isna().any():
        missing = frame.loc[merged["value"].isna(), ["dataset", "record_id"]].drop_duplicates()
        raise ValueError(f"Missing point predictions for {missing.to_dict(orient='records')}")
    return merged["value"].to_numpy(dtype=float)


def sample_windows_per_point(frame: pd.DataFrame, max_windows: int | None, random_state: int) -> pd.DataFrame:
    if max_windows is None:
        return frame
    return (
        frame.groupby(["dataset", "record_id"], group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), max_windows), random_state=random_state))
        .reset_index(drop=True)
    )


def save_plots(point_df: pd.DataFrame, output_dir: Path, prefix: str, title: str, rows: int, cols: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for dataset, sub in point_df.groupby("dataset"):
        ax.scatter(sub["x"], sub["y"], s=55, label=f"True {dataset}")
        ax.scatter(sub["pred_x"], sub["pred_y"], s=65, marker="x", label=f"Pred {dataset}")
        for _, row in sub.iterrows():
            ax.plot([row["x"], row["pred_x"]], [row["y"], row["pred_y"]], alpha=0.35)
            ax.text(row["x"] + 0.04, row["y"] + 0.04, str(int(row["record_id"])), fontsize=8)
    ax.set_xlim(-0.6, cols - 0.4)
    ax.set_ylim(-0.6, rows - 0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_point_predictions.png", dpi=180)
    plt.close(fig)

    heat = np.full((rows, cols), np.nan)
    for _, row in point_df.groupby(["record_id"], as_index=False).agg(x=("x", "first"), y=("y", "first"), err_dist=("err_dist", "mean")).iterrows():
        heat[int(round(row["y"])), int(round(row["x"]))] = row["err_dist"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(heat, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, label="Mean Euclidean error")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}: mean error heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_error_heatmap.png", dpi=180)
    plt.close(fig)


def save_outputs_by_dataset(point_df: pd.DataFrame, output_dir: Path, split_name: str, args: argparse.Namespace) -> dict[str, dict[str, float]]:
    point_df.to_csv(output_dir / f"{split_name}_point_predictions.csv", index=False, encoding="utf-8-sig")
    save_plots(point_df, output_dir, split_name, f"{split_name}: true vs predicted point mean", args.rows, args.cols)
    out = {}
    for dataset, sub in point_df.groupby("dataset"):
        sub = sub.reset_index(drop=True)
        sub.to_csv(output_dir / f"{split_name}_{dataset}_point_predictions.csv", index=False, encoding="utf-8-sig")
        out[dataset] = compute_metrics(sub[["x", "y"]].to_numpy(), sub[["pred_x", "pred_y"]].to_numpy())
    return out


def main() -> None:
    args = parse_args()
    harmonic_orders = resolved_harmonic_orders(args)
    actual_split_random_state = secrets.randbits(32) if args.randomize_split else args.split_random_state
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplconfig"))
    (args.output_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)

    print("Building window features...")
    features, dataset_meta = build_features(args)
    features.to_csv(args.output_dir / "window_features.csv", index=False, encoding="utf-8-sig")
    if args.split_mode == "random_point":
        train_df, test_df, actual_test_keys = split_random_points(
            features,
            test_size=args.test_size,
            random_state=actual_split_random_state,
            rows=args.rows,
            cols=args.cols,
            exclude_corner_test=args.exclude_corner_test,
            split_unit=args.split_unit,
        )
    elif args.split_mode == "stratified_edge":
        train_df, test_df, actual_test_keys = split_stratified_edge_points(
            features,
            test_size=args.test_size,
            random_state=actual_split_random_state,
            rows=args.rows,
            cols=args.cols,
            exclude_corner_test=args.exclude_corner_test,
            split_unit=args.split_unit,
        )
    elif args.split_mode == "limited_edge":
        train_df, test_df, actual_test_keys = split_limited_edge_points(
            features,
            test_size=args.test_size,
            max_edge_fraction=args.max_edge_test_fraction,
            random_state=actual_split_random_state,
            rows=args.rows,
            cols=args.cols,
            exclude_corner_test=args.exclude_corner_test,
            split_unit=args.split_unit,
        )
    elif args.split_mode == "constrained_random":
        train_df, test_df, actual_test_keys = split_constrained_random_points(
            features,
            test_size=args.test_size,
            random_state=actual_split_random_state,
            rows=args.rows,
            cols=args.cols,
            split_unit=args.split_unit,
            max_edge_fraction=args.constrained_max_edge_test_fraction,
            max_attempts=args.constrained_max_attempts,
        )
    else:
        train_df, test_df = split_by_keys(features, args.test_keys)
        actual_test_keys = list(args.test_keys)
    cols = feature_columns(features, args.feature_set)
    if args.training_unit == "window":
        train_df = sample_windows_per_point(train_df, args.max_windows_per_point, args.model_random_state)
        test_df = sample_windows_per_point(test_df, args.max_windows_per_point, args.model_random_state)
    if args.training_unit == "point_mean":
        train_df = aggregate_point_mean_features(train_df, cols)
        test_df = aggregate_point_mean_features(test_df, cols)
    x_train = train_df[cols].to_numpy(dtype=float)
    x_test = test_df[cols].to_numpy(dtype=float)
    y_train_rel, y_train_abs = make_targets(train_df, args)
    y_test_rel, y_test_abs = make_targets(test_df, args)

    print("Dataset feature settings:")
    print(json.dumps(dataset_meta, ensure_ascii=False, indent=2))
    print(f"Feature columns: {len(cols)}")
    print(f"Train dataset-points/windows: {count_dataset_points(train_df)} / {len(train_df)}")
    print(f"Test dataset-points/windows:  {count_dataset_points(test_df)} / {len(test_df)}")
    print(f"Datasets: {list(features['dataset'].drop_duplicates())}")
    print(f"Target mode: {args.target_mode}")
    print(f"Split mode: {args.split_mode}")
    print(f"Split unit: {args.split_unit}")
    print(f"Split random state: {actual_split_random_state}")
    corner_exclusion_active = args.exclude_corner_test or args.split_mode == "constrained_random"
    print(f"Exclude corner test: {corner_exclusion_active}")
    if args.split_mode == "constrained_random":
        print(f"Constrained max edge test fraction: {args.constrained_max_edge_test_fraction}")
        print("Constrained no-three-in-line: horizontal, vertical, and diagonal")
    print(f"Corner keys excluded from test: {get_corner_keys(features, args.rows, args.cols) if corner_exclusion_active else []}")
    print(f"Training unit: {args.training_unit}")
    print(f"Edge sample weight: {args.edge_sample_weight}")
    print(f"Near-edge sample weight: {args.near_edge_sample_weight}")
    print(f"Separate x model: {args.separate_x_model}")
    print(f"X model type: {args.x_model_type}")
    print(f"Augment x mirror: {args.augment_x_mirror}")
    print(f"X ensemble seeds: {list(args.x_ensemble_seeds) if args.x_ensemble_seeds else [args.x_model_random_state]}")
    print(f"X stretch calibration: {args.x_stretch_calibration}")
    if args.x_stretch_calibration.endswith("y_bins_mean_dist"):
        print(f"X stretch y bins: {args.x_stretch_y_bins}")
    print(f"X anchor correction: {args.x_anchor_correction}")
    print(f"X anchor record ids: {list(args.x_anchor_record_ids)}")
    print(f"X anchor auto count: {args.x_anchor_auto_count}")
    print(f"Test keys: {actual_test_keys}")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=tuple(args.hidden_layers),
                    activation="relu",
                    solver="adam",
                    alpha=args.alpha,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=args.max_iter,
                    early_stopping=not args.no_early_stopping,
                    n_iter_no_change=15,
                    random_state=args.model_random_state,
                    verbose=False,
                ),
            ),
        ]
    )
    print("Training MLP...")
    sample_weight = training_sample_weights(train_df, args)
    if sample_weight is None:
        model.fit(x_train, y_train_rel)
    else:
        model.fit(x_train, y_train_rel, mlp__sample_weight=sample_weight)
    print(f"Training done. n_iter={model.named_steps['mlp'].n_iter_}")

    train_pred_abs = pred_to_abs(model.predict(x_train), args)
    test_pred_abs = pred_to_abs(model.predict(x_test), args)
    x_model = None
    x_probability_temperature_gamma = None
    x_probability_temperature_train_metrics = None
    if args.separate_x_model:
        x_train_for_x_model = x_train
        y_train_x_for_x_model = train_df["x"].to_numpy(dtype=float)
        x_sample_weight = sample_weight
        if args.augment_x_mirror:
            x_train_for_x_model, y_train_x_for_x_model, x_sample_weight = mirror_x_training_arrays(
                train_df,
                cols,
                args,
                sample_weight,
            )
            print(f"Mirrored x training rows: {len(x_train)} -> {len(x_train_for_x_model)}")
        if args.x_model_type in ("window_mlp", "window_mlp_classifier_blend"):
            x_models = []
            train_x_preds = []
            test_x_preds = []
            seeds = list(args.x_ensemble_seeds) if args.x_ensemble_seeds else [args.x_model_random_state]
            print(f"Training separate x MLP ensemble ({len(seeds)} model(s))...")
            for seed in seeds:
                one_x_model = Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "mlp",
                            MLPRegressor(
                                hidden_layer_sizes=tuple(args.x_hidden_layers),
                                activation="relu",
                                solver="adam",
                                alpha=args.x_alpha,
                                batch_size=256,
                                learning_rate_init=1e-3,
                                max_iter=args.max_iter,
                                early_stopping=not args.no_early_stopping,
                                n_iter_no_change=20,
                                random_state=seed,
                                verbose=False,
                            ),
                        ),
                    ]
                )
                if x_sample_weight is None:
                    one_x_model.fit(x_train_for_x_model, y_train_x_for_x_model)
                else:
                    one_x_model.fit(x_train_for_x_model, y_train_x_for_x_model, mlp__sample_weight=x_sample_weight)
                x_models.append(one_x_model)
                train_x_preds.append(np.clip(one_x_model.predict(x_train), 0.0, args.cols - 1))
                test_x_preds.append(np.clip(one_x_model.predict(x_test), 0.0, args.cols - 1))
                print(f"  x seed {seed}: n_iter={one_x_model.named_steps['mlp'].n_iter_}")
            train_pred_abs[:, 0] = np.mean(np.vstack(train_x_preds), axis=0)
            test_pred_abs[:, 0] = np.mean(np.vstack(test_x_preds), axis=0)
            x_model = x_models[0] if len(x_models) == 1 else x_models
            print("Separate x training done.")
            if args.x_model_type == "window_mlp_classifier_blend":
                blend_weight = float(np.clip(args.x_blend_classifier_weight, 0.0, 1.0))
                mlp_train_x = train_pred_abs[:, 0].copy()
                mlp_test_x = test_pred_abs[:, 0].copy()
                classifier_models = []
                train_proba = np.zeros((len(train_df), args.cols), dtype=float)
                test_proba = np.zeros((len(test_df), args.cols), dtype=float)
                x_labels = np.rint(y_train_x_for_x_model).astype(int)
                print(f"Training x-column classifier blend component ({len(seeds)} model(s)); classifier weight={blend_weight:.3f}...")
                for seed in seeds:
                    clf = Pipeline(
                        steps=[
                            ("scaler", StandardScaler()),
                            (
                                "mlp",
                                MLPClassifier(
                                    hidden_layer_sizes=tuple(args.x_hidden_layers),
                                    activation="relu",
                                    solver="adam",
                                    alpha=args.x_alpha,
                                    batch_size=256,
                                    learning_rate_init=1e-3,
                                    max_iter=args.max_iter,
                                    early_stopping=not args.no_early_stopping,
                                    n_iter_no_change=20,
                                    random_state=seed,
                                    verbose=False,
                                ),
                            ),
                        ]
                    )
                    if x_sample_weight is None:
                        clf.fit(x_train_for_x_model, x_labels)
                    else:
                        clf.fit(x_train_for_x_model, x_labels, mlp__sample_weight=x_sample_weight)
                    train_proba += align_classifier_proba(clf.named_steps["mlp"], clf.predict_proba(x_train), args.cols)
                    test_proba += align_classifier_proba(clf.named_steps["mlp"], clf.predict_proba(x_test), args.cols)
                    classifier_models.append(clf)
                    print(f"  x classifier seed {seed}: n_iter={clf.named_steps['mlp'].n_iter_}")
                train_proba /= len(seeds)
                test_proba /= len(seeds)
                x_probability_temperature_gamma, x_probability_temperature_train_metrics = fit_x_probability_temperature(
                    train_df,
                    train_proba,
                    train_pred_abs[:, 1],
                    args,
                )
                classifier_train_x = x_proba_to_continuous(train_proba, x_probability_temperature_gamma)
                classifier_test_x = x_proba_to_continuous(test_proba, x_probability_temperature_gamma)
                train_pred_abs[:, 0] = (1.0 - blend_weight) * mlp_train_x + blend_weight * classifier_train_x
                test_pred_abs[:, 0] = (1.0 - blend_weight) * mlp_test_x + blend_weight * classifier_test_x
                x_model = {"window_mlp": x_models, "column_classifiers": classifier_models}
                print(f"Selected x probability temperature gamma: {x_probability_temperature_gamma:.3f}")
                print("Blended x model training done.")
        elif args.x_model_type == "point_histgbr":
            print("Training point-level x HistGradientBoosting model...")
            train_point_features = aggregate_point_mean_features_with_pred_y(train_df, cols, train_pred_abs[:, 1])
            test_point_features = aggregate_point_mean_features_with_pred_y(test_df, cols, test_pred_abs[:, 1])
            point_feature_cols = cols + ["pred_y_point_mean"]
            point_x_train = train_point_features[point_feature_cols].to_numpy(dtype=float)
            point_y_train = train_point_features["x"].to_numpy(dtype=float)
            point_x_test = test_point_features[point_feature_cols].to_numpy(dtype=float)
            point_sample_weight = training_sample_weights(train_point_features, args)
            x_model = HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.04,
                l2_regularization=0.01,
                max_leaf_nodes=15,
                random_state=args.x_model_random_state,
            )
            if point_sample_weight is None:
                x_model.fit(point_x_train, point_y_train)
            else:
                x_model.fit(point_x_train, point_y_train, sample_weight=point_sample_weight)
            train_point_x_pred = np.clip(x_model.predict(point_x_train), 0.0, args.cols - 1)
            test_point_x_pred = np.clip(x_model.predict(point_x_test), 0.0, args.cols - 1)
            train_pred_abs[:, 0] = point_predictions_to_window_values(train_df, train_point_features, train_point_x_pred)
            test_pred_abs[:, 0] = point_predictions_to_window_values(test_df, test_point_features, test_point_x_pred)
            print("Point-level x training done.")
        elif args.x_model_type == "column_classifier":
            x_models = []
            train_proba = np.zeros((len(train_df), args.cols), dtype=float)
            test_proba = np.zeros((len(test_df), args.cols), dtype=float)
            seeds = list(args.x_ensemble_seeds) if args.x_ensemble_seeds else [args.x_model_random_state]
            x_labels = np.rint(y_train_x_for_x_model).astype(int)
            print(f"Training x-column classifier ensemble ({len(seeds)} model(s))...")
            for seed in seeds:
                clf = Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "mlp",
                            MLPClassifier(
                                hidden_layer_sizes=tuple(args.x_hidden_layers),
                                activation="relu",
                                solver="adam",
                                alpha=args.x_alpha,
                                batch_size=256,
                                learning_rate_init=1e-3,
                                max_iter=args.max_iter,
                                early_stopping=not args.no_early_stopping,
                                n_iter_no_change=20,
                                random_state=seed,
                                verbose=False,
                            ),
                        ),
                    ]
                )
                if x_sample_weight is None:
                    clf.fit(x_train_for_x_model, x_labels)
                else:
                    clf.fit(x_train_for_x_model, x_labels, mlp__sample_weight=x_sample_weight)
                train_proba += align_classifier_proba(clf.named_steps["mlp"], clf.predict_proba(x_train), args.cols)
                test_proba += align_classifier_proba(clf.named_steps["mlp"], clf.predict_proba(x_test), args.cols)
                x_models.append(clf)
                print(f"  x classifier seed {seed}: n_iter={clf.named_steps['mlp'].n_iter_}")
            train_proba /= len(seeds)
            test_proba /= len(seeds)
            x_probability_temperature_gamma, x_probability_temperature_train_metrics = fit_x_probability_temperature(
                train_df,
                train_proba,
                train_pred_abs[:, 1],
                args,
            )
            train_pred_abs[:, 0] = x_proba_to_continuous(train_proba, x_probability_temperature_gamma)
            test_pred_abs[:, 0] = x_proba_to_continuous(test_proba, x_probability_temperature_gamma)
            x_model = x_models[0] if len(x_models) == 1 else x_models
            print(f"Selected x probability temperature gamma: {x_probability_temperature_gamma:.3f}")
            print("X-column classifier training done.")
        else:
            raise ValueError(f"Unsupported x_model_type={args.x_model_type!r}")
    x_stretch_calibration = {"mode": "none", "global_scale": 1.0, "bins": []}
    if args.x_stretch_calibration != "none":
        uncalibrated_train_point = point_predictions(train_df, train_pred_abs)
        x_stretch_calibration = x_stretch_calibration_from_train(uncalibrated_train_point, args)
        train_pred_abs = apply_x_stretch(train_pred_abs, x_stretch_calibration, args)
        test_pred_abs = apply_x_stretch(test_pred_abs, x_stretch_calibration, args)
        print(f"Applied x stretch calibration: {json.dumps(x_stretch_calibration, ensure_ascii=False)}")
    x_anchor_correction_fit = {"method": "none", "anchors": []}
    resolved_x_anchor_items = list(args.x_anchor_record_ids)
    if args.x_anchor_correction != "none":
        uncorrected_test_point = point_predictions(test_df, test_pred_abs)
        if args.x_anchor_auto_count > 0:
            if args.x_anchor_record_ids:
                raise ValueError("Use either --x-anchor-record-ids or --x-anchor-auto-count, not both.")
            resolved_x_anchor_items = auto_anchor_items(uncorrected_test_point, args.x_anchor_auto_count, args)
            print(f"Auto-selected x anchor keys: {resolved_x_anchor_items}")
        x_anchor_correction_fit = fit_x_anchor_correction(
            uncorrected_test_point,
            resolved_x_anchor_items,
            args.x_anchor_correction,
            args,
        )
        test_pred_abs = apply_x_anchor_correction(test_pred_abs, x_anchor_correction_fit, args)
        print(f"Applied x anchor correction: {json.dumps(x_anchor_correction_fit, ensure_ascii=False)}")
    train_point = point_predictions(train_df, train_pred_abs)
    test_point = point_predictions(test_df, test_pred_abs)
    train_by_dataset = save_outputs_by_dataset(train_point, args.output_dir, "train", args)
    test_by_dataset = save_outputs_by_dataset(test_point, args.output_dir, "test", args)
    anchor_mask, anchor_keys_used = anchor_point_mask(test_point, resolved_x_anchor_items)
    non_anchor_test_point = test_point.loc[~anchor_mask].reset_index(drop=True)
    test_point_metrics_non_anchor = None
    test_by_dataset_non_anchor = None
    if args.x_anchor_correction != "none" and not non_anchor_test_point.empty:
        test_point_metrics_non_anchor = compute_metrics(
            non_anchor_test_point[["x", "y"]].to_numpy(),
            non_anchor_test_point[["pred_x", "pred_y"]].to_numpy(),
        )
        test_by_dataset_non_anchor = {
            dataset: compute_metrics(sub[["x", "y"]].to_numpy(), sub[["pred_x", "pred_y"]].to_numpy())
            for dataset, sub in non_anchor_test_point.groupby("dataset")
        }

    result = {
        "config": {
            "data_dirs": [str(p.resolve()) for p in (args.data_dir or [ROOT / "20260428", ROOT / "20260429"])],
            "rows": args.rows,
            "cols": args.cols,
            "grid_step": args.grid_step,
            "scan_order": "horizontal_up",
            "left_sensor_xy": [args.left_sensor_x, args.left_sensor_y],
            "right_sensor_xy": [args.right_sensor_x, args.right_sensor_y],
            "reference_xy": [args.reference_x, args.reference_y],
            "target_mode": args.target_mode,
            "channels": CHANNELS,
            "dataset_channel_rules": {
                "20260428": "replace polluted measured 29 with theoretical 29; synthesize 30",
                "20260429": "use measured 29; synthesize 30",
            },
            "row_offsets": parse_row_offsets(args.row_offsets),
            "split_mode": args.split_mode,
            "split_unit": args.split_unit,
            "test_size": args.test_size,
            "max_edge_test_fraction": args.max_edge_test_fraction,
            "split_random_state": actual_split_random_state,
            "randomize_split": args.randomize_split,
            "constrained_max_edge_test_fraction": args.constrained_max_edge_test_fraction,
            "constrained_rule_no_three_in_line": "horizontal_vertical_diagonal",
            "exclude_corner_test": corner_exclusion_active,
            "excluded_corner_test_keys": get_corner_keys(features, args.rows, args.cols) if corner_exclusion_active else [],
            "test_keys": actual_test_keys,
            "feature_frequency_mode": "dataset_fundamental_harmonics_" + "_".join(f"h{i}" for i in harmonic_orders),
            "harmonic_orders": list(harmonic_orders),
            "signal_normalization": args.signal_normalization,
            "default_fundamental_freq": args.fundamental_freq,
            "dataset_fundamental_freqs": parse_float_mapping(args.dataset_fundamental_freqs),
            "dataset_feature_settings": dataset_meta,
            "feature_count": len(cols),
            "feature_set": args.feature_set,
            "training_unit": args.training_unit,
            "max_windows_per_point": args.max_windows_per_point,
            "hidden_layers": list(args.hidden_layers),
            "alpha": args.alpha,
            "separate_x_model": args.separate_x_model,
            "x_model_type": args.x_model_type,
            "x_blend_classifier_weight": args.x_blend_classifier_weight,
            "augment_x_mirror": args.augment_x_mirror,
            "x_hidden_layers": list(args.x_hidden_layers),
            "x_alpha": args.x_alpha,
            "x_model_random_state": args.x_model_random_state,
            "x_ensemble_seeds": list(args.x_ensemble_seeds) if args.x_ensemble_seeds else [args.x_model_random_state],
            "x_probability_temperature_gamma": x_probability_temperature_gamma,
            "x_probability_temperature_train_metrics": x_probability_temperature_train_metrics,
            "x_stretch_calibration": args.x_stretch_calibration,
            "x_stretch_calibration_fit": x_stretch_calibration,
            "x_anchor_correction": args.x_anchor_correction,
            "x_anchor_record_ids": list(args.x_anchor_record_ids),
            "x_anchor_auto_count": args.x_anchor_auto_count,
            "x_anchor_keys_used": anchor_keys_used,
            "x_anchor_correction_fit": x_anchor_correction_fit,
            "x_anchor_exclude_from_metrics": args.x_anchor_exclude_from_metrics,
            "edge_sample_weight": args.edge_sample_weight,
            "near_edge_sample_weight": args.near_edge_sample_weight,
            "early_stopping": not args.no_early_stopping,
            "train_points": count_dataset_points(train_df),
            "test_points": count_dataset_points(test_df),
            "train_unique_record_ids": int(train_df["record_id"].nunique()),
            "test_unique_record_ids": int(test_df["record_id"].nunique()),
            "train_windows": int(len(train_df)),
            "test_windows": int(len(test_df)),
        },
        "train_window_metrics": compute_metrics(y_train_abs, train_pred_abs),
        "test_window_metrics": compute_metrics(y_test_abs, test_pred_abs),
        "train_point_metrics": compute_metrics(train_point[["x", "y"]].to_numpy(), train_point[["pred_x", "pred_y"]].to_numpy()),
        "test_point_metrics": compute_metrics(test_point[["x", "y"]].to_numpy(), test_point[["pred_x", "pred_y"]].to_numpy()),
        "test_point_metrics_non_anchor": test_point_metrics_non_anchor,
        "train_point_metrics_by_dataset": train_by_dataset,
        "test_point_metrics_by_dataset": test_by_dataset,
        "test_point_metrics_by_dataset_non_anchor": test_by_dataset_non_anchor,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    joblib.dump({"model": model, "x_model": x_model, "feature_columns": cols, "config": result["config"]}, args.output_dir / "model.joblib")

    print("\nTrain point metrics:")
    print(json.dumps(result["train_point_metrics"], ensure_ascii=False, indent=2))
    print("\nTest point metrics:")
    print(json.dumps(result["test_point_metrics"], ensure_ascii=False, indent=2))
    if result["test_point_metrics_non_anchor"] is not None:
        print("\nTest point metrics excluding anchor points:")
        print(json.dumps(result["test_point_metrics_non_anchor"], ensure_ascii=False, indent=2))
    print("\nTest point metrics by dataset:")
    print(json.dumps(result["test_point_metrics_by_dataset"], ensure_ascii=False, indent=2))
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
