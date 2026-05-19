#!/usr/bin/env python3
# coding: utf-8
"""Train on 20260430_true_inputs 162 points and predict 20260429 63 points.

This is a thin experiment wrapper around the existing sklearn MLP localization
script.  It keeps the same window-feature extraction, MLP training, point
aggregation, plotting, and optional x-anchor correction implementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
BASE_SCRIPT = (
    ROOT
    / "mlp"
    / "20260428_20260429_mlp_localization"
    / "mlp_localization_20260428_20260429_sklearn_mlp_style.py"
)

FIVE_ANCHORS = (1, 9, 32, 55, 63)
NINE_ANCHORS = (1, 5, 9, 28, 32, 36, 55, 59, 63)


def load_base_module():
    spec = importlib.util.spec_from_file_location("base_mlp_localization", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base script from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data-dir", type=Path, default=ROOT / "mlp" / "20260430_true_inputs")
    p.add_argument("--predict-data-dir", type=Path, default=ROOT / "mlp" / "20260429")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--train-rows", type=int, default=18)
    p.add_argument("--predict-record-count", type=int, default=63)
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--target-mode", choices=("center_relative", "absolute", "dual_relative"), default="center_relative")
    p.add_argument("--feature-set", default="all")
    p.add_argument("--fundamental-freq", type=float, default=15.0)
    p.add_argument(
        "--dataset-fundamental-freqs",
        nargs="*",
        default=(),
        help="Optional dataset vibration frequencies, e.g. 20260429_2:12.",
    )
    p.add_argument("--harmonic-orders", type=int, nargs="+", default=(1, 2, 3, 4))
    p.add_argument("--signal-normalization", choices=("none", "record_median", "dataset_median"), default="none")
    p.add_argument("--max-windows-per-point", type=int, default=120)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--hidden-layers", type=int, nargs="+", default=(128, 128, 64))
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--model-random-state", type=int, default=42)
    p.add_argument("--edge-sample-weight", type=float, default=1.0)
    p.add_argument("--near-edge-sample-weight", type=float, default=1.0)
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--separate-x-model", action="store_true")
    p.add_argument(
        "--x-model-type",
        choices=("window_mlp", "point_histgbr", "column_classifier", "window_mlp_classifier_blend"),
        default="window_mlp",
    )
    p.add_argument("--x-hidden-layers", type=int, nargs="+", default=(128, 128, 64))
    p.add_argument("--x-alpha", type=float, default=1e-4)
    p.add_argument("--x-model-random-state", type=int, default=42)
    p.add_argument("--x-ensemble-seeds", type=int, nargs="*", default=())
    p.add_argument("--x-blend-classifier-weight", type=float, default=0.3)
    p.add_argument("--augment-x-mirror", action="store_true")
    p.add_argument(
        "--anchor-correction",
        choices=("none", "bias", "scale_bias", "y_bias_linear", "xy_affine_residual"),
        default="none",
        help="Post-prediction x correction fitted on selected 20260429 anchor points.",
    )
    p.add_argument(
        "--anchor-layout",
        choices=("five", "nine"),
        default="five",
        help="five = four corners + center; nine also adds edge midpoints.",
    )
    p.add_argument(
        "--anchor-record-ids",
        type=int,
        nargs="*",
        default=(),
        help="Override anchor record ids. Defaults: five=1 9 32 55 63, nine=1 5 9 28 32 36 55 59 63.",
    )
    p.add_argument(
        "--train-calibration-record-ids",
        type=int,
        nargs="*",
        default=(),
        help=(
            "20260429 record ids kept in the training set as calibration points. "
            "The model is evaluated on the remaining 20260429 records."
        ),
    )
    p.add_argument(
        "--calibration-sample-weight",
        type=float,
        default=1.0,
        help="Sample-weight multiplier for 20260429 calibration rows that are kept in the training set.",
    )
    p.add_argument(
        "--keep-20260428-input29",
        action="store_true",
        help="For 20260428, keep measured left_input29 instead of replacing it with theoretical input29.",
    )
    return p.parse_args()


def default_output_dir(args: argparse.Namespace) -> Path:
    parts = ["results", args.feature_set, args.signal_normalization]
    if args.separate_x_model:
        parts.extend(["sepx", args.x_model_type])
    else:
        parts.append("joint")
    if args.anchor_correction != "none":
        parts.extend([args.anchor_layout, args.anchor_correction])
    else:
        parts.append("no_anchor")
    if args.train_calibration_record_ids:
        parts.append(f"traincal{len(args.train_calibration_record_ids)}")
    if args.calibration_sample_weight != 1.0:
        parts.append(f"calw{args.calibration_sample_weight:g}")
    return SCRIPT_DIR / "_".join(parts)


def anchor_ids(args: argparse.Namespace) -> tuple[int, ...]:
    if args.anchor_record_ids:
        return tuple(args.anchor_record_ids)
    return FIVE_ANCHORS if args.anchor_layout == "five" else NINE_ANCHORS


def build_base_argv(args: argparse.Namespace) -> list[str]:
    output_dir = args.output_dir or default_output_dir(args)
    calibration_ids = set(args.train_calibration_record_ids)
    test_keys = [
        f"{args.predict_data_dir.name}:{rid}"
        for rid in range(1, args.predict_record_count + 1)
        if rid not in calibration_ids
    ]
    if not test_keys:
        raise ValueError("No test records left after applying --train-calibration-record-ids.")
    argv = [
        str(BASE_SCRIPT),
        "--data-dir",
        str(args.train_data_dir),
        "--data-dir",
        str(args.predict_data_dir),
        "--output-dir",
        str(output_dir),
        "--rows",
        str(args.train_rows),
        "--cols",
        str(args.cols),
        "--target-mode",
        args.target_mode,
        "--split-mode",
        "keys",
        "--split-unit",
        "dataset_record",
        "--test-keys",
        *test_keys,
        "--feature-set",
        args.feature_set,
        "--fundamental-freq",
        str(args.fundamental_freq),
        "--harmonic-orders",
        *[str(v) for v in args.harmonic_orders],
        "--signal-normalization",
        args.signal_normalization,
        "--max-iter",
        str(args.max_iter),
        "--model-random-state",
        str(args.model_random_state),
        "--hidden-layers",
        *[str(v) for v in args.hidden_layers],
        "--alpha",
        str(args.alpha),
        "--edge-sample-weight",
        str(args.edge_sample_weight),
        "--near-edge-sample-weight",
        str(args.near_edge_sample_weight),
    ]
    if args.max_windows_per_point is not None:
        argv.extend(["--max-windows-per-point", str(args.max_windows_per_point)])
    if args.dataset_fundamental_freqs:
        argv.extend(["--dataset-fundamental-freqs", *list(args.dataset_fundamental_freqs)])
    if args.no_early_stopping:
        argv.append("--no-early-stopping")
    if args.separate_x_model:
        argv.extend(
            [
                "--separate-x-model",
                "--x-model-type",
                args.x_model_type,
                "--x-hidden-layers",
                *[str(v) for v in args.x_hidden_layers],
                "--x-alpha",
                str(args.x_alpha),
                "--x-model-random-state",
                str(args.x_model_random_state),
                "--x-blend-classifier-weight",
                str(args.x_blend_classifier_weight),
            ]
        )
        if args.x_ensemble_seeds:
            argv.extend(["--x-ensemble-seeds", *[str(v) for v in args.x_ensemble_seeds]])
        if args.augment_x_mirror:
            argv.append("--augment-x-mirror")
    if args.anchor_correction != "none":
        argv.extend(
            [
                "--x-anchor-correction",
                args.anchor_correction,
                "--x-anchor-record-ids",
                *[str(v) for v in anchor_ids(args)],
                "--x-anchor-exclude-from-metrics",
            ]
        )
    return argv


def main() -> None:
    args = parse_args()
    base = load_base_module()
    base_argv = build_base_argv(args)
    output_dir = args.output_dir or default_output_dir(args)
    original_training_sample_weights = base.training_sample_weights
    original_load_record = base.load_record
    if args.keep_20260428_input29:

        def load_record_keep_20260428_input29(path, dataset, signal_normalization="none", baseline=None):
            if dataset == "20260428":
                return original_load_record(path, "__20260428_keep_input29__", signal_normalization, baseline)
            return original_load_record(path, dataset, signal_normalization, baseline)

        base.load_record = load_record_keep_20260428_input29
    if args.calibration_sample_weight != 1.0:
        calibration_dataset = args.predict_data_dir.name

        def training_sample_weights_with_calibration(frame, base_args):
            weights = original_training_sample_weights(frame, base_args)
            if weights is None:
                weights = np.ones(len(frame), dtype=float)
            else:
                weights = np.asarray(weights, dtype=float).copy()
            weights[frame["dataset"].eq(calibration_dataset).to_numpy()] *= float(args.calibration_sample_weight)
            return weights

        base.training_sample_weights = training_sample_weights_with_calibration
    print("Running base MLP localization with argv:")
    print(" ".join(base_argv))
    old_argv = sys.argv
    try:
        sys.argv = base_argv
        base.main()
    finally:
        sys.argv = old_argv
        base.training_sample_weights = original_training_sample_weights
        base.load_record = original_load_record
    try:
        import visualize_63_results

        old_argv = sys.argv
        try:
            sys.argv = [
                "visualize_63_results.py",
                "--result-dir",
                str(output_dir),
                "--in-place",
            ]
            visualize_63_results.main()
        finally:
            sys.argv = old_argv
    except Exception as exc:
        print(f"Warning: failed to create in-place 63-point visualizations: {exc}")


if __name__ == "__main__":
    main()
