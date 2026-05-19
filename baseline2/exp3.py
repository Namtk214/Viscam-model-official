"""
Experiment 3: Real data + VisCam synthetic combination.
[ThaiVersion — two-stage pipeline: binary + multiclass scenario]

Tests whether augmenting limited real labeled data with VisCam-generated
synthetic dialogues improves detection performance.

Data (dataset_scamstream-main/exp3_real+syn/):
  Train options:
    real1   — real1.json   (real labeled data, half 1)
    viscam  — Viscam.json  (VisCam synthetic data, size-matched to real1)
    both    — real1 + Viscam concatenated  (recommended)
  Test: real2.json

Usage examples:
  python exp3.py --train-source both
  python exp3.py --train-source real1
  python exp3.py --train-source viscam --bin-epochs 15
  python exp3.py --train-source both --bin-lr 3e-5 --no-truncate-aug
"""

import argparse
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from utils import (
    DATA_ROOT, add_common_args, cfg_from_args, load_json,
    run_two_stage_pipeline,
)

# ── Data paths ──────────────────────────────────────────────────────────
_EXP3 = os.path.join(DATA_ROOT, "exp3_real+syn")

TRAIN_FILES = {
    "real1":  [os.path.join(_EXP3, "train", "real1.json")],
    "viscam": [os.path.join(_EXP3, "train", "Viscam.json")],
    "both":   [
        os.path.join(_EXP3, "train", "real1.json"),
        os.path.join(_EXP3, "train", "Viscam.json"),
    ],
}
TEST_FILE = os.path.join(_EXP3, "test", "real2.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="[ThaiVersion] Exp3: Real data (real1) + VisCam synthetic combination",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-source",
        choices=["real1", "viscam", "both"],
        default="both",
        help="Training data: real1 only, viscam only, or both concatenated",
    )
    add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = args.output_dir or os.path.join(
        _DIR, "outputs", f"exp3_{args.train_source}_two_stage",
    )
    cfg = cfg_from_args(args, output_dir=out_dir)

    print(f"\n{'='*60}")
    print("EXPERIMENT 3 [ThaiVersion]: Real + VisCam Synthetic")
    print(f"  Train source : {args.train_source}")
    print(f"  Pipeline     : two-stage (binary → multiclass)")
    print(f"  Output       : {cfg.output_dir}")
    print(f"{'='*60}")

    train_dlg = []
    for path in TRAIN_FILES[args.train_source]:
        data = load_json(path)
        print(f"  Loaded {os.path.basename(path)}: {len(data)} dialogues")
        train_dlg.extend(data)

    test_dlg = load_json(TEST_FILE)
    print(f"  Loaded real2.json (test): {len(test_dlg)} dialogues")

    run_two_stage_pipeline(cfg, train_dlg, test_dlg, exp_name="exp3")


if __name__ == "__main__":
    main()
