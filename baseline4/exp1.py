"""
Experiment 1: Prompt-based generation (Vanilla) vs VisCam synthetic data.
[ThaiVersion — two-stage pipeline: binary + multiclass scenario]

Compares two synthetic data generation strategies on the same architecture
and the same real-world test set (real2).

Data (dataset_scamstream-main/exp1_prompt_vs_viscam/):
  Train options:
    vanilla  — Vanilla.json  (prompt-based generation)
    viscam   — Viscam.json   (VisCam 4-agent pipeline)
    both     — Vanilla + Viscam concatenated
  Test: real2.json

Usage examples:
  python exp1.py --train-source viscam
  python exp1.py --train-source vanilla --bin-lr 1e-5
  python exp1.py --train-source both --bin-epochs 20 --output-dir ./out/exp1_both
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
_EXP1 = os.path.join(DATA_ROOT, "exp1_prompt_vs_viscam")

TRAIN_FILES = {
    "vanilla": [os.path.join(_EXP1, "train", "Vanilla.json")],
    "viscam":  [os.path.join(_EXP1, "train", "Viscam.json")],
    "both":    [
        os.path.join(_EXP1, "train", "Vanilla.json"),
        os.path.join(_EXP1, "train", "Viscam.json"),
    ],
}
TEST_FILE = os.path.join(_EXP1, "test", "real2.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="[ThaiVersion] Exp1: Prompt-based (Vanilla) vs VisCam synthetic data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-source",
        choices=["vanilla", "viscam", "both"],
        default="viscam",
        help="Training data source",
    )
    add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = args.output_dir or os.path.join(
        _DIR, "outputs", f"exp1_{args.train_source}_two_stage",
    )
    cfg = cfg_from_args(args, output_dir=out_dir)

    print(f"\n{'='*60}")
    print("EXPERIMENT 1 [ThaiVersion]: Prompt-based (Vanilla) vs VisCam")
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

    run_two_stage_pipeline(cfg, train_dlg, test_dlg, exp_name="exp1")


if __name__ == "__main__":
    main()
