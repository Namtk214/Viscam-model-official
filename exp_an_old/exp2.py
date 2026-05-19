"""
Experiment 2: Adversarial data comparison.
[ThaiVersion — two-stage pipeline: binary + multiclass scenario]

Evaluates whether adversarial (Bothbosu) or out-of-domain (Tele28k) training
data improves detection on real scam calls compared to VisCam-only.

Data (dataset_scamstream-main/exp2_vs_adversarial/):
  Train options (any subset):
    bothbosu  — Bothbosu.json  (adversarial examples)
    tele28k   — Tele28k.json   (telemarketing-domain data)
    viscam    — Viscam.json    (VisCam pipeline)
  Test: real2.json

Usage examples:
  python exp2.py --train-sources viscam
  python exp2.py --train-sources bothbosu viscam
  python exp2.py --train-sources bothbosu tele28k viscam --bin-lr 1e-5
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
_EXP2 = os.path.join(DATA_ROOT, "exp2_vs_adversarial")

TRAIN_FILES = {
    "bothbosu": os.path.join(_EXP2, "train", "Bothbosu.json"),
    "tele28k":  os.path.join(_EXP2, "train", "Tele28k.json"),
    "viscam":   os.path.join(_EXP2, "train", "Viscam.json"),
}
TEST_FILE = os.path.join(_EXP2, "test", "real2.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="[ThaiVersion] Exp2: Adversarial data comparison "
                    "(Bothbosu / Tele28k / VisCam)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-sources",
        nargs="+",
        choices=["bothbosu", "tele28k", "viscam"],
        default=["viscam"],
        help="One or more training data sources (concatenated)",
    )
    add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    sources_tag = "_".join(sorted(args.train_sources))
    out_dir = args.output_dir or os.path.join(
        _DIR, "outputs", f"exp2_{sources_tag}_two_stage",
    )
    cfg = cfg_from_args(args, output_dir=out_dir)

    print(f"\n{'='*60}")
    print("EXPERIMENT 2 [ThaiVersion]: Adversarial Data Comparison")
    print(f"  Train sources: {args.train_sources}")
    print(f"  Pipeline     : two-stage (binary → multiclass)")
    print(f"  Output       : {cfg.output_dir}")
    print(f"{'='*60}")

    train_dlg = []
    for src in args.train_sources:
        path = TRAIN_FILES[src]
        data = load_json(path)
        print(f"  Loaded {os.path.basename(path)}: {len(data)} dialogues")
        train_dlg.extend(data)

    test_dlg = load_json(TEST_FILE)
    print(f"  Loaded real2.json (test): {len(test_dlg)} dialogues")

    run_two_stage_pipeline(cfg, train_dlg, test_dlg, exp_name="exp2",
                           mc_only=getattr(args, "mc_only", False))


if __name__ == "__main__":
    main()
