#!/usr/bin/env python3
"""
train.py — Two-stage scam detection pipeline training script.

Train a Vietnamese scam detection model using XLM-RoBERTa-base
(xlm-roberta-base) with two stages:
  - Stage 1: Binary classification (scam vs harmless)
  - Stage 2: Multiclass scenario classification (A/B/C/D)

Usage:
    python train.py --train data/train.json --test data/test.json

    python train.py --train data/train.json --test data/test.json \
                    --output-dir outputs/my_experiment \
                    --bin-epochs 30 --mc-epochs 30 \
                    --bin-lr 1e-5 --mc-lr 2e-4

    python train.py --help  # See all options
"""

import argparse
import os
import sys

# Make sure local modules are importable
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from config import Config
from dataset import load_json
from utils import run_two_stage_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Vietnamese scam detection model with XLM-RoBERTa (two-stage pipeline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required arguments ──────────────────────────────────────────────
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to training JSON file",
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test JSON file",
    )

    # ── Output ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs (default: outputs/experiment)",
    )

    # ── Model configuration ─────────────────────────────────────────────
    parser.add_argument(
        "--model-name",
        type=str,
        default="xlm-roberta-base",
        help="Transformer model name",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for projection layer",
    )
    parser.add_argument(
        "--max-turn-len",
        type=int,
        default=144,
        help="Maximum tokens per turn",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum number of turns per dialogue",
    )

    # ── Stage 1: Binary training ────────────────────────────────────────
    parser.add_argument(
        "--bin-epochs",
        type=int,
        default=50,
        help="Binary model training epochs",
    )
    parser.add_argument(
        "--bin-lr",
        type=float,
        default=2e-5,
        help="Binary model learning rate",
    )
    parser.add_argument(
        "--bin-batch-size",
        type=int,
        default=8,
        help="Binary model batch size",
    )
    parser.add_argument(
        "--bin-grad-accum",
        type=int,
        default=8,
        help="Binary model gradient accumulation steps",
    )
    parser.add_argument(
        "--bin-unfreeze-epoch",
        type=int,
        default=3,
        help="Epoch to unfreeze encoder in binary training",
    )
    parser.add_argument(
        "--bin-patience",
        type=int,
        default=5,
        help="Early stopping patience for binary model",
    )

    # ── Stage 2: Multiclass training ────────────────────────────────────
    parser.add_argument(
        "--mc-epochs",
        type=int,
        default=50,
        help="Multiclass model training epochs",
    )
    parser.add_argument(
        "--mc-lr",
        type=float,
        default=3e-4,
        help="Multiclass model learning rate",
    )
    parser.add_argument(
        "--mc-batch-size",
        type=int,
        default=8,
        help="Multiclass model batch size",
    )
    parser.add_argument(
        "--mc-grad-accum",
        type=int,
        default=8,
        help="Multiclass model gradient accumulation steps",
    )
    parser.add_argument(
        "--mc-unfreeze-epoch",
        type=int,
        default=5,
        help="Epoch to unfreeze encoder in multiclass training",
    )
    parser.add_argument(
        "--mc-patience",
        type=int,
        default=8,
        help="Early stopping patience for multiclass model",
    )

    # ── Data augmentation ───────────────────────────────────────────────
    parser.add_argument(
        "--no-truncate-aug",
        action="store_true",
        help="Disable truncate augmentation",
    )
    parser.add_argument(
        "--aug-k",
        type=int,
        default=3,
        help="Number of augmented sub-windows per dialogue",
    )

    # ── Other ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=0.5,
        help="Binary classification threshold",
    )
    parser.add_argument(
        "--no-grad-ckpt",
        action="store_true",
        help="Disable gradient checkpointing",
    )

    return parser.parse_args()


def build_config(args) -> Config:
    """Build Config object from command-line arguments."""
    cfg = Config()

    # Model
    cfg.model_name = args.model_name
    cfg.hidden_dim = args.hidden_dim
    cfg.max_turn_len = args.max_turn_len
    cfg.max_turns = args.max_turns
    cfg.use_grad_ckpt = not args.no_grad_ckpt

    # Stage 1
    cfg.bin_epochs = args.bin_epochs
    cfg.bin_lr = args.bin_lr
    cfg.bin_batch_size = args.bin_batch_size
    cfg.bin_grad_accum = args.bin_grad_accum
    cfg.bin_unfreeze_epoch = args.bin_unfreeze_epoch
    cfg.bin_patience = args.bin_patience

    # Stage 2
    cfg.mc_epochs = args.mc_epochs
    cfg.mc_lr = args.mc_lr
    cfg.mc_batch_size = args.mc_batch_size
    cfg.mc_grad_accum = args.mc_grad_accum
    cfg.mc_unfreeze_epoch = args.mc_unfreeze_epoch
    cfg.mc_patience = args.mc_patience

    # Augmentation
    cfg.truncate_aug = not args.no_truncate_aug
    cfg.aug_k = args.aug_k

    # Other
    cfg.seed = args.seed
    cfg.val_ratio = args.val_ratio
    cfg.binary_threshold = args.binary_threshold

    # Output
    if args.output_dir:
        cfg.output_dir = args.output_dir
    else:
        cfg.output_dir = os.path.join(_DIR, "outputs", "experiment")

    return cfg


def main():
    args = parse_args()

    # ── Load data ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VIETNAMESE SCAM DETECTION - XLM-RoBERTa-base")
    print("=" * 70)

    print(f"\nLoading data...")
    print(f"  Train: {args.train}")
    print(f"  Test:  {args.test}")

    if not os.path.exists(args.train):
        print(f"\nERROR: Training file not found: {args.train}")
        sys.exit(1)

    if not os.path.exists(args.test):
        print(f"\nERROR: Test file not found: {args.test}")
        sys.exit(1)

    train_dlg = load_json(args.train)
    test_dlg = load_json(args.test)

    print(f"\n  Loaded {len(train_dlg)} training dialogues")
    print(f"  Loaded {len(test_dlg)} test dialogues")

    # ── Build config ────────────────────────────────────────────────────
    cfg = build_config(args)

    print(f"\nConfiguration:")
    print(f"  Model:           {cfg.model_name}")
    print(f"  Hidden dim:      {cfg.hidden_dim}")
    print(f"  Max turns:       {cfg.max_turns} × {cfg.max_turn_len} tokens")
    print(f"  Binary epochs:   {cfg.bin_epochs} (LR={cfg.bin_lr:.2e}, unfreeze@{cfg.bin_unfreeze_epoch})")
    print(f"  Multiclass epochs: {cfg.mc_epochs} (LR={cfg.mc_lr:.2e}, unfreeze@{cfg.mc_unfreeze_epoch})")
    print(f"  Augmentation:    {'enabled' if cfg.truncate_aug else 'disabled'} (k={cfg.aug_k})")
    print(f"  Val ratio:       {cfg.val_ratio}")
    print(f"  Seed:            {cfg.seed}")
    print(f"  Output:          {cfg.output_dir}")

    # ── Run training ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Starting two-stage training pipeline...")
    print("=" * 70)

    run_two_stage_pipeline(
        cfg=cfg,
        train_dlg=train_dlg,
        test_dlg=test_dlg,
        exp_name="train",
    )

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Models saved to: {cfg.output_dir}")
    print("  - binary_model/model.pt")
    print("  - mc_model/model.pt")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
