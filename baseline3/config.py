from dataclasses import dataclass, field
import os

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    """Two-stage pipeline config: Stage 1 (binary) + Stage 2 (multiclass scenario)."""

    # Backbone
    model_name: str = "AITeamVN/Vietnamese_Embedding"
    max_turn_len: int = 144
    max_turns: int = 20

    # Shared model
    hidden_dim: int = 256
    attn_heads: int = 8
    dropout: float = 0.2

    # Full fine-tuning (encoder unfrozen at unfreeze_epoch)
    use_grad_ckpt: bool = True

    # Binary loss (hybrid_detection_loss)
    w_max: float = 3.0
    w_floor: float = 0.1
    patience_weight: float = 0.5
    class_weight_harmless: float = 1.0

    # Multiclass contrastive loss
    contrastive_temp: float = 0.07      # temperature tau
    contrastive_alpha: float = 0.5      # alpha*SupCon + (1-alpha)*CE

    # Augmentation (truncate sub-window aug)
    truncate_aug: bool = True
    aug_k: int = 3
    aug_min_turns: int = 3

    # Stage 1 — Binary training
    bin_batch_size: int = 8
    bin_grad_accum: int = 8
    bin_lr: float = 2e-5
    bin_weight_decay: float = 1e-2
    bin_grad_clip: float = 1.0
    bin_warmup_ratio: float = 0.1
    bin_epochs: int = 50
    bin_unfreeze_epoch: int = 3
    bin_patience: int = 5

    # Stage 2 — Multiclass training
    mc_batch_size: int = 8
    mc_grad_accum: int = 8
    mc_lr: float = 3e-4
    mc_weight_decay: float = 1e-2
    mc_grad_clip: float = 1.0
    mc_warmup_ratio: float = 0.1
    mc_epochs: int = 50
    mc_unfreeze_epoch: int = 5
    mc_patience: int = 8

    # Inference / split
    binary_threshold: float = 0.5
    seed: int = 42
    val_ratio: float = 0.20

    # Paths
    output_dir: str = field(default_factory=lambda: os.path.join(_EXP_DIR, "outputs"))


# Backwards-compat alias for any callers still importing M1Config
M1Config = Config
