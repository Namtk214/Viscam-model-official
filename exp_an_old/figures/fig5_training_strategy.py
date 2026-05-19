"""Figure 5: Training Strategy — freeze/unfreeze (full fine-tuning) + augmentation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig = plt.figure(figsize=(16, 9))

# ════════════════════════════════════════════════════════════════════════
# Panel A: Training timeline (freeze → full unfreeze)
# ════════════════════════════════════════════════════════════════════════
ax1 = fig.add_axes([0.06, 0.58, 0.55, 0.35])

epochs_bin = 15
unfreeze_bin = 3
epochs_mc = 20
unfreeze_mc = 5

# Binary stage
ax1.barh(1, unfreeze_bin - 1, left=0, height=0.4, color="#90CAF9", edgecolor="#1565C0", lw=1.2)
ax1.barh(1, epochs_bin - unfreeze_bin + 1, left=unfreeze_bin - 1, height=0.4,
         color="#42A5F5", edgecolor="#1565C0", lw=1.2)

# MC stage
ax1.barh(0, unfreeze_mc - 1, left=0, height=0.4, color="#A5D6A7", edgecolor="#2E7D32", lw=1.2)
ax1.barh(0, epochs_mc - unfreeze_mc + 1, left=unfreeze_mc - 1, height=0.4,
         color="#66BB6A", edgecolor="#2E7D32", lw=1.2)

# Unfreeze markers
ax1.axvline(x=unfreeze_bin - 1, color="#E65100", linestyle="--", lw=1.5, ymin=0.55, ymax=0.95)
ax1.axvline(x=unfreeze_mc - 1, color="#E65100", linestyle="--", lw=1.5, ymin=0.05, ymax=0.45)

ax1.text(unfreeze_bin - 1, 1.35, f"Unfreeze\n(epoch {unfreeze_bin})", ha="center",
         fontsize=8, color="#E65100", fontweight="bold")
ax1.text(unfreeze_mc - 1, -0.35, f"Unfreeze\n(epoch {unfreeze_mc})", ha="center",
         fontsize=8, color="#E65100", fontweight="bold")

ax1.text(1, 1.0, "Frozen Encoder", fontsize=8, ha="center", va="center", color="#0D47A1")
ax1.text(9, 1.0, "Full Fine-tuning (lr×0.1)", fontsize=8, ha="center", va="center", color="white", fontweight="bold")
ax1.text(2, 0.0, "Frozen Encoder", fontsize=8, ha="center", va="center", color="#1B5E20")
ax1.text(12, 0.0, "Full Fine-tuning (lr×0.1)", fontsize=8, ha="center", va="center", color="white", fontweight="bold")

ax1.set_yticks([0, 1])
ax1.set_yticklabels(["Stage 2: MC\n(20 epochs)", "Stage 1: Binary\n(15 epochs)"], fontsize=9)
ax1.set_xlabel("Epoch", fontsize=10)
ax1.set_title("(a) Training Timeline: Encoder Freeze → Full Unfreeze", fontsize=12, fontweight="bold")
ax1.set_xlim(-0.5, 20)
ax1.grid(axis="x", alpha=0.2)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#90CAF9", edgecolor="#1565C0", label="Encoder frozen (head-only training)"),
    Patch(facecolor="#42A5F5", edgecolor="#1565C0", label="Full fine-tuning (all encoder params + head)"),
]
ax1.legend(handles=legend_elements, fontsize=8, loc="upper right")

# ════════════════════════════════════════════════════════════════════════
# Panel B: LR schedule visualization (cosine warmup)
# ════════════════════════════════════════════════════════════════════════
ax2 = fig.add_axes([0.65, 0.58, 0.32, 0.35])

def cosine_warmup_lr(total_steps, warmup_steps, base_lr):
    lrs = []
    for step in range(total_steps):
        if step < warmup_steps:
            lr = base_lr * step / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        lrs.append(lr)
    return lrs

# Phase 1: frozen
frozen_steps = 50
warmup1 = 5
lrs_frozen = cosine_warmup_lr(frozen_steps, warmup1, 2e-5)

# Phase 2: Full unfreeze
unfreeze_steps = 100
warmup2 = 10
lrs_unfreeze = cosine_warmup_lr(unfreeze_steps, warmup2, 2e-6)

all_lrs = lrs_frozen + lrs_unfreeze
all_steps = range(len(all_lrs))

ax2.plot(all_steps[:frozen_steps], lrs_frozen, color="#42A5F5", lw=2, label="Phase 1 (frozen)")
ax2.plot(range(frozen_steps, frozen_steps + unfreeze_steps), lrs_unfreeze, color="#EF5350", lw=2, label="Phase 2 (unfrozen, lr×0.1)")
ax2.axvline(x=frozen_steps, color="#E65100", linestyle="--", lw=1.5)
ax2.text(frozen_steps + 2, 1.5e-5, "Full\nunfreeze", fontsize=8, color="#E65100")
ax2.set_xlabel("Step", fontsize=10)
ax2.set_ylabel("Learning Rate", fontsize=10)
ax2.set_title("(b) Cosine Warmup LR Schedule", fontsize=12, fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.2)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-5, -5))

# ════════════════════════════════════════════════════════════════════════
# Panel C: Truncate Sub-Window Augmentation
# ════════════════════════════════════════════════════════════════════════
ax3 = fig.add_axes([0.06, 0.06, 0.9, 0.40])
ax3.set_xlim(0, 14)
ax3.set_ylim(0, 5)
ax3.axis("off")

ax3.text(7, 4.7, "(c) Truncate Sub-Window Augmentation (k=3, min_turns=3)",
         ha="center", fontsize=13, fontweight="bold")

C_TURN = "#BBDEFB"
C_AUG = "#FFCDD2"
C_SEL = "#FFF9C4"

def draw_dialogue(ax, x, y, turns, label, highlight=None, color=C_TURN, alpha=1.0):
    for i, t in enumerate(turns):
        c = C_SEL if highlight and i in highlight else color
        rect = FancyBboxPatch((x + i * 0.8, y), 0.65, 0.6,
                              boxstyle="round,pad=0.05", facecolor=c,
                              edgecolor="#37474F", linewidth=1, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + i * 0.8 + 0.325, y + 0.3, f"t{t}", ha="center",
                va="center", fontsize=8, alpha=alpha)
    ax.text(x - 0.2, y + 0.3, label, ha="right", va="center", fontsize=9, fontweight="bold")

# Original dialogue
turns_orig = list(range(8))
draw_dialogue(ax3, 2.0, 3.5, turns_orig, "Original", color=C_TURN)
ax3.text(2.0 + 8 * 0.8 + 0.3, 3.8, "8 turns", fontsize=9, fontstyle="italic", color="#616161")
ax3.text(2.0 + 8 * 0.8 + 0.3, 3.5, "label: scam", fontsize=9, fontstyle="italic", color="#C62828")

# Aug 1: turns 1-5
draw_dialogue(ax3, 2.0, 2.5, [1, 2, 3, 4, 5], "Aug 1", highlight=[0, 1, 2, 3, 4], color=C_AUG)
ax3.text(2.0 + 5 * 0.8 + 0.3, 2.8, "window [1:6]", fontsize=9, fontstyle="italic", color="#C62828")

# Aug 2: turns 0-3
draw_dialogue(ax3, 2.0, 1.5, [0, 1, 2, 3], "Aug 2", highlight=[0, 1, 2, 3], color=C_AUG)
ax3.text(2.0 + 4 * 0.8 + 0.3, 1.8, "window [0:4]", fontsize=9, fontstyle="italic", color="#C62828")

# Aug 3: turns 3-7
draw_dialogue(ax3, 2.0, 0.5, [3, 4, 5, 6, 7], "Aug 3", highlight=[0, 1, 2, 3, 4], color=C_AUG)
ax3.text(2.0 + 5 * 0.8 + 0.3, 0.8, "window [3:8]", fontsize=9, fontstyle="italic", color="#C62828")

# Arrows from original to augmented
for y_tgt in [2.8, 1.8, 0.8]:
    ax3.annotate("", xy=(1.8, y_tgt), xytext=(1.8, 3.5),
                 arrowprops=dict(arrowstyle="-|>", color="#FF6F00", lw=1.2,
                                 connectionstyle="arc3,rad=-0.05"))

# Properties box
ax3.text(9.5, 2.5,
         "Properties:\n"
         "• k random sub-windows per dialogue\n"
         "• min_turns ≥ 3 per window\n"
         "• Original always kept\n"
         "• Same label inherited\n"
         "• Applied before train/val split",
         fontsize=9, color="#37474F", va="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                   edgecolor="#BDBDBD", linewidth=1))

fig.suptitle("Training Strategy & Data Augmentation", fontsize=15, fontweight="bold", y=1.00)

plt.savefig("/home/nvidia-lab/ai4life/CSLR/thaidc/VCS/exp_an/figures/fig5_training_strategy.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("/home/nvidia-lab/ai4life/CSLR/thaidc/VCS/exp_an/figures/fig5_training_strategy.pdf",
            bbox_inches="tight", facecolor="white")
print("Done: fig5_training_strategy")
