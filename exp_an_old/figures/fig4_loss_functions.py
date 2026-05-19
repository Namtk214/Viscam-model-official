"""Figure 4: Loss Functions — Hybrid Detection Loss & Multiclass Contrastive Loss."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig = plt.figure(figsize=(18, 9))

# ════════════════════════════════════════════════════════════════════════
# Panel A: Hybrid Detection Loss visualization
# ════════════════════════════════════════════════════════════════════════
ax1 = fig.add_axes([0.05, 0.55, 0.42, 0.38])

T = 20
t_idx = np.arange(T)
N = T
t_norm = t_idx / N

# Weight schedule
w_floor = 0.1
w_max = 3.0
is_first = t_norm < 0.5
w_second = 1.0 + (w_max - 1.0) * np.clip(2.0 * t_norm - 1.0, 0, 1)
weights = np.where(is_first, w_floor, w_second)

ax1.fill_between(t_idx[:10], 0, w_floor, alpha=0.3, color="#42A5F5", label="1st half: evidence gathering")
ax1.fill_between(t_idx[10:], 0, w_second[10:], alpha=0.3, color="#EF5350", label="2nd half: commit phase")
ax1.plot(t_idx, weights, color="#212121", lw=2.5, zorder=3)
ax1.axvline(x=T/2, color="#FF9800", linestyle="--", lw=1.5, label="N/2 boundary")

ax1.set_xlabel("Turn index t", fontsize=11)
ax1.set_ylabel("Loss weight w(t)", fontsize=11)
ax1.set_title("(a) Turn-Level Loss Weights", fontsize=12, fontweight="bold")
ax1.legend(fontsize=8, loc="upper left")
ax1.set_xlim(0, T-1)
ax1.set_ylim(0, w_max + 0.3)
ax1.text(4, 0.35, f"w_floor = {w_floor}", fontsize=9, color="#1565C0", ha="center")
ax1.text(15, 2.0, f"w_max = {w_max}", fontsize=9, color="#C62828", ha="center")
ax1.grid(True, alpha=0.2)

# ────────────────────────────────────────────────────────────────────────
# Panel B: Soft labels for scam dialogues
ax2 = fig.add_axes([0.05, 0.08, 0.42, 0.38])

y_soft_scam = np.clip(2.0 * t_norm, 0, 1.0)  # scam: ramp 0→1
y_soft_harm = np.zeros(T)  # harmless: always 0

ax2.plot(t_idx, y_soft_scam, color="#EF5350", lw=2.5, label="Scam: soft label → hard 1", marker="o", ms=4)
ax2.plot(t_idx, y_soft_harm, color="#42A5F5", lw=2.5, label="Harmless: label = 0", marker="s", ms=4)
ax2.axvline(x=T/2, color="#FF9800", linestyle="--", lw=1.5)

ax2.fill_between(t_idx[:10], y_soft_scam[:10], alpha=0.15, color="#EF5350")
ax2.fill_between(t_idx[10:], 1, alpha=0.08, color="#EF5350")

ax2.set_xlabel("Turn index t", fontsize=11)
ax2.set_ylabel("Target label y(t)", fontsize=11)
ax2.set_title("(b) Soft-to-Hard Label Schedule (Scam Dialogues)", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9, loc="center right")
ax2.set_xlim(0, T-1)
ax2.set_ylim(-0.05, 1.1)
ax2.text(4, 0.55, "Soft BCE\n(evidence\ngathering)", fontsize=9, color="#C62828",
         ha="center", fontstyle="italic")
ax2.text(15, 0.7, "Hard BCE\n(commit)", fontsize=9, color="#C62828",
         ha="center", fontstyle="italic")
ax2.grid(True, alpha=0.2)

# ════════════════════════════════════════════════════════════════════════
# Panel C: Overall loss structure diagram
# ════════════════════════════════════════════════════════════════════════
ax3 = fig.add_axes([0.54, 0.08, 0.44, 0.85])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis("off")

C_BORDER = "#37474F"

def box(x, y, w, h, text, color, fontsize=10, bold=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor=C_BORDER, linewidth=1.3)
    ax3.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax3.text(x + w/2, y + h/2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=weight, color="#212121",
             multialignment="center")

def arrow(x1, y1, x2, y2, color=C_BORDER, lw=1.5):
    ax3.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=12))

# Title
ax3.text(5, 9.7, "(c) Loss Function Architecture", ha="center",
         fontsize=13, fontweight="bold")

# ── Stage 1: Hybrid Detection Loss ──
ax3.text(2.5, 9.1, "Stage 1: Binary", ha="center", fontsize=11,
         fontweight="bold", color="#1565C0")

box(0.3, 7.6, 4.4, 1.0,
    "Hybrid Detection Loss\nhybrid_detection_loss()", "#BBDEFB", 10, True)

box(0.3, 6.2, 2.0, 0.9,
    "Main Loss\nweighted BCE\n(per-turn)", "#E3F2FD", 8)
box(2.7, 6.2, 2.0, 0.9,
    "Patience Loss\npenalize p_t > p_{t+1}\n(scam only)", "#E3F2FD", 8)

arrow(1.3, 7.6, 1.3, 7.1)
arrow(3.7, 7.6, 3.7, 7.1)

ax3.text(2.3, 5.8, "+", fontsize=14, fontweight="bold", color="#1565C0",
         ha="center")
ax3.text(4.8, 6.65, "× λ_patience\n  (0.5)", fontsize=8, fontstyle="italic",
         color="#616161")

box(0.3, 4.7, 4.4, 0.7,
    "L₁ = main_loss + patience_weight × patience_loss", "#E8EAF6", 9, True)
arrow(2.5, 6.2, 2.5, 5.4)

# ── Stage 2: Multiclass Contrastive Loss ──
ax3.text(7.5, 9.1, "Stage 2: Multiclass", ha="center", fontsize=11,
         fontweight="bold", color="#2E7D32")

box(5.3, 7.6, 4.4, 1.0,
    "Multiclass Contrastive Loss\nmulticlass_contrastive_loss()", "#C8E6C9", 10, True)

box(5.3, 6.2, 2.0, 0.9,
    "SupCon Loss\nKhosla et al.\n(L2-norm emb)", "#E8F5E9", 8)
box(7.7, 6.2, 2.0, 0.9,
    "Cross-Entropy\nF.cross_entropy\n(logits, labels)", "#E8F5E9", 8)

arrow(6.3, 7.6, 6.3, 7.1)
arrow(8.7, 7.6, 8.7, 7.1)

ax3.text(7.3, 5.8, "+", fontsize=14, fontweight="bold", color="#2E7D32",
         ha="center")

box(5.3, 4.7, 4.4, 0.7,
    "L₂ = α × SupCon + (1−α) × CE", "#E8F5E9", 9, True)
arrow(7.5, 6.2, 7.5, 5.4)

ax3.text(5.6, 4.45, "α = 0.5,  τ = 0.07", fontsize=8, fontstyle="italic",
         color="#616161")

# ── Bottom: key properties ──
box(0.3, 2.8, 4.4, 1.4,
    "Key properties:\n• Soft labels in 1st half (explore)\n• Hard labels in 2nd half (commit)\n"
    "• Monotonicity via patience penalty\n• Per-sample class weighting",
    "#FFF3E0", 8)

box(5.3, 2.8, 4.4, 1.4,
    "Key properties:\n• SupCon pulls same-scenario emb together\n• CE provides classification signal\n"
    "• Temperature τ sharpens similarity\n• Balanced sampler for scenario equity",
    "#F3E5F5", 8)

# ── Inference note ──
box(0.3, 1.4, 9.4, 0.9,
    "Inference:  Dialogue → Stage 1 (p_scam ≥ 0.5?) → if YES → Stage 2 (argmax scenario) → Scam A/B/C/D\n"
    "                                                                         → if NO  →  Harmless",
    "#F5F5F5", 9)

fig.suptitle("Loss Functions: Hybrid Detection & Multiclass Contrastive",
             fontsize=15, fontweight="bold", y=1.00)

plt.savefig("/home/nvidia-lab/ai4life/CSLR/thaidc/VCS/exp_an/figures/fig4_loss_functions.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("/home/nvidia-lab/ai4life/CSLR/thaidc/VCS/exp_an/figures/fig4_loss_functions.pdf",
            bbox_inches="tight", facecolor="white")
print("Done: fig4_loss_functions")
