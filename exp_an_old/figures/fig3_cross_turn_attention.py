"""Figure 3: CrossTurnAttention — causal attention mechanism."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig = plt.figure(figsize=(16, 8))

# ── Left panel: causal attention mask heatmap ──
ax1 = fig.add_axes([0.04, 0.12, 0.38, 0.75])
T = 6
labels = [f"Turn {i}" for i in range(T)]

mask = np.zeros((T, T))
for t in range(T):
    for k in range(t):
        mask[t, k] = 1.0

cmap = plt.cm.colors.ListedColormap(["#FAFAFA", "#81C784"])
ax1.imshow(mask, cmap=cmap, aspect="equal", origin="upper")

for t in range(T):
    for k in range(T):
        if mask[t, k] == 1:
            ax1.text(k, t, "✓", ha="center", va="center", fontsize=14, color="#1B5E20")
        elif t == k:
            ax1.text(k, t, "Q", ha="center", va="center", fontsize=11,
                     fontweight="bold", color="#1565C0")
        else:
            ax1.text(k, t, "✗", ha="center", va="center", fontsize=12, color="#BDBDBD")

ax1.set_xticks(range(T))
ax1.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
ax1.set_yticks(range(T))
ax1.set_yticklabels(labels, fontsize=9)
ax1.set_xlabel("Key / Value (past turns)", fontsize=11, labelpad=8)
ax1.set_ylabel("Query (current turn)", fontsize=11, labelpad=8)
ax1.set_title("(a) Causal Attention Mask\nTurn t attends to turns 0..t−1", fontsize=12, fontweight="bold", pad=10)

for spine in ax1.spines.values():
    spine.set_edgecolor("#BDBDBD")
ax1.tick_params(length=0)

# ── Right panel: per-turn processing detail ──
ax2 = fig.add_axes([0.48, 0.05, 0.52, 0.88])
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 10)
ax2.axis("off")

C_BORDER = "#37474F"
C_TEXT = "#212121"

def box(x, y, w, h, text, color, fontsize=10, bold=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor=C_BORDER, linewidth=1.3)
    ax2.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax2.text(x + w/2, y + h/2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=weight, color=C_TEXT, multialignment="center")

def arrow(x1, y1, x2, y2, color=C_BORDER):
    ax2.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=13))

ax2.text(6, 9.6, "(b) CrossTurnAttention — Processing Turn t",
         ha="center", fontsize=13, fontweight="bold", color=C_TEXT)

# h_seq input
box(0.3, 8.0, 2.5, 0.9, "h_seq[:, t, :]\n(query)", "#E3F2FD", 10, True)
box(4.0, 8.0, 3.5, 0.9, "h_seq[:, 0:t, :]\n(keys & values)", "#E8F5E9", 10, True)

# MHA
box(1.5, 6.2, 5.0, 1.0, "MultiheadAttention\n(d=256, heads=8, causal)", "#BBDEFB", 10, True)
arrow(1.55, 8.0, 3.0, 7.2)
arrow(5.75, 8.0, 5.0, 7.2)

ax2.text(7.0, 6.7, "Q, K, V\nbatch_first=True", fontsize=8,
         fontstyle="italic", color="#616161")

# ctx output
box(2.3, 4.8, 3.0, 0.7, "ctx  [B, 1, d]", "#B3E5FC", 10)
arrow(4.0, 6.2, 3.8, 5.5)

# Concatenate
box(1.0, 3.4, 5.5, 0.8, "Concat [h_seq[:, t], ctx.squeeze]  →  [B, 2d]", "#FFF3E0", 9, True)
arrow(3.8, 4.8, 3.75, 4.2)

# h_seq[t] also feeds into concat
ax2.annotate("", xy=(1.8, 4.2), xytext=(1.55, 8.0),
             arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=1.2,
                             linestyle="dashed", mutation_scale=11,
                             connectionstyle="arc3,rad=-0.3"))

# Projection
box(1.5, 2.2, 4.5, 0.7, "Linear(2d → d) + GELU + Dropout", "#C8E6C9", 10)
arrow(3.75, 3.4, 3.75, 2.9)

# Residual + LayerNorm
box(1.0, 0.8, 5.5, 0.9, "Residual Add + LayerNorm\nout[:, t] = LN(proj + h_seq[:, t])", "#E1BEE7", 10, True)
arrow(3.75, 2.2, 3.75, 1.7)

# Residual skip connection
ax2.annotate("", xy=(0.8, 1.25), xytext=(0.3, 8.45),
             arrowprops=dict(arrowstyle="-|>", color="#7B1FA2", lw=1.5,
                             linestyle="dotted", mutation_scale=12,
                             connectionstyle="arc3,rad=-0.15"))
ax2.text(0.05, 4.8, "residual", fontsize=8, color="#7B1FA2", rotation=90,
         ha="center", va="center")

# Output
box(1.5, 0.0, 4.5, 0.5, "out[:, t, :]  →  contextualized turn embedding", "#F5F5F5", 9)
arrow(3.75, 0.8, 3.75, 0.5)

# Note: t=0 special case
ax2.text(8.5, 3.5, "Note: Turn 0 has no past\n→ out[:, 0] = h_seq[:, 0]\n(identity, no attention)",
         fontsize=9, color="#E65100",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4",
                   edgecolor="#FFB300", linewidth=1))

# Dimension annotation
ax2.text(8.5, 1.5, "d_model = 256\nn_heads = 8\nhead_dim = 32\ndropout = 0.2",
         fontsize=9, color="#37474F",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                   edgecolor="#BDBDBD", linewidth=1))

fig.suptitle("CrossTurnAttention Module", fontsize=15, fontweight="bold", y=0.99)
plt.savefig("/home/nvidia-lab/ai4life/CSLR/thaidc/VCS/exp_an/figures/fig3_cross_turn_attention.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("/home/nvidia-lab/ai4life/CSLR/thaidc/VCS/exp_an/figures/fig3_cross_turn_attention.pdf",
            bbox_inches="tight", facecolor="white")
print("Done: fig3_cross_turn_attention")
