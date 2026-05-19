"""Figure 1: Overall Two-Stage Pipeline Architecture."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis("off")

# Color palette
C_INPUT  = "#E8F4FD"
C_STAGE1 = "#BBDEFB"
C_STAGE2 = "#C8E6C9"
C_OUTPUT = "#FFF9C4"
C_DECISION = "#FFCDD2"
C_ARROW  = "#455A64"
C_BORDER = "#37474F"
C_TEXT   = "#212121"

def box(x, y, w, h, text, color, fontsize=11, bold=False, border_color=C_BORDER):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor=border_color, linewidth=1.5)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=C_TEXT,
            multialignment="center")

def diamond(cx, cy, size, text, color):
    pts = [(cx, cy+size), (cx+size*1.3, cy), (cx, cy-size), (cx-size*1.3, cy)]
    poly = plt.Polygon(pts, facecolor=color, edgecolor=C_BORDER, linewidth=1.5)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10,
            fontweight="bold", color=C_TEXT)

def arrow(x1, y1, x2, y2, text="", text_offset=(0, 0.15)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=2, mutation_scale=15))
    if text:
        mx, my = (x1+x2)/2 + text_offset[0], (y1+y2)/2 + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=9,
                color=C_ARROW, fontstyle="italic")

# --- Title ---
ax.text(8, 6.6, "Two-Stage Streaming Scam Detection Pipeline",
        ha="center", va="center", fontsize=16, fontweight="bold", color=C_TEXT)

# --- Input ---
box(0.3, 4.0, 2.8, 1.2, "Input Dialogue\n(multi-turn\nconversation)", C_INPUT, 11, True)

# --- Stage 1: Binary ---
box(4.2, 3.7, 3.5, 1.8, "Stage 1: Binary Classifier\n\nHaLong Encoder\n→ CrossTurnAttention\n→ Sigmoid Head",
    C_STAGE1, 10, True)
ax.text(5.95, 5.7, "BinaryM1Classifier", ha="center", fontsize=9,
        fontstyle="italic", color="#1565C0")

# Arrow: input → stage1
arrow(3.1, 4.6, 4.2, 4.6)

# --- Decision diamond ---
diamond(9.3, 4.6, 0.6, "Scam?", C_DECISION)

# Arrow: stage1 → decision
arrow(7.7, 4.6, 8.0, 4.6, "p(scam)")

# --- Stage 2: Multiclass ---
box(11.0, 3.7, 3.8, 1.8,
    "Stage 2: Scenario Classifier\n\nHaLong Encoder\n→ CrossTurnAttention\n→ SupCon + CE Heads",
    C_STAGE2, 10, True)
ax.text(12.9, 5.7, "ConversationMulticlassClassifier", ha="center", fontsize=9,
        fontstyle="italic", color="#2E7D32")

# Arrow: decision → stage2 (Yes)
arrow(10.6, 4.6, 11.0, 4.6, "Yes", (0, 0.2))

# --- Output: Harmless ---
box(7.7, 1.2, 2.5, 0.9, "Harmless\n(no scenario)", C_OUTPUT, 11, True)

# Arrow: decision → harmless (No)
arrow(9.3, 4.0, 9.0, 2.1, "No", (-0.35, 0))

# --- Output: Scenario ---
box(11.3, 1.2, 3.2, 0.9, "Scam Scenario\nA / B / C / D", C_OUTPUT, 11, True)

# Arrow: stage2 → scenario
arrow(12.9, 3.7, 12.9, 2.1)

# --- Loss labels ---
box(4.5, 1.2, 2.8, 0.9, "Hybrid Detection\nLoss", "#E1BEE7", 10, False)
ax.annotate("", xy=(5.9, 2.1), xytext=(5.9, 3.7),
            arrowprops=dict(arrowstyle="-|>", color="#7B1FA2",
                            lw=1.5, linestyle="dashed", mutation_scale=12))

box(0.5, 1.2, 2.5, 0.9, "SupCon + CE\nLoss", "#E1BEE7", 10, False)
ax.text(1.75, 0.5, "α · SupCon + (1−α) · CE", ha="center", fontsize=9,
        fontstyle="italic", color="#7B1FA2")

# --- Per-turn streaming annotation ---
ax.text(5.95, 3.3, "per-turn streaming\n(sigmoid probability)", ha="center",
        fontsize=8, fontstyle="italic", color="#1565C0")
ax.text(12.9, 3.3, "per-conversation\n(last-turn pooling)", ha="center",
        fontsize=8, fontstyle="italic", color="#2E7D32")

# --- Training strategy note ---
box(0.3, 0.1, 15.2, 0.5, "Training: Encoder frozen → full unfreeze at epoch k  |  AdamW + cosine warmup  |  gradient checkpointing  |  truncate augmentation  |  early stopping",
    "#F5F5F5", 9, False, "#BDBDBD")

plt.tight_layout()
plt.savefig("/home/nvidia-lab/ai4life/CSLR/thaidc/VCS/exp_an/figures/fig1_two_stage_pipeline.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("/home/nvidia-lab/ai4life/CSLR/thaidc/VCS/exp_an/figures/fig1_two_stage_pipeline.pdf",
            bbox_inches="tight", facecolor="white")
print("Done: fig1_two_stage_pipeline")
