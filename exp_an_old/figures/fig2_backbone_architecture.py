"""Figure 2: Shared Backbone Architecture — _BackboneMixin detail."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.axis("off")

C = {
    "input": "#E3F2FD", "encoder": "#BBDEFB", "cls": "#90CAF9",
    "proj": "#B3E5FC", "attn": "#C8E6C9", "head_bin": "#FFCDD2",
    "head_mc": "#C8E6C9", "head_supcon": "#E1BEE7", "lora": "#FFF9C4",
    "border": "#37474F", "text": "#212121",
}

def box(x, y, w, h, text, color, fontsize=10, bold=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor=C["border"], linewidth=1.5)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=C["text"],
            multialignment="center")

def arrow(x1, y1, x2, y2, color=C["border"], lw=1.8, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=14))

def dashed_arrow(x1, y1, x2, y2, color="#7B1FA2"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                                linestyle="dashed", mutation_scale=12))

# ─── Title ───
ax.text(7.5, 9.7, "Model Architecture: Shared Backbone with Binary & Multiclass Heads",
        ha="center", fontsize=15, fontweight="bold", color=C["text"])

# ─── Input turns ───
for i in range(4):
    x = 1.0 + i * 2.2
    label = f"Turn {i+1}" if i < 3 else "Turn T"
    box(x, 8.3, 1.5, 0.7, label, C["input"], 10, True)
    if i < 3:
        ax.text(x + 0.75, 8.1, f"ids_{i+1}, mask_{i+1}", ha="center",
                fontsize=7, fontstyle="italic", color="#616161")
    else:
        ax.text(x + 0.75, 8.1, "ids_T, mask_T", ha="center",
                fontsize=7, fontstyle="italic", color="#616161")

ax.text(7.0, 8.65, "...", ha="center", fontsize=16, color="#616161")

# ─── HaLong Encoder per turn ───
for i in range(4):
    x = 1.0 + i * 2.2
    box(x, 6.8, 1.5, 0.8, "HaLong\nEncoder", C["encoder"], 9)
    arrow(x + 0.75, 8.3, x + 0.75, 7.6)

# Freeze/Unfreeze annotation
ax.text(10.2, 7.2, "  Frozen → Full Unfreeze\n  at epoch k (all params)\n  + gradient checkpointing",
        fontsize=9, color="#E65100",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C["lora"],
                  edgecolor="#FFB300", linewidth=1.2))

# ─── [CLS] extraction ───
for i in range(4):
    x = 1.0 + i * 2.2
    box(x, 5.6, 1.5, 0.7, "[CLS]", C["cls"], 10, True)
    arrow(x + 0.75, 6.8, x + 0.75, 6.3)

ax.text(10.2, 5.85, "last_hidden_state[:, 0]", fontsize=9,
        fontstyle="italic", color="#1565C0")

# ─── Stack → [B, T, embed_dim] ───
box(2.5, 4.5, 5.5, 0.65, "Stack → [B, T, embed_dim]", "#E0E0E0", 10)
for i in range(4):
    x = 1.0 + i * 2.2
    arrow(x + 0.75, 5.6, 5.25, 5.15)

# ─── Linear Projection ───
box(2.5, 3.5, 5.5, 0.65, "Linear Projection  +  GELU      →  [B, T, d=256]", C["proj"], 10)
arrow(5.25, 4.5, 5.25, 4.15)

# ─── CrossTurnAttention ───
box(2.5, 2.3, 5.5, 0.8, "CrossTurnAttention\n(causal, 8 heads, residual)", C["attn"], 11, True)
arrow(5.25, 3.5, 5.25, 3.1)
ax.text(8.2, 2.7, "h_ctx  [B, T, d]", fontsize=9, fontstyle="italic", color="#2E7D32")

# ─── Split into two heads ───
# Left: Binary head
box(0.5, 0.4, 4.5, 1.3,
    "Binary Sigmoid Head\n\nLinear(d→d/2) → GELU → Dropout\n→ Linear(d/2→1) → Sigmoid",
    C["head_bin"], 9, True)
ax.text(2.75, 1.85, "BinaryM1Classifier", fontsize=8, fontstyle="italic", color="#C62828")
ax.text(2.75, 0.1, "Output: turn_probs [B, T], dialogue_prob [B]",
        ha="center", fontsize=8, fontstyle="italic", color="#C62828")

# Right: Multiclass heads
box(5.8, 0.4, 4.5, 1.3,
    "Multiclass: Last-Turn Pooling\n\nproj_head → L2-norm (SupCon)\ncls_head → Linear(d/2→C) (CE)",
    C["head_mc"], 9, True)
ax.text(8.05, 1.85, "ConversationMulticlassClassifier", fontsize=8,
        fontstyle="italic", color="#2E7D32")
ax.text(8.05, 0.1, "Output: logits [B, C], embeddings [B, d/2]",
        ha="center", fontsize=8, fontstyle="italic", color="#2E7D32")

# Arrows to heads
arrow(4.0, 2.3, 2.75, 1.7)
arrow(6.5, 2.3, 8.05, 1.7)

# ─── "OR" label between heads ───
ax.text(5.3, 1.0, "OR", ha="center", va="center", fontsize=12,
        fontweight="bold", color="#FF6F00",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF3E0",
                  edgecolor="#FF6F00", linewidth=1.2))

# ─── Shared backbone bracket ───
ax.plot([0.3, 0.3], [2.3, 9.0], color="#78909C", lw=2, linestyle="--")
ax.plot([0.3, 0.6], [9.0, 9.0], color="#78909C", lw=2, linestyle="--")
ax.plot([0.3, 0.6], [2.3, 2.3], color="#78909C", lw=2, linestyle="--")
ax.text(0.15, 5.65, "Shared\nBackbone\n(_BackboneMixin)", ha="center", va="center",
        fontsize=8, color="#78909C", rotation=90)

# ─── Dimension annotations ───
box(11.0, 4.5, 3.5, 0.65, "embed_dim = 768\n(HaLong hidden_size)", "#F3E5F5", 9)
box(11.0, 3.5, 3.5, 0.65, "d = hidden_dim = 256\nattn_heads = 8", "#F3E5F5", 9)

plt.tight_layout()
plt.savefig("/home/nvidia-lab/ai4life/CSLR/thaidc/VCS/exp_an/figures/fig2_backbone_architecture.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("/home/nvidia-lab/ai4life/CSLR/thaidc/VCS/exp_an/figures/fig2_backbone_architecture.pdf",
            bbox_inches="tight", facecolor="white")
print("Done: fig2_backbone_architecture")
