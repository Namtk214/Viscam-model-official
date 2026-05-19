import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize

from config import Config


# ── Loss functions ─────────────────────────────────────────────────────

def hybrid_detection_loss(
    turn_probs: torch.Tensor,
    labels: torch.Tensor,
    turn_mask: torch.Tensor,
    w_max: float = 3.0,
    w_floor: float = 0.1,
    patience_weight: float = 0.5,
    class_weight_harmless: float = 1.0,
) -> torch.Tensor:
    """
    Binary streaming detection loss (per-turn):
      First half  (t < N/2): soft-label BCE × w_floor    — evidence gathering
      Second half (t >= N/2): hard BCE × w_second (1→w_max) — commit
      Patience: penalise p_t > p_{t+1} for scam dialogues.
    """
    B, T = turn_probs.shape
    p = turn_probs.clamp(1e-6, 1 - 1e-6)

    n      = turn_mask.sum(dim=1).clamp(min=1).float()
    t_idx  = torch.arange(T, device=p.device).float()
    t_norm = t_idx.unsqueeze(0) / n.unsqueeze(1)

    is_scam  = (labels == 1).unsqueeze(1).expand_as(p)
    is_first = (t_norm < 0.5)

    y_soft    = (2.0 * t_norm).clamp(max=1.0) * (labels == 1).float().unsqueeze(1)
    bce_first = -(y_soft * torch.log(p) + (1 - y_soft) * torch.log(1 - p))

    loss_second = torch.where(is_scam, -torch.log(p), -torch.log(1 - p))
    w_second    = 1.0 + (w_max - 1.0) * (2.0 * t_norm - 1.0).clamp(min=0.0, max=1.0)

    loss_t = torch.where(
        is_first, bce_first * w_floor, loss_second * w_second,
    ) * turn_mask.float()

    loss_per_sample = loss_t.sum(dim=1) / n
    cw = torch.where(
        labels == 0,
        torch.full_like(loss_per_sample, class_weight_harmless),
        torch.ones_like(loss_per_sample),
    )
    main_loss = (loss_per_sample * cw).mean()

    scam_b     = (labels == 1).float()
    pair_mask  = (turn_mask[:, :-1] & turn_mask[:, 1:]).float()
    n_pairs    = pair_mask.sum(dim=1).clamp(min=1)
    violations = F.relu(p[:, :-1] - p[:, 1:])
    patience_loss = ((violations * pair_mask).sum(dim=1) / n_pairs * scam_b).mean()

    return main_loss + patience_weight * patience_loss


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al. 2020).

    embeddings: [B, D] L2-normalized.
    labels:     [B]   class indices.
    """
    B = embeddings.size(0)
    if B < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    sim = embeddings @ embeddings.T / temperature   # [B, B]

    diag_mask = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
    pos_mask  = (labels.unsqueeze(0) == labels.unsqueeze(1)) & diag_mask

    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    sim_max  = sim.max(dim=1, keepdim=True)[0].detach()
    exp_sim  = torch.exp(sim - sim_max) * diag_mask.float()
    log_prob = (sim - sim_max) - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    n_pos           = pos_mask.float().sum(dim=1).clamp(min=1)
    loss_per_anchor = -(pos_mask.float() * log_prob).sum(dim=1) / n_pos

    has_pos = (pos_mask.sum(dim=1) > 0).float()
    return (loss_per_anchor * has_pos).sum() / has_pos.sum().clamp(min=1)


def multiclass_contrastive_loss(
    embeddings: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    alpha: float = 0.5,
) -> torch.Tensor:
    """alpha * SupCon(embeddings) + (1-alpha) * CrossEntropy(logits)."""
    con_loss = supervised_contrastive_loss(embeddings, labels, temperature)
    ce_loss  = F.cross_entropy(logits, labels)
    return alpha * con_loss + (1.0 - alpha) * ce_loss


# ── Shared modules ─────────────────────────────────────────────────────

class CrossTurnAttention(nn.Module):
    """Causal attention: turn t attends to turns 0..t-1."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h_seq: torch.Tensor, turn_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = h_seq.shape
        out = torch.zeros_like(h_seq)
        out[:, 0, :] = h_seq[:, 0, :]
        for t in range(1, T):
            query  = h_seq[:, t:t+1, :]
            keys   = h_seq[:, :t, :]
            key_pm = ~turn_mask[:, :t]
            ctx, _ = self.mha(query, keys, keys, key_padding_mask=key_pm)
            fused  = self.proj(torch.cat([h_seq[:, t, :], ctx.squeeze(1)], dim=-1))
            out[:, t, :] = self.norm(fused + h_seq[:, t, :])
        return out


class _BackboneMixin(nn.Module):
    """Shared SentenceTransformer encoder + projection + causal cross-turn attention."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Load SentenceTransformer model
        self.encoder = SentenceTransformer(cfg.model_name)
        self._frozen = True
        # Freeze encoder parameters
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Get embedding dimension from SentenceTransformer
        embed_dim = self.encoder.get_sentence_embedding_dimension()
        d = cfg.hidden_dim
        self.proj = nn.Linear(embed_dim, d)
        self.attn = CrossTurnAttention(d, cfg.attn_heads, cfg.dropout)

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters for full fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        self._frozen = False
        # Try to enable gradient checkpointing on the underlying transformer model
        if self.cfg.use_grad_ckpt:
            if hasattr(self.encoder, "_first_module") and hasattr(self.encoder._first_module(), "gradient_checkpointing_enable"):
                self.encoder._first_module().gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                print("  Gradient checkpointing enabled on encoder.")
        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.encoder.parameters())
        print(f"  Encoder unfrozen: trainable {trainable:,}/{total:,} encoder params "
              f"({100*trainable/max(total,1):.2f}%)")

    def _encode_turns(self, turn_texts: list) -> torch.Tensor:
        """
        Encode turns using SentenceTransformer with pyvi tokenization.

        Args:
            turn_texts: List[List[str]], shape [B, T] where B=batch, T=max_turns

        Returns:
            torch.Tensor: [B, T, embed_dim]
        """
        B = len(turn_texts)
        T = len(turn_texts[0])
        cls_list = []

        for t in range(T):
            # Collect all texts at turn t across batch
            texts_t = [turn_texts[b][t] for b in range(B)]
            # Apply pyvi tokenization (only for non-empty texts)
            tokenized_texts = [tokenize(text) if text.strip() else "" for text in texts_t]

            if self._frozen:
                with torch.no_grad():
                    # SentenceTransformer.encode returns numpy array by default
                    embeddings = self.encoder.encode(
                        tokenized_texts,
                        convert_to_tensor=True,
                        show_progress_bar=False
                    )
            else:
                embeddings = self.encoder.encode(
                    tokenized_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )

            cls_list.append(embeddings)  # [B, embed_dim]

        return torch.stack(cls_list, dim=1)  # [B, T, embed_dim]

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Stage 1: Binary streaming classifier ───────────────────────────────

class BinaryM1Classifier(_BackboneMixin):
    """HaLong per-turn encoder + causal CrossTurnAttention + binary sigmoid head."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        d = cfg.hidden_dim
        self.head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(d // 2, 1),
        )

    def forward(self, input_ids, attn_masks, turn_mask, labels=None, turn_texts=None):
        turn_mask = turn_mask.bool()
        n_turns   = turn_mask.sum(dim=1).long()
        B = input_ids.size(0)

        # Use turn_texts if available, otherwise fall back to input_ids (for backward compatibility)
        if turn_texts is not None:
            cls_seq = self._encode_turns(turn_texts)
        else:
            raise ValueError("turn_texts is required for SentenceTransformer encoder")

        h_seq   = F.gelu(self.proj(cls_seq))
        h_ctx   = self.attn(h_seq, turn_mask)
        logits  = self.head(h_ctx).squeeze(-1)   # [B, T]

        turn_probs     = torch.sigmoid(logits)
        dialogue_probs = torch.stack([turn_probs[b, n_turns[b] - 1] for b in range(B)])

        loss = None
        if labels is not None:
            loss = hybrid_detection_loss(
                turn_probs, labels, turn_mask,
                w_max=self.cfg.w_max,
                w_floor=self.cfg.w_floor,
                patience_weight=self.cfg.patience_weight,
                class_weight_harmless=self.cfg.class_weight_harmless,
            )
        return {"loss": loss, "turn_probs": turn_probs, "dialogue_probs": dialogue_probs}


# ── Stage 2: Multiclass scenario classifier (per-conversation) ─────────

class ConversationMulticlassClassifier(_BackboneMixin):
    """
    Multiclass per-conversation classifier for scam scenarios (A/B/C/D).
    HaLong encoder → CrossTurnAttention → last-turn pooling
        → proj_head (for SupCon) + cls_head (for CE).
    """

    def __init__(self, cfg: Config, num_classes: int):
        super().__init__(cfg)
        self.num_classes = num_classes
        d = cfg.hidden_dim

        # SupCon projection head (L2-normalized output)
        self.proj_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d // 2),
        )

        # CE classification head
        self.cls_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(d // 2, num_classes),
        )

    def forward(self, input_ids, attn_masks, turn_mask, labels=None, turn_texts=None):
        turn_mask = turn_mask.bool()
        n_turns   = turn_mask.sum(dim=1).long()
        B = input_ids.size(0)

        # Use turn_texts if available, otherwise fall back to input_ids (for backward compatibility)
        if turn_texts is not None:
            cls_seq = self._encode_turns(turn_texts)
        else:
            raise ValueError("turn_texts is required for SentenceTransformer encoder")

        h_seq   = F.gelu(self.proj(cls_seq))
        h_ctx   = self.attn(h_seq, turn_mask)

        # Pool: hidden state of last valid turn
        conv_emb = torch.stack([h_ctx[b, n_turns[b] - 1] for b in range(B)])  # [B, d]

        proj_emb = F.normalize(self.proj_head(conv_emb), dim=-1)              # [B, d//2]
        logits   = self.cls_head(conv_emb)                                    # [B, C]
        preds    = logits.argmax(dim=-1)

        loss = None
        if labels is not None:
            loss = multiclass_contrastive_loss(
                proj_emb, logits, labels,
                temperature=self.cfg.contrastive_temp,
                alpha=self.cfg.contrastive_alpha,
            )
        return {"loss": loss, "logits": logits, "preds": preds, "embeddings": proj_emb}


# Backwards-compat alias
M1Classifier = BinaryM1Classifier
