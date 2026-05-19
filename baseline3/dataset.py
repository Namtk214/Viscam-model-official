import json
import random
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def load_json(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def truncate_augment(dialogues: List[Dict], k: int, min_turns: int) -> List[Dict]:
    """Generate k sub-window augmentations per dialogue (kept + original)."""
    augmented = []
    for dlg in dialogues:
        augmented.append(dlg)
        n = len(dlg["turns"])
        candidates = [
            (s, e)
            for s in range(n)
            for e in range(s + min_turns, n + 1)
            if not (s == 0 and e == n)
        ]
        if not candidates:
            continue
        selected = random.sample(candidates, min(k, len(candidates)))
        for start, end in selected:
            augmented.append({**dlg, "turns": dlg["turns"][start:end]})
    return augmented


def stratified_val_split(
    dialogues: List[Dict],
    val_ratio: float = 0.2,
    seed: int = 42,
    label_fn: Callable = None,
) -> Tuple[List[Dict], List[Dict]]:
    if label_fn is None:
        label_fn = lambda d: d["label"]
    groups: Dict = defaultdict(list)
    for dlg in dialogues:
        groups[label_fn(dlg)].append(dlg)
    rng = random.Random(seed)
    train_out, val_out = [], []
    for group in groups.values():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_ratio))
        val_out.extend(group[:n_val])
        train_out.extend(group[n_val:])
    rng.shuffle(train_out)
    rng.shuffle(val_out)
    return train_out, val_out


def build_scenario_map(dialogues: List[Dict]) -> Dict[str, int]:
    """Map scam scenario letters to integer indices (sorted)."""
    scenarios = sorted({
        d["scenario"]
        for d in dialogues
        if d["label"] == "scam" and d.get("scenario") is not None
    })
    return {s: i for i, s in enumerate(scenarios)}


# ── Common encoder helper ─────────────────────────────────────────────

def _encode_dialogue(dlg: Dict, tok, max_turn_len: int, max_turns: int):
    turns  = dlg["turns"][:max_turns]
    n_real = len(turns)

    ids_list, mask_list = [], []
    for turn in turns:
        enc = tok(
            turn, max_length=max_turn_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        ids_list.append(enc["input_ids"].squeeze(0))
        mask_list.append(enc["attention_mask"].squeeze(0))

    pad_ids  = torch.zeros(max_turn_len, dtype=torch.long)
    pad_mask = torch.zeros(max_turn_len, dtype=torch.long)
    for _ in range(max_turns - n_real):
        ids_list.append(pad_ids)
        mask_list.append(pad_mask)

    turn_mask = torch.zeros(max_turns, dtype=torch.bool)
    turn_mask[:n_real] = True

    # Add raw text for SentenceTransformer (padded with empty strings)
    turn_texts = turns + [""] * (max_turns - n_real)

    return torch.stack(ids_list), torch.stack(mask_list), turn_mask, n_real, turn_texts


# ── Datasets ──────────────────────────────────────────────────────────

class BinaryDialogueDataset(Dataset):
    """Dialogue dataset with binary label (scam=1, harmless=0)."""

    def __init__(self, dialogues: List[Dict], tokenizer, max_turn_len: int, max_turns: int):
        self.dialogues    = dialogues
        self.tok          = tokenizer
        self.max_turn_len = max_turn_len
        self.max_turns    = max_turns

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> Dict:
        dlg = self.dialogues[idx]
        ids, masks, tmask, n_real, turn_texts = _encode_dialogue(
            dlg, self.tok, self.max_turn_len, self.max_turns,
        )
        return {
            "input_ids":  ids,
            "attn_masks": masks,
            "turn_mask":  tmask,
            "n_turns":    torch.tensor(n_real),
            "label":      torch.tensor(1 if dlg["label"] == "scam" else 0, dtype=torch.long),
            "turn_texts": turn_texts,
        }


class ScamMulticlassDataset(Dataset):
    """
    Scam-only dataset for scenario multiclass classification.
    Label = scenario index. Prediction is per-conversation (last valid turn).
    """

    def __init__(self, dialogues: List[Dict], tokenizer,
                 max_turn_len: int, max_turns: int, scenario_map: Dict[str, int]):
        self.dialogues = [
            d for d in dialogues
            if d["label"] == "scam" and d.get("scenario") in scenario_map
        ]
        self.tok          = tokenizer
        self.max_turn_len = max_turn_len
        self.max_turns    = max_turns
        self.scenario_map = scenario_map

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> Dict:
        dlg = self.dialogues[idx]
        ids, masks, tmask, n_real, turn_texts = _encode_dialogue(
            dlg, self.tok, self.max_turn_len, self.max_turns,
        )
        return {
            "input_ids":  ids,
            "attn_masks": masks,
            "turn_mask":  tmask,
            "n_turns":    torch.tensor(n_real),
            "label":      torch.tensor(self.scenario_map[dlg["scenario"]], dtype=torch.long),
            "turn_texts": turn_texts,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "input_ids":  torch.stack([b["input_ids"]  for b in batch]),
        "attn_masks": torch.stack([b["attn_masks"] for b in batch]),
        "turn_mask":  torch.stack([b["turn_mask"]  for b in batch]),
        "n_turns":    torch.stack([b["n_turns"]    for b in batch]),
        "labels":     torch.stack([b["label"]      for b in batch]),
        "turn_texts": [b["turn_texts"] for b in batch],  # List[List[str]]
    }


def make_balanced_sampler(dataset: ScamMulticlassDataset) -> WeightedRandomSampler:
    """Balanced sampler: each scenario sampled with equal expected frequency."""
    labels = [int(dataset[i]["label"].item()) for i in range(len(dataset))]
    class_count = Counter(labels)
    weights = [1.0 / class_count[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
