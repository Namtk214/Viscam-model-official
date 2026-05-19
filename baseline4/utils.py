"""
Shared utilities for ScamStream Thai-version experiments (exp1, exp2, exp3).

Two-stage pipeline (matching notebook4d4bce95da.ipynb):
  Stage 1 — Binary model (scam vs harmless), per-turn streaming, hybrid_detection_loss
  Stage 2 — Multiclass model (scenario A/B/C/D), per-conversation, SupCon + CE
  Inference: only run Stage 2 when Stage 1 predicts scam.
"""

import dataclasses
import json
import os
import random
import time
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# ── Local imports (self-contained) ────────────────────────────────────
_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
DATA_ROOT = os.path.join(_DIR, "dataset_scamstream-main")

from config  import Config
from dataset import (
    BinaryDialogueDataset, ScamMulticlassDataset,
    build_scenario_map, collate_fn, load_json,
    make_balanced_sampler, stratified_val_split, truncate_augment,
)
from model   import BinaryM1Classifier, ConversationMulticlassClassifier
from metrics import compute_streaming_metrics, print_streaming_report


# ── Utilities ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_binary(model, loader, device, threshold: float) -> Dict:
    model.eval()
    total_loss, n_dlg = 0.0, 0
    all_labels, all_d_probs, all_t_probs = [], [], []

    for batch in loader:
        ids     = batch["input_ids"].to(device)
        masks   = batch["attn_masks"].to(device)
        tmask   = batch["turn_mask"].to(device)
        labels  = batch["labels"].to(device)
        n_turns = batch["n_turns"]
        out = model(ids, masks, tmask, labels=labels)
        B = labels.size(0)
        if out["loss"] is not None:
            total_loss += out["loss"].item() * B
        n_dlg += B

        for b in range(B):
            n = int(n_turns[b].item())
            all_labels.append(int(labels[b].item()))
            all_d_probs.append(float(out["dialogue_probs"][b].item()))
            all_t_probs.append(out["turn_probs"][b, :n].cpu().numpy())

        del out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    m = compute_streaming_metrics(all_labels, all_d_probs, all_t_probs, threshold)
    m["loss"] = total_loss / max(n_dlg, 1)
    return m


@torch.no_grad()
def evaluate_multiclass(model, loader, device, scenario_map: Dict[str, int]) -> Dict:
    """Evaluate scam-only multiclass model: accuracy + macro F1 + per-class P/R/F1."""
    from sklearn.metrics import (
        confusion_matrix, precision_recall_fscore_support,
    )

    model.eval()
    total_loss, n_dlg = 0.0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        ids    = batch["input_ids"].to(device)
        masks  = batch["attn_masks"].to(device)
        tmask  = batch["turn_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(ids, masks, tmask, labels=labels)
        B = labels.size(0)
        if out["loss"] is not None:
            total_loss += out["loss"].item() * B
        n_dlg += B
        all_preds.extend(out["preds"].cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        del out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    inv_map = {v: k for k, v in scenario_map.items()}
    if not all_labels:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0,
                "weighted_f1": 0.0, "per_class": {}, "confusion": []}

    acc          = accuracy_score(all_labels, all_preds)
    macro_f1     = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    label_ids = sorted(scenario_map.values())
    precs, recs, f1s, supports = precision_recall_fscore_support(
        all_labels, all_preds, labels=label_ids, zero_division=0,
    )
    cnt_correct = Counter(p for p, l in zip(all_preds, all_labels) if p == l)
    cnt_total   = Counter(all_labels)
    per_class = {
        inv_map.get(cid, str(cid)): {
            "precision": float(precs[i]),
            "recall":    float(recs[i]),
            "f1":        float(f1s[i]),
            "accuracy":  cnt_correct[cid] / max(cnt_total[cid], 1),
            "support":   int(supports[i]),
            "count":     int(cnt_total.get(cid, 0)),
        }
        for i, cid in enumerate(label_ids)
    }
    cm = confusion_matrix(all_labels, all_preds, labels=label_ids).tolist()

    # Macro/weighted precision & recall summary
    macro_precision    = float(np.mean(precs)) if len(precs) else 0.0
    macro_recall       = float(np.mean(recs))  if len(recs)  else 0.0
    total_support      = supports.sum()
    weighted_precision = float((precs * supports).sum() / max(total_support, 1))
    weighted_recall    = float((recs  * supports).sum() / max(total_support, 1))

    return {
        "loss":               total_loss / max(n_dlg, 1),
        "accuracy":           acc,
        "macro_f1":           macro_f1,
        "weighted_f1":        weighted_f1,
        "macro_precision":    macro_precision,
        "macro_recall":       macro_recall,
        "weighted_precision": weighted_precision,
        "weighted_recall":    weighted_recall,
        "per_class":          per_class,
        "confusion":          cm,
        "label_order":        [inv_map.get(c, str(c)) for c in label_ids],
    }


# ── Training epochs ────────────────────────────────────────────────────

def run_binary_epoch(model, loader, optimizer, scheduler, device, cfg: Config):
    model.train()
    total_loss, n_dlg = 0.0, 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        ids    = batch["input_ids"].to(device)
        masks  = batch["attn_masks"].to(device)
        tmask  = batch["turn_mask"].to(device)
        labels = batch["labels"].to(device)
        turn_texts = batch.get("turn_texts", None)

        out  = model(ids, masks, tmask, labels=labels, turn_texts=turn_texts)
        loss = out["loss"] / cfg.bin_grad_accum
        loss.backward()

        total_loss += out["loss"].item() * ids.size(0)
        n_dlg      += ids.size(0)

        if (step + 1) % cfg.bin_grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.bin_grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        del out, loss
        if step % 20 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

        if (step + 1) % 10 == 0 or (step + 1) == len(loader):
            print(f"  [{step+1}/{len(loader)}] loss={total_loss/max(n_dlg,1):.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    return total_loss / max(n_dlg, 1)


def run_mc_epoch(model, loader, optimizer, scheduler, device, cfg: Config):
    model.train()
    total_loss, n_dlg = 0.0, 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        ids    = batch["input_ids"].to(device)
        masks  = batch["attn_masks"].to(device)
        tmask  = batch["turn_mask"].to(device)
        labels = batch["labels"].to(device)
        turn_texts = batch.get("turn_texts", None)

        out  = model(ids, masks, tmask, labels=labels, turn_texts=turn_texts)
        loss = out["loss"] / cfg.mc_grad_accum
        loss.backward()

        total_loss += out["loss"].item() * ids.size(0)
        n_dlg      += ids.size(0)

        if (step + 1) % cfg.mc_grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.mc_grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        del out, loss
        if step % 20 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

        if (step + 1) % 10 == 0 or (step + 1) == len(loader):
            print(f"  [{step+1}/{len(loader)}] loss={total_loss/max(n_dlg,1):.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    avg = total_loss / max(n_dlg, 1)
    print(f"  train_loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
    return avg


# ── Report helpers ─────────────────────────────────────────────────────

def print_binary_report(m: Dict, title: str = "BINARY MODEL"):
    sep = "=" * 64
    print(f"\n{sep}\n{title}\n{sep}")
    if "loss" in m:
        print(f"  Loss:                {m['loss']:.4f}")
    print("  -- Dialogue-level --")
    print(f"  Accuracy:            {m['dialogue_accuracy']:.4f}")
    print(f"  F1:                  {m['dialogue_f1']:.4f}")
    if not np.isnan(m.get("auroc", float("nan"))):
        print(f"  AUROC:               {m['auroc']:.4f}")
    if "precision" in m:
        print(f"  Precision (scam):    {m['precision']:.4f}")
        print(f"  Recall (scam):       {m['recall']:.4f}")
        print(f"  Specificity (harm):  {m['specificity']:.4f}")
        print("  -- Confusion matrix --")
        print(f"                  pred=harm   pred=scam")
        print(f"    gt=harmless   TN={m['tn']:<8d}  FP={m['fp']:<8d}")
        print(f"    gt=scam       FN={m['fn']:<8d}  TP={m['tp']:<8d}")
    print("  -- Streaming detection --")
    print(f"  Detection rate:      {m['detection_rate']:.4f}  "
          f"({m['num_detected']}/{m['num_scam']} scam)")
    if not np.isnan(m.get("avg_detection_delay", float("nan"))):
        print(f"  Avg detection delay: {m['avg_detection_delay']:.2f} turns")
    print(f"  False alarm rate:    {m['false_alarm_rate']:.4f}  "
          f"({m['num_false_alarms']}/{m['num_harmless']} harmless)")
    if "median_alert_turn" in m:
        print("  -- Early detection (TP only) --")
        print(f"  Median alert turn:   {m['median_alert_turn']:.1f}  "
              f"(mean={m.get('mean_alert_turn', float('nan')):.1f})")
        print(f"  Median lead frac:    {m['median_lead_frac']:.3f}")
        print(f"  Alert in 1st half:   {m['alert_at_half']:.3f}")
    print(sep)


def print_mc_report(m: Dict, title: str = "MULTICLASS MODEL"):
    sep = "=" * 64
    print(f"\n{sep}\n{title}\n{sep}")
    if "loss" in m:
        print(f"  Loss:               {m['loss']:.4f}")
    print(f"  Accuracy:           {m['accuracy']:.4f}")
    print(f"  Macro F1:           {m['macro_f1']:.4f}")
    if "weighted_f1" in m:
        print(f"  Weighted F1:        {m['weighted_f1']:.4f}")
    if "macro_precision" in m:
        print(f"  Macro Precision:    {m['macro_precision']:.4f}")
        print(f"  Macro Recall:       {m['macro_recall']:.4f}")
    if "weighted_precision" in m:
        print(f"  Weighted Precision: {m['weighted_precision']:.4f}")
        print(f"  Weighted Recall:    {m['weighted_recall']:.4f}")
    print("  -- Per-scenario --")
    print(f"    {'Scenario':<12s}{'Acc':>8s}{'Prec':>8s}{'Rec':>8s}"
          f"{'F1':>8s}{'N':>6s}")
    for name in sorted(m["per_class"]):
        v = m["per_class"][name]
        print(f"    {name:<12s}{v['accuracy']:>8.4f}{v.get('precision', 0):>8.4f}"
              f"{v.get('recall', 0):>8.4f}{v.get('f1', 0):>8.4f}"
              f"{v.get('count', v.get('support', 0)):>6d}")
    cm = m.get("confusion")
    if cm:
        labels = m.get("label_order", [str(i) for i in range(len(cm))])
        print("  -- Confusion matrix (rows=gt, cols=pred) --")
        header = " " * 14 + "".join(f"{lbl:>8s}" for lbl in labels)
        print(header)
        for i, row in enumerate(cm):
            print(f"    gt={labels[i]:<8s}" + "".join(f"{v:>8d}" for v in row))
    print(sep)


def _save_metrics(path: str, metrics: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = {
        k: (float(v) if isinstance(v, (float, np.floating)) else v)
        for k, v in metrics.items() if not isinstance(v, dict)
    }
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)


def _save_config(path: str, cfg: Config, extra: Dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = dataclasses.asdict(cfg)
    if extra:
        d.update(extra)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


# ── Stage 1: Binary training ───────────────────────────────────────────

def train_binary(cfg: Config, train_dlg: List[Dict], test_dlg: List[Dict], tokenizer,
                 device, save_dir: str):
    set_seed(cfg.seed)
    print("\n" + "=" * 60)
    print("STAGE 1: BINARY TRAINING")
    print("=" * 60)

    # Split test_dlg into val (50%) and final test (50%)
    val_dlg, final_test_dlg = stratified_val_split(
        test_dlg, val_ratio=0.5, seed=cfg.seed,
    )
    print(f"Split test set: Val={len(val_dlg)} | Test={len(final_test_dlg)}")

    if cfg.truncate_aug:
        n_before  = len(train_dlg)
        train_dlg = truncate_augment(train_dlg, cfg.aug_k, cfg.aug_min_turns)
        print(f"Augmented: {n_before} → {len(train_dlg)}")

    # Use 100% of train_dlg for training (no split from train anymore)
    train_split = train_dlg
    scam_tr = sum(1 for d in train_split if d["label"] == "scam")
    harm_tr = sum(1 for d in train_split if d["label"] == "harmless")
    print(f"Train: {len(train_split)} (scam={scam_tr}, harmless={harm_tr}) "
          f"| Val: {len(val_dlg)} | Test: {len(final_test_dlg)}")

    train_ds = BinaryDialogueDataset(train_split, tokenizer, cfg.max_turn_len, cfg.max_turns)
    val_ds   = BinaryDialogueDataset(val_dlg,     tokenizer, cfg.max_turn_len, cfg.max_turns)
    test_ds  = BinaryDialogueDataset(final_test_dlg, tokenizer, cfg.max_turn_len, cfg.max_turns)

    train_loader = DataLoader(train_ds, batch_size=cfg.bin_batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,  batch_size=cfg.bin_batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg.bin_batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = BinaryM1Classifier(cfg).to(device)
    print(f"Trainable params (frozen encoder): {model.count_trainable_params():,}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.bin_lr, weight_decay=cfg.bin_weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader) // cfg.bin_grad_accum)
    frozen_steps    = steps_per_epoch * max(1, cfg.bin_unfreeze_epoch - 1)
    warmup_steps    = max(1, int(frozen_steps * cfg.bin_warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=frozen_steps,
    )

    os.makedirs(save_dir, exist_ok=True)
    best_acc, best_epoch, no_improve = -1.0, 0, 0
    best_val_m: Dict = {}

    for epoch in range(1, cfg.bin_epochs + 1):
        t0 = time.time()

        if epoch == cfg.bin_unfreeze_epoch:
            model.unfreeze_encoder()
            remaining = cfg.bin_epochs - epoch + 1
            total_uf  = steps_per_epoch * remaining
            optimizer = AdamW(model.parameters(), lr=cfg.bin_lr * 0.1,
                              weight_decay=cfg.bin_weight_decay)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=max(1, int(total_uf * cfg.bin_warmup_ratio)),
                num_training_steps=total_uf,
            )
            print(f"\nEpoch {epoch}: encoder unfrozen, lr={cfg.bin_lr*0.1:.2e} "
                  f"| trainable={model.count_trainable_params():,}")

        print(f"\n--- Binary Epoch {epoch}/{cfg.bin_epochs} ---")
        tr_loss = run_binary_epoch(model, train_loader, optimizer, scheduler, device, cfg)
        val_m   = evaluate_binary(model, val_loader, device, cfg.binary_threshold)

        elapsed = time.time() - t0
        print(f"\n  [Binary Epoch {epoch}/{cfg.bin_epochs}] "
              f"time={elapsed:.1f}s  train_loss={tr_loss:.4f}  val_loss={val_m['loss']:.4f}")
        print_binary_report(val_m, title=f"VAL @ Epoch {epoch}")

        if val_m["dialogue_accuracy"] > best_acc:
            best_acc, best_epoch, no_improve = val_m["dialogue_accuracy"], epoch, 0
            best_val_m = val_m
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
            print(f"  * Saved best (val_acc={best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg.bin_patience:
                print(f"\nEarly stop at epoch {epoch}")
                break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    model.load_state_dict(torch.load(
        os.path.join(save_dir, "model.pt"), map_location=device, weights_only=True,
    ))
    print(f"\nBest binary epoch: {best_epoch} | val_acc={best_acc:.4f}")
    test_m = evaluate_binary(model, test_loader, device, cfg.binary_threshold)
    print_binary_report(test_m, title="BINARY MODEL — TEST SET")

    _save_config(os.path.join(save_dir, "config.json"), cfg, {"stage": "binary"})
    _save_metrics(os.path.join(save_dir, "val_metrics.json"),  best_val_m)
    _save_metrics(os.path.join(save_dir, "test_metrics.json"), test_m)
    tokenizer.save_pretrained(save_dir)
    return model, test_m


# ── Stage 2: Multiclass training ───────────────────────────────────────

def train_multiclass(cfg: Config, train_dlg: List[Dict], test_dlg: List[Dict],
                     tokenizer, scenario_map: Dict[str, int], device, save_dir: str):
    set_seed(cfg.seed)
    print("\n" + "=" * 60)
    print("STAGE 2: MULTICLASS TRAINING")
    print(f"Scenario map: {scenario_map}")
    print(f"Loss: {cfg.contrastive_alpha:.1f}*SupCon + "
          f"{1 - cfg.contrastive_alpha:.1f}*CE (temp={cfg.contrastive_temp})")
    print("=" * 60)

    scam_train = [d for d in train_dlg
                  if d["label"] == "scam" and d.get("scenario") in scenario_map]
    scam_test_all = [d for d in test_dlg
                     if d["label"] == "scam" and d.get("scenario") in scenario_map]

    # Split test into val (50%) and final test (50%)
    scam_val, scam_test = stratified_val_split(
        scam_test_all, val_ratio=0.5, seed=cfg.seed,
        label_fn=lambda d: d.get("scenario", "?"),
    )
    print(f"Split scam test: Val={len(scam_val)} | Test={len(scam_test)}")

    if cfg.truncate_aug:
        n_before   = len(scam_train)
        scam_train = truncate_augment(scam_train, cfg.aug_k, cfg.aug_min_turns)
        print(f"Scam aug: {n_before} → {len(scam_train)}")

    # Use 100% of scam_train for training (no split from train anymore)
    print(f"Scam Train: {len(scam_train)}  "
          f"dist={dict(Counter(d['scenario'] for d in scam_train))}")
    print(f"Scam Val:   {len(scam_val)}   "
          f"dist={dict(Counter(d['scenario'] for d in scam_val))}")
    print(f"Scam Test:  {len(scam_test)}  "
          f"dist={dict(Counter(d['scenario'] for d in scam_test))}")

    train_ds = ScamMulticlassDataset(scam_train, tokenizer,
                                     cfg.max_turn_len, cfg.max_turns, scenario_map)
    val_ds   = ScamMulticlassDataset(scam_val, tokenizer,
                                     cfg.max_turn_len, cfg.max_turns, scenario_map)
    test_ds  = ScamMulticlassDataset(scam_test, tokenizer,
                                     cfg.max_turn_len, cfg.max_turns, scenario_map)

    sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=cfg.mc_batch_size, sampler=sampler,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,  batch_size=cfg.mc_batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg.mc_batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    num_classes = len(scenario_map)
    model = ConversationMulticlassClassifier(cfg, num_classes).to(device)
    print(f"\nNum classes: {num_classes} | Trainable (frozen): "
          f"{model.count_trainable_params():,}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.mc_lr, weight_decay=cfg.mc_weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader) // cfg.mc_grad_accum)
    frozen_steps    = steps_per_epoch * max(1, cfg.mc_unfreeze_epoch - 1)
    warmup_steps    = max(1, int(frozen_steps * cfg.mc_warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=frozen_steps,
    )

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "scenario_map.json"), "w") as f:
        json.dump(scenario_map, f)

    best_acc, best_epoch, no_improve = -1.0, 0, 0
    best_val_m: Dict = {}

    for epoch in range(1, cfg.mc_epochs + 1):
        t0 = time.time()

        if epoch == cfg.mc_unfreeze_epoch:
            model.unfreeze_encoder()
            remaining = cfg.mc_epochs - epoch + 1
            total_uf  = steps_per_epoch * remaining
            optimizer = AdamW(model.parameters(), lr=cfg.mc_lr * 0.1,
                              weight_decay=cfg.mc_weight_decay)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=max(1, int(total_uf * cfg.mc_warmup_ratio)),
                num_training_steps=total_uf,
            )
            print(f"\nEpoch {epoch}: MC encoder unfrozen, lr={cfg.mc_lr*0.1:.2e}")

        print(f"\n--- MC Epoch {epoch}/{cfg.mc_epochs} ---")
        tr_loss = run_mc_epoch(model, train_loader, optimizer, scheduler, device, cfg)
        val_m   = evaluate_multiclass(model, val_loader, device, scenario_map)

        elapsed = time.time() - t0
        print(f"\n  [MC Epoch {epoch}/{cfg.mc_epochs}] "
              f"time={elapsed:.1f}s  train_loss={tr_loss:.4f}  val_loss={val_m['loss']:.4f}")
        print_mc_report(val_m, title=f"VAL @ Epoch {epoch}")

        if val_m["accuracy"] > best_acc:
            best_acc, best_epoch, no_improve = val_m["accuracy"], epoch, 0
            best_val_m = val_m
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
            print(f"  * Saved best (val_acc={best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg.mc_patience:
                print(f"\nEarly stop at epoch {epoch}")
                break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    model.load_state_dict(torch.load(
        os.path.join(save_dir, "model.pt"), map_location=device, weights_only=True,
    ))
    print(f"\nBest MC epoch: {best_epoch} | val_acc={best_acc:.4f}")
    test_m = evaluate_multiclass(model, test_loader, device, scenario_map)
    print_mc_report(test_m, title="MULTICLASS MODEL (GT scam) — TEST SET")

    _save_config(os.path.join(save_dir, "config.json"), cfg,
                 {"stage": "multiclass", "scenario_map": scenario_map})
    _save_metrics(os.path.join(save_dir, "val_metrics.json"),
                  {k: v for k, v in best_val_m.items() if k != "per_class"})
    _save_metrics(os.path.join(save_dir, "test_metrics.json"),
                  {k: v for k, v in test_m.items() if k != "per_class"})
    tokenizer.save_pretrained(save_dir)
    return model, test_m


# ── Two-stage inference ────────────────────────────────────────────────

@torch.no_grad()
def two_stage_evaluate(binary_model, mc_model, test_dlg: List[Dict],
                       tokenizer, cfg: Config, scenario_map: Dict[str, int], device):
    """
    Full two-stage evaluation:
      Stage 1: binary model → scam/harmless per dialogue
      Stage 2: if binary=scam → multiclass → scenario A/B/C/D
    """
    binary_model.eval()
    mc_model.eval()
    inv_scenario = {v: k for k, v in scenario_map.items()}

    bin_gt, bin_pred, bin_prob = [], [], []
    mc_gt_scenario, mc_pred_scenario = [], []
    scenario_dist = Counter()

    full_ds     = BinaryDialogueDataset(test_dlg, tokenizer, cfg.max_turn_len, cfg.max_turns)
    full_loader = DataLoader(full_ds, batch_size=cfg.bin_batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)

    dlg_idx = 0
    for batch in full_loader:
        ids    = batch["input_ids"].to(device)
        masks  = batch["attn_masks"].to(device)
        tmask  = batch["turn_mask"].to(device)
        labels = batch["labels"]
        B = ids.size(0)

        out_bin = binary_model(ids, masks, tmask)
        b_probs = out_bin["dialogue_probs"].cpu()
        b_preds = (b_probs >= cfg.binary_threshold).long()

        bin_gt.extend(labels.tolist())
        bin_pred.extend(b_preds.tolist())
        bin_prob.extend(b_probs.tolist())

        scam_idx_local = b_preds.nonzero(as_tuple=True)[0].tolist()
        if scam_idx_local:
            si = torch.tensor(scam_idx_local)
            out_mc = mc_model(ids[si], masks[si], tmask[si])
            mc_preds = out_mc["preds"].cpu().tolist()
            for local_i, global_i in enumerate(scam_idx_local):
                sc_pred = inv_scenario.get(mc_preds[local_i], "?")
                scenario_dist[sc_pred] += 1
                dlg = test_dlg[dlg_idx + global_i]
                gt_sc = dlg.get("scenario") if dlg["label"] == "scam" else None
                if gt_sc in scenario_map:
                    mc_gt_scenario.append(scenario_map[gt_sc])
                    mc_pred_scenario.append(mc_preds[local_i])

        del out_bin
        if device.type == "cuda":
            torch.cuda.empty_cache()
        dlg_idx += B

    sep = "=" * 60
    print(f"\n{sep}\nTWO-STAGE PIPELINE — FULL TEST EVALUATION\n{sep}")

    bin_arr_gt   = np.array(bin_gt)
    bin_arr_pred = np.array(bin_pred)
    bin_arr_prob = np.array(bin_prob)
    try:
        auroc = roc_auc_score(bin_arr_gt, bin_arr_prob)
    except ValueError:
        auroc = float("nan")

    n_scam = (bin_arr_gt == 1).sum()
    n_harm = (bin_arr_gt == 0).sum()
    tp = ((bin_arr_pred == 1) & (bin_arr_gt == 1)).sum()
    fp = ((bin_arr_pred == 1) & (bin_arr_gt == 0)).sum()
    print("\nStage 1 — Binary:")
    print(f"  Accuracy: {accuracy_score(bin_arr_gt, bin_arr_pred):.4f}")
    print(f"  F1:       {f1_score(bin_arr_gt, bin_arr_pred, zero_division=0):.4f}")
    if not np.isnan(auroc):
        print(f"  AUROC:    {auroc:.4f}")
    print(f"  TPR:      {tp/max(n_scam,1):.4f}  ({tp}/{n_scam} scam detected)")
    print(f"  FPR:      {fp/max(n_harm,1):.4f}  ({fp}/{n_harm} harmless false-alarmed)")

    print(f"\nStage 2 — Multiclass (ran on {sum(bin_pred)} binary-scam dialogues):")
    print(f"  Scenario prediction distribution: {dict(sorted(scenario_dist.items()))}")
    if mc_pred_scenario:
        mc_acc = accuracy_score(mc_gt_scenario, mc_pred_scenario)
        mc_f1  = f1_score(mc_gt_scenario, mc_pred_scenario, average="macro", zero_division=0)
        print(f"  (On GT-scam-correctly-detected subset: n={len(mc_gt_scenario)})")
        print(f"  Accuracy:  {mc_acc:.4f}")
        print(f"  Macro F1:  {mc_f1:.4f}")
        cnt_tot = Counter(mc_gt_scenario)
        cnt_cor = Counter(p for p, l in zip(mc_pred_scenario, mc_gt_scenario) if p == l)
        for sc_idx in sorted(cnt_tot):
            sc_name = inv_scenario.get(sc_idx, str(sc_idx))
            print(f"    Scenario {sc_name}: "
                  f"{cnt_cor[sc_idx]/cnt_tot[sc_idx]:.3f} ({cnt_tot[sc_idx]})")
    print(sep)


# ── Pipeline entry point ───────────────────────────────────────────────

def run_two_stage_pipeline(cfg: Config, train_dlg: List[Dict], test_dlg: List[Dict],
                           exp_name: str = "exp", mc_only: bool = False):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Scenario map built from raw train (before augmentation)
    scenario_map = build_scenario_map(train_dlg)
    print(f"\nScenario map: {scenario_map}")
    if not scenario_map:
        print("WARNING: No scam scenarios found in training data; "
              "Stage 2 will be skipped.")

    bin_dir = os.path.join(cfg.output_dir, "binary_model")
    mc_dir  = os.path.join(cfg.output_dir, "mc_model")

    # ── Tokenizer ─────────────────────────────────────────────────────
    # Load from saved checkpoint if available (avoids re-downloading)
    if os.path.exists(os.path.join(bin_dir, "tokenizer.json")):
        print(f"\nLoading tokenizer from checkpoint: {bin_dir}")
        tokenizer = AutoTokenizer.from_pretrained(bin_dir)
    else:
        print(f"\nLoading tokenizer: {cfg.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── Stage 1 ───────────────────────────────────────────────────────
    if mc_only:
        bin_pt = os.path.join(bin_dir, "model.pt")
        if not os.path.exists(bin_pt):
            raise FileNotFoundError(
                f"--mc-only requires an existing binary checkpoint at {bin_pt}")
        print(f"\n[--mc-only] Skipping Stage 1 training. "
              f"Loading binary model from {bin_pt}")
        binary_model = BinaryM1Classifier(cfg).to(device)
        binary_model.load_state_dict(
            torch.load(bin_pt, map_location=device, weights_only=True))
        binary_model.eval()
    else:
        binary_model, _ = train_binary(
            cfg, list(train_dlg), test_dlg, tokenizer, device, bin_dir,
        )

    # ── Stage 2 ───────────────────────────────────────────────────────
    mc_model = None
    if scenario_map:
        mc_model, _ = train_multiclass(
            cfg, list(train_dlg), test_dlg, tokenizer, scenario_map, device, mc_dir,
        )
        two_stage_evaluate(binary_model, mc_model, test_dlg, tokenizer,
                           cfg, scenario_map, device)

    print(f"\nDone. Models saved to: {cfg.output_dir}")
    return binary_model, mc_model


# ── Shared argparse ────────────────────────────────────────────────────

def add_common_args(parser):
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir",   default=DATA_ROOT)
    parser.add_argument("--mc-only", action="store_true",
                        help="Skip Stage 1 (binary) training and load existing "
                             "binary_model/model.pt; only run Stage 2 (multiclass).")

    parser.add_argument("--model-name",   default=None)
    parser.add_argument("--max-turn-len", type=int, default=None)
    parser.add_argument("--max-turns",    type=int, default=None)

    parser.add_argument("--hidden-dim", type=int,   default=None)
    parser.add_argument("--attn-heads", type=int,   default=None)
    parser.add_argument("--dropout",    type=float, default=None)

    # Binary loss
    parser.add_argument("--w-max",                 type=float, default=None)
    parser.add_argument("--w-floor",               type=float, default=None)
    parser.add_argument("--patience-weight",       type=float, default=None)
    parser.add_argument("--class-weight-harmless", type=float, default=None)

    # Multiclass contrastive
    parser.add_argument("--contrastive-temp",  type=float, default=None)
    parser.add_argument("--contrastive-alpha", type=float, default=None)

    # Augmentation
    parser.add_argument("--no-truncate-aug", action="store_true")
    parser.add_argument("--aug-k",         type=int, default=None)
    parser.add_argument("--aug-min-turns", type=int, default=None)

    # Stage 1
    parser.add_argument("--bin-batch-size",     type=int,   default=None)
    parser.add_argument("--bin-grad-accum",     type=int,   default=None)
    parser.add_argument("--bin-lr",             type=float, default=None)
    parser.add_argument("--bin-weight-decay",   type=float, default=None)
    parser.add_argument("--bin-grad-clip",      type=float, default=None)
    parser.add_argument("--bin-warmup-ratio",   type=float, default=None)
    parser.add_argument("--bin-epochs",         type=int,   default=None)
    parser.add_argument("--bin-unfreeze-epoch", type=int,   default=None)
    parser.add_argument("--bin-patience",       type=int,   default=None)

    # Stage 2
    parser.add_argument("--mc-batch-size",     type=int,   default=None)
    parser.add_argument("--mc-grad-accum",     type=int,   default=None)
    parser.add_argument("--mc-lr",             type=float, default=None)
    parser.add_argument("--mc-weight-decay",   type=float, default=None)
    parser.add_argument("--mc-grad-clip",      type=float, default=None)
    parser.add_argument("--mc-warmup-ratio",   type=float, default=None)
    parser.add_argument("--mc-epochs",         type=int,   default=None)
    parser.add_argument("--mc-unfreeze-epoch", type=int,   default=None)
    parser.add_argument("--mc-patience",       type=int,   default=None)

    parser.add_argument("--binary-threshold", type=float, default=None)
    parser.add_argument("--seed",             type=int,   default=None)
    parser.add_argument("--val-ratio",        type=float, default=None)

    return parser


def cfg_from_args(args, output_dir: str) -> Config:
    cfg = Config(output_dir=output_dir)

    field_map = {
        "model_name":           getattr(args, "model_name", None),
        "max_turn_len":         getattr(args, "max_turn_len", None),
        "max_turns":            getattr(args, "max_turns", None),
        "hidden_dim":           getattr(args, "hidden_dim", None),
        "attn_heads":           getattr(args, "attn_heads", None),
        "dropout":              getattr(args, "dropout", None),

        "w_max":                getattr(args, "w_max", None),
        "w_floor":              getattr(args, "w_floor", None),
        "patience_weight":      getattr(args, "patience_weight", None),
        "class_weight_harmless":getattr(args, "class_weight_harmless", None),

        "contrastive_temp":     getattr(args, "contrastive_temp", None),
        "contrastive_alpha":    getattr(args, "contrastive_alpha", None),

        "aug_k":                getattr(args, "aug_k", None),
        "aug_min_turns":        getattr(args, "aug_min_turns", None),

        "bin_batch_size":       getattr(args, "bin_batch_size", None),
        "bin_grad_accum":       getattr(args, "bin_grad_accum", None),
        "bin_lr":               getattr(args, "bin_lr", None),
        "bin_weight_decay":     getattr(args, "bin_weight_decay", None),
        "bin_grad_clip":        getattr(args, "bin_grad_clip", None),
        "bin_warmup_ratio":     getattr(args, "bin_warmup_ratio", None),
        "bin_epochs":           getattr(args, "bin_epochs", None),
        "bin_unfreeze_epoch":   getattr(args, "bin_unfreeze_epoch", None),
        "bin_patience":         getattr(args, "bin_patience", None),

        "mc_batch_size":        getattr(args, "mc_batch_size", None),
        "mc_grad_accum":        getattr(args, "mc_grad_accum", None),
        "mc_lr":                getattr(args, "mc_lr", None),
        "mc_weight_decay":      getattr(args, "mc_weight_decay", None),
        "mc_grad_clip":         getattr(args, "mc_grad_clip", None),
        "mc_warmup_ratio":      getattr(args, "mc_warmup_ratio", None),
        "mc_epochs":            getattr(args, "mc_epochs", None),
        "mc_unfreeze_epoch":    getattr(args, "mc_unfreeze_epoch", None),
        "mc_patience":          getattr(args, "mc_patience", None),

        "binary_threshold":     getattr(args, "binary_threshold", None),
        "seed":                 getattr(args, "seed", None),
        "val_ratio":            getattr(args, "val_ratio", None),
    }
    for name, val in field_map.items():
        if val is not None:
            setattr(cfg, name, val)

    if getattr(args, "no_truncate_aug", False):
        cfg.truncate_aug = False

    return cfg
