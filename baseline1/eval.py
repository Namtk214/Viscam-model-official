"""
eval.py — Two-stage pipeline evaluation script.

Usage:
    python eval.py --ckpt-dir outputs/exp2_bothbosu_two_stage --data path/to/data.json

Arguments:
    --ckpt-dir   Root output directory (must contain binary_model/ and mc_model/).
    --data       Path to the JSON dialogue file to evaluate on.
    --threshold  Binary scam threshold (default: from config, fallback 0.5).
    --batch-size Batch size for inference (default: 8).
    --device     'cuda' or 'cpu' (default: auto-detect).
    --save-json  Optional path to save metrics as JSON.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# ── make sure local modules are importable ──────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from config  import Config
from dataset import BinaryDialogueDataset, ScamMulticlassDataset, collate_fn, load_json
from metrics import _first_alert_turn
from model   import BinaryM1Classifier, ConversationMulticlassClassifier


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

def _load_config(ckpt_dir: str) -> Config:
    """Load Config from binary_model/config.json if present, else use defaults."""
    cfg_path = os.path.join(ckpt_dir, "binary_model", "config.json")
    cfg = Config()
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            d = json.load(f)
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def _load_scenario_map(ckpt_dir: str):
    path = os.path.join(ckpt_dir, "mc_model", "scenario_map.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_binary_model(ckpt_dir: str, cfg: Config, device):
    pt = os.path.join(ckpt_dir, "binary_model", "model.pt")
    model = BinaryM1Classifier(cfg).to(device)
    model.load_state_dict(torch.load(pt, map_location=device, weights_only=True))
    model.eval()
    return model


def _load_mc_model(ckpt_dir: str, cfg: Config, scenario_map, device):
    pt = os.path.join(ckpt_dir, "mc_model", "model.pt")
    model = ConversationMulticlassClassifier(cfg, num_classes=len(scenario_map)).to(device)
    model.load_state_dict(torch.load(pt, map_location=device, weights_only=True))
    model.eval()
    return model


# ───────────────────────────────────────────────────────────────────────
# Stage-1 binary inference
# ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_binary_inference(model, dialogues, tokenizer, cfg, device, batch_size):
    ds     = BinaryDialogueDataset(dialogues, tokenizer, cfg.max_turn_len, cfg.max_turns)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2, pin_memory=(device.type == "cuda"))

    all_labels, all_d_probs, all_t_probs = [], [], []
    for batch in loader:
        ids     = batch["input_ids"].to(device)
        masks   = batch["attn_masks"].to(device)
        tmask   = batch["turn_mask"].to(device)
        labels  = batch["labels"]
        n_turns = batch["n_turns"]
        turn_texts = batch.get("turn_texts", None)

        out = model(ids, masks, tmask, turn_texts=turn_texts)
        B   = ids.size(0)
        for b in range(B):
            n = int(n_turns[b].item())
            all_labels.append(int(labels[b].item()))
            all_d_probs.append(float(out["dialogue_probs"][b].item()))
            all_t_probs.append(out["turn_probs"][b, :n].cpu().numpy())

        del out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return all_labels, all_d_probs, all_t_probs


# ───────────────────────────────────────────────────────────────────────
# Stage-2 MC inference (on binary-scam predicted dialogues)
# ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_mc_inference(model, dialogues, tokenizer, cfg, scenario_map, device, batch_size):
    """Run MC model on a list of dialogues. Returns (gt_labels, pred_labels)."""
    scam_dialogues = [d for d in dialogues if d.get("scenario") in scenario_map]
    if not scam_dialogues:
        return [], []

    ds     = ScamMulticlassDataset(scam_dialogues, tokenizer,
                                   cfg.max_turn_len, cfg.max_turns, scenario_map)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2, pin_memory=(device.type == "cuda"))

    all_preds, all_labels = [], []
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        masks  = batch["attn_masks"].to(device)
        tmask  = batch["turn_mask"].to(device)
        labels = batch["labels"]
        turn_texts = batch.get("turn_texts", None)

        out = model(ids, masks, tmask, turn_texts=turn_texts)
        all_preds.extend(out["preds"].cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        del out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return all_labels, all_preds


# ───────────────────────────────────────────────────────────────────────
# Metric computation helpers
# ───────────────────────────────────────────────────────────────────────

def compute_binary_metrics(all_labels, all_d_probs, all_t_probs, threshold):
    labels = np.array(all_labels)
    probs  = np.array(all_d_probs)
    preds  = (probs >= threshold).astype(int)

    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, zero_division=0)
    pre  = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)

    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    specificity = tn / max(tn + fp, 1)

    # Streaming detection
    num_scam, num_detected   = 0, 0
    num_harm, num_false_alarm = 0, 0
    detection_delays         = []

    for label, t_probs in zip(all_labels, all_t_probs):
        first_alert = _first_alert_turn(t_probs, threshold)
        if label == 1:
            num_scam += 1
            if first_alert is not None:
                num_detected += 1
                detection_delays.append(first_alert)
        else:
            num_harm += 1
            if first_alert is not None:
                num_false_alarm += 1

    detection_rate  = num_detected    / max(num_scam, 1)
    false_alarm_rate= num_false_alarm / max(num_harm, 1)
    avg_delay       = float(np.mean(detection_delays)) if detection_delays else float("nan")

    return {
        "Acc":            acc,
        "F1":             f1,
        "Pre":            pre,
        "Recall":         rec,
        "AUROC":          auroc,
        "Specificity":    specificity,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "N_scam":         num_scam,
        "N_harmless":     num_harm,
        "Detection_rate": detection_rate,
        "Detected":       num_detected,
        "False_alarm":    false_alarm_rate,
        "False_alarms":   num_false_alarm,
        "Avg_delay":      avg_delay,
    }


def compute_mc_metrics(all_labels, all_preds, scenario_map):
    inv = {v: k for k, v in scenario_map.items()}
    label_ids = sorted(scenario_map.values())

    acc        = accuracy_score(all_labels, all_preds)
    macro_f1   = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1= f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_pre  = precision_score(all_labels, all_preds, average="macro",    zero_division=0)
    macro_rec  = recall_score(all_labels, all_preds,    average="macro",    zero_division=0)

    precs, recs, f1s, supports = precision_recall_fscore_support(
        all_labels, all_preds, labels=label_ids, zero_division=0,
    )
    from collections import Counter
    cnt_total   = Counter(all_labels)
    cnt_correct = Counter(p for p, l in zip(all_preds, all_labels) if p == l)

    per_class = {}
    for i, cid in enumerate(label_ids):
        name = inv.get(cid, str(cid))
        per_class[name] = {
            "Acc":  cnt_correct[cid] / max(cnt_total[cid], 1),
            "Pre":  float(precs[i]),
            "Rec":  float(recs[i]),
            "F1":   float(f1s[i]),
            "N":    int(cnt_total.get(cid, 0)),
        }

    cm = confusion_matrix(all_labels, all_preds, labels=label_ids).tolist()
    return {
        "Acc":              acc,
        "Macro_F1":         macro_f1,
        "Weighted_F1":      weighted_f1,
        "Macro_Pre":        macro_pre,
        "Macro_Rec":        macro_rec,
        "per_class":        per_class,
        "confusion":        cm,
        "label_order":      [inv.get(c, str(c)) for c in label_ids],
    }


# ───────────────────────────────────────────────────────────────────────
# Pretty print
# ───────────────────────────────────────────────────────────────────────

def print_binary_results(m):
    sep = "=" * 64
    print(f"\n{sep}")
    print("STAGE 1 — BINARY MODEL RESULTS")
    print(sep)
    print(f"  {'Metric':<22}  {'Value':>10}")
    print(f"  {'-'*22}  {'-'*10}")
    print(f"  {'Acc':<22}  {m['Acc']:>10.4f}")
    print(f"  {'F1':<22}  {m['F1']:>10.4f}")
    print(f"  {'Pre (scam)':<22}  {m['Pre']:>10.4f}")
    print(f"  {'Recall (scam)':<22}  {m['Recall']:>10.4f}")
    if not np.isnan(m["AUROC"]):
        print(f"  {'AUROC':<22}  {m['AUROC']:>10.4f}")
    print(f"  {'Specificity (harm)':<22}  {m['Specificity']:>10.4f}")
    print()
    print(f"  {'Detection rate':<22}  {m['Detection_rate']:>10.4f}  "
          f"({m['Detected']}/{m['N_scam']} scam)")
    if not np.isnan(m["Avg_delay"]):
        print(f"  {'Avg detection delay':<22}  {m['Avg_delay']:>10.2f} turns")
    print(f"  {'False alarm rate':<22}  {m['False_alarm']:>10.4f}  "
          f"({m['False_alarms']}/{m['N_harmless']} harmless)")
    print()
    print("  Confusion matrix:")
    print(f"                    pred=harm   pred=scam")
    print(f"    gt=harmless     TN={m['TN']:<8d}  FP={m['FP']:<8d}")
    print(f"    gt=scam         FN={m['FN']:<8d}  TP={m['TP']:<8d}")
    print(sep)


def print_mc_results(m):
    sep = "=" * 64
    print(f"\n{sep}")
    print("STAGE 2 — MULTICLASS MODEL RESULTS  (GT-scam subset)")
    print(sep)
    print(f"  {'Metric':<22}  {'Value':>10}")
    print(f"  {'-'*22}  {'-'*10}")
    print(f"  {'Acc':<22}  {m['Acc']:>10.4f}")
    print(f"  {'Macro F1':<22}  {m['Macro_F1']:>10.4f}")
    print(f"  {'Weighted F1':<22}  {m['Weighted_F1']:>10.4f}")
    print(f"  {'Macro Pre':<22}  {m['Macro_Pre']:>10.4f}")
    print(f"  {'Macro Recall':<22}  {m['Macro_Rec']:>10.4f}")
    print()
    print(f"  {'Scenario':<12} {'Acc':>8} {'Pre':>8} {'Recall':>8} {'F1':>8} {'N':>6}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for name in sorted(m["per_class"]):
        v = m["per_class"][name]
        print(f"  {name:<12} {v['Acc']:>8.4f} {v['Pre']:>8.4f} {v['Rec']:>8.4f} "
              f"{v['F1']:>8.4f} {v['N']:>6d}")
    labels = m["label_order"]
    cm     = m["confusion"]
    print()
    print("  Confusion matrix (rows=gt, cols=pred):")
    print("  " + " " * 14 + "".join(f"{l:>8s}" for l in labels))
    for i, row in enumerate(cm):
        print(f"    gt={labels[i]:<8s}" + "".join(f"{v:>8d}" for v in row))
    print(sep)


def print_summary_table(bin_m, mc_m=None):
    """Single-row summary table for quick comparison across experiments."""
    sep = "=" * 64
    print(f"\n{sep}")
    print("SUMMARY TABLE")
    print(sep)
    cols = ["Acc", "F1", "Pre", "Recall", "Detection rate", "False Alarm"]
    vals = [
        f"{bin_m['Acc']:.4f}",
        f"{bin_m['F1']:.4f}",
        f"{bin_m['Pre']:.4f}",
        f"{bin_m['Recall']:.4f}",
        f"{bin_m['Detection_rate']:.4f}",
        f"{bin_m['False_alarm']:.4f}",
    ]
    print("  " + "\t".join(cols))
    print("  " + "\t".join(vals))
    if mc_m:
        print()
        print(f"  MC Acc={mc_m['Acc']:.4f}  "
              f"Macro-F1={mc_m['Macro_F1']:.4f}  "
              f"Macro-Pre={mc_m['Macro_Pre']:.4f}  "
              f"Macro-Rec={mc_m['Macro_Rec']:.4f}")
    print(sep)


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a two-stage scam detection checkpoint on a data file."
    )
    parser.add_argument("--ckpt-dir", required=True,
        help="Root output dir (contains binary_model/ and mc_model/).")
    parser.add_argument("--data", required=True,
        help="Path to JSON dialogue file to evaluate.")
    parser.add_argument("--threshold", type=float, default=None,
        help="Binary scam threshold (default: value from config, fallback 0.5).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None,
        help="'cuda' or 'cpu'. Default: auto-detect.")
    parser.add_argument("--save-json", default=None,
        help="Optional path to save all metrics as JSON.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Config & scenario map ───────────────────────────────────────────
    cfg          = _load_config(args.ckpt_dir)
    scenario_map = _load_scenario_map(args.ckpt_dir)
    threshold    = args.threshold if args.threshold is not None else cfg.binary_threshold

    print(f"\nCheckpoint : {args.ckpt_dir}")
    print(f"Data file  : {args.data}")
    print(f"Threshold  : {threshold}")
    print(f"Scenario map: {scenario_map}")

    # ── Tokenizer ───────────────────────────────────────────────────────
    tok_path = os.path.join(args.ckpt_dir, "binary_model")
    if os.path.exists(os.path.join(tok_path, "tokenizer.json")):
        print(f"\nLoading tokenizer from {tok_path}")
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
    else:
        print(f"\nLoading tokenizer from HuggingFace: {cfg.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── Data ────────────────────────────────────────────────────────────
    dialogues = load_json(args.data)
    print(f"Loaded {len(dialogues)} dialogues from {args.data}")

    n_scam = sum(1 for d in dialogues if d.get("label") == "scam")
    n_harm = sum(1 for d in dialogues if d.get("label") == "harmless")
    print(f"  scam={n_scam}  harmless={n_harm}")

    # ── Stage 1: Binary ─────────────────────────────────────────────────
    print("\nLoading binary model …")
    bin_model = _load_binary_model(args.ckpt_dir, cfg, device)

    print("Running binary inference …")
    all_labels, all_d_probs, all_t_probs = run_binary_inference(
        bin_model, dialogues, tokenizer, cfg, device, args.batch_size
    )
    bin_metrics = compute_binary_metrics(all_labels, all_d_probs, all_t_probs, threshold)
    print_binary_results(bin_metrics)

    # ── Stage 2: Multiclass ─────────────────────────────────────────────
    mc_metrics = None
    if scenario_map:
        mc_pt = os.path.join(args.ckpt_dir, "mc_model", "model.pt")
        if os.path.exists(mc_pt):
            print("\nLoading multiclass model …")
            mc_model = _load_mc_model(args.ckpt_dir, cfg, scenario_map, device)

            # Evaluate on GT-scam dialogues (standard MC evaluation)
            scam_dialogues = [d for d in dialogues if d.get("label") == "scam"]
            print(f"Running MC inference on {len(scam_dialogues)} GT-scam dialogues …")
            mc_gt, mc_pred = run_mc_inference(
                mc_model, scam_dialogues, tokenizer, cfg, scenario_map, device, args.batch_size
            )
            if mc_gt:
                mc_metrics = compute_mc_metrics(mc_gt, mc_pred, scenario_map)
                print_mc_results(mc_metrics)
            else:
                print("  No GT-scam dialogues with known scenario found; skipping MC eval.")
        else:
            print(f"\n[WARN] mc_model/model.pt not found at {mc_pt}; skipping Stage 2.")
    else:
        print("\n[INFO] No scenario_map found; skipping Stage 2.")

    # ── Summary ─────────────────────────────────────────────────────────
    print_summary_table(bin_metrics, mc_metrics)

    # ── Save JSON ───────────────────────────────────────────────────────
    if args.save_json:
        def _clean(d):
            if isinstance(d, dict):
                return {k: _clean(v) for k, v in d.items()}
            if isinstance(d, (np.floating, float)):
                v = float(d)
                return None if np.isnan(v) else v
            if isinstance(d, (np.integer, int)):
                return int(d)
            if isinstance(d, list):
                return [_clean(x) for x in d]
            return d

        out = {"binary": _clean(bin_metrics)}
        if mc_metrics:
            out["multiclass"] = _clean(mc_metrics)
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics saved to: {args.save_json}")


if __name__ == "__main__":
    main()
