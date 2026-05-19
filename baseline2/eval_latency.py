"""
eval_latency.py — Binary model evaluation with consecutive-turn latency.

Thay vì alert ngay khi 1 turn vượt threshold, yêu cầu N turn LIÊN TIẾP
đều >= threshold thì mới coi là scam. In bảng metrics cho mỗi N.

Usage:
    python eval_latency.py \
        --ckpt-dir outputs/exp1_viscam_two_stage \
        --data /path/to/real2_new.json \
        [--threshold 0.5] [--batch-size 8] [--n-list 1 2 3 4 5]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from config  import Config
from dataset import BinaryDialogueDataset, collate_fn, load_json
from model   import BinaryM1Classifier


# ── Load helpers ───────────────────────────────────────────────────────

def _load_config(ckpt_dir: str) -> Config:
    cfg_path = os.path.join(ckpt_dir, "binary_model", "config.json")
    cfg = Config()
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            d = json.load(f)
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def _load_binary_model(ckpt_dir: str, cfg: Config, device):
    pt = os.path.join(ckpt_dir, "binary_model", "model.pt")
    model = BinaryM1Classifier(cfg).to(device)
    model.load_state_dict(torch.load(pt, map_location=device, weights_only=True))
    model.eval()
    return model


# ── Inference ──────────────────────────────────────────────────────────

@torch.no_grad()
def run_binary_inference(model, dialogues, tokenizer, cfg, device, batch_size):
    ds     = BinaryDialogueDataset(dialogues, tokenizer, cfg.max_turn_len, cfg.max_turns)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2,
                        pin_memory=(device.type == "cuda"))

    all_labels, all_d_probs, all_t_probs = [], [], []
    for batch in loader:
        ids     = batch["input_ids"].to(device)
        masks   = batch["attn_masks"].to(device)
        tmask   = batch["turn_mask"].to(device)
        labels  = batch["labels"]
        n_turns = batch["n_turns"]

        out = model(ids, masks, tmask)
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


# ── Latency alert ──────────────────────────────────────────────────────

def first_alert_consecutive(turn_probs: np.ndarray, threshold: float,
                             n_consec: int) -> int | None:
    """
    Trả về index turn đầu tiên của chuỗi n_consec turn liên tiếp
    đều >= threshold. Trả None nếu không có.
    """
    count = 0
    for i, p in enumerate(turn_probs):
        if float(p) >= threshold:
            count += 1
            if count >= n_consec:
                return i - n_consec + 1  # turn đầu chuỗi (hoặc dùng i nếu muốn turn kết thúc)
        else:
            count = 0
    return None


# ── Metrics với 1 giá trị N ───────────────────────────────────────────

def compute_metrics_for_n(all_labels, all_d_probs, all_t_probs,
                           threshold: float, n_consec: int) -> dict:
    labels = np.array(all_labels)
    probs  = np.array(all_d_probs)

    try:
        auroc = float(roc_auc_score(labels, probs))
    except ValueError:
        auroc = float("nan")

    num_scam, num_detected     = 0, 0
    num_harm, num_false_alarms = 0, 0
    detection_delays           = []
    stream_preds               = []   # prediction dựa trên streaming (có alert = scam)

    for label, t_probs in zip(all_labels, all_t_probs):
        alert = first_alert_consecutive(t_probs, threshold, n_consec)
        stream_preds.append(1 if alert is not None else 0)
        if label == 1:
            num_scam += 1
            if alert is not None:
                num_detected += 1
                detection_delays.append(alert)
        else:
            num_harm += 1
            if alert is not None:
                num_false_alarms += 1

    stream_preds = np.array(stream_preds)

    acc  = float(accuracy_score(labels, stream_preds))
    f1   = float(f1_score(labels, stream_preds, zero_division=0))
    dr   = num_detected    / max(num_scam, 1)
    far  = num_false_alarms / max(num_harm, 1)
    avg_delay = float(np.mean(detection_delays)) if detection_delays else float("nan")

    return {
        "acc":            acc,
        "f1":             f1,
        "auroc":          auroc,
        "detection_rate": dr,
        "false_alarm":    far,
        "avg_delay":      avg_delay,
        "detected":       num_detected,
        "false_alarms":   num_false_alarms,
        "n_scam":         num_scam,
        "n_harm":         num_harm,
    }


# ── Print table ────────────────────────────────────────────────────────

def print_latency_table(results: dict, threshold: float):
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"LATENCY EVAL  (threshold={threshold:.2f})")
    print(f"  N = số turn liên tiếp >= threshold cần thiết để alert scam")
    print(sep)

    header = f"{'N':>4}  {'Acc':>7}  {'F1':>7}  {'AUROC':>7}  " \
             f"{'DetRate':>8}  {'FalseAlm':>9}  {'AvgDelay':>9}  " \
             f"{'Detected':>9}  {'FalseAlms':>10}"
    print(header)
    print("-" * 80)

    for n, m in sorted(results.items()):
        auroc_str = f"{m['auroc']:.4f}" if not np.isnan(m["auroc"]) else "   NaN"
        delay_str = f"{m['avg_delay']:>9.2f}" if not np.isnan(m["avg_delay"]) else "      NaN"
        print(
            f"{n:>4}  "
            f"{m['acc']:>7.4f}  "
            f"{m['f1']:>7.4f}  "
            f"{auroc_str:>7}  "
            f"{m['detection_rate']:>8.4f}  "
            f"{m['false_alarm']:>9.4f}  "
            f"{delay_str}  "
            f"{m['detected']:>4}/{m['n_scam']:<4}  "
            f"{m['false_alarms']:>4}/{m['n_harm']:<5}"
        )

    print(sep)
    n1 = results[min(results)]
    print(f"\n  Note: N=1 = behaviour gốc (alert ngay turn đầu tiên >= threshold)")
    print(f"  Tổng: {n1['n_scam']} scam, {n1['n_harm']} harmless trong test set")
    print(sep)


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate binary model with consecutive-turn latency for varying N."
    )
    parser.add_argument("--ckpt-dir",   required=True)
    parser.add_argument("--data",       required=True)
    parser.add_argument("--threshold",  type=float, default=None)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--n-list",     type=int, nargs="+",
                        default=[1, 2, 3, 4, 5],
                        help="Danh sách N cần thử (default: 1 2 3 4 5)")
    parser.add_argument("--save-csv",   default=None,
                        help="Path lưu kết quả CSV")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg       = _load_config(args.ckpt_dir)
    threshold = args.threshold if args.threshold is not None else getattr(cfg, "binary_threshold", 0.5)

    print(f"\nCheckpoint : {args.ckpt_dir}")
    print(f"Data       : {args.data}")
    print(f"Threshold  : {threshold}")
    print(f"N values   : {args.n_list}")

    # ── Tokenizer ──
    tok_path = os.path.join(args.ckpt_dir, "binary_model")
    if os.path.exists(os.path.join(tok_path, "tokenizer.json")):
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    print(f"Tokenizer loaded.")

    # ── Data ──
    dialogues = load_json(args.data)
    n_scam = sum(1 for d in dialogues if d.get("label") == "scam")
    n_harm = sum(1 for d in dialogues if d.get("label") == "harmless")
    print(f"Loaded {len(dialogues)} dialogues  (scam={n_scam}, harmless={n_harm})")

    # ── Model + inference (1 lần duy nhất) ──
    print("\nLoading binary model …")
    model = _load_binary_model(args.ckpt_dir, cfg, device)

    print("Running inference …")
    all_labels, all_d_probs, all_t_probs = run_binary_inference(
        model, dialogues, tokenizer, cfg, device, args.batch_size
    )

    # ── Compute metrics cho mỗi N ──
    results = {}
    for n in args.n_list:
        results[n] = compute_metrics_for_n(
            all_labels, all_d_probs, all_t_probs, threshold, n
        )

    print_latency_table(results, threshold)

    if args.save_csv:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)), exist_ok=True)
        fieldnames = ["n_consec", "acc", "f1", "auroc", "detection_rate",
                      "false_alarm", "avg_delay", "detected", "n_scam",
                      "false_alarms", "n_harm"]
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for n, m in sorted(results.items()):
                writer.writerow({
                    "n_consec":       n,
                    "acc":            round(m["acc"], 6),
                    "f1":             round(m["f1"], 6),
                    "auroc":          round(m["auroc"], 6) if not np.isnan(m["auroc"]) else "",
                    "detection_rate": round(m["detection_rate"], 6),
                    "false_alarm":    round(m["false_alarm"], 6),
                    "avg_delay":      round(m["avg_delay"], 4) if not np.isnan(m["avg_delay"]) else "",
                    "detected":       m["detected"],
                    "n_scam":         m["n_scam"],
                    "false_alarms":   m["false_alarms"],
                    "n_harm":         m["n_harm"],
                })
        print(f"\nCSV saved → {args.save_csv}")


if __name__ == "__main__":
    main()
