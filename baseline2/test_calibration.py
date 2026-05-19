"""
test_calibration.py — Đánh giá calibration + binary (Scam vs Harmless) cho
Stage-1 binary detector của exp_an_old, trình bày theo style binary_test.py.

Cách dùng:
    python test_calibration.py --exp-dir outputs/exp1_viscam_two_stage
    python test_calibration.py                       # exp1 viscam (mặc định)
    python test_calibration.py --threshold 0.85 --no-plot

Lưu ý:
    - Tự phát hiện checkpoint train bằng LoRA (PEFT) và dựng lại encoder cho khớp.
    - Model là binary sigmoid → dialogue_probs chính là P(Scam).
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from config import Config
from dataset import BinaryDialogueDataset, collate_fn, load_json
from model import BinaryM1Classifier
from transformers import AutoTokenizer


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Calibration + binary eval cho binary scam detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--exp-dir",
        default=os.path.join(_DIR, "/home/nvidia-lab/ai4life/CSLR/thaidc/VCS/exp_an_oldva/outputs/exp1_vanilla_two_stage"),
        help="Thư mục experiment chứa binary_model/",
    )
    #python test_calibration.py  
    p.add_argument(
        "--test-file",
        default=os.path.join(
            _DIR, "/home/nvidia-lab/ai4life/CSLR/thaidc/VCS/dataset_scamstream-main/real2_new.json",
        ),
        help="File test JSON (list dialogues)",
    )
    p.add_argument("--n-bins", type=int, default=10,
                   help="Số bin cho calibration curve / ECE")
    p.add_argument("--strategy", choices=["uniform", "quantile"],
                   default="uniform", help="Cách chia bin")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threshold", type=float, default=None,
                   help="Ngưỡng phân loại (mặc định: lấy từ Config.binary_threshold)")
    p.add_argument("--no-plot", action="store_true", help="Bỏ qua vẽ đồ thị")
    p.add_argument("--out", default=None,
                   help="Đường dẫn PNG (mặc định: <exp-dir>/calibration_eval.png)")
    p.add_argument("--label", default=None,
                   help="Tên đường cong trong legend (mặc định: tên exp-dir)")
    return p.parse_args()


def sep(char="─", n=62):
    print(char * n)


def section(title, char="═", width=78):
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_kv(rows, indent=2, key_width=None):
    if not rows:
        return
    if key_width is None:
        key_width = max(len(str(k)) for k, _ in rows)
    for key, value in rows:
        print(f"{' ' * indent}{str(key):<{key_width}} : {value}")


def print_metric_table(title, headers, rows, indent=2):
    if title:
        print(f"{' ' * indent}{title}")
    widths = [len(h) for h in headers]
    str_rows = [[str(cell) for cell in row] for row in rows]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    prefix = " " * indent
    header_line = "  ".join(f"{headers[i]:<{widths[i]}}" for i in range(len(headers)))
    divider = "  ".join("─" * widths[i] for i in range(len(headers)))
    print(prefix + header_line)
    print(prefix + divider)
    for row in str_rows:
        print(prefix + "  ".join(f"{row[i]:<{widths[i]}}" for i in range(len(row))))


def print_confusion_matrix(cm, labels=("Harmless", "Scam"), indent=2):
    row_names = [f"True {label}" for label in labels]
    headers = [""] + [f"Pred {label}" for label in labels]
    rows = [[row_names[i]] + [f"{int(v)}" for v in cm[i]] for i in range(len(labels))]
    print_metric_table("Confusion matrix", headers, rows, indent=indent)


def print_classification_summary(y_true, y_pred, indent=2):
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Harmless", "Scam"],
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for cls in ("Harmless", "Scam"):
        stats = report[cls]
        rows.append([
            cls,
            f"{stats['precision']:.4f}",
            f"{stats['recall']:.4f}",
            f"{stats['f1-score']:.4f}",
            int(stats["support"]),
        ])
    for avg_name in ("macro avg", "weighted avg"):
        stats = report[avg_name]
        rows.append([
            avg_name,
            f"{stats['precision']:.4f}",
            f"{stats['recall']:.4f}",
            f"{stats['f1-score']:.4f}",
            int(stats["support"]),
        ])
    print_metric_table(
        "Per-class metrics",
        ["Class", "Precision", "Recall", "F1", "Support"],
        rows,
        indent=indent,
    )
    print(f"{' ' * indent}Accuracy: {report['accuracy']:.4f}")


# ── Model loading (LoRA-aware) ─────────────────────────────────────────

def _maybe_wrap_lora(model, cfg_json, ckpt_keys):
    """Nếu checkpoint train bằng PEFT-LoRA thì dựng lại encoder cho khớp."""
    if not any("lora_" in k for k in ckpt_keys):
        return model  # encoder thường — không cần làm gì

    from peft import LoraConfig, get_peft_model

    targets = cfg_json.get(
        "lora_target_modules", ["query", "key", "value", "dense"],
    )
    lora_cfg = LoraConfig(
        r=int(cfg_json.get("lora_r", 32)),
        lora_alpha=int(cfg_json.get("lora_alpha", 64)),
        lora_dropout=float(cfg_json.get("lora_dropout", 0.05)),
        target_modules=targets,
        bias="none",
    )
    model.encoder = get_peft_model(model.encoder, lora_cfg)
    print(f"  Dựng lại LoRA encoder (r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, "
          f"targets={targets})")
    return model


def load_binary_model(exp_dir, device):
    bin_dir = os.path.join(exp_dir, "binary_model")
    ckpt_path = os.path.join(bin_dir, "model.pt")
    cfg_path = os.path.join(bin_dir, "config.json")
    if not os.path.isfile(ckpt_path):
        sys.exit(f"ERROR: không tìm thấy checkpoint: {ckpt_path}")

    cfg_json = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg_json = json.load(f)

    cfg = Config()
    for fld in ("model_name", "max_turn_len", "max_turns",
                "hidden_dim", "attn_heads", "dropout"):
        if fld in cfg_json:
            setattr(cfg, fld, cfg_json[fld])

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    model = BinaryM1Classifier(cfg)
    model = _maybe_wrap_lora(model, cfg_json, state.keys())

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("encoder.")]
    if missing or unexpected:
        print(f"  load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected (encoder buffers bỏ qua)")
    model._frozen = False
    model.to(device).eval()

    tok_src = bin_dir if os.path.isfile(
        os.path.join(bin_dir, "tokenizer_config.json")) else cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    return model, tokenizer, cfg


# ── Inference ──────────────────────────────────────────────────────────

@torch.no_grad()
def collect_probs(model, loader, device):
    """Trả (y_true, y_prob, turn_probs_list):
    - y_true: nhãn scam
    - y_prob: P(Scam) ở turn cuối từng dialogue
    - turn_probs_list: list[np.array] — P(Scam) từng turn (chỉ các turn hợp lệ)
    """
    y_true, y_prob = [], []
    turn_probs_list = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        masks = batch["attn_masks"].to(device)
        tmask = batch["turn_mask"].to(device)
        labels = batch["labels"]

        out = model(ids, masks, tmask)
        probs = out["dialogue_probs"].float().cpu().numpy()
        tp_batch = out["turn_probs"].float().cpu().numpy()   # [B, T]
        tm_np = tmask.cpu().numpy()                          # [B, T]

        for b in range(len(labels)):
            n_valid = int(tm_np[b].sum())
            turn_probs_list.append(tp_batch[b, :n_valid])

        y_prob.extend(probs.tolist())
        y_true.extend(labels.tolist())

        del out
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float), turn_probs_list


def early_stop_predict(turn_probs_list, threshold):
    """Early-stop: dừng tại turn đầu tiên >= threshold → scam.
    Ngược lại dùng prob turn cuối → harmless nếu < threshold.

    Trả về:
    - early_probs: np.array — prob tại điểm dừng
    - stop_turns: list[int] — chỉ số turn dừng (1-indexed); = n_turns nếu không kích hoạt
    """
    early_probs, stop_turns = [], []
    for tps in turn_probs_list:
        triggered = np.where(tps >= threshold)[0]
        if len(triggered) > 0:
            t = int(triggered[0])
            early_probs.append(float(tps[t]))
            stop_turns.append(t + 1)
        else:
            early_probs.append(float(tps[-1]))
            stop_turns.append(len(tps))
    return np.asarray(early_probs, dtype=float), stop_turns


# ── Calibration metrics ────────────────────────────────────────────────

def calibration_metrics(y_true, y_prob, n_bins):
    """ECE/MCE (uniform bins) + Brier + AUROC."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.digitize(y_prob, bins[1:-1])

    n = len(y_true)
    ece, mce = 0.0, 0.0
    rows = []
    for b in range(n_bins):
        m = bin_id == b
        cnt = int(m.sum())
        if cnt == 0:
            rows.append((b, cnt, np.nan, np.nan, 0.0))
            continue
        conf = float(y_prob[m].mean())
        acc = float(y_true[m].mean())
        gap = abs(acc - conf)
        ece += gap * cnt / n
        mce = max(mce, gap)
        rows.append((b, cnt, conf, acc, gap))

    brier = float(brier_score_loss(y_true, y_prob))
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")
    return {"ece": ece, "mce": mce, "brier": brier, "auroc": auroc}, rows


# ── Plots (style giống binary_test.py) ─────────────────────────────────

def make_plots(y_true, prob_scam, threshold, cm, roc_auc, avg_prec,
               args, metrics, label, out_path):
    NAMES = ["Harmless", "Scam"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        f"Calibration + Binary Eval — {label}",
        fontsize=16, fontweight="bold",
    )

    # 1. Reliability diagram (calibration curve) — panel chính
    ax = axes[0, 0]
    prob_true, prob_pred = calibration_curve(
        y_true, prob_scam, n_bins=args.n_bins, strategy=args.strategy,
    )
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "s-", color="royalblue", label=label)
    ax.set_title(
        f"Reliability Diagram  (ECE={metrics['ece']:.4f}, "
        f"Brier={metrics['brier']:.4f})"
    )
    ax.set_xlabel("Mean predicted probability (Positive class: 1)")
    ax.set_ylabel("Fraction of positives (Positive class: 1)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # 2. Histogram độ tin cậy P(Scam)
    ax = axes[0, 1]
    ax.hist(prob_scam, range=(0, 1), bins=args.n_bins,
            color="royalblue", alpha=0.8)
    ax.axvline(threshold, color="black", linestyle="--",
               label=f"Threshold={threshold:.2f}")
    ax.set_title("Phân phối P(Scam) (toàn bộ)")
    ax.set_xlabel("P(Scam)"); ax.set_ylabel("Count")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(); ax.grid(alpha=0.3)

    # 3. Confusion matrix (matplotlib thuần — không cần seaborn)
    ax = axes[0, 2]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(NAMES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(NAMES)
    thr = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    fontsize=15,
                    color="white" if cm[i, j] > thr else "black")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 4. ROC curve
    ax = axes[1, 0]
    fpr, tpr, _ = roc_curve(y_true, prob_scam)
    ax.plot(fpr, tpr, lw=2, color="royalblue", label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(); ax.grid(alpha=0.3)

    # 5. Precision-Recall curve
    ax = axes[1, 1]
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, prob_scam)
    ax.plot(rec_arr, prec_arr, lw=2, color="darkorange",
            label=f"AP = {avg_prec:.4f}")
    ax.axhline(y_true.mean(), color="gray", linestyle="--")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(); ax.grid(alpha=0.3)

    # 6. Phân phối P(Scam) theo nhãn thật + threshold
    ax = axes[1, 2]
    ax.hist(prob_scam[y_true == 0], bins=40, alpha=0.6,
            color="steelblue", label="Harmless (True)")
    ax.hist(prob_scam[y_true == 1], bins=40, alpha=0.6,
            color="tomato", label="Scam (True)")
    ax.axvline(threshold, color="black", linestyle="--",
               label=f"Threshold={threshold:.2f}")
    ax.set_title("P(Scam) theo nhãn thật")
    ax.set_xlabel("P(Scam)"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Đã lưu biểu đồ: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    section("SETUP")
    print_kv([("Device", device)])

    label = args.label or os.path.basename(args.exp_dir.rstrip("/"))
    out_path = args.out or os.path.join(args.exp_dir, "calibration_eval.png")
    print_kv([("Experiment", label), ("Output plot", out_path)])

    print()
    print(f"Loading binary model: {args.exp_dir}/binary_model")
    model, tokenizer, cfg = load_binary_model(args.exp_dir, device)
    print(f"✅ Model loaded  ({sum(p.numel() for p in model.parameters()):,} params)")

    test_dlg = load_json(args.test_file)
    n_scam = sum(1 for d in test_dlg if d["label"] == "scam")
    print()
    print_kv([
        ("Test file", args.test_file),
        ("Samples", len(test_dlg)),
        ("Harmless", len(test_dlg) - n_scam),
        ("Scam", n_scam),
    ])

    ds = BinaryDialogueDataset(test_dlg, tokenizer, cfg.max_turn_len, cfg.max_turns)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2, pin_memory=True)

    print("\nRunning inference...")
    y_true, prob_scam, turn_probs_list = collect_probs(model, loader, device)

    # ── Binary metrics (style binary_test.py) ───────────────────────────
    cfg_threshold = float(getattr(cfg, "binary_threshold", 0.5))
    preds_binary = (prob_scam >= cfg_threshold).astype(int)
    acc = accuracy_score(y_true, preds_binary)
    roc_auc = roc_auc_score(y_true, prob_scam)
    avg_prec = average_precision_score(y_true, prob_scam)

    section("BINARY EVAL")
    print_kv([
        (f"Accuracy @{cfg_threshold:.2f}", f"{acc:.4f}  ({acc*100:.2f}%)"),
        ("ROC-AUC", f"{roc_auc:.4f}"),
        ("Average Precision", f"{avg_prec:.4f}"),
    ])
    print()
    print_classification_summary(y_true, preds_binary)

    cm = confusion_matrix(y_true, preds_binary)
    tn, fp, fn, tp = cm.ravel()
    print()
    print_confusion_matrix(cm)
    print()
    print_kv([
        ("Sensitivity (Recall Scam)", f"{tp/max(tp+fn,1):.4f}"),
        ("Specificity (Recall Harmless)", f"{tn/max(tn+fp,1):.4f}"),
    ])

    # ── Threshold search ────────────────────────────────────────────────
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        p = (prob_scam >= t).astype(int)
        f = f1_score(y_true, p, pos_label=1, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    threshold = args.threshold if args.threshold is not None else cfg_threshold
    preds_t = (prob_scam >= threshold).astype(int)

    section("THRESHOLD SEARCH")
    print_kv([
        ("Best threshold (reference)", f"{best_t:.2f}  (F1 Scam = {best_f1:.4f})"),
        (
            "Threshold in use",
            (
                f"{threshold:.2f}  (from Config.binary_threshold)"
                if args.threshold is None
                else f"{threshold:.2f}  (user override)"
            ),
        ),
    ])
    print()
    print(f"Report with threshold={threshold:.2f}")
    print_classification_summary(y_true, preds_t)

    # ── Early-stop evaluation ────────────────────────────────────────────
    from collections import Counter

    def run_early_stop(thr, title):
        early_probs, stop_turns = early_stop_predict(turn_probs_list, thr)
        early_preds = (early_probs >= thr).astype(int)

        triggered_idx = [i for i in range(len(stop_turns))
                         if stop_turns[i] < len(turn_probs_list[i])]
        n_early = len(triggered_idx)

        # Detection rate (Scam): tỉ lệ scam phát hiện được trên tổng scam
        n_scam_true = int((y_true == 1).sum())
        n_scam_detected = int(((early_preds == 1) & (y_true == 1)).sum())
        detection_rate = n_scam_detected / max(n_scam_true, 1)

        # False alarm rate: tỉ lệ harmless bị cảnh báo nhầm thành scam
        n_harmless_true = int((y_true == 0).sum())
        n_false_alarm = int(((early_preds == 1) & (y_true == 0)).sum())
        false_alarm_rate = n_false_alarm / max(n_harmless_true, 1)

        # Turn dừng trung bình — tính trên toàn bộ scam-prediction (kể cả ở turn cuối)
        scam_pred_idx = [i for i in range(len(stop_turns)) if early_preds[i] == 1]
        avg_stop_all_scam = (
            float(np.mean([stop_turns[i] for i in scam_pred_idx]))
            if scam_pred_idx else float("nan")
        )

        # Vị trí turn cảnh báo tương đối (%) = stop_turn / tổng số turn của dialogue
        def _rel_pos(idx_list):
            if not idx_list:
                return float("nan")
            return float(np.mean([
                stop_turns[i] / max(len(turn_probs_list[i]), 1) * 100.0
                for i in idx_list
            ]))
        avg_rel_pos_triggered = _rel_pos(triggered_idx)
        avg_rel_pos_scam_preds = _rel_pos(scam_pred_idx)

        section(f"EARLY-STOP | {title}")
        print_kv([
            ("Threshold", f"{thr:.2f}"),
            ("Triggered early", f"{n_early} / {len(y_true)}"),
        ])
        if n_early > 0:
            avg_stop = float(np.mean([stop_turns[i] for i in triggered_idx]))
            cnt = Counter(stop_turns[i] for i in triggered_idx)
            print()
            print_kv([
                ("Avg stop turn (triggered only)", f"{avg_stop:.2f}"),
                ("Avg alert position (triggered only)", f"{avg_rel_pos_triggered:.2f}% dialogue length"),
                ("Stop-turn distribution", ", ".join(f"turn{t}={c}" for t, c in sorted(cnt.items()))),
            ])
        if scam_pred_idx:
            print()
            print_kv([
                ("Avg stop turn (all scam predictions)", f"{avg_stop_all_scam:.2f}"),
                ("Avg alert position (all scam predictions)", f"{avg_rel_pos_scam_preds:.2f}% dialogue length"),
            ])
        print()
        print_kv([
            ("Detection rate", f"{detection_rate:.4f}  ({n_scam_detected}/{n_scam_true})"),
            ("False alarm rate", f"{false_alarm_rate:.4f}  ({n_false_alarm}/{n_harmless_true})"),
        ])
        print()
        print_classification_summary(y_true, early_preds)
        cm_early = confusion_matrix(y_true, early_preds)
        tn_e, fp_e, fn_e, tp_e = cm_early.ravel()
        print()
        print_confusion_matrix(cm_early)
        print()
        print_kv([
            ("Sensitivity (Recall Scam)", f"{tp_e/max(tp_e+fn_e,1):.4f}"),
            ("Specificity (Recall Harmless)", f"{tn_e/max(tn_e+fp_e,1):.4f}"),
        ])

        return {
            "threshold": float(thr),
            "n_triggered": int(n_early),
            "avg_stop_turn_triggered": (
                float(np.mean([stop_turns[i] for i in triggered_idx]))
                if triggered_idx else float("nan")
            ),
            "avg_stop_turn_scam_preds": avg_stop_all_scam,
            "avg_rel_pos_triggered_pct": avg_rel_pos_triggered,
            "avg_rel_pos_scam_preds_pct": avg_rel_pos_scam_preds,
            "detection_rate": float(detection_rate),
            "false_alarm_rate": float(false_alarm_rate),
        }

    es_used = run_early_stop(threshold, f"Threshold sử dụng = {threshold:.2f}")
    es_best = run_early_stop(best_t, f"Threshold tối ưu (F1) = {best_t:.2f}")

    # ── Calibration ─────────────────────────────────────────────────────
    metrics, rows = calibration_metrics(y_true, prob_scam, args.n_bins)
    section("CALIBRATION")
    print_kv([
        (f"ECE (n_bins={args.n_bins})", f"{metrics['ece']:.4f}"),
        ("MCE", f"{metrics['mce']:.4f}"),
        ("Brier score", f"{metrics['brier']:.4f}"),
        ("AUROC", f"{metrics['auroc']:.4f}"),
    ])
    edges = np.linspace(0, 1, args.n_bins + 1)
    bin_rows = []
    for b, cnt, conf, acc_b, gap in rows:
        rng = f"[{edges[b]:.2f},{edges[b+1]:.2f})"
        if cnt == 0:
            bin_rows.append([b, rng, cnt, "-", "-", "-"])
        else:
            bin_rows.append([b, rng, cnt, f"{conf:.4f}", f"{acc_b:.4f}", f"{gap:.4f}"])
    print()
    print_metric_table(
        "Per-bin breakdown",
        ["Bin", "Range", "N", "Conf", "Acc", "|Gap|"],
        bin_rows,
    )

    # ── Error analysis (style binary_test.py) ───────────────────────────
    fp_idx = np.where((preds_binary == 1) & (y_true == 0))[0]
    fn_idx = np.where((preds_binary == 0) & (y_true == 1))[0]
    section("ERROR ANALYSIS")
    print_kv([
        (f"Total errors @{cfg_threshold:.2f}", len(fp_idx) + len(fn_idx)),
        ("False Positive (Harmless -> Scam)", len(fp_idx)),
        ("False Negative (Scam -> Harmless)", len(fn_idx)),
    ])

    def show_errors(indices, title, n=3):
        if len(indices) == 0:
            return
        print()
        print(f"  [{title}] {min(n, len(indices))} sample(s)")
        chosen = np.random.choice(indices, size=min(n, len(indices)), replace=False)
        for i in chosen:
            turns = test_dlg[i]["turns"]
            print(f"    Sample {i} | true={test_dlg[i]['label']} | p_scam={prob_scam[i]:.3f}")
            for j, t in enumerate(turns[:3]):
                snippet = t[:90] + ("..." if len(t) > 90 else "")
                print(f"      T{j+1}: {snippet}")

    show_errors(fp_idx, "FALSE POSITIVE")
    show_errors(fn_idx, "FALSE NEGATIVE")
    print()
    sep("═")

    # ── Plots ───────────────────────────────────────────────────────────
    if not args.no_plot:
        make_plots(y_true, prob_scam, threshold, cm, roc_auc, avg_prec,
                   args, metrics, label, out_path)

    metrics_path = os.path.join(args.exp_dir, "calibration_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({**metrics, "n_bins": args.n_bins,
                   "strategy": args.strategy, "n_samples": len(y_true),
                   "best_threshold": float(best_t), "best_f1": float(best_f1),
                   "early_stop_used": es_used,
                   "early_stop_best": es_best},
                  f, indent=2)
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
