"""
test_calibration_multi.py — So sánh calibration + binary eval cho nhiều checkpoint.

Cách dùng:
    python test_calibration_multi.py \
        --exp-dirs outputs/exp1 outputs/exp2 \
        --labels "Model A" "Model B" \
        --test-file dataset.json

Biểu đồ:
    - Reliability Diagram (gộp)
    - ROC Curve (gộp)
    - Precision-Recall Curve (gộp)
    - Confusion Matrix (tách riêng từng model)
"""

import argparse, json, os, sys, warnings
from collections import Counter

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
        description="Multi-checkpoint calibration + binary eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--exp-dirs", nargs="+", required=True,
                   help="Danh sách thư mục experiment chứa binary_model/")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Tên hiển thị cho từng model (mặc định: tên thư mục)")
    p.add_argument("--test-file",
                   default="/home/nvidia-lab/ai4life/CSLR/thaidc/VCS/dataset_scamstream-main/real2_new.json",
                   help="File test JSON")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--strategy", choices=["uniform", "quantile"], default="uniform")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threshold", type=float, default=None,
                   help="Ngưỡng phân loại (mặc định: từ Config)")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--out", default=None,
                   help="Đường dẫn PNG output (mặc định: calibration_compare.png)")
    return p.parse_args()


def sep(char="─", n=62):
    print(char * n)


# ── Model loading (LoRA-aware) ─────────────────────────────────────────

def _maybe_wrap_lora(model, cfg_json, ckpt_keys):
    """Nếu checkpoint train bằng PEFT-LoRA thì dựng lại encoder cho khớp."""
    if not any("lora_" in k for k in ckpt_keys):
        return model
    from peft import LoraConfig, get_peft_model
    targets = cfg_json.get("lora_target_modules", ["query", "key", "value", "dense"])
    lora_cfg = LoraConfig(
        r=int(cfg_json.get("lora_r", 32)),
        lora_alpha=int(cfg_json.get("lora_alpha", 64)),
        lora_dropout=float(cfg_json.get("lora_dropout", 0.05)),
        target_modules=targets, bias="none",
    )
    model.encoder = get_peft_model(model.encoder, lora_cfg)
    print(f"  LoRA encoder (r={lora_cfg.r}, alpha={lora_cfg.lora_alpha})")
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
              f"{len(unexpected)} unexpected")
    model._frozen = False
    model.to(device).eval()

    tok_src = bin_dir if os.path.isfile(
        os.path.join(bin_dir, "tokenizer_config.json")) else cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    return model, tokenizer, cfg


# ── Inference ──────────────────────────────────────────────────────────

@torch.no_grad()
def collect_probs(model, loader, device):
    y_true, y_prob, turn_probs_list = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        masks = batch["attn_masks"].to(device)
        tmask = batch["turn_mask"].to(device)
        labels = batch["labels"]

        out = model(ids, masks, tmask)
        probs = out["dialogue_probs"].float().cpu().numpy()
        tp_batch = out["turn_probs"].float().cpu().numpy()
        tm_np = tmask.cpu().numpy()

        for b in range(len(labels)):
            n_valid = int(tm_np[b].sum())
            turn_probs_list.append(tp_batch[b, :n_valid])

        y_prob.extend(probs.tolist())
        y_true.extend(labels.tolist())
        del out
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.asarray(y_true, int), np.asarray(y_prob, float), turn_probs_list


def early_stop_predict(turn_probs_list, threshold):
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
    return np.asarray(early_probs, float), stop_turns


# ── Calibration metrics ────────────────────────────────────────────────

def calibration_metrics(y_true, y_prob, n_bins):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.digitize(y_prob, bins[1:-1])
    n = len(y_true)
    ece, mce = 0.0, 0.0
    rows = []
    for b in range(n_bins):
        m = bin_id == b
        cnt = int(m.sum())
        if cnt == 0:
            rows.append((b, cnt, np.nan, np.nan, 0.0)); continue
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


# ── Plots ──────────────────────────────────────────────────────────────

COLORS = ["royalblue", "tomato", "seagreen", "darkorange", "mediumpurple",
          "deeppink", "teal", "goldenrod", "slategray", "crimson"]


def make_plots(results, args, out_path):
    """
    results: list of dict, mỗi dict chứa:
        label, y_true, prob_scam, threshold, cm, roc_auc, avg_prec, metrics
    """
    n_models = len(results)
    NAMES = ["Harmless", "Scam"]

    # Layout: 1 row for 3 shared charts + 1 row for confusion matrices
    n_cm_cols = max(n_models, 3)
    fig = plt.figure(figsize=(7 * n_cm_cols, 12))
    gs = fig.add_gridspec(2, n_cm_cols, hspace=0.35, wspace=0.35)

    fig.suptitle("Calibration + Binary Eval — Multi-Model Comparison",
                 fontsize=16, fontweight="bold", y=0.98)

    # ── Row 0: 3 shared line charts ──
    ax_rel = fig.add_subplot(gs[0, 0])
    ax_roc = fig.add_subplot(gs[0, 1])
    ax_pr  = fig.add_subplot(gs[0, 2])

    for i, r in enumerate(results):
        c = COLORS[i % len(COLORS)]
        lbl = r["label"]
        y_true, prob_scam = r["y_true"], r["prob_scam"]
        metrics = r["metrics"]

        # 1. Reliability Diagram
        prob_true, prob_pred = calibration_curve(
            y_true, prob_scam, n_bins=args.n_bins, strategy=args.strategy)
        ax_rel.plot(prob_pred, prob_true, "s-", color=c,
                    label=f"{lbl} (ECE={metrics['ece']:.4f})")

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, prob_scam)
        ax_roc.plot(fpr, tpr, lw=2, color=c,
                    label=f"{lbl} (AUC={r['roc_auc']:.4f})")

        # 3. PR Curve
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, prob_scam)
        ax_pr.plot(rec_arr, prec_arr, lw=2, color=c,
                   label=f"{lbl} (AP={r['avg_prec']:.4f})")

    # Reliability diagram formatting
    ax_rel.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax_rel.set_title("Reliability Diagram")
    ax_rel.set_xlabel("Mean predicted probability")
    ax_rel.set_ylabel("Fraction of positives")
    ax_rel.set_xlim(-0.02, 1.02); ax_rel.set_ylim(-0.02, 1.02)
    ax_rel.legend(loc="lower right", fontsize=8); ax_rel.grid(alpha=0.3)

    # ROC formatting
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_title("ROC Curve")
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.legend(fontsize=8); ax_roc.grid(alpha=0.3)

    # PR formatting
    baseline = results[0]["y_true"].mean()
    ax_pr.axhline(baseline, color="gray", linestyle="--", label="Baseline")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=8); ax_pr.grid(alpha=0.3)

    # Hide extra cols in row 0 if n_cm_cols > 3
    for j in range(3, n_cm_cols):
        fig.add_subplot(gs[0, j]).set_visible(False)

    # ── Row 1: Confusion matrices (one per model) ──
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[1, i])
        cm = r["cm"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(NAMES)
        ax.set_yticks([0, 1]); ax.set_yticklabels(NAMES)
        thr_val = cm.max() / 2.0
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, f"{cm[ii, jj]:d}", ha="center", va="center",
                        fontsize=15,
                        color="white" if cm[ii, jj] > thr_val else "black")
        ax.set_title(f"CM: {r['label']}", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide extra cols in row 1
    for j in range(n_models, n_cm_cols):
        fig.add_subplot(gs[1, j]).set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Đã lưu biểu đồ: {out_path}")


# ── Per-model evaluation (text) ────────────────────────────────────────

def evaluate_one(label, y_true, prob_scam, turn_probs_list, test_dlg,
                 args, threshold_override):
    cfg_threshold = threshold_override if threshold_override else 0.5
    threshold = args.threshold if args.threshold is not None else cfg_threshold
    preds_binary = (prob_scam >= cfg_threshold).astype(int)

    acc = accuracy_score(y_true, preds_binary)
    roc_auc = roc_auc_score(y_true, prob_scam)
    avg_prec = average_precision_score(y_true, prob_scam)

    sep("═")
    print(f"   [{label}] KẾT QUẢ — Calibration + Binary")
    sep("═")
    print(f"  Accuracy (@{cfg_threshold:.2f}): {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC          : {roc_auc:.4f}")
    print(f"  Average Precision: {avg_prec:.4f}")
    sep()
    print(classification_report(
        y_true, preds_binary, target_names=["Harmless", "Scam"], digits=4))

    cm = confusion_matrix(y_true, preds_binary)
    tn, fp, fn, tp = cm.ravel()
    sep()
    print(f"  Confusion Matrix:")
    print(f"                Pred Harmless  Pred Scam")
    print(f"  True Harmless     {tn:>6}        {fp:>6}")
    print(f"  True Scam         {fn:>6}        {tp:>6}")
    print(f"\n  Sensitivity (Recall Scam)     : {tp/max(tp+fn,1):.4f}")
    print(f"  Specificity (Recall Harmless) : {tn/max(tn+fp,1):.4f}")

    # Threshold search
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        p = (prob_scam >= t).astype(int)
        f = f1_score(y_true, p, pos_label=1, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    preds_t = (prob_scam >= threshold).astype(int)
    sep()
    print(f"  Threshold tối ưu: {best_t:.2f} (F1={best_f1:.4f})")
    print(f"  Threshold sử dụng: {threshold:.2f}")
    print(classification_report(
        y_true, preds_t, target_names=["Harmless", "Scam"], digits=4))

    # Early-stop
    def run_early_stop(thr, title):
        early_probs, stop_turns = early_stop_predict(turn_probs_list, thr)
        early_preds = (early_probs >= thr).astype(int)
        triggered_idx = [i for i in range(len(stop_turns))
                         if stop_turns[i] < len(turn_probs_list[i])]
        n_early = len(triggered_idx)
        n_scam_true = int((y_true == 1).sum())
        n_scam_detected = int(((early_preds == 1) & (y_true == 1)).sum())
        detection_rate = n_scam_detected / max(n_scam_true, 1)
        n_harmless_true = int((y_true == 0).sum())
        n_false_alarm = int(((early_preds == 1) & (y_true == 0)).sum())
        false_alarm_rate = n_false_alarm / max(n_harmless_true, 1)
        scam_pred_idx = [i for i in range(len(stop_turns)) if early_preds[i] == 1]
        avg_stop_all = (float(np.mean([stop_turns[i] for i in scam_pred_idx]))
                        if scam_pred_idx else float("nan"))

        sep("═")
        print(f"   EARLY-STOP — {title}")
        sep("═")
        print(f"  Threshold : {thr:.2f}")
        print(f"  Triggered : {n_early} / {len(y_true)}")
        if n_early > 0:
            avg_s = float(np.mean([stop_turns[i] for i in triggered_idx]))
            print(f"  Avg stop turn (triggered): {avg_s:.2f}")
        print(f"  Detection rate : {detection_rate:.4f} ({n_scam_detected}/{n_scam_true})")
        print(f"  False alarm    : {false_alarm_rate:.4f} ({n_false_alarm}/{n_harmless_true})")
        sep()
        print(classification_report(
            y_true, early_preds, target_names=["Harmless", "Scam"], digits=4))

    run_early_stop(threshold, f"Threshold={threshold:.2f}")
    run_early_stop(best_t, f"Threshold tối ưu={best_t:.2f}")

    # Calibration
    metrics, rows = calibration_metrics(y_true, prob_scam, args.n_bins)
    sep("═")
    print(f"   CALIBRATION — {label}")
    sep("═")
    print(f"  ECE:   {metrics['ece']:.4f}")
    print(f"  MCE:   {metrics['mce']:.4f}")
    print(f"  Brier: {metrics['brier']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")

    # Error analysis
    fp_idx = np.where((preds_binary == 1) & (y_true == 0))[0]
    fn_idx = np.where((preds_binary == 0) & (y_true == 1))[0]
    sep("═")
    print(f"  Errors (@{cfg_threshold:.2f}): {len(fp_idx)+len(fn_idx)}")
    print(f"  FP (Harmless→Scam): {len(fp_idx)}")
    print(f"  FN (Scam→Harmless): {len(fn_idx)}")
    sep("═")

    return {
        "label": label, "y_true": y_true, "prob_scam": prob_scam,
        "threshold": threshold, "cm": cm,
        "roc_auc": roc_auc, "avg_prec": avg_prec, "metrics": metrics,
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_dirs = args.exp_dirs
    labels = args.labels or [os.path.basename(d.rstrip("/")) for d in exp_dirs]
    if len(labels) != len(exp_dirs):
        sys.exit("ERROR: số --labels phải bằng số --exp-dirs")

    # Load test data once
    test_dlg = load_json(args.test_file)
    n_scam = sum(1 for d in test_dlg if d["label"] == "scam")
    print(f"\nTest set: {args.test_file}  ({len(test_dlg)} mẫu)")
    print(f"  harmless: {len(test_dlg) - n_scam}  |  scam: {n_scam}")

    results = []
    for exp_dir, label in zip(exp_dirs, labels):
        sep("█", 70)
        print(f"\n▶ Model: {label}  ({exp_dir})")
        sep("█", 70)

        model, tokenizer, cfg = load_binary_model(exp_dir, device)
        print(f"✅ Loaded ({sum(p.numel() for p in model.parameters()):,} params)")

        ds = BinaryDialogueDataset(test_dlg, tokenizer,
                                   cfg.max_turn_len, cfg.max_turns)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)

        print("Running inference...")
        y_true, prob_scam, turn_probs_list = collect_probs(model, loader, device)

        cfg_thr = float(getattr(cfg, "binary_threshold", 0.5))
        r = evaluate_one(label, y_true, prob_scam, turn_probs_list,
                         test_dlg, args, cfg_thr)
        results.append(r)

        # Free GPU memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Combined plot ──────────────────────────────────────────────────
    if not args.no_plot:
        out_path = args.out or os.path.join(
            os.path.dirname(exp_dirs[0]), "calibration_compare.png")
        make_plots(results, args, out_path)

    # Save combined metrics
    combined = {}
    for r in results:
        combined[r["label"]] = {
            **r["metrics"],
            "roc_auc": r["roc_auc"],
            "avg_prec": r["avg_prec"],
        }
    metrics_path = os.path.join(
        os.path.dirname(exp_dirs[0]), "calibration_compare_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved combined metrics → {metrics_path}")


if __name__ == "__main__":
    main()
