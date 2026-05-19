import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict, List, Optional


def compute_streaming_metrics(
    all_dialogue_labels: List[int],
    all_dialogue_probs: List[float],
    all_turn_probs: List[np.ndarray],
    threshold: float = 0.5,
) -> Dict:
    d_labels = np.array(all_dialogue_labels)
    d_probs  = np.array(all_dialogue_probs)
    d_preds  = (d_probs >= threshold).astype(int)

    try:
        auroc = float(roc_auc_score(d_labels, d_probs))
    except ValueError:
        auroc = float("nan")

    # Binary confusion matrix elements & precision/recall (giống thaiversion2)
    tp = int(((d_preds == 1) & (d_labels == 1)).sum())
    fp = int(((d_preds == 1) & (d_labels == 0)).sum())
    tn = int(((d_preds == 0) & (d_labels == 0)).sum())
    fn = int(((d_preds == 0) & (d_labels == 1)).sum())
    precision   = tp / max(tp + fp, 1)
    recall      = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    detection_delays = []
    num_scam, num_detected         = 0, 0
    num_harmless, num_false_alarms = 0, 0
    all_alert_turns, all_n_turns   = [], []

    for label, turn_probs in zip(all_dialogue_labels, all_turn_probs):
        first_alert = _first_alert_turn(turn_probs, threshold)
        all_alert_turns.append(first_alert)
        all_n_turns.append(len(turn_probs))
        if label == 1:
            num_scam += 1
            if first_alert is not None:
                num_detected += 1
                detection_delays.append(first_alert)
        else:
            num_harmless += 1
            if first_alert is not None:
                num_false_alarms += 1

    return {
        # Dialogue-level
        "dialogue_accuracy": float(accuracy_score(d_labels, d_preds)),
        "dialogue_f1":       float(f1_score(d_labels, d_preds, zero_division=0)),
        "auroc":             auroc,
        # Binary precision / recall (giống thaiversion2)
        "precision":         precision,
        "recall":            recall,
        "specificity":       specificity,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        # Streaming detection (giống thaiversion2)
        "detection_rate":      num_detected / max(num_scam, 1),
        "avg_detection_delay": float(np.mean(detection_delays)) if detection_delays else float("nan"),
        "false_alarm_rate":    num_false_alarms / max(num_harmless, 1),
        "num_scam":            num_scam,
        "num_harmless":        num_harmless,
        "num_detected":        num_detected,
        "num_false_alarms":    num_false_alarms,
        # Early-detection stats
        **early_detection_stats(all_alert_turns, all_n_turns, all_dialogue_labels),
        # Aliases cho train.py early stopping
        "accuracy": float(accuracy_score(d_labels, d_preds)),
        "f1":       float(f1_score(d_labels, d_preds, zero_division=0)),
    }


def early_detection_stats(all_alert_turns: List[Optional[int]],
                           all_n_turns: List[int],
                           all_labels: List[int]) -> Dict:
    tp_turns, tp_fracs = [], []
    for alert_t, n, label in zip(all_alert_turns, all_n_turns, all_labels):
        if label == 1 and alert_t is not None:
            tp_turns.append(alert_t)
            tp_fracs.append(alert_t / max(n, 1))
    if not tp_turns:
        return {}
    return {
        "median_alert_turn": float(np.median(tp_turns)),
        "mean_alert_turn":   float(np.mean(tp_turns)),
        "median_lead_frac":  float(np.median(tp_fracs)),
        "alert_at_half":     float(np.mean([f <= 0.5 for f in tp_fracs])),
    }


def print_streaming_report(metrics: Dict, inv_label_map: Dict[int, str] = None):
    print("\n" + "=" * 60)
    print("M1 STREAMING EVALUATION REPORT")
    print("=" * 60)

    print("\n  Dialogue-Level Metrics (last-turn prediction):")
    print(f"    Accuracy:    {metrics['dialogue_accuracy']:.4f}")
    print(f"    F1:          {metrics['dialogue_f1']:.4f}")
    if not np.isnan(metrics.get("auroc", float("nan"))):
        print(f"    AUROC:       {metrics['auroc']:.4f}")

    print("\n  Binary Precision / Recall:")
    print(f"    Precision:   {metrics['precision']:.4f}")
    print(f"    Recall:      {metrics['recall']:.4f}")
    print(f"    Specificity: {metrics['specificity']:.4f}")
    print(f"    TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")

    print("\n  Streaming Detection (per-turn threshold):")
    print(f"    Detection rate:   {metrics['detection_rate']:.4f} "
          f"({metrics['num_detected']}/{metrics['num_scam']})")
    if not np.isnan(metrics["avg_detection_delay"]):
        print(f"    Avg delay:        {metrics['avg_detection_delay']:.2f} turns")
    print(f"    False alarm rate: {metrics['false_alarm_rate']:.4f} "
          f"({metrics['num_false_alarms']}/{metrics['num_harmless']})")

    if "median_alert_turn" in metrics:
        print("\n  Early Detection Stats (TP only):")
        print(f"    Median alert turn: {metrics['median_alert_turn']:.1f}")
        print(f"    Mean alert turn:   {metrics['mean_alert_turn']:.1f}")
        print(f"    Median lead frac:  {metrics['median_lead_frac']:.3f}")
        print(f"    Alert in 1st half: {metrics['alert_at_half']:.3f}")

    if "multiclass_accuracy" in metrics:
        print(f"\n  Multiclass Scenario Accuracy: {metrics['multiclass_accuracy']:.4f}")
        if "per_class_accuracy" in metrics:
            print("  Per-class accuracy:")
            per_acc   = metrics["per_class_accuracy"]
            per_count = metrics.get("per_class_counts", {})
            for cls_id in sorted(per_acc, key=int):
                name  = inv_label_map.get(int(cls_id), cls_id) if inv_label_map else cls_id
                n     = per_count.get(cls_id, "?")
                print(f"    {name:12s}: {per_acc[cls_id]:.4f}  (n={n})")

    if "loss" in metrics:
        print(f"\n  Loss: {metrics['loss']:.4f}")
    print("=" * 60)


def _first_alert_turn(turn_probs: np.ndarray, threshold: float) -> Optional[int]:
    for i, p in enumerate(turn_probs):
        if float(p) >= threshold:
            return i
    return None
