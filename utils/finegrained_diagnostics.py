"""Class-level FSCIL diagnostics from final test predictions.

Reports directed confusion (true class -> predicted class), not semantic
similarity or biological family membership. All rates are sample-based.
"""
import numpy as np
import torch


def summarize_class_confusions(logits, targets, class_names, base_class_ids,
                               top_k=25):
    """Return JSON-serializable class-wise accuracy and directed errors.

    Classification label IDs must index the columns of logits and the
    corresponding class_names list. No test labels are used in training.
    """
    scores = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
    truth = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    if scores.ndim != 2 or truth.ndim != 1 or scores.shape[0] != truth.size:
        raise ValueError("Expected logits [N, C] and targets [N]")
    if len(class_names) < scores.shape[1]:
        raise ValueError("class_names must cover every classifier column")
    if not 0 <= int(top_k):
        raise ValueError("top_k must be nonnegative")
    if np.any(truth < 0) or np.any(truth >= scores.shape[1]):
        raise ValueError("Test targets do not match the seen-class logits")

    preds = np.argmax(scores, axis=1)
    counts = np.bincount(truth.astype(np.int64), minlength=scores.shape[1])
    correct = np.bincount(truth[truth == preds].astype(np.int64),
                          minlength=scores.shape[1])
    base_ids = set(int(x) for x in base_class_ids)
    class_rows = []
    confusion_rows = []
    for cid, total in enumerate(counts.tolist()):
        if total == 0:
            continue
        group = "base" if cid in base_ids else "novel"
        class_rows.append({
            "class_id": cid,
            "class_name": str(class_names[cid]),
            "group": group,
            "support": total,
            "correct": int(correct[cid]),
            "accuracy_pct": round(100.0 * int(correct[cid]) / total, 4),
        })
        wrong = preds[(truth == cid) & (preds != cid)]
        if wrong.size == 0:
            continue
        destinations, miscounts = np.unique(wrong, return_counts=True)
        for predicted, n in zip(destinations, miscounts):
            j = int(predicted)
            confusion_rows.append({
                "true_id": cid,
                "true_name": str(class_names[cid]),
                "true_group": group,
                "pred_id": j,
                "pred_name": str(class_names[j]),
                "pred_group": "base" if j in base_ids else "novel",
                "count": int(n),
                "true_class_support": total,
                "rate_pct": round(100.0 * int(n) / total, 4),
            })

    # Stable ordering and human-readable directed confusion counts.
    confusion_rows.sort(key=lambda x: (-x["count"], -x["rate_pct"],
                                        x["true_id"], x["pred_id"]))
    class_rows.sort(key=lambda x: (x["accuracy_pct"], -x["support"], x["class_id"]))
    selected = confusion_rows[:int(top_k)]
    novel_errors = [r for r in confusion_rows if r["true_group"] == "novel"]
    base_errors = [r for r in confusion_rows if r["true_group"] == "base"]
    return {
        "num_test_samples": int(truth.size),
        "num_seen_classes": int(scores.shape[1]),
        "num_classes_with_test_samples": len(class_rows),
        "top_k": int(top_k),
        "top_confusions": selected,
        "top_novel_confusions": novel_errors[:int(top_k)],
        "top_base_confusions": base_errors[:int(top_k)],
        "hardest_classes": class_rows[:int(top_k)],
        # Full class accuracy and directed errors allow paired diagnosis.
        "per_class": sorted(class_rows, key=lambda x: x["class_id"]),
        "all_confusion_pairs": confusion_rows,
    }
