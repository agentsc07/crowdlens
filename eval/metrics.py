import numpy as np


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true)))) if len(pred) else 0.0


def in_range_rate(ranges, true):
    return float(np.mean([lo <= t <= hi for (lo, hi), t in zip(ranges, true)])) if len(true) else 0.0


def recall_at_k(ranked_ids, true_ids, k):
    true = set(true_ids)
    return 1.0 if not true else len(true & set(ranked_ids[:k])) / len(true)


def f1_acc(pred, y):
    pred, y = np.asarray(pred).astype(bool), np.asarray(y).astype(bool)
    tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum()
    f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return float((pred == y).mean()), float(f1)
