import numpy as np

from .schema import Verdict


def _f1(p, y, t):
    pred = np.asarray(p) >= t
    y = np.asarray(y).astype(bool)
    tp = (pred & y).sum()
    fp = (pred & ~y).sum()
    fn = (~pred & y).sum()
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def calibrate_threshold(p, y):
    grid = np.arange(0.05, 0.951, 0.05)
    scores = [_f1(p, y, t) for t in grid]
    best = max(scores)
    # middle of the plateau of best thresholds
    ts = [t for t, s in zip(grid, scores) if s == best]
    return float(ts[len(ts) // 2])


def apply_corrections(verdicts, flips):
    """flips: track_ids toggled relative to the ORIGINAL verdicts (so a double flip is removing from the set)."""
    out = []
    for v in verdicts:
        if v.track_id in flips:
            label = "no" if v.label == "yes" else "yes"
            out.append(Verdict(v.track_id, v.p_yes, label))
        else:
            out.append(v)
    return out


def tune_threshold(prev, p_yes, flips):
    """Nudge threshold toward the corrected labels. flips are indices into p_yes."""
    if not flips:
        return prev
    p = np.asarray(p_yes)
    y = np.array([(p[i] < prev) if i in flips else (p[i] >= prev) for i in range(len(p))])
    new = calibrate_threshold(p, y)
    return float(0.5 * prev + 0.5 * new)
