"""Per-question scoring. The VLM answers once per tracked person (on their best crop); every box of that
person in every frame shares the answer. Too-small crops are NaN = "unclear"."""
import json
import re
from pathlib import Path

import cv2
import numpy as np

from . import config

_LEAD = re.compile(r"^\s*(how many|show me|count|find)\b\s*(people|persons|everyone|of them|they)?\s*(are|is|have|has)?\s*", re.I)


def parse_attribute(question):
    q = question.strip().rstrip("?.! ")
    q = _LEAD.sub("", q, count=1)
    q = re.sub(r"^(people|persons|everyone|are|is)\s+", "", q, flags=re.I).strip()
    if not q:
        raise ValueError("Couldn't find an attribute, e.g. 'wearing a red shirt'")
    return q


def track_scores(out_dir, tracks, attribute, progress=None):
    """-> {track_id: P(yes) or NaN}. Cached per attribute in answers.json."""
    out_dir = Path(out_dir)
    cache_f = out_dir / "answers.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    if attribute in cache:
        return {int(k): (np.nan if v is None else v) for k, v in cache[attribute].items()}
    ids = sorted(set(int(t) for t in tracks))
    ok, small = [], []
    for t in ids:
        f = out_dir / "crops" / f"{t}.jpg"
        img = cv2.imread(str(f)) if f.exists() else None
        (ok if img is not None and img.shape[0] >= config.MIN_CROP_H else small).append(t)
    scores = {t: float("nan") for t in small}
    if ok:
        from . import vlm
        ps = vlm.p_yes([str(out_dir / "crops" / f"{t}.jpg") for t in ok], attribute, progress=progress)
        scores.update(zip(ok, ps))
    cache[attribute] = {str(k): (None if np.isnan(v) else v) for k, v in scores.items()}
    cache_f.write_text(json.dumps(cache))
    return scores


def detection_probs(tracks, scores):
    return np.array([scores.get(int(t), np.nan) for t in tracks], dtype="float64")


def is_yes(p, threshold=config.THRESHOLD):
    return np.nan_to_num(p, nan=0.0) >= threshold


def frame_counts(frames, p, n_frames, stride, threshold=config.THRESHOLD):
    """Per-frame (yes, total). Frames the detector skipped repeat the previous processed frame."""
    yes = np.zeros(n_frames, int)
    tot = np.zeros(n_frames, int)
    y = is_yes(p, threshold)
    for f in np.unique(frames):
        m = frames == f
        yes[f], tot[f] = int(y[m].sum()), int(m.sum())
    for f in range(n_frames):
        if f % stride != 0:
            yes[f], tot[f] = yes[f - f % stride], tot[f - f % stride]
    return yes, tot


def people_summary(scores, threshold=config.THRESHOLD):
    """(yes, no, unclear) counts over tracked people."""
    v = np.array(list(scores.values()), dtype="float64")
    unclear = int(np.isnan(v).sum())
    yes = int((np.nan_to_num(v, nan=0.0) >= threshold).sum())
    return yes, len(v) - yes - unclear, unclear
