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


MOTION_LABELS = {"right": "moving right", "left": "moving left", "toward": "coming toward the camera",
                 "away": "moving away from the camera", "still": "standing still", "moving": "moving"}
_MOTION_RULES = [
    ("still", re.compile(r"\b(standing still|stationary|not moving|motionless|stopped|staying still|idle|static)\b")),
    ("toward", re.compile(r"\b(toward|towards|approaching|approach|coming closer|closer to the camera)\b")),
    ("away", re.compile(r"\b(away from|walking away|going away|moving away|heading away|leaving)\b")),
    ("right", re.compile(r"\b(to the right|rightwards?|(going|went|moving|moved|walking|walked|heading|headed|turning|turned|go|walk|run|running)\s+(to\s+)?(the\s+)?right)\b")),
    ("left", re.compile(r"\b(to the left|leftwards?|(going|went|moving|moved|walking|walked|heading|headed|turning|turned|go|walk|run|running)\s+(to\s+)?(the\s+)?left)\b")),
    ("moving", re.compile(r"^(how many )?(people|persons|everyone|objects|cars|vehicles)?\s*(are |is )?(moving|in motion|walking)\??$")),
]
# actions are about what someone does over time, so one still crop is weak evidence: look at several moments
_ACTION = re.compile(r"\b(running|talking|waving|using|texting|eating|drinking|smoking|dancing|jumping|pointing|filming|"
                     r"photographing|shaking|hugging|pushing|pulling|sitting|crossing|looking at|on (a|the|their) phone|"
                     r"speaking|laughing|stretching|kneeling|bending)\b")


def parse_question(question):
    """Route a question. -> {"kind": "motion", "motion": ..., "label": ...} (answered from track geometry)
    or {"kind": "attr", "attribute": ..., "multi": bool, "label": ...} (answered by the VLM)."""
    q = question.strip().rstrip("?.! ").lower()
    for kind, rx in _MOTION_RULES:
        if rx.search(q):
            return {"kind": "motion", "motion": kind, "label": MOTION_LABELS[kind]}
    attr = parse_attribute(question)
    return {"kind": "attr", "attribute": attr, "multi": bool(_ACTION.search(attr.lower())), "label": attr}


def _cache_key(attribute, subject, multi):
    return attribute if subject == "person" and not multi else f"{subject}|{attribute}|{'multi' if multi else 'single'}"


def is_cached(out_dir, attribute, subject="person", multi=False):
    f = Path(out_dir) / "answers.json"
    return f.exists() and _cache_key(attribute, subject, multi) in json.loads(f.read_text())


def track_scores(out_dir, tracks, attribute, progress=None, on_result=None, subject="person", multi=False, only=None):
    """-> {track_id: P(yes) or NaN}. Cached per question in answers.json.
    multi: show the VLM several moments of each object (needs the _s<k> crops saved at index time).
    only: search-then-verify. Track ids the VLM should check; everyone else scores 0.0 (screened out by search)."""
    out_dir = Path(out_dir)
    cache_f = out_dir / "answers.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    key = _cache_key(attribute, subject, multi)
    if only is not None and key not in cache:  # a full verification already on file beats a shortlist run
        key += f"|shortlist{len(only)}"
    if key in cache:
        return {int(k): (np.nan if v is None else v) for k, v in cache[key].items()}
    ids = sorted(set(int(t) for t in tracks))
    skipped = [t for t in ids if only is not None and t not in only]
    ids = [t for t in ids if t not in skipped]
    ok, small = [], []
    for t in ids:
        f = out_dir / "crops" / f"{t}.jpg"
        img = cv2.imread(str(f)) if f.exists() else None
        (ok if img is not None and img.shape[0] >= config.MIN_CROP_H else small).append(t)
    scores = {t: float("nan") for t in small}
    scores.update({t: 0.0 for t in skipped})
    if on_result:
        for t in small:
            on_result(t, float("nan"))
    if ok:
        from . import vlm

        def on_batch(start, got):
            if on_result:
                for t, v in zip(ok[start:start + len(got)], got):
                    on_result(t, v)

        def paths(t):
            if multi:
                moments = sorted((out_dir / "crops").glob(f"{t}_s*.jpg"))
                if len(moments) > 1:
                    return [str(m) for m in moments]
            return [str(out_dir / "crops" / f"{t}.jpg")]

        ps = vlm.p_yes([paths(t) for t in ok], attribute, subject=subject, batch=4 if multi else 8,
                       progress=progress, on_batch=on_batch)
        scores.update(zip(ok, ps))
    cache[key] = {str(k): (None if np.isnan(v) else v) for k, v in scores.items()}
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
