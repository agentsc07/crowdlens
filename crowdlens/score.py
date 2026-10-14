"""Pass 2 (per question, milliseconds): yes/no for every person in every frame. Pure NumPy."""
import re

import numpy as np

from . import config
from .embed import embed_text, logit_scale

_LEAD = re.compile(r"^\s*(how many|show me|count|find)\b\s*(people|persons|persons|everyone|of them|they)?\s*(are|is|have|has)?\s*", re.I)


def parse_attribute(question):
    q = question.strip().rstrip("?.! ")
    q = _LEAD.sub("", q, count=1)
    q = re.sub(r"^(people|persons|everyone|are|is)\s+", "", q, flags=re.I).strip()
    if not q:
        raise ValueError("Couldn't find an attribute, e.g. 'wearing a red shirt'")
    return q


def prompts(attribute):
    return f"a photo of a person {attribute}", f"a photo of a person not {attribute}"


def yes_probs(embs, pos, neg, scale=100.0):
    """softmax over the two prompts; embs (n,d), pos/neg (d,) -> (n,) P(yes)."""
    if len(embs) == 0:
        return np.zeros(0, "float32")
    a, b = scale * embs @ pos, scale * embs @ neg
    return (1 / (1 + np.exp(b - a))).astype("float32")


def ask(dets, question):
    attr = parse_attribute(question)
    pos, neg = embed_text(prompts(attr))
    return attr, yes_probs(dets["emb"], pos, neg, logit_scale())


def frame_counts(frames, p, n_frames, stride, threshold=config.THRESHOLD):
    """Per-frame (yes, total). Frames the detector skipped repeat the previous processed frame."""
    yes = np.zeros(n_frames, int)
    tot = np.zeros(n_frames, int)
    for f in np.unique(frames):
        m = frames == f
        yes[f] = int((p[m] >= threshold).sum())
        tot[f] = int(m.sum())
    for f in range(n_frames):
        if f % stride != 0:
            yes[f], tot[f] = yes[f - f % stride], tot[f - f % stride]
    return yes, tot


def unique_yes(tracks, p, threshold=config.THRESHOLD):
    """People (tracks) whose mean P(yes) over the video is above threshold."""
    ids = np.unique(tracks)
    return [int(t) for t in ids if p[tracks == t].mean() >= threshold], len(ids)
