import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .calibrate import apply_corrections
from .embed import embed_text, shortlist
from .schema import Verdict, load_people
from .vote import count, vote

_LEAD = re.compile(r"^\s*(how many|show me|count)\b\s*(people|persons|everyone|are|is|of them|they)?\s*(are|is|have|has)?\s*", re.I)


def split_attributes(question: str):
    q = question.strip().rstrip("?.! ")
    q = _LEAD.sub("", q, count=1)
    q = re.sub(r"^(people|persons|everyone|are|is)\s+", "", q, flags=re.I).strip()
    parts = [p.strip(" ,") for p in re.split(r"\s+and\s+|,", q) if p.strip(" ,")]
    if not parts:
        raise ValueError("Could not find an attribute in the question, e.g. 'wearing a red shirt'")
    return parts


@dataclass
class Answer:
    low: int
    high: int
    verdicts: list
    n_verified: int
    n_total: int
    seconds: float
    attributes: list = field(default_factory=list)
    p_yes: dict = field(default_factory=dict)  # track_id -> min p_yes across attributes


def answer(video_dir, question, verifier, k=config.K_DEFAULT, threshold=config.THRESHOLD, crop_root=None):
    t0 = time.time()
    video_dir = Path(video_dir)
    crop_root = Path(crop_root or video_dir)
    attrs = split_attributes(question)
    people = {p.track_id: p for p in load_people(video_dir / "people.json")}
    short = shortlist(video_dir, embed_text(question), k)

    verdicts, pmap = [], {}
    for tid, _ in short:
        crops = [c for c in people[tid].crops if (c.box[3] - c.box[1]) >= config.MIN_CROP_H]
        if not crops:
            verdicts.append(Verdict(tid, 0.5, "unclear"))
            continue
        labels, mins = [], []
        for a in attrs:
            ps = [verifier.p_yes(str(crop_root / c.path), a) for c in crops]
            labels.append(vote(ps, threshold))
            mins.append(sum(ps) / len(ps))
        label = "no" if "no" in labels else "yes" if all(x == "yes" for x in labels) else "unclear"
        pmap[tid] = min(mins)
        verdicts.append(Verdict(tid, pmap[tid], label))
    low, high = count(verdicts)
    return Answer(low, high, verdicts, len(short), len(people), time.time() - t0, attrs, pmap)
