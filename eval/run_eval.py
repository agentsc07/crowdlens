"""Score every judgeable eval person on every question with (a) Qwen3-VL P(yes) and (b) SigLIP2 similarity.
Saves ~/.crowdlens/eval/scores.json = {question: {"vlm": {key: p}, "sim": {key: s}, "vlm_s": seconds, "sim_s": seconds}}"""
import json, time
from pathlib import Path

from crowdlens import search, vlm
from crowdlens.config import MIN_CROP_H

EVAL = Path.home() / ".crowdlens" / "eval"
Q = ["wearing a red top", "wearing a white shirt", "wearing a black jacket", "carrying a backpack",
     "carrying a handbag or shoulder bag", "wearing shorts", "wearing a hat or cap", "holding a shopping bag"]
people = [p for p in json.loads((EVAL / "people.json").read_text()) if p["judge"]]
keys = [f"{p['clip']}:{p['track']}" for p in people]
paths = [str(EVAL / "idx" / p["clip"] / "crops" / f"{p['track']}.jpg") for p in people]

vlm._load()  # exclude one-off model load from per-question timings
t = time.time(); emb = {c: search.index_embeddings(EVAL / "idx" / c) for c in sorted({p["clip"] for p in people})}
embed_s = time.time() - t
import numpy as np
out = {"_embed_seconds": embed_s}
for q in Q:
    t = time.time(); pv = vlm.p_yes(paths, q); vs = time.time() - t
    t = time.time(); tv = search.embed_text([f"a photo of a person {q}"])[0]
    sims = {}
    for c, (tr, e) in emb.items():
        for tid, s in zip(tr.tolist(), (e @ tv).tolist()):
            sims[f"{c}:{tid}"] = s
    ss = time.time() - t
    out[q] = {"vlm": dict(zip(keys, pv)), "sim": {k: sims[k] for k in keys}, "vlm_s": vs, "sim_s": ss}
    print(f"{q}: vlm {vs:.1f}s sim {ss:.2f}s", flush=True)
    (EVAL / "scores.json").write_text(json.dumps(out))
