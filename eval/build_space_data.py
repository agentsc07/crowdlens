"""Assemble space_data/ for the hosted demo: indexes of MOT17 clips 02, 04, 11 (first 15 s), the 1080p proxy as the
playable video, and answers.json pre-filled with the 8 eval questions (same model and crops as eval/run_eval.py)."""
import json, shutil
from pathlib import Path

import numpy as np

EVAL = Path.home() / ".crowdlens" / "eval"
OUT = Path(__file__).resolve().parent.parent / "space_data"
S = json.loads((EVAL / "scores.json").read_text())
Q = [q for q in S if not q.startswith("_")]
for clip in ("02", "04", "11"):
    vid = f"MOT17-{clip}"
    src, dst = EVAL / "idx" / clip, OUT / vid
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("answers.json", "annotated.mp4"))
    shutil.copy(src / "proxy.mp4", OUT / f"{vid}.mp4")
    tracks = sorted(int(t) for t in np.unique(np.load(dst / "dets.npz")["track"]))
    cache = {q: {str(t): S[q]["vlm"].get(f"{clip}:{t}") for t in tracks} for q in Q}
    (dst / "answers.json").write_text(json.dumps(cache))
    print(vid, len(tracks), "tracks,", sum(v is not None for v in cache[Q[0]].values()), "answered")
