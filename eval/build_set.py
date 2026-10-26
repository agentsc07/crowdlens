"""Index each eval clip with the normal CrowdLens pipeline and list the people (tracks) whose best crop is big
enough to judge. Writes ~/.crowdlens/eval/idx/<clip>/ and ~/.crowdlens/eval/people.json."""
import json, sys, time
from pathlib import Path

import cv2

from crowdlens import config
from crowdlens.index import index_video

EVAL = Path.home() / ".crowdlens" / "eval"
CLIPS = ["02", "04", "09", "11"]
people, times = [], {}
for c in CLIPS:
    out = EVAL / "idx" / c
    if not (out / "meta.json").exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        index_video(EVAL / f"mot17_{c}.mp4", out)
        times[c] = round(time.time() - t, 1)
        print("indexed", c, times[c], "s", flush=True)
    for f in sorted((out / "crops").glob("*.jpg")):
        if "_s" in f.stem:
            continue
        h = cv2.imread(str(f)).shape[0]
        people.append(dict(clip=c, track=int(f.stem), crop_h=h, judge=h >= config.MIN_CROP_H))
(EVAL / "people.json").write_text(json.dumps(people))
if times:
    (EVAL / "index_times.json").write_text(json.dumps(times))
for c in CLIPS:
    p = [x for x in people if x["clip"] == c]
    print(c, "tracks", len(p), "judgeable", sum(x["judge"] for x in p))
