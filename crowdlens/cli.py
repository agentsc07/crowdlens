"""Command line: `crowdlens index video.mp4`, `crowdlens ask video.mp4 "wearing a red shirt?"`.
Attributes only: it prints counts and track ids that exist for one video, never identities."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from . import config


def _dir(video, category="person"):
    video = Path(video)
    h = hashlib.sha1(f"{video.resolve()}|{video.stat().st_size}".encode()).hexdigest()[:8]
    return config.video_dir(f"{video.stem}-{h}", category)


def cmd_index(a):
    from .index import index_video

    out = _dir(a.video, a.category)
    t = time.time()
    last = [-1]

    def show(f, m, info=None):  # one line per 10%
        if int(f * 10) != last[0]:
            last[0] = int(f * 10)
            print(f"{m}", file=sys.stderr)

    n = index_video(a.video, out, stride=a.stride, category=a.category, progress=show)
    print(f"indexed {n} {a.category} tracks in {time.time() - t:.1f}s -> {out}")


def cmd_ask(a):
    from . import index, score
    from .score import is_yes

    out = _dir(a.video, a.category)
    if not (out / "meta.json").exists():
        cmd_index(a)
    dets, meta = index.load(out)
    route = score.parse_question(a.question)
    t = time.time()
    if route["kind"] == "motion":
        from .motion import motion_scores

        scores, label = motion_scores(dets, route["motion"], meta["fps"]), route["label"]
        note = "answered from track geometry, no model call"
    else:
        only = None
        if a.search:
            from . import search

            only = set(search.shortlist(search.similarity(out, route["attribute"], a.category), a.search))
        scores = score.track_scores(out, dets["track"], route["attribute"], subject=a.category,
                                    multi=route["multi"], only=only)
        label = route["label"]
        note = f"verified {len(only) if only else len(scores)} of {len(scores)} tracks" + (" (search shortlist)" if only else "")
    yes, no, unclear = score.people_summary(scores, a.threshold)
    secs = time.time() - t
    ids = sorted(t for t, p in scores.items() if is_yes(p, a.threshold))
    if a.json:
        print(json.dumps(dict(question=a.question, label=label, yes=yes, no=no, unclear=unclear, seconds=round(secs, 2), tracks=ids)))
    else:
        print(f"{label}: {yes} of {yes + no + unclear} ({unclear} too small to judge) in {secs:.1f}s; {note}")
        print("track ids:", ids)


def main(argv=None):
    p = argparse.ArgumentParser(prog="crowdlens", description="Ask plain-English questions of crowd video.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("index", cmd_index), ("ask", cmd_ask)):
        s = sub.add_parser(name)
        s.add_argument("video")
        if name == "ask":
            s.add_argument("question")
            s.add_argument("--search", type=float, default=0, metavar="K",
                           help="search-then-verify: only check the top K people (count, or fraction if <1)")
            s.add_argument("--threshold", type=float, default=config.THRESHOLD)
            s.add_argument("--json", action="store_true")
        s.add_argument("--category", default="person", choices=config.CATEGORIES)
        s.add_argument("--stride", type=int, default=config.DEFAULT_STRIDE)
        s.set_defaults(fn=fn)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
