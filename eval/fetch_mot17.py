"""Fetch the first 450 frames (15 s) of a few MOT17 train sequences straight out of the 5.9 GB zip with range
requests (resumable, parallel), then encode each to ~/.crowdlens/eval/mot17_<seq>.mp4.
MOT17 (Milan et al., 2016) is CC BY-NC-SA 3.0: non-commercial use with attribution."""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from remotezip import RemoteZip

URL = "https://motchallenge.net/data/MOT17.zip"
SEQS = sys.argv[1:] or ["02", "04", "09", "11"]
OUT = Path.home() / ".crowdlens" / "eval"
N = 450


def worker(job):
    seq, frames = job
    base = f"MOT17/train/MOT17-{seq}-FRCNN"
    z = None
    for f in frames:
        p = OUT / "raw" / seq / "img1" / f"{f:06d}.jpg"
        if p.exists() and p.stat().st_size > 0:
            continue
        for attempt in range(8):
            try:
                z = z or RemoteZip(URL)
                data = z.read(f"{base}/img1/{f:06d}.jpg")
                p.write_bytes(data)
                break
            except Exception:
                z = None
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"gave up on {seq} frame {f}")


for s in SEQS:
    (OUT / "raw" / s / "img1").mkdir(parents=True, exist_ok=True)
    with RemoteZip(URL) as z:
        for n in ("gt/gt.txt", "seqinfo.ini"):
            t = OUT / "raw" / s / Path(n).name
            if not t.exists():
                t.write_bytes(z.read(f"MOT17/train/MOT17-{s}-FRCNN/{n}"))
jobs = [(s, list(range(1 + k, N + 1, 8))) for s in SEQS for k in range(8)]
with ThreadPoolExecutor(8) as ex:
    list(ex.map(worker, jobs))
for s in SEQS:
    got = len(list((OUT / "raw" / s / "img1").glob("*.jpg")))
    assert got == N, (s, got)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(OUT / "raw" / s / "img1" / "%06d.jpg"),
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(OUT / f"mot17_{s}.mp4")], check=True)
    print("ok", s, flush=True)
