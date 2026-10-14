"""Annotated playback: every detected person boxed green (yes) / red (no) with P(yes); HUD with the question."""
import subprocess

import cv2
import numpy as np

from . import config

GREEN, RED = (60, 200, 60), (50, 50, 230)


def draw(frame, boxes, tracks, p, threshold, hud):
    for (x1, y1, x2, y2), t, pi in zip(boxes, tracks, p):
        c = GREEN if pi >= threshold else RED
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        label = f"{'YES' if pi >= threshold else 'no'} {pi:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)), c, -1)
        cv2.putText(frame, label, (int(x1) + 2, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(frame, hud, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _by_frame(dets, p, stride):
    out = {}
    for f in np.unique(dets["frame"]):
        m = dets["frame"] == f
        out[int(f)] = (dets["box"][m], dets["track"][m], p[m])
    return out


def render_video(video_path, dets, meta, p, question, out_path, threshold=config.THRESHOLD, progress=None):
    stride = meta["stride"]
    per = _by_frame(dets, p, stride)
    yes, tot = None, None
    from .score import frame_counts
    yes, tot = frame_counts(dets["frame"], p, meta["frames"], stride, threshold)
    cap = cv2.VideoCapture(str(video_path))
    tmp = str(out_path) + ".raw.mp4"
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["w"], meta["h"]))
    last = (np.zeros((0, 4)), np.zeros(0, int), np.zeros(0))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            last = per.get(i, (np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)))
        hud = f'"{question}"   frame {i}   yes {yes[min(i, len(yes) - 1)]} / {tot[min(i, len(tot) - 1)]} people'
        out.write(draw(frame, *last[:2], last[2], threshold, hud))
        i += 1
        if progress and i % 10 == 0:
            progress(min(i / meta["frames"], 1.0))
    cap.release()
    out.release()
    # browsers need H.264; re-encode with ffmpeg when available, else fall back to the raw file
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)])
    if r.returncode != 0:
        import shutil
        shutil.move(tmp, out_path)
    else:
        import os
        os.remove(tmp)
    return out_path
