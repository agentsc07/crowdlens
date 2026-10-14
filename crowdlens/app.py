import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import streamlit as st

from crowdlens import config
from crowdlens.index import index_video, load
from crowdlens.render import draw, render_video
from crowdlens.score import (detection_probs, frame_counts, is_yes, parse_attribute, people_summary,
                             track_scores)

st.set_page_config(page_title="CrowdLens", layout="wide")
st.title("CrowdLens")
st.caption("A small open VLM answers yes/no for every tracked person. Attributes only, never identities. Runs locally.")

with st.sidebar:
    stride = st.select_slider("Detect every Nth frame (higher = faster)", [1, 2, 3, 4, 6, 8], config.DEFAULT_STRIDE)
    threshold = st.slider("Yes threshold", 0.1, 0.9, config.THRESHOLD, 0.05)

up = st.file_uploader("Upload a crowd video", type=["mp4", "mov", "avi"])
if up and st.button("Index video", type="primary"):
    vid = Path(up.name).stem
    config.DATA_DIR.mkdir(exist_ok=True)
    video = config.DATA_DIR / f"{vid}{Path(up.name).suffix}"
    video.write_bytes(up.getvalue())
    bar = st.progress(0.0)
    n = index_video(video, config.DATA_DIR / vid, stride, lambda f, msg="": bar.progress(f, msg))
    st.session_state.update(vid=vid, video=str(video), result=None)
    st.success(f"Tracked {n} people")

vid = st.session_state.get("vid")
if not vid:
    st.info("Upload and index a clip to start.")
    st.stop()

d = config.DATA_DIR / vid
dets, meta = load(d)

q = st.text_input("Ask a question", placeholder="wearing a hat")
if q and st.button("Ask"):
    try:
        attr = parse_attribute(q)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    bar = st.progress(0.0, "Asking the VLM about each person…")
    scores = track_scores(d, dets["track"], attr, lambda f: bar.progress(f, "Asking the VLM about each person…"))
    p = detection_probs(dets["track"], scores)
    bar.progress(0.0, "Rendering annotated video…")
    out = render_video(st.session_state["video"], dets, meta, p, attr, d / "annotated.mp4", threshold,
                       lambda f: bar.progress(f, "Rendering annotated video…"))
    bar.empty()
    st.session_state["result"] = dict(attr=attr, scores=scores, p=p, out=str(out), thr=threshold)

res = st.session_state.get("result")
if not res:
    st.stop()

p, scores = res["p"], res["scores"]
yes_n, no_n, unclear_n = people_summary(scores, threshold)
yes, tot = frame_counts(dets["frame"], p, meta["frames"], meta["stride"], threshold)
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Yes: “{res['attr']}”", yes_n)
c2.metric("No", no_n)
c3.metric("Too small to judge", unclear_n)
c4.metric("Peak yes in one frame", int(yes.max()) if len(yes) else 0)
if res["thr"] != threshold:
    st.caption("Threshold changed since the video was rendered: counts are live, the video is not. Ask again to re-render.")

left, right = st.columns([3, 2])
with left:
    st.subheader("Annotated video")
    st.video(res["out"])
with right:
    st.subheader("Yes count per frame")
    st.line_chart({"yes": yes, "all people": tot})

st.subheader("Every person, with the crop the VLM saw")
order = sorted(scores, key=lambda t: -np.nan_to_num(scores[t], nan=-1))
show = st.radio("Show", ["Yes", "No", "Too small", "All"], horizontal=True)
cols = st.columns(8)
n = 0
for t in order:
    v = scores[t]
    label = "Too small" if np.isnan(v) else "Yes" if v >= threshold else "No"
    if show not in ("All", label):
        continue
    f = d / "crops" / f"{t}.jpg"
    if f.exists():
        cols[n % 8].image(str(f), caption=f"{label}" + ("" if np.isnan(v) else f" {v:.2f}"))
        n += 1

st.subheader("Inspect a frame")
fr = st.slider("Frame", 0, max(meta["frames"] - 1, 0), 0)
cap = cv2.VideoCapture(st.session_state["video"])
cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
ok, img = cap.read()
cap.release()
if ok:
    m = dets["frame"] == fr - fr % meta["stride"]
    img = draw(img, dets["box"][m], dets["track"][m], p[m], threshold, f"frame {fr}: yes {yes[fr]} / {tot[fr]}")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
