import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import streamlit as st

from crowdlens import config
from crowdlens.index import index_video, load
from crowdlens.render import draw, render_video
from crowdlens.score import ask, frame_counts, unique_yes

st.set_page_config(page_title="CrowdLens", layout="wide")
st.title("CrowdLens")
st.caption("Per-frame yes/no for every person, from a text question. Attributes only, never identities. Runs locally.")

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

q = st.text_input("Ask a question", placeholder="wearing a red shirt")
if q and st.button("Ask"):
    try:
        attr, p = ask(dets, q)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    bar = st.progress(0.0, "Rendering annotated video…")
    out = render_video(st.session_state["video"], dets, meta, p, attr, d / "annotated.mp4", threshold,
                       lambda f: bar.progress(f, "Rendering annotated video…"))
    bar.empty()
    st.session_state["result"] = dict(attr=attr, p=p, out=str(out))

res = st.session_state.get("result")
if not res:
    st.stop()

p = res["p"]
yes_ids, n_people = unique_yes(dets["track"], p, threshold)
yes, tot = frame_counts(dets["frame"], p, meta["frames"], meta["stride"], threshold)
c1, c2, c3 = st.columns(3)
c1.metric(f"People matching “{res['attr']}”", f"{len(yes_ids)} of {n_people}")
c2.metric("Peak in a single frame", int(yes.max()) if len(yes) else 0)
c3.metric("Detections scored", len(p))

left, right = st.columns([3, 2])
with left:
    st.subheader("Annotated video")
    st.video(res["out"])
with right:
    st.subheader("Yes count per frame")
    st.line_chart({"yes": yes, "all people": tot})

st.subheader("Inspect a frame")
f = st.slider("Frame", 0, max(meta["frames"] - 1, 0), 0)
cap = cv2.VideoCapture(st.session_state["video"])
cap.set(cv2.CAP_PROP_POS_FRAMES, f)
ok, img = cap.read()
cap.release()
if ok:
    src = f - f % meta["stride"]
    m = dets["frame"] == src
    img = draw(img, dets["box"][m], dets["track"][m], p[m], threshold, f"frame {f}: yes {yes[f]} / {tot[f]}")
    a, b = st.columns([3, 2])
    a.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    b.dataframe({"person (temp id)": dets["track"][m].tolist(), "P(yes)": p[m].round(3).tolist(),
                 "answer": ["yes" if x >= threshold else "no" for x in p[m]]}, hide_index=True)
