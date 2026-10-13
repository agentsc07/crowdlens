import json
import time
from pathlib import Path

import streamlit as st

from crowdlens import config, pipeline
from crowdlens.calibrate import apply_corrections
from crowdlens.cost import cost
from crowdlens.overlay import render_overlay
from crowdlens.query import answer
from crowdlens.schema import load_people
from crowdlens.timeline import per_minute
from crowdlens.vote import count

st.set_page_config(page_title="CrowdLens", layout="wide")
st.title("CrowdLens")
st.caption("Counts attributes, never identities. Everything runs locally.")

with st.sidebar:
    vname = st.selectbox("Verifier", ["Small VLM (default)", "SigLIP zero-shot", "Fine-tuned classifier"])
    k = st.slider("Shortlist size k", 10, 1000, config.K_DEFAULT)
    threshold = st.slider("Yes threshold", 0.1, 0.9, config.THRESHOLD)

up = st.file_uploader("Upload a crowd video", type=["mp4", "mov", "avi"])
if up and st.button("Index video"):
    vid = Path(up.name).stem
    tmp = config.DATA_DIR / f"{vid}_upload{Path(up.name).suffix}"
    config.DATA_DIR.mkdir(exist_ok=True)
    tmp.write_bytes(up.getvalue())
    bar = st.progress(0.0, "Detecting and tracking people...")
    people = pipeline.index_video(tmp, vid, progress=lambda f: bar.progress(f))
    st.session_state.update(vid=vid, video=str(tmp), flips=set())
    st.success(f"Indexed {len(people)} people")

vid = st.session_state.get("vid")
if not vid:
    st.info("Upload and index a clip to start.")
    st.stop()

d = pipeline.video_dir(vid)
fps = pipeline.fps(vid)
people = {p.track_id: p for p in load_people(d / "people.json")}
hist_f = d / "history.json"
history = json.loads(hist_f.read_text()) if hist_f.exists() else {}

q = st.text_input("Ask a question", placeholder="how many people are wearing red shirts?")
if q and st.button("Ask"):
    key = f"{vname}|{k}|{threshold}|{q.strip().lower()}"
    if key in history:  # question history: instant
        st.session_state["ans"] = history[key]
    else:
        try:
            ver = pipeline.make_verifier(vname)
            ans = answer(d, q, ver, k=k, threshold=threshold)
        except (ValueError, KeyError) as e:
            st.error(str(e))
            st.stop()
        rec = dict(q=q, verdicts=[v.__dict__ for v in ans.verdicts], n_verified=ans.n_verified,
                   n_total=ans.n_total, seconds=ans.seconds)
        history[key] = rec
        hist_f.write_text(json.dumps(history))
        st.session_state["ans"] = rec
    st.session_state["flips"] = set()

rec = st.session_state.get("ans")
if rec:
    from crowdlens.schema import Verdict
    base = [Verdict(**v) for v in rec["verdicts"]]
    flips = st.session_state.setdefault("flips", set())
    verdicts = apply_corrections(base, flips)
    low, high = count(verdicts)
    st.header(f"{low}" if low == high else f"{low} ({low}-{high})")
    st.caption(f"{rec['n_verified']} of {rec['n_total']} people verified in {rec['seconds']:.1f}s")

    c = cost(rec["n_total"], rec["n_verified"], rec["seconds"] / max(rec["n_verified"], 1))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Embedded", c["embedded"]); m2.metric("Verified", c["verified"])
    m3.metric("Verifier calls saved", c["saved_calls"]); m4.metric("Time saved", f"{c['saved_seconds']:.0f}s")

    matched = [v for v in verdicts if v.label in ("yes", "unclear")]
    st.subheader("Evidence")
    cols = st.columns(6)
    for i, v in enumerate(sorted(matched, key=lambda v: -v.p_yes)):
        p = people[v.track_id]
        with cols[i % 6]:
            if p.crops:
                st.image(str(d / p.crops[0].path), caption=f"{v.label} · {v.p_yes:.2f} · {p.first_frame / fps:.0f}s")
            if st.checkbox("wrong", key=f"flip{v.track_id}", value=v.track_id in flips):
                flips.add(v.track_id)
            else:
                flips.discard(v.track_id)
            if st.button("jump", key=f"jump{v.track_id}"):
                st.session_state["start"] = int(p.first_frame / fps)

    st.subheader("Video")
    ov = st.toggle("Only boxes around matching people")
    if ov:
        out = d / "overlay.mp4"
        render_overlay(st.session_state["video"], d / "people.json", [v.track_id for v in verdicts if v.label == "yes"], out)
        st.video(str(out), start_time=st.session_state.get("start", 0))
    else:
        st.video(st.session_state["video"], start_time=st.session_state.get("start", 0))

    st.subheader("Count over time")
    tl = per_minute(list(people.values()), verdicts, fps)
    if tl:
        st.bar_chart({f"min {m}": n for m, n in sorted(tl.items())})
