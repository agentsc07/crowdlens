import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import streamlit as st

from crowdlens import config, ui
from crowdlens.index import index_video, load
from crowdlens.render import draw, render_video
from crowdlens.score import (detection_probs, frame_counts, parse_attribute, people_summary, track_scores)

st.set_page_config(page_title="CrowdLens", page_icon="🔍", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)
st.markdown("""<style>
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--glass);border:1px solid var(--edge) !important;border-radius:28px;
padding:6px 10px;box-shadow:0 24px 60px rgba(10,60,255,.12),inset 0 1px 0 rgba(255,255,255,.9);
-webkit-backdrop-filter:blur(22px) saturate(170%);backdrop-filter:blur(22px) saturate(170%)}
</style>""", unsafe_allow_html=True)

THINK_INDEX = ["Scanning every corner of the frame", "Zooming in on distant people", "Following people between frames",
               "Picking each person's sharpest view"]
SUGGESTIONS = ["wearing a hat", "wearing glasses", "wearing a red top", "carrying a bag", "smiling"]


@st.cache_resource
def _warm_state():
    return {"started": False}


def prewarm():
    """Load the VLM in the background while the user is busy indexing / typing, so the first question
    doesn't pay the model-load cost."""
    state = _warm_state()
    if state["started"]:
        return
    state["started"] = True

    def run():
        try:
            from crowdlens import vlm
            vlm.prewarm()
        except Exception:
            state["started"] = False

    threading.Thread(target=run, daemon=True).start()


def eta_text(t0, frac):
    if frac < 0.03:
        return ""
    left = (time.time() - t0) / frac * (1 - frac)
    return f"about {int(left) + 1}s left" if left > 1 else "almost there"


def indexed_videos():
    out = []
    for m in sorted(config.DATA_DIR.glob("*/meta.json")) if config.DATA_DIR.exists() else []:
        vid = m.parent.name
        vids = [f for f in config.DATA_DIR.glob(f"{vid}.*") if f.suffix.lower() in (".mp4", ".mov", ".avi")]
        if vids:
            out.append((vid, str(vids[0])))
    return out


def tiles_for(scores, threshold, d, limit=160):
    out = []
    order = sorted(scores, key=lambda t: -np.nan_to_num(scores[t], nan=-1))
    for t in order:
        v = scores[t]
        kind = "unc" if np.isnan(v) else "yes" if v >= threshold else "no"
        src = ui.thumb(d / "crops" / f"{t}.jpg")
        if src:
            label = "too small" if kind == "unc" else f"{'Yes' if kind == 'yes' else 'No'} {v:.2f}"
            out.append((kind, ui.tile(src, kind, label)))
    return out[:limit] if limit else out


# ---------------------------------------------------------------- header
st.markdown(ui.hero(), unsafe_allow_html=True)
res = st.session_state.get("result")
vid = st.session_state.get("vid")
steps_ph = st.empty()
steps_ph.markdown(ui.stepper(3 if res else 2 if vid else 1, done=(1, 2) if res else (1,) if vid else ()),
                  unsafe_allow_html=True)

with st.expander("Settings", expanded=False):
    stride = st.select_slider("Look at every Nth frame (higher = faster, slightly less precise)", [1, 2, 3, 4, 6, 8],
                              config.DEFAULT_STRIDE)
    threshold = st.slider("Yes threshold", 0.1, 0.9, config.THRESHOLD, 0.05)

# ---------------------------------------------------------------- 1) video
with st.container(border=True):
    st.markdown('<div class="sec">1 · Choose a video</div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload a crowd video", type=["mp4", "mov", "avi"], label_visibility="collapsed")
    existing = indexed_videos()
    pick = None
    if existing:
        pick = st.selectbox("…or reopen one you've already indexed", [None] + existing,
                            format_func=lambda x: "Select…" if x is None else x[0])
    go_index = bool(up) and st.button("Index video", type="primary")
    if pick and st.button("Open", key="open"):
        st.session_state.update(vid=pick[0], video=pick[1], result=None)
        st.rerun()

if go_index:
    prewarm()
    name = Path(up.name)
    config.DATA_DIR.mkdir(exist_ok=True)
    video = config.DATA_DIR / f"{name.stem}{name.suffix}"
    video.write_bytes(up.getvalue())
    steps_ph.markdown(ui.stepper(1), unsafe_allow_html=True)
    live_ph = st.empty()
    left, right = st.columns([3, 2])
    img_ph, stat_ph = left.empty(), right.empty()
    img_ph.markdown(ui.skeleton(380), unsafe_allow_html=True)
    t0 = time.time()
    n_cb = [0]

    def on_progress(f, msg="", info=None):
        n_cb[0] += 1
        phrase = THINK_INDEX[(n_cb[0] // 6) % len(THINK_INDEX)]
        live_ph.markdown(ui.live(f"{phrase}…" if info else msg, f, eta_text(t0, f)), unsafe_allow_html=True)
        if info:
            img_ph.image(info["img"], use_container_width=True)
            stat_ph.markdown(ui.stats([("People in this frame", info["people"], "blue"),
                                       ("Different people so far", info["tracks"], "")]), unsafe_allow_html=True)

    n = index_video(video, config.DATA_DIR / name.stem, stride, on_progress)
    st.session_state.update(vid=name.stem, video=str(video), result=None)
    st.rerun()

vid = st.session_state.get("vid")
if not vid:
    st.stop()

d = config.DATA_DIR / vid
dets, meta = load(d)
n_people = len(set(dets["track"].tolist()))
# frames, playback and inspector read the 1080p proxy written at index time when there is one
play_src = str(d / "proxy.mp4") if meta.get("proxy") and (d / "proxy.mp4").exists() else st.session_state["video"]
prewarm()

# ---------------------------------------------------------------- 2) ask
with st.container(border=True):
    st.markdown(f'<div class="sec">2 · Ask a question<small>{n_people} people found in '
                f'{meta["frames"] / meta["fps"]:.0f}s of video</small></div>', unsafe_allow_html=True)
    with st.form("ask", border=False):
        c1, c2 = st.columns([5, 1])
        q = c1.text_input("Question", key="q", placeholder="e.g. wearing a hat", label_visibility="collapsed")
        submitted = c2.form_submit_button("Ask", type="primary", use_container_width=True)
    chip_cols = st.columns(len(SUGGESTIONS))
    for col, s_ in zip(chip_cols, SUGGESTIONS):
        if col.button(s_, key=f"chip-{s_}", use_container_width=True):
            st.session_state["q"] = s_
            st.session_state["go"] = True
            st.rerun()

go = submitted or st.session_state.pop("go", False)
if go and st.session_state.get("q"):
    try:
        attr = parse_attribute(st.session_state["q"])
    except ValueError as e:
        st.error(str(e))
        st.stop()
    steps_ph.markdown(ui.stepper(2, done=(1,)), unsafe_allow_html=True)
    live_ph, count_ph, grid_ph = st.empty(), st.empty(), st.empty()
    seen, total, t0 = {}, max(n_people, 1), time.time()
    live_ph.markdown(ui.live("Waking up the vision model…", 0.0), unsafe_allow_html=True)

    def on_result(t, v):
        seen[t] = v

    def on_progress(f):
        yes_now = sum(1 for v in seen.values() if not np.isnan(v) and v >= threshold)
        phrase = ["Looking closely at each person", f"Checking for “{attr}”", "Counting as we go"][int(f * 9) % 3]
        live_ph.markdown(ui.live(f"{phrase}… {len(seen)}/{total}", f, eta_text(t0, f)), unsafe_allow_html=True)
        count_ph.markdown(ui.stats([("Yes so far", yes_now, "yes"), ("Checked", len(seen), "blue")]), unsafe_allow_html=True)
        recent = [(k, ui.thumb(d / "crops" / f"{t}.jpg")) for t, v in list(seen.items())[-60:][::-1]
                  for k in ["unc" if np.isnan(v) else "yes" if v >= threshold else "no"]]
        grid_ph.markdown(ui.grid([ui.tile(s, k, {"yes": "Yes", "no": "No", "unc": "?"}[k]) for k, s in recent if s]),
                         unsafe_allow_html=True)

    scores = track_scores(d, dets["track"], attr, on_progress, on_result)
    live_ph.empty(); count_ph.empty(); grid_ph.empty()
    st.session_state["result"] = dict(attr=attr, scores=scores, p=detection_probs(dets["track"], scores), out=None,
                                      thr=threshold)
    st.rerun()

res = st.session_state.get("result")
if not res:
    st.stop()

# ---------------------------------------------------------------- 3) explore
p, scores = res["p"], res["scores"]
yes_n, no_n, unclear_n = people_summary(scores, threshold)
yes, tot = frame_counts(dets["frame"], p, meta["frames"], meta["stride"], threshold)
st.markdown(f'<div class="sec">3 · “{res["attr"]}”</div>', unsafe_allow_html=True)
st.markdown(ui.stats([("Yes", yes_n, "yes"), ("No", no_n, "no"), ("Too small to judge", unclear_n, "unc"),
                      ("Peak in one frame", int(yes.max()) if len(yes) else 0, "blue")]), unsafe_allow_html=True)
if res["thr"] != threshold:
    st.caption("Threshold changed since the video was rendered: counts are live, the video is not. Ask again to re-render.")

left, right = st.columns([3, 2])
with left:
    with st.container(border=True):
        st.markdown('<div class="sec">Annotated video</div>', unsafe_allow_html=True)
        vid_ph = st.empty()
        if res["out"] and Path(res["out"]).exists():
            vid_ph.video(res["out"])
        else:
            vid_ph.markdown(ui.skeleton(340), unsafe_allow_html=True)
with right:
    with st.container(border=True):
        st.markdown('<div class="sec">Yes count over time</div>', unsafe_allow_html=True)
        st.line_chart({"yes": yes, "all people": tot}, color=["#12b76a", "#0a3cff"])

with st.container(border=True):
    st.markdown('<div class="sec">Everyone, with the crop the model saw</div>', unsafe_allow_html=True)
    show = st.radio("Show", ["Yes", "No", "Too small", "All"], horizontal=True, label_visibility="collapsed")
    want = {"Yes": "yes", "No": "no", "Too small": "unc"}.get(show)
    allt = tiles_for(scores, threshold, d, limit=0)
    shown = [h for k, h in allt if want in (None, k)]
    st.markdown('<div class="grid">' + "".join(shown[:240]) + "</div>" if shown else "Nobody in this group.",
                unsafe_allow_html=True)
    if len(shown) > 240:
        st.caption(f"Showing 240 of {len(shown)}.")

with st.expander("Inspect a single frame"):
    fr = st.slider("Frame", 0, max(meta["frames"] - 1, 0), 0)
    cap = cv2.VideoCapture(play_src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    ok, img = cap.read()
    cap.release()
    if ok:
        m = dets["frame"] == fr - fr % meta["stride"]
        img = draw(img, dets["box"][m], dets["track"][m], p[m], threshold, f"frame {fr}: yes {yes[fr]} / {tot[fr]}")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# The video is rendered last: counts and faces above are already on screen, so the wait feels short.
if not (res["out"] and Path(res["out"]).exists()):
    t0 = time.time()
    ph = vid_ph
    out = render_video(play_src, dets, meta, p, res["attr"], d / "annotated.mp4", threshold,
                       lambda f: ph.markdown(ui.live("Drawing boxes on your video…", f, eta_text(t0, f)),
                                             unsafe_allow_html=True))
    res["out"] = str(out)
    ph.video(res["out"])
