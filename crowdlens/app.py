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
from crowdlens.motion import motion_scores
from crowdlens.render import draw, render_video
from crowdlens.score import detection_probs, frame_counts, parse_question, people_summary, track_scores

st.set_page_config(page_title="CrowdLens", page_icon="🔍", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

ATTRIBUTION = ("Sample clips: first 15 s of MOT17 sequences 02, 04 and 11 (Milan et al., 2016, "
               "[motchallenge.net](https://motchallenge.net/data/MOT17/)), licensed CC BY-NC-SA 3.0: non-commercial use, "
               "attribution, share-alike. CrowdLens counts attributes only, never identities; track numbers exist for one video only.")


def footer():
    st.markdown("---")
    st.caption(ATTRIBUTION)


STAGES = ["1 · Video", "2 · Ask", "3 · Explore"]
THINK_INDEX = ["Scanning every corner of the frame", "Zooming in on distant objects", "Following them between frames",
               "Picking each one's sharpest view"]
CHIPS = {
    "person": ["wearing a hat", "wearing glasses", "wearing red", "carrying a bag", "walking toward the camera",
               "standing still"],
}
DEFAULT_CHIPS = ["red", "white", "moving", "standing still"]
PLURAL = {"person": "people", "bus": "buses", "bicycle": "bicycles"}


def plural(cat):
    return PLURAL.get(cat, cat + "s")


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
    """[(label, vid, category, original video path)] for everything already indexed."""
    out = []
    for m in sorted(config.DATA_DIR.glob("*/meta.json")) if config.DATA_DIR.exists() else []:
        if m.parent.name.endswith(".tmp") or not any((m.parent / "crops").glob("*.jpg")):
            continue  # unfinished or damaged index
        vid, _, cat = m.parent.name.partition("__")
        vids = [f for f in config.DATA_DIR.glob(f"{vid}.*") if f.suffix.lower() in (".mp4", ".mov", ".avi")]
        if vids:
            out.append((f"{vid} · {plural(cat or 'person')}", vid, cat or "person", str(vids[0])))
    return out


def tiles_for(scores, threshold, d, numbers=True):
    out = []
    for t in sorted(scores, key=lambda t: -np.nan_to_num(scores[t], nan=-1)):
        v = scores[t]
        kind = "unc" if np.isnan(v) else "yes" if v >= threshold else "no"
        src = ui.thumb(d / "crops" / f"{t}.jpg")
        if src:
            label = "too short" if kind == "unc" else "Yes" if kind == "yes" else "No"
            if numbers and kind != "unc":
                label += f" {v:.2f}"
            out.append((kind, ui.tile(src, kind, label)))
    return out


@st.cache_resource
def _jobs():
    return {}


@st.fragment(run_every=0.7)
def show_index_job(job):
    """Live view of the background indexing thread. The job survives page reruns, so a click or refresh
    while indexing no longer destroys the index."""
    if job["done"]:
        ss["seen_job"] = id(job)
        if job["error"]:
            st.error(f"Indexing failed: {job['error']}")
            return
        ss.update(vid=job["vid"], category=job["cat"], video=job["video"], result=None)
        go_to(STAGES[1])
        return
    phrase = THINK_INDEX[(job["n"] // 6) % len(THINK_INDEX)]
    st.markdown(ui.live(f"{phrase}…" if job["info"] else job["msg"], job["f"], eta_text(job["t0"], job["f"])),
                unsafe_allow_html=True)
    pl, pr = st.columns([3, 2])
    if job["info"]:
        pl.image(job["info"]["img"], use_container_width=True)
        pr.markdown(ui.stats([(f"{plural(job['cat']).capitalize()} in this frame", job["info"]["people"], "blue"),
                              ("Different ones so far", job["info"]["tracks"], "")]), unsafe_allow_html=True)
    else:
        pl.markdown(ui.skeleton(380), unsafe_allow_html=True)


def go_to(stage):
    # the nav widget is already drawn in this run, so the change is applied at the top of the next one
    st.session_state["_goto"] = stage
    st.rerun()


# ---------------------------------------------------------------- state + header
ss = st.session_state
if "_goto" in ss:
    ss["stage"] = ss.pop("_goto")
ss.setdefault("stage", STAGES[0])
vid, category = ss.get("vid"), ss.get("category", "person")
res = ss.get("result")

st.markdown(ui.hero(), unsafe_allow_html=True)
nav, cfg = st.columns([5, 1])
with nav:
    st.segmented_control("Stage", STAGES, key="stage", label_visibility="collapsed")
with cfg:
    with st.popover("Settings", use_container_width=True):
        stride = st.select_slider("Look at every Nth frame (higher = faster)", [1, 2, 3, 4, 6, 8], config.DEFAULT_STRIDE)
        threshold = st.slider("Yes threshold", 0.1, 0.95, config.THRESHOLD, 0.05)
stage = ss.get("stage") or STAGES[0]
if stage != STAGES[0] and not vid:
    stage = STAGES[0]
    st.info("Start by choosing a video.")
if stage == STAGES[2] and not res:
    stage = STAGES[1]
    st.info("Ask a question first.")

# ---------------------------------------------------------------- 1) video
if stage == STAGES[0]:
    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="sec">Choose a video</div>', unsafe_allow_html=True)
            up = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], label_visibility="collapsed")
            existing = indexed_videos()
            if existing:
                pick = st.selectbox("…or reopen one you've already indexed", [None] + existing,
                                    format_func=lambda x: "Select…" if x is None else x[0])
                if pick and st.button("Open", key="open"):
                    ss.update(vid=pick[1], category=pick[2], video=pick[3], result=None)
                    go_to(STAGES[1])
    with right:
        with st.container(border=True):
            st.markdown('<div class="sec">What should it look for?</div>', unsafe_allow_html=True)
            cat = st.selectbox("Object type", config.CATEGORIES, format_func=lambda c: plural(c).capitalize(),
                               label_visibility="collapsed")
            st.caption("Everything of this type is found and followed through the video; then you ask about it.")
            go_index = bool(up) and st.button("Index video", type="primary", use_container_width=True)
    job = _jobs().get("index")
    if go_index and not (job and not job["done"]):
        prewarm()
        name = Path(up.name)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        video = config.DATA_DIR / f"{name.stem}{name.suffix}"
        video.write_bytes(up.getvalue())
        cap = cv2.VideoCapture(str(video))
        secs = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.release()
        if config.MAX_UPLOAD_SECONDS and secs > config.MAX_UPLOAD_SECONDS:
            st.error(f"This hosted demo takes clips up to 15 s on a free CPU; yours is {secs:.0f} s. "
                     "Trim it, or run CrowdLens locally for longer videos.")
            footer()
            st.stop()
        job = dict(done=False, error=None, f=0.0, msg="Starting…", info=None, t0=time.time(), cat=cat, vid=name.stem,
                   video=str(video), n=0)
        _jobs()["index"] = job

        def work():
            def cb(f, msg="", info=None):
                job.update(f=f, msg=msg, info=info or job["info"], n=job["n"] + 1)
            try:
                index_video(video, config.video_dir(name.stem, cat), stride, cb, cat)
            except Exception as e:  # shown in the UI; the previous index (if any) is untouched
                job["error"] = repr(e)
            job["done"] = True

        threading.Thread(target=work, daemon=True).start()
    if job and not (job["done"] and ss.get("seen_job") == id(job)):
        show_index_job(job)
    footer()
    st.stop()

d = config.video_dir(vid, category)
if not (d / "meta.json").exists() or not any((d / "crops").glob("*.jpg")):
    st.error("This video's index is incomplete (it was probably interrupted). Index it again.")
    ss["vid"] = None
    st.stop()
dets, meta = load(d)
n_found = len(set(dets["track"].tolist()))
prewarm()
play_src = str(d / "proxy.mp4") if meta.get("proxy") and (d / "proxy.mp4").exists() else ss["video"]
dur = meta["frames"] / meta["fps"]

# ---------------------------------------------------------------- 2) ask
if stage == STAGES[1]:
    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(f'<div class="sec">Ask about the {plural(category)}<small>{n_found} found in {dur:.0f}s of '
                        f'video</small></div>', unsafe_allow_html=True)
            with st.form("ask", border=False):
                c1, c2 = st.columns([5, 1])
                typed = c1.text_input("Question", value=ss.get("last_q", ""), label_visibility="collapsed",
                                      placeholder="e.g. wearing a hat · went to the right · standing still")
                submitted = c2.form_submit_button("Ask", type="primary", use_container_width=True)
            if submitted:
                ss["last_q"] = typed
            chips = CHIPS.get(category, DEFAULT_CHIPS)
            for col, s_ in zip(st.columns(len(chips)), chips):
                if col.button(s_, key=f"chip-{s_}", use_container_width=True):
                    ss["last_q"], ss["go"] = s_, True
                    st.rerun()
            st.checkbox("Look at several moments of each one (slower, better for actions like “using a phone”)",
                        key="multi_override", value=False)
    with right:
        st.markdown("""<div class="glass"><div class="sec">What you can ask</div>
<p class="hint"><b>Appearance</b> · “wearing a hat”, “red”, “carrying a bag”<br>Answered by an AI model that looks at each one.</p>
<p class="hint"><b>Movement</b> · “went to the right”, “walking toward the camera”, “standing still”<br>
Measured from how each one moves, corrected for camera shake. Instant.</p>
<p class="hint"><b>Action</b> · “using a phone”, “talking”<br>The AI model sees a few moments of each one.</p></div>""",
                    unsafe_allow_html=True)

    go = submitted or ss.pop("go", False)
    if go and ss.get("last_q"):
        try:
            route = parse_question(ss["last_q"])
        except ValueError as e:
            st.error(str(e))
            st.stop()
        multi = route.get("multi", False) or ss.get("multi_override", False)
        if route["kind"] == "motion":
            if "box_ref" not in dets:
                st.error("This video was indexed before movement questions existed. Index it again to enable them.")
                st.stop()
            with st.spinner("Measuring how everything moves…"):
                scores = motion_scores(dets, route["motion"], meta["fps"])
            mode = "motion"
        else:
            live_ph, count_ph, grid_ph = st.empty(), st.empty(), st.empty()
            attr, seen, total, t0 = route["attribute"], {}, max(n_found, 1), time.time()
            live_ph.markdown(ui.live("Waking up the vision model…", 0.0), unsafe_allow_html=True)

            def on_result(t, v):
                seen[t] = v

            def on_progress(f):
                yes_now = sum(1 for v in seen.values() if not np.isnan(v) and v >= threshold)
                phrase = ["Looking closely at each one", f"Checking for “{attr}”", "Counting as we go"][int(f * 9) % 3]
                live_ph.markdown(ui.live(f"{phrase}… {len(seen)}/{total}", f, eta_text(t0, f)), unsafe_allow_html=True)
                count_ph.markdown(ui.stats([("Yes so far", yes_now, "yes"), ("Checked", len(seen), "blue")]),
                                  unsafe_allow_html=True)
                recent = [("unc" if np.isnan(v) else "yes" if v >= threshold else "no",
                           ui.thumb(d / "crops" / f"{t}.jpg")) for t, v in list(seen.items())[-60:][::-1]]
                grid_ph.markdown(ui.grid([ui.tile(s, k, {"yes": "Yes", "no": "No", "unc": "?"}[k])
                                          for k, s in recent if s]), unsafe_allow_html=True)

            only = None
            if config.SHORTLIST and not multi:  # hosted CPU demo: verify only the SigLIP2 shortlist (search-then-verify)
                from crowdlens import search
                only = set(search.shortlist(search.similarity(d, attr, category), config.SHORTLIST))
            scores = track_scores(d, dets["track"], attr, on_progress, on_result, subject=category, multi=multi, only=only)
            live_ph.empty(); count_ph.empty(); grid_ph.empty()
            mode = "multi" if multi else "attr"
        ss["result"] = dict(label=route["label"], mode=mode, scores=scores, out=None, thr=threshold,
                            p=detection_probs(dets["track"], scores))
        go_to(STAGES[2])
    footer()
    st.stop()

# ---------------------------------------------------------------- 3) explore
p, scores = res["p"], res["scores"]
yes_n, no_n, unclear_n = people_summary(scores, threshold)
yes, tot = frame_counts(dets["frame"], p, meta["frames"], meta["stride"], threshold)
how = {"motion": "Movement · measured from tracks, no AI model needed", "attr": "Appearance · AI model, one view each",
       "multi": "Action · AI model, several moments each"}[res["mode"]]
st.markdown(f'<div class="sec">{plural(category).capitalize()} · “{res["label"]}”<small>{how}</small></div>',
            unsafe_allow_html=True)
st.markdown(ui.stats([("Yes", yes_n, "yes"), ("No", no_n, "no"),
                      ("Too short to judge" if res["mode"] == "motion" else "Too small to judge", unclear_n, "unc"),
                      ("Peak in one frame", int(yes.max()) if len(yes) else 0, "blue")]), unsafe_allow_html=True)
if res["mode"] == "motion" and meta.get("cam_speed", 0) > 0.02:
    st.caption("The camera moved during this clip. Movement is corrected for it, but treat direction counts as approximate.")
if res["thr"] != threshold and res["mode"] != "motion":
    st.caption("Threshold changed since the video was rendered: counts are live, the video is not. Ask again to re-render.")

left, right = st.columns([3, 2], gap="large")
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
        t_people, t_time, t_frame = st.tabs(["Everyone", "Over time", "Frame"])
        with t_people:
            show = st.radio("Show", ["Yes", "No", "Too small", "All"], horizontal=True, label_visibility="collapsed")
            want = {"Yes": "yes", "No": "no", "Too small": "unc"}.get(show)
            shown = [h for k, h in tiles_for(scores, threshold, d, numbers=res["mode"] != "motion") if want in (None, k)]
            st.markdown('<div class="grid">' + "".join(shown[:240]) + "</div>" if shown else "Nobody in this group.",
                        unsafe_allow_html=True)
            if len(shown) > 240:
                st.caption(f"Showing 240 of {len(shown)}.")
        with t_time:
            st.line_chart({"yes": yes, "all": tot}, color=["#12b76a", "#0a3cff"])
        with t_frame:
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
    out = render_video(play_src, dets, meta, p, res["label"], d / "annotated.mp4", threshold,
                       lambda f: vid_ph.markdown(ui.live("Drawing boxes on your video…", f, eta_text(t0, f)),
                                                 unsafe_allow_html=True))
    res["out"] = str(out)
    vid_ph.video(res["out"])
footer()
