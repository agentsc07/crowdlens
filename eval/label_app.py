"""Labelling page: one question at a time, a grid of every judgeable person's crop; tick the ones that are YES
(the default is NO). Labels are per tracked person, per question; they carry no identity.
Run: ~/.venvs/crowdlens/bin/python -m streamlit run eval/label_app.py"""
import json
from pathlib import Path

import streamlit as st

EVAL = Path.home() / ".crowdlens" / "eval"
LABELS = EVAL / "labels.json"
QUESTIONS = ["wearing a red top", "wearing a white shirt", "wearing a black jacket", "carrying a backpack",
             "carrying a handbag or shoulder bag", "wearing shorts", "wearing a hat or cap", "holding a shopping bag"]
COLS = 8
st.set_page_config(page_title="CrowdLens labelling", layout="wide")
people = [p for p in json.loads((EVAL / "people.json").read_text()) if p["judge"]]
labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}  # {question: {"clip:track": 0/1}} once saved
done = [q for q in QUESTIONS if q in labels]
st.title("Label people")
st.caption(f"{len(people)} people to judge. {len(done)}/{len(QUESTIONS)} questions done. Tick = YES. Leave unticked = NO. "
           "Skip people you truly can't judge by ticking 'unsure'.")
q = st.radio("Question", QUESTIONS, horizontal=True, format_func=lambda x: ("✅ " if x in labels else "") + x)
st.subheader(f"Who is {q}?")
prev = labels.get(q, {})
with st.form(f"form-{q}"):
    yes, unsure = {}, {}
    for i in range(0, len(people), COLS):
        cols = st.columns(COLS)
        for col, p in zip(cols, people[i:i + COLS]):
            k = f"{p['clip']}:{p['track']}"
            with col:
                st.image(str(EVAL / "idx" / p["clip"] / "crops" / f"{p['track']}.jpg"), width="stretch")
                yes[k] = st.checkbox("yes", value=prev.get(k) == 1, key=f"y-{q}-{k}")
                unsure[k] = st.checkbox("unsure", value=prev.get(k) == -1, key=f"u-{q}-{k}")
    if st.form_submit_button("Save this question", type="primary"):
        labels[q] = {k: (1 if yes[k] else -1 if unsure[k] else 0) for k in yes}
        LABELS.write_text(json.dumps(labels))
        st.success(f"Saved. {sum(v == 1 for v in labels[q].values())} yes of {len(people)}.")
        st.rerun()
