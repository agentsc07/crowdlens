"""3-minute audit of the Claude-made labels: 30 random people, each shown with the attributes Claude assigned.
Tick 'wrong' if any of them is wrong or missing. Saves ~/.crowdlens/eval/audit.json.
Run: ~/.venvs/crowdlens/bin/python -m streamlit run eval/audit_app.py"""
import json
import random
from pathlib import Path

import streamlit as st

EVAL = Path.home() / ".crowdlens" / "eval"
AUDIT = EVAL / "audit.json"
st.set_page_config(page_title="Audit labels", layout="wide")
people = [p for p in json.loads((EVAL / "people.json").read_text()) if p["judge"]]
labels = json.loads((EVAL / "labels_claude.json").read_text())
keys = [f"{p['clip']}:{p['track']}" for p in people]
skip = {k for k in keys if all(labels[q][k] == -1 for q in labels)}
pick = random.Random(7).sample([k for k in keys if k not in skip], 30)
saved = json.loads(AUDIT.read_text()) if AUDIT.exists() else {}
st.title("Audit Claude's labels")
st.caption("Each person is shown with the attributes Claude gave them. Tick WRONG if any attribute is wrong or missing "
           "(the 8 attributes: red top, white shirt, black jacket, backpack, handbag/shoulder bag, shorts, hat/cap, shopping bag).")
with st.form("audit"):
    wrong = {}
    for i in range(0, len(pick), 6):
        for col, k in zip(st.columns(6), pick[i:i + 6]):
            clip, tid = k.split(":")
            yes = [q.split(" ", 1)[1] for q in labels if labels[q][k] == 1]
            unsure = [q.split(" ", 1)[1] for q in labels if labels[q][k] == -1]
            with col:
                st.image(str(EVAL / "idx" / clip / "crops" / f"{tid}.jpg"), width="stretch")
                st.markdown("**" + (", ".join(yes) or "none of the 8") + "**" + (f"  \n_unsure: {', '.join(unsure)}_" if unsure else ""))
                wrong[k] = st.checkbox("wrong", value=saved.get(k, False), key=f"w-{k}")
    if st.form_submit_button("Save", type="primary"):
        AUDIT.write_text(json.dumps(wrong))
        st.success(f"Saved. Claude's labels judged correct for {sum(not v for v in wrong.values())} of {len(wrong)} people.")
