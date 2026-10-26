# SUBMISSION_LOG

Working log for `SUBMISSION_TASK.md`. Decisions are dated 2026-10-26 unless noted.

## Task list
- [x] Step 0: read README/spec/code, run tests, log state and gaps
- [ ] Step 1: evaluation set (3-4 public clips) + labelling page (<30 min for the user)
- [ ] Step 2: metrics (P/R/F1, count error %, time per question), threshold chosen on a held-out clip
- [ ] Step 3: search-then-verify with SigLIP2 shortlist, accuracy vs time vs verify-everyone
- [ ] Step 4: Hugging Face Space (Streamlit, CPU), 2-3 pre-indexed samples + upload up to ~15 s
- [ ] Step 5: `pip install crowdlens` + CLI (`index`, `ask`); ask before PyPI
- [ ] Step 6: README (grader), docs/report.md, docs/demo-script.md, privacy section, links at top
- [ ] GitHub repo under agentsc07 (private first; ask before public)
- [ ] startup.md (global CLAUDE.md rule for runnable projects)

## Step 0: state of the project
- Branch `main`, last commit `aa0a090`. No git remote yet. `gh` is logged in as `agentsc07`.
- Tests: `pytest` gives **10 passed** (parse/route logic, NaN handling, caching with a fake VLM, motion classify). Nothing tests accuracy.
- Env: macOS arm64 (MPS), Python venv at `~/.venvs/crowdlens`, data at `~/.crowdlens/data`. Local clips: `demo10s` (10 s, 167 person tracks), plus a 4K clip.
- Pipeline in code: RF-DETR Nano with tiling -> ByteTrack -> best crop(s) per track -> Qwen3-VL-2B yes/no from next-token logits -> annotated video, per-frame chart, evidence grid. Extras beyond the spec: object categories, motion questions from tracks with camera stabilisation, multi-moment action mode.
- Hygiene fixed in this step: `.gitignore` lacked `.DS_Store`, `.env`, `*.pem`, keys, build output. Added them. Created `_holding/` for anything unwanted (rule: never delete).

## Spec vs code gaps
| Spec promise | Code today |
|---|---|
| Search stage: SigLIP 2 embeddings, FAISS shortlist, search-then-verify | **Missing.** Every tracked person is verified. No embedding code, no FAISS. |
| Cost meter (people embedded vs verified, time saved) | Missing |
| Count error < 10% on hand-labelled clips | **No labels, no measurement at all** |
| Answer in < 30 s on an indexed video | Unmeasured; one run took 61 s for 167 people (glasses) and 135 s in multi-moment mode, on a loaded machine |
| Verifier answers yes / no / unclear (3 options) | Binary yes/no; "unclear" only means crop < 64 px |
| Multi-frame vote on 2-3 best crops, disagreement = unclear | Only in the optional multi-moment mode, which shows the VLM all crops at once; no per-crop vote |
| Count with a range, e.g. 23 (21-25) | Single count only |
| Threshold calibrated on a labelled set | Fixed 0.5 |
| Compound questions ("red and cap") as one check per attribute | Not supported; one route per sentence (see router notes) |
| Correct and recount (click to flip, tunes threshold) | Missing |
| Verifier switch (SigLIP / VLM / fine-tuned) | Missing |
| Tracking quality vs MOT17 (ID switches) | Not measured. 167 tracks on the 10 s clip is suspiciously many, so likely over-fragmented |
| Question history | Partial: answers cached per question in `answers.json` |
| Click thumbnail to jump to that moment | Frame inspector exists; check this is wired |
| Deliverables | README is 11 lines with no results or links; no docs/report.md, docs/demo-script.md, startup.md, CLI, package entry point or license |

## Known weak spots to keep in mind
- Router is regex based: typos and mixed movement + appearance questions fail (explained to the user earlier). Not a submission blocker; compound-question support is a cheap win if time allows.
- "Using a phone" yes rate looked too high (23/167) before any labels existed. The eval should settle it.
- Movement thresholds are unvalidated.

## Decisions
- D1: Eval and search-then-verify will run on **tracked-person crops**, matching how the app works. Labels are per person (track), not per frame.
- D2: Labelling effort budget: ~8 questions x ~150 people is too many clicks, so the labelling page will show a **grid with one question at a time** (click the "yes" ones, default is no) rather than 8 buttons per person. Aim for under 30 minutes.
