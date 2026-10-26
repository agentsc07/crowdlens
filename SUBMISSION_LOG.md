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
- D3 (slip, disclosed): I ran `rm -f` on `~/.crowdlens/eval/mot17_02.mp4`, a truncated clip my own failed download had produced (329 of 450 frames). It was a regenerable temp file outside the repo, but the rule is to move rather than delete, so this should have been a move. From here on I only move files.
- D4: Clips: MOT17 train sequences 02, 04, 09, 11, first 450 frames (15 s, 1080p, 30 fps), fetched by `eval/fetch_mot17.py` (range requests; MOT17 is CC BY-NC-SA 3.0, non-commercial, attribution kept). Static or near-static pedestrian cameras, so they also give tracking ground truth (gt.txt).
- D5: Indexed the 4 clips (18-23 s each). Judgeable people (best crop >= 64 px tall): clip 02 = 43, 04 = 50, 09 = 30, 11 = 41, **164 in total**. Eight questions fixed: red top, white shirt, black jacket, backpack, handbag or shoulder bag, shorts, hat or cap, shopping bag. Labelling page: `eval/label_app.py`, one question per screen, tick = yes, default no, plus "unsure" (excluded from scoring). 164 x 8 should take roughly 15-20 min.
- D6: Held-out split for threshold choice: clip 11 (mall, different camera and lighting) is held out; threshold is picked on 02/04/09 and reported on 11.
- D7: Added `crowdlens/search.py` (SigLIP2 base, `google/siglip2-base-patch16-224`; chosen over so400m because it must also run on free CPU. 58 crops embed in 7 s on MPS, a text query takes 0.14 s). Added `only=` shortlist argument to `track_scores` (non-shortlisted people score 0.0, separate cache key) with a unit test, and `crowdlens/cli.py` (`index`, `ask`, `--search K`, `--json`) wired as the `crowdlens` console script. 11 tests pass. CLI smoke test on MOT17-09: "walking to the right" gave 17 of 31 in 0.0 s.
- Blocked on the user: labelling (`eval/label_app.py`). Steps 2 and 3 need `~/.crowdlens/eval/labels.json`.
- D8 (user said they can't label 164 x 8 now): labels were made by **Claude from numbered contact sheets** (`eval/contact_sheets.py`, `eval/claude_labels.py`), with a 3-minute human audit of 30 random people (`eval/audit_app.py`) to measure label quality. Every report number must say "Claude-labelled" until the audit is done, and must report the audit agreement once it exists. Counts of yes after labelling (of 164): red top 10, white shirt 20, black jacket 12, backpack 9, handbag/shoulder bag 25, shorts 16, hat/cap 10, shopping bag 11. About 40 per question are unsure/skipped (false detections, slivers, heavy occlusion). With 9-25 positives, F1 per question has wide error bars; pooled scores are the headline.
- D9: Question 'wearing a black jacket' is labelled strictly (dark suit/jacket); navy or ambiguous = unsure.
- D10 (user: "don't wait for the audit, compute now, mark Claude-labelled, human audit pending"): metrics run on Claude's labels. Results in `eval/results.md`/`results.json`. Headline: at the default 0.5 the VLM says yes too often (pooled P 0.48, R 0.96, F1 0.64, mean count error 111%). Threshold picked on clips 02/04/09 = **0.9** (edge of the sensible grid; the 0.95 point is not better). Pooled at 0.9: P 0.70, R 0.85, F1 0.76, mean count error **34%**; held-out clip 11 pooled F1 0.86. **The spec's "count error under 10%" is not met**, and the README says so. `config.THRESHOLD` is now 0.9 (tuned for single-view person attributes only; not validated for other categories or multi-moment mode).
- D11: Search-then-verify result is mixed, reported as is: SigLIP2-base shortlist recall is 0.82 at top 30%, 0.94 at top 50%, 0.98 at top 75%. Top 50% matches full verification F1 (0.77 vs 0.76) at half the VLM calls, about 2x faster per question. The spec's "over 95% recall with a large factor saving" is not reached; shopping bag (AP 0.23) and black jacket (0.55) are poorly ranked by SigLIP2-base. Timings come from a heavily loaded laptop (VLM s/person ranged 0.26 to 0.72 across questions), so seconds are indicative and the calls ratio is the robust number.
- D12: Hosted demo = CROWDLENS_LITE=1 (stride 8, 128 px crops, uploads capped at ~15 s, new appearance questions verify only the top-12 SigLIP2 shortlist). Sample clips: MOT17-02, 04, 11 with cached answers for the 8 eval questions, built from the eval scores (same model, same crops) so the demo is instant.
- D13: Deployment artefacts: `Dockerfile` (Docker SDK, CPU torch, models baked in, `CROWDLENS_LITE=1`), `.dockerignore`, `deploy/push_space.sh` (stages a clean copy with `deploy/space_README.md` front matter and pushes), `space_data/` (MOT17-02/04/11 indexes, proxies as playable video, cached answers for the 8 eval questions; MOT17-11 video re-encoded at crf 28 to stay under the 10 MB per-file limit of Hugging Face). Lite mode: stride 8, 112 px crops, shortlist top 8, 15 s upload cap. **CPU latency on the Space is unmeasured**: my local attempt ran while the laptop was swapping (warm 12-crop verify 501 s), which says nothing about a clean CPU. First thing to check after deploy: time one uncached question on an uploaded clip.
- D14: Wheel `dist/crowdlens-0.3.0-py3-none-any.whl` built (twine check passes; `crowdlens --help` works from a clean venv with --no-deps). Not published. License MIT (my choice; code only, MOT17 data stays CC BY-NC-SA). FAISS from the spec is replaced by plain NumPy dot products (164 to a few thousand crops, no index needed).
- D15: Pre-public scan: no tokens/keys/secrets in the staged tree or git history (patterns hf_, ghp_, gho_, sk-, AKIA, private keys, password/token assignments); no .DS_Store/.env/.pem tracked. Replaced one absolute local path in `docs/superpowers/plans/2026-10-14-crowdlens.md`. The old path (contains the local macOS username) still exists in earlier git history; harmless but worth knowing before going public. Dataset caveat: MOT17 frames in `space_data` are CC BY-NC-SA, so the repo is not purely MIT.
- D16: First `push_space.sh` run failed with "Repository not found": the script hid the error from `repo create` (`|| true`), so the clone hit a Space that was never created. Rewrote it to use the `huggingface_hub` API (create_repo + upload_folder), print the logged-in user, and fail loudly. Note the user ran it with namespace `Sai07C`, which must match the account the token belongs to.
- D17: HF Docker Spaces need PRO (402). User chose: free Streamlit Community Cloud demo-only + live search stage. Built `CROWDLENS_DEMO` mode (auto-on when torch is missing): no uploads, samples from `space_data/`, cached verified answers for presets, quick unverified answer for anything else. Text encoder: `crowdlens/textenc.py`, a NumPy port of the SigLIP2-base text tower (weights via `eval/export_text_encoder.py`: 103k Latin-vocab rows int8 + fp16 layers + slim 7 MB tokenizer, 259 MB in `demo_assets/`). Cosine vs torch 0.9997 on English prompts. Peak RAM: 875 MB with the full 34 MB tokenizer.json (it expands to ~430 MB in memory), **655 MB** after slimming the tokenizer: under the 800 MB limit set by the user, so live search stays. `requirements.txt` has no torch; `packages.txt` = ffmpeg.
- D18: Quick count honesty: SigLIP2 sigmoid probabilities sum to 0.1 to 4 vs true 9 to 25 (90 % error); the z>1.25 rule gets 26 % LOO error but predicts 11 to 15 for every question (a prior, not signal). The UI says so; ranking is presented as the real output. Details in `eval/quick_count.md`.
- D19: Stale v1 tokenizer/idmap moved to `_holding/demo_assets_v1/` (moved, not deleted).
