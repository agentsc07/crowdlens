# Submission task: CrowdLens (side project)

Context: CrowdLens answers plain-English attribute questions about people in crowd video (RF-DETR + ByteTrack index once; Qwen3-VL-2B yes/no with P(yes) per tracked person; annotated video). See README.md and spec.md. The spec promised count error < 10% and a search-then-verify speed-up, but the code dropped the embedding search step and there are **no accuracy numbers**. Fix that, then deploy.

## Steps
0. Read README.md, spec.md, crowdlens/*. Run the tests. Log the state and the spec-vs-code gaps in `SUBMISSION_LOG.md`.
1. **Evaluation set.** Pick 3-4 public, licence-clear crowd clips (e.g. MOT17/MOT20 or CC-licensed footage). Build a tiny labelling page that shows each tracked person's crop and asks me yes/no for ~8 questions (red top, backpack, cap, phone, umbrella, etc.). Make labelling take me under 30 minutes.
2. **Metrics:** per-person precision/recall/F1, count error % per question, time per question. Compare thresholds; pick one on a held-out clip.
3. **Bring back search-then-verify** as an option: SigLIP2 embeddings per person, shortlist by similarity, verify only the shortlist with Qwen3-VL. Report accuracy vs time against verify-everyone. This is the project's main claim, so it needs the numbers.
4. **Deploy** to a free Hugging Face Space (Streamlit, CPU): 2-3 pre-indexed sample videos with cached answers, plus upload of clips up to ~15 s. Make sure CPU inference time is acceptable; downscale if needed.
5. **Package:** `pip install crowdlens` with a CLI (`crowdlens index video.mp4`, `crowdlens ask video.mp4 "wearing a red shirt?"`). Ask before publishing to PyPI.
6. Deliverables below, plus a privacy section (attributes only, no identities).

## Rules
- Never delete files. Move anything unwanted into `_holding/` inside this repo.
- Never commit secrets (`*.pem`, `.env`, keys). Check `git status` and `.gitignore` before every commit.
- Push to `github.com/agentsc07/<repo>` with `gh`. Ask me before making the repo public or publishing a package.
- Ask me only when blocked on something only I can do (hardware, accounts, labelling, listening). Otherwise decide, and note the decision in `SUBMISSION_LOG.md`.
- Keep a task list. Commit small and often.

## Deliverables (every project)
1. `README.md` rewritten for a grader: one-line pitch, the problem, how it works (diagram), **results with numbers**, live links, how to run.
2. `docs/report.md`: one page. Problem, method, evaluation, limitations, next steps.
3. `docs/demo-script.md`: a 2-minute screen-recording script, shot by shot.
4. All deployed links working and listed at the top of the README.
