# CrowdLens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload a crowd video, ask a free-text attribute question, get a count with range plus an evidence grid, all on local open-source models.

**Architecture:** Index once (RF-DETR detect → ByteTrack → best crops → SigLIP 2 embeddings → FAISS), then per question: text-embed → top-k shortlist → closed-question verifier returning probabilities → multi-frame vote → count with range. Every stage is its own module and saves artifacts to `data/<video_id>/` so later stages re-run without reprocessing video. Verifier is a swappable interface with three tiers.

**Tech Stack:** Python 3.11, FastAPI-free (Streamlit calls modules directly; open question resolved as Streamlit for speed), PyTorch, `rfdetr`, `supervision` (ByteTrack), `transformers` (SigLIP 2, Qwen-VL), `faiss-cpu`, OpenCV, pytest.

**Spec:** `/Users/saicharan/Desktop/02_Coursework/DSAI/Projects/CrowdQ/spec.md`

## Global Constraints

- All models open source and local; no paid API.
- Privacy: count attributes, never identities. No face recognition, no cross-video linking. Track IDs are per-video and deleted with the index.
- Count error target under 10%; new question on indexed video under 30 s on laptop GPU/Colab.
- Verifier answers only from the fixed set `yes | no | unclear`, via next-token probabilities normalised to sum 1; no free text.
- Each person is checked on their 2–3 best crops; multi-frame vote on mean yes-probability vs threshold; strong disagreement → `unclear`.
- Headline count = confident yeses; range = confident yeses to confident + unclear (e.g. "23 (21–25)" style).
- Compound questions split into one check per attribute; person must pass all.
- Check for newer versions of each model when starting to build.

## Review Focus

- Video with zero people detected: index and query must return count 0 (0–0), not crash.
- Person with fewer than 2 usable crops: vote uses what exists; tiny crops go to `unclear`.
- Query when shortlist k exceeds number of people: clamp k.
- Compound question with one attribute unparseable/empty: reject with clear message, not silent count.
- Re-indexing the same video id: overwrite cleanly, no stale FAISS/crops.
- Correction flip on a person then recount: count and threshold update consistently; flipping twice restores.

---

## File Structure

```
crowdlens/
  config.py          # paths, thresholds, model names
  schema.py          # Person, Crop, Verdict dataclasses
  detect_track.py    # RF-DETR + ByteTrack -> tracks.json
  crops.py           # best-crop selection, crop saving
  embed.py           # SigLIP 2 image/text embeddings, FAISS index
  verifiers/
    base.py          # Verifier protocol, normalise()
    siglip_zs.py     # tier 1
    vlm.py           # tier 2 (default, Qwen-VL)
    classifier.py    # tier 3 (PA-100K fine-tuned, stub-free: loads trained head)
  vote.py            # multi-frame vote, count + range
  query.py           # parse question, shortlist, verify, compound, answer
  calibrate.py       # threshold calibration + correction tuning
  direction.py       # left-to-right counting from tracks
  timeline.py        # count per minute
  cost.py            # cost meter
  app.py             # Streamlit UI
eval/
  attributes.py  tracking.py  counting.py  shortlist.py
tests/
```

---

### Task 1: Scaffold, schema, config (Milestone 0)

**Files:** Create `pyproject.toml`, `crowdlens/__init__.py`, `config.py`, `schema.py`, `tests/test_schema.py`

**Interfaces:**
- Produces: `Crop(track_id:int, frame_idx:int, box:tuple[float,float,float,float], path:str, sharpness:float, visibility:float)`, `Person(track_id:int, first_frame:int, last_frame:int, crops:list[Crop], centroid_x:list[float])`, `Verdict(track_id:int, p_yes:float, label:str)` where label ∈ {"yes","no","unclear"}; `save_people(path, people)`, `load_people(path) -> list[Person]`.

- [ ] **Step 1: Failing test**

```python
from crowdlens.schema import Person, Crop, save_people, load_people

def test_roundtrip(tmp_path):
    c = Crop(1, 10, (0, 0, 50, 100), "a.jpg", 0.9, 1.0)
    p = Person(1, 10, 20, [c], [5.0, 9.0])
    f = tmp_path / "people.json"
    save_people(f, [p])
    assert load_people(f) == [p]
```

- [ ] **Step 2:** `pytest tests/test_schema.py -v` → FAIL (module missing)
- [ ] **Step 3: Implement** dataclasses with `asdict`/reconstruction in `schema.py`; `config.py` holds `DATA_DIR=Path("data")`, `MIN_CROP_H=64`, `TOP_CROPS=3`, `K_DEFAULT=200`, `THRESHOLD=0.5`, `UNCLEAR_SPREAD=0.4`, model-name constants.
- [ ] **Step 4:** run test → PASS
- [ ] **Step 5:** Commit `feat: scaffold and schema`

### Task 2: Detection + tracking (Milestone 1)

**Files:** Create `crowdlens/detect_track.py`, `crowdlens/crops.py`; Test `tests/test_crops.py`, `tests/test_detect_track.py`

**Interfaces:**
- Consumes: `Person`, `Crop`, `config`.
- Produces: `index_people(video_path, out_dir) -> list[Person]` (writes `people.json`, `crops/<id>_<n>.jpg`); `select_best(crops, n=TOP_CROPS) -> list[Crop]` ranking by `sharpness*visibility`, dropping boxes shorter than `MIN_CROP_H` unless none remain; `sharpness(img)` = variance of Laplacian.

- [ ] **Step 1: Failing tests** for `select_best`: given 5 crops with varied scores returns top 3 ordered; all-tiny crops still return the largest one; empty list returns `[]`.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: Implement** `select_best`, `sharpness`. `detect_track.py`: loop frames with OpenCV, RF-DETR `predict` filtered to class `person`, `sv.ByteTrack().update_with_detections`, accumulate per-track crops (store visibility as 1 − max IoU with other boxes in frame), after the loop keep `select_best` crops and write jpgs. Zero detections → return `[]` and write an empty `people.json`.
- [ ] **Step 4:** unit tests PASS; smoke: run on a 10 s public pedestrian clip, assert person count within 10% of hand count (record hand count in `tests/fixtures/README.md`).
- [ ] **Step 5:** Commit `feat: detection, tracking, best crops`

### Task 3: Embedding + shortlist search (Milestone 2)

**Files:** Create `crowdlens/embed.py`; Test `tests/test_embed.py`

**Interfaces:**
- Produces: `embed_images(paths)->np.ndarray (n,d, L2-normalised)`, `embed_text(q)->np.ndarray (d,)`, `build_index(people, out_dir)` (FAISS `IndexFlatIP` + `ids.json` mapping row→track_id; person vector = mean of its crop vectors, renormalised), `shortlist(out_dir, q_vec, k)->list[tuple[int,float]]` with `k=min(k,ntotal)`; empty index → `[]`.

- [ ] **Step 1: Failing tests** with fake vectors (no model): `shortlist` returns ranked ids, clamps k, empty index returns `[]`, rebuilding overwrites old index.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: Implement** with SigLIP 2 via `transformers` (`AutoModel`/`AutoProcessor`, model name from config); FAISS functions as specified; `build_index` deletes existing index files first.
- [ ] **Step 4:** tests PASS; manual check: "red shirt" top-10 mostly red shirts.
- [ ] **Step 5:** Commit `feat: siglip2 embeddings and faiss shortlist`

### Task 4: Verifier interface, voting, counting (Milestone 3)

**Files:** Create `verifiers/base.py`, `verifiers/siglip_zs.py`, `verifiers/vlm.py`, `vote.py`; Test `tests/test_vote.py`, `tests/test_verifier_base.py`

**Interfaces:**
- Produces: `Verifier.p_yes(crop_path:str, attribute:str)->float` (protocol); `normalise(logits_yes, logits_no, logits_unclear)->(py,pn,pu)` summing to 1 via softmax; `vote(probs:list[float], threshold, spread=UNCLEAR_SPREAD)->str` returns "yes" if mean ≥ threshold and max−min ≤ spread, "no" if mean < threshold and spread ok, else "unclear"; `count(verdicts)->(low:int, high:int)` = (#yes, #yes+#unclear).

- [ ] **Step 1: Failing tests**

```python
from crowdlens.vote import vote, count
from crowdlens.schema import Verdict

def test_vote_cases():
    assert vote([0.9, 0.8], 0.5) == "yes"
    assert vote([0.1, 0.2], 0.5) == "no"
    assert vote([0.95, 0.1], 0.5) == "unclear"   # strong disagreement
    assert vote([0.8], 0.5) == "yes"            # single crop
    assert vote([], 0.5) == "unclear"

def test_count_range():
    v = [Verdict(i, 0.9, "yes") for i in range(21)] + [Verdict(99+i, 0.5, "unclear") for i in range(4)]
    assert count(v) == (21, 25)

def test_count_empty():
    assert count([]) == (0, 0)
```

- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: Implement** `vote.py`, `normalise`. `siglip_zs.py`: softmax over "a person wearing X" vs "a person not wearing X" image-text logits. `vlm.py`: Qwen-VL prompt "Is this person {attribute}? Answer yes, no, or unclear.", single forward pass, take next-token logits for the first token of each of the three answers, `normalise`, return `p_yes`. Crops below `MIN_CROP_H` skipped by the caller (→ unclear).
- [ ] **Step 4:** unit tests PASS; one VLM smoke test marked `@pytest.mark.gpu`.
- [ ] **Step 5:** Commit `feat: verifiers, vote, count range`

### Task 5: Query pipeline incl. compound questions (Milestone 3/6)

**Files:** Create `crowdlens/query.py`; Test `tests/test_query.py`

**Interfaces:**
- Consumes: `shortlist`, `embed_text`, `Verifier`, `vote`, `count`.
- Produces: `split_attributes(question:str)->list[str]` (splits on " and "/","; strips filler like "how many people are"; raises `ValueError` if no attribute); `answer(video_dir, question, verifier, k=K_DEFAULT, threshold=THRESHOLD)->Answer(low, high, verdicts, n_verified, n_total, seconds)`. Person is "yes" only if yes on all attributes; "no" if any no; else "unclear".

- [ ] **Step 1: Failing tests** with a fake verifier: `split_attributes("how many people in red and with a cap")==["in red","with a cap"]`; empty question raises; compound all-must-pass; zero-person index → `(0,0)`; `n_verified <= k`.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: Implement.** Embed the full question for shortlisting; verify only shortlisted people, using their crops (skip tiny → unclear).
- [ ] **Step 4:** PASS
- [ ] **Step 5:** Commit `feat: query pipeline`

### Task 6: Calibration and correct-and-recount (Milestone 6)

**Files:** Create `crowdlens/calibrate.py`; Test `tests/test_calibrate.py`

**Interfaces:**
- Produces: `calibrate_threshold(p:list[float], y:list[int])->float` (threshold maximising F1 over a grid 0.05–0.95); `apply_corrections(verdicts, flips:set[int])->list[Verdict]` toggling yes↔no for flipped ids (unclear → yes); `tune_threshold(prev, p, flips)`; flipping twice restores original.

- [ ] **Step 1: Failing tests** for separable data, flip toggle, double flip identity, recount via `count`.
- [ ] **Step 2:** FAIL → **Step 3:** implement → **Step 4:** PASS → **Step 5:** Commit `feat: calibration and corrections`

### Task 7: Direction, timeline, cost meter (Milestone 6)

**Files:** Create `direction.py`, `timeline.py`, `cost.py`; Test `tests/test_extras.py`

**Interfaces:**
- Produces: `left_to_right(people, min_dx=50)->list[int]` using `centroid_x[-1]-centroid_x[0]`; `per_minute(people, verdicts, fps)->dict[int,int]` counting matching people by minute of `first_frame`... counted in each minute they are visible (first–last frame span); `cost(n_total, n_verified, sec_per_verify)->{"embedded","verified","saved_calls","saved_seconds"}`.

- [ ] Steps 1–5 as above with tests: a person moving +100px counts, −100 doesn't, stationary doesn't; person spanning minutes 0–1 appears in both buckets; cost with `n_verified==n_total` saves 0.
- [ ] Commit `feat: direction, timeline, cost`

### Task 8: Streamlit app (Milestone 4/6)

**Files:** Create `crowdlens/app.py`

- [ ] Upload + index with progress bar; question box; verifier selector (SigLIP/VLM/fine-tuned); result "N (low–high)"; evidence grid of thumbnails with p_yes, each button seeking `st.video(start_time=first_frame/fps)`; checkbox per thumbnail to flip → recount via Task 6; count-over-time `st.line_chart`; cost meter; question history cached in `data/<id>/history.json`; overlay playback by rendering matched boxes with OpenCV to an mp4 (needs per-frame boxes saved in `tracks.json`). Add per-person boxes to `Person` if not already there.
- [ ] Verify by running `streamlit run crowdlens/app.py` on the demo clip and walking the demo script in the spec.
- [ ] Commit `feat: streamlit app`

### Task 9: Evaluation (Milestone 5)

**Files:** Create `eval/attributes.py`, `tracking.py`, `counting.py`, `shortlist.py`

- [ ] Attribute accuracy/F1 per tier on PA-100K subset; tracking ID switches on MOT17; MAE and "true count inside range" on 3–5 hand-counted clips × 5 questions; shortlist recall at several k vs verifier calls (headline chart: >95% recall with large call reduction); seconds per question. Each script writes CSV to `eval/results/`; unit-test the metric helpers (`mae`, `in_range_rate`, `recall_at_k`) with tiny inputs first.
- [ ] Commit `feat: evaluation scripts`

---

## Self-Review

- Spec coverage: detection/tracking (T2), embedding+FAISS (T3), 3 verifier tiers (T4; tier 3 classifier needs a trained PA-100K head — add training script in T9 before use), vote/range (T4), compound (T5), calibration/corrections (T6), direction/timeline/cost (T7), UI incl. overlay/history/switch (T8), evaluation (T9). Privacy: no face/identity code anywhere; track IDs live only in the per-video folder, deleted with it.
- Open question resolved: Streamlit. Demo clip choice still open (public pedestrian dataset vs consented footage).
