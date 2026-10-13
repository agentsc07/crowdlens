# CrowdLens — Ask Questions of Crowd Video

Oct 14, 2026 · @Sai Charan

## Overview

CrowdLens lets you upload a crowd video and ask plain-English questions such as "how many people are wearing red shirts?"; it answers with a count, a confidence range, and a thumbnail of every person it counted. Everything runs locally on open-source models.

**The problem.** Counting people with a given attribute in video today means either watching it by hand or training a custom classifier for every attribute. Neither handles a new question asked on the spot.

**Why two models.** The work splits into two halves with opposite needs:

- **Search (embedding model):** every person is embedded once; any new question becomes a fast similarity lookup over all of them. Cheap and open-ended, but fuzzy.
- **Verify (local classifier):** only the shortlist is checked with a strict, closed question (yes / no / unclear) that returns probabilities. Precise, but costs one pass per person.

Search-then-verify is what lets the system answer new questions quickly without checking every person against every question.

## Goals, non-goals and the privacy rule

The goal is accurate attribute counts on moderate-density crowd video, fully local, with every answer backed by visible evidence.

**Goals**

- Answer free-text attribute questions with count error under 10% on hand-labelled test clips.
- Answer a new question on an already-indexed video in under 30 seconds on a laptop GPU or Colab.
- Show the cost saving of search-then-verify against verifying every person.
- Run with no paid API: all models open source and local.

**Non-goals**

- Very dense crowds (stampede scale), where individuals can't be detected; those need density-map models and can't give attributes.
- Live CCTV streams; the MVP works on uploaded clips.

**Privacy rule (a design constraint, not a footnote)**

- CrowdLens counts *attributes*, never *identities*: no face recognition, no "find this person", no linking people across videos.
- Track IDs are temporary numbers that exist only within one video and are discarded with the index.
- Test footage is public datasets or clips filmed with consent.

## Example questions

Every answer is a number with a range, plus the evidence behind it.

| Question | What CrowdLens returns |
| --- | --- |
| How many people are wearing red shirts? | 23 people (21–25), grid of 23 thumbnails with confidence |
| How many are carrying a backpack? | Count + thumbnails + first-seen timestamp for each |
| How many people in red **and** with a cap? | Two attributes verified per candidate, both must pass |
| How many people are using umbrellas, minute by minute? | A small time chart of the count across the clip |
| How many people walked left to right? | Count from track direction, no classifier needed |
| Show me everyone holding a phone | Thumbnail grid, each clickable to that moment in the video |

## Pipeline

The heavy work happens once per video; each question then touches only the index and a short list of candidates.

&#91;embedded content: pipeline · index once, query many times\]

The local verifier is the only step that runs per candidate, which is why the shortlist matters for speed.

## Model choices

Every stage uses an open-source model that runs locally; check for newer versions of each when you start building.

| Stage | Default choice | Alternatives | Why |
| --- | --- | --- | --- |
| Person detection | RF-DETR | YOLO (v8 or newer) | Fast, accurate person boxes; RF-DETR matches the egg-inspection pipeline |
| Tracking | ByteTrack | BoT-SORT | Turns per-frame boxes into one ID per person, so people are counted once |
| Embedding (search) | SigLIP 2 | OpenCLIP | Puts images and text in one space, so a text question can search person crops; text-only embedders like EmbeddingGemma cannot |
| Verifier (local classifier) | Small open vision-language model, e.g. Qwen-VL family (2–8B) | SmolVLM; SigLIP 2 zero-shot | Answers a closed yes / no / unclear question per crop with probabilities |
| Vector store | FAISS | Plain NumPy | Thousands of crops per video is small; no database server needed |

**Three verifier tiers, one interface.** The verifier is a swappable module, which turns model choice into an experiment:

1. **Zero-shot SigLIP 2:** score the crop against prompts ("a person wearing a red shirt" vs "a person not wearing a red shirt") and softmax. Fastest; weakest on fine detail.
2. **Small vision-language model (default):** ask a constrained question and read the probabilities of the allowed answers. This is the closest local equivalent to the Decisions API.
3. **Fine-tuned attribute classifier:** a small vision model trained on a person-attribute dataset (e.g. PA-100K). Very fast and accurate, but only for the attributes it was trained on.

## How the local verifier decides

The verifier never writes free text; it picks from a fixed list of answers and returns a probability for each, the same contract as the Decisions API.

1. **Closed question.** The user's question becomes a per-person check: "Is this person wearing a red shirt? Answer yes, no, or unclear."
2. **Read probabilities, not text.** Instead of letting the model generate, read the model's next-token probabilities for the three allowed answers and normalise them to sum to 1. One forward pass per crop, no long outputs.
3. **Multi-frame vote.** Each person is checked on their 2–3 best crops (sharpest, least blocked). A person counts if the average yes-probability is above the threshold; strong disagreement between frames marks them "unclear".
4. **Count with a range.** The headline count is the confident yeses; the range adds the unclear cases. Example: 21 confident + 4 unclear → "23 (21–25)".
5. **Calibrate the threshold** once on a small labelled set, so 0.5 actually means 50/50 for this model.

Compound questions ("red **and** cap") are split into one check per attribute, and a person must pass all of them.

## Features

The MVP answers one question type well; the signature features make the answers visible and trustworthy.

| Feature | What it does | Priority |
| --- | --- | --- |
| Upload and index | Upload a clip; detect, track, pick best crops, embed; progress bar | MVP |
| Ask a question | Free-text box; returns count, range and thumbnails | MVP |
| **Evidence grid** | Every counted person as a thumbnail with confidence; click to jump to that moment | MVP, the wow |
| **Overlay playback** | Video replays with boxes drawn only around people who match the question | Signature |
| **Correct and recount** | Click a wrong thumbnail to flip it; the count updates and the correction tunes the threshold | Signature |
| **Count over time** | Chart of matching people per minute across the clip | Signature |
| **Cost meter** | Shows people embedded, people verified, and time saved vs verifying everyone | Signature |
| Compound questions | "Red shirt and backpack" handled as two checks | Should |
| Verifier switch | Pick SigLIP / VLM / fine-tuned per question and compare | Should |
| Question history | Earlier questions on the same video answer instantly | Could |

Stack: Python backend (FastAPI) running the models; React or Streamlit front end. Streamlit is faster to build; React gives a better evidence grid.

## Evaluation

The report rests on three experiments: counting accuracy, verifier comparison, and the cost of search-then-verify versus verifying everyone.

| What is measured | Data | Metric |
| --- | --- | --- |
| Attribute verification | A person-attribute dataset such as PA-100K | Accuracy and F1 per attribute, per verifier tier |
| Tracking quality | A tracking benchmark such as MOT17 | ID switches; counted people vs true people |
| End-to-end counting | 3–5 clips hand-counted for 5 questions each | Mean absolute count error; whether the true count falls inside the range |
| Search shortlist | Same clips | Recall of true matches in the top-k shortlist, for several k |
| Speed and compute | Same clips | Seconds per question; verifier calls with and without the shortlist |

**The headline result to aim for:** the shortlist keeps over 95% of true matches while cutting verifier calls by a large factor. That single chart justifies the two-model design.

## Risks and open questions

The largest risk is counting errors from tracking, not classification.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| ID switches when people cross paths | One person counted twice, or two as one | Tune the tracker; report the count as a range |
| Small, distant people | Crops too blurry to judge colour or objects | Minimum crop size; low-resolution people go to "unclear" |
| Shortlist misses true matches | Undercount | Set k generously; measure shortlist recall |
| Colour under lighting | Red looks orange or brown in shade | Multi-frame vote; calibrate on real clips |
| Local VLM is slow on a laptop | Long verify times | Shortlist first; batch crops; run on Colab or the lab GPU |
| Misuse as surveillance | Ethical and publication problems | Attribute-only design rule, no face or identity features |

**Open questions**

- Which clips to use for the demo: a public pedestrian dataset, or campus footage filmed with consent?
- Streamlit for speed, or React for a better evidence grid?

## Milestones and build plan

Build in six milestones; the project is demoable from milestone 3.

1. **People, once.** Detection plus tracking on a clip; output one record per person with best crops. *Done when* the person count matches a hand count within 10%.
2. **Search.** Embed crops with SigLIP 2; a text query returns ranked thumbnails. *Done when* "red shirt" puts mostly red shirts on top.
3. **Verify and count.** Local VLM verifier with probability reading and multi-frame vote. *Done when* a question returns a count with range and an evidence grid.
4. **App.** Upload, ask, evidence grid, overlay playback.
5. **Evaluation.** Verifier tiers on the attribute dataset; shortlist recall and cost on hand-counted clips.
6. **Polish.** Correct-and-recount, count over time, cost meter, compound questions.

**Working with Claude Code.** Give it this spec as the brief. Build each stage as its own module with saved outputs (tracks, crops, embeddings) so later stages can be re-run without re-processing the video.

## Demo script

Two minutes, one clip, and the audience picks the questions.

1. Play 10 seconds of a busy walkway clip: nobody could count this by eye.
2. Ask "how many people are wearing red?" and show the count, range and evidence grid.
3. Replay the video with only the matching people boxed.
4. Take a question from the audience and answer it live on the same index.
5. Click a wrong thumbnail; the count updates.
6. Show the cost meter: how many people were verified instead of all of them, and the time saved.
