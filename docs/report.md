# CrowdLens: report (one page)

**Problem.** Counting people with a given attribute in video needs either manual viewing or one trained classifier per attribute. CrowdLens answers a new plain-English question on the spot, with visual evidence, locally, and counts attributes only (never identities).

**Method.** Index once: RF-DETR Nano on tiled views (finds small and partial people), ByteTrack, sharpest crop per person, SigLIP2 embedding, camera-motion estimate. Per question: Qwen3-VL-2B returns P(Yes) vs P(No) from one forward pass per crop (no free text). Optional search-then-verify: SigLIP2 ranks crops by similarity to the question and only the top share is verified. Movement questions are answered from camera-stabilised track geometry without a model.

**Evaluation (Claude-labelled, human audit pending).** 164 judgeable people from 4 MOT17 clips (15 s each), 8 attribute questions, 9 to 25 positives each; people marked unsure or unjudgeable are excluded. Threshold chosen on clips 02/04/09, checked on held-out clip 11.

| | P | R | F1 | mean count error |
|---|---|---|---|---|
| threshold 0.5 | 0.48 | 0.96 | 0.64 | 111 % |
| threshold 0.9 | 0.70 | 0.85 | 0.76 | 34 % |

Held-out F1 at 0.9: 0.86. Search-then-verify: top 30 % keeps 82 % of true matches (F1 0.74), top 50 % keeps 94 % (F1 0.77, equal to verify-everyone 0.76) at half the VLM calls. Tracks per true pedestrian: 0.94 to 1.41 (mean 1.24).

**Findings.** The VLM over-says yes at 0.5, so a high threshold matters more than any model change. Count error stays well above the 10 % target. Search-then-verify halves the cost without losing accuracy but SigLIP2-base is too weak to cut it further, especially for objects (shopping bag AP 0.23).

**Hosted demo.** Free hosting forced a demo-only design: preset questions use cached verified answers; other questions get an unverified SigLIP2 search answer from a NumPy text encoder (655 MB peak RAM). That quick count has 26 % mean error but mostly reflects a prior of about 10 % of people matching, so it is shown as a rough guide next to the ranking.

**Limitations.** Labels are machine-made from low-resolution crops (human audit of 30 people pending); few positives per question; one machine, noisy timings; hosted CPU speed not measured; track fragmentation inflates counts; regex router; 15 s clips; MOT17 is non-commercial.

**Next steps.** Run the human audit and re-run `eval/metrics.py`; larger SigLIP2 (so400m) or a fine-tuned attribute head for the shortlist; merge fragmented tracks (re-identification within a video only, attributes not identities); calibrate per question; measure CPU latency on the hosted Space; replace the regex router with a small language-model parse that splits compound questions.
