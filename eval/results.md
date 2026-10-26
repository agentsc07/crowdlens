# Results (Claude-labelled, human audit pending)

164 people from 4 MOT17 clips x 8 questions; people marked unsure/unjudgeable are excluded per question.

Threshold chosen on clips 02/04/09 (max pooled F1): **0.9**. Held-out clip 11 pooled F1: 0.86 (at 0.5: 0.78).

| Question | true | pred@0.5 | P | R | F1 | count err % | pred@0.9 | F1 | count err % |
|---|---|---|---|---|---|---|---|---|---|
| wearing a red top | 10 | 28 | 0.36 | 1.00 | 0.53 | 180 | 18 | 0.71 | 80 |
| wearing a white shirt | 20 | 46 | 0.43 | 1.00 | 0.61 | 130 | 29 | 0.78 | 45 |
| wearing a black jacket | 12 | 34 | 0.35 | 1.00 | 0.52 | 183 | 15 | 0.81 | 25 |
| carrying a backpack | 9 | 17 | 0.53 | 1.00 | 0.69 | 89 | 11 | 0.80 | 22 |
| carrying a handbag or shoulder bag | 25 | 37 | 0.62 | 0.92 | 0.74 | 48 | 20 | 0.71 | 20 |
| wearing shorts | 16 | 21 | 0.76 | 1.00 | 0.86 | 31 | 20 | 0.89 | 25 |
| wearing a hat or cap | 10 | 15 | 0.60 | 0.90 | 0.72 | 50 | 9 | 0.84 | 10 |
| holding a shopping bag | 11 | 30 | 0.33 | 0.91 | 0.49 | 173 | 16 | 0.59 | 45 |
| **pooled @0.5** | | | 0.48 | 0.96 | 0.64 | 111 (mean) | | | |
| **pooled @0.9** | | | 0.70 | 0.85 | 0.76 | 34 (mean) | | | |

## Search-then-verify (SigLIP2 shortlist, then Qwen3-VL on the shortlist only)

VLM cost 0.50 s/person on this machine; SigLIP2 query 1.57 s; embedding done once (9.0 s).

| shortlist | shortlist recall | F1 | VLM calls/question | est. s/question |
|---|---|---|---|---|
| top 10% | 0.50 | 0.59 | 12 (10%) | 7.7 |
| top 20% | 0.70 | 0.72 | 24 (20%) | 13.8 |
| top 30% | 0.82 | 0.74 | 37 (30%) | 20.0 |
| top 40% | 0.88 | 0.75 | 49 (40%) | 26.1 |
| top 50% | 0.94 | 0.77 | 62 (50%) | 32.2 |
| top 75% | 0.98 | 0.77 | 92 (75%) | 47.5 |
| top 100% | 1.00 | 0.76 | 123 (100%) | 62.8 |

## Ranking quality (average precision)

| Question | prevalence | SigLIP2 AP | Qwen3-VL AP |
|---|---|---|---|
| wearing a red top | 0.08 | 0.78 | 0.99 |
| wearing a white shirt | 0.16 | 0.80 | 0.97 |
| wearing a black jacket | 0.10 | 0.55 | 0.85 |
| carrying a backpack | 0.07 | 0.66 | 0.97 |
| carrying a handbag or shoulder bag | 0.21 | 0.65 | 0.74 |
| wearing shorts | 0.13 | 0.80 | 1.00 |
| wearing a hat or cap | 0.08 | 0.67 | 0.93 |
| holding a shopping bag | 0.09 | 0.23 | 0.77 |
