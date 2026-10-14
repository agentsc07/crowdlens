# CrowdLens
Ask a plain-English attribute question of a crowd video; get a yes/no for every person in every frame, drawn onto the video.

    uv pip install -e ".[dev]"
    python -m streamlit run crowdlens/app.py

1. **Index (once per video, `index.py`)** — RF-DETR Nano + ByteTrack across the whole clip; each person's best crop is saved.
2. **Ask (once per question, cached)** — Qwen3-VL 2B answers "Is this person X? Yes or No." for each tracked person on their best crop (P(yes) from next-token logits); every box of that person shares the answer. Crops under 64px tall are "too small to judge".
3. **Show (`render.py`, `app.py`)** — annotated video (green = yes, red = no, with P(yes) and a per-frame HUD), per-frame count chart, frame inspector.

Attributes only, never identities; track ids are per-video temporaries. See `spec.md` for the original, larger design.
