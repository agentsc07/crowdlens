# CrowdLens
Ask a plain-English attribute question of a crowd video; get a yes/no for every person in every frame, drawn onto the video.

    uv pip install -e ".[dev]"
    python -m streamlit run crowdlens/app.py

1. **Index (once per video, `index.py`)** — RF-DETR Nano + ByteTrack across the whole clip; SigLIP 2 embeds every person crop.
2. **Ask (milliseconds, `score.py`)** — the question becomes two text prompts ("a person X" / "a person not X"); P(yes) per person per frame is a softmax over dot products.
3. **Show (`render.py`, `app.py`)** — annotated video (green = yes, red = no, with P(yes) and a per-frame HUD), per-frame count chart, frame inspector.

Attributes only, never identities; track ids are per-video temporaries. See `spec.md` for the original, larger design.
