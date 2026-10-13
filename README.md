# CrowdLens
See `spec.md` and `docs/superpowers/plans/2026-10-14-crowdlens.md`.

    uv venv --python 3.11 .venv && uv pip install -e ".[models,dev]"
    pytest                      # logic tests, no GPU needed
    streamlit run crowdlens/app.py

Model stages (RF-DETR, SigLIP 2, Qwen-VL) need the `models` extra and are not covered by the unit tests.
