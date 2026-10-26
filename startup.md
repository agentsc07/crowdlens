# CrowdLens: startup

**Prerequisites:** Python 3.11+, `ffmpeg` on PATH, ~8 GB free disk for model weights (Qwen3-VL-2B, SigLIP2, RF-DETR download on first use). Apple silicon (MPS), CUDA or CPU all work; CPU is slow.

**Setup**
```bash
python3.11 -m venv ~/.venvs/crowdlens && source ~/.venvs/crowdlens/bin/activate
pip install -e ".[dev]"
```
Optional: `CROWDLENS_DATA=/path` (where indexes live; default `~/.crowdlens/data`), `CROWDLENS_LITE=1` (CPU demo mode), `CROWDLENS_DEVICE=cpu|cuda|mps`.

**Run**
```bash
python -m streamlit run crowdlens/app.py        # http://localhost:8501
crowdlens index video.mp4
crowdlens ask video.mp4 "wearing a red shirt?" [--search 0.5] [--json]
```
Hosted demo (Docker): `docker build -t crowdlens . && docker run -p 7860:7860 crowdlens` then http://localhost:7860.

**Demo mode (no torch, ~650 MB):** `pip install -r demo/requirements.txt && streamlit run demo/streamlit_app.py` serves only the pre-indexed samples in `space_data/`. Streamlit Cloud entry point: `demo/streamlit_app.py` (it forces demo mode and uses `demo/requirements.txt`, not the full pyproject dependencies). It switches on automatically when torch is not installed.

**Stop:** Ctrl+C. **Tests:** `pytest` (12 pass).

**Common problems**
- `No module named crowdlens`: run `pip install -e .` from the repo root.
- `module 'supervision' has no attribute 'ByteTrack'`: keep `supervision==0.30.9` (0.31 removed it); restart the server after changing.
- Metal "Insufficient Memory" or very slow: close other apps; crops are already shrunk to 224 px.
- Streamlit hangs on start from a Desktop/iCloud folder: keep the venv outside iCloud (`~/.venvs`).
- Upload 403: the repo's `.streamlit/config.toml` disables XSRF/CORS; start from the repo root.
