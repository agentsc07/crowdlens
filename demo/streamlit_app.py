"""Entry point for the hosted demo (Streamlit Community Cloud). Forces demo mode (pre-indexed samples, cached answers,
NumPy text search; no torch, no detector) and keeps its light requirements.txt next to this file so the platform
doesn't install the full pyproject dependencies."""
import os
import runpy
from pathlib import Path

os.environ["CROWDLENS_DEMO"] = "1"
runpy.run_path(str(Path(__file__).resolve().parent.parent / "crowdlens" / "app.py"), run_name="__main__")
