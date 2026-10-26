import os
from pathlib import Path

import importlib.util

# CROWDLENS_DEMO=1 (or no torch installed) is the free hosted demo: pre-indexed samples only, cached verified answers,
# and a quick text-to-crop search for any other question. No model weights are loaded except the numpy text encoder.
# Streamlit Community Cloud mounts repos under /mount/src and may install the full dependency list, so torch can be
# importable there even though the host cannot run the models: treat it as the demo host too.
DEMO = (os.environ.get("CROWDLENS_DEMO") == "1" or importlib.util.find_spec("torch") is None
        or Path(__file__).resolve().parts[:3] == ("/", "mount", "src"))
# kept out of iCloud-synced folders (Desktop/Documents): proxies and crops churn thousands of files
DATA_DIR = Path(os.environ.get("CROWDLENS_DATA", Path(__file__).resolve().parent.parent / "space_data" if DEMO
                               else Path.home() / ".crowdlens" / "data"))
# CROWDLENS_LITE=1 is the free-CPU hosted demo: coarser frame stride, smaller crops, a short upload limit, and new
# questions only verify the SigLIP2 shortlist
LITE = os.environ.get("CROWDLENS_LITE") == "1"
MAX_UPLOAD_SECONDS = 15.5 if LITE else None
SHORTLIST = 8 if LITE else 0
DEVICE = os.environ.get("CROWDLENS_DEVICE")  # force "cpu" / "cuda" / "mps"; default is auto
VLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
EMBED_MODEL = "google/siglip2-base-patch16-224"
DETECT_WIDTH = 1080  # detection runs on a copy this wide, plus overlapping tiles
TILE = 640
DETECT_CONF = 0.4
MIN_TRACK_DETS = 3  # a person must appear in at least this many processed frames
DEFAULT_STRIDE = 8 if LITE else 4   # run the detector on every Nth frame; boxes are held in between
THRESHOLD = 0.9  # picked on MOT17 clips 02/04/09 (eval/metrics.py); 0.5 over-says yes (precision 0.48)
MIN_CROP_H = 64      # best crop shorter than this is too small to judge -> "unclear"
VLM_CROP_MAX = 112 if LITE else 224  # longest side of a crop fed to the VLM, in pixels
PREVIEW_MAX = 960  # longest side of the annotated preview video, in pixels
MIN_MOTION_DETS = 4      # a track needs this many detections ...
MIN_MOTION_SECONDS = 1.0  # ... spanning at least this long before its movement is judged
SLOT_SPAN = 6            # multi-moment mode keeps the sharpest crop from each run of this many detections
LOST_BUFFER = 90         # tracker keeps a lost person alive this many frames (at 30fps-equivalent)
# COCO classes the detector can index; the first is the default
CATEGORIES = ["person", "car", "bicycle", "motorcycle", "bus", "truck", "dog", "cat", "bird", "horse", "boat", "umbrella"]


def video_dir(vid, category="person"):
    """Each (video, category) is its own index; people keep the plain name."""
    return DATA_DIR / (vid if category == "person" else f"{vid}__{category}")
VLM_MULTI_CROP_MAX = 160  # smaller crops when several moments of one object go in one prompt
