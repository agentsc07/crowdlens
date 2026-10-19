from pathlib import Path

DATA_DIR = Path("data")
VLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DETECT_WIDTH = 1080  # detection runs on a copy this wide, plus overlapping tiles
TILE = 640
DETECT_CONF = 0.4
DEFAULT_STRIDE = 2   # run the detector on every Nth frame; boxes are held in between
THRESHOLD = 0.5
MIN_CROP_H = 64      # best crop shorter than this is too small to judge -> "unclear"
VLM_CROP_MAX = 224  # longest side of a crop fed to the VLM, in pixels
