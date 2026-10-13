from pathlib import Path

DATA_DIR = Path("data")
MIN_CROP_H = 64
TOP_CROPS = 3
K_DEFAULT = 200
THRESHOLD = 0.5
UNCLEAR_SPREAD = 0.4

EMBED_MODEL = "google/siglip2-base-patch16-224"
VLM_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
