from pathlib import Path

DATA_DIR = Path("data")
EMBED_MODEL = "google/siglip2-base-patch16-224"
DETECT_CONF = 0.4
DEFAULT_STRIDE = 2   # run the detector on every Nth frame; boxes are held in between
THRESHOLD = 0.5
