import cv2
import numpy as np

from . import config


def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_best(crops, n=config.TOP_CROPS):
    if not crops:
        return []
    big = [c for c in crops if (c.box[3] - c.box[1]) >= config.MIN_CROP_H]
    pool = big or [max(crops, key=lambda c: c.box[3] - c.box[1])]
    return sorted(pool, key=lambda c: c.sharpness * c.visibility, reverse=True)[:n]
