from crowdlens.crops import select_best, sharpness
from crowdlens.schema import Crop
import numpy as np


def mk(i, h, s, v):
    return Crop(1, i, (0, 0, 30, h), f"{i}.jpg", s, v)


def test_top3_ordered_by_score():
    cs = [mk(0, 100, 0.2, 1), mk(1, 100, 0.9, 1), mk(2, 100, 0.5, 1), mk(3, 100, 0.8, 0.5), mk(4, 100, 0.7, 1)]
    assert [c.frame_idx for c in select_best(cs)] == [1, 4, 2]


def test_tiny_dropped_unless_all_tiny():
    cs = [mk(0, 20, 0.9, 1), mk(1, 100, 0.1, 1)]
    assert [c.frame_idx for c in select_best(cs)] == [1]
    assert [c.frame_idx for c in select_best([mk(0, 20, 0.1, 1), mk(1, 30, 0.1, 1)])] == [1]


def test_empty():
    assert select_best([]) == []


def test_sharpness_flat_vs_noise():
    flat = np.full((64, 64, 3), 100, np.uint8)
    noise = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert sharpness(noise) > sharpness(flat)
