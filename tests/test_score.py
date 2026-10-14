import numpy as np
import pytest
from crowdlens.score import parse_attribute, prompts, yes_probs, frame_counts, unique_yes


def test_parse_attribute():
    assert parse_attribute("How many people are wearing red shirts?") == "wearing red shirts"
    assert parse_attribute("show me everyone holding a phone") == "holding a phone"
    assert parse_attribute("carrying a backpack") == "carrying a backpack"
    for bad in ("", "  ", "how many people are"):
        with pytest.raises(ValueError):
            parse_attribute(bad)


def test_prompts():
    pos, neg = prompts("wearing red")
    assert "not wearing red" in neg and "not" not in pos


def test_yes_probs_direction_and_empty():
    pos, neg = np.array([1., 0.]), np.array([0., 1.])
    p = yes_probs(np.array([[1., 0.], [0., 1.], [.7, .7]], "float32"), pos, neg, 10.0)
    assert p[0] > .9 and p[1] < .1 and abs(p[2] - .5) < 1e-3
    assert len(yes_probs(np.zeros((0, 2), "float32"), pos, neg)) == 0


def test_frame_counts_hold_between_strides():
    frames = np.array([0, 0, 2])
    p = np.array([.9, .1, .8])
    yes, tot = frame_counts(frames, p, 4, 2)
    assert yes.tolist() == [1, 1, 1, 1] and tot.tolist() == [2, 2, 1, 1]


def test_frame_counts_no_people():
    yes, tot = frame_counts(np.zeros(0, int), np.zeros(0), 5, 2)
    assert yes.sum() == 0 and tot.sum() == 0


def test_unique_yes_uses_track_mean():
    ids, n = unique_yes(np.array([1, 1, 2, 2]), np.array([.9, .7, .2, .6]))
    assert ids == [1] and n == 2
    assert unique_yes(np.zeros(0, int), np.zeros(0)) == ([], 0)
