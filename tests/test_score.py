import json

import numpy as np
import pytest

from crowdlens import score
from crowdlens.score import (detection_probs, frame_counts, is_yes, parse_attribute, people_summary)


def test_parse_attribute():
    assert parse_attribute("How many people are wearing red shirts?") == "wearing red shirts"
    assert parse_attribute("show me everyone holding a phone") == "holding a phone"
    assert parse_attribute("carrying a backpack") == "carrying a backpack"
    for bad in ("", "  ", "how many people are"):
        with pytest.raises(ValueError):
            parse_attribute(bad)


def test_nan_is_never_yes():
    p = np.array([0.9, np.nan, 0.2])
    assert is_yes(p).tolist() == [True, False, False]


def test_detection_probs_maps_tracks_and_missing():
    p = detection_probs(np.array([1, 2, 1, 9]), {1: 0.8, 2: float("nan")})
    assert p[0] == p[2] == 0.8 and np.isnan(p[1]) and np.isnan(p[3])


def test_frame_counts_hold_between_strides():
    yes, tot = frame_counts(np.array([0, 0, 2]), np.array([.95, .1, .92]), 4, 2)
    assert yes.tolist() == [1, 1, 1, 1] and tot.tolist() == [2, 2, 1, 1]


def test_frame_counts_no_people():
    yes, tot = frame_counts(np.zeros(0, int), np.zeros(0), 5, 2)
    assert yes.sum() == 0 and tot.sum() == 0


def test_people_summary():
    assert people_summary({1: .9, 2: .1, 3: float("nan"), 4: .6}, 0.5) == (2, 1, 1)
    assert people_summary({}) == (0, 0, 0)


def test_track_scores_small_crops_unclear_and_cached(tmp_path, monkeypatch):
    import cv2
    (tmp_path / "crops").mkdir()
    cv2.imwrite(str(tmp_path / "crops/1.jpg"), np.zeros((200, 80, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "crops/2.jpg"), np.zeros((20, 10, 3), np.uint8))
    calls = []
    from crowdlens import vlm
    monkeypatch.setattr(vlm, "p_yes", lambda paths, a, **k: calls.append(paths) or [0.7] * len(paths))
    s = score.track_scores(tmp_path, [1, 2, 2], "wearing a hat")
    assert s[1] == 0.7 and np.isnan(s[2]) and len(calls) == 1 and len(calls[0]) == 1
    s2 = score.track_scores(tmp_path, [1, 2], "wearing a hat")  # cached: VLM not called again
    assert s2[1] == 0.7 and np.isnan(s2[2]) and len(calls) == 1


def test_track_scores_no_tracks(tmp_path):
    (tmp_path / "crops").mkdir()
    assert score.track_scores(tmp_path, [], "wearing a hat") == {}


def test_parse_question_routes():
    from crowdlens.score import parse_question

    assert parse_question("how many people went to the right")["motion"] == "right"
    assert parse_question("who is standing still")["motion"] == "still"
    assert parse_question("people walking toward the camera")["motion"] == "toward"
    assert parse_question("wearing a hat") == {"kind": "attr", "attribute": "wearing a hat", "multi": False, "label": "wearing a hat"}
    assert parse_question("how many people are using a phone")["multi"] is True


def test_motion_classify_and_scores():
    from crowdlens import motion

    # a person 100px tall drifting 100px right over 2s, a stationary one, and one seen too briefly
    f = np.arange(0, 20, 2)
    box = lambda cx, cy: np.stack([cx - 20, cy - 50, cx + 20, cy + 50], 1).astype(float)
    dets = dict(frame=np.concatenate([f, f, f[:2]]), track=np.array([1] * 10 + [2] * 10 + [3] * 2),
                box_ref=np.vstack([box(np.linspace(0, 100, 10), np.full(10, 200.0)),
                                   box(np.full(10, 300.0), np.full(10, 200.0)),
                                   box(np.array([0.0, 5.0]), np.array([0.0, 0.0]))]))
    right = motion.motion_scores(dets, "right", fps=10)
    assert right[1] == 1.0 and right[2] == 0.0 and np.isnan(right[3])
    assert motion.motion_scores(dets, "still", fps=10)[2] == 1.0
    assert motion.motion_scores(dets, "moving", fps=10)[1] == 1.0


def test_track_scores_shortlist_only_verifies_shortlist(tmp_path, monkeypatch):
    import cv2
    (tmp_path / "crops").mkdir()
    for t in (1, 2, 3):
        cv2.imwrite(str(tmp_path / f"crops/{t}.jpg"), np.zeros((200, 80, 3), np.uint8))
    calls = []
    from crowdlens import vlm
    monkeypatch.setattr(vlm, "p_yes", lambda paths, a, **k: calls.append(paths) or [0.9] * len(paths))
    s = score.track_scores(tmp_path, [1, 2, 3], "wearing a hat", only={2})
    assert s == {1: 0.0, 2: 0.9, 3: 0.0} and len(calls[0]) == 1
    full = score.track_scores(tmp_path, [1, 2, 3], "wearing a hat")  # separate cache entry from the shortlist run
    assert full == {1: 0.9, 2: 0.9, 3: 0.9}
