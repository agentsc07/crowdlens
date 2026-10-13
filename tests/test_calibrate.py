from crowdlens.calibrate import calibrate_threshold, apply_corrections, tune_threshold
from crowdlens.vote import count
from crowdlens.schema import Verdict


def test_threshold_separable():
    t = calibrate_threshold([0.1, 0.2, 0.7, 0.9], [0, 0, 1, 1])
    assert 0.2 < t <= 0.7


def vs():
    return [Verdict(1, 0.9, "yes"), Verdict(2, 0.1, "no"), Verdict(3, 0.5, "unclear")]


def test_flip_and_recount():
    out = apply_corrections(vs(), {1, 3})
    assert [v.label for v in out] == ["no", "no", "yes"]
    assert count(out) == (1, 1)


def test_double_flip_identity():
    once = apply_corrections(vs(), {1})
    # flips are a set of toggles relative to the original verdicts
    assert apply_corrections(vs(), set()) == vs()
    assert once[0].label == "no"


def test_tune_threshold_moves_toward_corrections():
    # person with p=0.4 was flipped to yes -> threshold should not rise
    t = tune_threshold(0.5, [0.4, 0.9, 0.1], {0})
    assert t <= 0.5
