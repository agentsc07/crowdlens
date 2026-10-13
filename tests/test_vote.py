from crowdlens.vote import vote, count
from crowdlens.schema import Verdict
from crowdlens.verifiers.base import normalise


def test_vote_cases():
    assert vote([0.9, 0.8], 0.5) == "yes"
    assert vote([0.1, 0.2], 0.5) == "no"
    assert vote([0.95, 0.1], 0.5) == "unclear"
    assert vote([0.8], 0.5) == "yes"
    assert vote([], 0.5) == "unclear"


def test_count_range():
    v = [Verdict(i, 0.9, "yes") for i in range(21)] + [Verdict(99 + i, 0.5, "unclear") for i in range(4)]
    assert count(v) == (21, 25)


def test_count_empty():
    assert count([]) == (0, 0)


def test_normalise_sums_to_one():
    py, pn, pu = normalise(2.0, 0.0, -1.0)
    assert abs(py + pn + pu - 1) < 1e-9 and py > pn > pu
