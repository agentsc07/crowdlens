from crowdlens.direction import left_to_right
from crowdlens.timeline import per_minute
from crowdlens.cost import cost
from crowdlens.schema import Person, Verdict


def P(i, first, last, xs):
    return Person(i, first, last, [], xs)


def test_direction():
    ps = [P(1, 0, 9, [0, 100]), P(2, 0, 9, [100, 0]), P(3, 0, 9, [50, 60])]
    assert left_to_right(ps) == [1]


def test_per_minute_spans_buckets():
    ps = [P(1, 0, 1800, []), P(2, 100, 200, [])]  # fps 30 -> 1800 frames = minute 1
    vs = [Verdict(1, .9, "yes"), Verdict(2, .9, "yes")]
    assert per_minute(ps, vs, 30) == {0: 2, 1: 1}


def test_per_minute_ignores_non_yes():
    ps = [P(1, 0, 10, [])]
    assert per_minute(ps, [Verdict(1, .1, "no")], 30) == {}


def test_cost():
    c = cost(100, 20, 0.5)
    assert c == {"embedded": 100, "verified": 20, "saved_calls": 80, "saved_seconds": 40.0}
    assert cost(10, 10, 1.0)["saved_calls"] == 0
