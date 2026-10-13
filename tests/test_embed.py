import numpy as np
from crowdlens.embed import save_index, shortlist


def unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v)


def test_shortlist_ranked_and_clamped(tmp_path):
    vecs = np.stack([unit([1, 0]), unit([0, 1]), unit([1, 1])])
    save_index(tmp_path, vecs, [10, 20, 30])
    res = shortlist(tmp_path, unit([1, 0]), k=2)
    assert [r[0] for r in res] == [10, 30]
    assert len(shortlist(tmp_path, unit([1, 0]), k=99)) == 3


def test_empty_index(tmp_path):
    save_index(tmp_path, np.zeros((0, 2), "float32"), [])
    assert shortlist(tmp_path, unit([1, 0]), k=5) == []


def test_rebuild_overwrites(tmp_path):
    save_index(tmp_path, np.stack([unit([1, 0])]), [1])
    save_index(tmp_path, np.stack([unit([0, 1])]), [2])
    assert [r[0] for r in shortlist(tmp_path, unit([0, 1]), k=5)] == [2]
