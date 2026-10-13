import numpy as np
import pytest
from crowdlens import query
from crowdlens.embed import save_index
from crowdlens.schema import Person, Crop, save_people


def test_split_attributes():
    assert query.split_attributes("how many people in red and with a cap") == ["in red", "with a cap"]
    assert query.split_attributes("How many people are wearing red shirts?") == ["wearing red shirts"]


def test_empty_question_raises():
    with pytest.raises(ValueError):
        query.split_attributes("how many people are")
    with pytest.raises(ValueError):
        query.split_attributes("   ")


class Fake:
    def __init__(self, table):
        self.table = table

    def p_yes(self, path, attr):
        return self.table[(path.rsplit("/", 1)[-1], attr)]


def setup(tmp_path, n=5, tall=100):
    people = [Person(i, 0, 10, [Crop(i, 0, (0, 0, 30, tall), f"c{i}.jpg", 1, 1)], [0, 0]) for i in range(n)]
    save_people(tmp_path / "people.json", people)
    v = np.eye(max(n, 1), dtype="float32")[:n] if n else np.zeros((0, 1), "float32")
    save_index(tmp_path, v, list(range(n)))
    return people


def fake_embed(monkeypatch):
    monkeypatch.setattr(query, "embed_text", lambda q: np.ones(5, dtype="float32") / np.sqrt(5))


def test_compound_all_must_pass(tmp_path, monkeypatch):
    setup(tmp_path)
    fake_embed(monkeypatch)
    t = {(f"c{i}.jpg", a): 0.9 for i in range(5) for a in ("in red", "with a cap")}
    t[("c0.jpg", "with a cap")] = 0.1
    ans = query.answer(tmp_path, "people in red and with a cap", Fake(t), k=5, crop_root=tmp_path)
    assert (ans.low, ans.high) == (4, 4)
    assert ans.n_verified == 5 and ans.n_total == 5


def test_zero_people(tmp_path, monkeypatch):
    setup(tmp_path, n=0)
    monkeypatch.setattr(query, "embed_text", lambda q: np.ones(1, dtype="float32"))
    ans = query.answer(tmp_path, "red shirts", Fake({}), k=5, crop_root=tmp_path)
    assert (ans.low, ans.high, ans.n_verified) == (0, 0, 0)


def test_k_limits_verified_and_tiny_is_unclear(tmp_path, monkeypatch):
    setup(tmp_path, n=5, tall=20)
    fake_embed(monkeypatch)
    ans = query.answer(tmp_path, "red shirts", Fake({}), k=3, crop_root=tmp_path)
    assert ans.n_verified <= 3
    assert (ans.low, ans.high) == (0, 3)
