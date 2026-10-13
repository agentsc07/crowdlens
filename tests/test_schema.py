from crowdlens.schema import Person, Crop, save_people, load_people


def test_roundtrip(tmp_path):
    c = Crop(1, 10, (0, 0, 50, 100), "a.jpg", 0.9, 1.0)
    p = Person(1, 10, 20, [c], [5.0, 9.0], {10: (0, 0, 50, 100)})
    f = tmp_path / "people.json"
    save_people(f, [p])
    assert load_people(f) == [p]


def test_empty(tmp_path):
    f = tmp_path / "people.json"
    save_people(f, [])
    assert load_people(f) == []
