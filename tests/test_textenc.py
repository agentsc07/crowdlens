import numpy as np

from crowdlens import textenc


def test_text_encoder_is_unit_norm_and_semantic():
    e = textenc.embed_text(["a photo of a person wearing a red top", "a photo of a person wearing a crimson shirt",
                            "a photo of a person pushing a stroller"])
    assert e.shape == (3, 768) and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)
    assert e[0] @ e[1] > e[0] @ e[2]  # red top is nearer to crimson shirt than to stroller


def test_unknown_script_does_not_crash():
    e = textenc.embed_text(["a photo of a person wearing a 東京 jacket"])
    assert np.isfinite(e).all()
