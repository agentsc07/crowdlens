from eval.metrics import mae, in_range_rate, recall_at_k, f1_acc


def test_metrics():
    assert mae([10, 20], [12, 18]) == 2.0
    assert in_range_rate([(1, 3), (5, 6)], [2, 9]) == 0.5
    assert recall_at_k([1, 2, 3, 4], [2, 4], 2) == 0.5
    assert recall_at_k([1], [], 1) == 1.0
    assert f1_acc([1, 0, 1], [1, 0, 0]) == (2 / 3, 2 * 1 / (2 * 1 + 1))
