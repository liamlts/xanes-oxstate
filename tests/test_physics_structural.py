from xanes_oxstate.physics.analysis import group_failures_by_field


def _failures():
    return [
        {"true": 0, "pred": 1, "coord": 4, "nn_elem": "O"},
        {"true": 0, "pred": 1, "coord": 4, "nn_elem": "O"},
        {"true": 0, "pred": 1, "coord": 6, "nn_elem": "S"},
        {"true": 0, "pred": 0, "coord": 6, "nn_elem": "O"},
    ]


def test_group_failures_counts_by_field():
    out = group_failures_by_field(_failures(), field="coord",
                                  true_class=0, pred_class=1)
    assert out == {4: 2, 6: 1}


def test_group_failures_handles_missing_field():
    out = group_failures_by_field(
        [{"true": 0, "pred": 1}], field="coord",
        true_class=0, pred_class=1,
    )
    assert out == {}
