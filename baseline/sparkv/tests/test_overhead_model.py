from baseline.sparkv.overhead_model import SparseAttentionMLP


def test_predictor_architecture_matches_paper():
    model = SparseAttentionMLP()
    layers = list(model.net)

    assert layers[0].in_features == 3
    assert layers[0].out_features == 48
    assert layers[2].in_features == 48
    assert layers[2].out_features == 24
    assert layers[4].in_features == 24
    assert layers[4].out_features == 1
