import numpy as np
import pytest
import torch

from xmodel_kv.receiver_lens_gradient_structure import (
    aggregate_cosine_bootstrap,
    analyze_receiver_lens_gradients,
    domain_centroid_margins,
    normalized_gradient_spectrum,
    normalize_rows,
)


def group(rows):
    final = torch.tensor(rows, dtype=torch.float32)
    bridge = final.clone()
    return {"final": final, "bridge": bridge, "joint": final + 2 * bridge}


def test_opposed_domain_gradients_authorize_conditional_writer():
    dev = group([[1.0, 0.2], [1.0, -0.2], [0.9, 0.1], [1.1, -0.1]])
    musique = group([[-1.0, 0.2], [-1.0, -0.2], [-0.9, 0.1], [-1.1, -0.1]])
    result = analyze_receiver_lens_gradients(
        dev, musique, bootstrap_replicates=500, seed=7
    )
    assert result["comparisons"]["dev_final_vs_musique_bridge"][
        "bootstrap_95ci"
    ][1] < 0
    assert result["domain_structure"]["bootstrap_95ci"][0] > 0
    assert result["gates"]["conditional_writer_authorized"] is True


def test_aligned_domains_fail_both_authorization_conditions():
    dev = group([[1.0, 0.2], [1.0, -0.2], [0.9, 0.1], [1.1, -0.1]])
    musique = group([[1.0, 0.1], [0.9, -0.1], [1.1, 0.2], [1.0, -0.2]])
    result = analyze_receiver_lens_gradients(
        dev, musique, bootstrap_replicates=500, seed=8
    )
    assert result["comparisons"]["dev_final_vs_musique_bridge"][
        "aggregate_cosine"
    ] > 0
    assert result["gates"]["cross_domain_conflict"] is False
    assert result["gates"]["conditional_writer_authorized"] is False


def test_paired_bootstrap_requires_equal_case_counts():
    with pytest.raises(ValueError, match="case dimension"):
        aggregate_cosine_bootstrap(
            torch.ones(2, 3),
            torch.ones(3, 3),
            bootstrap_replicates=10,
            rng=np.random.default_rng(0),
            paired=True,
        )


def test_centroid_margin_and_spectrum_are_well_formed():
    dev = normalize_rows(torch.tensor([[1.0, 0.0], [0.8, 0.2]]))
    musique = normalize_rows(torch.tensor([[-1.0, 0.0], [-0.8, -0.2]]))
    margins = domain_centroid_margins(dev, musique)
    assert torch.all(margins["dev"] > 0)
    assert torch.all(margins["musique"] > 0)
    spectrum = normalized_gradient_spectrum(torch.cat((dev, musique)))
    assert spectrum["rank_50_energy"] >= 1
    assert spectrum["rank_90_energy"] >= spectrum["rank_50_energy"]
    assert spectrum["parameter_dimension"] == 2
