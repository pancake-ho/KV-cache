import torch

from xmodel_kv.receiver_lens import ReceiverLens
from xmodel_kv.receiver_lens_bank import ReceiverLensBank


def test_registered_identity_selects_exact_lens_and_unknown_bypasses():
    lens = ReceiverLens(torch.randn(4, 8))
    bank = ReceiverLensBank({"longbench_e_paired_lookup_v1": lens})

    assert bank.select("longbench_e_paired_lookup_v1") is lens
    assert bank.select("unregistered_agent") is None
    assert bank.receiver_ids == ("longbench_e_paired_lookup_v1",)


def test_bank_registers_lens_parameters_without_router_parameters():
    lens = ReceiverLens(torch.randn(4, 8))
    bank = ReceiverLensBank({"agent": lens})

    assert [name for name, _ in bank.named_parameters()] == [
        "lenses.agent.embeddings"
    ]
    assert sum(parameter.numel() for parameter in bank.parameters()) == 32
