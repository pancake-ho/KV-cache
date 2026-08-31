import torch

from baseline.sparkv.experiment import (
    ModelRuntime,
    cpu_dtype_from_name,
    decode_chunk,
    describe_runtime,
)


def test_cpu_runtime_description_and_dtype_selection():
    runtime = ModelRuntime(
        device=torch.device("cpu"),
        dtype=torch.float32,
        backend="cpu-float32",
    )

    assert not runtime.is_cuda
    assert cpu_dtype_from_name("float32") == torch.float32
    assert cpu_dtype_from_name("bfloat16") == torch.bfloat16
    assert describe_runtime(runtime) == {
        "device": "cpu",
        "dtype": "float32",
        "backend": "cpu-float32",
    }


def test_raw_decode_converts_to_runtime_dtype():
    source = {
        "k_00": torch.ones(1, 1, 2, 2, dtype=torch.float16),
        "v_00": torch.ones(1, 1, 2, 2, dtype=torch.float16),
    }

    decoded = decode_chunk(source, "raw", layers=1, target_dtype=torch.float32)

    assert decoded["k_00"].dtype == torch.float32
    assert decoded["v_00"].dtype == torch.float32


def test_q5_decode_uses_runtime_dtype():
    source = {
        "qk_00": torch.full((1, 1, 2, 2), 16, dtype=torch.uint8),
        "qv_00": torch.full((1, 1, 2, 2), 17, dtype=torch.uint8),
        "sk_00": torch.ones(1, 1, 1, 1, dtype=torch.float32),
        "sv_00": torch.full((1, 1, 1, 1), 0.5, dtype=torch.float32),
    }

    decoded = decode_chunk(source, "q5", layers=1, target_dtype=torch.float32)

    assert decoded["k_00"].dtype == torch.float32
    assert decoded["v_00"].dtype == torch.float32
    assert torch.equal(decoded["k_00"], torch.zeros_like(decoded["k_00"]))
    assert torch.equal(decoded["v_00"], torch.full_like(decoded["v_00"], 0.5))
