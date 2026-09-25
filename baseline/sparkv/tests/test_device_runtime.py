import torch

from baseline.sparkv.runtime import (
    ModelRuntime,
    cpu_dtype_from_name,
    describe_runtime,
    qa_f1,
)


def test_cpu_runtime_description_and_dtype_selection():
    runtime = ModelRuntime(
        device=torch.device("cpu"),
        dtype=torch.float32,
        backend="cpu-float32",
    )

    assert not runtime.is_cuda
    assert (
        cpu_dtype_from_name("float32")
        == torch.float32
    )
    assert (
        cpu_dtype_from_name("bfloat16")
        == torch.bfloat16
    )

    assert describe_runtime(runtime) == {
        "device": "cpu",
        "dtype": "float32",
        "backend": "cpu-float32",
    }


def test_qa_f1_exact_and_partial_match():
    assert qa_f1(
        "Seoul",
        ["Seoul"],
    ) == 1.0

    score = qa_f1(
        "Seoul Korea",
        ["Seoul"],
    )

    assert 0.0 < score < 1.0
