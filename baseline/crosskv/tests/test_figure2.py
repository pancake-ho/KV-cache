import hashlib
from pathlib import Path
import numpy as np
import pytest

from xmodel_kv.figure2.common import config
from xmodel_kv.figure2.prepare import pack_split
from xmodel_kv.figure2.regression import Statistics, fit_pair, ols_inverse, score


def reference_r2(x, y, xt, yt):
    values = []
    for head in range(x.shape[1]):
        design = np.column_stack((x[:, head], np.ones(len(x))))
        weights = np.linalg.lstsq(design, y[:, head], rcond=None)[0]
        pred = np.column_stack((xt[:, head], np.ones(len(xt)))) @ weights
        sse = ((yt[:, head] - pred)**2).sum(axis=0)
        sst = ((yt[:, head] - yt[:, head].mean(axis=0))**2).sum(axis=0)
        values.append((1 - sse / sst).mean())
    return np.asarray(values)


def calculate(x, y, xt, yt, device="cpu", dtype="float64"):
    engine = Statistics(device, dtype)
    sx = engine.moments(x, covariance=True)
    sy = engine.moments(y, covariance=False)
    tx = engine.moments(xt, covariance=True)
    ty = engine.moments(yt, covariance=False)
    inverse, rank, _ = ols_inverse(sx.covariance, 1e-10 if dtype == 'float64' else 1e-6)
    return fit_pair(engine, sx, sy, tx, ty, inverse), rank


def test_statistics_match_direct_ols_with_holdout_mean_shift():
    rng = np.random.default_rng(2026)
    x, xt = rng.normal(size=(80, 3, 5)), rng.normal(size=(50, 3, 5)) + 7
    weights, bias = rng.normal(size=(3, 5, 4)), rng.normal(size=(3, 4))
    y = np.einsum('nhd,hdo->nho', x, weights) + bias + rng.normal(size=(80, 3, 4)) * 0.2
    yt = np.einsum('nhd,hdo->nho', xt, weights) + bias + rng.normal(size=(50, 3, 4)) * 0.2
    (train, test), rank = calculate(x, y, xt, yt)
    np.testing.assert_allclose(train['uniform'], reference_r2(x, y, x, y), atol=1e-10)
    np.testing.assert_allclose(test['uniform'], reference_r2(x, y, xt, yt), atol=1e-10)
    assert np.all(rank == 5)


def test_rank_deficient_ols_and_negative_holdout_are_preserved():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(70, 2, 2))
    xt = rng.normal(size=(35, 2, 2))
    x = np.concatenate((x, x[..., :1], np.zeros_like(x[..., :1])), axis=-1)
    xt = np.concatenate((xt, xt[..., :1], np.zeros_like(xt[..., :1])), axis=-1)
    y = x[..., :2] * 3 + 5
    yt = xt[..., :2] * -8 - 30
    (train, test), rank = calculate(x, y, xt, yt)
    np.testing.assert_allclose(train['uniform'], 1, atol=1e-10)
    np.testing.assert_allclose(test['uniform'], reference_r2(x, y, xt, yt), atol=1e-10)
    assert np.all(rank == 2)
    assert np.all(test['uniform'] < 0)


def test_constant_channels_are_excluded_explicitly():
    engine = Statistics('cpu', 'float64')
    rng = np.random.default_rng(9)
    x = rng.normal(size=(60, 2, 3))
    y = x.copy()
    y[..., 0] = 2
    sx, sy = engine.moments(x, covariance=True), engine.moments(y, covariance=False)
    inv, _, _ = ols_inverse(sx.covariance, 1e-10)
    weight = inv @ engine.cross(sx, sy)
    bias = sy.mean - np.einsum('hd,hdo->ho', sx.mean, weight)
    result = score(sx, sy, engine.cross(sx, sy), weight, bias)
    assert np.all(result['valid_channels'] == 2)
    np.testing.assert_allclose(result['uniform'], 1, atol=1e-10)


def test_no_cross_head_information_is_used():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 2, 3))
    xt = rng.normal(size=(1500, 2, 3))
    # Target head zero depends on source head one: a matched-head fit must fail.
    y, yt = x[:, ::-1].copy(), xt[:, ::-1].copy()
    (train, test), _ = calculate(x, y, xt, yt)
    assert np.max(train['uniform']) < 0.03
    assert np.max(test['uniform']) < 0.03


class Tokenizer:
    eos_token_id = 1
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [2 + ord(c) for c in text]}


def test_document_disjoint_packing_is_deterministic():
    rng = np.random.default_rng(24)
    records = [{'text': ''.join(chr(65 + value) for value in rng.integers(0, 26, 85))} for _ in range(1000)]
    cfg = {'train_sequences': 12, 'heldout_sequences': 5, 'sequence_length': 32,
           'seed': 2026, 'document_train_fraction': 0.8, 'text_field': 'text'}
    a, docs_a = pack_split(records + records, Tokenizer(), cfg)
    b, docs_b = pack_split(records + records, Tokenizer(), cfg)
    np.testing.assert_array_equal(a, b)
    assert docs_a == docs_b
    assert not set(docs_a['train']) & set(docs_a['heldout'])
    assert a.shape == (17, 32)
    fit = {hashlib.sha256(row.tobytes()).hexdigest() for row in a[:12]}
    test = {hashlib.sha256(row.tobytes()).hexdigest() for row in a[12:]}
    assert not fit & test


def test_identical_packed_sequences_across_splits_are_rejected():
    class ConstantTokenizer:
        eos_token_id = 1
        def __call__(self, text, add_special_tokens=False):
            return {'input_ids': [2] * 31}
    cfg = {'train_sequences': 2, 'heldout_sequences': 2, 'sequence_length': 32,
           'seed': 2026, 'document_train_fraction': 0.8, 'text_field': 'text'}
    with pytest.raises(ValueError, match='identical token sequences'):
        pack_split([{'text': f'different text {i}'} for i in range(100)], ConstantTokenizer(), cfg)


def test_small_corpus_and_invalid_shape_fail():
    cfg = {'train_sequences': 12, 'heldout_sequences': 5, 'sequence_length': 32,
           'seed': 2026, 'document_train_fraction': 0.8, 'text_field': 'text'}
    with pytest.raises(ValueError, match='not enough'):
        pack_split([{'text': 'short'}], Tokenizer(), cfg)
    with pytest.raises(ValueError):
        Statistics('cpu', 'float64').moments(np.zeros((1, 2, 3)), covariance=True)


def test_main_and_smoke_config_observation_counts():
    path = Path(__file__).parents[1] / 'configs/figure2_qwen3_1p7b_4b.json'
    cfg, smoke = config(path), config(path, smoke=True)
    assert cfg['train_sequences'] * len(range(0, cfg['sequence_length'], cfg['stride'])) == 128000
    assert cfg['heldout_sequences'] * len(range(0, cfg['sequence_length'], cfg['stride'])) == 16384
    assert smoke['train_sequences'] * len(range(0, smoke['sequence_length'], smoke['stride'])) == 256


def test_cuda_statistics_match_cpu_reference_if_available():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; CPU reference tests remain active')
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(500, 2, 6)), rng.normal(size=(500, 2, 4))
    xt, yt = rng.normal(size=(100, 2, 6)), rng.normal(size=(100, 2, 4))
    (cpu_train, cpu_test), _ = calculate(x, y, xt, yt)
    (gpu_train, gpu_test), _ = calculate(x, y, xt, yt, 'cuda:0', 'float32')
    np.testing.assert_allclose(gpu_train['uniform'], cpu_train['uniform'], atol=2e-5)
    np.testing.assert_allclose(gpu_test['uniform'], cpu_test['uniform'], atol=2e-5)
