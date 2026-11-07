from pathlib import Path
import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

from xmodel_kv.figure2.common import digest, read_json, sha256, write_json
from xmodel_kv.figure2.extract import cache_pairs, extract_with_model
from xmodel_kv.figure2.plot import plot
from xmodel_kv.figure2.probe import run_probe
from xmodel_kv.figure2.store import ProbeStore, validate_pair


def tiny_model(layers):
    return Qwen3Model(Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=8, max_position_embeddings=128, rope_theta=1_000_000,
                     attention_dropout=0.0, use_cache=True)).eval()


def test_tiny_qwen_extraction_probe_resume_and_plot(tmp_path):
    torch.manual_seed(12)
    torch.set_num_threads(1)
    rng = np.random.default_rng(12)
    tokens = rng.integers(2, 90, size=(6, 24), dtype=np.int32)
    np.save(tmp_path / 'tokens.npy', tokens)
    cfg = {'stride': 2, 'flush_sequences': 2, 'covariance_dtype': 'float64',
           'covariance_rcond': 1e-8, 'source_model': 'tiny-source', 'target_model': 'tiny-target',
           'sequence_length': 24, 'smoke': True}
    prepared = {'config': cfg, 'models': {'source': {'model_id': 'tiny-source'},
                 'target': {'model_id': 'tiny-target'}}, 'train_observations': 48,
                 'heldout_observations': 24, 'tokens_sha256': sha256(tmp_path / 'tokens.npy')}
    prepared['prepare_digest'] = digest(prepared)
    write_json(tmp_path / 'prepare.json', prepared)
    models = {'source': tiny_model(2), 'target': tiny_model(3)}
    for role, model in models.items():
        extract_with_model(model, tmp_path / 'tokens.npy', tmp_path / role,
                           role=role, prepared=prepared, device=torch.device('cpu'))
        store = ProbeStore(tmp_path / role)
        assert store.open('k_rope', 0).shape == (72, 2, 8)
        assert store.metadata['complete']
        with torch.inference_mode():
            output = model(input_ids=torch.tensor(tokens[0].astype(np.int64))[None], use_cache=True)
        key, value = cache_pairs(output.past_key_values)[0]
        expected = key[0, :, ::2].permute(1, 0, 2).half().numpy()
        np.testing.assert_array_equal(store.open('k_rope', 0)[:12], expected)
        expected_v = value[0, :, ::2].permute(1, 0, 2).half().numpy()
        np.testing.assert_array_equal(store.open('v', 0)[:12], expected_v)
        assert not np.array_equal(store.open('k_rope', 0), store.open('k_stripped', 0))
    first = run_probe(tmp_path, 'cpu')
    assert first['train_uniform'].shape == (3, 2, 3, 2)
    assert first['done'].all()
    assert np.isfinite(first['test_uniform']).all()
    second = run_probe(tmp_path, 'cpu')
    np.testing.assert_array_equal(first['train_uniform'], second['train_uniform'])
    # Extraction completion is verified and safely reused.
    extract_with_model(models['source'], tmp_path / 'tokens.npy', tmp_path / 'source',
                       role='source', prepared=prepared, device=torch.device('cpu'))
    # Emulate an interrupted extraction at a persisted boundary, then resume it.
    metadata_path = tmp_path / 'source/metadata.json'
    metadata = read_json(metadata_path)
    metadata.update(complete=False, completed_sequences=2)
    write_json(metadata_path, metadata)
    before = np.array(ProbeStore(tmp_path / 'source', complete=False).open('k_rope', 0))
    extract_with_model(models['source'], tmp_path / 'tokens.npy', tmp_path / 'source',
                       role='source', prepared=prepared, device=torch.device('cpu'))
    np.testing.assert_array_equal(before, ProbeStore(tmp_path / 'source').open('k_rope', 0))
    plot(tmp_path)
    for name in ('figure2_calibration', 'figure2_heldout'):
        for extension in ('png', 'pdf'):
            assert (tmp_path / f'results/{name}.{extension}').stat().st_size > 1000


def test_pair_validation_rejects_different_tokens(tmp_path):
    shared = {'schema_version': 1, 'complete': True, 'num_observations': 10,
              'num_kv_heads': 2, 'head_dim': 8, 'prepare_digest': 'same',
              'tokens_sha256': 'same', 'train_observations': 6, 'heldout_observations': 4,
              'sequence_length': 5, 'stride': 1, 'config': {}}
    write_json(tmp_path / 'source/metadata.json', {**shared, 'role': 'source'})
    write_json(tmp_path / 'target/metadata.json', {**shared, 'role': 'target', 'tokens_sha256': 'different'})
    with pytest.raises(ValueError, match='tokens_sha256'):
        validate_pair(ProbeStore(tmp_path / 'source'), ProbeStore(tmp_path / 'target'))
