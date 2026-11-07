from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import numpy as np

from ..data import iter_records, record_to_text
from .common import config, digest, lock, provenance, read_json, sha256, write_json


def pack_split(records, tokenizer, cfg):
    """Hash-assign complete documents; never split one document across fit and holdout."""
    names = ("train", "heldout")
    counts = dict(zip(names, (cfg["train_sequences"], cfg["heldout_sequences"])))
    buffers = {name: [] for name in names}
    output = {name: [] for name in names}
    documents = {name: set() for name in names}
    seen = set()
    length = cfg["sequence_length"]
    for record in records:
        text = record_to_text(record, text_field=cfg["text_field"])
        if text is None:
            continue
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in seen:
            continue
        seen.add(text_hash)
        split_hash = hashlib.sha256(f"{cfg['seed']}:{text_hash}".encode()).digest()
        fraction = int.from_bytes(split_hash[:8], "big") / 2**64
        name = "train" if fraction < cfg["document_train_fraction"] else "heldout"
        if len(output[name]) >= counts[name]:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        if tokenizer.eos_token_id is not None:
            ids = [*ids, tokenizer.eos_token_id]
        documents[name].add(text_hash)
        buffers[name].extend(ids)
        while len(buffers[name]) >= length and len(output[name]) < counts[name]:
            output[name].append(buffers[name][:length])
            del buffers[name][:length]
        if all(len(output[n]) == counts[n] for n in names):
            break
    if any(len(output[n]) != counts[n] for n in names):
        raise ValueError(f"not enough corpus data: { {n: len(output[n]) for n in names} }")
    if documents["train"] & documents["heldout"]:
        raise AssertionError("document leakage")
    train, test = (np.asarray(output[n], dtype=np.int32) for n in names)
    # Duplicate whole token sequences can occur even in distinct text documents.
    train_hashes = {hashlib.sha256(row.tobytes()).hexdigest() for row in train}
    test_hashes = {hashlib.sha256(row.tobytes()).hexdigest() for row in test}
    if train_hashes & test_hashes:
        raise ValueError("identical token sequences in train and heldout; change seed/corpus")
    return np.concatenate((train, test)), {n: sorted(documents[n]) for n in names}


def prepare(cfg, dataset, root):
    from transformers import AutoConfig, AutoTokenizer
    root = Path(root)
    dataset = Path(dataset).resolve()
    if not dataset.is_file():
        raise FileNotFoundError("pass one explicit local FineWeb-Edu Parquet/JSONL/JSON file")
    with lock(root / ".prepare.lock"):
        if (root / "prepare.json").exists():
            old = read_json(root / "prepare.json")
            if old["config"] != cfg or old["dataset_path"] != str(dataset):
                raise ValueError("run directory already belongs to a different experiment")
            if old["dataset_sha256"] != sha256(dataset) or old["tokens_sha256"] != sha256(root / "tokens.npy"):
                raise ValueError("corpus/tokens changed; use a fresh run directory")
            print("verified existing prepared tokens", flush=True)
            return
        model_info = {}
        tokenizers = []
        for role in ("source", "target"):
            model_id = cfg[f"{role}_model"]
            model_cfg = AutoConfig.from_pretrained(model_id, revision=cfg[f"{role}_revision"])
            revision = getattr(model_cfg, "_commit_hash", None)
            if not revision:
                raise ValueError("HF model revision could not be resolved to a commit")
            expected = (cfg[f"expected_{role}_layers"], cfg["expected_kv_heads"], cfg["expected_head_dim"])
            actual = (model_cfg.num_hidden_layers, model_cfg.num_key_value_heads, model_cfg.head_dim)
            if model_cfg.model_type != "qwen3" or actual != expected:
                raise ValueError(f"unexpected {role} architecture: {actual}, expected {expected}")
            if model_cfg.rope_scaling is not None or getattr(model_cfg, "use_sliding_window", False):
                raise ValueError("this reproduction requires default full-attention Qwen3 RoPE")
            if cfg["sequence_length"] > model_cfg.max_position_embeddings:
                raise ValueError("sequence length exceeds model position support")
            tokenizers.append(AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True))
            model_info[role] = {"model_id": model_id, "revision": revision, "config": model_cfg.to_dict()}
        st, tt = tokenizers
        if st.get_vocab() != tt.get_vocab() or st.all_special_ids != tt.all_special_ids:
            raise ValueError("source/target tokenizer vocabulary/special IDs differ")
        if st.backend_tokenizer.to_str() != tt.backend_tokenizer.to_str():
            raise ValueError("source/target tokenizer pipelines differ")
        tokens, documents = pack_split(iter_records(dataset), st, cfg)
        root.mkdir(parents=True, exist_ok=True)
        temporary = root / "tokens.pending.npy"
        np.save(temporary, tokens)
        temporary.replace(root / "tokens.npy")
        samples = len(range(0, cfg["sequence_length"], cfg["stride"]))
        manifest = {
            "schema_version": 1, "config": cfg, "models": model_info,
            "dataset_path": str(dataset), "dataset_sha256": sha256(dataset),
            "tokens_sha256": sha256(root / "tokens.npy"),
            "tokenizer_sha256": hashlib.sha256(st.backend_tokenizer.to_str().encode()).hexdigest(),
            "split": "document-disjoint seeded SHA256 assignment; fixed-length packing; no chat template",
            "documents": documents, "train_observations": cfg["train_sequences"] * samples,
            "heldout_observations": cfg["heldout_sequences"] * samples,
        }
        manifest["prepare_digest"] = digest(manifest)
        manifest["provenance"] = provenance()
        write_json(root / "prepare.json", manifest)
        print(f"tokens={tokens.shape}, fit N={manifest['train_observations']}, holdout N={manifest['heldout_observations']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare shared, document-disjoint calibration/heldout tokens")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare(config(args.config, args.smoke), args.dataset, args.run_dir)


if __name__ == "__main__":
    main()
