"""Offline smoke with tiny local Qwen3 checkpoints. No downloads or CUDA needed."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from proposed.src.exp.cachebridge.attention import QueryCapture, boundaries, raw_sensitivity, shrinkage
from proposed.src.exp.mapper_optim.cli import run
from proposed.src.exp.mapper_optim.regression import fit_ridge
from proposed.src.exp.rarecache.rank import disagreement, fit_basis, select_positions
from proposed.src.exp.rarecache.recompute import recompute


class MapperOptimSmoke(unittest.TestCase):
    def test_weighted_ridge_matches_dense_and_rank_score(self):
        rng = torch.Generator().manual_seed(42)
        x = torch.randn(13, 5, generator=rng, dtype=torch.float64)
        y = torch.randn(13, 4, generator=rng, dtype=torch.float64)
        w = torch.linspace(0.0, 2.0, 13, dtype=torch.float64).reshape(-1, 1)

        def batches():
            yield x[:6], y[:6], w[:6]
            yield x[6:], y[6:], w[6:]

        weight, bias, mx, my, gram = fit_ridge(
            batches, (5, 4), ridge=.01, device=torch.device("cpu"), dtype=torch.float64)
        xc, yc = x - (x * w).sum(0) / w.sum(), y - (y * w).sum(0) / w.sum()
        expected = torch.linalg.solve(xc.T @ (xc * w) + .01 * torch.eye(5, dtype=torch.float64),
                                      xc.T @ (yc * w))
        torch.testing.assert_close(weight, expected)
        torch.testing.assert_close(bias, my - mx @ expected)
        subspace = fit_basis(weight, gram, 2)
        centered = x @ weight + bias - my
        explicit = (centered - (centered @ subspace) @ subspace.T).square().sum(-1)
        torch.testing.assert_close(disagreement(x @ weight + bias, my, subspace), explicit)

    def test_attention_weights_match_direct_formula(self):
        rng = torch.Generator().manual_seed(6)
        q = torch.randn(4, 3, generator=rng).reshape(2, 2, 3)
        k = torch.randn(2, 7, 3, generator=rng)
        v = torch.randn(2, 7, 3, generator=rng)
        p = torch.tensor([0, 3, 6])
        a = (q @ k.transpose(-1, -2) / (3 ** .5)).softmax(-1)
        o = a @ v
        delta2 = (v[:, None, p, :] - o[:, :, None, :]).square().sum(-1)
        rk = (a[..., p].square() * delta2 * q.square().sum(-1, keepdim=True) / 3).sum(1).T
        rv = a[..., p].square().sum(1).T
        expected = torch.stack([rk / rk.sum(0), rv / rv.sum(0)])
        torch.testing.assert_close(raw_sensitivity(q, k, v, p, 3 ** -.5), expected)
        raw = np.r_[0., 0., 0., 0., 0., 6., 0., 0.]
        mean, alpha = shrinkage(raw, 3)
        weights = 1 + alpha * (raw / mean - 1)
        self.assertAlmostEqual(weights.sum() ** 2 / (weights ** 2).sum(), 6.)

    def test_local_qwen_pipeline(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, target = root / "s", root / "t"
            self._model(source, layers=2)
            self._model(target, layers=3)
            fit_set, eval_set = root / "fit.jsonl", root / "eval.jsonl"
            fit_set.write_text("".join(json.dumps({"text": " ".join(f"t{(i * 13 + j * (i + 7)) % 50}"
                                                for j in range(24))}) + "\n" for i in range(6)))
            eval_set.write_text(json.dumps({"text": " ".join(f"t{49 - j}" for j in range(24))}) + "\n")
            options = {
                "rarecache": dict(basis=fit_basis, rank_score=disagreement,
                                  replay=(select_positions, recompute)),
                "cachebridge": dict(attention=(QueryCapture, boundaries, shrinkage)),
            }
            cases = [
                ("rarecache", None, ["--rank", "4"]),
                ("cachebridge", "head-local", ["--variant", "head-local"]),
                ("cachebridge", "attn-repair", ["--variant", "attn-repair",
                                                "--first-boundary", "2", "--boundaries", "5"]),
                ("cachebridge", "full-head", ["--variant", "full-head"]),
            ]
            size = {}
            for method, variant, extra in cases:
                tag = variant or method
                fit = root / (tag + "_fit")
                ev = root / (tag + "_eval")
                args = ["--source-model", str(source), "--target-model", str(target),
                        "--calibration-jsonl", str(fit_set), "--sequence-length", "24",
                        "--sequences", "6", "--stride", "2", "--top-k", "2",
                        "--probe-rows", "36", "--chunk-rows", "8", "--dtype", "float32"]
                with patch("sys.argv", ["test", "fit", "--output", str(fit), *args, *extra]):
                    run(method, device_name="cpu", **options[method])
                summary = json.loads((fit / "fit_summary.json").read_text())
                size[tag] = summary["safetensors_bytes"]
                if variant == "attn-repair":
                    self.assertEqual(len(summary["weight_shrinkage"]), 12)
                    self.assertTrue(any(0 < row["alpha"] <= 1 for row in summary["weight_shrinkage"]))
                args = ["--artifact", str(fit / "artifact"), "--eval-jsonl", str(eval_set),
                        "--prefix-length", "12", "--continuation-tokens", "4", "--examples", "1",
                        "--warmup", "0", "--repeats", "1", "--decode-tokens", "3", "--dtype", "float32"]
                if method == "rarecache":
                    args += ["--recompute-ratios", "0", "0.3", "1"]
                with patch("sys.argv", ["test", "eval", "--output", str(ev), *args]):
                    run(method, device_name="cpu", **options[method])
                row = json.loads((ev / "evaluation.json").read_text())["examples"][0]
                self.assertEqual(len(row["native"]["greedy_token_ids"]), 3)
                if method == "rarecache":
                    self.assertEqual(len(row["mapped"][0]["quality"]["recomputed_positions"]), 0)
                    self.assertEqual(len(row["mapped"][1]["quality"]["recomputed_positions"]), 4)
                    self.assertEqual(len(row["mapped"][2]["quality"]["recomputed_positions"]), 11)
                    self.assertAlmostEqual(row["mapped"][2]["quality"]["nll_delta"], 0., places=5)
                    self.assertEqual(row["mapped"][2]["benchmark"]["greedy_token_ids"],
                                     row["native"]["greedy_token_ids"])
            self.assertLess(size["head-local"], size["full-head"])

    @staticmethod
    def _model(path: Path, layers: int):
        vocab = {f"t{i}": i for i in range(63)} | {"[UNK]": 63}
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(path)
        cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=layers,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                          max_position_embeddings=64)
        Qwen3ForCausalLM(cfg).save_pretrained(path)
