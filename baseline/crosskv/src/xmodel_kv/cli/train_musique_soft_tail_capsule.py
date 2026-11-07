from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

from xmodel_kv.capsule_codec import pack_int4_capsule, unpack_int4_capsule
from xmodel_kv.coqa import coqa_f1
from xmodel_kv.composable_delta import compose_int4_base_delta
from xmodel_kv.differentiable_delta import compose_fake_int4_base_delta
from xmodel_kv.native_tail_distillation import canonical_int4_tail_state_loss
from xmodel_kv.policy_imprinting import build_chat_segments, prefill_legacy_cache
from xmodel_kv.semantic_kv_summary import concatenate_caches, teacher_forced_answer_logits
from xmodel_kv.soft_tail_capsule import (
    BoundaryKVWriteAdapter,
    LayerwiseKVWriteAdapter,
    SoftTailCapsule,
    detach_cache_layer_prefix,
    load_soft_tail_capsule,
    materialize_token_tail,
    move_legacy_cache,
    reposition_tail_cache,
)

from .evaluate_musique_agent_a import SOURCE_ANSWER_SYSTEM
from .evaluate_amortized_musique_handoff import _clean_generation, _format_target
from .evaluate_musique_tail_transplant import (
    active_layer_indices,
    mask_and_fake_quantize_suffix,
)
from .probe_semantic_kv_summary import _generate, _normalized
from .train_musique_semantic_handoff import (
    DEPENDENT_B_SYSTEM,
    INTERMEDIATE_B_SYSTEM,
    INTERMEDIATE_QUERY,
    _load_model,
    _load_slice,
    _system_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train shared soft state-emission tokens whose contextual native KV is "
            "transplanted before Agent A emits any answer text."
        )
    )
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--eval-offset", type=int, default=32)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--writer-rank", type=int, default=0)
    parser.add_argument("--writer-scale", type=float, default=1.0)
    parser.add_argument("--boundary-writer-rank", type=int, default=0)
    parser.add_argument("--boundary-writer-layer", type=int, default=-1)
    parser.add_argument(
        "--freeze-initialized-base",
        action="store_true",
        help=(
            "Freeze initialized slot embeddings and the layerwise writer; train "
            "only a newly requested boundary writer."
        ),
    )
    parser.add_argument(
        "--initialize-checkpoint",
        default=None,
        help="Optional compatible capsule checkpoint used as the trainable initialization.",
    )
    parser.add_argument(
        "--expand-initialized-slots",
        action="store_true",
        help=(
            "Expand a smaller initialized capsule to --slots by cycling its "
            "learned embeddings while preserving the complete writer."
        ),
    )
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--stage-weight", type=float, default=1.0)
    parser.add_argument("--bridge-weight", type=float, default=1.0)
    parser.add_argument(
        "--teacher-kl-weight",
        type=float,
        default=0.0,
        help="Behavioral KL weight from a gold-answer Tail-KV teacher.",
    )
    parser.add_argument("--teacher-temperature", type=float, default=2.0)
    parser.add_argument(
        "--native-tail-state-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for direct canonical K4/V4 state matching against the gold "
            "answer Tail-KV teacher. Layer zero is excluded by default because "
            "fixed soft embeddings cannot encode example-specific answer tokens."
        ),
    )
    parser.add_argument(
        "--native-tail-state-max-tokens",
        type=int,
        default=0,
        help="Teacher answer-token cap; zero uses the student slot count.",
    )
    parser.add_argument(
        "--native-tail-state-start-layer",
        type=int,
        default=1,
        help="First K/V layer included in direct Tail-KV state matching.",
    )
    parser.add_argument(
        "--eval-native-tail-state",
        action="store_true",
        help="Report held-out canonical K4/V4 Tail-state relative MSE.",
    )
    parser.add_argument(
        "--preanswer-teacher-checkpoint",
        default=None,
        help=(
            "Optional frozen soft-capsule teacher. Its pre-answer receiver "
            "behavior is distilled without exposing answer-token Tail-KV."
        ),
    )
    parser.add_argument(
        "--preanswer-teacher-kl-weight",
        type=float,
        default=0.0,
        help="Behavioral KL weight from the frozen pre-answer capsule teacher.",
    )
    parser.add_argument(
        "--orbit-base-checkpoint",
        default=None,
        help=(
            "Frozen reusable base for positional-orbit distillation of an "
            "independently quantized canonical task delta."
        ),
    )
    parser.add_argument(
        "--orbit-delta-layers",
        default=None,
        help="Comma-separated delta layers used by positional-orbit distillation.",
    )
    parser.add_argument(
        "--orbit-kl-weight",
        type=float,
        default=0.0,
        help="KL weight from target-frame base+delta to canonical base+delta logits.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--train-layer-pattern",
        default="all",
        help="Layer subset exposed to receiver losses during training.",
    )
    parser.add_argument(
        "--train-layer-patterns",
        default=None,
        help=(
            "Comma-separated layer subsets sampled uniformly per update. "
            "Mutually exclusive with a non-default --train-layer-pattern."
        ),
    )
    parser.add_argument(
        "--train-layer-bridge-weights",
        default=None,
        help=(
            "Optional comma-separated pattern:weight map for budget-conditioned "
            "bridge loss, for example first_16:0,all:4."
        ),
    )
    parser.add_argument(
        "--train-depth-gradient-boundary",
        type=int,
        default=0,
        help=(
            "If positive, deep-budget updates stop gradients through the first "
            "N emitted cache layers and through sender hidden state before layer N. "
            "This trains the deep packet as a residual over an isolated early base."
        ),
    )
    parser.add_argument(
        "--source-read-layers",
        type=int,
        default=None,
        help=(
            "If positive, only the first N sender emission layers may attend the "
            "long source cache; deeper layers transform capsule state only. Zero "
            "explicitly disables a bottleneck when continuing a checkpoint."
        ),
    )
    parser.add_argument(
        "--train-quant-bits",
        type=int,
        choices=(4, 8, 16),
        default=16,
        help="Straight-through packet precision exposed during training.",
    )
    parser.add_argument("--eval-quant-bits", default="16,4")
    parser.add_argument("--eval-layer-patterns", default="all")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--resident-source-caches", action="store_true")
    parser.add_argument(
        "--resident-source-cache-storage",
        choices=("device", "cpu"),
        default="device",
        help="Keep resident source caches on the model device or offload them to CPU.",
    )
    args = parser.parse_args()
    positive = (
        args.train_count,
        args.eval_count,
        args.slots,
        args.steps,
        args.learning_rate,
        args.grad_clip,
        args.max_new_tokens,
        args.log_every,
    )
    if min(positive) <= 0 or min(args.train_offset, args.eval_offset) < 0:
        parser.error("counts, rates, and limits must be positive; offsets non-negative")
    if (
        args.writer_rank < 0
        or args.boundary_writer_rank < 0
        or args.writer_scale <= 0
    ):
        parser.error("writer rank must be non-negative and scale positive")
    if bool(args.boundary_writer_rank) != (args.boundary_writer_layer >= 0):
        parser.error(
            "positive --boundary-writer-rank and non-negative "
            "--boundary-writer-layer must be used together"
        )
    if args.freeze_initialized_base and not args.initialize_checkpoint:
        parser.error(
            "--freeze-initialized-base requires an initialization checkpoint"
        )
    if args.expand_initialized_slots and not args.initialize_checkpoint:
        parser.error("--expand-initialized-slots requires an initialization checkpoint")
    if args.train_depth_gradient_boundary < 0:
        parser.error("--train-depth-gradient-boundary must be non-negative")
    if args.source_read_layers is not None and args.source_read_layers < 0:
        parser.error("--source-read-layers must be non-negative")
    if args.train_depth_gradient_boundary and not args.writer_rank:
        parser.error("depth-gradient isolation requires a positive --writer-rank")
    if min(
        args.weight_decay,
        args.stage_weight,
        args.bridge_weight,
        args.teacher_kl_weight,
        args.native_tail_state_weight,
        args.preanswer_teacher_kl_weight,
        args.orbit_kl_weight,
    ) < 0:
        parser.error("weights must be non-negative")
    if args.teacher_temperature <= 0:
        parser.error("--teacher-temperature must be positive")
    if args.native_tail_state_max_tokens < 0 or args.native_tail_state_start_layer < 0:
        parser.error("native Tail-state token cap and start layer must be non-negative")
    native_tail_state_max_tokens = (
        args.slots
        if args.native_tail_state_max_tokens == 0
        else args.native_tail_state_max_tokens
    )
    if native_tail_state_max_tokens > args.slots:
        parser.error("native Tail-state teacher cap cannot exceed student slots")
    if args.native_tail_state_weight and args.train_quant_bits != 4:
        parser.error("native Tail-state matching requires production-rate train INT4")
    if bool(args.preanswer_teacher_checkpoint) != bool(
        args.preanswer_teacher_kl_weight
    ):
        parser.error(
            "--preanswer-teacher-checkpoint and a positive "
            "--preanswer-teacher-kl-weight must be used together"
        )
    orbit_arguments = (
        bool(args.orbit_base_checkpoint),
        bool(args.orbit_delta_layers),
        bool(args.orbit_kl_weight),
    )
    if len(set(orbit_arguments)) != 1:
        parser.error(
            "--orbit-base-checkpoint, --orbit-delta-layers, and a positive "
            "--orbit-kl-weight must be used together"
        )
    orbit_delta_layers = None
    if args.orbit_delta_layers:
        raw_orbit_layers = args.orbit_delta_layers.split(",")
        try:
            orbit_delta_layers = frozenset(int(value) for value in raw_orbit_layers)
        except ValueError:
            parser.error("--orbit-delta-layers must contain comma-separated integers")
        if (
            not orbit_delta_layers
            or min(orbit_delta_layers) < 0
            or len(orbit_delta_layers) != len(raw_orbit_layers)
        ):
            parser.error("--orbit-delta-layers must contain unique non-negative integers")
    if args.orbit_kl_weight and (
        args.teacher_kl_weight or args.preanswer_teacher_kl_weight
    ):
        parser.error("orbit KL cannot be combined with another teacher in this study")
    if not args.resident_source_caches and args.resident_source_cache_storage != "device":
        parser.error("CPU cache storage requires --resident-source-caches")
    quant_bits_values = tuple(int(value) for value in args.eval_quant_bits.split(","))
    eval_layer_patterns = tuple(
        value.strip() for value in args.eval_layer_patterns.split(",") if value.strip()
    )
    try:
        train_layer_patterns = resolve_train_layer_patterns(
            args.train_layer_pattern, args.train_layer_patterns
        )
        train_layer_bridge_weights = resolve_train_layer_bridge_weights(
            train_layer_patterns,
            args.train_layer_bridge_weights,
            default_weight=args.bridge_weight,
        )
    except ValueError as error:
        parser.error(str(error))
    if (
        not quant_bits_values
        or any(value not in (4, 8, 16) for value in quant_bits_values)
        or len(set(quant_bits_values)) != len(quant_bits_values)
    ):
        parser.error("--eval-quant-bits must be a unique subset of 16,8,4")
    if not eval_layer_patterns or len(set(eval_layer_patterns)) != len(
        eval_layer_patterns
    ):
        parser.error("--eval-layer-patterns must contain unique non-empty patterns")
    if args.device.startswith("cuda"):
        # Keep synchronization/empty-cache calls and small implicit buffers on
        # the requested worker GPU instead of creating a context on cuda:0.
        torch.cuda.set_device(torch.device(args.device))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "train_layer_patterns": list(train_layer_patterns),
        "train_layer_bridge_weights": train_layer_bridge_weights,
        "eval_quant_bits": list(quant_bits_values),
        "eval_layer_patterns": list(eval_layer_patterns),
        "orbit_delta_layers": (
            None if orbit_delta_layers is None else sorted(orbit_delta_layers)
        ),
        "native_tail_state_max_tokens_resolved": native_tail_state_max_tokens,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, args.device, getattr(torch, args.dtype))
    if args.native_tail_state_start_layer >= model.config.num_hidden_layers:
        parser.error("native Tail-state start layer must be below model depth")
    native_tail_state_layers = frozenset(
        range(args.native_tail_state_start_layer, model.config.num_hidden_layers)
    )
    train_active_layers_by_pattern = {
        layer_pattern: active_layer_indices(
            layer_pattern, total_layers=model.config.num_hidden_layers
        )
        for layer_pattern in train_layer_patterns
    }
    for layer_pattern in eval_layer_patterns:
        active_layer_indices(
            layer_pattern, total_layers=model.config.num_hidden_layers
        )
    train_active_heads = frozenset(range(model.config.num_key_value_heads))
    if orbit_delta_layers is not None:
        if args.train_quant_bits != 4:
            parser.error("positional-orbit distillation requires train INT4")
        if len(train_layer_patterns) != 1:
            parser.error("positional-orbit distillation requires one layer pattern")
        orbit_base_layers = train_active_layers_by_pattern[train_layer_patterns[0]]
        if not orbit_delta_layers.issubset(orbit_base_layers):
            parser.error("orbit delta layers must be included in the training packet")
    else:
        orbit_base_layers = None
    if args.train_depth_gradient_boundary:
        if args.train_depth_gradient_boundary >= model.config.num_hidden_layers:
            parser.error(
                "--train-depth-gradient-boundary must be below the model layer count"
            )
        if not any(
            any(
                layer_index >= args.train_depth_gradient_boundary
                for layer_index in active_layers
            )
            for active_layers in train_active_layers_by_pattern.values()
        ):
            parser.error(
                "depth-gradient isolation requires a training pattern with deep layers"
            )
    if args.initialize_checkpoint:
        capsule, initialization_config = load_soft_tail_capsule(
            args.initialize_checkpoint
        )
        if args.expand_initialized_slots:
            original_slots = capsule.slots
            expand_soft_tail_capsule_slots(capsule, slots=args.slots)
            config["initialize_checkpoint_original_slots"] = original_slots
        _validate_initialized_capsule(
            capsule,
            initialization_config=initialization_config,
            model=model,
            model_path=args.model,
            slots=args.slots,
            writer_rank=args.writer_rank,
            writer_scale=args.writer_scale,
        )
        config["initialize_checkpoint_sha256"] = hashlib.sha256(
            Path(args.initialize_checkpoint).read_bytes()
        ).hexdigest()
    else:
        initial = _initial_embeddings(model, tokenizer, slots=args.slots)
        writer = None
        if args.writer_rank:
            writer = LayerwiseKVWriteAdapter(
                hidden_size=model.config.hidden_size,
                num_hidden_layers=model.config.num_hidden_layers,
                rank=args.writer_rank,
                scale=args.writer_scale,
            )
        capsule = SoftTailCapsule(initial, write_adapter=writer)
    if args.source_read_layers is not None:
        capsule.source_read_layers = args.source_read_layers or None
    if capsule.source_read_layers is not None and not (
        1 <= capsule.source_read_layers < model.config.num_hidden_layers
    ):
        parser.error(
            "--source-read-layers must leave at least one source-reading and "
            "one source-masked model layer"
        )
    if args.boundary_writer_rank:
        if capsule.boundary_write_adapter is not None:
            parser.error("initialization checkpoint already contains a boundary writer")
        if args.boundary_writer_layer >= model.config.num_hidden_layers - 1:
            parser.error("boundary writer must leave a following K/V layer")
        if capsule.source_read_layers != args.boundary_writer_layer + 1:
            parser.error(
                "emission-aligned boundary writer layer must equal "
                "source_read_layers - 1"
            )
        capsule.boundary_write_adapter = BoundaryKVWriteAdapter(
            hidden_size=model.config.hidden_size,
            layer_index=args.boundary_writer_layer,
            rank=args.boundary_writer_rank,
            scale=args.writer_scale,
        ).to(args.device)
    if args.freeze_initialized_base:
        try:
            freeze_initialized_base(capsule)
        except ValueError as error:
            parser.error(str(error))
    if orbit_delta_layers is not None:
        if capsule.boundary_write_adapter is None:
            parser.error("positional-orbit training requires a boundary writer")
        emitted_layer = capsule.boundary_write_adapter.layer_index + 1
        if orbit_delta_layers != frozenset({emitted_layer}):
            parser.error(
                "orbit delta layers must contain exactly the boundary-emission layer"
            )
    trainable_parameters = [
        parameter for parameter in capsule.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        parser.error("capsule has no trainable parameters")
    config["trainable_parameter_count"] = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    config["effective_source_read_layers"] = capsule.source_read_layers
    preanswer_teacher = None
    if args.preanswer_teacher_checkpoint:
        preanswer_teacher, preanswer_teacher_config = load_soft_tail_capsule(
            args.preanswer_teacher_checkpoint
        )
        _validate_initialized_capsule(
            preanswer_teacher,
            initialization_config=preanswer_teacher_config,
            model=model,
            model_path=args.model,
            slots=args.slots,
            writer_rank=args.writer_rank,
            writer_scale=args.writer_scale,
        )
        preanswer_teacher = preanswer_teacher.to(args.device).eval()
        preanswer_teacher.requires_grad_(False)
        config["preanswer_teacher_checkpoint_sha256"] = hashlib.sha256(
            Path(args.preanswer_teacher_checkpoint).read_bytes()
        ).hexdigest()
        config["preanswer_teacher_source_read_layers"] = (
            preanswer_teacher.source_read_layers
        )
    orbit_base_capsule = None
    if args.orbit_base_checkpoint:
        orbit_base_capsule, orbit_base_config = load_soft_tail_capsule(
            args.orbit_base_checkpoint
        )
        _validate_initialized_capsule(
            orbit_base_capsule,
            initialization_config=orbit_base_config,
            model=model,
            model_path=args.model,
            slots=args.slots,
            writer_rank=args.writer_rank,
            writer_scale=args.writer_scale,
        )
        if orbit_base_capsule.boundary_write_adapter is not None:
            parser.error("orbit base checkpoint must not contain a boundary writer")
        orbit_base_capsule = orbit_base_capsule.to(args.device).eval()
        orbit_base_capsule.requires_grad_(False)
        config["orbit_base_checkpoint_sha256"] = hashlib.sha256(
            Path(args.orbit_base_checkpoint).read_bytes()
        ).hexdigest()
    capsule = capsule.to(args.device)
    frozen_capsule_sha256_before = parameter_sha256(
        capsule, requires_grad=False
    )
    trainable_capsule_sha256_before = parameter_sha256(
        capsule, requires_grad=True
    )
    orbit_base_sha256_before = (
        None if orbit_base_capsule is None else parameter_sha256(orbit_base_capsule)
    )
    config["frozen_capsule_parameter_sha256_before"] = (
        frozen_capsule_sha256_before
    )
    config["trainable_capsule_parameter_sha256_before"] = (
        trainable_capsule_sha256_before
    )
    config["orbit_base_parameter_sha256_before"] = orbit_base_sha256_before
    stage_system_ids, stage_system_cache = _system_cache(
        model, tokenizer, DEPENDENT_B_SYSTEM, "placeholder"
    )
    bridge_system_ids, bridge_system_cache = _system_cache(
        model, tokenizer, INTERMEDIATE_B_SYSTEM, INTERMEDIATE_QUERY
    )

    train_cases = _load_slice(
        args.train_dataset, offset=args.train_offset, count=args.train_count
    )
    eval_cases = _load_slice(
        args.eval_dataset, offset=args.eval_offset, count=args.eval_count
    )
    if {case["id"] for _, case in train_cases} & {case["id"] for _, case in eval_cases}:
        parser.error("train and eval case IDs overlap")
    config["train_case_ids"] = [case["id"] for _, case in train_cases]
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    train_documents = [
        _prepare_document(
            tokenizer,
            index=index,
            case=case,
            stage_system_ids=stage_system_ids,
            bridge_system_ids=bridge_system_ids,
        )
        for index, case in train_cases
    ]
    resident_source_cache_bytes = 0
    if args.resident_source_caches:
        for position, document in enumerate(train_documents, 1):
            document["source_cache"] = _prefill_detached_cache(
                model, document["source_ids"]
            )
            if args.resident_source_cache_storage == "cpu":
                document["source_cache"] = move_legacy_cache(
                    document["source_cache"], "cpu"
                )
            resident_source_cache_bytes += _cache_bytes(document["source_cache"])
            print(
                json.dumps(
                    {
                        "event": "prepared_source_cache",
                        "position": position,
                        "resident_source_cache_gib": resident_source_cache_bytes / 2**30,
                        "storage": args.resident_source_cache_storage,
                    }
                ),
                flush=True,
            )

    optimizer = AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    schedule = list(train_documents)
    generator = random.Random(args.seed)
    layer_pattern_generator = random.Random(args.seed + 1)
    train_layer_pattern_counts = dict.fromkeys(train_layer_patterns, 0)
    totals = {
        "loss": 0.0,
        "stage_ce": 0.0,
        "bridge_ce": 0.0,
        "teacher_kl": 0.0,
        "preanswer_teacher_kl": 0.0,
        "orbit_kl": 0.0,
        "native_tail_state": 0.0,
        "grad_norm": 0.0,
    }
    training_rows = []
    started = time.perf_counter()
    capsule.train()
    for step in range(1, args.steps + 1):
        if (step - 1) % len(schedule) == 0:
            generator.shuffle(schedule)
        document = schedule[(step - 1) % len(schedule)]
        train_layer_pattern = layer_pattern_generator.choice(train_layer_patterns)
        train_active_layers = train_active_layers_by_pattern[train_layer_pattern]
        train_bridge_weight = train_layer_bridge_weights[train_layer_pattern]
        train_gradient_boundary = (
            args.train_depth_gradient_boundary
            if args.train_depth_gradient_boundary
            and any(
                layer_index >= args.train_depth_gradient_boundary
                for layer_index in train_active_layers
            )
            else None
        )
        train_layer_pattern_counts[train_layer_pattern] += 1
        source_cache = document.get("source_cache")
        if source_cache is None:
            source_cache = _prefill_detached_cache(model, document["source_ids"])
        elif args.resident_source_cache_storage == "cpu":
            source_cache = move_legacy_cache(
                source_cache, model.model.embed_tokens.weight.device
            )
        optimizer.zero_grad(set_to_none=True)
        source_tail = capsule.materialize(
            model,
            prefix_cache=source_cache,
            prefix_tokens=len(document["source_ids"]),
            gradient_boundary_layers=train_gradient_boundary,
        )
        if train_gradient_boundary is not None:
            source_tail = detach_cache_layer_prefix(
                source_tail, train_gradient_boundary
            )
        orbit_base_source_tail = None
        if orbit_base_capsule is not None:
            with torch.no_grad():
                orbit_base_source_tail = orbit_base_capsule.materialize(
                    model,
                    prefix_cache=source_cache,
                    prefix_tokens=len(document["source_ids"]),
                )
        orbit_teacher_stage_tail = None
        orbit_teacher_bridge_tail = None
        orbit_teacher_stage_logits = None
        orbit_teacher_bridge_logits = None
        if orbit_base_source_tail is None:
            stage_tail = reposition_tail_cache(
                model,
                source_tail,
                source_start=len(document["source_ids"]),
                target_start=len(stage_system_ids),
            )
            stage_tail = mask_and_fake_quantize_suffix(
                stage_tail,
                suffix_tokens=args.slots,
                bits=args.train_quant_bits,
                active_layers=train_active_layers,
                active_heads=train_active_heads,
                straight_through=True,
            )
        else:
            stage_tail, orbit_teacher_stage_tail = positional_orbit_caches(
                model,
                base_source_tail=orbit_base_source_tail,
                task_source_tail=source_tail,
                source_start=len(document["source_ids"]),
                target_start=len(stage_system_ids),
                slots=args.slots,
                base_layers=orbit_base_layers,
                delta_layers=orbit_delta_layers,
            )
        if args.stage_weight:
            stage_logits = _task_logits(
                model,
                prefix_cache=stage_system_cache,
                prefix_tokens=len(stage_system_ids),
                compact_cache=stage_tail,
                slots=args.slots,
                query_ids=document["stage_query_ids"],
                answer_ids=document["stage_answer_ids"],
            )
            stage_ce = _answer_ce(stage_logits, document["stage_answer_ids"])
            if orbit_teacher_stage_tail is not None:
                with torch.no_grad():
                    orbit_teacher_stage_logits = _task_logits(
                        model,
                        prefix_cache=stage_system_cache,
                        prefix_tokens=len(stage_system_ids),
                        compact_cache=orbit_teacher_stage_tail,
                        slots=args.slots,
                        query_ids=document["stage_query_ids"],
                        answer_ids=document["stage_answer_ids"],
                    )
        else:
            stage_logits = None
            stage_ce = source_tail[0][0].sum() * 0
        if train_bridge_weight:
            if orbit_base_source_tail is None:
                bridge_tail = reposition_tail_cache(
                    model,
                    source_tail,
                    source_start=len(document["source_ids"]),
                    target_start=len(bridge_system_ids),
                )
                bridge_tail = mask_and_fake_quantize_suffix(
                    bridge_tail,
                    suffix_tokens=args.slots,
                    bits=args.train_quant_bits,
                    active_layers=train_active_layers,
                    active_heads=train_active_heads,
                    straight_through=True,
                )
            else:
                bridge_tail, orbit_teacher_bridge_tail = positional_orbit_caches(
                    model,
                    base_source_tail=orbit_base_source_tail,
                    task_source_tail=source_tail,
                    source_start=len(document["source_ids"]),
                    target_start=len(bridge_system_ids),
                    slots=args.slots,
                    base_layers=orbit_base_layers,
                    delta_layers=orbit_delta_layers,
                )
            bridge_logits = _task_logits(
                model,
                prefix_cache=bridge_system_cache,
                prefix_tokens=len(bridge_system_ids),
                compact_cache=bridge_tail,
                slots=args.slots,
                query_ids=document["bridge_query_ids"],
                answer_ids=document["bridge_answer_ids"],
            )
            bridge_ce = _answer_ce(bridge_logits, document["bridge_answer_ids"])
            if orbit_teacher_bridge_tail is not None:
                with torch.no_grad():
                    orbit_teacher_bridge_logits = _task_logits(
                        model,
                        prefix_cache=bridge_system_cache,
                        prefix_tokens=len(bridge_system_ids),
                        compact_cache=orbit_teacher_bridge_tail,
                        slots=args.slots,
                        query_ids=document["bridge_query_ids"],
                        answer_ids=document["bridge_answer_ids"],
                    )
        else:
            bridge_tail = None
            bridge_logits = None
            bridge_ce = source_tail[0][0].sum() * 0
        teacher_kl = source_tail[0][0].sum() * 0
        if args.teacher_kl_weight:
            with torch.no_grad():
                teacher_source_tail = materialize_token_tail(
                    model,
                    document["bridge_answer_ids"],
                    prefix_cache=source_cache,
                    prefix_tokens=len(document["source_ids"]),
                )
                if args.stage_weight:
                    teacher_stage_tail = reposition_tail_cache(
                        model,
                        teacher_source_tail,
                        source_start=len(document["source_ids"]),
                        target_start=len(stage_system_ids),
                    )
                    teacher_stage_logits = _task_logits(
                        model,
                        prefix_cache=stage_system_cache,
                        prefix_tokens=len(stage_system_ids),
                        compact_cache=teacher_stage_tail,
                        slots=len(document["bridge_answer_ids"]),
                        query_ids=document["stage_query_ids"],
                        answer_ids=document["stage_answer_ids"],
                    )
                if train_bridge_weight:
                    teacher_bridge_tail = reposition_tail_cache(
                        model,
                        teacher_source_tail,
                        source_start=len(document["source_ids"]),
                        target_start=len(bridge_system_ids),
                    )
                    teacher_bridge_logits = _task_logits(
                        model,
                        prefix_cache=bridge_system_cache,
                        prefix_tokens=len(bridge_system_ids),
                        compact_cache=teacher_bridge_tail,
                        slots=len(document["bridge_answer_ids"]),
                        query_ids=document["bridge_query_ids"],
                        answer_ids=document["bridge_answer_ids"],
                    )
            if args.stage_weight:
                teacher_kl = teacher_kl + args.stage_weight * behavioral_kl(
                    stage_logits,
                    teacher_stage_logits,
                    temperature=args.teacher_temperature,
                )
            if train_bridge_weight:
                teacher_kl = teacher_kl + train_bridge_weight * behavioral_kl(
                    bridge_logits,
                    teacher_bridge_logits,
                    temperature=args.teacher_temperature,
                )
        native_tail_state = source_tail[0][0].sum() * 0
        if args.native_tail_state_weight:
            teacher_ids = document["bridge_answer_ids"][
                :native_tail_state_max_tokens
            ]
            with torch.no_grad():
                native_teacher_tail = materialize_token_tail(
                    model,
                    teacher_ids,
                    prefix_cache=source_cache,
                    prefix_tokens=len(document["source_ids"]),
                )
            native_tail_state = canonical_int4_tail_state_loss(
                model,
                source_tail,
                native_teacher_tail,
                source_start=len(document["source_ids"]),
                active_layers=native_tail_state_layers,
                straight_through=True,
            )
        preanswer_teacher_kl = source_tail[0][0].sum() * 0
        if preanswer_teacher is not None:
            with torch.no_grad():
                preanswer_source_tail = preanswer_teacher.materialize(
                    model,
                    prefix_cache=source_cache,
                    prefix_tokens=len(document["source_ids"]),
                )
                if args.stage_weight:
                    preanswer_stage_tail = reposition_tail_cache(
                        model,
                        preanswer_source_tail,
                        source_start=len(document["source_ids"]),
                        target_start=len(stage_system_ids),
                    )
                    preanswer_stage_tail = mask_and_fake_quantize_suffix(
                        preanswer_stage_tail,
                        suffix_tokens=args.slots,
                        bits=args.train_quant_bits,
                        active_layers=train_active_layers,
                        active_heads=train_active_heads,
                    )
                    preanswer_stage_logits = _task_logits(
                        model,
                        prefix_cache=stage_system_cache,
                        prefix_tokens=len(stage_system_ids),
                        compact_cache=preanswer_stage_tail,
                        slots=args.slots,
                        query_ids=document["stage_query_ids"],
                        answer_ids=document["stage_answer_ids"],
                    )
                if train_bridge_weight:
                    preanswer_bridge_tail = reposition_tail_cache(
                        model,
                        preanswer_source_tail,
                        source_start=len(document["source_ids"]),
                        target_start=len(bridge_system_ids),
                    )
                    preanswer_bridge_tail = mask_and_fake_quantize_suffix(
                        preanswer_bridge_tail,
                        suffix_tokens=args.slots,
                        bits=args.train_quant_bits,
                        active_layers=train_active_layers,
                        active_heads=train_active_heads,
                    )
                    preanswer_bridge_logits = _task_logits(
                        model,
                        prefix_cache=bridge_system_cache,
                        prefix_tokens=len(bridge_system_ids),
                        compact_cache=preanswer_bridge_tail,
                        slots=args.slots,
                        query_ids=document["bridge_query_ids"],
                        answer_ids=document["bridge_answer_ids"],
                    )
            if args.stage_weight:
                preanswer_teacher_kl = (
                    preanswer_teacher_kl
                    + args.stage_weight
                    * behavioral_kl(
                        stage_logits,
                        preanswer_stage_logits,
                        temperature=args.teacher_temperature,
                    )
                )
            if train_bridge_weight:
                preanswer_teacher_kl = (
                    preanswer_teacher_kl
                    + train_bridge_weight
                    * behavioral_kl(
                        bridge_logits,
                        preanswer_bridge_logits,
                        temperature=args.teacher_temperature,
                    )
                )
        orbit_kl = source_tail[0][0].sum() * 0
        if orbit_base_source_tail is not None:
            if args.stage_weight:
                orbit_kl = orbit_kl + args.stage_weight * behavioral_kl(
                    stage_logits,
                    orbit_teacher_stage_logits,
                    temperature=args.teacher_temperature,
                )
            if train_bridge_weight:
                orbit_kl = orbit_kl + train_bridge_weight * behavioral_kl(
                    bridge_logits,
                    orbit_teacher_bridge_logits,
                    temperature=args.teacher_temperature,
                )
        loss = (
            args.stage_weight * stage_ce
            + train_bridge_weight * bridge_ce
            + args.teacher_kl_weight * teacher_kl
            + args.preanswer_teacher_kl_weight * preanswer_teacher_kl
            + args.orbit_kl_weight * orbit_kl
            + args.native_tail_state_weight * native_tail_state
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, args.grad_clip
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        values = {
            "loss": float(loss.detach()),
            "stage_ce": float(stage_ce.detach()),
            "bridge_ce": float(bridge_ce.detach()),
            "teacher_kl": float(teacher_kl.detach()),
            "preanswer_teacher_kl": float(preanswer_teacher_kl.detach()),
            "orbit_kl": float(orbit_kl.detach()),
            "native_tail_state": float(native_tail_state.detach()),
            "grad_norm": float(grad_norm.detach()),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError(f"non-finite metric at step {step}")
        for name, value in values.items():
            totals[name] += value
        training_rows.append(
            {
                "step": step,
                "document": document["index"],
                "train_layer_pattern": train_layer_pattern,
                "train_bridge_weight": train_bridge_weight,
                **values,
            }
        )
        del (
            source_cache,
            source_tail,
            orbit_base_source_tail,
            stage_tail,
            bridge_tail,
            stage_logits,
            bridge_logits,
            orbit_teacher_stage_tail,
            orbit_teacher_bridge_tail,
            orbit_teacher_stage_logits,
            orbit_teacher_bridge_logits,
            loss,
            stage_ce,
            bridge_ce,
            teacher_kl,
            preanswer_teacher_kl,
            orbit_kl,
            native_tail_state,
            grad_norm,
        )
        if step == 1 or step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step,
                        "document": document["index"],
                        "train_layer_pattern": train_layer_pattern,
                        "train_bridge_weight": train_bridge_weight,
                        "train_gradient_boundary": train_gradient_boundary,
                        "train_layer_pattern_counts": train_layer_pattern_counts,
                        **{f"mean_{name}": total / step for name, total in totals.items()},
                    }
                ),
                flush=True,
            )
    training_seconds = time.perf_counter() - started
    config["train_layer_pattern_counts"] = train_layer_pattern_counts
    (output_dir / "training_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in training_rows)
    )
    frozen_capsule_sha256_after = parameter_sha256(
        capsule, requires_grad=False
    )
    trainable_capsule_sha256_after = parameter_sha256(
        capsule, requires_grad=True
    )
    orbit_base_sha256_after = (
        None if orbit_base_capsule is None else parameter_sha256(orbit_base_capsule)
    )
    config["frozen_capsule_parameter_sha256_after"] = frozen_capsule_sha256_after
    config["trainable_capsule_parameter_sha256_after"] = (
        trainable_capsule_sha256_after
    )
    config["orbit_base_parameter_sha256_after"] = orbit_base_sha256_after
    config["frozen_capsule_parameters_unchanged"] = (
        frozen_capsule_sha256_after == frozen_capsule_sha256_before
    )
    config["trainable_capsule_parameters_changed"] = (
        trainable_capsule_sha256_after != trainable_capsule_sha256_before
    )
    config["orbit_base_parameters_unchanged"] = (
        orbit_base_sha256_after == orbit_base_sha256_before
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if not config["frozen_capsule_parameters_unchanged"]:
        raise RuntimeError("frozen capsule parameters changed during training")
    if not config["orbit_base_parameters_unchanged"]:
        raise RuntimeError("frozen orbit-base parameters changed during training")

    capsule.eval()
    rows = []
    for position, (index, case) in enumerate(eval_cases, 1):
        rows.extend(
            _evaluate_case(
                model,
                tokenizer,
                capsule,
                index=index,
                case=case,
                stage_system_ids=stage_system_ids,
                stage_system_cache=stage_system_cache,
                bridge_system_ids=bridge_system_ids,
                bridge_system_cache=bridge_system_cache,
                quant_bits_values=quant_bits_values,
                max_new_tokens=args.max_new_tokens,
                layer_patterns=eval_layer_patterns,
                native_tail_state_layers=(
                    native_tail_state_layers
                    if args.eval_native_tail_state or args.native_tail_state_weight
                    else None
                ),
                native_tail_state_max_tokens=native_tail_state_max_tokens,
            )
        )
        print(json.dumps({"event": "evaluation_progress", "position": position}), flush=True)

    checkpoint = {"embeddings": capsule.embeddings.detach().cpu(), "config": config}
    checkpoint["materialization_config"] = capsule.materialization_config()
    if capsule.write_adapter is not None:
        checkpoint["write_adapter_config"] = capsule.write_adapter.checkpoint_config()
        checkpoint["write_adapter_state"] = {
            name: tensor.detach().cpu()
            for name, tensor in capsule.write_adapter.state_dict().items()
        }
    if capsule.boundary_write_adapter is not None:
        checkpoint["boundary_write_adapter_config"] = (
            capsule.boundary_write_adapter.checkpoint_config()
        )
        checkpoint["boundary_write_adapter_state"] = {
            name: tensor.detach().cpu()
            for name, tensor in capsule.boundary_write_adapter.state_dict().items()
        }
    torch.save(checkpoint, output_dir / "capsule.pt")
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    summary = _summarize(
        rows,
        config=config,
        model=model,
        slots=args.slots,
        training_seconds=training_seconds,
        training_means={name: total / args.steps for name, total in totals.items()},
        resident_source_cache_bytes=resident_source_cache_bytes,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "complete", "summary": summary}), flush=True)


def freeze_initialized_base(capsule: SoftTailCapsule) -> None:
    """Freeze reusable state while leaving exactly the boundary residual trainable."""

    if capsule.boundary_write_adapter is None:
        raise ValueError(
            "--freeze-initialized-base requires an existing or newly requested "
            "boundary writer"
        )
    capsule.embeddings.requires_grad_(False)
    if capsule.write_adapter is not None:
        capsule.write_adapter.requires_grad_(False)
    capsule.boundary_write_adapter.requires_grad_(True)


def expand_soft_tail_capsule_slots(capsule: SoftTailCapsule, *, slots: int) -> None:
    """Deterministically expand slot embeddings without changing writer weights."""

    if slots <= capsule.slots:
        raise ValueError("expanded slot count must exceed the initialized count")
    index = torch.arange(slots, device=capsule.embeddings.device) % capsule.slots
    expanded = capsule.embeddings.detach().index_select(0, index).clone()
    capsule.embeddings = torch.nn.Parameter(
        expanded, requires_grad=capsule.embeddings.requires_grad
    )


def parameter_sha256(module, *, requires_grad: bool | None = None) -> str:
    """Hash selected logical parameter bytes for frozen-state auditability."""

    digest = hashlib.sha256()
    selected = 0
    for name, parameter in module.named_parameters():
        if requires_grad is not None and parameter.requires_grad != requires_grad:
            continue
        tensor = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        selected += 1
    digest.update(f"parameters={selected}".encode())
    return digest.hexdigest()


def _initial_embeddings(model, tokenizer, *, slots: int) -> torch.Tensor:
    ids = tokenizer.encode(
        " internal computed state summary evidence result", add_special_tokens=False
    )
    if not ids:
        raise ValueError("capsule initialization text tokenized to empty")
    ids = [ids[index % len(ids)] for index in range(slots)]
    device = model.model.embed_tokens.weight.device
    index = torch.tensor(ids, dtype=torch.long, device=device)
    return model.model.embed_tokens(index).detach().float()


def _validate_initialized_capsule(
    capsule: SoftTailCapsule,
    *,
    initialization_config: dict,
    model,
    model_path: str,
    slots: int,
    writer_rank: int,
    writer_scale: float,
) -> None:
    """Reject initialization checkpoints that change the frozen experiment."""

    if capsule.slots != slots:
        raise ValueError("initialization checkpoint slot count does not match")
    if capsule.embeddings.shape[1] != model.config.hidden_size:
        raise ValueError("initialization checkpoint hidden size does not match")
    checkpoint_model = initialization_config.get("model")
    if checkpoint_model is not None and str(checkpoint_model) != str(model_path):
        raise ValueError("initialization checkpoint model does not match")
    writer = capsule.write_adapter
    if writer_rank == 0:
        if writer is not None:
            raise ValueError("initialization checkpoint unexpectedly contains a writer")
        return
    if writer is None:
        raise ValueError("initialization checkpoint is missing the requested writer")
    if (
        writer.hidden_size != model.config.hidden_size
        or writer.num_hidden_layers != model.config.num_hidden_layers
        or writer.rank != writer_rank
        or not math.isclose(writer.scale, writer_scale, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("initialization checkpoint writer configuration does not match")


def resolve_train_layer_patterns(
    train_layer_pattern: str, train_layer_patterns: str | None
) -> tuple[str, ...]:
    if train_layer_patterns is None:
        values = (train_layer_pattern.strip(),)
    else:
        if train_layer_pattern != "all":
            raise ValueError(
                "--train-layer-patterns conflicts with non-default "
                "--train-layer-pattern"
            )
        values = tuple(
            value.strip() for value in train_layer_patterns.split(",") if value.strip()
        )
    if not values:
        raise ValueError("training layer patterns must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError("training layer patterns must be unique")
    return values


def resolve_train_layer_bridge_weights(
    train_layer_patterns: tuple[str, ...],
    specification: str | None,
    *,
    default_weight: float,
) -> dict[str, float]:
    if specification is None:
        return {pattern: float(default_weight) for pattern in train_layer_patterns}
    weights = {}
    for item in specification.split(","):
        item = item.strip()
        if not item or ":" not in item:
            raise ValueError(
                "training layer bridge weights must use pattern:weight entries"
            )
        pattern, raw_weight = (part.strip() for part in item.split(":", 1))
        if pattern in weights:
            raise ValueError("training layer bridge weight patterns must be unique")
        try:
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError("training layer bridge weights must be numeric") from error
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                "training layer bridge weights must be finite and non-negative"
            )
        weights[pattern] = weight
    if set(weights) != set(train_layer_patterns):
        raise ValueError(
            "training layer bridge weight patterns must exactly match training patterns"
        )
    return weights


def _source_prompt_ids(tokenizer, case: dict) -> tuple[int, ...]:
    documents = "\n\n".join(
        f"[DOC-{number}] {document['title']}\n{document['text']}"
        for number, document in enumerate(case["agent_a"]["documents"], 1)
    )
    user = (
        f"<DOSSIER>\n{documents}\n</DOSSIER>\n\n"
        f"Question: {case['agent_a']['question']}\n"
        "Output only the exact short answer."
    )
    segments = build_chat_segments(
        tokenizer,
        system_prompt=SOURCE_ANSWER_SYSTEM,
        tools=[],
        history=[{"role": "user", "content": user}],
        enable_thinking=False,
    )
    return tuple((*segments.prefix_ids, *segments.history_ids, *segments.readout_ids))


def _prepare_document(
    tokenizer,
    *,
    index: int,
    case: dict,
    stage_system_ids,
    bridge_system_ids,
) -> dict:
    stage = _task_tokens(
        tokenizer,
        system_prompt=DEPENDENT_B_SYSTEM,
        system_ids=stage_system_ids,
        query=_format_target(case),
        answer=case["agent_b"]["gold_answer"],
    )
    bridge = _task_tokens(
        tokenizer,
        system_prompt=INTERMEDIATE_B_SYSTEM,
        system_ids=bridge_system_ids,
        query=INTERMEDIATE_QUERY,
        answer=case["agent_a"]["gold_answer"],
    )
    return {
        "index": index,
        "id": case["id"],
        "source_ids": _source_prompt_ids(tokenizer, case),
        "stage_query_ids": stage[0],
        "stage_answer_ids": stage[1],
        "bridge_query_ids": bridge[0],
        "bridge_answer_ids": bridge[1],
    }


def _task_tokens(tokenizer, *, system_prompt: str, system_ids, query: str, answer: str):
    segments = build_chat_segments(
        tokenizer,
        system_prompt=system_prompt,
        tools=[],
        history=[{"role": "user", "content": query}],
        enable_thinking=False,
    )
    if tuple(segments.prefix_ids) != tuple(system_ids):
        raise ValueError("target system prefix changed across tasks")
    answer_ids = tuple(tokenizer.encode(answer, add_special_tokens=False))
    if not answer_ids:
        raise ValueError("answer tokenization is empty")
    return tuple((*segments.history_ids, *segments.readout_ids)), answer_ids


@torch.no_grad()
def _prefill_detached_cache(model, token_ids):
    device = model.model.embed_tokens.weight.device
    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    output = model.model(input_ids=inputs, use_cache=True, return_dict=True)
    cache = output.past_key_values
    layers = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
    return tuple((key.detach(), value.detach()) for key, value in layers)


def _cache_bytes(cache) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in cache
        for tensor in layer
    )


def _task_logits(
    model,
    *,
    prefix_cache,
    prefix_tokens: int,
    compact_cache,
    slots: int,
    query_ids,
    answer_ids,
) -> torch.Tensor:
    assembled = concatenate_caches(prefix_cache, compact_cache)
    logits = teacher_forced_answer_logits(
        model,
        legacy_cache=assembled,
        cached_tokens=prefix_tokens + slots,
        query_ids=query_ids,
        answer_ids=answer_ids,
    )
    return logits


def positional_orbit_caches(
    model,
    *,
    base_source_tail,
    task_source_tail,
    source_start: int,
    target_start: int,
    slots: int,
    base_layers: frozenset[int],
    delta_layers: frozenset[int],
):
    """Build canonical student and detached target-frame teacher caches."""

    detached_task = tuple(
        tuple(tensor.detach() for tensor in layer) for layer in task_source_tail
    )
    detached_base = tuple(
        tuple(tensor.detach() for tensor in layer) for layer in base_source_tail
    )
    target_base = reposition_tail_cache(
        model,
        detached_base,
        source_start=source_start,
        target_start=target_start,
    )
    target_task = reposition_tail_cache(
        model,
        detached_task,
        source_start=source_start,
        target_start=target_start,
    )
    _require_delta_topology(
        target_base,
        target_task,
        base_layers=base_layers,
        delta_layers=delta_layers,
    )
    teacher = compose_fake_int4_base_delta(
        target_base,
        target_task,
        suffix_tokens=slots,
        base_layers=base_layers,
        delta_layers=delta_layers,
        straight_through_delta=False,
    ).cache

    canonical_base = reposition_tail_cache(
        model,
        detached_base,
        source_start=source_start,
        target_start=0,
    )
    canonical_task = reposition_tail_cache(
        model,
        task_source_tail,
        source_start=source_start,
        target_start=0,
    )
    _require_delta_topology(
        canonical_base,
        canonical_task,
        base_layers=base_layers,
        delta_layers=delta_layers,
    )
    canonical_student = compose_fake_int4_base_delta(
        canonical_base,
        canonical_task,
        suffix_tokens=slots,
        base_layers=base_layers,
        delta_layers=delta_layers,
        straight_through_delta=True,
    ).cache
    student = reposition_tail_cache(
        model,
        canonical_student,
        source_start=0,
        target_start=target_start,
    )
    return student, teacher


def _answer_ce(logits: torch.Tensor, answer_ids) -> torch.Tensor:
    targets = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits.squeeze(0).float(), targets)


def behavioral_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """KL(teacher || student) over every teacher-forced answer position."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher = F.softmax(teacher_logits.float() / temperature, dim=-1)
    per_position = F.kl_div(student, teacher, reduction="none").sum(dim=-1)
    return per_position.mean() * temperature**2


@torch.inference_mode()
def _evaluate_case(
    model,
    tokenizer,
    capsule,
    *,
    index: int,
    case: dict,
    stage_system_ids,
    stage_system_cache,
    bridge_system_ids,
    bridge_system_cache,
    quant_bits_values,
    max_new_tokens: int,
    source_index: int | None = None,
    source_case: dict | None = None,
    layer_patterns: tuple[str, ...] = ("all",),
    real_int4_codec: bool = False,
    delta_base_capsule=None,
    delta_layers: frozenset[int] | None = None,
    packet_position_mode: str = "target",
    key_codec: str = "cartesian",
    native_tail_state_layers: frozenset[int] | None = None,
    native_tail_state_max_tokens: int = 0,
) -> list[dict]:
    if (delta_base_capsule is None) != (delta_layers is None):
        raise ValueError("delta base capsule and delta layers must be specified together")
    if delta_base_capsule is not None and not real_int4_codec:
        raise ValueError("base-plus-delta composition requires the real INT4 codec")
    if packet_position_mode not in ("target", "canonical"):
        raise ValueError("packet position mode must be target or canonical")
    if packet_position_mode == "canonical" and not real_int4_codec:
        raise ValueError("canonical packet positions require the real INT4 codec")
    if key_codec not in ("cartesian", "rope_polar", "cartesian_k8"):
        raise ValueError(
            "key codec must be cartesian, rope_polar, or cartesian_k8"
        )
    if key_codec == "rope_polar" and packet_position_mode != "canonical":
        raise ValueError("rope_polar Keys require canonical packet positions")
    if key_codec == "cartesian_k8" and packet_position_mode != "canonical":
        raise ValueError("cartesian_k8 Keys require canonical packet positions")
    document = _prepare_document(
        tokenizer,
        index=index,
        case=case,
        stage_system_ids=stage_system_ids,
        bridge_system_ids=bridge_system_ids,
    )
    if source_case is not None:
        document["source_ids"] = _source_prompt_ids(tokenizer, source_case)
    else:
        source_case = case
    if source_index is None:
        source_index = index
    source_cache = prefill_legacy_cache(model, document["source_ids"])
    source_tail = capsule.materialize(
        model,
        prefix_cache=source_cache,
        prefix_tokens=len(document["source_ids"]),
    )
    native_tail_state_relative_mse = None
    if native_tail_state_layers is not None:
        teacher_ids = tuple(
            tokenizer.encode(
                source_case["agent_a"]["gold_answer"], add_special_tokens=False
            )
        )[:native_tail_state_max_tokens]
        if not teacher_ids:
            raise ValueError("native Tail-state teacher tokenized to empty")
        native_teacher_tail = materialize_token_tail(
            model,
            teacher_ids,
            prefix_cache=source_cache,
            prefix_tokens=len(document["source_ids"]),
        )
        native_tail_state_relative_mse = float(
            canonical_int4_tail_state_loss(
                model,
                source_tail,
                native_teacher_tail,
                source_start=len(document["source_ids"]),
                active_layers=native_tail_state_layers,
                straight_through=False,
            ).item()
        )
    base_source_tail = (
        None
        if delta_base_capsule is None
        else delta_base_capsule.materialize(
            model,
            prefix_cache=source_cache,
            prefix_tokens=len(document["source_ids"]),
        )
    )
    no_summary = _clean_generation(
        _generate(
            model,
            tokenizer,
            legacy_cache=stage_system_cache,
            cached_tokens=len(stage_system_ids),
            query_ids=document["stage_query_ids"],
            max_new_tokens=max_new_tokens,
        )
    )
    rows = []
    active_heads = frozenset(range(model.config.num_key_value_heads))
    for quant_bits in quant_bits_values:
        for layer_pattern in layer_patterns:
            active_layers = active_layer_indices(
                layer_pattern, total_layers=model.config.num_hidden_layers
            )
            stage_tail = reposition_tail_cache(
                model,
                source_tail,
                source_start=len(document["source_ids"]),
                target_start=(
                    0
                    if packet_position_mode == "canonical"
                    else len(stage_system_ids)
                ),
            )
            bridge_tail = reposition_tail_cache(
                model,
                source_tail,
                source_start=len(document["source_ids"]),
                target_start=(
                    0
                    if packet_position_mode == "canonical"
                    else len(bridge_system_ids)
                ),
            )
            actual_packet_bytes = None
            base_packet_bytes = None
            delta_packet_bytes = None
            incremental_packet_bytes = None
            direct_packet_bytes = None
            stage_direct_nrms = None
            bridge_direct_nrms = None
            stage_task_nrms = None
            bridge_task_nrms = None
            packet_target_invariant = None
            if delta_base_capsule is not None:
                if quant_bits != 4:
                    raise ValueError("base-plus-delta composition supports only INT4")
                base_stage_tail = reposition_tail_cache(
                    model,
                    base_source_tail,
                    source_start=len(document["source_ids"]),
                    target_start=(
                        0
                        if packet_position_mode == "canonical"
                        else len(stage_system_ids)
                    ),
                )
                base_bridge_tail = reposition_tail_cache(
                    model,
                    base_source_tail,
                    source_start=len(document["source_ids"]),
                    target_start=(
                        0
                        if packet_position_mode == "canonical"
                        else len(bridge_system_ids)
                    ),
                )
                _require_delta_topology(
                    base_stage_tail,
                    stage_tail,
                    base_layers=active_layers,
                    delta_layers=delta_layers,
                )
                _require_delta_topology(
                    base_bridge_tail,
                    bridge_tail,
                    base_layers=active_layers,
                    delta_layers=delta_layers,
                )
                stage_composition = compose_int4_base_delta(
                    base_stage_tail,
                    stage_tail,
                    suffix_tokens=capsule.slots,
                    base_layers=active_layers,
                    delta_layers=delta_layers,
                    key_codec=key_codec,
                )
                bridge_composition = compose_int4_base_delta(
                    base_bridge_tail,
                    bridge_tail,
                    suffix_tokens=capsule.slots,
                    base_layers=active_layers,
                    delta_layers=delta_layers,
                    key_codec=key_codec,
                )
                direct_stage_tail, direct_packet_bytes = _real_int4_round_trip(
                    stage_tail,
                    slots=capsule.slots,
                    active_layers=active_layers,
                    key_codec=key_codec,
                )
                direct_bridge_tail, direct_bridge_packet_bytes = _real_int4_round_trip(
                    bridge_tail,
                    slots=capsule.slots,
                    active_layers=active_layers,
                    key_codec=key_codec,
                )
                packet_size_tuple = (
                    stage_composition.base_packet_bytes,
                    bridge_composition.base_packet_bytes,
                    stage_composition.delta_packet_bytes,
                    bridge_composition.delta_packet_bytes,
                    direct_packet_bytes,
                    direct_bridge_packet_bytes,
                )
                if len(set(packet_size_tuple[:2])) != 1 or len(set(packet_size_tuple[2:4])) != 1:
                    raise RuntimeError("stage and bridge component packet sizes differ")
                if direct_packet_bytes != direct_bridge_packet_bytes:
                    raise RuntimeError("stage and bridge direct packet sizes differ")
                if packet_position_mode == "canonical":
                    packet_target_invariant = bool(
                        stage_composition.base_packet
                        == bridge_composition.base_packet
                        and stage_composition.delta_packet
                        == bridge_composition.delta_packet
                    )
                    if not packet_target_invariant:
                        raise RuntimeError(
                            "canonical packet bytes depend on receiver target"
                        )
                _require_composition_invariants(
                    stage_composition, active_layers=active_layers, delta_layers=delta_layers
                )
                _require_composition_invariants(
                    bridge_composition, active_layers=active_layers, delta_layers=delta_layers
                )
                stage_direct_nrms = _cache_nrms(
                    stage_composition.cache, direct_stage_tail, layers=delta_layers
                )
                bridge_direct_nrms = _cache_nrms(
                    bridge_composition.cache, direct_bridge_tail, layers=delta_layers
                )
                stage_task_nrms = _cache_nrms(
                    stage_composition.cache, stage_tail, layers=delta_layers
                )
                bridge_task_nrms = _cache_nrms(
                    bridge_composition.cache, bridge_tail, layers=delta_layers
                )
                stage_tail = stage_composition.cache
                bridge_tail = bridge_composition.cache
                base_packet_bytes = stage_composition.base_packet_bytes
                delta_packet_bytes = stage_composition.delta_packet_bytes
                incremental_packet_bytes = stage_composition.incremental_packet_bytes
                actual_packet_bytes = stage_composition.cold_packet_bytes
            elif real_int4_codec:
                if quant_bits != 4:
                    raise ValueError("real codec currently supports only INT4")
                stage_tail, actual_packet_bytes = _real_int4_round_trip(
                    stage_tail,
                    slots=capsule.slots,
                    active_layers=active_layers,
                    key_codec=key_codec,
                )
                bridge_tail, bridge_packet_bytes = _real_int4_round_trip(
                    bridge_tail,
                    slots=capsule.slots,
                    active_layers=active_layers,
                    key_codec=key_codec,
                )
                if bridge_packet_bytes != actual_packet_bytes:
                    raise RuntimeError("stage and bridge packet sizes differ")
                if packet_position_mode == "canonical":
                    stage_packet = pack_int4_capsule(
                        stage_tail,
                        suffix_tokens=capsule.slots,
                        active_layers=active_layers,
                        key_codec=key_codec,
                    )
                    bridge_packet = pack_int4_capsule(
                        bridge_tail,
                        suffix_tokens=capsule.slots,
                        active_layers=active_layers,
                        key_codec=key_codec,
                    )
                    packet_target_invariant = stage_packet == bridge_packet
                    if not packet_target_invariant:
                        raise RuntimeError(
                            "canonical direct packet bytes depend on receiver target"
                        )
            else:
                stage_tail = mask_and_fake_quantize_suffix(
                    stage_tail,
                    suffix_tokens=capsule.slots,
                    bits=quant_bits,
                    active_layers=active_layers,
                    active_heads=active_heads,
                )
                bridge_tail = mask_and_fake_quantize_suffix(
                    bridge_tail,
                    suffix_tokens=capsule.slots,
                    bits=quant_bits,
                    active_layers=active_layers,
                    active_heads=active_heads,
                )
            if packet_position_mode == "canonical":
                stage_tail = reposition_tail_cache(
                    model,
                    stage_tail,
                    source_start=0,
                    target_start=len(stage_system_ids),
                )
                bridge_tail = reposition_tail_cache(
                    model,
                    bridge_tail,
                    source_start=0,
                    target_start=len(bridge_system_ids),
                )
            student = _clean_generation(
                _generate(
                    model,
                    tokenizer,
                    legacy_cache=concatenate_caches(stage_system_cache, stage_tail),
                    cached_tokens=len(stage_system_ids) + capsule.slots,
                    query_ids=document["stage_query_ids"],
                    max_new_tokens=max_new_tokens,
                )
            )
            bridge_student = _clean_generation(
                _generate(
                    model,
                    tokenizer,
                    legacy_cache=concatenate_caches(bridge_system_cache, bridge_tail),
                    cached_tokens=len(bridge_system_ids) + capsule.slots,
                    query_ids=document["bridge_query_ids"],
                    max_new_tokens=max_new_tokens,
                )
            )
            final_gold = case["agent_b"]["gold_answer"]
            bridge_gold = case["agent_a"]["gold_answer"]
            row = {
                "index": index,
                "id": case["id"],
                "source_index": source_index,
                "source_id": source_case["id"],
                "quant_bits": quant_bits,
                "layer_pattern": layer_pattern,
                "active_layers": len(active_layers),
                "source_tokens": len(document["source_ids"]),
                "slots": capsule.slots,
                "wire_codec": (
                    "real_int4_base_delta"
                    if delta_base_capsule is not None
                    else "real_int4" if real_int4_codec else "fake_quant"
                ),
                "packet_position_mode": packet_position_mode,
                "key_codec": key_codec,
                "packet_target_invariant": packet_target_invariant,
                "actual_packet_bytes": actual_packet_bytes,
                "base_packet_bytes": base_packet_bytes,
                "delta_packet_bytes": delta_packet_bytes,
                "incremental_packet_bytes": incremental_packet_bytes,
                "direct_packet_bytes": direct_packet_bytes,
                "delta_layers": (
                    None if delta_layers is None else sorted(delta_layers)
                ),
                "native_tail_state_relative_mse": native_tail_state_relative_mse,
                "stage_delta_vs_direct_nrms": stage_direct_nrms,
                "bridge_delta_vs_direct_nrms": bridge_direct_nrms,
                "stage_delta_vs_task_nrms": stage_task_nrms,
                "bridge_delta_vs_task_nrms": bridge_task_nrms,
                "student": student,
                "bridge_student": bridge_student,
                "no_summary": no_summary,
                "final_gold": final_gold,
                "bridge_gold": bridge_gold,
                "student_f1": coqa_f1(student, [final_gold]),
                "bridge_f1": coqa_f1(bridge_student, [bridge_gold]),
                "no_summary_f1": coqa_f1(no_summary, [final_gold]),
                "student_exact": float(
                    _normalized(student) == _normalized(final_gold)
                ),
                "bridge_exact": float(
                    _normalized(bridge_student) == _normalized(bridge_gold)
                ),
                "no_summary_exact": float(
                    _normalized(no_summary) == _normalized(final_gold)
                ),
            }
            rows.append(row)
            print(json.dumps({"event": "evaluate", **row}), flush=True)
    return rows


def _real_int4_round_trip(
    cache,
    *,
    slots: int,
    active_layers: frozenset[int],
    key_codec: str = "cartesian",
):
    packet = pack_int4_capsule(
        cache,
        suffix_tokens=slots,
        active_layers=active_layers,
        key_codec=key_codec,
    )
    device = cache[0][0].device
    dtype = cache[0][0].dtype
    return (
        unpack_int4_capsule(packet, dtype=dtype, device=device),
        len(packet),
    )


def _require_delta_topology(
    base_cache,
    task_cache,
    *,
    base_layers: frozenset[int],
    delta_layers: frozenset[int],
) -> None:
    """Prove that the frozen task checkpoint differs only on declared wire layers."""

    if not delta_layers.issubset(base_layers):
        raise ValueError("delta layers must be a subset of active base layers")
    for layer_index in base_layers - delta_layers:
        for base_tensor, task_tensor in zip(
            base_cache[layer_index], task_cache[layer_index], strict=True
        ):
            if not torch.equal(base_tensor, task_tensor):
                raise RuntimeError(
                    f"task cache unexpectedly changes non-delta layer {layer_index}"
                )


def _require_composition_invariants(
    composition,
    *,
    active_layers: frozenset[int],
    delta_layers: frozenset[int],
) -> None:
    for layer_index, (composed_layer, base_layer) in enumerate(
        zip(composition.cache, composition.base_cache, strict=True)
    ):
        for composed_tensor, base_tensor in zip(
            composed_layer, base_layer, strict=True
        ):
            if layer_index in active_layers - delta_layers and not torch.equal(
                composed_tensor, base_tensor
            ):
                raise RuntimeError("composition changed an immutable base layer")
            if layer_index not in active_layers and torch.count_nonzero(
                composed_tensor
            ).item():
                raise RuntimeError("composition populated an inactive layer")


def _cache_nrms(candidate, reference, *, layers: frozenset[int]) -> float:
    squared_error = torch.zeros((), dtype=torch.float64)
    squared_reference = torch.zeros((), dtype=torch.float64)
    for layer_index in sorted(layers):
        for candidate_tensor, reference_tensor in zip(
            candidate[layer_index], reference[layer_index], strict=True
        ):
            delta = candidate_tensor.float() - reference_tensor.float()
            squared_error += delta.double().square().sum().cpu()
            squared_reference += reference_tensor.float().double().square().sum().cpu()
    return float(
        torch.sqrt(squared_error / squared_reference.clamp_min(1e-30)).item()
    )


def _summarize(
    rows,
    *,
    config,
    model,
    slots: int,
    training_seconds: float,
    training_means,
    resident_source_cache_bytes: int,
):
    by_layer_pattern = {}
    for layer_pattern in dict.fromkeys(row.get("layer_pattern", "all") for row in rows):
        pattern_rows = [
            row for row in rows if row.get("layer_pattern", "all") == layer_pattern
        ]
        by_quant = {}
        for bits in sorted({row["quant_bits"] for row in pattern_rows}, reverse=True):
            selected = [row for row in pattern_rows if row["quant_bits"] == bits]
            mean = lambda key: sum(row[key] for row in selected) / len(selected)
            optional_mean = lambda key: (
                None
                if selected[0].get(key) is None
                else sum(float(row[key]) for row in selected) / len(selected)
            )
            active_layers = mean("active_layers") if "active_layers" in selected[0] else model.config.num_hidden_layers
            kv_elements = (
                active_layers
                * model.config.num_key_value_heads
                * model.config.head_dim
                * 2
                * slots
            )
            payload = kv_elements * bits / 8
            if bits < 16:
                payload += active_layers * model.config.num_key_value_heads * 2 * slots * 2
            framed_payload = optional_mean("actual_packet_bytes")
            by_quant[f"int{bits}"] = {
                "cases": len(selected),
                "active_layers": active_layers,
                "student_accuracy": mean("student_exact"),
                "bridge_accuracy": mean("bridge_exact"),
                "no_summary_accuracy": mean("no_summary_exact"),
                "student_mean_f1": mean("student_f1"),
                "bridge_mean_f1": mean("bridge_f1"),
                "no_summary_mean_f1": mean("no_summary_f1"),
                "wins_vs_no_summary": sum(
                    row["student_f1"] > row["no_summary_f1"] for row in selected
                ),
                "mean_source_tokens": mean("source_tokens"),
                "tensor_payload_bytes": payload,
                "wire_payload_bytes": (
                    payload if framed_payload is None else framed_payload
                ),
                "base_packet_bytes": optional_mean("base_packet_bytes"),
                "delta_packet_bytes": optional_mean("delta_packet_bytes"),
                "incremental_packet_bytes": optional_mean(
                    "incremental_packet_bytes"
                ),
                "direct_packet_bytes": optional_mean("direct_packet_bytes"),
                "stage_delta_vs_direct_nrms": optional_mean(
                    "stage_delta_vs_direct_nrms"
                ),
                "bridge_delta_vs_direct_nrms": optional_mean(
                    "bridge_delta_vs_direct_nrms"
                ),
                "stage_delta_vs_task_nrms": optional_mean(
                    "stage_delta_vs_task_nrms"
                ),
                "bridge_delta_vs_task_nrms": optional_mean(
                    "bridge_delta_vs_task_nrms"
                ),
                "native_tail_state_relative_mse": optional_mean(
                    "native_tail_state_relative_mse"
                ),
            }
        by_layer_pattern[layer_pattern] = by_quant
    return {
        "config": config,
        "training_seconds": training_seconds,
        "training_means": training_means,
        "resident_source_cache_bytes": resident_source_cache_bytes,
        "by_quant": by_layer_pattern.get("all"),
        "by_layer_pattern": by_layer_pattern,
    }


if __name__ == "__main__":
    main()
