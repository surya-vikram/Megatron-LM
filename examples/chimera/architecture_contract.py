# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate and serialize the locked Chimera architecture contract.

This module is intentionally usable both from ``pretrain_chimera.py`` and as a
small command-line preflight tool for HF/Megatron conversion scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


FULL_PROFILE = {
    "num_layers": 25,
    "hidden_size": 2048,
    "ffn_hidden_size": 8192,
    "num_attention_heads": 16,
    "num_query_groups": 2,
    "kv_channels": 256,
    "num_moe_experts": 32,
    "moe_router_topk": 4,
    "moe_ffn_hidden_size": 2048,
    "moe_layer_freq": [0] * 2 + [1] * 23,
}

TINY_PROFILE = {
    "num_layers": 8,
    "hidden_size": 512,
    "ffn_hidden_size": 2048,
    "num_attention_heads": 8,
    "num_query_groups": 2,
    "kv_channels": 64,
    "num_moe_experts": 8,
    "moe_router_topk": 2,
    "moe_ffn_hidden_size": 256,
    "moe_layer_freq": [0] * 2 + [1] * 6,
}

PROFILES = {"full": FULL_PROFILE, "tiny": TINY_PROFILE}


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    return actual == expected


def _require(actual: Any, expected: Any, name: str, errors: list[str]) -> None:
    if not _equal(actual, expected):
        errors.append(f"{name}: expected {expected!r}, found {actual!r}")


def _profile_from_values(
    values: dict[str, Any], requested: str = "auto"
) -> tuple[str, dict[str, Any]]:
    if requested != "auto":
        return requested, PROFILES[requested]

    matches = []
    for name, profile in PROFILES.items():
        if all(_equal(values.get(key), value) for key, value in profile.items()):
            matches.append((name, profile))
    if len(matches) != 1:
        summary = {key: values.get(key) for key in FULL_PROFILE}
        raise ValueError(
            f"Architecture does not match the full or tiny Chimera profile: {summary}"
        )
    return matches[0]


def _normalized_layer_freq(value: Any) -> list[int]:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    raise ValueError(f"moe_layer_freq must be a parsed list, found {value!r}")


def validate_training_args(args: Any) -> str:
    """Fail before model construction when a launcher drifts from the contract."""
    values = {
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "ffn_hidden_size": args.ffn_hidden_size,
        "num_attention_heads": args.num_attention_heads,
        "num_query_groups": args.num_query_groups,
        "kv_channels": args.kv_channels,
        "num_moe_experts": args.num_experts,
        "moe_router_topk": args.moe_router_topk,
        "moe_ffn_hidden_size": args.moe_ffn_hidden_size,
        "moe_layer_freq": _normalized_layer_freq(args.moe_layer_freq),
    }
    profile_name, profile = _profile_from_values(values)
    errors: list[str] = []
    for key, expected in profile.items():
        _require(values[key], expected, key, errors)

    common = {
        "max_position_embeddings": 8192,
        "rotary_base": 10_000_000,
        "rotary_scaling_factor": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 0.0,
        "vocab_size": 50176,
        "moe_router_score_function": "sigmoid",
        "moe_router_topk_scaling_factor": 2.5,
        "moe_router_bias_update_rate": 0.0,
        "moe_aux_loss_coeff": 0.0,
        "moe_z_loss_coeff": 0.001,
    }
    for key, expected in common.items():
        _require(getattr(args, key, None), expected, key, errors)

    _require(getattr(args, "qk_layernorm", False), True, "qk_layernorm", errors)
    _require(
        getattr(args, "moe_router_enable_expert_bias", False),
        True,
        "moe_router_enable_expert_bias",
        errors,
    )
    _require(
        getattr(args, "moe_shared_expert_intermediate_size", None),
        None,
        "moe_shared_expert_intermediate_size",
        errors,
    )
    _require(
        getattr(args, "share_embeddings_and_output_weights", True),
        False,
        "share_embeddings_and_output_weights",
        errors,
    )

    stage_is_finetune = bool(
        getattr(args, "finetune", False)
        or getattr(args, "sft", False)
        or getattr(args, "simpo", False)
    )
    expected_balancing = "none" if stage_is_finetune else "quantile_balancing"
    _require(
        getattr(args, "moe_router_load_balancing_type", None),
        expected_balancing,
        "moe_router_load_balancing_type",
        errors,
    )
    if expected_balancing == "quantile_balancing":
        _require(
            getattr(args, "moe_qb_num_bins", None), 1000, "moe_qb_num_bins", errors
        )
        _require(
            getattr(args, "moe_qb_ema_decay", None), 0.0, "moe_qb_ema_decay", errors
        )

    if errors:
        raise ValueError(
            "Chimera architecture contract violation:\n- " + "\n- ".join(errors)
        )
    return profile_name


def write_runtime_run_config(args: Any, template: Path) -> Path | None:
    """Write Bridge-readable architecture metadata next to every saved checkpoint."""
    if not getattr(args, "save", None) or int(os.environ.get("RANK", "0")) != 0:
        return None

    import yaml

    with template.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model = config["model"]
    model.update(
        {
            "_target_": "megatron.bridge.models.chimera.chimera_bridge.ChimeraModelProvider",
            "chimera_load_with_bias": True,
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "ffn_hidden_size": args.ffn_hidden_size,
            "num_attention_heads": args.num_attention_heads,
            "num_query_groups": args.num_query_groups,
            "kv_channels": args.kv_channels,
            "seq_length": args.max_position_embeddings,
            "vocab_size": args.vocab_size,
            "qk_layernorm": args.qk_layernorm,
            "num_moe_experts": args.num_experts,
            "moe_router_topk": args.moe_router_topk,
            "moe_ffn_hidden_size": args.moe_ffn_hidden_size,
            "moe_layer_freq": _normalized_layer_freq(args.moe_layer_freq),
            "moe_router_load_balancing_type": args.moe_router_load_balancing_type,
            "moe_aux_loss_coeff": args.moe_aux_loss_coeff,
            "moe_z_loss_coeff": args.moe_z_loss_coeff,
            "moe_qb_num_bins": getattr(args, "moe_qb_num_bins", 1000),
            "moe_qb_ema_decay": getattr(args, "moe_qb_ema_decay", 0.0),
            "moe_router_bias_update_rate": args.moe_router_bias_update_rate,
            "moe_router_enable_expert_bias": args.moe_router_enable_expert_bias,
            "moe_router_score_function": args.moe_router_score_function,
            "moe_router_topk_scaling_factor": args.moe_router_topk_scaling_factor,
            "moe_shared_expert_intermediate_size": None,
            "moe_shared_expert_overlap": False,
            "rotary_base": args.rotary_base,
            "rotary_scaling_factor": args.rotary_scaling_factor,
            "position_embedding_type": "yarn",
            "yarn_rotary_scaling_factor": args.yarn_rotary_scaling_factor,
            "yarn_original_max_position_embeddings": args.yarn_original_max_position_embeddings,
            "yarn_beta_fast": args.yarn_beta_fast,
            "yarn_beta_slow": args.yarn_beta_slow,
            "yarn_mscale": args.yarn_mscale,
            "yarn_mscale_all_dim": args.yarn_mscale_all_dim,
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
            "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
            "expert_model_parallel_size": args.expert_model_parallel_size,
            "expert_tensor_parallel_size": args.expert_tensor_parallel_size,
        }
    )

    destination = Path(args.save) / "run_config.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".yaml.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    temporary.replace(destination)
    return destination


def _hf_values(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "ffn_hidden_size": config.get("intermediate_size"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_query_groups": config.get("num_key_value_heads"),
        "kv_channels": config.get("head_dim"),
        "num_moe_experts": config.get("n_routed_experts"),
        "moe_router_topk": config.get("num_experts_per_tok"),
        "moe_ffn_hidden_size": config.get("moe_intermediate_size"),
        "moe_layer_freq": [0] * config.get("first_k_dense_replace", 0)
        + [1]
        * (
            config.get("num_hidden_layers", 0)
            - config.get("first_k_dense_replace", 0)
            - config.get("last_k_dense_replace", 0)
        )
        + [0] * config.get("last_k_dense_replace", 0),
    }


def validate_hf_config(path: Path, profile: str = "auto") -> tuple[str, dict[str, Any]]:
    config_path = path / "config.json" if path.is_dir() else path
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    values = _hf_values(config)
    profile_name, expected_profile = _profile_from_values(values, profile)
    errors: list[str] = []
    for key, expected in expected_profile.items():
        _require(values[key], expected, key, errors)

    expected = {
        "architectures": ["ChimeraForCausalLM"],
        "max_position_embeddings": 8192,
        "original_max_position_embeddings": 8192,
        "qk_layernorm": True,
        "n_shared_experts": 0,
        "shared_expert_intermediate_size": 0,
        "scoring_func": "sigmoid",
        "routed_scaling_factor": 2.5,
        "router_bias_update_rate": 0.0,
        "router_aux_loss_coef": 0.0,
        "router_z_loss_coef": 0.001,
        "moe_qb_num_bins": 1000,
        "moe_qb_ema_decay": 0.0,
        "tie_word_embeddings": False,
        "vocab_size": 50176,
    }
    for key, value in expected.items():
        _require(config.get(key), value, key, errors)
    if config.get("load_with_bias") not in (True, False):
        errors.append("load_with_bias: must be an explicit boolean")
    if config.get("router_load_balancing_type") not in ("quantile_balancing", "none"):
        errors.append(
            "router_load_balancing_type: expected 'quantile_balancing' or 'none'"
        )

    rope = config.get("rope_scaling") or {}
    rope_expected = {
        "factor": 1.0,
        "original_max_position_embeddings": 8192,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 0.0,
    }
    for key, value in rope_expected.items():
        _require(rope.get(key), value, f"rope_scaling.{key}", errors)
    if rope.get("type", rope.get("rope_type")) != "yarn":
        errors.append("rope_scaling.type: expected 'yarn'")

    if errors:
        raise ValueError(
            "HF Chimera architecture contract violation:\n- " + "\n- ".join(errors)
        )
    return profile_name, config


def validate_run_config(path: Path, profile: str = "auto") -> str:
    import yaml

    config_path = path / "run_config.yaml" if path.is_dir() else path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = config.get("model", {})
    values = {key: model.get(key) for key in FULL_PROFILE}
    profile_name, expected_profile = _profile_from_values(values, profile)
    errors: list[str] = []
    for key, expected in expected_profile.items():
        _require(values[key], expected, f"model.{key}", errors)
    expected = {
        "_target_": "megatron.bridge.models.chimera.chimera_bridge.ChimeraModelProvider",
        "seq_length": 8192,
        "vocab_size": 50176,
        "qk_layernorm": True,
        "moe_shared_expert_intermediate_size": None,
        "moe_shared_expert_overlap": False,
        "moe_router_enable_expert_bias": True,
        "moe_router_bias_update_rate": 0.0,
        "moe_router_score_function": "sigmoid",
        "moe_router_topk_scaling_factor": 2.5,
        "moe_aux_loss_coeff": 0.0,
        "moe_z_loss_coeff": 0.001,
        "moe_qb_num_bins": 1000,
        "moe_qb_ema_decay": 0.0,
        "rotary_base": 10_000_000,
        "yarn_rotary_scaling_factor": 1.0,
        "yarn_original_max_position_embeddings": 8192,
        "yarn_mscale": 1.0,
        "yarn_mscale_all_dim": 0.0,
    }
    for key, value in expected.items():
        _require(model.get(key), value, f"model.{key}", errors)
    if model.get("chimera_load_with_bias") not in (True, False):
        errors.append("model.chimera_load_with_bias: must be an explicit boolean")
    if model.get("moe_router_load_balancing_type") not in (
        "quantile_balancing",
        "none",
    ):
        errors.append(
            "model.moe_router_load_balancing_type: expected 'quantile_balancing' or 'none'"
        )
    if errors:
        raise ValueError(
            "MCore Chimera architecture contract violation:\n- " + "\n- ".join(errors)
        )
    return profile_name


def _safetensor_files(path: Path) -> list[Path]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        with index.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return sorted({path / name for name in data["weight_map"].values()})
    single = path / "model.safetensors"
    return [single] if single.exists() else []


def _safetensor_key_map(path: Path) -> dict[str, Path]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required for exact HF weight verification"
        ) from exc

    files = _safetensor_files(path)
    if not files:
        raise FileNotFoundError(f"No model.safetensors files found under {path}")
    key_map: dict[str, Path] = {}
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in key_map:
                    raise ValueError(
                        f"Duplicate safetensors key {key!r} in {filename} and {key_map[key]}"
                    )
                key_map[key] = filename
    return key_map


def _load_safetensor(filename: Path, key: str) -> Any:
    from safetensors import safe_open

    with safe_open(filename, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def expected_hf_keys(config: dict[str, Any]) -> set[str]:
    keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    dense_count = config["first_k_dense_replace"]
    last_dense_start = config["num_hidden_layers"] - config.get(
        "last_k_dense_replace", 0
    )
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        keys.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
            }
        )
        if layer < dense_count or layer >= last_dense_start:
            keys.update(
                {
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                }
            )
        else:
            keys.add(f"{prefix}.mlp.gate.weight")
            keys.add(f"{prefix}.mlp.gate.e_score_correction_bias")
            for expert in range(config["n_routed_experts"]):
                keys.update(
                    {
                        f"{prefix}.mlp.experts.{expert}.gate_proj.weight",
                        f"{prefix}.mlp.experts.{expert}.up_proj.weight",
                        f"{prefix}.mlp.experts.{expert}.down_proj.weight",
                    }
                )
    return keys


def validate_hf_weights(path: Path, config: dict[str, Any]) -> int:
    actual = set(_safetensor_key_map(path))
    expected = expected_hf_keys(config)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    shared = sorted(key for key in actual if ".shared_experts." in key)
    if missing or extra or shared:
        details = []
        if missing:
            details.append(f"missing ({len(missing)}): {missing[:20]}")
        if extra:
            details.append(f"extra ({len(extra)}): {extra[:20]}")
        if shared:
            details.append(
                f"forbidden shared-expert keys ({len(shared)}): {shared[:20]}"
            )
        raise ValueError("HF checkpoint key-set violation:\n- " + "\n- ".join(details))
    return len(actual)


def _tensor_sha256(tensor: Any) -> str:
    tensor = tensor.detach().cpu().contiguous()
    byte_view = tensor.view(dtype=__import__("torch").uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def compare_hf_weights(
    expected_path: Path, actual_path: Path, report_path: Path | None = None
) -> int:
    import torch

    expected_profile, expected_config = validate_hf_config(expected_path)
    actual_profile, actual_config = validate_hf_config(actual_path)
    if expected_profile != actual_profile:
        raise ValueError(
            f"HF profiles differ: expected={expected_profile}, actual={actual_profile}"
        )
    config_fields = {
        "architectures",
        "first_k_dense_replace",
        "last_k_dense_replace",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "original_max_position_embeddings",
        "qk_layernorm",
        "load_with_bias",
        "n_routed_experts",
        "num_experts_per_tok",
        "n_shared_experts",
        "scoring_func",
        "routed_scaling_factor",
        "router_aux_loss_coef",
        "router_z_loss_coef",
        "router_bias_update_rate",
        "router_load_balancing_type",
        "moe_qb_num_bins",
        "moe_qb_ema_decay",
        "tie_word_embeddings",
        "vocab_size",
        "rope_theta",
    }
    config_failures = [
        key
        for key in sorted(config_fields)
        if expected_config.get(key) != actual_config.get(key)
    ]
    expected_rope = expected_config.get("rope_scaling") or {}
    actual_rope = actual_config.get("rope_scaling") or {}
    rope_fields = {
        "factor",
        "original_max_position_embeddings",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim",
    }
    if any(
        expected_rope.get(key) != actual_rope.get(key) for key in rope_fields
    ) or expected_rope.get("type", expected_rope.get("rope_type")) != actual_rope.get(
        "type", actual_rope.get("rope_type")
    ):
        config_failures.append("rope_scaling")
    if config_failures:
        raise ValueError(f"HF architecture configs differ in fields: {config_failures}")

    expected = _safetensor_key_map(expected_path)
    actual = _safetensor_key_map(actual_path)
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    if missing or extra:
        raise ValueError(
            f"Checkpoint key sets differ: missing={missing[:20]}, extra={extra[:20]}"
        )

    report: list[dict[str, Any]] = []
    failures: list[str] = []
    for key in sorted(expected):
        left = _load_safetensor(expected[key], key)
        right = _load_safetensor(actual[key], key)
        left_hash = _tensor_sha256(left)
        right_hash = _tensor_sha256(right)
        equal = (
            left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left, right)
        )
        report.append(
            {
                "key": key,
                "shape": list(left.shape),
                "dtype": str(left.dtype),
                "expected_sha256": left_hash,
                "actual_sha256": right_hash,
                "equal": equal,
            }
        )
        if not equal:
            failures.append(key)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"tensor_count": len(report), "tensors": report}, handle, indent=2
            )
            handle.write("\n")
    if failures:
        raise ValueError(
            f"Exact tensor comparison failed for {len(failures)} keys: {failures[:20]}"
        )
    return len(report)


def set_load_with_bias(path: Path, enabled: bool) -> None:
    """Atomically change inference routing mode without touching checkpoint tensors."""
    config_path = path / "config.json" if path.is_dir() else path
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["load_with_bias"] = enabled
    temporary = config_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(config_path)


def validate_repo_scripts(path: Path) -> int:
    """Statically guard every active launcher against architecture drift."""
    chimera_dir = (
        path / "examples/chimera" if (path / "examples/chimera").is_dir() else path
    )
    required: dict[str, list[str]] = {
        "train.sh": [
            "--num-layers 25",
            '"[0]*2+[1]*23"',
            "--qk-layernorm",
            "--num-experts 32",
            "--moe-ffn-hidden-size 2048",
            "--moe-router-load-balancing-type quantile_balancing",
            "--moe-qb-num-bins 1000",
            "--moe-qb-ema-decay 0.0",
            "--moe-aux-loss-coeff 0.0",
            "--moe-router-score-function sigmoid",
            "--moe-router-bias-update-rate 0.0",
            "--moe-router-topk-scaling-factor 2.5",
            "--moe-z-loss-coeff 0.001",
            "--max-position-embeddings 8192",
        ],
        "train_440B.sh": [
            "--num-layers 25",
            '"[0]*2+[1]*23"',
            "--qk-layernorm",
            "--num-experts 32",
            "--moe-ffn-hidden-size 2048",
            "--moe-router-load-balancing-type quantile_balancing",
            "--moe-qb-num-bins 1000",
            "--moe-qb-ema-decay 0.0",
            "--moe-aux-loss-coeff 0.0",
            "--moe-router-score-function sigmoid",
            "--moe-router-bias-update-rate 0.0",
            "--moe-router-topk-scaling-factor 2.5",
            "--moe-z-loss-coeff 0.001",
            "--max-position-embeddings 8192",
        ],
        "tiny_chimera.sh": [
            "--qk-layernorm",
            "--num-experts 8",
            '"[0]*2+[1]*6"',
            "--moe-ffn-hidden-size 256",
            "MOE_AUX_LOSS_COEFF:-0.0",
            "--moe-router-score-function sigmoid",
            "MOE_ROUTER_LOAD_BALANCING_TYPE:-quantile_balancing",
            "MOE_ROUTER_BIAS_UPDATE_RATE:-0.0",
            "--moe-router-topk-scaling-factor 2.5",
            "MOE_Z_LOSS_COEFF:-0.001",
            "MAX_POSITION_EMBEDDINGS:-8192",
        ],
        "sft.sh": [
            "--qk-layernorm",
            "NUM_EXPERTS:-32",
            "MOE_FFN_HIDDEN_SIZE:-2048",
            "MOE_AUX_LOSS_COEFF:-0.0",
            "--moe-router-score-function sigmoid",
            "MOE_ROUTER_LOAD_BALANCING_TYPE:-none",
            "MOE_ROUTER_BIAS_UPDATE_RATE:-0.0",
            "--moe-router-topk-scaling-factor 2.5",
            "--moe-z-loss-coeff 0.001",
            "MAX_POSITION_EMBEDDINGS:-8192",
        ],
        "simpo.sh": [
            "--qk-layernorm",
            "NUM_EXPERTS:-32",
            "MOE_FFN_HIDDEN_SIZE:-2048",
            "MOE_AUX_LOSS_COEFF:-0.0",
            "--moe-router-score-function sigmoid",
            "MOE_ROUTER_LOAD_BALANCING_TYPE:-none",
            "MOE_ROUTER_BIAS_UPDATE_RATE:-0.0",
            "--moe-router-topk-scaling-factor 2.5",
            "--moe-z-loss-coeff 0.001",
            "MAX_POSITION_EMBEDDINGS:-8192",
        ],
    }
    forbidden = [
        "--num-experts 64",
        "--moe-ffn-hidden-size 1024",
        "--moe-shared-expert-intermediate-size",
        "--moe-shared-expert-overlap",
        "seq_aux_loss",
        "--max-position-embeddings 32768",
        "--rotary-scaling-factor 4.0",
    ]
    errors: list[str] = []
    for filename, tokens in required.items():
        script = chimera_dir / filename
        if not script.is_file():
            errors.append(f"missing launcher: {script}")
            continue
        text = script.read_text(encoding="utf-8")
        errors.extend(
            f"{filename}: missing {token!r}" for token in tokens if token not in text
        )
        errors.extend(
            f"{filename}: forbidden stale token {token!r}"
            for token in forbidden
            if token in text
        )
    if errors:
        raise ValueError(
            "Chimera launcher contract violation:\n- " + "\n- ".join(errors)
        )
    validate_run_config(chimera_dir / "run_config.yaml", "full")
    return len(required)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hf_parser = subparsers.add_parser("validate-hf")
    hf_parser.add_argument("path", type=Path)
    hf_parser.add_argument(
        "--profile", choices=["auto", "full", "tiny"], default="auto"
    )
    hf_parser.add_argument("--weights", action="store_true")

    run_parser = subparsers.add_parser("validate-run-config")
    run_parser.add_argument("path", type=Path)
    run_parser.add_argument(
        "--profile", choices=["auto", "full", "tiny"], default="auto"
    )

    compare_parser = subparsers.add_parser("compare-hf")
    compare_parser.add_argument("expected", type=Path)
    compare_parser.add_argument("actual", type=Path)
    compare_parser.add_argument("--report", type=Path)

    bias_parser = subparsers.add_parser("set-load-with-bias")
    bias_parser.add_argument("path", type=Path)
    bias_parser.add_argument("value", choices=["true", "false"])

    repo_parser = subparsers.add_parser("validate-repo")
    repo_parser.add_argument("path", type=Path, nargs="?", default=Path.cwd())

    args = parser.parse_args()
    if args.command == "validate-hf":
        profile, config = validate_hf_config(args.path, args.profile)
        count = validate_hf_weights(args.path, config) if args.weights else None
        print(
            f"Validated HF Chimera profile={profile}"
            + (f" tensors={count}" if count is not None else "")
        )
    elif args.command == "validate-run-config":
        profile = validate_run_config(args.path, args.profile)
        print(f"Validated MCore Chimera profile={profile}")
    elif args.command == "compare-hf":
        count = compare_hf_weights(args.expected, args.actual, args.report)
        print(f"Exact HF tensor comparison passed tensors={count}")
    elif args.command == "set-load-with-bias":
        set_load_with_bias(args.path, args.value == "true")
        profile, _ = validate_hf_config(args.path)
        print(
            f"Set load_with_bias={args.value} for validated HF Chimera profile={profile}"
        )
    else:
        count = validate_repo_scripts(args.path)
        print(f"Validated Chimera launchers={count} and canonical run_config.yaml")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
