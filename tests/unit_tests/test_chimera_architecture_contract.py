# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from examples.chimera.architecture_contract import (
    CONTEXT_PHASES,
    FULL_PROFILE,
    validate_hf_config,
    validate_run_config,
    validate_training_args,
    write_runtime_run_config,
)

CHIMERA_DIR = Path(__file__).parents[2] / "examples" / "chimera"


def _training_args(phase: str, **overrides):
    geometry = CONTEXT_PHASES[phase]
    values = {
        **FULL_PROFILE,
        "num_experts": FULL_PROFILE["num_moe_experts"],
        "max_position_embeddings": geometry["max_position_embeddings"],
        "position_embedding_type": "yarn",
        "rotary_base": 10_000_000,
        "rotary_scaling_factor": geometry["rotary_scaling_factor"],
        "yarn_original_max_position_embeddings": 8192,
        "yarn_correction_range_round_to_int": False,
        "mscale": 1.0,
        "mscale_all_dim": 0.0,
        "vocab_size": 50176,
        "layernorm_epsilon": 1e-5,
        "moe_router_score_function": "sigmoid",
        "moe_router_topk_scaling_factor": 2.5,
        "moe_router_bias_update_rate": 0.0,
        "moe_aux_loss_coeff": 0.0,
        "moe_z_loss_coeff": 0.001,
        "qk_layernorm": True,
        "moe_router_enable_expert_bias": True,
        "moe_shared_expert_intermediate_size": None,
        "untie_embeddings_and_output_weights": True,
        "finetune": False,
        "sft": False,
        "simpo": False,
        "moe_router_load_balancing_type": "quantile_balancing",
        "moe_qb_num_bins": 1000,
        "moe_qb_ema_decay": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("phase", CONTEXT_PHASES)
def test_training_contract_accepts_every_context_phase(phase):
    assert validate_training_args(_training_args(phase)) == "full"


def test_context_extension_finetune_keeps_pretraining_router_contract():
    args = _training_args("32k", finetune=True)
    assert validate_training_args(args) == "full"


def test_sft_uses_posttraining_router_contract():
    args = _training_args(
        "128k", sft=True, moe_router_load_balancing_type="none"
    )
    assert validate_training_args(args) == "full"


@pytest.mark.parametrize("phase", CONTEXT_PHASES)
def test_run_config_accepts_every_context_phase(tmp_path, phase):
    with (CHIMERA_DIR / "run_config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    geometry = CONTEXT_PHASES[phase]
    model = config["model"]
    model["seq_length"] = geometry["max_position_embeddings"]
    model["chimera_context_phase"] = phase
    model["rotary_scaling_factor"] = geometry["rotary_scaling_factor"]
    model["yarn_rotary_scaling_factor"] = geometry["rotary_scaling_factor"]
    output = tmp_path / "run_config.yaml"
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)

    assert validate_run_config(output, "full", phase) == "full"


def test_context_phase_rejects_mismatched_factor(tmp_path):
    with (CHIMERA_DIR / "run_config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["model"]["seq_length"] = 32768
    output = tmp_path / "run_config.yaml"
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)

    with pytest.raises(ValueError, match="YaRN geometry"):
        validate_run_config(output, "full")


def _hf_config(phase):
    geometry = CONTEXT_PHASES[phase]
    rope = {
        "rope_type": "yarn",
        "factor": geometry["rotary_scaling_factor"],
        "original_max_position_embeddings": 8192,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 0.0,
        "truncate": False,
    }
    return {
        "architectures": ["ChimeraForCausalLM"],
        "num_hidden_layers": 25,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "n_routed_experts": 32,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 2048,
        "first_k_dense_replace": 2,
        "last_k_dense_replace": 0,
        "context_phase": phase,
        "position_embedding_type": "yarn",
        "max_position_embeddings": geometry["max_position_embeddings"],
        "original_max_position_embeddings": 8192,
        "rope_scaling": rope,
        "rms_norm_eps": 1e-5,
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
        "router_load_balancing_type": "quantile_balancing",
        "tie_word_embeddings": False,
        "vocab_size": 50176,
        "load_with_bias": True,
    }


@pytest.mark.parametrize("phase", CONTEXT_PHASES)
def test_hf_config_accepts_explicit_yarn_phase(tmp_path, phase):
    output = tmp_path / "config.json"
    output.write_text(json.dumps(_hf_config(phase)), encoding="utf-8")

    assert validate_hf_config(output, "full", phase)[0] == "full"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("context_phase", "8k", "context_phase"),
        ("position_embedding_type", "rope", "position_embedding_type"),
    ],
)
def test_hf_config_rejects_inconsistent_phase_metadata(tmp_path, field, value, error):
    config = _hf_config("32k")
    config[field] = value
    output = tmp_path / "config.json"
    output.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        validate_hf_config(output, "full", "32k")


def test_hf_config_rejects_integer_correction_bounds(tmp_path):
    config = _hf_config("128k")
    config["rope_scaling"]["truncate"] = True
    output = tmp_path / "config.json"
    output.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="truncate"):
        validate_hf_config(output, "full", "128k")


def test_runtime_run_config_serializes_explicit_context_phase(tmp_path):
    args = _training_args(
        "64k",
        save=str(tmp_path),
        yarn_rotary_scaling_factor=8.0,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        yarn_mscale=1.0,
        yarn_mscale_all_dim=0.0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )

    output = write_runtime_run_config(args, CHIMERA_DIR / "run_config.yaml")

    assert output == tmp_path / "run_config.yaml"
    assert validate_run_config(output, "full", "64k") == "full"
    with output.open(encoding="utf-8") as handle:
        model = yaml.safe_load(handle)["model"]
    assert model["chimera_context_phase"] == "64k"
    assert model["yarn_correction_range_round_to_int"] is False
