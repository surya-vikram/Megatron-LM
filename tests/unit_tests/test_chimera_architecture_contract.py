# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from examples.chimera.architecture_contract import (
    CONTEXT_PHASES,
    FULL_PROFILE,
    validate_run_config,
    validate_training_args,
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
