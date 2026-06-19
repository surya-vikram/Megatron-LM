# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Chimera GPT/MoE pretraining entrypoint.

Megatron-LM exposes YaRN model support, but the generic GPT CLI does not expose
the Chimera-specific YaRN fields that are needed to recreate the HF checkpoint.
This entrypoint injects those fields and then reuses the standard GPT dataset,
forward step, and training loop.
"""

import os
import sys
import time
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from gpt_builders import gpt_builder
from megatron.core.enums import ModelType
from megatron.training import inprocess_restart, pretrain, set_startup_timestamps
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from model_provider import model_provider
from pretrain_gpt import (
    _PROGRAM_START_TIME,
    forward_step,
    get_embedding_ranks,
    train_valid_test_datasets_provider,
)

try:
    from megatron.post_training.arguments import add_modelopt_args

    HAS_NVIDIA_MODELOPT = True
except ImportError:
    HAS_NVIDIA_MODELOPT = False


CHIMERA_YARN = {
    "yarn_rotary_scaling_factor": 4.0,
    "yarn_original_max_position_embeddings": 8192,
    "yarn_beta_fast": 32.0,
    "yarn_beta_slow": 1.0,
    "yarn_mscale": None,
    "yarn_mscale_all_dim": None,
    "yarn_correction_range_round_to_int": False,
}


def apply_chimera_yarn_args(args):
    """Attach Chimera YaRN metadata to args for config creation and checkpoint metadata."""
    for name, value in CHIMERA_YARN.items():
        setattr(args, name, value)

    # TransformerConfig also carries non-prefixed YaRN fields. Set both naming
    # conventions because GPTModel currently reads the prefixed attributes.
    args.rotary_scaling_factor = CHIMERA_YARN["yarn_rotary_scaling_factor"]
    args.original_max_position_embeddings = CHIMERA_YARN["yarn_original_max_position_embeddings"]
    args.beta_fast = CHIMERA_YARN["yarn_beta_fast"]
    args.beta_slow = CHIMERA_YARN["yarn_beta_slow"]
    args.mscale = CHIMERA_YARN["yarn_mscale"]
    args.mscale_all_dim = CHIMERA_YARN["yarn_mscale_all_dim"]
    return args


def chimera_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Build Chimera as a standard MCore GPT MoE model with Chimera YaRN metadata."""
    args = apply_chimera_yarn_args(args)
    if config is None:
        config = core_transformer_config_from_args(args)

    for name, value in CHIMERA_YARN.items():
        setattr(config, name, value)
    config.rotary_scaling_factor = CHIMERA_YARN["yarn_rotary_scaling_factor"]
    config.original_max_position_embeddings = CHIMERA_YARN["yarn_original_max_position_embeddings"]
    config.beta_fast = CHIMERA_YARN["yarn_beta_fast"]
    config.beta_slow = CHIMERA_YARN["yarn_beta_slow"]
    config.mscale = CHIMERA_YARN["yarn_mscale"]
    config.mscale_all_dim = CHIMERA_YARN["yarn_mscale_all_dim"]

    return gpt_builder(args, pre_process, post_process, vp_stage=vp_stage, config=config, pg_collection=pg_collection)


if __name__ == "__main__":
    main_entry_time = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=main_entry_time)

    setattr(train_valid_test_datasets_provider, "is_distributed", True)

    wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    args = parse_and_validate_args(
        extra_args_provider=add_modelopt_args if HAS_NVIDIA_MODELOPT else None,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    args = apply_chimera_yarn_args(args)
    full_config = pretrain_cfg_container_from_args(args)

    wrapped_pretrain(
        full_config,
        train_valid_test_datasets_provider,
        partial(model_provider, chimera_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
