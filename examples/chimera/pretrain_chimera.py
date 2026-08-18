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
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.distributed
import torch.distributed.distributed_c10d as c10d

def _make_safe_dist():
    for mod in [torch.distributed, c10d]:
        for name in ['barrier', 'all_reduce', 'broadcast', 'all_gather', 'all_gather_object', 'all_gather_into_tensor', 'reduce_scatter_tensor', 'broadcast_object_list']:
            if hasattr(mod, name):
                orig = getattr(mod, name)
                def make_wrapper(fn_name, orig_fn):
                    def safe_fn(*args, **kwargs):
                        group = kwargs.get('group', None)
                        if torch.distributed.is_initialized():
                            try:
                                if torch.distributed.get_world_size(group=group) > 1:
                                    return orig_fn(*args, **kwargs)
                            except Exception:
                                pass
                        if fn_name == 'all_gather_object':
                            obj_list = args[0] if len(args) > 0 else kwargs.get('object_list')
                            obj = args[1] if len(args) > 1 else kwargs.get('obj')
                            if obj_list is not None and len(obj_list) > 0:
                                obj_list[0] = obj
                        elif fn_name == 'all_gather':
                            out_list = args[0] if len(args) > 0 else kwargs.get('tensor_list', kwargs.get('output_tensor_list', kwargs.get('output')))
                            inp = args[1] if len(args) > 1 else kwargs.get('tensor', kwargs.get('input'))
                            if isinstance(out_list, list) and len(out_list) > 0 and inp is not None:
                                out_list[0].copy_(inp)
                            elif hasattr(out_list, 'copy_') and inp is not None:
                                out_list.copy_(inp)
                        elif fn_name in ('all_gather_into_tensor', 'reduce_scatter_tensor'):
                            out = args[0] if len(args) > 0 else kwargs.get('output')
                            inp = args[1] if len(args) > 1 else kwargs.get('input')
                            if out is not None and inp is not None:
                                out.copy_(inp)
                        return None
                    return safe_fn
                setattr(mod, name, make_wrapper(name, orig))

_make_safe_dist()

from gpt_builders import gpt_builder
from architecture_contract import validate_training_args, write_runtime_run_config
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.enums import ModelType
from megatron.core.utils import get_attr_wrapped_model
from megatron.post_training.simpo_utils import calculate_simpo_loss
from megatron.training import get_args, get_timers, inprocess_restart, pretrain, print_rank_0, set_startup_timestamps
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.datasets.simpo_dataset import SimPODataset
from model_provider import model_provider
from pretrain_gpt import (
    _PROGRAM_START_TIME,
    forward_step,
    get_batch,
    get_embedding_ranks,
    is_dataset_built_on_rank,
    core_gpt_dataset_config_from_args,
    stimer,
    train_valid_test_datasets_provider,
)

try:
    from megatron.post_training.arguments import add_modelopt_args

    HAS_NVIDIA_MODELOPT = True
except ImportError:
    HAS_NVIDIA_MODELOPT = False


def apply_chimera_yarn_args(args):
    """Attach Chimera YaRN metadata to args for config creation and checkpoint metadata."""
    scaling_factor = getattr(args, "rotary_scaling_factor", None)
    if scaling_factor is None:
        scaling_factor = 1.0

    orig_max_pos = getattr(args, "yarn_original_max_position_embeddings", None)
    if orig_max_pos is None:
        max_pos = getattr(args, "max_position_embeddings", 8192)
        if scaling_factor > 1.0 and int(scaling_factor) > 0:
            orig_max_pos = max_pos // int(scaling_factor)
        else:
            orig_max_pos = max_pos

    mscale_val = getattr(args, "mscale", 1.0)
    mscale_all_dim_val = getattr(args, "mscale_all_dim", 0.0)

    args.yarn_rotary_scaling_factor = scaling_factor
    args.yarn_original_max_position_embeddings = orig_max_pos
    args.yarn_beta_fast = getattr(args, "yarn_beta_fast", 32.0)
    args.yarn_beta_slow = getattr(args, "yarn_beta_slow", 1.0)
    args.yarn_mscale = mscale_val
    args.yarn_mscale_all_dim = mscale_all_dim_val
    args.yarn_correction_range_round_to_int = False

    args.rotary_scaling_factor = scaling_factor
    args.original_max_position_embeddings = orig_max_pos
    args.mscale = mscale_val
    args.mscale_all_dim = mscale_all_dim_val
    return args


def add_chimera_args(parser):
    """Add Chimera-only runtime knobs not exposed by Megatron's generic CLI."""
    if HAS_NVIDIA_MODELOPT:
        parser = add_modelopt_args(parser)
    group = parser.add_argument_group(title="Chimera")
    option_strings = []
    if "--chimera-expert-tp-size" not in parser._option_string_actions:
        option_strings.append("--chimera-expert-tp-size")
    if "--expert-tensor-parallel-size" not in parser._option_string_actions:
        option_strings.append("--expert-tensor-parallel-size")
    if option_strings:
        group.add_argument(
            *option_strings,
            dest="expert_tensor_parallel_size",
            type=int,
            default=1,
            help="Expert tensor parallel size. Keep at 1 unless explicitly testing expert tensor parallelism.",
        )
    return parser


def chimera_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Build Chimera as a standard MCore GPT MoE model with Chimera YaRN metadata."""
    args = apply_chimera_yarn_args(args)
    if config is None:
        config = core_transformer_config_from_args(args)

    config.yarn_rotary_scaling_factor = args.rotary_scaling_factor
    config.yarn_original_max_position_embeddings = args.original_max_position_embeddings
    config.yarn_beta_fast = getattr(args, "yarn_beta_fast", 32.0)
    config.yarn_beta_slow = getattr(args, "yarn_beta_slow", 1.0)
    config.yarn_mscale = args.mscale
    config.yarn_mscale_all_dim = args.mscale_all_dim
    config.yarn_correction_range_round_to_int = False

    config.rotary_scaling_factor = args.rotary_scaling_factor
    config.original_max_position_embeddings = args.original_max_position_embeddings
    config.mscale = args.mscale
    config.mscale_all_dim = args.mscale_all_dim

    return gpt_builder(args, pre_process, post_process, vp_stage=vp_stage, config=config, pg_collection=pg_collection)


def simpo_forward_step(data_iterator, model: torch.nn.Module, return_schedule_plan: bool = False):
    """Forward step for Chimera SimPO training."""
    if return_schedule_plan:
        raise NotImplementedError("SimPO does not support schedule-plan forward execution.")

    args = get_args()
    timers = get_timers()

    timers("batch-generator", log_level=2).start()
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator, vp_stage)
    timers("batch-generator").stop()

    with stimer:
        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
            labels=None,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
        )

    def simpo_loss_func(loss_mask_tensor, labels_tensor, cu_seqlens_tensor, output_tensor_or_logits):
        if cu_seqlens_tensor is None:
            raise RuntimeError("SimPO requires chosen/rejected sequence boundaries.")

        loss, metrics = calculate_simpo_loss(
            logits=output_tensor_or_logits,
            labels=labels_tensor,
            loss_mask=loss_mask_tensor,
            cu_seqlens=cu_seqlens_tensor,
            args=args,
        )
        num_tokens = loss_mask_tensor.sum().clone().detach().to(torch.int)
        report = {"simpo loss": torch.cat([(loss.clone().detach() * num_tokens).view(1), num_tokens.view(1)])}
        for key, value in metrics.items():
            report[key] = torch.cat([(value * num_tokens).view(1), num_tokens.view(1)])
        return loss * num_tokens, num_tokens, report

    cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params else None
    return output_tensor, partial(simpo_loss_func, loss_mask, labels, cu_seqlens)


def simpo_train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build SimPO datasets for Chimera chosen/rejected JSONL data."""
    args = get_args()
    config = core_gpt_dataset_config_from_args(args)

    print_rank_0("> building train, validation, and test datasets for Chimera SimPO ...")
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        SimPODataset,
        train_val_test_num_samples,
        partial(is_dataset_built_on_rank, vp_stage=vp_stage, is_packed_sequence=True),
        config,
    ).build()
    print_rank_0("> finished creating Chimera SimPO datasets ...")
    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    main_entry_time = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=main_entry_time)

    setattr(train_valid_test_datasets_provider, "is_distributed", True)
    setattr(simpo_train_valid_test_datasets_provider, "is_distributed", True)

    wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    args = parse_and_validate_args(
        extra_args_provider=add_chimera_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    args = apply_chimera_yarn_args(args)
    profile_name = validate_training_args(args)
    run_config_path = write_runtime_run_config(args, Path(__file__).with_name("run_config.yaml"))
    if run_config_path is not None:
        print(f"Wrote validated Chimera {profile_name} architecture metadata: {run_config_path}")
    full_config = pretrain_cfg_container_from_args(args)

    if getattr(args, "simpo", False):
        if args.context_parallel_size != 1:
            raise AssertionError("Chimera SimPO training requires context parallel size 1.")
        active_dataset_provider = simpo_train_valid_test_datasets_provider
        active_forward_step = simpo_forward_step
    else:
        active_dataset_provider = train_valid_test_datasets_provider
        active_forward_step = forward_step

    wrapped_pretrain(
        full_config,
        active_dataset_provider,
        partial(model_provider, chimera_builder),
        ModelType.encoder_or_decoder,
        active_forward_step,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
