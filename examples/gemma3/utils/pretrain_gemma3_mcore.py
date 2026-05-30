# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain Gemma3 using Megatron-LM training loop and Megatron-Bridge provider."""

import os
import sys

# Ensure Megatron-LM and Megatron-Bridge are in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import torch
from functools import partial

from pretrain_gpt import (
    forward_step,
    train_valid_test_datasets_provider,
    get_embedding_ranks,
    _PROGRAM_START_TIME,
    get_batch,
    stimer,
    is_dataset_built_on_rank,
    core_gpt_dataset_config_from_args
)

from megatron.training.utils import get_timers
from megatron.core.utils import get_attr_wrapped_model
from megatron.post_training.simpo_utils import calculate_simpo_loss
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.training.datasets.simpo_dataset import SimPODataset

def simpo_forward_step(data_iterator, model: torch.nn.Module, return_schedule_plan: bool = False):
    if data_iterator is None:
        return torch.tensor(0.0, device=torch.cuda.current_device()), lambda x: (torch.tensor(0.0, device=torch.cuda.current_device()), {})
    args = get_args()
    timers = get_timers()

    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator, vp_stage)
    timers('batch-generator').stop()

    use_cce = getattr(args, 'use_cce_simpo', False)

    if use_cce:
        # Avoid materializing the logits: return the transformer hidden states instead
        def simpo_output_processor(hidden_states, **kwargs):
            return hidden_states

        with stimer:
            output_tensor = model(
                tokens, position_ids, attention_mask, labels=None, loss_mask=loss_mask, 
                packed_seq_params=packed_seq_params, output_processor=simpo_output_processor
            )
    else:
        with stimer:
            output_tensor = model(
                tokens, position_ids, attention_mask, labels=None, loss_mask=loss_mask, packed_seq_params=packed_seq_params
            )

    def simpo_loss_func(loss_mask_tensor, labels_tensor, cu_seqlens_tensor, output_tensor_or_logits):
        if use_cce:
            # output_tensor_or_logits is the hidden states! Fetch output layer weights for CCE.
            output_layer_weight = get_attr_wrapped_model(model, "output_layer").weight
            loss, metrics = calculate_simpo_loss(
                logits=output_tensor_or_logits,
                labels=labels_tensor,
                loss_mask=loss_mask_tensor,
                cu_seqlens=cu_seqlens_tensor,
                args=args,
                output_layer_weight=output_layer_weight
            )
        else:
            loss, metrics = calculate_simpo_loss(
                logits=output_tensor_or_logits,
                labels=labels_tensor,
                loss_mask=loss_mask_tensor,
                cu_seqlens=cu_seqlens_tensor,
                args=args
            )
        
        num_tokens = loss_mask_tensor.sum().clone().detach().to(torch.int)
        
        # Format metrics for Megatron's logger: expect [metric_sum, num_tokens]
        # Our metrics are already averaged, so we multiply by num_tokens to get the sum
        report = {}
        report['simpo loss'] = torch.cat([(loss.clone().detach() * num_tokens).view(1), num_tokens.view(1)])
        for k, v in metrics.items():
            report[k] = torch.cat([(v * num_tokens).view(1), num_tokens.view(1)])
            
        return loss * num_tokens, num_tokens, report

    cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params else None
    return output_tensor, partial(simpo_loss_func, loss_mask, labels, cu_seqlens)

def simpo_train_valid_test_datasets_provider(train_val_test_num_samples):
    args = get_args()
    config = core_gpt_dataset_config_from_args(args)
    
    print_rank_0("> building train, validation, and test datasets for SimPO ...")
    
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        SimPODataset,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()
    
    print_rank_0("> finished creating SimPO datasets ...")
    return train_ds, valid_ds, test_ds

from megatron.training import (
    get_args,
    get_tokenizer,
    print_rank_0,
    set_startup_timestamps,
    pretrain
)
from megatron.training.arguments import parse_and_validate_args
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.core.enums import ModelType
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from model_provider import model_provider

# Patch tokenizers for SFT compatibility with offline Gemma 3 templates
try:
    from megatron.core.tokenizers.text.text_tokenizer import HuggingFaceTokenizer
    def tokenize_conversation(self, conversation):
        """Patch for Megatron SFTDataset using native HF Chat Templates offline."""
        template_path = os.path.join(os.path.dirname(__file__), 'gemma3_chat_template.jinja')
        if os.path.exists(template_path):
            with open(template_path, 'r') as tf:
                local_template = tf.read()
        else:
            raise FileNotFoundError("Local chat template not found at " + template_path)
            
        prompt = self.tokenizer.apply_chat_template(conversation, chat_template=local_template, tokenize=False, add_generation_prompt=False)
        tokens = self.tokenize(prompt)
        return tokens, tokens
    HuggingFaceTokenizer.tokenize_conversation = tokenize_conversation
except Exception:
    pass

try:
    from megatron.core.tokenizers.text.libraries.sft_tokenizer import SFTTokenizer
    orig_init = SFTTokenizer.__init__
    def new_init(self, tokenizer_path, prompt_format):
        orig_init(self, tokenizer_path, prompt_format)
        if prompt_format == "gemma3":
            template_path = os.path.join(os.path.dirname(__file__), 'gemma3_chat_template.jinja')
            if os.path.exists(template_path):
                with open(template_path, 'r') as tf:
                    local_template = tf.read()
                self._tokenizer.chat_template = local_template
                self._prompt_config.custom_chat_template = local_template
                
                # Re-derive assistant header tokens after setting the template!
                try:
                    conv = [{"role": "user", "content": ""}]
                    full = self._tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False, chat_template=self._prompt_config.custom_chat_template)
                    base = self._tokenizer.apply_chat_template(conv, add_generation_prompt=False, tokenize=False, chat_template=self._prompt_config.custom_chat_template)
                    prefix_text = full[len(base):]
                    self._assistant_header = self._tokenizer.encode(prefix_text, add_special_tokens=False)
                    if self._prompt_config.has_bos and len(self._assistant_header) > 0 and self._assistant_header[0] == self._tokenizer.bos_token_id:
                        self._assistant_header = self._assistant_header[1:]
                except Exception:
                    pass
            else:
                raise FileNotFoundError("Local chat template not found at " + template_path)
    SFTTokenizer.__init__ = new_init
except Exception:
    pass

try:
    from megatron.post_training.arguments import add_modelopt_args
    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

def add_cce_simpo_args(parser):
    group = parser.add_argument_group(title='SimPO CCE Options')
    group.add_argument('--use-cce-simpo', action='store_true', help='Use Apple Cut Cross-Entropy for memory-efficient SimPO.')
    group.add_argument('--debug-dataset', action='store_true', help='Print step-by-step token packing trace.')
    group.add_argument('--log-dataset-stats', action='store_true', help='Dumps aggregate packing density every 100 steps.')
    group.add_argument('--warn-oversized-samples', action='store_true', help='Flags any samples that exceed sequence length.')
    if has_nvidia_modelopt:
        parser = add_modelopt_args(parser)
    return parser

def gemma3_model_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Gemma3 model builder that relies on Megatron-Bridge for the model implementation."""
    print_rank_0('building Gemma3 model via Megatron-Bridge ...')
    
    try:
        from megatron.bridge.models.gemma.gemma3_provider import (
            Gemma3ModelProvider1B,
            Gemma3ModelProvider4B,
            Gemma3ModelProvider12B,
            Gemma3ModelProvider27B
        )
    except ImportError:
        raise ImportError("Megatron-Bridge is required for Gemma3 training.")

    if args.hidden_size == 1152:
        provider_cls = Gemma3ModelProvider1B
    elif args.hidden_size == 2560:
        provider_cls = Gemma3ModelProvider4B
    elif args.hidden_size == 3840:
        provider_cls = Gemma3ModelProvider12B
    else:
        provider_cls = Gemma3ModelProvider1B

    provider = provider_cls()
    provider.num_layers = args.num_layers
    provider.hidden_size = args.hidden_size
    provider.num_attention_heads = args.num_attention_heads
    provider.num_query_groups = args.num_query_groups
    provider.kv_channels = args.kv_channels
    provider.ffn_hidden_size = args.ffn_hidden_size
    provider.seq_length = args.seq_length
    provider.vocab_size = args.padded_vocab_size
    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    provider.context_parallel_size = args.context_parallel_size
    
    if hasattr(args, 'window_size') and args.window_size is not None:
        provider.window_size = args.window_size[0] if isinstance(args.window_size, (list, tuple)) else args.window_size

    provider.bf16 = args.bf16
    provider.fp16 = args.fp16
    
    # Forward CCE flag: without this the provider never enables fused linear cross
    # entropy and the model materialises the full (seq*batch x vocab) logit tensor.
    # For SimPO, we MUST disable this to get raw logits for the margin loss.
    if getattr(args, 'simpo', False):
        provider.use_linear_cross_entropy = False
    else:
        provider.use_linear_cross_entropy = getattr(args, 'use_linear_cross_entropy', False)
        
    provider.finalize()
    return provider.provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

if __name__ == "__main__":
    import time
    _MAIN_ENTRY_TIME = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)
    
    args = parse_and_validate_args(
        extra_args_provider=add_cce_simpo_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    
    full_config = pretrain_cfg_container_from_args(args)
    
    # Ensure datasets are built in a distributed-aware way across all ranks to prevent deadlocks
    setattr(train_valid_test_datasets_provider, "is_distributed", True)
    setattr(simpo_train_valid_test_datasets_provider, "is_distributed", True)

    if getattr(args, 'simpo', False):
        assert args.context_parallel_size == 1, "SimPO training does not support Context Parallelism (CP > 1) because sequence-level length-normalization requires sequence boundaries to be local to the rank."
        active_dataset_provider = simpo_train_valid_test_datasets_provider
        active_forward_step = simpo_forward_step
    else:
        active_dataset_provider = train_valid_test_datasets_provider
        active_forward_step = forward_step
        
    pretrain(full_config,
        active_dataset_provider,
        partial(model_provider, gemma3_model_builder),
        ModelType.encoder_or_decoder,
        active_forward_step,
        get_embedding_ranks=get_embedding_ranks,
    )
