#!/usr/bin/env python3
import os

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from transformers import AutoTokenizer

from megatron.bridge.models.gemma.gemma3_provider import Gemma3ModelProvider1B
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedInitConfig,
    GPTDatasetConfig,
    LoggerConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


def getenv_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def getenv_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def main() -> None:
    data_path = os.environ["DATA_PATH"]
    hf_model_path = os.environ["HF_MODEL_PATH"]
    mcore_checkpoint = os.environ["MCORE_CHECKPOINT"]
    save_path = os.environ["SAVE_PATH"]

    seq_length = getenv_int("SEQ_LENGTH", 2048)
    micro_batch_size = getenv_int("MICRO_BATCH_SIZE", 1)
    global_batch_size = getenv_int("GLOBAL_BATCH_SIZE", 1)
    train_iters = getenv_int("TRAIN_ITERS", 100)
    learning_rate = getenv_float("LEARNING_RATE", 1e-5)
    min_learning_rate = getenv_float("MIN_LEARNING_RATE", 1e-6)
    lr_warmup_iters = getenv_int("LR_WARMUP_ITERS", 10)
    lr_decay_iters = getenv_int("LR_DECAY_ITERS", train_iters)
    save_interval = getenv_int("SAVE_INTERVAL", train_iters)
    eval_interval = getenv_int("EVAL_INTERVAL", 1000)
    eval_iters = getenv_int("EVAL_ITERS", 0)

    model = Gemma3ModelProvider1B()
    model.tensor_model_parallel_size = 1
    model.pipeline_model_parallel_size = 1
    model.pipeline_model_parallel_layout = None
    model.pipeline_dtype = None
    model.virtual_pipeline_model_parallel_size = None
    model.context_parallel_size = 1
    model.sequence_parallel = False
    model.seq_length = seq_length
    model.transformer_impl = "transformer_engine"
    model.cuda_graph_impl = "none"
    model.cuda_graph_scope = "full"
    model.cuda_graph_warmup_steps = 3
    model.attention_backend = None
    model.cross_entropy_loss_fusion = True
    model.cross_entropy_fusion_impl = "native"
    model.recompute_granularity = None
    model.recompute_modules = None
    model.fine_grained_activation_offloading = False
    model.offload_modules = None
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if len(tokenizer) > model.vocab_size:
        model.vocab_size = len(tokenizer)

    optimizer = OptimizerConfig(
        optimizer="adam",
        lr=learning_rate,
        min_lr=min_learning_rate,
        weight_decay=0.1,
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        use_distributed_optimizer=True,
        clip_grad=1.0,
    )
    scheduler = SchedulerConfig(
        start_weight_decay=0.033,
        end_weight_decay=0.033,
        weight_decay_incr_style="constant",
        lr_decay_style="cosine",
        lr_wsd_decay_style="minus_sqrt",
        lr_warmup_iters=lr_warmup_iters,
        lr_warmup_init=0.0,
        lr_decay_iters=lr_decay_iters,
        lr_wsd_decay_iters=lr_decay_iters,
        override_opt_param_scheduler=True,
    )

    cfg = ConfigContainer(
        model=model,
        train=TrainingConfig(
            train_iters=train_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        validation=ValidationConfig(
            eval_interval=eval_interval,
            eval_iters=eval_iters,
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            data_parallel_sharding_strategy="no_shard",
            use_distributed_optimizer=True,
        ),
        dataset=GPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            seq_length=seq_length,
            num_dataset_builder_threads=1,
            blend=None,
            blend_per_split=None,
            split="100,0,0",
            data_sharding=True,
            dataloader_type="single",
            skip_getting_attention_mask_from_dataset=True,
            data_path=data_path,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=os.path.join(save_path, "tb_logs"),
            log_timers_to_tensorboard=True,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_path,
        ),
        checkpoint=CheckpointConfig(
            save=save_path,
            save_interval=save_interval,
            load=save_path,
            pretrained_checkpoint=mcore_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=1234),
        dist=DistributedInitConfig(),
        comm_overlap=None,
        mixed_precision="bf16_mixed",
    )

    pretrain(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
