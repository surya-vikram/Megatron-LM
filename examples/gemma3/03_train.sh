#!/bin/bash
set -e
export DATA_PATH="/home/jovyan/data/gemma_medical_data_text_document"
export MCORE_CHECKPOINT="/home/jovyan/models/gemma-3-1b-pt-mcore"
export SAVE_PATH="/home/jovyan/models/gemma-3-1b-trained"
torchrun --nproc_per_node=1 /root/Megatron-LM/pretrain_gpt.py     --tensor-model-parallel-size 1     --pipeline-model-parallel-size 1     --sequence-parallel     --num-layers 18     --hidden-size 2048     --num-attention-heads 16     --num-query-groups 2     --seq-length 2048     --max-position-embeddings 8192     --micro-batch-size 1     --global-batch-size 1     --train-iters 100     --lr 1e-5     --min-lr 1e-6     --lr-decay-style cosine     --data-path $DATA_PATH     --tokenizer-type HuggingFaceTokenizer     --tokenizer-model /home/jovyan/models/gemma-3-1b-pt-hf     --load $MCORE_CHECKPOINT     --save $SAVE_PATH     --bf16     --save-interval 10     --eval-interval 1000     --eval-iters 0     --split 100,0,0
