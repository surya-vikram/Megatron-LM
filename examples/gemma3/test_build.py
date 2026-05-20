import sys
import os

# Mocking enough to test model building
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from examples.gemma3.pretrain_gemma3_mcore import gemma3_model_builder
from megatron.training.arguments import parse_and_validate_args

class MockArgs:
    num_layers = 2
    hidden_size = 1152 # 1B provider
    num_attention_heads = 4
    num_query_groups = 1
    kv_channels = 256
    ffn_hidden_size = 6912
    seq_length = 1024
    padded_vocab_size = 262144
    tensor_model_parallel_size = 1
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    bf16 = True
    fp16 = False
    window_size = 512

print("Testing Gemma3 model building...")
try:
    model = gemma3_model_builder(MockArgs(), pre_process=True, post_process=True)
    print("Model built successfully!")
    print(model)
except Exception as e:
    print(f"Failed to build model: {e}")
    import traceback
    traceback.print_exc()
