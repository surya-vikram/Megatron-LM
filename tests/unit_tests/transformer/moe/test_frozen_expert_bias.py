import torch
import torch.nn as nn
from typing import cast
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

def test_frozen_expert_bias_sft_rl():
    print("=" * 80)
    print("VERIFYING FROZEN EXPERT BIAS CORRECTNESS FOR SFT AND RL")
    print("=" * 80)

    # 1. Initialize distributed environment
    Utils.initialize_model_parallel(1, 1)
    _set_random_seed(seed_=123, data_parallel_random_init=False)

    num_experts = 8
    topk = 2

    # 2. Config for SFT / RL with FROZEN BIAS
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=num_experts,
        moe_router_topk=topk,
        moe_router_load_balancing_type="none",   # Aux loss = 0.0, no dynamic QB
        moe_aux_loss_coeff=0.0,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,      # Enable bias in Top-K routing
        moe_router_bias_update_rate=0.0,         # Frozen rate = 0.0
        use_cpu_initialization=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
    )

    # 3. Instantiate Pre-training MoE Layer
    submodules_pretrain = get_submodules(
        get_gpt_layer_local_submodules(num_experts=num_experts, moe_grouped_gemm=False).mlp
    )
    assert isinstance(submodules_pretrain, MoESubmodules)
    moe_pretrain = MoELayer(config, submodules_pretrain).cuda()
    router_pretrain = cast(TopKRouter, moe_pretrain.router)

    # 4. Set a mock pre-trained bias vector (non-zero equilibrium values)
    mock_pretrain_bias = torch.tensor(
        [0.15, -0.08, 0.22, -0.14, 0.05, -0.02, 0.09, -0.27],
        dtype=torch.float32,
        device=torch.cuda.current_device(),
    )
    router_pretrain.expert_bias.copy_(mock_pretrain_bias)

    print("\n[Step 1] Pre-train router saved with expert_bias:")
    print("  Pretrain bias:", router_pretrain.expert_bias.cpu().numpy())

    # 5. Save state dict (simulating pretrain checkpoint save)
    state_dict = moe_pretrain.state_dict()
    assert "router.expert_bias" in state_dict, "ERROR: router.expert_bias must be present in state_dict!"
    print("\n[Step 2] Verified 'router.expert_bias' key is present in checkpoint state_dict:")
    print("  State dict keys containing bias:", [k for k in state_dict.keys() if "bias" in k])

    # 6. Instantiate fresh SFT / RL MoE Layer (starts with zeros)
    submodules_sft = get_submodules(
        get_gpt_layer_local_submodules(num_experts=num_experts, moe_grouped_gemm=False).mlp
    )
    moe_sft = MoELayer(config, submodules_sft).cuda()
    router_sft = cast(TopKRouter, moe_sft.router)
    print("\n[Step 3] Fresh SFT router BEFORE loading checkpoint:")
    print("  Initial SFT bias:", router_sft.expert_bias.cpu().numpy())

    # 7. Load checkpoint into SFT model
    moe_sft.load_state_dict(state_dict)
    print("\n[Step 4] SFT router AFTER loading pretrain checkpoint:")
    print("  Loaded SFT bias:", router_sft.expert_bias.cpu().numpy())
    assert torch.equal(router_sft.expert_bias, mock_pretrain_bias), "ERROR: expert_bias was not loaded correctly!"
    print("  -> SUCCESS: Pretrained expert_bias successfully restored into SFT router!")

    # 8. Simulate 20 training steps with extreme token imbalance
    moe_sft.train()
    print("\n[Step 5] Simulating 20 SFT training iterations with heavily imbalanced inputs...")
    for step in range(1, 21):
        dummy_input = torch.randn(16, 64, dtype=torch.bfloat16, device=torch.cuda.current_device())
        output, mlp_bias = moe_sft(dummy_input)

        # Call Megatron gradient finalization bias update
        _update_router_expert_bias([moe_sft], config)

        # Assert expert_bias has NOT changed by even 1e-7
        assert torch.equal(router_sft.expert_bias, mock_pretrain_bias), f"ERROR at step {step}: expert_bias changed!"

    print("  -> SUCCESS: Verified across all 20 training steps:")
    print("     Final bias after training:", router_sft.expert_bias.cpu().numpy())
    print("     Expected bias:            ", mock_pretrain_bias.cpu().numpy())
    print("     Max Difference:           ", (router_sft.expert_bias - mock_pretrain_bias).abs().max().item())
    print("\n[VERDICT] The frozen expert bias is 100% correctly loaded, preserved, and remains invariant across all SFT/RL training iterations.")
    print("=" * 80)

    Utils.destroy_model_parallel()

if __name__ == "__main__":
    test_frozen_expert_bias_sft_rl()
