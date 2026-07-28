# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import (
    accumulate_router_balance_metrics,
    clear_router_balance_metrics,
    get_router_balance_metrics,
)
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(autouse=True)
def clear_tracker():
    clear_router_balance_metrics()
    yield
    clear_router_balance_metrics()


def test_router_balance_metrics_accumulate_over_interval():
    accumulate_router_balance_metrics(
        torch.tensor([[4.0, 2.0, 0.0, 2.0], [1.0, 1.0, 1.0, 1.0]]),
        torch.tensor([[0.1, -0.2, 0.0, 0.1], [0.3, -0.1, 0.0, 0.1]]),
    )
    accumulate_router_balance_metrics(
        torch.tensor([[0.0, 2.0, 2.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
        torch.tensor([[0.2, -0.1, 0.0, 0.1], [0.4, -0.1, 0.0, 0.1]]),
    )

    metrics = get_router_balance_metrics()

    assert metrics is not None
    assert metrics["load_cv"] == pytest.approx(1.0 / 6.0)
    assert metrics["worst_load_over_mean"] == pytest.approx(4.0 / 3.0)
    assert metrics["dead_expert_slots"] == 0
    assert metrics["expert_slots"] == 8
    assert metrics["bias_max_abs"] == pytest.approx(0.4)


def test_router_balance_metrics_reset_only_clears_interval_counts():
    counts = torch.tensor([[4.0, 0.0]])
    bias = torch.tensor([[0.25, -0.5]])
    accumulate_router_balance_metrics(counts, bias)

    first = get_router_balance_metrics(reset=True)
    assert first is not None
    assert first["dead_expert_slots"] == 1

    accumulate_router_balance_metrics(torch.tensor([[1.0, 1.0]]), bias)
    second = get_router_balance_metrics()
    assert second is not None
    assert second["load_cv"] == 0.0
    assert second["dead_expert_slots"] == 0


def test_router_balance_logging_interval_requires_expert_bias():
    with pytest.raises(ValueError, match="requires moe_router_enable_expert_bias"):
        TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            moe_router_balance_logging_interval=100,
        )


def test_router_balance_logging_interval_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            moe_router_enable_expert_bias=True,
            moe_router_score_function="sigmoid",
            moe_router_balance_logging_interval=0,
        )
