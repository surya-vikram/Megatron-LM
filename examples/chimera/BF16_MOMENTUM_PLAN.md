# Technical Proposal & Implementation Plan: BF16 Momentum ($M_t$) for Muon Optimizer

## 📌 Executive Summary
This document presents a detailed, production-grade implementation and verification plan to convert Muon's 1st-moment momentum state ($M_t$) from **FP32 (32-bit)** to **BF16 (16-bit)** within Megatron-LM (`TensorParallelMuon` / `TensorParallelAdaptiveMuon`).

### 🎯 Primary Goal
- **Reduce Base & Max Peak VRAM Usage**: Save **$2N$ bytes per 2D parameter** across all attention projection and MoE expert weights (**-50% momentum memory**), freeing ~20 GB VRAM at 10B/440B scale (~130 MB on Tiny Chimera).
- **Maintain 100% Convergence & Numerical Stability**: Use a **Mixed-Precision Hybrid Wrapper**—store $M_t$ in BF16 for persistent storage, but upcast to FP32/TF32 internally during the Newton-Schulz matrix multiplication step.

---

## 🏗️ Architectural Changes

### 1. File to Modify
- [`/home/surya/workspace/repos/Megatron-LM/megatron/core/optimizer/emerging_optimizers.py`](file:///home/surya/workspace/repos/Megatron-LM/megatron/core/optimizer/emerging_optimizers.py)

### 2. State Initialization (`_init_group`)
Override state allocation in `OrthogonalizedOptimizer` so that 2D weight momentum buffers are explicitly allocated in `torch.bfloat16` instead of `torch.float32`:

```python
# Standard FP32 state allocation (Current)
state['momentum_buffer'] = torch.zeros_like(p, dtype=torch.float32)

# BF16 state allocation (Proposed)
state['momentum_buffer'] = torch.zeros_like(p, dtype=torch.bfloat16)
```

### 3. Mixed-Precision Newton-Schulz Wrapper (`scaled_orthogonalize_fn`)
In `TensorParallelMuon`, modify `scaled_orthogonalize_fn` to handle BF16 inputs smoothly by performing internal FP32 upcasting during Newton-Schulz matrix multiplications:

```python
def scaled_orthogonalize_fn(
    grad: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup,
    partition_dim: int | None = None,
) -> torch.Tensor:
    # 1. Upcast BF16 momentum input to FP32 for Newton-Schulz GEMMs
    grad_fp32 = grad.to(torch.float32)
    
    # 2. Execute Newton-Schulz 6-step matrix orthogonalization in FP32/TF32
    orth_grad = newton_schulz_tp(
        grad_fp32,
        steps=num_ns_steps,
        coefficient_type=coefficient_type,
        tp_group=tp_group,
        partition_dim=partition_dim,
        tp_mode="duplicated" if tp_mode == "blockwise" else tp_mode,
    )
    scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
    
    # 3. Downcast orthogonalized result back to BF16 for parameter update
    return (orth_grad * scale_factor * extra_scale_factor).to(grad.dtype)
```

---

## 🔬 Rigorous Verification & Stress Test Plan

To ensure this change is 100% safe before deploying to production pretraining, we will execute a 4-stage stress test matrix:

### Stage 1: Exact Bitwise Memory Verification
- **Test**: Run PyTorch memory profiler (`torch.cuda.memory_allocated()`, `torch.cuda.max_memory_allocated()`).
- **Pass Criterion**: Base persistent VRAM drops by exactly $2N$ bytes per 2D weight parameter.

### Stage 2: Orthogonality & Singular Value Equivalence Test
- **Test**: Compare orthogonalization output $\|O_{\text{BF16}} - O_{\text{FP32}}\|_F$ and singular value spectrum ($\sigma_{\max}/\sigma_{\min}$) across 1,000 synthetic 2D matrices (well-conditioned & ill-conditioned, $\kappa \in [1, 10000]$).
- **Pass Criterion**: $\|O_{\text{BF16}} - O_{\text{FP32}}\|_F / \|O_{\text{FP32}}\|_F < 10^{-4}$.

### Stage 3: 1,000-Step FineWeb-Edu Convergence Comparison (Clean Data)
- **Test**: Run full 1,000-step pretraining on FineWeb-Edu dataset with BF16 momentum vs FP32 momentum baseline.
- **Pass Criterion**: Final LM loss delta $|\text{Loss}_{\text{BF16}} - \text{Loss}_{\text{FP32}}| < 0.005$. No NaN iterations or loss spikes.

### Stage 4: 1,000-Step Noisy Data Stress Test (10% Token Corruption + 50 Garbage Blocks)
- **Test**: Run 1,000-step pretraining under noisy FineWeb-Edu dataset (`fineweb_edu_noisy_text_document`).
- **Pass Criterion**: Max gradient norm spike $< 15.0$, zero NaN iterations, clean loss recovery after noise blocks.

---

## 🚦 Risk Assessment & Mitigations

| Risk | Likelihood | Impact | Mitigation Strategy |
|---|---|---|---|
| **Underflow/Overflow in Momentum** | Low | High | BF16 shares the exact same 8-bit dynamic exponent as FP32 ($10^{-38}$ to $10^{38}$), eliminating underflow/overflow risk. |
| **Newton-Schulz Matrix Multiplication Precision Loss** | Low | High | We upcast $M_t \rightarrow \text{FP32}$ inside `scaled_orthogonalize_fn` so GEMM Tensor Cores compute in FP32/TF32. |
| **Checkpoint Backward Compatibility** | Medium | Medium | Include dtype casting logic in state loading (`torch_dist` checkpoint load function) so existing FP32 checkpoints load seamlessly into BF16 state buffers. |

---

## 🙋 Reviewer Decision Required
- **Option A**: Proceed with writing BF16 Momentum implementation and launching Stage 1–4 Stress Test Matrix.
- **Option B**: Modify plan constraints before implementation.
