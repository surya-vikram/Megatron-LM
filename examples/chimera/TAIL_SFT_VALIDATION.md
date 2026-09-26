# TailSFT validation — 26 September 2026

Completed on two H200 GPUs using the 10.04B Chimera pretrained MCore checkpoint,
BF16, Transformer Engine and DP=2. The implementation branch is
`tailsft-implementation`, based on Chimera commit
`a60e083449557e5110068ab9dfece210ec646e1c`.

## Completed DP checks

| Case | Configuration | Result |
| --- | --- | --- |
| Native reference scoring | 24 conversations; packed and unpacked | All 24 full logit tensors and token NLL tensors matched bitwise between layouts |
| Packed ramp | MBS=1, GBS=4, 5 optimizer steps, Muon | Passed selection, gradients, accumulation and evaluation |
| Unpacked ramp | MBS=1, GBS=4, 5 optimizer steps, Muon | Passed selection, gradients, accumulation and evaluation |
| Packed static fraction | MBS=2, GBS=8, 3 steps, fraction=0.5, fused linear CE | Passed selection, gradients, accumulation and evaluation |
| Ordinary SFT / fraction zero | Two steps each, deterministic kernels, Muon | Identical logged losses and gradient norms; all 249 saved model tensors bitwise equal |
| Local tests | TailSFT math/dataset tests plus architecture contract tests | 22 + 21 passed |
| CUDA/NCCL math checks | Variable DP candidate counts, selection and gradients | Passed |

The training runs used sequence length 256 and the checkpoint's 32K YaRN
configuration. These are correctness checks with short, varied conversations,
including multiple assistant turns; they do not measure production context-length
memory or throughput.

## Intermediate values verified

The audit independently checked actual training inputs and labels against physical
JSONL rows, conversation boundaries, assistant masks, target counts, reference
scores, current losses, margins, DP candidate gathering, stable ranking, rejected
rows, retained-token totals, scalar loss, token gradients and the gradients received
during the actual backward pass. All 52 audited rank/microbatch records passed.
Padding segments were excluded. Rejected target tokens had zero gradients.

Captured initial training logits and token NLLs matched the native reference
tensors exactly. The initial margins were zero. Recomputed cross entropy from
captured logits differed by at most `1.91e-6` from training CE. Independent CPU
recomputation of reference means differed by at most `1.17e-5` because CPU and CUDA
float32 reductions differ.

Both ramp runs used `0, 0.125, 0.25, 0.375, 0.5` on iterations 0 through 4.
For example, packed iteration 2 had seven candidates and rejected two at a requested
fraction of 0.25. Their margins were `-1.3994703` and `-1.4543257`, the two lowest;
168 target tokens remained. Its actual rejected fraction was `2/7`.

Counts use Python/torch round-to-even, followed by the retain-at-least-one cap.
Consequently, with only two candidates, fraction 0.25 rejects zero
(`round(0.5) = 0`), while fraction 0.5 rejects one. Packing changes the number of
candidates, which changes this rounding effect.

The gradient finalizer received two normalization units per rank and four globally
for each GBS=4 step. This verifies averaging of the two DP-wide selection objectives
per optimizer step. Parameter-gradient probes were finite, and evaluation used
ordinary SFT loss.

## Corrections made during validation

- The existing fused packed RoPE path omitted YaRN's non-unit scaling. Active
  TailSFT and native scoring now use unfused packed YaRN. Ordinary SFT's settings
  remain unchanged.
- Conversation losses now use independent segment reductions with float64 sums.
  This avoids cancellation from pack-wide float32 prefix sums and keeps initial
  margins exactly tied when references match.
- Native Megatron scoring is the recommended reference path. Matching exported
  weights alone did not establish HF/Megatron forward parity on this checkpoint.
  Native cache metadata checks packing, sequence length, MBS and RoPE fusion.

Flash-kernel ordinary/zero-fraction runs had identical logged losses but small
backward/weight differences. Repeating with deterministic settings established
exact equivalence, including the saved weights; the report does not claim bitwise
repeatability for the nondeterministic kernel configuration.

## Evidence and scope

Local evidence is in `../../../tailsft-validation-20260926/` relative to this
file: audit JSONL records, reference metadata/arrays, verification reports, logs,
source SHA256 values and `dp-validation-evidence.tar.gz`.
Full captured logits, batches and token gradients remain on the instance under
`/home/jovyan/chimera-tailsft-validation/`; they are larger than the local bundle.

TP/PP/EP checks were discontinued when the requested scope became DP only.
Checkpoint resume was skipped at the user's request; its ramp and optimizer-state
continuity are not GPU validated. CP remains restricted to 1. All our GPU processes
were stopped; both GPUs reported 0% utilization and 1 MiB used afterward.
