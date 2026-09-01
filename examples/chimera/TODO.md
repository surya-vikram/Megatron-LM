# Chimera workflow status

- [x] Default TP, PP, EP, and CP to 1 while allowing valid operator-selected topology.
- [x] Default optimizer gradients and state to FP32 while allowing Adam precision overrides.
- [x] Derive pretraining and extension iterations from `TRAIN_TOKENS`.
- [x] Derive warmup, decay, save, and validation intervals from configurable fractions.
- [x] Support both cosine and WSD schedules without editing launch scripts.
- [x] Support direct 8K-to-128K YaRN context extension with source-phase validation.
- [x] Support separate packed training and validation data for SFT and SimPO.
- [x] Provide four stage env templates for `cluster_manager.sh`.
- [x] Complete local static, schedule, manager dry-run, and focused dataset tests.
- [x] Review the final diff; the verified workflow is ready to push.
