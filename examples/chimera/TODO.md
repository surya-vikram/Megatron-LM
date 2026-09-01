# Chimera workflow status

- [x] Keep TP, PP, EP, and CP at 1 so the production topology is DP-only.
- [x] Keep optimizer gradients and state in FP32 for Adam, Muon, and AdaMuon internals.
- [x] Derive pretraining and extension iterations from `TRAIN_TOKENS`.
- [x] Derive warmup, decay, save, and validation intervals from configurable fractions.
- [x] Support both cosine and WSD schedules without editing launch scripts.
- [x] Support direct 8K-to-128K YaRN context extension with source-phase validation.
- [x] Support separate packed training and validation data for SFT and SimPO.
- [x] Provide four stage env templates for `cluster_manager.sh`.
- [x] Complete local static, schedule, manager dry-run, and focused dataset tests.
- [x] Review the final diff; the verified workflow is ready to push.
