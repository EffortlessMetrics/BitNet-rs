<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-dense-slm-regression | M4-SLM-REG-005 | #4198 | `codex/apple-m4-dense-slm-regression/M4-SLM-REG-005-threshold-tightening` | Tighten Apple M4 dense SLM performance thresholds only after multiple matching release-mode receipts exist, with separate bands for timing noise, quality failures, memory drift, and backend/fallback mismatches. |
| nvidia-5070ti | CUDA-DENSE-012 | #4212 | `codex/cuda-dense-012-linear-sweep-command` | Add an aggregate dense GGUF linear role-sweep command and receipt validator that routes multiple verified Qwen2.5 0.5B Q8_0 dense GGUF linear roles through the existing dense FP16 CUDA bridge in one receipt, recording planner counts, per-role kernel stats, aggregate transfer accounting, fallback_used=false, BitNet packed QK256 proof false, dense GGUF inference false, speedup_claim=false, and full_cuda_residency_claimed=false. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
