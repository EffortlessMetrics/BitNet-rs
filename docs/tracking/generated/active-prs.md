<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m3-macbook-air | M3MBA-003 | #4596 | `codex/apple-m3-macbook-air/M3MBA-003-receipt-label` | Add or confirm the smallest receipt validation path for apple-m3-air-cpu-neon, preserving existing apple-m4-cpu-neon checks and making MacBook timing impossible to label as M4 evidence. |
| intel-258v-platform | CPU-BITNET-REF-001 | #4599 | `codex/lunar-lake/CPU-BITNET-REF-001-reference-boundary` | Record a narrow 258V BitNet CPU external reference boundary that links the corrected CPU reference bundle, external BitNet.cpp generated-text reference, and instrumentation classifier, preserving missing generated-token/logit fields as blockers without claiming generated-token parity, first-token logits parity, answer quality, speed, Arc, NPU, QK256 changes, or full model correctness. |
| nvidia-5070ti | CUDA-PROD-009 | #4604 | `cuda/bitnet-strict-preflight` | Harden strict BitNet CUDA user preflight with `bitnet cuda doctor`, fail-closed strict backend/tokenizer checks before generation, default strict receipt path visibility, and preserved speedup_claim=false. |
