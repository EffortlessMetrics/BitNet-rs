<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4 | M4-008 | #3721 | `codex/apple-m4/M4-008-backend-receipts` | Record Apple requested backend, selected backend, runtime API, resolved device identity, fallback status, and artifact path in receipts. |
| cpu-proof | CPU-BITNET-004 | #3696 | `codex/cpu-bitnet-004-scalar-packed-truth` | Canonical scalar packed QK256 GEMV/GEMM kernels are deterministic correctness oracles for decode and prefill. |
| intel-258v-platform | LNL258V-RUN-001 | #3714 | `codex/intel-258v/LNL258V-RUN-001-platform-probe` | Add a JSON-ready Lunar Lake 258V platform probe that records CPU AVX2 facts, Arc 140V OpenCL/Level Zero/OpenVINO GPU visibility, Intel NPU OS/OpenVINO visibility, memory, power, OS, proof_stage=runtime_detected, and fallback_used=false without inference claims. |
| intel-npu | NPU-002 | #3722 | `codex/intel-npu/NPU-002-lite-backend-identity` | Preserve Intel NPU requested and selected backend identity without mapping it to Metal, CUDA, generic GPU, or CPU fallback. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
