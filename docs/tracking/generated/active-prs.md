<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | ARC140V-005 | #4103 | `codex/intel-258v-platform/ARC140V-005-opencl-cpu-parity` | Run one isolated Arc 140V native OpenCL kernel or static subgraph and compare its output against the 258V CPU reference, recording requested/selected backend identity, runtime API, device identity, input shape, tolerance, timing, fallback=false, and no BitNet/QK256/acceleration claims. |
| nvidia-5070ti | CUDA-DENSE-002 | #4106 | `codex/cuda-dense-002-gemm-parity` | Add the first dense CUDA FP16/BF16 or cuBLAS-backed GEMM smoke/parity fixture with dense_regular_llm_cuda receipts, fallback_used=false, CPU reference comparison, and no BitNet packed I2S/QK256 or speedup claim. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
