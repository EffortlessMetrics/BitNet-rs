<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-PERF-004 | #4445 | `codex/cuda-dense-perf-004-qualification-review` | Add a governed dense Qwen benchmark qualification review after CUDA-DENSE-PERF-001/002/003, consuming the baseline, repeated CPU/CUDA comparator, and D2H transfer-timing receipts; explicitly keep speedup_claim=false and benchmark_qualified_speedup=false because reviewed CUDA mean total times are slower than CPU means and host-to-device timing remains unmeasured; preserve fallback_used=false, full_cuda_residency_claimed=false, server_ready_claimed=false, dense ask/chat claim boundaries, and bitnet_packed_i2s_qk256_proof=false; add validator coverage rejecting speedup/profile acceptance and missing transfer timing source fields. |
