<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-PERF-003 | #4443 | `codex/cuda-dense-perf-003-transfer-timing` | Record measured device-to-host logits download timing in the dense Qwen one-token, short-decode, and warm-session strict CUDA runtime receipts using the verified qwen2.5-0.5b-instruct-q8_0 GGUF artifact on RTX 5070 Ti; keep host-to-device timing explicitly unmeasured with source fields, preserve fallback_used=false, speedup_claim=false, benchmark_qualified_speedup=false, full_cuda_residency_claimed=false, server_ready_claimed=false, dense ask/chat claim boundaries, and bitnet_packed_i2s_qk256_proof=false; add validator coverage rejecting missing transfer timing source fields. |
