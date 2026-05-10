<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-PERF-005 | #4450 | `codex/cuda-dense-perf-005-h2d-envelope` | Record a measured host-to-device model-load wall-clock envelope in the dense Qwen one-token, short-decode, and warm-session strict CUDA runtime receipts using the verified qwen2.5-0.5b-instruct-q8_0 GGUF artifact on RTX 5070 Ti; preserve measured device-to-host logits download timing, fallback_used=false, speedup_claim=false, benchmark_qualified_speedup=false, full_cuda_residency_claimed=false, server_ready_claimed=false, dense ask/chat claim boundaries, and bitnet_packed_i2s_qk256_proof=false; clearly label the H2D value as a model-load envelope that includes non-transfer overhead rather than pure CUDA event copy timing; add validator coverage rejecting missing H2D envelope fields and timing/accounting mismatches. |
