<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-PERF-006 | #4452 | `codex/cuda-dense-perf-006-h2d-envelope-qualification` | Refresh dense Qwen benchmark qualification tooling to consume the CUDA-DENSE-PERF-005 H2D model-load envelope receipts for one-token, short-decode, and warm-session profiles; record the envelope timing, source, scope, and non-transfer-overhead flag in profile reviews and evidence summaries; keep pure host-to-device CUDA event copy timing blocked, keep speedup_claim=false and benchmark_qualified_speedup=false, and preserve full-residency/server/BitNet-proof claim boundaries. |
