<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-slm-performance | M4-SLM-PERF-001 | #4044 | `codex/apple-m4-slm-performance/M4-SLM-PERF-001-release-baseline` | Add a release-mode Apple M4 SLM warm-session baseline for warm_16, warm_32, warm_64, and warm_128 profiles, separating cold load from warm prompt timing and recording model/tokenizer/backend/fallback/session fields without making broad performance claims. |
| nvidia-5070ti | CUDA-ANSWER-011 | #4039 | `codex/cuda-answer/CUDA-ANSWER-011-cpu-cuda-parity` | Record same-box 9950X3D AVX-512 CPU and RTX 5070 Ti CUDA answer-corpus receipts for the official Microsoft I2_S artifact, compare them with the generic answer parity tool, and preserve the first divergence evidence without claiming exact CPU/CUDA parity when generated tokens or top-k logits differ. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
