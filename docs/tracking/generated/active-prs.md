<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-022 | #5140 | `model/slm-cpu-022-smollm2-reference-comparator` | Define the SmolLM2 360M reference-compatible first-token/top-k comparator contract after the SLM-CPU-021 wrong-first-token diagnosis, and add CLI fixture coverage showing `reference-compare` validates the exact SmolLM2 model-family/top-k artifact shape and fails closed under `--require-match` when it diverges. This support slice must not claim a fresh external reference run, CPU answer readiness, CUDA planning, CUDA execution, speedup, server readiness, broad dense GGUF support, or BitNet QK256 behavior. |
