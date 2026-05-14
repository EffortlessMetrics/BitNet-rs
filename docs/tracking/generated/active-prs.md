<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-052 | #4695 | `cuda/dense-qwen25-short-decode-unblock` | Add an 8-32 token deterministic Qwen2.5 0.5B Q8_0 short-decode strict CUDA receipt with fallback_used=false, stable greedy token sequence, valid UTF-8 answer, no raw special-token garbage, and recorded prefill, KV, logits, sampler, kernel, and transfer evidence. The current-source rerun supersedes the stale-binary diagnostic blocker and records decoded text `The answer is 4. What is` with CPU/CUDA generated-token equality. |
