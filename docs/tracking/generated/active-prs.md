<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-007B | #4138 | `codex/slm-cpu-007b-first-drift-capture` | Capture bitnet-rs Qwen3 checkpoint JSONL plus known-good reference prompt/top-logit evidence for the same Qwen3-0.6B Q8_0 model SHA, prompt/template/BOS policy, prompt IDs, greedy settings, and selected CPU backend as SLM-CPU-006B; validate and classify the first comparable divergence as logits and the first internal drift as reference-missing until reference internal checkpoint dumps exist, without claiming answer quality, throughput, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
