<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-npu | NPU-010 | #4080 | `codex/intel-npu/NPU-010-live-openvino-receipts` | Record live 258V OpenVINO 2026.1 Intel NPU runtime visibility, tiny static graph smoke, and selected BitNet RMSNorm and linear-projection static subgraph parity receipts with selected_backend=intel-npu-openvino, runtime_api=openvino, runtime_device=NPU, fallback_used=false, and no full BitNet inference, acceleration, QK256 decode, or CPU fallback proof claims. |
| slm-cpu | SLM-CPU-006 | #4071 | `codex/slm-cpu-006-first-token-artifact` | Capture a real first-token divergence artifact for Qwen3-0.6B Q8_0 on the i5-8250U by running bitnet-rs and a known-good external reference with identical model SHA, prompt text, Qwen template, BOS policy, prompt IDs, generated IDs, decoded text, chosen token, and first-step top-k/logit evidence where available. The artifact classifies the first divergence as prompt/tokenizer/template, logits/sampler, output-head/vocab indexing, or shared transformer math without claiming answer quality, throughput, server, GPU, NPU, or Qwen3.5 support. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
