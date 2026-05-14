<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-008Y | #4696 | `codex/slm-cpu-008y-blocked` | Capture or ingest a real known-good internal Qwen3 checkpoint pack for the same Qwen3-0.6B Q8_0 model SHA, rendered prompt, prompt IDs, and greedy first-token settings used by the i5-8250U artifact, then validate it with the SLM checkpoint-aware reference-compare path from SLM-CPU-008X. The artifact must identify the first divergent checkpoint before lm_head.top_logits when compared with bitnet-rs. This item must not claim first-token parity, answer quality, tiny corpus success, multi-token stability, warm-session performance, Q4/Q5 expansion, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
