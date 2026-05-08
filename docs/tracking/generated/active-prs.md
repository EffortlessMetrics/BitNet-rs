<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-PLANNER-002 | #4129 | `codex/cuda-planner-002-capability-spec` | Add conservative model-family and quantization metadata label mapping into the model-aware CUDA dispatch spec so BitNet I2_S/QK256 and dense regular-LLM FP16/BF16 routes can be selected without treating unknown or mismatched metadata as CUDA-supported. |
| slm-cpu | SLM-CPU-007 | #4126 | `codex/slm-cpu-007-logits-root-cause` | Localize the Qwen3-0.6B Q8_0 first-token logits divergence by capturing bounded bitnet-rs checkpoint summaries for the same model SHA, prompt/template/BOS policy, prompt IDs, greedy settings, and selected CPU backend as SLM-CPU-006B; compare against reference checkpoint evidence where available; identify the first layer or operation where drift appears without claiming answer quality, throughput, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
