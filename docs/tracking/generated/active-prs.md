<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | SLM258V-005 | #4552 | `codex/lunar-lake/SLM258V-005-receipt-identity` | Harden dense Qwen SLM answer and phase receipts so top-level requested_backend, selected_backend, runtime_api, fallback_used, backend_lane, model_family, model_architecture, quantization, prompt_template, tokenizer_source, and selected_kernel_or_runtime are explicit, while preserving existing child/case backend fields and avoiding new quality, speed, Arc/NPU, acceleration, or BitNet QK256/I2_S claims. |
