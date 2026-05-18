<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-inference-excellence | M4-DENSE-REF-000 | #5640 | `codex/apple-m4-inference-excellence/M4-DENSE-REF-000-reference-receipt-contract` | Define and validate the dense SLM reference-runner vs Rust M4 comparison receipt contract before committing live comparison evidence: artifact kind, supported Qwen identity fields, reference-runner identity and command shape, prompt template and tokenizer authority, Rust generated token IDs, generated text, mechanical scores, summary totals, token-ID availability status, and claim-boundary flags. |
| intel-258v-platform | LNL258V-OPENVINO-QUAL-FIX-001 | #5644 | `codex/lunar-lake/LNL258V-OPENVINO-QUAL-FIX-001-exact-answer-fixture` | Apply the accepted OpenVINO exact-answer policy to the canonical Lunar Lake corpus-v2 fixture by tightening only yes_no_clear_sky to the tested passing one-token budget, leaving stop_token_one_word_done unchanged as a true instruction-miss blocker, validating corpus shape with answer-corpus --dry-run, and preserving no model inference, route promotion, OpenVINO quality claim, speedup, power, acceleration, native OpenCL/NPU, or BitNet QK256/I2_S claims. |
