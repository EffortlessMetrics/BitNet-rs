<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU258V-018 | #4178 | `codex/lunar-lake/CPU258V-018-hf-prompt-token-parity` | Compare official HF AutoTokenizer.apply_chat_template rendered prompts and token IDs against BitNet-rs metadata-authoritative prompt-authority-audit output for math_2_plus_2, say_ok, capital_france, and yes_no_water, recording mismatch indices and preserving no inference, logits, answer-quality, speed, Arc/NPU, or QK256 claims. |
| nvidia-5070ti | CUDA-UX-002 | #4191 | `codex/cuda-ux-002-ask-proof-summary` | Reuse the `bitnet receipts explain` normalization layer for strict `bitnet ask` proof summaries so the live ask path prints a compact proof covering model, planner route, backend, runtime, kernels, fallback, quality, upload-once weight status, measured kernel/transfer fields, full-residency claim, speed claim, and receipt path without changing generation, tokenizer, loader, transformer, kernel, benchmark, dense GGUF inference, or server behavior. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
