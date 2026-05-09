<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-silicon-macbook | MB-AS-003 | #4190 | `codex/apple-silicon-macbook/MB-AS-003-bitnet-candidate-matrix` | Add a MacBook-oriented Apple BitNet candidate matrix covering official Microsoft 2B I2_S, 1bitLLM 0.7B, 1bitLLM 3B TL1/TL2 diagnostic routes, and Falcon-E candidates with storage estimates, supported kernel routes, tokenizer authority requirements, reference-runner commands, and cleanup rules. |
| intel-258v-platform | CPU258V-018 | #4178 | `codex/lunar-lake/CPU258V-018-hf-prompt-token-parity` | Compare official HF AutoTokenizer.apply_chat_template rendered prompts and token IDs against BitNet-rs metadata-authoritative prompt-authority-audit output for math_2_plus_2, say_ok, capital_france, and yes_no_water, recording mismatch indices and preserving no inference, logits, answer-quality, speed, Arc/NPU, or QK256 claims. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
