<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| cpu-proof | CPU-ANSWER-006 | #4005 | `codex/cpu-answer-006-reference-token-artifact` | A Microsoft BitNet.cpp reference-divergence artifact records the MODEL-ARTIFACT-007 prompt envelope, BOS policy, external Llama-BPE tokenizer/pre-tokenizer authority, prompt token IDs, generated token IDs, decoded text, and first-step top-k/logit evidence where available, so strict Rust CPU failures can be classified as prompt/tokenizer divergence, shared decode/logits divergence, or backend-specific execution without claiming answer quality. |
| cpu-proof | CPU-ANSWER-007 | #4019 | `codex/cpu-answer-007-bitnet-subnorms` | Strict Rust CPU answer-corpus runs pass the MODEL-ARTIFACT-007 official Microsoft I2_S answer-ready artifact by aligning BitNet B1.58 model math with BitNet.cpp: RMSNorm/ReLU^2 defaults, attention and FFN sub-layernorms, BitNet.cpp I2_S QK256 layout, inline scale and I8_S GEMV semantics, and GGML token-major F16 token embeddings. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
