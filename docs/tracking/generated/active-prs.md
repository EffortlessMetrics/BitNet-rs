<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-productization | M4-PROD-005 | #4034 | `codex/apple-m4-productization/M4-PROD-005-metal-phase` | Implement the first Apple Metal prefill linear projection microphase only with CPU-only versus CPU-plus-Metal greedy parity, Metal phase fallback_used=false, the rest of the pipeline recorded as CPU/NEON, layout handling recorded, and no full Metal inference claim. |
| intel-258v-platform | CPU258V-013 | #4036 | `codex/intel-258v-platform/CPU258V-013-warm-phase-artifacts` | Record release-built 258V warm-session strict CPU phase receipts for prefill_512 and decode_128 after the BitNet b1.58 mechanics correction, preserving real GGUF loading, explicit tokenizer resolution, selected i2_s-avx2-reference kernel, fallback=false, and phase timing claim boundaries. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
