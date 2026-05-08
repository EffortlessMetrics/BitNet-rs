<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU258V-006 | #4001 | `codex/intel-258v-platform/CPU258V-006-warm-phase` | Add a strict CPU warm phase runner that loads the 258V BitNet GGUF model/tokenizer once, emits per-profile strict CPU receipts for prefill_512 and decode_128, preserves selected backend/kernel and fallback=false, and keeps phase evidence separate from speedup, Arc, or NPU claims. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
