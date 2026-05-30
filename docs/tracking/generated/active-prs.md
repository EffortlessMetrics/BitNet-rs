<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-REG-014 | #6245 | `codex/lunar-lake/LNL258V-REG-014-operator-ask-hash-coverage` | Make strict Lunar Lake regression fail closed when successful auto operator-ask receipts lose model hash identity coverage. The regression surface must record model_hash_or_explicit_gap and model_hash_coverage for ask_short, ask_normal, and warm_resident auto ask receipts using top-level model.hash_coverage when available, legacy single model.sha256 when present, or embedded OpenVINO IR per-file SHA256 coverage from source_receipt.model.files. Refresh regression-v2 and operator-comparison artifacts without running inference or changing route promotion, speedup, power-advantage, measured-temperature, native accelerator, broad quality, Qwen3, or BitNet QK256/I2_S behavior claims. |
