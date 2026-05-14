<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU-BITNET-REF-003 | #4715 | `codex/lunar-lake/CPU-BITNET-REF-003-direct-reference-tokens` | Record direct Microsoft BitNet.cpp generated-token IDs and first-token top-k/logit evidence for the fixed 258V prompts using a patched local llama-server boundary, then rerun the first-token divergence classifier so direct reference token evidence validates against the corrected 258V scalar/AVX2 CPU receipts without claiming broad answer quality, speed, Arc/NPU execution, QK256 changes, or full model correctness. |
