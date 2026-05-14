<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU-BITNET-REF-002 | #4710 | `codex/lunar-lake/CPU-BITNET-REF-002-direct-token-classifier` | Harden the first-token divergence classifier so direct external reference generated_token_ids are treated as authoritative token evidence, a derived first-generated-token field is recorded, and all matching direct first-token cases classify as no divergence without claiming generated-token or logits parity for the current text-only BitNet.cpp evidence. |
