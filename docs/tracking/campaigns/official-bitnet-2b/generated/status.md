<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Official Microsoft BitNet 2B productization Campaign Status

- Campaign: `official-bitnet-2b`
- State: `active`
- Objective: Govern microsoft/BitNet-b1.58-2B-4T as the official BitNet-rs reference model family while keeping I2_S/QK256, TL1, TL2, BF16/GPU-int2, CPU, CUDA, Apple, A770, speed, residency, and server claims route-specific and fallback-explicit.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| OFFICIAL-2B-000 | ready | TBD | `codex/official-bitnet-2b/OFFICIAL-2B-000-source-map` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add the official 2B source map, implementation plan, campaign manifest, campaign overview, specs index entry, and BitNet capability status page without model binaries, runtime changes, or claim promotion. |
| OFFICIAL-2B-001 | blocked | TBD | `codex/official-bitnet-2b/OFFICIAL-2B-001-proposal` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add BITNET-PROP-0014 explaining why the official Microsoft 2B model is the anchor and why speed, full residency, broad server readiness, TL1, TL2, and BF16/GPU-int2 remain separate proof families. |
| OFFICIAL-2B-002 | blocked | TBD | `codex/official-bitnet-2b/OFFICIAL-2B-002-artifact-tokenizer-specs` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add official 2B artifact and tokenizer/prompt contracts with exact-artifact, tokenizer, pre-tokenizer, prompt, stop-policy, and route claim boundaries. |

## Hard Constraints

- Do not commit model binaries.
- Do not weaken the I2_S/QK256 product_cli_ready row.
- Do not promote speedup without exact-profile benchmark review.
- Do not promote full residency without per-phase residency proof.
- Do not promote broad server readiness from exact-profile smoke.
- Do not let TL1/TL2 inherit I2_S/QK256 proof.
- Do not let dense SLM proof satisfy BitNet packed proof.
- Do not let CUDA proof satisfy Apple, A770, CPU, TL1, TL2, or BF16/GPU-int2 proof.
- Do not claim no-scale F32 diagnostic QK256 as production I2_S.
- Keep receipts fallback-explicit and selected-route-explicit.
