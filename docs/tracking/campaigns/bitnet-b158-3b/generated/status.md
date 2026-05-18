<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# BitNet b1.58 3B TL candidate Campaign Status

- Campaign: `bitnet-b158-3b`
- State: `active`
- Objective: Make 1bitLLM/bitnet_b1_58-3B a first-class BitNet-rs TL-model candidate with guarded artifact, conversion, runner, TL layout, tokenizer, quality, backend, and performance authority before any answer or speed claim.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| B158-3B-001 | ready | TBD | `codex/bitnet-b158-3b/docs-rails-and-specs` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add the 3B TL model proposal, source map, campaign, and implementation plan.; Add artifact, conversion, TL layout, tokenizer/prompt, reference-quality, CPU, CUDA, Apple, and performance specs.; Update specs index and Apple candidate matrices without promoting model coverage or runtime claims.; Keep 3B I2_S/QK256 unsupported and TL1/TL2 verification pending. |

## Hard Constraints

- Do not commit model binaries.
- Do not claim 3B I2_S or QK256 support except diagnostic rejection receipts.
- Do not claim x86 TL2 or ARM TL1 answer readiness until runner/conversion and reference-quality evidence pass.
- Do not claim CPU, CUDA, Apple, server, or speed readiness before route-specific receipts with fallback=false exist.
