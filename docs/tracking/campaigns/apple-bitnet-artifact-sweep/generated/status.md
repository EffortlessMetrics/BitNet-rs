<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple BitNet artifact sweep Campaign Status

- Campaign: `apple-bitnet-artifact-sweep`
- State: `active`
- Objective: Use the MacBook Apple Silicon lane to qualify 1-bit / 1.58-bit BitNet-family artifacts before any M4 Mac mini Apple CPU/NEON or Metal BitNet local-answer claim.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| ABAS-001 | proposed | TBD | `codex/apple-bitnet-artifact-sweep/ABAS-001-microsoft-2b-i2s` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Validate the official Microsoft BitNet b1.58 2B / 2B4T I2_S GGUF on MacBook under external Microsoft tokenizer pre-tokenizer authority, recording source, revision, SHA256, size, tokenizer authority, reference-runner prompt outputs, bad/no-authority rejection evidence, and cleanup status. |
| ABAS-002 | proposed | TBD | `codex/apple-bitnet-artifact-sweep/ABAS-002-1bitllm-07b` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Evaluate 1bitLLM/bitnet_b1_58-large as the smaller Apple BitNet control candidate under BITNET-PROP-0009 and B158-large artifact/conversion/tokenizer/reference contracts, recording exact file, size, SHA256, I2_S/TL1 route evidence, tokenizer authority, coherent reference output or rejection evidence, and cleanup status without promoting CPU, CUDA, Apple, server, or speed claims. |
| ABAS-003 | proposed | TBD | `codex/apple-bitnet-artifact-sweep/ABAS-003-3b-tl-diagnostic` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Evaluate 1bitLLM/bitnet_b1_58-3B only on supported TL1/TL2 diagnostic routes, recording why I2_S remains unsupported, tokenizer authority, reference-runner outcome, and cleanup status. |
| ABAS-004 | proposed | TBD | `codex/apple-bitnet-artifact-sweep/ABAS-004-falcon-e-secondary` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Evaluate Falcon-E 1B/3B GGUFs as secondary BitNet-like family candidates only after Microsoft and 1bitLLM behavior is understood, recording runner path, tokenizer authority, output sanity, license/source notes, and cleanup status. |
| ABAS-005 | proposed | TBD | `codex/apple-bitnet-artifact-sweep/ABAS-005-m4-proof-handoff` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Promote the best accepted Apple BitNet artifact into an M4 Mac mini strict Apple CPU/NEON local-answer proof plan, preserving source, hash, tokenizer authority, kernel route, and claim boundary without running the proof in this handoff item. |

## Hard Constraints

- Use MacBook first for larger artifact sweeps; do not manufacture MacBook receipts from the M4 Mac mini.
- Do not claim Rust Apple BitNet local answers before the target backend runs its own strict receipt gate.
- Do not claim BitNet quality from dense Qwen SLM evidence.
- Do not claim QK256 support, full Apple Metal inference, Neural Engine execution, MPSGraph model inference, or broad Apple Silicon performance.
- Do not weaken the shared answer-artifact gate or model/kernel compatibility ledger.
- Never commit model binaries.
