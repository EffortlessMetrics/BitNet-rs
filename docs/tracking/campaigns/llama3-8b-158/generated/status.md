<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Llama3 8B 1.58 supported-model candidate Campaign Status

- Campaign: `llama3-8b-158`
- State: `active`
- Objective: Make HF1BitLLM/Llama3-8B-1.58-100B-tokens a first-class large BitNet-family candidate lane without inheriting Microsoft 2B, dense Llama3, backend, server, or speed proof.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| LLAMA3-158-000 | ready | TBD | `codex/llama3-158-source-map-specs` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add Llama3 8B 1.58 proposal, source map, implementation plan, active campaign, conservative model/kernel/Apple matrices, and spec contracts without runtime changes, model binaries, or support promotion beyond registered. |
| LLAMA3-158-001 | proposed | TBD | `codex/llama3-158-artifact-inventory` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Record exact artifact inventory receipt or blocked receipt with revision, file list, sizes, SHA256 values, tokenizer/config hashes, identity discrepancy, storage context, cleanup status, and no answer/backend claims. |
| LLAMA3-158-002 | proposed | TBD | `codex/llama3-158-conversion-runner-authority` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Verify or block I2_S/TL1/TL2 conversion and reference runner paths with exact commands, tool commits, input hashes, output hashes when produced, runner command, route, and diagnostic-only claim boundary. |

## Hard Constraints

- Do not commit model binaries.
- Do not claim answer readiness from safetensors inventory alone.
- Do not inherit official Microsoft 2B I2_S/QK256 proof.
- Do not inherit dense Llama3 or dense Qwen proof.
- Do not claim I2_S/QK256 route compatibility before layout proof.
- Do not route TL1/TL2 through QK256/I2_S kernels.
- Do not claim CPU, CUDA, Apple, server, or speed readiness before exact receipts.
