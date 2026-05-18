<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Falcon-E Family compact 1.58-bit lane Campaign Status

- Campaign: `falcon-e-family`
- State: `active`
- Objective: Register Falcon-E Family as BitNet-rs's compact direct-GGUF 1.58-bit validation lane while preserving artifact, tokenizer, prompt, route, backend, and performance claim boundaries.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| FE-000 | ready | TBD | `codex/falcon-e-family/FE-000-source-map-and-specs` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add Falcon-E source-of-truth docs, implementation plan, active campaign, specs, and registered-only candidate matrix rows without runtime/kernel changes or support promotion. |
| FE-001 | proposed | TBD | `codex/falcon-e-family/FE-001-artifact-inventory` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Record exact source revision, file list, size, SHA256, license, GGUF metadata, tokenizer metadata if embedded, and cleanup status for Falcon-E 1B and 3B I2_S GGUF artifacts without answer/backend/speed claims. |

## Hard Constraints

- Do not commit model binaries.
- Do not claim Falcon-E 1B proof proves Falcon-E 3B.
- Do not claim Falcon-E proof inherits Microsoft BitNet 2B, Falcon3, 1bitLLM, or dense SLM proof.
- Do not claim I2_S/QK256 compatibility before Falcon-E layout proof.
- Do not route TL1/TL2 through QK256/I2_S kernels.
- Do not claim CPU, CUDA, Apple, A770, speed, server, or full-residency readiness without exact receipts.
