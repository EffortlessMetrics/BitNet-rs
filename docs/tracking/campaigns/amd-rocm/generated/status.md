<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# AMD ROCm productization Campaign Status

- Campaign: `amd-rocm`
- State: `active`
- Objective: Register and sequence AMD ROCm as BitNet-rs's selected-device HIP/ROCm lane for exact AMD Radeon, Radeon PRO, and Instinct routes without inheriting CUDA, OpenCL, WGPU, CPU, OpenVINO, BitNet QK256, dense SLM, speed, residency, or server proof.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| ROCM-DOCS-000 | ready | TBD | `codex/amd-rocm/ROCM-DOCS-000-source-of-truth` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add AMD ROCm source-of-truth docs, proposal, specs, plan, campaign tracker, specs index entry, and hardware matrix row as docs-only registered/scaffold evidence with no runtime, model, speed, residency, or server promotion. |

## Hard Constraints

- Do not claim generic AMD GPU support.
- Do not claim ROCm from source text tests.
- Do not claim HIP compile from source embedding.
- Do not claim execution from compile smoke.
- Do not claim BitNet QK256 from dense SLM ROCm proof.
- Do not claim dense SLM from BitNet QK256 proof.
- Do not claim speedup without exact-profile review.
- Do not claim full residency without per-phase residency evidence.
- Do not add live ROCm hardware execution to ordinary PR CI.
