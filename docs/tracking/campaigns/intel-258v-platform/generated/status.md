<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Intel 258V platform validation Campaign Status

- Campaign: `intel-258v-platform`
- State: `active`
- Objective: Validate Core Ultra 7 258V as a tri-device platform while keeping CPU AVX2, Arc 140V GPU, and Intel AI Boost NPU proof labels separate.

## Work Items

| Item | State | PR | Branch | Acceptance |
|---|---|---:|---|---|
| LNL258V-RUN-001 | merged | #3714 | `codex/intel-258v/LNL258V-RUN-001-platform-probe` | Add a JSON-ready Lunar Lake 258V platform probe that records CPU AVX2 facts, Arc 140V OpenCL/Level Zero/OpenVINO GPU visibility, Intel NPU OS/OpenVINO visibility, memory, power, OS, proof_stage=runtime_detected, and fallback_used=false without inference claims. |
| ARC140V-002 | pr_open | #3727 | `codex/intel-arc/ARC140V-002-runtime-probe` | Probe exact Arc 140V runtime visibility by name or PCI ID 0x64A0 across OpenCL, Level Zero, and OpenVINO GPU.0 while recording proof_stage=runtime_detected, requested/selected backend identity, runtime API, and fallback_used=false. |
| LNL258V-002 | ready | TBD | `codex/intel-258v-platform/LNL258V-002-probe-bundle` | Add a 258V platform probe bundle that records CPU, Arc 140V GPU, Intel NPU, memory, OS, driver, OpenVINO, OpenCL, Level Zero, WSL, and power context without runtime claims. |

## Hard Constraints

- Arc 140V OpenCL proof is not NPU proof.
- OpenVINO GPU smoke is not packed BitNet kernel proof.
- WSL only counts for NPU validation if OpenVINO reports NPU inside WSL.
