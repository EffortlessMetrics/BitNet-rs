<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Intel 258V platform validation Campaign Status

- Campaign: `intel-258v-platform`
- State: `active`
- Objective: Validate Core Ultra 7 258V as a tri-device platform while keeping CPU AVX2, Arc 140V GPU, and Intel AI Boost NPU proof labels separate.

## Work Items

| Item | State | PR | Branch | Acceptance |
|---|---|---:|---|---|
| LNL258V-002 | ready | TBD | `codex/intel-258v-platform/LNL258V-002-probe-bundle` | Document and then add a 258V platform probe bundle that records CPU, Arc 140V GPU, Intel NPU, memory, OS, driver, OpenVINO, OpenCL, Level Zero, WSL, power context, artifact paths, and receipt fields without runtime claims. |

## Hard Constraints

- Arc 140V OpenCL proof is not NPU proof.
- OpenVINO GPU smoke is not packed BitNet kernel proof.
- WSL only counts for NPU validation if OpenVINO reports NPU inside WSL.
- The 258V CPU validation harness records evidence only and does not take over shared CPU implementation ownership.
