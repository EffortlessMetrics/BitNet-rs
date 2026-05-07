# Intel 258V Platform Campaign

Campaign ID: `intel-258v-platform`

Status: active

## Objective

Validate Core Ultra 7 258V as the BitNet CPU lead and tri-device platform while keeping CPU AVX2, Arc 140V GPU, and Intel AI Boost NPU proof labels separate.

## End State

- Same-machine CPU, GPU, and NPU facts are captured.
- 258V CPU strict real-GGUF validation, scalar/AVX2 answer parity, and phase receipts provide the CPU reference plate.
- Arc 140V OpenCL, OpenVINO GPU, and OpenVINO NPU evidence are not conflated.
- Receipts record OS, drivers, memory, power, thermal, and WSL/native visibility context.

## Hard Constraints

- 258V CPU proof is first priority; NPU and Arc proofs must compare against the 258V CPU reference before BitNet-adjacent parity claims.
- Arc 140V OpenCL proof is not NPU proof.
- OpenVINO GPU smoke is not packed BitNet kernel proof.
- WSL only counts for NPU validation if OpenVINO reports NPU inside WSL.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| LNL258V-RUN-001 | merged | Add JSON-ready Lunar Lake platform probe structs. |
| ARC140V-002 | merged | Add exact Arc 140V runtime identity probe logic. |
| LNL258V-002 | merged | Add 258V probe bundle and same-machine comparison hooks. |
| LNL258V-003 | merged | Add CLI platform probe emission for the current 258V machine. |
| CPU258V-001 | merged | Add a validation-only CPU BitNet preflight harness for the 258V lane. |
| LNL258V-OWNERSHIP-001 | merged | Made the 258V CPU the BitNet CPU lead and set priority order: CPU, NPU, Arc 140V; merged in #3914. |
| CPU258V-002 | merged | Add scalar-vs-AVX2 strict CPU answer parity on the 258V; merged in #3929. |
| CPU258V-003 | ready | Add 258V CPU phase benchmark receipts for the CPU reference plate. |

## Review Policy

Platform PRs document and compare lanes; they must not collapse CPU, GPU, and NPU implementation claims into one backend.
