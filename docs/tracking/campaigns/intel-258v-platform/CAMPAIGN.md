# Intel 258V Platform Campaign

Campaign ID: `intel-258v-platform`

Status: active

## Objective

Validate Core Ultra 7 258V as a tri-device platform while keeping CPU AVX2, Arc 140V GPU, and Intel AI Boost NPU proof labels separate.

## End State

- Same-machine CPU, GPU, and NPU facts are captured.
- Arc 140V OpenCL, OpenVINO GPU, and OpenVINO NPU evidence are not conflated.
- Receipts record OS, drivers, memory, power, thermal, and WSL/native visibility context.

## Hard Constraints

- Arc 140V OpenCL proof is not NPU proof.
- OpenVINO GPU smoke is not packed BitNet kernel proof.
- WSL only counts for NPU validation if OpenVINO reports NPU inside WSL.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| LNL258V-RUN-001 | merged | Add JSON-ready Lunar Lake platform probe structs. |
| ARC140V-002 | merged | Add exact Arc 140V runtime identity probe logic. |
| LNL258V-002 | merged | Add 258V probe bundle and same-machine comparison hooks. |
| LNL258V-003 | pr_open | Add CLI platform probe emission for the current 258V machine. |

## Review Policy

Platform PRs document and compare lanes; they must not collapse CPU, GPU, and NPU implementation claims into one backend.
