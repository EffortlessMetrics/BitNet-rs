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
- The 258V CPU validation harness records evidence only and does not take over shared CPU implementation ownership.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| LNL258V-002 | ready | Document and add the 258V probe bundle, artifact paths, receipt fields, and same-machine comparison hooks without runtime claims. |

## Review Policy

Platform PRs document and compare lanes; they must not collapse CPU, GPU, and NPU implementation claims into one backend.


## Build-Out Sequence

1. Preserve Intel NPU identity before runtime work (`NPU-002-lite` / `NPU-002`).
2. Add exact Arc 140V probe evidence (`ARC140V-002`).
3. Add the visibility-only 258V platform probe (`LNL258V-RUN-001` / `LNL258V-002`).
4. Let the CPU proof lane land strict loader/tokenizer/kernel/runtime work.
5. Add the 258V CPU validation harness (`CPU258V-001`) to record strict receipts and benchmarks on Lunar Lake.

Each item must keep CPU, Arc 140V, and NPU claims separate in generated artifacts.
