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


## Needed Build-Out Parts

| Part | Status to unblock | Acceptance boundary |
|---|---|---|
| `LNL258V-RUN-001` platform probe | Ready after docs merge. | Visibility-only JSON records CPU, Arc 140V, Level Zero, OpenVINO GPU, OpenVINO NPU, `/dev/accel`, power, thermal, memory, and OS context. |
| `NPU-002-lite` backend identity | Must precede NPU runtime smoke. | `npu` preserves Intel NPU/OpenVINO NPU identity and never aliases to Metal or generic GPU. |
| `ARC140V-002` Arc runtime probe | Must precede Arc kernel smoke. | PCI ID `0x64A0` or exact Arc 140V full name is recorded through OpenCL/Level Zero/OpenVINO GPU without CPU fallback. |
| `CPU258V-001` CPU validation harness | Starts only after CPU-proof strict loader/tokenizer authority. | Strict CPU receipt records loader, tokenizer, requested/selected kernel, requested/selected backend, phase metrics, and zero fallback. |

The campaign can collect same-machine artifacts, but CPU loader, tokenizer, QK256 layout, AVX2 dispatch, and transformer hot-path implementation remain owned by the CPU-proof campaign unless a future work item explicitly changes that scope.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| LNL258V-002 | ready | Add 258V probe bundle and same-machine comparison hooks. |

## Review Policy

Platform PRs document and compare lanes; they must not collapse CPU, GPU, and NPU implementation claims into one backend.
