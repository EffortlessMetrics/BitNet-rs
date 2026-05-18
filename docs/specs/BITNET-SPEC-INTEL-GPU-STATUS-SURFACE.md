# BITNET-SPEC-INTEL-GPU-STATUS-SURFACE

## Purpose

Make Intel GPU route truth visible to users, maintainers, and agents.

## Target commands

Status and explanation surfaces should eventually include:

```bash
bitnet model status --device intel-arc-a770-opencl
bitnet model status --device intel-arc-140v-openvino-gpu
bitnet receipts explain <receipt>
bitnet lunar-lake routes --format json
bitnet gpu doctor --vendor intel
```

## Required output fields

Outputs should include route ID, proof family, claim level, selected backend,
runtime API, quality status, performance status, residency status, server status,
not-claims, and next required proof.

## Explanation examples

`receipts explain` must distinguish statements such as:

- This is A770 native OpenCL BitNet proof.
- This is Arc 140V OpenVINO GPU dense SLM proof.
- This is Arc 140V native OpenCL smoke proof.
- This is not NPU proof.
- This is not CUDA proof.
- This is not full residency.
