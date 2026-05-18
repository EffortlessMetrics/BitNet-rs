# BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT

## Purpose

Define concrete Intel GPU route identities so receipts, route matrices, status
surfaces, and `receipts explain` cannot collapse A770, Arc 140V, OpenVINO GPU,
OpenCL, Level Zero, CPU, NPU, CUDA, BitNet QK256, or dense SLM proof families.

## Route IDs

- `intel_arc_a770_opencl_bitnet_qk256`
- `intel_arc_a770_opencl_embedding`
- `intel_arc_a770_opencl_lm_head`
- `intel_arc_a770_openvino_gpu_reference`
- `intel_arc_140v_opencl_smoke`
- `intel_arc_140v_opencl_bitnet_candidate`
- `intel_arc_140v_openvino_gpu_dense_slm`
- `intel_gpu_level_zero_candidate`

## Backend and runtime rules

- `selected_backend` must be concrete; generic `gpu`, `opencl`, or `intel` is
  not a claim backend.
- `runtime_api` must be concrete: `opencl`, `openvino_genai`,
  `openvino_runtime`, or `level_zero`.
- `fallback_used=false` is required for any Intel GPU route claim.
- OpenVINO GPU receipts must record `GPU.X` and the full device name.
- OpenCL receipts must record platform index, device index, and full device
  name.
- A770 receipts must record PCI ID `0x56A0` when available.
- Arc 140V receipts must record PCI ID `0x64A0` when available.
- CPU fallback, CPU comparator evidence, and CPU reference results must be
  labelled as CPU evidence and cannot satisfy Intel GPU execution.

## Claim levels

Routes use the common Intel GPU ladder: `unsupported`, `runtime_detected`,
`compile_smoke`, `kernel_smoke`, `parity_tested`, `answer_ready`,
`behavior_proven`, `benchmark_candidate`, `performance_proven`,
`resident_proven`, and `complete`. `performance_proven`, `resident_proven`, and
`complete` must not be collapsed.
