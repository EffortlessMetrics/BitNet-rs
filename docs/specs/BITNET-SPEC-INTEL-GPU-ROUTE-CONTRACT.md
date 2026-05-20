# BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT

Status: proposed
Owner: intel-gpu/product
Created: 2026-05-18
Linked proposal: `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-INTEL-GPU-DEVICE-IDENTITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-QUALITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-PERFORMANCE.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-RESIDENCY.md`
Linked ADRs: n/a
Linked plan: `plans/intel-gpu/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines labels; no promotion.
Policy impact: No exception.

## Route identities

Valid Intel GPU route IDs are concrete and model/runtime scoped:

```text
intel_arc_a770_opencl_bitnet_qk256
intel_arc_a770_opencl_embedding
intel_arc_a770_opencl_lm_head
intel_arc_a770_openvino_gpu_reference
intel_arc_140v_opencl_smoke
intel_arc_140v_opencl_bitnet_candidate
intel_arc_140v_openvino_gpu_dense_slm
intel_gpu_level_zero_candidate
```

## Backend labels

- `selected_backend` must be concrete. Values such as `gpu`, `opencl`, or
  `intel-gpu` are convenience selectors only and cannot appear as selected
  proof backends.
- Native A770 OpenCL proof uses `selected_backend=intel-arc-a770-opencl` and
  `runtime_api=opencl`.
- Native Arc 140V OpenCL proof uses `selected_backend=intel-arc-140v-opencl` and
  `runtime_api=opencl`.
- OpenVINO GPU proof uses `selected_backend=openvino-gpu` and
  `runtime_api=openvino_genai` or `runtime_api=openvino_runtime`.
- Level Zero evidence remains candidate evidence until a route-specific spec and
  proof receipts promote it.

## Claim rules

- `fallback_used=false` is required for any Intel GPU route claim.
- `fallback_reason` must be `null` when `fallback_used=false`.
- OpenVINO GPU receipts must record `GPU.X` and full device name.
- OpenCL receipts must record platform index, device index, and full device
  name.
- A770 receipts must record PCI ID `0x56A0` when available.
- Arc 140V receipts must record PCI ID `0x64A0` when available.
- CPU fallback, generic GPU selection, or generic OpenCL selection cannot
  satisfy selected Intel GPU proof.

## Claim levels

| Level | Meaning | Public claim |
| --- | --- | --- |
| `unsupported` | no valid route or proof | none |
| `runtime_detected` | device visible | detection only |
| `compile_smoke` | kernel/graph compiles | compile only |
| `kernel_smoke` | tiny kernel/graph executes | smoke only |
| `parity_tested` | CPU/GPU fixture parity | fixture parity |
| `answer_ready` | strict answer corpus or bounded useful answers | answer route |
| `behavior_proven` | prompt conditioning, stop/repetition, long decode | behavior route |
| `benchmark_candidate` | timing fields recorded | diagnostic perf |
| `performance_proven` | quality-gated profile beats baseline with history | exact-profile perf |
| `resident_proven` | named op resident | named residency only |
| `complete` | all required ops/residency/server gates pass | full route |

`performance_proven`, `resident_proven`, and `complete` must never be collapsed.

## Minimum route receipt shape

```json
{
  "requested_backend": "intel-arc-a770 | intel-arc-140v | openvino-gpu",
  "selected_backend": "intel-arc-a770-opencl | intel-arc-140v-opencl | openvino-gpu",
  "runtime_api": "opencl | openvino_genai | openvino_runtime | level_zero",
  "runtime_device": "GPU.0 | GPU.1 | OpenCL platform/device index",
  "fallback_used": false,
  "fallback_reason": null,
  "model_family": "bitnet | dense_slm | small_llm",
  "proof_family": "bitnet_qk256_opencl | dense_slm_openvino_gpu | arc140v_opencl_smoke"
}
```
