# Intel GPU source-of-truth map

Intel GPU support in BitNet-rs is a vendor-specific accelerator family, not a
single generic "GPU works" claim. The family is split into receipt-backed lanes
that preserve the exact device, runtime, model family, route, fallback status,
quality state, timing profile, residency boundary, and not-claims.

This map aligns the existing A770 and Lunar Lake 258V campaigns before any new
route promotion or receipt schema changes. It is documentation-only: it does not
promote runtime support, alter model coverage, or change kernel behavior.

## Lane boundaries

| Lane | Hardware | Runtime | First target | Claim family |
| --- | --- | --- | --- | --- |
| A770 native GPU | Intel Arc A770, especially 16GB boards | OpenCL first; Level Zero only as a candidate later | BitNet I2_S/QK256 trusted partial acceleration | `intel-arc-a770-opencl` |
| A770 OpenVINO GPU reference | Intel Arc A770 | OpenVINO GPU `GPU.X` | Reference graph/runtime comparison only | `intel-arc-a770-openvino-gpu` |
| Lunar Lake Arc 140V OpenVINO GPU | Arc 140V on Core Ultra 7 258V class systems | OpenVINO GPU `GPU.0` when it resolves to Arc 140V | Dense SLM candidate routing, initially Qwen-family OpenVINO exports | `openvino-gpu` |
| Lunar Lake Arc 140V native OpenCL | Arc 140V on Core Ultra 7 258V class systems | Native OpenCL selected device | Smoke/parity first; BitNet-adjacent fixtures later | `arc140v-opencl` |
| Lunar Lake NPU | Intel AI Boost NPU | OpenVINO NPU | Separate NPU evidence lane | `openvino-npu` |
| CPU reference plate | Host CPU, including 258V CPU routes | CPU scalar/SIMD | Comparator and correctness baseline | not GPU proof |

## Non-conflation rules

These boundaries apply to specs, plans, receipts, route matrices, status pages,
and future `receipts explain` output:

- A770 OpenCL proof is not Arc 140V proof.
- Arc 140V OpenCL proof is not A770 proof.
- OpenVINO GPU proof is not native OpenCL proof.
- OpenVINO GPU proof is not NPU proof.
- Intel GPU proof is not CUDA proof.
- Dense SLM OpenVINO proof is not BitNet QK256/I2_S proof.
- BitNet QK256 proof is not dense SLM proof.
- CPU fallback cannot count as Intel GPU execution.
- Generic OpenCL is not selected Intel GPU proof.
- Generic GPU is not selected Intel GPU proof.

## Current source-of-truth stack

| Artifact | Role |
| --- | --- |
| `docs/specs/intel-arc-a770-gpu-roadmap.md` | Existing A770 hardware/runtime lane; identifies native OpenCL as primary and OpenVINO GPU as reference. |
| `docs/specs/a770-bitnet-claim-boundary.md` | Existing A770 BitNet product claim boundary; limits the first claim to trusted partial BitNet I2_S acceleration with explicit not-claims. |
| `docs/tracking/campaigns/intel-a770/active.toml` | Active A770 campaign execution state. |
| `docs/specs/intel-lunar-lake-gpu-roadmap.md` | Existing Arc 140V GPU roadmap; separates OpenVINO GPU and native OpenCL evidence. |
| `docs/tracking/campaigns/intel-258v-platform/CAMPAIGN.md` | Generated 258V campaign state; keeps CPU, Arc 140V GPU, and NPU proof labels separate. |
| `docs/tracking/campaigns/intel-258v-platform/active.toml` | Active 258V platform execution state. |
| `plans/intel-gpu/implementation-plan.md` | Shared Intel GPU documentation/spec rollout plan. |

## Route truth before promotion

The shared Intel GPU work starts from this posture:

- A770 native OpenCL is the discrete BitNet path, but A770 QK256, embedding,
  and LM-head rows remain diagnostic until committed claim-grade receipts prove
  selected-device execution, fallback-free operation, quality, timing, and
  not-claims.
- A770 OpenVINO GPU is a reference runtime path, not native OpenCL proof.
- Arc 140V OpenVINO GPU is a dense SLM candidate route; promising timings do not
  promote it until exact-profile quality, timing applicability, comparator, and
  telemetry gates pass.
- Arc 140V native OpenCL is an integrated GPU smoke/parity lane; it does not
  imply A770 support or BitNet QK256 support.
- NPU evidence remains a separate OpenVINO NPU lane.
- CPU evidence is the reference plate and never satisfies Intel GPU execution.

## Shared claim ladder

Intel GPU routes use a common claim ladder so detection, execution, answer
quality, performance, and residency are not collapsed:

| Level | Meaning | Public claim |
| --- | --- | --- |
| `unsupported` | no valid route or proof | none |
| `runtime_detected` | device visible | detection only |
| `compile_smoke` | kernel/graph compiles | compile only |
| `kernel_smoke` | tiny kernel/graph executes | smoke only |
| `parity_tested` | CPU/GPU fixture parity | fixture parity |
| `answer_ready` | strict answer corpus or bounded useful answers | answer route |
| `behavior_proven` | prompt conditioning, stop/repetition, long decode | behavior route |
| `benchmark_candidate` | timing fields recorded | diagnostic performance |
| `performance_proven` | quality-gated profile beats baseline with history | exact-profile performance |
| `resident_proven` | named op or phase resident | named residency only |
| `complete` | all required ops, residency, and server gates pass | full route |

`performance_proven`, `resident_proven`, and `complete` are separate states.
Named QK256 linears, embedding, LM-head, dense graph runtime, KV cache, attention
scores, softmax, sampling, and server paths must each carry their own proof.

## Receipt identity minimum

Every Intel GPU route receipt should converge on these fields as specs and code
catch up:

```json
{
  "requested_backend": "intel-arc-a770 | intel-arc-140v | openvino-gpu",
  "selected_backend": "intel-arc-a770-opencl | intel-arc-140v-opencl | openvino-gpu",
  "runtime_api": "opencl | openvino_genai | openvino_runtime | level_zero",
  "runtime_device": "GPU.0 | GPU.1 | OpenCL platform/device index",
  "fallback_used": false,
  "fallback_reason": null,
  "model_family": "bitnet | dense_slm | small_llm",
  "proof_family": "bitnet_qk256_opencl | dense_slm_openvino_gpu | arc140v_opencl_smoke",
  "device_identity": {
    "name": "...",
    "vendor": "Intel",
    "pci_device_id": "0x56A0 | 0x64A0 | ...",
    "driver_version": "...",
    "vram_or_shared_memory_bytes": 0
  },
  "claim_boundary": {
    "native_opencl_proof": true,
    "openvino_gpu_proof": false,
    "bitnet_qk256_proof": true,
    "dense_slm_proof": false,
    "full_residency_claim": false,
    "speedup_claim": false
  }
}
```

Route-specific specs may add stricter fields; they must not weaken selected
backend identity or fallback truth.

## Immediate rollout

The shared rollout is staged in `plans/intel-gpu/implementation-plan.md`:

1. Add this source-of-truth map and the shared implementation plan.
2. Add Intel GPU proposal/spec documents for route identity, device identity,
   BitNet QK256, dense SLM, quality, performance, residency, and status
   surfaces.
3. Reconcile A770 route truth against committed claim-grade receipts.
4. Productize A770 native OpenCL only through selected-device, answer-quality,
   timing, and named-residency receipts.
5. Productize Arc 140V OpenVINO GPU only per dense-SLM exact profile after
   quality and timing blockers are closed.
6. Keep Arc 140V native OpenCL as smoke/parity until BitNet-adjacent fixtures and
   answer proof exist.
7. Add shared UX surfaces such as capability matrices, `receipts explain`, and
   `gpu doctor` without broadening claims.
