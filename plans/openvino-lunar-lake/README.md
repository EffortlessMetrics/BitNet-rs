# OpenVINO Lunar Lake Plan

This plan sequences the OpenVINO productization lane described by
[BITNET-PROP-0004](../../docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md).
The lane governs dense SLM and selected small-LLM OpenVINO proof on Lunar Lake
CPU/GPU/NPU, plus a separate static BitNet-shaped subgraph reference lane.

## Source-Of-Truth Links

| Surface | Path |
| --- | --- |
| Product proposal | `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md` |
| Route contract | `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md` |
| Source-of-truth rules | `docs/reference/SPEC_SYSTEM.md` |
| 258V validation profile | `docs/hardware/intel-258v-validation.md` |
| 258V platform roadmap | `docs/specs/intel-lunar-lake-258v-platform-roadmap.md` |
| 258V buildout plan | `docs/specs/intel-lunar-lake-258v-buildout-plan.md` |
| Arc 140V roadmap | `docs/specs/intel-lunar-lake-gpu-roadmap.md` |
| NPU roadmap | `docs/specs/intel-lunar-lake-npu-roadmap.md` |
| Campaign manifest | `docs/tracking/campaigns/intel-258v-platform/active.toml` |
| Receipts | `ci/hardware/intel-258v/**` |

## Plan Files

| File | Owns |
| --- | --- |
| `implementation-plan.md` | PR order, dependencies, proof commands, and rollback for OpenVINO productization |

## Operating Rules

- Keep OpenVINO CPU, OpenVINO GPU.0 Arc 140V, OpenVINO NPU, native OpenCL,
  BitNet QK256, and server proof families separate.
- Do not promote OpenVINO GPU/NPU routes from docs/spec PRs.
- Do not claim speedup, power advantage, full residency, broad dense SLM
  quality, native OpenCL proof, or BitNet QK256 proof unless the exact contract
  and receipts support that claim.
- Treat OpenVINO AUTO/HETERO as diagnostic unless execution devices are
  recorded.
- Treat retokenized generated text as diagnostic token evidence, not direct
  pipeline-internal generated token IDs.
- Keep model binaries uncommitted.
- Keep Python proof harnesses until Rust surfaces emit equivalent receipts and
  pass equivalent validators.
