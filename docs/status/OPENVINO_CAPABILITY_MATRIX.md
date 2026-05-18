# OpenVINO Capability Matrix

This page is the user-facing status map for the Lunar Lake OpenVINO lane. It
summarizes the current Qwen2.5 dense SLM OpenVINO CPU/GPU/NPU evidence and the
separate BitNet OpenVINO subgraph research boundary without changing route
promotion.

The operational source of truth remains the Intel 258V campaign tracker, the
OpenVINO Lunar Lake specs, the route-promotion ledger, and the committed
receipts. Start here before running:

```powershell
bitnet lunar-lake validate --strict
bitnet lunar-lake regress --strict
bitnet lunar-lake compare --strict
bitnet validate open-vino-lunar-lake --receipt <receipt.json>
bitnet receipts explain <receipt.json>
```

## Source Of Truth

| Surface | Source |
| --- | --- |
| OpenVINO route identity and fallback rules | `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md` |
| Dense SLM artifact/export contract | `docs/specs/BITNET-SPEC-OPENVINO-DENSE-SLM.md` |
| NPU cold/cache/warm evidence contract | `docs/specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md` |
| Quality corpus and token-ID visibility contract | `docs/specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md` |
| Phase timing contract | `docs/specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md` |
| Route promotion gates | `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md` |
| BitNet proof boundary | `docs/specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md` |
| Rust bridge boundary | `docs/specs/BITNET-SPEC-OPENVINO-RUST-BRIDGE.md` |
| Server boundary | `docs/specs/BITNET-SPEC-OPENVINO-SERVER.md` |
| Implementation plan | `plans/openvino-lunar-lake/implementation-plan.md` |
| Route promotion ledger | `ci/hardware/intel-258v/2026-05-08/lunar-lake-route-promotion.json` |
| Route profile comparison | `ci/hardware/intel-258v/2026-05-08/lunar-lake-route-profile-comparison.json` |
| OpenVINO corpus-v2 receipt | `ci/hardware/intel-258v/2026-05-08/slm-openvino-cpu-gpu-npu-corpus-v2.json` |
| OpenVINO phase runner | `ci/hardware/intel-258v/2026-05-08/slm-openvino-cpu-gpu-npu-phase-runner.json` |

## Current OpenVINO Rows

| Row | Model / proof family | Backend | Runtime device | Status | Evidence | Boundary |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5 OpenVINO CPU | Dense SLM OpenVINO GenAI | `openvino-cpu` | `CPU` | candidate | IR manifest, LLMPipeline smoke, corpus-v2 receipt, phase runner, Rust-vs-OpenVINO CPU comparison | Not the promoted default; dense GGUF CPU remains default until a route-promotion review changes it. |
| Qwen2.5 OpenVINO GPU | Dense SLM OpenVINO GenAI | `openvino-gpu` | `GPU.0` / Arc 140V | candidate | Arc 140V OpenVINO ask receipt, corpus-v2 receipt, phase runner, route-profile blockers, validator gate | No native OpenCL claim; no speedup or route promotion until exact-profile quality and timing pass. |
| Qwen2.5 OpenVINO NPU | Dense SLM OpenVINO GenAI | `openvino-npu` | `NPU` | candidate | NPU ask receipt, corpus-v2 receipt, phase runner, cold-start diagnosis, route-profile blockers, validator gate | No cold one-off usability claim; warm/resident or low-power promotion requires cache plus warm/resident evidence. |
| Qwen2.5 OpenVINO server | Dense SLM exact-profile server | OpenVINO route under server | exact profile only | planned | Server spec only | No server readiness until ask/chat readiness, underlying route linkage, exposure fields, cold/warm timing, streaming/concurrency boundaries, and exact-profile receipts exist. |
| BitNet OpenVINO NPU subgraphs | BitNet-shaped static subgraphs | `openvino-npu` | `NPU` | diagnostic | RMSNorm, linear projection, and FFN/static subgraph parity receipts in the 258V lane | No full BitNet inference, QK256 decode, dynamic decode, or acceleration claim. |
| BitNet OpenVINO GPU/subgraphs | BitNet-shaped OpenVINO research | `openvino-gpu` | `GPU.0` | planned/diagnostic | Current Arc GPU proof is OpenVINO smoke plus separate native OpenCL parity outside this row | No full BitNet GPU inference, no native OpenCL proof, and no QK256 accelerator decode claim. |

## Route Status

| Route | Promotion status | Promoted profiles | Current blockers |
| --- | --- | --- | --- |
| `dense_slm_default_cpu` | promoted | `regression_tiny`, `ask_short`, `ask_normal` | CPU path is trusted but remains performance-sensitive; OpenVINO routes may replace it only through exact-profile promotion. |
| `dense_slm_openvino_gpu_candidate` | candidate | none | Corpus-v2 failures, direct generated-token visibility gaps, incomplete profile timing splits, and missing benchmark-qualified advantage. |
| `dense_slm_openvino_npu_candidate` | candidate | none | Corpus-v2 failures, high cold load, missing cache/resident proof, and missing low-power advantage evidence. |
| `bitnet_reference_cpu` | promoted for BitNet reference only | `bitnet_strict_reference` | Not a dense SLM route; dense SLM success never counts as BitNet proof. |

## Claim Boundaries

- OpenVINO dense SLM receipts are not BitNet packed I2_S/QK256 proof.
- OpenVINO GPU receipts are not native OpenCL execution proof.
- OpenVINO NPU receipts are not native NPU kernel proof.
- OpenVINO GPU/NPU candidate evidence is not route promotion.
- `fallback_used=false` or an explicit strict no-fallback policy is mandatory
  for OpenVINO status rows.
- Retokenized generated token IDs must be labeled as retokenized and must not
  be described as direct OpenVINO GenAI pipeline internals.
- NPU hot-path timing does not prove cold one-off usability.
- NPU route promotion requires cache identity plus warm or resident evidence.
- Speedup, power advantage, low-power promotion, and server readiness require
  separate exact-profile receipts.

## Validator

The receipt-boundary validator rejects hidden fallback, backend/device drift,
retokenized-token ambiguity, dense-SLM-to-BitNet claim leakage,
OpenVINO-GPU-to-native-OpenCL claim leakage, and premature NPU promotion
without cache plus warm/resident evidence:

```powershell
bitnet validate open-vino-lunar-lake `
  --receipt ci\hardware\intel-258v\2026-05-08\slm-openvino-cpu-gpu-npu-corpus-v2.json
```

## Next Proofs

| Row | Next proof |
| --- | --- |
| Qwen2.5 OpenVINO CPU | Decide whether OpenVINO CPU can beat the dense GGUF CPU default for any exact profile after quality and timing review. |
| Qwen2.5 OpenVINO GPU | Run profile-specific quality/timing review and promote only exact profiles that pass corpus v2 with fallback-free GPU execution and a benchmark-qualified advantage. |
| Qwen2.5 OpenVINO NPU | Add cache, warm/resident, and low-power evidence before any warm or low-power route promotion. |
| OpenVINO server | Add exact-profile server receipts only after the underlying ask/chat route is promoted or explicitly candidate-scoped. |
| BitNet OpenVINO subgraphs | Continue static-shape CPU-reference parity for selected subgraphs; keep QK256/dynamic decode out of scope until separately proven. |

## Validation

Run these checks after editing this page or the underlying status surfaces:

```powershell
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
npx --yes markdownlint-cli2@0.18.1 --config .markdownlint.jsonc docs/status/OPENVINO_CAPABILITY_MATRIX.md docs/status/README.md plans/openvino-lunar-lake/implementation-plan.md
git diff --check
```
