# BITNET-PROP-0007: NPU Productization

Status: proposed
Owner: BitNet-rs maintainers
Campaign: `intel-npu`
Plan: `plans/npu/implementation-plan.md`
Roadmap: `docs/specs/intel-lunar-lake-npu-roadmap.md`

## Thesis

NPUs are BitNet-rs's governed low-power / resident inference lane. They must not
be treated as generic accelerators or replacements for CUDA, OpenCL, AVX2, or
AVX-512 throughput work. Their first product value is small dense SLM
warm/resident inference and selected static BitNet-shaped subgraph parity. Full
BitNet QK256 decode is a later research target unless receipts prove otherwise.

## Why this exists

Modern laptops increasingly include NPUs that can keep small models available
without tying up CPU or GPU resources. For BitNet-rs, that value is different
from peak throughput. The NPU lane should optimize for short-answer,
low-power, cached, warm, and resident profiles where a local assistant can stay
ready in the background.

The current concrete target is Intel AI Boost NPU on Lunar Lake / Core Ultra 7
258V through OpenVINO NPU. OpenVINO's NPU path currently makes static-shape and
cache behavior central to the product contract, so the first BitNet work is
static subgraph parity rather than dynamic autoregressive QK256 decode.

## First implementation target

- Hardware: Intel AI Boost NPU on Lunar Lake / Core Ultra 7 258V.
- Runtime: OpenVINO NPU.
- First useful dense SLM route: Qwen2.5 0.5B / Qwen3-class OpenVINO GenAI
  experiments for warm/resident short-answer profiles.
- First BitNet route: selected static BitNet-shaped subgraph parity.
- First status surface: receipts, `npu doctor`, model status, and route status
  that show what is proven and what is not claimed.

## Future NPU families

Apple Neural Engine, Qualcomm Hexagon, and AMD Ryzen AI are future NPU families.
They require family-specific proposals/specs/proofs and do not inherit Intel
OpenVINO NPU proof. A single Intel route must not create a generic NPU support
claim.

## Claim boundary

Allowed early claims:

- Intel NPU backend identity is distinct from CPU, GPU, CUDA, Metal, OpenCL, and
  generic accelerator labels.
- OpenVINO can report NPU visibility when receipts record it.
- Tiny static OpenVINO NPU graphs can be claimed only from graph execution
  receipts with `fallback_used=false`.
- Selected static BitNet-shaped subgraph parity can be claimed only for the
  named subgraph, shape, tolerance, reference backend, target backend, and
  receipt.
- Dense SLM NPU routes can be candidates before profile-specific quality and
  timing gates pass.

Must not claim without later receipts:

- Native bitnet-rs NPU inference.
- Full BitNet inference on NPU.
- Packed QK256 decode on NPU.
- Native NPU packed kernels.
- Broad NPU speedup.
- Cold one-off NPU usability when compile/load dominates.
- Full residency.
- Broad dense SLM quality.
- CPU fallback as NPU proof.

## User impact

The end state is a status surface where a user can distinguish:

```text
NPU visible: yes/no
OpenVINO sees NPU: yes/no
NPU graph smoke: pass/fail
static BitNet subgraph parity: pass/fail
dense SLM NPU route: candidate/promoted/blocked
cold start: measured
cached start: measured
resident/warm session: measured
quality corpus: passed/failed
fallback_used: false
not claimed: BitNet QK256 full inference, native NPU packed kernels, broad speedup, full residency
```

## References

- OpenVINO NPU device documentation: <https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html>
- OpenVINO GenAI on NPU documentation: <https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html>
- Intel NPU Driver for Windows: <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>
