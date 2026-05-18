# OpenVINO Lunar Lake Productization Plan

This plan turns the existing Lunar Lake 258V OpenVINO proof lane into governed,
receipt-backed product surfaces for dense SLMs and selected small LLMs, while
keeping BitNet-shaped OpenVINO graph work separate as a reference/research lane.

The product identities are intentionally narrow:

```text
CPU control:        OpenVINO CPU for exact dense SLM export/profile
GPU candidate:     OpenVINO GPU.0 / Intel Arc 140V for exact dense SLM profiles
NPU candidate:     OpenVINO NPU / Intel AI Boost for warm or resident profiles
BitNet reference:  selected static OpenVINO subgraphs only
```

## Source-Of-Truth Links

- Proposal:
  [BITNET-PROP-0004](../../docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
- Route contract:
  [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../../docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md)
- Platform roadmap:
  [intel-lunar-lake-258v-platform-roadmap](../../docs/specs/intel-lunar-lake-258v-platform-roadmap.md)
- GPU roadmap:
  [intel-lunar-lake-gpu-roadmap](../../docs/specs/intel-lunar-lake-gpu-roadmap.md)
- NPU roadmap:
  [intel-lunar-lake-npu-roadmap](../../docs/specs/intel-lunar-lake-npu-roadmap.md)
- Live campaign state:
  `docs/tracking/campaigns/intel-258v-platform/active.toml`
- Receipt root:
  `ci/hardware/intel-258v/**`

## Files

- [implementation-plan.md](implementation-plan.md) lists PR-sized work items.

Future plan shards may split dense SLM model contracts, quality, timing,
NPU cold/cache/warm proof, route promotion, Rust bridge, server readiness, and
BitNet subgraph research after the first docs-only rails are merged.

## Claim Boundary

This plan does not claim new runtime behavior. It only defines the order,
acceptance, receipts, and rollback paths for future PRs.

Do not use this plan to claim:

- OpenVINO dense SLM proof as BitNet QK256 proof;
- OpenVINO GPU proof as native OpenCL proof;
- OpenVINO NPU proof as Arc 140V proof;
- CPU fallback as GPU or NPU execution;
- hot-path NPU timing as cold one-off readiness;
- retokenized generated text as direct pipeline-internal generated token IDs;
- speedup, low power, full residency, broad server readiness, or broad dense SLM
  quality without exact-profile receipts.

## Validation For This Plan PR

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```
