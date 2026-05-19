# BITNET-SPEC-NPU-PROOF-LADDER

Status: draft
Proposal: `docs/proposals/BITNET-PROP-0007-npu-productization.md`
Plan: `plans/npu/implementation-plan.md`

## Purpose

Standardize NPU maturity levels. Promotion is per model + route + profile, not
global. A higher level for one dense SLM profile does not promote BitNet QK256,
other model families, other NPUs, or cold one-off usage.

## Levels

| Level | Name | Allowed claim |
| ---: | --- | --- |
| 0 | `detected` | Device/runtime visible. |
| 1 | `compile_smoke` | Runtime accepts compile path. |
| 2 | `graph_smoke` | Tiny static graph executes on selected NPU. |
| 3 | `subgraph_parity` | Selected static model subgraph matches CPU reference. |
| 4 | `answer_candidate` | Dense SLM route generates bounded useful answers. |
| 5 | `quality_candidate` | Corpus/profile quality passes. |
| 6 | `benchmark_candidate` | Phase timing/cold/warm/cache data exists. |
| 7 | `promoted_for_profile` | Exact profile promoted with fallback false and quality/timing proof. |
| 8 | `resident_ready` | Warm/resident route is validated. |
| 9 | `full_inference_candidate` | Full model route exists but is not broadly claimed. |
| 10 | `complete` | Full route, quality, speed, residency, and server gates pass. |

## Promotion requirements

Every promotion record must include:

- model identity and artifact/export identity,
- route ID,
- profile ID,
- proof ladder level,
- selected backend/runtime/device,
- `fallback_used=false`,
- quality result where answer generation is involved,
- cold/cache/warm timing split where performance is claimed,
- explicit not-claims.
