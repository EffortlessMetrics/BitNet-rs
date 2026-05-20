# OpenVINO Lunar Lake Productization Plan

Status: active
Owner: intel/openvino
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../../docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../../docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md)
Linked ADRs: n/a
Linked plan: [implementation-plan.md](implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion; plan-only governance
Policy impact: no policy exception

## Goal

Lay down the docs, specs, and receipt-governance rails for OpenVINO
productization on the Lunar Lake 258V platform. This is a docs/specs and
receipt-governance campaign before it is a runtime promotion campaign.

OpenVINO should become BitNet-rs's governed Intel-runtime lane for dense SLMs
and selected small LLMs on Lunar Lake CPU/GPU/NPU, plus a clearly separate
reference lane for future BitNet graph/subgraph experiments.

## Current Product Targets

1. Qwen2.5 0.5B Instruct OpenVINO on Lunar Lake:
   - CPU = correctness/reference route;
   - GPU.0 / Arc 140V = likely first interactive speed candidate;
   - NPU / Intel AI Boost = warm/resident low-power candidate.
2. Qwen3 / SmolLM / Llama/Gemma/Phi small models through the same proof ladder.
3. BitNet-shaped OpenVINO subgraphs as a separate research/reference ladder.

Do not collapse these targets.

## Hard Rails

- Do not promote OpenVINO GPU/NPU routes from docs PRs.
- Do not claim speedup.
- Do not claim broad dense SLM quality.
- Do not claim BitNet QK256 from OpenVINO dense SLM receipts.
- Do not claim native OpenCL from OpenVINO GPU receipts.
- Do not claim cold one-off NPU usability from hot-path numbers.
- Do not treat retokenized generated text as direct pipeline-internal generated
  token IDs.
- Keep model binaries uncommitted.
- Keep Python proof harnesses until Rust surfaces emit equivalent receipts and
  pass the same validators.

## Validation for Docs/Rails PRs

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

See [implementation-plan.md](implementation-plan.md) for PR-sized work items.
