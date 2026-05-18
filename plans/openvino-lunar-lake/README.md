# OpenVINO Lunar Lake Plan Index

Status: proposed
Owner: intel-runtime/product
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../../docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs:
- [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../../docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md)
Linked ADRs: n/a
Linked plan: [implementation-plan.md](implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: No support-tier promotion.
Policy impact: No policy exception.

This directory sequences the OpenVINO Lunar Lake productization campaign.
OpenVINO is the governed Intel-runtime lane for dense SLMs and selected small
LLMs on Lunar Lake CPU/GPU/NPU, plus a separate static BitNet-shaped
subgraph/reference lane.

Start with [implementation-plan.md](implementation-plan.md). The first work item
is docs-only and adds the productization proposal, route identity contract, and
PR-sized plan. Runtime changes, route promotion, speedup claims, server claims,
and BitNet QK256 claims are out of scope for the first PR.
