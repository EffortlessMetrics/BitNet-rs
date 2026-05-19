# BITNET-SPEC-TL2-ROUTE-CONTRACT

Status: draft
Owner: BitNet-rs contributors
Created: 2026-05-19
Linked proposal: docs/proposals/BITNET-PROP-0018-tl2-productization.md
Linked specs: self
Linked ADRs: docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md
Linked plan: plans/tl2/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: none

Route IDs: tl2_artifact_inventory, tl2_layout_fixture, tl2_scalar_reference, tl2_avx2_reference, tl2_avx512_reference, tl2_x86_cpu_answer, tl2_cuda_candidate, tl2_warm_session, tl2_benchmark_profile.

Strict `--route tl2 --strict` must fail closed when artifact or kernel is unavailable.
