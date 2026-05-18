# Intel GPU productization plans

Status: proposed
Owner: intel-gpu/product
Created: 2026-05-18
Linked proposal: `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-DEVICE-IDENTITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-BITNET-QK256.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-DENSE-SLM.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-QUALITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-PERFORMANCE.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-RESIDENCY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-STATUS-SURFACE.md`
Linked ADRs: n/a
Linked plan: `plans/intel-gpu/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Documentation only; no support-tier promotion.
Policy impact: Keeps A770, Arc 140V, OpenVINO GPU, NPU, CPU, and CUDA proof families separate.

This directory sequences Intel GPU productization work. It spans the A770
native OpenCL BitNet lane and the Lunar Lake Arc 140V OpenVINO/native OpenCL
lanes while preserving strict proof-family boundaries.

Start with `implementation-plan.md` for PR order, proof commands, non-goals,
and rollback guidance.
