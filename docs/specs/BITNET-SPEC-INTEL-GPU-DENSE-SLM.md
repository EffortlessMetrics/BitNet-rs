# BITNET-SPEC-INTEL-GPU-DENSE-SLM

Status: proposed
Owner: intel-gpu/product
Created: 2026-05-18
Linked proposal: `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-QUALITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-PERFORMANCE.md`
Linked ADRs: n/a
Linked plan: `plans/intel-gpu/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines dense SLM OpenVINO GPU requirements; no promotion.
Policy impact: No exception.

## Initial target

The initial dense SLM Intel GPU target is:

```text
Qwen2.5 0.5B Instruct OpenVINO INT4 symmetric export on Lunar Lake Arc 140V GPU.0
```

This is dense SLM OpenVINO GPU proof. It is not BitNet QK256 proof, native
OpenCL proof, NPU proof, CUDA proof, or A770 proof.

## Proof ladder

Dense SLM OpenVINO GPU routes graduate through:

```text
export manifest
runtime/device identity
OpenVINO GPU bounded smoke
operator ask
corpus v2
phase timing
profile comparison
promotion review
model status
server exact-profile optional
```

## Promotion rule

OpenVINO GPU can be promoted only per profile after:

- `fallback_used=false`;
- quality passes for that profile;
- profile timing is applicable;
- benchmark-qualified advantage exists, or a reviewed UX/power advantage is
  accepted for that exact profile;
- telemetry context is present or explicitly unavailable;
- generated-token limitation is recorded.

## Known blocker classes to formalize

Candidate route reviews must classify and preserve blockers such as:

- corpus quality failures;
- missing direct generated-token IDs;
- missing prompt-token timing applicability;
- incomplete phase splits;
- missing profile regression bundle;
- missing benchmark-qualified speed or power advantage.

A bounded OpenVINO GPU answer receipt can be useful evidence while the route
remains unpromoted.
