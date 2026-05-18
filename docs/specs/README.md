# Specs

Specs define what must be true before BitNet-rs accepts a behavior, claim,
proof boundary, or policy contract. They are the acceptance layer between a
proposal and implementation work.

Specs should not duplicate product rationale, active queues, or operational
ledgers. They should point to the authoritative gate, policy TOML, matrix,
manifest, or receipt format and define how that authority is used.

## Source-of-truth role

| Question | Source of truth |
| --- | --- |
| Why does this lane exist? | `docs/proposals/` |
| What must be true? | `docs/specs/` |
| What decision did we make? | `docs/adr/` |
| What PRs execute it? | `plans/` plus the active goal or campaign manifest |
| What is actively executing now? | `.bitnet/goals/active.toml` or `docs/tracking/campaigns/<campaign>/active.toml` |
| What is currently supported? | `docs/status/` plus proof artifacts |
| What does CI enforce? | `policy/*.toml` and workflow gates |
| What happened? | Receipts, artifacts, campaign events, closeouts |

## BitNet claim rules

Specs for model, hardware, or CI claims must preserve these boundaries:

- Structural GGUF loading is not answer readiness.
- Tokenizer authority is not answer readiness by itself.
- Hardware execution proof is not answer quality.
- Dense SLM proof is not BitNet or 1-bit proof.
- Backend receipts are lane-specific and must not collapse CPU, CUDA, Metal,
  MPSGraph, OpenCL, WGPU, NPU, or platform identities.
- Expensive CI evidence belongs on risk-routed, main, nightly, release, or
  campaign lanes unless policy explicitly promotes it.

## Existing operational authorities

Specs should link to these maintained authorities instead of copying them:

- `docs/model-artifacts/ANSWER_ARTIFACT_GATE.md`
- `docs/model-artifacts/MODEL_COVERAGE_MATRIX.md`
- `docs/hardware/HARDWARE_MATRIX.md`
- `docs/hardware/PROOF_STAGES.md`
- `docs/ci/cost-and-verification-policy.md`
- `docs/tracking/TRACKER_MODEL.md`
- `policy/ci-lanes.toml`
- `policy/ci-budget.toml`
- `policy/ci-risk-packs.toml`

## Spec shape

New specs should include:

```text
Status:
Owner:
Created:
Linked proposal:
Linked ADRs:
Linked plan:
Linked issues:
Linked PRs:
Support-tier impact:
Policy impact:
```

Recommended sections:

- Problem
- Behavior
- Non-goals
- Required evidence
- Acceptance examples
- Test mapping
- Implementation mapping
- CI proof
- Metrics / promotion rule
- Claim boundaries
