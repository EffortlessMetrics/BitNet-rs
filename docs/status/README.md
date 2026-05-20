# Status Documents

Status documents are the user-facing claim map for BitNet-rs. They summarize
which model families, hardware lanes, backend routes, and proof stages are
usable, diagnostic, experimental, or unsupported.

Status pages must link to proof. If a claim cannot point to a model artifact,
hardware receipt, CI lane, policy ledger, or campaign closeout, keep it
diagnostic, advisory, or planned.

## Source-Of-Truth Role

| Status surface | Owns |
| --- | --- |
| Capability matrix | Product claim tier, proof command, proof artifact, claim boundary |
| Claim boundaries | What a proof does and does not allow the README or CLI docs to say |
| README summary | Short user entry point, not the final proof map |
| Handoff | Operator transfer context and validation gaps, not claim authority |
| Receipt/artifact | Evidence for one run, model, backend, or lane |

## Required Claim Boundary

Status documents must preserve BitNet-specific distinctions:

- Dense SLM support is not BitNet or 1-bit proof.
- BitNet I2_S CPU proof is not CUDA proof.
- CUDA receipt validation is not coherent answer readiness.
- Hardware detection is not speed proof.
- Structural model validity is not answer readiness.
- Diagnostic receipts are useful evidence but must remain diagnostic until the
  relevant artifact and answer gates pass.

## Status Surfaces

Current maintained status surfaces:

```text
docs/status/CUDA_CAPABILITY_MATRIX.md
docs/status/OPENVINO_CAPABILITY_MATRIX.md
```

Planned status surfaces include:

```text
docs/status/CAPABILITY_MATRIX.md
docs/status/CLAIM_BOUNDARIES.md
```

These pages summarize proof. They do not replace the operational authorities:

- `README.md` for high-level user positioning.
- `ROADMAP.md` for release direction and current limitations.
- `docs/model-artifacts/ANSWER_ARTIFACT_GATE.md` for answer-readiness.
- `docs/model-artifacts/MODEL_COVERAGE_MATRIX.md` for model coverage.
- `docs/hardware/HARDWARE_MATRIX.md` for hardware lane identity.
- `docs/ci/cost-and-verification-policy.md` for CI economics.
- `docs/tracking/TRACKER_MODEL.md` for campaign execution state.
- `docs/handoffs/` for operator transfer notes and closeout summaries.
