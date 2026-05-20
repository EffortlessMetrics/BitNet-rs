# BITNET-SPEC-FALCON3-FAMILY-ROUTE-COMPATIBILITY: Falcon3 Route Compatibility

Status: draft
Owner: model-artifacts
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0012](../proposals/BITNET-PROP-0012-falcon3-family-supported-models.md)
Linked specs: n/a
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [Falcon3 family implementation plan](../../plans/falcon3-family/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: defines future gates only; no promotion
Policy impact: no policy exception

## Purpose

Mirror upstream route support as candidate information while preventing BitNet-rs from treating listed upstream support as proof authority.

## Initial Route Table

| Route | Status |
| --- | --- |
| x86 `I2_S` | `listed_supported_verify_runner` |
| x86 `TL1` | `unsupported_upstream` |
| x86 `TL2` | `listed_supported_verify_runner` |
| ARM `I2_S` | `listed_supported_verify_runner` |
| ARM `TL1` | `listed_supported_verify_runner` |
| ARM `TL2` | `unsupported_upstream` |

`listed_supported_verify_runner` means BitNet-rs may plan artifact, conversion, reference-runner, layout, and backend work. It is not answer readiness and not backend readiness.

## Promotion Requirements

- Artifact inventory receipt for the exact model size and route.
- Tokenizer/prompt authority receipt.
- Reference-runner receipt that loads the exact artifact or converted output.
- Layout proof for the exact route.
- CPU scalar oracle before accelerator proof.
- Backend receipt with requested backend, selected backend, selected route, selected kernel, fallback=false, generated IDs, decoded text, and claim boundary.

## Hard Rules

```text
I2_S routes can use QK256 kernels only after layout proof.
TL1/TL2 routes require separate layout specs and scalar oracles.
Unsupported upstream routes may produce diagnostic rejection receipts only.
x86 TL2 remains unpromoted until runner/conversion proof exists.
ARM TL1 remains unpromoted until runner/conversion proof exists.
```
