# BITNET-SPEC-FALCON-E-FAMILY-ROUTE-COMPATIBILITY

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: docs/proposals/BITNET-PROP-0013-falcon-e-family-supported-models.md
Linked specs: docs/specs/BITNET-SPEC-FALCON-E-FAMILY-ARTIFACT-CONTRACT.md
Linked ADRs: n/a
Linked plan: plans/falcon-e-family/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: route registration only
Policy impact: no policy exception

## Route matrix

| Route | Initial BitNet-rs status |
|---|---|
| x86 `I2_S` | `listed_supported_verify_runner` |
| x86 `TL1` | `unsupported_upstream` |
| x86 `TL2` | `listed_supported_verify_runner` |
| ARM `I2_S` | `listed_supported_verify_runner` |
| ARM `TL1` | `listed_supported_verify_runner` |
| ARM `TL2` | `unsupported_upstream` |

`I2_S` is the first implementation route. TL routes remain registered
candidates until TL layout fixtures and scalar oracles exist.

## Required route evidence

- Upstream support row or diagnostic rejection authority.
- Artifact route metadata from GGUF or approved conversion path.
- Architecture-specific route selected by proof command.
- Kernel ID and fallback status when execution claims begin.
- Unsupported-op count and diagnostic rejection receipt for unsupported routes.

## Hard rules

```text
I2_S routes can use QK256/scaled I8S kernels only after Falcon-E layout proof.
TL1/TL2 routes require separate layout specs and scalar oracles.
Unsupported upstream routes may produce diagnostic rejection receipts only.
```
