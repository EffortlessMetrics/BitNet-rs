# BITNET-SPEC-LLAMA3-8B-158-ROUTE-COMPATIBILITY

Status: proposed
Owner: model-artifacts
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0011](../proposals/BITNET-PROP-0011-llama3-8b-158-supported-model.md)
Linked plan: [Llama3 8B 1.58 implementation plan](../../plans/llama3-8b-158/implementation-plan.md)
Support-tier impact: registered route candidates only
Policy impact: no policy exception

## Purpose

Make route support explicit while preventing upstream-listing evidence from
becoming BitNet-rs backend proof.

## Initial route table

| Route | Status |
| --- | --- |
| x86 I2_S | `listed_supported_verify_runner` |
| x86 TL1 | `unsupported_upstream` |
| x86 TL2 | `listed_supported_verify_runner` |
| ARM I2_S | `listed_supported_verify_runner` |
| ARM TL1 | `listed_supported_verify_runner` |
| ARM TL2 | `unsupported_upstream` |

## Route proof requirements

`I2_S` routes need exact converted artifact metadata, QK256 layout proof, scalar
oracle fixtures, and backend-specific receipts before support claims. `TL1` and
`TL2` routes need independent layout fixtures and scalar oracles before any TL
backend work.

## Hard rules

- `listed_supported_verify_runner` is not `supported_reference`.
- Unsupported upstream routes may produce diagnostic rejection receipts only.
- x86 `TL2` does not follow from x86 `I2_S` proof.
- ARM `TL1` does not follow from ARM `I2_S` proof.
- CUDA, Apple, and server routes require exact-route receipts.
