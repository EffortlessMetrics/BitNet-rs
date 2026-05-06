# Apple M4 Campaign

Campaign ID: `apple-m4`

Status: active

## Objective

Make Apple Silicon a receipt-backed BitNet target by moving in order from machine profile to backend identity, Metal probe, tiny compute smoke, CPU/Metal parity, MPSGraph reference smoke, receipts, and benchmarks.

## End State

- M4 machine profile and probe bundle exist.
- Apple backend identity preserves Metal, MPSGraph, and CPU/NEON separately.
- Metal probe and tiny smoke execute with `fallback_used=false`.
- CPU/Metal parity exists for one kernel or subgraph.
- MPSGraph graph smoke is recorded as reference evidence, not native Metal proof.
- Receipts record requested backend, selected backend, runtime API, resolved device, fallback status, model, quantization, kernel family, and execution phase before BitNet claims.

## Hard Constraints

- Do not touch QK256 during identity, probe, or smoke setup.
- Do not touch server inference.
- Do not add dependencies without an item that explicitly allows it.
- Do not claim Neural Engine execution from MPSGraph.
- Do not claim Metal inference from a Metal device probe.
- Do not benchmark before parity exists.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-001 | merged | Apple M4 lane scaffold merged in #3625; docs/contracts only. |
| M4-002 | merged | Machine profile and probe bundle merged in #3627. |
| M4-003 | merged | Apple Metal, MPSGraph, and CPU/NEON backend identity merged in #3652. |
| M4-004 | ready | Add Metal device probe. |
| M4-005 | proposed | Run tiny Metal compute smoke. |
| M4-006 | proposed | Add CPU/Metal parity. |
| M4-007 | proposed | Run MPSGraph tiny graph smoke. |
| M4-008 | proposed | Record Apple backend identity in receipts. |
| M4-009 | proposed | Add benchmark baseline after parity and receipt identity. |

## Review Policy

Each PR owns one work item. Runtime work is non-stackable unless the campaign owner explicitly marks it stackable. Keep Metal, MPSGraph, CPU/NEON, and Neural Engine evidence separate in every review.
