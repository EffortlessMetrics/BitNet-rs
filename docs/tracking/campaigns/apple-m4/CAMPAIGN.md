# Apple M4 Campaign

Campaign ID: `apple-m4`

Status: complete

## Objective

Make Apple Silicon a receipt-backed BitNet target by moving in order from machine profile to backend identity, Metal probe, tiny compute smoke, CPU/Metal parity, MPSGraph reference smoke, receipts, and benchmarks.

## End State

- M4 machine profile and probe bundle exist.
- Apple backend identity preserves Metal, MPSGraph, and CPU/NEON separately.
- Metal probe and tiny smoke execute with `fallback_used=false`.
- CPU/Metal parity exists for one kernel or subgraph.
- MPSGraph graph smoke is recorded as reference evidence, not native Metal proof.
- Receipts record requested backend, selected backend, runtime API, resolved device, fallback status, model, quantization, kernel family, and execution phase before BitNet claims.
- Strict real-GGUF Apple CPU/NEON proof records real tokenizer, strict loader, selected Apple backend, fallback_used=false, deterministic decode, model identity, kernel family, and execution phase.
- Native Apple Metal I2_S parity, prefill contribution, and projection-residual subgraph receipts compare against Apple CPU/NEON without QK256, MPSGraph, Neural Engine, or full Metal inference claims.
- Apple profile and allocation receipts separate timing, prompt prefill, decode, sampling, machine context, and hot-loop allocation evidence.
- CLI and package surfaces document Apple backend labels, strict-mode errors, artifact paths, and non-M4 failure boundaries.

## Hard Constraints

- Do not touch QK256 during identity, probe, or smoke setup.
- Do not touch server inference.
- Do not add dependencies without an item that explicitly allows it.
- Do not claim Neural Engine execution from MPSGraph.
- Do not claim Metal inference from a Metal device probe.
- Do not benchmark before parity exists.

## Completion Boundary

This campaign is complete through the M4-018 operator surface. It proves Apple
CPU/NEON strict BitNet execution and native Metal I2_S fixture/subgraph proof
lanes with explicit receipts and claim boundaries. It does not claim full
`bitnet run --device apple-m4-metal` model inference, QK256 on Apple Silicon,
Neural Engine execution, or general M4 performance.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-001 | merged | Apple M4 lane scaffold merged in #3625; docs/contracts only. |
| M4-002 | merged | Machine profile and probe bundle merged in #3627. |
| M4-003 | merged | Apple Metal, MPSGraph, and CPU/NEON backend identity merged in #3652. |
| M4-004 | merged | Add Metal device probe. |
| M4-005 | merged | Tiny Metal compute smoke merged in #3699. |
| M4-006 | merged | CPU/Metal parity merged in #3709. |
| M4-007 | merged | MPSGraph tiny graph smoke merged in #3719. |
| M4-008 | merged | Apple backend receipt identity merged in #3721. |
| M4-009 | merged | Benchmark baseline for the validated tiny Metal add kernel merged in #3732. |
| M4-010 | merged | Apple CPU/NEON BitNet reference merged in #3746. |
| M4-011 | merged | Native Metal I2_S smoke/parity merged in #3769. |
| M4-012 | merged | TL1 Apple layout-boundary investigation merged in #3775. |
| M4-013 | merged | Metal prefill contribution merged in #3783. |
| M4-014 | merged | Strict real-model BitNet M4 proof receipts merged in #3789. |
| M4-015 | merged | Steady decode and prefill profile receipts merged in #3804. |
| M4-016 | merged | Hot-loop allocation audit receipts merged in #3811. |
| M4-017 | merged | Metal I2_S projection-residual subgraph parity merged in #3818. |
| M4-018 | merged | Apple backend CLI/package polish merged in #3826 and closed out in #3828. |

## Review Policy

Each PR owns one work item. Runtime work is non-stackable unless the campaign owner explicitly marks it stackable. Keep Metal, MPSGraph, CPU/NEON, and Neural Engine evidence separate in every review.
