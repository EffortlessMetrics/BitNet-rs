# Apple M4 Operational Campaign

Campaign ID: `apple-m4-operational`

Status: active

## Objective

Turn the completed Apple M4 proof lane into a repeatable operator workflow with one-command validation, durable receipts, clear docs, stable failure modes, and benchmark profiles.

## Why This Exists

The `apple-m4` proof campaign is complete and should remain closed. It established receipt-backed Apple CPU/NEON strict BitNet proof plus Metal and MPSGraph proof lanes, with explicit boundaries around full Metal inference, QK256, Neural Engine execution, and broad performance claims.

This campaign starts after that proof work. Its job is operational readiness: make the proof repeatable, checkable, documented, and useful for someone running the Mac lane on a real M4 machine.

## End State

- A single Apple M4 validation command emits the expected machine, probe, smoke, parity, graph, strict BitNet CPU/NEON, profile, allocation, and summary receipts.
- A receipt-bundle checker rejects missing fallback status, unsupported Apple claims, missing BitNet fields, and accidental QK256 or Neural Engine claims.
- The operator runbook documents model placement, commands, receipt paths, backend labels, failure modes, and unsupported claims.
- CLI examples make effective Apple CPU/NEON use and Metal phase proof repeatable without relying on hidden test knowledge.
- Benchmark profiles are conservative, named, and tied to receipt-backed proof boundaries.
- The next Apple implementation frontier is selected explicitly instead of reopening the completed proof campaign.

## Hard Constraints

- Do not reopen the completed `apple-m4` proof campaign.
- Do not claim full `apple-m4-metal` model inference unless a strict real-model receipt proves it.
- Do not claim Neural Engine execution from MPSGraph.
- Do not claim QK256 on Apple Silicon.
- Do not claim general M4 performance from tiny-kernel benchmarks.

## Backend Wording

Use these labels consistently:

| Label | Meaning |
|---|---|
| `apple-m4-cpu-neon` | Strict BitNet CPU/NEON proof path on Apple Silicon. |
| `apple-m4-metal` | Native Metal proof or phase path only where receipt-backed. |
| `apple-m4-mpsgraph` | Graph/reference evidence only; not native Metal kernel proof. |

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-OP-001 | merged | One-command Apple M4 validation bundle merged in #3845. |
| M4-OP-002 | merged | Hardened receipt-bundle validation merged in #3848. |
| M4-OP-003 | merged | Apple M4 operator runbook merged in #3857. |
| M4-OP-004 | pr_open | Add effective-use CLI examples and strict failure-mode docs in #3871. |
| M4-OP-005 | merged | Conservative benchmark profile names and summary artifact validation merged in #3861. |
| M4-OP-006 | proposed | Decide the next implementation frontier. |

## Review Policy

Each PR owns one work item. `stackable = false` means the dependent next item waits until the current item lands; it does not mean Codex should stop before merge. Normal work uses Codex pre-merge review, auto-merge when green, and a human gate only for blockers.
