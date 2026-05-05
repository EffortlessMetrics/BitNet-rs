# bitnet-rs Alignment Status

Updated: 2026-05-05

## Current focus

P0 truth boundary, then crate consolidation inventory.

## Active PR queue

| Item | PR | State | Notes |
|---|---:|---|---|
| TRUTH-002 | TBD | ready | Make GGUF fallback explicit |
| TRUTH-003 | #3626 | pr_open | Null-byte model path validation |
| INV-001 | TBD | ready | Crate consolidation map |

## Queue hygiene

| Cluster | Decision | Notes |
|---|---|---|
| Codecov duplicates | Deferred canonical #3620 | #3609-#3612 and #3617-#3619 are recorded as superseded by #3620; handle this in a separate CI coverage review. |
| Null-byte Sentinel duplicates | Tied to TRUTH-003 | Do not merge pre-ledger Sentinel branches; close duplicates after the canonical TRUTH-003 PR lands. |
| Accessibility Palette duplicates | Deferred | Review after truth boundary and inventory are complete; #3607 is the latest matching candidate for later review. |
| Sampling/performance Bolt duplicates | Deferred | Hold until truth boundary and inventory are complete. |

## Completed

| Item | PR | Merge SHA | Notes |
|---|---:|---|---|
| TRUTH-001 | #3621 | 10ea2b409a1d4095205722da77a600d31bb57d04 | Fence server simulated inference merged. |
| QUEUE-001 | #3623 | 678a3ba1592ab10e9a7b473db077ec93b1d867fb | Codecov duplicate cluster normalized; #3620 retained as deferred canonical candidate. |

## Blocked

| Item | Blocker | Next action |
|---|---|---|

## Superseded

| Item/PR | Superseded by | Reason |
|---|---|---|
