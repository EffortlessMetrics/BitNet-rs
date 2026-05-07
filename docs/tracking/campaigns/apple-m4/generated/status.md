<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 Mac mini validation Campaign Status

- Campaign: `apple-m4`
- State: `complete`
- Objective: Make Apple Silicon a receipt-backed BitNet target through Metal, MPSGraph, and CPU/NEON proof lanes without conflating detection, execution, parity, inference, or performance.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-001 | merged | #3625 | `codex/hardware-scaffold/M4-001-apple-lane` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add the Apple M4 Mac mini lane scaffold without runtime execution claims. |
| M4-002 | merged | #3627 | `codex/apple-m4/M4-002-profile-probe-bundle` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add the Apple M4 Mac mini machine profile and probe bundle without runtime code, kernels, QK256, server inference, dependencies, or bulky artifacts. |
| M4-003 | merged | #3652 | `codex/apple-m4/M4-003-backend-identity` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Preserve Apple M4 Metal, MPSGraph, and CPU/NEON backend identity without adding runtime execution or kernels. |
| M4-004 | merged | #3692 | `codex/apple-m4/M4-004-metal-probe` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add Apple M4 Metal device probe without claiming Metal execution or BitNet inference. |
| M4-005 | merged | #3699 | `codex/apple-m4/M4-005-metal-smoke` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run a tiny native Metal compute smoke on M4 with fallback_used=false. |
| M4-006 | merged | #3709 | `codex/apple-m4/M4-006-metal-cpu-parity` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Compare one M4 Metal kernel or subgraph output against Apple CPU/NEON without claiming full inference. |
| M4-007 | merged | #3719 | `codex/apple-m4/M4-007-mpsgraph-smoke` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run a tiny MPSGraph graph smoke as reference-lane evidence without claiming native Metal or Neural Engine proof. |
| M4-008 | merged | #3721 | `codex/apple-m4/M4-008-backend-receipts` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Record Apple requested backend, selected backend, runtime API, resolved device identity, fallback status, and artifact path in receipts. |
| M4-009 | merged | #3732 | `codex/apple-m4/M4-009-benchmark-baseline` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Compare Apple CPU/NEON and M4 Metal for a validated kernel or subgraph after parity exists. |
| M4-010 | merged | #3746 | `codex/apple-m4/M4-010-cpu-neon-bitnet-reference` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Prove the Apple CPU/NEON BitNet reference path before native Metal BitNet kernels. |
| M4-011 | merged | #3769 | `codex/apple-m4/M4-011-metal-i2s-smoke-parity` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run an I2_S-adjacent native Metal smoke or parity target against Apple CPU/NEON without claiming full inference. |
| M4-012 | merged | #3775 | `codex/apple-m4/M4-012-tl1-arm-investigation` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Investigate TL1 as an Apple CPU/NEON-oriented BitNet path and document any Metal conversion boundaries honestly. |
| M4-013 | merged | #3783 | `codex/apple-m4/M4-013-metal-prefill-decode` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Move from isolated Apple Metal kernels toward a named BitNet inference phase with CPU reference and explicit fallback status. |
| M4-014 | merged | #3789 | `codex/apple-m4/M4-014-strict-bitnet-proof` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run strict real GGUF, real tokenizer, selected Apple backend, fallback_used=false, deterministic prompt, and receipt emission. |
| M4-015 | merged | #3804 | `codex/apple-m4/M4-015-steady-decode-profile` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add steady decode and prefill timing profiles for the selected Apple backend with strict model, tokenizer, phase, fallback, and machine context recorded. |
| M4-016 | merged | #3811 | `codex/apple-m4/M4-016-hot-loop-allocation-audit` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Measure or bound hot-loop allocations for the Apple BitNet path and distinguish compute timing from allocation overhead. |
| M4-017 | merged | #3818 | `codex/apple-m4/M4-017-metal-kernel-family-expansion` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Expand Apple Metal BitNet kernel or subgraph coverage only where CPU reference parity, fallback=false receipts, phase labels, and claim boundaries are present. |
| M4-018 | merged | #3826 | `codex/apple-m4/M4-018-cli-package-polish` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Polish Apple backend CLI and package surfaces so backend labels, strict-mode errors, artifact paths, and non-M4 failure modes are clear. |

## Hard Constraints

- Do not touch QK256 before a BitNet-specific Apple item explicitly allows it.
- Do not touch server inference.
- Do not claim Neural Engine execution from MPSGraph.
- Do not claim Metal execution from a Metal probe.
- Do not benchmark before parity exists.
