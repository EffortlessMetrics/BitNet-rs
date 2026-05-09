# Apple M4 SLM Excellence Campaign

Campaign ID: `apple-m4-slm-excellence`

Status: active

## Objective

Turn the working Apple M4 dense SLM path into an appliance-grade local model
runner experience: native-feeling CLI, reliable health checks, lower perceived
latency, better allocation hygiene, stronger quality coverage, longer resident
session stability, local regression reporting, leading dense SLM support, and a
measured operator envelope while keeping generic CI efficient.

## Why This Exists

The completed Apple M4 dense SLM lanes made the Mac mini useful: Qwen2.5 dense
SLM answers run through Rust-native `apple-m4-cpu-neon`, model cache management
is documented, `bitnet mac ask`, `bitnet mac chat`, and `bitnet mac smoke`
exist, receipts validate backend/fallback identity, and regression guardrails
track the recorded M4 envelope.

This campaign moves from "working" to "boring local appliance." It does not
reopen the completed M4 proof, operational, productization, performance,
hardening, continuity, or dense regression campaigns.

## Model-Family Boundary

Dense Qwen SLM evidence validates Mac UX, model-cache behavior, receipts,
warm-session behavior, quality checks, and Apple CPU/NEON routing for a regular
dense SLM. It does not prove BitNet, 1-bit / 1.58-bit kernels, I2_S/TL1/TL2
layouts, QK256, Neural Engine execution, MPSGraph model inference, or full
Apple Metal inference.

BitNet remains a separate model family. The M4 BitNet proof command is prepared
but blocked until an accepted BitNet artifact exists.

## End State

- `bitnet mac doctor` gives an operator a single health verdict for the M4 dense SLM path.
- `bitnet mac chat` behaves like a quiet, streaming, resident local tool.
- Time-to-first-token and hot-loop overhead are measured and reduced without changing greedy output.
- Dense model support is explicit: default, supported, candidate, and rejected states are distinct.
- Leading dense SLM candidates can be evaluated through the same cache, tokenizer, quality, receipt, and M4 support matrix.
- A second dense model is supported only after reference and Rust M4 quality gates pass.
- Quality corpus 2.0 covers small but useful local-answer behavior without broad eval claims.
- Long resident sessions record memory and timing drift without fleet-wide performance claims.
- Local regression reports compare matching receipts against the stored M4 envelope.
- The operator expectation doc says what the M4 mini should do and what remains unsupported.

## Hard Constraints

- This is an M4 Mac mini local campaign.
- Do not execute MacBook artifact sweeps or MacBook receipts here.
- Do not reopen completed Apple M4 proof, operational, SLM answer, productization, performance, hardening, continuity, or dense regression campaigns.
- Do not weaken blocked BitNet local-answer gates.
- Do not claim BitNet local-answer quality from dense Qwen SLM evidence.
- Do not claim full `apple-m4-metal` inference, Neural Engine execution, MPSGraph model inference, QK256 support, or broad M4 performance.
- Do not touch QK256, `bitnet-qk256-dispatch`, server inference, or Metal kernels unless a later phase-scoped Metal item explicitly allows it.
- Do not add live model downloads, long resident soaks, or hardware performance runs to generic required CI; keep those as local, advisory, or scheduled Apple-hardware checks.
- Never commit model binaries.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-SLM-EX-001 | merged | Add `bitnet mac doctor` as a one-command health verdict for cache, disk, smoke, receipt, backend, fallback, and unsupported-backend behavior. |
| M4-SLM-EX-002 | merged | Polish `bitnet mac chat` interactive behavior, streaming defaults, EOF/Ctrl-C handling, and receipt options. |
| M4-SLM-EX-003 | merged | Reduce time-to-first-token while preserving greedy token IDs and quality corpus behavior. |
| M4-SLM-EX-004 | merged | Clean hot-loop allocations and document allocation budget without changing receipt schema. |
| M4-SLM-EX-005 | merged | Document dense model support matrix for default and leading candidate SLMs: default, supported, candidate, diagnostic-only, and rejected model states. |
| M4-SLM-EX-006 | pending | Add a second supported dense model only after reference and Rust M4 quality gates pass. |
| M4-SLM-EX-007 | pending | Expand to quality corpus 2.0 for small local-answer behavior. |
| M4-SLM-EX-008 | pending | Record long resident-session soak behavior for memory and timing drift. |
| M4-SLM-EX-009 | pending | Add local advisory `bitnet mac regression` comparison against matching M4 envelope receipts. |
| M4-SLM-EX-010 | pending | Publish the measured M4 mini user expectation envelope. |

## Review Policy

Each PR should own one item. Runtime PRs must preserve `apple-m4-cpu-neon`
routing, explicit fallback status, model/tokenizer identity, generated text,
token IDs, timing receipts, and dense-SLM-only claim boundaries. Performance
work must preserve greedy output and quality behavior before interpreting
timing changes.

Generic CI should stay fast: parser, snapshot, synthetic receipt, and tracker
checks belong in ordinary PR validation; live model runs, long resident soaks,
and hardware timing comparisons belong in local, advisory, or scheduled
Apple-hardware lanes.
