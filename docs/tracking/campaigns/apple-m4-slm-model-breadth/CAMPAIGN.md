# Apple M4 SLM Model Breadth

Campaign ID: `apple-m4-slm-model-breadth`

Status: active

## Objective

Add more storage-conscious dense instruct model families to the Apple M4 Mac
mini path without weakening the completed Qwen appliance baseline.

## End State

- Candidate models are exact, pinned artifacts.
- Reference output sanity passes before Rust M4 work.
- Rust M4 quality, tokenizer authority, backend/fallback identity, generated
  token IDs, timing, cache metadata, and receipt validation pass before support.
- The model matrix distinguishes `default`, `supported`, `candidate`,
  `diagnostic-only`, and `rejected`.
- No model binaries are committed.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-MODEL-001 | merged | Selected the next exact dense instruct GGUF candidate set. |
| M4-MODEL-002 | in_progress | Run reference output sanity and record exact artifact metadata. |
| M4-MODEL-003 | pending | Run Rust M4 quality, receipts, and deterministic gates. |
| M4-MODEL-004 | pending | Register cache/model selection only after quality passes. |
| M4-MODEL-005 | pending | Update model matrix and user envelope. |

## Current Selection

`M4-MODEL-001` selects `qwen3-0.6b-q8_0` and
`smollm2-360m-instruct-q8_0` for the next reference-output sanity step. These
are exact evaluation candidates only; no model is downloaded, accepted, or
registered by this item.

`M4-MODEL-002` rejects `qwen3-0.6b-q8_0` for this round because the available
reference runner cannot load `qwen3`, and promotes
`smollm2-360m-instruct-q8_0` as the only reference-good candidate for the Rust
M4 quality gate.

## Claim Boundary

Dense model breadth does not prove BitNet, QK256, Neural Engine execution,
MPSGraph model inference, full Apple Metal inference, or broad M4 performance.
