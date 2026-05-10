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
| M4-MODEL-002 | merged | Ran reference output sanity and recorded exact artifact metadata. |
| M4-MODEL-003 | merged | Rust M4 quality evidence rejects SmolLM2 for this round. |
| M4-MODEL-004 | blocked | No accepted candidate is available to register. |
| M4-MODEL-005 | blocked | Update model matrix and user envelope after registration. |
| M4-MODEL-006 | in progress | Select the next exact candidate after the failed Qwen3/SmolLM2 round. |
| M4-MODEL-007 | blocked | Run reference output sanity for the newly selected candidate. |
| M4-MODEL-008 | blocked | Run Rust M4 quality gates for the newly selected candidate. |

## Current Selection

`M4-MODEL-001` selects `qwen3-0.6b-q8_0` and
`smollm2-360m-instruct-q8_0` for the next reference-output sanity step. These
are exact evaluation candidates only; no model is downloaded, accepted, or
registered by this item.

`M4-MODEL-002` rejects `qwen3-0.6b-q8_0` for this round because the available
reference runner cannot load `qwen3`, and promotes
`smollm2-360m-instruct-q8_0` as the only reference-good candidate for the Rust
M4 quality gate.

`M4-MODEL-003` records that `smollm2-360m-instruct-q8_0` does not pass the
Rust M4 quality gate for this round. The current strict Rust loader rejects the
artifact before generation, and diagnostic compatibility probes still produce
incoherent output. `M4-MODEL-004` remains blocked because there is no accepted
new model to register.

The model-breadth lane must not promote cache/model registration until a later
candidate passes both reference output sanity and Rust M4 quality gates.

`M4-MODEL-006` starts that later candidate cycle with a pinned Gemma 3 270M IT
GGUF candidate from `ggml-org/gemma-3-270m-it-GGUF`. This is selection metadata
only: no model artifact is downloaded, accepted, registered, or claimed
supported by the M4 lane.

## Claim Boundary

Dense model breadth does not prove BitNet, QK256, Neural Engine execution,
MPSGraph model inference, full Apple Metal inference, or broad M4 performance.
