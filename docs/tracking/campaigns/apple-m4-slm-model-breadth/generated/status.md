<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 SLM model breadth Campaign Status

- Campaign: `apple-m4-slm-model-breadth`
- State: `complete`
- Objective: Add more storage-conscious dense instruct model families to the Apple M4 Mac mini path through exact artifact selection, reference output sanity, Rust M4 quality gates, tokenizer authority, cache metadata, receipt validation, and deterministic checks without weakening the completed Qwen appliance baseline.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-MODEL-001 | merged | #4339 | `codex/apple-m4-slm-model-breadth/M4-MODEL-001-candidate-selection` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Select the next exact dense instruct GGUF candidate or candidates for M4 evaluation, recording source, license notes, revision, file, expected size, tokenizer expectations, prompt template, storage budget, and rejection criteria without downloading or accepting a model. |
| M4-MODEL-002 | merged | #4343 | `codex/apple-m4-slm-model-breadth/M4-MODEL-002-reference-sanity` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run reference output sanity for the selected candidate, recording exact artifact source, revision, file, size, SHA256, tokenizer authority, prompt template, reference command, prompt outputs, and reject/accept evidence. |
| M4-MODEL-003 | merged | #4355 | `codex/apple-m4-slm-model-breadth-M4-MODEL-003-rust-m4-quality` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run the candidate through Rust M4 apple-m4-cpu-neon quality gates with valid UTF-8, non-empty output, non-degenerate output, backend/fallback receipts, generated token IDs, timing, and deterministic behavior where required. |
| M4-MODEL-009 | merged | #4404 | `codex/apple-m4-slm-model-breadth/M4-MODEL-009-qwen15-candidate` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Select a larger but still storage-conscious Qwen2.5 dense instruct GGUF candidate for M4 evaluation, recording source, license notes, revision, file, expected size, tokenizer expectations, prompt template, storage budget, and rejection criteria without downloading or accepting a model. |
| M4-MODEL-010 | merged | #4408 | `codex/apple-m4-slm-model-breadth/M4-MODEL-010-qwen15-reference` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run reference output sanity for the selected larger Qwen2.5 candidate, recording exact artifact source, revision, file, size, SHA256, tokenizer authority, prompt template, reference command, prompt outputs, and reject/accept evidence. |
| M4-MODEL-011 | merged | #4411 | `codex/apple-m4-slm-model-breadth/M4-MODEL-011-qwen15-rust-m4-quality` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run the larger Qwen2.5 candidate through Rust M4 apple-m4-cpu-neon quality gates with valid UTF-8, non-empty output, non-degenerate output, backend/fallback receipts, generated token IDs, timing, and deterministic behavior where required. |
| M4-MODEL-004 | merged | #4414 | `codex/apple-m4-slm-model-breadth/M4-MODEL-004-cache-registration` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Register the accepted candidate in model cache metadata and Mac model selection only after quality passes, with verify/fetch/list behavior, receipt validation, and no default-model change unless explicitly accepted. |
| M4-MODEL-005 | merged | #4416 | `codex/apple-m4-slm-model-breadth/M4-MODEL-005-docs-envelope` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Update the dense model support matrix and M4 user expectation envelope with the accepted model, receipt evidence, cache size, supported/candidate/rejected state, and claim boundaries. |
| M4-MODEL-006 | merged | #4399 | `codex/apple-m4-slm-model-breadth/M4-MODEL-006-gemma-candidate` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Select the next exact dense instruct GGUF candidate after the failed Qwen3/SmolLM2 round, recording source, license notes, revision, file, expected size, tokenizer expectations, prompt template, storage budget, and rejection criteria without downloading or accepting a model. |
| M4-MODEL-007 | merged | #4402 | `codex/apple-m4-slm-model-breadth/M4-MODEL-007-gemma-reference` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run reference output sanity for the newly selected candidate, recording exact artifact source, revision, file, size, SHA256, tokenizer authority, prompt template, reference command, prompt outputs, and reject/accept evidence. |
| M4-MODEL-008 | blocked | TBD | `codex/apple-m4-slm-model-breadth/M4-MODEL-008-gemma-rust-m4-quality` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run the selected candidate through Rust M4 apple-m4-cpu-neon quality gates with valid UTF-8, non-empty output, non-degenerate output, backend/fallback receipts, generated token IDs, timing, and deterministic behavior where required. |

## Hard Constraints

- This is an M4 Mac mini dense SLM campaign.
- Do not execute MacBook artifact sweeps or MacBook receipts here.
- Do not reopen completed Apple M4 dense SLM baseline, performance, excellence, continuity, or regression campaigns.
- Never commit model binaries.
- Do not accept unpinned or random community GGUFs.
- Do not mark a model supported just because it loads.
- Do not claim BitNet local-answer quality from dense SLM evidence.
- Do not claim full apple-m4-metal inference, Neural Engine execution, MPSGraph model inference, QK256 support, or broad M4 performance.
