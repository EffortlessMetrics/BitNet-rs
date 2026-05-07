<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 SLM local answer usability Campaign Status

- Campaign: `apple-m4-slm-answer`
- State: `active`
- Objective: Make the M4 Mac mini produce prompt-in, intelligible-answer-out local runs with a small storage-conscious dense SLM while preserving receipt-backed hardware routing and keeping blocked BitNet local-answer gates honest.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| SLM-M4-001 | merged | #3925 | `codex/apple-m4-slm-answer/SLM-M4-001-seed-campaign` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Seed the Apple M4 SLM answer campaign with objective, end state, storage policy, hard claim boundaries, campaign-local Codex goal, and an ordered work queue whose first ready item validates a small dense instruct GGUF under a reference runner. |
| SLM-M4-002 | merged | #3930 | `codex/apple-m4-slm-answer/SLM-M4-002-validate-artifact` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Validate a sub-1 GiB dense instruct GGUF under a reference runner against the M4 SLM prompt suite, record source, SHA256, size, GGUF architecture, quantization, tokenizer metadata, pre-tokenizer authority, prompt template, and reference output, and reject candidates that fail quality. |
| SLM-M4-003 | pr_open | #3937 | `codex/apple-m4-slm-answer/SLM-M4-003-rust-cli-answer` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run the validated dense SLM through the Rust CLI with apple-m4-cpu-neon, strict loader/tokenizer behavior, explicit fallback status, coherent answer text, generated token IDs, and timing in receipts. |
| SLM-M4-004 | proposed | TBD | `codex/apple-m4-slm-answer/SLM-M4-004-warm-session` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add warm-session behavior so the validated model and tokenizer are loaded once, multiple prompts can run in one process, per-prompt receipts are emitted, and model_load, tokenize, prefill, decode, sampling, and total timing are separated. |
| SLM-M4-005 | proposed | TBD | `codex/apple-m4-slm-answer/SLM-M4-005-quality-determinism` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a deterministic Apple M4 SLM quality corpus so temperature=0 runs produce stable token IDs and receipts record valid UTF-8, non-empty, non-degenerate answers, model/tokenizer identity, backend routing, fallback status, and timing. |
| SLM-M4-006 | proposed | TBD | `codex/apple-m4-slm-answer/SLM-M4-006-warm-speed` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Measure and improve warm-answer speed by avoiding repeated model/tokenizer loads, reusing buffers and sampling/logits scratch where safe, separating prefill/decode timing, and recording warm tokens/sec without broad performance claims. |
| SLM-M4-007 | proposed | TBD | `codex/apple-m4-slm-answer/SLM-M4-007-metal-phase-decision` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Decide the first safe Apple Metal phase contribution for the validated SLM only after CPU/NEON answers are stable; require CPU-only vs CPU+Metal greedy comparison, Metal phase fallback=false, and explicit CPU fallback for the rest. |

## Hard Constraints

- Do not reopen the completed apple-m4 or apple-m4-operational campaigns.
- Do not weaken the blocked BitNet apple-m4-local-answer gates.
- Do not claim BitNet local-answer quality from dense SLM evidence.
- Do not touch QK256, bitnet-qk256-dispatch, Metal kernels, MPSGraph execution, Neural Engine routing, or server inference for the campaign seed.
- Do not claim full apple-m4-metal inference until a strict real-model receipt proves it.
- Do not claim general performance from a tiny answer smoke or cold-start run.
- Never commit model binaries.
