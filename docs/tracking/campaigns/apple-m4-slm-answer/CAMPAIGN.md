# Apple M4 SLM Answer Campaign

Campaign ID: `apple-m4-slm-answer`

Status: active

## Objective

Make the M4 Mac mini useful for prompt-in, intelligible-answer-out local runs with a small dense SLM while BitNet local-answer artifact quality remains blocked.

## Why This Exists

The completed `apple-m4` and `apple-m4-operational` campaigns proved Apple hardware, backend identity, Metal/MPSGraph proof lanes, CPU/NEON strict BitNet proof, receipts, validation, and operator surfaces.

The `apple-m4-local-answer` campaign remains active but blocked on BitNet model artifact quality. The tested BitNet GGUF artifacts and regenerated Bessie Q8_0 attempt do not produce coherent reference output, so that campaign must not weaken its quality gate or claim coherent BitNet local answers.

This campaign is the practical Mac user-facing path: use a small known-good dense instruct model first, with the same routing and receipt discipline, then make the run warm and usable.

## End State

- A sub-1 GiB dense instruct GGUF is reference-validated for short local-answer prompts.
- The selected artifact records source, SHA256, size, GGUF architecture, quantization, tokenizer metadata, pre-tokenizer authority, and prompt template.
- `apple-m4-cpu-neon` can run the validated SLM through the Rust CLI with strict loader/tokenizer behavior and explicit fallback status.
- Short prompt suites produce valid UTF-8, non-empty, non-degenerate generated text under deterministic greedy settings.
- Warm-session operation loads model and tokenizer once and records timing split by load, tokenize, prefill, decode, sampling, and total time.
- Any future Metal work is phase-level and receipt-backed; CPU fallback is explicit and never counted as full Metal inference.

## Hard Constraints

- Do not reopen the completed `apple-m4` or `apple-m4-operational` campaigns.
- Do not weaken the blocked BitNet `apple-m4-local-answer` gates.
- Do not claim BitNet local-answer quality from dense SLM evidence.
- Do not touch QK256, `bitnet-qk256-dispatch`, Metal kernels, MPSGraph execution, Neural Engine routing, or server inference for this campaign seed.
- Do not claim full `apple-m4-metal` inference until a strict real-model receipt proves it.
- Do not claim general performance from a tiny answer smoke or cold-start run.
- Never commit model binaries.

## Storage Policy

| Limit | Policy |
|---|---|
| Preferred artifact | `<= 500 MiB` |
| Soft cap | `<= 750 MiB` |
| Hard local cap | `<= 1 GiB` without explicit campaign update |
| Download location | `target/apple-m4-slm-answer/...` or another ignored `target/` path |
| Rejected candidates | Delete after recording source, size, SHA, tokenizer metadata, and result |
| Accepted candidates | Record source and SHA only; do not commit the binary |

The first candidate class is the official Qwen small instruct GGUF family. `Qwen2.5-0.5B-Instruct-GGUF` Q4_K_M is preferred when it passes reference quality because it is under the preferred storage budget. Qwen3 0.6B GGUF remains a fallback candidate. Random community uploads are lower priority than official artifacts with clear tokenizer metadata.

## Backend Wording

| Label | Meaning |
|---|---|
| `apple-m4-cpu-neon` | Initial reliable local-answer path for the dense SLM on Apple Silicon CPU/NEON. |
| `apple-m4-metal` | Future phase/subgraph contribution only where receipt-backed; not full model inference until proven. |
| `apple-m4-mpsgraph` | Out of scope for this campaign until separately proposed as graph/reference evidence. |

## Work Items

| Work item | Status | Notes |
|---|---|---|
| SLM-M4-001 | in_progress | Seed this campaign, storage policy, claim boundaries, and Codex goal. |
| SLM-M4-002 | ready | Validate a sub-1 GiB dense instruct GGUF under a reference runner and record exact artifact metadata. |
| SLM-M4-003 | proposed | Run the validated SLM through the Rust CLI with `apple-m4-cpu-neon` and answer receipts. |
| SLM-M4-004 | proposed | Add warm-session behavior so model/tokenizer are not reloaded per prompt. |
| SLM-M4-005 | proposed | Add deterministic quality corpus and receipts. |
| SLM-M4-006 | proposed | Measure and improve warm-answer speed without broad performance claims. |
| SLM-M4-007 | proposed | Decide first safe Metal phase contribution after CPU/NEON answers are stable. |

## Review Policy

Each PR owns one work item. SLM evidence must not be used to unblock BitNet local-answer claims. Dense SLM work may reuse loader, tokenizer, CLI, and receipt infrastructure, but must keep BitNet proof/research lanes and QK256 surfaces separate.
