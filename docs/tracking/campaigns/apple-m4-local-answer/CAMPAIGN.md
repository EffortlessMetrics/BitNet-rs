# Apple M4 Local Answer Campaign

Campaign ID: `apple-m4-local-answer`

Status: active

## Objective

Make Apple Silicon useful for a local Mac user: install or run BitNet-rs, point it at a supported BitNet model, ask a normal question, get coherent generated text back, and receive truthful routing/proof artifacts.

## Why This Exists

The `apple-m4` proof campaign and `apple-m4-operational` campaign are complete. They established hardware proof, CPU/NEON strict proof, Metal and MPSGraph proof lanes, operator validation, receipt checking, runbooks, CLI examples, benchmark profile validation, and the next-frontier decision.

This campaign starts the selected next frontier: CPU/NEON local-answer usability first. Metal subgraph expansion comes after the CPU/NEON answer path is quality-gated, and Apple QK256 investigation remains last.

The model-artifact blocker discovered here is shared, not Apple-only.
`MODEL-ARTIFACT-007` now records the official Microsoft I2_S GGUF as
`answer_ready` for backend gates when paired with external
`tokenizer.ggml.pre=llama-bpe` authority and the `bitnetcpp-answer` prompt
envelope. A local Apple CPU/NEON release receipt now records the strict shared
answer corpus passing with fallback disabled and explicit artifact/tokenizer
authority; landing that evidence is still separate from enabling BitNet through
`bitnet mac ask/chat`.

## End State

- `apple-m4-cpu-neon` can run a supported real GGUF with real tokenizer and strict loader for multi-token local answers.
- Short prompt suites produce non-empty, valid UTF-8, non-degenerate generated text under deterministic greedy settings.
- Greedy output is stable for the same model, prompt, tokenizer, backend, and runtime settings.
- Local-answer receipts record requested backend, selected backend, runtime API, model, tokenizer, kernel family, execution phase, fallback status, generated text, token counts, and timing.
- Strict failure modes are predictable when model files, tokenizer authority, backend identity, or unsupported Apple lanes are wrong.
- Any future Metal contribution is phase-level and receipt-backed; CPU fallback is visible and never counted as Metal inference.

## Hard Constraints

- Do not reopen the completed `apple-m4` or `apple-m4-operational` campaigns.
- Start with `apple-m4-cpu-neon` as the reliable local-answer path.
- Do not claim full `apple-m4-metal` model inference unless a strict real-model receipt proves it.
- Do not claim Neural Engine execution from MPSGraph.
- Do not claim QK256 on Apple Silicon.
- Do not hide CPU fallback or treat fallback as acceleration.
- Do not add broad benchmark or performance claims before quality and receipt gates exist.

## Backend Wording

| Label | Meaning |
|---|---|
| `apple-m4-cpu-neon` | Reliable local-answer path for strict BitNet CPU/NEON proof on Apple Silicon. |
| `apple-m4-metal` | Native Metal phase/subgraph proof only where receipt-backed; not full model inference until proven. |
| `apple-m4-mpsgraph` | Graph/reference evidence only; not native Metal or Neural Engine proof. |

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-QA-ROOT-001 | merged | Evidence shows the current local GGUF also garbles under the reference runner; model artifact validation is now the blocker. |
| M4-QA-MODEL-001 | merged | Historical Apple-specific evidence rejected the default/missing-pretokenizer prompt path. |
| M4-QA-MODEL-002 | merged | Satisfied by shared `MODEL-ARTIFACT-007` answer-ready authority for backend gates. |
| M4-QA-001 | in_progress | Release-built Apple M4 CPU/NEON full shared BitNet answer corpus passed with receipt-quality fields reviewed; closeout still needs campaign checks and PR landing. |
| M4-QA-002 | proposed | Local repeat-run parity evidence exists; item remains stack-blocked until M4-QA-001 lands. |
| M4-QA-003 | proposed | Local receipt-quality checks now require generated text, token counts and ID consistency, tokenizer pretokenizer authority, model source/SHA, backend routing/fallback status, and timing fields; item remains stack-blocked until M4-QA-001 lands. |
| M4-QA-004 | proposed | Local preflight coverage rejects missing model/tokenizer authority before hidden fallback; unsupported Apple Metal/MPSGraph answer-corpus lanes fail closed, and receipt checks reject speedup/full-inference acceleration claims. Item remains stack-blocked until M4-QA-001 lands. |
| M4-QA-005 | proposed | Local decision recorded: the first eligible Metal route is a prefill projection fixture after CPU/NEON proof, greedy determinism, and receipt-quality gates land; current user-facing local-answer path remains CPU/NEON only. |

## Review Policy

Each PR owns one work item. `stackable = false` means dependent work waits until the current item lands; it does not mean Codex should stop before merge. Keep CPU/NEON, Metal, MPSGraph, and Neural Engine evidence separate in every review.
