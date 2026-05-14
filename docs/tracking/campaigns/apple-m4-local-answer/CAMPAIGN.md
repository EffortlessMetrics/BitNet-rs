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
authority. #4637 wires that evidence into an explicit one-shot
`bitnet mac ask` route guarded by accepted GGUF/tokenizer identity and a
`supported-ask` catalog state. #4647 adds the first user-route runtime receipt,
and #4651 adds progress/status UX for slow one-shot runs. BitNet `mac chat`
and `mac serve` remain disabled.

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
| M4-QA-001 | merged | Release-built Apple M4 CPU/NEON full shared BitNet answer corpus passed with receipt-quality fields reviewed; merged in #4618. |
| M4-QA-002 | merged | Repeat-run parity evidence from #4618 shows both Apple M4 CPU/NEON runs used the same answer-ready model, tokenizer, prompt template, greedy settings, and fallback-free backend routing; all five compared cases matched generated token IDs. |
| M4-QA-003 | merged | Receipt-quality checks from #4618 require generated text, token counts and ID consistency, tokenizer pretokenizer authority, model source/SHA, backend routing/fallback status, and timing fields; the committed Apple M4 CPU/NEON BitNet answer-corpus receipt records those fields for all five passing cases. |
| M4-QA-004 | merged | Local preflight coverage rejects missing model/tokenizer authority before hidden fallback; unsupported Apple Metal/MPSGraph answer-corpus lanes fail closed, and receipt checks reject hidden fallback plus speedup/full-inference acceleration claims. |
| M4-QA-005 | merged | Local decision recorded: the first eligible Metal contribution is a prefill projection fixture with CPU-only greedy reference comparison, CPU/Metal phase parity, unchanged generated token IDs and decoded text, and explicit fallback=false phase receipts; current user-facing local-answer path remains CPU/NEON only. |
| M4-BITNET-ASK-000 | merged | #4637 added an explicit one-shot BitNet `bitnet mac ask` route gated by accepted Microsoft I2_S GGUF and external tokenizer identity, marked BitNet as `supported-ask`, and kept BitNet chat/serve disabled without claiming a fresh runtime smoke. |
| M4-BITNET-ASK-001 | merged | Local receipt `ci/hardware/apple-m4-mac-mini/2026-05-13/bitnet-mac-ask/bitnet-mac-ask-runtime-receipt.json` proves the user-facing BitNet `bitnet mac ask` route completed one strict Apple M4 CPU/NEON prompt with fallback_used=false, text `2+2 equals 4.`, generated token IDs, accepted model/tokenizer identity, and timing fields; merged in #4647. |
| M4-BITNET-ASK-002 | merged | #4651 added explicit `--progress` status milestones for slow one-shot `bitnet mac ask` runs and `--quiet` suppression for scripts while keeping generated text on stdout and BitNet chat/server/Metal claims disabled. |
| M4-BITNET-ASK-003 | merged | #4680 added durable partial-failure receipts for BitNet `bitnet mac ask` setup and generation failures, with repair guidance and unchanged chat/server/Metal claim boundaries. |
| M4-BITNET-ASK-004 | merged | #4686 prints compact operator repair guidance for BitNet `bitnet mac ask` setup/generation failures, matching the receipt guidance for accepted tokenizer, model cache, explicit GGUF verification, and unchanged chat/server/Metal boundaries. |
| M4-BITNET-ASK-005 | merged | #4688 added advisory BitNet one-shot ask readiness to `bitnet mac doctor` receipts without making dense doctor fail when optional BitNet artifacts are absent or enabling BitNet chat/server/Metal claims. |
| M4-BITNET-ASK-006 | merged | #4691 added `bitnet mac smoke --model-family bitnet` as a one-shot BitNet ask smoke with accepted model/tokenizer identity, failure receipts, and unchanged chat/server/Metal boundaries. |
| M4-BITNET-SMOKE-001 | in_progress | Add a committed runtime receipt proving `bitnet mac smoke --model-family bitnet` completes through the accepted Microsoft I2_S GGUF and accepted external tokenizer under `apple-m4-cpu-neon`, with generated text/token IDs, fallback=false, and unchanged chat/server/Metal boundaries. |

## Review Policy

Each PR owns one work item. `stackable = false` means dependent work waits until the current item lands; it does not mean Codex should stop before merge. Keep CPU/NEON, Metal, MPSGraph, and Neural Engine evidence separate in every review.
