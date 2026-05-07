<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 local answer usability Campaign Status

- Campaign: `apple-m4-local-answer`
- State: `active`
- Objective: Make Apple Silicon useful for a local Mac user by turning the completed proof and operational lanes into prompt-in, intelligible-answer-out behavior with truthful hardware routing and receipts.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-QA-001 | blocked | #3904 | `codex/apple-m4-local-answer/M4-QA-001-output-smoke` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a multi-prompt Apple M4 CPU/NEON local-answer smoke suite that runs real GGUF and tokenizer paths under strict mode, requires generated_tokens >= 16, valid UTF-8, non-empty output, non-degenerate token variation, and a receipt with explicit fallback status. |
| M4-QA-ROOT-001 | merged | #3908 | `codex/apple-m4-local-answer/M4-QA-ROOT-001-bitnetcpp-parity` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Compare the same real GGUF, tokenizer, prompt template, prompt, and greedy settings against bitnet.cpp/reference behavior; either produce token/logit parity evidence for the first divergence and fix the Rust path, or prove the local GGUF artifact itself also garbles under the reference implementation. |
| M4-QA-MODEL-001 | merged | #3913 | `codex/apple-m4-local-answer/M4-QA-MODEL-001-artifact-validation` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Validate or replace the supported local-answer model artifact so a reference runner produces coherent short answers for the campaign prompt suite before M4-QA-001 is allowed to claim Apple M4 CPU/NEON local-answer smoke coverage. |
| M4-QA-MODEL-002 | ready | TBD | `codex/apple-m4-local-answer/M4-QA-MODEL-002-known-good-artifact` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Acquire or regenerate a supported local-answer GGUF/tokenizer artifact that passes the campaign prompt suite under a reference runner, records exact SHA256 and tokenizer metadata, and is allowed to unblock M4-QA-001. |
| M4-QA-002 | proposed | TBD | `codex/apple-m4-local-answer/M4-QA-002-greedy-determinism` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add deterministic greedy local-answer checks so the same model, prompt, tokenizer, backend, and runtime settings produce stable token IDs and receipt identity. |
| M4-QA-003 | proposed | TBD | `codex/apple-m4-local-answer/M4-QA-003-receipt-quality` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Harden local-answer receipts so generated text, token counts, tokenizer authority, model identity, backend routing, fallback status, and timing fields are present and checked. |
| M4-QA-004 | proposed | TBD | `codex/apple-m4-local-answer/M4-QA-004-strict-failure-modes` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add strict local-answer failure-mode coverage for missing model files, tokenizer authority failures, unsupported Apple backend selections, and attempts to count fallback as acceleration. |
| M4-QA-005 | proposed | TBD | `codex/apple-m4-local-answer/M4-QA-005-metal-phase-routing` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Decide and document the first safe route for one receipt-backed Apple Metal phase to participate in real generation while preserving CPU fallback visibility and greedy output comparison. |

## Hard Constraints

- Do not reopen the completed apple-m4 or apple-m4-operational campaigns.
- Start with apple-m4-cpu-neon as the reliable local-answer path.
- Do not claim full apple-m4-metal model inference unless a strict real-model receipt proves it.
- Do not claim Neural Engine execution from MPSGraph.
- Do not claim QK256 on Apple Silicon.
- Do not hide CPU fallback or treat fallback as acceleration.
- Do not add broad benchmark or performance claims before quality and receipt gates exist.
