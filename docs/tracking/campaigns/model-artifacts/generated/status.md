<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Model artifact answer authority Campaign Status

- Campaign: `model-artifacts`
- State: `active`
- Objective: Define and maintain shared answer-artifact authority so hardware lanes cannot claim coherent local answers from structurally valid but answer-bad GGUF artifacts.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| MODEL-ARTIFACT-001 | merged | #3922 | `codex/model-artifacts/MODEL-ARTIFACT-001-shared-answer-gate-v2` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Define the shared reference-good answer artifact gate, record artifact states, and list the current Microsoft BitNet I2_S GGUF as rejected for answer-readiness claims without changing runtime behavior. |
| MODEL-ARTIFACT-002 | blocked | #3928 | `codex/model-artifacts/MODEL-ARTIFACT-002-reference-good-bitnet` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Acquire or regenerate a reference-good BitNet GGUF/tokenizer artifact that passes the deterministic answer prompt suite under a reference runner, records exact SHA256 and tokenizer/pre-tokenizer authority, and can unblock backend answer-readiness lanes. |
| MODEL-ARTIFACT-003 | merged | #3932 | `codex/model-artifacts/MODEL-ARTIFACT-003-reference-runner-compat` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Record reference-runner compatibility evidence for official-derived BitNet GGUF candidates, including the latest stock llama.cpp runner result, without claiming answer readiness or changing runtime behavior. |

## Hard Constraints

- Do not weaken CPU, CUDA, Apple, NPU, SLM, server, or quality gates.
- Do not touch kernels, transformer routing, tokenizer behavior, model loader behavior, server inference, or accelerator runtime code in tracker-only PRs.
- Do not claim a structurally valid GGUF is answer-ready without reference runner prompt-suite evidence.
- Do not turn backend execution proof into answer-readiness proof.
- Do not make speed claims from answer artifacts.
