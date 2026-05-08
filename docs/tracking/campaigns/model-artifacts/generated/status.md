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
| MODEL-ARTIFACT-004 | merged | #3939 | `codex/model-artifacts/MODEL-ARTIFACT-004-ikllama-intended-runner` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Record intended ik_llama.cpp runner evidence for official-derived BitNet GGUF candidates, including tdh111 IQ2_BN_R4 prompt-suite output and official Microsoft I2_S prompt-suite output, without promoting an answer_ready artifact or changing runtime behavior. |
| MODEL-ARTIFACT-005 | in_progress | — | `claude/untangle-authority-gates-5oPxH` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Split artifact authority into explicit machine-readable dimensions (target_alignment, runner_authority, tokenizer_authority, pretokenizer_authority, prompt_suite_result, can_unblock_official_i2s_cuda, can_unblock_alt_quant_control) across all manifest files. Produce MODEL_ARTIFACT_005_AUTHORITY_DIMENSIONS.md. Preserve claim boundary: official I2_S remains rejected; no artifact promoted to answer_ready. |

## Current Claim Boundary

`answer_ready_artifact_available = false`

Official Microsoft I2_S (`ggml-model-i2_s.gguf`) is the official CUDA target and remains `rejected_prompt_suite_failed`. It loads under ik_llama.cpp but emits non-coherent output. Pre-tokenizer authority is missing.

`tdh111` IQ2_BN_R4 passes the tiny prompt suite under its intended ik_llama.cpp runner. It is recorded as `alternate_quant_control` with `can_unblock_alt_quant_control = true`, but `can_unblock_official_i2s_cuda = false`. It does not satisfy the official I2_S CUDA answer target.

Next unblocker: MODEL-ARTIFACT-006 — tokenizer/pre-tokenizer authority audit.

## Hard Constraints

- Do not weaken CPU, CUDA, Apple, NPU, SLM, server, or quality gates.
- Do not touch kernels, transformer routing, tokenizer behavior, model loader behavior, server inference, or accelerator runtime code in tracker-only PRs.
- Do not claim a structurally valid GGUF is answer-ready without reference runner prompt-suite evidence.
- Do not turn backend execution proof into answer-readiness proof.
- Do not make speed claims from answer artifacts.
