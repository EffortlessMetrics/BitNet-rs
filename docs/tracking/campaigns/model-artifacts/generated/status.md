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
| MODEL-ARTIFACT-005 | merged | #3977 | `codex/model-artifacts/MODEL-ARTIFACT-005-authority-dimensions` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Split artifact authority into explicit target alignment, runner authority, tokenizer/pre-tokenizer authority, prompt-suite result, and per-lane unblock fields so alternate-quant control evidence cannot be confused with the official Microsoft I2_S CUDA target. |
| MODEL-ARTIFACT-006 | merged | #3979 | `codex/model-artifacts/MODEL-ARTIFACT-006-tokenizer-authority` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Audit tokenizer and pre-tokenizer authority for the official Microsoft I2_S target and tdh111 alternate-quant control, including GGUF metadata, external tokenizer assets, prompt-template authority, and whether external authority can unblock official I2_S CUDA answer-readiness. |
| MODEL-ARTIFACT-007 | merged | #3988 | `codex/model-artifacts/MODEL-ARTIFACT-007-msft-bitnetcpp-external-pretokenizer` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Record Microsoft BitNet.cpp reference-runner evidence showing the official Microsoft I2_S GGUF passes the committed deterministic answer corpus when the externally supplied Microsoft tokenizer pre-tokenizer authority is injected with tokenizer.ggml.pre=llama-bpe, without changing Rust runtime behavior or making backend answer claims. |
| BITNET-COMPAT-001 | merged | #4134 | `codex/bitnet-compat-001-model-kernel-ledger` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Record upstream model/kernel compatibility constraints so 1bitLLM/bitnet_b1_58-3B on x86 I2_S is unsupported_upstream and cannot be used as answer_ready, reference_authority, backend_parity, or speedup evidence, while diagnostic and unsupported-path receipts remain allowed. |
| BITNET-CONTRACT-001 | merged | #4196 | `codex/bitnet-contract-001-model-contracts` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a machine-readable BitNet-family model contract matrix and typed registry that separates official 2B I2_S, official 2B TL1/TL2, unsupported 3B I2_S, listed-but-unverified 3B TL routes, and alternate-quant control evidence without changing runtime behavior or claim gates. |
| BITNET-CONTRACT-002 | in_progress | TBD | `codex/bitnet-contract-002-model-verify` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Wire the official Microsoft BitNet 2B I2_S artifact into `bitnet model verify` and cache metadata so artifact verification exposes the model-contract summary, permitted claims, required receipts, and claim boundary without changing runtime inference, tokenizer, loader, transformer, QK256, CUDA, dense GGUF, server, or speed-claim behavior. |

## Hard Constraints

- Do not weaken CPU, CUDA, Apple, NPU, SLM, server, or quality gates.
- Do not touch kernels, transformer routing, tokenizer behavior, model loader behavior, server inference, or accelerator runtime code in tracker-only PRs.
- Do not claim a structurally valid GGUF is answer-ready without reference runner prompt-suite evidence.
- Do not turn backend execution proof into answer-readiness proof.
- Do not make speed claims from answer artifacts.
