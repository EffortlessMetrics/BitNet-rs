<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple Silicon MacBook cross-reference Campaign Status

- Campaign: `apple-silicon-macbook`
- State: `active`
- Objective: Use a MacBook as the Apple Silicon cross-reference and larger-artifact validation lane for dense SLM behavior and Apple BitNet candidate sweeps, while keeping M4 Mac mini product/performance claims and BitNet artifact claims separate.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| MB-AS-001 | merged | #4184 | `codex/apple-silicon-macbook/MB-AS-001-machine-profile` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a MacBook Apple Silicon machine/storage/profile receipt contract that records chip, memory, macOS, free disk, cache root, thermal/mobile context when available, and CPU/NEON, Metal, and MPSGraph visibility without running model inference. |
| MB-AS-002 | proposed | TBD | `codex/apple-silicon-macbook/MB-AS-002-qwen-baseline` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Mirror the supported dense Qwen2.5 SLM Mac baseline on the live M3 MacBook Air with the same model hash, tokenizer metadata, quality corpus, deterministic greedy settings, the apple-m3-air-cpu-neon receipt label or documented successor, backend/fallback receipt schema, thermal/power context, receipts-check output, and MacBook-specific timing context. |
| MB-AS-003 | merged | #4190 | `codex/apple-silicon-macbook/MB-AS-003-bitnet-candidate-matrix` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a MacBook-oriented Apple BitNet candidate matrix covering official Microsoft 2B I2_S, 1bitLLM 0.7B, 1bitLLM 3B TL1/TL2 diagnostic routes, and Falcon-E candidates with storage estimates, supported kernel routes, tokenizer authority requirements, reference-runner commands, and cleanup rules. |
| MB-AS-004 | proposed | TBD | `codex/apple-silicon-macbook/MB-AS-004-microsoft-2b-i2s` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Validate the official Microsoft BitNet b1.58 2B / 2B4T I2_S GGUF on MacBook under the required external tokenizer pre-tokenizer authority, recording source, SHA256, tokenizer authority, reference-runner prompt outputs, bad/no-authority rejection evidence, and storage cleanup status. |
| MB-AS-005 | proposed | TBD | `codex/apple-silicon-macbook/MB-AS-005-1bitllm-07b` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Evaluate 1bitLLM/bitnet_b1_58-large as the smaller Apple BitNet candidate, recording source, size, SHA256, I2_S/TL1 route evidence, tokenizer authority, reference-runner prompt outputs, acceptance or rejection, and storage cleanup status. |
| MB-AS-006 | proposed | TBD | `codex/apple-silicon-macbook/MB-AS-006-3b-tl-diagnostic` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Evaluate 1bitLLM/bitnet_b1_58-3B only on supported TL1/TL2 diagnostic routes, documenting why 3B I2_S is not an Apple proof target unless compatibility evidence changes. |
| MB-AS-007 | in_progress | TBD | `codex/apple-silicon-macbook/m3-air-roadmap` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a live M3 MacBook Air lane roadmap that records the current host class, storage budget, roadmap lanes, first-run checklist, thermal/power policy, explicit apple-m3-air-cpu-neon receipt label, near-term PR stack, receipt locations, dense SLM mirror sequence, BitNet artifact sweep sequence, decision gates, open engineering questions, M4 strict-proof handoff boundary, and explicit non-claims. |
| MB-AS-008 | proposed | TBD | `codex/apple-silicon-macbook/MB-AS-008-m3-air-machine-profile` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Capture a real M3 MacBook Air machine/profile receipt with chip, model identifier, memory, macOS, free disk, cache root, power/thermal context when available, Low Power Mode when available, CPU/NEON visibility, Metal visibility, MPSGraph visibility when available, requested_backend=none, selected_backend=none, and inference_run=false before model validation starts. |

## Hard Constraints

- Do not reopen the completed apple-m4 proof, operational, SLM answer, productization, performance, hardening, or regression campaigns.
- Do not claim BitNet local-answer quality from dense Qwen SLM evidence.
- Do not claim full apple-m4-metal inference, Neural Engine execution, MPSGraph model inference, QK256 support, or broad Apple Silicon performance.
- Do not claim a BitNet candidate is answer-ready unless the reference runner produces coherent short answers with recorded tokenizer authority.
- Do not claim MacBook evidence validates the M4 Mac mini performance envelope without matching model, tokenizer, backend, fallback, profile, and receipt context.
- Never commit model binaries.
