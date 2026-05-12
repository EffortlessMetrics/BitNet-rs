<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple Silicon MacBook cross-reference Campaign Status

- Campaign: `apple-silicon-macbook`
- State: `active`
- Objective: Use a MacBook as the Apple Silicon cross-reference and larger-artifact validation lane for dense SLM behavior and Apple BitNet candidate sweeps, while keeping M4 Mac mini product/performance claims and BitNet artifact claims separate.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| MB-AS-001 | merged | #4184 | `codex/apple-silicon-macbook/MB-AS-001-machine-profile` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a MacBook Apple Silicon machine/storage/profile receipt contract that records chip, memory, macOS, free disk, cache root, thermal/mobile context when available, and CPU/NEON, Metal, and MPSGraph visibility without running model inference. |
| MB-AS-002 | blocked | TBD | `codex/apple-silicon-macbook/MB-AS-002-qwen-baseline` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Superseded for live M3 Air execution by M3MBA-004A and M3MBA-004B in the apple-m3-macbook-air campaign; keep this umbrella item closed to new execution PRs unless a proxy note is needed. |
| MB-AS-003 | merged | #4190 | `codex/apple-silicon-macbook/MB-AS-003-bitnet-candidate-matrix` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a MacBook-oriented Apple BitNet candidate matrix covering official Microsoft 2B I2_S, 1bitLLM 0.7B, 1bitLLM 3B TL1/TL2 diagnostic routes, and Falcon-E candidates with storage estimates, supported kernel routes, tokenizer authority requirements, reference-runner commands, and cleanup rules. |
| MB-AS-004 | blocked | TBD | `codex/apple-silicon-macbook/MB-AS-004-microsoft-2b-i2s` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Superseded for live M3 Air execution by M3MBA-005A, M3MBA-005B, and M3MBA-005C in the apple-m3-macbook-air campaign; keep this umbrella item closed to new execution PRs unless a proxy note is needed. |
| MB-AS-005 | blocked | TBD | `codex/apple-silicon-macbook/MB-AS-005-1bitllm-07b` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Superseded for live M3 Air execution by M3MBA-006 in the apple-m3-macbook-air campaign; keep this umbrella item closed to new execution PRs unless a proxy note is needed. |
| MB-AS-006 | blocked | TBD | `codex/apple-silicon-macbook/MB-AS-006-3b-tl-diagnostic` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Superseded for live M3 Air execution by M3MBA-007 in the apple-m3-macbook-air campaign; keep this umbrella item closed to new execution PRs unless a proxy note is needed. |
| MB-AS-007 | merged | #4511 | `codex/apple-silicon-macbook/m3-air-roadmap` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a live M3 MacBook Air lane roadmap that records the current host class, storage budget, roadmap lanes, milestone gates, first-run checklist, thermal/power policy, measurement plan, artifact ledger, explicit apple-m3-air-cpu-neon receipt label, near-term PR stack, receipt locations, dense SLM mirror sequence, BitNet artifact sweep sequence, decision gates, open engineering questions, M4 strict-proof handoff boundary, review checklist, and explicit non-claims. |
| MB-AS-008 | blocked | TBD | `codex/apple-silicon-macbook/MB-AS-008-m3-air-machine-profile` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Superseded for live M3 Air execution by M3MBA-002 in the apple-m3-macbook-air campaign; keep this umbrella item closed to new execution PRs unless a proxy note is needed. |

## Hard Constraints

- Do not reopen the completed apple-m4 proof, operational, SLM answer, productization, performance, hardening, or regression campaigns.
- Do not claim BitNet local-answer quality from dense Qwen SLM evidence.
- Do not claim full apple-m4-metal inference, Neural Engine execution, MPSGraph model inference, QK256 support, or broad Apple Silicon performance.
- Do not claim a BitNet candidate is answer-ready unless the reference runner produces coherent short answers with recorded tokenizer authority.
- Do not claim MacBook evidence validates the M4 Mac mini performance envelope without matching model, tokenizer, backend, fallback, profile, and receipt context.
- Never commit model binaries.
