<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M3 MacBook Air Campaign Status

- Campaign: `apple-m3-macbook-air`
- State: `active`
- Objective: Turn the available M3 MacBook Air into a disciplined Apple Silicon lane for machine-profile evidence, dense SLM cross-checks, large BitNet artifact qualification, and M4 strict-proof handoff planning without converting MacBook receipts into M4 Mac mini performance or BitNet local-answer claims.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M3MBA-001 | in_progress | TBD | `codex/apple-m3-macbook-air/roadmap-campaign` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add the M3 MacBook Air campaign control plane, roadmap linkage, generated status, and first lifecycle event without running models or changing runtime behavior. |
| M3MBA-002 | proposed | TBD | `codex/apple-m3-macbook-air/M3MBA-002-machine-profile` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Commit a real M3 MacBook Air machine-profile receipt with model identifier, chip, core split, memory, macOS version, free disk, cache root, power source, thermal state when available, CPU/NEON visibility, Metal visibility, MPSGraph visibility when available, and inference_run=false. |
| M3MBA-003 | proposed | TBD | `codex/apple-m3-macbook-air/M3MBA-003-receipt-label` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add or confirm the smallest receipt validation path for apple-m3-air-cpu-neon, preserving existing apple-m4-cpu-neon checks and making MacBook timing impossible to label as M4 evidence. |
| M3MBA-004 | proposed | TBD | `codex/apple-m3-macbook-air/M3MBA-004-dense-qwen-mirror` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Mirror the known dense Qwen2.5 0.5B Mac path on M3 Air with smoke and, if smoke passes, operator receipts that record model hash, tokenizer metadata, deterministic settings, backend label, fallback status, power and thermal context, storage before/after, receipts-check output, and dense-only claim boundary. |
| M3MBA-005 | proposed | TBD | `codex/apple-m3-macbook-air/M3MBA-005-microsoft-2b-i2s` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Qualify the official Microsoft BitNet 2B I2_S GGUF on M3 Air with source revision, filename, size, SHA256, tokenizer/pre-tokenizer authority, reference-runner command, prompt outputs, bad/no-authority rejection evidence, and cleanup status. |
| M3MBA-006 | proposed | TBD | `codex/apple-m3-macbook-air/M3MBA-006-1bitllm-07b` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Evaluate 1bitLLM/bitnet_b1_58-large as the smaller M3 Air control candidate only after the Microsoft 2B path records acceptance, rejection, or a clear blocker. |
| M3MBA-007 | proposed | TBD | `codex/apple-m3-macbook-air/M3MBA-007-3b-diagnostic` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Evaluate the 3B candidate only on supported TL1/TL2 diagnostic routes, recording why I2_S remains unsupported and why the result is diagnostic rather than proof. |
| M3MBA-008 | proposed | TBD | `codex/apple-m3-macbook-air/M3MBA-008-m4-proof-handoff` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Promote accepted M3 Air BitNet artifact evidence into a separate M4 Mac mini strict Apple CPU/NEON proof item, preserving source, hash, tokenizer authority, route, and claim boundary without running or claiming the proof in this handoff item. |

## Hard Constraints

- This is the Apple M3 MacBook Air lane, not the M4 Mac mini product, performance, or strict-proof lane.
- Do not claim BitNet local-answer quality from dense Qwen SLM receipts.
- Do not claim M4 Mac mini performance, broad Apple Silicon performance, QK256 support, full Apple Metal inference, Neural Engine execution, or MPSGraph model inference from this lane.
- Do not weaken existing M4 receipt checks to make M3 receipts fit; add the smallest MacBook-specific label or validation path instead.
- Do not add live model downloads, large artifact sweeps, or hardware timing runs to generic required CI.
- Never commit model binaries.
