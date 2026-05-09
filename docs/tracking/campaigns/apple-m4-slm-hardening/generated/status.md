<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 SLM hardening Campaign Status

- Campaign: `apple-m4-slm-hardening`
- State: `active`
- Objective: Make the completed Apple M4 SLM path boring for local users through simple Mac commands, default verified model-cache behavior, clear first-run guidance, stable receipts, and conservative hardware claim boundaries.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-SLM-HARDEN-001 | merged | #4149 | `codex/apple-m4-slm-hardening/M4-SLM-HARDEN-001-positional-mac-ask` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Allow bitnet mac ask "What is 2+2?" as the shortest supported Mac local-answer command, keep bitnet mac ask --question compatible, reject ambiguous double question input, preserve default verified Qwen2.5 cache resolution, and keep device-boundary errors before cache/model work. |
| M4-SLM-HARDEN-002 | merged | #4155 | `codex/apple-m4-slm-hardening/M4-SLM-HARDEN-002-cache-repair` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Improve first-run, corrupted-cache, low-disk, offline, verify, and prune guidance for Mac SLM operators without changing model artifacts or backend claims. |
| M4-SLM-HARDEN-003 | merged | #4158 | `codex/apple-m4-slm-hardening/M4-SLM-HARDEN-003-quality-corpus` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Expand the tiny Mac SLM operator quality corpus with a few deterministic factual, short-instruction, and format-constrained prompts while preserving fast runtime and avoiding broad eval claims. |
| M4-SLM-HARDEN-004 | in_progress | TBD | `codex/apple-m4-slm-hardening/M4-SLM-HARDEN-004-regression-seed` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Define the next regression campaign and thresholds from the measured Apple M4 SLM performance envelope without turning one machine run into a broad performance guarantee. |

## Hard Constraints

- Do not reopen completed Apple M4 proof, operational, SLM answer, productization, or performance campaigns.
- Do not weaken blocked BitNet local-answer gates.
- Do not claim BitNet local-answer quality from dense SLM evidence.
- Do not claim full apple-m4-metal inference, Neural Engine execution, MPSGraph model inference, QK256 support, or broad M4 performance.
- Do not touch QK256, bitnet-qk256-dispatch, server inference, or Metal kernels.
- Never commit model binaries.
