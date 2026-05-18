<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Falcon3 multi-size BitNet-family onboarding Campaign Status

- Campaign: `falcon3-family`
- State: `active`
- Objective: Register Falcon3 Family as a first-class multi-size BitNet-family onboarding lane while keeping artifact, tokenizer/prompt, route, backend, performance, and server claims exact and unpromoted until receipts exist.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| F3-000 | ready | TBD | `codex/falcon3-family/F3-000-source-map-specs` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add Falcon3 family proposal, source map, implementation plan, active campaign tracker, spec contracts, and registered-only model/hardware matrix rows without runtime changes, model binaries, answer readiness, backend readiness, speedup, server readiness, or proof inheritance claims. |

## Hard Constraints

- Do not commit model binaries.
- Do not claim all Falcon3 works from one model size.
- Do not claim Falcon3 proof inherits Microsoft 2B, Falcon-E, Llama3-8B-1.58, or dense SLM proof.
- Do not claim I2_S/QK256 compatibility before layout proof.
- Do not route TL1/TL2 through QK256/I2_S kernels.
- Do not claim CPU/CUDA/Apple/A770 answer readiness before artifact, tokenizer/prompt, reference-good, and exact backend receipts pass.
- Do not claim speedup before exact-profile benchmark review.
