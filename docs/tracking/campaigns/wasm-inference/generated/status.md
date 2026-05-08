<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# WASM inference proof lane Campaign Status

- Campaign: `wasm-inference`
- State: `active`
- Objective: Establish WASM as an explicit receipt-backed BitNet backend lane for browser, Node, WASI, and embedded-sandbox hosts without conflating WASM detection, placeholder generation, model parsing, or scalar fallback with real inference.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| WASM-001 | ready | TBD | `codex/wasm-inference/WASM-001-proof-contract` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Document the WASM proof contract, current scaffold boundary, explicit backend identities, receipt requirements, model/tokenizer strategy, memory plan, milestones, and PR ladder without changing runtime code. |
| WASM-002 | ready | TBD | `codex/wasm-inference/WASM-002-honest-inference-compile` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Make cargo check --target wasm32-unknown-unknown -p bitnet-wasm --no-default-features --features browser,inference compile, with unsupported runtime paths returning explicit not-implemented errors rather than placeholder success. |
| WASM-003 | ready | TBD | `codex/wasm-inference/WASM-003-byte-loaders` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add byte-backed model and tokenizer loader APIs for WASM so browser paths do not depend on virtual filesystem placeholders. |
| WASM-004 | ready | TBD | `codex/wasm-inference/WASM-004-worker-api` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Expose a worker-safe JS API for loadModel, generate, generateStream, unload, and getMemoryStats, with abort support and no fake generation responses. |
| WASM-005 | ready | TBD | `codex/wasm-inference/WASM-005-tiny-fixture-proof` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Use a tiny committed fixture to emit a strict receipt with runtime_api=wasm, selected_backend=wasm-cpu-scalar, fallback_used=false, and generated_tokens=1. |
| WASM-006 | ready | TBD | `codex/wasm-inference/WASM-006-simd-kernel-smoke` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a WASM SIMD packed BitNet linear smoke proof with scalar parity, SIMD invocation count greater than zero, and strict fallback count zero. |
| WASM-007 | ready | TBD | `codex/wasm-inference/WASM-007-browser-short-decode` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Prove browser-worker short greedy decode with separately supplied and cached model artifacts, strict model hash, streamed real output, memory high-water mark, and fallback status. |
| WASM-008 | ready | TBD | `codex/wasm-inference/WASM-008-official-bitnet-feasibility` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Attempt official or canonical BitNet GGUF WASM feasibility with strict loader, real tokenizer, packed weights, one greedy token, native CPU reference parity, and no speed claim unless separately benchmarked. |

## Hard Constraints

- WASM detection is not inference.
- Placeholder generation cannot count as inference, parity, or performance proof.
- A model loader probe cannot count as sandboxed inference.
- Generic cpu or generic wasm backend labels cannot support WASM proof.
- Strict mode must fail rather than silently downgrade SIMD or threads.
- Model weights must not be embedded in the WASM runtime bundle for product claims.
- Do not claim browser BitNet 2B feasibility, usability, or speed until smaller fixture and short-decode proofs exist.
