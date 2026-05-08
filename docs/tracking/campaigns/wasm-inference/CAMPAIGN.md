# WASM Inference Campaign

Campaign ID: `wasm-inference`

Status: active

## Objective

Establish WASM as an explicit receipt-backed BitNet backend lane for browser,
Node, WASI, and embedded-sandbox hosts without conflating WASM detection,
placeholder generation, model parsing, or scalar fallback with real inference.

## End State

- WASM backend identities are explicit: `wasm-cpu-scalar`, `wasm-cpu-simd`,
  `wasm-cpu-threads`, `wasm-browser-worker`, and `wasm-wasi-sandbox`.
- Byte-backed model and tokenizer loaders avoid browser filesystem assumptions.
- Tiny-fixture inference proves at least one deterministic generated token with
  `fallback_used=false` before any official BitNet browser claim.
- Browser product claims run in a worker, use separately supplied model artifacts,
  record strict hashes, and emit memory high-water receipts.
- SIMD, threads, official BitNet, and speed claims are gated by strict receipts
  that prove exactly what ran.

## Hard Constraints

- WASM detection is not inference.
- Placeholder generation cannot count as inference, parity, or performance proof.
- A model loader probe cannot count as sandboxed inference.
- Generic `cpu` or generic `wasm` backend labels cannot support WASM proof.
- Strict mode must fail rather than silently downgrade SIMD or threads.
- Model weights must not be embedded in the WASM runtime bundle for product
  claims.
- Do not claim browser BitNet 2B feasibility, usability, or speed until smaller
  fixture and short-decode proofs exist.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| WASM-001 | ready | Document the proof contract, claim boundary, backend identities, receipt fields, memory plan, and PR ladder. |
| WASM-002 | ready | Make the opt-in inference feature compile honestly for `wasm32-unknown-unknown` without claiming runtime inference. |
| WASM-003 | ready | Add byte-backed GGUF and tokenizer loading so browser paths do not rely on virtual filesystem placeholders. |
| WASM-004 | ready | Expose a worker-safe JS API for load/generate/stream/unload/memory stats with no fake generation. |
| WASM-005 | ready | Prove a tiny fixture token path with a strict WASM scalar receipt. |
| WASM-006 | ready | Add a strict WASM SIMD packed-kernel smoke proof with scalar parity. |
| WASM-007 | ready | Prove browser-worker short decode with cached model artifacts, streaming, and memory receipts. |
| WASM-008 | ready | Attempt official/canonical BitNet WASM feasibility only after smaller proofs land. |

## Review Policy

WASM PRs are non-stackable when they alter claim boundaries, receipt semantics,
backend identity, or artifact acceptance. Compile-cleanup, loader, worker, kernel,
and benchmark PRs should stay in their own work items so a placeholder or probe
cannot accidentally become an inference claim.
