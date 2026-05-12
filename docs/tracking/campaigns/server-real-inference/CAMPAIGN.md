# Server Real Inference Campaign

Campaign ID: `server-real-inference`

Status: active

## Objective

Replace server-side simulated inference surfaces with real engine execution or explicit unavailable responses without weakening strict proof boundaries.

## End State

- Server endpoints never return fake model output in non-test builds.
- Real inference entrypoints share loader, tokenizer, backend, receipt, and fallback truth with the CLI.
- Unsupported server configurations fail explicitly.

## Hard Constraints

- Do not reintroduce simulated inference.
- Do not bypass strict model loading, tokenizer authority, or fallback receipts.
- Do not mix server runtime work with hardware kernel claims.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| SERVER-001 | merged | Wired single-request server inference surfaces to real execution or explicit unavailable responses; no simulated response path in non-test builds. |
| SERVER-002 | merged | Failed closed placeholder model lifecycle scaffolds that previously reported HuggingFace/cache readiness without real I/O or a real inference engine. |
| SERVER-003 | merged | Added `/readiness` and `/v1/readiness` certification responses for active model, backend, inference, fallback, and claim-boundary state. |
| SERVER-004 | merged | Added `POST /v1/chat/completions` as an OpenAI-compatible surface that returns `SERVER_INFERENCE_UNAVAILABLE` with readiness/certification details until real inference is wired. |
| SERVER-005 | proposed | Wire one non-streaming chat-completions path to the same verified local inference surface as CLI ask/chat, with strict fallback rejection and per-request receipts. |

`SERVER-005` is the first runtime reopening item after the fail-closed server
surface. It must share the validated CLI loader, tokenizer, planner, receipt,
and fallback path instead of creating a second inference implementation.

## Current Claim Boundary

- Server endpoints must not return fake model output in non-test builds.
- `/v1/chat/completions` is a compatibility surface only; it does not claim real
  chat inference, server answer readiness, CUDA execution, speedup, or full
  residency.
- `/readiness` and `/v1/readiness` expose fail-closed certification state until
  the server shares the validated CLI loader, tokenizer, planner, receipt, and
  fallback paths.
- Future server-answer items must keep hardware kernel claims separate from
  server routing claims.

## Review Policy

Server PRs should be non-stackable when they touch inference routing, runtime configuration, streaming, or receipt emission.
