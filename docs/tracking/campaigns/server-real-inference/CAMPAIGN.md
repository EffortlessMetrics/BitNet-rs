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
| SERVER-001 | proposed | Wire single-request server inference to real engine surfaces or explicit 501/503. |

## Review Policy

Server PRs should be non-stackable when they touch inference routing, runtime configuration, streaming, or receipt emission.
