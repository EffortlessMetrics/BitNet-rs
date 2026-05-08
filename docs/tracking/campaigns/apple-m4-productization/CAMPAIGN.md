# Apple M4 Productization Campaign

Campaign ID: `apple-m4-productization`

Status: active

## Objective

Turn the completed Apple M4 SLM answer proof into a practical operator path for Mac users: documented baseline commands, model cache management, Mac CLI entry points, warm-session speed polish, and a parity-gated Metal phase handoff.

## Why This Exists

The `apple-m4-slm-answer` campaign proved that the M4 can run a sub-1 GiB dense instruct GGUF through the Rust CLI on `apple-m4-cpu-neon`, with strict loader/tokenizer behavior, explicit fallback status, warm-session operation, quality/determinism receipts, and bounded warm timing.

That campaign is proof complete. This campaign owns productization: making the working path discoverable, repeatable, storage-conscious, and safe for normal Mac users without implying BitNet quality or full Metal inference.

## End State

- The working Rust-native Apple M4 CPU/NEON SLM local-answer baseline is documented as the practical Mac path.
- A model cache surface can fetch, verify, list, and prune supported artifacts without committing binaries.
- Mac-oriented CLI commands wrap check, ask, validate, and receipt-bundle flows with clear strict failures.
- Warm-session speed remains separated from cold load and records load, tokenize, prefill, decode, sampling, and total timing.
- The first Apple Metal contribution is implemented only as a named phase with CPU parity and explicit fallback boundaries.

## Hard Constraints

- Do not reopen the completed `apple-m4`, `apple-m4-operational`, or `apple-m4-slm-answer` campaigns.
- Do not weaken the blocked BitNet `apple-m4-local-answer` gates.
- Do not claim BitNet local-answer quality from dense SLM evidence.
- Do not claim full `apple-m4-metal` inference until a strict real-model receipt proves it.
- Do not claim Neural Engine execution from MPSGraph or any unresolved Apple graph target.
- Do not claim QK256 support on Apple Silicon from SLM evidence.
- Do not claim broad performance from warm-session or tiny phase receipts.
- Never commit model binaries.

## User-Facing Baseline

Works today:

```text
Rust-native Apple M4 CPU/NEON SLM local answers with strict loader/tokenizer routing, visible fallback status, warm-session receipts, quality checks, determinism checks, and bounded warm timing.
```

Not claimed:

```text
BitNet local-answer quality
full apple-m4-metal model inference
Neural Engine execution
QK256 on Apple Silicon
general M4 performance
```

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-PROD-001 | in_progress | Document the working Apple M4 SLM local-answer baseline and claim boundary. |
| M4-PROD-002 | proposed | Add model fetch, verify, list, and prune commands for the supported SLM artifact cache. |
| M4-PROD-003 | proposed | Add Mac-oriented check, ask, validate, and receipts-check CLI wrappers. |
| M4-PROD-004 | proposed | Polish warm-session speed measurement and operator thresholds without broad performance claims. |
| M4-PROD-005 | proposed | Implement the first parity-gated Apple Metal prefill projection microphase handoff. |

## Review Policy

Each PR owns one work item. Productization work may improve CLI and docs around the validated SLM path, but it must keep BitNet proof lanes, QK256, MPSGraph, Neural Engine, and full Metal inference claims separate unless a future item explicitly permits and proves them.
