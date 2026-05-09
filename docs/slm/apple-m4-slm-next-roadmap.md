# Apple M4 Dense SLM Next Roadmap

The M4 dense SLM appliance baseline is complete. The supported local path now
has `bitnet mac ask`, `bitnet mac chat`, `bitnet mac smoke`, `bitnet mac
doctor`, `bitnet mac regression`, model-cache verification, quality corpus 2.0,
long-session soak receipts, a measured expectation envelope, and strict claim
boundaries.

The next work is expansion, not baseline completion.

## Maintenance Baseline

Keep the appliance healthy with:

```bash
bitnet mac doctor
bitnet mac smoke
bitnet mac regression <receipt.json> --baseline <baseline.json>
bitnet mac receipts-check <receipt-or-directory>
```

Run these after changes to tokenization, prompt templates, sampling, model-cache
behavior, receipt schema, Mac CLI behavior, or runtime timing.

## Next Campaigns

### `apple-m4-slm-model-breadth`

Goal: add more supported dense instruct model families without weakening the
existing gates.

Model candidates must stay exact and pinned. A model can become supported only
after source, revision, file, size, SHA256, tokenizer authority, prompt
template, reference output sanity, Rust M4 output quality, cache metadata,
receipt validation, and deterministic behavior where required are recorded.

The first selected breadth candidates are `qwen3-0.6b-q8_0` and
`smollm2-360m-instruct-q8_0`. They are evaluation candidates only, not supported
M4 models.

### `apple-m4-slm-metal-phases`

Goal: expand Apple Metal contribution phase by phase while preserving CPU/NEON
as the honest default path.

Every Metal phase needs CPU-only versus CPU-plus-Metal greedy parity,
`fallback_used=false` for the Metal phase, explicit CPU/NEON routing for the
rest of the pipeline, timing delta receipts, and no full `apple-m4-metal`
inference claim.

### `apple-m4-local-server`

Goal: expose the M4 dense SLM appliance as a local service while reusing the
same model-cache, tokenizer authority, resident-session, streaming, and receipt
discipline.

The first server surface is a command/config contract for `bitnet mac serve`:
loopback by default, `apple-m4-cpu-neon`, supported dense model IDs only,
startup cache/tokenizer/backend verification, streaming by default, receipt
export, and no hidden fallback. Health/ready endpoints, streaming completions,
and receipt export are later implementation items. OpenAI-shaped endpoints must
not be described as fully compatible until their request/response semantics are
tested.

## Still Separate

Dense SLM success is not BitNet success. The M4 BitNet path still requires an
accepted BitNet artifact, strict `apple-m4-cpu-neon` proof, coherent generated
output, token IDs, `fallback_used=false`, determinism, receipt hardening, warm
sessions, and then phase-scoped Metal.
