# Apple M4 SLM Excellence Roadmap

The Apple M4 Mac mini now has a working Rust-native dense SLM path:

```bash
bitnet mac ask "What is 2+2?"
bitnet mac chat --prompt "What is 2+2?" --prompt "Name the capital of France."
bitnet mac smoke
```

The next target is not another proof that the path exists. The target is an
appliance-grade local model runner experience: predictable cache behavior,
fast warm interaction, quiet default logs, useful health checks, stable
resident sessions, regression receipts, leading dense SLM support, and clear
unsupported-path errors.

## Scope

This roadmap is M4 Mac mini local work.

It owns:

- dense Qwen-class local answer UX on `apple-m4-cpu-neon`;
- Mac cache, first-run, health, smoke, chat, and receipt flows;
- resident-session stability and memory/timing envelopes;
- advisory local regression checks against the recorded M4 envelope;
- support-matrix work for leading dense SLM candidates;
- future phase-scoped Metal work only after CPU/NEON behavior remains boring.

It does not own:

- MacBook artifact sweeps;
- x86, CUDA, A770, Lunar Lake, or NPU lanes;
- BitNet artifact qualification;
- QK256 work;
- server inference;
- full `apple-m4-metal` inference claims.

## Model-Family Boundary

The current Mac user-facing path is a dense regular SLM path. It uses the
validated Qwen2.5 dense model artifact to exercise the real runner surface:
model cache, tokenizer, prompt template, generation, warm sessions, receipts,
quality checks, and Apple CPU/NEON routing.

That evidence does not prove BitNet local-answer quality, 1-bit / 1.58-bit
kernels, I2_S/TL1/TL2 layouts, QK256, Neural Engine execution, MPSGraph model
inference, or full Apple Metal inference.

BitNet remains a separate model family. The M4-side BitNet proof command and
receipt contract are prepared, but the strict proof stays blocked until an
accepted BitNet artifact exists.

## WASM Relationship

WASM is a first-class product target for the broader Rust-native model runner.
It is not part of this M4 execution lane. The M4 roadmap should keep WASM
visible as product architecture while leaving implementation to the WASM
campaign.

The shared invariant across native Mac, x86, GPU/NPU, and WASM lanes is the
receipt contract: model identity, tokenizer authority, runtime API, requested
and selected backend, fallback status, generated text, token IDs, timing,
memory where available, and explicit claim boundaries.

## CI Efficiency

The excellent Mac experience should not make every PR slow. Ordinary CI should
use fast checks: argument parsing, help snapshots, synthetic receipts, campaign
validation, and unit tests that do not download models. Live model generation,
long resident sessions, and hardware timing comparisons should remain local,
advisory, or scheduled on Apple hardware until there is an explicit hardware
runner policy.

## PR Ladder

### M4-SLM-EX-001: Mac Doctor

Add:

```bash
bitnet mac doctor
```

It should answer whether the M4 dense SLM path is healthy by checking cache
presence, model hash, disk space, backend/fallback identity, and
unsupported-backend rejection. Live smoke answer behavior and generated receipt
validation should be explicit through `--run-smoke`, not part of the default
repair/readiness path.

The command must not download a model unless a later item explicitly adds an
opt-in repair flag. It should tell the operator which existing command to run
for repair, such as `bitnet model fetch`, `bitnet model verify`, or
`bitnet model prune`.

### M4-SLM-EX-002: Interactive Chat Polish

Improve `bitnet mac chat` as a resident local tool:

- clean prompt loop behavior through `--interactive`;
- EOF, `/exit`, and `/quit` handling before generation;
- quiet default logs;
- streaming by default;
- optional per-turn receipt output through `--no-turn-receipts`;
- aggregate session receipt at exit;
- clear model/tokenizer loaded-once status in receipts and `--progress` output.

### M4-SLM-EX-003: Time-To-First-Token Pass

Reduce perceived latency by measuring and tightening:

- prompt template construction;
- tokenization overhead;
- first decode step timing;
- streaming flush behavior;
- cache verification placement;
- receipt construction outside the hot path.

The first concrete pass reuses the SHA256 already produced by the Mac model
cache verifier when `bitnet mac chat` and `bitnet mac validate` launch resident
warm sessions. That avoids a second full-file GGUF hash before generation and
records `model.sha256_source`, `model.sha256_rehash_skipped`, and
`timing.model_sha256_ms` in the warm-session receipt so the optimization is
auditable. The same pass keeps the existing deterministic quality corpus usable
by normalizing the observed leading Qwen assistant separator before applying
format-prefix gates; generated text and token IDs remain unchanged in receipts.

Acceptance requires unchanged greedy token IDs and unchanged quality corpus
behavior.

### M4-SLM-EX-004: Hot-Loop Allocation Cleanup

Tighten resident decode hygiene:

- sampling scratch reuse;
- logits buffer reuse where supported;
- token vector growth control;
- detokenization string churn;
- temporary tensor creation;
- JSON receipt construction outside generation.

The first cleanup preallocates the sampler logits scratch buffer before the
resident decode loop. Local allocation-audit receipts showed `sampler.sample`
performed one `vocab_size * sizeof(f32)` allocation per prompt; after this pass
that allocation is moved into prompt setup and the sampler reuses the buffer for
each decode step. This does not claim reusable storage for model logits tensor
extraction, which remains a separate and larger allocation source.

Acceptance requires quality unchanged, timing not worse, and receipt schema
unchanged.

### M4-SLM-EX-005: Dense Model Support Matrix

Make supported and candidate dense SLMs explicit, including leading small
instruct families that are plausible for local Apple Silicon use. Each entry
should record:

- source and file;
- size and SHA256;
- tokenizer authority;
- prompt template;
- quantization;
- Rust support status;
- M4 support status;
- cache policy;
- quality status.

The default remains the currently verified dense Qwen M4 model. Candidate
models are not accepted until reference and Rust M4 quality gates pass.

The current matrix lives in
[`apple-m4-dense-slm-model-support-matrix.md`](apple-m4-dense-slm-model-support-matrix.md).
It keeps the Qwen2.5 Q8_0 default separate from the Qwen2.5 Q4_K_M
storage-conscious candidate, Qwen3 diagnostic work, cross-family candidates,
and rejected unpinned or unsupported artifact classes.

### M4-SLM-EX-006: Second Supported Dense Model

Add a second storage-conscious dense instruct model only if it passes:

- reference output sanity;
- Rust M4 output quality;
- tokenizer authority checks;
- cache metadata verification;
- receipt validation.

No model binaries may be committed.

The second supported model is the registered Qwen2.5 0.5B Instruct Q4_K_M GGUF
artifact. This keeps Qwen2.5 Q8_0 as the default while adding a smaller
non-default model ID for storage-conscious M4 use. The support slice adds the
standard GGUF dequantization needed by this artifact (`Q5_0`, `Q4_K`, and
`Q6_K`), then requires both reference output sanity and Rust M4 quality receipts
before marking the model supported.

### M4-SLM-EX-007: Quality Corpus 2.0

Expand the local smoke surface without turning it into a benchmark suite:

- simple factual answers;
- short instruction following;
- format-constrained output;
- one-sentence generation;
- basic arithmetic;
- small summarization;
- short rewrite.

Acceptance requires valid UTF-8, non-empty output, non-degenerate output,
stable greedy IDs where expected, and validated receipts.

The v2 corpus is `apple-m4-slm-quality-determinism-v2`. It keeps the original
five deterministic groups and adds bounded summarization and rewrite prompts,
for seven cases repeated twice in one warm session.

### M4-SLM-EX-008: Long-Session Soak

Record longer resident behavior:

- 25-prompt sessions;
- 50-prompt sessions;
- 64-token and 128-token response budgets;
- memory drift;
- time-to-first-token drift;
- decode throughput drift;
- quality failures;
- model/tokenizer reuse.

The output is a scoped M4 Mac mini envelope, not a fleet-wide Apple Silicon
performance claim.

The scoped M4 Mac mini soak receipts live under:

```text
ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/
```

The recorded profiles are `resident-25-64.json` and `resident-50-128.json`,
with a compact `summary.json`. Both profiles keep `fallback_used = false`,
`model_loaded_once = true`, `tokenizer_loaded_once = true`, and
`quality_summary.passed = true`. The 50-prompt profile records 1185 generated
tokens, `peak_memory_mb = 4020.25`, and `decode_generated_tok_s = 6.313` for
this M4 Mac mini receipt only.

### M4-SLM-EX-009: Local Regression Command

Add:

```bash
bitnet mac regression \
  ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/resident-25-64.json \
  --baseline ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/resident-25-64.json
```

It should compare current receipts with the stored M4 dense SLM envelope and
report drift in model identity, tokenizer identity, backend routing, fallback
status, quality, time-to-first-token, decode throughput, and peak memory.

`bitnet mac regression` is receipt-only: it does not download models and does
not run live generation. It validates receipts first, compares only matching M4
dense SLM performance or warm-session envelopes, and reports timing/memory drift
as advisory warnings by default. Use `--fail-on-drift` when an operator wants a
local hard failure for threshold drift.

### M4-SLM-EX-010: Measured User Envelope

Publish an operator expectation page for the M4 Mac mini:

- cold load time;
- warm ask timing;
- time-to-first-token;
- warm 16/32/64/128 token timing;
- decode tokens per second;
- peak memory;
- cache size;
- known unsupported models and backends.

This is expectation-setting, not a broad performance claim.

The measured envelope is published in
`docs/slm/apple-m4-mini-user-expectation-envelope.md`. It records the supported
models, cache sizes, release-mode warm 16/32/64/128 profile timings, long
resident-session soak evidence, health/regression commands, and unsupported
claim boundaries for the M4 Mac mini dense SLM path.

## Later Metal Work

Dense SLM Metal work should remain phase-scoped until strict receipts justify a
backend-level claim. Candidate phases include Q/K/V prefill projection, MLP
up/gate projection, and `lm_head` projection.

Every phase must prove:

- CPU-only greedy tokens match CPU-plus-Metal-phase greedy tokens;
- `fallback_used=false` for the Metal phase;
- the rest of the pipeline remains explicit CPU/NEON;
- timing delta is recorded;
- no full `apple-m4-metal` inference claim is made.
