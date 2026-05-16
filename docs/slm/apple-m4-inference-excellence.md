# Apple M4 Inference Excellence

This page is the operator-facing map for the
`apple-m4-inference-excellence` campaign. It starts after the durable evidence
closeout: dense benchmark, BitNet eval, and BitNet benchmark groups have
matching-history comparisons, while dense SLM eval v2 and BitNet variable warm
were intentionally kept as `insufficient_history` until another matching
refresh lands.

The goal is not to prove that the M4 can run local inference. That is already
done for the supported dense SLM path and narrowly done for the accepted BitNet
one-shot and warm proof surfaces. The goal is to make the M4 a measured,
operator-ready appliance: repeatable evidence, larger mechanical evals,
complete benchmark envelopes, reproducible run identity, artifact provenance,
service conformance, BitNet-specific gates, better operator UX, and strict
claim boundaries.

## First Proof Gap

The first two items remove the remaining important matching-history gaps:

```text
M4-EXCELLENCE-001  second dense SLM eval-v2 refresh
M4-EXCELLENCE-002  second BitNet variable-warm refresh
```

After those land, `M4-EXCELLENCE-003` refreshes the dashboard so operators can
see comparable trend status instead of relying on one-off receipts.

## Accuracy Depth

Before the large corpus work, the campaign freezes corpus and scorer identity:

```text
corpus IDs
seed generation rules
expected-output provenance
normalization rules
scoring schema
scorer self-tests
receipt version fields
```

Dense SLM accuracy work expands the deterministic corpus in two stages:

```text
100 mechanical cases
500 mechanical cases
```

Scoring stays mechanical:

```text
exact match
normalized match
numeric tolerance
JSON/schema validation
required keywords
forbidden tokens
closed-label classification
```

LLM-as-judge can be advisory only; it is not a required gate.

Small golden-token canaries stay separate from the full corpus. They record
prompt text, template identity, input token IDs, generated token IDs and text,
stop reason, sampler config, backend, fallback state, and artifact/tokenizer
identity so drift can be localized before running hundreds of cases.

## Benchmark Depth

The benchmark envelope should cover:

```text
cold load
tokenizer load
prompt tokenization
prefill/input tokens per second
TTFT
output/decode tokens per second
sampling overhead
total wall time
peak memory
memory drift
```

Reports should include p50, p90, p99, and min/max where the receipt schema
supports them. Regression comparisons must match model, tokenizer, backend,
runtime API, fallback state, corpus or profile set, and machine identity before
describing drift.

Benchmarkability also needs environment and variance evidence:

```text
macOS build
thermal and memory pressure when available
power state
disk/cache state
model cache root
background-load notes
run count and sample count
variance band
outlier handling
threshold derivation
```

## Reproducibility

Excellent M4 evidence needs enough identity to rerun or reject a comparison.
Receipts should record:

```text
machine ID and SoC
OS version
git commit
binary hash or build profile
command class
model ID and SHA256
tokenizer authority and SHA256
prompt template and stop criteria
generation parameters
backend and fallback state
corpus/profile seed
timing source
```

Artifact provenance is separate from runtime quality. A supported dense model
or accepted BitNet artifact should have a manifest for source, license or
redistribution boundary, file size, SHA256, tokenizer authority, prompt
template identity, local cache path, symlink target when used, and repair
command.

Dense SLMs also get a bounded reference-vs-Rust control so reference runner,
template, tokenizer, and Rust behavior can be distinguished without using that
control as broad model-quality evidence.

## BitNet Ladder

BitNet remains separate from dense Qwen evidence. The campaign keeps this proof
ladder:

```text
BitNet-specific corpus
reference-vs-Rust comparison
one-shot benchmark envelope
variable warm 25/50/100
progress and timeout UX
chat gate
serve gate
```

BitNet chat and serve stay disabled until their specific receipt gates pass.

## Stability And Service

The appliance should prove that it stays useful after the first successful
command:

```text
mixed dense-model switching
cache reuse and unload/reload behavior
memory drift
cache repair and low-disk guidance
interrupted generation
client cancellation
interrupted receipt write
process restart
scheduled trend retention
stale-identity aging
```

Service proof is separate from CLI proof. Dense SLM serve and later BitNet
serve need receipts for:

```text
health and ready
one-shot request
streaming completion
client cancellation
timeout stage
invalid request
missing cache
per-request receipt export
local-only safety defaults
```

Local service claims stay bounded: local appliance operation, not production
hosting and not broad OpenAI compatibility.

## Operator UX

The M4 should explain itself without requiring a user to read the whole
receipt tree. Operator-facing commands should surface:

```text
default model
supported models
cache state
disk pressure
last successful dense report
last successful BitNet report
current regressions
unsupported claims
recommended next command
route envelope class
```

`bitnet mac status`, `doctor`, `report-refresh`, and
`regression-dashboard` remain model-free by default. Live model runs belong in
local, advisory, scheduled, or release lanes.

Envelope classes should translate evidence into local user expectations:

```text
interactive
advisory
batch
disabled
unsupported
```

## Release Gates

Before the public M4 expectation envelope changes, a go/no-go matrix should say
which dense SLM, BitNet, benchmark, stability, service, operator, and
claim-boundary gates passed. A missing BitNet chat or serve gate remains a
missing feature, not a documentation issue.

## Metal Boundary

Metal work is phase-scoped only:

```text
one named phase
CPU reference parity
same generated token IDs/text where required
fallback_used=false
phase-local timing
explicit CPU/NEON remainder
```

No full `apple-m4-metal`, QK256, Neural Engine, MPSGraph, MacBook, broad Apple
Silicon, broad quality, or speedup claim is allowed until a separate full-route
receipt proves it.
