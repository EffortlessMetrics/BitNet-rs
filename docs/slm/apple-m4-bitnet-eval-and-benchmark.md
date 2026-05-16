# Apple M4 BitNet Eval And Benchmark

This is the next BitNet proof layer after the Apple M4 one-shot `bitnet mac ask`
route and fixed-prompt `bitnet mac bitnet-warm` proof. The goal is to make
BitNet measurable enough for operators before any broader product surface is
enabled.

## Current Boundary

The accepted BitNet artifact is:

- Model id: `microsoft-bitnet-b1.58-2B-4T-i2s`
- Repo: `microsoft/bitnet-b1.58-2B-4T-gguf`
- Revision: `a1f2f1c765812aa8af3f6eda4a313707064bba15`
- File: `ggml-model-i2_s.gguf`
- SHA256: `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`
- Tokenizer repo: `microsoft/bitnet-b1.58-2B-4T`
- Tokenizer revision: `04c3b9ad9361b824064a1f25ea60a8be9599b127`
- Tokenizer file: `tokenizer.json`
- Tokenizer SHA256: `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`
- Pre-tokenizer authority: `llama-bpe`
- Prompt authority: `bitnetcpp-answer`

The current product surface remains narrow:

- `bitnet mac ask` supports explicit one-shot BitNet asks for the accepted
  artifact and tokenizer.
- `bitnet mac bitnet-warm` supports a fixed-prompt warm proof route.
- BitNet chat and BitNet serve remain disabled.

## Campaign Shape

`M4-BITNET-EVAL-001` adds a deterministic BitNet-specific corpus and dry-runs it
through the existing answer-corpus parser/scoring path. This is a fixture and
tracking PR only. It does not run the model and does not claim runtime accuracy
or performance.

`M4-ACCURACY-000` freezes the BitNet eval corpus/scorer contract before any
larger BitNet accuracy expansion. The YAML records `metadata.corpus_contract`
with:

```text
contract_version: m4-eval-corpus-scorer-contract-v1
corpus_id: apple-m4-bitnet-eval-seeded-corpus
corpus_version: 1.0.0
seed: 912587
generator_policy: deterministic-static-fixture-bitnet-v1
scoring_schema: answer_corpus_mechanical_scoring_v1
receipt_contract: answer_corpus_aggregate_receipt_v1
```

Expected outputs are closed-form deterministic fixture answers from the YAML
prompt data. Reference-runner answers can be added as comparison evidence, but
they do not replace the mechanical expected-output authority. `answer-corpus`
aggregate receipts propagate the contract under `corpus.contract` and
`scoring_contract`; this keeps BitNet eval identity separate from dense SLM
evidence.

Later work items add:

- BitNet eval/report schema fields for reference-vs-Rust comparison.
- M4 eval receipts for the accepted I2_S artifact.
- One-shot and fixed-warm benchmark receipts.
- Advisory/nightly regression dashboards for BitNet quality and performance.

## Report Schema Slice

`M4-BITNET-EVAL-002` extends `answer-corpus` receipts so later M4 BitNet eval
runs can be scored, compared, and regressed without inventing a second report
format.

Aggregate receipts now include:

- model authority for the accepted BitNet artifact, including repo, revision,
  file, SHA256, byte size, architecture, and quantization.
- `task_family_summary`: per-family totals, pass/fail/timeout/not-run counts,
  scoring totals, scoring kinds, and failure taxonomy counts.
- `reference_comparison`: a `bitnet_reference_vs_rust_v1` summary with the
  Rust runner backend/runtime API, fallback status, prompt template, tokenizer
  authority, status counts, text/token-ID match counts, mismatched fields, and
  claim boundaries.

Each case row now includes:

- `task_family`, `category`, and `seed_material`.
- generated text and token IDs when a live run exists.
- `reference_comparison.schema = bitnet_reference_vs_rust_v1`.
- reference metadata when supplied by a later reference-runner receipt.
- Rust output metadata and comparison status:
  `reference_not_supplied`, `not_run`, `matched`, `mismatched`, or
  `partially_compared`.

Dry-run validation for this slice still records all 100 corpus cases as
`not_run`; the comparison summary records that reference answers are not yet
supplied. This is schema readiness only, not runtime BitNet accuracy or
performance evidence.

## M4 Eval Report Slice

`M4-BITNET-EVAL-003` runs the 100-case seeded BitNet corpus on the M4 Mac mini
through the accepted I2_S GGUF, external tokenizer authority, and
`apple-m4-cpu-neon` backend.

Recorded report:

- Aggregate receipt:
  `ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-eval/answer-corpus.json`
- Child receipts:
  `ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-eval/answer-corpus-runs/*.json`
- Artifact kind: `bitnet_apple_m4_local_answer_corpus`
- Corpus: `apple-m4-bitnet-eval-seeded-corpus`
- Cases: 100
- Passed: 75
- Failed: 25
- Timeout: 0
- Not run: 0
- Generated tokens: 765
- Backend: `apple-m4-cpu-neon`
- Runtime API: `cpu`
- Fallback used: `false`
- Reference comparison schema: `bitnet_reference_vs_rust_v1`
- Reference answers supplied: 0

Task-family pass rates:

| Task family | Passed | Failed |
|---|---:|---:|
| arithmetic_exact | 10 | 0 |
| closed_label_classification | 9 | 1 |
| constrained_summary | 9 | 1 |
| fixed_table_qa | 6 | 4 |
| format_constrained_json | 5 | 5 |
| numeric_tolerance | 5 | 5 |
| ordering_sorting | 8 | 2 |
| required_forbidden_tokens | 7 | 3 |
| rewrite_normalized | 9 | 1 |
| synthetic_extraction | 7 | 3 |

The receipt records generated text, generated token IDs, tokenizer authority,
model SHA, per-case timing, task-family scoring, and failure taxonomy for the
bounded corpus. It also keeps the explicit claim boundary: this is not a broad
BitNet quality benchmark, not a performance envelope, not dense SLM evidence,
and not chat, serve, Metal, QK256, Neural Engine, MPSGraph, MacBook, or broad
Apple Silicon proof.

## M4 Benchmark Report Slice

`M4-BITNET-EVAL-004` benchmarks the explicit one-shot `bitnet mac ask` route and
the fixed-prompt `bitnet mac bitnet-warm` route for the accepted BitNet artifact.
It adds a BitNet-specific aggregate benchmark receipt and teaches
`bitnet mac receipts-check` to validate it.

Recorded report:

- Aggregate receipt:
  `ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-benchmark/summary.json`
- One-shot receipt:
  `ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-benchmark/receipts/bitnet-mac-ask-benchmark.json`
- Fixed-warm receipt:
  `ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-benchmark/receipts/bitnet-mac-bitnet-warm-benchmark.json`
- Fixed-warm prompt receipts:
  `ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-benchmark/receipts/bitnet-mac-bitnet-warm-benchmark-prompts/*.json`
- Artifact kind: `bitnet_apple_m4_benchmark_v1`
- Benchmark set: `bitnet-one-shot-fixed-warm-v1`
- Prompts: 4 total, 1 one-shot and 3 fixed-warm
- Generated tokens: 8
- Backend: `apple-m4-cpu-neon`
- Runtime API: `cpu`
- Fallback used: `false`
- Timeout boundary: not reached, not enforced
- Chat enabled: `false`
- Serve enabled: `false`

Aggregate speed and memory summary:

| Metric | p50 | p90 | p99 |
|---|---:|---:|---:|
| Model/cold load ms | 4410.309 | 4443.249 | 4443.249 |
| Tokenizer load ms | 163.834 | 169.638 | 169.638 |
| Prompt tokenize ms | 0.056 | 0.205 | 0.205 |
| Prefill ms | 10063.830 | 11316.592 | 11316.592 |
| TTFT ms | 10777.000 | 12042.000 | 12042.000 |
| Input tok/s | 1.625 | 2.408 | 2.408 |
| Output tok/s | 0.158 | 0.216 | 0.216 |
| Decode tok/s | 1.411 | 2.055 | 2.055 |
| Total wall ms | 11489.261 | 12739.703 | 12739.703 |
| Peak memory MiB | 4246.359 | 4322.078 | 4322.078 |

Path-level summary:

| Path | Prompts | Generated tokens | TTFT p50 ms | Output tok/s p50 | Decode tok/s p50 |
|---|---:|---:|---:|---:|---:|
| `mac ask` one-shot | 1 | 2 | 8800.000 | 0.216 | 2.055 |
| `mac bitnet-warm` fixed warm | 3 | 6 | 11878.000 | 0.158 | 1.411 |

The one-shot prompt answered `4`. The fixed-warm prompts answered `4`, `Paris`,
and `4`. These benchmark receipts are timing evidence for the exact accepted
artifact/tokenizer/backend on the recorded M4 Mac mini only. They are not a broad
Apple Silicon performance claim, not a speedup claim, not a BitNet quality
benchmark, and not evidence that BitNet chat, BitNet serve, full Metal, QK256,
Neural Engine, MPSGraph, MacBook, or broader Apple Silicon routes work.

## Regression Dashboard Slice

`M4-BITNET-EVAL-005` wires the existing `bitnet mac regression` and
`bitnet mac receipts-check --regression-baseline` paths to compare the BitNet
eval and benchmark reports. Generic PR CI remains model-free; these comparisons
are for advisory, nightly, scheduled, or release-refresh lanes that already have
the receipt artifacts.

Supported BitNet regression artifacts:

- `bitnet_apple_m4_local_answer_corpus`
- `bitnet_apple_m4_benchmark_v1`

Example commands:

```bash
bitnet mac regression \
  ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-eval/answer-corpus.json \
  --baseline ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-eval/answer-corpus.json \
  --json

bitnet mac regression \
  ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-benchmark/summary.json \
  --baseline ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-benchmark/summary.json \
  --fail-on-drift
```

The eval comparison is strict about context before it reports drift:

- exact model repo, revision, file, path, SHA256, architecture, and I2_S
  quantization.
- exact external tokenizer repo, revision, path, SHA256, and `llama-bpe`
  authority.
- exact backend/runtime identity, prompt template, corpus name/path/case count,
  selected case IDs, scoring kinds, and reference-comparison schema.
- claim boundaries that keep dense SLM evidence, chat, serve, Metal, QK256,
  Neural Engine, MPSGraph, MacBook, runtime-accuracy, and broad Apple Silicon
  claims out of scope.

Eval warnings are advisory by default and become failures with
`--fail-on-drift`. Warning thresholds cover:

- aggregate quality pass drops and failed/timeout/not-run increases.
- strict scoring pass drops and failed/not-run increases.
- task-family pass drops and failed/timeout/not-run increases.
- reference-vs-Rust comparable/matched/text/token-ID drops and mismatch/not-run
  increases.

The benchmark comparison is also context-gated before drift checks:

- exact accepted model and tokenizer identity.
- exact one-shot `mac ask` and fixed-warm `mac bitnet-warm` path definitions.
- exact prompt/generation counts, timeout status, release-mode evidence, backend
  identity, and no fallback.
- claim boundaries that keep BitNet chat, BitNet serve, broad quality,
  broad performance, speedup, Metal, QK256, Neural Engine, MPSGraph, MacBook,
  and broad Apple Silicon claims disabled.

Benchmark warnings cover p50/p90/p99 and path-level summaries for TTFT,
prefill, prompt tokenization, load time, input/output/decode throughput, decode
time, total wall time, sampling time, peak memory, memory drift, and process
peak drift. The dashboard can therefore catch both quality regressions and
operator-cost regressions without running live models in generic PR checks.

## Claim Policy

Allowed after the first slice:

```text
A 100-case deterministic BitNet eval corpus exists and parser/scoring dry-run
validation passes.
```

Not allowed after the first slice:

```text
Runtime BitNet accuracy has been measured for this corpus.
BitNet performance has been benchmarked.
BitNet chat works.
BitNet serve works.
Full apple-m4-metal inference works.
QK256, Neural Engine, MPSGraph, MacBook, or broad Apple Silicon claims.
```

Dense SLM eval reports stay separate. They can prove dense Qwen behavior only;
they are not BitNet quality or performance evidence.

Allowed after the report-schema slice:

```text
BitNet eval receipts can represent task-family scoring, timeout/failure
taxonomy, generated text/token IDs, backend identity, fallback status, and
reference-vs-Rust comparison fields.
```

Still not allowed after the report-schema slice:

```text
The full BitNet eval corpus has run on M4.
Runtime BitNet accuracy has been measured for this corpus.
BitNet performance has been benchmarked.
BitNet chat works.
BitNet serve works.
Full apple-m4-metal inference works.
QK256, Neural Engine, MPSGraph, MacBook, or broad Apple Silicon claims.
```

Allowed after the M4 eval report slice:

```text
The recorded Apple M4 BitNet eval receipts describe the accepted I2_S
artifact's bounded 100-case corpus behavior for those exact runs.
```

Still not allowed after the M4 eval report slice:

```text
The reports are broad BitNet quality benchmarks.
The reports are BitNet performance benchmarks.
Dense SLM evidence supports BitNet quality.
BitNet chat works.
BitNet serve works.
Full apple-m4-metal inference works.
QK256, Neural Engine, MPSGraph, MacBook, or broad Apple Silicon claims.
```

Allowed after the benchmark report slice:

```text
Recorded Apple M4 BitNet one-shot and fixed-warm benchmark receipts exist for
the accepted I2_S artifact and external tokenizer authority.
```

Still not allowed after the benchmark report slice:

```text
The benchmark is broad Apple Silicon performance evidence.
The benchmark is a broad BitNet quality benchmark.
BitNet chat works.
BitNet serve works.
Full apple-m4-metal inference works.
QK256, Neural Engine, MPSGraph, MacBook, or broad Apple Silicon claims.
```
