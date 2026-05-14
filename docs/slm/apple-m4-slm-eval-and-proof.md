# Apple M4 SLM Eval And Proof

This page defines the next dense SLM proof lane for the M4 Mac mini. It turns
the current receipt-backed Qwen appliance into a measured local model runner
without claiming broad Apple Silicon benchmark status.

## Scope

This lane is dense SLM only. It covers the supported Qwen model set on
`apple-m4-cpu-neon` and keeps BitNet, full Metal inference, QK256, Neural
Engine execution, MPSGraph inference, MacBook evidence, and broad performance
claims out of scope.

The first committed corpus spec is:

```text
ci/quality/apple-m4-slm-eval-seeded-corpus.yaml
```

It is seeded with `424242`, deterministic, and parser-compatible with the
existing `answer-corpus --dry-run` shape. The scoring contract now has
deterministic fixture support, but runtime model evaluation remains a later
gate.

## Proof Plan

Dense M4 SLM quality needs four planes:

- accuracy and answer quality;
- latency and throughput;
- resident-session stability;
- regression economics.

The current dense appliance already has smoke, quality corpus, warm-session,
performance-envelope, model-matrix, and regression evidence. This lane adds the
missing structured eval bridge: seeded reproducible cases, exact/normalized and
schema-style scoring, per-model summary reports, CI tiering, and regression
comparison.

## Seeded Corpus Families

`apple-m4-slm-eval-seeded-corpus-v1` includes deterministic cases for:

- arithmetic with exact answers;
- fixed-table factual QA;
- compact JSON shape;
- closed-label classification;
- synthetic extraction;
- ordering and sorting;
- copy/edit/rewrite;
- constrained summarization;
- required and forbidden token instruction following.

The corpus records deterministic scoring kinds for exact, normalized,
schema-style JSON, numeric tolerance, required-keyword, and forbidden-token
checks. These scorers can validate fixture answers and future generated text,
but they do not create an accuracy-rate claim until supported-model runtime
reports are published.

## Report Target

Each supported model should eventually publish one summary:

```text
ci/hardware/apple-m4-mac-mini/<date>/slm-eval/<model-id>/summary.json
```

The schema artifact kind is:

```text
apple_m4_slm_eval_summary
```

The report must validate through `bitnet mac receipts-check` and record:

- schema version, `machine_id=apple-m4-mac-mini`, model ID, model repo/file,
  model SHA256, model family, architecture, and quantization;
- tokenizer source, tokenizer authority, pretokenizer authority, strict
  tokenizer mode, and prompt template;
- seeded corpus name, seed, and case count;
- accuracy totals plus exact, normalized, JSON/schema, numeric tolerance,
  required-keyword, and forbidden-token pass rates;
- generated-text and generated-token-ID evidence coverage, generated token
  total, and source receipt links;
- first-class speed fields for cold load, tokenizer load, prompt tokenization,
  prefill, TTFT p50/p90, input token throughput, output/decode throughput,
  sampling cost, and total wall time;
- peak memory and resident-session stability fields;
- dense-SLM-only claim boundaries.

Required claim-boundary flags keep the report narrow:

```text
dense_slm_only=true
bounded_seeded_corpus_only=true
broad_model_quality_claim=false
broad_performance_claim=false
bitnet_evidence=false
full_metal_inference_claimed=false
qk256_apple_claimed=false
neural_engine_claimed=false
mpsgraph_inference_claimed=false
macbook_evidence=false
speedup_claim=false
```

A passing report is a bounded seeded-corpus dense SLM artifact. It is not a
broad model-quality benchmark, broad Apple Silicon performance claim, BitNet
proof, full Metal inference proof, QK256 claim, Neural Engine claim, MPSGraph
claim, or MacBook claim.

## Published 2026-05-14 Reports

`M4-SLM-EVAL-004` publishes the first per-model report set:

```text
ci/hardware/apple-m4-mac-mini/2026-05-14/slm-eval/<model-id>/summary.json
```

Each summary combines:

- seeded deterministic scoring from `answer-corpus` over
  `ci/quality/apple-m4-slm-eval-seeded-corpus.yaml`;
- resident stability from `bitnet mac validate` over
  `ci/quality/apple-m4-slm-quality-corpus.yaml`;
- source answer-corpus receipts, per-case run receipts, resident aggregate
  receipts, and resident per-prompt receipts.

For the non-default model runs, the summary model identity is taken from the
supported model-cache entry and per-case run receipts. The aggregate
`answer-corpus` receipt still carries the corpus default model block, so it is
not treated as the authority for non-default model SHA or quantization.

All three summaries validate with `bitnet mac receipts-check` and record
`requested_backend=apple-m4-cpu-neon`, `selected_backend=apple-m4-cpu-neon`,
`runtime_api=cpu`, and `fallback_used=false`.

| Model ID | Seeded score | TTFT p50 | TTFT p90 | Input tok/s p50 | Output tok/s p50 | Decode tok/s p50 | Resident prompts | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen2.5-0.5b-instruct-q8_0` | 2 / 10 | 3895.5 ms | 4529.0 ms | 12.398 | 1.619 | 9.034 | 14 | 4012.047 MB |
| `qwen2.5-0.5b-instruct-q4_k_m` | 2 / 10 | 3596.0 ms | 4313.0 ms | 12.358 | 1.886 | 9.162 | 14 | 4013.844 MB |
| `qwen2.5-1.5b-instruct-q4_k_m` | 2 / 10 | 14879.0 ms | 17867.0 ms | 3.335 | 0.329 | 3.039 | 14 | 7995.656 MB |

The seeded score is intentionally strict. The current misses mostly expose
format and stop-token behavior such as raw `<|im_end|>` tails or JSON fenced
code blocks, not a broad quality verdict. The resident quality corpus still
passes for all three models, so the report set should be read as bounded
evidence with concrete failure modes for future regression work.

The warm-session receipt records peak memory through `getrusage.ru_maxrss`; it
does not record a separate memory-drift series. The summary schema therefore
records `memory_drift_mb=0.0` with
`memory_drift_source=not_recorded_by_warm_session_receipt`.

## CI Tiers

Generic PR CI stays lightweight:

- Tier 0: schema, tracker, receipt-schema, parser, and tiny fixture checks;
- Tier 1: advisory M4 label for a quick supported-model smoke;
- Tier 2: nightly M4 full supported-model matrix, seeded corpus, speed profiles,
  and resident soak;
- Tier 3: release gate for full reports, regression comparison, and published
  user envelope refresh.

Live model downloads, long soaks, and hardware timing runs do not belong in
ordinary required CI.
