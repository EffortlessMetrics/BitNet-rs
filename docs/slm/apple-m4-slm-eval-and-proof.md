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
existing `answer-corpus --dry-run` shape. Runtime model evaluation and stronger
scorers are intentionally later gates.

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

The first item only records the corpus and future scoring intent. Later items
must implement and validate the stronger scorers before the corpus can support
accuracy-rate claims.

## Report Target

Each supported model should eventually publish one summary:

```text
ci/hardware/apple-m4-mac-mini/<date>/slm-eval/<model-id>/summary.json
```

Required fields include model and tokenizer identity, prompt template, backend,
fallback status, accuracy buckets, timing percentiles, resident stability, and
claim boundaries. A passing report must not be treated as a broad model-quality
benchmark unless the report explicitly records that scope.

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
