# Apple M4 SLM Warm-Answer Speed

`SLM-M4-006` extended the Apple M4 SLM warm-session receipt with bounded timing fields for the validated Qwen2.5 0.5B Q8_0 artifact. `M4-PROD-004` turns those fields into an operator profile set for 16, 32, and 64 token warm answers. The goal is to make warm-answer speed measurable after cold load is separated, not to claim broad M4 performance.

Operator profile command:

```bash
bitnet mac validate \
  --profile-set operator \
  --json-out target/apple-m4-productization/mac-operator-profiles.json
```

This writes:

```text
target/apple-m4-productization/mac-operator-profiles.json
target/apple-m4-productization/mac-operator-profiles-profiles/warm_16.json
target/apple-m4-productization/mac-operator-profiles-profiles/warm_32.json
target/apple-m4-productization/mac-operator-profiles-profiles/warm_64.json
```

Each `warm_<tokens>.json` receipt is one warm-session run for that token budget.
It records `model_loaded_once=true` and `tokenizer_loaded_once=true` within that
profile. The aggregate summary discloses `profiles_loaded_independently=true`
`profile_set_model_loads=3`, and `reuse_scope=within_each_profile`; it must not
imply one resident process was shared across all three budgets.

Recommended proof command:

```bash
RUST_LOG=warn cargo run --locked -p bitnet-cli \
  --no-default-features --features cpu,full-cli -- \
  --device apple-m4-cpu-neon \
  slm-warm-session \
  --model target/apple-m4-slm-answer/SLM-M4-003/candidates/qwen2_5_0_5b_q8_0/qwen2.5-0.5b-instruct-q8_0.gguf \
  --corpus ci/quality/apple-m4-slm-quality-corpus.yaml \
  --strict-loader \
  --strict-tokenizer \
  --fail-on-quality \
  --require-determinism \
  --json-out target/apple-m4-slm-answer/SLM-M4-006/slm-warm-speed.json
```

The aggregate receipt includes:

- `speed.reuse.model_loaded_once = true`
- `speed.reuse.tokenizer_loaded_once = true`
- `speed.counts.generated_tokens`
- `speed.timing.warm_prompt_wall_ms`
- `speed.timing.prefill_ms`
- `speed.timing.decode_total_ms`
- `speed.timing.sampling_ms`
- `speed.throughput.warm_prompt_generated_tok_s`
- `speed.throughput.decode_generated_tok_s`

Local proof on the M4 lane wrote:

```text
target/apple-m4-slm-answer/SLM-M4-006/slm-warm-speed.json
```

Observed receipt summary:

```text
quality_summary.passed = true
determinism.passed = true
speed.counts.generated_tokens = 68
speed.throughput.warm_prompt_generated_tok_s = 1.977
speed.throughput.decode_generated_tok_s = 6.397
speed.reuse.model_loaded_once = true
speed.reuse.tokenizer_loaded_once = true
```

Claim boundary:

- The receipt measures warm-answer timing for the recorded model, corpus, backend, and machine context only.
- Operator profile summaries are claim bounds for named warm-answer budgets, not latency guarantees.
- `speedup_claim` and `broad_performance_claim` remain false.
- This does not claim BitNet performance, full Apple Metal inference, QK256 support, or general M4 performance.
