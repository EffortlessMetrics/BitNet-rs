# Apple M4 SLM Quality Determinism

Campaign: `apple-m4-slm-answer`

Work item: `SLM-M4-005`

Status: `in_progress`

## Summary

`SLM-M4-005` adds a committed warm-session quality corpus for the Apple M4 SLM lane. The corpus is intentionally small: three short prompts, each repeated twice under deterministic greedy decoding, using the Rust-supported Qwen2.5 0.5B Q8_0 companion artifact.

This proves only bounded Apple M4 CPU/NEON dense-SLM answer quality and deterministic repeatability for the recorded prompts. It does not claim BitNet local-answer quality, full `apple-m4-metal` inference, QK256 support, Neural Engine execution, MPSGraph execution, or broad performance.

## Corpus

Source:

```text
ci/quality/apple-m4-slm-quality-corpus.yaml
```

Defaults:

```text
prompt_template = qwen2.5
max_new_tokens = 16
temperature = 0.0
top_k = 1
greedy = true
deterministic = true
repeat_runs = 2
min_generated_tokens = 1
min_distinct_generated_tokens = 2
```

## Proof Command

```bash
RUST_LOG=warn cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- --device apple-m4-cpu-neon \
  slm-warm-session \
  --model target/apple-m4-slm-answer/SLM-M4-003/candidates/qwen2_5_0_5b_q8_0/qwen2.5-0.5b-instruct-q8_0.gguf \
  --corpus ci/quality/apple-m4-slm-quality-corpus.yaml \
  --strict-loader \
  --strict-tokenizer \
  --fail-on-quality \
  --require-determinism \
  --json-out target/apple-m4-slm-answer/SLM-M4-005/slm-quality-corpus.json
```

The command writes a local aggregate receipt and per-prompt receipts under `target/`. These are proof artifacts for the local operator and are not committed.

## Observed Local Output

The local proof produced `quality_summary.passed=true`, `determinism.checked=true`, `determinism.passed=true`, and three repeated-prompt determinism groups.

Representative normalized answers:

```text
2+2 equals 4.
The capital of France is Paris.
Rust is a powerful, versatile programming language that has gained popularity in recent years for
```

The Rust sentence is truncated by the 16-token smoke cap. That is acceptable for this item because the corpus gate checks valid, non-empty, non-degenerate text and deterministic token IDs, not broad chat quality or long-form completion quality.
