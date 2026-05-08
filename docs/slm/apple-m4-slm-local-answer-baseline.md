# Apple M4 SLM Local-Answer Baseline

The completed `apple-m4-slm-answer` campaign makes the practical Mac baseline explicit:

```text
prompt in
-> validated small dense instruct GGUF
-> Rust CLI
-> apple-m4-cpu-neon
-> strict loader and tokenizer routing
-> warm-session answer receipts
-> intelligible text out
```

This is the current user-facing Mac path. It is separate from the blocked BitNet local-answer lane and from future Metal acceleration work.

## Supported Baseline

| Field | Current baseline |
|---|---|
| Model family | Qwen2.5 0.5B Instruct GGUF |
| Proof artifact | Q8_0 companion used by the Rust-native warm-session proof |
| Backend label | `apple-m4-cpu-neon` |
| Runtime API | `cpu` |
| Fallback policy | `fallback_used=false` must be recorded |
| Prompt template | `qwen2.5` |
| Execution mode | Warm session: model and tokenizer load once, then multiple prompts run |

The proof artifact remains a local file under `target/` today:

```text
target/apple-m4-slm-answer/SLM-M4-003/candidates/qwen2_5_0_5b_q8_0/qwen2.5-0.5b-instruct-q8_0.gguf
```

Do not commit model binaries. Productized model cache commands are tracked as `M4-PROD-002`.

## Model Cache

`M4-PROD-002` adds a user cache for supported SLM artifacts. By default it uses:

```text
~/.cache/bitnet-rs/models/
```

The cache root can be overridden with `BITNET_MODEL_CACHE_DIR` or `--cache-dir`.

Supported model IDs:

```text
qwen2.5-0.5b-instruct-q8_0    Rust-native Apple M4 CPU/NEON baseline artifact
qwen2.5-0.5b-instruct-q4_k_m  Reference-good storage-preferred artifact; strict Rust execution remains unsupported
```

Useful commands:

```bash
bitnet model list
bitnet model fetch qwen2.5-0.5b-instruct-q8_0
bitnet model verify qwen2.5-0.5b-instruct-q8_0
bitnet model prune qwen2.5-0.5b-instruct-q8_0
```

Cache metadata records source repository, revision, filename, SHA256, size, quantization, tokenizer metadata, chat-template presence, and Apple M4 CPU/NEON support status. Fetch warns on low disk headroom and honors `--offline` / `BITNET_OFFLINE`.

## Working Commands

Fetch and verify the supported runtime artifact once:

```bash
bitnet model fetch qwen2.5-0.5b-instruct-q8_0
bitnet mac check
```

Ask one question through the supported Mac wrapper:

```bash
bitnet mac ask \
  --question "What is 2+2? Answer briefly." \
  --json-out target/apple-m4-productization/mac-ask.json
```

Run the deterministic warm-session validation corpus:

```bash
bitnet mac validate \
  --json-out target/apple-m4-productization/mac-validate.json
```

Run the operator timing profile set:

```bash
bitnet mac validate \
  --profile-set operator \
  --json-out target/apple-m4-productization/mac-operator-profiles.json
```

This writes a summary receipt plus per-profile warm-session receipts for
`warm_16`, `warm_32`, and `warm_64`. These profiles record cold model/tokenizer
load separately from warm prompt timing, show model/tokenizer reuse within each
profile, and keep latency numbers scoped to this model, backend, prompt set, and
machine context. The operator profile set intentionally runs one warm session per
token budget, so reuse is `within_profile`, not a single shared process across
all three budgets. The summary records `profile_set_model_loads=3` and
`profiles_loaded_independently=true` to keep that scope visible. These are not
broad performance or speedup claims.

Check answer or warm-session receipts:

```bash
bitnet mac receipts-check target/apple-m4-productization/mac-validate.json
bitnet mac receipts-check target/apple-m4-productization/mac-operator-profiles.json
```

The lower-level warm-session command remains available for debugging:

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
  --json-out target/apple-m4-productization/M4-PROD-001/slm-local-answer-baseline.json
```

Expected receipt properties:

```text
requested_backend = apple-m4-cpu-neon
selected_backend = apple-m4-cpu-neon
runtime_api = cpu
fallback_used = false
model_loaded_once = true
tokenizer_loaded_once = true
generated text and token IDs present
quality_summary.passed = true
determinism.passed = true
timing separates load, tokenize, prefill, decode, sampling, and total time
operator profile summaries include warm_16, warm_32, and warm_64 when requested
operator profile summaries disclose one warm session per token budget
operator profile summaries record profile_set_model_loads = 3
broad_performance_claim = false
speedup_claim = false
```

`bitnet mac ask` and `bitnet mac validate` intentionally route to
`apple-m4-cpu-neon`. Passing `--device apple-m4-metal`, `apple-m4-mpsgraph`, or
another accelerator label is rejected because full Metal/MPSGraph model
inference is not a proven user-facing path yet.

## Failure Boundaries

The Mac baseline must fail clearly when:

- the model file is missing;
- the model hash does not match the supported artifact manifest;
- strict loader or strict tokenizer mode would fall back;
- `selected_backend` differs from `requested_backend` without `fallback_reason`;
- `apple-m4-metal` is requested for full inference before a strict receipt proves it;
- MPSGraph output is counted as Neural Engine execution;
- QK256 support is inferred from SLM evidence.

## Claim Boundary

This path may claim:

```text
Rust-native Apple M4 CPU/NEON SLM local answers work for the validated model, corpus, backend, and receipt settings.
```

It must not claim:

```text
BitNet local-answer quality
full apple-m4-metal model inference
Neural Engine execution
QK256 on Apple Silicon
general M4 performance
```

Warm timing is measured for the recorded machine, model, corpus, backend, and run settings only. Broad performance claims require a later campaign item.
