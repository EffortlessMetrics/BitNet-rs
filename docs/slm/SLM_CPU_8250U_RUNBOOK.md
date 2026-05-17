# SLM CPU 8250U Runbook

The Intel Core i5-8250U lane is a conservative dense SLM correctness host. It is not a performance host.

## Default Settings

Use small deterministic runs:

```text
context: 256-512
max_new_tokens: 4-16
temperature: 0.0
greedy: true
threads: 4 for the current Qwen3 Q8_0 operator profile
batch: 1
```

The 4-thread default is evidence-scoped to the Qwen3 Q8_0 operator profile and
the 1/2/4/8-thread envelope recorded under
`ci/slm-cpu/intel-i5-8250u/2026-05-15/`. It is not a sustained-performance
claim. Record power and thermal context when available, but do not turn a cold
run into a sustained-performance claim.

## Candidate Preflight

Before inference, verify:

```text
model path exists
sha256 matches manifest
GGUF general.architecture is recorded
tokenizer source is gguf_metadata, explicit, or sibling
tokenizer.strict = true
context length is capped for the 8250U
quant format is recorded
dense adapter candidate is selected
fallback_used = false
BitNet QK256/I2_S path is not selected
```

## First Tiny Run Shape

The first run should be diagnosable even if the answer is wrong:

```powershell
$env:BITNET_STRICT_MODE = "1"
$env:RAYON_NUM_THREADS = "8"

cargo run --locked -p bitnet-cli --no-default-features --features "cpu,full-cli" -- `
  --device cpu `
  run `
  --model models\slm\Qwen3-0.6B-Q8_0.gguf `
  --prompt-template qwen `
  --prompt "2+2=" `
  --max-tokens 4 `
  --temperature 0.0 `
  --greedy `
  --strict-loader `
  --strict-tokenizer `
  --json-out ci\slm-cpu\intel-i5-8250u\qwen3_0_6b_2plus2.json
```

Required receipt facts:

```text
model.sha256 present
general.architecture present
tokenizer.source recorded
tokenizer.strict = true
selected_backend = cpu or cpu-rust
fallback_used = false
prompt_ids present
generated_ids present
decoded text present
```

If the decoded text is wrong, keep the artifact. The next step is reference divergence, not a performance claim.

## First-Token Divergence Triage

After `SLM-CPU-005`, the Qwen3 answer corpus evidence is execution evidence,
not answer-readiness evidence. Use `SLM-CPU-006` to capture a first-token
comparison against a known-good external runner before changing transformer
math.

Rust capture:

```powershell
$env:BITNET_STRICT_MODE = "1"
$env:BITNET_DISABLE_MINIMAL_LOADER = "1"
$env:RAYON_NUM_THREADS = "8"

cargo run --locked -p bitnet-cli --no-default-features --features "cpu,full-cli" -- `
  --device cpu `
  run `
  --model models\slm\Qwen3-0.6B-Q8_0.gguf `
  --prompt-template qwen `
  --prompt "What is 2+2? Answer with only the number." `
  --max-new-tokens 1 `
  --temperature 0.0 `
  --greedy `
  --deterministic `
  --strict-loader `
  --strict-tokenizer `
  --logits-dump-steps 1 `
  --logits-topk 10 `
  --assert-greedy `
  --json-out ci\slm-cpu\intel-i5-8250u\2026-05-07\qwen3-bitnet-rs-first-token-topk.json
```

For root-cause work after the first-token divergence is confirmed, add bounded
checkpoint tracing to the same run:

```powershell
  --qwen-trace-jsonl ci\slm-cpu\intel-i5-8250u\2026-05-07\qwen3-bitnet-rs-checkpoints.jsonl `
  --qwen-trace-layer 0 `
  --qwen-trace-full-prompt
```

The JSONL trace records summaries only: shape, dtype, finite counts, mean, RMS,
min/max, checksum, and a short sample. Use it to find the first drift against a
known-good reference checkpoint pack; do not treat it as answer-quality or
throughput evidence.

Reference comparison:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features "cpu,full-cli" -- `
  reference-compare `
  --artifact ci\slm-cpu\intel-i5-8250u\2026-05-07\qwen3-reference-compare.json `
  --json-out ci\slm-cpu\intel-i5-8250u\2026-05-07\qwen3-reference-divergence-validation.json
```

Do not pass `--require-match` while the lane is intentionally capturing a
failure. A valid divergence artifact should include identical model SHA,
prompt text, Qwen template, BOS policy, prompt IDs, generated IDs, decoded
text, chosen token, and first-step top-k evidence from both sides where the
external runner can provide it.

Use the validator classification to choose the next fix:

```text
prompt_tokenizer_template
  -> Qwen template, BOS/EOS policy, chat markers, tokenizer extraction.

logits_or_shared_transformer_math
  -> Q8_0 dequantization, tensor orientation, Q/K/V/O projection shape,
     RoPE, RMSNorm, GQA, output head, or vocab indexing.

sampler
  -> greedy argmax, temperature=0 path, tie-breaking, EOS/stop handling.

tokenizer_decode
  -> byte fallback, special-token filtering, UTF-8 cleanup.
```

## Post-008 Artifact Revalidation

`SLM-CPU-008` landed the Qwen3 architecture-default fix in #4434. That merge
did not include a fresh real-model i5-8250U artifact because the verified GGUF
was not present in that worktree/cache. Before advancing to the tiny answer
corpus, run `SLM-CPU-008R` to verify the post-#4434 runtime against the actual
Qwen3-0.6B Q8_0 artifact.

Original prompt policy:

```powershell
$env:BITNET_STRICT_MODE = "1"
$env:BITNET_DISABLE_MINIMAL_LOADER = "1"
$env:RAYON_NUM_THREADS = "8"

cargo run --locked -p bitnet-cli --no-default-features --features "cpu,full-cli" -- `
  --device cpu `
  run `
  --model models\slm\Qwen3-0.6B-Q8_0.gguf `
  --prompt-template qwen `
  --prompt "What is 2+2? Answer with only the number." `
  --max-new-tokens 1 `
  --temperature 0.0 `
  --greedy `
  --deterministic `
  --strict-loader `
  --strict-tokenizer `
  --logits-dump-steps 1 `
  --logits-topk 10 `
  --assert-greedy `
  --qwen-trace-jsonl ci\slm-cpu\intel-i5-8250u\2026-05-07\qwen3-post-008-original-trace.jsonl `
  --qwen-trace-layer 0 `
  --json-out ci\slm-cpu\intel-i5-8250u\2026-05-07\qwen3-post-008-original-first-token.json
```

No-thinking prompt policy:

Use `--no-think` when a Qwen3 first-token comparison is meant to exercise
answer mode rather than thinking mode. Run the same artifact protocol with the
no-thinking rendered prompt and refresh the known-good reference first:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features "cpu,full-cli" -- `
  --device cpu `
  run `
  --model models\slm\Qwen3-0.6B-Q8_0.gguf `
  --prompt-template qwen `
  --no-think `
  --prompt "What is 2+2? Answer with only the number." `
  --max-new-tokens 1 `
  --temperature 0.0 `
  --greedy `
  --deterministic `
  --strict-loader `
  --strict-tokenizer `
  --logits-dump-steps 1 `
  --logits-topk 10 `
  --assert-greedy `
  --json-out ci\slm-cpu\intel-i5-8250u\2026-05-07\qwen3-post-008-no-think-first-token.json
```

For either policy, record the model SHA, rendered prompt, prompt IDs,
generated IDs, decoded text, chosen token, first-step top-k, selected backend,
kernel/backend provenance, tokenizer source, and `fallback_used = false`.
Validate the result with `reference-compare`. Do not judge `--no-think` against
the older SLM-CPU-006B reference unless the known-good reference was regenerated
from the exact no-thinking rendered prompt and BOS policy. The receipt records
`qwen_no_think = true` when this policy is active.

If the original prompt now emits token `19` / `4`, first-token parity is
revalidated and the lane can move to the tiny corpus. If either policy still
diverges, keep the artifact as the next root-cause input. This revalidation is
not an answer-quality or throughput claim.

## Observed Qwen3 Q8_0 Boundary

On the i5-8250U, the official `Qwen3-0.6B-Q8_0.gguf` artifact verifies against the pinned SHA256 and reaches the strict CPU loader with `selected_backend = cpu-rust` and `fallback_used = false`.

`SLM-CPU-002B` adds eager dense GGUF Q8_0 dequantization in the model loader. With that support, the artifact reaches full strict tensor loading:

```text
Successfully loaded 310 tensors (detected 0 QK256 tensors)
```

The current boundary is after tensor loading and before inference:

```text
shape mismatch for layers.0.attention.q_proj.weight, expected: [1024, 1024], got: [2048, 1024]
```

This reflects Qwen3 attention dimensions that are not yet represented by the current transformer construction path. Do not claim a tiny dense CPU run until the same command emits prompt IDs, generated IDs, and decoded text.
