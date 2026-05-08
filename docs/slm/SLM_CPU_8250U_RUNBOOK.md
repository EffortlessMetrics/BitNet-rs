# SLM CPU 8250U Runbook

The Intel Core i5-8250U lane is a conservative dense SLM correctness host. It is not a performance host.

## Default Settings

Use small deterministic runs:

```text
context: 256-512
max_new_tokens: 4-16
temperature: 0.0
greedy: true
threads: 8
batch: 1
```

Record power and thermal context when available, but do not turn a cold run into a sustained-performance claim.

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
