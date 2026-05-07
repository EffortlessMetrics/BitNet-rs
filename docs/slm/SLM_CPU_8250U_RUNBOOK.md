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
  --model models\slm\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --prompt-template raw `
  --prompt "2+2=" `
  --max-tokens 4 `
  --temperature 0.0 `
  --greedy `
  --strict-loader `
  --strict-tokenizer `
  --json-out ci\slm-cpu\intel-i5-8250u\qwen2_5_0_5b_2plus2.json
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
