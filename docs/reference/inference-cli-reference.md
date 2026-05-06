# Inference CLI Reference

Comprehensive reference for bitnet-rs inference commands (`run`, `chat`, `generate`).

## Commands

### run / generate

`generate` is an alias for `run`. Both accept identical flags.

```bash
RUST_LOG=warn cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" --max-tokens 8
```

### strict CPU proof command

Use this command when checking the CPU proof path. It requires a real model path,
an explicit tokenizer path, strict loader behavior, minimal fallback disabled,
greedy decoding, and JSON proof output.

```bash
BITNET_DISABLE_MINIMAL_LOADER=1 \
BITNET_STRICT_MODE=1 \
RUST_LOG=warn \
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --strict-loader \
  --strict-tokenizer \
  --prompt "Answer with a single digit: 2+2=" \
  --max-tokens 1 \
  --temperature 0.0 \
  --greedy \
  --json-out target/cpu-proof.json
```

Passing this command is a strict smoke/proof signal for the local CPU CLI path.
It is not a model-quality claim, a sustained performance claim, or receipt
validation by itself. Treat `target/cpu-proof.json` as a receipt-adjacent proof
artifact: it must show the enhanced loader path and must not show compatibility
fallback before follow-up CPU receipt work can promote the claim.

### chat

Interactive chat with REPL and auto-template detection.

```bash
RUST_LOG=warn cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json
```

**Chat commands:** `/help`, `/clear`, `/metrics`, `/exit` (or `/quit`, Ctrl+C)

## Prompt Templates

bitnet-rs supports 59+ prompt template variants with auto-detection.

**Detection priority:**

1. GGUF `chat_template` metadata (detects LLaMA-3 special tokens, generic instruct)
2. Model/tokenizer path heuristics (detects llama3, instruct, chat patterns)
3. Fallback to `instruct` template

**Override with `--prompt-template`:**

| Template | Use case |
|----------|----------|
| `auto` (default) | Detects from GGUF metadata or tokenizer |
| `raw` | No formatting — completion-style models |
| `instruct` | Q&A format — best for base models |
| `llama3-chat` | LLaMA-3 chat with system prompt |
| Others | `phi-4`, `qwen`, `gemma`, `mistral`, `deepseek`, and 50+ more |

### Examples

```bash
# Auto-detect (recommended)
RUST_LOG=warn cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is the capital of France?" --max-tokens 32

# Raw completion (no formatting)
RUST_LOG=warn cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt-template raw --prompt "2+2=" --max-tokens 16

# LLaMA-3 chat with system prompt
RUST_LOG=warn cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt-template llama3-chat \
  --system-prompt "You are a helpful assistant" \
  --prompt "Explain photosynthesis" \
  --max-tokens 128 --temperature 0.7 --top-p 0.95

# Instruct (Q&A format)
RUST_LOG=warn cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt-template instruct \
  --prompt "What is 2+2?" --max-tokens 8 --temperature 0.0 --greedy
```

## Flag Aliases

| Primary | Aliases |
|---------|---------|
| `--max-tokens N` | `--max-new-tokens N`, `--n-predict N` |
| `--stop "..."` | `--stop-sequence "..."`, `--stop_sequences "..."` |

## Sampling Controls

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature` | 1.0 | Sampling temperature (0.0 = greedy) |
| `--top-k` | 50 | Top-k filtering |
| `--top-p` | 1.0 | Nucleus sampling threshold |
| `--greedy` | false | Greedy decoding (equivalent to temperature 0.0) |
| `--seed` | random | RNG seed for reproducibility |
| `--repetition-penalty` | 1.0 | Repetition penalty factor |

### Deterministic Inference

```bash
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
RUST_LOG=warn cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model model.gguf --prompt "Test" --greedy --seed 42
```

## Stop Sequences

The engine evaluates stops in order:

1. **Token IDs** (`--stop-id N`) — O(1) lookup, checked first
2. **EOS** (from tokenizer or explicit) — fallback after token ID check
3. **String sequences** (`--stop "..."`) — rolling UTF-8-safe tail buffer

| Flag | Description |
|------|-------------|
| `--stop "..."` | String-based stop sequence (repeatable) |
| `--stop-id N` | Token ID stop (repeatable) |
| `--stop-string-window N` | Tail buffer size in bytes (default: 64) |

**Template defaults:**

- `raw`: no stop sequences
- `instruct`: `"\n\nQ:"`, `"\n\nHuman:"`
- `llama3-chat`: `<|eot_id|>` (auto-resolved to token ID 128009)

### Combined Stop Example

```bash
RUST_LOG=warn cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt-template llama3-chat \
  --prompt "What is 2+2?" \
  --stop "\n\n" --stop-id 128009 --max-tokens 32
```

## Logging

| Level | Effect |
|-------|--------|
| `RUST_LOG=warn` | Clean output (recommended) |
| `RUST_LOG=info` | Verbose (default) |
| `RUST_LOG=error` | Errors only |
