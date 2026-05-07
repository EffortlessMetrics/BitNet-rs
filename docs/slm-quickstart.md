# Getting Started with SLM Models

BitNet-rs supports inference on a range of Small Language Model (SLM) families
beyond the native 1-bit BitNet format. This guide walks you through downloading,
running, and chatting with Phi-4, Qwen, Gemma, Mistral, LLaMA, and other
supported models.

> **Pre-alpha notice:** SLM support loads SafeTensors weights and runs inference
> on CPU. Performance, accuracy, and GPU offload are under active development.

---

## Supported Architectures

| Family | Parameters | Format | Auth Required | Repo ID |
|--------|-----------|--------|---------------|---------|
| **Phi-4** | 14B | SafeTensors | Yes (`HF_TOKEN`) | `microsoft/phi-4` |
| **Phi-4-mini** | 3.8B | SafeTensors | No | `microsoft/Phi-4-mini-instruct` |
| **Qwen 2.5** | 7B | SafeTensors | No | `Qwen/Qwen2.5-7B-Instruct` |
| **Qwen 2.5** | 1.5B | SafeTensors | No | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Gemma 2** | 2B | SafeTensors | Yes (`HF_TOKEN`) | `google/gemma-2-2b-it` |
| **Mistral** | 7B | SafeTensors | No | `mistralai/Mistral-7B-Instruct-v0.3` |
| **LLaMA 3.2** | 1B | SafeTensors | Yes (`HF_TOKEN`) | `meta-llama/Llama-3.2-1B-Instruct` |
| **SmolLM2** | 1.7B | SafeTensors | No | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| **BitNet** | 2B | GGUF | No | `microsoft/bitnet-b1.58-2B-4T-gguf` |

---

## Quick Start

### 1. Download a Model

The `xtask download-model` command fetches model weights and tokenizer files
from HuggingFace:

```bash
# List all known models
cargo run --no-default-features -p xtask -- download-model --list

# Download SmolLM2 (~3.4 GB, no auth needed — good for testing)
cargo run --no-default-features -p xtask -- download-model --id HuggingFaceTB/SmolLM2-1.7B-Instruct

# Download Phi-4-mini (~7.6 GB, no auth needed — good balance of size/quality)
cargo run --no-default-features -p xtask -- download-model --id microsoft/Phi-4-mini-instruct

# Download Qwen 2.5 1.5B (~3 GB, no auth needed)
cargo run --no-default-features -p xtask -- download-model --id Qwen/Qwen2.5-1.5B-Instruct
```

For gated models (Phi-4 14B, Gemma, LLaMA), set your HuggingFace token first:

```bash
# Linux/macOS
export HF_TOKEN=hf_your_token_here

# Windows PowerShell
$env:HF_TOKEN = "hf_your_token_here"

# Then download
cargo run --no-default-features -p xtask -- download-model --id meta-llama/Llama-3.2-1B-Instruct
```

Models are saved to `./models/<repo-name>/`.

### 2. Run Inference

Always specify `--no-default-features --features cpu` (default features are empty by design):

```bash
# Basic inference with SmolLM2
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/HuggingFaceTB-SmolLM2-1.7B-Instruct/ \
  --prompt "What is 2+2?" \
  --max-tokens 32 --temperature 0.0

# Phi-4-mini with explicit SafeTensors format
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-Phi-4-mini-instruct/ \
  --model-format safetensors \
  --prompt "Explain quantum computing briefly." \
  --max-tokens 100

# Reduce log noise with RUST_LOG
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/Qwen-Qwen2.5-1.5B-Instruct/ \
  --prompt "Write a haiku about Rust." \
  --max-tokens 32 --temperature 0.7
```

### 3. Interactive Chat

The `chat` subcommand starts an interactive REPL with `/help`, `/clear`, and
`/metrics` commands:

```bash
# Chat with Phi-4-mini (template auto-detected from model metadata)
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/microsoft-Phi-4-mini-instruct/

# Explicitly set a prompt template
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/Qwen-Qwen2.5-1.5B-Instruct/ \
  --prompt-template qwen2.5

# LLaMA 3.2 chat
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/meta-llama-Llama-3.2-1B-Instruct/ \
  --prompt-template llama3-chat
```

---

## Model Family Reference

### Prompt Templates

Each model family has a recommended prompt template that structures the
conversation correctly. The CLI auto-detects the template from model metadata
when possible, but you can override with `--prompt-template`:

| Family | Template Name | Format Style | Aliases |
|--------|--------------|--------------|---------|
| Phi-4 / Phi-4-mini | `phi4-chat` | ChatML (`<\|im_start\|>` / `<\|im_end\|>`) | `phi4`, `chatml` |
| Phi-3 | `phi3-instruct` | `<\|system\|>` / `<\|user\|>` / `<\|assistant\|>` | `phi3` |
| Qwen 2.5 | `qwen2.5-chat` | ChatML (`<\|im_start\|>` / `<\|im_end\|>`) | `qwen2.5` |
| Qwen 2 | `qwen-chat` | ChatML | `qwen` |
| Gemma 2 | `gemma2-chat` | `<start_of_turn>` / `<end_of_turn>` | `gemma2`, `gemma-2` |
| Gemma | `gemma-chat` | `<start_of_turn>` / `<end_of_turn>` | `gemma` |
| Mistral | `mistral-chat` | `[INST]` ... `[/INST]` | `mistral` |
| Mixtral | `mixtral-instruct` | `[INST]` ... `[/INST]` | `mixtral` |
| LLaMA 3.2 | `llama3.2-chat` | LLaMA 3 special tokens | `llama32` |
| LLaMA 3 / 3.1 | `llama3-chat` | LLaMA 3 special tokens | `llama`, `llama3` |
| LLaMA 2 | `llama2-chat` | `[INST]<<SYS>>` / `<</SYS>>` | `llama2` |
| SmolLM | `smollm-chat` | Custom format | `smollm` |
| BitNet (default) | `instruct` | Simple Q&A | `raw` for no formatting |

### Architecture Details

| Family | Activation | Normalization | GQA | Context Length | Vocab Size |
|--------|-----------|---------------|-----|---------------|------------|
| Phi-4 | SiLU (SwiGLU) | RMSNorm | 40:10 (4:1) | 16K | 100,352 |
| Phi-4-mini | SiLU (SwiGLU) | RMSNorm | GQA | 128K | 100,352 |
| Qwen 2.5 | SiLU (SwiGLU) | RMSNorm | GQA | 32K–128K | 151,936 |
| Gemma 2 | GELU | RMSNorm | GQA | 8K | 256,000 |
| Mistral 7B | SiLU (SwiGLU) | RMSNorm | 32:8 (4:1) | 32K | 32,768 |
| LLaMA 3.2 | SiLU (SwiGLU) | RMSNorm | GQA | 128K | 128,256 |
| SmolLM2 | SiLU (SwiGLU) | RMSNorm | MHA | 8K | 49,152 |

---

## Hardware Requirements

Approximate RAM usage for SafeTensors models loaded in FP16/BF16:

| Model | Parameters | Download Size | RAM (FP16) | Recommended |
|-------|-----------|---------------|------------|-------------|
| SmolLM2 1.7B | 1.7B | ~3.4 GB | ~4 GB | 8 GB RAM |
| Qwen 2.5 1.5B | 1.5B | ~3 GB | ~4 GB | 8 GB RAM |
| LLaMA 3.2 1B | 1.3B | ~2.5 GB | ~3 GB | 8 GB RAM |
| Phi-4-mini 3.8B | 3.8B | ~7.6 GB | ~8 GB | 16 GB RAM |
| Gemma 2 2B | 2.6B | ~5 GB | ~6 GB | 16 GB RAM |
| Qwen 2.5 7B | 7.6B | ~15 GB | ~16 GB | 32 GB RAM |
| Mistral 7B | 7.2B | ~15 GB | ~16 GB | 32 GB RAM |
| Phi-4 14B | 14B | ~29 GB | ~30 GB | 64 GB RAM |

> **Tip:** For quick testing, start with SmolLM2 1.7B or Qwen 2.5 1.5B — both
> fit comfortably in 8 GB and require no HuggingFace token.

### Release Build (Recommended for Performance)

For production-grade performance, build with native SIMD optimizations:

```bash
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features cpu,full-cli
```

Then run the release binary directly:

```bash
./target/release/bitnet-cli run \
  --model models/microsoft-Phi-4-mini-instruct/ \
  --prompt "Hello, world!" --max-tokens 64
```

---

## Troubleshooting

### "Unknown model" error

The `--id` must match a known repo ID. Run `cargo run --no-default-features -p xtask -- download-model --list`
to see all registered models. You can also download any HuggingFace repo by
passing the full `owner/repo` path.

### "HF_TOKEN required" / 401 Unauthorized

Some models (Phi-4 14B, Gemma, LLaMA) are gated on HuggingFace and require
authentication:

1. Create a token at <https://huggingface.co/settings/tokens>
2. Accept the model's license on the model page
3. Set the token:
   ```bash
   export HF_TOKEN=hf_your_token_here          # Linux/macOS
   $env:HF_TOKEN = "hf_your_token_here"        # Windows PowerShell
   ```

### "No GGUF model found" when using SafeTensors models

If the CLI expects GGUF but you downloaded SafeTensors, pass `--model-format safetensors`
explicitly, or ensure you're pointing `--model` at the directory containing the
`.safetensors` files and `config.json`.

### Out of memory

SafeTensors models load in FP16/BF16 and require roughly 2× the parameter count
in bytes. See the [Hardware Requirements](#hardware-requirements) table and
choose a smaller model, or use the 1-bit BitNet GGUF format which is
significantly more memory-efficient.

### Slow inference

- Ensure you're using a **release build** with SIMD:
  ```bash
  RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
    cargo build --release --no-default-features --features cpu,full-cli
  ```
- SLM models use standard FP16 kernels, not the optimized 1-bit paths.
  Performance varies by model size and hardware.
- Set `RUST_LOG=warn` to reduce log overhead.

### Wrong or garbled output

Try specifying the correct prompt template explicitly:
```bash
--prompt-template phi4-chat    # for Phi-4 / Phi-4-mini
--prompt-template qwen2.5      # for Qwen 2.5
--prompt-template gemma2-chat   # for Gemma 2
--prompt-template mistral-chat  # for Mistral 7B
--prompt-template llama3-chat   # for LLaMA 3.x
```

The template structures the prompt in the format the model expects. Using the
wrong template often produces incoherent output.

---

## See Also

- [Quickstart (BitNet 1-bit models)](quickstart.md)
- [QK256 model usage](howto/use-qk256-models.md)
- [Environment variables](environment-variables.md)
- [CLI reference](specs/CLI.md)
- [Prompt template auto-detection](explanation/i2s-dual-flavor.md)
