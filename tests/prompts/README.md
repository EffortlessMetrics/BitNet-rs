# Test Prompts

This directory contains prompt files for manual smoke testing of BitNet.rs inference.

## Usage

```bash
# Test with raw template (no formatting)
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt-template raw \
  --prompt "2+2=" \
  --max-tokens 16 \
  --temperature 0.0

# Test with instruct template
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt-template instruct \
  --prompt "What is 2+2?" \
  --max-tokens 32 \
  --temperature 0.0

# Test with LLaMA-3 chat template
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt-template llama3-chat \
  --system-prompt "You are a helpful assistant" \
  --prompt "Explain photosynthesis in simple terms" \
  --max-tokens 128 \
  --temperature 0.7 \
  --top-p 0.95
```

## Expected Behavior

- **Math questions** (2+2): Should produce numeric answer "4"
- **Geography** (capital of France): Should produce "Paris"
- **Name completion**: Should produce common names
- **Open-ended** (photosynthesis): Should produce coherent explanation

## Deterministic Testing

For reproducible results:

```bash
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt-template instruct \
  --prompt "What is 2+2?" \
  --max-tokens 16 \
  --temperature 0.0 \
  --greedy
```
