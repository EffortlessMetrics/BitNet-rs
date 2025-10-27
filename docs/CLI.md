# BitNet CLI Documentation

The BitNet CLI provides comprehensive tools for model inference, validation, conversion, and benchmarking.

## Installation

```bash
# Install from crates.io
cargo install bitnet-cli

# Build from source
cargo build --release -p bitnet-cli --no-default-features --features cpu
```

## Commands

### `run` - Simple Text Generation

Generate text using a BitNet model with various sampling options.

```bash
bitnet run --model model.gguf --prompt "Hello, world!"

# Options:
#   --tokenizer PATH        External tokenizer (overrides embedded)
#   --max-new-tokens N      Maximum tokens to generate (default: 100)
#   --temperature F         Sampling temperature (default: 0.8)
#   --top-k N              Top-k sampling (default: 50)
#   --top-p F              Top-p (nucleus) sampling (default: 0.9)
#   --repetition-penalty F  Repetition penalty (default: 1.0)
#   --seed N               Random seed for reproducibility
#   --greedy               Use greedy decoding (overrides temperature)
#   --deterministic        Enable deterministic mode (single-threaded)
#   --threads N            Number of threads (0 = all cores)
#   --bos                  Insert BOS token at start
#   --json-out PATH        Output results as JSON
#   --dump-ids             Dump token IDs to stdout
#   --dump-logit-steps N   Dump logits for first N generation steps
#   --logits-topk K        Top-k tokens in logit dump (default: 10)
```

### `compat-check` - Validate GGUF Files

Quickly validate GGUF file compatibility without loading the full model.

**Exit Codes:**
| Code | Meaning |
|------|---------|
| 0 | Valid GGUF file |
| 1 | I/O error (file not found, permission denied, etc.) |
| 2 | Malformed file (bad magic, short header, corrupt) |
| 3 | Unsupported GGUF version |
| 4 | Strict mode: unsupported version or suspicious counts |

```bash
# Human-readable output
bitnet compat-check model.gguf

# Strict validation (exits 4 if suspicious)
bitnet compat-check model.gguf --strict
# Output:
# File:      model.gguf
# Status:    ✓ Valid GGUF
# Version:   2 (supported)
# Tensors:   1234
# KV pairs:  56

# Show key-value metadata
bitnet compat-check model.gguf --show-kv
# Output:
# File:      model.gguf
# Status:    ✓ Valid GGUF
# Version:   2 (supported)
# Tensors:   1234
# KV pairs:  56
#
# Metadata (showing 20 of 56):
#   model.arch                     = "llama"
#   model.vocab_size               = 32000
#   model.layers                   = 12
#   ...

# Custom KV limit
bitnet compat-check model.gguf --show-kv --kv-limit 5

# JSON output for scripting
bitnet compat-check model.gguf --json
# {
#   "path": "model.gguf",
#   "status": "valid",
#   "gguf": {
#     "version": 2,
#     "n_tensors": 1234,
#     "n_kv": 56
#   },
#   "compatibility": {
#     "supported_version": true,
#     "tensors_reasonable": true,
#     "kvs_reasonable": true
#   }
# }

# JSON with metadata
bitnet compat-check model.gguf --json --show-kv --kv-limit 3
# {
#   "path": "model.gguf",
#   "status": "valid",
#   "gguf": { ... },
#   "compatibility": { ... },
#   "metadata": [
#     {"key": "model.arch", "value": "llama"},
#     {"key": "model.vocab_size", "value": 32000},
#     {"key": "model.layers", "value": 12}
#   ]
# }
```

### `inspect` - Model Metadata

Inspect model metadata without loading tensors.

```bash
bitnet inspect --model model.gguf

# JSON output with detailed metadata
bitnet inspect --model model.gguf --json
```

### `tokenize` - Text Tokenization

Tokenize text and output token IDs.

```bash
# Inline text
bitnet tokenize --model model.gguf --text "Hello, world!"

# From file
bitnet tokenize --model model.gguf --file input.txt

# With BOS token
bitnet tokenize --model model.gguf --text "Hello" --bos

# Output to JSON file
bitnet tokenize --model model.gguf --text "Hello" --json-out tokens.json
```

### `score` - Teacher-Forcing Evaluation and Perplexity

Perform teacher-forcing evaluation with perplexity and negative log-likelihood (NLL) calculation. This command evaluates model quality by computing the probability the model assigns to ground truth text sequences.

**Core Functionality:**
- Real teacher-forcing evaluation using the inference engine
- Negative Log-Likelihood (NLL) and Perplexity (PPL) calculation
- Device-aware processing with automatic fallback
- Batch processing for improved throughput
- Structured JSON output for automation

```bash
# Basic perplexity calculation
bitnet score --model model.gguf --file test.txt

# Device selection with batch processing
bitnet score --model model.gguf --file test.txt --device cuda --batch-size 8

# Use external tokenizer
bitnet score --model model.gguf --file test.txt --tokenizer tokenizer.json

# Limit evaluation to first 1000 tokens
bitnet score --model model.gguf --file test.txt --max-tokens 1000

# Save results to JSON file
bitnet score --model model.gguf --file test.txt --json-out results.json
```

**Arguments:**
- `--model PATH` - GGUF model file (required)
- `--file PATH` - Text file with one prompt per line (required)
- `--tokenizer PATH` - External SentencePiece tokenizer (overrides embedded)
- `--max-tokens N` - Cap on tokens evaluated (0 = unlimited, default: 0)
- `--device DEVICE` - Compute device: `cpu`, `cuda`, `metal`, `auto` (default: auto)
- `--batch-size SIZE` - Lines processed per batch (default: 1)
- `--json-out PATH` - Output JSON file (stdout if omitted)

**Device Selection:**
- `cpu` - Force CPU computation
- `cuda` / `gpu` - Force CUDA GPU (fails if unavailable)
- `metal` - Force Metal GPU (not currently supported)
- `auto` - Try CUDA, fallback to CPU (recommended)

**Input Format:**
The input file should contain one text sequence per line:
```
The quick brown fox jumps over the lazy dog.
To be or not to be, that is the question.
Machine learning is a subset of artificial intelligence.
```

**Output Format:**
```json
{
  "type": "score",
  "model": "model.gguf",
  "dataset": "test.txt",
  "tokens": 1234,
  "mean_nll": 2.345,
  "ppl": 10.42,
  "latency": {
    "total_ms": 1500.0
  },
  "tokenizer": {
    "type": "sentencepiece",
    "origin": "embedded"
  },
  "gen_policy": {
    "bos": false,
    "temperature": 0.0,
    "seed": null
  },
  "counts": {
    "n_kv": 56,
    "n_tensors": 1234,
    "unmapped": 0
  }
}
```

**Key Metrics:**
- `tokens` - Total number of tokens evaluated (T-1, where T is sequence length)
- `mean_nll` - Mean negative log-likelihood per token
- `ppl` - Perplexity (exp(mean_nll))
- `latency.total_ms` - Total evaluation time in milliseconds

**Examples:**

```bash
# Evaluate multiple models on the same dataset
for model in *.gguf; do
  echo "Evaluating $model..."
  bitnet score --model "$model" --file validation.txt --json-out "${model%.gguf}_scores.json"
done

# Compare CPU vs GPU performance
bitnet score --model model.gguf --file test.txt --device cpu --json-out cpu_results.json
bitnet score --model model.gguf --file test.txt --device cuda --json-out gpu_results.json

# Large dataset processing with batching
bitnet score --model large-model.gguf \
  --file large-dataset.txt \
  --batch-size 16 \
  --device cuda \
  --max-tokens 10000 \
  --json-out evaluation.json

# Extract specific metrics with jq
bitnet score --model model.gguf --file test.txt | jq '.ppl'
bitnet score --model model.gguf --file test.txt | jq '.latency.total_ms'
```

### `config` - Configuration Management

Manage CLI configuration settings.

```bash
# Show current configuration
bitnet config show

# Set a configuration value
bitnet config set device cuda

# Reset to defaults
bitnet config reset
```

### `info` - System Information

Display system and build information.

```bash
bitnet info
```

## Global Options

These options can be used with any command:

- `--config PATH` - Configuration file path
- `--device DEVICE` - Device to use (cpu, cuda, auto)
- `--log-level LEVEL` - Log level (trace, debug, info, warn, error)
- `--threads N` - Number of CPU threads
- `--batch-size SIZE` - Batch size for processing

## Environment Variables

- `BITNET_GGUF` - Default model path for testing
- `BITNET_DETERMINISTIC` - Enable deterministic mode
- `BITNET_SEED` - Set random seed
- `RAYON_NUM_THREADS` - Control CPU parallelism

## Examples

### Basic Inference
```bash
bitnet run --model llama-3b.gguf --prompt "What is the capital of France?"
```

### Deterministic Generation
```bash
bitnet run --model model.gguf \
  --prompt "Define entropy." \
  --deterministic \
  --greedy \
  --threads 1 \
  --seed 42
```

### Batch Processing
```bash
# Process multiple prompts from file
cat prompts.txt | while read prompt; do
  bitnet run --model model.gguf --prompt "$prompt" --json-out >> results.jsonl
done
```

### Model Validation Pipeline
```bash
# 1. Check GGUF compatibility
bitnet compat-check model.gguf || exit 1

# 2. Inspect metadata
bitnet inspect --model model.gguf --json | jq '.quantization'

# 3. Test tokenization
bitnet tokenize --model model.gguf --text "Test" --json-out test.json

# 4. Run inference
bitnet run --model model.gguf --prompt "Hello" --max-new-tokens 10
```

### Automation with JSON Output
```bash
# Get model info programmatically
MODEL_VERSION=$(bitnet compat-check model.gguf --json | jq -r '.gguf.version')
if [ "$MODEL_VERSION" -gt 3 ]; then
  echo "Warning: Unsupported GGUF version $MODEL_VERSION"
fi

# Check tensor count
TENSOR_COUNT=$(bitnet compat-check model.gguf --json | jq -r '.gguf.n_tensors')
echo "Model has $TENSOR_COUNT tensors"
```

## Error Handling

The CLI provides detailed error messages with recovery suggestions:

- **Invalid GGUF**: Shows specific validation failure (bad magic, unsupported version, etc.)
- **Missing tokenizer**: Suggests using `--tokenizer` flag or embedding tokenizer in GGUF
- **Resource limits**: Recommends batch size or thread adjustments
- **CUDA errors**: Provides GPU troubleshooting steps

## Shell Completions

Generate shell completions for your shell:

```bash
# Bash
bitnet --completions bash > ~/.local/share/bash-completion/completions/bitnet

# Zsh
bitnet --completions zsh > ~/.zfunc/_bitnet

# Fish
bitnet --completions fish > ~/.config/fish/completions/bitnet.fish

# PowerShell
bitnet --completions powershell > $PROFILE\bitnet.ps1
```

## Performance Tips

1. **Use native CPU features**: Build with `-C target-cpu=native` for optimal SIMD
2. **Thread tuning**: Set `--threads` to physical core count for best performance
3. **Batch processing**: Use larger `--batch-size` for throughput
4. **GPU offloading**: Use `--device cuda` (requires gpu feature compilation)
5. **Memory mapping**: Models are memory-mapped by default for efficiency

## Troubleshooting

### Model Won't Load
```bash
# Validate GGUF format
bitnet compat-check model.gguf

# Check for corruption
sha256sum model.gguf
```

### Tokenization Issues
```bash
# Test tokenizer separately
bitnet tokenize --model model.gguf --text "test" --json-out test.json

# Use external tokenizer if needed
bitnet run --model model.gguf --tokenizer tokenizer.json --prompt "test"
```

### Performance Issues
```bash
# Check system info
bitnet info

# Monitor resource usage
bitnet run --model model.gguf --prompt "test" --log-level debug
```

## See Also

- [README.md](README.md) - Project overview
- [CLAUDE.md](CLAUDE.md) - Development guide
- [VALIDATION.md](VALIDATION.md) - Testing documentation
