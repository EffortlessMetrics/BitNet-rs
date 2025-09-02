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

### `score` - Perplexity Calculation

Calculate perplexity scores for model evaluation. Supports device and batch
size selection.

```bash
bitnet score --model model.gguf --file test.txt --batch-size 8 --device cuda
```

- `--batch-size SIZE` - Number of lines to process per batch
- `--device DEVICE` - Compute device (`cpu`, `cuda`, `metal`, `auto`)

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
4. **GPU offloading**: Use `--device cuda` when available
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