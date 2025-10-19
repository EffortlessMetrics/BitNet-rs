# CLAUDE.md

Essential guidance for working with the BitNet.rs neural network inference codebase.

## Quick Reference

### Essential Commands

```bash
# Build (default features are EMPTY - always specify features)
cargo build --no-default-features --features cpu     # CPU inference
cargo build --no-default-features --features gpu     # GPU inference

# Build with CPU optimization (recommended for production performance)
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features cpu,full-cli

# Test
cargo test --workspace --no-default-features --features cpu

# Quality
cargo fmt --all && cargo clippy --all-targets --all-features -- -D warnings

# Development workflow
cargo run -p xtask -- download-model

# Inference with prompt templates (clean output with reduced log noise)
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt-template instruct \
  --prompt "What is 2+2?" \
  --max-tokens 32 \
  --temperature 0.7

# Interactive chat (auto-detects prompt template)
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json

# Model validation (3-stage: LayerNorm, projection, linguistic sanity)
./scripts/validate_gguf.sh <model.gguf> <tokenizer.json>  # Full validation pipeline
cargo run -p bitnet-cli --features cpu,full-cli -- inspect --ln-stats --gate auto <model.gguf>

# Model export and validation (clean GGUF with F16 LayerNorm)
./scripts/export_clean_gguf.sh <model_dir> <tokenizer.json> <output_dir>  # Export + validate

# Inference receipt verification (honest compute gates)
cargo run -p xtask -- benchmark --model <model.gguf> --tokens 128  # Run benchmark, writes ci/inference.json
cargo run -p xtask -- verify-receipt             # Verify receipt against quality gates
cargo run -p xtask -- verify-receipt --require-gpu-kernels  # Explicitly require GPU kernels
# The benchmark command automatically writes production receipts with measured TPS and real kernel IDs.
# Receipt verification includes:
#   - Schema validation (v1.0.0)
#   - compute_path == "real" (no mock inference)
#   - Kernel ID hygiene (no empty strings, length ≤ 128, count ≤ 10K)
#   - Auto-GPU enforcement: backend="cuda" requires GPU kernels automatically

# SafeTensors to GGUF converter (Rust st2gguf - preferred)
cargo run -p bitnet-st2gguf -- --input model.safetensors --output model.gguf --strict
cargo run --release -p bitnet-st2gguf -- --help      # See all options
```

## Core Architecture

### Design Principles

1. **Feature-Gated**: Default features are **EMPTY** - always specify `--features cpu|gpu`
2. **Zero-Copy**: Memory-mapped models, efficient lifetime management
3. **Device-Aware**: Automatic GPU/CPU selection with graceful fallback
4. **Cross-Validated**: Systematic comparison with C++ reference

### Key Crates

- `bitnet` (root): Main library with unified API
- `bitnet-inference`: Autoregressive generation engine
- `bitnet-quantization`: 1-bit quantization (I2_S, TL1, TL2)
- `bitnet-kernels`: SIMD/CUDA compute kernels
- `bitnet-models`: GGUF/SafeTensors model loading
- `bitnet-tokenizers`: Universal tokenizer with auto-discovery
- `bitnet-st2gguf`: SafeTensors to GGUF converter with LayerNorm preservation
- `bitnet-cli`: Command-line interface and utilities
- `crossval`: C++ reference validation framework

## Key Configurations

### MSRV: 1.90.0 (Rust 2024 edition)

### Feature Flags

- `cpu`: SIMD-optimized CPU inference (AVX2/AVX-512/NEON)
- `gpu`: CUDA acceleration with mixed precision (FP16/BF16)
- `cuda`: Backward-compatible alias for `gpu` (temporary - prefer `gpu` in new code)
- `ffi`: C++ FFI bridge for gradual migration
- `crossval`: Cross-validation against Microsoft BitNet C++

**Important**: Always use unified GPU predicate in code:

```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_function() { /* ... */ }
```

Use `bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime}` for runtime checks.

### Quantization Support

BitNet.rs supports multiple I2_S quantization formats with automatic flavor detection:

- **I2_S BitNet32-F16**: Production 2-bit signed quantization (32-elem blocks, inline
  F16 scales) - ✅ CPU/GPU
- **I2_S QK256 (GGML)**: Pure Rust 2-bit signed quantization (256-elem blocks, separate
  scales) - ✅ MVP (scalar)
  - Automatic flavor detection from tensor size
  - Routes to C++ via FFI for validation when `BITNET_CPP_DIR` set
  - See: `docs/howto/use-qk256-models.md` for usage guide
  - See: `docs/explanation/i2s-dual-flavor.md` for architecture details
- **TL1/TL2**: Table lookup quantization with device-aware selection (ARM NEON / x86 AVX)
- **IQ2_S**: GGML-compatible via FFI bridge

**Parity Validation:**

```bash
# One-command smoke test (tests both BitNet and QK256 formats)
scripts/parity_smoke.sh models/model.gguf

# Full cross-validation with receipts
BITNET_CPP_DIR=/path/to/bitnet.cpp cargo run -p xtask -- crossval
```

**Receipts show parity metrics:**

```json
{
  "parity": {
    "cpp_available": true,
    "cosine_similarity": 0.9923,
    "exact_match_rate": 1.0,
    "status": "ok"
  }
}
```

## Documentation Structure

### Getting Started

- `docs/quickstart.md`: 5-minute setup guide
- `docs/getting-started.md`: Comprehensive introduction
- `docs/explanation/FEATURES.md`: Feature flag documentation

### Development

- `docs/development/build-commands.md`: Comprehensive build reference
- `docs/development/gpu-development.md`: CUDA development guide
- `docs/development/test-suite.md`: Testing framework
- `docs/development/validation-framework.md`: Quality assurance
- `docs/development/xtask.md`: Developer tooling
- `docs/howto/export-clean-gguf.md`: Clean GGUF export and validation
- `docs/howto/validate-models.md`: Complete validation workflow guide

### Architecture

- `docs/architecture-overview.md`: System design and components
- `docs/reference/quantization-support.md`: Quantization algorithms
- `docs/reference/validation-gates.md`: Validation system technical reference
- `docs/gpu-kernel-architecture.md`: CUDA kernel design
- `docs/tokenizer-architecture.md`: Universal tokenizer system

### Operations

- `docs/performance-benchmarking.md`: Performance testing
- `docs/health-endpoints.md`: Monitoring and observability
- `docs/GPU_SETUP.md`: GPU configuration
- `docs/environment-variables.md`: Runtime configuration
- `docs/baselines/`: Model baselines and fingerprints

## Inference Usage

### Prompt Templates and Auto-Detection

BitNet.rs supports multiple prompt templates for optimal model behavior. The CLI
automatically detects the appropriate template using:

1. **Priority 1**: GGUF `chat_template` metadata (detects LLaMA-3 special tokens and
   generic instruct patterns)
2. **Priority 2**: Model/tokenizer path heuristics (detects llama3, instruct, chat
   patterns)
3. **Priority 3**: Fallback to Instruct template (safer than Raw for most models)

**Note**: As of v0.9.x, the default auto-detection fallback changed from `raw` to
`instruct` for better out-of-box experience with instruction-tuned models. Use
`--prompt-template raw` if you need raw completion behavior.

You can override auto-detection with `--prompt-template`:

```bash
# Auto-detect template (recommended - uses GGUF metadata and tokenizer hints)
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is the capital of France?" \
  --max-tokens 32

# Raw (no formatting) - for completion-style models
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt-template raw \
  --prompt "2+2=" \
  --max-tokens 16

# Instruct (Q&A format) - for instruction-tuned models
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt-template instruct \
  --prompt "What is the capital of France?" \
  --max-tokens 32

# LLaMA-3 chat format - for LLaMA-3 models
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt-template llama3-chat \
  --system-prompt "You are a helpful assistant" \
  --prompt "Explain photosynthesis" \
  --max-tokens 128 \
  --temperature 0.7 \
  --top-p 0.95
```

### Flag Aliases

BitNet.rs CLI supports convenient aliases for common flags to improve compatibility with other tools:

```bash
# --max-tokens (primary) with backward-compatible aliases
# All three forms are equivalent:
--max-tokens 32              # Primary flag
--max-new-tokens 32          # Alias (common in other tools)
--n-predict 32               # Alias (GGML compatibility)

# --stop (primary) with backward-compatible aliases
# All three forms are equivalent:
--stop "</s>"                # Primary flag
--stop-sequence "</s>"       # Alias
--stop_sequences "</s>"      # Alias
```

### Sampling Controls

```bash
# Greedy decoding (deterministic) - with reduced log noise
RUST_LOG=warn cargo run -p bitnet-cli -- run --model model.gguf --prompt "Test" \
  --temperature 0.0 --greedy --seed 42

# Greedy math sanity check (validates model correctness)
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Answer with a single digit: 2+2=" \
  --max-tokens 1 \
  --temperature 0.0 --greedy

# Nucleus sampling (creative) - with clean output
RUST_LOG=warn cargo run -p bitnet-cli -- run --model model.gguf --prompt "Test" \
  --temperature 0.7 --top-p 0.95 --top-k 50 --repetition-penalty 1.05

# Deterministic inference (reproducible)
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
RUST_LOG=warn cargo run -p bitnet-cli -- run --model model.gguf --prompt "Test" --greedy --seed 42
```

**Logging:**

- `RUST_LOG=warn`: Suppresses debug/info logs, shows only warnings/errors (recommended for clean output)
- `RUST_LOG=info`: Shows general information (default verbose)
- `RUST_LOG=error`: Only shows errors (minimal output)

### Stop Sequences

```bash
# Manual stop sequences
--stop "</s>" --stop "\n\nQ:"

# Template defaults (automatic based on --prompt-template)
# - raw: no stop sequences
# - instruct: stops on "\n\nQ:", "\n\nHuman:"
# - llama3-chat: stops on "<|eot_id|>", "<|end_of_text|>"
```

### Interactive Chat

BitNet.rs provides an interactive chat mode with REPL and built-in commands. The CLI
auto-detects the appropriate chat template and displays streaming responses:

```bash
# Interactive chat with auto-template detection
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json

# Chat with specific template override
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt-template llama3-chat

# Chat with custom sampling parameters
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --temperature 0.7 --top-p 0.95
```

**Available chat commands:**

- `/help` - Show available commands
- `/clear` - Clear conversation history
- `/metrics` - Display performance metrics
- `/exit` or `/quit` - Exit chat mode (also Ctrl+C)

## Common Workflows

### Inference

```bash
# Generate text using `run` or its alias `generate`
bitnet run --model model.gguf --prompt "Hello world" --max-tokens 32
bitnet generate --model model.gguf --prompt "Hello world" --max-tokens 32  # Same as above

# Interactive chat (auto-detects template)
bitnet chat --model model.gguf --tokenizer tokenizer.json
```

**Note**: The `generate` subcommand is an alias for `run`, providing semantic clarity when using the CLI directly.

### Development Workflow

```bash
# Standard development cycle
cargo test --workspace --no-default-features --features cpu
cargo fmt --all && cargo clippy --all-targets --all-features -- -D warnings

# Cross-validation (when changing inference)
# First-time setup: provision model
cargo run -p xtask -- download-model
# Tests auto-discover model in models/ directory
cargo test -p bitnet-models --no-default-features --features crossval
# Or use custom model path
export BITNET_GGUF="path/to/model.gguf"
cargo run -p xtask -- crossval

# Model operations
cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf
cargo run -p bitnet-cli -- compat-check model.gguf
```

### Model Validation Workflow

```bash
# 1. Validate existing GGUF (3-stage: LayerNorm, projection, linguistic sanity)
./scripts/validate_gguf.sh models/model.gguf models/tokenizer.json

# 2. Inspect LayerNorm and projection statistics (architecture-aware)
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto models/model.gguf

# 3. Strict mode (fail on warnings - for CI/CD)
BITNET_STRICT_MODE=1 \
  cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto models/model.gguf

# 4. Custom validation policy
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate policy \
  --policy examples/policies/custom-model.yml \
  --policy-key my-model:f16 \
  models/model.gguf

# 5. Export clean GGUF from SafeTensors (F16 with LayerNorm preservation)
./scripts/export_clean_gguf.sh \
  models/safetensors-checkpoint \
  models/tokenizer.json \
  models/clean

# Then validate the exported model
./scripts/validate_gguf.sh models/clean/clean-f16.gguf models/tokenizer.json
```

**See also:** `docs/howto/validate-models.md` for complete validation guide.

### Troubleshooting

- FFI linker errors: Use `--no-default-features --features cpu` or
  `cargo xtask fetch-cpp`
- CUDA issues: Ensure CUDA toolkit installed and `nvcc` in PATH
- Model validation: `cargo run -p bitnet-cli -- compat-check model.gguf`
- GPU detection: Run `cargo run -p xtask -- preflight` to check GPU compilation and
  runtime availability
- Silent CPU fallback: Check receipts for GPU kernel IDs (`gemm_*`, `i2s_gpu_*`); use
  `BITNET_GPU_FAKE` for testing
- Feature gate mismatches: Always use `#[cfg(any(feature = "gpu", feature = "cuda"))]`
  pattern
- Template auto-detection: If the wrong template is detected, override with
  `--prompt-template` (raw/instruct/llama3-chat). Check GGUF metadata with
  `cargo run -p bitnet-cli -- compat-check model.gguf --show-kv` to diagnose detection
  priority issues.
- LayerNorm validation errors: If you see "suspicious LayerNorm gamma" warnings, your
  GGUF has quantized LN weights (should be FP16/FP32)
  - **RMS-based validation**: Validator checks LayerNorm gamma RMS (root mean square)
    with architecture-aware envelopes
  - **Proper fix**: Regenerate GGUF with LayerNorm weights in float format (not
    quantized)
  - **Diagnosis**: Use `cargo run -p bitnet-cli --features cpu,full-cli -- inspect
    --ln-stats --gate auto model.gguf`
  - **Validation modes**: `none` (skip), `auto` (architecture detection), `policy`
    (custom rules)
  - **Temporary workaround**: Policy-driven corrections for known-bad models (see
    `docs/explanation/correction-policy.md`)
    - Requires both `BITNET_CORRECTION_POLICY=/path/to/policy.yml` and
      `BITNET_ALLOW_RUNTIME_CORRECTIONS=1`
    - CI blocks correction flags - use only for fingerprinted known-bad models
  - **Strict mode**: `BITNET_STRICT_MODE=1` will fail immediately on suspicious LN
    weights (exit code 8)
  - **See also**: `docs/howto/validate-models.md` for complete troubleshooting guide

## Environment Variables

### Inference Configuration

- `BITNET_DETERMINISTIC=1 BITNET_SEED=42`: Reproducible inference
- `BITNET_GGUF`: Model path override for cross-validation and inference (auto-discovers
  `models/` if not set)
- `RAYON_NUM_THREADS=1`: Single-threaded determinism
- `BITNET_GPU_FAKE=cuda|none`: Override GPU detection for deterministic testing
  (Issue #439)

### Validation Configuration

- `BITNET_STRICT_MODE=1`: Enable strict validation (fails on LayerNorm/projection
  warnings, exit code 8)
- `BITNET_VALIDATION_GATE=none|auto|policy`: Validation mode (default: `auto`)
- `BITNET_VALIDATION_POLICY=/path/to/policy.yml`: Policy file for custom validation
  rules
- `BITNET_VALIDATION_POLICY_KEY=arch:variant`: Policy key for rules lookup

### Correction Configuration (Development Only)

- `BITNET_CORRECTION_POLICY=/path/to/policy.yml`: Policy file for model-specific
  corrections (requires `BITNET_ALLOW_RUNTIME_CORRECTIONS=1`)
- `BITNET_ALLOW_RUNTIME_CORRECTIONS=1`: Enable runtime corrections for known-bad models
  (CI blocks this flag)

## Repository Contracts

- **Always specify features**: `--no-default-features --features cpu|gpu`
- **Use xtask for operations**: `cargo run -p xtask --` instead of scripts
- **Check compatibility**: Review `COMPATIBILITY.md` before API changes
- **Never modify GGUF in-place**: Use `bitnet-compat export-fixed` for new files

For comprehensive documentation, see the `docs/` directory organized by audience and use case.
