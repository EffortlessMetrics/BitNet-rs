# CLAUDE.md

Essential guidance for working with the BitNet.rs neural network inference codebase.

## Project Status

**Current Release**: v0.1.0-qna-mvp (Q&A-ready MVP)

### What's Working

- CPU inference with SIMD optimization (AVX2/AVX-512/NEON)
- GPU inference with CUDA acceleration (GPU support via feature gates)
- QK256 (GGML I2_S) MVP with scalar kernels (~0.1 tok/s for 2B models)
- Interactive chat and Q&A workflows with prompt templates
- Model validation and inspection tools
- Cross-validation framework against C++ reference

### Current Limitations (MVP Phase)

- **QK256 Performance**: Scalar-only kernels. For quick validation, limit to
  `--max-new-tokens 4-16`.
- **Model Quality**: The microsoft-bitnet-b1.58-2B-4T-gguf produces non-sensical
  output in some configurations. This is a known model quality issue, not an
  inference bug.
- **Test Scaffolding**: ~548 TODO/FIXME/unimplemented markers and ~70 ignored tests
  represent TDD-style scaffolding for planned features. See **Test Status** section
  below.
- **Active Issues**: See GitHub issue tracker for current blockers and their status. Run `just tdd-receipts` for test execution receipts.

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
  - Automatic flavor detection from tensor size with QK256 priority in close-match scenarios
  - Routes to C++ via FFI for validation when `BITNET_CPP_DIR` set
  - See: `docs/howto/use-qk256-models.md` for usage guide
  - See: `docs/explanation/i2s-dual-flavor.md` for architecture details
- **TL1/TL2**: Table lookup quantization with device-aware selection (ARM NEON / x86 AVX)
- **IQ2_S**: GGML-compatible via FFI bridge

**Flavor Detection Priority**: When tensor sizes match multiple quantization formats, the loader
checks QK256 (GgmlQk256NoScale) first for more specific matches before falling back to other formats.

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

### QK256 AVX2 Fast Path (v0.2 Foundation)

BitNet.rs implements AVX2-accelerated QK256 dequantization with runtime dispatch:

- **Runtime dispatch**: Scalar fallback if `avx2` is unavailable at runtime
- **Correctness parity**: ≤ 1e-5 max absolute difference vs scalar on randomized shapes
- **Initial uplift**: ~1.2× observed; target ≥3× with nibble-LUT + FMA tiling and prefetch
- **Benchmarks**: Run `cargo bench --bench kernel_benchmarks --features cpu,avx2`
- **Tests**: Property-based tests validate numerical correctness across random inputs

**Planned optimizations for ≥3× uplift:**
- Nibble LUT unpack via `pshufb` (2-bit → signed i8 mapping)
- FMA tiling (8-16 rows, unroll dot-products)
- Load combine (reduce AVX crossings)
- Prefetch (next code block & input)

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

**Auto-Detection for BitNet Base Models**: BitNet base models auto-detect to `instruct` template,
providing better Q&A behavior out-of-box. This is more reliable than raw completion mode for
getting coherent answers to questions.

**Note**: As of v0.9.x, the default auto-detection fallback changed from `raw` to
`instruct` for better out-of-box experience with instruction-tuned models. Use
`--prompt-template raw` if you need raw completion behavior.

**Default template:** `auto` — uses `llama3-chat` if the tokenizer exposes `<|eot_id|>`,
otherwise falls back to `instruct`. Override with `--prompt-template`.

### Quick Start: Q&A with BitNet Models

**Base vs Instruction-tuned Models**: Base models tend to "complete" prompts rather than
"answer" questions. For best Q&A results with base models, use `--prompt-template instruct`
with concise prompts. Instruction-tuned models perform better with conversational templates.

**For one-shot Q&A and math problems**, use `--prompt-template instruct`:

```bash
# One-shot Q&A (recommended for base/BitNet models)
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt-template instruct \
  --prompt "What is 2+2?" \
  --max-tokens 8 \
  --temperature 0.0 --greedy
```

**For conversational prompts with LLaMA-3 compatible models**, use `--prompt-template llama3-chat`:

```bash
# LLaMA-3 chat (auto-stops on <|eot_id|> token ID 128009)
# Note: This uses the Microsoft model with llama3-chat template
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt-template llama3-chat \
  --system-prompt "You are a helpful assistant" \
  --prompt "What is the capital of France?" \
  --max-tokens 32 \
  --temperature 0.7 --top-p 0.95
```

**For explicit stop control**, use `--stop-id` to specify token IDs:

```bash
# Explicit stop token ID (e.g., LLaMA-3 <|eot_id|> = 128009)
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt-template llama3-chat \
  --prompt "Capital of France?" \
  --stop-id 128009 \
  --max-tokens 16
```

You can override auto-detection with `--prompt-template`:

```bash
# Auto-detect template (recommended - uses GGUF metadata and tokenizer hints)
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is the capital of France?" \
  --max-tokens 32

# Raw (no formatting) - for completion-style models
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt-template raw \
  --prompt "2+2=" \
  --max-tokens 16

# Instruct (Q&A format) - for instruction-tuned models
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt-template instruct \
  --prompt "What is the capital of France?" \
  --max-tokens 32

# LLaMA-3 chat format - for LLaMA-3 compatible models
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
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
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
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
# Manual stop sequences (string-based)
--stop "</s>" --stop "\n\nQ:"

# Manual stop token IDs (numeric token IDs for LLaMA-3 EOT, etc.)
--stop-id 128009  # <|eot_id|> for LLaMA-3

# Combined: manual strings + manual IDs + template defaults
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt-template llama3-chat \
  --prompt "What is 2+2?" \
  --stop "\n\n" \
  --stop-id 128009 \
  --max-tokens 32

# Template defaults (automatic based on --prompt-template)
# - raw: no stop sequences
# - instruct: stops on "\n\nQ:", "\n\nHuman:"
# - llama3-chat: stops on "<|eot_id|>" (auto-resolved to token ID 128009)
#
# Note: Template-resolved token IDs are automatically merged with manual --stop-id values

**Stop-Sequence Optimization**: Stop token IDs are checked first (fast), then EOS fallback, then
string-based stops using a rolling tail window (optimized for memory and performance). The tail
window size is bounded by the longest string stop sequence (default 64-byte window), reducing
decoding overhead for large batches.

#### Stop Semantics (Unified Across run/chat/streaming)

The engine evaluates stops in this order for **all** generation paths:

1. **Token IDs** (`stop_token_ids`) — O(1) lookup, checked first
2. **EOS** (from tokenizer or explicit) — fallback after token ID check
3. **String sequences** (`stop_sequences`) — matched on a **rolling, UTF-8-safe tail buffer**
   configured by `stop_string_window` (bytes). This avoids decoding the full history per step.

**Configuration:**
- Default tail window: 64 bytes (sufficient for most stop sequences like `</s>`, `\n\n`)
- Increase with `--stop-string-window <N>` for longer stop sequences
- All generation modes (run, chat, streaming) use the same evaluation order

### Interactive Chat

BitNet.rs provides an interactive chat mode with REPL and built-in commands. The CLI
auto-detects the appropriate chat template and displays streaming responses:

```bash
# Interactive chat with auto-template detection
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json

# Chat with specific template override
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt-template llama3-chat

# Chat with custom sampling parameters
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
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
- Slow QK256 inference: The QK256 MVP uses scalar kernels (~0.1 tok/s for 2B models).
  For quick validation, use `--max-new-tokens 4-16`. SIMD optimizations are planned.
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

## Test Status (MVP Phase)

### Overview

<!-- TEST-STATUS:BEGIN -->
**As of 2025-10-21T04:14:57Z (CPU features)**

- **Discovered**: 0 tests
- **Executed**: 0 tests
  - Passed: 0
  - Failed: 0
  - Ignored: 0

_Auto-generated by `scripts/tdd_receipts.py`. Run `just tdd-receipts` to refresh._
<!-- TEST-STATUS:END -->

### Test Execution

```bash
# Run all enabled tests (skips #[ignore] tests)
cargo test --workspace --no-default-features --features cpu

# Run including ignored tests (will encounter blocked tests)
cargo test --workspace --no-default-features --features cpu -- --ignored --include-ignored

# Run specific test category
cargo test -p bitnet-inference --no-default-features --features cpu
cargo test -p bitnet-quantization --no-default-features --features cpu
cargo test -p bitnet-kernels --no-default-features --features cpu

# Skip slow tests (QK256 scalar kernels)
BITNET_SKIP_SLOW_TESTS=1 cargo test --workspace --no-default-features --features cpu
```

### TDD Scaffolds and Cleanup Items

The following tests are marked `#[ignore]` and reference specific issues. Run `just tdd-receipts` for current status:

1. **Issue #254** - **1 test**:
   - `test_real_transformer_forward_pass` (bitnet-inference/tests/test_real_inference.rs)
   - **See**: GitHub issue #254 for current status and investigation details

2. **Issue #260** - **2 tests**:
   - `test_cpu_simd_kernel_integration` (bitnet-kernels/tests/issue_260_feature_gated_tests.rs)
   - `test_tl2_avx_optimization` (bitnet-kernels/tests/issue_260_feature_gated_tests.rs)
   - **Note**: Docs previously referenced `test_cpu_tl2_optimization_x86_avx` but actual test name is `test_tl2_avx_optimization`
   - **See**: GitHub issue #260 for current status and implementation details

**Note**: Previous documentation referenced `test_real_vs_mock_comparison` which doesn't exist. The actual test is named `test_real_vs_mock_inference_comparison` (note "inference" in the middle).

**Current Status**: See auto-generated test receipts in TEST-STATUS section above.

### Ignored Test Patterns

Common reasons for #[ignore] markers (with counts):

```rust
// Pattern 1: Infrastructure-gated (52 tests) - FULLY IMPLEMENTED
#[test]
#[ignore] // Requires CUDA hardware
fn test_gpu_quantization() { /* ... */ }

#[test]
#[ignore] // Requires BITNET_GGUF environment variable
fn test_real_model_loading() { /* ... */ }

#[test]
#[ignore] // Network-dependent test
fn test_tokenizer_download() { /* ... */ }

// Pattern 2: TDD scaffolds (3 tests) - See issue tracker for status
#[test]
#[ignore] // Issue #254 - see GitHub issue for current status
fn test_real_transformer_forward_pass() { /* ... */ }

#[test]
#[ignore] // Issue #260 - see GitHub issue for current status
fn test_cpu_simd_kernel_integration() { /* ... */ }

// Pattern 3: Intentionally disabled by design (12 tests)
#[test]
#[ignore] // Benchmark, not unit test
fn bench_avx2_dequantize() { /* ... */ }

#[test]
#[ignore] // Disabled due to edge case handling - focus on successful mutation killers
fn test_mutation_edge_case() { /* ... */ }
```

**How to Enable Infrastructure-Gated Tests**:
```bash
# GPU tests (14 tests)
cargo test --workspace --features gpu --ignored

# Environment variable tests (14 tests)
export BITNET_GGUF=/path/to/model.gguf
export CROSSVAL_GGUF=/path/to/crossval-model.gguf
cargo test --workspace --features cpu --ignored

# Cross-validation tests (3 tests)
export BITNET_CPP_DIR=/path/to/bitnet.cpp
export CROSSVAL_GGUF=/path/to/model.gguf
cargo test --workspace --features crossval test_ac5 -- --ignored

# Network tests (9 tests) - require internet connection
cargo test --workspace --features cpu test_ac4 -- --ignored
```

### Working Test Categories

These test suites pass reliably:

- **quantization tests**: I2_S flavor detection, TL1/TL2, IQ2_S via FFI
- **model loading tests**: GGUF and SafeTensors parsing
- **tokenizer tests**: Universal tokenizer, auto-discovery (except parity tests blocked by #469)
- **cli tests**: Command-line parsing, flag validation
- **device feature tests**: CPU/GPU compilation detection
- **validation tests**: LayerNorm inspection, projection statistics (when not in strict mode)

### Test Dependencies

**Test Blocking Analysis**: See auto-generated receipts in TEST-STATUS section above for current counts.

**Referenced Issues**:
- Issue #254: 2 tests
- Issue #260: 2 tests

See individual GitHub issues for detailed status, root cause analysis, and resolution plans.

**Infrastructure-Gated Tests**: Tests requiring GPU hardware, environment variables, or network connectivity are marked `#[ignore]` with clear skip reasons in test output.

**Hidden Tests**: ~1000 additional tests exist in `tests/` directory but are not
auto-discovered due to `autotests = false` configuration (intentional for MVP).

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

### Test Configuration

- `BITNET_SKIP_SLOW_TESTS=1`: Skip slow tests (QK256 scalar kernel tests that exceed timeout)
- `BITNET_RUN_IGNORED_TESTS=1`: Include ignored tests when running suite (e.g., blocked tests waiting for issue resolution)

## Known Issues

These are active issues affecting current development. See issue tracker for details and workarounds.

### Active Issues

For current status of all issues, see the GitHub issue tracker:

- **Issue #254**: Test fixtures and RMSNorm configuration
- **Issue #260**: Kernel implementation and feature gates
- **Issue #439**: Feature gate consistency across GPU/CPU predicates
- **Issue #469**: Tokenizer parity and FFI build hygiene

Check individual issues for:
- Current status and resolution progress
- Root cause analysis and investigation details
- Workarounds and temporary mitigation strategies
- Impact on test execution (see `just tdd-receipts` for counts)

### Model Quality: microsoft-bitnet-b1.58-2B-4T-gguf

**Status**: Known limitation
**Symptom**: Non-sensical output in some configurations

- Some models produce garbled text instead of coherent responses
- This is a **model quality issue**, not an inference engine bug
- Try alternative models or simpler prompts for validation
- For testing inference correctness, use synthetic/controlled inputs

## Common Pitfalls

### 1. Confusing Test Scaffolding with Bugs

**Problem**: Seeing unimplemented!() calls or #[ignore] tests

```rust
// This is NORMAL during MVP - it's intentional scaffolding
#[test]
#[ignore] // Blocked by Issue #254
fn test_real_inference_path() {
    unimplemented!("Waiting for shape mismatch fix")
}
```

**Solution**: Check the blocking issue (e.g., #254). These are placeholder tests that will be enabled once issues are resolved.

### 2. Expecting Production Performance from QK256 MVP

**Problem**: QK256 inference very slow (~0.1 tok/s for 2B models)

```bash
# Wrong - will timeout waiting for inference
cargo run -p bitnet-cli --features cpu -- run \
  --model model.gguf --prompt "Long text" --max-tokens 1000

# Right - quick validation with small token budget
cargo run -p bitnet-cli --features cpu -- run \
  --model model.gguf --prompt "What is 2+2?" --max-tokens 4
```

**Why**: QK256 MVP uses scalar-only kernels. SIMD optimization is planned for post-MVP.

### 3. Model Quality Issues Aren't Inference Bugs

**Problem**: Getting garbled output from microsoft-bitnet model

```text
Prompt: "What is the capital of France?"
Output: "jjjjkkkk llll mmmm nnnn..."
```

**Solution**: This is a known model quality limitation, not an inference engine bug:

- Try alternative models
- Use shorter, simpler prompts
- Validate inference correctness with synthetic inputs
- Report reproducible inference bugs separately

### 4. Ignoring Feature Flags

**Problem**: Getting linker errors or silent GPU fallback

```bash
# Wrong - uses default (empty) features, causes errors
cargo build

# Right - always specify features
cargo build --no-default-features --features cpu
cargo build --no-default-features --features gpu
```

**Why**: BitNet.rs deliberately has **empty default features** to prevent surprise dependencies. Always be explicit.

### 5. Running Ignored Tests Expecting Success

**Problem**: Running all tests with `--ignored` flag

```bash
# Will encounter blocked tests
cargo test --workspace -- --ignored --include-ignored
```

**Solution**: Check blocking issue numbers in test comments. These are intentional placeholders:

```bash
# Run only non-ignored tests (recommended for CI)
cargo test --workspace --no-default-features --features cpu

# Run specific working test suites
cargo test -p bitnet-quantization --no-default-features --features cpu
cargo test -p bitnet-models --no-default-features --features cpu
```

### 6. Expecting All Tests to Pass

**Current State (MVP)**:

- ~500+ tests with passing infrastructure
- ~70 tests intentionally ignored (scaffolding)
- Real inference tests blocked by #254, #260, #439, #469

**CI Status**: Only non-ignored tests run in CI. Ignored tests are tracked separately.

### 7. FFI Linker Issues

**Problem**: "undefined reference" to C++ functions

```text
error: undefined reference to `bitnet_cpp::...`
```

**Solution**: Choose the appropriate feature set:

```bash
# Pure Rust (recommended for development)
cargo build --no-default-features --features cpu

# With FFI support (requires C++ reference setup)
export BITNET_CPP_DIR=/path/to/bitnet.cpp
cargo build --no-default-features --features cpu,ffi

# Or just avoid FFI
cargo build --no-default-features --features cpu
```

## Repository Contracts

- **Always specify features**: `--no-default-features --features cpu|gpu`
- **Use xtask for operations**: `cargo run -p xtask --` instead of scripts
- **Check compatibility**: Review `COMPATIBILITY.md` before API changes
- **Never modify GGUF in-place**: Use `bitnet-compat export-fixed` for new files
- **Expect test scaffolding during MVP**: ~548 TODO/FIXME markers and ~70 ignored tests are intentional
- **unimplemented!() in tests is not a bug**: It's TDD scaffolding for planned features
- **Check issue tracker for blockers**: Before investigating test failures, see #254, #260, #439, #469

For comprehensive documentation, see the `docs/` directory organized by audience and use case.
