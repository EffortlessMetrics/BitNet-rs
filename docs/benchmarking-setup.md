# BitNet.rs Benchmarking Infrastructure Setup

This document provides comprehensive guidance for setting up and using the BitNet.rs benchmarking infrastructure, addressing GitHub issue #155.

## Overview

The `scripts/setup-benchmarks.sh` script provides a one-stop solution for setting up a complete benchmarking environment for BitNet.rs. It addresses the "non-functional" benchmarking issues by:

1. **Downloading and verifying required models and tokenizers**
2. **Setting up the C++ implementation for cross-validation**
3. **Running diagnostic checks to ensure everything works**
4. **Providing clear instructions for running benchmarks**
5. **Integrating with existing CI/CD workflows**

## Quick Start

### Basic Setup (Recommended)

```bash
# Full setup with all features
./scripts/setup-benchmarks.sh
```

This will:
- Download the default BitNet model (microsoft/bitnet-b1.58-2B-4T-gguf)
- Set up the C++ implementation for cross-validation
- Run diagnostic checks
- Provide ready-to-use benchmark commands

### Rust-Only Setup (Faster)

```bash
# Skip C++ setup for faster installation
./scripts/setup-benchmarks.sh --skip-cpp
```

Use this if you only need Rust benchmarks and don't require cross-validation with the C++ implementation.

### Preview What Will Be Done

```bash
# See what the script would do without making changes
./scripts/setup-benchmarks.sh --dry-run
```

## Script Options

| Option | Description | Example |
|--------|-------------|---------|
| `--force` | Force re-download and rebuild everything | `--force` |
| `--skip-cpp` | Skip C++ implementation (Rust-only) | `--skip-cpp` |
| `--skip-model` | Skip model download (use existing) | `--skip-model` |
| `--model-id` | Use different model | `--model-id custom/model` |
| `--model-file` | Use different model file | `--model-file model.gguf` |
| `--cpp-dir` | Custom C++ directory | `--cpp-dir /custom/path` |
| `--dry-run` | Preview actions without executing | `--dry-run` |
| `--verbose` | Enable detailed output | `--verbose` |
| `--help` | Show help message | `--help` |

## What the Script Does

### 1. System Requirements Check

The script verifies:
- ✅ Required commands are available (`cargo`, `curl`, `python3`)
- ✅ Rust version is compatible (≥1.89.0)
- ✅ Sufficient disk space is available (≥10GB recommended)
- ✅ Repository structure is correct

### 2. Model Fixtures Setup

Downloads and verifies:
- **Primary model**: `microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf`
- **Tokenizer**: `tokenizer.json` (if available)
- **Crossval fixtures**: Creates symlinks for benchmark tests

The script uses the existing `cargo xtask download-model` infrastructure for reliable downloads with progress tracking.

### 3. C++ Implementation Setup

Sets up BitNet.cpp for cross-validation:
- Downloads Microsoft's C++ implementation
- Builds the `llama-cli` binary
- Verifies the build works with the downloaded model
- Sets up `BITNET_CPP_DIR` environment variable

This uses the existing `cargo xtask fetch-cpp` command.

### 4. Diagnostic Checks

Validates the complete setup:
- ✅ Rust builds successfully
- ✅ Model is accessible and reasonable size
- ✅ Rust inference works
- ✅ C++ implementation works (if available)
- ✅ Python benchmark script is functional
- ✅ Crossval benchmarks compile
- ✅ GPU support (if available)

### 5. Configuration Generation

Creates `benchmark-results/benchmark-config.json` with:
- Environment information
- Available features
- Recommended commands
- Platform details

## Running Benchmarks

After setup, you'll have several benchmark options:

### 1. Python Comparison Benchmark (Recommended)

The `benchmark_comparison.py` script provides the most comprehensive benchmarking:

```bash
# Basic comparison (Rust vs C++)
./benchmark_comparison.py

# With custom parameters
./benchmark_comparison.py \
    --model "models/microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf" \
    --cpp-dir "/home/user/.cache/bitnet_cpp" \
    --prompt "The capital of France is" \
    --tokens 32 \
    --iterations 3

# GPU comparison
./benchmark_comparison.py --gpu

# Rust-only (if C++ not available)
./benchmark_comparison.py --skip-cpp
```

**Output includes**:
- Performance metrics (mean, std dev, min, max)
- Throughput comparison
- Response correctness validation
- Speedup calculations
- Winner determination
- JSON results for CI integration

### 2. Rust-Only Benchmarks

```bash
# Standard Rust benchmarks
cargo bench --workspace --no-default-features --features cpu

# With GPU support
cargo bench --workspace --no-default-features --features gpu

# Quick tests
cargo test --workspace --no-default-features --features cpu --release
```

### 3. Cross-Validation Benchmarks

```bash
# Criterion-based cross-validation benchmarks
cargo bench --features crossval

# Comprehensive cross-validation tests
cargo test --package crossval --features crossval --release -- --nocapture benchmark
```

### 4. Specialized Benchmarks

```bash
# SIMD performance
cargo bench -p bitnet-quantization --bench simd_comparison --no-default-features --features cpu

# GPU kernels (if available)
cargo bench -p bitnet-kernels --bench mixed_precision_bench --no-default-features --features gpu

# Memory usage
cargo test -p bitnet-kernels --no-default-features --features cpu test_memory_tracking_comprehensive
```

## Integration with CI/CD

The setup script integrates seamlessly with existing workflows:

### GitHub Actions Integration

The script works with `.github/workflows/performance-tracking.yml`:

```yaml
- name: Setup benchmarking infrastructure
  run: ./scripts/setup-benchmarks.sh --skip-cpp

- name: Run performance benchmarks
  run: |
    ./benchmark_comparison.py --skip-cpp --format json > benchmark-results.json
    cargo bench --workspace --no-default-features --features cpu
```

### Environment Variables

The script sets up standard environment variables:

```bash
export BITNET_GGUF="/path/to/model.gguf"
export BITNET_CPP_DIR="/path/to/cpp/implementation"
```

These are compatible with existing scripts and CI workflows.

## Troubleshooting

### Common Issues and Solutions

#### Model Download Fails

```bash
# Check internet connection and retry
./scripts/setup-benchmarks.sh --force

# Try with HF token for private repos
export HF_TOKEN="your_token_here"
./scripts/setup-benchmarks.sh --force
```

#### C++ Build Fails

```bash
# Skip C++ and use Rust-only benchmarks
./scripts/setup-benchmarks.sh --skip-cpp

# Or check build dependencies
sudo apt install build-essential cmake  # Linux
brew install cmake                       # macOS
```

#### Disk Space Issues

```bash
# Check available space
df -h .

# Use smaller model or external storage
./scripts/setup-benchmarks.sh --model-id smaller/model
```

#### Permission Issues

```bash
# Fix script permissions
chmod +x scripts/setup-benchmarks.sh

# Check directory permissions
ls -la scripts/
```

### Diagnostic Commands

```bash
# Verify model integrity
cargo run -p xtask -- verify --model "path/to/model.gguf"

# Test inference
cargo run -p xtask -- infer \
    --model "path/to/model.gguf" \
    --prompt "Test" \
    --max-new-tokens 10 \
    --allow-mock

# Check C++ implementation
"${BITNET_CPP_DIR}/build/bin/llama-cli" --help
```

### Log Files

Check setup logs for detailed information:

```bash
# View setup log
tail -f benchmark-results/setup.log

# Check benchmark results
ls -la benchmark-results/
cat benchmark-results/benchmark-config.json
```

## Advanced Usage

### Custom Models

```bash
# Use different model
./scripts/setup-benchmarks.sh \
    --model-id "custom/model-repo" \
    --model-file "custom-model.gguf"
```

### Custom C++ Directory

```bash
# Use existing C++ installation
./scripts/setup-benchmarks.sh \
    --cpp-dir "/path/to/existing/cpp"
```

### Performance Optimization

```bash
# Build with maximum optimizations
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
cargo build --release --no-default-features --features cpu

# Run optimized benchmarks
./benchmark_comparison.py --iterations 5
```

## Files Created by Setup

The script creates the following files and directories:

```
models/
├── microsoft/
│   └── bitnet-b1.58-2B-4T-gguf/
│       ├── ggml-model-i2_s.gguf      # Main model file
│       └── tokenizer.json             # Tokenizer (if available)
├── crossval/
│   └── fixtures/
│       └── benchmark_model.gguf       # Symlink to main model
└── benchmark-results/
    ├── setup.log                      # Setup log
    └── benchmark-config.json          # Configuration
```

## Environment Variables Used

| Variable | Purpose | Default |
|----------|---------|---------|
| `BITNET_GGUF` | Path to model file | Auto-detected |
| `BITNET_CPP_DIR` | C++ implementation directory | `~/.cache/bitnet_cpp` |
| `HF_TOKEN` | Hugging Face token | None |
| `RUSTFLAGS` | Rust compilation flags | None |

## Performance Expectations

After setup, typical benchmark performance:

- **Model download**: 5-15 minutes (depends on connection)
- **C++ build**: 10-20 minutes (depends on system)
- **Basic benchmark**: 1-5 minutes
- **Comprehensive benchmark**: 10-30 minutes
- **Cross-validation**: 20-60 minutes

## Security Considerations

The setup script:
- ✅ Downloads models from official Hugging Face repositories
- ✅ Uses official Microsoft C++ implementation
- ✅ Verifies checksums where available
- ✅ Runs in user space (no sudo required)
- ✅ Creates isolated directories

## Integration with Existing Tools

The setup integrates with:

- **benchmark_comparison.py**: Uses the same environment variables
- **crossval/benches/performance.rs**: Creates compatible fixtures
- **CI workflows**: Compatible with performance-tracking.yml
- **xtask commands**: Uses existing download and build infrastructure

## Contributing

To improve the benchmarking setup:

1. Test the script on your platform
2. Report issues with detailed logs
3. Suggest improvements for edge cases
4. Add support for new models or platforms

## References

- **GitHub Issue**: [#155 - Non-functional benchmarking infrastructure](https://github.com/BitNet-rs/BitNet-rs/issues/155)
- **Related Scripts**: `benchmark_comparison.py`, `crossval/benches/performance.rs`
- **CI Workflows**: `.github/workflows/performance-tracking.yml`
- **Documentation**: `CLAUDE.md`, `README.md`