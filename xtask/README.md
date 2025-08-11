# BitNet.rs Development Tasks (xtask)

A collection of development automation tools for BitNet.rs, providing convenient commands for model downloads, cross-validation, and development workflows.

## Installation

The xtask is automatically available in the workspace. No additional installation required.

## Usage

All commands are available via `cargo xtask`:

```bash
cargo xtask <command> [options]
```

## Commands

### Model Management

#### `download-model` - Download GGUF models from Hugging Face

Downloads BitNet GGUF models with resumable download support and optional SHA256 verification.

```bash
# Download default BitNet model
cargo xtask download-model

# Download specific model
cargo xtask download-model \
  --id microsoft/bitnet-b1.58-2B-4T-gguf \
  --file ggml-model-i2_s.gguf \
  --out models

# With SHA256 verification
cargo xtask download-model \
  --sha256 <expected_hash> \
  --force  # Overwrite existing
```

**Features:**
- Resumable downloads (partial downloads are cached)
- HF_TOKEN support for private repos
- SHA256 verification
- Progress bar with ETA

### Cross-Validation

#### `fetch-cpp` - Fetch and build Microsoft BitNet C++

Downloads and builds the official C++ implementation for cross-validation testing.

```bash
# Fetch with default tag
cargo xtask fetch-cpp

# Specific tag with clean rebuild
cargo xtask fetch-cpp \
  --tag b1-65-ggml \
  --force \
  --clean
```

#### `crossval` - Run cross-validation tests

Runs deterministic tests comparing Rust and C++ implementations.

```bash
# Use default model path
cargo xtask crossval

# Specify model and options
cargo xtask crossval \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --cpp-dir $HOME/.cache/bitnet_cpp \
  --release
```

**Environment:**
- Single-threaded execution for determinism
- Controlled via `OMP_NUM_THREADS=1` and `GGML_NUM_THREADS=1`

#### `full-crossval` - Complete workflow

Runs the entire cross-validation workflow: download → fetch → test.

```bash
# Run complete workflow
cargo xtask full-crossval

# Force redownload/rebuild
cargo xtask full-crossval --force
```

### Development Tools

#### `gen-fixtures` - Generate test fixtures

Creates deterministic test model fixtures for development.

```bash
cargo xtask gen-fixtures \
  --size small \
  --output crossval/fixtures/
```

Sizes: `tiny` (5KB), `small` (20KB), `medium` (100KB)

#### `setup-crossval` - Setup environment

Prepares the cross-validation environment.

```bash
cargo xtask setup-crossval
```

#### `clean-cache` - Clean caches

Removes all build caches and temporary files.

```bash
cargo xtask clean-cache
```

Cleans:
- `target/`
- `~/.cache/bitnet_cpp/`
- `crossval/fixtures/`

#### `check-features` - Verify feature flags

Ensures feature flag consistency across the workspace.

```bash
cargo xtask check-features
```

#### `benchmark` - Run benchmarks

Executes performance benchmarks.

```bash
cargo xtask benchmark --platform current
```

## Environment Variables

### For Downloads
- `HF_TOKEN`: Hugging Face authentication token for private repos

### For Cross-Validation
- `BITNET_CPP_DIR`: Path to C++ implementation (default: `~/.cache/bitnet_cpp`)
- `CROSSVAL_GGUF`: Path to GGUF model file
- `OMP_NUM_THREADS`: OpenMP threads (set to 1 for determinism)
- `GGML_NUM_THREADS`: GGML threads (set to 1 for determinism)

## CI/CD Integration

The xtask commands are designed for CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Download model
  run: cargo xtask download-model
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}

- name: Run cross-validation
  run: cargo xtask full-crossval
```

## Windows Support

On Windows, the `fetch-cpp` command requires WSL or Git Bash for the shell scripts. All other commands work natively.

## Troubleshooting

### Download Issues
- Check network connectivity
- Verify HF_TOKEN for private repos
- Use `--force` to override existing files

### Cross-Validation Failures
- Ensure C++ build dependencies are installed
- Check BITNET_CPP_DIR path
- Verify model file exists and is valid

### Memory Issues
- Use release builds for better performance
- Reduce batch size in tests
- Monitor with `cargo xtask benchmark`