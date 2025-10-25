# C++ Reference Setup for Cross-Validation

This guide shows how to set up the Microsoft BitNet C++ reference implementation for cross-validation with BitNet.rs.

## Overview

BitNet.rs provides comprehensive cross-validation infrastructure to compare Rust inference against the official C++ implementation:

- **Official Reference**: [microsoft/BitNet](https://github.com/microsoft/BitNet) (includes `bitnet.cpp` built on top of `llama.cpp`)
- **Use Cases**: Per-token parity checking, trace diffing, logits divergence detection
- **One-Command Setup**: `cargo run -p xtask -- fetch-cpp`

## Quick Start

### 1. Bootstrap C++ Reference (Automated)

The easiest way is to use xtask's automated setup:

```bash
# Download and build C++ reference with shared library
cargo run -p xtask -- fetch-cpp

# The build completes in ~/.cache/bitnet_cpp by default
# xtask will display the export command you need to run
```

After the build completes, set the environment variable:

```bash
# Linux
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/bin:$LD_LIBRARY_PATH"

# macOS
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
export DYLD_LIBRARY_PATH="$BITNET_CPP_DIR/build/bin:$DYLD_LIBRARY_PATH"

# Windows (PowerShell)
$env:BITNET_CPP_DIR = "$env:USERPROFILE\.cache\bitnet_cpp"
$env:PATH = "$env:BITNET_CPP_DIR\build\bin;$env:PATH"
```

**Pro Tip**: Add the `export` commands to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to make them permanent.

### 2. Verify Installation

```bash
# Check that the shared library is discoverable
# Linux:
ldd ~/.cache/bitnet_cpp/build/bin/libllama.so
# macOS:
otool -L ~/.cache/bitnet_cpp/build/bin/libllama.dylib

# Verify xtask can access C++ functionality (requires --features inference)
cargo run -p xtask --features inference -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1 \
  --cos-tol 0.999
```

### 3. Run Cross-Validation

Once setup is complete, you can use the full cross-validation toolkit:

```bash
# Per-token logits divergence detection
cargo run -p xtask --features inference -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --cos-tol 0.999 \
  --format json

# Full cross-validation sweep (3 scenarios, 90+ traces per scenario)
./scripts/run_crossval_sweep.sh \
  models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  /tmp/crossval-sweep

# Trace diffing (compare Rust vs C++ layer-by-layer)
# 1. Generate Rust traces
BITNET_TRACE_DIR=/tmp/rs RUST_LOG=warn BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
  cargo run -p bitnet-cli --features cpu,trace -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" --max-tokens 1 --greedy

# 2. Generate C++ traces (if patched for trace emission)
# ... emit equivalent JSON traces from bitnet.cpp ...

# 3. Compare traces
python3 scripts/trace_diff.py /tmp/rs /tmp/cpp
```

## Manual Setup (Advanced)

If you need fine-grained control over the C++ build, you can build manually:

### Building llama.cpp Shared Library

The modern way to build llama.cpp with shared library support:

```bash
# Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Build shared library (CPU-only)
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j

# Build with CUDA support
cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON
cmake --build build --config Release -j
```

**Key Points**:
- Use **`-DBUILD_SHARED_LIBS=ON`** (not `LLAMA_BUILD_SHARED_LIB`)
- The shared library appears in **`build/bin/`** as:
  - Linux: `libllama.so`
  - macOS: `libllama.dylib`
  - Windows: `llama.dll`

### Building Microsoft BitNet

```bash
# Clone the official reference
git clone https://github.com/microsoft/BitNet
cd BitNet

# Build (follows llama.cpp build process)
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j

# Set environment variables
export BITNET_CPP_DIR="$(pwd)"
export LD_LIBRARY_PATH="$(pwd)/build/bin:$LD_LIBRARY_PATH"  # Linux
export DYLD_LIBRARY_PATH="$(pwd)/build/bin:$DYLD_LIBRARY_PATH"  # macOS
```

## Troubleshooting

### Error: "libllama.so: cannot open shared object file"

**Cause**: The dynamic loader can't find the shared library.

**Fix**:
```bash
# Verify the library exists
ls -lh ~/.cache/bitnet_cpp/build/bin/libllama.*

# Set the dynamic loader path
# Linux:
export LD_LIBRARY_PATH="$HOME/.cache/bitnet_cpp/build/bin:$LD_LIBRARY_PATH"
# macOS:
export DYLD_LIBRARY_PATH="$HOME/.cache/bitnet_cpp/build/bin:$DYLD_LIBRARY_PATH"

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export LD_LIBRARY_PATH="$HOME/.cache/bitnet_cpp/build/bin:$LD_LIBRARY_PATH"' >> ~/.bashrc
```

### Error: "C++ FFI not available"

**Cause**: xtask built without the `inference` feature.

**Fix**:
```bash
# Build with inference feature
cargo run -p xtask --features inference -- crossval-per-token --help
```

### Error: "bitnet-crossval: Using mock C wrapper"

**Cause**: This is a **warning**, not an error. It means:
- C++ library not found at build time
- FFI will fall back to rust-only mode
- Cross-validation will run in "rust-only" mode (no C++ comparison)

**Fix** (if you want actual C++ parity):
```bash
# Ensure BITNET_CPP_DIR is set at BUILD time
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/bin:$LD_LIBRARY_PATH"

# Rebuild xtask
cargo clean -p xtask -p bitnet-crossval -p bitnet-sys
cargo build -p xtask --features inference
```

### Windows-Specific Issues

For Windows, you'll need:
1. Visual Studio 2019 or later (with C++ CMake tools)
2. CMake 3.14+
3. Add `-A x64` to cmake if needed:
   ```powershell
   cmake -B build -A x64 -DBUILD_SHARED_LIBS=ON
   cmake --build build --config Release -j
   ```

**One-Click Helper**: [Llama-Build](https://github.com/SciSharp/Llama-Build) automates prerequisites and build on Windows.

## Cross-Validation Architecture

BitNet.rs provides multiple levels of validation:

1. **Smoke Test** (`scripts/parity_smoke.sh`):
   - One-command validation
   - Tests both BitNet and QK256 formats
   - Pretty-printed JSON receipts

2. **Per-Token Parity** (`xtask crossval-per-token`):
   - Compares Rust vs C++ logits position-by-position
   - Finds first divergence token
   - Cosine similarity, L2 distance, max absolute difference

3. **Trace Diffing** (`scripts/trace_diff.py`):
   - Layer-by-layer comparison
   - Blake3 hash verification
   - RMS statistics
   - Identifies exact divergence point (layer, stage, sequence position)

4. **Multi-Scenario Sweep** (`scripts/run_crossval_sweep.sh`):
   - 3 deterministic scenarios (1, 2, 4 tokens)
   - 90+ trace files per scenario
   - Summary markdown output

## References

- **C++ Reference**: <https://github.com/microsoft/BitNet>
- **llama.cpp Build Docs**: <https://github.com/ggml-org/llama.cpp>
- **BitNet.rs Cross-Validation**: `docs/development/validation-framework.md`
- **Trace Infrastructure**: `crates/bitnet-trace/README.md`
- **Per-Token Parity**: `crossval/src/logits_compare.rs`

## Related Documentation

- [`docs/development/validation-framework.md`](../development/validation-framework.md) - Complete validation architecture
- [`docs/howto/validate-models.md`](validate-models.md) - Model validation workflow
- [`CLAUDE.md`](../../CLAUDE.md) - Feature flags and cross-validation quick reference
