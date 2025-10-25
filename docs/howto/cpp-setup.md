# C++ Reference Setup for Cross-Validation

This guide shows how to set up C++ reference implementations for cross-validation with BitNet.rs. BitNet.rs supports dual-backend cross-validation, allowing you to validate against either BitNet.cpp or llama.cpp depending on your model.

## Overview

BitNet.rs provides comprehensive cross-validation infrastructure to compare Rust inference against official C++ implementations:

### Two Backends Available

**Lane A: BitNet.rs vs bitnet.cpp**
- **Models**: microsoft-bitnet-b1.58-2B-4T-gguf (BitNet-specific models)
- **Source**: [microsoft/BitNet](https://github.com/microsoft/BitNet)
- **Libraries**: `libbitnet*.so` (or `.dylib` on macOS)
- **Use Cases**: BitNet quantization validation, 1-bit inference parity

**Lane B: BitNet.rs vs llama.cpp**
- **Models**: llama-3, llama-2, SmolLM3, and other GGUF-compatible models
- **Source**: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **Libraries**: `libllama*.so`, `libggml*.so`
- **Use Cases**: General GGUF model validation, LLaMA-family inference parity

### Auto-Detection

The `crossval-per-token` command automatically selects the appropriate backend based on your model path:
- Path contains `"bitnet"` or `"microsoft/bitnet"` → Uses bitnet.cpp
- Path contains `"llama"` → Uses llama.cpp
- Default: llama.cpp (conservative fallback)

You can override auto-detection with `--cpp-backend bitnet|llama`.

### One-Command Setup

```bash
# Auto-bootstrap both backends
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
```

This command fetches, builds, and configures the appropriate C++ reference implementation and emits environment variable exports for your shell.

## Quick Start

### Option 1: One-Command Setup (Recommended)

The easiest way is to use `setup-cpp-auto`, which automatically selects and builds the appropriate backend(s):

```bash
# Bash/Zsh
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Fish shell
cargo run -p xtask -- setup-cpp-auto --emit=fish | source

# PowerShell
cargo run -p xtask -- setup-cpp-auto --emit=pwsh | Invoke-Expression
```

This command:
1. Detects if you need bitnet.cpp or llama.cpp (or both)
2. Fetches and builds the C++ reference(s)
3. Emits shell-specific environment variable exports
4. Sets `BITNET_CPP_DIR` and dynamic loader paths automatically

**Pro Tip**: The emitted exports are printed to stdout. You can add them to your shell profile to make them permanent:

```bash
# Bash/Zsh - add to ~/.bashrc or ~/.zshrc
cargo run -p xtask -- setup-cpp-auto --emit=sh >> ~/.bashrc

# Fish - add to ~/.config/fish/config.fish
cargo run -p xtask -- setup-cpp-auto --emit=fish >> ~/.config/fish/config.fish
```

### Option 2: Manual Per-Backend Setup

If you need fine-grained control, set up backends individually:

**For BitNet Models** (BitNet.cpp backend):

```bash
# Download and build bitnet.cpp with shared library
cargo run -p xtask -- fetch-cpp --backend cpu

# The build completes in ~/.cache/bitnet_cpp by default

# Set environment variables (Linux)
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/bin:$LD_LIBRARY_PATH"

# Set environment variables (macOS)
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
export DYLD_LIBRARY_PATH="$BITNET_CPP_DIR/build/bin:$DYLD_LIBRARY_PATH"

# Set environment variables (Windows PowerShell)
$env:BITNET_CPP_DIR = "$env:USERPROFILE\.cache\bitnet_cpp"
$env:PATH = "$env:BITNET_CPP_DIR\build\bin;$env:PATH"
```

**For LLaMA Models** (llama.cpp backend):

If you need separate llama.cpp setup (optional—bitnet.cpp includes llama.cpp):

```bash
# Clone and build llama.cpp independently
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j
cd ..

# Set LLAMA_CPP_DIR if using standalone llama.cpp
export LLAMA_CPP_DIR="$(pwd)/llama.cpp"
export LD_LIBRARY_PATH="$LLAMA_CPP_DIR/build/bin:$LD_LIBRARY_PATH"  # Linux
```

**Make Permanent**: Add the `export` commands to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.).

### 3. Verify Installation

Verify that the C++ libraries are discoverable:

```bash
# Check that shared libraries exist and are discoverable
# Linux (bitnet.cpp):
ldd ~/.cache/bitnet_cpp/build/bin/libbitnet.so

# Linux (llama.cpp):
ldd ~/.cache/bitnet_cpp/build/bin/libllama.so

# macOS (bitnet.cpp):
otool -L ~/.cache/bitnet_cpp/build/bin/libbitnet.dylib

# macOS (llama.cpp):
otool -L ~/.cache/bitnet_cpp/build/bin/libllama.dylib
```

Verify that xtask can access C++ functionality:

```bash
# BitNet model auto-detection (will use bitnet.cpp)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1 \
  --verbose

# LLaMA model auto-detection (will use llama.cpp)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/llama-3-8b-instruct.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1 \
  --verbose
```

The `--verbose` flag shows backend selection, library detection, and preflight results.

### 4. Run Cross-Validation

Once setup is complete, you can use the full cross-validation toolkit. The toolkit automatically selects the appropriate C++ backend:

**BitNet Model Validation** (auto-detects bitnet.cpp):

```bash
# Per-token logits divergence detection (BitNet)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --cos-tol 0.999 \
  --format json
```

**LLaMA Model Validation** (auto-detects llama.cpp):

```bash
# Per-token logits divergence detection (LLaMA)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/llama-3-8b-instruct.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is the capital of France?" \
  --max-tokens 8 \
  --cos-tol 0.999 \
  --format json
```

**Explicit Backend Selection** (override auto-detection):

```bash
# Force bitnet.cpp even for non-BitNet models
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/some-model.gguf \
  --tokenizer models/tokenizer.json \
  --cpp-backend bitnet \
  --prompt "Test" \
  --max-tokens 4

# Force llama.cpp even for BitNet models
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --cpp-backend llama \
  --prompt "Test" \
  --max-tokens 4
```

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

### Building BitNet.cpp

BitNet.cpp is the primary reference for 1-bit models and includes llama.cpp as a dependency:

```bash
# Clone the official reference
git clone https://github.com/microsoft/BitNet
cd BitNet

# Build bitnet.cpp (CPU-only)
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j

# Build with CUDA support
cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80;86
cmake --build build --config Release -j

# Set environment variables
export BITNET_CPP_DIR="$(pwd)"
export LD_LIBRARY_PATH="$(pwd)/build/bin:$LD_LIBRARY_PATH"  # Linux
export DYLD_LIBRARY_PATH="$(pwd)/build/bin:$DYLD_LIBRARY_PATH"  # macOS
```

**Key Points**:
- BitNet.cpp is built on top of llama.cpp, so it includes both `libbitnet.so` and `libllama.so`
- Use **`-DBUILD_SHARED_LIBS=ON`** to generate shared libraries
- Shared libraries appear in **`build/bin/`** as:
  - Linux: `libbitnet.so`, `libllama.so`, `libggml.so`
  - macOS: `libbitnet.dylib`, `libllama.dylib`, `libggml.dylib`
  - Windows: `bitnet.dll`, `llama.dll`, `ggml.dll`

### Building llama.cpp Standalone (Optional)

If you need llama.cpp without bitnet.cpp (for LLaMA-family models only):

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

# Set environment variables for standalone llama.cpp
export LLAMA_CPP_DIR="$(pwd)"
export LD_LIBRARY_PATH="$(pwd)/build/bin:$LD_LIBRARY_PATH"  # Linux
export DYLD_LIBRARY_PATH="$(pwd)/build/bin:$DYLD_LIBRARY_PATH"  # macOS
```

**Note**: BitNet.rs `--cpp-backend llama` will look for `libllama.so` in `$BITNET_CPP_DIR/build/bin`. If using standalone llama.cpp, either set `BITNET_CPP_DIR` to your llama.cpp directory or use `LLAMA_CPP_DIR` and ensure the loader can find the libraries.

## Troubleshooting

### Error: "libbitnet.so: cannot open shared object file" or "libllama.so: cannot open shared object file"

**Cause**: The dynamic loader can't find the C++ libraries.

**Fix**:

```bash
# Verify the library exists (BitNet)
ls -lh ~/.cache/bitnet_cpp/build/bin/libbitnet.*

# Verify the library exists (LLaMA)
ls -lh ~/.cache/bitnet_cpp/build/bin/libllama.*

# Set the dynamic loader path (Linux)
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/bin:$LD_LIBRARY_PATH"

# Set the dynamic loader path (macOS)
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
export DYLD_LIBRARY_PATH="$BITNET_CPP_DIR/build/bin:$DYLD_LIBRARY_PATH"

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export LD_LIBRARY_PATH="$HOME/.cache/bitnet_cpp/build/bin:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify ldd/otool can find the libraries
# Linux:
ldd ~/.cache/bitnet_cpp/build/bin/libbitnet.so | grep "not found"

# macOS:
otool -L ~/.cache/bitnet_cpp/build/bin/libbitnet.dylib | grep "not found"
```

### Error: "Backend 'bitnet' requires libbitnet but build has no C++ libs"

**Cause**: BitNet.cpp wasn't built or the build directory is missing.

**Fix**:

```bash
# Rebuild bitnet.cpp
cargo run -p xtask -- fetch-cpp --backend cpu --force

# Verify both backends are available
ls -lh ~/.cache/bitnet_cpp/build/bin/lib*.so  # Linux
# Output should show: libbitnet.so, libllama.so, libggml.so
```

### Error: "Backend 'llama' requires libllama but build has no C++ libs"

**Cause**: llama.cpp wasn't built or is missing from the build directory.

**Fix**:

```bash
# If you built bitnet.cpp, it includes llama.cpp by default
# Verify the library exists:
ls -lh ~/.cache/bitnet_cpp/build/bin/libllama.so  # Linux

# If using standalone llama.cpp:
export BITNET_CPP_DIR="/path/to/llama.cpp"
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/bin:$LD_LIBRARY_PATH"
```

### Error: "crossval-per-token: feature 'inference' not enabled"

**Cause**: xtask built without the `--features crossval-all` (or `--features inference`).

**Fix**:

```bash
# Build xtask with required features
cargo run -p xtask --features crossval-all -- crossval-per-token --help
```

Note: The feature must be specified at build time, not at runtime.

### Error: Backend auto-detection wrong (using llama when you want bitnet, or vice versa)

**Cause**: Model path doesn't match detection heuristics.

**Fix**: Explicitly specify the backend:

```bash
# Force bitnet backend
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/my-model.gguf \
  --tokenizer models/tokenizer.json \
  --cpp-backend bitnet \
  --prompt "Test"

# Force llama backend
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/my-model.gguf \
  --tokenizer models/tokenizer.json \
  --cpp-backend llama \
  --prompt "Test"
```

### Error: "Library mismatch: expecting libbitnet but found llama"

**Cause**: The wrong backend library was loaded, possibly due to `LD_LIBRARY_PATH` conflicts.

**Fix**:

```bash
# Check which libraries are in your library path
echo $LD_LIBRARY_PATH | tr ':' '\n'

# Clean up conflicting paths and use only bitnet.cpp
export LD_LIBRARY_PATH="$HOME/.cache/bitnet_cpp/build/bin:$LD_LIBRARY_PATH"

# Rebuild to clear any cached library info
cargo clean -p xtask -p bitnet-crossval
cargo build -p xtask --features crossval-all
```

### Error: "Preflight check failed: backend unavailable"

**Cause**: Required C++ libraries weren't found during preflight checks (build time or runtime).

**Fix**:

```bash
# 1. Ensure BITNET_CPP_DIR is set at BUILD time
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/bin:$LD_LIBRARY_PATH"

# 2. Rebuild xtask to re-run build-time checks
cargo clean -p xtask -p bitnet-crossval
cargo build -p xtask --features crossval-all

# 3. Use --verbose to diagnose which libs are missing
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" \
  --max-tokens 1 \
  --verbose
```

### Error: Token IDs don't match between Rust and C++

**Cause**: Tokenizer mismatch between Rust and C++ implementations.

**Fix**:

```bash
# Use --dump-ids and --dump-cpp-ids to compare token sequences
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --dump-ids \
  --dump-cpp-ids \
  --verbose 2>&1 | head -50

# The output will show Rust vs C++ token IDs side-by-side
# If they diverge, investigate tokenizer differences
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

- [`docs/explanation/dual-backend-crossval.md`](../explanation/dual-backend-crossval.md) - Dual-backend architecture and design
- [`docs/development/validation-framework.md`](../development/validation-framework.md) - Complete validation architecture
- [`docs/howto/validate-models.md`](validate-models.md) - Model validation workflow
- [`CLAUDE.md`](../../CLAUDE.md) - Feature flags and cross-validation CLI reference
