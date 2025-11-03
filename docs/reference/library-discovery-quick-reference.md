# Library Discovery Quick Reference

## TL;DR - Library Search Chain

```
┌─ BITNET_CROSSVAL_LIBDIR (explicit)
├─ $BITNET_CPP_DIR/build/3rdparty/llama.cpp/src
├─ $BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src
├─ $BITNET_CPP_DIR/build/bin
├─ $BITNET_CPP_DIR/build/lib
└─ $BITNET_CPP_DIR/lib
    └─ Default: $HOME/.cache/bitnet_cpp
```

## Library Categories & Status

| Category | Status | Files | Location |
|----------|--------|-------|----------|
| **BitNet-specific** | ❌ NOT LINKED | libbitnet.{a,so,dylib} | `$BITNET_CPP_DIR/build/lib` |
| **llama.cpp core** | ✅ LINKED | libllama.{a,so,dylib} | `build/3rdparty/llama.cpp/src/` |
| **GGML** | ✅ LINKED (optional) | libggml.{a,so,dylib} | `build/3rdparty/llama.cpp/ggml/src/` |
| **Platform runtime** | ✅ LINKED | stdc++, libc++, pthread, etc | System paths |

## FFI Available Symbols

| Function | Purpose | From |
|----------|---------|------|
| `bitnet_model_new_from_file()` | Load model | bitnet_c_shim.cc |
| `bitnet_context_new()` | Create context | bitnet_c_shim.cc |
| `bitnet_tokenize()` | Tokenize text | bitnet_c_shim.cc |
| `bitnet_eval()` | Get logits | bitnet_c_shim.cc |
| `bitnet_prefill()` | Prefill context | bitnet_c_shim.cc |
| `bitnet_decode_greedy()` | Generate tokens | bitnet_c_shim.cc |

## Feature Gate Behavior

| Feature | Build Script | Behavior |
|---------|-------------|----------|
| `cpu` | bitnet-kernels/build.rs | Skip GPU link paths |
| `gpu`/`cuda` | bitnet-kernels/build.rs | Add CUDA paths & link libs |
| `ffi` | bitnet-sys/build.rs, crossval/build.rs | Run library discovery |
| No `ffi` | (skipped) | No C++ linking |

## Environment Variable Precedence

```
BITNET_CROSSVAL_LIBDIR (highest priority)
  ↓
BITNET_CPP_DIR
  ↓
BITNET_CPP_PATH (legacy)
  ↓
$HOME/.cache/bitnet_cpp (default)
```

## Current Limitations (Missing)

- ❌ No CUDA vs CPU backend detection
- ❌ No kernel capability registry
- ❌ No symbol availability checking
- ❌ No ABI compatibility validation
- ❌ No quantization format detection (I2_S, TL1, TL2, IQ2_S)

## Build Script File Handling

| Script | Discovers | Links | Fallback |
|--------|-----------|-------|----------|
| `crossval/build.rs` | ALL found libs | ALL found | Mock wrapper |
| `bitnet-sys/build.rs` | llama + ggml | Required (error if missing) | Panic |
| `bitnet-kernels/build.rs` | bitnet + llama + ggml | If found | None (CPU-only) |

## Runtime Setup

After building C++:

```bash
# Option 1: Use xtask helper
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Option 2: Manual setup
export BITNET_CPP_DIR=/path/to/bitnet_cpp
export LD_LIBRARY_PATH=$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1  # for determinism
```

## Testing with FFI

```bash
# Build with C++ support
BITNET_CPP_DIR=$HOME/.cache/bitnet_cpp \
  cargo test --features ffi -p crossval

# Check what was linked
cargo build --features ffi -p bitnet-sys -vv 2>&1 | grep "cargo:rustc-link"

# Run cross-validation
export BITNET_CPP_DIR=$HOME/.cache/bitnet_cpp
cargo run -p xtask -- crossval --model model.gguf
```

## Symbol Analysis (Not Yet Implemented)

To check library capabilities:

```bash
# Would show available functions
nm -D $BITNET_CPP_DIR/build/3rdparty/llama.cpp/src/libllama.so | grep bitnet_

# Would detect CUDA support
nm -D libllama.so | grep -E "cuda|cublas|__global__"

# Would show GGML quantization kernels
nm -D libggml.so | grep -E "ggml_quants|i2_s|tl1"
```

## Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `undefined reference to bitnet_model_new_from_file` | FFI not enabled | Build with `--features ffi` |
| `library not found for -lllama` | C++ not built | Run `cargo xtask fetch-cpp` |
| `BITNET_CPP_DIR not set` | Missing env var | Set `$BITNET_CPP_DIR` explicitly |
| Build succeeds but tests fail | Mock wrapper loaded | Check `LD_LIBRARY_PATH` |
| Wrong backend loaded | Feature mismatch | Rebuild with correct features |

