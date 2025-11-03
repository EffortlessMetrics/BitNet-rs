# BitNet.cpp AVAILABLE Mode Wiring Guide

**Date**: 2025-10-25
**Status**: Production-Ready Reference
**Target Audience**: Future maintainers and contributors

## Purpose

This document provides comprehensive guidance for integrating the BitNet.cpp C++ reference into the crossval crate's AVAILABLE mode. It covers all aspects of the FFI wiring, from header includes to platform-specific linking, troubleshooting common errors, and verification procedures.

## Overview

The crossval crate implements dual-compilation modes:

- **STUB mode** (default): No external dependencies, returns actionable errors
- **AVAILABLE mode** (when `BITNET_CPP_DIR` is set): Full BitNet.cpp integration via FFI

The AVAILABLE mode wiring connects Rust code to the llama.cpp C API that powers BitNet.cpp.

## Table of Contents

1. [Required Headers](#required-headers)
2. [Library Dependencies](#library-dependencies)
3. [Build System Configuration](#build-system-configuration)
4. [Symbol Visibility and Linking](#symbol-visibility-and-linking)
5. [Platform-Specific Notes](#platform-specific-notes)
6. [Common Compilation Errors](#common-compilation-errors)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Verification Checklist](#verification-checklist)

---

## Required Headers

### Primary Header

The only header you need is:

```cpp
#include <llama.h>
```

**Location**: `$BITNET_CPP_DIR/3rdparty/llama.cpp/include/llama.h`

**Why llama.h?**: BitNet.cpp is built on top of the llama.cpp framework. All tokenization and inference APIs are exposed through the llama.cpp C interface, not a separate BitNet API.

### Where Headers Come From

The build system automatically discovers headers from `BITNET_CPP_DIR`:

```rust
// In crossval/build.rs (AVAILABLE mode)
let bitnet_root = env::var("BITNET_CPP_DIR")?;
let include_path = Path::new(&bitnet_root).join("3rdparty/llama.cpp/include");

build.include(include_path);  // cc::Build adds -I flag
```

### Header Structure in BitNet.cpp

```
$BITNET_CPP_DIR/
├── 3rdparty/llama.cpp/
│   └── include/
│       └── llama.h              ← Primary C API
├── include/
│   └── ggml-bitnet.h            ← BitNet kernels (not needed for FFI)
└── build/
    └── lib/
        ├── libllama.so
        └── libggml.so
```

### Conditional Inclusion

In `crossval/src/bitnet_cpp_wrapper.cc`:

```cpp
#ifdef BITNET_AVAILABLE
#include <llama.h>
#endif
```

This prevents compilation errors when building in STUB mode.

---

## Library Dependencies

### Required Libraries

The AVAILABLE mode requires these shared libraries:

1. **libllama** (primary LLaMA.cpp API)
2. **libggml** (GGML tensor backend)

**Location**: `$BITNET_CPP_DIR/build/lib/`

**Naming conventions**:
- Linux: `libllama.so`, `libggml.so`
- macOS: `libllama.dylib`, `libggml.dylib`
- Windows: `llama.dll`, `ggml.dll` (or static `.lib`)

### Where Libraries Come From

The build system scans multiple directories for libraries:

```rust
// In crossval/build.rs
let possible_lib_dirs = vec![
    format!("{}/build", bitnet_root),
    format!("{}/build/lib", bitnet_root),
    format!("{}/build/3rdparty/llama.cpp/src", bitnet_root),
    format!("{}/build/3rdparty/llama.cpp/ggml/src", bitnet_root),
];

for lib_dir in &possible_lib_dirs {
    println!("cargo:rustc-link-search=native={}", lib_dir);

    // Auto-detect libllama*, libggml*
    if name.starts_with("libllama") || name.starts_with("libggml") {
        println!("cargo:rustc-link-lib=dylib={}", lib_name);
    }
}
```

### Library Discovery Priority

1. **Explicit override**: `BITNET_CROSSVAL_LIBDIR` (highest priority)
2. **Build directory**: `$BITNET_CPP_DIR/build`
3. **Build lib subdirectory**: `$BITNET_CPP_DIR/build/lib`
4. **LLaMA.cpp build**: `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src`
5. **GGML build**: `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src`

---

## Build System Configuration

### build.rs Overview

The crossval crate uses `build.rs` to:

1. Detect AVAILABLE vs STUB mode via `BITNET_CPP_DIR`
2. Compile `bitnet_cpp_wrapper.cc` with appropriate defines
3. Link C++ wrapper as static library
4. Discover and link external shared libraries
5. Emit environment variables for runtime detection

### AVAILABLE Mode Detection

```rust
// crossval/build.rs
let bitnet_available = env::var("BITNET_CPP_DIR").is_ok();

let mut build = cc::Build::new();
build.file("src/bitnet_cpp_wrapper.cc")
    .cpp(true)
    .flag_if_supported("-std=c++17");

if bitnet_available {
    build.define("BITNET_AVAILABLE", None);
    println!("cargo:warning=crossval: Compiling C++ wrapper in AVAILABLE mode");
} else {
    build.define("BITNET_STUB", None);
    println!("cargo:warning=crossval: Compiling C++ wrapper in STUB mode");
}

build.compile("bitnet_cpp_wrapper_cc");
```

### Compiler Flags

**Required minimum**: `-std=c++17`

**Platform-specific optimizations** (optional):
- `-O3` (optimization level)
- `-march=native` (CPU-specific SIMD)
- `-flto` (link-time optimization)

**Important**: The wrapper uses standard C++17 features (no exotic dependencies).

### Linking Configuration

#### Static Wrapper Library

The C++ wrapper is compiled into a static library and linked into the crate:

```rust
println!("cargo:rustc-link-lib=static=bitnet_cpp_wrapper_cc");
```

#### External Shared Libraries

Dynamic linking to BitNet.cpp libraries:

```rust
// Linux
#[cfg(target_os = "linux")]
println!("cargo:rustc-link-lib=dylib=stdc++");

// macOS
#[cfg(target_os = "macos")]
println!("cargo:rustc-link-lib=dylib=c++");
```

### Build-Time Environment Variables

The build script emits these for runtime detection:

```rust
println!("cargo:rustc-env=CROSSVAL_HAS_BITNET={}", found_bitnet);
println!("cargo:rustc-env=CROSSVAL_HAS_LLAMA={}", found_llama);
```

These are checked in Rust code:

```rust
// In cpp_bindings.rs
if !matches!(option_env!("CROSSVAL_HAS_BITNET"), Some("true")) {
    return Err(CrossvalError::CppNotAvailable);
}
```

---

## Symbol Visibility and Linking

### C ABI Export

All FFI functions use `extern "C"` linkage to prevent C++ name mangling:

```cpp
extern "C" {

int crossval_bitnet_tokenize(
    const char* model_path,
    // ... other args ...
) {
    // Implementation
}

int crossval_bitnet_eval_with_tokens(
    const char* model_path,
    // ... other args ...
) {
    // Implementation
}

}  // extern "C"
```

### Name Prefixing

All wrapper functions use the `crossval_` prefix to avoid symbol conflicts:

- `crossval_bitnet_tokenize` (not `bitnet_tokenize`)
- `crossval_bitnet_eval_with_tokens` (not `bitnet_eval_with_tokens`)

**Rationale**: Prevents collisions if BitNet.cpp or llama.cpp export similarly named symbols.

### Rust FFI Declarations

Corresponding Rust declarations in `crossval/src/cpp_bindings.rs`:

```rust
unsafe extern "C" {
    fn crossval_bitnet_tokenize(
        model_path: *const c_char,
        prompt: *const c_char,
        add_bos: c_int,
        parse_special: c_int,
        out_tokens: *mut i32,
        out_capacity: i32,
        out_len: *mut i32,
        err: *mut c_char,
        err_len: i32,
    ) -> c_int;

    fn crossval_bitnet_eval_with_tokens(
        model_path: *const c_char,
        tokens: *const i32,
        n_tokens: i32,
        n_ctx: i32,
        out_logits: *mut f32,
        logits_capacity: i32,
        out_rows: *mut i32,
        out_cols: *mut i32,
        err: *mut c_char,
        err_len: i32,
    ) -> c_int;
}
```

### Symbol Resolution at Link Time

When linking succeeds, you'll see in build output:

```
cargo:rustc-link-lib=dylib=llama
cargo:rustc-link-lib=dylib=ggml
cargo:warning=crossval: Linked libraries: llama, ggml
```

---

## Platform-Specific Notes

### Linux

#### Standard Library Linking

```rust
#[cfg(target_os = "linux")]
println!("cargo:rustc-link-lib=dylib=stdc++");
```

**Why**: Links the GNU C++ standard library required for C++ code.

#### Runtime Library Path

Set `LD_LIBRARY_PATH` so the dynamic linker finds shared libraries:

```bash
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/lib:$LD_LIBRARY_PATH"
```

**Common issue**: If `LD_LIBRARY_PATH` is not set, you'll get:

```
error while loading shared libraries: libllama.so: cannot open shared object file
```

#### Compiler Compatibility

- **GCC**: 7.x or newer (C++17 support)
- **Clang**: 6.x or newer

#### Build Command Example

```bash
export BITNET_CPP_DIR=/path/to/bitnet.cpp
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/lib:$LD_LIBRARY_PATH"
cargo build --package crossval --features ffi
```

---

### macOS

#### Standard Library Linking

```rust
#[cfg(target_os = "macos")]
println!("cargo:rustc-link-lib=dylib=c++");
```

**Why**: Links the LLVM C++ standard library (libc++) used on macOS.

#### Runtime Library Path

Set `DYLD_LIBRARY_PATH` (macOS equivalent of `LD_LIBRARY_PATH`):

```bash
export DYLD_LIBRARY_PATH="$BITNET_CPP_DIR/build/lib:$DYLD_LIBRARY_PATH"
```

**macOS SIP Note**: System Integrity Protection may ignore `DYLD_LIBRARY_PATH` for signed binaries. Workarounds:

1. Copy `.dylib` files to `/usr/local/lib` (requires sudo)
2. Use `install_name_tool` to embed library paths (advanced)
3. Build with static linking (see below)

#### Compiler Compatibility

- **Xcode Command Line Tools**: 12.x or newer
- **Apple Clang**: Built-in with Xcode
- **Homebrew LLVM**: Alternative, but may require extra flags

#### Build Command Example

```bash
export BITNET_CPP_DIR=/path/to/bitnet.cpp
export DYLD_LIBRARY_PATH="$BITNET_CPP_DIR/build/lib:$DYLD_LIBRARY_PATH"
cargo build --package crossval --features ffi
```

---

### Windows

#### Standard Library Linking (MSVC)

Windows uses Visual Studio's C Runtime (CRT). You must match the CRT linking mode:

**Dynamic CRT** (`/MD` or `/MDd`):
```rust
// In build.rs (Windows-specific)
build.flag("/MD");  // Release
// or
build.flag("/MDd"); // Debug
```

**Static CRT** (`/MT` or `/MTd`):
```rust
build.flag("/MT");  // Release
// or
build.flag("/MTd"); // Debug
```

**Critical**: All linked libraries (Rust crate, wrapper, BitNet.cpp) must use the **same CRT mode**.

**Mismatch symptoms**:
- Linker errors: `LNK2005` (duplicate symbols)
- Runtime crashes: Heap corruption, invalid pointers
- Missing symbols: `_CrtDbgReport`, `__imp__malloc`

#### DLL vs Static Linking

**DLL (shared) linking**:
```rust
println!("cargo:rustc-link-lib=dylib=llama");
println!("cargo:rustc-link-lib=dylib=ggml");
```

**Static linking**:
```rust
println!("cargo:rustc-link-lib=static=llama");
println!("cargo:rustc-link-lib=static=ggml");
```

**Recommendation**: Use DLLs to avoid CRT mismatch issues.

#### pragma comment(lib, ...) Directives

If BitNet.cpp uses `#pragma comment(lib, "...")` to auto-link dependencies, ensure those libraries are in your linker path:

```cpp
// Example (hypothetical)
#pragma comment(lib, "llama.lib")
#pragma comment(lib, "ggml.lib")
```

**Workaround**: Explicitly add to `build.rs` or use `/NODEFAULTLIB` to override.

#### Runtime Library Path

Set `PATH` so Windows finds DLLs:

```powershell
$env:PATH = "$env:BITNET_CPP_DIR\build\bin;$env:PATH"
```

**Common issue**: DLL not found errors at runtime:

```
The code execution cannot proceed because llama.dll was not found
```

#### Compiler Compatibility

- **MSVC**: Visual Studio 2019 or newer (C++17 support)
- **MinGW-w64**: GCC 7.x or newer (C++17 support)
- **Clang-cl**: LLVM for Windows 10.x or newer

**Note**: MSVC is the primary supported compiler on Windows. MinGW may work but is less tested.

#### Build Command Example (PowerShell)

```powershell
$env:BITNET_CPP_DIR = "C:\path\to\bitnet.cpp"
$env:PATH = "$env:BITNET_CPP_DIR\build\bin;$env:PATH"
cargo build --package crossval --features ffi
```

---

## Common Compilation Errors

### Error: "Undefined reference to llama_*"

**Symptom**:
```
undefined reference to `llama_tokenize'
undefined reference to `llama_load_model_from_file'
```

**Cause**: Linker cannot find `libllama.so` or `libggml.so`.

**Fix**:

1. **Check BITNET_CPP_DIR is set**:
   ```bash
   echo $BITNET_CPP_DIR
   ```

2. **Verify libraries exist**:
   ```bash
   ls -la $BITNET_CPP_DIR/build/lib/libllama*
   ls -la $BITNET_CPP_DIR/build/lib/libggml*
   ```

3. **Check build output for link directives**:
   ```
   cargo:rustc-link-search=native=/path/to/bitnet.cpp/build/lib
   cargo:rustc-link-lib=dylib=llama
   cargo:rustc-link-lib=dylib=ggml
   ```

4. **Rebuild BitNet.cpp if libraries missing**:
   ```bash
   cargo run -p xtask -- fetch-cpp --backend cpu
   ```

---

### Error: "Header not found: llama.h"

**Symptom**:
```
fatal error: llama.h: No such file or directory
```

**Cause**: Compiler cannot find `llama.h` in include paths.

**Fix**:

1. **Check header exists**:
   ```bash
   ls -la $BITNET_CPP_DIR/3rdparty/llama.cpp/include/llama.h
   ```

2. **Verify build.rs adds include path** (should see in build output):
   ```
   cargo:warning=crossval: Include path: /path/to/bitnet.cpp/3rdparty/llama.cpp/include
   ```

3. **Manual override** (temporary diagnostic):
   ```rust
   // In crossval/build.rs
   build.include("/path/to/bitnet.cpp/3rdparty/llama.cpp/include");
   ```

4. **Rebuild crossval**:
   ```bash
   cargo clean -p crossval
   cargo build -p crossval --features ffi
   ```

---

### Error: "Duplicate symbol: ..."

**Symptom**:
```
duplicate symbol '_some_function' in:
  libbitnet_cpp_wrapper_cc.a
  libllama.so
```

**Cause**: Symbol conflict between wrapper and external libraries.

**Fix**:

1. **Use crossval_ prefix** for all wrapper functions (already done)
2. **Avoid global symbols** in wrapper (use static or anonymous namespaces)
3. **Check for name collisions** in BitNet.cpp:
   ```bash
   nm -g $BITNET_CPP_DIR/build/lib/libllama.so | grep crossval
   ```

4. **If collision confirmed**, rename wrapper functions

---

### Error: "Runtime library not found"

**Symptom** (Linux):
```
error while loading shared libraries: libllama.so: cannot open shared object file
```

**Symptom** (macOS):
```
dyld: Library not loaded: libllama.dylib
```

**Symptom** (Windows):
```
The code execution cannot proceed because llama.dll was not found
```

**Cause**: Dynamic linker cannot find shared libraries at runtime.

**Fix**:

**Linux**:
```bash
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/lib:$LD_LIBRARY_PATH"
```

**macOS**:
```bash
export DYLD_LIBRARY_PATH="$BITNET_CPP_DIR/build/lib:$DYLD_LIBRARY_PATH"
```

**Windows**:
```powershell
$env:PATH = "$env:BITNET_CPP_DIR\build\bin;$env:PATH"
```

**Persistent solution**: Add to shell profile (~/.bashrc, ~/.zshrc, etc.)

**Alternative**: Use static linking (requires rebuilding BitNet.cpp with static libraries)

---

### Error: "Undefined behavior: null pointer dereference"

**Symptom**: Segmentation fault or access violation.

**Cause**: NULL pointer passed to C++ code or returned from llama.cpp.

**Common locations**:
- `llama_load_model_from_file()` returns NULL on failure
- `llama_new_context_with_model()` returns NULL on failure
- `llama_get_logits()` returns NULL if context not initialized

**Fix**: Always check pointers before use:

```cpp
struct llama_model* model = llama_load_model_from_file(path, params);
if (!model) {
    snprintf(err, err_len, "Failed to load model");
    return -1;
}
```

**Verification**: Run with AddressSanitizer:

```bash
RUSTFLAGS="-Z sanitizer=address" cargo build --package crossval --features ffi
```

---

## Troubleshooting Guide

### Diagnostic: Check Build Mode

**Command**:
```bash
cargo build -p crossval --features ffi 2>&1 | grep "wrapper in"
```

**Expected output (AVAILABLE mode)**:
```
warning: crossval: Compiling C++ wrapper in AVAILABLE mode
```

**Expected output (STUB mode)**:
```
warning: crossval: Compiling C++ wrapper in STUB mode
```

**If STUB mode but you want AVAILABLE**:
```bash
export BITNET_CPP_DIR=/path/to/bitnet.cpp
```

---

### Diagnostic: Check Library Discovery

**Command**:
```bash
cargo build -p crossval --features ffi 2>&1 | grep "Linked libraries"
```

**Expected output (dual-backend)**:
```
warning: crossval: Both bitnet.cpp and llama.cpp libraries found (dual-backend support)
warning: crossval: Linked libraries: bitnet, llama, ggml
```

**Expected output (llama only)**:
```
warning: crossval: Found llama.cpp libraries only (LLaMA parity supported)
warning: crossval: Linked libraries: llama, ggml
```

**If no libraries found**:
```
warning: crossval: No C++ libraries found (crossval will use mock/stub)
```

**Fix**: Check `BITNET_CPP_DIR` and rebuild BitNet.cpp.

---

### Diagnostic: Check Symbol Resolution

**Command** (Linux/macOS):
```bash
nm -gC target/debug/libcrossval.rlib | grep crossval_bitnet
```

**Expected output**:
```
T crossval_bitnet_tokenize
T crossval_bitnet_eval_with_tokens
```

**If symbols missing**: Wrapper not compiled correctly. Check build.rs.

---

### Diagnostic: Runtime Library Path

**Command** (Linux):
```bash
ldd target/debug/libcrossval.so | grep llama
```

**Expected output**:
```
libllama.so => /path/to/bitnet.cpp/build/lib/libllama.so
```

**If "not found"**: Set `LD_LIBRARY_PATH`.

**Command** (macOS):
```bash
otool -L target/debug/libcrossval.dylib | grep llama
```

**Expected output**:
```
libllama.dylib (compatibility version 0.0.0, current version 0.0.0)
```

---

### Diagnostic: Test FFI Functions

**Smoke test**:
```bash
cargo test -p crossval --features ffi --test cpp_wrapper_smoke_test
```

**Expected output (STUB mode)**:
```
test ffi_tests::test_bitnet_tokenize_stub_mode ... ok
```

**Expected output (AVAILABLE mode)**:
```
test ffi_tests::test_bitnet_tokenize_stub_mode ... SKIPPED (AVAILABLE mode)
test ffi_tests::test_bitnet_tokenize_integration ... ok
```

---

### Diagnostic: Check Runtime Detection

**Command**:
```bash
cargo test -p crossval --features ffi has_bitnet -- --nocapture
```

**Look for**:
```
CROSSVAL_HAS_BITNET = true
CROSSVAL_HAS_LLAMA = true
```

**If "false" but AVAILABLE mode compiled**:
- Build system didn't find libraries during build
- Rebuild with verbose output: `cargo build -p crossval --features ffi -vv`

---

## Verification Checklist

Use this checklist when wiring or troubleshooting AVAILABLE mode:

### Build-Time Checks

- [ ] **BITNET_CPP_DIR is set**
  ```bash
  echo $BITNET_CPP_DIR
  # Should output: /path/to/bitnet.cpp (not empty)
  ```

- [ ] **Headers found during build**
  ```bash
  ls -la $BITNET_CPP_DIR/3rdparty/llama.cpp/include/llama.h
  # Should show file (not "No such file or directory")
  ```

- [ ] **Libraries exist**
  ```bash
  ls -la $BITNET_CPP_DIR/build/lib/libllama*
  ls -la $BITNET_CPP_DIR/build/lib/libggml*
  # Should list .so/.dylib files
  ```

- [ ] **Build output shows AVAILABLE mode**
  ```bash
  cargo build -p crossval --features ffi 2>&1 | grep "AVAILABLE mode"
  # Should show: "Compiling C++ wrapper in AVAILABLE mode"
  ```

- [ ] **BITNET_AVAILABLE defined in C++ code**
  ```bash
  cargo build -p crossval --features ffi -vv 2>&1 | grep "BITNET_AVAILABLE"
  # Should show: -DBITNET_AVAILABLE
  ```

- [ ] **CROSSVAL_HAS_BITNET=true emitted**
  ```bash
  cargo build -p crossval --features ffi 2>&1 | grep "CROSSVAL_HAS_BITNET"
  # Should show: cargo:rustc-env=CROSSVAL_HAS_BITNET=true
  ```

- [ ] **Libraries linked successfully**
  ```bash
  cargo build -p crossval --features ffi 2>&1 | grep "Linked libraries"
  # Should show: "Linked libraries: llama, ggml" (or bitnet, llama, ggml)
  ```

---

### Runtime Checks

- [ ] **Dynamic loader can find libraries**
  ```bash
  # Linux
  ldd target/debug/libcrossval.so | grep -E "llama|ggml"

  # macOS
  otool -L target/debug/libcrossval.dylib | grep -E "llama|ggml"

  # Should NOT show "not found"
  ```

- [ ] **LD_LIBRARY_PATH set** (Linux)
  ```bash
  echo $LD_LIBRARY_PATH | grep bitnet
  # Should show path containing bitnet.cpp/build/lib
  ```

- [ ] **DYLD_LIBRARY_PATH set** (macOS)
  ```bash
  echo $DYLD_LIBRARY_PATH | grep bitnet
  # Should show path containing bitnet.cpp/build/lib
  ```

- [ ] **PATH set** (Windows)
  ```powershell
  echo $env:PATH | Select-String bitnet
  # Should show path containing bitnet.cpp\build\bin
  ```

---

### Functional Checks

- [ ] **Tokenization smoke test passes**
  ```bash
  cargo test -p crossval --features ffi test_bitnet_tokenize
  # Should show: test result: ok
  ```

- [ ] **Evaluation smoke test passes**
  ```bash
  cargo test -p crossval --features ffi test_bitnet_eval
  # Should show: test result: ok
  ```

- [ ] **Integration test with real model** (requires model file)
  ```bash
  export BITNET_GGUF=/path/to/model.gguf
  cargo test -p crossval --features ffi,crossval dual_backend
  # Should show: Backend Library Status: ✓ bitnet.cpp: AVAILABLE
  ```

- [ ] **No memory leaks** (optional, requires Valgrind/AddressSanitizer)
  ```bash
  RUSTFLAGS="-Z sanitizer=address" cargo test -p crossval --features ffi
  # Should complete without LeakSanitizer errors
  ```

---

### Cross-Platform Verification

- [ ] **Linux: GCC or Clang builds successfully**
- [ ] **macOS: Xcode Command Line Tools builds successfully**
- [ ] **Windows: MSVC builds successfully** (if targeting Windows)

---

## Quick Reference Card

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `BITNET_CPP_DIR` | Root of BitNet.cpp installation | `/home/user/.cache/bitnet_cpp` |
| `BITNET_CROSSVAL_LIBDIR` | Override library search path | `/custom/lib/path` |
| `LD_LIBRARY_PATH` | Linux runtime library path | `$BITNET_CPP_DIR/build/lib` |
| `DYLD_LIBRARY_PATH` | macOS runtime library path | `$BITNET_CPP_DIR/build/lib` |
| `PATH` | Windows runtime library path | `%BITNET_CPP_DIR%\build\bin` |

### Build Flags

| Platform | Standard Library | Command |
|----------|------------------|---------|
| Linux | `libstdc++` | `cargo:rustc-link-lib=dylib=stdc++` |
| macOS | `libc++` | `cargo:rustc-link-lib=dylib=c++` |
| Windows | MSVC CRT | `/MD` (dynamic) or `/MT` (static) |

### Common Commands

```bash
# Setup environment (one-time)
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Build crossval with FFI
cargo build -p crossval --features ffi

# Run smoke tests
cargo test -p crossval --features ffi --test cpp_wrapper_smoke_test

# Check library discovery
cargo build -p crossval --features ffi 2>&1 | grep "Linked libraries"

# Verify runtime linking (Linux)
ldd target/debug/libcrossval.so | grep llama

# Verify runtime linking (macOS)
otool -L target/debug/libcrossval.dylib | grep llama
```

---

## Next Steps After Wiring

Once AVAILABLE mode is wired and verified:

1. **Replace TODOs in bitnet_cpp_wrapper.cc**: Uncomment placeholder implementation code
2. **Test with real GGUF models**: Run integration tests with actual BitNet models
3. **Benchmark performance**: Compare Rust vs C++ inference speed
4. **Add session caching** (post-MVP): Avoid reloading models per call
5. **GPU support**: Set `n_gpu_layers > 0` for GPU acceleration
6. **Expand test coverage**: Add property-based tests, fuzz testing

---

## References

- **Primary documentation**: `docs/specs/bitnet-cpp-api-requirements.md`
- **Build patterns**: `crossval/build.rs`
- **FFI bindings**: `crossval/src/cpp_bindings.rs`
- **Wrapper implementation**: `crossval/src/bitnet_cpp_wrapper.cc`
- **Smoke tests**: `crossval/tests/cpp_wrapper_smoke_test.rs`
- **Setup guide**: `docs/howto/cpp-setup.md`
- **llama.cpp API**: `$BITNET_CPP_DIR/3rdparty/llama.cpp/include/llama.h`

---

## Acceptance Criteria Summary

This wiring guide is complete and production-ready if:

- ✅ All required headers documented
- ✅ Library dependencies identified
- ✅ Build system configuration explained
- ✅ Symbol visibility best practices covered
- ✅ Platform-specific notes (Linux, macOS, Windows) provided
- ✅ Common compilation errors documented with fixes
- ✅ Troubleshooting guide with diagnostic commands
- ✅ Verification checklist with concrete steps
- ✅ Quick reference card for maintainers

---

**Maintainer Note**: When modifying the FFI layer, always verify against this checklist and update this document if new edge cases are discovered.
