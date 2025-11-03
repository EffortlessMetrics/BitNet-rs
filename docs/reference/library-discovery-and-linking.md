# BitNet.rs Library Discovery & Linking Architecture

## Executive Summary

The BitNet.rs build system implements a **flexible, multi-backend library discovery pattern** with support for:
- Optional FFI integration to Microsoft BitNet C++ reference
- Automatic library detection from standard llama.cpp build locations
- Environment variable overrides for custom paths
- Runtime fallbacks to stub implementations when libraries are unavailable
- GPU/CUDA detection with feature-gated compilation

**Key Finding:** The system is designed for **optional FFI** - libraries are discovered but the build succeeds even if not found, using mock implementations instead.

---

## 1. Current Library Discovery Status

### 1.1 What Gets Discovered

The build system searches for three categories of libraries:

#### Category 1: BitNet-Specific Libraries
```
libbitnet*.a (static)
libbitnet*.so (dynamic, Linux)
libbitnet*.dylib (dynamic, macOS)
```
**Current Status:** NOT ACTIVELY LINKED
- Search path: `$BITNET_CPP_DIR/build/lib` (bitnet-kernels/build.rs)
- Note: These are placeholders; actual BitNet functionality routes through llama.cpp

#### Category 2: llama.cpp Libraries
```
libllama.a
libllama.so
libllama.dylib
libggml.a
libggml.so
libggml.dylib
```
**Current Status:** ACTIVELY DISCOVERED AND LINKED
- Priority paths (in order):
  1. `$BITNET_CROSSVAL_LIBDIR` (explicit override)
  2. `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src`
  3. `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src`
  4. `$BITNET_CPP_DIR/build/bin`
  5. `$BITNET_CPP_DIR/build/lib` (legacy)

#### Category 3: Platform Runtime Dependencies
```
Linux:    libstdc++, libpthread, libdl, libm, libgomp
macOS:    libc++ (C++ stdlib), Accelerate framework
Windows:  msvcrt
```
**Current Status:** CONDITIONALLY LINKED based on platform

### 1.2 Library Search Flowchart

```
┌─────────────────────────────────────────────────────────────┐
│ Build Script Invoked (crossval/build.rs or bitnet-sys)      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├─ Is ffi feature enabled?
                 │  ├─ NO → Skip library discovery, return
                 │  └─ YES → Continue
                 │
                 ├─ Resolve base directory:
                 │  ├─ BITNET_CPP_DIR (preferred)
                 │  ├─ BITNET_CPP_PATH (legacy)
                 │  └─ $HOME/.cache/bitnet_cpp (default)
                 │
                 ├─ Search library paths (in priority order):
                 │  ├─ BITNET_CROSSVAL_LIBDIR (if set)
                 │  ├─ {base}/build/3rdparty/llama.cpp/src
                 │  ├─ {base}/build/3rdparty/llama.cpp/ggml/src
                 │  ├─ {base}/build/bin
                 │  └─ {base}/build/lib
                 │
                 ├─ Find libraries matching:
                 │  ├─ libbitnet*
                 │  ├─ libllama*
                 │  └─ libggml*
                 │
                 └─ Link if found, else emit warning and use stubs
```

---

## 2. Where Libraries Are Located

### 2.1 Standard BitNet C++ Build Structure

After running `./ci/fetch_bitnet_cpp.sh`, the directory structure is:

```
$HOME/.cache/bitnet_cpp/  (or $BITNET_CPP_DIR/)
├── 3rdparty/
│   └── llama.cpp/
│       ├── include/           (llama.h, ggml.h headers)
│       ├── ggml/
│       │   ├── include/       (ggml headers)
│       │   └── src/
│       │       ├── libggml.a  ← DISCOVERED HERE (static)
│       │       └── libggml.so ← OR HERE (shared)
│       └── src/
│           ├── libllama.a     ← DISCOVERED HERE (static)
│           └── libllama.so    ← OR HERE (shared)
│
├── build/
│   ├── bin/                   (CLI binaries)
│   ├── lib/
│   │   ├── libbitnet.a
│   │   └── libbitnet.so
│   ├── 3rdparty/
│   │   └── llama.cpp/
│   │       ├── src/           (libllama output)
│   │       ├── ggml/src/      (libggml output)
│   │       └── include/       (build-time generated headers)
│   └── CMakeCache.txt
│
├── include/
│   ├── llama.h
│   ├── ggml-bitnet.h          (optional, not always present)
│   └── bitnet-lut-kernels.h
│
├── src/
│   └── llama.h                (alternative location)
│
└── CMakeLists.txt
```

### 2.2 Actual Library Locations (Verified)

From `fetch_bitnet_cpp.sh` (lines 270-307):

**Linux/macOS:**
```bash
# Primary locations checked after build
$CACHE_DIR/build/3rdparty/llama.cpp/src/libllama.a
$CACHE_DIR/build/3rdparty/llama.cpp/ggml/src/libggml.a

# Alternative locations (fallback search)
$CACHE_DIR/build/3rdparty/llama.cpp/libllama.a
$CACHE_DIR/build/lib/libllama.a
$CACHE_DIR/build/libllama.a
```

### 2.3 Current Build Configuration

From `ci/fetch_bitnet_cpp.sh` (lines 251-261):

```cmake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=ON \          # ← Enable shared library build
    -DBITNET_BUILD_TESTS=OFF \
    -DBITNET_BUILD_EXAMPLES=ON \
    -DBITNET_CUDA=OFF \               # ← CPU-only by default
    -DBITNET_METAL=OFF \
    -DBITNET_BLAS=OFF \
    $CMAKE_FLAGS
```

**Note:** `BUILD_SHARED_LIBS=ON` builds `.so`/`.dylib` files, but the search still looks for `.a` files first (fallback pattern).

---

## 3. How Library Selection Works

### 3.1 Discovery Order (crossval/build.rs, lines 40-94)

```rust
// Priority 1: Explicit libdir environment variable
if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
    possible_lib_dirs.push(Path::new(&lib_dir).to_path_buf());
}

// Priority 2-3: Standard CMake output paths
possible_lib_dirs.push("$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src");
possible_lib_dirs.push("$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src");

// Priority 4-6: Alternative/legacy paths
possible_lib_dirs.push("$BITNET_CPP_DIR/build/bin");
possible_lib_dirs.push("$BITNET_CPP_DIR/build/lib");
possible_lib_dirs.push("$BITNET_CPP_DIR/build");
possible_lib_dirs.push("$BITNET_CPP_DIR/lib");

// For each directory that exists, search for libraries
for lib_dir in &possible_lib_dirs {
    for entry in fs::read_dir(lib_dir) {
        if name.starts_with("libbitnet") 
           || name.starts_with("libllama") 
           || name.starts_with("libggml") {
            println!("cargo:rustc-link-lib=dylib={}", lib_name);
        }
    }
}
```

### 3.2 Library Selection Pattern

**File Extension Priority (implicit):**
1. Any found `.so` or `.dylib` → linked as `dylib`
2. Any found `.a` → NOT linked in crossval/build.rs (only in bitnet-sys/build.rs)

**Actual Linking Logic (bitnet-sys/build.rs, lines 115-122):**

```rust
// Link the main llama library (REQUIRED if FFI enabled)
println!("cargo:rustc-link-lib=dylib=llama");

// Only link ggml if it exists as a separate library
if lib_search_paths.iter().any(|dir| lib_present(dir, "ggml")) {
    println!("cargo:rustc-link-lib=dylib=ggml");
}
```

**Key Difference from crossval:**
- `bitnet-sys` links LLAMA library **always** (fails if missing and FFI enabled)
- `crossval` links **all found** libraries (gracefully falls back to mock)

### 3.3 Backend Selection (GPU Detection)

From `bitnet-kernels/build.rs` (lines 30-61):

```rust
// Unified GPU detection: honor both "gpu" and "cuda" features
let gpu = env::var_os("CARGO_FEATURE_GPU").is_some() 
       || env::var_os("CARGO_FEATURE_CUDA").is_some();

if gpu {
    // Emit build-time cfg flag
    println!("cargo:rustc-cfg=bitnet_build_gpu");
    
    // Add CUDA library paths
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64/stubs");
    // ... more paths
    
    // Link CUDA libraries
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=nvrtc");
    println!("cargo:rustc-link-lib=cublas");
}
```

**No backend selection logic** - GPU is compile-time only, not discovered.

---

## 4. Available Symbols & Backend Determination

### 4.1 C Interface Symbols

The FFI exposes these symbols (from `crates/bitnet-sys/include/bitnet_c.h`):

**Model Management:**
- `bitnet_model_t* bitnet_model_new_from_file(const char* gguf_path)`
- `void bitnet_model_free(bitnet_model_t*)`

**Context Management:**
- `bitnet_ctx_t* bitnet_context_new(bitnet_model_t*, const bitnet_params_t*)`
- `void bitnet_context_free(bitnet_ctx_t*)`

**Inference:**
- `int bitnet_tokenize(bitnet_model_t*, const char* text, int add_bos, int parse_special, int32_t* out_ids, int out_cap)`
- `int bitnet_eval(bitnet_ctx_t*, const int32_t* ids, int n_ids, float* logits_out, int logits_cap)`
- `int bitnet_prefill(bitnet_ctx_t*, const int32_t* ids, int n_ids)`
- `int bitnet_decode_greedy(bitnet_model_t* model, bitnet_ctx_t* ctx, int eos_id, int eot_id, int max_steps, int* out_token_ids, int out_cap)`

**Utility:**
- `int bitnet_vocab_size(bitnet_ctx_t* ctx)`

### 4.2 Backend Detection at Runtime

No explicit backend symbol checking exists. Instead:

1. **Feature gates** (`--features gpu` vs `--features cpu`) control compilation
2. **Symbol availability** is detected via dynamic library loading (not checked at link time)
3. **Graceful degradation** - if C++ library not found, build uses mock stub implementation

### 4.3 How to Determine Which Backend Is Loaded

**At Build Time:**
```bash
# Check if GPU code compiled
cargo build --features gpu,ffi
# Look for: "cargo:rustc-cfg=bitnet_build_gpu"
```

**At Runtime:**
```rust
// Feature-gated code paths (from source)
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_function() { /* ... */ }

// Check available kernels via receipt
// (see next section)
```

---

## 5. Build System Changes Needed for Dual-Backend Support

### 5.1 Current Limitations

**Issue: No explicit kernel backend detection**

The system currently:
- ✅ Detects whether libraries exist (presence/absence)
- ✅ Supports compile-time GPU vs CPU selection (feature gates)
- ✅ Links platform-specific runtime dependencies
- ❌ Does NOT distinguish between CUDA-enabled vs CPU-only bitnet.cpp builds
- ❌ Does NOT detect which quantization kernels are compiled (I2_S_QK256, TL1, TL2, IQ2_S)
- ❌ Does NOT perform runtime kernel capability checking

### 5.2 Proposed Changes for Backend Selection

#### Step 1: Add Backend Detection Environment Variable

```bash
# New env var to track backend choice
export BITNET_BACKEND=cuda    # or cpu (default)
```

#### Step 2: Enhance Library Search to Distinguish Backends

**Modify `crossval/build.rs` and `bitnet-sys/build.rs`:**

```rust
// Detect which backend the C++ library was built for
let backend = env::var("BITNET_BACKEND").unwrap_or("cpu".to_string());

// Search for backend-specific library markers
for lib_dir in &possible_lib_dirs {
    // Check for CUDA marker files
    if backend == "cuda" {
        if Path::new(&lib_dir).join("libcuda.so").exists() {
            println!("cargo:rustc-cfg=bitnet_backend_cuda");
        }
    }
    
    // Check for GGML flags in library metadata
    if has_ggml_cuda_support(&lib_dir) {
        println!("cargo:rustc-cfg=bitnet_has_ggml_cuda");
    }
}
```

#### Step 3: Add Kernel Capability Registry

Create `bitnet-kernel-registry.rs`:

```rust
pub enum KernelBackend {
    I2S_QK256_Scalar,
    I2S_QK256_AVX2,
    I2S_QK256_CUDA,
    TL1_ARM_NEON,
    TL1_X86_AVX2,
    TL2_AVX512,
    IQ2S_GGML,
}

pub struct KernelCapabilities {
    pub available: Vec<KernelBackend>,
    pub preferred: KernelBackend,
    pub cuda_available: bool,
    pub cpu_simd_level: CpuSimdLevel,
}
```

#### Step 4: Implement nm-based Symbol Extraction

Add to `xtask`:

```rust
pub fn analyze_library_symbols(lib_path: &Path) -> Result<Vec<String>> {
    let output = Command::new("nm")
        .arg("-D")
        .arg(lib_path)
        .output()?;
    
    // Parse symbols to detect:
    // - bitnet_* (bitnet-specific)
    // - llama_* (llama.cpp generic)
    // - __cuda* (CUDA support)
    // - _ZN (C++ mangled names)
}
```

---

## 6. Current Library Linking Verified

### 6.1 Automatic Detection Works For:

✅ **Presence/Absence Detection**
- Checks if libraries exist in expected locations
- Fails gracefully with mock implementations if missing

✅ **Platform-Specific Linking**
- Linux: links libstdc++, libpthread, libdl, libm, libgomp
- macOS: links libc++, Accelerate
- Windows: links msvcrt

✅ **Environment Variable Overrides**
- `BITNET_CPP_DIR` - base directory
- `BITNET_CPP_PATH` - legacy name
- `BITNET_CROSSVAL_LIBDIR` - explicit lib directory

✅ **Multiple Library Format Support**
- `.a` (static)
- `.so` (Linux shared)
- `.dylib` (macOS shared)

### 6.2 NOT Currently Detected:

❌ **CUDA vs CPU build variant** of bitnet.cpp
❌ **Which quantization kernels** are compiled in
❌ **Kernel capability** (SIMD level, GPU availability)
❌ **Symbol availability** in loaded library
❌ **ABI compatibility** between Rust and C++ code

---

## 7. Environment Variable Reference

### 7.1 Library Path Configuration

| Variable | Default | Purpose | Example |
|----------|---------|---------|---------|
| `BITNET_CPP_DIR` | `$HOME/.cache/bitnet_cpp` | Root of built C++ repository | `/opt/bitnet_cpp` |
| `BITNET_CPP_PATH` | (same as above) | Legacy name for BITNET_CPP_DIR | `/home/user/bitnet` |
| `BITNET_CROSSVAL_LIBDIR` | (auto-detect) | Explicit library directory override | `/usr/local/lib` |

### 7.2 Runtime Configuration

| Variable | Values | Purpose |
|----------|--------|---------|
| `OMP_NUM_THREADS` | N | OpenMP thread count (set to 1 for determinism) |
| `GGML_NUM_THREADS` | N | GGML thread count (set to 1 for determinism) |
| `LD_LIBRARY_PATH` | paths | Linux dynamic loader paths (output by setup-cpp-auto) |
| `DYLD_LIBRARY_PATH` | paths | macOS dynamic loader paths (output by setup-cpp-auto) |

---

## 8. Build Process Flowchart

```
User runs: cargo build --features ffi

    ↓
    
1. Parse feature flags
   - GPU enabled? → Add CUDA link paths
   - FFI enabled? → Run library discovery

    ↓

2. Resolve base directories
   - Check BITNET_CPP_DIR env var
   - Fall back to $HOME/.cache/bitnet_cpp
   
    ↓

3. Search for libraries in order:
   a) BITNET_CROSSVAL_LIBDIR (explicit override)
   b) build/3rdparty/llama.cpp/src (CMake standard)
   c) build/3rdparty/llama.cpp/ggml/src (GGML location)
   d) build/bin (alternative)
   e) build/lib (legacy)

    ↓

4. For each searchable directory:
   - List files
   - Match: libbitnet*, libllama*, libggml*
   - Link found libraries with dylib or static flag
   - Add RPATH for runtime resolution (Linux/macOS)

    ↓

5. Add platform runtime dependencies
   - Linux: stdc++, pthread, dl, m, gomp
   - macOS: c++, Accelerate
   - Windows: msvcrt

    ↓

6. Result
   - If libraries found → Emit link directives
   - If libraries not found → Warn and use stub implementation
   - Build succeeds either way (graceful degradation)
```

---

## 9. Recommended Next Steps for Dual-Backend Support

### Priority 1: Non-Breaking Additions

1. **Add kernel registry** (new file, no existing code changes)
   - Define `KernelBackend` enum with all variants
   - Create capability detection function
   
2. **Enhance xtask** with symbol analysis
   ```bash
   cargo xtask analyze-symbols $LIB_PATH
   ```

3. **Add build-time kernel detection**
   - Check if library was CUDA-compiled (via symbol presence)
   - Emit `cfg(bitnet_backend_cuda)` conditionally

### Priority 2: Breaking Changes (Post-MVP)

1. **Require explicit BITNET_BACKEND env var** in CI
2. **Validate kernel ABI** at library load time
3. **Reject incompatible library combinations**

---

## Appendix: File Locations

### Build Scripts
- Primary discovery: `/home/steven/code/Rust/BitNet-rs/crossval/build.rs`
- FFI setup: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/build.rs`
- GPU detection: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/build.rs`
- Fetch script: `/home/steven/code/Rust/BitNet-rs/ci/fetch_bitnet_cpp.sh`

### C Interface
- Header: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/include/bitnet_c.h`
- Shim (C++): `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/csrc/bitnet_c_shim.cc`
- Wrapper (Rust): `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/src/wrapper.rs`

### xtask Integration
- Setup automation: `/home/steven/code/Rust/BitNet-rs/xtask/src/cpp_setup_auto.rs`
- Main xtask: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs`

