# Build.rs Library Discovery and Linking Analysis

**Report Date**: 2025-10-25
**Repository**: BitNet-rs (Rust/BitNet-rs)
**Scope**: Comprehensive analysis of build.rs patterns for library discovery, linking, and backend selection

## Executive Summary

BitNet-rs employs a **sophisticated but fragmented** library discovery pattern across multiple build.rs scripts. Key findings:

1. **Current Discovery Algorithm**: Multi-tier path scanning with platform-aware search paths
2. **Environment Variables**: BITNET_CPP_DIR, BITNET_CPP_PATH, BITNET_CROSSVAL_LIBDIR, BITNET_GPU_FAKE
3. **Gap Identified**: No explicit backend-aware library discovery (CPU vs GPU kernel selection)
4. **Linking Strategy**: Both dynamic (dylib) and static (a) library support with platform-specific runtime dependencies
5. **Error Handling**: Mix of panic!, bail!, Result types with varying fail-fast strategies

---

## 1. Current Library Discovery Logic

### 1.1 Multi-Tier Search Path Strategy

The codebase uses a **priority-ordered directory scanning** approach:

**Priority Tier 1: Explicit Environment Override**
```rust
// crossval/build.rs (lines 45-46)
if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
    possible_lib_dirs.push(Path::new(&lib_dir).to_path_buf());
}
```

**Priority Tier 2: Standard CMake Output Locations** (llama.cpp integration)
```rust
// crossval/build.rs (lines 50-52)
possible_lib_dirs.push(Path::new(&root).join("build/3rdparty/llama.cpp/src"));
possible_lib_dirs.push(Path::new(&root).join("build/3rdparty/llama.cpp/ggml/src"));
possible_lib_dirs.push(Path::new(&root).join("build/bin"));
```

**Priority Tier 3: Legacy/Fallback Paths**
```rust
// crossval/build.rs (lines 55-57)
possible_lib_dirs.push(Path::new(&root).join("build/lib"));
possible_lib_dirs.push(Path::new(&root).join("build"));
possible_lib_dirs.push(Path::new(&root).join("lib"));
```

**Root Directory Resolution**
```rust
// crossval/build.rs (lines 36-38)
let root = env::var("BITNET_CPP_DIR")
    .or_else(|_| env::var("BITNET_CPP_PATH")) // Legacy support
    .unwrap_or_else(|_| format!("{}/.cache/bitnet_cpp", env::var("HOME").unwrap()));
```

### 1.2 Library Discovery Pattern

**Directional Scan with Fallback Chain**:
```
1. Check BITNET_CROSSVAL_LIBDIR (explicit, highest priority)
   ↓
2. Check BITNET_CPP_DIR/build/3rdparty/llama.cpp/src (CMake primary output)
   ↓
3. Check BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src (GGML path)
   ↓
4. Check BITNET_CPP_DIR/build/bin (alternative CMake output)
   ↓
5. Check BITNET_CPP_DIR/build/lib (legacy path)
   ↓
6. Check BITNET_CPP_DIR/build (fallback)
   ↓
7. Check BITNET_CPP_DIR/lib (last resort)
```

### 1.3 Library Pattern Matching

```rust
// crossval/build.rs (lines 80-82)
if (name.starts_with("libbitnet")
    || name.starts_with("libllama")
    || name.starts_with("libggml"))
    && (path.extension()
        .is_some_and(|ext| ext == "so" || ext == "dylib" || ext == "a"))
{
    let lib_name = name.strip_prefix("lib").unwrap_or(name);
    println!("cargo:rustc-link-lib=dylib={}", lib_name);
}
```

**Library Name Parsing**:
- Strips "lib" prefix (platform convention)
- Supports `.so` (Linux), `.dylib` (macOS), `.a` (static)
- Pattern matching on: `libbitnet*`, `libllama*`, `libggml*`

---

## 2. Search Path Priority (DETAILED BREAKDOWN)

### 2.1 Environment Variable Hierarchy

| Priority | Var | Value | Usage | Scope |
|----------|-----|-------|-------|-------|
| 1 (Override) | `BITNET_CROSSVAL_LIBDIR` | Absolute path to lib dir | Direct library location | crossval/build.rs, crates/bitnet-sys/build.rs |
| 2 | `BITNET_CPP_DIR` | Root C++ source directory | Primary C++ discovery root | Multiple build.rs files |
| 2 (Legacy) | `BITNET_CPP_PATH` | Root C++ source directory | Fallback for BITNET_CPP_DIR | crates/bitnet-sys/build.rs, crates/bitnet-kernels/build.rs |
| 3 | `HOME/.cache/bitnet_cpp` | Default cache location | Fallback if env vars not set | All build.rs with FFI |
| Special | `BITNET_GPU_FAKE` | "cuda", "none", "metal", "rocm" | Test GPU detection override | Device features, NOT library discovery |

### 2.2 crates/bitnet-sys/build.rs Search Paths

**Structured search path array** (lines 78-84):
```rust
let lib_search_paths = [
    build_dir.join("3rdparty/llama.cpp/src"),
    build_dir.join("3rdparty/llama.cpp/ggml/src"),
    build_dir.join("3rdparty/llama.cpp"),
    build_dir.join("lib"),
    build_dir.clone(),
];
```

**RPATH Integration** (lines 94-97):
```rust
#[cfg(any(target_os = "linux", target_os = "macos"))]
{
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
}
```

Eliminates need for LD_LIBRARY_PATH/DYLD_LIBRARY_PATH at runtime.

### 2.3 crates/bitnet-kernels/build.rs Strategy

**Dual-Phase Discovery**:

**Phase 1: CUDA Standard Paths (unconditional when GPU feature enabled)**
```rust
if gpu {
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64/stubs");
    println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
    // ... ARM64 Jetson paths
    // ... x86_64-linux paths
}
```

**Phase 2: Conditional BitNet C++ Linking (only if FFI + libraries found)**
```rust
if ffi_enabled {
    let have_header = inc.join("ggml-bitnet.h").exists();
    let have_static = lib.join("libbitnet.a").exists();
    let have_shared = lib.join("libbitnet.so").exists();
    
    if have_header && (have_static || have_shared) {
        println!("cargo:rustc-cfg=have_cpp");
        // Link libraries...
    } else {
        eprintln!("FFI enabled but C++ library not found");
    }
}
```

---

## 3. Linking Directives (cargo:rustc-link-*)

### 3.1 Link Instruction Types

**Dynamic Library Linking**:
```rust
println!("cargo:rustc-link-lib=dylib=llama");    // Prefer dynamic
println!("cargo:rustc-link-lib=dylib=ggml");     // Prefer dynamic
println!("cargo:rustc-link-lib=dylib=stdc++");   // C++ stdlib
```

**Static Library Linking**:
```rust
println!("cargo:rustc-link-lib=static=bitnet");           // (if .a exists)
println!("cargo:rustc-link-lib=static=bitnet_static");    // (variant)
```

**Framework Linking (macOS)**:
```rust
println!("cargo:rustc-link-lib=framework=Accelerate");
```

### 3.2 Link Search Path Directives

```rust
// Relative to $OUT_DIR
println!("cargo:rustc-link-search=native={}", lib.display());

// RPATH for runtime resolution (Linux/macOS)
println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
```

### 3.3 Platform-Specific Linking

**Linux** (crates/bitnet-sys/build.rs, lines 126-131):
```rust
println!("cargo:rustc-link-lib=dylib=stdc++");
println!("cargo:rustc-link-lib=dylib=pthread");
println!("cargo:rustc-link-lib=dylib=dl");
println!("cargo:rustc-link-lib=dylib=m");
println!("cargo:rustc-link-lib=dylib=gomp");    // OpenMP for parallel code
```

**macOS** (lines 134-137):
```rust
println!("cargo:rustc-link-lib=dylib=c++");
println!("cargo:rustc-link-lib=framework=Accelerate");
```

**Windows** (lines 140-142):
```rust
println!("cargo:rustc-link-lib=dylib=msvcrt");
```

### 3.4 CUDA Linking (GPU Path)

**crates/bitnet-kernels/build.rs** (lines 55-60):
```rust
println!("cargo:rustc-link-lib=cuda");
println!("cargo:rustc-link-lib=nvrtc");
println!("cargo:rustc-link-lib=curand");
println!("cargo:rustc-link-lib=cublas");
println!("cargo:rustc-link-lib=cublasLt");
```

**Search Paths** (lines 41-53):
- `/usr/local/cuda/lib64`
- `/usr/local/cuda/lib64/stubs`
- `/usr/lib/x86_64-linux-gnu`
- `/usr/lib64`
- `/usr/local/cuda/targets/aarch64-linux/lib` (Jetson ARM64)
- `/usr/local/cuda/targets/x86_64-linux/lib`

---

## 4. Backend Selection Analysis

### 4.1 Current State: NO EXPLICIT BACKEND-AWARE DISCOVERY

**Gap Identified**: Library discovery does NOT currently distinguish between:
- CPU-only kernels vs GPU-capable kernels
- Different GPU backends (CUDA vs ROCm vs Metal)
- Fallback libraries for missing backends

### 4.2 Implicit Backend Selection via Feature Gates

**Compile-Time Selection**:
```rust
// crates/bitnet-kernels/build.rs (lines 33-34)
let gpu = env::var_os("CARGO_FEATURE_GPU").is_some() 
    || env::var_os("CARGO_FEATURE_CUDA").is_some();
```

**Feature Predicates**:
- `--features cpu` → CPU-only build
- `--features gpu` or `--features cuda` → GPU-enabled (searches CUDA paths)
- No feature → Default (currently empty, fails)

### 4.3 Runtime Backend Detection (Issue #439)

**Device Features Module** (`crates/bitnet-kernels/src/device_features.rs`):

```rust
/// Compile-time check
pub fn gpu_compiled() -> bool {
    cfg!(any(feature = "gpu", feature = "cuda"))
}

/// Runtime check
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_available_runtime() -> bool {
    // Check BITNET_GPU_FAKE first (test override)
    if let Ok(fake) = env::var("BITNET_GPU_FAKE") {
        return fake.eq_ignore_ascii_case("cuda") || fake.eq_ignore_ascii_case("gpu");
    }
    
    // Real detection via nvidia-smi
    crate::gpu_utils::get_gpu_info().cuda
}
```

### 4.4 The Missing Piece: Backend-Specific Library Discovery

**Current Pattern (static at build time)**:
```
build.rs
  → feature gate check (compile time)
    → CUDA paths added IF feature="gpu" || feature="cuda"
    → No runtime backend selection
```

**What's Missing (no dynamic backend switching)**:
```
build.rs
  → query BITNET_GPU_BACKEND env var (or detect from system)
    → search backend-specific library dirs
      → libcuda*.so, librocm*.so, etc.
    → select appropriate linking directives
      → cargo:rustc-link-lib=cuda vs rocm vs metal
    → generate backend cfg! markers for runtime code
```

---

## 5. Error Reporting Patterns

### 5.1 Panic-Based Fast-Fail (bitnet-sys/build.rs)

**Hard Requirements** (panic on missing):
```rust
if !cpp_dir.exists() {
    panic!(
        "bitnet-sys: BitNet C++ directory not found: {}\n\
         Run: ./ci/fetch_bitnet_cpp.sh",
        cpp_dir.display()
    );
}
```

**Rationale**: FFI feature requires C++ reference; no fallback available.

### 5.2 Error-Based Fallback (crossval/build.rs)

**Soft Warnings** (continue with mock):
```rust
if found_libs {
    println!(
        "cargo:warning=bitnet-crossval: C++ libraries found in {} dir(s): {}",
        searched_dirs.len(),
        searched_dirs.join(", ")
    );
} else {
    println!(
        "cargo:warning=bitnet-crossval: Using mock C wrapper (searched {} dirs but found no recognized libraries)",
        searched_dirs.len()
    );
}
```

**Rationale**: Crossval can work with mock wrapper; graceful degradation.

### 5.3 Conditional Linking (bitnet-kernels/build.rs)

**Silent Fallback** (lines 108-119):
```rust
if have_header && (have_static || have_shared || have_components) {
    // Link the C++ library
} else if !searched_dirs.is_empty() {
    eprintln!(
        "bitnet-kernels: FFI enabled but C++ library not found at {}; using stub implementation",
        root
    );
}
```

### 5.4 Result-Based Handling (bitnet-sys/link_cpp_implementation)

```rust
fn link_cpp_implementation(cpp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // ... discovery logic ...
    if !found_any {
        return Err("No library directories found. Is BitNet C++ built?".into());
    }
    Ok(())
}
```

---

## 6. Preflight Check Patterns

### 6.1 xtask Preflight Command

**Device Features Summary** (`crates/bitnet-kernels/src/device_features.rs`, lines 133-164):
```rust
pub fn device_capability_summary() -> String {
    // Compile-time: GPU compiled? CPU always available
    // Runtime: CUDA available? CPU always available
    // Returns formatted string with ✓/✗ markers
}
```

### 6.2 C++ Reference Bootstrap

**xtask setup-cpp-auto** (`xtask/src/cpp_setup_auto.rs`):

**Library Discovery** (lines 59-95):
```rust
fn find_lib_dir(build: &Path) -> Result<PathBuf> {
    // 1. Check common fast-path candidates:
    //    - build/
    //    - build/bin/
    //    - build/Release/
    //    - build/Debug/
    //    - build/lib/
    
    // 2. If not found: recursive search max_depth=3
    //    - Look for libllama.{so,dylib,dll}
}
```

**Shell Export Generation** (lines 1-40):
```rust
pub enum Emit {
    Sh,     // POSIX sh/bash/zsh
    Fish,   // fish shell
    Pwsh,   // PowerShell
    Cmd,    // Windows cmd
}
```

Outputs environment variable exports for:
- `BITNET_CROSSVAL_LIBDIR` (direct library location)
- `BITNET_CPP_DIR` (root directory)
- `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` (dynamic loader path)

---

## 7. The Gap: No Backend-Specific Discovery

### 7.1 What Exists

- **CUDA Detection**: Hardcoded to `/usr/local/cuda/lib64` when GPU feature enabled
- **FFI Detection**: Search for libbitnet/libllama/libggml by name
- **Runtime GPU Detection**: `BITNET_GPU_FAKE` override + `nvidia-smi` check

### 7.2 What's Missing

| Feature | Current | Needed |
|---------|---------|--------|
| ROCm Support | Hardcoded CUDA only | Query `BITNET_GPU_BACKEND` or auto-detect | 
| Metal Support (Apple) | Framework links only | Query Apple GPU backend |
| Multi-GPU Scenarios | No consideration | Prioritize multiple backends per platform |
| Backend-Specific Libs | All backends link same libs | Backend-specific library names/paths |
| Error Context | Generic "libraries not found" | Backend-specific: "CUDA libs not found, ROCm available?" |

### 7.3 Template for Backend-Aware Discovery

**Pseudocode Structure**:
```rust
fn discover_gpu_backend() -> Result<GpuBackend> {
    // 1. Check explicit override
    if let Ok(backend_name) = env::var("BITNET_GPU_BACKEND") {
        return GpuBackend::from_str(&backend_name);
    }
    
    // 2. Auto-detect from system
    if Command::new("nvidia-smi").output().is_ok() {
        return Ok(GpuBackend::Cuda);
    }
    if Command::new("rocm-smi").output().is_ok() {
        return Ok(GpuBackend::Rocm);
    }
    if cfg!(target_os = "macos") && check_metal() {
        return Ok(GpuBackend::Metal);
    }
    
    // 3. Fallback
    Err("No GPU backend detected")
}

fn link_gpu_libraries(backend: GpuBackend) {
    match backend {
        GpuBackend::Cuda => {
            println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-lib=cuda");
            // ... CUDA-specific libs
        }
        GpuBackend::Rocm => {
            println!("cargo:rustc-link-search=/opt/rocm/lib");
            println!("cargo:rustc-link-lib=amdhip64");
            // ... ROCm-specific libs
        }
        GpuBackend::Metal => {
            println!("cargo:rustc-link-lib=framework=Metal");
            // ... Metal-specific frameworks
        }
    }
}
```

---

## 8. Environment Variable Usage Summary

| Variable | Where Set | Read In | Purpose | Example |
|----------|-----------|---------|---------|---------|
| `BITNET_CPP_DIR` | User/CI | All FFI build.rs | C++ root directory | `/path/to/bitnet.cpp` |
| `BITNET_CPP_PATH` | User/CI | Legacy fallback | Alias for CPP_DIR | (deprecated) |
| `BITNET_CROSSVAL_LIBDIR` | setup-cpp-auto | crossval/build.rs | Direct lib directory | `/path/to/.../build/lib` |
| `BITNET_GPU_FAKE` | Test runner | Runtime device detection | Override GPU detection | `none`, `cuda`, `rocm` |
| `BITNET_STRICT_MODE` | Test runner | Runtime validation | Force strict GPU detection | `1` |
| `CARGO_FEATURE_GPU` | Cargo | build.rs | GPU feature enabled | (automatic) |
| `CARGO_FEATURE_CUDA` | Cargo | build.rs | Legacy GPU feature | (automatic) |
| `CARGO_FEATURE_FFI` | Cargo | build.rs | FFI feature enabled | (automatic) |
| `HOME` | System | build.rs | Default cache root | `/home/user` |

---

## 9. Document the Current Implementation

### 9.1 crossval/build.rs Algorithm (Canonical Reference)

**Location**: `/home/steven/code/Rust/BitNet-rs/crossval/build.rs` (lines 27-117)

**Algorithm Summary**:
1. Emit metadata (rustc version, target triple)
2. If `feature="ffi"`:
   a. Compile C wrapper (bitnet_cpp_wrapper.c)
   b. Search library directories (priority-ordered)
   c. Scan for libbitnet/libllama/libggml
   d. Link found libraries dynamically
   e. Link platform C++ standard libraries
   f. Emit cfg!(have_cpp) if libraries found

**Key Code Section** (lines 40-94):
```rust
// Try multiple possible library locations
let mut possible_lib_dirs = Vec::new();

// Highest priority: explicit lib dir from setup-cpp-auto
if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
    possible_lib_dirs.push(Path::new(&lib_dir).to_path_buf());
}

// Standard llama.cpp build structure
possible_lib_dirs.push(Path::new(&root).join("build/3rdparty/llama.cpp/src"));
possible_lib_dirs.push(Path::new(&root).join("build/3rdparty/llama.cpp/ggml/src"));
possible_lib_dirs.push(Path::new(&root).join("build/bin"));

// Legacy/fallback paths
possible_lib_dirs.push(Path::new(&root).join("build/lib"));
possible_lib_dirs.push(Path::new(&root).join("build"));
possible_lib_dirs.push(Path::new(&root).join("lib"));

// Search ALL directories and collect all libraries
let mut found_libs = false;
for lib_dir in &possible_lib_dirs {
    if !lib_dir.exists() {
        continue;
    }
    
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    
    if let Ok(lib_files) = std::fs::read_dir(lib_dir) {
        for entry in lib_files.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                if (name.starts_with("libbitnet")
                    || name.starts_with("libllama")
                    || name.starts_with("libggml"))
                    && (path.extension()
                        .is_some_and(|ext| ext == "so" || ext == "dylib" || ext == "a"))
                {
                    let lib_name = name.strip_prefix("lib").unwrap_or(name);
                    println!("cargo:rustc-link-lib=dylib={}", lib_name);
                    found_libs = true;
                }
            }
        }
    }
}
```

### 9.2 crates/bitnet-sys/build.rs Algorithm

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/build.rs` (lines 73-145)

**Key Functions**:
1. `link_cpp_implementation()` - Searches paths, links libraries, adds RPATH
2. `compile_cpp_shim()` - Compiles bitnet_c_shim.cc with correct include paths
3. `generate_bindings()` - Generates Rust FFI bindings from C headers

**Search Path Array** (lines 78-84):
```rust
let lib_search_paths = [
    build_dir.join("3rdparty/llama.cpp/src"),
    build_dir.join("3rdparty/llama.cpp/ggml/src"),
    build_dir.join("3rdparty/llama.cpp"),
    build_dir.join("lib"),
    build_dir.clone(),
];
```

**Link Strategy** (lines 116-122):
```rust
// Link the main llama library (required)
println!("cargo:rustc-link-lib=dylib=llama");

// Only link ggml if it exists as a separate library
if lib_search_paths.iter().any(|dir| lib_present(dir, "ggml")) {
    println!("cargo:rustc-link-lib=dylib=ggml");
}
```

### 9.3 crates/bitnet-kernels/build.rs Algorithm

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/build.rs` (lines 23-120)

**Two-Phase Approach**:
1. **GPU Detection Phase** (lines 33-61): Add CUDA paths if GPU feature enabled
2. **FFI Optional Phase** (lines 64-119): Link BitNet C++ IF available

**Key Decision Points**:
```rust
let gpu = env::var_os("CARGO_FEATURE_GPU").is_some() 
    || env::var_os("CARGO_FEATURE_CUDA").is_some();

let ffi_enabled = env::var_os("CARGO_FEATURE_FFI").is_some();
```

---

## 10. Complete Integration Blueprint

### 10.1 File-by-File Pattern

| File | Responsibility | Discovery Scope | Linking Scope | Error Strategy |
|------|-----------------|-----------------|----------------|-----------------|
| crossval/build.rs | C++ reference linking | Multi-path scan | Dynamic + platform libs | Soft warning |
| bitnet-sys/build.rs | FFI bindings + shim | Indexed paths + RPATH | Dynamic + static | Hard panic |
| bitnet-kernels/build.rs | GPU + optional FFI | CUDA hardcoded + optional | Platform-specific | Conditional fallback |
| bitnet-ffi/build.rs | FFI header generation | N/A | Package config only | Warning if cbindgen unavailable |
| bitnet-ggml-ffi/build.rs | GGML vendored shim | Vendored only | Local compile only | Panic in CI if marker missing |

### 10.2 Library Search Path Consolidation

**Canonical Search Order** (across all build.rs):
```
1. Explicit environment (BITNET_CROSSVAL_LIBDIR, BITNET_CPP_DIR)
2. CMake standard: build/3rdparty/llama.cpp/{src,ggml/src}
3. CMake alternative: build/bin/, build/lib/
4. Fallback: build/, lib/
5. System defaults: /usr/local/cuda/lib64, /usr/lib/x86_64-linux-gnu
6. Jetson ARM: /usr/local/cuda/targets/aarch64-linux/lib
```

### 10.3 Linking Directives Pattern

**All build.rs follow this pattern**:
```rust
// 1. Add search paths
println!("cargo:rustc-link-search=native={}", path);

// 2. Link libraries
println!("cargo:rustc-link-lib=[static|dylib]=libname");

// 3. Add RPATH (Linux/macOS only)
#[cfg(any(target_os = "linux", target_os = "macos"))]
println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path);

// 4. Link platform dependencies
#[cfg(target_os = "linux")]
println!("cargo:rustc-link-lib=dylib=stdc++");
// ... (platform-specific)

// 5. Emit cfg markers
println!("cargo:rustc-cfg=have_cpp");
```

---

## 11. Root Cause: Missing Backend Selection

### 11.1 Why This Matters

**Current Behavior**:
- Build succeeds with GPU feature → CUDA paths added
- But no check if CUDA is actually available
- Runtime fallback exists (device_features) but build.rs doesn't know

**Example Problem**:
```bash
cargo build --features gpu  # On a ROCm-only system
# ✗ Linker error: undefined reference to `cudaMalloc`
# (Should have searched ROCm paths instead)
```

### 11.2 Solution Architecture

**Proposed** `bitnet-backend-discover` crate or module:

```rust
pub enum GpuBackend {
    Cuda,
    Rocm,
    Metal,
    Oneapi,
    None,
}

impl GpuBackend {
    /// Auto-detect from system
    pub fn detect() -> Result<Self> {
        // Check nvidia-smi, rocm-smi, metal capabilities, etc.
    }
    
    /// From explicit override
    pub fn from_env() -> Option<Self> {
        // Read BITNET_GPU_BACKEND
    }
    
    /// Get search paths for this backend
    pub fn lib_search_paths(&self) -> Vec<PathBuf> {
        match self {
            Self::Cuda => vec!["/usr/local/cuda/lib64", ...],
            Self::Rocm => vec!["/opt/rocm/lib", ...],
            Self::Metal => vec![], // Framework-based
            // ...
        }
    }
    
    /// Get library names for this backend
    pub fn library_names(&self) -> Vec<&'static str> {
        match self {
            Self::Cuda => vec!["cuda", "nvrtc", "cublas", "cublasLt"],
            Self::Rocm => vec!["amdhip64", "rocblas", "rocsolver"],
            // ...
        }
    }
}
```

Then use in build.rs:
```rust
if gpu {
    let backend = GpuBackend::from_env()
        .or_else(|_| GpuBackend::detect())
        .unwrap_or(GpuBackend::Cuda);
    
    for path in backend.lib_search_paths() {
        println!("cargo:rustc-link-search={}", path);
    }
    
    for lib in backend.library_names() {
        println!("cargo:rustc-link-lib={}", lib);
    }
}
```

---

## 12. Key Observations and Recommendations

### 12.1 Current Strengths

1. **Robust multi-tier fallback**: Environment variables → CMake paths → legacy paths
2. **Platform awareness**: Different linking for Linux/macOS/Windows
3. **RPATH integration**: Eliminates LD_LIBRARY_PATH for deploy
4. **Feature gating**: Clean separation of CPU/GPU builds
5. **Error messages**: Helpful guidance for missing setup

### 12.2 Current Weaknesses

1. **No backend selection**: Only CUDA, no ROCm/Metal auto-detection
2. **Fragmented patterns**: Each build.rs reinvents discovery
3. **Silent fallbacks**: May hide configuration errors
4. **Limited diagnostics**: "Libraries not found" doesn't say which ones
5. **Hard-coded paths**: `/usr/local/cuda/lib64` assumes Linux/x86_64

### 12.3 Recommended Improvements

1. **Extract library discovery to utility crate**: DRY principle
2. **Add backend detection**: Auto-select GPU backend
3. **Emit better diagnostics**: List searched directories on failure
4. **Standardize error strategy**: Define panic vs warning rules
5. **Document search algorithm**: This report is a start; add to CLAUDE.md
6. **Add CI test for discovery**: Verify search paths work on multiple platforms

---

## 13. Complete File Listing

### All build.rs Files Analyzed

| Path | Purpose | Size | Key Functions |
|------|---------|------|----------------|
| `/home/steven/code/Rust/BitNet-rs/build.rs` | Workspace metadata | 7 lines | Build timestamp only |
| `/home/steven/code/Rust/BitNet-rs/crates/bitnet-py/build.rs` | Python FFI linking | 21 lines | PyO3 auto-config |
| `/home/steven/code/Rust/BitNet-rs/crates/bitnet-ffi/build.rs` | FFI header generation | 75 lines | cbindgen integration |
| `/home/steven/code/Rust/BitNet-rs/crates/bitnet-server/build.rs` | Server metadata | 30 lines | vergen integration |
| `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/build.rs` | GPU + optional FFI | 120 lines | GPU detection + conditional linking |
| `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/build.rs` | FFI binding generation | 302 lines | Complete C++ integration |
| `/home/steven/code/Rust/BitNet-rs/crates/bitnet-ggml-ffi/build.rs` | GGML shim compilation | 95 lines | Vendored GGML + platform CC flags |
| `/home/steven/code/Rust/BitNet-rs/crossval/build.rs` | Cross-validation linking | 119 lines | Multi-path library discovery |
| `/home/steven/code/Rust/BitNet-rs/xtask-build-helper/src/lib.rs` | Build utility crate | 225+ lines | FFI build hygiene patterns |

### Related Discovery Code

| Path | Purpose |
|------|---------|
| `crates/bitnet-kernels/src/device_features.rs` | Runtime GPU detection |
| `xtask/src/cpp_setup_auto.rs` | C++ bootstrap + lib discovery |
| `docs/environment-variables.md` | Environment variable reference |

---

## 14. Appendix: Code Snippets Reference

### A.1 Complete crossval/build.rs Compile Phase

```rust
#[cfg(feature = "ffi")]
fn compile_ffi() {
    use std::{env, path::Path};

    // Compile our C wrapper
    cc::Build::new().file("src/bitnet_cpp_wrapper.c").compile("bitnet_cpp_wrapper");

    let root = env::var("BITNET_CPP_DIR")
        .or_else(|_| env::var("BITNET_CPP_PATH"))
        .unwrap_or_else(|_| format!("{}/.cache/bitnet_cpp", env::var("HOME").unwrap()));

    let mut possible_lib_dirs = Vec::new();
    
    if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
        possible_lib_dirs.push(Path::new(&lib_dir).to_path_buf());
    }

    possible_lib_dirs.push(Path::new(&root).join("build/3rdparty/llama.cpp/src"));
    possible_lib_dirs.push(Path::new(&root).join("build/3rdparty/llama.cpp/ggml/src"));
    possible_lib_dirs.push(Path::new(&root).join("build/bin"));
    possible_lib_dirs.push(Path::new(&root).join("build/lib"));
    possible_lib_dirs.push(Path::new(&root).join("build"));
    possible_lib_dirs.push(Path::new(&root).join("lib"));

    println!("cargo:rustc-cfg=have_cpp");

    let mut found_libs = false;
    let mut searched_dirs = Vec::new();

    for lib_dir in &possible_lib_dirs {
        if !lib_dir.exists() {
            continue;
        }

        searched_dirs.push(lib_dir.display().to_string());
        println!("cargo:rustc-link-search=native={}", lib_dir.display());

        if let Ok(lib_files) = std::fs::read_dir(lib_dir) {
            for entry in lib_files.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    if (name.starts_with("libbitnet")
                        || name.starts_with("libllama")
                        || name.starts_with("libggml"))
                        && (path
                            .extension()
                            .is_some_and(|ext| ext == "so" || ext == "dylib" || ext == "a"))
                    {
                        let lib_name = name.strip_prefix("lib").unwrap_or(name);
                        println!("cargo:rustc-link-lib=dylib={}", lib_name);
                        found_libs = true;
                    }
                }
            }
        }
    }

    if found_libs {
        println!(
            "cargo:warning=bitnet-crossval: C++ libraries found in {} dir(s): {}",
            searched_dirs.len(),
            searched_dirs.join(", ")
        );
        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-lib=dylib=stdc++");
        #[cfg(target_os = "macos")]
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if !searched_dirs.is_empty() {
        println!(
            "cargo:warning=bitnet-crossval: Using mock C wrapper (searched {} dirs but found no recognized libraries)",
            searched_dirs.len()
        );
    } else {
        println!(
            "cargo:warning=bitnet-crossval: Using mock C wrapper (no library directories exist)"
        );
    }
}
```

---

**Report End**
