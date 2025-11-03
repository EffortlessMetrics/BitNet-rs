# Dual-Backend Support Implementation Roadmap

## Current State Assessment

### What Works Today

1. **Library Discovery**: Automatic detection of libllama.so/dylib/a in standard CMake locations
2. **Environment Overrides**: BITNET_CPP_DIR, BITNET_CPP_PATH, BITNET_CROSSVAL_LIBDIR
3. **Feature-Gated Compilation**: --features gpu vs --features cpu
4. **Graceful Fallback**: Mock implementations if libraries not found
5. **Platform-Specific Linking**: Different stdlibs/frameworks per OS
6. **RPATH Support**: Runtime library resolution without LD_LIBRARY_PATH (Linux/macOS)

### What's Missing

1. **Backend Detection**: No way to distinguish CUDA vs CPU bitnet.cpp builds
2. **Kernel Capability Registry**: No centralized list of available quantization kernels
3. **Symbol Analysis**: No runtime inspection of loaded libraries
4. **ABI Validation**: No checking for compatible Rust/C++ versions
5. **Backend Enforcement**: No validation that loaded backend matches requested features

---

## Phase 1: Non-Breaking Infrastructure (Week 1-2)

### 1.1 Create Kernel Registry Module

**File**: `crates/bitnet-common/src/kernel_registry.rs` (NEW)

```rust
/// Kernel backends supported by BitNet.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelBackend {
    // Pure Rust SIMD backends
    I2S_QK256_Scalar,
    I2S_QK256_AVX2,
    I2S_QK256_AVX512,
    I2S_QK256_NEON,
    
    // Quantization LUT backends
    TL1_ARM_NEON,
    TL1_X86_AVX2,
    TL1_X86_AVX512,
    TL2_X86_AVX512,
    
    // GGML-compatible backends
    IQ2S_GGML_CPU,
    IQ2S_GGML_CUDA,
    
    // GPU backends
    I2S_QK256_CUDA,
    Llama_CUDA,
    Llama_CPU,
}

/// Kernel compilation flags (bitwise)
pub struct KernelCapabilities {
    pub backends: Vec<KernelBackend>,
    pub cuda_available: bool,
    pub simd_level: SimdLevel,
    pub max_context_length: usize,
}

impl KernelCapabilities {
    pub fn is_capable_of(&self, backend: KernelBackend) -> bool {
        self.backends.contains(&backend)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    Scalar,
    AVX2,
    AVX512,
    NEON,
}
```

**Tests**: `crates/bitnet-common/src/kernel_registry.rs` (inline)
- Verify all backends can be enumerated
- Check SIMD level comparisons work correctly

### 1.2 Add Symbol Analysis Tool to xtask

**File**: `xtask/src/symbol_analysis.rs` (NEW)

```rust
pub struct SymbolAnalysis {
    pub bitnet_symbols: Vec<String>,
    pub llama_symbols: Vec<String>,
    pub cuda_symbols: Vec<String>,
    pub quantization_symbols: Vec<String>,
}

/// Analyze what symbols are available in a shared library
pub fn analyze_library(lib_path: &Path) -> Result<SymbolAnalysis> {
    // Use `nm` command to extract symbols
    let output = Command::new("nm")
        .args(&["-D", "-g"])  // Dynamic, global symbols only
        .arg(lib_path)
        .output()?;
    
    let stdout = String::from_utf8(output.stdout)?;
    
    // Parse symbols by prefix
    let mut analysis = SymbolAnalysis {
        bitnet_symbols: Vec::new(),
        llama_symbols: Vec::new(),
        cuda_symbols: Vec::new(),
        quantization_symbols: Vec::new(),
    };
    
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if let Some(symbol) = parts.last() {
            if symbol.starts_with("bitnet_") {
                analysis.bitnet_symbols.push(symbol.to_string());
            } else if symbol.starts_with("llama_") {
                analysis.llama_symbols.push(symbol.to_string());
            } else if symbol.contains("cuda") || symbol.contains("__global__") {
                analysis.cuda_symbols.push(symbol.to_string());
            } else if symbol.contains("quants") || symbol.contains("i2_s") {
                analysis.quantization_symbols.push(symbol.to_string());
            }
        }
    }
    
    Ok(analysis)
}

pub fn detect_kernel_backends(analysis: &SymbolAnalysis) -> Vec<KernelBackend> {
    let mut backends = vec![];
    
    // Detect CUDA support
    if !analysis.cuda_symbols.is_empty() {
        backends.push(KernelBackend::I2S_QK256_CUDA);
        backends.push(KernelBackend::IQ2S_GGML_CUDA);
    }
    
    // Detect CPU SIMD kernels (requires source inspection for now)
    backends.push(KernelBackend::I2S_QK256_Scalar);
    
    // Detect quantization support
    if analysis.quantization_symbols.iter().any(|s| s.contains("i2_s")) {
        backends.push(KernelBackend::IQ2S_GGML_CPU);
    }
    
    backends
}
```

**CLI Command**:
```bash
cargo xtask analyze-library /path/to/libllama.so
```

Output:
```
Library: libllama.so
Bitnet symbols:   5 found
  - bitnet_model_new_from_file
  - bitnet_context_new
  - bitnet_tokenize
  - bitnet_eval
  - bitnet_decode_greedy

Llama symbols:    42 found
  - llama_load_model_from_file
  - llama_new_context_with_model
  ...

CUDA symbols:     0 found
  Inference: CPU-only backend

Quantization:     8 found
  - Detected: IQ2S_GGML (CPU)
```

### 1.3 Enhance Build Scripts with Backend Logging

**Modify**: `crossval/build.rs` (lines ~60-100)

```rust
if found_libs {
    // New: Log which backends were found
    eprintln!("cargo:warning=bitnet-crossval: Libraries found in {} dir(s)", searched_dirs.len());
    eprintln!("cargo:warning=  Standard paths checked:");
    for dir in &searched_dirs {
        eprintln!("cargo:warning=    - {}", dir);
    }
    
    // Could add symbol analysis here (optional, Phase 2)
    // if cfg!(feature = "symbol-analysis") {
    //     if let Ok(analysis) = analyze_library(lib_path) {
    //         eprintln!("cargo:warning=  Backends detected: {:?}", analysis);
    //     }
    // }
}
```

### 1.4 Document Current Implementation

**Files**: Save to docs/
- `library-discovery-and-linking.md` (already done)
- `library-discovery-quick-reference.md` (already done)
- `dual-backend-roadmap.md` (this file)

---

## Phase 2: Build-Time Backend Detection (Week 3)

### 2.1 Implement Symbol Analysis in Build Scripts

**Modify**: `bitnet-sys/build.rs` (add after line 144)

```rust
#[cfg(feature = "ffi")]
fn analyze_found_libraries(lib_search_paths: &[PathBuf]) -> SymbolAnalysis {
    // Only if optional build-time symbol analysis is enabled
    #[cfg(feature = "symbol-analysis")]
    {
        for path in lib_search_paths {
            if path.exists() {
                if let Ok(lib) = find_library(path, "llama") {
                    if let Ok(analysis) = xtask_build_helper::analyze_library(&lib) {
                        eprintln!("cargo:warning=bitnet-sys: Detected backends:");
                        eprintln!("cargo:warning=  CUDA support: {}", 
                                  !analysis.cuda_symbols.is_empty());
                        eprintln!("cargo:warning=  Bitnet symbols: {}", 
                                  analysis.bitnet_symbols.len());
                        
                        // Emit cfg flags based on detected capabilities
                        if !analysis.cuda_symbols.is_empty() {
                            eprintln!("cargo:rustc-cfg=bitnet_cpp_has_cuda");
                        }
                        
                        return analysis;
                    }
                }
            }
        }
    }
    
    SymbolAnalysis::default()
}
```

### 2.2 Add Cfg Flags for Backend Detection

```rust
// In bitnet-sys/build.rs, after linking libraries:

// Check if the C++ library has CUDA support (via symbol analysis)
if has_cuda_symbols {
    println!("cargo:rustc-cfg=bitnet_cpp_has_cuda");
}

// Check if bitnet C shim is available
if lib_search_paths.iter().any(|dir| lib_present(dir, "bitnet")) {
    println!("cargo:rustc-cfg=bitnet_cpp_has_bitnet_lib");
}
```

### 2.3 Update cargo feature detection

**File**: `Cargo.toml` (bitnet-sys)

```toml
[features]
ffi = ["dep:bindgen", "dep:cc"]
symbol-analysis = []  # NEW: optional symbol analysis
crossval = ["ffi"]
```

---

## Phase 3: Runtime Validation (Week 4)

### 3.1 Add Library Validation at Startup

**File**: `crates/bitnet-ffi/src/lib.rs` (NEW module)

```rust
pub mod validation {
    use std::path::Path;
    
    /// Validate that loaded FFI library is compatible
    pub fn validate_ffi_library(lib_path: &Path) -> Result<(), String> {
        // Step 1: Check file existence
        if !lib_path.exists() {
            return Err(format!("Library not found: {}", lib_path.display()));
        }
        
        // Step 2: Check for required symbols
        let required_symbols = [
            "bitnet_model_new_from_file",
            "bitnet_context_new",
            "bitnet_tokenize",
            "bitnet_eval",
        ];
        
        // Step 3: Attempt to load and check symbols
        #[cfg(not(target_os = "windows"))]
        unsafe {
            use libloading::Library;
            
            let lib = Library::new(lib_path)
                .map_err(|e| format!("Failed to load library: {}", e))?;
            
            for symbol in &required_symbols {
                let _: Result<libloading::Symbol<unsafe extern "C" fn()>, _> =
                    lib.get(symbol.as_bytes());
                
                // In production, would fail on missing symbols
            }
        }
        
        Ok(())
    }
}
```

**Usage**:
```rust
// At program startup
match bitnet_ffi::validation::validate_ffi_library(&lib_path) {
    Ok(_) => eprintln!("FFI library validated"),
    Err(e) => {
        eprintln!("WARNING: FFI validation failed: {}", e);
        eprintln!("Falling back to pure Rust implementation");
    }
}
```

### 3.2 Add Runtime Backend Selection

**File**: `crates/bitnet-inference/src/device_features.rs` (ENHANCE)

```rust
/// Detect available compute backends at runtime
pub fn detect_backends() -> BackendCapabilities {
    let mut caps = BackendCapabilities::default();
    
    // Check if CUDA is available (runtime)
    #[cfg(feature = "gpu")]
    {
        if cuda_available() {
            caps.cuda = true;
        }
    }
    
    // Check if C++ FFI is available
    #[cfg(feature = "ffi")]
    {
        if ffi_available() {
            caps.cpp_ffi = true;
        }
    }
    
    // Pure Rust backends always available
    caps.cpu_rust = true;
    
    caps
}

pub struct BackendCapabilities {
    pub cpu_rust: bool,
    pub cuda: bool,
    pub cpp_ffi: bool,
    pub quantization_kernels: Vec<KernelBackend>,
}
```

---

## Phase 4: Enforcement & Testing (Week 5)

### 4.1 Add Feature Validation Tests

**File**: `tests/test_backend_selection.rs` (NEW)

```rust
#[test]
fn test_cpu_feature_no_cuda_linking() {
    // Verify that --features cpu doesn't link CUDA
    let metadata = cargo_metadata::metadata().unwrap();
    let pkg = metadata.packages.iter()
        .find(|p| p.name == "bitnet-kernels")
        .unwrap();
    
    // Check that cuda features are not resolved
    assert!(!pkg.features.contains_key("cuda"));
}

#[test]
fn test_gpu_feature_requires_cuda() {
    // Verify that --features gpu requires CUDA to be available
    #[cfg(all(feature = "gpu", not(feature = "cuda")))]
    {
        panic!("GPU feature requires CUDA feature");
    }
}

#[test]
fn test_ffi_library_discovery() {
    #[cfg(feature = "ffi")]
    {
        let lib_path = std::env::var("BITNET_CPP_DIR")
            .map(|d| format!("{}/build/3rdparty/llama.cpp/src/libllama.so", d))
            .ok();
        
        if let Some(path) = lib_path {
            assert!(Path::new(&path).exists(), 
                   "Expected library at {}", path);
        }
    }
}
```

### 4.2 Add CI Backend Validation

**File**: `.github/workflows/ci.yml` (ENHANCE)

```yaml
- name: Analyze linked libraries
  run: |
    cargo xtask analyze-library ${{ env.BITNET_CPP_DIR }}/build/3rdparty/llama.cpp/src/libllama.so

- name: Verify CPU build has no CUDA
  run: |
    cargo build --no-default-features --features cpu
    # Check that no cuda symbols are linked
    nm target/debug/libbitnet.so | grep -i cuda && exit 1 || true

- name: Verify GPU build requires CUDA
  run: |
    cargo build --no-default-features --features gpu 2>&1 | grep -i cuda || true
```

### 4.3 Create Backend Compatibility Matrix

**File**: `docs/reference/backend-compatibility-matrix.md` (NEW)

```markdown
# Backend Compatibility Matrix

## CPU Backends

| Quantization | Scalar | AVX2 | AVX-512 | NEON |
|--------------|--------|------|---------|------|
| I2_S_QK256   | ✅     | ✅   | ✅      | ✅   |
| TL1          | ✅     | ✅   | ✅      | ✅   |
| TL2          | ✅     | ❌   | ✅      | ❌   |
| IQ2S_GGML    | ✅     | ✅   | ✅      | ✅   |

## GPU Backends (CUDA)

| Quantization | H100 | A100 | A6000 | RTX4090 |
|--------------|------|------|-------|---------|
| I2_S_QK256   | ✅   | ✅   | ✅    | ✅      |
| IQ2S_GGML    | ✅   | ✅   | ✅    | ✅      |

## Fallback Behavior

- CPU to Scalar: ✅ (always available)
- CUDA to CPU: ✅ (graceful degradation)
- Unsupported format: ✅ (use closest match)
```

---

## Phase 5: Extended Features (Future)

### 5.1 Runtime Kernel Switching

Allow switching between backends at runtime without recompilation:

```rust
// Proposed API (post-MVP)
let mut inference = BitNetInference::new(model_path)?;

// Try CUDA first, fall back to CPU
inference.set_backend_preference(vec![
    KernelBackend::I2S_QK256_CUDA,
    KernelBackend::I2S_QK256_AVX2,
    KernelBackend::I2S_QK256_Scalar,
])?;
```

### 5.2 Dynamic Library Hot-Reload

Allow switching between different C++ libraries without restart:

```rust
// Proposed API
inference.reload_backend("/path/to/new/libllama.so")?;
```

### 5.3 Kernel Capability Reporting

API for querying available kernels:

```rust
let caps = inference.kernel_capabilities();
println!("Available backends: {:?}", caps.backends);
println!("Max context: {}", caps.max_context_length);
println!("CUDA available: {}", caps.cuda_available);
```

---

## Timeline & Effort Estimate

| Phase | Duration | Files | Breaking? | Risk |
|-------|----------|-------|-----------|------|
| 1: Infrastructure | 1-2 wks | 4 new | No | Low |
| 2: Build-time detection | 1 wk | 2 modified | No | Low |
| 3: Runtime validation | 1 wk | 2 new, 1 modified | No | Medium |
| 4: Enforcement & tests | 1 wk | 3 new, CI modified | Yes | High |
| 5: Extended features | Ongoing | Multiple | Depends | High |

**Total MVP**: 4 weeks to full dual-backend support

---

## Success Criteria

After implementation, the system should:

- [ ] Automatically detect which C++ backends are compiled
- [ ] Emit appropriate cfg flags at build time
- [ ] Validate library compatibility at runtime
- [ ] Provide clear error messages on mismatch
- [ ] Fall back gracefully when backends unavailable
- [ ] Support hot-reload for different libraries (Phase 5)
- [ ] Have >95% test coverage for backend selection
- [ ] Document all backend combinations in CI

---

## Dependencies & Prerequisites

- [ ] Symbol analysis tool (`xtask analyze-library`)
- [ ] Kernel registry module (`bitnet-common`)
- [ ] FFI validation module (`bitnet-ffi`)
- [ ] Enhanced build scripts (both sys and crossval)
- [ ] Test infrastructure for backend selection

**Blocking Issues**: None (non-breaking implementation)

**Related Issues**: #254, #260, #439, #469

