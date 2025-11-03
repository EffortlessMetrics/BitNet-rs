# Backend Detection and Device Selection Patterns - BitNet.rs Analysis Report

## Executive Summary

BitNet.rs implements a comprehensive backend detection system with unified patterns for GPU/CPU selection, runtime detection, and environment overrides. This report documents the existing patterns and provides reusable templates for adding C++ backend detection similar to the GPU/CPU architecture.

## Key Findings

### 1. Device Feature Detection Architecture (Issue #439)

**Location**: `crates/bitnet-kernels/src/device_features.rs`

The cornerstone of backend detection is a two-level approach:

#### Compile-Time Detection (`gpu_compiled()`)
```rust
#[inline]
pub fn gpu_compiled() -> bool {
    cfg!(any(feature = "gpu", feature = "cuda"))
}
```
- Determines if GPU support was built into the binary
- Uses unified predicate: `any(feature = "gpu", feature = "cuda")`
- **Reusable pattern**: Can be extended for C++ backend with `cfg!(feature = "cpp")`

#### Runtime Detection (`gpu_available_runtime()`)
```rust
pub fn gpu_available_runtime() -> bool {
    use std::env;
    
    // Strict mode: refuse BITNET_GPU_FAKE
    let strict_mode = env::var("BITNET_STRICT_MODE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    
    if strict_mode {
        return crate::gpu_utils::get_gpu_info().cuda;
    }
    
    // Check BITNET_GPU_FAKE override first
    if let Ok(fake) = env::var("BITNET_GPU_FAKE") {
        return fake.eq_ignore_ascii_case("cuda") || fake.eq_ignore_ascii_case("gpu");
    }
    
    // Real detection
    crate::gpu_utils::get_gpu_info().cuda
}
```

**Key Features**:
- **Deterministic Testing**: `BITNET_GPU_FAKE=cuda|none` for testing GPU paths
- **Strict Mode Override**: `BITNET_STRICT_MODE=1` forces real detection
- **Graceful Fallback**: Stubs when GPU not compiled
- **Stub Implementation**: When `not(any(feature = "gpu", feature = "cuda"))`

---

### 2. Runtime GPU Detection (`gpu_utils.rs`)

**Location**: `crates/bitnet-kernels/src/gpu_utils.rs`

#### Detection Strategy
```rust
pub fn get_gpu_info() -> GpuInfo {
    // Priority 1: Check BITNET_GPU_FAKE environment variable
    if let Ok(fake) = env::var("BITNET_GPU_FAKE") {
        if env::var("BITNET_STRICT_NO_FAKE_GPU").as_deref() == Ok("1") {
            panic!("BITNET_GPU_FAKE is set but strict mode forbids fake GPU");
        }
        // Parse fake value and return mocked GpuInfo
        return GpuInfo { /* ... */ };
    }
    
    // Priority 2: Real detection via system checks
    let cuda = Command::new("nvidia-smi")
        .arg("--query-gpu=gpu_name")
        .arg("--format=csv,noheader")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);
    
    let rocm = Command::new("rocm-smi")
        .arg("--showid")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);
    
    // Metal detection for macOS
    let metal = cfg!(target_os = "macos") || System::name().contains("mac");
}
```

#### GpuInfo Structure
```rust
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub cuda: bool,
    pub cuda_version: Option<String>,
    pub metal: bool,
    pub rocm: bool,
    pub rocm_version: Option<String>,
    pub wgpu: bool,
}

impl GpuInfo {
    pub fn any_available(&self) -> bool {
        self.cuda || self.metal || self.rocm || self.wgpu
    }
    
    pub fn summary(&self) -> String {
        // Human-readable output for diagnostics
    }
}
```

**Reusable Patterns**:
- Command-based detection (checks for executables via `Command::new()`)
- Version parsing from tool output
- Graceful fallback to false if command fails
- Summary generation for diagnostics

---

### 3. Backend Abstraction Layer (`bitnet-inference/src/backends.rs`)

**Location**: `crates/bitnet-inference/src/backends.rs`

#### Backend Trait
```rust
#[async_trait]
pub trait Backend: Send + Sync {
    fn backend_type(&self) -> String;
    fn clone_backend(&self) -> Box<dyn Backend>;
    async fn forward(&self, input: &ConcreteTensor, cache: &mut KVCache) -> Result<ConcreteTensor>;
    fn capabilities(&self) -> BackendCapabilities;
    async fn warmup(&self) -> Result<()>;
}
```

#### Backend Implementations
1. **CpuBackend**
   - Thread configuration
   - Fallback when GPU unavailable
   - Rayon thread pool management

2. **GpuBackend**
   - CUDA device selection
   - Mixed precision support
   - GPU memory tracking

#### Selection Function
```rust
pub fn select_backend(
    model: Arc<dyn Model>,
    preferred_device: Option<Device>,
) -> Result<Box<dyn Backend>> {
    let device = preferred_device
        .unwrap_or_else(|| if GpuBackend::is_available() { Device::Cuda(0) } else { Device::Cpu });
    
    match device {
        Device::Cpu => Ok(Box::new(CpuBackend::new(model)?)),
        Device::Cuda(_) | Device::Metal => {
            if GpuBackend::is_available() {
                Ok(Box::new(GpuBackend::new(model, device)?))
            } else {
                warn!("GPU requested but not available, falling back to CPU");
                Ok(Box::new(CpuBackend::new(model)?))
            }
        }
    }
}
```

**Key Patterns**:
- Trait-based abstraction for swappable backends
- Automatic fallback when preferred device unavailable
- Capability reporting per backend
- Clone support for threading

---

### 4. Kernel Provider Selection (`bitnet-kernels/src/lib.rs`)

**Location**: `crates/bitnet-kernels/src/lib.rs`

#### KernelManager Architecture
```rust
pub struct KernelManager {
    providers: Vec<Box<dyn KernelProvider>>,
    selected: OnceLock<usize>,  // Cached selection
}

impl KernelManager {
    pub fn new() -> Self {
        let mut providers: Vec<Box<dyn KernelProvider>> = vec![Box::new(cpu::FallbackKernel)];
        
        // GPU kernels first (highest priority)
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            if let Ok(cuda_kernel) = gpu::CudaKernel::new() {
                if cuda_kernel.is_available() {
                    providers.insert(0, Box::new(cuda_kernel));
                }
            }
        }
        
        // Optimized CPU kernels in order of preference
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            if is_x86_feature_detected!("avx2") {
                providers.insert(insert_pos, Box::new(cpu::Avx2Kernel));
            }
        }
        
        // Fallback CPU kernel (always available)
        // FFI kernel as fallback option
    }
    
    pub fn select_best(&self) -> Result<&dyn KernelProvider> {
        let selected_idx = self.selected.get_or_init(|| {
            for (i, provider) in self.providers.iter().enumerate() {
                if provider.is_available() {
                    log::info!("Selected kernel provider: {}", provider.name());
                    return i;
                }
            }
            self.providers.len() - 1  // Fallback
        });
    }
}
```

**Selection Strategy**:
- **Priority ordering**: GPU → AVX512 → AVX2 → NEON → FFI → Fallback
- **Lazy caching**: `OnceLock` for single selection
- **Availability checks**: Each provider verifies runtime support
- **Feature gating**: Platform-specific kernels conditional on features

---

### 5. Device-Aware Quantizer (`bitnet-kernels/src/device_aware.rs`)

**Location**: `crates/bitnet-kernels/src/device_aware.rs`

#### Device-Aware Selection Pattern
```rust
pub struct DeviceAwareQuantizer {
    primary_provider: Option<Box<dyn KernelProvider>>,
    fallback_provider: Box<dyn KernelProvider>,
    target_device: Device,
    stats: Arc<Mutex<DeviceStatsInternal>>,
}

impl DeviceAwareQuantizer {
    pub fn new(device: Device) -> Result<Self> {
        let (primary_provider, fallback_provider) = match device {
            #[cfg(any(feature = "gpu", feature = "cuda"))]
            Device::Cuda(device_id) => {
                // Try GPU first
                let gpu_provider = match gpu::CudaKernel::new_with_device(device_id) {
                    Ok(kernel) if kernel.is_available() => Some(Box::new(kernel)),
                    Ok(_) => { log::warn!("CUDA device {} not available", device_id); None }
                    Err(e) => { log::warn!("Failed to create CUDA kernel: {}", e); None }
                };
                
                // Always have CPU fallback
                let cpu_provider = Self::create_best_cpu_provider()?;
                (gpu_provider, cpu_provider)
            }
            #[cfg(not(any(feature = "gpu", feature = "cuda")))]
            Device::Cuda(_device_id) => {
                let cpu_provider = Self::create_best_cpu_provider()?;
                (None, cpu_provider)
            }
            Device::Cpu | Device::Metal => {
                let cpu_provider = Self::create_best_cpu_provider()?;
                (None, cpu_provider)
            }
        };
        
        Ok(Self {
            primary_provider,
            fallback_provider,
            target_device: device,
            stats: Arc::new(Mutex::new(DeviceStatsInternal::default())),
        })
    }
}
```

**Key Features**:
- **Dual-provider model**: Primary (preferred) + Fallback
- **Graceful degradation**: Falls through on errors
- **Performance tracking**: Stats for fallback decisions
- **Memory monitoring**: Tracks device utilization

---

### 6. Preflight/Diagnostic Commands (`xtask/src/main.rs` & `xtask/tests/preflight.rs`)

**Location**: `xtask/tests/preflight.rs`

#### Preflight Test Pattern
```rust
#[test]
fn ac5_preflight_detects_no_gpu_with_fake_none() {
    let output = Command::new("cargo")
        .args(["run", "-p", "xtask", "--", "preflight"])
        .current_dir(workspace_root())
        .env("BITNET_GPU_FAKE", "none")
        .output()
        .expect("Failed to run xtask preflight");
    
    // Check output for GPU indicators
    let indicates_no_gpu = combined_output.contains("GPU: Not available")
        || combined_output.contains("GPU: ✗")
        || combined_output.contains("No GPU");
    
    assert!(indicates_no_gpu, "Preflight should report no GPU with BITNET_GPU_FAKE=none");
}
```

#### Diagnostic Features
```rust
pub fn device_capability_summary() -> String {
    let mut summary = String::from("Device Capabilities:\n");
    
    // Compile-time capabilities
    summary.push_str("  Compiled: ");
    if gpu_compiled() {
        summary.push_str("GPU ✓, ");
    }
    summary.push_str("CPU ✓\n");
    
    // Runtime capabilities
    summary.push_str("  Runtime: ");
    if gpu_available_runtime() {
        let info = crate::gpu_utils::get_gpu_info();
        if let Some(version) = &info.cuda_version {
            summary.push_str(&format!("CUDA {} ✓", version));
        }
    } else {
        summary.push_str("CUDA ✗");
    }
    summary.push_str(", CPU ✓");
}
```

**Preflight Pattern**:
- Two-tier validation: compilation + runtime
- Version reporting when available
- Human-readable indicators (✓/✗)
- Environment override testing

---

## Backend Selection Patterns

### Pattern 1: Two-Level Feature Gates
```rust
// Compile-time check
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_function() { }

// Runtime check
if gpu_compiled() && gpu_available_runtime() {
    // Use GPU
}
```

**Reusable for C++**:
```rust
#[cfg(feature = "cpp")]
pub fn cpp_compiled() -> bool { true }

#[cfg(not(feature = "cpp"))]
pub fn cpp_compiled() -> bool { false }

#[cfg(feature = "cpp")]
pub fn cpp_available_runtime() -> bool {
    std::env::var("BITNET_CPP_FAKE")
        .map(|v| v.eq_ignore_ascii_case("yes") || v == "1")
        .or_else(|_| check_cpp_backend_available())
        .unwrap_or(false)
}
```

### Pattern 2: Environment Variable Hierarchy
```rust
// 1. Strict mode override (prevents fake)
BITNET_STRICT_MODE=1  // Forces real detection only

// 2. Fake override (for testing)
BITNET_GPU_FAKE=cuda|none|metal|rocm

// 3. Backend-specific env vars
BITNET_CPP_DIR=/path/to/bitnet.cpp  // C++ reference location
BITNET_IQ2S_IMPL=rust|ffi           // Quantization backend
```

### Pattern 3: Graceful Fallback Chain
```
GPU (preferred)
  └─> Fail? Check strict mode
      └─> Strict? Return GPU unavailable
      └─> Not strict? Check BITNET_GPU_FAKE
          └─> Yes? Use fake
          └─> No? Try real detection
              └─> Failed? Fallback to CPU

CPU (always available)
  └─> Try AVX512 (if compiled)
      └─> Try AVX2 (if compiled)
          └─> Try NEON (if compiled)
              └─> Fallback to scalar
```

### Pattern 4: Provider Registry
```rust
struct ProviderRegistry {
    providers: Vec<Box<dyn Provider>>,
    selected: OnceLock<usize>,  // Single selection
}

// Benefit: Allows adding C++Backend, FfiBridge, etc. without code duplication
```

---

## Environment Variables Overview

| Variable | Purpose | Values | Precedence |
|----------|---------|--------|-----------|
| `BITNET_GPU_FAKE` | Override GPU detection | `cuda`, `none`, `metal`, `rocm` | Highest (unless strict mode) |
| `BITNET_STRICT_MODE` | Force real detection | `1`, `true` | Overrides BITNET_GPU_FAKE |
| `BITNET_STRICT_NO_FAKE_GPU` | Panic if fake set | `1` | Safety guard |
| `BITNET_CPP_DIR` | C++ reference location | Path | Configuration |
| `BITNET_IQ2S_IMPL` | Quantization backend | `rust`, `ffi` | Implementation selection |
| `BITNET_CORRECTION_POLICY` | Model corrections | Path to YAML | Configuration |
| `BITNET_ALLOW_RUNTIME_CORRECTIONS` | Enable corrections | `1` | Safety gate |

---

## Proposed C++ Backend Enum

Based on patterns observed in `backends.rs`, here's how to add C++ backend:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CppBackend {
    Reference,           // Microsoft BitNet C++ reference
    Optimized,          // Custom-optimized C++ variant
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Gpu(usize),          // GPU device ID
    Cpp(CppBackend),     // C++ backend variant
    Metal,
}

// Selection logic
pub fn select_cpp_backend() -> Result<Box<dyn BackendImpl>> {
    // 1. Check if C++ compiled
    if !cpp_compiled() {
        return Err("C++ backend not compiled (missing --features cpp)");
    }
    
    // 2. Check runtime availability
    if !cpp_available_runtime() {
        // Check if strict mode
        if std::env::var("BITNET_STRICT_MODE").is_ok() {
            return Err("C++ backend not available in strict mode");
        }
        // Otherwise fallback to CPU/GPU
    }
    
    // 3. Check for BITNET_CPP_FAKE override
    if let Ok(fake) = std::env::var("BITNET_CPP_FAKE") {
        if fake.eq_ignore_ascii_case("reference") {
            return Ok(Box::new(CppReferenceBackend::new()?));
        }
    }
    
    // 4. Real detection: check for bitnet.cpp installation
    if let Ok(cpp_dir) = std::env::var("BITNET_CPP_DIR") {
        // Check if libbitnet.so or .dll exists
        if PathBuf::from(&cpp_dir).join("lib/libbitnet.so").exists() {
            return Ok(Box::new(CppReferenceBackend::new()?));
        }
    }
    
    // 5. Try standard installation locations
    for standard_path in &["/usr/local/lib", "~/.local/lib", "./bitnet.cpp/lib"] {
        if PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(standard_path)
            .join("libbitnet.so")
            .exists()
        {
            return Ok(Box::new(CppReferenceBackend::new()?));
        }
    }
    
    Err("C++ backend not found - set BITNET_CPP_DIR or ensure installation")
}
```

---

## Implementation Checklist for C++ Backend

1. **Feature Gate** (`Cargo.toml`)
   ```toml
   [features]
   cpp = ["bitnet-ffi"]
   cpp-reference = ["cpp"]
   crossval-all = ["inference", "crossval", "ffi"]
   ```

2. **Compilation Check** (`src/cpp_features.rs`)
   ```rust
   #[inline]
   pub fn cpp_compiled() -> bool {
       cfg!(feature = "cpp")
   }
   ```

3. **Runtime Detection** (`src/cpp_detection.rs`)
   ```rust
   pub fn cpp_available_runtime() -> bool {
       // Check BITNET_CPP_FAKE first
       // Then check BITNET_CPP_DIR
       // Then search standard paths
       // Use Command to probe for libbitnet.so
   }
   
   pub struct CppInfo {
       pub available: bool,
       pub version: Option<String>,
       pub installation_path: Option<PathBuf>,
   }
   ```

4. **Backend Implementation**
   ```rust
   pub struct CppBackendImpl {
       cpp_dir: PathBuf,
       version: String,
       lib_handle: unsafe { *mut libc::c_void },  // libbitnet.so handle
   }
   
   impl Backend for CppBackendImpl {
       fn backend_type(&self) -> String {
           format!("cpp_reference_v{}", self.version)
       }
   }
   ```

5. **Selection Logic** (extend `backends.rs`)
   ```rust
   pub fn select_best_backend(config: &InferenceConfig) -> Result<Box<dyn Backend>> {
       match config.backend_preference {
           BackendPreference::Cpp => select_cpp_backend(),
           BackendPreference::Auto => {
               // Try: GPU → C++ → CPU
               if let Ok(backend) = select_gpu_backend_with_fallback() {
                   return Ok(backend);
               }
               if let Ok(backend) = select_cpp_backend_with_fallback() {
                   return Ok(backend);
               }
               Ok(Box::new(CpuBackend::new()?))
           }
       }
   }
   ```

6. **Diagnostics/Preflight**
   ```rust
   pub fn cpp_capability_summary() -> String {
       let mut summary = String::new();
       summary.push_str("C++ Backend:\n");
       if cpp_compiled() {
           summary.push_str("  Compiled: ✓\n");
           if let Ok(info) = cpp_detection::get_cpp_info() {
               summary.push_str(&format!("  Runtime: ✓ ({})\n", info.installation_path.display()));
               if let Some(version) = &info.version {
                   summary.push_str(&format!("  Version: {}\n", version));
               }
           } else {
               summary.push_str("  Runtime: ✗\n");
           }
       } else {
           summary.push_str("  Compiled: ✗\n");
           summary.push_str("  Runtime: N/A\n");
       }
       summary
   }
   ```

---

## Reusable Patterns Summary

| Pattern | Location | Reusable For |
|---------|----------|-------------|
| Two-level feature gates | `device_features.rs` | Any compile+runtime feature |
| Environment variable hierarchy | `gpu_utils.rs` | Fake overrides, strict mode |
| Command-based detection | `gpu_utils.rs` (nvidia-smi) | Any system check |
| Version parsing | `gpu_utils.rs` | Tool version extraction |
| Graceful fallback chain | `device_aware.rs` | Multi-provider selection |
| Provider registry | `kernels/lib.rs` | Dynamic provider selection |
| Lazy caching | `KernelManager` | Single-selection optimization |
| Diagnostic summary | `device_features.rs` | Preflight reporting |
| Trait-based abstraction | `backends.rs` | Swappable implementations |

---

## Testing Patterns

**Preflight Tests** (`xtask/tests/preflight.rs`):
- Validate compile-time detection with feature gates
- Validate runtime detection with/without fake overrides
- Validate strict mode enforcement
- Validate output formatting (human-readable indicators)
- Validate edge cases (invalid fake values, missing tools)

**Adoption for C++**:
```rust
#[test]
fn test_cpp_backend_compile_detection() {
    let compiled = cpp_compiled();
    #[cfg(feature = "cpp")]
    assert!(compiled);
    #[cfg(not(feature = "cpp"))]
    assert!(!compiled);
}

#[test]
fn test_cpp_backend_fake_override() {
    std::env::set_var("BITNET_CPP_FAKE", "reference");
    assert!(cpp_available_runtime());
    
    std::env::set_var("BITNET_CPP_FAKE", "none");
    assert!(!cpp_available_runtime());
}
```

---

## Key Takeaways

1. **BitNet.rs uses unified patterns** for all backend detection:
   - Compile-time checks via feature gates
   - Runtime checks via system commands or environment variables
   - Graceful fallback chains
   - Fake overrides for testing

2. **Environment variables are hierarchical**:
   - Fake overrides (for testing)
   - Strict mode (safety gate)
   - Real detection (as fallback)

3. **Provider registry pattern** is extensible:
   - Add C++Backend to providers list
   - Maintain priority ordering
   - Lazy cache selection

4. **Diagnostics are built-in**:
   - Compile vs runtime distinction
   - Version reporting
   - Human-readable summaries
   - Preflight commands

5. **Fallback is always available**:
   - CPU as ultimate fallback
   - FFI as optional layer
   - Scalar kernels as base case

---

## Files for Reference

| File | Purpose | Key Patterns |
|------|---------|-------------|
| `crates/bitnet-kernels/src/device_features.rs` | Two-level detection API | Compile+runtime checks |
| `crates/bitnet-kernels/src/gpu_utils.rs` | GPU runtime detection | Command-based detection, version parsing |
| `crates/bitnet-inference/src/backends.rs` | Backend abstraction | Trait-based design, fallback selection |
| `crates/bitnet-kernels/src/lib.rs` | Kernel selection | Provider registry, lazy caching |
| `crates/bitnet-kernels/src/device_aware.rs` | Device-aware selection | Dual-provider pattern, graceful fallback |
| `xtask/tests/preflight.rs` | Diagnostic tests | Fake override validation, preflight patterns |

