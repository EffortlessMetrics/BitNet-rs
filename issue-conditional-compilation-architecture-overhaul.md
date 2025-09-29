# [Architecture] Conditional Compilation to Runtime Detection Migration

## Problem Description

BitNet.rs currently relies heavily on conditional compilation (`#[cfg(...)]`) for feature detection and capability selection, creating four critical architectural problems:

1. **Download Functionality**: `download_file()` conditionally compiled with `#[cfg(feature = "downloads")]`
2. **Tokenizer Support**: `load_tokenizer_from_gguf_reader()` conditionally compiled for SentencePiece (`spm` feature)
3. **Kernel Management**: `KernelManager::new()` uses conditional compilation for CUDA, AVX2, AVX-512, NEON, and FFI kernels
4. **SIMD Operations**: `quantize_avx2/dequantize_avx2` functions require compile-time feature detection

This approach forces users to recompile for different target environments and prevents runtime optimization selection.

## Environment

- **Affected Crates**: `bitnet-tokenizers`, `bitnet-kernels`, `bitnet-quantization`
- **Primary Files**:
  - `crates/bitnet-tokenizers/src/download.rs`
  - `crates/bitnet-tokenizers/src/loader.rs`
  - `crates/bitnet-kernels/src/lib.rs`
  - `crates/bitnet-quantization/src/simd_ops.rs`
- **Build Configuration**: All feature combinations (`cpu`, `gpu`, `ffi`, `avx2`, `avx512`, `neon`, `spm`, `downloads`)
- **Target Architectures**: x86_64, ARM64, with various instruction set extensions

## Root Cause Analysis

### Architectural Limitations

1. **Static Feature Selection**: Compile-time feature flags prevent runtime adaptation
   ```rust
   #[cfg(not(feature = "downloads"))]
   async fn download_file(&self, _url: &str, _path: &Path) -> Result<()> {
       Err(BitNetError::Config("Download feature not enabled".to_string()))
   }
   ```

2. **Hardware Lock-in**: SIMD kernels selected at compile-time rather than runtime
   ```rust
   #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
   if is_x86_feature_detected!("avx2") {
       providers.insert(insert_pos, Box::new(cpu::Avx2Kernel));
   }
   ```

3. **Deployment Complexity**: Different binaries required for different environments

4. **Testing Fragmentation**: Feature combinations create exponential test matrix

### Current Impact

- **Distribution**: Separate builds required for CPU-only vs GPU-enabled environments
- **Performance**: Suboptimal kernel selection prevents runtime optimization
- **Maintenance**: Feature flag combinations create complex build matrix
- **User Experience**: Runtime capability errors instead of graceful fallbacks

## Impact Assessment

- **Severity**: High - Affects deployment flexibility and runtime performance
- **Deployment Impact**: Requires multiple binary distributions
- **Performance Impact**: Prevents optimal runtime kernel selection
- **Maintenance Burden**: Exponential feature flag test combinations
- **User Experience**: Poor error messages and rigid capability requirements

## Proposed Solution

### 1. Runtime Download Detection System

**Dynamic Download Capability**:
```rust
pub struct DownloadCapability {
    available: bool,
    http_client: Option<Arc<reqwest::Client>>,
    strategy: DownloadStrategy,
}

impl DownloadCapability {
    pub fn detect() -> Self {
        let available = Self::probe_network_capability();
        let http_client = if available {
            Some(Arc::new(reqwest::Client::new()))
        } else {
            None
        };

        Self {
            available,
            http_client,
            strategy: if available {
                DownloadStrategy::Http
            } else {
                DownloadStrategy::LocalOnly
            },
        }
    }

    async fn download_file(&self, url: &str, path: &Path) -> Result<()> {
        match &self.http_client {
            Some(client) => {
                let response = client.get(url).send().await?;
                let bytes = response.bytes().await?;
                tokio::fs::write(path, bytes).await?;
                Ok(())
            }
            None => Err(BitNetError::Capability(
                "Network download not available. Use local files or enable network access."
            ))
        }
    }

    fn probe_network_capability() -> bool {
        // Probe for network availability, HTTP client dependencies, etc.
        std::env::var("BITNET_OFFLINE").is_err() &&
            std::net::TcpStream::connect_timeout(
                &"8.8.8.8:53".parse().unwrap(),
                Duration::from_millis(100)
            ).is_ok()
    }
}
```

### 2. Universal Tokenizer Loading System

**Runtime Tokenizer Detection**:
```rust
pub struct TokenizerCapabilities {
    sentencepiece: bool,
    tiktoken: bool,
    huggingface: bool,
}

impl TokenizerCapabilities {
    pub fn detect() -> Self {
        Self {
            sentencepiece: Self::probe_sentencepiece(),
            tiktoken: Self::probe_tiktoken(),
            huggingface: Self::probe_huggingface(),
        }
    }

    fn probe_sentencepiece() -> bool {
        // Check if SentencePiece binaries/libraries are available
        std::process::Command::new("sentencepiece")
            .arg("--version")
            .output()
            .is_ok()
    }
}

pub fn load_tokenizer_from_gguf_reader(
    reader: &bitnet_models::GgufReader,
) -> Result<Box<dyn Tokenizer + Send + Sync>> {
    let capabilities = TokenizerCapabilities::detect();

    if let Some(bytes) = reader.get_bin_or_u8_array("tokenizer.ggml.model") {
        if capabilities.sentencepiece {
            let bos = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
            let eos = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
            return Ok(crate::sp_tokenizer::SpTokenizer::from_gguf_blob(&bytes, bos, eos)?);
        } else {
            return Err(anyhow::anyhow!(
                "SentencePiece tokenizer detected but support not available. Install SentencePiece or use alternative tokenizer."
            ));
        }
    }

    // Try alternative tokenizer formats
    if capabilities.tiktoken {
        return Self::try_tiktoken_loading(reader);
    }

    Err(anyhow::anyhow!("No compatible tokenizer found in GGUF"))
}
```

### 3. Dynamic Kernel Provider System

**Runtime Kernel Detection and Loading**:
```rust
pub struct KernelCapabilities {
    cuda: Option<CudaInfo>,
    avx512: bool,
    avx2: bool,
    neon: bool,
    ffi: bool,
}

impl KernelCapabilities {
    pub fn detect() -> Self {
        Self {
            cuda: Self::probe_cuda(),
            avx512: Self::probe_avx512(),
            avx2: Self::probe_avx2(),
            neon: Self::probe_neon(),
            ffi: Self::probe_ffi(),
        }
    }

    fn probe_cuda() -> Option<CudaInfo> {
        // Runtime CUDA detection
        if let Ok(device_count) = cudaruntimesys::cudaGetDeviceCount() {
            if device_count > 0 {
                return Some(CudaInfo {
                    device_count,
                    compute_capability: Self::get_compute_capability(),
                    memory_info: Self::get_memory_info(),
                });
            }
        }
        None
    }

    #[cfg(target_arch = "x86_64")]
    fn probe_avx512() -> bool {
        is_x86_feature_detected!("avx512f") &&
        is_x86_feature_detected!("avx512bw") &&
        is_x86_feature_detected!("avx512vl")
    }

    #[cfg(target_arch = "aarch64")]
    fn probe_neon() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    fn probe_ffi() -> bool {
        // Check if FFI bridge is available
        std::env::var("BITNET_FFI_PATH").is_ok() ||
        std::path::Path::new("/usr/local/lib/libbitnet.so").exists()
    }
}

impl KernelManager {
    pub fn new() -> Self {
        let capabilities = KernelCapabilities::detect();
        let mut providers: Vec<Box<dyn KernelProvider>> = vec![];

        // Add providers based on runtime capabilities
        if let Some(cuda_info) = capabilities.cuda {
            match gpu::CudaKernel::new_with_info(cuda_info) {
                Ok(cuda_kernel) => {
                    log::info!("CUDA kernel loaded: {} devices", cuda_info.device_count);
                    providers.push(Box::new(cuda_kernel));
                }
                Err(e) => log::warn!("CUDA available but failed to initialize: {}", e),
            }
        }

        if capabilities.avx512 {
            providers.push(Box::new(cpu::Avx512Kernel::new()));
            log::info!("AVX-512 kernel loaded");
        }

        if capabilities.avx2 {
            providers.push(Box::new(cpu::Avx2Kernel::new()));
            log::info!("AVX2 kernel loaded");
        }

        if capabilities.neon {
            providers.push(Box::new(cpu::NeonKernel::new()));
            log::info!("NEON kernel loaded");
        }

        if capabilities.ffi {
            match ffi::FfiKernel::new() {
                Ok(ffi_kernel) => {
                    providers.push(Box::new(ffi_kernel));
                    log::info!("FFI kernel loaded");
                }
                Err(e) => log::warn!("FFI bridge available but failed to load: {}", e),
            }
        }

        // Always include fallback
        providers.push(Box::new(cpu::FallbackKernel::new()));

        Self { providers, selected: OnceLock::new() }
    }
}
```

### 4. Dynamic SIMD Operation Selection

**Runtime SIMD Capability Detection**:
```rust
pub struct SimdCapabilities {
    avx512: bool,
    avx2: bool,
    neon: bool,
    baseline: bool,
}

impl SimdCapabilities {
    pub fn detect() -> Self {
        Self {
            avx512: Self::probe_avx512(),
            avx2: Self::probe_avx2(),
            neon: Self::probe_neon(),
            baseline: true,
        }
    }
}

impl QuantizationKernels {
    pub fn new() -> Self {
        let capabilities = SimdCapabilities::detect();

        Self {
            capabilities,
            strategy: Self::select_optimal_strategy(&capabilities),
        }
    }

    pub fn quantize_i2s(
        &self,
        data: &[f32],
        scales: &[f32],
        block_size: usize,
    ) -> Result<Vec<i8>> {
        match self.strategy {
            SimdStrategy::Avx512 if self.capabilities.avx512 => {
                unsafe { self.quantize_avx512_impl(data, scales, block_size) }
            }
            SimdStrategy::Avx2 if self.capabilities.avx2 => {
                unsafe { self.quantize_avx2_impl(data, scales, block_size) }
            }
            SimdStrategy::Neon if self.capabilities.neon => {
                unsafe { self.quantize_neon_impl(data, scales, block_size) }
            }
            _ => self.quantize_scalar_impl(data, scales, block_size),
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn quantize_avx2_impl(
        &self,
        data: &[f32],
        scales: &[f32],
        block_size: usize,
    ) -> Result<Vec<i8>> {
        // AVX2 implementation with runtime safety check
        debug_assert!(is_x86_feature_detected!("avx2"));

        let mut result = Vec::with_capacity(data.len());
        // ... AVX2 intrinsics implementation ...
        Ok(result)
    }
}
```

## Implementation Plan

### Phase 1: Capability Detection Framework (Week 1-2)
- [ ] Create `SystemCapabilities` trait for unified capability detection
- [ ] Implement `NetworkCapability` for download functionality
- [ ] Add `HardwareCapabilities` for CPU instruction sets
- [ ] Create `LibraryCapabilities` for external dependencies

### Phase 2: Download System Overhaul (Week 2-3)
- [ ] Replace conditional compilation in `download.rs`
- [ ] Implement runtime network detection
- [ ] Add graceful offline mode fallbacks
- [ ] Create comprehensive download strategy system

### Phase 3: Tokenizer Loading Modernization (Week 3-4)
- [ ] Remove SentencePiece conditional compilation
- [ ] Implement runtime tokenizer capability detection
- [ ] Add multi-format tokenizer fallback chain
- [ ] Create universal tokenizer loading interface

### Phase 4: Kernel Management Refactoring (Week 4-5)
- [ ] Replace all kernel conditional compilation
- [ ] Implement dynamic kernel provider loading
- [ ] Add runtime CUDA capability detection
- [ ] Create optimal kernel selection algorithms

### Phase 5: SIMD Operations Enhancement (Week 5-6)
- [ ] Remove SIMD conditional compilation
- [ ] Implement runtime instruction set detection
- [ ] Add fallback chain for quantization operations
- [ ] Create performance profiling for optimal selection

### Phase 6: Integration and Testing (Week 6-7)
- [ ] Integrate all runtime detection systems
- [ ] Add comprehensive capability reporting
- [ ] Create end-to-end test suite
- [ ] Implement performance benchmarking

## Testing Strategy

### Capability Detection Tests
```rust
#[test]
fn test_capability_detection_consistency() {
    let caps = SystemCapabilities::detect();

    // Verify capability detection is deterministic
    assert_eq!(caps, SystemCapabilities::detect());

    // Verify capabilities match actual hardware
    if caps.hardware.avx2 {
        assert!(is_x86_feature_detected!("avx2"));
    }
}

#[test]
fn test_graceful_capability_degradation() {
    let mut manager = KernelManager::new();

    // Disable specific capabilities
    manager.disable_capability(Capability::Cuda);

    // Verify graceful fallback
    let kernel = manager.select_optimal_kernel(TaskType::Quantization);
    assert!(kernel.capability_level() >= CapabilityLevel::Baseline);
}
```

### Runtime Adaptation Tests
```rust
#[test]
fn test_download_offline_mode() {
    std::env::set_var("BITNET_OFFLINE", "1");

    let downloader = SmartTokenizerDownload::new();
    let result = downloader.download_file("http://example.com/model", &Path::new("/tmp/test"));

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("offline mode"));
}

#[test]
fn test_tokenizer_fallback_chain() {
    let reader = create_test_gguf_without_sentencepiece();

    // Should gracefully fall back to alternative tokenizers
    let tokenizer = load_tokenizer_from_gguf_reader(&reader).unwrap();
    assert!(tokenizer.tokenize("test").is_ok());
}
```

### Cross-Platform Validation
```rust
#[test]
#[cfg(feature = "crossval")]
fn test_runtime_kernel_consistency() {
    let capabilities = KernelCapabilities::detect();
    let manager = KernelManager::new();

    for task_type in [TaskType::Quantization, TaskType::Attention, TaskType::Linear] {
        let kernel = manager.select_optimal_kernel(task_type);

        // Verify selected kernel matches capabilities
        assert!(kernel.is_compatible_with(&capabilities));

        // Cross-validate with reference implementation
        let test_data = generate_test_data(task_type);
        let bitnet_result = kernel.process(&test_data)?;
        let reference_result = reference_kernel.process(&test_data)?;

        assert_tensors_close(&bitnet_result, &reference_result, 1e-5);
    }
}
```

## Risk Assessment

### Technical Risks
1. **Runtime Detection Overhead**: Capability probing may add startup latency
   - *Mitigation*: Cache detection results, parallel probing, lazy initialization
2. **Fallback Performance**: Graceful degradation may significantly impact performance
   - *Mitigation*: Comprehensive benchmarking, performance warnings, optimization recommendations
3. **Binary Size**: Including all kernel implementations may increase binary size
   - *Mitigation*: Dynamic loading, feature-specific distributions for constrained environments

### Compatibility Risks
1. **Breaking Changes**: Runtime detection may change existing behavior
   - *Mitigation*: Compatibility mode, extensive testing, gradual migration path
2. **Platform-Specific Issues**: Different capability detection on various platforms
   - *Mitigation*: Comprehensive platform testing, conservative capability assumptions

## Acceptance Criteria

### Functional Requirements
- [ ] All conditional compilation removed from core functionality
- [ ] Runtime capability detection for all major subsystems
- [ ] Graceful fallback chains for all capabilities
- [ ] Comprehensive error messages for missing capabilities
- [ ] Deterministic capability detection across runs

### Performance Requirements
- [ ] Capability detection overhead <10ms on cold start
- [ ] Runtime kernel selection within 5% of compile-time optimal
- [ ] Memory overhead for capability detection <1MB
- [ ] No performance regression in optimal capability scenarios

### Quality Requirements
- [ ] Cross-platform capability detection accuracy >99%
- [ ] Comprehensive test coverage for all fallback scenarios
- [ ] All existing functionality preserved through runtime detection
- [ ] Clear documentation for capability requirements and fallbacks

## Related Issues

- BitNet.rs #218: Device-aware quantization system
- BitNet.rs #251: Production-ready inference server
- BitNet.rs #260: Mock elimination project

## Implementation Notes

### BitNet.rs Integration
- Maintain compatibility with existing feature flag architecture during transition
- Leverage `crossval` framework for runtime detection validation
- Use `tracing` for capability detection logging and debugging
- Integrate with existing error handling patterns using `anyhow`

### Migration Strategy
1. **Parallel Implementation**: Implement runtime detection alongside existing conditional compilation
2. **Feature Flag Transition**: Gradually migrate feature flags to runtime detection
3. **Backward Compatibility**: Maintain existing behavior during transition period
4. **Documentation Update**: Comprehensive documentation for new capability system