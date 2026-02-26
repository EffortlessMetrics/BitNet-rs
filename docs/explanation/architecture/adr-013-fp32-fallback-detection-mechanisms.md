# ADR-013: FP32 Fallback Detection Mechanisms

## Status

**ACCEPTED** - Issue #453 Implementation

## Context

BitNet-rs quantized layers can silently fall back to FP32 dequantization when native quantized kernels are unavailable. Issue #453 requires runtime detection mechanisms to identify and reject FP32 fallback in strict mode.

### Fallback Scenarios

**I2S Quantization Fallback:**
1. **Kernel Not Compiled:** Missing `--features cpu|gpu` flag
2. **Device Mismatch:** Tensor on GPU, layer on CPU (or vice versa)
3. **Unsupported Dimensions:** Non-multiple of SIMD block size
4. **CUDA Unavailable:** GPU requested but CUDA runtime unavailable

**TL1/TL2 Quantization Fallback:**
1. **Architecture Mismatch:** TL1 requires ARM NEON (fails on x86), TL2 requires AVX2 (fails on ARM)
2. **Lookup Table Failure:** Memory allocation fails for table
3. **Numerical Instability:** Overflow in table construction

### Requirements

1. **Compile-Time Detection:** Identify missing kernels at compile time
2. **Runtime Detection:** Detect device mismatches and unavailable kernels at runtime
3. **Precise Errors:** Provide detailed context for fallback scenarios
4. **Performance:** <1% overhead for detection checks

## Decision

We implement **multi-layer fallback detection** with compile-time checks, runtime queries, and post-inference validation.

### Layer 1: Compile-Time Detection (Feature Flags)

**Purpose:** Catch missing kernel compilation at build time

**Implementation:**
```rust
// crates/bitnet-inference/src/layers/quantized_linear.rs

impl QuantizedLinear {
    async fn forward_i2s(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        // Compile-time check: GPU feature flag
        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        {
            if self.device.is_cuda() {
                return Err(BitNetError::FeatureMissing(
                    "GPU inference requires --features gpu".to_string()
                ));
            }
        }

        // Compile-time check: CPU feature flag
        #[cfg(not(feature = "cpu"))]
        {
            if self.device.is_cpu() {
                return Err(BitNetError::FeatureMissing(
                    "CPU inference requires --features cpu".to_string()
                ));
            }
        }

        // Proceed with quantized inference
        self.quantized_matmul_i2s(input).await
    }
}
```

**Characteristics:**
- **Timing:** Compile time (zero runtime overhead)
- **Scope:** Feature flag presence (`cpu`, `gpu`)
- **Limitations:** Cannot detect runtime device mismatches

### Layer 2: Runtime Kernel Availability Query

**Purpose:** Detect unavailable kernels at runtime (device mismatches, CUDA unavailable)

**Implementation:**
```rust
// crates/bitnet-kernels/src/lib.rs

/// Query whether native quantized kernel is available
pub fn is_quantized_kernel_available(
    qtype: QuantizationType,
    device: Device,
    dims: (usize, usize)
) -> bool {
    match (qtype, device) {
        (QuantizationType::I2S, Device::Cuda(device_id)) => {
            // Runtime check: CUDA available?
            if !cuda_runtime_available() {
                return false;
            }

            // Check GPU capabilities
            if let Ok(info) = get_gpu_info(device_id) {
                info.supports_i2s_quantization()
            } else {
                false
            }
        }
        (QuantizationType::I2S, Device::Cpu) => {
            // Check SIMD availability (AVX2/AVX-512/NEON)
            cfg!(target_feature = "avx2")
                || cfg!(target_feature = "avx512f")
                || cfg!(target_feature = "neon")
        }
        (QuantizationType::TL1, Device::Cpu) => {
            // TL1 requires ARM NEON
            cfg!(target_arch = "aarch64") && cfg!(target_feature = "neon")
        }
        (QuantizationType::TL2, Device::Cpu) => {
            // TL2 requires x86 AVX2/AVX-512
            (cfg!(target_arch = "x86_64") || cfg!(target_arch = "x86"))
                && (cfg!(target_feature = "avx2") || cfg!(target_feature = "avx512f"))
        }
        _ => false,
    }
}

/// Check if CUDA runtime is available at runtime
fn cuda_runtime_available() -> bool {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        // Try to initialize CUDA context
        match cuda_runtime::cuda_init() {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    {
        false
    }
}
```

**Characteristics:**
- **Timing:** Runtime (during forward pass)
- **Scope:** Device availability, SIMD features, GPU capabilities
- **Overhead:** <1% (single function call per forward pass)

### Layer 3: Device Capability Detection

**Purpose:** Validate device capabilities for quantization operations

**Implementation:**
```rust
// crates/bitnet-kernels/src/gpu/mod.rs

/// Get GPU device information
pub fn get_gpu_info(device_id: usize) -> Result<GpuInfo> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        use cuda_runtime::cuda_get_device_properties;

        let props = cuda_get_device_properties(device_id as i32)?;

        Ok(GpuInfo {
            device_id,
            name: props.name,
            compute_capability: (props.major, props.minor),
            total_memory: props.total_global_mem,
            supports_fp16: props.major >= 6,  // Pascal+
            supports_bf16: props.major >= 8,  // Ampere+
            supports_tensor_cores: props.major >= 7,  // Volta+
        })
    }

    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    {
        Err(anyhow!("GPU support not compiled (missing --features gpu)"))
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub device_id: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory: usize,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_tensor_cores: bool,
}

impl GpuInfo {
    /// Check if GPU supports I2S quantization
    pub fn supports_i2s_quantization(&self) -> bool {
        // I2S requires FP16 support (compute capability 6.1+)
        self.compute_capability.0 >= 6
            && (self.compute_capability.0 > 6 || self.compute_capability.1 >= 1)
    }
}
```

**Characteristics:**
- **Timing:** Runtime (during device initialization)
- **Scope:** GPU compute capability, memory, precision support
- **Overhead:** One-time cost (cached per device)

### Layer 4: Fallback Reason Tracking

**Purpose:** Provide detailed context for fallback scenarios

**Implementation:**
```rust
// crates/bitnet-quantization/src/lib.rs

/// Quantization fallback reason
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationFallbackReason {
    /// Quantized kernel not compiled (missing feature flag)
    KernelNotAvailable,

    /// Device mismatch (tensor on GPU, layer on CPU)
    DeviceMismatch,

    /// Unsupported dimensions (non-multiple of SIMD block size)
    UnsupportedDimensions,

    /// Numerical instability (overflow in quantization)
    NumericalInstability,

    /// CUDA runtime unavailable (GPU requested but CUDA missing)
    CudaUnavailable,

    /// Architecture mismatch (TL1 on x86, TL2 on ARM)
    ArchitectureMismatch,
}

impl QuantizationFallbackReason {
    /// Get human-readable error message
    pub fn error_message(&self) -> &'static str {
        match self {
            Self::KernelNotAvailable => "quantized kernel not compiled (use --features cpu|gpu)",
            Self::DeviceMismatch => "device mismatch (tensor and layer on different devices)",
            Self::UnsupportedDimensions => "unsupported dimensions (non-multiple of SIMD block size)",
            Self::NumericalInstability => "numerical instability (overflow in quantization)",
            Self::CudaUnavailable => "CUDA runtime unavailable (GPU requested but CUDA missing)",
            Self::ArchitectureMismatch => "architecture mismatch (TL1 requires ARM, TL2 requires x86)",
        }
    }
}
```

### Layer 5: Post-Inference Receipt Validation

**Purpose:** Validate receipts accurately reflect computation path

**Implementation:**
```rust
// xtask/src/main.rs

fn verify_quantization_claims(receipt: &Receipt) -> Result<()> {
    // Detect FP32 fallback from kernel IDs
    let has_quantized = receipt.kernels.iter().any(is_quantized_kernel);
    let has_fallback = receipt.kernels.iter().any(is_fallback_kernel);

    if has_fallback && !has_quantized {
        bail!("Receipt claims quantized computation but only fallback kernels detected");
    }

    // Validate kernel_path field (v1.1.0)
    if let Some(kernel_path) = &receipt.kernel_path {
        if kernel_path == "fp32_fallback" && receipt.compute_path == "quantized" {
            bail!("Receipt claims quantized compute_path but kernel_path is fp32_fallback");
        }
    }

    Ok(())
}

fn is_fallback_kernel(kernel_id: &str) -> bool {
    kernel_id.contains("dequant_")
        || kernel_id.contains("fp32_matmul")
        || kernel_id.contains("scalar_")
        || kernel_id.contains("fallback_")
}
```

## Integration with Three-Tier Validation

### Tier 1: Debug Assertions

```rust
#[cfg(debug_assertions)]
{
    if !self.has_native_quantized_kernel() {
        let reason = self.detect_fallback_reason();
        panic!(
            "fallback to FP32 in debug mode: layer={}, qtype={:?}, reason={:?}",
            self.name, self.qtype, reason
        );
    }
}
```

### Tier 2: Strict Mode

```rust
if !self.has_native_quantized_kernel() {
    let reason = self.detect_fallback_reason();

    let strict_mode = StrictModeEnforcer::new();
    if strict_mode.get_config().enforce_quantized_inference {
        return Err(BitNetError::StrictMode(format!(
            "FP32 fallback rejected - qtype={:?}, device={:?}, reason={:?}",
            self.qtype, self.device, reason
        )));
    }
}
```

### Tier 3: Receipt Validation

```rust
// Post-inference validation
let result = verify_quantization_claims(&receipt)?;
```

## Rationale

### Why Multi-Layer Detection?

**Single-Layer (Runtime Only):**
- **Rejected:** No compile-time feedback (slow development cycle)
- **Rejected:** Higher runtime overhead (checking every forward pass)

**Two-Layer (Compile + Runtime):**
- **Rejected:** No post-inference verification (cannot validate historical baselines)

**Three-Layer (Compile + Runtime + Post-Inference):**
- ✅ Compile-time: Fast feedback for missing feature flags
- ✅ Runtime: Detect device mismatches and unavailable kernels
- ✅ Post-inference: Independent validation of computation claims

**Selected: Multi-Layer (Best Practice)**
- Combines all three layers for comprehensive fallback detection
- Minimal overhead (<1% runtime, zero compile-time after build)
- Detailed error messages with fallback reasons

## Implementation Strategy

### Phase 1: Compile-Time Detection (Week 1, Days 1-2)

**Files Modified:**
- `crates/bitnet-inference/src/layers/quantized_linear.rs`

**Validation:**
```bash
# Test missing GPU feature flag
cargo build --no-default-features
# Expected error: "GPU inference requires --features gpu"

# Test missing CPU feature flag
cargo build --no-default-features
# Expected error: "CPU inference requires --features cpu"
```

### Phase 2: Runtime Kernel Availability (Week 1, Days 3-5)

**Files Modified:**
- `crates/bitnet-kernels/src/lib.rs`
- `crates/bitnet-kernels/src/gpu/mod.rs`

**Validation:**
```bash
# Test kernel availability query
cargo test -p bitnet-kernels test_is_quantized_kernel_available

# Test GPU capability detection
cargo test -p bitnet-kernels --features gpu test_get_gpu_info
```

### Phase 3: Fallback Reason Tracking (Week 1, Days 6-7)

**Files Modified:**
- `crates/bitnet-quantization/src/lib.rs`

**Validation:**
```bash
# Test fallback reason detection
cargo test -p bitnet-quantization test_detect_fallback_reason
```

### Phase 4: Receipt Validation (Week 3, Days 18-20)

**Files Modified:**
- `xtask/src/main.rs`

**Validation:**
```bash
# Test receipt validation for fallback detection
cargo test -p xtask test_verify_quantization_claims_fallback
```

## Consequences

### Positive

1. **Comprehensive Detection:** Multi-layer approach catches all fallback scenarios
2. **Fast Feedback:** Compile-time checks catch feature flag issues early
3. **Precise Errors:** Detailed fallback reasons aid debugging
4. **Low Overhead:** <1% runtime overhead for kernel availability queries

### Negative

1. **Implementation Complexity:** Multiple detection layers require coordination
2. **Test Overhead:** Each layer requires dedicated test coverage
3. **Maintenance:** Fallback reason enum requires updates for new scenarios

### Mitigation

- **Complexity:** Clear separation of concerns (compile-time, runtime, post-inference)
- **Test Overhead:** TDD approach with `// AC:ID` tags ensures coverage
- **Maintenance:** Comprehensive error messages reduce debugging overhead

## Success Metrics

- ✅ Compile-time: 100% detection of missing feature flags
- ✅ Runtime: 100% detection of device mismatches and unavailable kernels
- ✅ Post-inference: 100% accuracy in receipt validation
- ✅ Performance: <1% overhead for runtime detection
- ✅ Usability: Detailed error messages for all fallback scenarios

## Validation Commands

```bash
# Compile-time detection
cargo build --no-default-features
# Expected: Feature flag error

# Runtime detection
BITNET_STRICT_MODE=1 \
cargo test -p bitnet-inference test_ac3_strict_mode_rejects_fallback

# Receipt validation
cargo test -p xtask test_verify_quantization_claims_fallback
cargo run -p xtask -- verify-receipt ci/inference.json
```

## Related ADRs

- **ADR-010:** Three-Tier Validation Strategy
- **ADR-011:** Receipt Schema Backward Compatibility
- **ADR-012:** Kernel ID Naming Conventions

## References

- **Issue #453:** Strict Quantization Guards
- **Issue #439:** GPU Feature-Gate Hardening
- **PR #452:** Receipt Verification Infrastructure
