# Quantization Device-Aware Patterns (Issue #439 Neural Network Context)

## Purpose

Demonstrate BitNet.rs neural network patterns for device-aware quantization backend selection using unified GPU feature gates.

## Fixture Files

### I2S Quantization

1. **`i2s_device_selection.rs`**
   - Device-aware I2S quantization backend selection
   - Uses `gpu_compiled()` and `gpu_available_runtime()` from AC3 device_features
   - Automatic GPU/CPU fallback
   - Unified feature gate: `#[cfg(any(feature = "gpu", feature = "cuda"))]`
   - Usage: Reference pattern for I2S quantization integration

### TL Quantization

2. **`tl_device_selection.rs`**
   - Device-aware TL1/TL2 quantization backend selection
   - Architecture-specific SIMD: ARM NEON (TL1), x86 AVX2/AVX-512 (TL2)
   - GPU acceleration when available
   - Runtime feature detection (`is_x86_feature_detected!`)
   - Usage: Reference pattern for architecture + device selection

### Mixed Precision

3. **`mixed_precision_selection.rs`**
   - GPU precision mode selection (FP32, FP16, BF16, Mixed)
   - Device capability detection (Tensor Cores, compute capability)
   - Automatic precision fallback for legacy GPUs
   - Demonstrates mixed precision GEMM kernels
   - Usage: Reference pattern for GPU acceleration with mixed precision

## Testing Usage

### Load Fixture Patterns
```rust
// I2S device selection
mod i2s_fixture {
    include!("tests/fixtures/quantization/i2s_device_selection.rs");
}

#[test]
fn test_i2s_backend_selection() {
    let backend = i2s_fixture::select_i2s_backend();
    println!("I2S backend: {}", backend);

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    assert!(backend == "i2s_gpu" || backend == "i2s_cpu");

    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    assert_eq!(backend, "i2s_cpu");
}
```

### Test Device-Aware Quantization
```rust
#[test]
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn test_gpu_quantization_path() {
    use std::env;

    // Force GPU path
    env::set_var("BITNET_GPU_FAKE", "cuda");

    let backend = i2s_fixture::select_i2s_backend();
    assert_eq!(backend, "i2s_gpu");

    env::remove_var("BITNET_GPU_FAKE");
}

#[test]
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn test_cpu_fallback_path() {
    use std::env;

    // Force CPU fallback
    env::set_var("BITNET_GPU_FAKE", "none");

    let backend = i2s_fixture::select_i2s_backend();
    assert_eq!(backend, "i2s_cpu");

    env::remove_var("BITNET_GPU_FAKE");
}
```

## Integration with Tests

These fixtures demonstrate patterns for:
- `crates/bitnet-quantization/src/` (Quantization backend selection)
- `crates/bitnet-kernels/src/` (Device-aware kernel dispatch)
- `crates/bitnet-inference/src/` (Inference engine device routing)

## Specification Reference

- **Issue**: #439 GPU feature-gate hardening
- **Acceptance Criteria**: AC3 - Shared helpers (device features)
- **Specification**: `docs/explanation/issue-439-spec.md#neural-network-context`

## Device Selection Patterns

### Compile-Time + Runtime Check
```rust
use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};

pub fn select_backend() -> &'static str {
    if gpu_compiled() && gpu_available_runtime() {
        "gpu"
    } else {
        "cpu"
    }
}
```

### Architecture-Specific SIMD
```rust
#[cfg(target_arch = "x86_64")]
fn select_cpu_backend() -> &'static str {
    if is_x86_feature_detected!("avx512f") {
        "cpu_avx512"
    } else if is_x86_feature_detected!("avx2") {
        "cpu_avx2"
    } else {
        "cpu_generic"
    }
}

#[cfg(target_arch = "aarch64")]
fn select_cpu_backend() -> &'static str {
    "cpu_neon"
}
```

### GPU Capability-Based Selection
```rust
pub fn select_precision(device_id: usize) -> PrecisionMode {
    let caps = query_device_capabilities(device_id);

    if caps.supports_bf16 && caps.supports_tensor_cores {
        PrecisionMode::MixedBF16
    } else if caps.supports_fp16 {
        PrecisionMode::FP16
    } else {
        PrecisionMode::FP32
    }
}
```

## Quantization Algorithm Matrix

| Algorithm | CPU Arch | GPU Support | Mixed Precision | Accuracy Target |
|-----------|----------|-------------|-----------------|-----------------|
| I2S | x86_64 (AVX2/512), aarch64 (NEON) | ✓ CUDA | ✓ FP16/BF16 | ≥99.8% |
| TL1 | aarch64 (NEON optimized) | ✓ CUDA | ✓ FP16/BF16 | ≥99.6% |
| TL2 | x86_64 (AVX2/512 optimized) | ✓ CUDA | ✓ FP16/BF16 | ≥99.6% |

## Feature Gate Patterns

### Unified GPU Predicate (CORRECT)
```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn gpu_quantize() {
    // GPU implementation
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn gpu_quantize() {
    panic!("GPU not compiled");
}
```

### Runtime Feature Detection
```rust
pub fn has_gpu_support() -> bool {
    cfg!(any(feature = "gpu", feature = "cuda"))
}
```

## Common Patterns

### Automatic Fallback
```rust
pub fn quantize_auto(input: &[f32]) -> Vec<i8> {
    if gpu_compiled() && gpu_available_runtime() {
        quantize_gpu(input)
    } else {
        quantize_cpu(input)
    }
}
```

### Device Preference List
```rust
pub fn select_backend_with_preferences() -> &'static str {
    // 1. Try GPU with mixed precision
    if gpu_compiled() && gpu_available_runtime() {
        return "gpu_mixed_bf16";
    }

    // 2. Fallback to CPU with best SIMD
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        return "cpu_avx512";
    }

    // 3. Generic CPU fallback
    "cpu_generic"
}
```

## Validation Checklist

Device-aware quantization patterns should:
- [ ] Use unified GPU predicate: `#[cfg(any(feature = "gpu", feature = "cuda"))]`
- [ ] Check compile-time: `gpu_compiled()`
- [ ] Check runtime: `gpu_available_runtime()`
- [ ] Respect `BITNET_GPU_FAKE` environment variable
- [ ] Provide CPU fallback when GPU unavailable
- [ ] Use architecture-specific SIMD when possible
- [ ] Support mixed precision on capable GPUs

## Example Validation Test

```rust
#[test]
fn test_device_aware_quantization() {
    use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};

    let backend = select_i2s_backend();

    if gpu_compiled() {
        // GPU compiled - backend depends on runtime
        if gpu_available_runtime() {
            assert_eq!(backend, "i2s_gpu");
        } else {
            assert_eq!(backend, "i2s_cpu");
        }
    } else {
        // GPU not compiled - must use CPU
        assert_eq!(backend, "i2s_cpu");
    }
}
```
