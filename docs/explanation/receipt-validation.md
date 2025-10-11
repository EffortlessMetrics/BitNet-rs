# Receipt Validation and GPU Honesty Checking

**Status**: Draft
**Related Issue**: #439 (GPU feature-gate hardening)
**Audience**: BitNet.rs developers implementing performance validation
**Type**: Explanation (Diátaxis)

## Overview

Receipt validation ensures that performance receipts accurately reflect actual device usage. This prevents "GPU on paper, CPU in reality" scenarios where performance baselines become misleading due to silent CPU fallback.

## Problem Statement

### Silent CPU Fallback Scenario

**Setup**:
- User builds with `cargo build --features gpu`
- CUDA runtime unavailable at runtime (no GPU, driver issue, etc.)
- Inference silently falls back to CPU without error

**Result**:
```json
{
  "backend": "cuda",
  "kernels": ["i2s_cpu_quantize", "avx2_matmul"],
  "tokens_per_second": 12.3,
  "latency_ms": 81.2
}
```

**Problem**: Receipt claims GPU backend but kernels are CPU implementations → misleading performance baseline → regression detection fails.

**Impact**:
- Performance regressions masked (12 tok/s looks normal for "GPU")
- Cross-validation false positives (comparing CPU perf to GPU baseline)
- User confusion (expecting 50-100 tok/s, getting 10-20 tok/s)

## Architecture Decision

### Receipt Structure

**File**: `xtask/src/verify_receipt.rs`

```rust
/// Performance receipt structure (matches bitnet-inference Receipt)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    /// Backend used: "cpu", "cuda", or "unknown"
    pub backend: String,

    /// Kernels executed during inference (kernel IDs for traceability)
    pub kernels: Vec<String>,

    /// Performance metrics
    pub tokens_per_second: Option<f64>,
    pub latency_ms: Option<f64>,
}
```

### GPU Kernel Naming Convention

**Design Principle**: Kernel IDs must be self-documenting for automated validation.

**Naming Patterns**:

| Kernel Type | Prefix | Examples |
|-------------|--------|----------|
| GEMM kernels | `gemm_` | `gemm_fp16`, `gemm_bf16`, `gemm_i2s` |
| Tensor Core kernels | `wmma_` | `wmma_matmul`, `wmma_quantize` |
| CUDA utilities | `cuda_` | `cuda_memcpy`, `cuda_sync` |
| I2_S GPU quantization | `i2s_gpu_` | `i2s_gpu_quantize`, `i2s_gpu_dequantize` |
| TL1 GPU quantization | `tl1_gpu_` | `tl1_gpu_pack`, `tl1_gpu_unpack` |
| TL2 GPU quantization | `tl2_gpu_` | `tl2_gpu_matmul`, `tl2_gpu_transform` |

**Counter-Examples (CPU kernels)**:
- `i2s_cpu_quantize` - I2_S CPU implementation
- `avx2_matmul` - AVX2 SIMD matrix multiplication
- `fallback_*` - Generic CPU fallback kernels

### Validation Rules (AC6)

**Rule 1: GPU Backend Requires GPU Kernel Evidence**

If `receipt.backend == "cuda"` OR `receipt.backend == "gpu"`:
- `receipt.kernels` array MUST NOT be empty
- `receipt.kernels` MUST contain at least one kernel matching GPU naming convention
- Validation FAILS otherwise with actionable error message

**Rule 2: CPU Backend Requires No Validation**

If `receipt.backend == "cpu"`:
- No kernel validation required
- GPU kernels in kernels array trigger warning (not failure)
- Allows mixed CPU/GPU fallback scenarios

**Rule 3: Unknown Backend Skips Validation**

If `receipt.backend` is not recognized:
- Skip validation (forward compatibility)
- Log warning for diagnostic purposes

## Public API Specification

### Core Validation Function

```rust
/// Verify GPU receipt honesty (AC6)
///
/// Ensures that if a receipt claims GPU backend execution, it contains
/// evidence of actual GPU kernel usage via naming convention.
///
/// # Validation Rules
///
/// - If `backend == "cuda"` OR `backend == "gpu"`:
///   - `kernels` array MUST contain at least one GPU kernel
///   - GPU kernels identified by prefix matching (see GPU_KERNEL_PREFIXES)
///   - Empty `kernels` array is a validation failure
///
/// - If `backend == "cpu"`:
///   - GPU kernels in `kernels` array trigger a warning but not failure
///   - This allows mixed CPU/GPU fallback scenarios
///
/// # Examples
///
/// ```rust
/// // VALID: GPU backend with GPU kernel evidence
/// let receipt = Receipt {
///     backend: "cuda".to_string(),
///     kernels: vec!["i2s_gpu_quantize".to_string(), "gemm_fp16".to_string()],
///     tokens_per_second: Some(87.5),
///     latency_ms: Some(11.4),
/// };
/// verify_gpu_receipt(&receipt)?; // ✓ PASS
///
/// // INVALID: GPU backend without GPU kernel evidence
/// let bad_receipt = Receipt {
///     backend: "cuda".to_string(),
///     kernels: vec!["i2s_cpu_quantize".to_string()], // CPU kernel!
///     tokens_per_second: Some(12.3),
///     latency_ms: Some(81.2),
/// };
/// verify_gpu_receipt(&bad_receipt)?; // ✗ FAIL
/// ```
pub fn verify_gpu_receipt(receipt: &Receipt) -> Result<()> {
    let backend_claims_gpu = receipt.backend == "cuda" || receipt.backend == "gpu";

    if !backend_claims_gpu {
        // CPU backend - no validation needed
        return Ok(());
    }

    // GPU backend claimed - verify kernel evidence
    ensure!(
        !receipt.kernels.is_empty(),
        "GPU backend '{}' requires non-empty kernels array, got: {:?}",
        receipt.backend,
        receipt.kernels
    );

    let has_gpu_kernel = receipt.kernels.iter().any(|kernel_id| {
        GPU_KERNEL_PREFIXES.iter().any(|prefix| kernel_id.starts_with(prefix))
    });

    ensure!(
        has_gpu_kernel,
        "GPU backend '{}' requires at least one GPU kernel matching naming convention.\n\
         Expected kernel prefixes: {}\n\
         Actual kernels: {:?}\n\n\
         This likely indicates silent CPU fallback. Verify:\n\
         1. GPU feature compiled: cargo build --features gpu\n\
         2. CUDA runtime available: nvidia-smi\n\
         3. Device selection: Device::Cuda(0) passed to inference",
        receipt.backend,
        GPU_KERNEL_PREFIXES.join(", "),
        receipt.kernels
    );

    Ok(())
}
```

### File-Based Validation

```rust
/// Verify receipt file from path (xtask integration)
pub fn verify_receipt_file(path: &std::path::Path) -> Result<()> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read receipt file: {}", path.display()))?;

    let receipt: Receipt = serde_json::from_str(&contents)
        .with_context(|| format!("Failed to parse receipt JSON: {}", path.display()))?;

    verify_gpu_receipt(&receipt)
        .with_context(|| format!("Receipt validation failed: {}", path.display()))?;

    println!("✓ Receipt validation passed: {}", path.display());
    println!("  Backend: {}", receipt.backend);
    println!("  Kernels: {:?}", receipt.kernels);

    Ok(())
}
```

## Integration Points

### Inference Engine Receipt Generation

**File**: `crates/bitnet-inference/src/engine.rs`

```rust
pub struct InferenceEngine {
    device: Device,
    kernel_usage: Vec<String>, // Track kernels for receipt
}

impl InferenceEngine {
    /// Record kernel usage during inference
    fn record_kernel(&mut self, kernel_id: &str) {
        self.kernel_usage.push(kernel_id.to_string());
    }

    pub fn generate(&mut self, prompt: &str) -> Result<(String, Receipt)> {
        // Clear kernel tracking for new generation
        self.kernel_usage.clear();

        // Inference logic records kernels via self.record_kernel()
        // Example: self.record_kernel("i2s_gpu_quantize");

        let backend = match self.device {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            _ => "unknown",
        };

        let receipt = Receipt {
            backend: backend.to_string(),
            kernels: self.kernel_usage.clone(),
            tokens_per_second: Some(self.calculate_tps()),
            latency_ms: Some(self.total_latency()),
        };

        // Verify receipt honesty before returning
        verify_gpu_receipt(&receipt)?;

        Ok((output, receipt))
    }
}
```

### Quantization Kernel Recording

**File**: `crates/bitnet-quantization/src/i2s.rs`

```rust
impl I2SQuantizer {
    pub fn quantize(&self, input: &[f32], device: &Device) -> Result<Vec<i8>> {
        match device {
            Device::Cpu => {
                // Record CPU kernel usage
                if let Some(recorder) = &self.kernel_recorder {
                    recorder.record("i2s_cpu_quantize");
                }
                self.quantize_cpu(input)
            }
            Device::Cuda(gpu_id) => {
                // Record GPU kernel usage
                if let Some(recorder) = &self.kernel_recorder {
                    recorder.record("i2s_gpu_quantize");
                }
                self.quantize_cuda(input, *gpu_id)
            }
            _ => Err(QuantizationError::UnsupportedDevice),
        }
    }
}
```

### xtask verify-receipt Command

**File**: `xtask/src/main.rs`

```rust
#[derive(Parser)]
enum Command {
    /// Verify GPU receipt honesty
    VerifyReceipt {
        /// Path to receipt JSON file
        #[arg(value_name = "PATH")]
        receipt_path: PathBuf,
    },
    // ... other commands
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::VerifyReceipt { receipt_path } => {
            verify_receipt_file(&receipt_path)?;
        }
        // ... other command handlers
    }

    Ok(())
}
```

## Performance Baseline Validation

### Baseline Thresholds (from Issue #261)

```rust
/// Performance validation for receipt honesty
pub fn validate_performance_reasonable(receipt: &Receipt) -> Result<()> {
    if let Some(tps) = receipt.tokens_per_second {
        match receipt.backend.as_str() {
            "cpu" => {
                // CPU baseline: 10-20 tok/s (allowing 5-30 tok/s variance)
                ensure!(
                    tps >= 5.0 && tps <= 30.0,
                    "CPU performance {:.1} tok/s outside expected range 5-30 tok/s",
                    tps
                );
            }
            "cuda" | "gpu" => {
                // GPU baseline: 50-100 tok/s (allowing 30-150 tok/s variance)
                ensure!(
                    tps >= 30.0 && tps <= 150.0,
                    "GPU performance {:.1} tok/s outside expected range 30-150 tok/s",
                    tps
                );
            }
            _ => {} // Unknown backend - skip validation
        }
    }

    Ok(())
}
```

### Silent Fallback Detection

```rust
/// Detect suspicious performance indicating silent CPU fallback
pub fn detect_silent_cpu_fallback(receipt: &Receipt) -> Option<String> {
    if receipt.backend == "cuda" || receipt.backend == "gpu" {
        if let Some(tps) = receipt.tokens_per_second {
            // GPU performance < 25 tok/s is suspicious (CPU-like)
            if tps < 25.0 {
                return Some(format!(
                    "WARNING: GPU backend but CPU-like performance ({:.1} tok/s < 25 tok/s threshold). \
                     Possible silent CPU fallback - verify:\n\
                     1. GPU kernels in receipt: {:?}\n\
                     2. CUDA available: nvidia-smi\n\
                     3. Feature compiled: cargo build --features gpu",
                    tps, receipt.kernels
                ));
            }
        }
    }
    None
}
```

## Testing Strategy

### Unit Tests (AC6)

**File**: `xtask/tests/verify_receipt.rs`

```rust
use xtask::verify_receipt::{Receipt, verify_gpu_receipt};

#[test]
fn ac6_gpu_backend_requires_gpu_kernel() {
    // AC:6 - GPU backend must have GPU kernel evidence
    let receipt = Receipt {
        backend: "cuda".to_string(),
        kernels: vec!["i2s_cpu_quantize".to_string()],
        tokens_per_second: Some(12.0),
        latency_ms: Some(80.0),
    };

    assert!(
        verify_gpu_receipt(&receipt).is_err(),
        "GPU backend with CPU kernels should fail validation"
    );
}

#[test]
fn ac6_gpu_backend_with_valid_kernel() {
    // AC:6 - GPU backend with GPU kernel should pass
    let receipt = Receipt {
        backend: "cuda".to_string(),
        kernels: vec!["gemm_fp16".to_string()],
        tokens_per_second: Some(87.5),
        latency_ms: Some(11.4),
    };

    assert!(
        verify_gpu_receipt(&receipt).is_ok(),
        "GPU backend with GPU kernel should pass validation"
    );
}

#[test]
fn ac6_cpu_backend_no_validation() {
    // AC:6 - CPU backend requires no kernel validation
    let receipt = Receipt {
        backend: "cpu".to_string(),
        kernels: vec![],
        tokens_per_second: Some(15.0),
        latency_ms: Some(66.0),
    };

    assert!(
        verify_gpu_receipt(&receipt).is_ok(),
        "CPU backend should pass validation regardless of kernels"
    );
}

#[test]
fn ac6_valid_gpu_kernel_prefixes() {
    // AC:6 - All GPU kernel prefixes should pass validation
    let valid_prefixes = vec![
        "gemm_fp16",
        "wmma_matmul",
        "cuda_sync",
        "i2s_gpu_quantize",
        "tl1_gpu_pack",
        "tl2_gpu_matmul",
    ];

    for kernel in valid_prefixes {
        let receipt = Receipt {
            backend: "cuda".to_string(),
            kernels: vec![kernel.to_string()],
            tokens_per_second: Some(87.5),
            latency_ms: Some(11.4),
        };

        assert!(
            verify_gpu_receipt(&receipt).is_ok(),
            "Kernel '{}' should pass validation (AC6)",
            kernel
        );
    }
}

#[test]
fn ac6_empty_kernels_gpu_backend_fails() {
    // AC:6 - GPU backend with empty kernels array should fail
    let receipt = Receipt {
        backend: "cuda".to_string(),
        kernels: vec![],
        tokens_per_second: Some(87.5),
        latency_ms: Some(11.4),
    };

    let result = verify_gpu_receipt(&receipt);
    assert!(result.is_err(), "GPU backend with empty kernels should fail");

    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("non-empty kernels array"),
        "Error should mention non-empty requirement"
    );
}
```

### Integration Tests

**File**: `xtask/tests/verify_receipt_integration.rs`

```rust
#[test]
fn ac6_verify_receipt_file_valid() {
    // Create temporary valid receipt file
    let temp_dir = tempfile::tempdir().unwrap();
    let receipt_path = temp_dir.path().join("valid-gpu.json");

    let receipt = Receipt {
        backend: "cuda".to_string(),
        kernels: vec!["gemm_fp16".to_string(), "i2s_gpu_quantize".to_string()],
        tokens_per_second: Some(87.5),
        latency_ms: Some(11.4),
    };

    std::fs::write(&receipt_path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();

    // Verify receipt file
    let result = verify_receipt_file(&receipt_path);
    assert!(result.is_ok(), "Valid receipt file should pass verification");
}

#[test]
fn ac6_verify_receipt_file_invalid() {
    // Create temporary invalid receipt file (GPU backend, CPU kernels)
    let temp_dir = tempfile::tempdir().unwrap();
    let receipt_path = temp_dir.path().join("invalid-gpu.json");

    let receipt = Receipt {
        backend: "cuda".to_string(),
        kernels: vec!["i2s_cpu_quantize".to_string()],
        tokens_per_second: Some(12.3),
        latency_ms: Some(81.2),
    };

    std::fs::write(&receipt_path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();

    // Verify receipt file
    let result = verify_receipt_file(&receipt_path);
    assert!(result.is_err(), "Invalid receipt file should fail verification");
}
```

## Validation Commands

### Create Test Receipts

```bash
# Create test receipt directory
mkdir -p /tmp/bitnet-receipts

# Valid GPU receipt
cat > /tmp/bitnet-receipts/valid-gpu.json <<EOF
{
  "backend": "cuda",
  "kernels": ["gemm_fp16", "i2s_gpu_quantize"],
  "tokens_per_second": 87.5,
  "latency_ms": 11.4
}
EOF

# Invalid GPU receipt (CPU kernels)
cat > /tmp/bitnet-receipts/invalid-gpu.json <<EOF
{
  "backend": "cuda",
  "kernels": ["i2s_cpu_quantize", "avx2_matmul"],
  "tokens_per_second": 12.3,
  "latency_ms": 81.2
}
EOF

# CPU receipt (valid)
cat > /tmp/bitnet-receipts/valid-cpu.json <<EOF
{
  "backend": "cpu",
  "kernels": ["i2s_cpu_quantize", "avx2_matmul"],
  "tokens_per_second": 15.2,
  "latency_ms": 65.8
}
EOF
```

### Run Validation

```bash
# Verify valid GPU receipt (should pass)
cargo run -p xtask -- verify-receipt /tmp/bitnet-receipts/valid-gpu.json
# Expected: "✓ Receipt validation passed"

# Verify invalid GPU receipt (should fail)
cargo run -p xtask -- verify-receipt /tmp/bitnet-receipts/invalid-gpu.json
# Expected: Error with "naming convention" message

# Verify CPU receipt (should pass)
cargo run -p xtask -- verify-receipt /tmp/bitnet-receipts/valid-cpu.json
# Expected: "✓ Receipt validation passed"
```

### Run Unit Tests

```bash
# Run receipt validation tests
cargo test --package xtask --test verify_receipt

# Run specific AC6 tests
cargo test --package xtask --test verify_receipt ac6_
```

## Cross-Validation Integration

### Receipt Comparison in crossval

**File**: `xtask/src/crossval.rs`

```rust
pub fn run_crossval(model_path: &Path) -> Result<()> {
    // Run Rust inference
    let (rust_output, rust_receipt) = run_rust_inference(model_path)?;

    // Run C++ reference
    let (cpp_output, cpp_receipt) = run_cpp_reference(model_path)?;

    // Verify both receipts are honest
    verify_gpu_receipt(&rust_receipt)
        .context("Rust receipt validation failed")?;
    verify_gpu_receipt(&cpp_receipt)
        .context("C++ receipt validation failed")?;

    // Compare outputs (accuracy)
    let accuracy = compare_outputs(&rust_output, &cpp_output);

    // Compare performance (should be within 2x)
    let rust_tps = rust_receipt.tokens_per_second.unwrap();
    let cpp_tps = cpp_receipt.tokens_per_second.unwrap();

    ensure!(
        (rust_tps / cpp_tps) > 0.5 && (rust_tps / cpp_tps) < 2.0,
        "Performance divergence: Rust {:.1} tok/s vs C++ {:.1} tok/s",
        rust_tps,
        cpp_tps
    );

    println!("Cross-validation PASS:");
    println!("  Accuracy: {:.2}%", accuracy * 100.0);
    println!("  Rust: {:.1} tok/s (backend: {})", rust_tps, rust_receipt.backend);
    println!("  C++:  {:.1} tok/s (backend: {})", cpp_tps, cpp_receipt.backend);

    Ok(())
}
```

## Error Messages

### GPU Backend Without GPU Kernels

```text
GPU backend 'cuda' requires at least one GPU kernel matching naming convention.
Expected kernel prefixes: gemm_, wmma_, cuda_, i2s_gpu_, tl1_gpu_, tl2_gpu_
Actual kernels: ["i2s_cpu_quantize", "avx2_matmul"]

This likely indicates silent CPU fallback. Verify:
1. GPU feature compiled: cargo build --features gpu
2. CUDA runtime available: nvidia-smi
3. Device selection: Device::Cuda(0) passed to inference
```

### Empty Kernels Array

```text
GPU backend 'cuda' requires non-empty kernels array, got: []

Ensure inference engine is recording kernel usage via record_kernel() calls.
```

### Performance Threshold Warning

```text
WARNING: GPU backend but CPU-like performance (12.3 tok/s < 25 tok/s threshold).
Possible silent CPU fallback - verify:
1. GPU kernels in receipt: ["i2s_cpu_quantize", "avx2_matmul"]
2. CUDA available: nvidia-smi
3. Feature compiled: cargo build --features gpu
```

## Related Documentation

- **Main Spec**: `docs/explanation/issue-439-spec.md` - Full feature gate hardening specification
- **Device Features**: `docs/explanation/device-feature-detection.md` - GPU capability detection API
- **GPU Architecture**: `docs/gpu-kernel-architecture.md` - CUDA kernel design patterns
- **Performance Benchmarking**: `docs/performance-benchmarking.md` - Performance testing framework

## References

- **Issue #439**: GPU feature-gate hardening (workspace-wide)
- **Issue #261**: Mock elimination, receipt-driven baselines
- **AC6**: Receipt guard requirement for GPU kernel naming convention
