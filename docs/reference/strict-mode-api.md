# Strict Mode API Contracts

**Document Version:** 1.0.0
**Target BitNet-rs Version:** 0.1.0+
**Related Issue:** #453
**Type:** Reference (Diátaxis)
**Audience:** BitNet-rs developers implementing strict mode validation

---

## Overview

This document defines the API contracts for strict mode quantization guards in BitNet-rs. It provides precise Rust type signatures, environment variable contracts, receipt schema extensions, and kernel ID naming conventions required for implementation.

---

## Table of Contents

1. [StrictModeConfig API](#strictmodeconfig-api)
2. [BitNetError::StrictMode](#bitneterrorstrictmode)
3. [Receipt Schema v1.1.0](#receipt-schema-v110)
4. [Kernel ID Naming Conventions](#kernel-id-naming-conventions)
5. [Environment Variables](#environment-variables)
6. [Kernel Availability Query API](#kernel-availability-query-api)
7. [Validation Function Contracts](#validation-function-contracts)

---

## StrictModeConfig API

### Type Definition

**Location:** `crates/bitnet-common/src/strict_mode.rs`

```rust
/// Strict mode configuration for BitNet-rs inference
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrictModeConfig {
    /// Enable all strict mode checks
    pub enabled: bool,

    /// Fail on mock computation detection (Issue #261)
    pub fail_on_mock: bool,

    /// Require quantization (no FP32 fallback allowed)
    pub require_quantization: bool,

    /// Enforce quantized inference (NEW: Issue #453)
    /// Rejects FP32 fallback in quantized layers and attention projections
    pub enforce_quantized_inference: bool,

    /// Validate performance metrics against baselines
    pub validate_performance: bool,

    /// CI-enhanced mode (fail-fast, verbose logging)
    pub ci_enhanced_mode: bool,

    /// Log all validation checks (debugging)
    pub log_all_validations: bool,

    /// Fail immediately on any mock detection (no accumulation)
    pub fail_fast_on_any_mock: bool,
}
```

### Constructor Methods

```rust
impl StrictModeConfig {
    /// Create configuration from environment variables (simple)
    ///
    /// Reads `BITNET_STRICT_MODE` environment variable:
    /// - `"1"` or `"true"`: Enables all strict mode checks
    /// - Other values: Disabled
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_common::strict_mode::StrictModeConfig;
    ///
    /// std::env::set_var("BITNET_STRICT_MODE", "1");
    /// let config = StrictModeConfig::from_env();
    /// assert!(config.enabled);
    /// assert!(config.enforce_quantized_inference);
    /// ```
    pub fn from_env() -> Self;

    /// Create detailed configuration from environment variables (granular)
    ///
    /// Reads multiple environment variables for fine-grained control:
    /// - `BITNET_STRICT_MODE`: Master switch (enables all if `"1"`)
    /// - `BITNET_STRICT_FAIL_ON_MOCK`: Override for mock detection
    /// - `BITNET_STRICT_REQUIRE_QUANTIZATION`: Override for quantization requirement
    /// - `BITNET_STRICT_VALIDATE_PERFORMANCE`: Override for performance validation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_common::strict_mode::StrictModeConfig;
    ///
    /// // Enable only quantization enforcement
    /// std::env::set_var("BITNET_STRICT_MODE", "0");
    /// std::env::set_var("BITNET_STRICT_REQUIRE_QUANTIZATION", "1");
    ///
    /// let config = StrictModeConfig::from_env_detailed();
    /// assert!(!config.enabled);
    /// assert!(config.enforce_quantized_inference);
    /// ```
    pub fn from_env_detailed() -> Self;

    /// Create configuration with CI enhancements
    ///
    /// Automatically enables enhanced mode when running in CI environment:
    /// - Checks `CI` environment variable (GitHub Actions, GitLab CI, etc.)
    /// - Checks `BITNET_CI_ENHANCED_STRICT` for explicit opt-in
    /// - Enables fail-fast and verbose logging
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_common::strict_mode::StrictModeConfig;
    ///
    /// std::env::set_var("CI", "true");
    /// std::env::set_var("BITNET_CI_ENHANCED_STRICT", "1");
    ///
    /// let config = StrictModeConfig::from_env_with_ci_enhancements();
    /// assert!(config.ci_enhanced_mode);
    /// assert!(config.log_all_validations);
    /// ```
    pub fn from_env_with_ci_enhancements() -> Self;
}
```

### Validation Methods

```rust
impl StrictModeConfig {
    /// Validate inference path for mock usage (Issue #261)
    ///
    /// # Arguments
    ///
    /// * `path` - Mock inference path metadata
    ///
    /// # Returns
    ///
    /// - `Ok(())` if validation passes
    /// - `Err(BitNetError::StrictMode)` if mock computation detected
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_common::strict_mode::{StrictModeConfig, MockInferencePath};
    ///
    /// let config = StrictModeConfig {
    ///     enabled: true,
    ///     fail_on_mock: true,
    ///     ..Default::default()
    /// };
    ///
    /// let path = MockInferencePath {
    ///     uses_mock_computation: true,
    ///     description: "Test mock path".into(),
    /// };
    ///
    /// let result = config.validate_inference_path(&path);
    /// assert!(result.is_err());
    /// ```
    pub fn validate_inference_path(&self, path: &MockInferencePath) -> Result<()>;

    /// Validate kernel availability for quantization (NEW: Issue #453)
    ///
    /// # Arguments
    ///
    /// * `scenario` - Missing kernel scenario metadata
    ///
    /// # Returns
    ///
    /// - `Ok(())` if validation passes
    /// - `Err(BitNetError::StrictMode)` if required quantization kernel unavailable
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_common::strict_mode::{StrictModeConfig, MissingKernelScenario};
    /// use bitnet_common::device::Device;
    /// use bitnet_quantization::QuantizationType;
    ///
    /// let config = StrictModeConfig {
    ///     enabled: true,
    ///     enforce_quantized_inference: true,
    ///     ..Default::default()
    /// };
    ///
    /// let scenario = MissingKernelScenario {
    ///     quantization_type: QuantizationType::I2S,
    ///     device: Device::Cuda(0),
    ///     fallback_available: true,
    /// };
    ///
    /// let result = config.validate_kernel_availability(&scenario);
    /// assert!(result.is_err());
    /// ```
    pub fn validate_kernel_availability(&self, scenario: &MissingKernelScenario) -> Result<()>;

    /// Validate quantization fallback scenario (NEW: Issue #453)
    ///
    /// # Arguments
    ///
    /// * `qtype` - Quantization type (I2S, TL1, TL2)
    /// * `device` - Target device (CPU, CUDA)
    ///
    /// # Returns
    ///
    /// - `Ok(())` if validation passes or strict mode disabled
    /// - `Err(BitNetError::StrictMode)` if FP32 fallback would occur in strict mode
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_common::strict_mode::StrictModeConfig;
    /// use bitnet_common::device::Device;
    /// use bitnet_quantization::QuantizationType;
    ///
    /// let config = StrictModeConfig {
    ///     enabled: true,
    ///     enforce_quantized_inference: true,
    ///     ..Default::default()
    /// };
    ///
    /// let result = config.validate_quantization_fallback(
    ///     QuantizationType::I2S,
    ///     Device::Cpu
    /// );
    /// // Passes if native I2S CPU kernel available, fails otherwise
    /// ```
    pub fn validate_quantization_fallback(
        &self,
        qtype: QuantizationType,
        device: Device
    ) -> Result<()>;

    /// Validate performance metrics for suspicious values (Issue #261)
    ///
    /// # Arguments
    ///
    /// * `metrics` - Performance metrics from inference
    ///
    /// # Returns
    ///
    /// - `Ok(())` if validation passes
    /// - `Err(BitNetError::StrictMode)` if suspicious metrics detected
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_common::strict_mode::{StrictModeConfig, PerformanceMetrics};
    ///
    /// let config = StrictModeConfig {
    ///     enabled: true,
    ///     validate_performance: true,
    ///     ..Default::default()
    /// };
    ///
    /// let metrics = PerformanceMetrics {
    ///     tokens_per_second: 200.0, // Suspiciously high
    ///     computation_type: ComputationType::Real,
    /// };
    ///
    /// let result = config.validate_performance_metrics(&metrics);
    /// assert!(result.is_err()); // >150 tok/s threshold
    /// ```
    pub fn validate_performance_metrics(&self, metrics: &PerformanceMetrics) -> Result<()>;
}
```

### Supporting Types

```rust
/// Mock inference path metadata (Issue #261)
#[derive(Debug, Clone)]
pub struct MockInferencePath {
    /// Whether mock computation is used
    pub uses_mock_computation: bool,

    /// Human-readable description of the inference path
    pub description: String,
}

/// Missing kernel scenario metadata (NEW: Issue #453)
#[derive(Debug, Clone)]
pub struct MissingKernelScenario {
    /// Quantization type requiring kernel
    pub quantization_type: QuantizationType,

    /// Target device
    pub device: Device,

    /// Whether FP32 fallback is available
    pub fallback_available: bool,
}

/// Performance metrics for validation (Issue #261)
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Tokens per second throughput
    pub tokens_per_second: f64,

    /// Computation type used
    pub computation_type: ComputationType,
}

/// Computation type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputationType {
    /// Real computation (quantized or FP32)
    Real,

    /// Mock computation (testing only)
    Mock,
}
```

---

## BitNetError::StrictMode

### Type Definition

**Location:** `crates/bitnet-common/src/error.rs`

```rust
use thiserror::Error;

/// BitNet-rs error types
#[derive(Debug, Error)]
pub enum BitNetError {
    /// Strict mode validation failure (Issue #261, #453)
    ///
    /// This error is returned when strict mode enforcement detects:
    /// - Mock computation in production inference
    /// - FP32 fallback in quantized layers
    /// - Missing required quantization kernels
    /// - Suspicious performance metrics
    #[error("Strict mode validation failed: {0}")]
    StrictMode(String),

    // ... other variants
}
```

### Error Message Format

**Quantization Fallback Error (AC3, AC4):**
```
Strict mode: FP32 fallback rejected - qtype=I2S, device=Cuda(0),
layer_dims=[2048, 2048], reason=kernel_unavailable
```

**Mock Computation Error (Issue #261):**
```
Strict mode: Mock computation detected in inference path: <description>
```

**Performance Validation Error (Issue #261):**
```
Strict mode: Suspicious performance detected: 200.50 tok/s
```

### Error Construction Examples

```rust
use bitnet_common::error::BitNetError;
use bitnet_common::device::Device;
use bitnet_quantization::QuantizationType;

/// Construct quantization fallback error
fn quantization_fallback_error(
    qtype: QuantizationType,
    device: Device,
    layer_dims: (usize, usize),
    reason: &str
) -> BitNetError {
    BitNetError::StrictMode(format!(
        "FP32 fallback rejected - qtype={:?}, device={:?}, layer_dims=[{}, {}], reason={}",
        qtype, device, layer_dims.0, layer_dims.1, reason
    ))
}

/// Construct kernel unavailable error
fn kernel_unavailable_error(
    qtype: QuantizationType,
    device: Device,
    kernel_name: &str
) -> BitNetError {
    BitNetError::StrictMode(format!(
        "Native {} kernel unavailable on {:?} - kernel_name={}",
        qtype, device, kernel_name
    ))
}

/// Construct attention projection error
fn attention_projection_error(
    projection_name: &str,
    qtype: QuantizationType,
    device: Device
) -> BitNetError {
    BitNetError::StrictMode(format!(
        "{} projection would fall back to FP32 - qtype={:?}, device={:?}",
        projection_name, qtype, device
    ))
}
```

---

## Receipt Schema v1.1.0

### Schema Definition

**Location:** `xtask/src/verify_receipt.rs`

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Performance receipt (schema v1.1.0)
///
/// Backward compatible extension of v1.0.0 with quantization metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    /// Schema version: "1.0.0" or "1.1.0"
    pub schema_version: String,

    /// Timestamp (RFC3339 format)
    pub timestamp: String,

    /// Compute path: "real" (actual inference) or "mock" (testing)
    pub compute_path: String,

    /// Backend: "cpu", "cuda", or "unknown"
    pub backend: String,

    /// Deterministic inference flag
    #[serde(default)]
    pub deterministic: bool,

    /// Tokens requested for generation
    #[serde(default)]
    pub tokens_requested: usize,

    /// Tokens actually generated
    pub tokens_generated: usize,

    /// Measured tokens per second throughput
    pub tokens_per_second: f64,

    /// Kernels executed during inference (kernel IDs for traceability)
    pub kernels: Vec<String>,

    /// Kernel path: "native_quantized" or "fp32_fallback" (NEW: v1.1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_path: Option<String>,

    /// Quantization metadata (NEW: v1.1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<QuantizationMetadata>,

    /// Environment metadata
    #[serde(default)]
    pub environment: HashMap<String, String>,

    /// Model metadata
    #[serde(default)]
    pub model: HashMap<String, String>,
}

/// Quantization metadata (NEW: v1.1.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationMetadata {
    /// Quantization types used: ["I2S"], ["TL1"], ["I2S", "TL2"], etc.
    pub types_used: Vec<String>,

    /// Number of FP32 fallback operations (0 for strict mode)
    pub fallback_count: usize,

    /// Device-aware quantization selection enabled
    #[serde(default)]
    pub device_aware_selection: bool,
}
```

### Schema Version Compatibility

**v1.0.0 Reader Behavior (Backward Compatibility):**
```rust
// v1.0.0 readers ignore unknown fields (kernel_path, quantization)
let receipt: Receipt = serde_json::from_str(receipt_json)?;
// kernel_path and quantization fields are None (skipped)
```

**v1.1.0 Reader Behavior (Forward Compatibility):**
```rust
// v1.1.0 readers handle both v1.0.0 and v1.1.0 receipts
let receipt: Receipt = serde_json::from_str(receipt_json)?;

match receipt.schema_version.as_str() {
    "1.0.0" => {
        // Infer kernel_path from kernels array
        let inferred_path = infer_kernel_path(&receipt.kernels);
    }
    "1.1.0" => {
        // Use explicit kernel_path field
        let kernel_path = receipt.kernel_path.as_deref();
    }
    _ => return Err(anyhow!("Unsupported schema version")),
}
```

### JSON Examples

**v1.0.0 Receipt (Existing):**
```json
{
  "schema_version": "1.0.0",
  "timestamp": "2025-10-14T01:33:28.076791999+00:00",
  "compute_path": "real",
  "backend": "cuda",
  "deterministic": true,
  "tokens_requested": 128,
  "tokens_generated": 128,
  "tokens_per_second": 87.5,
  "kernels": ["gemm_fp16", "i2s_gpu_quantize"],
  "environment": {
    "BITNET_VERSION": "0.1.0",
    "OS": "linux-x86_64"
  },
  "model": {
    "path": "tests/models/mini.gguf"
  }
}
```

**v1.1.0 Receipt with Native Quantization:**
```json
{
  "schema_version": "1.1.0",
  "timestamp": "2025-10-14T02:15:42.123456789+00:00",
  "compute_path": "real",
  "backend": "cuda",
  "deterministic": true,
  "tokens_requested": 16,
  "tokens_generated": 16,
  "tokens_per_second": 87.5,
  "kernels": ["gemm_fp16", "i2s_gpu_quantize", "wmma_matmul"],
  "kernel_path": "native_quantized",
  "quantization": {
    "types_used": ["I2S"],
    "fallback_count": 0,
    "device_aware_selection": true
  },
  "environment": {
    "BITNET_VERSION": "0.1.0",
    "OS": "linux-x86_64"
  },
  "model": {
    "path": "tests/models/mini.gguf"
  }
}
```

**v1.1.0 Receipt with FP32 Fallback:**
```json
{
  "schema_version": "1.1.0",
  "timestamp": "2025-10-14T03:00:00.000000000+00:00",
  "compute_path": "real",
  "backend": "cuda",
  "tokens_generated": 16,
  "tokens_per_second": 35.0,
  "kernels": ["dequant_i2s", "fp32_matmul", "cuda_sync"],
  "kernel_path": "fp32_fallback",
  "quantization": {
    "types_used": [],
    "fallback_count": 16,
    "device_aware_selection": false
  }
}
```

---

## Kernel ID Naming Conventions

### Quantized Kernels (Native 1/2-bit Arithmetic)

**GPU Quantized Kernels:**
```
gemm_*          - GPU GEMM kernels (FP16/BF16/FP32 matmul with quantized weights)
wmma_*          - Tensor Core kernels (mixed precision with quantized inputs)
cuda_*          - CUDA-specific quantization operations
i2s_gpu_*       - I2S GPU quantization (pack, unpack, matmul)
tl1_gpu_*       - TL1 GPU quantization (table lookup)
tl2_gpu_*       - TL2 GPU quantization (table lookup)
```

**CPU Quantized Kernels:**
```
i2s_gemv        - I2S CPU GEMV (SIMD-optimized)
tl1_neon_*      - ARM NEON TL1 kernels (pack, matmul)
tl2_avx_*       - x86 AVX TL2 kernels (matmul, pack)
tl2_avx512_*    - x86 AVX-512 TL2 kernels (enhanced)
quantized_matmul_* - Generic quantized matmul implementations
```

### FP32 Fallback Kernels (Dequantization + FP32 Arithmetic)

**Fallback Kernel Indicators:**
```
dequant_*       - Explicit dequantization to FP32 (staging)
fp32_matmul     - Standard FP32 matrix multiplication
scalar_*        - Scalar fallback (no SIMD, no quantization)
fallback_*      - Explicit fallback naming convention
mock_*          - Mock kernels (testing only, not production)
```

### Kernel ID Pattern Matching

```rust
/// Check if kernel ID represents native quantized computation
pub fn is_quantized_kernel(kernel_id: &str) -> bool {
    const QUANTIZED_PREFIXES: &[&str] = &[
        "gemm_",
        "wmma_",
        "cuda_",
        "i2s_gpu_",
        "tl1_gpu_",
        "tl2_gpu_",
        "i2s_gemv",
        "tl1_neon_",
        "tl2_avx_",
        "tl2_avx512_",
        "quantized_matmul_",
    ];

    QUANTIZED_PREFIXES.iter().any(|prefix| kernel_id.starts_with(prefix))
}

/// Check if kernel ID indicates FP32 fallback
pub fn is_fallback_kernel(kernel_id: &str) -> bool {
    const FALLBACK_INDICATORS: &[&str] = &[
        "dequant_",
        "fp32_matmul",
        "scalar_",
        "fallback_",
        "mock_",
    ];

    FALLBACK_INDICATORS.iter().any(|indicator| kernel_id.contains(indicator))
}

/// Infer kernel path from kernels array (v1.0.0 compatibility)
pub fn infer_kernel_path(kernel_ids: &[String]) -> String {
    let has_quantized = kernel_ids.iter().any(|id| is_quantized_kernel(id));
    let has_fallback = kernel_ids.iter().any(|id| is_fallback_kernel(id));

    if has_quantized && !has_fallback {
        "native_quantized".to_string()
    } else if has_fallback {
        "fp32_fallback".to_string()
    } else {
        "unknown".to_string()
    }
}
```

### Kernel ID Examples

**Valid Native Quantization (GPU):**
- `gemm_fp16` - FP16 GEMM with quantized weights
- `i2s_gpu_quantize` - I2S GPU quantization operation
- `wmma_matmul` - Tensor Core mixed precision matmul
- `tl2_gpu_matmul` - TL2 GPU table lookup matmul

**Valid Native Quantization (CPU):**
- `i2s_gemv` - I2S CPU GEMV (SIMD)
- `tl1_neon_pack` - ARM NEON TL1 packing
- `tl2_avx_matmul` - x86 AVX TL2 matmul
- `quantized_matmul_i2s` - Generic I2S quantized matmul

**FP32 Fallback Kernels:**
- `dequant_i2s` - Dequantize I2S weights to FP32
- `fp32_matmul` - Standard FP32 matrix multiplication
- `scalar_matmul` - Scalar fallback (no SIMD)
- `fallback_gemm` - Explicit fallback GEMM

---

## Environment Variables

### Master Switch

**`BITNET_STRICT_MODE`**
- **Type:** Boolean (`0`, `1`, `true`, `false`)
- **Default:** `0` (disabled)
- **Description:** Enable all strict mode checks (master switch)
- **Effect:**
  - Sets `enabled = true`
  - Sets `fail_on_mock = true`
  - Sets `require_quantization = true`
  - Sets `enforce_quantized_inference = true`
  - Sets `validate_performance = true`

**Example:**
```bash
# Enable all strict mode checks
BITNET_STRICT_MODE=1 cargo run -p xtask -- benchmark --model model.gguf
```

### Granular Controls

**`BITNET_STRICT_FAIL_ON_MOCK`**
- **Type:** Boolean (`0`, `1`)
- **Default:** `0` (disabled, unless `BITNET_STRICT_MODE=1`)
- **Description:** Fail on mock computation detection (Issue #261)
- **Overrides:** `BITNET_STRICT_MODE` for mock detection

**`BITNET_STRICT_REQUIRE_QUANTIZATION`**
- **Type:** Boolean (`0`, `1`)
- **Default:** `0` (disabled, unless `BITNET_STRICT_MODE=1`)
- **Description:** Require quantization (no FP32 fallback allowed)
- **Overrides:** `BITNET_STRICT_MODE` for quantization requirement
- **Alias:** `enforce_quantized_inference` field in `StrictModeConfig`

**`BITNET_STRICT_VALIDATE_PERFORMANCE`**
- **Type:** Boolean (`0`, `1`)
- **Default:** `0` (disabled, unless `BITNET_STRICT_MODE=1`)
- **Description:** Validate performance metrics against baselines
- **Overrides:** `BITNET_STRICT_MODE` for performance validation

**Example:**
```bash
# Enable only quantization enforcement (granular)
BITNET_STRICT_MODE=0 \
BITNET_STRICT_REQUIRE_QUANTIZATION=1 \
cargo test -p bitnet-inference test_strict_quantization
```

### CI-Enhanced Mode

**`BITNET_CI_ENHANCED_STRICT`**
- **Type:** Boolean (`0`, `1`)
- **Default:** `0` (disabled)
- **Description:** Enable CI-enhanced strict mode (fail-fast, verbose logging)
- **Requires:** `CI` environment variable (GitHub Actions, GitLab CI, etc.)
- **Effect:**
  - Sets `ci_enhanced_mode = true`
  - Sets `log_all_validations = true`
  - Sets `fail_fast_on_any_mock = true`

**Example:**
```bash
# CI workflow (GitHub Actions)
- name: Run strict mode tests
  env:
    CI: true
    BITNET_CI_ENHANCED_STRICT: 1
  run: cargo test --workspace --no-default-features --features cpu
```

### Testing/Debugging Variables

**`BITNET_FORCE_QUANTIZATION_FALLBACK`**
- **Type:** Boolean (`0`, `1`)
- **Default:** `0` (disabled)
- **Description:** Force FP32 fallback for testing strict mode detection
- **Usage:** Testing only (not production)

**`BITNET_TRACK_KERNEL_IDS`**
- **Type:** Boolean (`0`, `1`)
- **Default:** `0` (disabled)
- **Description:** Track kernel IDs for validation (testing/debugging)
- **Usage:** Testing only (performance overhead)

**Example:**
```bash
# Test strict mode fallback detection
BITNET_STRICT_MODE=1 \
BITNET_FORCE_QUANTIZATION_FALLBACK=1 \
cargo test -p bitnet-inference test_ac3_strict_mode_rejects_fallback
```

### Related Variables (Existing)

**`BITNET_DETERMINISTIC`**
- **Type:** Boolean (`0`, `1`)
- **Default:** `0` (disabled)
- **Description:** Enable deterministic inference (fixed random seed)
- **Usage:** Cross-validation, reproducible testing

**`BITNET_SEED`**
- **Type:** Integer (`0`-`u64::MAX`)
- **Default:** System random
- **Description:** Random seed for deterministic inference
- **Requires:** `BITNET_DETERMINISTIC=1`

**`BITNET_GPU_FAKE`**
- **Type:** String (`cuda`, `none`)
- **Default:** Auto-detect
- **Description:** Override GPU detection for deterministic testing (Issue #439)
- **Usage:** CI reproducibility

**Example (Deterministic Testing):**
```bash
BITNET_STRICT_MODE=1 \
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
BITNET_GPU_FAKE=none \
cargo test -p bitnet-inference test_ac5_16_token_decode_cpu_strict_mode
```

---

## Kernel Availability Query API

### Type Definition

**Location:** `crates/bitnet-kernels/src/lib.rs`

```rust
use bitnet_common::device::Device;
use bitnet_quantization::QuantizationType;

/// Query whether native quantized kernel is available
///
/// # Arguments
///
/// * `qtype` - Quantization type (I2S, TL1, TL2)
/// * `device` - Target device (CPU, CUDA)
/// * `dims` - Layer dimensions (in_features, out_features)
///
/// # Returns
///
/// - `true` if native quantized kernel is available
/// - `false` if FP32 fallback would occur
///
/// # Examples
///
/// ```rust
/// use bitnet_kernels::is_quantized_kernel_available;
/// use bitnet_common::device::Device;
/// use bitnet_quantization::QuantizationType;
///
/// // Check I2S GPU kernel availability
/// let available = is_quantized_kernel_available(
///     QuantizationType::I2S,
///     Device::Cuda(0),
///     (2048, 2048)
/// );
///
/// if available {
///     println!("Native I2S GPU kernel available");
/// } else {
///     println!("Would fall back to FP32 dequantization");
/// }
/// ```
pub fn is_quantized_kernel_available(
    qtype: QuantizationType,
    device: Device,
    dims: (usize, usize)
) -> bool;

/// Get kernel ID for quantized matmul operation
///
/// # Arguments
///
/// * `qtype` - Quantization type
/// * `device` - Target device
///
/// # Returns
///
/// Kernel ID string for recording in receipts
///
/// # Examples
///
/// ```rust
/// use bitnet_kernels::get_quantized_kernel_id;
/// use bitnet_common::device::Device;
/// use bitnet_quantization::QuantizationType;
///
/// let kernel_id = get_quantized_kernel_id(
///     QuantizationType::I2S,
///     Device::Cuda(0)
/// );
///
/// assert!(kernel_id.starts_with("i2s_gpu_"));
/// ```
pub fn get_quantized_kernel_id(
    qtype: QuantizationType,
    device: Device
) -> String;

/// Check if device supports quantization type
///
/// # Arguments
///
/// * `qtype` - Quantization type
/// * `device` - Target device
///
/// # Returns
///
/// - `true` if device supports quantization type
/// - `false` otherwise
///
/// # Examples
///
/// ```rust
/// use bitnet_kernels::device_supports_quantization;
/// use bitnet_common::device::Device;
/// use bitnet_quantization::QuantizationType;
///
/// // TL1 requires ARM NEON
/// let supported = device_supports_quantization(
///     QuantizationType::TL1,
///     Device::Cpu
/// );
///
/// #[cfg(target_arch = "aarch64")]
/// assert!(supported);
///
/// #[cfg(not(target_arch = "aarch64"))]
/// assert!(!supported);
/// ```
pub fn device_supports_quantization(
    qtype: QuantizationType,
    device: Device
) -> bool;
```

---

## Validation Function Contracts

### Receipt Validation

**Location:** `xtask/src/main.rs`

```rust
use anyhow::{bail, ensure, Result};

/// Verify quantization claims in receipt (AC6)
///
/// # Arguments
///
/// * `receipt` - Performance receipt to validate
///
/// # Returns
///
/// - `Ok(())` if validation passes
/// - `Err` if quantization claims don't match kernel IDs
///
/// # Validation Rules
///
/// 1. If `kernel_path == "native_quantized"`, kernels array must contain quantized kernel IDs
/// 2. If `kernel_path == "fp32_fallback"`, compute_path cannot be "quantized"
/// 3. If `kernel_path` is None (v1.0.0), infer from kernels array and log warning
/// 4. If `quantization.fallback_count > 0`, kernel_path must be "fp32_fallback"
///
/// # Examples
///
/// ```rust
/// use xtask::verify_receipt::verify_quantization_claims;
///
/// let receipt = Receipt {
///     schema_version: "1.1.0".into(),
///     kernel_path: Some("native_quantized".into()),
///     kernels: vec!["gemm_fp16".into(), "i2s_gpu_quantize".into()],
///     // ... other fields
/// };
///
/// let result = verify_quantization_claims(&receipt);
/// assert!(result.is_ok());
/// ```
pub fn verify_quantization_claims(receipt: &Receipt) -> Result<()> {
    // Schema v1.1.0: explicit kernel_path field
    if let Some(kernel_path) = &receipt.kernel_path {
        match kernel_path.as_str() {
            "native_quantized" => {
                // Verify kernels array contains quantized kernel IDs
                ensure!(
                    receipt.kernels.iter().any(|id| is_quantized_kernel(id)),
                    "kernel_path='native_quantized' requires quantized kernel IDs in kernels array"
                );
            }
            "fp32_fallback" => {
                // Validate that compute_path reflects fallback
                ensure!(
                    receipt.compute_path != "quantized",
                    "kernel_path='fp32_fallback' cannot claim compute_path='quantized'"
                );
            }
            _ => {
                bail!("Invalid kernel_path: {}", kernel_path);
            }
        }
    } else {
        // Schema v1.0.0: infer from kernels array
        let has_quantized = receipt.kernels.iter().any(|id| is_quantized_kernel(id));
        let has_fallback = receipt.kernels.iter().any(|id| is_fallback_kernel(id));

        if has_fallback && !has_quantized {
            log::warn!("Receipt uses FP32 fallback kernels without native quantized kernels");
        }
    }

    // Validate quantization section (v1.1.0 only)
    if let Some(quant) = &receipt.quantization {
        ensure!(
            quant.fallback_count == 0 || receipt.kernel_path == Some("fp32_fallback".into()),
            "Non-zero fallback_count requires kernel_path='fp32_fallback'"
        );
    }

    Ok(())
}
```

### Layer Validation

**Location:** `crates/bitnet-inference/src/layers/quantized_linear.rs`

```rust
use bitnet_common::strict_mode::StrictModeEnforcer;
use bitnet_common::error::BitNetError;

impl QuantizedLinear {
    /// Validate quantized path availability before forward pass (AC1, AC3)
    ///
    /// # Arguments
    ///
    /// * `qtype` - Quantization type
    /// * `device` - Target device
    ///
    /// # Returns
    ///
    /// - `Ok(())` if native quantized kernel available or strict mode disabled
    /// - `Err(BitNetError::StrictMode)` if FP32 fallback would occur in strict mode
    ///
    /// # Panics
    ///
    /// Panics in debug builds if FP32 fallback would occur (AC1)
    fn validate_quantized_path(
        &self,
        qtype: QuantizationType,
        device: Device
    ) -> Result<()> {
        let has_native_kernel = bitnet_kernels::is_quantized_kernel_available(
            qtype,
            device,
            (self.in_features, self.out_features)
        );

        if !has_native_kernel {
            #[cfg(debug_assertions)]
            panic!(
                "fallback to FP32 in debug mode: layer={}, qtype={:?}, device={:?}",
                self.name, qtype, device
            );

            let strict_mode = StrictModeEnforcer::new();
            if strict_mode.get_config().enforce_quantized_inference {
                return Err(BitNetError::StrictMode(format!(
                    "FP32 fallback rejected - qtype={:?}, device={:?}, layer_dims=[{}, {}], reason=kernel_unavailable",
                    qtype, device, self.in_features, self.out_features
                )));
            }
        }

        Ok(())
    }
}
```

**Location:** `crates/bitnet-inference/src/layers/attention.rs`

```rust
use bitnet_common::strict_mode::StrictModeEnforcer;
use bitnet_common::error::BitNetError;

impl BitNetAttention {
    /// Validate all projections use quantized kernels (AC2, AC4)
    ///
    /// # Returns
    ///
    /// - `Ok(())` if all projections have native quantized kernels or strict mode disabled
    /// - `Err(BitNetError::StrictMode)` if any projection would fall back to FP32
    ///
    /// # Panics
    ///
    /// Panics in debug builds if any projection would fall back (AC2)
    fn validate_projections_quantized(&self) -> Result<()> {
        let projections = [
            ("Q", &self.q_proj),
            ("K", &self.k_proj),
            ("V", &self.v_proj),
            ("O", &self.o_proj),
        ];

        for (name, proj) in &projections {
            let has_native_kernel = proj.has_native_quantized_kernel();

            if !has_native_kernel {
                #[cfg(debug_assertions)]
                panic!(
                    "fallback to FP32 in debug mode: {} projection would fall back",
                    name
                );

                let strict_mode = StrictModeEnforcer::new();
                if strict_mode.get_config().enforce_quantized_inference {
                    return Err(BitNetError::StrictMode(format!(
                        "{} projection would fall back to FP32 - qtype={:?}, device={:?}",
                        name, proj.quantization_type, proj.device
                    )));
                }
            }
        }

        Ok(())
    }
}
```

---

## Document Status

**Status:** Approved - Ready for Implementation
**Next Steps:** Implementation team uses these contracts for development

---

## Related Documentation

- **Feature Specification:** `docs/explanation/strict-quantization-guards.md`
- **ADR-010:** Three-Tier Validation Strategy
- **ADR-011:** Receipt Schema Backward Compatibility
- **ADR-012:** Kernel ID Naming Conventions
- **ADR-013:** FP32 Fallback Detection Mechanisms
- **Receipt Validation:** `docs/explanation/receipt-validation.md`
- **Quantization Support:** `docs/reference/quantization-support.md`
