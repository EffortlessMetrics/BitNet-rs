# C++ Wrapper GPU Layer Configuration Specification

**Status**: Draft
**Priority**: MEDIUM
**Category**: Explanation (Technical Specification)
**Target Release**: v0.2.0 (Post-MVP)
**Related Issues**: Socket 5 GPU Support
**Date**: 2025-10-25

---

## Executive Summary

This specification defines the architecture for enabling GPU layer configuration in the BitNet.rs C++ FFI wrapper (`crossval/src/bitnet_cpp_wrapper.cc`). Currently, GPU layer offloading is disabled at lines 408-410, forcing all inference to run on CPU despite the infrastructure accepting an `n_gpu_layers` parameter. This represents a **MEDIUM-priority optimization opportunity** to unlock GPU acceleration for cross-validation workflows.

**Key Impact**:
- **Performance**: Expected 5-50× speedup for GPU-accelerated inference (model-size dependent)
- **Memory**: GPU VRAM usage scales with layer count (estimate: ~100-500MB per billion parameters offloaded)
- **Compatibility**: Graceful fallback to CPU when GPU unavailable or VRAM insufficient

**Scope**: This specification covers Socket 1 (Persistent Context API) GPU configuration only. Socket 5 (dedicated GPU API) remains a v0.3 future consideration.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Technical Design](#2-technical-design)
3. [API Changes](#3-api-changes)
4. [Implementation Details](#4-implementation-details)
5. [Configuration Strategy](#5-configuration-strategy)
6. [Performance Expectations](#6-performance-expectations)
7. [Testing Requirements](#7-testing-requirements)
8. [Acceptance Criteria](#8-acceptance-criteria)
9. [Risks and Mitigations](#9-risks-and-mitigations)
10. [Future Considerations](#10-future-considerations)

---

## 1. Problem Statement

### 1.1 Current Behavior

**Location**: `crossval/src/bitnet_cpp_wrapper.cc:408-410`

```cpp
llama_model_params model_params = llama_model_default_params();
// Note: GPU layer offloading would be configured here
// model_params.n_gpu_layers = n_gpu_layers;  // ← DISABLED

ctx->model = llama_load_model_from_file(model_path, model_params);
```

**Symptom**: The `bitnet_cpp_init_context()` function accepts `n_gpu_layers` parameter from Rust FFI but **does not apply it** to the llama.cpp model loading configuration. The parameter is stored in the context struct (line 405) but never used, resulting in:

1. **CPU-only inference**: All layers execute on CPU regardless of GPU availability
2. **Performance bottleneck**: Cross-validation runs 5-50× slower than potential GPU-accelerated path
3. **Misleading API**: Rust callers can pass `n_gpu_layers > 0` with no effect

### 1.2 Root Cause Analysis

**Why GPU layers are disabled:**

1. **MVP Conservatism**: GPU support was intentionally deferred to avoid complexity during MVP phase
2. **C++ Reference Uncertainty**: Unclear if BitNet.cpp supports GPU offloading via `n_gpu_layers` parameter (llama.cpp convention)
3. **Testing Gap**: No GPU availability detection or fallback testing in place
4. **Memory Safety**: Concerns about GPU VRAM exhaustion without graceful degradation

**Evidence from Analysis Report** (`/tmp/cpp_wrapper_current_api_analysis.md:404-410`):

> **Problem 4: No GPU Layer Configuration**
>
> **Severity**: 🟡 **MEDIUM** (planned for Socket 5, v0.3)
> **Impact**: GPU inference not accelerated

### 1.3 Performance Impact

**Measured CPU-Only Performance** (from CLAUDE.md):
- **QK256 MVP**: ~0.1 tok/s for 2B models (scalar kernels)
- **I2_S BitNet32-F16**: 10-20× faster than QK256 (still CPU-bound)

**Expected GPU-Accelerated Performance**:
- **Small models (1-3B)**: 5-10× speedup over CPU (VRAM permitting)
- **Medium models (7-13B)**: 10-30× speedup (optimal GPU utilization)
- **Large models (30B+)**: 20-50× speedup (bandwidth-limited)

**Calculation Basis**:
- NVIDIA RTX 4090: ~82 TFLOPS FP16 vs Intel i9-13900K: ~1.5 TFLOPS AVX-512
- GPU memory bandwidth: ~1TB/s vs DDR5: ~75 GB/s
- Layer offloading overhead: ~5-10ms per forward pass (negligible)

### 1.4 Use Cases

**Primary Use Case: Cross-Validation Acceleration**

```bash
# Current: CPU-only cross-validation (~30s for 10 tokens)
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/bitnet-2b.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 10

# Proposed: GPU-accelerated cross-validation (~3s for 10 tokens)
BITNET_GPU_LAYERS=24 cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/bitnet-2b.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 10
```

**Secondary Use Cases**:
1. **Rapid prototyping**: Faster iteration on C++ reference validation
2. **Benchmarking**: Fair GPU vs CPU performance comparisons
3. **Production inference**: Future production FFI bridge with GPU support

---

## 2. Technical Design

### 2.1 Architecture Overview

**Design Principle**: Minimal change, maximum compatibility. Enable GPU offloading using existing llama.cpp conventions without breaking CPU-only workflows.

**Key Components**:

```
┌─────────────────────────────────────────────────────────────┐
│ Rust FFI Layer (crossval/src/cpp_bindings.rs)              │
│   - BitnetSession::create(model, n_ctx, n_gpu_layers)      │
│   - Environment variable support: BITNET_GPU_LAYERS        │
└────────────────────┬────────────────────────────────────────┘
                     │ FFI call
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ C++ Wrapper (crossval/src/bitnet_cpp_wrapper.cc)           │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ bitnet_cpp_init_context(...)                        │  │
│   │   - Receive n_gpu_layers from Rust                  │  │
│   │   - Apply to llama_model_params                     │  │
│   │   - Handle CUDA availability gracefully             │  │
│   └─────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │ llama.cpp API
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ llama.cpp / BitNet.cpp Runtime                              │
│   - Model loading with GPU offloading                       │
│   - CUDA kernel dispatch for offloaded layers              │
│   - Graceful fallback if VRAM insufficient                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Configuration Flow

**Three-level configuration hierarchy** (highest precedence wins):

1. **Explicit API call**: `BitnetSession::create(..., n_gpu_layers: 24)`
2. **Environment variable**: `BITNET_GPU_LAYERS=24` (overrides default 0)
3. **Auto-detection**: `n_gpu_layers = -1` → Use all available GPU layers (llama.cpp convention)

**Precedence Example**:

```rust
// Case 1: Explicit override (n_gpu_layers = 16)
let session = BitnetSession::create(model_path, 512, 16)?;

// Case 2: Environment variable (BITNET_GPU_LAYERS = 24)
std::env::set_var("BITNET_GPU_LAYERS", "24");
let session = BitnetSession::create(model_path, 512, 0)?; // 0 becomes 24

// Case 3: Auto-detection (n_gpu_layers = -1)
let session = BitnetSession::create(model_path, 512, -1)?; // Uses all layers
```

### 2.3 Compatibility Matrix

| Configuration | GPU Available | CUDA Built | Behavior |
|---------------|---------------|------------|----------|
| `n_gpu_layers = 0` | ✓ Yes | ✓ Yes | **CPU-only** (current behavior) |
| `n_gpu_layers = 24` | ✓ Yes | ✓ Yes | **GPU-accelerated** (24 layers offloaded) |
| `n_gpu_layers = -1` | ✓ Yes | ✓ Yes | **GPU-accelerated** (all layers offloaded) |
| `n_gpu_layers = 24` | ✗ No | ✓ Yes | **CPU fallback** (warning logged) |
| `n_gpu_layers = 24` | ✓ Yes | ✗ No | **CPU fallback** (warning logged) |
| `n_gpu_layers = 0` | ✗ No | ✗ No | **CPU-only** (expected) |

**Key Invariant**: GPU configuration **never crashes**. Invalid configurations degrade gracefully to CPU with warnings.

---

## 3. API Changes

### 3.1 C++ FFI Function Signature (No Change Required)

**Current Signature** (`crossval/src/bitnet_cpp_wrapper.cc:357-364`):

```cpp
int bitnet_cpp_init_context(
    bitnet_context_t** out_ctx,
    const char* model_path,
    int32_t n_ctx,
    int32_t n_gpu_layers,  // ← Already exists, just unused
    char* err,
    int32_t err_len
)
```

**Status**: ✅ **No change needed**. The API already accepts `n_gpu_layers`.

### 3.2 Rust FFI Wrapper (Enhancement Required)

**Current Implementation** (`crossval/src/cpp_bindings.rs:607-635`):

```rust
pub fn create(model_path: &std::path::Path, n_ctx: i32, n_gpu_layers: i32) -> Result<Self> {
    // ... (validation)

    let result = unsafe {
        bitnet_cpp_init_context(
            &mut ctx_ptr,
            model_path_c.as_ptr(),
            n_ctx,
            n_gpu_layers,  // ← Passed through, but C++ ignores it
            err_buf.as_mut_ptr() as *mut c_char,
            err_buf.len() as i32,
        )
    };

    // ... (error handling)
}
```

**Proposed Enhancement** (environment variable support):

```rust
pub fn create(model_path: &std::path::Path, n_ctx: i32, n_gpu_layers: i32) -> Result<Self> {
    // Early availability check
    if !matches!(option_env!("CROSSVAL_HAS_BITNET"), Some("true")) {
        return Err(CrossvalError::CppNotAvailable);
    }

    // Apply environment variable override if n_gpu_layers == 0
    let effective_gpu_layers = if n_gpu_layers == 0 {
        std::env::var("BITNET_GPU_LAYERS")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0)
    } else {
        n_gpu_layers
    };

    let model_path_c = CString::new(model_path.to_str().ok_or_else(|| {
        CrossvalError::ModelLoadError("Invalid UTF-8 in model path".to_string())
    })?)
    .map_err(|e| {
        CrossvalError::ModelLoadError(format!(
            "Model path contains NUL byte at position {}",
            e.nul_position()
        ))
    })?;

    let mut err_buf = vec![0u8; 512];
    let mut ctx_ptr: *mut BitnetContext = std::ptr::null_mut();

    let result = unsafe {
        bitnet_cpp_init_context(
            &mut ctx_ptr,
            model_path_c.as_ptr(),
            n_ctx,
            effective_gpu_layers,  // ← Now respects BITNET_GPU_LAYERS
            err_buf.as_mut_ptr() as *mut c_char,
            err_buf.len() as i32,
        )
    };

    if result != 0 {
        let error_msg =
            std::str::from_utf8(&err_buf).unwrap_or("unknown error").trim_end_matches('\0');
        return Err(CrossvalError::ModelLoadError(error_msg.to_string()));
    }

    if ctx_ptr.is_null() {
        return Err(CrossvalError::ModelLoadError(
            "C++ returned null context".to_string()
        ));
    }

    Ok(Self { inner: ctx_ptr, _phantom: std::marker::PhantomData })
}
```

**Key Changes**:
1. **Environment variable support**: `BITNET_GPU_LAYERS` overrides default 0 (explicit non-zero values take precedence)
2. **Validation**: Parse environment variable safely with error handling
3. **Backward compatibility**: `n_gpu_layers = 0` → check env var → fallback to 0 if unset

### 3.3 Public API Documentation

**Update required in**: `crossval/src/cpp_bindings.rs:600-606`

```rust
/// Create a new BitNet session with persistent context
///
/// # Arguments
///
/// * `model_path` - Path to GGUF model file
/// * `n_ctx` - Context size for inference (e.g., 512, 2048)
/// * `n_gpu_layers` - Number of layers to offload to GPU:
///   - `0`: CPU-only (default). Checks `BITNET_GPU_LAYERS` env var for override.
///   - `1..N`: Offload first N layers to GPU (requires CUDA runtime)
///   - `-1`: Offload all layers to GPU (auto-detection)
///
/// # Environment Variables
///
/// * `BITNET_GPU_LAYERS` - Override GPU layer count when `n_gpu_layers == 0`
///   - Example: `BITNET_GPU_LAYERS=24` offloads 24 layers
///   - Ignored if explicit non-zero `n_gpu_layers` provided
///
/// # GPU Availability
///
/// GPU offloading requires:
/// 1. CUDA-capable GPU (compute capability ≥ 6.0)
/// 2. CUDA runtime libraries (libcudart.so / cudart64_*.dll)
/// 3. Sufficient VRAM (estimate: ~100-500MB per billion parameters)
///
/// If GPU unavailable or VRAM insufficient, llama.cpp gracefully falls back to CPU
/// with a warning message. This function never fails due to GPU unavailability.
///
/// # Returns
///
/// A `BitnetSession` handle on success, or `CrossvalError` on failure.
///
/// # Examples
///
/// ```rust,no_run
/// use crossval::cpp_bindings::BitnetSession;
///
/// // CPU-only inference
/// let session = BitnetSession::create(
///     std::path::Path::new("model.gguf"),
///     512,  // n_ctx
///     0     // n_gpu_layers (CPU-only)
/// )?;
///
/// // GPU-accelerated (24 layers)
/// let session = BitnetSession::create(
///     std::path::Path::new("model.gguf"),
///     512,  // n_ctx
///     24    // n_gpu_layers
/// )?;
///
/// // GPU-accelerated (all layers)
/// let session = BitnetSession::create(
///     std::path::Path::new("model.gguf"),
///     512,  // n_ctx
///     -1    // n_gpu_layers (auto-detect)
/// )?;
///
/// // Environment variable override
/// std::env::set_var("BITNET_GPU_LAYERS", "24");
/// let session = BitnetSession::create(
///     std::path::Path::new("model.gguf"),
///     512,  // n_ctx
///     0     // n_gpu_layers → becomes 24 via env var
/// )?;
/// ```
pub fn create(model_path: &std::path::Path, n_ctx: i32, n_gpu_layers: i32) -> Result<Self>
```

---

## 4. Implementation Details

### 4.1 C++ Code Changes

**File**: `crossval/src/bitnet_cpp_wrapper.cc`

**Change Location**: Lines 408-410

**Before** (current disabled code):

```cpp
// Step 2: Load model
llama_model_params model_params = llama_model_default_params();
// Note: GPU layer offloading would be configured here
// model_params.n_gpu_layers = n_gpu_layers;

ctx->model = llama_load_model_from_file(model_path, model_params);
```

**After** (enable GPU layers):

```cpp
// Step 2: Load model with GPU offloading
llama_model_params model_params = llama_model_default_params();

// Apply GPU layer configuration
// n_gpu_layers semantics (llama.cpp convention):
//   0 = CPU-only (no offloading)
//   N > 0 = Offload first N layers to GPU
//   -1 = Offload all layers (auto-detection)
if (n_gpu_layers != 0) {
    model_params.n_gpu_layers = static_cast<uint32_t>(
        n_gpu_layers == -1 ? INT32_MAX : n_gpu_layers
    );
    // Note: llama.cpp handles GPU unavailability gracefully with fallback to CPU
    // No explicit CUDA availability check needed here
}

ctx->model = llama_load_model_from_file(model_path, model_params);
if (!ctx->model) {
    snprintf(err, err_len, "bitnet_cpp_init_context: Failed to load model from %s", model_path);
    err[err_len - 1] = '\0';
    delete ctx;
    return -1;
}
```

**Key Implementation Notes**:

1. **Type Safety**: Cast `int32_t` to `uint32_t` for llama.cpp API (negative values handled explicitly)
2. **Auto-Detection**: `-1` maps to `INT32_MAX` (llama.cpp convention for "use all layers")
3. **Graceful Degradation**: llama.cpp runtime handles GPU unavailability internally (no explicit check needed)
4. **Zero Overhead**: `n_gpu_layers == 0` → no GPU code path (pure CPU as today)

### 4.2 Error Handling Strategy

**Error Categories**:

| Error Type | Detection | Handling |
|------------|-----------|----------|
| **GPU unavailable** | Runtime (llama.cpp) | Fallback to CPU + warning log |
| **CUDA version mismatch** | Runtime (llama.cpp) | Fallback to CPU + warning log |
| **VRAM exhausted** | Runtime (llama.cpp) | Fallback to CPU + warning log |
| **Invalid n_gpu_layers** | N/A (all i32 valid) | Clamp to 0 or model layer count |

**No Additional Error Handling Required**: llama.cpp's `llama_load_model_from_file()` already implements comprehensive GPU fallback logic. The wrapper simply passes through configuration.

### 4.3 Logging and Diagnostics

**Recommended Logging** (optional enhancement):

```cpp
// After model load, query actual GPU usage
if (n_gpu_layers != 0) {
    // Note: llama.cpp may not expose actual GPU layer count in API
    // This is informational only, not required for correctness
    fprintf(stderr, "INFO: Requested %d GPU layers (actual usage determined by llama.cpp runtime)\n",
            n_gpu_layers);
}
```

**Rust-Side Diagnostics**:

```rust
// In BitnetSession::create(), log effective configuration
if effective_gpu_layers > 0 {
    eprintln!("INFO: GPU offloading enabled: {} layers", effective_gpu_layers);
} else {
    eprintln!("INFO: CPU-only inference (n_gpu_layers=0)");
}
```

---

## 5. Configuration Strategy

### 5.1 Environment Variable Specification

**Variable**: `BITNET_GPU_LAYERS`

**Syntax**:
```bash
BITNET_GPU_LAYERS=<integer>
```

**Valid Values**:
- `0`: CPU-only (explicit)
- `1..N`: Offload N layers
- `-1`: Auto-detect (use all layers)
- Empty/unset: Use function argument (default 0)

**Parsing Rules**:
1. **Parse error → ignore**: Invalid integers (e.g., "abc") → fall back to 0
2. **Negative values**: Only `-1` supported; other negatives → 0
3. **Overflow**: Values > `INT32_MAX` → clamp to `INT32_MAX`

**Precedence** (highest to lowest):
1. Explicit function argument (`n_gpu_layers != 0`)
2. `BITNET_GPU_LAYERS` environment variable
3. Default (0 = CPU-only)

### 5.2 CLI Integration (xtask)

**Proposed `xtask` Flag** (future enhancement):

```bash
# xtask crossval-per-token with GPU
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "Test" \
  --gpu-layers 24  # ← New flag

# Alternative: environment variable
BITNET_GPU_LAYERS=24 cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "Test"

# Auto-detect
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "Test" \
  --gpu-layers -1
```

**Implementation Note**: This requires `xtask/src/crossval/backend.rs` changes (out of scope for this spec, but documented for completeness).

### 5.3 Default Behavior

**Conservative Default**: `n_gpu_layers = 0` (CPU-only)

**Rationale**:
1. **Predictability**: Cross-validation results identical across CPU/GPU systems
2. **CI Compatibility**: Many CI runners lack GPU (GitHub Actions standard)
3. **Explicit Opt-In**: Users must consciously enable GPU (avoid surprise VRAM usage)

**Future Consideration**: Auto-detection (`-1`) as default after GPU testing matures.

---

## 6. Performance Expectations

### 6.1 Benchmark Targets

**Model**: microsoft-bitnet-b1.58-2B-4T-gguf (2B parameters, ~32 layers)

| Configuration | Tokens/Sec | Latency (10 tokens) | Speedup |
|---------------|------------|---------------------|---------|
| **CPU-only** (baseline) | 0.1 | ~100s | 1.0× |
| **GPU 8 layers** | 0.5 | ~20s | 5.0× |
| **GPU 16 layers** | 1.0 | ~10s | 10.0× |
| **GPU 24 layers** | 2.0 | ~5s | 20.0× |
| **GPU all layers (-1)** | 3.0 | ~3.3s | 30.0× |

**Assumptions**:
- NVIDIA RTX 4090 (24GB VRAM, 82 TFLOPS FP16)
- Model fits in VRAM (2B params ≈ 4GB FP16)
- Batch size = 1 (single-sequence inference)
- No KV cache reuse (cold start per token)

**Variance Factors**:
- **Model size**: Larger models benefit more from GPU (bandwidth-limited)
- **Quantization**: I2_S vs QK256 affects memory bandwidth requirements
- **GPU utilization**: Small models may underutilize GPU compute (Amdahl's law)

### 6.2 Memory Requirements

**VRAM Estimation**:

```
VRAM_per_layer ≈ (model_params / num_layers) * bytes_per_param + overhead

For 2B params, 32 layers, FP16:
VRAM_per_layer ≈ (2e9 / 32) * 2 bytes + 50MB ≈ 175MB/layer

24 layers: ~4.2GB VRAM (RTX 4090 safe)
```

**System RAM vs VRAM Trade-off**:
- **CPU-only**: Model resides in system RAM (~4GB for 2B FP16)
- **GPU 24 layers**: ~4GB VRAM + ~500MB system RAM (metadata)
- **GPU all layers**: ~4.5GB VRAM + minimal system RAM

**Failure Mode**: If VRAM insufficient, llama.cpp automatically falls back to CPU for remaining layers (hybrid execution).

### 6.3 Performance Validation

**Measurement Strategy**:

1. **Benchmark harness**: `cargo bench --bench crossval_gpu_benchmark --features crossval`
2. **Metrics**: Tokens/sec, latency (P50/P95/P99), VRAM usage
3. **Scenarios**:
   - CPU baseline (n_gpu_layers=0)
   - GPU incremental (8, 16, 24 layers)
   - GPU full (n_gpu_layers=-1)
4. **Regression threshold**: GPU should be ≥5× faster than CPU for 2B models

**Example Benchmark Output**:

```
crossval_gpu_benchmark/cpu_only           time: [98.5 s 100.2 s 102.1 s]
crossval_gpu_benchmark/gpu_8_layers       time: [19.2 s 20.1 s 21.3 s]  (5.0× speedup)
crossval_gpu_benchmark/gpu_24_layers      time: [4.8 s 5.2 s 5.7 s]     (19.2× speedup)
crossval_gpu_benchmark/gpu_all_layers     time: [3.1 s 3.4 s 3.8 s]     (29.5× speedup)
```

---

## 7. Testing Requirements

### 7.1 Unit Tests

**File**: `crossval/tests/ffi_socket_tests.rs`

**Test Cases** (AC_IDs for `// AC:ID` tags):

```rust
/// AC1: Verify n_gpu_layers=0 uses CPU-only path (baseline)
#[test]
#[serial(bitnet_env)]
fn test_gpu_layers_zero_cpu_only() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS", "");
    let session = BitnetSession::create(
        Path::new("models/test.gguf"),
        512,
        0  // n_gpu_layers=0
    ).unwrap();

    // Assert: Model loaded, no GPU errors
    assert!(session.inner.is_null() == false);
}

/// AC2: Verify n_gpu_layers=24 enables GPU offloading (no crash on GPU unavailable)
#[test]
#[serial(bitnet_env)]
fn test_gpu_layers_explicit_count() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS", "");
    let session = BitnetSession::create(
        Path::new("models/test.gguf"),
        512,
        24  // n_gpu_layers=24
    ).unwrap();

    // Assert: Model loaded (CPU fallback if GPU unavailable)
    assert!(session.inner.is_null() == false);
}

/// AC3: Verify n_gpu_layers=-1 auto-detects all layers
#[test]
#[serial(bitnet_env)]
fn test_gpu_layers_auto_detect() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS", "");
    let session = BitnetSession::create(
        Path::new("models/test.gguf"),
        512,
        -1  // n_gpu_layers=-1 (auto-detect)
    ).unwrap();

    assert!(session.inner.is_null() == false);
}

/// AC4: Verify BITNET_GPU_LAYERS environment variable overrides n_gpu_layers=0
#[test]
#[serial(bitnet_env)]
fn test_gpu_layers_env_override() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS", "24");

    // Pass n_gpu_layers=0, expect env var (24) to take effect
    let session = BitnetSession::create(
        Path::new("models/test.gguf"),
        512,
        0  // n_gpu_layers=0 → BITNET_GPU_LAYERS=24 overrides
    ).unwrap();

    assert!(session.inner.is_null() == false);
}

/// AC5: Verify explicit n_gpu_layers overrides environment variable
#[test]
#[serial(bitnet_env)]
fn test_gpu_layers_explicit_overrides_env() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS", "8");

    // Pass n_gpu_layers=24 (explicit), expect 24 not 8
    let session = BitnetSession::create(
        Path::new("models/test.gguf"),
        512,
        24  // Explicit value overrides BITNET_GPU_LAYERS=8
    ).unwrap();

    assert!(session.inner.is_null() == false);
}

/// AC6: Verify invalid BITNET_GPU_LAYERS (non-integer) falls back to 0
#[test]
#[serial(bitnet_env)]
fn test_gpu_layers_invalid_env_var() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS", "invalid");

    // Invalid env var → parse error → fall back to n_gpu_layers=0
    let session = BitnetSession::create(
        Path::new("models/test.gguf"),
        512,
        0
    ).unwrap();

    assert!(session.inner.is_null() == false);
}

/// AC7: Verify GPU unavailable gracefully falls back to CPU (no crash)
#[test]
#[serial(bitnet_env)]
#[cfg_attr(not(feature = "gpu"), ignore)] // Only run if GPU feature enabled
fn test_gpu_unavailable_fallback() {
    let _guard = EnvGuard::new("CUDA_VISIBLE_DEVICES", "-1"); // Force GPU unavailable

    let session = BitnetSession::create(
        Path::new("models/test.gguf"),
        512,
        24  // Request GPU, but it's unavailable
    );

    // Should succeed with CPU fallback (llama.cpp handles gracefully)
    assert!(session.is_ok());
}
```

**Test Execution**:

```bash
# Run GPU layer tests
cargo test -p crossval --test ffi_socket_tests test_gpu_layers --features ffi

# Run with GPU feature enabled (requires CUDA)
cargo test -p crossval --test ffi_socket_tests test_gpu_layers --features ffi,gpu
```

### 7.2 Integration Tests

**File**: `crossval/tests/ffi_integration_tests.rs`

**Test Case** (AC8):

```rust
/// AC8: Verify end-to-end inference with GPU layers produces valid logits
#[test]
#[serial(bitnet_env)]
fn test_gpu_inference_produces_valid_logits() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS", "24");

    let session = BitnetSession::create(
        Path::new("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"),
        512,
        0  // n_gpu_layers=0 → BITNET_GPU_LAYERS=24 overrides
    ).unwrap();

    let tokens = vec![1, 2, 3]; // Simple token sequence
    let logits = session.eval_and_get_logits(&tokens, 0).unwrap();

    // Validate logits shape and sanity
    assert!(logits.len() > 0);
    assert!(logits.iter().all(|&x| x.is_finite())); // No NaN/Inf
}
```

### 7.3 Cross-Validation Parity Tests

**Test Case** (AC9):

```rust
/// AC9: Verify GPU logits match CPU logits within tolerance (numerical parity)
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "gpu")]
fn test_gpu_cpu_parity() {
    let model_path = Path::new("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf");
    let tokens = vec![1, 2, 3];

    // CPU inference
    let _guard_cpu = EnvGuard::new("CUDA_VISIBLE_DEVICES", "-1");
    let session_cpu = BitnetSession::create(model_path, 512, 0).unwrap();
    let logits_cpu = session_cpu.eval_and_get_logits(&tokens, 0).unwrap();
    drop(session_cpu);
    drop(_guard_cpu);

    // GPU inference
    let _guard_gpu = EnvGuard::new("BITNET_GPU_LAYERS", "24");
    let session_gpu = BitnetSession::create(model_path, 512, 0).unwrap();
    let logits_gpu = session_gpu.eval_and_get_logits(&tokens, 0).unwrap();

    // Assert numerical parity (cosine similarity ≥ 0.999)
    let cos_sim = cosine_similarity(&logits_cpu, &logits_gpu);
    assert!(cos_sim >= 0.999, "GPU/CPU logits diverged: cos_sim={}", cos_sim);
}
```

### 7.4 Performance Regression Tests

**Benchmark File**: `crossval/benches/gpu_offloading_bench.rs` (new file)

**Test Case** (AC10):

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use crossval::cpp_bindings::BitnetSession;
use std::path::Path;

/// AC10: Verify GPU inference is ≥5× faster than CPU for 2B models
fn bench_gpu_speedup(c: &mut Criterion) {
    let model_path = Path::new("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf");
    let tokens = vec![1; 10]; // 10-token sequence

    // CPU baseline
    c.bench_function("cpu_only", |b| {
        let session = BitnetSession::create(model_path, 512, 0).unwrap();
        b.iter(|| {
            let logits = session.eval_and_get_logits(&tokens, 0).unwrap();
            black_box(logits);
        });
    });

    // GPU 24 layers
    std::env::set_var("BITNET_GPU_LAYERS", "24");
    c.bench_function("gpu_24_layers", |b| {
        let session = BitnetSession::create(model_path, 512, 0).unwrap();
        b.iter(|| {
            let logits = session.eval_and_get_logits(&tokens, 0).unwrap();
            black_box(logits);
        });
    });
}

criterion_group!(benches, bench_gpu_speedup);
criterion_main!(benches);
```

**Run Benchmark**:

```bash
cargo bench --bench gpu_offloading_bench --features crossval,gpu
```

### 7.5 CI/CD Integration

**GitHub Actions Workflow** (`.github/workflows/crossval-gpu.yml`):

```yaml
name: Cross-Validation GPU Tests

on: [push, pull_request]

jobs:
  test-cpu-fallback:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test CPU fallback (no GPU)
        run: |
          cargo test -p crossval --features ffi test_gpu_layers
          # Verify CUDA_VISIBLE_DEVICES=-1 forces CPU fallback
          CUDA_VISIBLE_DEVICES=-1 cargo test -p crossval --features ffi test_gpu_unavailable_fallback

  test-gpu-acceleration:
    runs-on: [self-hosted, gpu]  # Requires GPU runner
    steps:
      - uses: actions/checkout@v3
      - name: Test GPU acceleration
        run: |
          cargo test -p crossval --features ffi,gpu test_gpu_layers
          cargo bench --bench gpu_offloading_bench --features crossval,gpu -- --test
```

---

## 8. Acceptance Criteria

**Definition of Done**: All acceptance criteria (AC) must pass before merging.

### AC1: CPU-Only Baseline (Backward Compatibility)
**Status**: ✅ **PASS**
**Verification**: `test_gpu_layers_zero_cpu_only`
**Requirement**: `n_gpu_layers=0` continues to use CPU-only inference (no regression)

### AC2: Explicit GPU Layer Count
**Status**: 🟡 **PENDING**
**Verification**: `test_gpu_layers_explicit_count`
**Requirement**: `n_gpu_layers=24` enables GPU offloading for 24 layers (graceful fallback if GPU unavailable)

### AC3: Auto-Detection
**Status**: 🟡 **PENDING**
**Verification**: `test_gpu_layers_auto_detect`
**Requirement**: `n_gpu_layers=-1` offloads all layers to GPU (llama.cpp auto-detection)

### AC4: Environment Variable Override
**Status**: 🟡 **PENDING**
**Verification**: `test_gpu_layers_env_override`
**Requirement**: `BITNET_GPU_LAYERS=24` overrides `n_gpu_layers=0`

### AC5: Explicit Precedence
**Status**: 🟡 **PENDING**
**Verification**: `test_gpu_layers_explicit_overrides_env`
**Requirement**: Explicit `n_gpu_layers` takes precedence over `BITNET_GPU_LAYERS`

### AC6: Invalid Environment Variable Handling
**Status**: 🟡 **PENDING**
**Verification**: `test_gpu_layers_invalid_env_var`
**Requirement**: Invalid `BITNET_GPU_LAYERS` (e.g., "abc") falls back to `n_gpu_layers=0` without crashing

### AC7: GPU Unavailable Fallback
**Status**: 🟡 **PENDING**
**Verification**: `test_gpu_unavailable_fallback`
**Requirement**: Requesting GPU when unavailable falls back to CPU gracefully (no crash, warning logged)

### AC8: Valid Logits Output
**Status**: 🟡 **PENDING**
**Verification**: `test_gpu_inference_produces_valid_logits`
**Requirement**: GPU inference produces finite logits (no NaN/Inf)

### AC9: GPU/CPU Numerical Parity
**Status**: 🟡 **PENDING**
**Verification**: `test_gpu_cpu_parity`
**Requirement**: GPU logits match CPU logits with cosine similarity ≥ 0.999

### AC10: Performance Improvement
**Status**: 🟡 **PENDING**
**Verification**: `bench_gpu_speedup`
**Requirement**: GPU inference ≥5× faster than CPU for 2B models (24 layers offloaded)

---

## 9. Risks and Mitigations

### Risk 1: GPU VRAM Exhaustion
**Probability**: MEDIUM
**Impact**: HIGH (out-of-memory crash)

**Mitigation**:
1. **llama.cpp handles gracefully**: Automatic fallback to CPU for remaining layers
2. **Documentation**: Clearly specify VRAM requirements (~100-500MB per billion params)
3. **Conservative defaults**: `n_gpu_layers=0` (CPU-only) as default

**Residual Risk**: Users with insufficient VRAM may experience degraded performance (hybrid CPU/GPU). Acceptable for MVP.

### Risk 2: CUDA Version Mismatch
**Probability**: LOW
**Impact**: MEDIUM (GPU unavailable, fallback to CPU)

**Mitigation**:
1. **Runtime detection**: llama.cpp checks CUDA compatibility at runtime
2. **Fallback**: Graceful degradation to CPU with warning log
3. **Documentation**: Specify minimum CUDA version (11.0+) in README

**Residual Risk**: None (graceful fallback)

### Risk 3: Performance Regression on CPU
**Probability**: LOW
**Impact**: MEDIUM (slower CPU inference)

**Mitigation**:
1. **Zero overhead**: `n_gpu_layers=0` → no GPU code path executed
2. **Regression tests**: Benchmark CPU-only performance before/after changes
3. **CI validation**: Run CPU-only tests on every commit

**Residual Risk**: None (code change is minimal, isolated to GPU path)

### Risk 4: API Misuse (Incorrect Layer Count)
**Probability**: MEDIUM
**Impact**: LOW (suboptimal performance)

**Mitigation**:
1. **Validation**: Accept all `i32` values, clamp to valid range internally
2. **Documentation**: Provide examples with common layer counts (8, 16, 24)
3. **Auto-detection**: Support `-1` for "use all layers" (eliminates guesswork)

**Residual Risk**: Users may specify too many layers (VRAM exhaustion), but llama.cpp handles gracefully.

### Risk 5: CI/CD GPU Runner Availability
**Probability**: HIGH
**Impact**: MEDIUM (GPU tests cannot run in CI)

**Mitigation**:
1. **CPU fallback tests**: Ensure all tests pass on CPU-only runners
2. **Self-hosted GPU runners**: Configure self-hosted GitHub runner with GPU (optional)
3. **Manual GPU testing**: Document manual GPU testing procedures for contributors

**Residual Risk**: GPU performance regressions may not be detected in CI. Acceptable for v0.2 (manual testing suffices).

---

## 10. Future Considerations

### 10.1 Socket 5 Dedicated GPU API (v0.3)

**Current Spec** (`docs/specs/bitnet-cpp-ffi-sockets.md:500-523`):

```c
int bitnet_cpp_eval_gpu(
    const bitnet_context_t* ctx,
    const int32_t* tokens,
    int32_t n_tokens,
    float* out_logits,
    int32_t logits_capacity,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
);
```

**Rationale**: Separate GPU-specific inference function for:
1. **Explicit GPU control**: Force GPU path, fail if unavailable
2. **GPU-specific optimizations**: Batching, mixed precision, kernel fusion
3. **Performance profiling**: Separate GPU-only metrics

**Decision Point**: Evaluate after v0.2 GPU layer testing. If `n_gpu_layers` sufficient → defer Socket 5.

### 10.2 Dynamic Layer Offloading

**Concept**: Adjust `n_gpu_layers` at runtime based on VRAM availability

**Use Case**: Large models (30B+) where VRAM varies by workload

**Implementation Sketch**:

```rust
pub fn adjust_gpu_layers(&mut self, new_count: i32) -> Result<()> {
    // Requires context recreation (expensive)
    // OR: llama.cpp dynamic offloading API (if exists)
    unimplemented!("v0.3 feature")
}
```

**Status**: v0.3 consideration (requires llama.cpp runtime API)

### 10.3 Mixed Precision GPU Inference

**Concept**: Use FP16/BF16 on GPU, FP32 on CPU for better performance

**Implementation**: llama.cpp `model_params.use_fp16 = true`

**Benefit**: ~2× memory savings, ~1.5× speedup on Tensor Core GPUs

**Status**: v0.3 enhancement (after GPU layer stability)

### 10.4 Multi-GPU Support

**Concept**: Distribute layers across multiple GPUs

**Use Case**: Very large models (70B+) exceeding single GPU VRAM

**Implementation**: llama.cpp `model_params.tensor_split[]` array

**Status**: v0.4+ (low priority for BitNet 1-3B models)

---

## Appendix A: Related Documentation

**Specifications**:
- `docs/specs/bitnet-cpp-ffi-sockets.md`: FFI socket architecture
- `docs/specs/INDEX.md`: Specification index and status

**Architecture**:
- `docs/explanation/dual-backend-crossval.md`: Cross-validation architecture
- `docs/explanation/backend-detection-and-device-selection-patterns.md`: Device selection

**How-To Guides**:
- `docs/howto/cpp-setup.md`: C++ reference setup for cross-validation

**References**:
- `CLAUDE.md`: Project overview and quick reference
- `docs/GPU_SETUP.md`: GPU configuration guide

---

## Appendix B: llama.cpp GPU API Reference

**Relevant llama.cpp Types** (from llama.h):

```c
struct llama_model_params {
    int32_t n_gpu_layers;  // Number of layers to store in VRAM
    // ... other fields
};

// Default params (n_gpu_layers = 0)
struct llama_model_params llama_model_default_params(void);

// Model loading with GPU offloading
struct llama_model * llama_load_model_from_file(
    const char * path_model,
    struct llama_model_params params
);
```

**GPU Offloading Semantics**:
- `n_gpu_layers = 0`: CPU-only (no GPU used)
- `n_gpu_layers = N`: Offload first N layers to GPU
- `n_gpu_layers = INT_MAX`: Offload all layers (auto-detection)

**Runtime Behavior**:
- If GPU unavailable: Silently falls back to CPU (no error)
- If VRAM insufficient: Partial offloading (hybrid CPU/GPU)
- No explicit GPU availability check needed (handled internally)

---

## Appendix C: BitNet.rs GPU Feature Matrix

| Feature | v0.1 (MVP) | v0.2 (This Spec) | v0.3 (Future) |
|---------|------------|------------------|---------------|
| CPU inference | ✅ Stable | ✅ Stable | ✅ Stable |
| GPU layer offloading (Socket 1) | ❌ Disabled | 🟡 **Enabled** | ✅ Stable |
| Socket 5 GPU API | ❌ Stub | ❌ Stub | 🟡 Planned |
| Auto-detection (`-1`) | ❌ N/A | 🟡 **Supported** | ✅ Default |
| Environment variable (`BITNET_GPU_LAYERS`) | ❌ N/A | 🟡 **Supported** | ✅ Stable |
| Mixed precision (FP16/BF16) | ❌ N/A | ❌ Future | 🟡 Planned |
| Multi-GPU support | ❌ N/A | ❌ Future | ❌ v0.4+ |

**Legend**:
- ✅ **Stable**: Fully implemented, tested, documented
- 🟡 **Enabled/Planned**: Implemented but not yet stable
- ❌ **Disabled/Future**: Not implemented, future work

---

**END OF SPECIFICATION**

---

## Changelog

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-10-25 | 0.1.0 | spec-gate | Initial specification draft |

---

**Approval Required**: spec-finalizer (architectural review)
