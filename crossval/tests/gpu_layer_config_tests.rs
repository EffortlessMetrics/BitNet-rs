// GPU Layer Configuration Tests
//
// Test scaffolding for GPU layer offloading feature (v0.2.0 Socket 1 enhancement)
//
// **Specification**: docs/explanation/cpp-wrapper-gpu-layer-config.md
//
// **Feature Description**:
// Enable GPU layer offloading via the BitNet.cpp FFI wrapper, allowing users to
// configure the number of transformer layers to offload to GPU for accelerated
// inference. The feature provides:
// - Three-level configuration hierarchy: API > BITNET_GPU_LAYERS env > default 0
// - Graceful fallback to CPU when GPU unavailable
// - Auto-detection with n_gpu_layers=-1
//
// **Acceptance Criteria Coverage**:
// - AC1: CPU-only baseline (n_gpu_layers=0)
// - AC2: Explicit GPU layer count (n_gpu_layers=24)
// - AC3: Auto-detection (n_gpu_layers=-1)
// - AC4: Environment variable override (BITNET_GPU_LAYERS)
// - AC5: Explicit precedence over env var
// - AC6: Invalid env var handling
// - AC7: GPU unavailable fallback
// - AC8: Valid logits output
// - AC9: GPU/CPU numerical parity
//
// **Test Patterns**:
// - Use `#[serial(bitnet_env)]` for BITNET_GPU_LAYERS env var tests
// - Use `tests::support::env_guard::EnvGuard` for environment isolation
// - Use `#[cfg(feature = "gpu")]` for GPU-specific tests
// - Use `#[ignore]` for tests requiring GPU hardware
// - Use `#[cfg_attr(not(feature = "gpu"), ignore)]` for conditional skip
//
// **TDD Status**: Tests compile but fail due to missing GPU layer configuration implementation

#[allow(unused_imports)] // TDD scaffolding - used in unimplemented tests
use serial_test::serial;
#[allow(unused_imports)]
use std::path::Path;

// Re-export crossval types for test convenience
#[cfg(feature = "ffi")]
use crossval::cpp_bindings::BitnetSession;

// ============================================================================
// Test Helpers - Environment Guard (RAII-style for test isolation)
// ============================================================================

/// RAII-style environment variable guard for test isolation
///
/// This provides thread-safe environment variable management for tests,
/// ensuring automatic restoration of original values.
///
/// **Safety**: Always use with `#[serial(bitnet_env)]` to prevent
/// process-level races across multiple cargo test processes.
#[allow(dead_code)]
struct EnvGuard {
    key: String,
    old: Option<String>,
}

#[allow(dead_code)]
impl EnvGuard {
    /// Create a new environment variable guard
    fn new(key: &str) -> Self {
        let old = std::env::var(key).ok();
        Self { key: key.to_string(), old }
    }

    /// Set the environment variable to a new value
    fn set(&self, val: &str) {
        unsafe {
            std::env::set_var(&self.key, val);
        }
    }

    /// Remove the environment variable temporarily
    fn remove(&self) {
        unsafe {
            std::env::remove_var(&self.key);
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(ref v) = self.old {
                std::env::set_var(&self.key, v);
            } else {
                std::env::remove_var(&self.key);
            }
        }
    }
}

// Test helper: Check if GPU is available at runtime
#[cfg(feature = "ffi")]
fn check_gpu_available() -> bool {
    // NOTE: This is a placeholder for GPU runtime detection
    // Implementation should check CUDA runtime availability
    std::env::var("CUDA_VISIBLE_DEVICES").map(|v| !v.is_empty() && v != "-1").unwrap_or(false)
}

// Test helper: Get test model path (small model for fast testing)
#[cfg(feature = "ffi")]
fn get_test_model_path() -> &'static Path {
    // NOTE: This should point to a small test model for fast validation
    // Implementation will use models discovered by xtask download-model
    Path::new("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf")
}

// Test helper: Compare logits for GPU/CPU parity validation
#[cfg(feature = "ffi")]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Logits vectors must have same length");

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

// =============================================================================
// Category 1: Configuration Tests (AC1-AC7)
// =============================================================================

/// AC:AC1 - Verify n_gpu_layers=0 uses CPU-only path (baseline)
///
/// **Test Objective**: Validate backward compatibility - default CPU-only behavior
///
/// **Specification Reference**: docs/explanation/cpp-wrapper-gpu-layer-config.md#ac1
///
/// **Test Strategy**:
/// 1. Create BitnetSession with n_gpu_layers=0
/// 2. Verify model loads successfully
/// 3. Verify no GPU initialization (CPU fallback is transparent)
///
/// **Expected Outcome**: Model loads successfully in CPU-only mode
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "ffi")]
#[ignore] // TODO: Enable after implementing GPU layer configuration in BitnetSession::create
fn test_cpu_baseline_zero_gpu_layers() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS");
    _guard.remove(); // Ensure no env var override

    let model_path = get_test_model_path();

    // Create session with n_gpu_layers=0 (CPU-only)
    let session =
        BitnetSession::create(model_path, 512, 0).expect("Failed to create CPU-only session");

    // Verify session is valid
    assert!(!session.ctx.is_null(), "Session context should be non-null");

    // NOTE: In current implementation, n_gpu_layers is stored but not applied
    // After implementation, this should verify CPU-only execution via receipts or logs
}

/// AC:AC2 - Verify n_gpu_layers=24 enables GPU offloading
///
/// **Test Objective**: Validate explicit GPU layer count configuration
///
/// **Specification Reference**: docs/explanation/cpp-wrapper-gpu-layer-config.md#ac2
///
/// **Test Strategy**:
/// 1. Create BitnetSession with n_gpu_layers=24
/// 2. Verify model loads successfully (GPU or CPU fallback)
/// 3. No crash on GPU unavailable (graceful degradation)
///
/// **Expected Outcome**: Model loads successfully, uses GPU if available
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "ffi")]
#[ignore] // TODO: Enable after implementing GPU layer configuration
fn test_explicit_gpu_layers_via_api() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS");
    _guard.remove();

    let model_path = get_test_model_path();

    // Create session with explicit GPU layers (24)
    let session =
        BitnetSession::create(model_path, 512, 24).expect("Failed to create GPU-enabled session");

    assert!(!session.ctx.is_null());

    // NOTE: After implementation, verify GPU layer count via C++ wrapper diagnostics
    // or receipt metadata showing GPU kernel usage
}

/// AC:AC3 - Verify n_gpu_layers=-1 auto-detects all GPU layers
///
/// **Test Objective**: Validate auto-detection (use all available layers)
///
/// **Specification Reference**: docs/explanation/cpp-wrapper-gpu-layer-config.md#ac3
///
/// **Test Strategy**:
/// 1. Create BitnetSession with n_gpu_layers=-1
/// 2. Verify model loads successfully
/// 3. Verify auto-detection maps to INT32_MAX for llama.cpp
///
/// **Expected Outcome**: All layers offloaded to GPU (or CPU fallback)
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "ffi")]
#[ignore] // TODO: Enable after implementing GPU layer configuration
fn test_auto_detect_all_gpu_layers() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS");
    _guard.remove();

    let model_path = get_test_model_path();

    // Create session with n_gpu_layers=-1 (auto-detect)
    let session = BitnetSession::create(model_path, 512, -1)
        .expect("Failed to create auto-detect GPU session");

    assert!(!session.ctx.is_null());

    // NOTE: After implementation, verify all layers offloaded via diagnostics
}

/// AC:AC4 - Verify BITNET_GPU_LAYERS environment variable overrides n_gpu_layers=0
///
/// **Test Objective**: Validate environment variable override for default value
///
/// **Specification Reference**: docs/explanation/cpp-wrapper-gpu-layer-config.md#ac4
///
/// **Test Strategy**:
/// 1. Set BITNET_GPU_LAYERS=24 via EnvGuard
/// 2. Create BitnetSession with n_gpu_layers=0
/// 3. Verify env var takes precedence (24 layers offloaded)
///
/// **Expected Outcome**: Environment variable overrides n_gpu_layers=0
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "ffi")]
#[ignore] // TODO: Enable after implementing env var support in BitnetSession::create
fn test_env_var_gpu_layers() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS");
    _guard.set("24");

    let model_path = get_test_model_path();

    // Pass n_gpu_layers=0, expect BITNET_GPU_LAYERS=24 to override
    let session = BitnetSession::create(model_path, 512, 0)
        .expect("Failed to create session with env var override");

    assert!(!session.ctx.is_null());

    // NOTE: After implementation, verify 24 layers offloaded via diagnostics
}

/// AC:AC5 - Verify explicit n_gpu_layers overrides BITNET_GPU_LAYERS
///
/// **Test Objective**: Validate API precedence over environment variable
///
/// **Specification Reference**: docs/explanation/cpp-wrapper-gpu-layer-config.md#ac5
///
/// **Test Strategy**:
/// 1. Set BITNET_GPU_LAYERS=8 via EnvGuard
/// 2. Create BitnetSession with n_gpu_layers=24 (explicit)
/// 3. Verify explicit value (24) takes precedence over env var (8)
///
/// **Expected Outcome**: Explicit API parameter overrides environment variable
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "ffi")]
#[ignore] // TODO: Enable after implementing precedence logic
fn test_env_var_overrides_default() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS");
    _guard.set("8");

    let model_path = get_test_model_path();

    // Pass explicit n_gpu_layers=24 (should override BITNET_GPU_LAYERS=8)
    let session = BitnetSession::create(model_path, 512, 24)
        .expect("Failed to create session with explicit override");

    assert!(!session.ctx.is_null());

    // NOTE: After implementation, verify 24 layers (not 8) via diagnostics
}

/// AC:AC6 - Verify invalid BITNET_GPU_LAYERS falls back to n_gpu_layers=0
///
/// **Test Objective**: Validate error handling for malformed environment variables
///
/// **Specification Reference**: docs/explanation/cpp-wrapper-gpu-layer-config.md#ac6
///
/// **Test Strategy**:
/// 1. Set BITNET_GPU_LAYERS="invalid" (non-integer)
/// 2. Create BitnetSession with n_gpu_layers=0
/// 3. Verify graceful fallback to CPU-only (no panic)
///
/// **Expected Outcome**: Invalid env var ignored, session falls back to 0 layers
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "ffi")]
#[ignore] // TODO: Enable after implementing env var parsing with error handling
fn test_env_var_parsing_edge_cases() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS");
    _guard.set("invalid");

    let model_path = get_test_model_path();

    // Invalid env var should be ignored, fall back to n_gpu_layers=0
    let session = BitnetSession::create(model_path, 512, 0)
        .expect("Failed to create session with invalid env var");

    assert!(!session.ctx.is_null());

    // NOTE: After implementation, verify 0 layers (CPU-only) via diagnostics
}

/// AC:AC7 - Verify GPU unavailable gracefully falls back to CPU
///
/// **Test Objective**: Validate graceful degradation when GPU not available
///
/// **Specification Reference**: docs/explanation/cpp-wrapper-gpu-layer-config.md#ac7
///
/// **Test Strategy**:
/// 1. Force GPU unavailable via CUDA_VISIBLE_DEVICES=-1
/// 2. Create BitnetSession with n_gpu_layers=24 (request GPU)
/// 3. Verify session creation succeeds (CPU fallback)
///
/// **Expected Outcome**: Session creates successfully with CPU fallback (no crash)
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "ffi")]
#[cfg_attr(not(feature = "gpu"), ignore)] // Only run if GPU feature enabled
#[ignore] // TODO: Enable after verifying llama.cpp graceful fallback behavior
fn test_gpu_unavailable_fallback() {
    let _cuda_guard = EnvGuard::new("CUDA_VISIBLE_DEVICES");
    _cuda_guard.set("-1"); // Force GPU unavailable

    let _gpu_layers_guard = EnvGuard::new("BITNET_GPU_LAYERS");
    _gpu_layers_guard.remove();

    let model_path = get_test_model_path();

    // Request GPU layers but GPU is unavailable
    let session = BitnetSession::create(model_path, 512, 24);

    // Should succeed with CPU fallback (llama.cpp handles gracefully)
    assert!(
        session.is_ok(),
        "Session creation should succeed with CPU fallback when GPU unavailable"
    );

    // NOTE: After implementation, verify CPU fallback logged in stderr or diagnostics
}

// =============================================================================
// Category 2: Integration Tests (AC8-AC9)
// =============================================================================

/// AC:AC8 - Verify GPU inference produces valid logits
///
/// **Test Objective**: Validate end-to-end GPU inference correctness
///
/// **Specification Reference**: docs/explanation/cpp-wrapper-gpu-layer-config.md#ac8
///
/// **Test Strategy**:
/// 1. Create GPU-enabled BitnetSession (BITNET_GPU_LAYERS=24)
/// 2. Run inference with simple token sequence
/// 3. Verify logits are finite (no NaN/Inf)
/// 4. Verify logits shape matches vocabulary size
///
/// **Expected Outcome**: GPU inference produces valid logits without numerical errors
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "ffi")]
#[ignore] // TODO: Enable after implementing GPU layer configuration + logits validation
fn test_gpu_inference_logits_validation() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS");
    _guard.set("24");

    let model_path = get_test_model_path();

    let session =
        BitnetSession::create(model_path, 512, 0).expect("Failed to create GPU-enabled session");

    // Simple token sequence for inference
    let tokens = vec![1, 2, 3];

    // NOTE: This requires implementing eval_and_get_logits() method on BitnetSession
    // For now, this is a placeholder showing the expected API
    unimplemented!("eval_and_get_logits() not yet implemented - blocked by Socket 1 inference API");

    // Expected implementation:
    // let logits = session.eval_and_get_logits(&tokens, 0)
    //     .expect("Failed to get logits from GPU inference");
    //
    // // Validate logits are finite
    // assert!(logits.iter().all(|&x| x.is_finite()),
    //     "GPU logits contain NaN or Inf values");
    //
    // // Validate logits shape (should match vocabulary size)
    // assert!(logits.len() > 0, "GPU logits vector is empty");
}

/// AC:AC9 - Verify GPU logits match CPU logits within tolerance
///
/// **Test Objective**: Validate numerical parity between GPU and CPU inference
///
/// **Specification Reference**: docs/explanation/cpp-wrapper-gpu-layer-config.md#ac9
///
/// **Test Strategy**:
/// 1. Run inference with n_gpu_layers=0 (CPU-only)
/// 2. Run inference with n_gpu_layers=24 (GPU-accelerated)
/// 3. Compare logits using cosine similarity (≥0.999 threshold)
/// 4. Verify L2 distance within acceptable tolerance
///
/// **Expected Outcome**: GPU and CPU produce numerically identical results
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "ffi")]
#[cfg(feature = "gpu")]
#[ignore] // TODO: Enable after implementing GPU layer configuration + parity validation
fn test_gpu_cpu_parity() {
    let model_path = get_test_model_path();
    let tokens = vec![1, 2, 3];

    // =========================================================================
    // Step 1: CPU inference (n_gpu_layers=0)
    // =========================================================================
    let _guard_cpu = EnvGuard::new("CUDA_VISIBLE_DEVICES");
    _guard_cpu.set("-1"); // Force CPU-only

    let session_cpu =
        BitnetSession::create(model_path, 512, 0).expect("Failed to create CPU-only session");

    // NOTE: Placeholder for CPU inference
    unimplemented!("CPU inference not yet implemented - blocked by Socket 1 inference API");

    // Expected implementation:
    // let logits_cpu = session_cpu.eval_and_get_logits(&tokens, 0)
    //     .expect("Failed to get CPU logits");
    //
    // drop(session_cpu);
    // drop(_guard_cpu);

    // =========================================================================
    // Step 2: GPU inference (n_gpu_layers=24)
    // =========================================================================
    let _guard_gpu = EnvGuard::new("BITNET_GPU_LAYERS");
    _guard_gpu.set("24");

    let session_gpu =
        BitnetSession::create(model_path, 512, 0).expect("Failed to create GPU-enabled session");

    // NOTE: Placeholder for GPU inference
    unimplemented!("GPU inference not yet implemented - blocked by Socket 1 inference API");

    // Expected implementation:
    // let logits_gpu = session_gpu.eval_and_get_logits(&tokens, 0)
    //     .expect("Failed to get GPU logits");
    //
    // // =========================================================================
    // // Step 3: Parity validation
    // // =========================================================================
    // let cos_sim = cosine_similarity(&logits_cpu, &logits_gpu);
    // assert!(
    //     cos_sim >= 0.999,
    //     "GPU/CPU logits diverged: cosine_similarity={:.6} (threshold: 0.999)",
    //     cos_sim
    // );
    //
    // // Additional validation: L2 distance
    // let l2_dist: f32 = logits_cpu
    //     .iter()
    //     .zip(logits_gpu.iter())
    //     .map(|(a, b)| (a - b).powi(2))
    //     .sum::<f32>()
    //     .sqrt();
    //
    // assert!(
    //     l2_dist < 1e-3,
    //     "GPU/CPU logits L2 distance too large: {:.6} (threshold: 1e-3)",
    //     l2_dist
    // );
}

// =============================================================================
// Additional Edge Case Tests
// =============================================================================

/// Test: Verify large n_gpu_layers value (> model layer count) handled gracefully
///
/// **Test Objective**: Validate clamping or graceful handling of excessive layer counts
///
/// **Specification Reference**: docs/explanation/cpp-wrapper-gpu-layer-config.md#risk-4
///
/// **Expected Outcome**: llama.cpp clamps to model layer count, no crash
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "ffi")]
#[ignore] // TODO: Enable after verifying llama.cpp layer count clamping
fn test_excessive_gpu_layers_count() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS");
    _guard.remove();

    let model_path = get_test_model_path();

    // Request more layers than model has (e.g., 1000 layers for 32-layer model)
    let session = BitnetSession::create(model_path, 512, 1000);

    // Should succeed - llama.cpp clamps to actual layer count
    assert!(session.is_ok(), "Session creation should handle excessive layer count gracefully");
}

/// Test: Verify negative n_gpu_layers values (other than -1) handled gracefully
///
/// **Test Objective**: Validate handling of invalid negative layer counts
///
/// **Specification Reference**: docs/explanation/cpp-wrapper-gpu-layer-config.md#51
///
/// **Expected Outcome**: Negative values other than -1 fall back to 0 (CPU-only)
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "ffi")]
#[ignore] // TODO: Enable after implementing negative value validation
fn test_negative_gpu_layers_invalid() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS");
    _guard.remove();

    let model_path = get_test_model_path();

    // Request invalid negative value (not -1)
    let session = BitnetSession::create(model_path, 512, -5);

    // Should succeed with fallback to CPU-only
    assert!(session.is_ok(), "Session creation should handle invalid negative values");
}

/// Test: Verify BITNET_GPU_LAYERS with whitespace is parsed correctly
///
/// **Test Objective**: Validate environment variable parsing robustness
///
/// **Expected Outcome**: Whitespace trimmed, valid integer extracted
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "ffi")]
#[ignore] // TODO: Enable after implementing robust env var parsing
fn test_env_var_whitespace_handling() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS");
    _guard.set(" 24 "); // Whitespace padding

    let model_path = get_test_model_path();

    let session = BitnetSession::create(model_path, 512, 0);

    // Should parse "24" successfully after trimming whitespace
    assert!(session.is_ok(), "Session creation should handle whitespace in env var");
}

/// Test: Verify BITNET_GPU_LAYERS with empty string falls back to n_gpu_layers
///
/// **Test Objective**: Validate empty string is treated as unset
///
/// **Expected Outcome**: Empty string ignored, n_gpu_layers used
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "ffi")]
#[ignore] // TODO: Enable after implementing empty string handling
fn test_env_var_empty_string() {
    let _guard = EnvGuard::new("BITNET_GPU_LAYERS");
    _guard.set(""); // Empty string

    let model_path = get_test_model_path();

    let session = BitnetSession::create(model_path, 512, 0);

    // Should treat empty string as unset, use n_gpu_layers=0
    assert!(session.is_ok(), "Session creation should handle empty env var");
}

// =============================================================================
// Test Organization Notes
// =============================================================================
//
// **Test Execution**:
//
// Run all GPU configuration tests:
// ```bash
// cargo test -p crossval --test gpu_layer_config_tests --features ffi
// ```
//
// Run with GPU feature enabled (requires CUDA):
// ```bash
// cargo test -p crossval --test gpu_layer_config_tests --features ffi,gpu
// ```
//
// Run including ignored tests (requires test model):
// ```bash
// cargo test -p crossval --test gpu_layer_config_tests --features ffi -- --ignored --include-ignored
// ```
//
// **Blocked Dependencies**:
// - Socket 1 inference API (eval_and_get_logits method)
// - GPU layer configuration in bitnet_cpp_wrapper.cc (lines 408-410)
// - Environment variable support in BitnetSession::create
// - Test model provisioning via xtask download-model
//
// **Next Steps**:
// 1. Implement GPU layer configuration in C++ wrapper
// 2. Add environment variable support in Rust FFI wrapper
// 3. Implement eval_and_get_logits method for Socket 1
// 4. Provision test model for integration tests
// 5. Enable tests incrementally as blockers are resolved
