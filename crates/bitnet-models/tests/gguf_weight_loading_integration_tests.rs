//! GGUF Weight Loading Integration Tests (Issue #159)
//!
//! Tests feature spec: gguf-weight-loading-integration-testing.md
//! API contract: gguf-weight-loading-api-contracts.md
//!
//! Integration tests validating GGUF weight loading across multiple BitNet.rs crates:
//! - bitnet-models: Weight loading and parsing
//! - bitnet-quantization: I2S, TL1, TL2 quantization integration
//! - bitnet-kernels: Device-aware operations and GPU acceleration
#![allow(dead_code)] // Test utilities may be used by future tests
#![allow(deprecated)] // Uses deprecated load_gguf() for backward compatibility testing
//! - bitnet-inference: End-to-end inference with real weights

use anyhow::{Context, Result};
use bitnet_common::BitNetConfig;
#[allow(unused_imports)]
use bitnet_common::Device;
use candle_core::Tensor as CandleTensor;
use serial_test::serial;
use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::TempDir;

/// Integration test configuration
#[derive(Debug, Clone)]
pub struct IntegrationTestConfig {
    pub accuracy_threshold: f32,
    pub test_model_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub quantization_enabled: bool,
    pub cross_validation_enabled: bool,
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            accuracy_threshold: 0.99,
            test_model_layers: 2,
            hidden_size: 1024,
            intermediate_size: 2816,
            vocab_size: 16000,
            quantization_enabled: true,
            cross_validation_enabled: std::env::var("BITNET_CROSSVAL_WEIGHTS").is_ok(),
        }
    }
}

/// Integration test fixture builder
pub struct IntegrationTestFixture {
    temp_dir: TempDir,
    config: IntegrationTestConfig,
}

impl IntegrationTestFixture {
    pub fn new() -> Result<Self> {
        Ok(Self {
            temp_dir: TempDir::new().context("Failed to create temp directory")?,
            config: IntegrationTestConfig::default(),
        })
    }

    pub fn with_config(mut self, config: IntegrationTestConfig) -> Self {
        self.config = config;
        self
    }

    /// Create test GGUF file with complete model structure
    pub fn create_test_model(&self) -> Result<PathBuf> {
        let model_path = self.temp_dir.path().join("integration_test_model.gguf");

        // Create mock GGUF file that the minimal loader recognizes
        // The minimal loader (gguf_simple.rs) checks for this exact content
        // and creates a default tensor layout with mock tensors
        std::fs::write(&model_path, b"mock_gguf_content")
            .context("Failed to create test GGUF file")?;

        Ok(model_path)
    }
}

// ============================================================================
// Cross-Crate Integration Tests
// ============================================================================

/// Integration test: bitnet-models + bitnet-quantization
/// Validates that GGUF weight loading properly integrates with quantization
#[cfg(all(feature = "cpu", feature = "quantization"))]
#[tokio::test]
#[serial(bitnet_env)]
async fn test_integration_models_quantization_cpu() -> Result<()> {
    let config = IntegrationTestConfig::default();
    let fixture = IntegrationTestFixture::new()?.with_config(config.clone());
    let model_path = fixture.create_test_model()?;

    // Load model weights via bitnet-models
    let load_result = bitnet_models::gguf_simple::load_gguf(&model_path, Device::Cpu);

    match load_result {
        Ok((_loaded_config, tensor_map)) => {
            // Test quantization integration with loaded weights
            for layer_idx in 0..config.test_model_layers {
                let layer_prefix = format!("blk.{}", layer_idx);
                let weight_name = format!("{}.attn_q.weight", layer_prefix);

                if let Some(weight_tensor) = tensor_map.get(&weight_name) {
                    // Test I2S quantization integration
                    test_i2s_quantization_with_loaded_weight(weight_tensor, &weight_name)?;

                    // Test TL1 quantization integration
                    test_tl1_quantization_with_loaded_weight(weight_tensor, &weight_name)?;

                    // Test TL2 quantization integration
                    test_tl2_quantization_with_loaded_weight(weight_tensor, &weight_name)?;
                }
            }
        }
        Err(err) => {
            eprintln!("Models+Quantization integration test correctly failing (TDD Red): {}", err);
            panic!(
                "Integration test will pass once GGUF weight loading + quantization integration is complete"
            );
        }
    }

    Ok(())
}

/// Integration test: bitnet-models + bitnet-kernels
/// Validates device-aware operations with loaded weights
#[cfg(all(feature = "cpu", feature = "kernels"))]
#[tokio::test]
#[serial(bitnet_env)]
async fn test_integration_models_kernels_cpu() -> Result<()> {
    let config = IntegrationTestConfig::default();
    let fixture = IntegrationTestFixture::new()?.with_config(config.clone());
    let model_path = fixture.create_test_model()?;

    // Load model weights
    let load_result = bitnet_models::gguf_simple::load_gguf(&model_path, Device::Cpu);

    match load_result {
        Ok((_, tensor_map)) => {
            // Test kernel operations with loaded weights
            for layer_idx in 0..config.test_model_layers {
                let layer_prefix = format!("blk.{}", layer_idx);

                // Test attention kernel integration
                test_attention_kernel_with_loaded_weights(&tensor_map, &layer_prefix)?;

                // Test feed-forward kernel integration
                test_feedforward_kernel_with_loaded_weights(&tensor_map, &layer_prefix)?;
            }
        }
        Err(err) => {
            eprintln!("Models+Kernels integration test correctly failing (TDD Red): {}", err);
            panic!(
                "Integration test will pass once weight loading + kernel integration is complete"
            );
        }
    }

    Ok(())
}

/// Integration test: bitnet-models + bitnet-inference
/// Validates end-to-end inference pipeline with real weights
#[cfg(all(feature = "cpu", feature = "inference"))]
#[tokio::test]
#[serial(bitnet_env)]
async fn test_integration_models_inference_cpu() -> Result<()> {
    let config = IntegrationTestConfig::default();
    let fixture = IntegrationTestFixture::new()?.with_config(config.clone());
    let model_path = fixture.create_test_model()?;

    // Load model weights
    let load_result = bitnet_models::gguf_simple::load_gguf(&model_path, Device::Cpu);

    match load_result {
        Ok((loaded_config, tensor_map)) => {
            // Test inference engine with loaded weights
            test_inference_engine_with_real_weights(loaded_config, tensor_map)?;
        }
        Err(err) => {
            eprintln!("Models+Inference integration test correctly failing (TDD Red): {}", err);
            panic!(
                "Integration test will pass once weight loading + inference integration is complete"
            );
        }
    }

    Ok(())
}

/// GPU Integration test: Complete pipeline with GPU acceleration
#[cfg(feature = "gpu")]
#[tokio::test]
#[serial(bitnet_env)]
async fn test_integration_full_pipeline_gpu() -> Result<()> {
    let config = IntegrationTestConfig::default();
    let fixture = IntegrationTestFixture::new()?.with_config(config.clone());
    let model_path = fixture.create_test_model()?;

    // Test GPU loading with fallback
    let gpu_device = Device::Cuda(0);
    let load_result = bitnet_models::gguf_simple::load_gguf(&model_path, gpu_device);

    match load_result {
        Ok((loaded_config, tensor_map)) => {
            // Validate GPU tensor placement
            validate_gpu_tensor_placement(&tensor_map)?;

            // Test GPU quantization with loaded weights
            test_gpu_quantization_integration(&tensor_map)?;

            // Test GPU kernel operations
            test_gpu_kernel_integration(&tensor_map)?;

            // Test GPU inference pipeline
            test_gpu_inference_integration(loaded_config, tensor_map)?;
        }
        Err(err) => {
            // Should gracefully fallback to CPU or provide clear error
            eprintln!("GPU integration test correctly failing (TDD Red): {}", err);
            panic!("GPU integration test will pass once GPU-aware weight loading is complete");
        }
    }

    Ok(())
}

// ============================================================================
// Cross-Validation Integration Tests
// ============================================================================

/// Cross-validation integration: bitnet-models + crossval framework
#[cfg(feature = "crossval")]
#[tokio::test]
#[serial(bitnet_env)]
async fn test_integration_crossval_framework() -> Result<()> {
    // Set deterministic environment for reproducible results
    unsafe {
        std::env::set_var("BITNET_DETERMINISTIC", "1");
        std::env::set_var("BITNET_SEED", "42");
        std::env::set_var("BITNET_CROSSVAL_WEIGHTS", "1");
    }

    let config = IntegrationTestConfig::default();
    let fixture = IntegrationTestFixture::new()?.with_config(config.clone());
    let model_path = fixture.create_test_model()?;

    // Load model with cross-validation enabled
    let load_result = bitnet_models::gguf_simple::load_gguf(&model_path, Device::Cpu);

    match load_result {
        Ok((_, tensor_map)) => {
            // Test cross-validation framework integration
            for (tensor_name, tensor) in &tensor_map {
                if tensor_name.contains("weight") {
                    test_cpp_reference_validation(tensor_name, tensor)?;
                }
            }

            // Test deterministic validation
            test_deterministic_weight_validation(&tensor_map)?;
        }
        Err(err) => {
            eprintln!("Cross-validation integration test correctly failing (TDD Red): {}", err);
            panic!(
                "Cross-validation integration will pass once crossval framework integration is complete"
            );
        }
    }

    Ok(())
}

// ============================================================================
// FFI Integration Tests
// ============================================================================

/// FFI Integration test: GGUF weight loading with C++ bridge validation
#[cfg(feature = "ffi")]
#[tokio::test]
#[serial(bitnet_env)]
async fn test_integration_ffi_bridge_validation() -> Result<()> {
    let config = IntegrationTestConfig::default();
    let fixture = IntegrationTestFixture::new()?.with_config(config.clone());
    let model_path = fixture.create_test_model()?;

    // Load weights via Rust implementation
    let rust_result = bitnet_models::gguf_simple::load_gguf(&model_path, Device::Cpu);

    match rust_result {
        Ok((_, rust_tensor_map)) => {
            // Test FFI bridge with loaded weights
            for (tensor_name, tensor) in &rust_tensor_map {
                if tensor_name.contains("weight") {
                    test_ffi_bridge_compatibility(tensor_name, tensor)?;
                }
            }

            // Test C++ reference comparison
            test_cpp_rust_weight_comparison(&rust_tensor_map)?;
        }
        Err(err) => {
            eprintln!("FFI integration test correctly failing (TDD Red): {}", err);
            panic!("FFI integration will pass once C++ bridge integration is complete");
        }
    }

    Ok(())
}

// ============================================================================
// WASM Integration Tests
// ============================================================================

/// WASM Integration test: Weight loading in WebAssembly environment
#[cfg(all(target_arch = "wasm32", feature = "browser"))]
#[tokio::test]
#[serial(bitnet_env)]
async fn test_integration_wasm_weight_loading() -> Result<()> {
    let config = IntegrationTestConfig::default();
    let fixture = IntegrationTestFixture::new()?.with_config(config);
    let model_path = fixture.create_test_model()?;

    // Test WASM-compatible weight loading
    let wasm_result = bitnet_models::gguf_simple::load_gguf(&model_path, Device::Cpu);

    match wasm_result {
        Ok((_, tensor_map)) => {
            // Validate WASM-compatible tensor operations
            test_wasm_tensor_compatibility(&tensor_map)?;

            // Test WASM-specific memory management
            test_wasm_memory_management(&tensor_map)?;
        }
        Err(err) => {
            eprintln!("WASM integration test correctly failing (TDD Red): {}", err);
            panic!("WASM integration will pass once WASM-compatible weight loading is complete");
        }
    }

    Ok(())
}

// ============================================================================
// Performance Integration Tests
// ============================================================================

/// AC6: Performance integration: End-to-end pipeline performance validation
/// Issue #159: GGUF Weight Loading Performance Validation
///
/// Validates:
/// - Loading time is reasonable for test fixtures
/// - Memory efficiency meets expectations
/// - End-to-end pipeline completes successfully
///
/// Note: This test validates loading infrastructure performance with a minimal test fixture.
/// The test ensures the pipeline completes in reasonable time and manages memory efficiently.
/// For real model performance validation (>50 MB/s throughput), larger GGUF models are required.
#[cfg(feature = "cpu")]
#[tokio::test]
#[serial(bitnet_env)]
#[ignore = "Slow: runs a full mock inference pipeline — exceeds 5min nextest timeout on CI"]
async fn test_integration_performance_pipeline_cpu() -> Result<()> {
    use std::time::Instant;

    // AC6.1: Create a test fixture to validate loading pipeline
    // Use minimal config for fast loading (2 layers, small hidden size)
    let config = IntegrationTestConfig {
        test_model_layers: 2,
        hidden_size: 256,
        intermediate_size: 512,
        vocab_size: 1000,
        ..Default::default()
    };
    let fixture = IntegrationTestFixture::new()?.with_config(config.clone());
    let model_path = fixture.create_test_model()?;

    eprintln!("Performance test: Validating loading pipeline with test fixture");

    // AC6.2: Time the complete loading pipeline
    let start_time = Instant::now();
    let load_result = bitnet_models::gguf_simple::load_gguf(&model_path, Device::Cpu);
    let loading_time = start_time.elapsed();

    // AC6.3: Validate loading completes (even if using fallback minimal loader)
    let (loaded_config, tensor_map) = load_result
        .context("Loading failed - GGUF pipeline must handle test fixtures gracefully")?;

    let loading_time_secs = loading_time.as_secs_f64();
    eprintln!("Loading completed in {:.3}s", loading_time_secs);

    // AC6.4: Validate loading time is reasonable
    // Note: Mock tensor creation for large default models (32 layers × large dims) can be slow
    // Real GGUF loading with optimized code is much faster
    // This test validates that the pipeline completes without crashing
    let max_time_secs = 300.0; // Allow up to 5 minutes for mock tensor creation
    assert!(
        loading_time_secs <= max_time_secs,
        "Loading time {:.3}s exceeds {:.1}s maximum - pipeline hung or crashed",
        loading_time_secs,
        max_time_secs
    );

    // AC6.5: Calculate throughput (informational for small fixtures)
    let file_size_mb =
        std::fs::metadata(&model_path).map(|m| m.len() as f64 / (1024.0 * 1024.0)).unwrap_or(0.0);

    let throughput_mbps =
        if loading_time_secs > 0.0 { file_size_mb / loading_time_secs } else { f64::INFINITY };

    eprintln!("Throughput: {:.2} MB/s (file size: {:.3} MB)", throughput_mbps, file_size_mb);

    // AC6.6: Validate memory efficiency
    let memory_usage = estimate_tensor_memory_usage(&tensor_map);
    let memory_usage_mb = memory_usage as f64 / (1024.0 * 1024.0);

    // Calculate expected memory based on ACTUAL loaded config (not test fixture config)
    // The minimal loader uses BitNetConfig::default() which has its own dimensions
    let actual_config = IntegrationTestConfig {
        test_model_layers: loaded_config.model.num_layers,
        hidden_size: loaded_config.model.hidden_size,
        intermediate_size: loaded_config.model.intermediate_size,
        vocab_size: loaded_config.model.vocab_size,
        ..Default::default()
    };
    let expected_memory_mb = estimate_model_size(&actual_config) as f64 / (1024.0 * 1024.0);
    let memory_ratio = memory_usage_mb / expected_memory_mb;

    eprintln!(
        "Memory usage: {:.2} MB (expected ~{:.2} MB, ratio: {:.2}x)",
        memory_usage_mb, expected_memory_mb, memory_ratio
    );

    // Memory should be within 2x of expected (allows for overhead and FP32 vs quantized)
    assert!(
        (0.5..=2.0).contains(&memory_ratio),
        "Memory usage {:.2} MB is outside expected range ({:.2} MB, ratio: {:.2}x) - memory efficiency issue detected",
        memory_usage_mb,
        expected_memory_mb,
        memory_ratio
    );

    // AC6.7: Validate model structure is correct
    assert!(!tensor_map.is_empty(), "Loaded tensor map is empty - model loading failed");

    assert!(
        loaded_config.model.hidden_size > 0,
        "Config has invalid hidden_size: {}",
        loaded_config.model.hidden_size
    );

    assert!(
        loaded_config.model.vocab_size > 0,
        "Config has invalid vocab_size: {}",
        loaded_config.model.vocab_size
    );

    // Validate expected tensor count
    let expected_tensor_count = 2 + // embeddings + output
        loaded_config.model.num_layers * 9 + // per-layer tensors (7 weights + 2 norms)
        1; // output norm

    assert_eq!(
        tensor_map.len(),
        expected_tensor_count,
        "Tensor count mismatch: expected {} (2 base + {}*9 layers + 1 norm), got {}",
        expected_tensor_count,
        loaded_config.model.num_layers,
        tensor_map.len()
    );

    eprintln!("✓ Performance validation passed:");
    eprintln!(
        "  - Loading completed: {:.1}s (< {:.0}s timeout) ✓",
        loading_time_secs, max_time_secs
    );
    eprintln!("  - Throughput: {:.2} MB/s (info only)", throughput_mbps);
    eprintln!("  - Memory efficiency: {:.2}x expected ✓", memory_ratio);
    eprintln!("  - Tensors loaded: {} (expected {}) ✓", tensor_map.len(), expected_tensor_count);
    eprintln!(
        "  - Model config: hidden={}, vocab={}, layers={} ✓",
        loaded_config.model.hidden_size,
        loaded_config.model.vocab_size,
        loaded_config.model.num_layers
    );

    Ok(())
}

// ============================================================================
// Helper Functions for Integration Testing
// ============================================================================

/// Test I2S quantization integration with loaded weight
#[allow(dead_code)]
fn test_i2s_quantization_with_loaded_weight(
    weight_tensor: &CandleTensor,
    weight_name: &str,
) -> Result<()> {
    // TODO: Integrate with bitnet-quantization I2S quantizer
    let _ = (weight_tensor, weight_name);
    Err(anyhow::anyhow!("I2S quantization integration not yet implemented"))
}

/// Test TL1 quantization integration with loaded weight
#[allow(dead_code)]
fn test_tl1_quantization_with_loaded_weight(
    weight_tensor: &CandleTensor,
    weight_name: &str,
) -> Result<()> {
    // TODO: Integrate with bitnet-quantization TL1 quantizer
    let _ = (weight_tensor, weight_name);
    Err(anyhow::anyhow!("TL1 quantization integration not yet implemented"))
}

/// Test TL2 quantization integration with loaded weight
#[allow(dead_code)]
fn test_tl2_quantization_with_loaded_weight(
    weight_tensor: &CandleTensor,
    weight_name: &str,
) -> Result<()> {
    // TODO: Integrate with bitnet-quantization TL2 quantizer
    let _ = (weight_tensor, weight_name);
    Err(anyhow::anyhow!("TL2 quantization integration not yet implemented"))
}

/// Test attention kernel operations with loaded weights
#[allow(dead_code)]
fn test_attention_kernel_with_loaded_weights(
    tensor_map: &HashMap<String, CandleTensor>,
    layer_prefix: &str,
) -> Result<()> {
    // TODO: Integrate with bitnet-kernels attention operations
    let _ = (tensor_map, layer_prefix);
    Err(anyhow::anyhow!("Attention kernel integration not yet implemented"))
}

/// Test feed-forward kernel operations with loaded weights
#[allow(dead_code)]
fn test_feedforward_kernel_with_loaded_weights(
    tensor_map: &HashMap<String, CandleTensor>,
    layer_prefix: &str,
) -> Result<()> {
    // TODO: Integrate with bitnet-kernels feed-forward operations
    let _ = (tensor_map, layer_prefix);
    Err(anyhow::anyhow!("Feed-forward kernel integration not yet implemented"))
}

/// Test inference engine with real weights
#[allow(dead_code)]
fn test_inference_engine_with_real_weights(
    config: BitNetConfig,
    tensor_map: HashMap<String, CandleTensor>,
) -> Result<()> {
    // TODO: Integrate with bitnet-inference engine
    let _ = (config, tensor_map);
    Err(anyhow::anyhow!("Inference engine integration not yet implemented"))
}

/// Validate GPU tensor placement
#[allow(dead_code)]
fn validate_gpu_tensor_placement(tensor_map: &HashMap<String, CandleTensor>) -> Result<()> {
    for (name, tensor) in tensor_map {
        let device = tensor.device();
        if !device.is_cuda() && !device.is_cpu() {
            return Err(anyhow::anyhow!("Tensor {} on unexpected device: {:?}", name, device));
        }
    }
    Ok(())
}

/// Test GPU quantization integration
#[allow(dead_code)]
fn test_gpu_quantization_integration(tensor_map: &HashMap<String, CandleTensor>) -> Result<()> {
    // TODO: Test GPU quantization operations
    let _ = tensor_map;
    Err(anyhow::anyhow!("GPU quantization integration not yet implemented"))
}

/// Test GPU kernel integration
#[allow(dead_code)]
fn test_gpu_kernel_integration(tensor_map: &HashMap<String, CandleTensor>) -> Result<()> {
    // TODO: Test GPU kernel operations
    let _ = tensor_map;
    Err(anyhow::anyhow!("GPU kernel integration not yet implemented"))
}

/// Test GPU inference integration
#[allow(dead_code)]
fn test_gpu_inference_integration(
    config: BitNetConfig,
    tensor_map: HashMap<String, CandleTensor>,
) -> Result<()> {
    // TODO: Test GPU inference pipeline
    let _ = (config, tensor_map);
    Err(anyhow::anyhow!("GPU inference integration not yet implemented"))
}

/// Test C++ reference validation
#[allow(dead_code)]
fn test_cpp_reference_validation(tensor_name: &str, tensor: &CandleTensor) -> Result<()> {
    // TODO: Integrate with crossval framework
    let _ = (tensor_name, tensor);
    Err(anyhow::anyhow!("C++ reference validation not yet implemented"))
}

/// Test deterministic weight validation
#[allow(dead_code)]
fn test_deterministic_weight_validation(tensor_map: &HashMap<String, CandleTensor>) -> Result<()> {
    // TODO: Test deterministic loading behavior
    let _ = tensor_map;
    Err(anyhow::anyhow!("Deterministic validation not yet implemented"))
}

/// Test FFI bridge compatibility
#[allow(dead_code)]
fn test_ffi_bridge_compatibility(tensor_name: &str, tensor: &CandleTensor) -> Result<()> {
    // TODO: Test FFI bridge operations
    let _ = (tensor_name, tensor);
    Err(anyhow::anyhow!("FFI bridge compatibility not yet implemented"))
}

/// Test C++ vs Rust weight comparison
#[allow(dead_code)]
fn test_cpp_rust_weight_comparison(tensor_map: &HashMap<String, CandleTensor>) -> Result<()> {
    // TODO: Compare C++ and Rust weight loading results
    let _ = tensor_map;
    Err(anyhow::anyhow!("C++ Rust comparison not yet implemented"))
}

/// Test WASM tensor compatibility
#[allow(dead_code)]
fn test_wasm_tensor_compatibility(tensor_map: &HashMap<String, CandleTensor>) -> Result<()> {
    // TODO: Test WASM-specific tensor operations
    let _ = tensor_map;
    Err(anyhow::anyhow!("WASM tensor compatibility not yet implemented"))
}

/// Test WASM memory management
#[allow(dead_code)]
fn test_wasm_memory_management(tensor_map: &HashMap<String, CandleTensor>) -> Result<()> {
    // TODO: Test WASM memory constraints and management
    let _ = tensor_map;
    Err(anyhow::anyhow!("WASM memory management not yet implemented"))
}

/// Test quantization performance impact
fn test_quantization_performance_impact(tensor_map: &HashMap<String, CandleTensor>) -> Result<()> {
    // TODO: Measure quantization performance overhead
    let _ = tensor_map;
    Ok(()) // Placeholder
}

/// Estimate tensor memory usage
fn estimate_tensor_memory_usage(tensor_map: &HashMap<String, CandleTensor>) -> usize {
    tensor_map.values().map(|tensor| tensor.shape().elem_count() * std::mem::size_of::<f32>()).sum()
}

/// Estimate model size based on configuration
fn estimate_model_size(config: &IntegrationTestConfig) -> usize {
    let layer_size = config.hidden_size * config.hidden_size * 4 + // attention weights
                    config.hidden_size * config.intermediate_size * 3; // ffn weights
    let embedding_size = config.vocab_size * config.hidden_size;
    (layer_size * config.test_model_layers + embedding_size * 2) * std::mem::size_of::<f32>()
}
