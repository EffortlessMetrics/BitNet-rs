//! Integration test fixtures for multi-crate BitNet.rs testing.
//!
//! This module provides fixtures that span multiple workspace crates:
//! - bitnet-models + bitnet-quantization integration
//! - CPU/GPU device-aware test data
//! - Memory efficiency scenarios
//! - Performance baseline data
//!
//! Designed for BitNet.rs Issue #159 comprehensive test coverage.

use anyhow::{Context, Result};
use bitnet_common::{BitNetConfig, BitNetError, Device};
use candle_core::{DType, Device as CDevice, Tensor as CandleTensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Integration test configuration
#[derive(Debug, Clone)]
pub struct IntegrationTestConfig {
    pub use_deterministic_data: bool,
    pub test_seed: u64,
    pub memory_limit_mb: usize,
    pub performance_timeout_ms: u64,
    pub device_fallback_enabled: bool,
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            use_deterministic_data: true,
            test_seed: 42,
            memory_limit_mb: 512,
            performance_timeout_ms: 5000,
            device_fallback_enabled: true,
        }
    }
}

/// Multi-crate integration test fixture
#[derive(Debug)]
pub struct IntegrationFixture {
    pub name: String,
    pub model_path: PathBuf,
    pub quantization_types: Vec<String>,
    pub expected_tensors: HashMap<String, TensorSpec>,
    pub device_requirements: DeviceRequirements,
    pub memory_profile: MemoryProfile,
    pub performance_expectations: PerformanceProfile,
}

#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub quantization_type: Option<String>,
    pub device_placement: String,
    pub memory_alignment: usize,
}

#[derive(Debug, Clone)]
pub struct DeviceRequirements {
    pub cpu_supported: bool,
    pub gpu_supported: bool,
    pub fallback_behavior: String,
    pub minimum_memory_mb: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryProfile {
    pub loading_peak_mb: usize,
    pub runtime_steady_mb: usize,
    pub overhead_factor: f32,
    pub zero_copy_eligible: bool,
}

#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    pub loading_time_ms: u64,
    pub first_inference_ms: u64,
    pub subsequent_inference_ms: u64,
    pub memory_bandwidth_gb_s: f32,
}

/// Lazy-loaded integration test fixtures
static INTEGRATION_FIXTURES: LazyLock<Vec<IntegrationFixture>> =
    LazyLock::new(load_integration_fixtures);

/// Load all integration test fixtures
fn load_integration_fixtures() -> Vec<IntegrationFixture> {
    let fixtures_dir = get_fixtures_dir();

    vec![
        create_bitnet_models_quantization_fixture(&fixtures_dir),
        create_cpu_gpu_device_fixture(&fixtures_dir),
        create_memory_efficiency_fixture(&fixtures_dir),
        create_performance_baseline_fixture(&fixtures_dir),
        create_cross_crate_validation_fixture(&fixtures_dir),
    ]
}

/// Get integration fixtures directory
fn get_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("integration")
}

/// Create fixture for bitnet-models + bitnet-quantization integration
fn create_bitnet_models_quantization_fixture(fixtures_dir: &Path) -> IntegrationFixture {
    let model_path = fixtures_dir.parent().unwrap().join("gguf/valid/small_bitnet_test.gguf");

    let mut expected_tensors = HashMap::new();

    // Define expected tensors with quantization specifications
    expected_tensors.insert(
        "token_embd.weight".to_string(),
        TensorSpec {
            shape: vec![1000, 256],
            dtype: "F32".to_string(),
            quantization_type: Some("I2_S".to_string()),
            device_placement: "auto".to_string(),
            memory_alignment: 32,
        },
    );

    expected_tensors.insert(
        "output.weight".to_string(),
        TensorSpec {
            shape: vec![256, 1000],
            dtype: "F32".to_string(),
            quantization_type: Some("I2_S".to_string()),
            device_placement: "auto".to_string(),
            memory_alignment: 32,
        },
    );

    // Add transformer layer tensors
    for layer_idx in 0..2 {
        let layer_prefix = format!("blk.{}", layer_idx);

        for weight_name in
            &["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"]
        {
            expected_tensors.insert(
                format!("{}.{}", layer_prefix, weight_name),
                TensorSpec {
                    shape: vec![256, 256],
                    dtype: "F32".to_string(),
                    quantization_type: Some("I2_S".to_string()),
                    device_placement: "auto".to_string(),
                    memory_alignment: 32,
                },
            );
        }

        for weight_name in &["ffn_gate.weight", "ffn_up.weight"] {
            expected_tensors.insert(
                format!("{}.{}", layer_prefix, weight_name),
                TensorSpec {
                    shape: vec![512, 256],
                    dtype: "F32".to_string(),
                    quantization_type: Some("TL1".to_string()),
                    device_placement: "auto".to_string(),
                    memory_alignment: 32,
                },
            );
        }

        expected_tensors.insert(
            format!("{}.ffn_down.weight", layer_prefix),
            TensorSpec {
                shape: vec![256, 512],
                dtype: "F32".to_string(),
                quantization_type: Some("TL1".to_string()),
                device_placement: "auto".to_string(),
                memory_alignment: 32,
            },
        );
    }

    IntegrationFixture {
        name: "models_quantization_integration".to_string(),
        model_path,
        quantization_types: vec!["I2_S".to_string(), "TL1".to_string(), "TL2".to_string()],
        expected_tensors,
        device_requirements: DeviceRequirements {
            cpu_supported: true,
            gpu_supported: true,
            fallback_behavior: "cpu_on_gpu_failure".to_string(),
            minimum_memory_mb: 64,
        },
        memory_profile: MemoryProfile {
            loading_peak_mb: 32,
            runtime_steady_mb: 16,
            overhead_factor: 1.2,
            zero_copy_eligible: true,
        },
        performance_expectations: PerformanceProfile {
            loading_time_ms: 100,
            first_inference_ms: 50,
            subsequent_inference_ms: 10,
            memory_bandwidth_gb_s: 2.0,
        },
    }
}

/// Create fixture for CPU/GPU device-aware testing
fn create_cpu_gpu_device_fixture(fixtures_dir: &Path) -> IntegrationFixture {
    let model_path = fixtures_dir.parent().unwrap().join("gguf/valid/minimal_bitnet_i2s.gguf");

    let mut expected_tensors = HashMap::new();

    // Define device-specific tensor placements
    expected_tensors.insert(
        "token_embd.weight".to_string(),
        TensorSpec {
            shape: vec![32000, 2048],
            dtype: "F32".to_string(),
            quantization_type: Some("I2_S".to_string()),
            device_placement: "gpu_preferred".to_string(),
            memory_alignment: 32,
        },
    );

    IntegrationFixture {
        name: "cpu_gpu_device_aware".to_string(),
        model_path,
        quantization_types: vec!["I2_S".to_string()],
        expected_tensors,
        device_requirements: DeviceRequirements {
            cpu_supported: true,
            gpu_supported: true,
            fallback_behavior: "graceful_cpu_fallback".to_string(),
            minimum_memory_mb: 512,
        },
        memory_profile: MemoryProfile {
            loading_peak_mb: 256,
            runtime_steady_mb: 128,
            overhead_factor: 1.5,
            zero_copy_eligible: true,
        },
        performance_expectations: PerformanceProfile {
            loading_time_ms: 500,
            first_inference_ms: 100,
            subsequent_inference_ms: 20,
            memory_bandwidth_gb_s: 10.0, // GPU bandwidth
        },
    }
}

/// Create fixture for memory efficiency testing
fn create_memory_efficiency_fixture(fixtures_dir: &Path) -> IntegrationFixture {
    let model_path = fixtures_dir.parent().unwrap().join("gguf/valid/small_bitnet_test.gguf");

    IntegrationFixture {
        name: "memory_efficiency_test".to_string(),
        model_path,
        quantization_types: vec!["I2_S".to_string()],
        expected_tensors: HashMap::new(), // Focus on memory, not specific tensors
        device_requirements: DeviceRequirements {
            cpu_supported: true,
            gpu_supported: false, // CPU-only for memory testing
            fallback_behavior: "cpu_only".to_string(),
            minimum_memory_mb: 16,
        },
        memory_profile: MemoryProfile {
            loading_peak_mb: 20,
            runtime_steady_mb: 16,
            overhead_factor: 1.1, // Very efficient
            zero_copy_eligible: true,
        },
        performance_expectations: PerformanceProfile {
            loading_time_ms: 50,
            first_inference_ms: 25,
            subsequent_inference_ms: 5,
            memory_bandwidth_gb_s: 1.0,
        },
    }
}

/// Create fixture for performance baseline testing
fn create_performance_baseline_fixture(fixtures_dir: &Path) -> IntegrationFixture {
    let model_path = fixtures_dir.parent().unwrap().join("gguf/valid/minimal_bitnet_i2s.gguf");

    IntegrationFixture {
        name: "performance_baseline".to_string(),
        model_path,
        quantization_types: vec!["I2_S".to_string(), "TL1".to_string(), "TL2".to_string()],
        expected_tensors: HashMap::new(),
        device_requirements: DeviceRequirements {
            cpu_supported: true,
            gpu_supported: true,
            fallback_behavior: "performance_optimized".to_string(),
            minimum_memory_mb: 256,
        },
        memory_profile: MemoryProfile {
            loading_peak_mb: 300,
            runtime_steady_mb: 200,
            overhead_factor: 1.3,
            zero_copy_eligible: true,
        },
        performance_expectations: PerformanceProfile {
            loading_time_ms: 200,
            first_inference_ms: 80,
            subsequent_inference_ms: 15,
            memory_bandwidth_gb_s: 8.0,
        },
    }
}

/// Create fixture for cross-crate validation
fn create_cross_crate_validation_fixture(fixtures_dir: &Path) -> IntegrationFixture {
    let model_path = fixtures_dir.parent().unwrap().join("gguf/valid/small_bitnet_test.gguf");

    IntegrationFixture {
        name: "cross_crate_validation".to_string(),
        model_path,
        quantization_types: vec!["I2_S".to_string()],
        expected_tensors: HashMap::new(),
        device_requirements: DeviceRequirements {
            cpu_supported: true,
            gpu_supported: true,
            fallback_behavior: "validation_mode".to_string(),
            minimum_memory_mb: 32,
        },
        memory_profile: MemoryProfile {
            loading_peak_mb: 40,
            runtime_steady_mb: 24,
            overhead_factor: 1.2,
            zero_copy_eligible: true,
        },
        performance_expectations: PerformanceProfile {
            loading_time_ms: 100,
            first_inference_ms: 40,
            subsequent_inference_ms: 8,
            memory_bandwidth_gb_s: 3.0,
        },
    }
}

/// Get all integration test fixtures
pub fn get_integration_fixtures() -> &'static [IntegrationFixture] {
    &INTEGRATION_FIXTURES
}

/// Get specific integration fixture by name
pub fn get_fixture_by_name(name: &str) -> Option<&'static IntegrationFixture> {
    INTEGRATION_FIXTURES.iter().find(|f| f.name == name)
}

/// Validate fixture model file exists
pub fn validate_fixture_files() -> Result<()> {
    for fixture in INTEGRATION_FIXTURES.iter() {
        if !fixture.model_path.exists() {
            return Err(anyhow::anyhow!(
                "Fixture model file not found: {}",
                fixture.model_path.display()
            ));
        }
    }
    Ok(())
}

/// Load fixture model for testing
pub fn load_fixture_model(
    fixture: &IntegrationFixture,
    device: Device,
) -> Result<(BitNetConfig, HashMap<String, CandleTensor>)> {
    // Use the existing GGUF loader from bitnet-models
    bitnet_models::gguf_simple::load_gguf(&fixture.model_path, device)
        .context("Failed to load fixture model")
}

/// Validate tensor against fixture specification
pub fn validate_tensor_spec(
    tensor_name: &str,
    tensor: &CandleTensor,
    spec: &TensorSpec,
) -> Result<()> {
    // Validate shape
    if tensor.shape().dims() != spec.shape {
        return Err(anyhow::anyhow!(
            "Tensor {} shape mismatch: expected {:?}, got {:?}",
            tensor_name,
            spec.shape,
            tensor.shape().dims()
        ));
    }

    // Validate device placement
    let device = tensor.device();
    match spec.device_placement.as_str() {
        "cpu" => {
            if !device.is_cpu() {
                return Err(anyhow::anyhow!(
                    "Tensor {} should be on CPU, found: {:?}",
                    tensor_name,
                    device
                ));
            }
        }
        "gpu" | "cuda" => {
            if !device.is_cuda() {
                return Err(anyhow::anyhow!(
                    "Tensor {} should be on GPU, found: {:?}",
                    tensor_name,
                    device
                ));
            }
        }
        "auto" | "gpu_preferred" => {
            // Either CPU or GPU is acceptable
            if !device.is_cpu() && !device.is_cuda() {
                return Err(anyhow::anyhow!(
                    "Tensor {} should be on CPU or GPU, found: {:?}",
                    tensor_name,
                    device
                ));
            }
        }
        _ => {
            // Unknown device placement requirement
            return Err(anyhow::anyhow!(
                "Unknown device placement requirement: {}",
                spec.device_placement
            ));
        }
    }

    Ok(())
}

/// Check memory usage against fixture profile
pub fn check_memory_usage(fixture: &IntegrationFixture) -> Result<()> {
    // Get current process memory usage (simplified - would need proper implementation)
    let current_memory_mb = get_process_memory_usage_mb()?;

    if current_memory_mb > fixture.memory_profile.loading_peak_mb + 50 {
        // 50MB tolerance
        return Err(anyhow::anyhow!(
            "Memory usage {} MB exceeds fixture {} peak {} MB",
            current_memory_mb,
            fixture.name,
            fixture.memory_profile.loading_peak_mb
        ));
    }

    Ok(())
}

/// Get process memory usage in MB (placeholder implementation)
fn get_process_memory_usage_mb() -> Result<usize> {
    // TODO: Implement cross-platform memory usage detection
    // For now, return a placeholder value
    Ok(32) // 32MB placeholder
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_fixtures_loaded() {
        let fixtures = get_integration_fixtures();
        assert!(!fixtures.is_empty());

        // Verify we have expected fixture types
        let fixture_names: Vec<&str> = fixtures.iter().map(|f| f.name.as_str()).collect();
        assert!(fixture_names.contains(&"models_quantization_integration"));
        assert!(fixture_names.contains(&"cpu_gpu_device_aware"));
        assert!(fixture_names.contains(&"memory_efficiency_test"));
    }

    #[test]
    fn test_fixture_by_name() {
        let fixture = get_fixture_by_name("models_quantization_integration");
        assert!(fixture.is_some());
        assert_eq!(fixture.unwrap().name, "models_quantization_integration");

        let missing = get_fixture_by_name("non_existent_fixture");
        assert!(missing.is_none());
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_tensor_spec_validation() {
        use candle_core::{Device as CDevice, Tensor as CandleTensor};

        let device = CDevice::Cpu;
        let tensor = CandleTensor::zeros((100, 100), candle_core::DType::F32, &device).unwrap();

        let spec = TensorSpec {
            shape: vec![100, 100],
            dtype: "F32".to_string(),
            quantization_type: None,
            device_placement: "cpu".to_string(),
            memory_alignment: 32,
        };

        let result = validate_tensor_spec("test_tensor", &tensor, &spec);
        assert!(result.is_ok());

        // Test shape mismatch
        let bad_spec = TensorSpec {
            shape: vec![50, 50],
            dtype: "F32".to_string(),
            quantization_type: None,
            device_placement: "cpu".to_string(),
            memory_alignment: 32,
        };

        let result = validate_tensor_spec("test_tensor", &tensor, &bad_spec);
        assert!(result.is_err());
    }
}
