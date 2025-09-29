//! Test Fixtures for BitNet.rs Inference Server
//!
//! This module provides comprehensive test fixtures for the production inference
//! server including models, quantization data, HTTP requests/responses, and
//! deployment configurations. All fixtures support feature-gated compilation
//! and BitNet.rs neural network testing patterns.

pub mod deployment;
pub mod models;
pub mod quantization;
pub mod requests;
pub mod responses;

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Once;

pub use deployment::EnvironmentConfig;
/// Re-export key fixture types for convenience
pub use models::{GgufTestModel, QuantizationType as ModelQuantizationType, get_all_test_models};
#[cfg(feature = "crossval")]
pub use quantization::CrossValidationFixture;
pub use quantization::{DeviceType, QuantizationTestVector};
pub use requests::InferenceRequest;
pub use responses::InferenceResponse;

static INIT: Once = Once::new();

/// Initialize test fixtures (call once per test session)
pub fn init_fixtures() {
    INIT.call_once(|| {
        // Set deterministic environment for reproducible tests
        unsafe {
            std::env::set_var("BITNET_DETERMINISTIC", "1");
            std::env::set_var("BITNET_SEED", "42");
        }

        // Configure logging for tests
        if std::env::var("RUST_LOG").is_err() {
            unsafe {
                std::env::set_var("RUST_LOG", "bitnet_server=info,warn");
            }
        }

        // Set reasonable thread count for CI
        if std::env::var("RAYON_NUM_THREADS").is_err() {
            unsafe {
                std::env::set_var("RAYON_NUM_THREADS", "2");
            }
        }
    });
}

/// Fixture loading utilities with feature-gated compilation
pub struct FixtureLoader;

impl FixtureLoader {
    /// Get project root directory for fixture paths
    pub fn project_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
    }

    /// Get fixture directory path
    pub fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
    }

    /// Load model fixtures based on feature flags
    #[cfg(feature = "cpu")]
    pub fn load_cpu_models() -> Vec<&'static GgufTestModel> {
        init_fixtures();
        models::get_cpu_compatible_models()
    }

    #[cfg(feature = "gpu")]
    pub fn load_gpu_models() -> Vec<&'static GgufTestModel> {
        init_fixtures();
        models::get_gpu_compatible_models()
    }

    /// Load quantization test vectors based on device capabilities
    #[cfg(feature = "cpu")]
    pub fn load_cpu_quantization_vectors() -> Vec<&'static QuantizationTestVector> {
        init_fixtures();
        quantization::get_cpu_test_vectors()
    }

    #[cfg(feature = "gpu")]
    pub fn load_gpu_quantization_vectors() -> Vec<&'static QuantizationTestVector> {
        init_fixtures();
        quantization::get_gpu_test_vectors()
    }

    /// Load cross-validation fixtures (requires crossval feature)
    #[cfg(feature = "crossval")]
    pub fn load_crossval_fixtures() -> Vec<&'static CrossValidationFixture> {
        init_fixtures();
        quantization::get_crossval_fixtures()
    }

    /// Load HTTP request fixtures for API testing
    pub fn load_basic_requests() -> Vec<&'static str> {
        init_fixtures();
        requests::get_all_basic_request_names()
    }

    /// Load error condition request fixtures
    pub fn load_error_requests() -> Vec<&'static str> {
        init_fixtures();
        requests::get_all_error_request_names()
    }

    /// Load security test request fixtures
    pub fn load_security_requests() -> Vec<&'static str> {
        init_fixtures();
        requests::get_all_security_request_names()
    }

    /// Load deployment configuration fixtures
    pub fn load_docker_configs() -> Vec<&'static str> {
        init_fixtures();
        deployment::get_all_docker_configs()
    }

    pub fn load_kubernetes_configs() -> Vec<&'static str> {
        init_fixtures();
        deployment::get_all_kubernetes_configs()
    }

    pub fn load_environment_configs() -> Vec<&'static str> {
        init_fixtures();
        deployment::get_all_environment_configs()
    }

    /// Load performance benchmark fixtures
    pub fn load_performance_benchmarks() -> Vec<&'static str> {
        init_fixtures();
        deployment::get_all_performance_benchmarks()
    }

    /// Generate concurrent request fixtures for load testing
    pub fn generate_load_test_requests(
        count: usize,
        base_prompt: &str,
    ) -> Vec<requests::InferenceRequest> {
        init_fixtures();
        requests::generate_concurrent_requests(count, base_prompt)
    }

    /// Validate fixture file existence (for actual file fixtures)
    pub fn validate_fixture_files() -> Result<()> {
        let fixture_dir = Self::fixture_dir();

        // Check that fixture directory exists
        if !fixture_dir.exists() {
            return Err(anyhow::anyhow!("Fixture directory does not exist: {:?}", fixture_dir));
        }

        // Note: Actual GGUF model files are not included in git due to size
        // In a real deployment, these would be downloaded or generated
        println!("Fixture directory validated: {:?}", fixture_dir);

        Ok(())
    }
}

/// Test fixture selection utilities
pub struct FixtureSelector;

impl FixtureSelector {
    /// Select appropriate model fixture based on test requirements
    pub fn select_model_for_test(
        device_preference: Option<&str>,
        quantization_type: Option<ModelQuantizationType>,
        memory_limit_mb: Option<u64>,
    ) -> Option<&'static GgufTestModel> {
        let models = get_all_test_models();

        for model in models {
            // Check device compatibility
            if let Some(device) = device_preference {
                match device {
                    "cpu" => {
                        #[cfg(not(feature = "cpu"))]
                        continue;
                    }
                    "gpu" => {
                        #[cfg(not(feature = "gpu"))]
                        continue;
                    }
                    _ => {}
                }
            }

            // Check quantization type match
            if let Some(quant_type) = &quantization_type {
                if &model.quantization_type != quant_type {
                    continue;
                }
            }

            // Check memory requirements
            if let Some(limit) = memory_limit_mb {
                if model.model_size_bytes > limit * 1024 * 1024 {
                    continue;
                }
            }

            return Some(model);
        }

        None
    }

    /// Select quantization test vectors for specific test scenario
    pub fn select_quantization_vectors_for_scenario(
        scenario: &str,
    ) -> Vec<&'static QuantizationTestVector> {
        match scenario {
            "accuracy_validation" => {
                let mut vectors = Vec::new();
                vectors.extend(quantization::get_quantization_vectors(
                    quantization::QuantizationType::I2S,
                ));
                vectors.into_iter().filter(|v| v.accuracy_target >= 0.99).collect()
            }
            "performance_benchmarking" => {
                let mut vectors = Vec::new();
                vectors.extend(quantization::get_quantization_vectors(
                    quantization::QuantizationType::TL1,
                ));
                vectors.extend(quantization::get_quantization_vectors(
                    quantization::QuantizationType::TL2,
                ));
                vectors.into_iter().filter(|v| v.input_data.len() >= 256).collect()
            }
            "edge_case_testing" => quantization::get_edge_case_vectors(),
            _ => {
                // Default: return all I2S vectors
                quantization::get_quantization_vectors(quantization::QuantizationType::I2S)
            }
        }
    }

    /// Select appropriate deployment configuration for environment
    pub fn select_deployment_config_for_env(env: &str) -> Option<&'static EnvironmentConfig> {
        deployment::get_environment_config(env)
    }
}

/// Fixture validation utilities
pub struct FixtureValidator;

impl FixtureValidator {
    /// Validate quantization test vector accuracy
    pub fn validate_quantization_accuracy(vector: &QuantizationTestVector) -> Result<f32> {
        let accuracy = quantization::validate_quantization_accuracy(
            &vector.input_data,
            &vector.expected_quantized,
            &vector.expected_scales,
            vector.tolerance,
        );

        if accuracy < vector.accuracy_target {
            return Err(anyhow::anyhow!(
                "Quantization accuracy {} below target {} for vector '{}'",
                accuracy,
                vector.accuracy_target,
                vector.name
            ));
        }

        Ok(accuracy)
    }

    /// Validate HTTP request fixture structure
    pub fn validate_request_fixture(request: &InferenceRequest) -> Result<()> {
        if request.prompt.is_empty() {
            return Err(anyhow::anyhow!("Request prompt cannot be empty"));
        }

        if let Some(max_tokens) = request.max_tokens {
            if max_tokens == 0 {
                return Err(anyhow::anyhow!("Max tokens must be greater than 0"));
            }
        }

        if let Some(temperature) = request.temperature {
            if temperature < 0.0 || temperature > 2.0 {
                return Err(anyhow::anyhow!(
                    "Temperature {} outside valid range [0.0, 2.0]",
                    temperature
                ));
            }
        }

        Ok(())
    }

    /// Validate response fixture structure
    pub fn validate_response_fixture(response: &InferenceResponse) -> Result<()> {
        if response.text.is_empty() {
            return Err(anyhow::anyhow!("Response text cannot be empty"));
        }

        if response.tokens_generated == 0 {
            return Err(anyhow::anyhow!("Tokens generated must be greater than 0"));
        }

        if response.tokens_per_second <= 0.0 {
            return Err(anyhow::anyhow!("Tokens per second must be positive"));
        }

        if response.accuracy_metrics.quantization_accuracy < 0.0
            || response.accuracy_metrics.quantization_accuracy > 1.0
        {
            return Err(anyhow::anyhow!(
                "Quantization accuracy {} outside valid range [0.0, 1.0]",
                response.accuracy_metrics.quantization_accuracy
            ));
        }

        Ok(())
    }
}

/// Convenience macros for fixture loading in tests
#[macro_export]
macro_rules! load_test_model {
    ($device:expr, $quant:expr) => {
        $crate::fixtures::FixtureSelector::select_model_for_test(Some($device), Some($quant), None)
            .expect("Failed to load test model")
    };
}

#[macro_export]
macro_rules! load_quantization_vectors {
    ($scenario:expr) => {
        $crate::fixtures::FixtureSelector::select_quantization_vectors_for_scenario($scenario)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixture_initialization() {
        init_fixtures();

        // Verify environment variables are set
        assert_eq!(std::env::var("BITNET_DETERMINISTIC").unwrap(), "1");
        assert_eq!(std::env::var("BITNET_SEED").unwrap(), "42");
    }

    #[test]
    fn test_fixture_loader_basic_functionality() {
        let basic_requests = FixtureLoader::load_basic_requests();
        assert!(!basic_requests.is_empty());

        let error_requests = FixtureLoader::load_error_requests();
        assert!(!error_requests.is_empty());

        let docker_configs = FixtureLoader::load_docker_configs();
        assert!(!docker_configs.is_empty());
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_cpu_fixture_loading() {
        let cpu_models = FixtureLoader::load_cpu_models();
        assert!(!cpu_models.is_empty());

        let cpu_vectors = FixtureLoader::load_cpu_quantization_vectors();
        assert!(!cpu_vectors.is_empty());

        for vector in cpu_vectors {
            assert_eq!(vector.device_type, DeviceType::CPU);
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_fixture_loading() {
        let gpu_models = FixtureLoader::load_gpu_models();
        assert!(!gpu_models.is_empty());

        let gpu_vectors = FixtureLoader::load_gpu_quantization_vectors();
        assert!(!gpu_vectors.is_empty());

        for vector in gpu_vectors {
            assert_eq!(vector.device_type, DeviceType::GPU);
        }
    }

    #[test]
    fn test_model_selection() {
        let cpu_model = FixtureSelector::select_model_for_test(
            Some("cpu"),
            Some(ModelQuantizationType::I2S),
            Some(100), // 100MB limit
        );
        assert!(cpu_model.is_some());

        let model = cpu_model.unwrap();
        assert_eq!(model.quantization_type, ModelQuantizationType::I2S);
        assert!(model.model_size_bytes <= 100 * 1024 * 1024);
    }

    #[test]
    fn test_quantization_vector_selection() {
        let accuracy_vectors =
            FixtureSelector::select_quantization_vectors_for_scenario("accuracy_validation");
        assert!(!accuracy_vectors.is_empty());

        for vector in accuracy_vectors {
            assert!(vector.accuracy_target >= 0.99);
        }

        let edge_vectors =
            FixtureSelector::select_quantization_vectors_for_scenario("edge_case_testing");
        assert!(!edge_vectors.is_empty());

        for vector in edge_vectors {
            assert!(vector.name.starts_with("edge_"));
        }
    }

    #[test]
    fn test_fixture_validation() {
        // Test valid request
        let request = InferenceRequest {
            prompt: "Test prompt".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            device_preference: Some("cpu".to_string()),
            quantization_preference: Some("i2s".to_string()),
            stream: Some(false),
            model_id: None,
            stop_sequences: None,
            seed: Some(42),
            request_id: Some("test-001".to_string()),
            top_k: None,
        };

        assert!(FixtureValidator::validate_request_fixture(&request).is_ok());

        // Test invalid request (empty prompt)
        let invalid_request = InferenceRequest {
            prompt: "".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            device_preference: None,
            quantization_preference: None,
            stream: None,
            model_id: None,
            stop_sequences: None,
            seed: None,
            request_id: None,
            top_k: None,
        };

        assert!(FixtureValidator::validate_request_fixture(&invalid_request).is_err());
    }

    #[test]
    fn test_load_test_request_generation() {
        let requests = FixtureLoader::generate_load_test_requests(10, "Test load testing prompt");

        assert_eq!(requests.len(), 10);

        // Verify each request has unique characteristics
        for (i, request) in requests.iter().enumerate() {
            assert!(request.prompt.contains(&format!("Request {}", i + 1)));
            assert!(request.request_id.is_some());
        }
    }

    #[test]
    fn test_fixture_file_validation() {
        // This test validates the fixture directory structure exists
        let result = FixtureLoader::validate_fixture_files();
        assert!(result.is_ok());
    }

    #[cfg(feature = "crossval")]
    #[test]
    fn test_crossval_fixture_loading() {
        let crossval_fixtures = FixtureLoader::load_crossval_fixtures();
        assert!(!crossval_fixtures.is_empty());

        for fixture in crossval_fixtures {
            assert_eq!(fixture.rust_output.len(), fixture.cpp_reference.len());
            assert!(fixture.tolerance > 0.0);
        }
    }
}
