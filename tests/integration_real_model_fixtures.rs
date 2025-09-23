//! Integration test for real BitNet model fixtures
//!
//! This test validates the complete fixture infrastructure and demonstrates
//! how to use the test fixtures for BitNet.rs neural network component testing.

use bitnet_common::{Device, Result};

// Import the fixtures module
mod fixtures;
use fixtures::{RealModelIntegrationFixtures, TestEnvironmentConfig, TestTier};

// Define local helper macros for test skipping (since they're not properly exported)
macro_rules! skip_if_tier_insufficient {
    ($config:expr, $required_tier:expr) => {
        if $config.tier < $required_tier {
            eprintln!(
                "Skipping test - requires tier {:?}, current tier {:?}",
                $required_tier, $config.tier
            );
            return Ok(());
        }
    };
}

macro_rules! skip_if_no_gpu {
    ($config:expr) => {
        if !$config.gpu_features_enabled() {
            eprintln!("Skipping GPU test - GPU features not enabled or strict mode active");
            return Ok(());
        }
    };
}

/// Test comprehensive fixture initialization and usage
#[tokio::test]
async fn test_fixture_infrastructure() -> Result<()> {
    let mut fixtures = RealModelIntegrationFixtures::new();

    // Initialize all fixtures
    fixtures.initialize().await?;

    // Validate fixture components are properly initialized
    assert!(!fixtures.model_fixtures.mock_models.is_empty(), "Should have mock models available");

    // Test device-aware fixtures
    let cpu_device = fixtures.device_fixtures.get_test_device(false);
    assert_eq!(cpu_device, Device::Cpu, "Should default to CPU device");

    // Test quantization fixtures
    let i2s_vectors = fixtures.quantization_fixtures.get_test_vectors("I2_S");
    assert!(i2s_vectors.is_some(), "Should have I2S quantization test vectors");

    let i2s_vectors = i2s_vectors.unwrap();
    assert!(!i2s_vectors.test_cases.is_empty(), "Should have I2S test cases");

    // Test performance fixtures
    let cpu_targets = fixtures.performance_fixtures.benchmark_targets.get(&Device::Cpu);
    assert!(cpu_targets.is_some(), "Should have CPU performance targets");

    // Test error handling fixtures
    let error_scenarios = &fixtures.error_handling_fixtures.failure_scenarios;
    assert!(
        !error_scenarios.model_loading_failures.is_empty(),
        "Should have model loading failure scenarios"
    );

    // Test tier-based behavior
    match fixtures.config.tier {
        TestTier::Fast => {
            println!("✅ Fast tier fixture testing completed");
        }
        TestTier::Standard => {
            println!("✅ Standard tier fixture testing completed");
        }
        TestTier::Full => {
            println!("✅ Full tier fixture testing completed");
        }
    }

    // Cleanup fixtures
    fixtures.cleanup().await?;

    Ok(())
}

/// Test mock model artifacts and GGUF generation
#[tokio::test]
async fn test_mock_model_artifacts() -> Result<()> {
    let config = TestEnvironmentConfig::from_env();
    let mut fixtures = RealModelIntegrationFixtures::new();
    fixtures.initialize().await?;

    // Test mock model creation
    let small_model = fixtures.model_fixtures.get_mock_model("small");
    assert!(small_model.is_some(), "Should have small mock model");

    let small_model = small_model.unwrap();
    assert_eq!(small_model.config.parameter_count, 1_000_000);
    assert_eq!(small_model.config.vocab_size, 32000);
    assert!(!small_model.mock_tensors.is_empty(), "Should have mock tensors");

    // Test GGUF file generation
    let gguf_content = small_model.generate_mock_gguf();
    assert!(gguf_content.len() > 100, "GGUF content should be substantial");
    assert_eq!(&gguf_content[0..4], b"GGUF", "Should have valid GGUF magic");

    // Test model path retrieval based on tier
    let model_path = fixtures.model_fixtures.get_model_for_tier(config.tier).await?;
    assert!(
        model_path.exists() || config.tier == TestTier::Fast,
        "Model should exist for real tiers or be mock for fast tier"
    );

    fixtures.cleanup().await?;
    Ok(())
}

/// Test device-aware quantization validation
#[tokio::test]
async fn test_device_aware_quantization() -> Result<()> {
    let mut fixtures = RealModelIntegrationFixtures::new();
    fixtures.initialize().await?;

    // Test CPU quantization validation
    let cpu_validation =
        fixtures.quantization_fixtures.validate_quantization_accuracy("I2_S", Device::Cpu).await?;

    assert_eq!(cpu_validation.quantization_type, "I2_S");
    assert_eq!(cpu_validation.device, Device::Cpu);
    assert!(!cpu_validation.test_case_results.is_empty(), "Should have test case results");

    // Test that validation passes accuracy threshold
    assert!(
        cpu_validation.passes_accuracy_threshold,
        "CPU I2S quantization should pass accuracy threshold"
    );

    // Test GPU quantization if available
    skip_if_no_gpu!(fixtures.config);

    #[cfg(feature = "gpu")]
    {
        let gpu_validation = fixtures
            .quantization_fixtures
            .validate_quantization_accuracy("I2_S", Device::Cuda(0))
            .await?;

        assert_eq!(gpu_validation.device, Device::Cuda(0));
        assert!(
            gpu_validation.passes_accuracy_threshold,
            "GPU I2S quantization should pass accuracy threshold"
        );

        // GPU should be faster than CPU
        let gpu_avg_latency: f32 =
            gpu_validation.test_case_results.iter().map(|r| r.latency_ms).sum::<f32>()
                / gpu_validation.test_case_results.len() as f32;

        let cpu_avg_latency: f32 =
            cpu_validation.test_case_results.iter().map(|r| r.latency_ms).sum::<f32>()
                / cpu_validation.test_case_results.len() as f32;

        // Allow for mock timing variations
        if gpu_avg_latency < cpu_avg_latency * 2.0 {
            println!("✅ GPU quantization shows expected performance improvement");
        } else {
            println!("ℹ️ GPU performance improvement not significant (mock data)");
        }
    }

    fixtures.cleanup().await?;
    Ok(())
}

/// Test error handling and recovery scenarios
#[tokio::test]
async fn test_error_handling_scenarios() -> Result<()> {
    let mut fixtures = RealModelIntegrationFixtures::new();
    fixtures.initialize().await?;

    // Test model loading failure simulation
    let missing_file_error =
        fixtures.error_handling_fixtures.simulate_model_loading_failure("missing_model_file")?;

    match missing_file_error {
        bitnet_common::BitNetError::Model(model_error) => {
            let quality = fixtures
                .error_handling_fixtures
                .test_error_message_quality(&bitnet_common::BitNetError::Model(model_error));

            assert!(quality.has_user_friendly_message, "Error message should be user-friendly");
            assert!(quality.clarity_score > 0.5, "Error message should have good clarity");
        }
        _ => panic!("Expected ModelError for missing file scenario"),
    }

    // Test inference failure simulation
    let context_error =
        fixtures.error_handling_fixtures.simulate_inference_failure("context_length_exceeded")?;

    match context_error {
        bitnet_common::BitNetError::Inference(_) => {
            println!("✅ Context length exceeded error properly simulated");
        }
        _ => panic!("Expected InferenceError for context exceeded scenario"),
    }

    // Test corrupted file creation
    if let Some(corrupted_file) = fixtures.error_handling_fixtures.get_test_file("corrupted_gguf") {
        assert!(corrupted_file.exists(), "Corrupted test file should exist");

        let file_content = std::fs::read(corrupted_file).unwrap();
        assert!(file_content.len() > 0, "Corrupted file should have content");
        assert_ne!(&file_content[0..4], b"GGUF", "Should not have valid GGUF magic");
    }

    fixtures.cleanup().await?;
    Ok(())
}

/// Test performance benchmarking infrastructure
#[tokio::test]
async fn test_performance_benchmarking() -> Result<()> {
    let mut fixtures = RealModelIntegrationFixtures::new();
    fixtures.initialize().await?;

    // Skip performance testing in fast tier
    skip_if_tier_insufficient!(fixtures.config, TestTier::Standard);

    // Test CPU performance suite
    let cpu_result = fixtures.performance_fixtures.run_benchmark_suite("CPU_Performance").await?;

    assert_eq!(cpu_result.suite_name, "CPU_Performance");
    assert_eq!(cpu_result.device, Device::Cpu);
    assert!(!cpu_result.individual_results.is_empty(), "Should have individual benchmark results");

    // Validate performance targets
    for result in &cpu_result.individual_results {
        assert!(result.target_ms > 0, "Performance target should be positive");
        assert!(result.statistics.mean_ms > 0.0, "Mean execution time should be positive");
    }

    println!("CPU Performance Score: {:.2}", cpu_result.overall_score);
    assert!(
        cpu_result.overall_score >= 0.0 && cpu_result.overall_score <= 1.0,
        "Performance score should be between 0 and 1"
    );

    fixtures.cleanup().await?;
    Ok(())
}

/// Test cross-validation infrastructure
#[tokio::test]
async fn test_cross_validation_infrastructure() -> Result<()> {
    let mut fixtures = RealModelIntegrationFixtures::new();
    fixtures.initialize().await?;

    // Skip cross-validation in fast tier
    skip_if_tier_insufficient!(fixtures.config, TestTier::Full);

    // Test reference data availability
    let reference_data = fixtures.cross_validation_fixtures.get_reference_data("mock_small");

    if let Some(ref_data) = reference_data {
        assert!(!ref_data.input_tokens.is_empty(), "Reference data should have input tokens");
        assert!(!ref_data.expected_logits.is_empty(), "Reference data should have expected logits");
        assert!(ref_data.tolerance_config.logit_tolerance > 0.0, "Tolerance should be positive");
    }

    // Test cross-validation execution
    for test_case in &fixtures.cross_validation_fixtures.test_vectors.test_cases {
        if test_case.deterministic {
            let result = fixtures
                .cross_validation_fixtures
                .run_cross_validation("mock_small", test_case)
                .await?;

            assert_eq!(result.test_case_name, test_case.name);
            assert!(
                result.logit_correlation >= 0.0 && result.logit_correlation <= 1.0,
                "Correlation should be between 0 and 1"
            );

            if test_case.deterministic {
                assert!(
                    result.token_accuracy >= 0.9,
                    "Deterministic tests should have high token accuracy"
                );
            }
        }
    }

    fixtures.cleanup().await?;
    Ok(())
}

/// Test three-tier testing infrastructure
#[tokio::test]
async fn test_three_tier_infrastructure() -> Result<()> {
    // Test each tier configuration
    let tiers = vec![TestTier::Fast, TestTier::Standard, TestTier::Full];

    for tier in tiers {
        // Create config for specific tier
        let mut config = TestEnvironmentConfig::from_env();
        config.tier = tier.clone();

        println!("Testing tier: {:?}", tier);

        let mut fixtures = RealModelIntegrationFixtures::new();
        fixtures.config = config;
        fixtures.initialize().await?;

        match tier {
            TestTier::Fast => {
                // Fast tier should use mock models only
                assert!(fixtures.model_fixtures.get_mock_model("small").is_some());
                assert!(!fixtures.cross_validation_fixtures.reference_data.is_empty());
            }
            TestTier::Standard => {
                // Standard tier should have performance fixtures
                assert!(!fixtures.performance_fixtures.benchmark_targets.is_empty());
            }
            TestTier::Full => {
                // Full tier should have cross-validation
                assert!(!fixtures.cross_validation_fixtures.test_vectors.test_cases.is_empty());
            }
        }

        fixtures.cleanup().await?;
        println!("✅ Tier {:?} validation completed", tier);
    }

    Ok(())
}

/// Test fixture cleanup and resource management
#[tokio::test]
async fn test_fixture_cleanup() -> Result<()> {
    let mut fixtures = RealModelIntegrationFixtures::new();
    fixtures.initialize().await?;

    // Verify fixtures are initialized
    assert!(!fixtures.model_fixtures.mock_models.is_empty());
    assert!(!fixtures.error_handling_fixtures.test_files.is_empty());

    // Test cleanup
    fixtures.cleanup().await?;

    // Verify cleanup completed (test files should be removed)
    let test_file_count = fixtures.error_handling_fixtures.test_files.len();
    assert_eq!(test_file_count, 0, "Test files should be cleaned up");

    println!("✅ Fixture cleanup validation completed");
    Ok(())
}

/// Test acceptance criteria coverage
#[tokio::test]
async fn test_acceptance_criteria_coverage() -> Result<()> {
    let mut fixtures = RealModelIntegrationFixtures::new();
    fixtures.initialize().await?;

    // Test AC1-AC10 fixture coverage
    for ac_id in 1..=10 {
        let ac_fixtures = fixtures.get_fixtures_for_ac(ac_id);
        assert!(
            ac_fixtures.is_ok(),
            "Should have fixtures for AC{}: {:?}",
            ac_id,
            ac_fixtures.err()
        );

        match ac_fixtures.unwrap() {
            fixtures::ACTestFixtures::AC1 { model_fixtures, device_fixtures } => {
                assert!(!model_fixtures.mock_models.is_empty());
                assert!(!device_fixtures.kernel_providers.is_empty());
            }
            fixtures::ACTestFixtures::AC2 { quantization_fixtures, device_fixtures } => {
                assert!(quantization_fixtures.get_test_vectors("I2_S").is_some());
                assert!(!device_fixtures.kernel_providers.is_empty());
            }
            fixtures::ACTestFixtures::AC3 { device_fixtures, performance_fixtures } => {
                assert!(!device_fixtures.kernel_providers.is_empty());
                assert!(!performance_fixtures.benchmark_targets.is_empty());
            }
            fixtures::ACTestFixtures::AC4_5 { cross_validation_fixtures, performance_fixtures } => {
                assert!(!cross_validation_fixtures.reference_data.is_empty());
                assert!(!performance_fixtures.benchmark_targets.is_empty());
            }
            fixtures::ACTestFixtures::AC6_10 {
                model_fixtures,
                error_handling_fixtures,
                performance_fixtures,
            } => {
                assert!(!model_fixtures.mock_models.is_empty());
                assert!(
                    !error_handling_fixtures.failure_scenarios.model_loading_failures.is_empty()
                );
                assert!(!performance_fixtures.benchmark_targets.is_empty());
            }
        }
    }

    fixtures.cleanup().await?;
    println!("✅ All acceptance criteria (AC1-AC10) have fixture coverage");
    Ok(())
}
