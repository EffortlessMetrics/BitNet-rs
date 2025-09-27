//! Simple validation test for BitNet.rs neural network fixtures
//!
//! Basic test to verify fixtures can be loaded and used correctly.

use super::cross_validation_data::CrossValidationFixtures;
use super::network_mocks::NetworkMockFixtures;
use super::quantization_test_vectors::QuantizationFixtures;
use super::tokenizer_fixtures::TokenizerFixtures;
use bitnet_common::QuantizationType;

/// Simple test for tokenizer fixtures
#[test]
fn test_basic_tokenizer_fixtures() {
    let fixtures = TokenizerFixtures::new();

    // Test fixture count
    assert!(fixtures.fixtures.len() >= 3, "Should have at least 3 tokenizer types");

    // Test GGUF model count
    assert!(fixtures.gguf_models.len() >= 3, "Should have at least 3 GGUF models");

    println!("✅ Basic tokenizer fixtures test passed");
}

/// Simple test for quantization fixtures
#[test]
fn test_basic_quantization_fixtures() {
    let fixtures = QuantizationFixtures::new();

    // Test that we have test vectors for different quantization types
    assert!(fixtures.test_vectors.contains_key(&QuantizationType::I2S));
    assert!(fixtures.test_vectors.contains_key(&QuantizationType::TL1));
    assert!(fixtures.test_vectors.contains_key(&QuantizationType::TL2));

    // Test that vectors are not empty
    let i2s_vectors = fixtures.get_test_vectors(&QuantizationType::I2S).unwrap();
    assert!(!i2s_vectors.is_empty(), "I2S vectors should not be empty");

    println!("✅ Basic quantization fixtures test passed");
}

/// Simple test for cross-validation fixtures
#[test]
fn test_basic_cross_validation_fixtures() {
    let fixtures = CrossValidationFixtures::new();

    // Test that we have cases for different architectures
    let llama3_cases = fixtures.get_cases_for_architecture("BitNet-b1.58");
    assert!(!llama3_cases.is_empty(), "Should have LLaMA-3 cross-validation cases");

    let llama2_cases = fixtures.get_cases_for_architecture("BitNet-TL1");
    assert!(!llama2_cases.is_empty(), "Should have LLaMA-2 cross-validation cases");

    println!("✅ Basic cross-validation fixtures test passed");
}

/// Simple test for network mock fixtures
#[test]
fn test_basic_network_mock_fixtures() {
    let fixtures = NetworkMockFixtures::new();

    // Test that we have repository mocks
    assert!(!fixtures.model_repositories.is_empty(), "Should have model repositories");

    // Test specific repository
    let bitnet_repo = fixtures.get_repository_mock("microsoft/bitnet-b1.58-2B-4T-gguf");
    assert!(bitnet_repo.is_some(), "Should have BitNet repository mock");

    // Test API responses
    assert!(!fixtures.api_responses.is_empty(), "Should have API response mocks");

    // Test error scenarios
    assert!(!fixtures.error_scenarios.is_empty(), "Should have error scenarios");

    println!("✅ Basic network mock fixtures test passed");
}

/// Integration test using all fixture types
#[test]
fn test_fixture_integration() {
    let tokenizer_fixtures = TokenizerFixtures::new();
    let quantization_fixtures = QuantizationFixtures::new();
    let crossval_fixtures = CrossValidationFixtures::new();
    let network_fixtures = NetworkMockFixtures::new();

    // Test that fixtures can work together
    let llama3_gguf = tokenizer_fixtures.get_gguf_model_by_vocab(128256);
    assert!(llama3_gguf.is_some(), "Should find LLaMA-3 GGUF model");

    let i2s_vectors = quantization_fixtures.get_vocab_compatible_vectors(128256);
    assert!(!i2s_vectors.is_empty(), "Should find I2S vectors for large vocabulary");

    let llama3_crossval = crossval_fixtures.get_cases_for_architecture("BitNet-b1.58");
    assert!(!llama3_crossval.is_empty(), "Should find cross-validation cases for BitNet-b1.58");

    let bitnet_repo = network_fixtures.get_repository_mock("microsoft/bitnet-b1.58-2B-4T-gguf");
    assert!(bitnet_repo.is_some(), "Should find repository mock");

    println!("✅ Fixture integration test passed");
}

/// Test deterministic data generation
#[test]
fn test_deterministic_generation() {
    let fixtures = QuantizationFixtures::new();

    // Generate same data twice with same seed
    let data1 = fixtures.generate_deterministic_vectors(42, 5);
    let data2 = fixtures.generate_deterministic_vectors(42, 5);

    assert_eq!(data1.len(), data2.len(), "Same seed should produce same number of vectors");
    assert_eq!(data1.len(), 5, "Should generate requested number of vectors");

    // Compare first vector in detail
    assert_eq!(data1[0].input_data, data2[0].input_data, "Input data should be identical");
    assert_eq!(
        data1[0].expected_quantized, data2[0].expected_quantized,
        "Quantized data should be identical"
    );
    assert_eq!(
        data1[0].test_scenario, data2[0].test_scenario,
        "Test scenarios should be identical"
    );

    // Different seed should produce different data
    let data3 = fixtures.generate_deterministic_vectors(123, 5);
    assert_ne!(
        data1[0].input_data, data3[0].input_data,
        "Different seeds should produce different data"
    );

    println!("✅ Deterministic generation test passed");
}

/// Feature gate tests
#[cfg(feature = "cpu")]
#[test]
fn test_cpu_feature_gates() {
    use super::quantization_test_vectors::cpu_quantization;

    let simd_vectors = cpu_quantization::get_simd_compatible_vectors();
    assert!(!simd_vectors.is_empty(), "Should have SIMD compatible vectors");

    let avx2_data = cpu_quantization::get_avx2_test_data();
    assert_eq!(avx2_data.len(), 2, "Should have AVX2 test data");

    println!("✅ CPU feature gates test passed");
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_feature_gates() {
    use super::quantization_test_vectors::gpu_quantization;

    let cuda_vectors = gpu_quantization::get_cuda_compatible_vectors();
    assert!(!cuda_vectors.is_empty(), "Should have CUDA compatible vectors");

    let tensor_core_data = gpu_quantization::get_tensor_core_data();
    assert!(!tensor_core_data.is_empty(), "Should have Tensor Core data");

    println!("✅ GPU feature gates test passed");
}

#[cfg(feature = "ffi")]
#[test]
fn test_ffi_feature_gates() {
    use super::quantization_test_vectors::ffi_quantization;

    let ffi_vectors = ffi_quantization::get_ffi_test_vectors();
    assert!(!ffi_vectors.is_empty(), "Should have FFI test vectors");

    let cpp_data = ffi_quantization::create_cpp_compatible_test_data();
    assert!(!cpp_data.is_empty(), "Should have C++ compatible data");

    println!("✅ FFI feature gates test passed");
}
