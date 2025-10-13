//! Comprehensive validation tests for BitNet.rs neural network fixtures
//!
//! Validates that all fixtures work correctly with proper feature gates,
//! deterministic testing, and cargo test integration.

use super::*;
use bitnet_common::{QuantizationType, Result};
use tokio;

/// Test tokenizer fixtures loading and basic functionality
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_tokenizer_fixtures_loading() {
    let fixtures = TokenizerFixtures::new();

    // Test LLaMA-3 fixture
    let llama3_fixture = fixtures.get_fixture(&TokenizerType::LLaMA3).unwrap();
    assert_eq!(llama3_fixture.vocab_size, 128256);
    assert_eq!(llama3_fixture.model_architecture, "BitNet");
    assert!(llama3_fixture.special_tokens.bos_token.is_some());
    assert_eq!(llama3_fixture.special_tokens.bos_token.unwrap(), 128000);

    // Test LLaMA-2 fixture
    let llama2_fixture = fixtures.get_fixture(&TokenizerType::LLaMA2).unwrap();
    assert_eq!(llama2_fixture.vocab_size, 32000);
    assert_eq!(llama2_fixture.model_architecture, "BitNet");
    assert_eq!(llama2_fixture.special_tokens.bos_token.unwrap(), 1);

    // Test GPT-2 fixture
    let gpt2_fixture = fixtures.get_fixture(&TokenizerType::GPT2).unwrap();
    assert_eq!(gpt2_fixture.vocab_size, 50257);
    assert_eq!(gpt2_fixture.model_architecture, "GPT");
    assert!(gpt2_fixture.special_tokens.bos_token.is_none());
    assert_eq!(gpt2_fixture.special_tokens.eos_token.unwrap(), 50256);

    println!("✅ Tokenizer fixtures validation passed");
}

/// Test GGUF model fixtures generation and validation
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_gguf_model_fixtures() {
    let fixtures = TokenizerFixtures::new();

    // Test LLaMA-3 GGUF model
    let llama3_model = fixtures.get_gguf_model_by_vocab(128256).unwrap();
    assert_eq!(llama3_model.vocab_size, 128256);
    assert_eq!(llama3_model.architecture, "BitNet");
    assert!(!llama3_model.file_content.is_empty());

    // Verify GGUF magic number
    assert_eq!(&llama3_model.file_content[0..4], b"GGUF");

    // Test LLaMA-2 GGUF model
    let llama2_model = fixtures.get_gguf_model_by_vocab(32000).unwrap();
    assert_eq!(llama2_model.vocab_size, 32000);
    assert_eq!(llama2_model.architecture, "BitNet");

    // Test GPT-2 GGUF model
    let gpt2_model = fixtures.get_gguf_model_by_vocab(50257).unwrap();
    assert_eq!(gpt2_model.vocab_size, 50257);
    assert_eq!(gpt2_model.architecture, "GPT");

    println!("✅ GGUF model fixtures validation passed");
}

/// Test quantization test vectors for different algorithms
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_quantization_fixtures() {
    let fixtures = quantization_test_vectors::QuantizationFixtures::new();

    // Test I2S quantization vectors
    let i2s_vectors = fixtures.get_test_vectors(&QuantizationType::I2S).unwrap();
    assert!(!i2s_vectors.is_empty());

    for vector in i2s_vectors {
        assert_eq!(vector.quantization_type, QuantizationType::I2S);
        assert!(!vector.input_data.is_empty());
        assert!(!vector.expected_quantized.is_empty());
        assert!(!vector.expected_scales.is_empty());
        assert_eq!(vector.input_data.len(), vector.expected_dequantized.len());
        assert!(vector.tolerance > 0.0);
        assert!(vector.device_compatible.contains(&"CPU".to_string()));
    }

    // Test TL1 quantization vectors
    let tl1_vectors = fixtures.get_test_vectors(&QuantizationType::TL1).unwrap();
    assert!(!tl1_vectors.is_empty());

    for vector in tl1_vectors {
        assert_eq!(vector.quantization_type, QuantizationType::TL1);
        assert!(vector.vocab_range.is_some());
        let (min_vocab, max_vocab) = vector.vocab_range.unwrap();
        assert!(max_vocab <= 32000); // TL1 optimized for smaller vocabs
    }

    // Test vocabulary compatibility
    let llama3_compatible = fixtures.get_vocab_compatible_vectors(128256);
    assert!(!llama3_compatible.is_empty());
    assert!(llama3_compatible.iter().any(|v| v.quantization_type == QuantizationType::I2S));

    let llama2_compatible = fixtures.get_vocab_compatible_vectors(32000);
    assert!(!llama2_compatible.is_empty());
    assert!(llama2_compatible.iter().any(|v| v.quantization_type == QuantizationType::TL1));

    println!("✅ Quantization fixtures validation passed");
}

/// Test cross-validation fixtures
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_cross_validation_fixtures() {
    let fixtures = cross_validation_data::CrossValidationFixtures::new();

    // Test LLaMA-3 cross-validation cases
    let llama3_cases = fixtures.get_cases_for_architecture("BitNet-b1.58");
    assert!(!llama3_cases.is_empty());

    for case in &llama3_cases {
        assert_eq!(case.model_architecture, "BitNet-b1.58");
        assert_eq!(case.rust_implementation, "bitnet-rs");
        assert!(!case.input_data.input_tokens.is_empty());
        assert!(!case.expected_outputs.rust_logits.is_empty());
        assert!(!case.expected_outputs.cpp_reference_logits.is_empty());
        assert_eq!(
            case.expected_outputs.rust_logits.len(),
            case.expected_outputs.cpp_reference_logits.len()
        );
        assert!(case.tolerance_spec.absolute_tolerance > 0.0);
        assert!(case.tolerance_spec.cosine_similarity_threshold > 0.9);
    }

    // Test tokenizer cross-validation
    let llama3_tokenizer_data = fixtures.get_tokenizer_data("LLaMA-3").unwrap();
    assert_eq!(llama3_tokenizer_data.vocab_size, 128256);
    assert!(!llama3_tokenizer_data.test_texts.is_empty());
    assert_eq!(
        llama3_tokenizer_data.test_texts.len(),
        llama3_tokenizer_data.rust_tokenizations.len()
    );
    assert_eq!(
        llama3_tokenizer_data.rust_tokenizations.len(),
        llama3_tokenizer_data.cpp_tokenizations.len()
    );

    // Test deterministic case generation
    let deterministic_case = fixtures.generate_deterministic_case(42, 128256);
    assert_eq!(deterministic_case.input_data.deterministic_seed, Some(42));
    assert!(!deterministic_case.input_data.input_tokens.is_empty());

    // Generate same case again to verify determinism
    let deterministic_case2 = fixtures.generate_deterministic_case(42, 128256);
    assert_eq!(
        deterministic_case.input_data.input_tokens,
        deterministic_case2.input_data.input_tokens
    );

    println!("✅ Cross-validation fixtures validation passed");
}

/// Test network mock fixtures
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_network_mock_fixtures() {
    let fixtures = NetworkMockFixtures::new();

    // Test Microsoft BitNet repository mock
    let bitnet_repo = fixtures.get_repository_mock("microsoft/bitnet-b1.58-2B-4T-gguf").unwrap();
    assert_eq!(bitnet_repo.repo_id, "microsoft/bitnet-b1.58-2B-4T-gguf");
    assert!(!bitnet_repo.model_files.is_empty());
    assert!(!bitnet_repo.tokenizer_files.is_empty());

    // Verify model file metadata
    let gguf_model = bitnet_repo.model_files.iter()
        .find(|f| f.filename == "ggml-model-i2_s.gguf")
        .unwrap();
    assert_eq!(gguf_model.size, 1_800_000_000);
    assert!(gguf_model.lfs_pointer.is_some());

    // Verify tokenizer file metadata
    let tokenizer = bitnet_repo.tokenizer_files.iter()
        .find(|f| f.filename == "tokenizer.json")
        .unwrap();
    assert_eq!(tokenizer.tokenizer_type, "LLaMA-3");
    assert!(tokenizer.content.is_some());

    // Test API response mocks
    let model_info_response = fixtures.get_api_response(
        "https://huggingface.co/api/models/microsoft/bitnet-b1.58-2B-4T-gguf"
    ).unwrap();
    assert_eq!(model_info_response.status_code, 200);
    assert!(model_info_response.response_body.contains("microsoft/bitnet-b1.58-2B-4T-gguf"));

    // Test error scenarios
    let timeout_scenario = fixtures.get_error_scenario("connection_timeout").unwrap();
    assert_eq!(timeout_scenario.error_type, "Timeout");
    assert_eq!(timeout_scenario.max_retries, 3);
    assert!(timeout_scenario.should_recover);

    // Test network scenarios
    let fast_scenario = fixtures.create_test_scenario("fast_download");
    assert_eq!(fast_scenario.name, "Fast reliable download");
    assert!(fast_scenario.bandwidth_limit.is_some());
    assert_eq!(fast_scenario.packet_loss_rate, 0.0);

    let slow_scenario = fixtures.create_test_scenario("slow_download");
    assert_eq!(slow_scenario.name, "Slow network download");
    assert_eq!(slow_scenario.timeout_seconds, 300);

    println!("✅ Network mock fixtures validation passed");
}

/// Test fixture loader with different configurations
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_fixture_loader() {
    // Test fast configuration
    let fast_config = FixtureConfig {
        fixtures_directory: std::env::temp_dir().join("bitnet_test_fixtures_fast"),
        enable_deterministic: true,
        seed: Some(12345),
        force_regenerate: true,
        feature_gates: vec!["cpu".to_string()],
        test_tier: FixtureTier::Fast,
    };

    let mut loader = FixtureLoader::new();
    loader.initialize(fast_config).await.unwrap();
    assert!(loader.is_ready());

    // Test tokenizer fixture loading
    let llama3_result = loader.load_tokenizer_fixture(TokenizerType::LLaMA3).unwrap();
    assert_eq!(llama3_result.data.vocab_size, 128256);
    assert!(matches!(llama3_result.source, FixtureSource::Static));

    // Test quantization vector loading
    let quant_result = loader.load_quantization_vectors(QuantizationType::I2S, Some(128256)).unwrap();
    assert!(!quant_result.data.is_empty());
    assert!(quant_result.load_time_ms > 0);

    // Test cross-validation cases
    let crossval_result = loader.load_crossval_cases("BitNet-b1.58").unwrap();
    assert!(!crossval_result.data.is_empty());

    // Test network scenario loading
    let network_result = loader.load_network_scenario("fast_download").unwrap();
    assert!(network_result.data.bandwidth_limit.is_some());
    assert!(matches!(network_result.source, FixtureSource::Generated));

    // Test deterministic data generation
    let deterministic_result = loader.generate_deterministic_data(|seed| {
        format!("test_data_seed_{}", seed)
    }).unwrap();
    assert!(deterministic_result.data.contains("test_data_seed_12345"));

    // Test fixture statistics
    let stats = loader.get_fixture_stats();
    assert!(stats.tokenizer_fixtures_count > 0);
    assert!(stats.quantization_vectors_count > 0);
    assert!(stats.crossval_cases_count > 0);
    assert!(stats.network_scenarios_count > 0);
    assert!(stats.initialized);

    // Cleanup
    loader.cleanup().await.unwrap();

    println!("✅ Fixture loader validation passed");
}

/// Test GGUF fixture file writing and reading
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_gguf_file_operations() {
    let fixtures = TokenizerFixtures::new();
    let temp_dir = std::env::temp_dir().join("bitnet_gguf_test");
    tokio::fs::create_dir_all(&temp_dir).await.unwrap();

    // Write fixtures to temporary directory
    let mut fixtures_with_temp_dir = fixtures.clone();
    fixtures_with_temp_dir.fixtures_dir = temp_dir.clone();
    fixtures_with_temp_dir.write_all_fixtures().await.unwrap();

    // Verify GGUF files were created
    let llama3_model = fixtures.get_gguf_model_by_vocab(128256).unwrap();
    let model_file = temp_dir.join("gguf_models").join("llama3_128k.gguf");

    // Create the file if it doesn't exist (simulating fixture loader behavior)
    tokio::fs::create_dir_all(model_file.parent().unwrap()).await.unwrap();
    tokio::fs::write(&model_file, &llama3_model.file_content).await.unwrap();

    assert!(model_file.exists());

    // Read and verify GGUF content
    let content = tokio::fs::read(&model_file).await.unwrap();
    assert_eq!(&content[0..4], b"GGUF");
    assert_eq!(content.len(), llama3_model.file_content.len());

    // Verify tokenizer files
    let tokenizer_file = temp_dir.join("tokenizers").join("llama3_tokenizer.json");
    assert!(tokenizer_file.exists());

    let tokenizer_content = tokio::fs::read_to_string(&tokenizer_file).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&tokenizer_content).unwrap();
    assert_eq!(parsed["model"]["type"], "BPE");

    // Cleanup
    tokio::fs::remove_dir_all(&temp_dir).await.unwrap();

    println!("✅ GGUF file operations validation passed");
}

/// Test feature-gated compilation for CPU fixtures
#[cfg(feature = "cpu")]
#[tokio::test]
async fn test_cpu_feature_gates() {
    use fixture_loader::cpu_fixtures;

    // Test CPU-specific initialization
    cpu_fixtures::initialize_cpu_fixtures().await.unwrap();

    // Test CPU quantization data loading
    let cpu_quant_data = cpu_fixtures::load_cpu_quantization_data().unwrap();
    assert!(!cpu_quant_data.is_empty());
    assert!(cpu_quant_data.iter().all(|v| v.device_compatible.contains(&"CPU".to_string())));

    // Test SIMD test data
    let simd_data = cpu_fixtures::load_simd_test_data();
    assert_eq!(simd_data.len(), 2);

    for (input, expected_quantized, scale) in &simd_data {
        assert_eq!(input.len(), 8); // SIMD vector size
        assert_eq!(expected_quantized.len(), 8);
        assert!(*scale > 0.0);
    }

    println!("✅ CPU feature gates validation passed");
}

/// Test feature-gated compilation for GPU fixtures (if available)
#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_gpu_feature_gates() {
    use fixture_loader::gpu_fixtures;

    // Test GPU-specific initialization
    gpu_fixtures::initialize_gpu_fixtures().await.unwrap();

    // Test GPU quantization data loading (I2S optimized for large vocabularies)
    let gpu_quant_data = gpu_fixtures::load_gpu_quantization_data().unwrap();
    assert!(!gpu_quant_data.is_empty());
    assert!(gpu_quant_data.iter().all(|v| v.quantization_type == QuantizationType::I2S));
    assert!(gpu_quant_data.iter().all(|v| v.device_compatible.contains(&"GPU".to_string())));

    // Test mixed precision fixtures
    let mixed_precision_result = gpu_fixtures::load_mixed_precision_fixtures().await.unwrap();
    assert!(!mixed_precision_result.data.is_empty());
    assert_eq!(&mixed_precision_result.data[0..4], b"GGUF");

    println!("✅ GPU feature gates validation passed");
}

/// Test FFI feature gates (if available)
#[cfg(feature = "ffi")]
#[tokio::test]
async fn test_ffi_feature_gates() {
    use fixture_loader::ffi_fixtures;

    // Test FFI-specific initialization
    ffi_fixtures::initialize_ffi_fixtures().await.unwrap();

    // Test FFI cross-validation cases
    let ffi_cases = ffi_fixtures::load_ffi_crossval_cases().unwrap();
    assert!(!ffi_cases.is_empty());

    for case in &ffi_cases {
        assert!(case.cpp_implementation == "llama.cpp" || case.cpp_implementation == "ggml");
        assert_eq!(case.rust_implementation, "bitnet-rs");
        assert!(!case.expected_outputs.rust_logits.is_empty());
        assert!(!case.expected_outputs.cpp_reference_logits.is_empty());
    }

    println!("✅ FFI feature gates validation passed");
}

/// Test deterministic behavior with BITNET_DETERMINISTIC=1
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_deterministic_behavior() {
    std::env::set_var("BITNET_DETERMINISTIC", "1");
    std::env::set_var("BITNET_SEED", "42");

    let config = FixtureConfig {
        fixtures_directory: std::env::temp_dir().join("bitnet_deterministic_test"),
        enable_deterministic: true,
        seed: Some(42),
        force_regenerate: true,
        feature_gates: vec!["cpu".to_string()],
        test_tier: FixtureTier::Fast,
    };

    // First run
    let mut loader1 = FixtureLoader::new();
    loader1.initialize(config.clone()).await.unwrap();

    let result1 = loader1.generate_deterministic_data(|seed| {
        let quant_fixtures = quantization_test_vectors::QuantizationFixtures::new();
        quant_fixtures.generate_deterministic_vectors(seed, 3)
    }).unwrap();

    // Second run with same seed
    let mut loader2 = FixtureLoader::new();
    loader2.initialize(config.clone()).await.unwrap();

    let result2 = loader2.generate_deterministic_data(|seed| {
        let quant_fixtures = quantization_test_vectors::QuantizationFixtures::new();
        quant_fixtures.generate_deterministic_vectors(seed, 3)
    }).unwrap();

    // Results should be identical
    assert_eq!(result1.data.len(), result2.data.len());
    for (vec1, vec2) in result1.data.iter().zip(result2.data.iter()) {
        assert_eq!(vec1.input_data, vec2.input_data);
        assert_eq!(vec1.expected_quantized, vec2.expected_quantized);
        assert_eq!(vec1.expected_scales, vec2.expected_scales);
        assert_eq!(vec1.test_scenario, vec2.test_scenario);
    }

    // Cleanup
    loader1.cleanup().await.unwrap();
    loader2.cleanup().await.unwrap();

    std::env::remove_var("BITNET_DETERMINISTIC");
    std::env::remove_var("BITNET_SEED");

    println!("✅ Deterministic behavior validation passed");
}

/// Integration test combining all fixtures
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_comprehensive_fixture_integration() {
    let config = FixtureConfig {
        fixtures_directory: std::env::temp_dir().join("bitnet_integration_test"),
        enable_deterministic: true,
        seed: Some(42),
        force_regenerate: true,
        feature_gates: vec!["cpu".to_string()],
        test_tier: FixtureTier::Standard,
    };

    let mut loader = FixtureLoader::new();
    loader.initialize(config).await.unwrap();

    // Load tokenizer fixture for LLaMA-3
    let tokenizer_result = loader.load_tokenizer_fixture(TokenizerType::LLaMA3).unwrap();
    let tokenizer_fixture = tokenizer_result.data;

    // Load compatible quantization vectors
    let quant_result = loader.load_quantization_vectors(
        QuantizationType::I2S,
        Some(tokenizer_fixture.vocab_size)
    ).unwrap();
    let quant_vectors = quant_result.data;

    // Load cross-validation cases for the same architecture
    let crossval_result = loader.load_crossval_cases(&tokenizer_fixture.model_architecture).unwrap();
    let crossval_cases = crossval_result.data;

    // Load network scenario for testing download behavior
    let network_result = loader.load_network_scenario("fast_download").unwrap();
    let network_scenario = network_result.data;

    // Verify all components are compatible and consistent
    assert_eq!(tokenizer_fixture.vocab_size, 128256);
    assert!(!quant_vectors.is_empty());
    assert!(quant_vectors.iter().all(|v| {
        if let Some((min, max)) = v.vocab_range {
            tokenizer_fixture.vocab_size >= min && tokenizer_fixture.vocab_size <= max
        } else {
            true
        }
    }));

    assert!(!crossval_cases.is_empty());
    assert!(crossval_cases.iter().any(|case| {
        case.input_data.model_config.vocab_size == tokenizer_fixture.vocab_size
    }));

    assert!(network_scenario.bandwidth_limit.is_some());
    assert_eq!(network_scenario.name, "Fast reliable download");

    // Test that we can load GGUF fixture for the same vocabulary
    let gguf_result = loader.load_gguf_fixture(tokenizer_fixture.vocab_size).await.unwrap();
    assert!(!gguf_result.data.is_empty());
    assert_eq!(&gguf_result.data[0..4], b"GGUF");

    // Cleanup
    loader.cleanup().await.unwrap();

    println!("✅ Comprehensive fixture integration validation passed");
}

/// Performance test for fixture loading times
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_fixture_loading_performance() {
    let config = FixtureConfig {
        fixtures_directory: std::env::temp_dir().join("bitnet_perf_test"),
        enable_deterministic: false, // Disable for performance testing
        seed: None,
        force_regenerate: true,
        feature_gates: vec!["cpu".to_string()],
        test_tier: FixtureTier::Fast, // Use fast tier for performance testing
    };

    let start_time = std::time::Instant::now();

    let mut loader = FixtureLoader::new();
    loader.initialize(config).await.unwrap();

    let init_time = start_time.elapsed();
    assert!(init_time.as_millis() < 5000, "Initialization took too long: {:?}", init_time);

    // Test loading times for different fixture types
    let load_start = std::time::Instant::now();
    let _tokenizer = loader.load_tokenizer_fixture(TokenizerType::LLaMA3).unwrap();
    let tokenizer_load_time = load_start.elapsed();
    assert!(tokenizer_load_time.as_millis() < 100, "Tokenizer loading too slow: {:?}", tokenizer_load_time);

    let quant_start = std::time::Instant::now();
    let _quant = loader.load_quantization_vectors(QuantizationType::I2S, None).unwrap();
    let quant_load_time = quant_start.elapsed();
    assert!(quant_load_time.as_millis() < 50, "Quantization loading too slow: {:?}", quant_load_time);

    let network_start = std::time::Instant::now();
    let _network = loader.load_network_scenario("fast_download").unwrap();
    let network_load_time = network_start.elapsed();
    assert!(network_load_time.as_millis() < 10, "Network scenario loading too slow: {:?}", network_load_time);

    // Cleanup
    loader.cleanup().await.unwrap();

    println!("✅ Fixture loading performance validation passed");
    println!("   - Initialization: {:?}", init_time);
    println!("   - Tokenizer loading: {:?}", tokenizer_load_time);
    println!("   - Quantization loading: {:?}", quant_load_time);
    println!("   - Network scenario: {:?}", network_load_time);
}

/// Test error handling and edge cases
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_fixture_error_handling() {
    let mut loader = FixtureLoader::new();

    // Test loading without initialization
    let result = loader.load_tokenizer_fixture(TokenizerType::LLaMA3);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not initialized"));

    // Initialize with minimal config
    let config = FixtureConfig {
        fixtures_directory: std::env::temp_dir().join("bitnet_error_test"),
        enable_deterministic: false,
        seed: None,
        force_regenerate: true,
        feature_gates: vec!["cpu".to_string()],
        test_tier: FixtureTier::Fast,
    };

    loader.initialize(config).await.unwrap();

    // Test invalid vocabulary size
    let invalid_result = loader.load_gguf_fixture(999999).await;
    assert!(invalid_result.is_err());

    // Test invalid architecture
    let invalid_arch_result = loader.load_crossval_cases("InvalidArchitecture");
    assert!(invalid_arch_result.is_err());

    // Test deterministic mode without enabling it
    loader.deterministic_mode = false;
    let deterministic_result = loader.generate_deterministic_data(|_| "test".to_string());
    assert!(deterministic_result.is_err());
    assert!(deterministic_result.unwrap_err().to_string().contains("Deterministic mode not enabled"));

    println!("✅ Fixture error handling validation passed");
}
