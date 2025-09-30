#![cfg(feature = "integration-tests")]
//! Integration tests for bitnet-common

use bitnet_common::*;
use candle_core::DType;
use serial_test::serial;
use std::fs;
use temp_env::with_var;
use tempfile::TempDir;

#[test]
fn test_config_with_tensor_operations() {
    let config =
        BitNetConfig::builder().vocab_size(32000).hidden_size(4096).use_gpu(false).build().unwrap();

    let vocab_tensor =
        ConcreteTensor::mock(vec![config.model.vocab_size, config.model.hidden_size]);
    assert_eq!(vocab_tensor.shape(), &[32000, 4096]);

    let hidden_tensor = ConcreteTensor::mock(vec![config.model.hidden_size]);
    assert_eq!(hidden_tensor.shape(), &[4096]);
}

#[test]
fn test_error_propagation_through_config() {
    let mut config = BitNetConfig::default();
    config.model.vocab_size = 0;

    let validation_result = config.validate();
    assert!(validation_result.is_err());

    match validation_result {
        Err(BitNetError::Config(msg)) => {
            assert!(msg.contains("vocab_size"));
        }
        _ => panic!("Expected Config error"),
    }
}

#[test]
fn test_serialization_with_all_types() {
    let config = BitNetConfig {
        model: ModelConfig {
            path: Some("/test/model.gguf".into()),
            format: ModelFormat::SafeTensors,
            vocab_size: 50000,
            hidden_size: 8192,
            num_layers: 48,
            num_heads: 64,
            num_key_value_heads: 64,
            intermediate_size: 22016,
            max_position_embeddings: 4096,
            rope_theta: Some(10000.0),
            rope_scaling: None,
        },
        inference: InferenceConfig {
            max_length: 4096,
            max_new_tokens: 1024,
            temperature: 0.8,
            top_k: Some(40),
            top_p: Some(0.95),
            repetition_penalty: 1.1,
            seed: Some(42),
        },
        quantization: QuantizationConfig {
            quantization_type: QuantizationType::TL2,
            block_size: 128,
            precision: 1e-5,
        },
        performance: PerformanceConfig {
            num_threads: Some(16),
            use_gpu: true,
            batch_size: 8,
            memory_limit: Some(8 * 1024 * 1024 * 1024),
        },
    };

    let json = serde_json::to_string_pretty(&config).unwrap();
    let deserialized: BitNetConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config.model.vocab_size, deserialized.model.vocab_size);
    assert_eq!(config.model.format, deserialized.model.format);
    assert_eq!(config.inference.temperature, deserialized.inference.temperature);
    assert_eq!(config.quantization.quantization_type, deserialized.quantization.quantization_type);
    assert_eq!(config.performance.use_gpu, deserialized.performance.use_gpu);

    assert!(deserialized.validate().is_ok());
}

#[test]
#[serial(bitnet_env)]
fn test_end_to_end_config_workflow() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("integration_test.toml");

    let config_content = r#"
[model]
path = "/models/bitnet-7b.gguf"
format = "Gguf"
vocab_size = 32000
hidden_size = 4096

[inference]
temperature = 0.8
seed = 123

[quantization]
quantization_type = "I2S"

[performance]
use_gpu = false
"#;

    fs::write(&config_path, config_content).unwrap();

    let config = BitNetConfig::from_file(&config_path).unwrap();

    assert_eq!(config.model.path, Some("/models/bitnet-7b.gguf".into()));
    assert_eq!(config.model.format, ModelFormat::Gguf);
    assert_eq!(config.model.vocab_size, 32000);
    assert_eq!(config.inference.temperature, 0.8);
    assert_eq!(config.quantization.quantization_type, QuantizationType::I2S);
    assert!(!config.performance.use_gpu);

    // Test env var overrides using scoped env changes
    with_var("BITNET_TEMPERATURE", Some("0.9"), || {
        with_var("BITNET_USE_GPU", Some("true"), || {
            let config_with_env = BitNetConfig::from_file_with_env(&config_path).unwrap();

            assert_eq!(config_with_env.inference.temperature, 0.9);
            assert!(config_with_env.performance.use_gpu);
            assert_eq!(config_with_env.model.vocab_size, 32000);
            assert_eq!(config_with_env.inference.seed, Some(123));
        });
    });
}

#[test]
fn test_error_handling_integration() {
    let mut config = BitNetConfig::default();
    config.model.vocab_size = 0;
    let config_error = config.validate().unwrap_err();
    assert!(matches!(config_error, BitNetError::Config(_)));

    let model_error = ModelError::NotFound { path: "missing.gguf".to_string() };
    let bitnet_error: BitNetError = model_error.into();
    assert!(matches!(bitnet_error, BitNetError::Model(_)));

    let quant_error = QuantizationError::InvalidBlockSize { size: 0 };
    let bitnet_error: BitNetError = quant_error.into();
    assert!(matches!(bitnet_error, BitNetError::Quantization(_)));

    let error_msg = format!("{}", bitnet_error);
    assert!(error_msg.contains("Quantization error"));
    assert!(error_msg.contains("Invalid block size"));
}

#[test]
fn test_performance_metrics_with_config() {
    let config =
        BitNetConfig::builder().batch_size(4).use_gpu(true).num_threads(Some(8)).build().unwrap();

    let metrics = PerformanceMetrics {
        tokens_per_second: 150.0,
        latency_ms: 6.67,
        memory_usage_mb: 2048.0,
        computation_type: ComputationType::Real,
        gpu_utilization: if config.performance.use_gpu { Some(85.0) } else { None },
    };

    assert_eq!(config.performance.batch_size, 4);
    assert!(config.performance.use_gpu);
    assert_eq!(config.performance.num_threads, Some(8));
    assert!(metrics.gpu_utilization.is_some());

    let json = serde_json::to_string(&metrics).unwrap();
    let deserialized: PerformanceMetrics = serde_json::from_str(&json).unwrap();
    assert_eq!(metrics.tokens_per_second, deserialized.tokens_per_second);
    assert_eq!(metrics.computation_type, deserialized.computation_type);
    assert_eq!(metrics.gpu_utilization, deserialized.gpu_utilization);
}

#[test]
fn test_model_metadata_integration() {
    let config = BitNetConfig::builder()
        .vocab_size(50000)
        .quantization_type(QuantizationType::TL2)
        .build()
        .unwrap();

    let metadata = ModelMetadata {
        name: "bitnet-test".to_string(),
        version: "1.0.0".to_string(),
        architecture: "BitNet".to_string(),
        vocab_size: config.model.vocab_size,
        context_length: config.inference.max_length,
        quantization: Some(config.quantization.quantization_type),
    };

    assert_eq!(metadata.vocab_size, 50000);
    assert_eq!(metadata.context_length, 2048);
    assert_eq!(metadata.quantization, Some(QuantizationType::TL2));

    let json = serde_json::to_string_pretty(&metadata).unwrap();
    let deserialized: ModelMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(metadata.name, deserialized.name);
    assert_eq!(metadata.quantization, deserialized.quantization);
}

#[test]
fn test_generation_config_integration() {
    let inference_config = InferenceConfig {
        max_length: 4096,
        max_new_tokens: 1024,
        temperature: 0.7,
        top_k: Some(40),
        top_p: Some(0.95),
        repetition_penalty: 1.2,
        seed: Some(42),
    };

    let generation_config = GenerationConfig {
        max_new_tokens: inference_config.max_new_tokens,
        temperature: inference_config.temperature,
        top_k: inference_config.top_k,
        top_p: inference_config.top_p,
        repetition_penalty: inference_config.repetition_penalty,
        do_sample: true,
        seed: inference_config.seed,
    };

    assert_eq!(generation_config.max_new_tokens, 1024);
    assert_eq!(generation_config.temperature, 0.7);
    assert_eq!(generation_config.top_k, Some(40));
    assert_eq!(generation_config.top_p, Some(0.95));
    assert_eq!(generation_config.repetition_penalty, 1.2);
    assert_eq!(generation_config.seed, Some(42));
}

#[test]
fn test_complex_tensor_operations() {
    let shapes = vec![vec![1, 1], vec![10, 10], vec![2, 3, 4], vec![1, 1000, 4096]];

    for shape in shapes {
        let tensor = ConcreteTensor::mock(shape.clone());

        assert_eq!(tensor.shape(), shape.as_slice());
        assert_eq!(tensor.dtype(), DType::F32);

        let expected_size: usize = shape.iter().product();
        let slice: &[f32] = tensor.as_slice().unwrap();
        assert_eq!(slice.len(), expected_size);

        let candle_tensor = tensor.to_candle().unwrap();
        assert_eq!(candle_tensor.shape().dims(), shape.as_slice());
        assert_eq!(candle_tensor.dtype(), DType::F32);
    }
}

#[test]
fn test_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let config = Arc::new(BitNetConfig::default());
    let tensor = Arc::new(ConcreteTensor::mock(vec![10, 10]));
    let metadata = Arc::new(ModelMetadata {
        name: "test".to_string(),
        version: "1.0".to_string(),
        architecture: "BitNet".to_string(),
        vocab_size: 32000,
        context_length: 2048,
        quantization: Some(QuantizationType::I2S),
    });

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let config = Arc::clone(&config);
            let tensor = Arc::clone(&tensor);
            let metadata = Arc::clone(&metadata);

            thread::spawn(move || {
                assert_eq!(config.model.vocab_size, 32000);
                assert_eq!(tensor.shape(), &[10, 10]);
                assert_eq!(metadata.vocab_size, 32000);
                i
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}
