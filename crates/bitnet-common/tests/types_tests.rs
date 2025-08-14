//! Comprehensive tests for type definitions in bitnet-common

use bitnet_common::*;
use proptest::prelude::*;

#[test]
fn test_quantization_type_variants() {
    // Test all variants exist and are distinct
    let i2s = QuantizationType::I2S;
    let tl1 = QuantizationType::TL1;
    let tl2 = QuantizationType::TL2;

    assert_ne!(i2s, tl1);
    assert_ne!(i2s, tl2);
    assert_ne!(tl1, tl2);
}

#[test]
fn test_quantization_type_display() {
    assert_eq!(format!("{}", QuantizationType::I2S), "I2_S");
    assert_eq!(format!("{}", QuantizationType::TL1), "TL1");
    assert_eq!(format!("{}", QuantizationType::TL2), "TL2");
}

#[test]
fn test_quantization_type_serialization() {
    // Test JSON serialization/deserialization
    let i2s = QuantizationType::I2S;
    let json = serde_json::to_string(&i2s).unwrap();
    let deserialized: QuantizationType = serde_json::from_str(&json).unwrap();
    assert_eq!(i2s, deserialized);

    let tl1 = QuantizationType::TL1;
    let json = serde_json::to_string(&tl1).unwrap();
    let deserialized: QuantizationType = serde_json::from_str(&json).unwrap();
    assert_eq!(tl1, deserialized);

    let tl2 = QuantizationType::TL2;
    let json = serde_json::to_string(&tl2).unwrap();
    let deserialized: QuantizationType = serde_json::from_str(&json).unwrap();
    assert_eq!(tl2, deserialized);
}

#[test]
fn test_device_type_variants() {
    let cpu = DeviceType::Cpu;
    let cuda0 = DeviceType::Cuda(0);
    let cuda1 = DeviceType::Cuda(1);
    let metal = DeviceType::Metal;

    assert_ne!(cpu, cuda0);
    assert_ne!(cuda0, cuda1);
    assert_ne!(cpu, metal);
    assert_eq!(cuda0, DeviceType::Cuda(0));
}

#[test]
fn test_device_type_default() {
    assert_eq!(DeviceType::default(), DeviceType::Cpu);
}

#[test]
fn test_device_type_serialization() {
    let cpu = DeviceType::Cpu;
    let json = serde_json::to_string(&cpu).unwrap();
    let deserialized: DeviceType = serde_json::from_str(&json).unwrap();
    assert_eq!(cpu, deserialized);

    let cuda = DeviceType::Cuda(2);
    let json = serde_json::to_string(&cuda).unwrap();
    let deserialized: DeviceType = serde_json::from_str(&json).unwrap();
    assert_eq!(cuda, deserialized);
}

#[test]
fn test_device_variants() {
    let cpu = Device::Cpu;
    let cuda0 = Device::Cuda(0);
    let cuda1 = Device::Cuda(1);
    let metal = Device::Metal;

    assert_ne!(cpu, cuda0);
    assert_ne!(cuda0, cuda1);
    assert_ne!(cpu, metal);
    assert_eq!(cuda0, Device::Cuda(0));
}

#[test]
fn test_device_default() {
    assert_eq!(Device::default(), Device::Cpu);
}

#[test]
fn test_generation_config_default() {
    let config = GenerationConfig::default();

    assert_eq!(config.max_new_tokens, 512);
    assert_eq!(config.temperature, 1.0);
    assert_eq!(config.top_k, Some(50));
    assert_eq!(config.top_p, Some(0.9));
    assert_eq!(config.repetition_penalty, 1.1);
    assert!(config.do_sample);
    assert_eq!(config.seed, None);
}

#[test]
fn test_generation_config_serialization() {
    let config = GenerationConfig {
        max_new_tokens: 256,
        temperature: 0.8,
        top_k: Some(40),
        top_p: Some(0.95),
        repetition_penalty: 1.2,
        do_sample: false,
        seed: Some(42),
    };

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: GenerationConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config.max_new_tokens, deserialized.max_new_tokens);
    assert_eq!(config.temperature, deserialized.temperature);
    assert_eq!(config.top_k, deserialized.top_k);
    assert_eq!(config.top_p, deserialized.top_p);
    assert_eq!(config.repetition_penalty, deserialized.repetition_penalty);
    assert_eq!(config.do_sample, deserialized.do_sample);
    assert_eq!(config.seed, deserialized.seed);
}

#[test]
fn test_generation_config_optional_fields() {
    let config = GenerationConfig {
        max_new_tokens: 100,
        temperature: 0.5,
        top_k: None,
        top_p: None,
        repetition_penalty: 1.0,
        do_sample: true,
        seed: None,
    };

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: GenerationConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config.top_k, None);
    assert_eq!(config.top_p, None);
    assert_eq!(config.seed, None);
    assert_eq!(deserialized.top_k, None);
    assert_eq!(deserialized.top_p, None);
    assert_eq!(deserialized.seed, None);
}

#[test]
fn test_model_metadata_serialization() {
    let metadata = ModelMetadata {
        name: "test-model".to_string(),
        version: "1.0.0".to_string(),
        architecture: "BitNet".to_string(),
        vocab_size: 32000,
        context_length: 2048,
        quantization: Some(QuantizationType::TL1),
    };

    let json = serde_json::to_string(&metadata).unwrap();
    let deserialized: ModelMetadata = serde_json::from_str(&json).unwrap();

    assert_eq!(metadata.name, deserialized.name);
    assert_eq!(metadata.version, deserialized.version);
    assert_eq!(metadata.architecture, deserialized.architecture);
    assert_eq!(metadata.vocab_size, deserialized.vocab_size);
    assert_eq!(metadata.context_length, deserialized.context_length);
    assert_eq!(metadata.quantization, deserialized.quantization);
}

#[test]
fn test_model_metadata_optional_quantization() {
    let metadata = ModelMetadata {
        name: "unquantized-model".to_string(),
        version: "2.0.0".to_string(),
        architecture: "Transformer".to_string(),
        vocab_size: 50000,
        context_length: 4096,
        quantization: None,
    };

    let json = serde_json::to_string(&metadata).unwrap();
    let deserialized: ModelMetadata = serde_json::from_str(&json).unwrap();

    assert_eq!(metadata.quantization, None);
    assert_eq!(deserialized.quantization, None);
}

#[test]
fn test_performance_metrics_default() {
    let metrics = PerformanceMetrics::default();

    assert_eq!(metrics.tokens_per_second, 0.0);
    assert_eq!(metrics.latency_ms, 0.0);
    assert_eq!(metrics.memory_usage_mb, 0.0);
    assert_eq!(metrics.gpu_utilization, None);
}

#[test]
fn test_performance_metrics_serialization() {
    let metrics = PerformanceMetrics {
        tokens_per_second: 125.5,
        latency_ms: 8.2,
        memory_usage_mb: 1024.0,
        gpu_utilization: Some(85.5),
    };

    let json = serde_json::to_string(&metrics).unwrap();
    let deserialized: PerformanceMetrics = serde_json::from_str(&json).unwrap();

    assert_eq!(metrics.tokens_per_second, deserialized.tokens_per_second);
    assert_eq!(metrics.latency_ms, deserialized.latency_ms);
    assert_eq!(metrics.memory_usage_mb, deserialized.memory_usage_mb);
    assert_eq!(metrics.gpu_utilization, deserialized.gpu_utilization);
}

#[test]
fn test_performance_metrics_optional_gpu() {
    let metrics = PerformanceMetrics {
        tokens_per_second: 100.0,
        latency_ms: 10.0,
        memory_usage_mb: 512.0,
        gpu_utilization: None,
    };

    let json = serde_json::to_string(&metrics).unwrap();
    let deserialized: PerformanceMetrics = serde_json::from_str(&json).unwrap();

    assert_eq!(metrics.gpu_utilization, None);
    assert_eq!(deserialized.gpu_utilization, None);
}

// Property-based tests
proptest! {
    #[test]
    fn test_generation_config_roundtrip_serialization(
        max_new_tokens in 1usize..10000,
        temperature in 0.1f32..2.0,
        top_k in proptest::option::of(1usize..1000),
        top_p in proptest::option::of(0.1f32..1.0),
        repetition_penalty in 0.1f32..2.0,
        do_sample in any::<bool>(),
        seed in proptest::option::of(any::<u64>())
    ) {
        let config = GenerationConfig {
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            do_sample,
            seed,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: GenerationConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.max_new_tokens, deserialized.max_new_tokens);
        assert_eq!(config.temperature, deserialized.temperature);
        assert_eq!(config.top_k, deserialized.top_k);
        assert_eq!(config.top_p, deserialized.top_p);
        assert_eq!(config.repetition_penalty, deserialized.repetition_penalty);
        assert_eq!(config.do_sample, deserialized.do_sample);
        assert_eq!(config.seed, deserialized.seed);
    }

    #[test]
    fn test_model_metadata_roundtrip_serialization(
        name in "\\PC{1,100}",
        version in "\\PC{1,20}",
        architecture in "\\PC{1,50}",
        vocab_size in 1usize..1000000,
        context_length in 1usize..100000,
        quantization in proptest::option::of(prop_oneof![
            Just(QuantizationType::I2S),
            Just(QuantizationType::TL1),
            Just(QuantizationType::TL2)
        ])
    ) {
        let metadata = ModelMetadata {
            name: name.clone(),
            version: version.clone(),
            architecture: architecture.clone(),
            vocab_size,
            context_length,
            quantization,
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: ModelMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(metadata.name, deserialized.name);
        assert_eq!(metadata.version, deserialized.version);
        assert_eq!(metadata.architecture, deserialized.architecture);
        assert_eq!(metadata.vocab_size, deserialized.vocab_size);
        assert_eq!(metadata.context_length, deserialized.context_length);
        assert_eq!(metadata.quantization, deserialized.quantization);
    }

    #[test]
    fn test_performance_metrics_roundtrip_serialization(
        tokens_per_second in 0.0f64..10000.0,
        latency_ms in 0.0f64..1000.0,
        memory_usage_mb in 0.0f64..100000.0,
        gpu_utilization in proptest::option::of(0.0f64..100.0)
    ) {
        let metrics = PerformanceMetrics {
            tokens_per_second,
            latency_ms,
            memory_usage_mb,
            gpu_utilization,
        };

        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: PerformanceMetrics = serde_json::from_str(&json).unwrap();

        // Use approximate equality for floating-point values due to serialization precision
        assert!((metrics.tokens_per_second - deserialized.tokens_per_second).abs() < 1e-10);
        assert!((metrics.latency_ms - deserialized.latency_ms).abs() < 1e-10);
        assert!((metrics.memory_usage_mb - deserialized.memory_usage_mb).abs() < 1e-10);

        // Compare optional GPU utilization with approximate equality
        match (metrics.gpu_utilization, deserialized.gpu_utilization) {
            (Some(a), Some(b)) => assert!((a - b).abs() < 1e-10),
            (None, None) => {},
            _ => panic!("GPU utilization mismatch: {:?} vs {:?}", metrics.gpu_utilization, deserialized.gpu_utilization),
        }
    }

    #[test]
    fn test_device_cuda_id_bounds(id in any::<usize>()) {
        let device = Device::Cuda(id);
        match device {
            Device::Cuda(actual_id) => assert_eq!(id, actual_id),
            _ => panic!("Expected CUDA device"),
        }
    }
}

#[test]
fn test_type_debug_formatting() {
    // Test that all types implement Debug properly
    let qtype = QuantizationType::I2S;
    let debug_str = format!("{:?}", qtype);
    assert!(debug_str.contains("I2S"));

    let device = Device::Cuda(1);
    let debug_str = format!("{:?}", device);
    assert!(debug_str.contains("Cuda"));
    assert!(debug_str.contains("1"));

    let config = GenerationConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("GenerationConfig"));
}

#[test]
fn test_type_clone() {
    // Test that all types implement Clone properly
    let qtype = QuantizationType::TL1;
    let cloned = qtype.clone();
    assert_eq!(qtype, cloned);

    let device = Device::Metal;
    let cloned = device.clone();
    assert_eq!(device, cloned);

    let config = GenerationConfig::default();
    let cloned = config.clone();
    assert_eq!(config.max_new_tokens, cloned.max_new_tokens);
}

#[test]
fn test_send_sync_traits() {
    // Ensure all types are Send + Sync for use in async contexts
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<QuantizationType>();
    assert_send_sync::<DeviceType>();
    assert_send_sync::<Device>();
    assert_send_sync::<GenerationConfig>();
    assert_send_sync::<ModelMetadata>();
    assert_send_sync::<PerformanceMetrics>();
}
