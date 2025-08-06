//! Comprehensive tests for bitnet-common
//! Covers edge cases, error conditions, and integration scenarios

use bitnet_common::*;
use candle_core::DType;
use std::env;
use std::fs;
use tempfile::{NamedTempFile, TempDir};
use std::io::Write;

/// Test comprehensive configuration scenarios
mod config_comprehensive {
    use super::*;

    #[test]
    fn test_config_validation_edge_cases() {
        // Test zero values (should fail)
        let mut config = BitNetConfig::default();
        config.model.vocab_size = 0;
        assert!(config.validate().is_err());

        config = BitNetConfig::default();
        config.model.hidden_size = 0;
        assert!(config.validate().is_err());

        config = BitNetConfig::default();
        config.model.num_layers = 0;
        assert!(config.validate().is_err());

        config = BitNetConfig::default();
        config.model.num_heads = 0;
        assert!(config.validate().is_err());

        // Test invalid hidden_size/num_heads ratio
        config = BitNetConfig::default();
        config.model.hidden_size = 100;
        config.model.num_heads = 7; // 100 is not divisible by 7
        assert!(config.validate().is_err());

        // Test invalid temperature
        config = BitNetConfig::default();
        config.inference.temperature = 0.0;
        assert!(config.validate().is_err());

        config.inference.temperature = -1.0;
        assert!(config.validate().is_err());

        // Test invalid top_p
        config = BitNetConfig::default();
        config.inference.top_p = Some(0.0);
        assert!(config.validate().is_err());

        config.inference.top_p = Some(1.5);
        assert!(config.validate().is_err());

        // Test invalid repetition_penalty
        config = BitNetConfig::default();
        config.inference.repetition_penalty = 0.0;
        assert!(config.validate().is_err());

        // Test invalid block_size (not power of 2)
        config = BitNetConfig::default();
        config.quantization.block_size = 63; // Not power of 2
        assert!(config.validate().is_err());

        // Test invalid precision
        config = BitNetConfig::default();
        config.quantization.precision = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_file_error_handling() {
        // Test non-existent file
        let result = BitNetConfig::from_file("non_existent_file.toml");
        assert!(result.is_err());

        // Test invalid TOML
        let mut temp_file = NamedTempFile::with_suffix(".toml").unwrap();
        temp_file.write_all(b"invalid toml content [[[").unwrap();
        temp_file.flush().unwrap();
        
        let result = BitNetConfig::from_file(temp_file.path());
        assert!(result.is_err());

        // Test invalid JSON
        let mut temp_file = NamedTempFile::with_suffix(".json").unwrap();
        temp_file.write_all(b"{ invalid json }").unwrap();
        temp_file.flush().unwrap();
        
        let result = BitNetConfig::from_file(temp_file.path());
        assert!(result.is_err());

        // Test unsupported file extension
        let mut temp_file = NamedTempFile::with_suffix(".yaml").unwrap();
        temp_file.write_all(b"model:\n  vocab_size: 1000").unwrap();
        temp_file.flush().unwrap();
        
        let result = BitNetConfig::from_file(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_env_variable_error_handling() {
        let _lock = acquire_env_lock();
        
        // Test invalid numeric values
        env::set_var("BITNET_VOCAB_SIZE", "not_a_number");
        let result = BitNetConfig::from_env();
        assert!(result.is_err());
        env::remove_var("BITNET_VOCAB_SIZE");

        env::set_var("BITNET_TEMPERATURE", "invalid_float");
        let result = BitNetConfig::from_env();
        assert!(result.is_err());
        env::remove_var("BITNET_TEMPERATURE");

        // Test invalid enum values
        env::set_var("BITNET_MODEL_FORMAT", "invalid_format");
        let result = BitNetConfig::from_env();
        assert!(result.is_err());
        env::remove_var("BITNET_MODEL_FORMAT");

        env::set_var("BITNET_QUANTIZATION_TYPE", "INVALID_TYPE");
        let result = BitNetConfig::from_env();
        assert!(result.is_err());
        env::remove_var("BITNET_QUANTIZATION_TYPE");

        env::set_var("BITNET_USE_GPU", "maybe");
        let result = BitNetConfig::from_env();
        assert!(result.is_err());
        env::remove_var("BITNET_USE_GPU");
    }

    #[test]
    fn test_config_builder_comprehensive() {
        // Test builder with all options
        let config = BitNetConfig::builder()
            .vocab_size(50000)
            .hidden_size(4096)
            .num_layers(32)
            .num_heads(32)
            .max_length(2048)
            .temperature(0.8)
            .top_k(Some(40))
            .top_p(Some(0.95))
            .quantization_type(QuantizationType::TL2)
            .use_gpu(true)
            .num_threads(Some(8))
            .batch_size(4)
            .build()
            .unwrap();

        assert_eq!(config.model.vocab_size, 50000);
        assert_eq!(config.model.hidden_size, 4096);
        assert_eq!(config.model.num_layers, 32);
        assert_eq!(config.model.num_heads, 32);
        assert_eq!(config.inference.max_length, 2048);
        assert_eq!(config.inference.temperature, 0.8);
        assert_eq!(config.inference.top_k, Some(40));
        assert_eq!(config.inference.top_p, Some(0.95));
        assert_eq!(config.quantization.quantization_type, QuantizationType::TL2);
        assert_eq!(config.performance.use_gpu, true);
        assert_eq!(config.performance.num_threads, Some(8));
        assert_eq!(config.performance.batch_size, 4);

        // Test builder validation failure
        let result = BitNetConfig::builder()
            .vocab_size(0) // Invalid
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_limit_parsing() {
        let _lock = acquire_env_lock();
        
        // Test various memory limit formats
        env::set_var("BITNET_MEMORY_LIMIT", "1GB");
        let mut config = BitNetConfig::default();
        config.apply_env_overrides().unwrap();
        assert_eq!(config.performance.memory_limit, Some(1024 * 1024 * 1024));

        env::set_var("BITNET_MEMORY_LIMIT", "512MB");
        let mut config = BitNetConfig::default();
        config.apply_env_overrides().unwrap();
        assert_eq!(config.performance.memory_limit, Some(512 * 1024 * 1024));

        env::set_var("BITNET_MEMORY_LIMIT", "1024KB");
        let mut config = BitNetConfig::default();
        config.apply_env_overrides().unwrap();
        assert_eq!(config.performance.memory_limit, Some(1024 * 1024));

        env::set_var("BITNET_MEMORY_LIMIT", "1048576");
        let mut config = BitNetConfig::default();
        config.apply_env_overrides().unwrap();
        assert_eq!(config.performance.memory_limit, Some(1048576));

        env::set_var("BITNET_MEMORY_LIMIT", "none");
        let mut config = BitNetConfig::default();
        config.apply_env_overrides().unwrap();
        assert_eq!(config.performance.memory_limit, None);

        // Test invalid memory limit
        env::set_var("BITNET_MEMORY_LIMIT", "invalid_size");
        let mut config = BitNetConfig::default();
        assert!(config.apply_env_overrides().is_err());

        env::remove_var("BITNET_MEMORY_LIMIT");
    }

    // Helper function to acquire environment lock for tests
    fn acquire_env_lock() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        ENV_LOCK.lock().unwrap()
    }
}

/// Test tensor operations comprehensively
mod tensor_comprehensive {
    use super::*;

    #[test]
    fn test_concrete_tensor_operations() {
        // Test creation with different data types
        let data_f32 = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = ConcreteTensor::new(data_f32.clone(), vec![2, 2], DType::F32);
        
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.device(), &Device::Cpu);

        // Test as_slice
        let slice: &[f32] = tensor.as_slice().unwrap();
        assert_eq!(slice, &data_f32);

        // Test with different shapes
        let tensor = ConcreteTensor::new(vec![1.0f32; 24], vec![2, 3, 4], DType::F32);
        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.numel(), 24);

        // Test empty tensor
        let tensor = ConcreteTensor::new(Vec::<f32>::new(), vec![0], DType::F32);
        assert_eq!(tensor.numel(), 0);
        assert!(tensor.as_slice::<f32>().unwrap().is_empty());
    }

    #[test]
    fn test_tensor_type_conversions() {
        // Test different data types
        let data_i32 = vec![1i32, -2, 3, -4];
        let tensor = ConcreteTensor::new(data_i32.clone(), vec![4], DType::I32);
        
        let slice: &[i32] = tensor.as_slice().unwrap();
        assert_eq!(slice, &data_i32);

        // Test u8 data
        let data_u8 = vec![0u8, 255, 128, 64];
        let tensor = ConcreteTensor::new(data_u8.clone(), vec![4], DType::U8);
        
        let slice: &[u8] = tensor.as_slice().unwrap();
        assert_eq!(slice, &data_u8);
    }

    #[test]
    fn test_tensor_error_conditions() {
        let tensor = ConcreteTensor::new(vec![1.0f32; 4], vec![2, 2], DType::F32);
        
        // Test wrong type access
        let result: Result<&[i32]> = tensor.as_slice();
        assert!(result.is_err());

        // Test device operations (should work for CPU)
        assert_eq!(tensor.device(), &Device::Cpu);
        
        // Test CUDA device (should be placeholder)
        let cuda_tensor = ConcreteTensor::new(vec![1.0f32; 4], vec![2, 2], DType::F32);
        // Note: CUDA operations are not implemented yet, so we just test the interface
    }

    #[test]
    fn test_mock_tensor() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = MockTensor::new(data.clone());
        
        assert_eq!(tensor.shape(), &[4]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.device(), &Device::Cpu);
        assert_eq!(tensor.numel(), 4);

        let slice: &[f32] = tensor.as_slice().unwrap();
        assert_eq!(slice, &data);
    }
}

/// Test error handling comprehensively
mod error_comprehensive {
    use super::*;

    #[test]
    fn test_error_types() {
        // Test BitNetError variants
        let config_error = BitNetError::Config("test config error".to_string());
        assert!(matches!(config_error, BitNetError::Config(_)));

        let model_error = BitNetError::Model(ModelError::InvalidFormat("test".to_string()));
        assert!(matches!(model_error, BitNetError::Model(_)));

        let kernel_error = BitNetError::Kernel(KernelError::NoProvider);
        assert!(matches!(kernel_error, BitNetError::Kernel(_)));

        let quantization_error = BitNetError::Quantization(QuantizationError::InvalidBlockSize(0));
        assert!(matches!(quantization_error, BitNetError::Quantization(_)));

        let inference_error = BitNetError::Inference(InferenceError::ModelNotLoaded);
        assert!(matches!(inference_error, BitNetError::Inference(_)));
    }

    #[test]
    fn test_error_display() {
        let error = BitNetError::Config("Configuration validation failed".to_string());
        let display_str = format!("{}", error);
        assert!(display_str.contains("Configuration validation failed"));

        let error = ModelError::InvalidFormat("GGUF".to_string());
        let display_str = format!("{}", error);
        assert!(display_str.contains("GGUF"));
    }

    #[test]
    fn test_error_conversions() {
        // Test From implementations
        let model_error = ModelError::InvalidFormat("test".to_string());
        let bitnet_error: BitNetError = model_error.into();
        assert!(matches!(bitnet_error, BitNetError::Model(_)));

        let kernel_error = KernelError::NoProvider;
        let bitnet_error: BitNetError = kernel_error.into();
        assert!(matches!(bitnet_error, BitNetError::Kernel(_)));

        let quantization_error = QuantizationError::InvalidBlockSize(0);
        let bitnet_error: BitNetError = quantization_error.into();
        assert!(matches!(bitnet_error, BitNetError::Quantization(_)));

        let inference_error = InferenceError::ModelNotLoaded;
        let bitnet_error: BitNetError = inference_error.into();
        assert!(matches!(bitnet_error, BitNetError::Inference(_)));
    }
}

/// Test device handling
mod device_comprehensive {
    use super::*;

    #[test]
    fn test_device_types() {
        let cpu_device = Device::Cpu;
        assert!(matches!(cpu_device, Device::Cpu));

        let cuda_device = Device::Cuda(0);
        assert!(matches!(cuda_device, Device::Cuda(0)));

        let cuda_device_1 = Device::Cuda(1);
        assert!(matches!(cuda_device_1, Device::Cuda(1)));
    }

    #[test]
    fn test_device_equality() {
        assert_eq!(Device::Cpu, Device::Cpu);
        assert_eq!(Device::Cuda(0), Device::Cuda(0));
        assert_ne!(Device::Cpu, Device::Cuda(0));
        assert_ne!(Device::Cuda(0), Device::Cuda(1));
    }
}

/// Integration tests
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_config_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");

        // Create a config file
        let config_content = r#"
[model]
vocab_size = 32000
hidden_size = 4096
num_layers = 32
num_heads = 32

[inference]
temperature = 0.8
max_length = 2048

[quantization]
quantization_type = "TL2"
block_size = 64

[performance]
use_gpu = false
batch_size = 1
"#;
        fs::write(&config_path, config_content).unwrap();

        // Load config from file
        let config = BitNetConfig::from_file(&config_path).unwrap();
        assert_eq!(config.model.vocab_size, 32000);
        assert_eq!(config.inference.temperature, 0.8);
        assert_eq!(config.quantization.quantization_type, QuantizationType::TL2);

        // Test with environment overrides
        let _lock = acquire_env_lock();
        env::set_var("BITNET_TEMPERATURE", "0.9");
        env::set_var("BITNET_USE_GPU", "true");

        let config = BitNetConfig::from_file_with_env(&config_path).unwrap();
        assert_eq!(config.inference.temperature, 0.9);
        assert_eq!(config.performance.use_gpu, true);

        env::remove_var("BITNET_TEMPERATURE");
        env::remove_var("BITNET_USE_GPU");
    }

    #[test]
    fn test_config_precedence_comprehensive() {
        let _lock = acquire_env_lock();
        
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("precedence_test.toml");

        // Create config file with specific values
        let config_content = r#"
[model]
vocab_size = 30000

[inference]
temperature = 0.7
max_length = 1024

[performance]
use_gpu = false
batch_size = 2
"#;
        fs::write(&config_path, config_content).unwrap();

        // Set environment variables that should override file values
        env::set_var("BITNET_VOCAB_SIZE", "40000");
        env::set_var("BITNET_TEMPERATURE", "0.9");
        env::set_var("BITNET_USE_GPU", "true");
        // Don't set BITNET_MAX_LENGTH or BITNET_BATCH_SIZE

        let config = ConfigLoader::load_with_precedence(Some(&config_path)).unwrap();

        // Environment should override file
        assert_eq!(config.model.vocab_size, 40000);
        assert_eq!(config.inference.temperature, 0.9);
        assert_eq!(config.performance.use_gpu, true);

        // File should override defaults
        assert_eq!(config.inference.max_length, 1024);
        assert_eq!(config.performance.batch_size, 2);

        // Defaults should be used where neither file nor env specify
        assert_eq!(config.model.hidden_size, 4096); // Default value

        // Clean up
        env::remove_var("BITNET_VOCAB_SIZE");
        env::remove_var("BITNET_TEMPERATURE");
        env::remove_var("BITNET_USE_GPU");
    }

    // Helper function to acquire environment lock for tests
    fn acquire_env_lock() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        ENV_LOCK.lock().unwrap()
    }
}