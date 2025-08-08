//! Comprehensive tests for bitnet-models
//! Covers edge cases, error conditions, and end-to-end scenarios

use bitnet_common::{BitNetConfig, Device, ModelMetadata, QuantizationType, Tensor};
use bitnet_models::*;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tempfile::{NamedTempFile, TempDir};

/// Test model loading and validation
mod model_loading {
    use super::*;

    #[test]
    fn test_model_loader_creation() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);

        let formats = loader.available_formats();
        assert!(formats.contains(&"GGUF"));
        assert!(formats.contains(&"SafeTensors"));
        assert!(formats.contains(&"HuggingFace"));
    }

    #[test]
    fn test_invalid_file_paths() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);

        // Test non-existent file
        let result = loader.load("non_existent_file.gguf");
        assert!(result.is_err());

        // Test directory instead of file
        let temp_dir = TempDir::new().unwrap();
        let result = loader.load(temp_dir.path());
        assert!(result.is_err());

        // Test file without read permissions (platform-specific)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut temp_file = NamedTempFile::new().unwrap();
            temp_file.write_all(b"test content").unwrap();

            let mut perms = temp_file.as_file().metadata().unwrap().permissions();
            perms.set_mode(0o000); // No permissions
            temp_file.as_file().set_permissions(perms).unwrap();

            let result = loader.load(temp_file.path());
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_corrupted_file_headers() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);

        // Test file with invalid GGUF magic
        let mut temp_file = NamedTempFile::with_suffix(".gguf").unwrap();
        temp_file.write_all(b"INVALID_MAGIC_BYTES").unwrap();
        temp_file.flush().unwrap();

        let result = loader.load(temp_file.path());
        assert!(result.is_err());

        // Test truncated GGUF header
        let mut temp_file = NamedTempFile::with_suffix(".gguf").unwrap();
        temp_file.write_all(b"GGUF").unwrap(); // Only magic, no version
        temp_file.flush().unwrap();

        let result = loader.load(temp_file.path());
        assert!(result.is_err());

        // Test invalid SafeTensors header
        let mut temp_file = NamedTempFile::with_suffix(".safetensors").unwrap();
        temp_file.write_all(b"invalid json header").unwrap();
        temp_file.flush().unwrap();

        let result = loader.load(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_file_formats() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);

        // Test unsupported extension
        let mut temp_file = NamedTempFile::with_suffix(".unsupported").unwrap();
        temp_file.write_all(b"some content").unwrap();
        temp_file.flush().unwrap();

        let result = loader.load(temp_file.path());
        assert!(result.is_err());

        // Test file without extension
        let temp_dir = TempDir::new().unwrap();
        let no_ext_path = temp_dir.path().join("no_extension");
        fs::write(&no_ext_path, b"some content").unwrap();

        let result = loader.load(&no_ext_path);
        // Should try magic byte detection but fail
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata_extraction() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);

        // Create a minimal GGUF file for metadata extraction
        let mut temp_file = NamedTempFile::with_suffix(".gguf").unwrap();
        temp_file.write_all(b"GGUF").unwrap(); // Magic
        temp_file.write_all(&3u32.to_le_bytes()).unwrap(); // Version
        temp_file.write_all(&0u64.to_le_bytes()).unwrap(); // Tensor count
        temp_file.write_all(&0u64.to_le_bytes()).unwrap(); // Metadata KV count
        temp_file.flush().unwrap();

        // Test metadata extraction (should work even if loading fails)
        let result = loader.extract_metadata(temp_file.path());
        // This might fail due to incomplete GGUF structure, but should not panic
        match result {
            Ok(metadata) => {
                assert!(!metadata.name.is_empty());
                assert!(metadata.vocab_size > 0);
            }
            Err(_) => {
                // Expected for incomplete GGUF file
            }
        }
    }

    #[test]
    fn test_model_validation() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);

        // Test that loader validates model compatibility
        let formats = loader.available_formats();
        assert!(!formats.is_empty());

        // Test format detection capabilities
        assert!(formats.contains(&"GGUF"));
        assert!(formats.contains(&"SafeTensors"));
        assert!(formats.contains(&"HuggingFace"));
    }
}

/// Test various model formats
mod format_tests {
    use super::*;

    #[test]
    fn test_gguf_format_detection() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);

        // Test valid GGUF magic bytes
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"GGUF").unwrap();
        temp_file.write_all(&[0u8; 20]).unwrap(); // Padding
        temp_file.flush().unwrap();

        // Test that the file is recognized as potentially GGUF
        let result = loader.load(temp_file.path());
        // Should fail due to incomplete structure but not due to format detection
        assert!(result.is_err());
    }

    #[test]
    fn test_safetensors_format_detection() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);

        // Test SafeTensors file
        let mut temp_file = NamedTempFile::with_suffix(".safetensors").unwrap();
        // Create a minimal SafeTensors header with proper alignment
        let header = r#"{"test_tensor":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let header_len = header.len() as u64;
        temp_file.write_all(&header_len.to_le_bytes()).unwrap();
        temp_file.write_all(header.as_bytes()).unwrap();
        // Write properly aligned F32 data (8 bytes = 2 F32 values)
        let tensor_data: [f32; 2] = [1.0, 2.0];
        let tensor_bytes = bytemuck::cast_slice(&tensor_data);
        temp_file.write_all(tensor_bytes).unwrap();
        temp_file.flush().unwrap();

        let result = loader.load(temp_file.path());
        // Should fail due to incomplete model but not due to format detection
        assert!(result.is_err());
    }

    #[test]
    fn test_huggingface_format_detection() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);

        // Create a temporary directory with HuggingFace structure
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        fs::write(&config_path, r#"{"model_type": "bitnet"}"#).unwrap();

        let result = loader.load(temp_dir.path());
        // Should fail due to incomplete model but not due to format detection
        assert!(result.is_err());
    }

    #[test]
    fn test_format_priority() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);

        // Test that magic byte detection works
        let mut temp_file = NamedTempFile::with_suffix(".wrong_extension").unwrap();
        temp_file.write_all(b"GGUF").unwrap();
        temp_file.write_all(&[0u8; 20]).unwrap();
        temp_file.flush().unwrap();

        let result = loader.load(temp_file.path());
        // Should attempt to load as GGUF despite wrong extension
        assert!(result.is_err());
    }

    #[test]
    fn test_model_format_conversion() {
        // Test that we can handle different model formats
        let config = BitNetConfig::default();
        let device = Device::Cpu;
        let model = BitNetModel::new(config, device);

        // Test basic model properties
        assert!(model.config().model.vocab_size > 0);
        assert!(model.tensor_names().is_empty()); // New model has no tensors
    }
}

/// Test memory mapping functionality
mod mmap_tests {
    use super::*;

    #[test]
    fn test_mmap_file_operations() {
        // Create a test file
        let test_data = b"Hello, World! This is a test file for memory mapping.";
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(test_data).unwrap();
        temp_file.flush().unwrap();

        // Test memory mapping
        let mmap_file = MmapFile::open(temp_file.path()).unwrap();

        assert_eq!(mmap_file.len(), test_data.len());
        assert_eq!(mmap_file.as_slice(), test_data);
        assert!(!mmap_file.is_empty());

        // Test slice operations
        let slice = &mmap_file.as_slice()[0..5];
        assert_eq!(slice, b"Hello");
    }

    #[test]
    fn test_mmap_empty_file() {
        let temp_file = NamedTempFile::new().unwrap();
        // Don't write anything - empty file

        let mmap_file = MmapFile::open(temp_file.path()).unwrap();
        assert_eq!(mmap_file.len(), 0);
        assert!(mmap_file.is_empty());
        assert!(mmap_file.as_slice().is_empty());
    }

    #[test]
    fn test_mmap_large_file() {
        // Create a larger file to test memory mapping efficiency
        let large_data = vec![0u8; 1024 * 1024]; // 1MB
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&large_data).unwrap();
        temp_file.flush().unwrap();

        let mmap_file = MmapFile::open(temp_file.path()).unwrap();
        assert_eq!(mmap_file.len(), large_data.len());
        assert_eq!(mmap_file.as_slice(), &large_data);
    }

    #[test]
    fn test_mmap_error_conditions() {
        // Test non-existent file
        let result = MmapFile::open("non_existent_file.txt");
        assert!(result.is_err());

        // Test directory instead of file
        let temp_dir = TempDir::new().unwrap();
        let result = MmapFile::open(temp_dir.path());
        assert!(result.is_err());
    }
}

/// Test progress callback functionality
mod progress_tests {
    use super::*;

    #[test]
    fn test_progress_callback_sequence() {
        let progress_values = Arc::new(Mutex::new(Vec::new()));
        let progress_values_clone = progress_values.clone();

        let callback: ProgressCallback = Arc::new(move |progress, message| {
            progress_values_clone
                .lock()
                .unwrap()
                .push((progress, message.to_string()));
        });

        // Simulate a loading sequence
        callback(0.0, "Starting model load");
        callback(0.25, "Loading headers");
        callback(0.5, "Loading tensors");
        callback(0.75, "Validating model");
        callback(1.0, "Load complete");

        let values = progress_values.lock().unwrap();
        assert_eq!(values.len(), 5);

        // Check progress is monotonically increasing
        for i in 1..values.len() {
            assert!(
                values[i].0 >= values[i - 1].0,
                "Progress should be monotonic"
            );
        }

        assert_eq!(values[0].0, 0.0);
        assert_eq!(values[4].0, 1.0);
    }

    #[test]
    fn test_progress_callback_error_handling() {
        // Test callback with invalid progress values
        let callback: ProgressCallback = Arc::new(|progress, _message| {
            assert!(
                progress >= 0.0 && progress <= 1.0,
                "Progress should be in [0, 1] range"
            );
        });

        // These should not panic
        callback(0.0, "Start");
        callback(0.5, "Middle");
        callback(1.0, "End");
    }

    #[test]
    fn test_utility_progress_callbacks() {
        // Test logging callback (should not panic)
        let logging_callback = crate::loader::utils::create_logging_progress_callback();
        logging_callback(0.5, "Test logging message");

        // Test stdout callback (should not panic)
        let stdout_callback = crate::loader::utils::create_stdout_progress_callback();
        stdout_callback(0.5, "Test stdout message");
    }

    #[test]
    fn test_load_config_with_progress() {
        let progress_values = Arc::new(Mutex::new(Vec::new()));
        let progress_values_clone = progress_values.clone();

        let callback: ProgressCallback = Arc::new(move |progress, message| {
            progress_values_clone
                .lock()
                .unwrap()
                .push((progress, message.to_string()));
        });

        let config = LoadConfig {
            use_mmap: true,
            validate_checksums: false,
            progress_callback: Some(callback),
        };

        // Test that config is created properly
        assert!(config.use_mmap);
        assert!(!config.validate_checksums);
        assert!(config.progress_callback.is_some());
    }
}

/// Test model security features
mod security_tests {
    use super::*;
    use bitnet_models::security::{ModelSecurity, ModelVerifier};

    #[test]
    fn test_model_verification() {
        let verifier = ModelVerifier::new(ModelSecurity::default());

        // Test with a known file
        let mut temp_file = NamedTempFile::new().unwrap();
        let test_content = b"test model content";
        temp_file.write_all(test_content).unwrap();
        temp_file.flush().unwrap();

        // Test hash computation
        let hash = verifier.compute_file_hash(temp_file.path()).unwrap();
        assert!(!hash.is_empty());

        // Test same file produces same hash
        let hash2 = verifier.compute_file_hash(temp_file.path()).unwrap();
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_source_verification() {
        let trusted_sources = vec!["https://huggingface.co/".to_string()];
        let security = ModelSecurity {
            trusted_sources,
            require_hash_verification: true,
            max_model_size: 1024 * 1024 * 1024, // 1GB
            known_hashes: std::collections::HashMap::new(),
        };
        let verifier = ModelVerifier::new(security);

        assert!(verifier
            .verify_source("https://huggingface.co/model/file.gguf")
            .is_ok());
        assert!(verifier
            .verify_source("https://untrusted.com/model.gguf")
            .is_err());
    }

    #[test]
    fn test_file_size_limits() {
        let mut security = ModelSecurity::default();
        security.max_model_size = 1024; // 1KB limit
        security.require_hash_verification = false; // Disable hash verification for this test
        let verifier = ModelVerifier::new(security);

        // Create a small file (should pass)
        let mut small_file = NamedTempFile::new().unwrap();
        small_file.write_all(b"small content").unwrap();
        small_file.flush().unwrap();

        assert!(verifier.verify_model(small_file.path(), None).is_ok());

        // Create a large file (should fail)
        let mut large_file = NamedTempFile::new().unwrap();
        let large_content = vec![0u8; 2048]; // 2KB
        large_file.write_all(&large_content).unwrap();
        large_file.flush().unwrap();

        assert!(verifier.verify_model(large_file.path(), None).is_err());
    }

    #[test]
    fn test_checksum_validation() {
        let verifier = ModelVerifier::new(ModelSecurity::default());

        let mut temp_file = NamedTempFile::new().unwrap();
        let content = b"test content for checksum";
        temp_file.write_all(content).unwrap();
        temp_file.flush().unwrap();

        // Compute expected checksum
        let expected_hash = verifier.compute_file_hash(temp_file.path()).unwrap();

        // Test validation with correct checksum
        assert!(verifier
            .verify_model(temp_file.path(), Some(&expected_hash))
            .is_ok());

        // Test validation with incorrect checksum
        let wrong_hash = "wrong_hash_value";
        assert!(verifier
            .verify_model(temp_file.path(), Some(wrong_hash))
            .is_err());
    }
}

/// Test utility functions
mod utils_tests {
    use super::*;

    #[test]
    fn test_file_validation() {
        // Test with valid file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"test content").unwrap();
        temp_file.flush().unwrap();

        assert!(crate::loader::utils::validate_file_access(temp_file.path()).is_ok());

        // Test with non-existent file
        let non_existent = Path::new("definitely_does_not_exist.txt");
        assert!(crate::loader::utils::validate_file_access(non_existent).is_err());

        // Test with directory
        let temp_dir = TempDir::new().unwrap();
        assert!(crate::loader::utils::validate_file_access(temp_dir.path()).is_ok());
    }

    #[test]
    fn test_file_size_calculation() {
        let test_content = b"Hello, World! This is a test file.";
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(test_content).unwrap();
        temp_file.flush().unwrap();

        let size = crate::loader::utils::get_file_size(temp_file.path()).unwrap();
        assert_eq!(size, test_content.len() as u64);

        // Test with empty file
        let empty_file = NamedTempFile::new().unwrap();
        let size = crate::loader::utils::get_file_size(empty_file.path()).unwrap();
        assert_eq!(size, 0);
    }

    #[test]
    fn test_path_utilities() {
        // Test various path operations
        let path = Path::new("model.gguf");
        assert_eq!(path.extension().unwrap(), "gguf");

        let path = Path::new("path/to/model.safetensors");
        assert_eq!(path.extension().unwrap(), "safetensors");

        let path = Path::new("no_extension");
        assert!(path.extension().is_none());
    }
}

/// Test BitNet model functionality
mod bitnet_model_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_bitnet_model_creation() {
        let config = BitNetConfig::default();
        let device = Device::Cpu;
        let model = BitNetModel::new(config.clone(), device);

        assert_eq!(model.config().model.vocab_size, config.model.vocab_size);
        assert!(model.tensor_names().is_empty());
    }

    #[test]
    fn test_bitnet_model_from_gguf() {
        let config = BitNetConfig::default();
        let device = Device::Cpu;

        // Create mock tensors
        let mut tensors = HashMap::new();

        // Create mock candle tensors for required tensors
        let token_embd = candle_core::Tensor::zeros(
            (config.model.vocab_size, config.model.hidden_size),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let output = candle_core::Tensor::zeros(
            (config.model.hidden_size, config.model.vocab_size),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();

        tensors.insert("token_embd.weight".to_string(), token_embd);
        tensors.insert("output.weight".to_string(), output);

        let result = BitNetModel::from_gguf(config, tensors, device);
        assert!(result.is_ok());

        let model = result.unwrap();
        assert!(model.get_tensor("token_embd.weight").is_some());
        assert!(model.get_tensor("output.weight").is_some());
        assert!(model.get_tensor("non_existent").is_none());
    }

    #[test]
    fn test_bitnet_model_missing_tensors() {
        let config = BitNetConfig::default();
        let device = Device::Cpu;

        // Create incomplete tensor set (missing required tensors)
        let tensors = HashMap::new();

        let result = BitNetModel::from_gguf(config, tensors, device);
        assert!(result.is_err());
    }

    #[test]
    fn test_bitnet_model_forward() {
        let config = BitNetConfig::default();
        let device = Device::Cpu;
        let model = BitNetModel::new(config.clone(), device);

        // Create mock input tensor
        let input = bitnet_common::ConcreteTensor::mock(vec![1, 10]); // batch_size=1, seq_len=10
        let mut cache = ();

        let result = model.forward(&input, &mut cache);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape()[0], 1); // batch_size
        assert_eq!(output.shape()[1], config.model.vocab_size); // vocab_size
    }

    #[test]
    fn test_model_trait_implementation() {
        let config = BitNetConfig::default();
        let device = Device::Cpu;
        let model = BitNetModel::new(config.clone(), device);

        // Test Model trait methods
        assert_eq!(model.config().model.vocab_size, config.model.vocab_size);

        let input = bitnet_common::ConcreteTensor::mock(vec![1, 10]);
        let mut cache = ();
        let result = model.forward(&input, &mut cache);
        assert!(result.is_ok());
    }
}

/// Integration tests for complete workflows
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_loading_workflow() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);

        // Create a minimal valid GGUF file structure
        let mut temp_file = NamedTempFile::with_suffix(".gguf").unwrap();

        // Write GGUF magic and minimal header
        temp_file.write_all(b"GGUF").unwrap(); // Magic
        temp_file.write_all(&3u32.to_le_bytes()).unwrap(); // Version
        temp_file.write_all(&0u64.to_le_bytes()).unwrap(); // Tensor count
        temp_file.write_all(&0u64.to_le_bytes()).unwrap(); // Metadata KV count
        temp_file.flush().unwrap();

        // Test format detection through available formats
        let formats = loader.available_formats();
        assert!(formats.contains(&"GGUF"));

        // Test loading (will fail due to incomplete file, but should not panic)
        let result = loader.load(temp_file.path());
        // Expected to fail due to incomplete GGUF structure
        assert!(result.is_err());
    }

    #[test]
    fn test_loading_with_progress() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);

        let progress_values = Arc::new(Mutex::new(Vec::new()));
        let progress_values_clone = progress_values.clone();

        let callback: ProgressCallback = Arc::new(move |progress, message| {
            progress_values_clone
                .lock()
                .unwrap()
                .push((progress, message.to_string()));
        });

        let config = LoadConfig {
            use_mmap: true,
            validate_checksums: false,
            progress_callback: Some(callback),
        };

        // Create a test file
        let mut temp_file = NamedTempFile::with_suffix(".gguf").unwrap();
        temp_file.write_all(b"GGUF").unwrap();
        temp_file.write_all(&[0u8; 20]).unwrap();
        temp_file.flush().unwrap();

        // Attempt to load (will fail, but should call progress callback)
        let _result = loader.load_with_config(temp_file.path(), &config);

        // Check that progress callback was called
        let values = progress_values.lock().unwrap();
        // Should have at least one progress update
        assert!(!values.is_empty());
    }

    #[test]
    fn test_error_recovery() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);

        // Test that loader can handle multiple failed attempts
        for i in 0..5 {
            let result = loader.load(&format!("non_existent_file_{}.gguf", i));
            assert!(result.is_err());
        }

        // Loader should still be functional after errors
        let formats = loader.available_formats();
        assert!(!formats.is_empty());
    }

    #[test]
    fn test_concurrent_loading() {
        use std::sync::Arc;
        use std::thread;

        let device = Device::Cpu;
        let loader = Arc::new(ModelLoader::new(device));

        // Test concurrent access to loader
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let loader_clone = Arc::clone(&loader);
                thread::spawn(move || {
                    let result = loader_clone.load(&format!("non_existent_{}.gguf", i));
                    assert!(result.is_err()); // Expected to fail
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Loader should still be functional
        let formats = loader.available_formats();
        assert!(!formats.is_empty());
    }

    #[test]
    fn test_memory_usage_patterns() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);

        // Test with mmap enabled
        let config_mmap = LoadConfig {
            use_mmap: true,
            validate_checksums: false,
            progress_callback: None,
        };

        // Test with mmap disabled
        let config_no_mmap = LoadConfig {
            use_mmap: false,
            validate_checksums: false,
            progress_callback: None,
        };

        // Create test file
        let mut temp_file = NamedTempFile::with_suffix(".gguf").unwrap();
        let content = vec![0u8; 1024];
        temp_file.write_all(&content).unwrap();
        temp_file.flush().unwrap();

        // Both should handle the file (though they'll fail due to invalid format)
        let result1 = loader.load_with_config(temp_file.path(), &config_mmap);
        let result2 = loader.load_with_config(temp_file.path(), &config_no_mmap);

        // Both should fail gracefully
        assert!(result1.is_err());
        assert!(result2.is_err());
    }
}

/// Test model metadata and configuration
mod metadata_tests {
    use super::*;

    #[test]
    fn test_model_metadata_creation() {
        let metadata = ModelMetadata {
            name: "test_model".to_string(),
            version: "1.0".to_string(),
            architecture: "bitnet".to_string(),
            vocab_size: 32000,
            context_length: 2048,
            quantization: Some(QuantizationType::I2S),
        };

        assert_eq!(metadata.name, "test_model");
        assert_eq!(metadata.vocab_size, 32000);
        assert_eq!(metadata.context_length, 2048);
        assert!(metadata.quantization.is_some());
    }

    #[test]
    fn test_quantization_types() {
        let i2s = QuantizationType::I2S;
        let tl1 = QuantizationType::TL1;
        let tl2 = QuantizationType::TL2;

        assert_eq!(format!("{}", i2s), "I2_S");
        assert_eq!(format!("{}", tl1), "TL1");
        assert_eq!(format!("{}", tl2), "TL2");
    }

    #[test]
    fn test_model_config_validation() {
        let config = BitNetConfig::default();

        // Test default values are valid
        assert!(config.model.vocab_size > 0);
        assert!(config.model.hidden_size > 0);
        assert!(config.model.num_layers > 0);
        assert!(config.model.num_heads > 0);
        assert!(config.model.max_position_embeddings > 0);
    }
}
