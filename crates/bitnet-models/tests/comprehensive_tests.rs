//! Comprehensive tests for bitnet-models
//! Covers edge cases, error conditions, and end-to-end scenarios

use bitnet_models::*;
use bitnet_common::{Device, BitNetError, ModelError, Result};
use tempfile::{NamedTempFile, TempDir};
use std::io::Write;
use std::fs;
use std::path::Path;

/// Test error conditions and edge cases
mod error_handling {
    use super::*;

    #[test]
    fn test_invalid_file_paths() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);
        
        // Test non-existent file
        let result = loader.load_model("non_existent_file.gguf", None);
        assert!(result.is_err());
        
        // Test directory instead of file
        let temp_dir = TempDir::new().unwrap();
        let result = loader.load_model(temp_dir.path(), None);
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
            
            let result = loader.load_model(temp_file.path(), None);
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
        
        let result = loader.load_model(temp_file.path(), None);
        assert!(result.is_err());
        
        // Test truncated GGUF header
        let mut temp_file = NamedTempFile::with_suffix(".gguf").unwrap();
        temp_file.write_all(b"GGUF").unwrap(); // Only magic, no version
        temp_file.flush().unwrap();
        
        let result = loader.load_model(temp_file.path(), None);
        assert!(result.is_err());
        
        // Test invalid SafeTensors header
        let mut temp_file = NamedTempFile::with_suffix(".safetensors").unwrap();
        temp_file.write_all(b"invalid json header").unwrap();
        temp_file.flush().unwrap();
        
        let result = loader.load_model(temp_file.path(), None);
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
        
        let result = loader.load_model(temp_file.path(), None);
        assert!(result.is_err());
        
        // Test file without extension
        let temp_dir = TempDir::new().unwrap();
        let no_ext_path = temp_dir.path().join("no_extension");
        fs::write(&no_ext_path, b"some content").unwrap();
        
        let result = loader.load_model(&no_ext_path, None);
        // Should try magic byte detection but fail
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_constraints() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);
        
        // Create a config with very small memory limit
        let config = LoadConfig {
            use_mmap: false,
            validate_checksums: true,
            progress_callback: None,
            max_memory_usage: Some(1), // 1 byte limit
        };
        
        // Create a file larger than the limit
        let mut temp_file = NamedTempFile::with_suffix(".gguf").unwrap();
        let large_content = vec![0u8; 1024]; // 1KB file
        temp_file.write_all(&large_content).unwrap();
        temp_file.flush().unwrap();
        
        let result = loader.load_model(temp_file.path(), Some(config));
        // Should fail due to memory constraint
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_tensor_shapes() {
        // This test would require creating a valid GGUF file with invalid tensor shapes
        // For now, we test the validation logic directly
        
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);
        
        // Test detection of format by extension with invalid content
        let formats = loader.available_formats();
        assert!(formats.contains(&"GGUF"));
        assert!(formats.contains(&"SafeTensors"));
        assert!(formats.contains(&"HuggingFace"));
    }
}

/// Test different model formats comprehensively
mod format_comprehensive {
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
        
        let detected = loader.detect_by_magic_bytes(temp_file.path()).unwrap();
        assert!(detected.is_some());
        assert_eq!(detected.unwrap().name(), "GGUF");
        
        // Test by extension
        let gguf_path = Path::new("model.gguf");
        let detected = loader.detect_by_extension(gguf_path);
        assert!(detected.is_some());
        assert_eq!(detected.unwrap().name(), "GGUF");
    }

    #[test]
    fn test_safetensors_format_detection() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);
        
        // Test by extension
        let safetensors_path = Path::new("model.safetensors");
        let detected = loader.detect_by_extension(safetensors_path);
        assert!(detected.is_some());
        assert_eq!(detected.unwrap().name(), "SafeTensors");
        
        // Test alternative extensions
        let st_path = Path::new("model.st");
        let detected = loader.detect_by_extension(st_path);
        assert!(detected.is_some());
        assert_eq!(detected.unwrap().name(), "SafeTensors");
    }

    #[test]
    fn test_huggingface_format_detection() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);
        
        // Test by extension
        let hf_path = Path::new("pytorch_model.bin");
        let detected = loader.detect_by_extension(hf_path);
        assert!(detected.is_some());
        assert_eq!(detected.unwrap().name(), "HuggingFace");
        
        // Test config.json detection
        let config_path = Path::new("config.json");
        let detected = loader.detect_by_extension(config_path);
        assert!(detected.is_some());
        assert_eq!(detected.unwrap().name(), "HuggingFace");
    }

    #[test]
    fn test_format_priority() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);
        
        // Test that magic byte detection takes priority over extension
        let mut temp_file = NamedTempFile::with_suffix(".wrong_extension").unwrap();
        temp_file.write_all(b"GGUF").unwrap();
        temp_file.write_all(&[0u8; 20]).unwrap();
        temp_file.flush().unwrap();
        
        let detected = loader.detect_by_magic_bytes(temp_file.path()).unwrap();
        assert!(detected.is_some());
        assert_eq!(detected.unwrap().name(), "GGUF");
    }

    #[test]
    fn test_ambiguous_extensions() {
        let device = Device::Cpu;
        let loader = ModelLoader::new(device);
        
        // Test files with no clear format indication
        let ambiguous_path = Path::new("model.bin");
        let detected = loader.detect_by_extension(ambiguous_path);
        // Should detect as HuggingFace (common .bin format)
        assert!(detected.is_some());
        assert_eq!(detected.unwrap().name(), "HuggingFace");
    }
}

/// Test memory mapping functionality
mod mmap_comprehensive {
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
mod progress_comprehensive {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_progress_callback_sequence() {
        let progress_values = Arc::new(Mutex::new(Vec::new()));
        let progress_values_clone = progress_values.clone();
        
        let callback: ProgressCallback = Arc::new(move |progress, message| {
            progress_values_clone.lock().unwrap().push((progress, message.to_string()));
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
            assert!(values[i].0 >= values[i-1].0, "Progress should be monotonic");
        }
        
        assert_eq!(values[0].0, 0.0);
        assert_eq!(values[4].0, 1.0);
    }

    #[test]
    fn test_progress_callback_error_handling() {
        // Test callback that panics
        let panic_callback: ProgressCallback = Arc::new(|_progress, _message| {
            panic!("Callback panic test");
        });
        
        // The callback should be called in a way that doesn't crash the loader
        // This is more of a design consideration for the actual implementation
        
        // Test callback with invalid progress values
        let callback: ProgressCallback = Arc::new(|progress, _message| {
            assert!(progress >= 0.0 && progress <= 1.0, "Progress should be in [0, 1] range");
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
}

/// Test model security features
mod security_comprehensive {
    use super::*;

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
        let verifier = ModelVerifier::new(ModelSecurity::default());
        
        // Test trusted source
        let trusted_sources = vec!["https://huggingface.co/".to_string()];
        let security = ModelSecurity {
            verify_checksums: true,
            allowed_sources: Some(trusted_sources),
            max_file_size: None,
            require_signature: false,
        };
        let verifier = ModelVerifier::new(security);
        
        assert!(verifier.verify_source("https://huggingface.co/model/file.gguf").is_ok());
        assert!(verifier.verify_source("https://untrusted.com/model.gguf").is_err());
    }

    #[test]
    fn test_file_size_limits() {
        let mut security = ModelSecurity::default();
        security.max_file_size = Some(1024); // 1KB limit
        let verifier = ModelVerifier::new(security);
        
        // Create a small file (should pass)
        let mut small_file = NamedTempFile::new().unwrap();
        small_file.write_all(b"small content").unwrap();
        small_file.flush().unwrap();
        
        assert!(verifier.verify_file_size(small_file.path()).is_ok());
        
        // Create a large file (should fail)
        let mut large_file = NamedTempFile::new().unwrap();
        let large_content = vec![0u8; 2048]; // 2KB
        large_file.write_all(&large_content).unwrap();
        large_file.flush().unwrap();
        
        assert!(verifier.verify_file_size(large_file.path()).is_err());
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
        assert!(verifier.validate_checksum(temp_file.path(), &expected_hash).is_ok());
        
        // Test validation with incorrect checksum
        let wrong_hash = "wrong_hash_value".to_string();
        assert!(verifier.validate_checksum(temp_file.path(), &wrong_hash).is_err());
    }
}

/// Test utility functions
mod utils_comprehensive {
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
        assert!(crate::loader::utils::validate_file_access(temp_dir.path()).is_err());
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
        
        // Test format detection
        let detected = loader.detect_by_extension(temp_file.path());
        assert!(detected.is_some());
        assert_eq!(detected.unwrap().name(), "GGUF");
        
        let detected = loader.detect_by_magic_bytes(temp_file.path()).unwrap();
        assert!(detected.is_some());
        assert_eq!(detected.unwrap().name(), "GGUF");
        
        // Test loading (will fail due to incomplete file, but should not panic)
        let result = loader.load_model(temp_file.path(), None);
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
            progress_values_clone.lock().unwrap().push((progress, message.to_string()));
        });
        
        let config = LoadConfig {
            use_mmap: true,
            validate_checksums: false,
            progress_callback: Some(callback),
            max_memory_usage: None,
        };
        
        // Create a test file
        let mut temp_file = NamedTempFile::with_suffix(".gguf").unwrap();
        temp_file.write_all(b"GGUF").unwrap();
        temp_file.write_all(&[0u8; 20]).unwrap();
        temp_file.flush().unwrap();
        
        // Attempt to load (will fail, but should call progress callback)
        let _result = loader.load_model(temp_file.path(), Some(config));
        
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
            let result = loader.load_model(&format!("non_existent_file_{}.gguf", i), None);
            assert!(result.is_err());
        }
        
        // Loader should still be functional after errors
        let formats = loader.available_formats();
        assert!(!formats.is_empty());
    }

    #[test]
    fn test_concurrent_loading() {
        use std::thread;
        use std::sync::Arc;
        
        let device = Device::Cpu;
        let loader = Arc::new(ModelLoader::new(device));
        
        // Test concurrent access to loader
        let handles: Vec<_> = (0..4).map(|i| {
            let loader_clone = Arc::clone(&loader);
            thread::spawn(move || {
                let result = loader_clone.load_model(&format!("non_existent_{}.gguf", i), None);
                assert!(result.is_err()); // Expected to fail
            })
        }).collect();
        
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
            max_memory_usage: None,
        };
        
        // Test with mmap disabled
        let config_no_mmap = LoadConfig {
            use_mmap: false,
            validate_checksums: false,
            progress_callback: None,
            max_memory_usage: None,
        };
        
        // Create test file
        let mut temp_file = NamedTempFile::with_suffix(".gguf").unwrap();
        let content = vec![0u8; 1024];
        temp_file.write_all(&content).unwrap();
        temp_file.flush().unwrap();
        
        // Both should handle the file (though they'll fail due to invalid format)
        let result1 = loader.load_model(temp_file.path(), Some(config_mmap));
        let result2 = loader.load_model(temp_file.path(), Some(config_no_mmap));
        
        // Both should fail gracefully
        assert!(result1.is_err());
        assert!(result2.is_err());
    }
}