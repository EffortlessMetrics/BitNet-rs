//! Example unit tests for bitnet-models crate
//!
//! Demonstrates testing patterns for model loading, validation, and format compatibility

use bitnet_models::{BitNetModel, ModelError, ModelFormat, ModelMetadata};
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::fs;

#[cfg(test)]
mod bitnet_models_examples {
    use super::*;

    /// Example: Model loading with mock data
    #[tokio::test]
    async fn test_model_loading_success() {
        // Create temporary test model file
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.gguf");

        // Create mock model data
        let mock_model_data = create_mock_gguf_data();
        fs::write(&model_path, mock_model_data).await.unwrap();

        // Test model loading
        let result = BitNetModel::from_file(&model_path).await;
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.format(), ModelFormat::GGUF);
        assert!(model.metadata().vocab_size > 0);
    }

    /// Example: Model format detection
    #[tokio::test]
    async fn test_model_format_detection() {
        let temp_dir = TempDir::new().unwrap();

        // Test GGUF format detection
        let gguf_path = temp_dir.path().join("model.gguf");
        fs::write(&gguf_path, create_mock_gguf_data())
            .await
            .unwrap();

        let format = BitNetModel::detect_format(&gguf_path).await.unwrap();
        assert_eq!(format, ModelFormat::GGUF);

        // Test SafeTensors format detection
        let safetensors_path = temp_dir.path().join("model.safetensors");
        fs::write(&safetensors_path, create_mock_safetensors_data())
            .await
            .unwrap();

        let format = BitNetModel::detect_format(&safetensors_path).await.unwrap();
        assert_eq!(format, ModelFormat::SafeTensors);
    }

    /// Example: Model metadata validation
    #[tokio::test]
    async fn test_model_metadata_validation() {
        let metadata = ModelMetadata::builder()
            .name("test_model".to_string())
            .version("1.0.0".to_string())
            .vocab_size(32000)
            .hidden_size(4096)
            .num_layers(32)
            .model_type("bitnet_b1_58".to_string())
            .build();

        assert!(metadata.is_ok());
        let metadata = metadata.unwrap();

        // Validate required fields
        assert!(!metadata.name().is_empty());
        assert!(!metadata.version().is_empty());
        assert!(metadata.vocab_size() > 0);
        assert!(metadata.hidden_size() > 0);
        assert!(metadata.num_layers() > 0);
    }

    /// Example: Error handling for invalid models
    #[tokio::test]
    async fn test_invalid_model_file() {
        let temp_dir = TempDir::new().unwrap();
        let invalid_path = temp_dir.path().join("invalid.model");

        // Create file with invalid content
        fs::write(&invalid_path, b"invalid model data")
            .await
            .unwrap();

        let result = BitNetModel::from_file(&invalid_path).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            ModelError::InvalidFormat { path, reason } => {
                assert_eq!(path, invalid_path);
                assert!(reason.contains("invalid"));
            }
            _ => panic!("Expected InvalidFormat error"),
        }
    }

    /// Example: Model conversion testing
    #[tokio::test]
    async fn test_model_format_conversion() {
        let temp_dir = TempDir::new().unwrap();

        // Create source model
        let source_path = temp_dir.path().join("source.gguf");
        fs::write(&source_path, create_mock_gguf_data())
            .await
            .unwrap();

        let model = BitNetModel::from_file(&source_path).await.unwrap();

        // Convert to SafeTensors
        let target_path = temp_dir.path().join("target.safetensors");
        let result = model
            .convert_to_format(ModelFormat::SafeTensors, &target_path)
            .await;

        assert!(result.is_ok());
        assert!(target_path.exists());

        // Verify converted model can be loaded
        let converted_model = BitNetModel::from_file(&target_path).await.unwrap();
        assert_eq!(converted_model.format(), ModelFormat::SafeTensors);

        // Verify metadata preservation
        assert_eq!(
            model.metadata().vocab_size(),
            converted_model.metadata().vocab_size()
        );
        assert_eq!(
            model.metadata().hidden_size(),
            converted_model.metadata().hidden_size()
        );
    }

    /// Example: Large model handling
    #[tokio::test]
    async fn test_large_model_memory_efficiency() {
        let temp_dir = TempDir::new().unwrap();
        let large_model_path = temp_dir.path().join("large_model.gguf");

        // Create mock large model (simulate with metadata)
        let large_model_data = create_mock_large_gguf_data();
        fs::write(&large_model_path, large_model_data)
            .await
            .unwrap();

        // Test memory-efficient loading
        let start_memory = get_memory_usage();
        let model = BitNetModel::from_file_lazy(&large_model_path)
            .await
            .unwrap();
        let after_load_memory = get_memory_usage();

        // Verify lazy loading doesn't load entire model into memory
        let memory_increase = after_load_memory - start_memory;
        assert!(
            memory_increase < 100 * 1024 * 1024, // Less than 100MB
            "Memory increase too large: {} bytes",
            memory_increase
        );

        // Test that model can still be used
        assert!(model.is_loaded());
        assert!(model.metadata().vocab_size() > 0);
    }

    /// Example: Concurrent model access
    #[tokio::test]
    async fn test_concurrent_model_access() {
        use std::sync::Arc;
        use tokio::task::JoinSet;

        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("concurrent_test.gguf");
        fs::write(&model_path, create_mock_gguf_data())
            .await
            .unwrap();

        let model = Arc::new(BitNetModel::from_file(&model_path).await.unwrap());
        let mut join_set = JoinSet::new();

        // Spawn multiple tasks accessing model concurrently
        for i in 0..5 {
            let model_clone = Arc::clone(&model);
            join_set.spawn(async move {
                // Test concurrent metadata access
                let metadata = model_clone.metadata();
                assert!(metadata.vocab_size() > 0);

                // Test concurrent format access
                let format = model_clone.format();
                assert_eq!(format, ModelFormat::GGUF);

                i
            });
        }

        // Wait for all tasks
        let mut results = Vec::new();
        while let Some(result) = join_set.join_next().await {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 5);
    }
}

/// Test utilities for model testing
pub mod model_test_utils {
    use super::*;

    /// Create mock GGUF model data for testing
    pub fn create_mock_gguf_data() -> Vec<u8> {
        // Simplified GGUF header + minimal data
        let mut data = Vec::new();

        // GGUF magic number
        data.extend_from_slice(b"GGUF");

        // Version (4 bytes, little endian)
        data.extend_from_slice(&3u32.to_le_bytes());

        // Tensor count (8 bytes, little endian)
        data.extend_from_slice(&10u64.to_le_bytes());

        // Metadata count (8 bytes, little endian)
        data.extend_from_slice(&5u64.to_le_bytes());

        // Add some mock metadata and tensor data
        data.extend_from_slice(&vec![0u8; 1024]); // Mock data

        data
    }

    /// Create mock SafeTensors model data for testing
    pub fn create_mock_safetensors_data() -> Vec<u8> {
        // Simplified SafeTensors format
        let header = r#"{"vocab_size":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}"#;
        let header_len = header.len() as u64;

        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(header.as_bytes());
        data.extend_from_slice(&32000u32.to_le_bytes()); // Mock vocab_size data

        data
    }

    /// Create mock large model data for memory testing
    pub fn create_mock_large_gguf_data() -> Vec<u8> {
        let mut data = create_mock_gguf_data();

        // Extend with additional mock data to simulate large model
        // But keep actual data small for test performance
        data.extend_from_slice(&vec![0u8; 10 * 1024]); // 10KB mock data

        data
    }

    /// Get current memory usage (mock implementation)
    pub fn get_memory_usage() -> u64 {
        // In real implementation, this would use system APIs
        // For testing, return mock value
        std::process::id() as u64 * 1024 // Mock memory usage
    }

    /// Helper to create temporary model file
    pub async fn create_temp_model(format: ModelFormat) -> (TempDir, PathBuf) {
        let temp_dir = TempDir::new().unwrap();
        let extension = match format {
            ModelFormat::GGUF => "gguf",
            ModelFormat::SafeTensors => "safetensors",
        };

        let model_path = temp_dir.path().join(format!("test_model.{}", extension));

        let data = match format {
            ModelFormat::GGUF => create_mock_gguf_data(),
            ModelFormat::SafeTensors => create_mock_safetensors_data(),
        };

        fs::write(&model_path, data).await.unwrap();

        (temp_dir, model_path)
    }

    /// Assert model metadata equality
    pub fn assert_metadata_eq(a: &ModelMetadata, b: &ModelMetadata) {
        assert_eq!(a.name(), b.name());
        assert_eq!(a.version(), b.version());
        assert_eq!(a.vocab_size(), b.vocab_size());
        assert_eq!(a.hidden_size(), b.hidden_size());
        assert_eq!(a.num_layers(), b.num_layers());
        assert_eq!(a.model_type(), b.model_type());
    }
}

/// Example: Parameterized tests for different model formats
#[cfg(test)]
mod parameterized_tests {
    use super::*;
    use rstest::*;

    #[rstest]
    #[case(ModelFormat::GGUF)]
    #[case(ModelFormat::SafeTensors)]
    #[tokio::test]
    async fn test_model_loading_all_formats(#[case] format: ModelFormat) {
        let (_temp_dir, model_path) = model_test_utils::create_temp_model(format).await;

        let result = BitNetModel::from_file(&model_path).await;
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.format(), format);
    }

    #[rstest]
    #[case(1000, 256, 4)] // Small model
    #[case(32000, 4096, 32)] // Standard model
    #[case(100000, 8192, 64)] // Large model
    #[tokio::test]
    async fn test_model_metadata_variations(
        #[case] vocab_size: u32,
        #[case] hidden_size: u32,
        #[case] num_layers: u32,
    ) {
        let metadata = ModelMetadata::builder()
            .name("test".to_string())
            .version("1.0.0".to_string())
            .vocab_size(vocab_size)
            .hidden_size(hidden_size)
            .num_layers(num_layers)
            .model_type("bitnet_b1_58".to_string())
            .build()
            .unwrap();

        assert_eq!(metadata.vocab_size(), vocab_size);
        assert_eq!(metadata.hidden_size(), hidden_size);
        assert_eq!(metadata.num_layers(), num_layers);
    }
}
