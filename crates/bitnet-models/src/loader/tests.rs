//! Model loader tests

use super::*;
use bitnet_common::Device;
use std::io::Write;
use tempfile::NamedTempFile;

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
fn test_format_detection_by_extension() {
    let device = Device::Cpu;
    let loader = ModelLoader::new(device);

    // Test GGUF detection
    let gguf_path = std::path::Path::new("model.gguf");
    if let Some(format_loader) = loader.detect_by_extension(gguf_path) {
        assert_eq!(format_loader.name(), "GGUF");
    }

    // Test SafeTensors detection
    let safetensors_path = std::path::Path::new("model.safetensors");
    if let Some(format_loader) = loader.detect_by_extension(safetensors_path) {
        assert_eq!(format_loader.name(), "SafeTensors");
    }
}

#[test]
fn test_magic_bytes_detection() {
    let device = Device::Cpu;
    let loader = ModelLoader::new(device);

    // Create a temporary file with GGUF magic bytes
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(b"GGUF").unwrap();
    temp_file.write_all(&[0u8; 20]).unwrap(); // Padding
    temp_file.flush().unwrap();

    let result = loader.detect_by_magic_bytes(temp_file.path()).unwrap();
    assert!(result.is_some());
    assert_eq!(result.unwrap().name(), "GGUF");
}

#[test]
fn test_progress_callback() {
    use std::sync::{Arc, Mutex};

    let progress_values = Arc::new(Mutex::new(Vec::new()));
    let progress_values_clone = progress_values.clone();

    let callback: ProgressCallback = Arc::new(move |progress, message| {
        progress_values_clone.lock().unwrap().push((progress, message.to_string()));
    });

    // Simulate progress updates
    callback(0.0, "Starting");
    callback(0.5, "Halfway");
    callback(1.0, "Complete");

    let values = progress_values.lock().unwrap();
    assert_eq!(values.len(), 3);
    assert_eq!(values[0].0, 0.0);
    assert_eq!(values[1].0, 0.5);
    assert_eq!(values[2].0, 1.0);
}

#[test]
fn test_load_config_defaults() {
    let config = LoadConfig::default();
    assert!(config.use_mmap);
    assert!(config.validate_checksums);
    assert!(config.progress_callback.is_none());
}

#[test]
fn test_mmap_file() {
    let mut temp_file = NamedTempFile::new().unwrap();
    let test_data = b"Hello, World!";
    temp_file.write_all(test_data).unwrap();
    temp_file.flush().unwrap();

    let mmap_file = MmapFile::open(temp_file.path()).unwrap();
    assert_eq!(mmap_file.len(), test_data.len());
    assert_eq!(mmap_file.as_slice(), test_data);
    assert!(!mmap_file.is_empty());
}

#[test]
fn test_utils_validate_file_access() {
    // Test with existing file
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(b"test").unwrap();
    temp_file.flush().unwrap();

    assert!(crate::loader::utils::validate_file_access(temp_file.path()).is_ok());

    // Test with non-existent file
    let non_existent = std::path::Path::new("non_existent_file.txt");
    assert!(crate::loader::utils::validate_file_access(non_existent).is_err());
}

#[test]
fn test_utils_get_file_size() {
    let mut temp_file = NamedTempFile::new().unwrap();
    let test_data = b"Hello, World!";
    temp_file.write_all(test_data).unwrap();
    temp_file.flush().unwrap();

    let size = crate::loader::utils::get_file_size(temp_file.path()).unwrap();
    assert_eq!(size, test_data.len() as u64);
}

#[test]
fn test_utils_progress_callbacks() {
    let logging_callback = crate::loader::utils::create_logging_progress_callback();
    logging_callback(0.5, "Test message");

    let stdout_callback = crate::loader::utils::create_stdout_progress_callback();
    stdout_callback(0.5, "Test message");

    // These should not panic
}

#[test]
fn test_architecture_support() {
    let loader = ModelLoader::new(Device::Cpu);

    // All architectures shared with production loader
    for arch in ["bitnet", "bitnet-b1.58", "llama", "mistral", "qwen", "gpt", "bert", "phi"] {
        assert!(
            loader.is_supported_architecture(arch),
            "{arch} should be supported"
        );
    }

    // Case-insensitive
    assert!(loader.is_supported_architecture("BitNet-B1.58"));
    assert!(loader.is_supported_architecture("GPT"));
    assert!(loader.is_supported_architecture("BERT"));

    // Unknown rejected
    assert!(!loader.is_supported_architecture("unknown"));
}
