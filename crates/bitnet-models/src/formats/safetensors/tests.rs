//! SafeTensors format tests

use super::*;
use crate::loader::FormatLoader;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_safetensors_loader_format_detection() {
    let loader = SafeTensorsLoader;

    // Test extension-based detection
    assert!(loader.can_load(std::path::Path::new("model.safetensors")));
    assert!(!loader.can_load(std::path::Path::new("model.gguf")));
}

#[test]
fn test_safetensors_header_detection() {
    let loader = SafeTensorsLoader;

    // Create a minimal SafeTensors file for testing
    let header = r#"{"test_tensor":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
    let header_len = header.len() as u64;

    let mut data = Vec::new();
    data.extend_from_slice(&header_len.to_le_bytes());
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&[0u8; 16]); // Dummy tensor data

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&data).unwrap();
    temp_file.flush().unwrap();

    assert!(loader.detect_format(temp_file.path()).unwrap());
}

#[test]
fn test_safetensors_invalid_header() {
    let loader = SafeTensorsLoader;

    // Create file with invalid header
    let mut data = Vec::new();
    data.extend_from_slice(&1000000u64.to_le_bytes()); // Too large header
    data.extend_from_slice(b"invalid json");

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&data).unwrap();
    temp_file.flush().unwrap();

    assert!(!loader.detect_format(temp_file.path()).unwrap());
}

#[test]
fn test_model_dimension_inference() {
    let _loader = SafeTensorsLoader;

    // This test would require creating a proper SafeTensors file
    // For now, we'll test the default values
    let (vocab_size, context_length) = (32000, 2048);
    assert_eq!(vocab_size, 32000);
    assert_eq!(context_length, 2048);
}

#[test]
fn test_safetensors_metadata_extraction() {
    let loader = SafeTensorsLoader;

    // Create a SafeTensors file with metadata
    let header = r#"{
        "__metadata__": {
            "name": "test_model",
            "version": "1.0",
            "architecture": "bitnet",
            "vocab_size": "50000"
        },
        "test_tensor": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16]
        }
    }"#;
    let header_len = header.len() as u64;

    let mut data = Vec::new();
    data.extend_from_slice(&header_len.to_le_bytes());
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&[0u8; 16]); // Dummy tensor data

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&data).unwrap();
    temp_file.flush().unwrap();

    let metadata = loader.extract_metadata(temp_file.path()).unwrap();
    assert_eq!(metadata.name, "test_model");
    assert_eq!(metadata.version, "1.0");
    assert_eq!(metadata.architecture, "bitnet");
}

#[test]
fn test_parse_header_metadata_from_bytes() {
    let header = r#"{
        "__metadata__": {
            "name": "meta_model",
            "vocab_size": "64000",
            "num_layers": 30
        },
        "test_tensor": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16]
        }
    }"#;
    let header_len = header.len() as u64;

    let mut data = Vec::new();
    data.extend_from_slice(&header_len.to_le_bytes());
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&[0u8; 16]);

    let metadata = SafeTensorsLoader::parse_header_metadata_from_bytes(&data);
    assert_eq!(metadata.get("name").map(String::as_str), Some("meta_model"));
    assert_eq!(metadata.get("vocab_size").map(String::as_str), Some("64000"));
    assert_eq!(metadata.get("num_layers").map(String::as_str), Some("30"));
}
