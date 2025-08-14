//! GGUF format tests

use super::*;
use crate::loader::FormatLoader;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_gguf_header_parsing() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF"); // Magic
    data.extend_from_slice(&3u32.to_le_bytes()); // Version
    data.extend_from_slice(&5u64.to_le_bytes()); // Tensor count
    data.extend_from_slice(&2u64.to_le_bytes()); // Metadata count

    let mut offset = 0;
    let header = GgufHeader::read(&data, &mut offset).unwrap();

    assert_eq!(header.magic, *b"GGUF");
    assert_eq!(header.version, 3);
    assert_eq!(header.tensor_count, 5);
    assert_eq!(header.metadata_kv_count, 2);
    assert_eq!(offset, 24);
}

#[test]
fn test_gguf_header_invalid_magic() {
    let mut data = Vec::new();
    data.extend_from_slice(b"XXXX"); // Invalid magic
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&5u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    let mut offset = 0;
    let result = GgufHeader::read(&data, &mut offset);
    assert!(result.is_err());
}

#[test]
fn test_gguf_value_types() {
    // Test U32 value
    let mut data = vec![4u8]; // U32 type
    data.extend_from_slice(&42u32.to_le_bytes());

    let mut offset = 0;
    let value = GgufValue::read(&data, &mut offset).unwrap();
    match value {
        GgufValue::U32(v) => assert_eq!(v, 42),
        _ => panic!("Expected U32 value"),
    }

    // Test String value
    let mut data = vec![8u8]; // String type
    let test_string = "hello";
    data.extend_from_slice(&(test_string.len() as u64).to_le_bytes());
    data.extend_from_slice(test_string.as_bytes());

    let mut offset = 0;
    let value = GgufValue::read(&data, &mut offset).unwrap();
    match value {
        GgufValue::String(s) => assert_eq!(s, "hello"),
        _ => panic!("Expected String value"),
    }

    // Test Bool value
    let data = vec![7u8, 1u8]; // Bool type, true
    let mut offset = 0;
    let value = GgufValue::read(&data, &mut offset).unwrap();
    match value {
        GgufValue::Bool(b) => assert!(b),
        _ => panic!("Expected Bool value"),
    }
}

#[test]
fn test_tensor_type_conversion() {
    assert!(matches!(GgufTensorType::from_u32(0).unwrap(), GgufTensorType::F32));
    assert!(matches!(GgufTensorType::from_u32(1).unwrap(), GgufTensorType::F16));
    assert!(matches!(GgufTensorType::from_u32(2).unwrap(), GgufTensorType::Q4_0));

    assert!(GgufTensorType::from_u32(999).is_err());
}

#[test]
fn test_tensor_type_element_sizes() {
    assert_eq!(GgufTensorType::F32.element_size(), 4);
    assert_eq!(GgufTensorType::F16.element_size(), 2);
    assert_eq!(GgufTensorType::Q4_0.element_size(), 18);
    assert_eq!(GgufTensorType::Q8_0.element_size(), 34);
}

#[test]
fn test_gguf_loader_format_detection() {
    let loader = GgufLoader;

    // Test extension-based detection
    assert!(loader.can_load(std::path::Path::new("model.gguf")));
    assert!(!loader.can_load(std::path::Path::new("model.safetensors")));

    // Test magic bytes detection
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(b"GGUF").unwrap();
    temp_file.write_all(&[0u8; 20]).unwrap(); // Padding
    temp_file.flush().unwrap();

    assert!(loader.detect_format(temp_file.path()).unwrap());
}

#[test]
fn test_gguf_metadata_extraction() {
    let loader = GgufLoader;

    // Create a minimal GGUF file for testing
    let mut data = Vec::new();

    // Header
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes()); // Version
    data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count
    data.extend_from_slice(&3u64.to_le_bytes()); // Metadata count

    // Metadata 1: general.name
    let key = "general.name";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.push(8); // String type
    let value = "test_model";
    data.extend_from_slice(&(value.len() as u64).to_le_bytes());
    data.extend_from_slice(value.as_bytes());

    // Metadata 2: llama.vocab_size
    let key = "llama.vocab_size";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.push(4); // U32 type
    data.extend_from_slice(&50000u32.to_le_bytes());

    // Metadata 3: general.architecture
    let key = "general.architecture";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.push(8); // String type
    let value = "bitnet";
    data.extend_from_slice(&(value.len() as u64).to_le_bytes());
    data.extend_from_slice(value.as_bytes());

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&data).unwrap();
    temp_file.flush().unwrap();

    let metadata = loader.extract_metadata(temp_file.path()).unwrap();
    assert_eq!(metadata.name, "test_model");
    assert_eq!(metadata.vocab_size, 50000);
    assert_eq!(metadata.architecture, "bitnet");
}

#[test]
fn test_binary_reading_functions() {
    let data = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
    let mut offset = 0;

    assert_eq!(read_u8(&data, &mut offset).unwrap(), 0x12);
    assert_eq!(offset, 1);

    offset = 0;
    assert_eq!(read_u16(&data, &mut offset).unwrap(), 0x3412); // Little endian
    assert_eq!(offset, 2);

    offset = 0;
    assert_eq!(read_u32(&data, &mut offset).unwrap(), 0x78563412); // Little endian
    assert_eq!(offset, 4);

    offset = 0;
    assert_eq!(read_u64(&data, &mut offset).unwrap(), 0xF0DEBC9A78563412); // Little endian
    assert_eq!(offset, 8);
}

#[test]
fn test_string_reading() {
    let mut data = Vec::new();
    let test_string = "Hello, GGUF!";
    data.extend_from_slice(&(test_string.len() as u64).to_le_bytes());
    data.extend_from_slice(test_string.as_bytes());

    let mut offset = 0;
    let result = read_string(&data, &mut offset).unwrap();
    assert_eq!(result, test_string);
    assert_eq!(offset, 8 + test_string.len());
}

#[test]
fn test_string_reading_invalid_utf8() {
    let mut data = Vec::new();
    data.extend_from_slice(&3u64.to_le_bytes()); // Length
    data.extend_from_slice(&[0xFF, 0xFE, 0xFD]); // Invalid UTF-8

    let mut offset = 0;
    let result = read_string(&data, &mut offset);
    assert!(result.is_err());
}

#[test]
fn test_insufficient_data_errors() {
    let data = [0x12, 0x34];
    let mut offset = 0;

    // Should succeed
    assert!(read_u16(&data, &mut offset).is_ok());

    // Should fail - not enough data for u32
    assert!(read_u32(&data, &mut offset).is_err());
}
