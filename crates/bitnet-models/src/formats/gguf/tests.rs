//! GGUF format tests

use super::*;
use crate::loader::FormatLoader;
use std::io::Write;
use tempfile::NamedTempFile;

/// Helper to build valid GGUF v3 bytes for testing
fn build_gguf_bytes(metadata: Vec<(&str, GgufValue)>) -> Vec<u8> {
    let mut data = Vec::<u8>::new();
    const V3: u32 = 3;
    const ALIGN: usize = 32;

    // --- Header (v3) ---
    // Note: The reader doesn't expect alignment/data_offset fields in the header
    // It only reads: magic, version, tensor_count, metadata_kv_count
    data.extend_from_slice(b"GGUF");               // magic
    data.extend_from_slice(&V3.to_le_bytes());     // version
    let n_tensors = 0u64;
    let n_kv = metadata.len() as u64;
    data.extend_from_slice(&n_tensors.to_le_bytes()); // n_tensors
    data.extend_from_slice(&n_kv.to_le_bytes());      // n_kv
    // No alignment or data_offset fields here - reader doesn't expect them

    // --- KV section ---
    for (key, value) in metadata {
        let kb = key.as_bytes();
        data.extend_from_slice(&(kb.len() as u64).to_le_bytes());
        data.extend_from_slice(kb);
        write_gguf_value(&mut data, value);
    }

    // --- Align to 32 bytes for data section (even though we have no tensors) ---
    // Use safe padding calculation
    let pad = (ALIGN - (data.len() % ALIGN)) % ALIGN;
    data.resize(data.len() + pad, 0);

    // no tensors (n_tensors = 0)
    data
}

/// Helper to write a GGUF value to a byte vector
fn write_gguf_value(data: &mut Vec<u8>, value: GgufValue) {
    match value {
        GgufValue::U8(v) => {
            data.extend_from_slice(&0u32.to_le_bytes()); // Type 0
            data.push(v);
        }
        GgufValue::I8(v) => {
            data.extend_from_slice(&1u32.to_le_bytes()); // Type 1
            data.push(v as u8);
        }
        GgufValue::U16(v) => {
            data.extend_from_slice(&2u32.to_le_bytes()); // Type 2
            data.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::I16(v) => {
            data.extend_from_slice(&3u32.to_le_bytes()); // Type 3
            data.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::U32(v) => {
            data.extend_from_slice(&4u32.to_le_bytes()); // Type 4
            data.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::I32(v) => {
            data.extend_from_slice(&5u32.to_le_bytes()); // Type 5
            data.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::F32(v) => {
            data.extend_from_slice(&6u32.to_le_bytes()); // Type 6
            data.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::Bool(v) => {
            data.extend_from_slice(&7u32.to_le_bytes()); // Type 7
            data.push(if v { 1 } else { 0 });
        }
        GgufValue::String(ref s) => {
            data.extend_from_slice(&8u32.to_le_bytes()); // Type 8
            data.extend_from_slice(&(s.len() as u64).to_le_bytes());
            data.extend_from_slice(s.as_bytes());
        }
        GgufValue::Array(_) => {
            // Not needed for current tests
            panic!("Array values not implemented in test helper");
        }
    }
}

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
fn test_builder_writes_v3_header() {
    let bytes = build_gguf_bytes(vec![("k", GgufValue::U32(1))]);
    // Check header structure
    assert_eq!(&bytes[0..4], b"GGUF");
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 3); // version
    let n_tensors = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let n_kv = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    assert_eq!(n_tensors, 0);
    assert_eq!(n_kv, 1);
    // Check that the data is aligned to 32 bytes
    // The header (24 bytes) + KV data should be padded to next 32-byte boundary
    assert_eq!(bytes.len() % 32, 0, "Total size should be aligned to 32 bytes");
}

#[test]
fn test_gguf_value_types() {
    // Create a valid GGUF file with various value types
    let metadata = vec![
        ("test_u32", GgufValue::U32(42)),
        ("test_i32", GgufValue::I32(-7)),
        ("test_f32", GgufValue::F32(3.14)),
        ("test_bool", GgufValue::Bool(true)),
        ("test_string", GgufValue::String("hello".to_string())),
    ];
    
    let data = build_gguf_bytes(metadata);
    let reader = GgufReader::new(&data).expect("Failed to create reader");
    
    // Test reading values through the reader
    assert_eq!(reader.get_u32_metadata("test_u32"), Some(42));
    assert_eq!(reader.get_i32_metadata("test_i32"), Some(-7));
    assert!((reader.get_f32_metadata("test_f32").unwrap_or(0.0) - 3.14).abs() < 1e-6);
    assert_eq!(reader.get_bool_metadata("test_bool"), Some(true));
    assert_eq!(reader.get_string_metadata("test_string").as_deref(), Some("hello"));
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

    // Create a valid GGUF file with metadata
    let metadata = vec![
        ("general.name", GgufValue::String("test_model".to_string())),
        ("llama.vocab_size", GgufValue::U32(50000)),
        ("general.architecture", GgufValue::String("bitnet".to_string())),
    ];
    
    let data = build_gguf_bytes(metadata);
    
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
    // The implementation uses lossy UTF-8 decoding for GGUF compatibility
    // (GGUF files can contain byte strings that aren't valid UTF-8)
    assert!(result.is_ok());
    let string = result.unwrap();
    // Invalid bytes are replaced with the replacement character
    assert!(string.contains('\u{FFFD}')); // Unicode replacement character
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
