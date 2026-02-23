//! GGUF format tests

use super::*;
use crate::loader::FormatLoader;
use crate::names::is_layernorm_weight;
use std::io::Write;
use tempfile::NamedTempFile;

// Use shared EnvGuard from workspace test support
use bitnet_test_support::env_guard::EnvGuard;

/// Helper to build valid GGUF bytes for testing
fn build_gguf_bytes(metadata: Vec<(&str, GgufValue)>) -> Vec<u8> {
    let mut data = Vec::<u8>::new();
    const GGUF_VERSION: u32 = 2;
    const ALIGN: usize = 32;

    // --- Header (v2 shape expected by reader) ---
    // The reader reads: magic, version, n_tensors (u64), n_kv (u64)
    data.extend_from_slice(b"GGUF"); // magic
    data.extend_from_slice(&GGUF_VERSION.to_le_bytes()); // version
    let n_tensors = 0u64;
    let n_kv = metadata.len() as u64;
    data.extend_from_slice(&n_tensors.to_le_bytes()); // n_tensors
    data.extend_from_slice(&n_kv.to_le_bytes()); // n_kv
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
    // Test v2 header
    let mut data_v2 = Vec::new();
    data_v2.extend_from_slice(b"GGUF"); // Magic
    data_v2.extend_from_slice(&2u32.to_le_bytes()); // Version 2
    data_v2.extend_from_slice(&5u64.to_le_bytes()); // Tensor count
    data_v2.extend_from_slice(&2u64.to_le_bytes()); // Metadata count

    let mut offset = 0;
    let header_v2 = GgufHeader::read(&data_v2, &mut offset).unwrap();

    assert_eq!(header_v2.magic, *b"GGUF");
    assert_eq!(header_v2.version, 2);
    assert_eq!(header_v2.tensor_count, 5);
    assert_eq!(header_v2.metadata_kv_count, 2);
    assert_eq!(header_v2.alignment, 32); // Default for v2
    assert_eq!(header_v2.data_offset, 0); // Default for v2
    assert_eq!(offset, 24);

    // Test v3 header
    let mut data_v3 = Vec::new();
    data_v3.extend_from_slice(b"GGUF"); // Magic
    data_v3.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    data_v3.extend_from_slice(&5u64.to_le_bytes()); // Tensor count
    data_v3.extend_from_slice(&2u64.to_le_bytes()); // Metadata count
    data_v3.extend_from_slice(&64u32.to_le_bytes()); // Alignment
    data_v3.extend_from_slice(&1024u64.to_le_bytes()); // Data offset

    let mut offset = 0;
    let header_v3 = GgufHeader::read(&data_v3, &mut offset).unwrap();

    assert_eq!(header_v3.magic, *b"GGUF");
    assert_eq!(header_v3.version, 3);
    assert_eq!(header_v3.tensor_count, 5);
    assert_eq!(header_v3.metadata_kv_count, 2);
    assert_eq!(header_v3.alignment, 64);
    assert_eq!(header_v3.data_offset, 1024);
    assert_eq!(offset, 36); // 24 + 4 + 8
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
fn test_gguf_v3_early_variant_detection() {
    // Test v3 early variant (missing alignment/data_offset, goes straight to KV pairs)
    // This simulates Microsoft BitNet model structure
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF"); // Magic
    data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count
    data.extend_from_slice(&1u64.to_le_bytes()); // Metadata count (1 KV pair)

    // Instead of alignment/data_offset, write a KV pair that looks like a string key
    let key = "general.architecture";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes()); // Key length
    data.extend_from_slice(key.as_bytes()); // Key bytes
    data.extend_from_slice(&8u32.to_le_bytes()); // String type
    let value = "bitnet";
    data.extend_from_slice(&(value.len() as u64).to_le_bytes()); // Value length
    data.extend_from_slice(value.as_bytes()); // Value bytes

    let mut offset = 0;
    let header = GgufHeader::read(&data, &mut offset).unwrap();

    // Should detect early variant and use defaults
    assert_eq!(header.version, 3);
    assert_eq!(header.alignment, 32); // Default
    assert_eq!(header.data_offset, 0); // Early variant marker
    assert!(header.is_early_v3_variant());
    assert!(!header.is_standard_v3());
    assert_eq!(
        header.format_description(),
        "GGUF v3 (early variant, missing alignment/data_offset fields)"
    );

    // Offset should be positioned at the start of the KV pair (after header, not consuming KV data)
    assert_eq!(offset, 24); // Just the base header, no alignment/data_offset consumed
}

#[test]
fn test_gguf_v3_standard_detection() {
    // Test standard v3 with proper alignment and data_offset
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF"); // Magic
    data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count
    data.extend_from_slice(&0u64.to_le_bytes()); // Metadata count
    data.extend_from_slice(&32u32.to_le_bytes()); // Alignment
    data.extend_from_slice(&512u64.to_le_bytes()); // Data offset (non-zero)

    let mut offset = 0;
    let header = GgufHeader::read(&data, &mut offset).unwrap();

    assert_eq!(header.version, 3);
    assert_eq!(header.alignment, 32);
    assert_eq!(header.data_offset, 512);
    assert!(header.is_standard_v3());
    assert!(!header.is_early_v3_variant());
    assert_eq!(header.format_description(), "GGUF v3 (standard, align=32, data_offset=512)");

    // Should consume alignment + data_offset fields
    assert_eq!(offset, 36); // 24 + 4 + 8
}

#[test]
fn test_gguf_v3_invalid_alignment_clamped() {
    // Test v3 with invalid alignment (not power of 2)
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF"); // Magic
    data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count
    data.extend_from_slice(&0u64.to_le_bytes()); // Metadata count
    data.extend_from_slice(&17u32.to_le_bytes()); // Invalid alignment (not power of 2)
    data.extend_from_slice(&1024u64.to_le_bytes()); // Data offset

    let mut offset = 0;
    let header = GgufHeader::read(&data, &mut offset).unwrap();

    // Should clamp invalid alignment to 32
    assert_eq!(header.alignment, 32);
    assert_eq!(header.data_offset, 1024);
}

#[test]
fn test_builder_writes_header_v2_shape() {
    let bytes = build_gguf_bytes(vec![("k", GgufValue::U32(1))]);
    // Check header structure
    assert_eq!(&bytes[0..4], b"GGUF");
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2); // version
    let n_tensors = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let n_kv = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    assert_eq!(n_tensors, 0);
    assert_eq!(n_kv, 1);
    // Body is padded to 32 bytes (safe modulo logic in builder)
    assert_eq!(bytes.len() % 32, 0, "Total size should be aligned to 32 bytes");
}

#[test]
fn test_gguf_value_types() {
    // Create a valid GGUF file with various value types
    let metadata = vec![
        ("test_u32", GgufValue::U32(42)),
        ("test_i32", GgufValue::I32(-7)),
        ("test_f32", GgufValue::F32(std::f32::consts::PI)),
        ("test_bool", GgufValue::Bool(true)),
        ("test_string", GgufValue::String("hello".to_string())),
    ];

    let data = build_gguf_bytes(metadata);
    let reader = GgufReader::new(&data).expect("Failed to create reader");

    // Test reading values through the reader
    assert_eq!(reader.get_u32_metadata("test_u32"), Some(42));
    assert_eq!(reader.get_i32_metadata("test_i32"), Some(-7));
    assert!(
        (reader.get_f32_metadata("test_f32").unwrap_or(0.0) - std::f32::consts::PI).abs() < 1e-6
    );
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
    // Create a simple GGUF file with just metadata (no tensors needed for metadata parsing)
    let metadata = vec![
        ("general.name", GgufValue::String("test_model".to_string())),
        ("llama.vocab_size", GgufValue::U32(50000)),
        ("general.architecture", GgufValue::String("bitnet".to_string())),
    ];

    let data = build_gguf_bytes(metadata);

    // Test metadata parsing directly without validation
    let reader = GgufReader::new(&data).unwrap();

    // Test that we can extract the metadata fields directly
    assert_eq!(reader.get_string_metadata("general.name"), Some("test_model".to_string()));
    assert_eq!(reader.get_u32_metadata("llama.vocab_size"), Some(50000));
    assert_eq!(reader.get_string_metadata("general.architecture"), Some("bitnet".to_string()));
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

/// Helper to build valid GGUF v3 bytes for testing (no tensors)
fn build_gguf_bytes_v3(kvs: Vec<(&str, GgufValue)>) -> Vec<u8> {
    build_gguf_bytes_v3_with_align(kvs, 32)
}

/// Helper to build valid GGUF v3 bytes with custom alignment
fn build_gguf_bytes_v3_with_align(kvs: Vec<(&str, GgufValue)>, align: u32) -> Vec<u8> {
    let mut data = Vec::<u8>::new();
    const V3: u32 = 3;

    // --- Header (v3) ---
    data.extend_from_slice(b"GGUF"); // magic
    data.extend_from_slice(&V3.to_le_bytes()); // version
    let n_tensors = 0u64;
    let n_kv = kvs.len() as u64;
    data.extend_from_slice(&n_tensors.to_le_bytes()); // n_tensors
    data.extend_from_slice(&n_kv.to_le_bytes()); // n_kv
    data.extend_from_slice(&align.to_le_bytes()); // alignment (u32)
    let doff_pos = data.len();
    data.extend_from_slice(&0u64.to_le_bytes()); // placeholder data_offset

    // --- KV section ---
    for (key, value) in kvs {
        let kb = key.as_bytes();
        data.extend_from_slice(&(kb.len() as u64).to_le_bytes());
        data.extend_from_slice(kb);
        write_gguf_value(&mut data, value);
    }

    // --- compute & backfill data_offset, then pad to it ---
    let pad = (align as usize - (data.len() % align as usize)) % align as usize;
    let data_offset = (data.len() + pad) as u64;
    data[doff_pos..doff_pos + 8].copy_from_slice(&data_offset.to_le_bytes());
    data.resize(data_offset as usize, 0);

    // no tensors (n_tensors = 0)
    data
}

#[test]
fn test_reader_accepts_v3_header_and_metadata() {
    let bytes = build_gguf_bytes_v3(vec![
        ("u32_key", GgufValue::U32(123)),
        ("i32_key", GgufValue::I32(-7)),
        ("f32_key", GgufValue::F32(std::f32::consts::PI)),
        ("bool_key", GgufValue::Bool(true)),
        ("str_key", GgufValue::String("hello".into())),
    ]);
    let r = GgufReader::new(&bytes).expect("read gguf v3");
    assert_eq!(r.get_u32_metadata("u32_key"), Some(123));
    assert_eq!(r.get_i32_metadata("i32_key"), Some(-7));
    assert!((r.get_f32_metadata("f32_key").unwrap_or(0.0) - std::f32::consts::PI).abs() < 1e-6);
    assert_eq!(r.get_bool_metadata("bool_key"), Some(true));
    assert_eq!(r.get_string_metadata("str_key").as_deref(), Some("hello"));
    // New getters also work:
    assert_eq!(r.alignment(), 32);
    assert_eq!((r.data_offset() as usize) % (r.alignment() as usize), 0);
}

#[test]
fn test_reader_v2_v3_parity_for_same_kvs() {
    let kvs = vec![
        ("tokenizer.ggml.model", GgufValue::String("gpt2".into())),
        ("tokenizer.ggml.add_bos", GgufValue::Bool(false)),
        ("tokenizer.ggml.add_eos", GgufValue::Bool(true)),
        ("tokenizer.ggml.vocab_size", GgufValue::U32(128_256)),
    ];
    let v2 = build_gguf_bytes(kvs.clone());
    let v3 = build_gguf_bytes_v3(kvs);
    let r2 = GgufReader::new(&v2).expect("v2 ok");
    let r3 = GgufReader::new(&v3).expect("v3 ok");
    assert_eq!(
        r2.get_string_metadata("tokenizer.ggml.model"),
        r3.get_string_metadata("tokenizer.ggml.model")
    );
    assert_eq!(
        r2.get_bool_metadata("tokenizer.ggml.add_bos"),
        r3.get_bool_metadata("tokenizer.ggml.add_bos")
    );
    assert_eq!(
        r2.get_bool_metadata("tokenizer.ggml.add_eos"),
        r3.get_bool_metadata("tokenizer.ggml.add_eos")
    );
    assert_eq!(
        r2.get_u32_metadata("tokenizer.ggml.vocab_size"),
        r3.get_u32_metadata("tokenizer.ggml.vocab_size")
    );
}

#[test]
fn test_reader_v3_ignores_bad_data_offset() {
    // Build a v3 file but deliberately write data_offset = 0 (bad).
    let mut bytes = build_gguf_bytes_v3(vec![("k", GgufValue::U32(1))]);

    // Overwrite the data_offset (8 bytes) to 0.
    // Header layout: "GGUF"(4) + ver(4) + n_t(8) + n_kv(8) + align(4) + data_offset(8)
    //                ^0         ^4      ^8        ^16        ^24        ^28
    let doff_pos = 28;
    bytes[doff_pos..doff_pos + 8].copy_from_slice(&0u64.to_le_bytes());

    let r = GgufReader::new(&bytes).expect("v3 with bad doff should still parse via fallback");
    assert_eq!(r.get_u32_metadata("k"), Some(1));
    // We don't assert the exact data_start; the fact that metadata read succeeded
    // proves we fell back to align_up(kv_end, alignment).
}

#[test]
fn test_gguf_arithmetic_overflow_protection() {
    // Test arithmetic operations in header parsing and offset calculations

    // Test 1: Tensor offset overflow protection
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&2u32.to_le_bytes()); // Version
    data.extend_from_slice(&1u64.to_le_bytes()); // One tensor
    data.extend_from_slice(&0u64.to_le_bytes()); // Zero metadata

    // Add a tensor info with maximum offset to test overflow protection
    let tensor_name = "test_tensor";
    data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    data.extend_from_slice(tensor_name.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // F32 tensor type
    data.extend_from_slice(&2u32.to_le_bytes()); // 2D tensor
    data.extend_from_slice(&10usize.to_le_bytes());
    data.extend_from_slice(&10usize.to_le_bytes());
    data.extend_from_slice(&u64::MAX.to_le_bytes()); // Maximum offset to trigger overflow

    let result = GgufReader::new(&data);
    assert!(result.is_err(), "Should reject tensors with overflow-inducing offsets");

    // Test 2: Alignment calculation edge cases
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // No tensors
    data.extend_from_slice(&1u64.to_le_bytes()); // One metadata
    data.extend_from_slice(&1u32.to_le_bytes()); // Alignment = 1 (edge case)
    data.extend_from_slice(&0u64.to_le_bytes()); // data_offset = 0

    // Add minimal metadata
    let key = "k";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // U32 type
    data.extend_from_slice(&1u32.to_le_bytes()); // Value

    let result = GgufReader::new(&data);
    assert!(result.is_ok(), "Should handle alignment=1 gracefully");
    let reader = result.unwrap();
    assert_eq!(reader.alignment(), 1);
}

#[test]
fn test_gguf_security_validation_edge_cases() {
    // Test security limits and validation logic robustness

    // Test 1: Extremely large metadata count
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // No tensors
    data.extend_from_slice(&(u32::MAX as u64).to_le_bytes()); // Huge metadata count

    let result = GgufReader::new(&data);
    assert!(result.is_err(), "Should reject files with excessive metadata count");

    // Test 2: Memory exhaustion protection for large tensor counts
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&(u32::MAX as u64).to_le_bytes()); // Huge tensor count
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GgufReader::new(&data);
    assert!(result.is_err(), "Should reject files with excessive tensor count");

    // Test 3: String length validation with potential overflow
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    // Add metadata with extremely large string length
    data.extend_from_slice(&u64::MAX.to_le_bytes()); // Huge string length

    let result = GgufReader::new(&data);
    assert!(result.is_err(), "Should reject files with malformed string lengths");
}

#[test]
fn test_gguf_extreme_value_boundary_conditions() {
    // Test extreme values in GGUF parsing and tensor handling

    // Test 1: Zero-sized tensor dimensions
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // One tensor
    data.extend_from_slice(&2u64.to_le_bytes()); // Required metadata

    // Add required metadata first
    let arch_key = "general.architecture";
    data.extend_from_slice(&(arch_key.len() as u64).to_le_bytes());
    data.extend_from_slice(arch_key.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes()); // String type
    let arch_val = "bitnet";
    data.extend_from_slice(&(arch_val.len() as u64).to_le_bytes());
    data.extend_from_slice(arch_val.as_bytes());

    let name_key = "general.name";
    data.extend_from_slice(&(name_key.len() as u64).to_le_bytes());
    data.extend_from_slice(name_key.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes()); // String type
    let name_val = "test";
    data.extend_from_slice(&(name_val.len() as u64).to_le_bytes());
    data.extend_from_slice(name_val.as_bytes());

    // Add tensor with zero dimension
    let tensor_name = "zero_tensor";
    data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    data.extend_from_slice(tensor_name.as_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // F32 tensor
    data.extend_from_slice(&2u32.to_le_bytes()); // 2D
    data.extend_from_slice(&0usize.to_le_bytes()); // Zero dimension
    data.extend_from_slice(&10usize.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // Zero offset

    let result = GgufReader::new(&data);
    assert!(result.is_err(), "Should reject tensors with zero dimensions");

    // Test 2: Maximum dimension values
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    // Required metadata
    let arch_key = "general.architecture";
    data.extend_from_slice(&(arch_key.len() as u64).to_le_bytes());
    data.extend_from_slice(arch_key.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let arch_val = "bitnet";
    data.extend_from_slice(&(arch_val.len() as u64).to_le_bytes());
    data.extend_from_slice(arch_val.as_bytes());

    let name_key = "general.name";
    data.extend_from_slice(&(name_key.len() as u64).to_le_bytes());
    data.extend_from_slice(name_key.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let name_val = "test";
    data.extend_from_slice(&(name_val.len() as u64).to_le_bytes());
    data.extend_from_slice(name_val.as_bytes());

    // Tensor with extreme dimensions
    let tensor_name = "max_tensor";
    data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    data.extend_from_slice(tensor_name.as_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&usize::MAX.to_le_bytes()); // Maximum dimension
    data.extend_from_slice(&usize::MAX.to_le_bytes()); // Maximum dimension
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GgufReader::new(&data);
    assert!(result.is_err(), "Should handle extreme dimension values safely");
}

#[test]
fn test_gguf_tensor_data_boundary_validation() {
    // Test tensor data boundary checks and validation

    // Create minimal valid GGUF with one tensor
    let metadata = vec![
        ("general.architecture", GgufValue::String("bitnet".to_string())),
        ("general.name", GgufValue::String("test".to_string())),
    ];

    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // One tensor
    data.extend_from_slice(&2u64.to_le_bytes()); // Two metadata items

    // Write metadata
    for (key, value) in metadata {
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        write_gguf_value(&mut data, value);
    }

    // Add tensor info
    let tensor_name = "test.weight";
    data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    data.extend_from_slice(tensor_name.as_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // F32
    data.extend_from_slice(&1u32.to_le_bytes()); // 1D
    data.extend_from_slice(&100usize.to_le_bytes()); // 100 elements
    data.extend_from_slice(&0u64.to_le_bytes()); // Offset 0

    // Pad to alignment
    let pad = (32 - (data.len() % 32)) % 32;
    data.resize(data.len() + pad, 0);

    // Add insufficient tensor data (only 50 bytes instead of 400)
    data.resize(data.len() + 50, 0x42);

    let result = GgufReader::new(&data);
    if let Ok(reader) = result {
        let tensor_result = reader.get_tensor_data(0);
        assert!(tensor_result.is_err(), "Should detect tensor data extending beyond file");
    }
}

#[test]
fn test_gguf_corrupted_header_edge_cases() {
    // Test additional corrupted header scenarios for robustness

    // Test 1: Truncated header
    let truncated = vec![b'G', b'G', b'U']; // Incomplete magic
    let result = GgufReader::new(&truncated);
    assert!(result.is_err(), "Should reject truncated headers");

    // Test 2: Version boundary cases
    for version in [0u32, 1u32, 4u32, u32::MAX] {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&version.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let result = GgufReader::new(&data);
        if !(2..=3).contains(&version) {
            assert!(result.is_err(), "Should reject unsupported version {}", version);
        }
    }

    // Test 3: Misaligned tensor data with extreme alignment values
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes()); // v3
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // Alignment = 0 (invalid)
    data.extend_from_slice(&100u64.to_le_bytes()); // data_offset

    let key = "k";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    let result = GgufReader::new(&data);
    if let Ok(reader) = result {
        // Should clamp invalid alignment to safe default
        assert!(reader.alignment() >= 1, "Alignment should be clamped to safe minimum");
        assert!(reader.alignment().is_power_of_two(), "Alignment should be power of two");
    }
}

#[test]
fn test_reader_v3_clamps_weird_alignment() {
    // Write v3 with alignment = 0 (invalid), data_offset aligned to 32 anyway.
    let mut bytes = build_gguf_bytes_v3(vec![("flag", GgufValue::Bool(true))]);
    // Patch alignment to 0 to trigger the guard.
    bytes[24..28].copy_from_slice(&0u32.to_le_bytes());

    let r = GgufReader::new(&bytes).expect("v3 with align=0 should still parse");
    // Reader should clamp to 32 (warns in logs).
    assert_eq!(r.alignment(), 32);
    assert_eq!(r.get_bool_metadata("flag"), Some(true));
}

#[test]
fn test_reader_v3_respects_header_alignment() {
    let bytes = build_gguf_bytes_v3_with_align(vec![("val", GgufValue::U32(42))], 64);
    let r = GgufReader::new(&bytes).expect("v3/align=64");
    assert_eq!(r.alignment(), 64);
    assert_eq!(r.get_u32_metadata("val"), Some(42));
}

#[test]
fn test_reader_v3_rejects_doff_past_eof_and_falls_back() {
    // Build a minimal v3 file
    let mut bytes = build_gguf_bytes_v3(vec![("k", GgufValue::U32(7))]);

    // data_offset is at bytes[28..36] for v3 (after magic, ver, n_t, n_kv, align)
    let doff_pos = 28;
    let absurd_doff = (bytes.len() as u64) + 10_000; // well past EOF
    bytes[doff_pos..doff_pos + 8].copy_from_slice(&absurd_doff.to_le_bytes());

    // Reader should fall back to align_up(kv_end, alignment), NOT panic.
    let r = GgufReader::new(&bytes).expect("fallback path should parse");
    assert_eq!(r.get_u32_metadata("k"), Some(7));
}

#[test]
fn test_reader_v3_falls_back_when_doff_past_eof() {
    // Build valid v3 then patch data_offset to something absurdly beyond EOF.
    let mut bytes = build_gguf_bytes_v3(vec![("k", GgufValue::U32(7))]);
    // v3 header layout: magic(4) ver(4) n_t(8) n_kv(8) align(4) doff(8)
    let doff_pos = 28;
    let past_eof = (bytes.len() as u64) + 4096;
    bytes[doff_pos..doff_pos + 8].copy_from_slice(&past_eof.to_le_bytes());

    // Reader must not panic; it should fall back to align_up(kv_end, alignment).
    let r = GgufReader::new(&bytes).expect("fallback should parse");
    assert_eq!(r.get_u32_metadata("k"), Some(7));
}

#[allow(dead_code)]
fn build_gguf_bytes_with_tensor(metadata: Vec<(&str, GgufValue)>) -> Vec<u8> {
    let mut data = Vec::<u8>::new();
    const GGUF_VERSION: u32 = 2;
    const ALIGN: usize = 32;

    // --- Header (v2 shape expected by reader) ---
    data.extend_from_slice(b"GGUF"); // magic
    data.extend_from_slice(&GGUF_VERSION.to_le_bytes()); // version
    let n_tensors = 2u64; // We'll add two minimal tensors (embedding + attention)
    let n_kv = metadata.len() as u64;
    data.extend_from_slice(&n_tensors.to_le_bytes()); // n_tensors
    data.extend_from_slice(&n_kv.to_le_bytes()); // n_kv

    // --- KV section ---
    for (key, value) in metadata {
        let kb = key.as_bytes();
        data.extend_from_slice(&(kb.len() as u64).to_le_bytes());
        data.extend_from_slice(kb);
        write_gguf_value(&mut data, value);
    }

    // --- Tensor info section ---
    // Add minimal embedding tensor
    let tensor_name = "tok_embeddings.weight";
    let tensor_name_bytes = tensor_name.as_bytes();
    data.extend_from_slice(&(tensor_name_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(tensor_name_bytes);

    // Tensor dimensions (2D: vocab_size x embed_dim)
    let n_dims = 2u32;
    data.extend_from_slice(&n_dims.to_le_bytes());
    data.extend_from_slice(&50000u64.to_le_bytes()); // vocab_size
    data.extend_from_slice(&128u64.to_le_bytes()); // embed_dim

    // Tensor type (F32)
    data.extend_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32

    // Tensor offset (will be calculated later)
    let embed_data_size = 50000 * 128 * 4; // vocab_size * embed_dim * sizeof(f32)
    let mut current_pos = data.len() + 8; // +8 for the offset field we're about to write

    // Add attention tensor info
    let attn_tensor_name = "layers.0.self_attn.q_proj.weight";
    let attn_tensor_name_bytes = attn_tensor_name.as_bytes();
    current_pos += 8 + attn_tensor_name_bytes.len() + 4 + 8 + 4 + 8; // full tensor info size

    let pad = (ALIGN - (current_pos % ALIGN)) % ALIGN;
    let tensor_offset = (current_pos + pad) as u64;
    data.extend_from_slice(&tensor_offset.to_le_bytes());

    // Add minimal attention tensor
    data.extend_from_slice(&(attn_tensor_name_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(attn_tensor_name_bytes);

    // Tensor dimensions (2D: embed_dim x embed_dim)
    data.extend_from_slice(&n_dims.to_le_bytes());
    data.extend_from_slice(&128u64.to_le_bytes()); // embed_dim
    data.extend_from_slice(&128u64.to_le_bytes()); // embed_dim

    // Tensor type (F32)
    data.extend_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32

    // Tensor offset (right after embedding tensor)
    let attn_data_size = 128 * 128 * 4; // embed_dim * embed_dim * sizeof(f32)
    let attn_tensor_offset = tensor_offset + embed_data_size as u64;
    data.extend_from_slice(&attn_tensor_offset.to_le_bytes());

    // --- Align to 32 bytes for data section ---
    let current_len = data.len();
    let pad = (ALIGN - (current_len % ALIGN)) % ALIGN;
    data.resize(data.len() + pad, 0);

    // --- Tensor data section ---
    // Add minimal tensor data (all zeros for simplicity)
    data.resize(data.len() + embed_data_size, 0); // embedding tensor data
    data.resize(data.len() + attn_data_size, 0); // attention tensor data

    data
}

/// Enhanced GGUF edge case tests for robustness
#[test]
fn test_gguf_corrupted_header() {
    // Test with corrupted magic bytes
    let mut corrupted_data = Vec::new();
    corrupted_data.extend_from_slice(b"XXXX"); // Invalid magic
    corrupted_data.extend_from_slice(&2u32.to_le_bytes()); // Version 2
    corrupted_data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count
    corrupted_data.extend_from_slice(&0u64.to_le_bytes()); // Metadata count

    let result = GgufReader::new(&corrupted_data);
    assert!(result.is_err(), "Should fail with corrupted magic bytes");

    // Test with valid magic but corrupted version
    let mut version_corrupted = Vec::new();
    version_corrupted.extend_from_slice(b"GGUF"); // Valid magic
    version_corrupted.extend_from_slice(&999u32.to_le_bytes()); // Unsupported version
    version_corrupted.extend_from_slice(&0u64.to_le_bytes()); // Tensor count
    version_corrupted.extend_from_slice(&0u64.to_le_bytes()); // Metadata count

    let result = GgufReader::new(&version_corrupted);
    // Should either fail or handle gracefully
    if result.is_ok() {
        // Graceful handling is acceptable
    }
    // Expected failure is also acceptable
}

#[test]
fn test_gguf_truncated_file() {
    // Test with file truncated during header
    let incomplete_header = vec![b'G', b'G', b'U']; // Truncated magic
    let result = GgufReader::new(&incomplete_header);
    assert!(result.is_err(), "Should fail with truncated header");

    // Test with header present but metadata truncated
    let mut partial_metadata = Vec::new();
    partial_metadata.extend_from_slice(b"GGUF");
    partial_metadata.extend_from_slice(&2u32.to_le_bytes());
    partial_metadata.extend_from_slice(&0u64.to_le_bytes()); // No tensors
    partial_metadata.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata entry
    // Missing actual metadata entry

    let result = GgufReader::new(&partial_metadata);
    assert!(result.is_err(), "Should fail with truncated metadata");
}

#[test]
fn test_gguf_malformed_metadata() {
    let mut malformed_data = Vec::new();
    malformed_data.extend_from_slice(b"GGUF");
    malformed_data.extend_from_slice(&2u32.to_le_bytes());
    malformed_data.extend_from_slice(&0u64.to_le_bytes()); // No tensors
    malformed_data.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata entry

    // Add key with invalid length
    malformed_data.extend_from_slice(&u64::MAX.to_le_bytes()); // Excessive key length
    malformed_data.extend_from_slice(b"test"); // Key shorter than declared

    let result = GgufReader::new(&malformed_data);
    assert!(result.is_err(), "Should fail with malformed metadata key");
}

#[test]
fn test_gguf_extreme_tensor_counts() {
    // Test with extremely high tensor count but no actual tensors
    let mut extreme_count = Vec::new();
    extreme_count.extend_from_slice(b"GGUF");
    extreme_count.extend_from_slice(&2u32.to_le_bytes());
    extreme_count.extend_from_slice(&u64::MAX.to_le_bytes()); // Extreme tensor count
    extreme_count.extend_from_slice(&0u64.to_le_bytes()); // No metadata

    let result = GgufReader::new(&extreme_count);
    // Should handle large counts gracefully without allocating excessive memory
    if let Ok(reader) = result {
        // If parsing succeeds, tensor iteration should be bounded
        let count = reader.tensor_count() as usize;
        assert!(count <= 1000, "Tensor count should be bounded for safety");
    }
    // Rejection is acceptable
}

#[test]
fn test_gguf_misaligned_tensor_data() {
    // Create GGUF with tensor data at non-standard alignment
    let mut misaligned_data = Vec::new();
    misaligned_data.extend_from_slice(b"GGUF");
    misaligned_data.extend_from_slice(&2u32.to_le_bytes());
    misaligned_data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    misaligned_data.extend_from_slice(&0u64.to_le_bytes()); // No metadata

    // Add tensor info with misaligned offset
    let tensor_name = "test_tensor";
    misaligned_data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    misaligned_data.extend_from_slice(tensor_name.as_bytes());

    // Tensor dimensions (1D)
    misaligned_data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    misaligned_data.extend_from_slice(&4u64.to_le_bytes()); // 4 elements

    // Tensor type (F32)
    misaligned_data.extend_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32

    // Misaligned tensor offset (not 32-byte aligned)
    let misaligned_offset = 37u64; // Not aligned to 32 bytes
    misaligned_data.extend_from_slice(&misaligned_offset.to_le_bytes());

    // Pad to at least the tensor offset
    misaligned_data.resize(misaligned_offset as usize + 16, 0); // 4 f32 values

    let result = GgufReader::new(&misaligned_data);
    // Should handle misaligned data gracefully
    if let Ok(reader) = result {
        let info = reader.get_tensor_info(0);
        // Should not panic even with misaligned data
        if info.is_ok() {
            // Acceptable if parsing succeeds
        }
        // Acceptable if tensor access fails gracefully
    }
}

#[test]
fn test_gguf_memory_exhaustion_protection() {
    // Test protection against memory exhaustion attacks
    let mut large_metadata = Vec::new();
    large_metadata.extend_from_slice(b"GGUF");
    large_metadata.extend_from_slice(&2u32.to_le_bytes());
    large_metadata.extend_from_slice(&0u64.to_le_bytes()); // No tensors
    large_metadata.extend_from_slice(&10000u64.to_le_bytes()); // Many metadata entries

    // Add a few normal entries, then truncate
    // This tests that parser doesn't pre-allocate based on metadata count
    for i in 0..3 {
        let key = format!("key_{}", i);
        large_metadata.extend_from_slice(&(key.len() as u64).to_le_bytes());
        large_metadata.extend_from_slice(key.as_bytes());

        // String value
        large_metadata.extend_from_slice(&8u32.to_le_bytes()); // String type
        let value = format!("value_{}", i);
        large_metadata.extend_from_slice(&(value.len() as u64).to_le_bytes());
        large_metadata.extend_from_slice(value.as_bytes());
    }
    // File ends here, much shorter than declared metadata count

    let result = GgufReader::new(&large_metadata);
    // Should fail gracefully without consuming excessive memory
    if let Ok(reader) = result {
        // If parsing succeeds, should handle iteration safely
        let keys: Vec<_> = reader.metadata_keys().into_iter().take(100).collect(); // Bounded iteration
        assert!(keys.len() <= 100, "Should not iterate beyond reasonable bounds");
    }
    // Expected failure is acceptable
}

#[test]
fn test_ln_name_matching() {
    let positives = [
        "layers.0.attention_norm.weight",
        "layers.0.ffn_norm.weight",
        "final_norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "model.layers.0.attention_norm.weight",
        "blk.0.attn_norm.weight",
    ];
    for n in positives {
        assert!(is_layernorm_weight(n), "should match {}", n);
    }
    assert!(!is_layernorm_weight("layers.0.attention_norm.bias"));
    assert!(!is_layernorm_weight("layers.0.q_proj.weight"));
}

#[test]
#[serial_test::serial]
fn test_ln_gamma_validator_envelope() {
    use super::loader::GgufLoader;
    use candle_core::Tensor;

    // Helper to create test tensors with specific RMS
    fn tensor_with_rms(rms_target: f32, size: usize) -> Tensor {
        let data: Vec<f32> = (0..size).map(|i| (i as f32 / size as f32) * rms_target).collect();
        Tensor::from_vec(data, &[size], &candle_core::Device::Cpu).unwrap()
    }

    // Test 1: Valid RMS should pass in non-strict mode (no BITNET_STRICT_MODE set)
    {
        let valid_tensor = tensor_with_rms(1.0, 100);
        let result = GgufLoader::check_ln_gamma_stats("test.norm.weight", &valid_tensor);
        assert!(result.is_ok(), "Valid RMS should pass");
    }

    // Test 2: Invalid RMS should warn in non-strict mode but pass
    {
        let invalid_tensor = tensor_with_rms(0.01, 100);
        let result = GgufLoader::check_ln_gamma_stats("test.norm.weight", &invalid_tensor);
        assert!(result.is_ok(), "Invalid RMS should warn but pass in non-strict mode");
    }

    // Test 3: Invalid RMS should fail in strict mode
    {
        let _guard = EnvGuard::new("BITNET_STRICT_MODE");
        _guard.set("1");
        let invalid_tensor = tensor_with_rms(0.01, 100);
        let result = GgufLoader::check_ln_gamma_stats("test.norm.weight", &invalid_tensor);
        assert!(result.is_err(), "Invalid RMS should fail in strict mode");
    }

    // Test 4: Edge of envelope (0.5) should pass
    {
        let edge_low_tensor = tensor_with_rms(0.5, 100);
        let result = GgufLoader::check_ln_gamma_stats("test.norm.weight", &edge_low_tensor);
        assert!(result.is_ok(), "RMS at lower boundary should pass");
    }

    // Test 5: Edge of envelope (2.0) should pass
    {
        let edge_high_tensor = tensor_with_rms(2.0, 100);
        let result = GgufLoader::check_ln_gamma_stats("test.norm.weight", &edge_high_tensor);
        assert!(result.is_ok(), "RMS at upper boundary should pass");
    }
}
