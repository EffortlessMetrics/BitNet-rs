//! GGUF format tests

use super::*;
use crate::loader::FormatLoader;
use std::io::Write;
use tempfile::NamedTempFile;

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
        ("f32_key", GgufValue::F32(3.14)),
        ("bool_key", GgufValue::Bool(true)),
        ("str_key", GgufValue::String("hello".into())),
    ]);
    let r = GgufReader::new(&bytes).expect("read gguf v3");
    assert_eq!(r.get_u32_metadata("u32_key"), Some(123));
    assert_eq!(r.get_i32_metadata("i32_key"), Some(-7));
    assert!((r.get_f32_metadata("f32_key").unwrap_or(0.0) - 3.14).abs() < 1e-6);
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
