//! GGUF loader correctness regression tests.
//!
//! Tests header parsing, metadata extraction, tensor info, architecture
//! detection, config extraction, and error handling — all with synthetic
//! in-memory GGUF buffers (no model files required).

use std::collections::HashMap;
use std::io::Cursor;

use bitnet_models::config::{GgufModelConfig, GgufQuantizationConfig};
use bitnet_models::formats::gguf::{GgufReader, GgufTensorType as ReaderTensorType, GgufValue};
use bitnet_models::gguf_writer::{GgufBuilder, GgufTensorType};

// ---------------------------------------------------------------------------
// Helper: build a synthetic GGUF buffer using GgufBuilder → Cursor roundtrip
// ---------------------------------------------------------------------------

/// Minimal GGUF with only architecture metadata and no tensors.
fn build_minimal_gguf(arch: &str) -> Vec<u8> {
    let cursor = GgufBuilder::new()
        .architecture(arch)
        .write(Cursor::new(Vec::new()))
        .expect("write minimal GGUF");
    cursor.into_inner()
}

/// GGUF with full model metadata (llama-like architecture).
fn build_full_metadata_gguf() -> Vec<u8> {
    let tensor_data = vec![0u8; 128]; // 32 F32 elements = 128 bytes
    let cursor = GgufBuilder::new()
        .architecture("llama")
        .description("test model")
        .metadata_string("general.name", "test-llama-7b")
        .metadata_u32("llama.vocab_size", 32000)
        .metadata_u32("llama.embedding_length", 4096)
        .metadata_u32("llama.block_count", 32)
        .metadata_u32("llama.attention.head_count", 32)
        .metadata_u32("llama.attention.head_count_kv", 8)
        .metadata_u32("llama.feed_forward_length", 11008)
        .metadata_u32("llama.context_length", 2048)
        .metadata_f32("llama.rope.freq_base", 10000.0)
        .tensor("token_embd.weight", &[4096, 32], GgufTensorType::F32, &tensor_data)
        .write(Cursor::new(Vec::new()))
        .expect("write full GGUF");
    cursor.into_inner()
}

/// GGUF with multiple tensors of varying types.
fn build_multi_tensor_gguf() -> Vec<u8> {
    let f32_data = vec![0u8; 128]; // 32 F32 elements
    let f16_data = vec![0u8; 64]; // 32 F16 elements
    let cursor = GgufBuilder::new()
        .architecture("bitnet")
        .tensor("blk.0.attn_q.weight", &[64, 64], GgufTensorType::F32, &f32_data)
        .tensor("blk.0.attn_k.weight", &[64, 32], GgufTensorType::F16, &f16_data)
        .tensor("blk.0.attn_v.weight", &[64, 32], GgufTensorType::F16, &f16_data)
        .write(Cursor::new(Vec::new()))
        .expect("write multi-tensor GGUF");
    cursor.into_inner()
}

// ===========================================================================
// Header parsing
// ===========================================================================

#[test]
fn header_magic_is_gguf() {
    let buf = build_minimal_gguf("llama");
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(&reader.header.magic, b"GGUF");
}

#[test]
fn header_version_is_3() {
    let buf = build_minimal_gguf("llama");
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(reader.header.version, 3);
}

#[test]
fn header_tensor_count_matches() {
    let buf = build_multi_tensor_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(reader.header.tensor_count, 3);
}

#[test]
fn header_metadata_kv_count_matches() {
    let buf = build_full_metadata_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    // architecture + description + name + 8 arch-prefixed keys = 11
    assert!(reader.header.metadata_kv_count >= 10);
}

#[test]
fn header_zero_tensors() {
    let buf = build_minimal_gguf("llama");
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(reader.header.tensor_count, 0);
    assert_eq!(reader.tensor_count(), 0);
}

// ===========================================================================
// Metadata extraction
// ===========================================================================

#[test]
fn metadata_architecture_string() {
    let buf = build_minimal_gguf("bitnet");
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(reader.get_string_metadata("general.architecture"), Some("bitnet".to_string()));
}

#[test]
fn metadata_u32_values() {
    let buf = build_full_metadata_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(reader.get_u32_metadata("llama.vocab_size"), Some(32000));
    assert_eq!(reader.get_u32_metadata("llama.embedding_length"), Some(4096));
    assert_eq!(reader.get_u32_metadata("llama.block_count"), Some(32));
    assert_eq!(reader.get_u32_metadata("llama.attention.head_count"), Some(32));
    assert_eq!(reader.get_u32_metadata("llama.attention.head_count_kv"), Some(8));
    assert_eq!(reader.get_u32_metadata("llama.feed_forward_length"), Some(11008));
    assert_eq!(reader.get_u32_metadata("llama.context_length"), Some(2048));
}

#[test]
fn metadata_f32_values() {
    let buf = build_full_metadata_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    let rope = reader.get_f32_metadata("llama.rope.freq_base").unwrap();
    assert!((rope - 10000.0).abs() < f32::EPSILON);
}

#[test]
fn metadata_string_values() {
    let buf = build_full_metadata_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(reader.get_string_metadata("general.name"), Some("test-llama-7b".to_string()));
    assert_eq!(reader.get_string_metadata("general.description"), Some("test model".to_string()));
}

#[test]
fn metadata_missing_key_returns_none() {
    let buf = build_minimal_gguf("llama");
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(reader.get_u32_metadata("nonexistent.key"), None);
    assert_eq!(reader.get_string_metadata("nonexistent.key"), None);
    assert_eq!(reader.get_f32_metadata("nonexistent.key"), None);
    assert_eq!(reader.get_bool_metadata("nonexistent.key"), None);
}

#[test]
fn metadata_keys_are_enumerable() {
    let buf = build_full_metadata_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    let keys = reader.metadata_keys();
    assert!(keys.contains(&"general.architecture"));
    assert!(keys.contains(&"general.name"));
    assert!(keys.contains(&"llama.vocab_size"));
}

#[test]
fn metadata_bool_roundtrip() {
    let cursor = GgufBuilder::new()
        .architecture("llama")
        .metadata_bool("custom.flag", true)
        .write(Cursor::new(Vec::new()))
        .expect("write");
    let buf = cursor.into_inner();
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(reader.get_bool_metadata("custom.flag"), Some(true));
}

// ===========================================================================
// Tensor info parsing
// ===========================================================================

#[test]
fn tensor_names_listed() {
    let buf = build_multi_tensor_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    let names = reader.tensor_names();
    assert_eq!(names.len(), 3);
    assert!(names.contains(&"blk.0.attn_q.weight"));
    assert!(names.contains(&"blk.0.attn_k.weight"));
    assert!(names.contains(&"blk.0.attn_v.weight"));
}

#[test]
fn tensor_info_by_name() {
    let buf = build_multi_tensor_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    let info = reader.get_tensor_info_by_name("blk.0.attn_q.weight").unwrap();
    assert_eq!(info.name, "blk.0.attn_q.weight");
    assert_eq!(info.shape, vec![64, 64]);
    assert_eq!(info.tensor_type, ReaderTensorType::F32);
}

#[test]
fn tensor_info_by_index() {
    let buf = build_multi_tensor_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    let info = reader.get_tensor_info(0).unwrap();
    assert_eq!(info.name, "blk.0.attn_q.weight");
}

#[test]
fn tensor_data_readable() {
    let buf = build_multi_tensor_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    let data = reader.get_tensor_data(0).unwrap();
    assert_eq!(data.len(), 128); // 32 F32 elements × 4 bytes
}

#[test]
fn tensor_data_by_name() {
    let buf = build_multi_tensor_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    let data = reader.get_tensor_data_by_name("blk.0.attn_k.weight").unwrap();
    assert_eq!(data.len(), 64); // 32 F16 elements × 2 bytes
}

#[test]
fn tensor_type_preserved() {
    let buf = build_multi_tensor_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    let q = reader.get_tensor_info_by_name("blk.0.attn_q.weight").unwrap();
    let k = reader.get_tensor_info_by_name("blk.0.attn_k.weight").unwrap();
    assert_eq!(q.tensor_type, ReaderTensorType::F32);
    assert_eq!(k.tensor_type, ReaderTensorType::F16);
}

// ===========================================================================
// Architecture detection
// ===========================================================================

#[test]
fn detect_llama_architecture() {
    let buf = build_minimal_gguf("llama");
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(reader.get_string_metadata("general.architecture"), Some("llama".to_string()));
}

#[test]
fn detect_bitnet_architecture() {
    let buf = build_minimal_gguf("bitnet");
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(reader.get_string_metadata("general.architecture"), Some("bitnet".to_string()));
}

#[test]
fn detect_phi_architecture() {
    let buf = build_minimal_gguf("phi");
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(reader.get_string_metadata("general.architecture"), Some("phi".to_string()));
}

#[test]
fn detect_various_architectures() {
    for arch in &["llama", "bitnet", "phi", "gpt2", "mamba", "qwen2"] {
        let buf = build_minimal_gguf(arch);
        let reader = GgufReader::new(&buf).expect("parse");
        assert_eq!(
            reader.get_string_metadata("general.architecture"),
            Some(arch.to_string()),
            "failed for architecture: {arch}"
        );
    }
}

// ===========================================================================
// GgufModelConfig extraction
// ===========================================================================

#[test]
fn config_from_metadata_basic() {
    let mut meta = HashMap::new();
    meta.insert("general.architecture".to_string(), GgufValue::String("llama".to_string()));
    meta.insert("llama.vocab_size".to_string(), GgufValue::U32(32000));
    meta.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
    meta.insert("llama.block_count".to_string(), GgufValue::U32(32));
    meta.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
    meta.insert("llama.attention.head_count_kv".to_string(), GgufValue::U32(8));
    meta.insert("llama.feed_forward_length".to_string(), GgufValue::U32(11008));
    meta.insert("llama.context_length".to_string(), GgufValue::U32(2048));

    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert_eq!(config.architecture, "llama");
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_heads, 32);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.intermediate_size, 11008);
    assert_eq!(config.max_seq_len, 2048);
}

#[test]
fn config_defaults_for_missing_metadata() {
    let meta = HashMap::new();
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    // With no metadata, defaults should be applied
    assert_eq!(config.architecture, "llama"); // default
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_heads, 32);
    assert_eq!(config.num_kv_heads, 32); // defaults to num_heads
    assert_eq!(config.max_seq_len, 2048);
}

#[test]
fn config_head_dim_computed() {
    let mut meta = HashMap::new();
    meta.insert("general.architecture".to_string(), GgufValue::String("llama".to_string()));
    meta.insert("llama.embedding_length".to_string(), GgufValue::U32(2048));
    meta.insert("llama.attention.head_count".to_string(), GgufValue::U32(16));
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert_eq!(config.head_dim, 128); // 2048 / 16
}

#[test]
fn config_gqa_detection() {
    let mut meta = HashMap::new();
    meta.insert("general.architecture".to_string(), GgufValue::String("llama".to_string()));
    meta.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
    meta.insert("llama.attention.head_count_kv".to_string(), GgufValue::U32(8));
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(config.is_gqa());
    assert_eq!(config.gqa_group_size(), 4);
}

#[test]
fn config_mha_no_gqa() {
    let mut meta = HashMap::new();
    meta.insert("general.architecture".to_string(), GgufValue::String("llama".to_string()));
    meta.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
    meta.insert("llama.attention.head_count_kv".to_string(), GgufValue::U32(32));
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(!config.is_gqa());
    assert_eq!(config.gqa_group_size(), 1);
}

#[test]
fn config_rope_theta_extracted() {
    let mut meta = HashMap::new();
    meta.insert("general.architecture".to_string(), GgufValue::String("llama".to_string()));
    meta.insert("llama.rope.freq_base".to_string(), GgufValue::F32(500000.0));
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!((config.rope_theta - 500000.0).abs() < 0.1);
}

#[test]
fn config_rope_theta_default() {
    let meta = HashMap::new();
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!((config.rope_theta - 10000.0).abs() < 0.1);
}

#[test]
fn config_model_name_optional() {
    let meta = HashMap::new();
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(config.model_name.is_none());

    let mut meta2 = HashMap::new();
    meta2.insert("general.name".to_string(), GgufValue::String("my-model".to_string()));
    let config2 = GgufModelConfig::from_gguf_metadata(&meta2).unwrap();
    assert_eq!(config2.model_name, Some("my-model".to_string()));
}

// ===========================================================================
// Config validation
// ===========================================================================

#[test]
fn config_validate_valid() {
    let mut meta = HashMap::new();
    meta.insert("general.architecture".to_string(), GgufValue::String("llama".to_string()));
    meta.insert("llama.vocab_size".to_string(), GgufValue::U32(32000));
    meta.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
    meta.insert("llama.block_count".to_string(), GgufValue::U32(32));
    meta.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
    meta.insert("llama.attention.head_count_kv".to_string(), GgufValue::U32(8));
    meta.insert("llama.feed_forward_length".to_string(), GgufValue::U32(11008));
    meta.insert("llama.context_length".to_string(), GgufValue::U32(2048));
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(config.validate().is_ok());
}

#[test]
fn config_validate_kv_heads_le_heads() {
    let mut meta = HashMap::new();
    meta.insert("general.architecture".to_string(), GgufValue::String("llama".to_string()));
    meta.insert("llama.attention.head_count".to_string(), GgufValue::U32(8));
    meta.insert("llama.attention.head_count_kv".to_string(), GgufValue::U32(16));
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(config.validate().is_err());
}

#[test]
fn config_validate_heads_divisible_by_kv_heads() {
    let mut meta = HashMap::new();
    meta.insert("general.architecture".to_string(), GgufValue::String("llama".to_string()));
    meta.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
    meta.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
    meta.insert("llama.attention.head_count_kv".to_string(), GgufValue::U32(5));
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(config.validate().is_err());
}

#[test]
fn config_memory_estimate_nonzero() {
    let mut meta = HashMap::new();
    meta.insert("general.architecture".to_string(), GgufValue::String("llama".to_string()));
    meta.insert("llama.vocab_size".to_string(), GgufValue::U32(32000));
    meta.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
    meta.insert("llama.block_count".to_string(), GgufValue::U32(32));
    meta.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
    meta.insert("llama.attention.head_count_kv".to_string(), GgufValue::U32(8));
    meta.insert("llama.feed_forward_length".to_string(), GgufValue::U32(11008));
    meta.insert("llama.context_length".to_string(), GgufValue::U32(2048));
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    let estimate = config.memory_estimate();
    assert!(estimate.weight_bytes > 0);
    assert!(estimate.kv_cache_bytes > 0);
    assert!(estimate.total_bytes > 0);
    assert!(estimate.total_bytes == estimate.weight_bytes + estimate.kv_cache_bytes);
    assert!(!estimate.summary.is_empty());
}

// ===========================================================================
// Error handling for invalid / truncated data
// ===========================================================================

#[test]
fn error_on_empty_data() {
    let result = GgufReader::new(&[]);
    assert!(result.is_err());
}

#[test]
fn error_on_truncated_header() {
    // Only 8 bytes — not enough for a full GGUF header
    let data = b"GGUF\x03\x00\x00\x00";
    let result = GgufReader::new(data);
    assert!(result.is_err());
}

#[test]
fn error_on_wrong_magic() {
    let mut buf = build_minimal_gguf("llama");
    // Corrupt magic bytes
    buf[0] = b'X';
    let result = GgufReader::new(&buf);
    assert!(result.is_err());
}

#[test]
fn error_on_unsupported_version() {
    let mut buf = build_minimal_gguf("llama");
    // Set version to 99
    buf[4] = 99;
    buf[5] = 0;
    buf[6] = 0;
    buf[7] = 0;
    let result = GgufReader::new(&buf);
    assert!(result.is_err());
}

#[test]
fn error_on_version_1() {
    let mut buf = build_minimal_gguf("llama");
    // Set version to 1 (only v2/v3 supported)
    buf[4] = 1;
    buf[5] = 0;
    buf[6] = 0;
    buf[7] = 0;
    let result = GgufReader::new(&buf);
    assert!(result.is_err());
}

#[test]
fn error_on_nonexistent_tensor_index() {
    let buf = build_minimal_gguf("llama");
    let reader = GgufReader::new(&buf).expect("parse");
    assert!(reader.get_tensor_info(0).is_err());
    assert!(reader.get_tensor_info(999).is_err());
}

#[test]
fn error_on_nonexistent_tensor_name() {
    let buf = build_multi_tensor_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    assert!(reader.get_tensor_info_by_name("does.not.exist").is_none());
    assert!(reader.get_tensor_data_by_name("does.not.exist").is_err());
}

// ===========================================================================
// Tensor count consistency
// ===========================================================================

#[test]
fn tensor_count_matches_tensor_names() {
    let buf = build_multi_tensor_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(reader.tensor_count() as usize, reader.tensor_names().len());
}

#[test]
fn tensor_count_zero_for_metadata_only() {
    let buf = build_minimal_gguf("llama");
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(reader.tensor_count(), 0);
    assert!(reader.tensor_names().is_empty());
}

// ===========================================================================
// ReaderTensorType correctness
// ===========================================================================

#[test]
fn tensor_type_from_u32_known_types() {
    assert_eq!(ReaderTensorType::from_u32(0).unwrap(), ReaderTensorType::F32);
    assert_eq!(ReaderTensorType::from_u32(1).unwrap(), ReaderTensorType::F16);
    assert_eq!(ReaderTensorType::from_u32(36).unwrap(), ReaderTensorType::I2_S);
    assert_eq!(ReaderTensorType::from_u32(24).unwrap(), ReaderTensorType::IQ2_S);
}

#[test]
fn tensor_type_from_u32_unknown_type_errors() {
    assert!(ReaderTensorType::from_u32(255).is_err());
    assert!(ReaderTensorType::from_u32(999).is_err());
}

#[test]
fn tensor_type_element_size() {
    assert_eq!(ReaderTensorType::F32.element_size(), 4);
    assert_eq!(ReaderTensorType::F16.element_size(), 2);
    assert_eq!(ReaderTensorType::F64.element_size(), 8);
}

#[test]
fn tensor_type_is_quantized() {
    assert!(!ReaderTensorType::F32.is_quantized());
    assert!(!ReaderTensorType::F16.is_quantized());
    assert!(!ReaderTensorType::F64.is_quantized());
    assert!(ReaderTensorType::Q4_0.is_quantized());
    assert!(ReaderTensorType::I2_S.is_quantized());
    assert!(ReaderTensorType::IQ2_S.is_quantized());
}

#[test]
fn tensor_type_block_size() {
    assert_eq!(ReaderTensorType::I2_S.block_size(), 32);
    assert_eq!(ReaderTensorType::IQ2_S.block_size(), 256);
    assert_eq!(ReaderTensorType::Q4_0.block_size(), 32);
    assert_eq!(ReaderTensorType::F32.block_size(), 1);
}

// ===========================================================================
// Writer → Reader roundtrip
// ===========================================================================

#[test]
fn roundtrip_metadata_preserved() {
    let cursor = GgufBuilder::new()
        .architecture("bitnet")
        .description("roundtrip test")
        .metadata_string("custom.author", "test-harness")
        .metadata_u32("custom.version", 42)
        .metadata_f32("custom.temperature", 0.7)
        .metadata_bool("custom.enabled", false)
        .write(Cursor::new(Vec::new()))
        .expect("write");
    let buf = cursor.into_inner();
    let reader = GgufReader::new(&buf).expect("parse");

    assert_eq!(reader.get_string_metadata("general.architecture"), Some("bitnet".to_string()));
    assert_eq!(
        reader.get_string_metadata("general.description"),
        Some("roundtrip test".to_string())
    );
    assert_eq!(reader.get_string_metadata("custom.author"), Some("test-harness".to_string()));
    assert_eq!(reader.get_u32_metadata("custom.version"), Some(42));
    let temp = reader.get_f32_metadata("custom.temperature").unwrap();
    assert!((temp - 0.7).abs() < f32::EPSILON);
    assert_eq!(reader.get_bool_metadata("custom.enabled"), Some(false));
}

#[test]
fn roundtrip_tensor_data_preserved() {
    let original: Vec<u8> = (0..64).collect();
    let cursor = GgufBuilder::new()
        .architecture("llama")
        .tensor("test.weight", &[16], GgufTensorType::F32, &original)
        .write(Cursor::new(Vec::new()))
        .expect("write");
    let buf = cursor.into_inner();
    let reader = GgufReader::new(&buf).expect("parse");
    let data = reader.get_tensor_data(0).unwrap();
    assert_eq!(data, &original[..]);
}

#[test]
fn roundtrip_multiple_tensors_data_integrity() {
    let data_a: Vec<u8> = vec![0xAA; 128];
    let data_b: Vec<u8> = vec![0xBB; 64];
    let data_c: Vec<u8> = vec![0xCC; 32];
    let cursor = GgufBuilder::new()
        .architecture("llama")
        .tensor("tensor_a", &[32], GgufTensorType::F32, &data_a)
        .tensor("tensor_b", &[32], GgufTensorType::F16, &data_b)
        .tensor("tensor_c", &[32], GgufTensorType::F32, &data_c)
        .write(Cursor::new(Vec::new()))
        .expect("write");
    let buf = cursor.into_inner();
    let reader = GgufReader::new(&buf).expect("parse");

    assert_eq!(reader.get_tensor_data_by_name("tensor_a").unwrap(), &data_a[..]);
    assert_eq!(reader.get_tensor_data_by_name("tensor_b").unwrap(), &data_b[..]);
    assert_eq!(reader.get_tensor_data_by_name("tensor_c").unwrap(), &data_c[..]);
}

// ===========================================================================
// Quantization config
// ===========================================================================

#[test]
fn quantization_config_default() {
    let qc = GgufQuantizationConfig::default();
    assert_eq!(qc.bit_width, 2);
    assert_eq!(qc.block_size, 64);
    assert_eq!(qc.format, "I2_S");
}

// ===========================================================================
// ReaderTensorType from quant string
// ===========================================================================

#[test]
fn tensor_type_from_quant_string() {
    assert_eq!(ReaderTensorType::from_quant_string("i2_s"), Some(ReaderTensorType::I2_S));
    assert_eq!(ReaderTensorType::from_quant_string("I2_S"), Some(ReaderTensorType::I2_S));
    assert_eq!(ReaderTensorType::from_quant_string("f32"), Some(ReaderTensorType::F32));
    assert_eq!(ReaderTensorType::from_quant_string("f16"), Some(ReaderTensorType::F16));
    assert_eq!(ReaderTensorType::from_quant_string("q4_0"), Some(ReaderTensorType::Q4_0));
    assert_eq!(ReaderTensorType::from_quant_string("unknown"), None);
}

// ===========================================================================
// I2S flavor types
// ===========================================================================

#[test]
fn i2s_flavor_block_sizes() {
    use bitnet_models::formats::gguf::I2SFlavor;

    assert_eq!(I2SFlavor::BitNet32F16.block_size(), 32);
    assert_eq!(I2SFlavor::Split32WithSibling.block_size(), 32);
    assert_eq!(I2SFlavor::GgmlQk256NoScale.block_size(), 256);
}

#[test]
fn i2s_flavor_bytes_per_block() {
    use bitnet_models::formats::gguf::I2SFlavor;

    assert_eq!(I2SFlavor::BitNet32F16.total_bytes_per_block(), 10);
    assert_eq!(I2SFlavor::Split32WithSibling.total_bytes_per_block(), 8);
    assert_eq!(I2SFlavor::GgmlQk256NoScale.total_bytes_per_block(), 64);
}

// ===========================================================================
// Header format description
// ===========================================================================

#[test]
fn header_format_description_v3() {
    let buf = build_minimal_gguf("llama");
    let reader = GgufReader::new(&buf).expect("parse");
    let desc = reader.header.format_description();
    assert!(desc.contains("GGUF v3"), "got: {desc}");
}

// ===========================================================================
// Metadata count consistency
// ===========================================================================

#[test]
fn metadata_count_equals_kv_count() {
    let buf = build_full_metadata_gguf();
    let reader = GgufReader::new(&buf).expect("parse");
    assert_eq!(reader.metadata_count(), reader.metadata_kv_count() as usize);
}

// ===========================================================================
// Edge case: very long tensor name
// ===========================================================================

#[test]
fn long_tensor_name_roundtrip() {
    let long_name = "blk.99.ffn_gate_inp.weight".repeat(3);
    let data = vec![0u8; 16];
    let cursor = GgufBuilder::new()
        .architecture("llama")
        .tensor(&long_name, &[4], GgufTensorType::F32, &data)
        .write(Cursor::new(Vec::new()))
        .expect("write");
    let buf = cursor.into_inner();
    let reader = GgufReader::new(&buf).expect("parse");
    assert!(reader.get_tensor_info_by_name(&long_name).is_some());
}
