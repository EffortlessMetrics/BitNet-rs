//! Integration tests for `bitnet-st2gguf`.
//!
//! Coverage:
//! - GgufWriter: output file creation, GGUF magic, header fields, tensor storage
//! - TensorEntry and TensorDType: field access and type codes
//! - LayerNorm detection: GGUF blk-style, rms_norm, attention_norm variants
//! - count_layernorm_tensors: edge cases
//! - CLI binary: --help exits 0, nonexistent --input path exits non-zero
//! - Property test: blk-style ffn_norm always detected as LayerNorm

use std::fs;

use bitnet_st2gguf::{
    layernorm::{count_layernorm_tensors, is_layernorm_tensor},
    writer::{GgufWriter, MetadataValue, TensorDType, TensorEntry},
};
use proptest::prelude::*;
use tempfile::TempDir;

// ── GgufWriter file-creation tests ──────────────────────────────────────────

/// Writing a GGUF file creates the output path on disk.
#[test]
fn test_writer_creates_output_file() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("model.gguf");

    let writer = GgufWriter::new();
    writer.write_to_file(&out).unwrap();

    assert!(out.exists(), "output file must be created on disk");
}

/// The first four bytes of every GGUF file must be the magic b"GGUF".
#[test]
fn test_writer_output_starts_with_gguf_magic() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("model.gguf");

    GgufWriter::new().write_to_file(&out).unwrap();

    let data = fs::read(&out).unwrap();
    assert!(data.len() >= 4, "output must have at least 4 bytes");
    assert_eq!(&data[0..4], b"GGUF", "first 4 bytes must be the GGUF magic");
}

/// The GGUF version field (bytes 4–7) must equal 3.
#[test]
fn test_writer_header_version_is_3() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("model.gguf");

    GgufWriter::new().write_to_file(&out).unwrap();

    let data = fs::read(&out).unwrap();
    assert!(data.len() >= 8);
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    assert_eq!(version, 3, "GGUF version must be 3");
}

/// The tensor-count field (bytes 8–15) reflects the actual number of tensors added.
#[test]
fn test_writer_header_tensor_count() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("model.gguf");

    let mut writer = GgufWriter::new();
    writer.add_tensor(TensorEntry::new(
        "a".to_string(),
        vec![4],
        TensorDType::F16,
        vec![0u8; 8], // 4 × 2 bytes
    ));
    writer.add_tensor(TensorEntry::new("b".to_string(), vec![2], TensorDType::F16, vec![0u8; 4]));
    writer.write_to_file(&out).unwrap();

    let data = fs::read(&out).unwrap();
    assert!(data.len() >= 16);
    let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
    assert_eq!(tensor_count, 2, "header tensor_count must equal the number added");
}

/// The KV-count field (bytes 16–23) reflects the number of metadata entries.
#[test]
fn test_writer_header_kv_count() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("model.gguf");

    let mut writer = GgufWriter::new();
    writer.add_metadata("key.one", MetadataValue::U32(1));
    writer.add_metadata("key.two", MetadataValue::U32(2));
    writer.add_metadata("key.three", MetadataValue::U32(3));
    writer.write_to_file(&out).unwrap();

    let data = fs::read(&out).unwrap();
    assert!(data.len() >= 24);
    let kv_count = u64::from_le_bytes(data[16..24].try_into().unwrap());
    assert_eq!(kv_count, 3, "header kv_count must equal the number of metadata entries");
}

/// A writer with zero tensors still produces a valid non-empty GGUF file.
#[test]
fn test_writer_output_non_empty_with_no_tensors() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("empty.gguf");

    GgufWriter::new().write_to_file(&out).unwrap();

    let len = fs::metadata(&out).unwrap().len();
    assert!(len > 0, "GGUF file must be non-empty even with no tensors");
}

/// Three tensors added are all reflected in the header tensor count.
#[test]
fn test_writer_multiple_tensors_all_stored() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("multi.gguf");

    let mut writer = GgufWriter::new();
    for i in 0u32..3 {
        writer.add_tensor(TensorEntry::new(
            format!("layer.{i}.weight"),
            vec![2],
            TensorDType::F16,
            vec![0u8; 4],
        ));
    }
    writer.write_to_file(&out).unwrap();

    let data = fs::read(&out).unwrap();
    let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
    assert_eq!(tensor_count, 3);
}

/// A tensor with 1024 bytes of data can be written without error.
#[test]
fn test_writer_tensor_with_large_data() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("large.gguf");

    let mut writer = GgufWriter::new();
    writer.add_tensor(TensorEntry::new(
        "embed.weight".to_string(),
        vec![512], // 512 F16 elements
        TensorDType::F16,
        vec![0u8; 1024], // 512 × 2 bytes
    ));
    // Should not panic or return Err
    writer.write_to_file(&out).unwrap();

    assert!(out.exists());
}

/// The output file size is greater than the fixed header size (36 bytes) when
/// metadata is present.
#[test]
fn test_writer_output_file_size_nonzero() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("sized.gguf");

    let mut writer = GgufWriter::new();
    writer.add_metadata("general.architecture", MetadataValue::String("test".to_string()));
    writer.write_to_file(&out).unwrap();

    let len = fs::metadata(&out).unwrap().len();
    // Header (36 bytes) + KV for "general.architecture" + alignment padding
    assert!(len >= 36, "file must be at least as large as the GGUF header");
}

// ── TensorEntry and TensorDType ──────────────────────────────────────────────

/// TensorEntry stores and exposes the name field.
#[test]
fn test_tensor_entry_name_accessible() {
    let entry =
        TensorEntry::new("my.tensor.weight".to_string(), vec![4, 4], TensorDType::F16, vec![0; 32]);
    assert_eq!(entry.name, "my.tensor.weight");
}

/// TensorEntry stores and exposes the shape field.
#[test]
fn test_tensor_entry_shape_accessible() {
    let entry = TensorEntry::new("x".to_string(), vec![3, 5, 7], TensorDType::F32, vec![0; 420]);
    assert_eq!(entry.shape, vec![3, 5, 7]);
}

/// TensorEntry stores and exposes the data bytes.
#[test]
fn test_tensor_entry_data_length_matches() {
    let data = vec![42u8; 64];
    let entry = TensorEntry::new("t".to_string(), vec![32], TensorDType::F16, data.clone());
    assert_eq!(entry.data.len(), 64);
}

/// TensorDType::F16 and TensorDType::F32 have different GGUF type codes.
#[test]
fn test_tensor_dtype_codes_differ() {
    assert_ne!(
        TensorDType::F16.as_gguf_type(),
        TensorDType::F32.as_gguf_type(),
        "F16 and F32 must have distinct GGUF type codes"
    );
}

/// TensorDType::F16 has GGUF type code 1 (standard GGUF spec).
#[test]
fn test_tensor_dtype_f16_code_is_1() {
    assert_eq!(TensorDType::F16.as_gguf_type(), 1);
}

/// TensorDType::F32 has GGUF type code 0 (standard GGUF spec).
#[test]
fn test_tensor_dtype_f32_code_is_0() {
    assert_eq!(TensorDType::F32.as_gguf_type(), 0);
}

/// TensorDType::F16 element size is 2 bytes.
#[test]
fn test_tensor_dtype_f16_element_size_is_2() {
    assert_eq!(TensorDType::F16.element_size(), 2);
}

/// TensorDType::F32 element size is 4 bytes.
#[test]
fn test_tensor_dtype_f32_element_size_is_4() {
    assert_eq!(TensorDType::F32.element_size(), 4);
}

// ── LayerNorm detection: additional patterns ─────────────────────────────────

/// GGUF-style blk prefix: `blk.0.attn_norm.weight` must be detected as LayerNorm.
#[test]
fn test_layernorm_gguf_blk_style_attn_norm() {
    assert!(
        is_layernorm_tensor("blk.0.attn_norm.weight"),
        "blk-style attn_norm must be a LayerNorm tensor"
    );
}

/// GGUF-style blk prefix: `blk.5.ffn_norm.weight` must be detected as LayerNorm.
#[test]
fn test_layernorm_gguf_blk_style_ffn_norm() {
    assert!(
        is_layernorm_tensor("blk.5.ffn_norm.weight"),
        "blk-style ffn_norm must be a LayerNorm tensor"
    );
}

/// `rms_norm.weight` suffix pattern must be detected as LayerNorm.
#[test]
fn test_layernorm_rms_norm_variant() {
    assert!(
        is_layernorm_tensor("blk.3.rms_norm.weight"),
        ".rms_norm.weight suffix must be a LayerNorm tensor"
    );
}

/// `attention_norm.weight` (LLaMA-style) must be detected as LayerNorm.
#[test]
fn test_layernorm_attention_norm_variant() {
    assert!(
        is_layernorm_tensor("blk.2.attention_norm.weight"),
        ".attention_norm.weight suffix must be a LayerNorm tensor"
    );
}

/// `.norm.weight` generic suffix must be detected as LayerNorm.
#[test]
fn test_layernorm_norm_weight_suffix() {
    assert!(
        is_layernorm_tensor("model.layers.0.norm.weight"),
        ".norm.weight suffix must be detected as LayerNorm"
    );
}

/// `token_embd.weight` is not a LayerNorm tensor.
#[test]
fn test_non_layernorm_token_embd() {
    assert!(
        !is_layernorm_tensor("token_embd.weight"),
        "token_embd.weight must NOT be a LayerNorm tensor"
    );
}

/// `blk.0.attn_output.weight` (a projection) is not a LayerNorm tensor.
#[test]
fn test_non_layernorm_attn_output_projection() {
    assert!(
        !is_layernorm_tensor("blk.0.attn_output.weight"),
        "attention projection weights must NOT be LayerNorm"
    );
}

// ── count_layernorm_tensors ───────────────────────────────────────────────────

/// An empty slice produces a count of zero.
#[test]
fn test_count_empty_slice_is_zero() {
    assert_eq!(count_layernorm_tensors([].iter().copied()), 0);
}

/// When every name in the list is a LayerNorm tensor, count equals the list length.
#[test]
fn test_count_all_match_equals_len() {
    let names = [
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.1.attn_norm.weight",
        "blk.1.ffn_norm.weight",
    ];
    assert_eq!(count_layernorm_tensors(names.iter().copied()), names.len());
}

/// A mixed list counts only the LayerNorm names.
#[test]
fn test_count_partial_match() {
    let names = [
        "blk.0.attn_norm.weight", // LN
        "blk.0.attn_q.weight",    // projection, not LN
        "blk.0.ffn_norm.weight",  // LN
        "token_embd.weight",      // embedding, not LN
    ];
    assert_eq!(count_layernorm_tensors(names.iter().copied()), 2);
}

// ── CLI binary tests ─────────────────────────────────────────────────────────

/// `st2gguf --help` exits with status 0.
#[test]
fn test_cli_help_exits_zero() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_st2gguf"))
        .arg("--help")
        .output()
        .expect("failed to run st2gguf binary");
    assert!(output.status.success(), "--help must exit 0, got {:?}", output.status.code());
}

/// Running with a nonexistent --input path exits with a non-zero status.
#[test]
fn test_cli_nonexistent_input_file_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("out.gguf");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_st2gguf"))
        .args([
            "--input",
            "/definitely/does/not/exist/model.safetensors",
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run st2gguf binary");

    assert!(!output.status.success(), "nonexistent input path must cause a non-zero exit code");
}

// ── Property tests ────────────────────────────────────────────────────────────

proptest! {
    /// blk.{n}.ffn_norm.weight always matches the LayerNorm predicate, regardless of layer index.
    #[test]
    fn prop_blk_style_ffn_norm_always_layernorm(layer in 0usize..256) {
        let name = format!("blk.{layer}.ffn_norm.weight");
        prop_assert!(
            is_layernorm_tensor(&name),
            "{name} must always be detected as a LayerNorm tensor"
        );
    }

    /// blk.{n}.attn_norm.weight always matches regardless of layer index.
    #[test]
    fn prop_blk_style_attn_norm_always_layernorm(layer in 0usize..256) {
        let name = format!("blk.{layer}.attn_norm.weight");
        prop_assert!(
            is_layernorm_tensor(&name),
            "{name} must always be detected as a LayerNorm tensor"
        );
    }

    /// blk.{n}.attn_q.weight is never a LayerNorm tensor.
    #[test]
    fn prop_projection_weight_never_layernorm(layer in 0usize..256) {
        let name = format!("blk.{layer}.attn_q.weight");
        prop_assert!(
            !is_layernorm_tensor(&name),
            "{name} is a projection weight and must NOT match LayerNorm"
        );
    }
}
