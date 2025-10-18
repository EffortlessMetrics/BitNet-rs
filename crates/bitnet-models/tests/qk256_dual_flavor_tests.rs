//! QK256 dual-flavor detection and storage tests
//!
//! Tests for the QK256 I2_S format detection and storage in the i2s_qk256 HashMap.

use bitnet_common::Device;
use bitnet_models::gguf_simple::{GgufLoadResult, load_gguf_full};
use bitnet_models::quant::i2s_qk256::I2SQk256NoScale;
use std::collections::HashMap;
use std::io::Seek;
use tempfile::NamedTempFile;

/// Helper to create a minimal GGUF file with I2_S tensors
fn create_test_gguf_with_i2s(
    tensor_name: &str,
    shape: &[usize],
    data: Vec<u8>,
    gguf_type: u32,
) -> NamedTempFile {
    use std::io::Write;

    let mut file = NamedTempFile::new().unwrap();

    // Write minimal GGUF header
    // Magic: 0x46554747 ('GGUF')
    file.write_all(&0x47475546u32.to_le_bytes()).unwrap();
    // Version: 3
    file.write_all(&3u32.to_le_bytes()).unwrap();
    // Tensor count: 1
    file.write_all(&1u64.to_le_bytes()).unwrap();
    // Metadata KV count: 2
    file.write_all(&2u64.to_le_bytes()).unwrap();

    // Metadata 1: vocab_size
    // Key length
    file.write_all(&20u64.to_le_bytes()).unwrap();
    // Key: "tokenizer.ggml.tokens"
    file.write_all(b"tokenizer.ggml.tokens").unwrap();
    // Type: array (9)
    file.write_all(&9u32.to_le_bytes()).unwrap();
    // Array type: string (8)
    file.write_all(&8u32.to_le_bytes()).unwrap();
    // Array length: 100
    file.write_all(&100u64.to_le_bytes()).unwrap();
    // Write 100 empty strings
    for _ in 0..100 {
        file.write_all(&0u64.to_le_bytes()).unwrap(); // String length 0
    }

    // Metadata 2: hidden_size
    // Key length
    file.write_all(&30u64.to_le_bytes()).unwrap();
    // Key: "bitnet-b1.58.embedding_length"
    file.write_all(b"bitnet-b1.58.embedding_length").unwrap();
    file.write_all(&0u8.to_le_bytes()).unwrap(); // Padding
    // Type: u32 (4)
    file.write_all(&4u32.to_le_bytes()).unwrap();
    // Value: 512
    file.write_all(&512u32.to_le_bytes()).unwrap();

    // Tensor info
    // Name length
    file.write_all(&(tensor_name.len() as u64).to_le_bytes()).unwrap();
    // Name
    file.write_all(tensor_name.as_bytes()).unwrap();
    // Padding to 8-byte alignment
    let name_padding = (8 - (tensor_name.len() % 8)) % 8;
    file.write_all(&vec![0u8; name_padding]).unwrap();

    // Dimensions
    file.write_all(&(shape.len() as u32).to_le_bytes()).unwrap();
    for &dim in shape {
        file.write_all(&(dim as u64).to_le_bytes()).unwrap();
    }

    // Type (I2_S = 26)
    file.write_all(&gguf_type.to_le_bytes()).unwrap();

    // Offset (will be calculated)
    let header_end = file.stream_position().unwrap();
    let offset = 0u64; // Placeholder
    file.write_all(&offset.to_le_bytes()).unwrap();

    // Alignment padding to 32-byte boundary
    let pos = file.stream_position().unwrap();
    let padding = ((32 - (pos % 32)) % 32) as usize;
    file.write_all(&vec![0u8; padding]).unwrap();

    // Write tensor data
    let data_start = file.stream_position().unwrap();
    file.write_all(&data).unwrap();

    // Update offset
    file.seek(std::io::SeekFrom::Start(header_end)).unwrap();
    file.write_all(&data_start.to_le_bytes()).unwrap();

    file.flush().unwrap();
    file
}

#[test]
fn test_qk256_detection_by_size() {
    // Test QK256 format detection based on tensor size
    // Shape: [4, 256] → 1024 elements
    // QK256 expects: ceil(256/256) = 1 block per row × 4 rows = 4 blocks × 64 bytes = 256 bytes

    let rows: usize = 4;
    let cols: usize = 256;
    let blocks_per_row = cols.div_ceil(256); // 1
    let expected_bytes = rows * blocks_per_row * 64; // 256 bytes

    let data = vec![0xAAu8; expected_bytes]; // Fill with test pattern

    let file = create_test_gguf_with_i2s("test.weight", &[rows, cols], data, 26);

    let result =
        load_gguf_full(file.path(), Device::Cpu, bitnet_models::GGUFLoaderConfig::default())
            .unwrap();

    // Verify QK256 tensor was detected and stored in i2s_qk256 map
    assert_eq!(result.i2s_qk256.len(), 1, "Should have one QK256 tensor");
    assert!(result.i2s_qk256.contains_key("test.weight"), "Should contain test.weight");

    // Verify it's NOT in the regular tensors map
    assert!(
        !result.tensors.contains_key("test.weight"),
        "QK256 tensor should not be in regular tensors map"
    );

    // Verify the QK256 structure
    let qk256 = result.i2s_qk256.get("test.weight").unwrap();
    assert_eq!(qk256.rows, rows);
    assert_eq!(qk256.cols, cols);
    assert_eq!(qk256.row_stride_bytes, 64); // 1 block × 64 bytes
}

#[test]
fn test_bitnet32_still_uses_fp_path() {
    // Test that BitNet-32 I2_S tensors still go through FP dequantization path
    // Shape: [2, 64] → 128 elements
    // BitNet-32 expects: ceil(128/32) = 4 blocks × 10 bytes = 40 bytes

    let rows: usize = 2;
    let cols: usize = 64;
    let blocks = (rows * cols).div_ceil(32); // 4 blocks
    let expected_bytes = blocks * 10; // 40 bytes (inline F16 scales)

    let mut data = Vec::with_capacity(expected_bytes);
    // Create proper BitNet-32 I2_S data (8 bytes packed + 2 bytes F16 scale per block)
    for _ in 0..blocks {
        data.extend_from_slice(&[0xAAu8; 8]); // Packed data
        data.extend_from_slice(&[0x00, 0x3C]); // F16 scale (1.0)
    }

    let file = create_test_gguf_with_i2s("test.weight", &[rows, cols], data, 26);

    let result =
        load_gguf_full(file.path(), Device::Cpu, bitnet_models::GGUFLoaderConfig::default())
            .unwrap();

    // Verify BitNet-32 tensor was dequantized and stored in tensors map
    assert_eq!(result.i2s_qk256.len(), 0, "Should have no QK256 tensors");
    assert!(
        result.tensors.contains_key("test.weight"),
        "Should contain test.weight in regular tensors"
    );

    // Verify the tensor was dequantized to F32
    let tensor = result.tensors.get("test.weight").unwrap();
    assert_eq!(tensor.shape().dims(), &[rows, cols]);
}

#[test]
fn test_qk256_with_non_multiple_cols() {
    // Test QK256 with column count not a multiple of 256
    // Shape: [2, 300] → 600 elements
    // QK256 expects: ceil(300/256) = 2 blocks per row × 2 rows = 4 blocks × 64 bytes = 256 bytes

    let rows: usize = 2;
    let cols: usize = 300;
    let blocks_per_row = cols.div_ceil(256); // 2
    let expected_bytes = rows * blocks_per_row * 64; // 256 bytes

    let data = vec![0x55u8; expected_bytes]; // Different test pattern

    let file = create_test_gguf_with_i2s("test.weight", &[rows, cols], data, 26);

    let result =
        load_gguf_full(file.path(), Device::Cpu, bitnet_models::GGUFLoaderConfig::default())
            .unwrap();

    // Verify QK256 tensor was detected
    assert_eq!(result.i2s_qk256.len(), 1);
    let qk256 = result.i2s_qk256.get("test.weight").unwrap();
    assert_eq!(qk256.rows, rows);
    assert_eq!(qk256.cols, cols);
    assert_eq!(qk256.row_stride_bytes, 128); // 2 blocks × 64 bytes
}

#[test]
fn test_qk256_i2s_qk256_noscale_creation() {
    // Test I2SQk256NoScale creation directly
    let rows: usize = 3;
    let cols: usize = 512;
    let blocks_per_row = cols.div_ceil(256); // 2
    let row_stride_bytes = blocks_per_row * 64; // 128
    let total_bytes = rows * row_stride_bytes; // 384

    let qs = vec![0xFFu8; total_bytes];

    let qk256 = I2SQk256NoScale::new(rows, cols, qs).unwrap();

    assert_eq!(qk256.rows, rows);
    assert_eq!(qk256.cols, cols);
    assert_eq!(qk256.row_stride_bytes, row_stride_bytes);
    assert_eq!(qk256.qs.len(), total_bytes);
}

#[test]
fn test_qk256_size_mismatch_error() {
    // Test that incorrect size triggers error
    let rows: usize = 2;
    let cols: usize = 256;
    let wrong_size: usize = 100; // Way too small

    let qs = vec![0u8; wrong_size];

    let result = I2SQk256NoScale::new(rows, cols, qs);
    assert!(result.is_err(), "Should fail with size mismatch");

    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("data size mismatch"), "Error should mention size mismatch");
}

#[test]
fn test_gguf_load_result_structure() {
    // Test that GgufLoadResult has the expected structure
    let config = bitnet_common::BitNetConfig::default();
    let tensors = HashMap::new();
    let i2s_qk256 = HashMap::new();

    let result = GgufLoadResult { config: config.clone(), tensors, i2s_qk256 };

    assert_eq!(result.config.model.vocab_size, config.model.vocab_size);
    assert_eq!(result.tensors.len(), 0);
    assert_eq!(result.i2s_qk256.len(), 0);
}
