//! QK256 dual-flavor detection and storage tests
//!
//! Tests for the QK256 I2_S format detection and storage in the i2s_qk256 HashMap.
//!
//! ## Fixture Generation Approach
//!
//! Tests use **in-memory fixture generation** via `helpers::qk256_fixtures` for fast,
//! deterministic test execution. For CI/CD environments preferring disk-based fixtures,
//! persistent fixtures are available in `ci/fixtures/qk256/` (see `qk256_fixture_loader_tests.rs`
//! for disk-based loading examples).

use bitnet_common::Device;
use bitnet_models::gguf_simple::{GgufLoadResult, load_gguf_full};
use bitnet_models::quant::i2s_qk256::I2SQk256NoScale;
use std::collections::HashMap;
use std::io::Seek;
use tempfile::NamedTempFile;

mod helpers;

/// Helper to create a minimal GGUF file with I2_S tensors
/// Note: This function is intentionally unused - kept as reference for old manual fixture approach.
/// Tests now use helpers::qk256_fixtures generators instead.
#[allow(dead_code)]
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
#[cfg_attr(not(feature = "fixtures"), ignore = "Requires real or generated GGUF fixtures")]
fn test_qk256_detection_by_size() {
    // Test QK256 format detection based on tensor size using fixture generator
    // Shape: [4, 256] → 1024 elements
    // QK256 expects: ceil(256/256) = 1 block per row × 4 rows = 4 blocks × 64 bytes = 256 bytes

    use std::io::Write;

    let rows: usize = 4;
    let cols: usize = 256;

    // Generate fixture using helpers::qk256_fixtures with deterministic seed 42
    let fixture_bytes = helpers::qk256_fixtures::generate_qk256_4x256(42);

    // Write to temp file for GGUF loading
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&fixture_bytes).unwrap();
    file.flush().unwrap();

    let result = match load_gguf_full(
        file.path(),
        Device::Cpu,
        bitnet_models::GGUFLoaderConfig::default(),
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("\n❌ GGUF load error: {:?}", e);
            eprintln!("File path: {:?}", file.path());
            eprintln!("File exists: {}", file.path().exists());
            eprintln!("Fixture size: {} bytes", fixture_bytes.len());
            panic!("Failed to load GGUF fixture for test_qk256_detection_by_size: {}", e);
        }
    };

    // Verify QK256 tensor was detected and stored in i2s_qk256 map
    assert_eq!(result.i2s_qk256.len(), 1, "Should have one QK256 tensor");
    assert!(
        result.i2s_qk256.contains_key("tok_embeddings.weight"),
        "Should contain tok_embeddings.weight (canonical GGUF tensor name)"
    );

    // Verify it's NOT in the regular tensors map
    assert!(
        !result.tensors.contains_key("tok_embeddings.weight"),
        "QK256 tensor should not be in regular tensors map"
    );

    // Verify the QK256 structure
    let qk256 = result.i2s_qk256.get("tok_embeddings.weight").unwrap();
    assert_eq!(qk256.rows, rows, "Rows should match fixture");
    assert_eq!(qk256.cols, cols, "Cols should match fixture");
    assert_eq!(qk256.row_stride_bytes, 64, "Row stride should be 1 block × 64 bytes");
}

#[test]
#[cfg_attr(not(feature = "fixtures"), ignore = "Requires real or generated GGUF fixtures")]
fn test_bitnet32_still_uses_fp_path() {
    // Test that BitNet-32 I2_S tensors use FP dequantization path (not QK256) using fixture generator
    // Shape: [2, 64] → 128 elements
    // BitNet-32 expects: ceil(128/32) = 4 blocks × 10 bytes = 40 bytes

    use std::io::Write;

    let _rows: usize = 2;
    let _cols: usize = 64;

    // Generate fixture using helpers::qk256_fixtures with deterministic seed 43
    let fixture_bytes = helpers::qk256_fixtures::generate_bitnet32_2x64(43);

    // Write to temp file for GGUF loading
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&fixture_bytes).unwrap();
    file.flush().unwrap();

    let result =
        load_gguf_full(file.path(), Device::Cpu, bitnet_models::GGUFLoaderConfig::default())
            .unwrap();

    // Verify BitNet-32 tensor was dequantized and stored in tensors map (NOT in i2s_qk256)
    assert_eq!(result.i2s_qk256.len(), 0, "Should have no QK256 tensors");

    // The loader normalizes tensor names: "tok_embeddings.weight" -> "token_embd.weight"
    assert!(
        result.tensors.contains_key("token_embd.weight"),
        "Should contain token_embd.weight (normalized from tok_embeddings.weight)"
    );

    // Verify the tensor was dequantized to F32
    // Note: The loader normalizes tensor shapes to match config metadata (vocab=1000, hidden=512)
    // so the final shape may differ from the fixture's raw tensor shape (2×64)
    let tensor = result.tensors.get("token_embd.weight").unwrap();
    let vocab_size = result.config.model.vocab_size;
    let hidden_size = result.config.model.hidden_size;
    assert_eq!(
        tensor.shape().dims(),
        &[vocab_size, hidden_size],
        "Tensor shape should match config metadata after normalization"
    );
}

#[test]
#[cfg_attr(not(feature = "fixtures"), ignore = "Requires real or generated GGUF fixtures")]
fn test_qk256_with_non_multiple_cols() {
    // Test QK256 with column count not a multiple of 256 using fixture generator
    // Shape: [3, 300] → 900 elements (updated to match fixture)
    // QK256 expects: ceil(300/256) = 2 blocks per row × 3 rows = 6 blocks × 64 bytes = 384 bytes

    use std::io::Write;

    let rows: usize = 3;
    let cols: usize = 300;
    let _blocks_per_row = cols.div_ceil(256); // 2 (used in calculation comment, not in code)

    // Generate fixture using helpers::qk256_fixtures with deterministic seed 44
    let fixture_bytes = helpers::qk256_fixtures::generate_qk256_3x300(44);

    // Write to temp file for GGUF loading
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&fixture_bytes).unwrap();
    file.flush().unwrap();

    let result =
        load_gguf_full(file.path(), Device::Cpu, bitnet_models::GGUFLoaderConfig::default())
            .unwrap();

    // Verify QK256 tensor was detected
    assert_eq!(result.i2s_qk256.len(), 1, "Should have one QK256 tensor");
    let qk256 = result.i2s_qk256.get("tok_embeddings.weight").unwrap();
    assert_eq!(qk256.rows, rows, "Rows should match fixture");
    assert_eq!(qk256.cols, cols, "Cols should match fixture");
    assert_eq!(qk256.row_stride_bytes, 128, "Row stride should be 2 blocks × 64 bytes");
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
    // For 10×256 matrix: expected = 10 rows × 1 block × 64 bytes = 640 bytes
    // I2SQk256NoScale::new has TOLERANCE = 128 bytes, so we need diff > 128
    let rows: usize = 10;
    let cols: usize = 256;
    let expected_size = rows * 64; // 640 bytes
    let wrong_size: usize = expected_size - 200; // 200 bytes off = outside tolerance

    let qs = vec![0u8; wrong_size];

    let result = I2SQk256NoScale::new(rows, cols, qs);
    assert!(result.is_err(), "Should fail with size mismatch (diff=200B > tolerance=128B)");

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

#[test]
#[cfg(feature = "fixtures")]
fn test_dump_fixture_for_debug() {
    // Generate all three fixtures and write to files
    let fixtures = [
        ("qk256_4x256", helpers::qk256_fixtures::generate_qk256_4x256(42)),
        ("bitnet32_2x64", helpers::qk256_fixtures::generate_bitnet32_2x64(43)),
        ("qk256_3x300", helpers::qk256_fixtures::generate_qk256_3x300(44)),
    ];

    for (name, fixture_bytes) in &fixtures {
        let path = format!("/tmp/test_{}.gguf", name);
        std::fs::write(&path, fixture_bytes).unwrap();
        eprintln!("✓ Wrote {} ({} bytes) to {}", name, fixture_bytes.len(), path);
    }

    eprintln!(
        "\nInspect with: cargo run -p bitnet-cli --features cpu,full-cli -- compat-check /tmp/test_<name>.gguf"
    );
}

#[test]
#[cfg(feature = "fixtures")]
fn test_load_fixture_from_fixed_path() {
    use std::path::Path;

    // Fixture already written by test_dump_fixture_for_debug
    let path = Path::new("/tmp/test_generated_fixture.gguf");
    assert!(path.exists(), "Fixture should exist");

    let result = load_gguf_full(path, Device::Cpu, bitnet_models::GGUFLoaderConfig::default());

    match &result {
        Ok(r) => {
            eprintln!("✓ Loaded from fixed path!");
            eprintln!("  i2s_qk256: {}", r.i2s_qk256.len());
            eprintln!("  tensors: {}", r.tensors.len());
        }
        Err(e) => {
            eprintln!("✗ Failed: {:?}", e);
        }
    }

    result.expect("Should load fixture from fixed path");
}
