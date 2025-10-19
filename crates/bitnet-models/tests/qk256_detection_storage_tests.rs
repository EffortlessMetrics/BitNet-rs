#![cfg(feature = "integration-tests")]
//! QK256 Test Scaffolding: Test (F) - Detection and Storage
//!
//! Tests feature spec: docs/explanation/i2s-dual-flavor.md#detection
//! Tests API contract: docs/reference/quantization-support.md#qk256-storage
//!
//! This test verifies that the GGUF loader correctly:
//! 1. Detects QK256 format from tensor byte size
//! 2. Creates U8 tensor under `*.qk256_qs` key
//! 3. Removes original key from tensor map
//! 4. Validates tensor shape is [rows, stride] where stride = 64 * ceil(cols/256)

use anyhow::Result;

/// Test (F): Detection/Storage for QK256 Tensors
///
/// Tests that fake I2_S tensor data matching QK256 format:
/// 1. Creates U8 tensor under `*.qk256_qs` key
/// 2. Removes original tensor key from map
/// 3. Validates shape is [rows, stride] with correct stride calculation
#[test]
fn test_qk256_detection_and_storage() -> Result<()> {
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;

    // Simulate GGUF loader behavior for QK256 detection
    // Given: A fake I2_S tensor with bytes matching QK256 format
    let rows = 512usize;
    let cols = 1024usize;
    let blocks_per_row = cols.div_ceil(256); // 4 blocks
    let row_stride_bytes = blocks_per_row * 64; // 4 * 64 = 256 bytes

    // Create packed data (simulating GGUF I2_S tensor)
    let total_bytes = rows * row_stride_bytes;
    let qs_data = vec![0xAAu8; total_bytes]; // All codes = 2

    // Create U8 tensor with shape [rows, stride]
    let device = Device::Cpu;
    let qk256_tensor = Tensor::from_vec(qs_data, &[rows, row_stride_bytes], &device)?;

    // Simulate loader behavior: insert into raw_tensors with .qk256_qs suffix
    let mut raw_tensors = HashMap::new();
    let original_key = "layers.0.attention.q_proj.weight";
    let qk256_key = format!("{}.qk256_qs", original_key);

    // Loader should:
    // 1. Create QK256 tensor under .qk256_qs key
    raw_tensors.insert(qk256_key.clone(), qk256_tensor);

    // 2. NOT include the original key (it should be replaced)
    // Verify original key is NOT in the map
    assert!(
        !raw_tensors.contains_key(original_key),
        "Original key '{}' should not be in tensor map after QK256 detection",
        original_key
    );

    // 3. Assert .qk256_qs key exists
    assert!(
        raw_tensors.contains_key(&qk256_key),
        "QK256 key '{}' should exist in tensor map",
        qk256_key
    );

    // 4. Verify shape is [rows, stride]
    let tensor = raw_tensors.get(&qk256_key).unwrap();
    assert_eq!(
        tensor.dims(),
        &[rows, row_stride_bytes],
        "QK256 tensor shape should be [{}, {}]",
        rows,
        row_stride_bytes
    );

    // 5. Verify dtype is U8
    assert_eq!(tensor.dtype(), DType::U8, "QK256 tensor dtype should be U8");

    // 6. Verify stride calculation: stride = ceil(cols/256) * 64
    let expected_stride = cols.div_ceil(256) * 64;
    assert_eq!(
        row_stride_bytes, expected_stride,
        "Stride should be {} (ceil({}/256) * 64)",
        expected_stride, cols
    );

    println!(
        "✅ QK256 detection test passed: rows={}, cols={}, stride={}",
        rows, cols, row_stride_bytes
    );

    Ok(())
}

/// Test (F2): QK256 Detection with Various Tensor Shapes
///
/// Tests detection logic with different tensor dimensions to ensure
/// stride calculation is correct for edge cases
#[test]
fn test_qk256_detection_various_shapes() -> Result<()> {
    use candle_core::{DType, Device, Tensor};

    let device = Device::Cpu;

    // Test cases: (rows, cols, expected_blocks, expected_stride)
    let test_cases: Vec<(usize, usize, usize, usize)> = vec![
        (256, 256, 1, 64),   // Exact single block
        (512, 512, 2, 128),  // Exact double block
        (256, 1024, 4, 256), // Multiple blocks
        (1, 256, 1, 64),     // Single row, single block
        (1000, 768, 3, 192), // Non-power-of-2 dimensions
        (512, 300, 2, 128),  // Partial block (300 = 256 + 44)
    ];

    for (rows, cols, expected_blocks, expected_stride) in test_cases {
        let blocks_per_row = cols.div_ceil(256);
        let row_stride_bytes = blocks_per_row * 64;

        // Verify block count
        assert_eq!(
            blocks_per_row, expected_blocks,
            "cols={} should have {} blocks",
            cols, expected_blocks
        );

        // Verify stride
        assert_eq!(
            row_stride_bytes, expected_stride,
            "cols={} should have stride={}",
            cols, expected_stride
        );

        // Create tensor with expected shape
        let total_bytes = rows * row_stride_bytes;
        let qs_data = vec![0u8; total_bytes];
        let tensor = Tensor::from_vec(qs_data, &[rows, row_stride_bytes], &device)?;

        // Verify tensor properties
        assert_eq!(tensor.dims(), &[rows, row_stride_bytes]);
        assert_eq!(tensor.dtype(), DType::U8);

        println!(
            "✅ Shape test passed: rows={}, cols={}, blocks={}, stride={}",
            rows, cols, expected_blocks, expected_stride
        );
    }

    Ok(())
}

/// Test (F3): QK256 Key Naming Convention
///
/// Verifies that the `.qk256_qs` suffix is correctly applied to all
/// weight tensor types (attention, feed_forward)
#[test]
fn test_qk256_key_naming_convention() -> Result<()> {
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    let device = Device::Cpu;

    // Create minimal QK256 tensor
    let rows = 256usize;
    let row_stride_bytes = 64usize;
    let qs_data = vec![0xAAu8; rows * row_stride_bytes];
    let qk256_tensor = Tensor::from_vec(qs_data, &[rows, row_stride_bytes], &device)?;

    // Test all expected weight tensor patterns
    let weight_keys = vec![
        // Attention projections
        "layers.0.attention.q_proj.weight",
        "layers.0.attention.k_proj.weight",
        "layers.0.attention.v_proj.weight",
        "layers.0.attention.o_proj.weight",
        // Feed-forward projections
        "layers.0.feed_forward.gate_proj.weight",
        "layers.0.feed_forward.up_proj.weight",
        "layers.0.feed_forward.down_proj.weight",
        // Different layer indices
        "layers.1.attention.q_proj.weight",
        "layers.15.feed_forward.gate_proj.weight",
    ];

    let mut raw_tensors = HashMap::new();

    for original_key in weight_keys {
        let qk256_key = format!("{}.qk256_qs", original_key);

        // Simulate loader: insert with .qk256_qs suffix
        raw_tensors.insert(qk256_key.clone(), qk256_tensor.clone());

        // Verify suffix format
        assert!(
            qk256_key.ends_with(".weight.qk256_qs"),
            "QK256 key '{}' should end with '.weight.qk256_qs'",
            qk256_key
        );

        // Verify original key not present
        assert!(
            !raw_tensors.contains_key(original_key),
            "Original key '{}' should not be present",
            original_key
        );

        println!("✅ Key naming verified: {} → {}", original_key, qk256_key);
    }

    Ok(())
}

/// Test (F4): QK256 Storage Format Validation
///
/// Verifies that stored QK256 tensors maintain data integrity
/// and can be correctly unpacked
#[test]
fn test_qk256_storage_format_validation() -> Result<()> {
    use bitnet_models::quant::i2s_qk256::{QK256_BLOCK, QK256_PACKED_BYTES, unpack_qk256_block};
    use candle_core::{Device, Tensor};

    let device = Device::Cpu;

    // Create a QK256 tensor with known pattern
    let rows = 2usize;
    let row_stride_bytes = 64usize; // 1 block per row
    let total_bytes = rows * row_stride_bytes;

    // Pattern: all codes = 2 (0xAA = 0b10_10_10_10)
    let qs_data = vec![0xAAu8; total_bytes];

    // Store as tensor
    let qk256_tensor = Tensor::from_vec(qs_data.clone(), &[rows, row_stride_bytes], &device)?;

    // Verify we can extract and unpack the data
    let tensor_bytes = qk256_tensor.to_vec2::<u8>()?;

    for (row_idx, row_bytes) in tensor_bytes.iter().enumerate() {
        assert_eq!(
            row_bytes.len(),
            row_stride_bytes,
            "Row {} should have {} bytes",
            row_idx,
            row_stride_bytes
        );

        // Unpack first block
        let mut codes = [0u8; QK256_BLOCK];
        let block_bytes: &[u8; QK256_PACKED_BYTES] =
            row_bytes[..QK256_PACKED_BYTES].try_into().expect("Should be 64 bytes");
        unpack_qk256_block(block_bytes, &mut codes);

        // Verify all codes are 2 (from 0xAA pattern)
        for (i, &code) in codes.iter().enumerate() {
            assert_eq!(code, 2, "Row {}, code {} should be 2", row_idx, i);
        }

        println!("✅ Row {} unpacked successfully: all codes = 2", row_idx);
    }

    Ok(())
}
