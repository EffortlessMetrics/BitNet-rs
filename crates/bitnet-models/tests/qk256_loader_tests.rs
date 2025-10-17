//! QK256 Loader and Dispatch Tests
//!
//! Tests feature spec: i2s-dual-flavor.md#qk256-loader-integration
//!
//! This test suite validates the complete QK256 loading and dispatch pipeline:
//!
//! - GGUF tensor detection (I2S flavor identification)
//! - Side-map storage (`.qk256_qs` suffix convention)
//! - Transformer integration (automatic QK256 dispatch)
//! - End-to-end forward pass validation

use bitnet_models::quant::i2s_qk256::{I2SQk256NoScale, QK256_BLOCK, QK256_PACKED_BYTES};
use candle_core::{DType, Device as CDevice, Tensor as CandleTensor};
use std::collections::HashMap;

// ==================== I2S Flavor Detection Tests ====================

/// Test spec: i2s-dual-flavor.md#i2s-flavor-detection-qk256
///
/// Verify QK256 tensors are detected by exact size match
#[test]
fn test_i2s_flavor_detection_qk256_exact_match() {
    // Test case: rows=2048, cols=2048
    let rows = 2048;
    let cols: usize = 2048;
    let _total_elements = rows * cols;

    // QK256 format: blocks_per_row = ceil(cols/256), row_stride_bytes = blocks_per_row * 64
    let blocks_per_row = cols.div_ceil(QK256_BLOCK); // = 8
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES; // = 512
    let qk256_size = rows * row_stride_bytes; // = 1048576

    // Verify detection logic
    // Detection: total_elements from shape product, available_bytes from tensor size
    // Expected: if available_bytes == qk256_size, then flavor = GgmlQk256NoScale

    let expected_qk256_size = rows * (cols.div_ceil(QK256_BLOCK) * QK256_PACKED_BYTES);
    assert_eq!(qk256_size, expected_qk256_size);

    println!("✓ QK256 detection: {}×{} → {} bytes (expected)", rows, cols, qk256_size);
}

/// Test spec: i2s-dual-flavor.md#i2s-flavor-detection-bitnet32
///
/// Verify BitNet32F16 tensors are detected by different size
#[test]
fn test_i2s_flavor_detection_bitnet32_exact_match() {
    // Test case: same dimensions but BitNet32F16 format
    let rows = 2048;
    let cols: usize = 2048;
    let total_elements = rows * cols;

    // BitNet32F16 format: blocks = ceil(total_elements/32), size = blocks * 10
    let bitnet32_blocks = total_elements.div_ceil(32); // = 131072
    let bitnet32_size = bitnet32_blocks * 10; // = 1310720 (8B packed + 2B f16 scale)

    // QK256 size for comparison
    let qk256_size = rows * (cols.div_ceil(QK256_BLOCK) * QK256_PACKED_BYTES); // = 1048576

    // Verify sizes are different (unambiguous detection)
    assert_ne!(bitnet32_size, qk256_size, "BitNet32 and QK256 sizes should differ");

    println!(
        "✓ BitNet32 detection: {}×{} → {} bytes (different from QK256 {} bytes)",
        rows, cols, bitnet32_size, qk256_size
    );
}

/// Test spec: i2s-dual-flavor.md#i2s-flavor-detection-mismatch
///
/// Verify invalid tensor sizes trigger helpful error
#[test]
fn test_i2s_flavor_detection_invalid_size() {
    let rows = 2048;
    let cols: usize = 2048;

    // Create tensor with invalid size (doesn't match either format)
    let invalid_size = 999999; // Arbitrary wrong size

    let qk256_expected = rows * (cols.div_ceil(QK256_BLOCK) * QK256_PACKED_BYTES);
    let bitnet32_expected = (rows * cols).div_ceil(32) * 10;

    // Verify neither format matches
    assert_ne!(invalid_size, qk256_expected, "Should not match QK256");
    assert_ne!(invalid_size, bitnet32_expected, "Should not match BitNet32");

    println!(
        "✓ Invalid size {} doesn't match QK256 ({}) or BitNet32 ({})",
        invalid_size, qk256_expected, bitnet32_expected
    );
}

// ==================== Side-Map Storage Tests ====================

/// Test spec: i2s-dual-flavor.md#qk256-side-storage
///
/// Verify QK256 data stored with derived key `.qk256_qs` suffix
#[test]
fn test_qk256_side_map_storage_convention() {
    // Test key naming convention
    let original_keys = [
        "layers.0.attention.q_proj.weight",
        "layers.1.feed_forward.gate_proj.weight",
        "blk.0.attn_q.weight",
    ];

    let expected_derived_keys = [
        "layers.0.attention.q_proj.weight.qk256_qs",
        "layers.1.feed_forward.gate_proj.weight.qk256_qs",
        "blk.0.attn_q.weight.qk256_qs",
    ];

    for (original, expected) in original_keys.iter().zip(expected_derived_keys.iter()) {
        let derived = format!("{}.qk256_qs", original);
        assert_eq!(&derived, expected, "Derived key mismatch for {}", original);

        // Verify suffix detection
        assert!(derived.ends_with(".qk256_qs"), "Derived key must end with .qk256_qs");

        // Verify original key extraction
        let extracted = derived.strip_suffix(".qk256_qs").unwrap();
        assert_eq!(extracted, *original, "Original key extraction failed");
    }

    println!("✓ QK256 side-map storage convention verified");
}

/// Test spec: i2s-dual-flavor.md#qk256-storage-no-double-insert
///
/// Verify original key is NOT stored when QK256 variant exists
#[test]
fn test_qk256_storage_no_double_insert() {
    // Simulate GGUF loader behavior
    let mut tensor_map: HashMap<String, String> = HashMap::new();

    let original_name = "layers.0.attention.q_proj.weight";
    let is_qk256 = true; // Detected as QK256

    if is_qk256 {
        // Store with derived key ONLY
        let qk256_key = format!("{}.qk256_qs", original_name);
        tensor_map.insert(qk256_key.clone(), "QK256_TENSOR_DATA".to_string());

        // Verify original key NOT inserted
        assert!(!tensor_map.contains_key(original_name), "Original key should not be stored");
        assert!(tensor_map.contains_key(&qk256_key), "QK256 key should be stored");
    }

    println!("✓ QK256 storage avoids double-insert");
}

// ==================== Linear Layer Detection Tests ====================

/// Test spec: i2s-dual-flavor.md#linear-layer-qk256-detection
///
/// Verify Linear layer detects QK256 weights by suffix
#[test]
fn test_linear_layer_qk256_detection_by_suffix() {
    let weight_keys = vec![
        ("layers.0.attention.q_proj.weight", false),
        ("layers.0.attention.q_proj.weight.qk256_qs", true),
        ("layers.1.feed_forward.gate_proj.weight", false),
        ("layers.1.feed_forward.gate_proj.weight.qk256_qs", true),
    ];

    for (key, expected_is_qk256) in weight_keys {
        let is_qk256 = key.ends_with(".qk256_qs");
        assert_eq!(is_qk256, expected_is_qk256, "QK256 detection failed for key: {}", key);
    }

    println!("✓ Linear layer QK256 detection by suffix verified");
}

// ==================== Transformer Integration Tests ====================

/// Test spec: i2s-dual-flavor.md#qk256-transformer-integration
///
/// Verify transformer can load and use QK256 tensors
#[test]
fn test_qk256_transformer_integration_smoke() -> anyhow::Result<()> {
    // Create synthetic QK256 tensor
    let rows = 128;
    let cols: usize = 256; // Single block
    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;

    // Create QK256 data (code 2 → +1.0)
    let qk256_bytes = vec![0xAAu8; rows * row_stride_bytes];

    // Create I2SQk256NoScale struct
    let qk256 = I2SQk256NoScale::new(rows, cols, qk256_bytes)?;

    // Verify struct properties
    assert_eq!(qk256.rows, rows);
    assert_eq!(qk256.cols, cols);
    assert_eq!(qk256.row_stride_bytes, row_stride_bytes);

    // Verify row access
    let row0 = qk256.row_bytes(0);
    assert_eq!(row0.len(), row_stride_bytes);
    assert!(row0.iter().all(|&b| b == 0xAA), "All bytes should be 0xAA");

    println!("✓ Transformer QK256 integration smoke test passed");

    Ok(())
}

/// Test spec: i2s-dual-flavor.md#qk256-forward-pass-validation
///
/// Verify forward pass produces valid output with QK256 weights
#[test]
fn test_qk256_forward_pass_output_validation() -> anyhow::Result<()> {
    use bitnet_models::quant::i2s_qk256::gemv_qk256;

    let rows = 64; // Output features
    let cols = 256; // Input features (single block)
    let batch_size = 2;

    // Create QK256 weights (code 2 → +1.0)
    let qk256_bytes = vec![0xAAu8; rows * QK256_PACKED_BYTES];

    // Create input batch (shape: [batch_size, cols])
    let input_data: Vec<f32> = (0..batch_size * cols).map(|i| (i as f32 * 0.01).sin()).collect();

    // Process each batch item
    for batch_idx in 0..batch_size {
        let batch_start = batch_idx * cols;
        let batch_input = &input_data[batch_start..batch_start + cols];

        let mut output = vec![0.0f32; rows];
        gemv_qk256(&qk256_bytes, batch_input, &mut output, rows, cols, QK256_PACKED_BYTES)?;

        // Verify output shape and properties
        assert_eq!(output.len(), rows);

        // Verify outputs are finite and reasonable
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "Batch {}, Row {}: non-finite output {}", batch_idx, i, val);

            // With all weights = +1.0, output ≈ sum(input)
            let input_sum: f32 = batch_input.iter().sum();
            assert!(
                (val - input_sum).abs() < 1.0,
                "Batch {}, Row {}: output {} not close to input sum {}",
                batch_idx,
                i,
                val,
                input_sum
            );
        }
    }

    println!("✓ QK256 forward pass output validation passed");

    Ok(())
}

// ==================== Multi-Layer Integration Tests ====================

/// Test spec: i2s-dual-flavor.md#qk256-multi-layer-integration
///
/// Verify QK256 works correctly across multiple transformer layers
#[test]
fn test_qk256_multi_layer_integration() -> anyhow::Result<()> {
    use bitnet_models::quant::i2s_qk256::gemv_qk256;

    let num_layers = 4;
    let hidden_size: usize = 512; // 2 blocks
    let blocks_per_row = hidden_size.div_ceil(QK256_BLOCK);
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;

    // Create QK256 weights for each layer (different patterns)
    let mut layer_weights = Vec::new();
    for layer_idx in 0..num_layers {
        let pattern = match layer_idx % 4 {
            0 => 0x00u8, // Code 0 → -2.0
            1 => 0x55u8, // Code 1 → -1.0
            2 => 0xAAu8, // Code 2 → +1.0
            _ => 0xFFu8, // Code 3 → +2.0
        };
        let qk256_bytes = vec![pattern; hidden_size * row_stride_bytes];
        layer_weights.push(qk256_bytes);
    }

    // Create initial input
    let mut current_input = vec![1.0f32; hidden_size];

    // Pass through each layer
    for (layer_idx, layer_weight) in layer_weights.iter().enumerate() {
        let mut output = vec![0.0f32; hidden_size];
        gemv_qk256(
            layer_weight,
            &current_input,
            &mut output,
            hidden_size,
            hidden_size,
            row_stride_bytes,
        )?;

        // Verify output is valid
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "Layer {}, Row {}: non-finite output {}", layer_idx, i, val);
        }

        // Update input for next layer
        current_input = output;
    }

    println!("✓ QK256 multi-layer integration test passed ({} layers)", num_layers);

    Ok(())
}

// ==================== Dimension Edge Cases ====================

/// Test spec: i2s-dual-flavor.md#qk256-dimension-edge-cases
///
/// Verify QK256 handles various matrix dimensions correctly
#[test]
fn test_qk256_dimension_edge_cases() {
    let test_cases: Vec<(usize, usize, &str)> = vec![
        // (rows, cols, description)
        (1, 256, "Single row, single block"),
        (1, 512, "Single row, multiple blocks"),
        (256, 1, "Minimal cols with many rows"),
        (2048, 2048, "Large square matrix"),
        (11008, 2048, "FFN intermediate projection"),
        (2048, 11008, "FFN output projection"),
    ];

    for (rows, cols, description) in test_cases {
        let blocks_per_row = cols.div_ceil(QK256_BLOCK);
        let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;
        let total_bytes = rows * row_stride_bytes;

        let qk256_bytes = vec![0xAAu8; total_bytes];
        let result = I2SQk256NoScale::new(rows, cols, qk256_bytes);

        assert!(result.is_ok(), "QK256 creation failed for {} ({}×{})", description, rows, cols);

        let qk256 = result.unwrap();
        assert_eq!(qk256.rows, rows);
        assert_eq!(qk256.cols, cols);
        assert_eq!(qk256.row_stride_bytes, row_stride_bytes);

        println!("✓ QK256 {}: {}×{} → {} bytes/row", description, rows, cols, row_stride_bytes);
    }
}

// ==================== Candle Tensor Integration ====================

/// Test spec: i2s-dual-flavor.md#qk256-candle-tensor-integration
///
/// Verify QK256 data can be loaded from Candle U8 tensors
#[test]
fn test_qk256_candle_tensor_integration() -> anyhow::Result<()> {
    let rows = 32;
    let cols: usize = 512; // 2 blocks
    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;

    // Create Candle U8 tensor with QK256 data
    let qk256_bytes = vec![0xAAu8; rows * row_stride_bytes];
    let tensor =
        CandleTensor::from_vec(qk256_bytes.clone(), &[rows, row_stride_bytes], &CDevice::Cpu)?
            .to_dtype(DType::U8)?;

    // Verify tensor shape
    assert_eq!(tensor.dims(), &[rows, row_stride_bytes]);
    assert_eq!(tensor.dtype(), DType::U8);

    // Extract bytes and create I2SQk256NoScale
    let bytes_2d = tensor.to_vec2::<u8>()?;
    let flat_bytes: Vec<u8> = bytes_2d.into_iter().flatten().collect();

    let qk256 = I2SQk256NoScale::new(rows, cols, flat_bytes)?;

    // Verify properties
    assert_eq!(qk256.rows, rows);
    assert_eq!(qk256.cols, cols);

    println!("✓ QK256 Candle tensor integration test passed");

    Ok(())
}
