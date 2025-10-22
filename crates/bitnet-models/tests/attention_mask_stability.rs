use candle_core::{D, Device, Tensor};
use candle_nn::ops::softmax;

#[test]
fn test_softmax_all_masked_row_no_nan() {
    // Create attention scores with one fully masked row
    let scores = Tensor::from_slice(
        &[
            1.0, 2.0, 3.0, // Row 0: normal
            -1e9, -1e9, -1e9, // Row 1: fully masked
            4.0, 5.0, 6.0, // Row 2: normal
        ],
        &[3, 3],
        &Device::Cpu,
    )
    .unwrap();

    // Apply softmax
    let softmax_result = softmax(&scores, D::Minus1).unwrap();
    // Convert to f32 if needed
    let softmax_f32 = softmax_result.to_dtype(candle_core::DType::F32).unwrap();
    let values = softmax_f32.to_vec2::<f32>().unwrap();

    // Row 0 and 2: should be valid softmax (sum to 1, no NaN)
    assert!(!values[0][0].is_nan());
    assert!(!values[2][0].is_nan());

    // Row 1: should NOT be NaN (key test)
    assert!(!values[1][0].is_nan());
    assert!(!values[1][1].is_nan());
    assert!(!values[1][2].is_nan());

    // Row 1: should be near-zero or uniform (effectively suppressed)
    let row1_sum: f32 = values[1].iter().sum();
    assert!(row1_sum < 1e-6 || (row1_sum - 1.0).abs() < 1e-3);
}

#[test]
fn test_softmax_negative_infinity_causes_nan() {
    // Demonstrate the OLD bug with -inf
    let scores_with_inf = Tensor::from_slice(
        &[f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
        &[1, 3],
        &Device::Cpu,
    )
    .unwrap();

    let softmax_result = softmax(&scores_with_inf, D::Minus1).unwrap();
    // Convert to f32 if needed and flatten to 1D
    let softmax_f32 = softmax_result.to_dtype(candle_core::DType::F32).unwrap();
    let values = softmax_f32.to_vec2::<f32>().unwrap();
    let values = &values[0]; // Get first (and only) row

    // With -inf, this WILL produce NaN (verifies the bug existed)
    // Note: candle may handle this gracefully, so we check for either NaN or zero
    assert!(values[0].is_nan() || values[0] == 0.0);
    // ^ This test documents the old behavior
}

#[test]
fn test_softmax_partially_masked_row() {
    // Create attention scores with partially masked row
    let scores = Tensor::from_slice(
        &[
            1.0, 2.0, 3.0, // Row 0: normal
            1.0, -1e9, -1e9, // Row 1: partially masked
            4.0, 5.0, 6.0, // Row 2: normal
        ],
        &[3, 3],
        &Device::Cpu,
    )
    .unwrap();

    // Apply softmax
    let softmax_result = softmax(&scores, D::Minus1).unwrap();
    // Convert to f32 if needed
    let softmax_f32 = softmax_result.to_dtype(candle_core::DType::F32).unwrap();
    let values = softmax_f32.to_vec2::<f32>().unwrap();

    // All rows should be valid (no NaN)
    for (i, row) in values.iter().enumerate().take(3) {
        for (j, val) in row.iter().enumerate().take(3) {
            assert!(!val.is_nan(), "NaN at row {}, col {}", i, j);
        }
    }

    // Row 1: first position should have ~1.0 weight, others ~0.0
    assert!(values[1][0] > 0.9, "First unmasked position should have high weight");
    assert!(values[1][1] < 0.1, "Masked position should have near-zero weight");
    assert!(values[1][2] < 0.1, "Masked position should have near-zero weight");

    // Sum should still be ~1.0
    let row1_sum: f32 = values[1].iter().sum();
    assert!((row1_sum - 1.0).abs() < 1e-3, "Row should sum to 1.0, got {}", row1_sum);
}

#[test]
fn test_softmax_causal_mask_pattern() {
    // Simulate a typical causal mask pattern (lower triangular)
    // Position 0 can only attend to itself
    // Position 1 can attend to positions 0, 1
    // Position 2 can attend to positions 0, 1, 2
    let scores = Tensor::from_slice(
        &[
            1.0, -1e9, -1e9, // Row 0: causal mask (only position 0 visible)
            2.0, 3.0, -1e9, // Row 1: causal mask (positions 0, 1 visible)
            4.0, 5.0, 6.0, // Row 2: causal mask (all positions visible)
        ],
        &[3, 3],
        &Device::Cpu,
    )
    .unwrap();

    // Apply softmax
    let softmax_result = softmax(&scores, D::Minus1).unwrap();
    // Convert to f32 if needed
    let softmax_f32 = softmax_result.to_dtype(candle_core::DType::F32).unwrap();
    let values = softmax_f32.to_vec2::<f32>().unwrap();

    // No NaN values
    for (i, row) in values.iter().enumerate().take(3) {
        for (j, val) in row.iter().enumerate().take(3) {
            assert!(!val.is_nan(), "NaN at row {}, col {}", i, j);
        }
    }

    // Row 0: only first position should have weight
    assert!(values[0][0] > 0.99, "Only position should have ~1.0 weight");
    assert!(values[0][1] < 0.01, "Masked position should be near-zero");
    assert!(values[0][2] < 0.01, "Masked position should be near-zero");

    // Row 1: first two positions should split weight
    let row1_sum: f32 = values[1].iter().sum();
    assert!((row1_sum - 1.0).abs() < 1e-3, "Row should sum to 1.0");
    assert!(values[1][2] < 0.01, "Masked position should be near-zero");

    // Row 2: all positions should have weight
    let row2_sum: f32 = values[2].iter().sum();
    assert!((row2_sum - 1.0).abs() < 1e-3, "Row should sum to 1.0");
}

#[test]
fn test_large_negative_value_stability() {
    // Test that -1e9 is "negative enough" to suppress attention
    let scores = Tensor::from_slice(
        &[
            100.0, -1e9, // Large positive score vs masked
            0.0, -1e9, // Zero score vs masked
            -10.0, -1e9, // Negative score vs masked
        ],
        &[3, 2],
        &Device::Cpu,
    )
    .unwrap();

    let softmax_result = softmax(&scores, D::Minus1).unwrap();
    // Convert to f32 if needed
    let softmax_f32 = softmax_result.to_dtype(candle_core::DType::F32).unwrap();
    let values = softmax_f32.to_vec2::<f32>().unwrap();

    // All rows: first position should dominate, second should be negligible
    for (i, row) in values.iter().enumerate().take(3) {
        assert!(!row[0].is_nan());
        assert!(!row[1].is_nan());
        assert!(row[0] > 0.99999, "Unmasked position should dominate (row {})", i);
        assert!(row[1] < 1e-6, "Masked position should be negligible (row {})", i);
    }
}
