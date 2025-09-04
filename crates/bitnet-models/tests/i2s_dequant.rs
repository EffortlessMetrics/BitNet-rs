#![cfg(feature = "integration-tests")]
//! Tests for native I2_S dequantization

use bitnet_models::quant::i2s;
use half::f16;

/// Pack 2-bit values into bytes
/// Each byte contains 4 values: bits [1:0]=value 0, [3:2]=value 1, [5:4]=value 2, [7:6]=value 3
fn pack_2bit(vals: &[u8]) -> [u8; 64] {
    let mut bytes = [0u8; 64];
    for i in 0..256 {
        let v = if i < vals.len() { vals[i] & 0b11 } else { 0 };
        bytes[i >> 2] |= v << ((i & 3) * 2);
    }
    bytes
}

#[test]
fn i2s_single_block() {
    // Pattern: 0,1,2,3 repeating -> qi={-2,-1,0,1} with scale=0.5
    let mut qvals = [0u8; 256];
    for (i, item) in qvals.iter_mut().enumerate() {
        *item = (i % 4) as u8;
    }
    let qbits = pack_2bit(&qvals);
    let scale = f16::from_f32(0.5).to_bits().to_le_bytes();

    // Assemble one row with one block
    let mut bytes = Vec::with_capacity(66);
    bytes.extend_from_slice(&qbits);
    bytes.extend_from_slice(&scale);

    let out = i2s::dequantize_to_f32(&bytes, &[256]).unwrap();
    assert_eq!(out.len(), 256);

    for i in 0..256 {
        let qi = (qvals[i] as i8) - 2;
        let expected = 0.5 * (qi as f32);
        assert!((out[i] - expected).abs() < 1e-6, "i={i}: got {}, expected {expected}", out[i]);
    }
}

#[test]
fn i2s_multi_block() {
    // Test a 2x512 tensor (2 rows, each with 2 blocks)
    let scale1 = f16::from_f32(0.25);
    let scale2 = f16::from_f32(0.75);
    let scale3 = f16::from_f32(1.0);
    let scale4 = f16::from_f32(0.5);

    // Create pattern for testing
    let pattern1 = [0u8, 1, 2, 3]; // -2, -1, 0, 1
    let pattern2 = [3u8, 2, 1, 0]; // 1, 0, -1, -2

    let mut bytes = Vec::new();

    // Row 1, Block 1 (pattern1 repeating, scale=0.25)
    let mut vals = [0u8; 256];
    for i in 0..256 {
        vals[i] = pattern1[i % 4];
    }
    bytes.extend_from_slice(&pack_2bit(&vals));
    bytes.extend_from_slice(&scale1.to_bits().to_le_bytes());

    // Row 1, Block 2 (pattern2 repeating, scale=0.75)
    for i in 0..256 {
        vals[i] = pattern2[i % 4];
    }
    bytes.extend_from_slice(&pack_2bit(&vals));
    bytes.extend_from_slice(&scale2.to_bits().to_le_bytes());

    // Row 2, Block 1 (pattern2 repeating, scale=1.0)
    bytes.extend_from_slice(&pack_2bit(&vals)); // reuse last pattern
    bytes.extend_from_slice(&scale3.to_bits().to_le_bytes());

    // Row 2, Block 2 (pattern1 repeating, scale=0.5)
    for i in 0..256 {
        vals[i] = pattern1[i % 4];
    }
    bytes.extend_from_slice(&pack_2bit(&vals));
    bytes.extend_from_slice(&scale4.to_bits().to_le_bytes());

    // Dequantize 2x512 tensor
    let out = i2s::dequantize_to_f32(&bytes, &[2, 512]).unwrap();
    assert_eq!(out.len(), 1024);

    // Verify first element of each block
    let eps = 1e-5;

    // Row 1, Block 1, first element: scale1 * (-2)
    assert!((out[0] - 0.25 * (-2.0)).abs() < eps);

    // Row 1, Block 2, first element: scale2 * 1
    assert!((out[256] - 0.75 * 1.0).abs() < eps);

    // Row 2, Block 1, first element: scale3 * 1
    assert!((out[512] - 1.0 * 1.0).abs() < eps);

    // Row 2, Block 2, first element: scale4 * (-2)
    assert!((out[768] - 0.5 * (-2.0)).abs() < eps);
}

#[test]
fn i2s_partial_block() {
    // Test tensor with non-multiple-of-256 columns
    // Shape: [1, 300] = 2 blocks (256 + 44 elements)
    let scale1 = f16::from_f32(2.0);
    let scale2 = f16::from_f32(3.0);

    let mut bytes = Vec::new();

    // Block 1: full 256 elements, all zeros (q=2 -> qi=0)
    let zeros = [2u8; 256];
    bytes.extend_from_slice(&pack_2bit(&zeros));
    bytes.extend_from_slice(&scale1.to_bits().to_le_bytes());

    // Block 2: partial block (only 44 elements matter)
    let ones = [3u8; 256]; // q=3 -> qi=1
    bytes.extend_from_slice(&pack_2bit(&ones));
    bytes.extend_from_slice(&scale2.to_bits().to_le_bytes());

    let out = i2s::dequantize_to_f32(&bytes, &[300]).unwrap();
    assert_eq!(out.len(), 300);

    // First 256 should be zeros
    for (i, item) in out.iter().enumerate().take(256) {
        assert_eq!(*item, 0.0, "Element {i} should be 0");
    }

    // Next 44 should be scale2 * 1 = 3.0
    for (i, item) in out.iter().enumerate().take(300).skip(256) {
        assert!((*item - 3.0).abs() < 1e-5, "Element {i} should be 3.0");
    }
}

#[test]
fn i2s_error_on_size_mismatch() {
    // Test that we get an error for incorrect data size
    let bytes = vec![0u8; 100]; // Wrong size
    let result = i2s::dequantize_to_f32(&bytes, &[256]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("byte length mismatch"));
}

#[test]
fn i2s_constants() {
    assert_eq!(i2s::block_elems(), 256);
    assert_eq!(i2s::block_bytes(), 66);
}
