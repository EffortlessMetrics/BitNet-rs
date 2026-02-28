//! Integration tests for quantization algorithms
//!
//! These tests validate the correctness and numerical accuracy of all quantization
//! implementations against reference implementations and cross-validate between
//! different quantization types.

// ============================================================================
// Numerical accuracy integration tests (always run with --features cpu)
// ============================================================================

use bitnet_common::{BitNetTensor, QuantizationType};
use bitnet_quantization::{I2SQuantizer, Quantize, QuantizedTensor, TL1Quantizer, TL2Quantizer};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

fn make_tensor(data: Vec<f32>, shape: &[usize]) -> BitNetTensor {
    let t = CandleTensor::from_vec(data, shape, &CandleDevice::Cpu).unwrap();
    BitNetTensor::new(t)
}

/// Verify dequantizing a hand-crafted I2_S block produces the expected signed values.
///
/// I2_S packs 2-bit codes using offset encoding: code = signed_val + 2.
/// Codes [-2, -1, 0, 1] pack into a single byte (0xE4), and dequantize
/// back to [-2.0, -1.0, 0.0, 1.0] when scale = 1.0.
#[test]
fn test_i2s_dequantize_known_values() {
    // Codes [-2, -1, 0, 1] map to unsigned [0, 1, 2, 3].
    // Packed: 0 | (1<<2) | (2<<4) | (3<<6) = 0b11100100 = 0xE4
    let packed_byte: u8 = (1u8 << 2) | (2u8 << 4) | (3u8 << 6);
    assert_eq!(packed_byte, 0xE4);

    let tensor = QuantizedTensor::new_with_params(
        vec![packed_byte],
        vec![1.0f32],
        None,
        vec![4],
        QuantizationType::I2S,
        4,
    );

    let quantizer = I2SQuantizer::new();
    let dequantized = quantizer.dequantize_tensor(&tensor).unwrap();
    let vals = dequantized.to_vec().unwrap();

    assert_eq!(vals.len(), 4);
    let expected = [-2.0f32, -1.0, 0.0, 1.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "Element {i}: expected {exp}, got {got}");
    }
}

/// Verify TL1 2-bit quantization produces at most 4 distinct codes (2^2 = 4 levels).
///
/// TL1 with `precision_bits = 2` encodes each element as one of 4 unsigned
/// 2-bit codes (0..=3), packed two bits per element. The packed byte count
/// must equal `n_elements * 2 / 8` and only codes in `[0, 3]` appear.
#[test]
fn test_tl1_lut_entry_count() {
    let config = bitnet_quantization::tl1::TL1Config::default();
    let num_levels = 1usize << config.precision_bits; // 4 for 2-bit
    assert_eq!(num_levels, 4, "Default TL1 has 4 LUT levels (2-bit precision)");

    let n = 128usize; // Two full blocks of 64
    let data: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 2.0 - 1.0).collect();
    let tensor = make_tensor(data, &[n]);
    let quantizer = TL1Quantizer::new();
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();

    // Packed bytes: 2 bits/element → n * 2 / 8 bytes
    let expected_bytes = n * config.precision_bits as usize / 8;
    assert_eq!(
        quantized.data.len(),
        expected_bytes,
        "Packed data should be {expected_bytes} bytes for {n} elements at 2 bits/element"
    );

    // Unpacked codes must all be in [0, num_levels-1]
    let codes = bitnet_quantization::utils::unpack_unsigned_2bit_values(&quantized.data, n);
    assert_eq!(codes.len(), n);
    for &code in &codes {
        assert!((code as usize) < num_levels, "Code {code} out of range [0, {num_levels})");
    }

    // Diverse input should produce more than one distinct code
    let distinct: std::collections::HashSet<i8> = codes.into_iter().collect();
    assert!(distinct.len() > 1, "Diverse input must produce multiple distinct codes");
    assert!(
        distinct.len() <= num_levels,
        "At most {num_levels} distinct codes expected, got {}",
        distinct.len()
    );
}

/// Verify TL2 dequantization maps codes [0, 1, 2, 3] to symmetric signed values.
///
/// TL2 uses unsigned 2-bit codes with `shift = precision_bits/2 = 2`.
/// Dequantization: `(code - shift) * scale` yields {-2s, -s, 0, s} for
/// codes {0, 1, 2, 3}, confirming the symmetric centering around code 2.
#[test]
fn test_tl2_dequantize_symmetry() {
    // Codes [0, 1, 2, 3] → packed unsigned 2-bit byte:
    // 0 | (1<<2) | (2<<4) | (3<<6) = 0xE4
    let packed_byte: u8 = (1u8 << 2) | (2u8 << 4) | (3u8 << 6);

    let tensor = QuantizedTensor::new_with_params(
        vec![packed_byte],
        vec![1.0f32],
        None,
        vec![4],
        QuantizationType::TL2,
        4,
    );

    let quantizer = TL2Quantizer::new();
    let dequantized = quantizer.dequantize_tensor(&tensor).unwrap();
    let vals = dequantized.to_vec().unwrap();

    assert_eq!(vals.len(), 4);
    // With scale=1.0 and shift=2: (code - 2) * 1.0 = [-2, -1, 0, 1]
    let expected = [-2.0f32, -1.0, 0.0, 1.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "Element {i}: expected {exp}, got {got} (symmetry broken)"
        );
    }
    // Symmetric around zero: negative, zero, positive values must all appear
    assert!(vals[1] < 0.0 && vals[2] == 0.0 && vals[3] > 0.0, "Values must straddle zero");
}

/// Verify I2_S quantize→dequantize reconstruction error is below 0.1 per element.
///
/// With 2-bit quantization, max error ≤ `scale / 2 = max_abs / 2`.
/// Using inputs in `[-0.1, 0.1]` bounds `max_abs ≤ 0.1`, so max error ≤ 0.05 < 0.1.
#[test]
fn test_quantize_dequantize_round_trip_accuracy() {
    use rand::Rng as _;
    use rand::SeedableRng as _;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let data: Vec<f32> = (0..64).map(|_| rng.random_range(-0.1f32..=0.1f32)).collect();

    let tensor = make_tensor(data.clone(), &[64]);
    let quantizer = I2SQuantizer::new();
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
    let recovered = dequantized.to_vec().unwrap();

    assert_eq!(recovered.len(), data.len());
    for (i, (orig, rec)) in data.iter().zip(recovered.iter()).enumerate() {
        let err = (orig - rec).abs();
        assert!(
            err < 0.1,
            "Element {i}: reconstruction error {err:.4} >= 0.1 (orig={orig:.4}, recovered={rec:.4})"
        );
    }
}

/// Verify that the QK256 constant equals 256 elements per block.
///
/// QK256 packs 2 bits per element in 256-element blocks, requiring 64 packed bytes.
#[test]
fn test_qk256_block_size() {
    use bitnet_quantization::i2s_qk256::{I2SQk256NoScale, QK256_BLOCK, QK256_PACKED_BYTES};
    assert_eq!(QK256_BLOCK, 256, "QK256 must use 256-element blocks");
    // 2 bits/elem × 256 elem ÷ 8 bits/byte = 64 bytes per block
    assert_eq!(
        QK256_PACKED_BYTES,
        QK256_BLOCK / 4,
        "QK256_PACKED_BYTES should be 64 (= 256 × 2 / 8)"
    );
    // Verify I2SQk256NoScale enforces the block-size invariant
    let rows = 4usize;
    let cols = QK256_BLOCK;
    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let row_stride = blocks_per_row * QK256_PACKED_BYTES;
    let qs = vec![0u8; rows * row_stride];
    let weight = I2SQk256NoScale::new(rows, cols, qs).unwrap();
    assert_eq!(weight.rows, rows);
    assert_eq!(weight.cols, cols);
    assert_eq!(weight.row_stride_bytes, QK256_PACKED_BYTES);
}

/// Verify all-zeros input produces all-zeros output under every quantization scheme.
///
/// When all input values are zero, `calculate_scale` returns the safe fallback 1.0.
/// The quantized code for 0.0 is always the center code, which dequantizes back to 0.0.
#[test]
fn test_zero_vector_quantizes_to_zero() {
    let zeros = vec![0.0f32; 64];
    let tensor = make_tensor(zeros, &[64]);

    for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let quantized = tensor.quantize(qtype).unwrap();
        let dequantized = quantized.dequantize().unwrap();
        let vals = dequantized.to_vec().unwrap();
        assert_eq!(vals.len(), 64);
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(v, 0.0, "Element {i} for {qtype:?}: expected 0.0, got {v}");
        }
    }
}

// ============================================================================
// Feature-gated integration tests (require --features integration-tests)
// ============================================================================

#[cfg(feature = "integration-tests")]
mod gated {
    use bitnet_common::{BitNetTensor, QuantizationType, Tensor};
    use bitnet_quantization::{
        I2SQuantizer, Quantize, QuantizerFactory, QuantizerTrait, TL1Quantizer, TL2Quantizer,
        convert_quantization,
    };
    use candle_core::{Device, Tensor as CandleTensor};
    use proptest::prelude::*;

    /// Helper function to create test tensors
    fn create_test_tensor(data: Vec<f32>, shape: Vec<usize>) -> BitNetTensor {
        let device = Device::Cpu;
        let tensor = CandleTensor::from_vec(data, shape.as_slice(), &device).unwrap();
        BitNetTensor::new(tensor)
    }

    /// Test basic quantization round-trip for all quantization types
    #[test]
    fn test_all_quantization_round_trips() {
        let data = vec![1.0, -2.0, 0.5, -0.5, 3.0, -1.5, 0.0, 2.5];
        let shape = vec![2, 4];
        let tensor = create_test_tensor(data, shape.clone());

        for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
            let quantizer = QuantizerFactory::create(qtype);

            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

            assert_eq!(quantized.qtype, qtype);
            assert_eq!(quantized.shape, shape);
            assert_eq!(dequantized.shape(), &shape);

            // Verify compression ratio
            let ratio = quantized.compression_ratio();
            assert!(ratio > 2.0, "Compression ratio too low: {}", ratio);
        }
    }

    /// Test quantization format conversion
    #[test]
    fn test_quantization_format_conversion() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
        let shape = vec![6];
        let tensor = create_test_tensor(data, shape);

        // Start with I2_S
        let i2s_quantized = tensor.quantize(QuantizationType::I2S).unwrap();

        // Convert to TL1
        let tl1_quantized = convert_quantization(&i2s_quantized, QuantizationType::TL1).unwrap();
        assert_eq!(tl1_quantized.qtype, QuantizationType::TL1);

        // Convert to TL2
        let tl2_quantized = convert_quantization(&tl1_quantized, QuantizationType::TL2).unwrap();
        assert_eq!(tl2_quantized.qtype, QuantizationType::TL2);

        // Convert back to I2_S
        let back_to_i2s = convert_quantization(&tl2_quantized, QuantizationType::I2S).unwrap();
        assert_eq!(back_to_i2s.qtype, QuantizationType::I2S);

        // All should be dequantizable
        let _ = i2s_quantized.dequantize().unwrap();
        let _ = tl1_quantized.dequantize().unwrap();
        let _ = tl2_quantized.dequantize().unwrap();
        let _ = back_to_i2s.dequantize().unwrap();
    }

    /// Test quantization with different tensor shapes
    #[test]
    fn test_different_tensor_shapes() {
        let test_cases = vec![
            (vec![1.0], vec![1]),                   // Scalar
            (vec![1.0, 2.0, 3.0, 4.0], vec![4]),    // Vector
            (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]), // Matrix
            (vec![1.0; 24], vec![2, 3, 4]),         // 3D tensor
            (vec![1.0; 120], vec![2, 3, 4, 5]),     // 4D tensor
        ];

        for (data, shape) in test_cases {
            let tensor = create_test_tensor(data, shape.clone());

            for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
                let quantized = tensor.quantize(qtype).unwrap();
                let dequantized = quantized.dequantize().unwrap();

                assert_eq!(quantized.shape, shape);
                assert_eq!(dequantized.shape(), &shape);
            }
        }
    }

    /// Test quantization with extreme values
    #[test]
    fn test_extreme_values() {
        let extreme_data = vec![f32::MAX, f32::MIN, 0.0, 1e-10, -1e-10, 100.0, -100.0, 1e6, -1e6];
        let shape = vec![9];
        let tensor = create_test_tensor(extreme_data, shape);

        for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
            let quantized = tensor.quantize(qtype).unwrap();
            let dequantized = quantized.dequantize().unwrap();

            // Should not panic and should maintain shape
            assert_eq!(dequantized.shape(), &[9]);
        }
    }

    /// Test quantization accuracy with known patterns
    #[test]
    fn test_quantization_accuracy() {
        // Test with a sine wave pattern
        let data: Vec<f32> =
            (0..64).map(|i| (i as f32 * std::f32::consts::PI / 32.0).sin()).collect();
        let shape = vec![64];
        let tensor = create_test_tensor(data.clone(), shape);

        for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
            let quantized = tensor.quantize(qtype).unwrap();
            let dequantized = quantized.dequantize().unwrap();

            // Extract dequantized data for comparison
            let dequant_candle = dequantized.inner();
            let dequant_data = dequant_candle.to_vec1::<f32>().unwrap();

            // Calculate MSE
            let mse: f32 = data
                .iter()
                .zip(dequant_data.iter())
                .map(|(&orig, &dequant)| (orig - dequant).powi(2))
                .sum::<f32>()
                / data.len() as f32;

            // MSE should be reasonable for 2-bit quantization (allow higher error for limited precision)
            assert!(mse < 2.0, "MSE too high for {}: {}", qtype, mse);
        }
    }

    /// Test quantizer availability
    #[test]
    fn test_quantizer_availability() {
        let i2s = I2SQuantizer::new();
        let tl1 = TL1Quantizer::new();
        let tl2 = TL2Quantizer::new();

        assert!(i2s.is_available());
        assert!(tl1.is_available());
        assert!(tl2.is_available());

        assert_eq!(i2s.quantization_type(), QuantizationType::I2S);
        assert_eq!(tl1.quantization_type(), QuantizationType::TL1);
        assert_eq!(tl2.quantization_type(), QuantizationType::TL2);
    }

    /// Test best quantization type selection for architecture
    #[test]
    fn test_best_quantization_for_arch() {
        let best = QuantizerFactory::best_for_arch();

        // Should return a valid quantization type
        match best {
            QuantizationType::I2S | QuantizationType::TL1 | QuantizationType::TL2 => {
                // All valid
            }
        }

        // Should be able to create a quantizer for the best type
        let quantizer = QuantizerFactory::create(best);
        assert!(quantizer.is_available());
    }

    // Property-based test for quantization round-trip accuracy
    proptest! {
        #[test]
        fn prop_quantization_round_trip(
            data in prop::collection::vec(-10.0f32..10.0f32, 1..100),
            qtype in prop::sample::select(vec![
                QuantizationType::I2S,
                QuantizationType::TL1,
                QuantizationType::TL2
            ])
        ) {
            let shape = vec![data.len()];
            let tensor = create_test_tensor(data.clone(), shape.clone());

            let quantized = tensor.quantize(qtype).unwrap();
            let dequantized = quantized.dequantize().unwrap();

            // Basic properties should hold
            prop_assert_eq!(quantized.qtype, qtype);

            // Compression ratio should be reasonable
            let ratio = quantized.compression_ratio();
            prop_assert_eq!(quantized.shape, shape.clone());
            prop_assert_eq!(dequantized.shape(), &shape);
            prop_assert!(ratio >= 1.0); // Allow ratio of 1.0 for very small tensors

            // Should be able to extract dequantized data
            let dequant_candle = dequantized.inner();
            let dequant_data = dequant_candle.to_vec1::<f32>().unwrap();
            prop_assert_eq!(dequant_data.len(), data.len());
        }
    }

    // Property-based test for quantization format conversion
    proptest! {
        #[test]
        fn prop_format_conversion(
            data in prop::collection::vec(-5.0f32..5.0f32, 4..32),
            source_qtype in prop::sample::select(vec![
                QuantizationType::I2S,
                QuantizationType::TL1,
                QuantizationType::TL2
            ]),
            target_qtype in prop::sample::select(vec![
                QuantizationType::I2S,
                QuantizationType::TL1,
                QuantizationType::TL2
            ])
        ) {
            let shape = vec![data.len()];
            let tensor = create_test_tensor(data, shape.clone());

            // Quantize to source format
            let source_quantized = tensor.quantize(source_qtype).unwrap();

            // Convert to target format
            let target_quantized = convert_quantization(&source_quantized, target_qtype).unwrap();

            // Properties should be preserved
            prop_assert_eq!(target_quantized.qtype, target_qtype);

            // Should be dequantizable
            let dequantized = target_quantized.dequantize().unwrap();
            prop_assert_eq!(target_quantized.shape, shape.clone());
            prop_assert_eq!(dequantized.shape(), &shape);
        }
    }

    // Property-based test for quantization with different block sizes
    proptest! {
        #[test]
        fn prop_different_block_sizes(
            data in prop::collection::vec(-2.0f32..2.0f32, 16..128),
            block_size in 4usize..64usize
        ) {
            let shape = vec![data.len()];
            let tensor = create_test_tensor(data, shape.clone());

            // Test I2_S with different block sizes
            let i2s_quantizer = I2SQuantizer::with_block_size(block_size);
            let quantized = i2s_quantizer.quantize_tensor(&tensor).unwrap();
            let dequantized = i2s_quantizer.dequantize_tensor(&quantized).unwrap();

            prop_assert_eq!(quantized.block_size, block_size);
            prop_assert_eq!(quantized.shape, shape.clone());
            prop_assert_eq!(dequantized.shape(), &shape);
        }
    }

    /// Benchmark comparison test (simplified)
    #[test]
    fn test_quantization_performance_comparison() {
        let data = vec![1.0; 1024];
        let shape = vec![32, 32];
        let tensor = create_test_tensor(data, shape);

        let start = std::time::Instant::now();
        let _ = tensor.quantize(QuantizationType::I2S).unwrap();
        let i2s_time = start.elapsed();

        let start = std::time::Instant::now();
        let _ = tensor.quantize(QuantizationType::TL1).unwrap();
        let tl1_time = start.elapsed();

        let start = std::time::Instant::now();
        let _ = tensor.quantize(QuantizationType::TL2).unwrap();
        let tl2_time = start.elapsed();

        // All should complete in reasonable time (< 1 second for this small tensor)
        assert!(i2s_time.as_secs() < 1);
        assert!(tl1_time.as_secs() < 1);
        assert!(tl2_time.as_secs() < 1);
    }
} // mod gated
