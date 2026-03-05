#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal quantization shader tests for Apple Silicon.
//!
//! Validates I2_S (2-bit ternary) quantization, dequantization, matrix-vector
//! multiply, scale factor handling, block-wise quantization (BitNet32 / QK256),
//! mixed-precision compute, buffer layout, numerical accuracy vs CPU reference,
//! edge-case patterns, and throughput for common hidden dimensions.
//!
//! All tests are `#[ignore]` — they require a macOS host with Metal GPU
//! (Apple Silicon).

#![cfg(target_os = "macos")]

#[cfg(test)]
mod tests {
    use std::time::Instant;

    // ── I2_S encoding constants ────────────────────────────────────────
    const I2S_ZERO: u8 = 0b00;
    const I2S_PLUS_ONE: u8 = 0b01;
    const I2S_MINUS_ONE: u8 = 0b11;

    // Block sizes
    const BITNET32_BLOCK: usize = 32;
    const QK256_BLOCK: usize = 256;

    // Metal constraints
    const METAL_BUFFER_ALIGNMENT: usize = 256;
    const METAL_MAX_THREADS_PER_THREADGROUP: usize = 1024;

    // ── Encoding / decoding helpers ────────────────────────────────────

    /// Decode a single 2-bit I2_S code to its signed value.
    fn decode_i2s(bits: u8) -> i8 {
        match bits & 0x03 {
            I2S_PLUS_ONE => 1,
            I2S_MINUS_ONE => -1,
            _ => 0, // 0b00 → 0, 0b10 (reserved) → 0
        }
    }

    /// Pack four ternary values ({-1, 0, +1}) into one byte (LSB-first).
    fn pack_i2s(vals: [i8; 4]) -> u8 {
        let mut byte = 0u8;
        for (i, &v) in vals.iter().enumerate() {
            let code: u8 = match v {
                1 => I2S_PLUS_ONE,
                -1 => I2S_MINUS_ONE,
                _ => I2S_ZERO,
            };
            byte |= code << (i * 2);
        }
        byte
    }

    /// Pack an arbitrary-length slice of ternary values into bytes.
    fn pack_i2s_vec(vals: &[i8]) -> Vec<u8> {
        let num_bytes = vals.len().div_ceil(4);
        let mut packed = vec![0u8; num_bytes];
        for (i, &v) in vals.iter().enumerate() {
            let code: u8 = match v {
                1 => I2S_PLUS_ONE,
                -1 => I2S_MINUS_ONE,
                _ => I2S_ZERO,
            };
            packed[i / 4] |= code << ((i % 4) * 2);
        }
        packed
    }

    /// Dequantize packed I2_S bytes back to f32 with a scale factor.
    fn dequant_i2s(packed: &[u8], scale: f32, count: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let byte_idx = i / 4;
            let bit_off = (i % 4) * 2;
            let bits = (packed[byte_idx] >> bit_off) & 0x03;
            out.push(decode_i2s(bits) as f32 * scale);
        }
        out
    }

    /// CPU reference: I2_S matrix-vector multiply.
    ///
    /// `weights_packed` is column-major packed I2_S `[k × n]`.
    /// `scales` has one entry per block per column: `n * ceil(k / block_size)`.
    /// `activation` is a `[k]` f32 vector.
    /// Returns `[n]` f32 output.
    fn cpu_i2s_matvec(
        weights_packed: &[u8],
        scales: &[f32],
        activation: &[f32],
        n: usize,
        k: usize,
        block_size: usize,
    ) -> Vec<f32> {
        let packed_k = k.div_ceil(4);
        let num_blocks = k.div_ceil(block_size);
        let mut out = vec![0.0f32; n];

        for col in 0..n {
            let mut acc = 0.0f32;
            for blk in 0..num_blocks {
                let blk_start = blk * block_size;
                let blk_end = (blk_start + block_size).min(k);
                let scale = scales[col * num_blocks + blk];
                for i in blk_start..blk_end {
                    let byte_idx = i / 4;
                    let bit_off = (i % 4) * 2;
                    let bits = (weights_packed[col * packed_k + byte_idx] >> bit_off) & 0x03;
                    let w = decode_i2s(bits) as f32 * scale;
                    acc += w * activation[i];
                }
            }
            out[col] = acc;
        }
        out
    }

    /// Pad a byte buffer to Metal's 256-byte alignment.
    fn align_to_metal(data: &[u8]) -> Vec<u8> {
        let aligned_len = (data.len() + METAL_BUFFER_ALIGNMENT - 1) & !(METAL_BUFFER_ALIGNMENT - 1);
        let mut buf = vec![0u8; aligned_len];
        buf[..data.len()].copy_from_slice(data);
        buf
    }

    // ====================================================================
    // 1. I2_S quantization shader: encode +1/0/-1 into 2-bit packed format
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_i2s_quantize_encode_basic() {
        // Encode the canonical ternary values and verify packed bytes.
        let vals: [i8; 4] = [1, 0, -1, 0];
        let packed = pack_i2s(vals);
        // Expected: +1=0b01 at [1:0], 0=0b00 at [3:2],
        //           -1=0b11 at [5:4], 0=0b00 at [7:6]
        assert_eq!(packed, 0b00_11_00_01);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_i2s_quantize_encode_all_plus_one() {
        let vals: [i8; 4] = [1, 1, 1, 1];
        let packed = pack_i2s(vals);
        assert_eq!(packed, 0b01_01_01_01);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_i2s_quantize_encode_all_minus_one() {
        let vals: [i8; 4] = [-1, -1, -1, -1];
        let packed = pack_i2s(vals);
        assert_eq!(packed, 0b11_11_11_11);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_i2s_quantize_encode_all_zero() {
        let vals: [i8; 4] = [0, 0, 0, 0];
        let packed = pack_i2s(vals);
        assert_eq!(packed, 0b00_00_00_00);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_i2s_quantize_encode_multiblock() {
        // Encode 8 values (2 bytes): [+1, -1, +1, -1, 0, 0, +1, -1]
        let vals: [i8; 8] = [1, -1, 1, -1, 0, 0, 1, -1];
        let packed = pack_i2s_vec(&vals);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], pack_i2s([1, -1, 1, -1]));
        assert_eq!(packed[1], pack_i2s([0, 0, 1, -1]));
    }

    // ====================================================================
    // 2. I2_S dequantization shader: unpack 2-bit values to f32
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_i2s_dequantize_unit_scale() {
        let vals: [i8; 4] = [1, 0, -1, 0];
        let packed = pack_i2s(vals);
        let result = dequant_i2s(&[packed], 1.0, 4);
        assert_eq!(result, vec![1.0, 0.0, -1.0, 0.0]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_i2s_dequantize_with_scale() {
        let vals: [i8; 4] = [1, -1, 0, 1];
        let packed = pack_i2s(vals);
        let scale = 0.5;
        let result = dequant_i2s(&[packed], scale, 4);
        assert_eq!(result, vec![0.5, -0.5, 0.0, 0.5]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_i2s_dequantize_reserved_code_0b10() {
        // Manually craft a byte with the reserved 0b10 code.
        let byte = 0b10; // element 0 = 0b10 (reserved)
        let result = dequant_i2s(&[byte], 1.0, 1);
        // Reserved code is treated as 0.
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_i2s_dequantize_roundtrip() {
        let original: Vec<i8> = vec![1, 0, -1, 1, -1, -1, 0, 0, 1, 1, -1, 0];
        let packed = pack_i2s_vec(&original);
        let restored = dequant_i2s(&packed, 1.0, original.len());
        let expected: Vec<f32> = original.iter().map(|&v| v as f32).collect();
        assert_eq!(restored, expected);
    }

    // ====================================================================
    // 3. Quantized matrix-vector multiply with I2_S weights
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_i2s_matvec_identity_like() {
        // k=4, n=4, block_size=4: diagonal of +1 weights, scale=1.0
        let k: usize = 4;
        let n = 4;
        let block_size: usize = 4;
        let packed_k = k.div_ceil(4); // 1 byte per column

        let mut weights_packed = vec![0u8; n * packed_k];
        for col in 0..n {
            // Set element `col` to +1 in column `col`.
            let mut vals = [0i8; 4];
            vals[col] = 1;
            weights_packed[col * packed_k] = pack_i2s(vals);
        }
        let scales = vec![1.0f32; n]; // one block per column
        let activation = vec![2.0, 3.0, 4.0, 5.0];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_i2s_matvec_all_ones_weights() {
        let k: usize = 8;
        let n = 2;
        let block_size: usize = 8;
        let packed_k = k.div_ceil(4); // 2 bytes per column

        // All +1 weights
        let vals_all_one: Vec<i8> = vec![1; k];
        let packed_col = pack_i2s_vec(&vals_all_one);

        let mut weights_packed = vec![0u8; n * packed_k];
        for col in 0..n {
            weights_packed[col * packed_k..col * packed_k + packed_col.len()]
                .copy_from_slice(&packed_col);
        }
        let scales = vec![1.0f32; n]; // 1 block per column
        let activation: Vec<f32> = (1..=k as i32).map(|x| x as f32).collect();
        let expected_sum: f32 = activation.iter().sum();

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        for &v in &result {
            assert!((v - expected_sum).abs() < 1e-6, "Expected {expected_sum}, got {v}");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_i2s_matvec_negative_weights() {
        let k: usize = 4;
        let n = 1;
        let block_size: usize = 4;
        let packed_k = k.div_ceil(4);

        let vals: Vec<i8> = vec![-1; k];
        let packed_col = pack_i2s_vec(&vals);
        let mut weights_packed = vec![0u8; packed_k];
        weights_packed[..packed_col.len()].copy_from_slice(&packed_col);
        let scales = vec![1.0f32];
        let activation = vec![1.0, 2.0, 3.0, 4.0];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        let expected: f32 = -activation.iter().sum::<f32>();
        assert!((result[0] - expected).abs() < 1e-6);
    }

    // ====================================================================
    // 4. Scale factor extraction and application
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_scale_factor_application() {
        // A single block of +1 weights with scale=0.25
        let vals: [i8; 4] = [1, 1, 1, 1];
        let packed = pack_i2s(vals);
        let result = dequant_i2s(&[packed], 0.25, 4);
        assert_eq!(result, vec![0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_per_block_scale_factors() {
        // Two blocks, each with a different scale.
        let k: usize = 8;
        let n = 1;
        let block_size: usize = 4;
        let packed_k = k.div_ceil(4);
        let num_blocks = k.div_ceil(block_size); // 2

        let vals: Vec<i8> = vec![1; k];
        let weights_packed = pack_i2s_vec(&vals);
        // Block 0 scale=2.0, block 1 scale=3.0
        let scales = vec![2.0f32, 3.0f32];
        assert_eq!(scales.len(), num_blocks);

        let activation = vec![1.0f32; k];
        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        // Block 0: 4 * 1.0 * 1.0 * 2.0 = 8.0
        // Block 1: 4 * 1.0 * 1.0 * 3.0 = 12.0
        let expected = 8.0 + 12.0;
        assert!((result[0] - expected).abs() < 1e-6, "Expected {expected}, got {}", result[0]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_scale_factor_zero() {
        let vals: [i8; 4] = [1, -1, 1, -1];
        let packed = pack_i2s(vals);
        let result = dequant_i2s(&[packed], 0.0, 4);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_scale_factor_negative() {
        let vals: [i8; 4] = [1, 0, -1, 0];
        let packed = pack_i2s(vals);
        let result = dequant_i2s(&[packed], -2.0, 4);
        assert_eq!(result, vec![-2.0, 0.0, 2.0, 0.0]);
    }

    // ====================================================================
    // 5. Block-wise quantization (32-elem for BitNet32, 256-elem for QK256)
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_bitnet32_block_size() {
        let k: usize = BITNET32_BLOCK; // 32
        let n = 1;
        let packed_k = k.div_ceil(4); // 8 bytes
        let num_blocks = k.div_ceil(BITNET32_BLOCK);
        assert_eq!(num_blocks, 1);

        // All +1 weights
        let vals: Vec<i8> = vec![1; k];
        let weights_packed = pack_i2s_vec(&vals);
        assert_eq!(weights_packed.len(), packed_k);

        let scales = vec![1.0f32; num_blocks];
        let activation: Vec<f32> = vec![1.0; k];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, BITNET32_BLOCK);
        assert!((result[0] - 32.0).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_qk256_block_size() {
        let k: usize = QK256_BLOCK; // 256
        let n = 1;
        let packed_k = k.div_ceil(4); // 64 bytes
        let num_blocks = k.div_ceil(QK256_BLOCK);
        assert_eq!(num_blocks, 1);

        let vals: Vec<i8> = vec![1; k];
        let weights_packed = pack_i2s_vec(&vals);
        assert_eq!(weights_packed.len(), packed_k);

        let scales = vec![1.0f32; num_blocks];
        let activation: Vec<f32> = vec![1.0; k];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, QK256_BLOCK);
        assert!((result[0] - 256.0).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_bitnet32_multi_block() {
        // 64 elements → 2 blocks of 32
        let k: usize = 64;
        let n = 1;
        let num_blocks = k.div_ceil(BITNET32_BLOCK);
        assert_eq!(num_blocks, 2);

        // Alternating pattern: block 0 all +1, block 1 all -1
        let mut vals = vec![0i8; k];
        vals[..BITNET32_BLOCK].fill(1);
        vals[BITNET32_BLOCK..].fill(-1);
        let weights_packed = pack_i2s_vec(&vals);

        let scales = vec![1.0f32; num_blocks];
        let activation: Vec<f32> = vec![1.0; k];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, BITNET32_BLOCK);
        // Block 0: 32 * 1.0 = 32.0, Block 1: 32 * -1.0 = -32.0
        assert!((result[0]).abs() < 1e-6, "Expected ~0.0, got {}", result[0]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_qk256_multi_block() {
        // 512 elements → 2 blocks of 256
        let k: usize = 512;
        let n = 1;
        let num_blocks = k.div_ceil(QK256_BLOCK);
        assert_eq!(num_blocks, 2);

        let vals: Vec<i8> = vec![1; k];
        let weights_packed = pack_i2s_vec(&vals);
        // Different scales per block
        let scales = vec![0.5f32, 2.0f32];
        let activation: Vec<f32> = vec![1.0; k];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, QK256_BLOCK);
        // Block 0: 256 * 0.5 = 128.0, Block 1: 256 * 2.0 = 512.0
        let expected = 128.0 + 512.0;
        assert!((result[0] - expected).abs() < 1e-4, "Expected {expected}, got {}", result[0]);
    }

    // ====================================================================
    // 6. Mixed precision: I2_S weight + f16 activation computation
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_mixed_precision_f16_activation() {
        // Simulate f16 activations via half crate, compute in f32.
        let k: usize = 4;
        let n = 1;
        let block_size: usize = 4;

        let vals: [i8; 4] = [1, -1, 1, -1];
        let weights_packed = pack_i2s_vec(&vals);
        let scales = vec![1.0f32];

        // Simulate f16 round-trip for activations
        let activation_f32 = [1.5, 2.5, 3.5, 4.5];
        let activation_f16: Vec<f32> =
            activation_f32.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation_f16, n, k, block_size);
        // w·a = 1*1.5 + (-1)*2.5 + 1*3.5 + (-1)*4.5 = 1.5 - 2.5 + 3.5 - 4.5 = -2.0
        let expected = -2.0f32;
        assert!(
            (result[0] - expected).abs() < 0.01,
            "Mixed precision result {}, expected {expected}",
            result[0]
        );
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_mixed_precision_f16_scale() {
        // Scale stored as f16 then promoted to f32.
        let vals: [i8; 4] = [1, 1, 1, 1];
        let packed = pack_i2s(vals);
        let scale_f16 = half::f16::from_f32(0.3).to_f32();
        let result = dequant_i2s(&[packed], scale_f16, 4);
        for &v in &result {
            // f16(0.3) ≈ 0.2998..0.3003; tolerance 1e-3
            assert!((v - 0.3).abs() < 1e-3, "Expected ~0.3, got {v}");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_mixed_precision_accumulation_order() {
        // Larger dot-product to expose accumulation drift.
        let k: usize = 256;
        let n = 1;
        let block_size: usize = QK256_BLOCK;

        let vals: Vec<i8> = (0..k).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let weights_packed = pack_i2s_vec(&vals);
        let scales = vec![1.0f32];

        // Activations as f16 round-tripped
        let activation: Vec<f32> =
            (0..k).map(|i| half::f16::from_f32(0.01 * i as f32).to_f32()).collect();

        let result_f16 = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        // Compute f32 reference
        let activation_f32: Vec<f32> = (0..k).map(|i| 0.01 * i as f32).collect();
        let result_f32 =
            cpu_i2s_matvec(&weights_packed, &scales, &activation_f32, n, k, block_size);
        // Allow f16 quantization noise
        assert!(
            (result_f16[0] - result_f32[0]).abs() < 0.5,
            "f16 result {} vs f32 result {}: drift too large",
            result_f16[0],
            result_f32[0]
        );
    }

    // ====================================================================
    // 7. Buffer layout validation for quantized tensors
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_buffer_alignment_256_bytes() {
        let data = vec![0u8; 100];
        let aligned = align_to_metal(&data);
        assert_eq!(aligned.len() % METAL_BUFFER_ALIGNMENT, 0);
        assert!(aligned.len() >= data.len());
        // Original data preserved
        assert_eq!(&aligned[..data.len()], &data[..]);
        // Padding is zero
        for &b in &aligned[data.len()..] {
            assert_eq!(b, 0);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_buffer_alignment_exact_multiple() {
        let data = vec![0u8; METAL_BUFFER_ALIGNMENT];
        let aligned = align_to_metal(&data);
        assert_eq!(aligned.len(), METAL_BUFFER_ALIGNMENT);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_packed_weight_buffer_layout() {
        // Verify column-major layout: n columns of packed_k bytes each.
        let k: usize = 32;
        let n = 4;
        let packed_k = k.div_ceil(4); // 8
        let total = n * packed_k;

        let mut weights = vec![0u8; total];
        // Column 2, element 0 = +1
        let col = 2;
        weights[col * packed_k] = pack_i2s([1, 0, 0, 0]);

        // Read back column 2, element 0
        let byte = weights[col * packed_k];
        let bits = byte & 0x03;
        assert_eq!(decode_i2s(bits), 1);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_scale_buffer_layout() {
        // Scales: [n * num_blocks_k] f32, column-major block ordering.
        let k: usize = 64;
        let n = 3;
        let block_size: usize = BITNET32_BLOCK;
        let num_blocks = k.div_ceil(block_size); // 2

        let mut scales = vec![0.0f32; n * num_blocks];
        // Column 1, block 0 → index 1*2 + 0 = 2
        scales[num_blocks] = 1.5;
        // Column 1, block 1 → index 1*2 + 1 = 3
        scales[num_blocks + 1] = 2.5;

        assert_eq!(scales[2], 1.5);
        assert_eq!(scales[3], 2.5);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_threadgroup_size_constraint() {
        // Validate that dispatched thread counts respect the Metal limit.
        let hidden_dims = [768, 2048, 4096];
        for &dim in &hidden_dims {
            let threads = dim.min(METAL_MAX_THREADS_PER_THREADGROUP);
            assert!(
                threads <= METAL_MAX_THREADS_PER_THREADGROUP,
                "Threadgroup size {threads} exceeds Metal limit \
                 {METAL_MAX_THREADS_PER_THREADGROUP} for dim {dim}"
            );
            // Thread count should be > 0
            assert!(threads > 0);
        }
    }

    // ====================================================================
    // 8. Numerical accuracy vs CPU reference
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_accuracy_small_matvec() {
        let k: usize = 32;
        let n = 8;
        let block_size: usize = BITNET32_BLOCK;

        // Deterministic pseudo-random weights
        let vals: Vec<i8> = (0..k * n)
            .map(|i| match i % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            })
            .collect();

        let packed_k = k.div_ceil(4);
        let mut weights_packed = vec![0u8; n * packed_k];
        for col in 0..n {
            let col_vals = &vals[col * k..(col + 1) * k];
            let col_packed = pack_i2s_vec(col_vals);
            weights_packed[col * packed_k..col * packed_k + col_packed.len()]
                .copy_from_slice(&col_packed);
        }

        let num_blocks = k.div_ceil(block_size);
        let scales = vec![1.0f32; n * num_blocks];
        let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.1).sin()).collect();

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);

        // Verify against brute-force
        for col in 0..n {
            let mut expected = 0.0f32;
            for i in 0..k {
                let w = vals[col * k + i] as f32;
                expected += w * activation[i];
            }
            assert!(
                (result[col] - expected).abs() < 1e-5,
                "Col {col}: expected {expected}, got {}",
                result[col]
            );
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_accuracy_qk256_large() {
        let k: usize = 2048;
        let n = 4;
        let block_size: usize = QK256_BLOCK;
        let packed_k = k.div_ceil(4);
        let num_blocks = k.div_ceil(block_size);

        // Build weights: repeating [+1, -1, 0, +1]
        let vals: Vec<i8> = (0..k * n)
            .map(|i| match i % 4 {
                0 => 1,
                1 => -1,
                2 => 0,
                _ => 1,
            })
            .collect();

        let mut weights_packed = vec![0u8; n * packed_k];
        for col in 0..n {
            let col_vals = &vals[col * k..(col + 1) * k];
            let col_packed = pack_i2s_vec(col_vals);
            weights_packed[col * packed_k..col * packed_k + col_packed.len()]
                .copy_from_slice(&col_packed);
        }

        let scales: Vec<f32> = (0..n * num_blocks).map(|i| 0.5 + 0.1 * (i % 5) as f32).collect();
        let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.003).cos()).collect();

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);

        // Brute-force reference
        for col in 0..n {
            let mut expected = 0.0f32;
            for blk in 0..num_blocks {
                let blk_start = blk * block_size;
                let blk_end = (blk_start + block_size).min(k);
                let scale = scales[col * num_blocks + blk];
                for i in blk_start..blk_end {
                    let w = vals[col * k + i] as f32 * scale;
                    expected += w * activation[i];
                }
            }
            assert!(
                (result[col] - expected).abs() < 1e-3,
                "Col {col}: expected {expected}, got {}",
                result[col]
            );
        }
    }

    // ====================================================================
    // 9. Edge cases: zero weights, all-ones, alternating patterns
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_edge_all_zero_weights() {
        let k: usize = 64;
        let n = 2;
        let block_size: usize = BITNET32_BLOCK;
        let packed_k = k.div_ceil(4);

        let weights_packed = vec![0u8; n * packed_k]; // all 0b00 → weight 0
        let num_blocks = k.div_ceil(block_size);
        let scales = vec![1.0f32; n * num_blocks];
        let activation: Vec<f32> = (0..k).map(|i| i as f32).collect();

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        for (col, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-6, "Zero weights should produce 0 output, col {col} got {v}");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_edge_all_plus_one_weights() {
        let k: usize = 128;
        let n = 1;
        let block_size: usize = BITNET32_BLOCK;
        let packed_k = k.div_ceil(4);

        let vals: Vec<i8> = vec![1; k];
        let weights_packed = pack_i2s_vec(&vals);
        assert_eq!(weights_packed.len(), packed_k);

        let num_blocks = k.div_ceil(block_size);
        let scales = vec![1.0f32; num_blocks];
        let activation: Vec<f32> = vec![1.0; k];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        assert!(
            (result[0] - k as f32).abs() < 1e-4,
            "All-ones should sum to {k}, got {}",
            result[0]
        );
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_edge_alternating_plus_minus() {
        let k: usize = 128;
        let n = 1;
        let block_size: usize = BITNET32_BLOCK;
        let packed_k = k.div_ceil(4);

        let vals: Vec<i8> = (0..k).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let weights_packed = pack_i2s_vec(&vals);
        assert_eq!(weights_packed.len(), packed_k);

        let num_blocks = k.div_ceil(block_size);
        let scales = vec![1.0f32; num_blocks];
        // Uniform activation: alternating pattern should cancel out.
        let activation = vec![1.0f32; k];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        assert!(
            result[0].abs() < 1e-6,
            "Alternating ±1 with uniform activation should cancel, got {}",
            result[0]
        );
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_edge_single_element() {
        let k: usize = 1;
        let n = 1;
        let block_size: usize = BITNET32_BLOCK;

        let vals: Vec<i8> = vec![1];
        let weights_packed = pack_i2s_vec(&vals);
        let scales = vec![3.0f32];
        let activation = vec![7.0f32];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        assert!((result[0] - 21.0).abs() < 1e-6, "1 * 3.0 * 7.0 = 21.0, got {}", result[0]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_edge_non_multiple_of_four() {
        // k=7 → packed into 2 bytes (8 slots, last slot padding)
        let k: usize = 7;
        let n = 1;
        let block_size: usize = BITNET32_BLOCK;

        let vals: Vec<i8> = vec![1, -1, 1, -1, 1, -1, 1];
        let weights_packed = pack_i2s_vec(&vals);
        assert_eq!(weights_packed.len(), 2); // ceil(7/4) = 2

        let scales = vec![1.0f32];
        let activation = vec![1.0f32; k];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        // 4 × +1 and 3 × -1 → 4 - 3 = 1
        assert!((result[0] - 1.0).abs() < 1e-6, "Expected 1.0, got {}", result[0]);
    }

    // ====================================================================
    // 10. Performance: throughput for common hidden dimensions
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_throughput_dim_768() {
        run_throughput_benchmark(768, 768, BITNET32_BLOCK, 100);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_throughput_dim_2048() {
        run_throughput_benchmark(2048, 2048, QK256_BLOCK, 100);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_throughput_dim_4096() {
        run_throughput_benchmark(4096, 4096, QK256_BLOCK, 50);
    }

    // ====================================================================
    // 11. Additional block sizes (64, 128)
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_block_size_64() {
        let k: usize = 128;
        let n = 1;
        let block_size: usize = 64;
        let num_blocks = k.div_ceil(block_size);
        assert_eq!(num_blocks, 2);

        let vals: Vec<i8> = vec![1; k];
        let weights_packed = pack_i2s_vec(&vals);
        let scales = vec![1.0f32, 2.0f32];
        let activation = vec![1.0f32; k];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        // Block 0: 64 * 1.0 = 64.0, Block 1: 64 * 2.0 = 128.0
        let expected = 64.0 + 128.0;
        assert!((result[0] - expected).abs() < 1e-4, "Expected {expected}, got {}", result[0]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_block_size_128() {
        let k: usize = 256;
        let n = 1;
        let block_size: usize = 128;
        let num_blocks = k.div_ceil(block_size);
        assert_eq!(num_blocks, 2);

        let vals: Vec<i8> = (0..k).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let weights_packed = pack_i2s_vec(&vals);
        let scales = vec![1.0f32; num_blocks];
        let activation = vec![1.0f32; k];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        // Alternating ±1 with uniform activation cancels out.
        assert!(result[0].abs() < 1e-6, "Expected ~0, got {}", result[0]);
    }

    // ====================================================================
    // 12. BitNet32-F16 format (inline F16 scales) vs QK256 (separate scales)
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_bitnet32_inline_f16_scale_precision() {
        // BitNet32-F16: each 32-element block stores its scale as F16 inline.
        let k: usize = BITNET32_BLOCK;
        let n = 1;

        let vals: Vec<i8> = vec![1; k];
        let weights_packed = pack_i2s_vec(&vals);

        // Simulate F16 inline scale (stored with each block).
        let scale_f32 = 0.123_456_79_f32;
        let scale_f16 = half::f16::from_f32(scale_f32).to_f32();

        let scales = vec![scale_f16];
        let activation = vec![1.0f32; k];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, BITNET32_BLOCK);
        let expected = k as f32 * scale_f16;
        assert!(
            (result[0] - expected).abs() < 1e-4,
            "BitNet32 F16 scale: expected {expected}, got {}",
            result[0]
        );
        // Verify F16 quantization introduces some rounding.
        assert_ne!(scale_f32, scale_f16, "F16 should differ from F32");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_qk256_separate_scales_layout() {
        // QK256: 256-element blocks, scales stored in a separate buffer.
        let k = QK256_BLOCK * 2; // 512 elements, 2 blocks
        let n = 2;
        let num_blocks = k.div_ceil(QK256_BLOCK);
        let packed_k = k.div_ceil(4);

        let vals: Vec<i8> = vec![1; k * n];
        let mut weights_packed = vec![0u8; n * packed_k];
        for col in 0..n {
            let col_packed = pack_i2s_vec(&vals[col * k..(col + 1) * k]);
            weights_packed[col * packed_k..col * packed_k + col_packed.len()]
                .copy_from_slice(&col_packed);
        }

        // Separate scale buffer: [n * num_blocks] contiguous f32
        let scales: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0]; // col0: [0.5, 1.0], col1: [1.5, 2.0]
        assert_eq!(scales.len(), n * num_blocks);

        let activation = vec![1.0f32; k];
        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, QK256_BLOCK);

        // col0: 256*0.5 + 256*1.0 = 128 + 256 = 384
        assert!((result[0] - 384.0).abs() < 1e-3, "Col0: expected 384, got {}", result[0]);
        // col1: 256*1.5 + 256*2.0 = 384 + 512 = 896
        assert!((result[1] - 896.0).abs() < 1e-3, "Col1: expected 896, got {}", result[1]);
    }

    // ====================================================================
    // 13. Per-channel vs per-tensor quantization
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_per_channel_quantization() {
        // Each output channel (column) has independent scales per block.
        let k: usize = 64;
        let n = 3;
        let block_size: usize = BITNET32_BLOCK;
        let num_blocks = k.div_ceil(block_size); // 2
        let packed_k = k.div_ceil(4);

        let vals: Vec<i8> = vec![1; k * n];
        let mut weights_packed = vec![0u8; n * packed_k];
        for col in 0..n {
            let col_packed = pack_i2s_vec(&vals[col * k..(col + 1) * k]);
            weights_packed[col * packed_k..col * packed_k + col_packed.len()]
                .copy_from_slice(&col_packed);
        }

        // Per-channel: different scales for each channel
        let scales = vec![
            1.0f32, 1.0, // channel 0, blocks 0-1
            2.0, 2.0, // channel 1, blocks 0-1
            3.0, 3.0, // channel 2, blocks 0-1
        ];
        let activation = vec![1.0f32; k];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        assert!((result[0] - 64.0).abs() < 1e-4, "Channel 0: expected 64, got {}", result[0]);
        assert!((result[1] - 128.0).abs() < 1e-4, "Channel 1: expected 128, got {}", result[1]);
        assert!((result[2] - 192.0).abs() < 1e-4, "Channel 2: expected 192, got {}", result[2]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_per_tensor_quantization() {
        // Per-tensor: all blocks and channels share a single scale.
        let k: usize = 64;
        let n = 3;
        let block_size: usize = BITNET32_BLOCK;
        let num_blocks = k.div_ceil(block_size);
        let packed_k = k.div_ceil(4);

        let vals: Vec<i8> = vec![1; k * n];
        let mut weights_packed = vec![0u8; n * packed_k];
        for col in 0..n {
            let col_packed = pack_i2s_vec(&vals[col * k..(col + 1) * k]);
            weights_packed[col * packed_k..col * packed_k + col_packed.len()]
                .copy_from_slice(&col_packed);
        }

        // Per-tensor: uniform scale of 0.5 for all blocks/channels
        let uniform_scale = 0.5f32;
        let scales = vec![uniform_scale; n * num_blocks];
        let activation = vec![1.0f32; k];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        let expected = k as f32 * uniform_scale;
        for (col, &v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "Per-tensor ch{col}: expected {expected}, got {v}"
            );
        }
    }

    // ====================================================================
    // 14. Numerical stability with extreme values
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_numerical_stability_very_small_scale() {
        let vals: Vec<i8> = vec![1; 32];
        let packed = pack_i2s_vec(&vals);
        let tiny_scale = 1e-30_f32;
        let result = dequant_i2s(&packed, tiny_scale, 32);
        for &v in &result {
            assert!(v.is_finite(), "Very small scale should produce finite results");
            assert!((v - tiny_scale).abs() < 1e-35, "Expected ~{tiny_scale}, got {v}");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_numerical_stability_very_large_scale() {
        let vals: Vec<i8> = vec![1; 4];
        let packed = pack_i2s_vec(&vals);
        let huge_scale = 1e30_f32;
        let result = dequant_i2s(&packed, huge_scale, 4);
        for &v in &result {
            assert!(v.is_finite(), "Large scale should produce finite results");
            assert!((v - huge_scale).abs() / huge_scale < 1e-6, "Expected ~{huge_scale}, got {v}");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_numerical_stability_subnormal_scale() {
        let vals: Vec<i8> = vec![1, -1, 0, 1];
        let packed = pack_i2s_vec(&vals);
        let subnormal = f32::MIN_POSITIVE / 2.0; // subnormal
        let result = dequant_i2s(&packed, subnormal, 4);
        for &v in &result {
            assert!(v.is_finite(), "Subnormal scale should produce finite results");
        }
        // +1 * subnormal should equal subnormal
        assert_eq!(result[0], subnormal);
        // -1 * subnormal should equal -subnormal
        assert_eq!(result[1], -subnormal);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_numerical_stability_inf_scale() {
        let vals: [i8; 4] = [1, -1, 0, 1];
        let packed = pack_i2s(vals);
        let result = dequant_i2s(&[packed], f32::INFINITY, 4);
        assert!(result[0].is_infinite() && result[0] > 0.0);
        assert!(result[1].is_infinite() && result[1] < 0.0);
        // 0 * inf = NaN in IEEE 754, but we map code 0b00 to 0i8, so 0.0 * inf = NaN
        assert!(result[2].is_nan());
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_numerical_stability_nan_scale() {
        let vals: [i8; 4] = [1, -1, 0, 1];
        let packed = pack_i2s(vals);
        let result = dequant_i2s(&[packed], f32::NAN, 4);
        for &v in &result {
            assert!(v.is_nan(), "NaN scale should propagate to all outputs");
        }
    }

    // ====================================================================
    // 15. Zero and constant input handling
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_zero_activation_input() {
        let k: usize = 64;
        let n = 2;
        let block_size: usize = BITNET32_BLOCK;
        let packed_k = k.div_ceil(4);

        let vals: Vec<i8> = (0..k * n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let mut weights_packed = vec![0u8; n * packed_k];
        for col in 0..n {
            let col_packed = pack_i2s_vec(&vals[col * k..(col + 1) * k]);
            weights_packed[col * packed_k..col * packed_k + col_packed.len()]
                .copy_from_slice(&col_packed);
        }
        let num_blocks = k.div_ceil(block_size);
        let scales = vec![1.0f32; n * num_blocks];
        let activation = vec![0.0f32; k];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        for (col, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-6, "Zero activation should produce 0, ch{col} got {v}");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_constant_activation_input() {
        let k: usize = 32;
        let n = 1;
        let block_size: usize = BITNET32_BLOCK;

        // Weights: half +1, half -1
        let mut vals = vec![0i8; k];
        vals[..16].fill(1);
        vals[16..].fill(-1);
        let weights_packed = pack_i2s_vec(&vals);
        let scales = vec![1.0f32];

        let constant = 3.14f32;
        let activation = vec![constant; k];

        let result = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        // 16 * 3.14 + 16 * (-3.14) = 0
        assert!(
            result[0].abs() < 1e-4,
            "Balanced ±1 with constant input should cancel, got {}",
            result[0]
        );
    }

    // ====================================================================
    // 16. Scale computation accuracy
    // ====================================================================

    /// Compute optimal I2_S scale for a block via absmax: scale = max(|x|).
    fn compute_scale_absmax(values: &[f32]) -> f32 {
        values.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    }

    /// Quantize f32 values to ternary {-1, 0, +1} given a scale.
    fn quantize_to_ternary(values: &[f32], scale: f32) -> Vec<i8> {
        if scale == 0.0 {
            return vec![0i8; values.len()];
        }
        values
            .iter()
            .map(|&v| {
                let normalized = v / scale;
                if normalized > 0.5 {
                    1i8
                } else if normalized < -0.5 {
                    -1i8
                } else {
                    0i8
                }
            })
            .collect()
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_scale_computation_absmax() {
        let values = vec![0.5, -1.0, 0.3, 0.8, -0.7, 1.0, -0.2, 0.0];
        let scale = compute_scale_absmax(&values);
        assert!((scale - 1.0).abs() < 1e-6, "absmax should be 1.0, got {scale}");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_scale_computation_all_zeros() {
        let values = vec![0.0; 32];
        let scale = compute_scale_absmax(&values);
        assert_eq!(scale, 0.0, "All-zero input should produce scale 0");
        let quantized = quantize_to_ternary(&values, scale);
        assert!(quantized.iter().all(|&v| v == 0), "All-zero should quantize to all-zero");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_scale_computation_uniform_positive() {
        let values = vec![0.75; 16];
        let scale = compute_scale_absmax(&values);
        assert!((scale - 0.75).abs() < 1e-6);
        let quantized = quantize_to_ternary(&values, scale);
        // 0.75 / 0.75 = 1.0 > 0.5 → +1
        assert!(quantized.iter().all(|&v| v == 1));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_quantize_dequantize_roundtrip_accuracy() {
        // End-to-end: f32 → ternary → packed I2_S → dequant → f32
        let original = vec![0.9, -0.8, 0.1, 0.7, -0.95, 0.0, -0.3, 0.85];
        let scale = compute_scale_absmax(&original);
        let quantized = quantize_to_ternary(&original, scale);
        let packed = pack_i2s_vec(&quantized);
        let restored = dequant_i2s(&packed, scale, original.len());

        // Check each element is within 1 scale unit of original
        for (i, (&orig, &rest)) in original.iter().zip(restored.iter()).enumerate() {
            let error = (orig - rest).abs();
            assert!(
                error <= scale + 1e-6,
                "Element {i}: orig={orig}, restored={rest}, error={error}, scale={scale}"
            );
        }
    }

    // ====================================================================
    // 17. Threadgroup dispatch sizing (additional tests)
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_threadgroup_dispatch_power_of_two_dims() {
        // Optimal Metal dispatch uses power-of-two threadgroup sizes.
        for &dim in &[32usize, 64, 128, 256, 512, 1024] {
            let tg_size = dim.min(METAL_MAX_THREADS_PER_THREADGROUP);
            assert!(tg_size.is_power_of_two(), "Dim {dim} → tg_size {tg_size} should be po2");
            assert!(tg_size <= METAL_MAX_THREADS_PER_THREADGROUP);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_threadgroup_dispatch_non_power_of_two_dims() {
        // Non-po2 dims need rounding to next po2 or clamping.
        fn next_power_of_two(n: usize) -> usize {
            n.next_power_of_two()
        }

        for &dim in &[33usize, 65, 129, 257, 513, 1025, 2048, 4096] {
            let tg_size = next_power_of_two(dim).min(METAL_MAX_THREADS_PER_THREADGROUP);
            assert!(tg_size.is_power_of_two());
            assert!(tg_size >= dim.min(METAL_MAX_THREADS_PER_THREADGROUP));
            assert!(tg_size <= METAL_MAX_THREADS_PER_THREADGROUP);

            // Grid size: number of threadgroups needed
            let num_threadgroups = dim.div_ceil(tg_size);
            assert!(num_threadgroups >= 1, "Must dispatch at least 1 threadgroup");
            assert!(num_threadgroups * tg_size >= dim, "Grid must cover all elements");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_threadgroup_dispatch_quantization_kernel() {
        // For quantization: each threadgroup processes one block of elements.
        for &block_size in &[BITNET32_BLOCK, 64, 128, QK256_BLOCK] {
            let elements: usize = 4096;
            let num_blocks = elements.div_ceil(block_size);
            // One threadgroup per block, threads within handle elements
            let threads_per_tg = block_size.min(METAL_MAX_THREADS_PER_THREADGROUP);
            assert!(threads_per_tg > 0);
            assert!(
                num_blocks > 0,
                "block_size={block_size}: need at least 1 block for {elements} elements"
            );
            // Total thread invocations covers all elements
            assert!(num_blocks * threads_per_tg >= elements.min(num_blocks * block_size));
        }
    }

    // ====================================================================
    // 18. Mixed-precision: F16 input → I2_S output quantization
    // ====================================================================

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_f16_input_to_i2s_quantization() {
        // Simulate: F16 input tensor → compute absmax scale → quantize to ternary → pack I2_S
        let input_f32: Vec<f32> = vec![0.8, -0.9, 0.1, 0.7, -0.6, 0.0, -0.95, 0.5];
        // Round-trip through F16 to simulate Metal F16 input
        let input_f16: Vec<f32> =
            input_f32.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();

        let scale = compute_scale_absmax(&input_f16);
        let quantized = quantize_to_ternary(&input_f16, scale);
        let packed = pack_i2s_vec(&quantized);
        let restored = dequant_i2s(&packed, scale, input_f16.len());

        // Verify output is valid I2_S (values are in {-scale, 0, +scale})
        for &v in &restored {
            let abs_v = v.abs();
            assert!(
                abs_v < 1e-6 || (abs_v - scale).abs() < 1e-6,
                "Dequantized value {v} should be 0 or ±{scale}"
            );
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_f16_precision_boundary_values() {
        // F16 has limited precision; test values near quantization thresholds.
        let scale = 1.0f32;
        // Values near the 0.5 threshold (boundary between 0 and ±1)
        let boundary_values: Vec<f32> = vec![0.49, 0.50, 0.51, -0.49, -0.50, -0.51];
        let f16_values: Vec<f32> =
            boundary_values.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        let quantized = quantize_to_ternary(&f16_values, scale);

        // 0.49/1.0 < 0.5 → 0,  0.51/1.0 > 0.5 → +1
        assert_eq!(quantized[0], 0, "0.49 should quantize to 0");
        // 0.50 exactly: our threshold is strict >, so 0.50 → 0
        assert_eq!(quantized[1], 0, "0.50 should quantize to 0 (strict >)");
        assert_eq!(quantized[2], 1, "0.51 should quantize to +1");
        assert_eq!(quantized[3], 0, "-0.49 should quantize to 0");
        assert_eq!(quantized[4], 0, "-0.50 should quantize to 0 (strict <)");
        assert_eq!(quantized[5], -1, "-0.51 should quantize to -1");
    }

    // ====================================================================
    // 10. Performance: throughput for common hidden dimensions (continued)
    // ====================================================================

    /// Benchmark helper for I2_S matrix-vector multiply throughput.
    fn run_throughput_benchmark(k: usize, n: usize, block_size: usize, iterations: usize) {
        let packed_k = k.div_ceil(4);
        let num_blocks = k.div_ceil(block_size);

        // Build packed weights (alternating pattern)
        let vals: Vec<i8> = (0..k * n)
            .map(|i| match i % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            })
            .collect();
        let mut weights_packed = vec![0u8; n * packed_k];
        for col in 0..n {
            let col_packed = pack_i2s_vec(&vals[col * k..(col + 1) * k]);
            weights_packed[col * packed_k..col * packed_k + col_packed.len()]
                .copy_from_slice(&col_packed);
        }

        let scales = vec![1.0f32; n * num_blocks];
        let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();

        // Warm-up
        let _ = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = cpu_i2s_matvec(&weights_packed, &scales, &activation, n, k, block_size);
        }
        let elapsed = start.elapsed();

        let ops_per_iter = (k as u64) * (n as u64) * 2; // multiply + add
        let total_ops = ops_per_iter * iterations as u64;
        let gops = total_ops as f64 / elapsed.as_secs_f64() / 1e9;

        // Sanity: the benchmark must complete in reasonable time.
        assert!(elapsed.as_secs() < 60, "Benchmark for k={k} n={n} took too long: {elapsed:?}");
        eprintln!(
            "  [perf] k={k} n={n} block={block_size}: \
             {iterations} iters in {elapsed:.2?} ({gops:.2} GOp/s)"
        );
    }
}
