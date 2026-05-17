#![cfg(feature = "cpu")]
#![allow(clippy::needless_range_loop)]
//! AVX2 vs scalar parity tests for quantized matrix multiplication kernels.
//!
//! Verifies that `simd_quantized_matmul` (which has runtime AVX2 dispatch
//! via `block_accum_avx2`) produces identical results to the pure-scalar
//! reference implementations in `quantized_matmul`.
//!
//! Dimensions are chosen to exercise:
//! - SIMD body (multiples of 8 for AVX2 lane width)
//! - Scalar tails (non-multiples of 8)
//! - Block boundaries (k values crossing block_size boundaries)
//! - Both block sizes: 32 (BitNet32-F16) and 256 (QK256/GGML)
//!
//! All tests are pure-math — no model files required.

use bitnet_kernels::cpu::quantized_matmul;
use bitnet_kernels::cpu::simd_quantized_matmul;

#[path = "common/avx2_i8_parity.rs"]
mod avx2_i8_parity;
#[path = "common/avx2_parity.rs"]
mod avx2_parity;
use avx2_i8_parity::pseudo_rand_i8;
use avx2_parity::{assert_vec_parity, pseudo_rand};

// ── Tolerance constants ────────────────────────────────────────────────
//
// AVX2 FMA vs scalar multiply-then-add can differ by up to ~0.5 ULP per
// op.  With thousands of accumulations the absolute error can reach ~1e-4
// on moderate matrices.  We use the same budget as the other AVX2 parity
// test suites in this crate.

/// Absolute tolerance for matmul accumulation.
const ABS_TOL: f32 = 1e-4;
/// Relative tolerance for matmul accumulation.
const REL_TOL: f32 = 1e-3;

/// Deterministic pseudo-random ternary values in {-1, 0, +1}.
fn pseudo_rand_ternary(len: usize, seed: u64) -> Vec<i8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            match state % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            }
        })
        .collect()
}

/// Pack a ternary weight matrix (k×n row-major) into I2_S bytes
/// with column-major packing and the given scales.
fn pack_weight_matrix(
    weights: &[i8],
    k: usize,
    n: usize,
    block_size: usize,
) -> (Vec<u8>, Vec<f32>) {
    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);
    let mut packed = vec![0u8; packed_k * n];
    for col in 0..n {
        for row in 0..k {
            let val = weights[row * n + col];
            let code: u8 = match val {
                1 => 0b01,
                -1 => 0b11,
                _ => 0b00,
            };
            let byte_idx = col * packed_k + row / 4;
            let bit_off = (row % 4) * 2;
            packed[byte_idx] |= code << bit_off;
        }
    }
    let scales = vec![1.0f32; n * num_blocks_k];
    (packed, scales)
}

/// Pack with non-uniform scales for more thorough testing.
fn pack_weight_matrix_with_scales(
    weights: &[i8],
    k: usize,
    n: usize,
    block_size: usize,
    scale_seed: u64,
) -> (Vec<u8>, Vec<f32>) {
    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);
    let mut packed = vec![0u8; packed_k * n];
    for col in 0..n {
        for row in 0..k {
            let val = weights[row * n + col];
            let code: u8 = match val {
                1 => 0b01,
                -1 => 0b11,
                _ => 0b00,
            };
            let byte_idx = col * packed_k + row / 4;
            let bit_off = (row % 4) * 2;
            packed[byte_idx] |= code << bit_off;
        }
    }
    // Non-uniform scales in [0.5, 1.5]
    let raw_scales = pseudo_rand(n * num_blocks_k, scale_seed);
    let scales: Vec<f32> = raw_scales.iter().map(|&s| s * 0.5 + 1.0).collect();
    (packed, scales)
}

// ── 1. block_quantized_matmul vs i2s_matmul_f32 (scalar reference) ────
//
// block_quantized_matmul is the AVX2-dispatching kernel; i2s_matmul_f32
// is the pure-scalar reference. These should produce identical results
// within FMA tolerance.

#[test]
fn parity_block_vs_scalar_4x4_block32() {
    let (m, n, k, bs) = (4, 4, 4, 32);
    let w = pseudo_rand_ternary(k * n, 42);
    let act = pseudo_rand(m * k, 123);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_simd = vec![0.0f32; m * n];

    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();
    simd_quantized_matmul::block_quantized_matmul(
        &act,
        &packed,
        &scales,
        &mut out_simd,
        m,
        n,
        k,
        bs,
    )
    .unwrap();

    assert_vec_parity(&out_simd, &out_scalar, ABS_TOL, REL_TOL, "4x4 bs=32");
}

#[test]
fn parity_block_vs_scalar_16x16_block32() {
    let (m, n, k, bs) = (16, 16, 16, 32);
    let w = pseudo_rand_ternary(k * n, 100);
    let act = pseudo_rand(m * k, 200);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_simd = vec![0.0f32; m * n];

    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();
    simd_quantized_matmul::block_quantized_matmul(
        &act,
        &packed,
        &scales,
        &mut out_simd,
        m,
        n,
        k,
        bs,
    )
    .unwrap();

    assert_vec_parity(&out_simd, &out_scalar, ABS_TOL, REL_TOL, "16x16 bs=32");
}

#[test]
fn parity_block_vs_scalar_8x32_block256() {
    let (m, n, k, bs) = (8, 32, 256, 256);
    let w = pseudo_rand_ternary(k * n, 300);
    let act = pseudo_rand(m * k, 400);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_simd = vec![0.0f32; m * n];

    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();
    simd_quantized_matmul::block_quantized_matmul(
        &act,
        &packed,
        &scales,
        &mut out_simd,
        m,
        n,
        k,
        bs,
    )
    .unwrap();

    assert_vec_parity(&out_simd, &out_scalar, ABS_TOL, REL_TOL, "8x32 bs=256");
}

/// Non-power-of-2 k to exercise scalar tail in AVX2 inner loop.
#[test]
fn parity_block_vs_scalar_tail_k13() {
    let (m, n, k, bs) = (4, 8, 13, 32);
    let w = pseudo_rand_ternary(k * n, 500);
    let act = pseudo_rand(m * k, 600);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_simd = vec![0.0f32; m * n];

    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();
    simd_quantized_matmul::block_quantized_matmul(
        &act,
        &packed,
        &scales,
        &mut out_simd,
        m,
        n,
        k,
        bs,
    )
    .unwrap();

    assert_vec_parity(&out_simd, &out_scalar, ABS_TOL, REL_TOL, "tail k=13");
}

/// k crosses block boundary: k=48, block_size=32 → 2 blocks.
#[test]
fn parity_block_vs_scalar_multi_block() {
    let (m, n, k, bs) = (4, 8, 48, 32);
    let w = pseudo_rand_ternary(k * n, 700);
    let act = pseudo_rand(m * k, 800);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_simd = vec![0.0f32; m * n];

    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();
    simd_quantized_matmul::block_quantized_matmul(
        &act,
        &packed,
        &scales,
        &mut out_simd,
        m,
        n,
        k,
        bs,
    )
    .unwrap();

    assert_vec_parity(&out_simd, &out_scalar, ABS_TOL, REL_TOL, "multi-block k=48 bs=32");
}

/// Non-uniform scales stress FMA accumulation path differently.
#[test]
fn parity_block_vs_scalar_nonuniform_scales() {
    let (m, n, k, bs) = (8, 16, 64, 32);
    let w = pseudo_rand_ternary(k * n, 900);
    let act = pseudo_rand(m * k, 1000);
    let (packed, scales) = pack_weight_matrix_with_scales(&w, k, n, bs, 1100);

    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_simd = vec![0.0f32; m * n];

    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();
    simd_quantized_matmul::block_quantized_matmul(
        &act,
        &packed,
        &scales,
        &mut out_simd,
        m,
        n,
        k,
        bs,
    )
    .unwrap();

    assert_vec_parity(&out_simd, &out_scalar, ABS_TOL, REL_TOL, "non-uniform scales");
}

// ── 2. fused_dequant_matmul vs i2s_matmul_f32 ─────────────────────────

#[test]
fn parity_fused_vs_scalar_8x8_block32() {
    let (m, n, k, bs) = (8, 8, 32, 32);
    let w = pseudo_rand_ternary(k * n, 1200);
    let act = pseudo_rand(m * k, 1300);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_fused = vec![0.0f32; m * n];

    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();
    simd_quantized_matmul::fused_dequant_matmul(
        &act,
        &packed,
        &scales,
        &mut out_fused,
        m,
        n,
        k,
        bs,
    )
    .unwrap();

    assert_vec_parity(&out_fused, &out_scalar, ABS_TOL, REL_TOL, "fused 8x8 bs=32");
}

#[test]
fn parity_fused_vs_scalar_4x16_block256() {
    let (m, n, k, bs) = (4, 16, 256, 256);
    let w = pseudo_rand_ternary(k * n, 1400);
    let act = pseudo_rand(m * k, 1500);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_fused = vec![0.0f32; m * n];

    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();
    simd_quantized_matmul::fused_dequant_matmul(
        &act,
        &packed,
        &scales,
        &mut out_fused,
        m,
        n,
        k,
        bs,
    )
    .unwrap();

    assert_vec_parity(&out_fused, &out_scalar, ABS_TOL, REL_TOL, "fused 4x16 bs=256");
}

// ── 3. tiled_quantized_matmul vs i2s_matmul_f32 ───────────────────────

#[test]
fn parity_tiled_vs_scalar_16x16_block32() {
    let (m, n, k, bs) = (16, 16, 32, 32);
    let w = pseudo_rand_ternary(k * n, 1600);
    let act = pseudo_rand(m * k, 1700);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_tiled = vec![0.0f32; m * n];

    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();
    simd_quantized_matmul::tiled_quantized_matmul(
        &act,
        &packed,
        &scales,
        &mut out_tiled,
        m,
        n,
        k,
        bs,
        &simd_quantized_matmul::QuantizedTileConfig::DEFAULT,
    )
    .unwrap();

    assert_vec_parity(&out_tiled, &out_scalar, ABS_TOL, REL_TOL, "tiled 16x16 bs=32");
}

/// Non-tile-aligned dimensions to exercise remainder handling.
#[test]
fn parity_tiled_vs_scalar_remainder() {
    let (m, n, k, bs) = (13, 11, 48, 32);
    let w = pseudo_rand_ternary(k * n, 1800);
    let act = pseudo_rand(m * k, 1900);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_tiled = vec![0.0f32; m * n];

    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();
    simd_quantized_matmul::tiled_quantized_matmul(
        &act,
        &packed,
        &scales,
        &mut out_tiled,
        m,
        n,
        k,
        bs,
        &simd_quantized_matmul::QuantizedTileConfig::SMALL,
    )
    .unwrap();

    assert_vec_parity(&out_tiled, &out_scalar, ABS_TOL, REL_TOL, "tiled remainder 13x11");
}

// ── 4. mixed_precision_matmul vs scalar + bias ─────────────────────────

#[test]
fn parity_mixed_precision_no_bias() {
    let (m, n, k, bs) = (4, 8, 32, 32);
    let w = pseudo_rand_ternary(k * n, 2000);
    let act = pseudo_rand(m * k, 2100);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_mixed = vec![0.0f32; m * n];

    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();
    simd_quantized_matmul::mixed_precision_matmul(
        &act,
        &packed,
        &scales,
        None,
        &mut out_mixed,
        m,
        n,
        k,
        bs,
    )
    .unwrap();

    assert_vec_parity(&out_mixed, &out_scalar, ABS_TOL, REL_TOL, "mixed no-bias");
}

#[test]
fn parity_mixed_precision_with_bias() {
    let (m, n, k, bs) = (4, 8, 32, 32);
    let w = pseudo_rand_ternary(k * n, 2200);
    let act = pseudo_rand(m * k, 2300);
    let bias = pseudo_rand(n, 2400);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_mixed = vec![0.0f32; m * n];

    // Scalar reference: matmul + manual bias
    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();
    for row in 0..m {
        for col in 0..n {
            out_scalar[row * n + col] += bias[col];
        }
    }

    simd_quantized_matmul::mixed_precision_matmul(
        &act,
        &packed,
        &scales,
        Some(&bias),
        &mut out_mixed,
        m,
        n,
        k,
        bs,
    )
    .unwrap();

    assert_vec_parity(&out_mixed, &out_scalar, ABS_TOL, REL_TOL, "mixed with-bias");
}

// ── 5. batched_quantized_matmul vs per-head scalar ─────────────────────

#[test]
fn parity_batched_4heads() {
    let (num_heads, m, n, k, bs) = (4, 4, 8, 32, 32);
    let w = pseudo_rand_ternary(k * n, 2500);
    let act = pseudo_rand(num_heads * m * k, 2600);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    // Scalar reference: run i2s_matmul_f32 per head.
    let mut out_scalar = vec![0.0f32; num_heads * m * n];
    for head in 0..num_heads {
        let act_slice = &act[head * m * k..(head + 1) * m * k];
        let out_slice = &mut out_scalar[head * m * n..(head + 1) * m * n];
        quantized_matmul::i2s_matmul_f32(act_slice, &packed, &scales, out_slice, m, n, k, bs)
            .unwrap();
    }

    let mut out_batched = vec![0.0f32; num_heads * m * n];
    simd_quantized_matmul::batched_quantized_matmul(
        &act,
        &packed,
        &scales,
        &mut out_batched,
        num_heads,
        m,
        n,
        k,
        bs,
    )
    .unwrap();

    assert_vec_parity(&out_batched, &out_scalar, ABS_TOL, REL_TOL, "batched 4 heads");
}

// ── 6. int2_int8_matmul self-consistency ────────────────────────────────
//
// int2_int8_matmul takes i8 activations (integer domain), so we compare
// two runs with the same inputs to verify determinism, and compare against
// a manual i8-domain reference.

/// Manual i8-domain reference matmul for int2_int8 parity.
fn naive_int2_int8_matmul(
    act: &[i8],
    weights_packed: &[u8],
    scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Vec<f32> {
    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);
    let mut out = vec![0.0f32; m * n];

    for row in 0..m {
        for col in 0..n {
            let mut acc_f32 = 0.0f32;
            for blk in 0..num_blocks_k {
                let blk_start = blk * block_size;
                let blk_end = (blk_start + block_size).min(k);
                let scale = scales[col * num_blocks_k + blk];
                let mut acc_i32 = 0i32;
                for idx in blk_start..blk_end {
                    let byte_idx = col * packed_k + idx / 4;
                    let bit_off = (idx % 4) * 2;
                    let bits = (weights_packed[byte_idx] >> bit_off) & 0x03;
                    let w: i8 = match bits {
                        0b01 => 1,
                        0b11 => -1,
                        _ => 0,
                    };
                    acc_i32 += act[row * k + idx] as i32 * w as i32;
                }
                acc_f32 += acc_i32 as f32 * scale;
            }
            out[row * n + col] = acc_f32;
        }
    }
    out
}

#[test]
fn parity_int2_int8_vs_reference_4x8() {
    let (m, n, k, bs) = (4, 8, 32, 32);
    let w = pseudo_rand_ternary(k * n, 2700);
    let act = pseudo_rand_i8(m * k, 2800);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let expected = naive_int2_int8_matmul(&act, &packed, &scales, m, n, k, bs);

    let mut out_int2 = vec![0.0f32; m * n];
    simd_quantized_matmul::int2_int8_matmul(&act, &packed, &scales, &mut out_int2, m, n, k, bs)
        .unwrap();

    assert_vec_parity(&out_int2, &expected, ABS_TOL, REL_TOL, "int2_int8 4x8");
}

#[test]
fn parity_int2_int8_vs_reference_32x32() {
    let (m, n, k, bs) = (32, 32, 64, 32);
    let w = pseudo_rand_ternary(k * n, 2900);
    let act = pseudo_rand_i8(m * k, 3000);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let expected = naive_int2_int8_matmul(&act, &packed, &scales, m, n, k, bs);

    let mut out_int2 = vec![0.0f32; m * n];
    simd_quantized_matmul::int2_int8_matmul(&act, &packed, &scales, &mut out_int2, m, n, k, bs)
        .unwrap();

    assert_vec_parity(&out_int2, &expected, ABS_TOL, REL_TOL, "int2_int8 32x32");
}

// ── 7. dequantize_and_matmul vs block_quantized_matmul (cross-kernel) ─

#[test]
fn parity_dequant_vs_block_16x8_block256() {
    let (m, n, k, bs) = (16, 8, 256, 256);
    let w = pseudo_rand_ternary(k * n, 3100);
    let act = pseudo_rand(m * k, 3200);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let mut out_dequant = vec![0.0f32; m * n];
    quantized_matmul::dequantize_and_matmul(&act, &packed, &scales, &mut out_dequant, m, n, k, bs)
        .unwrap();

    let mut out_block = vec![0.0f32; m * n];
    simd_quantized_matmul::block_quantized_matmul(
        &act,
        &packed,
        &scales,
        &mut out_block,
        m,
        n,
        k,
        bs,
    )
    .unwrap();

    assert_vec_parity(&out_block, &out_dequant, ABS_TOL, REL_TOL, "dequant vs block 16x8 bs=256");
}

// ── 8. Stress: large matrix, QK256 block size, non-uniform scales ─────

#[test]
fn parity_stress_large_qk256() {
    let (m, n, k, bs) = (4, 64, 512, 256);
    let w = pseudo_rand_ternary(k * n, 3300);
    let act = pseudo_rand(m * k, 3400);
    let (packed, scales) = pack_weight_matrix_with_scales(&w, k, n, bs, 3500);

    let mut out_scalar = vec![0.0f32; m * n];
    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();

    let mut out_block = vec![0.0f32; m * n];
    simd_quantized_matmul::block_quantized_matmul(
        &act,
        &packed,
        &scales,
        &mut out_block,
        m,
        n,
        k,
        bs,
    )
    .unwrap();

    assert_vec_parity(&out_block, &out_scalar, ABS_TOL, REL_TOL, "stress large QK256");
}

#[test]
fn parity_stress_large_bitnet32() {
    let (m, n, k, bs) = (4, 64, 128, 32);
    let w = pseudo_rand_ternary(k * n, 3600);
    let act = pseudo_rand(m * k, 3700);
    let (packed, scales) = pack_weight_matrix_with_scales(&w, k, n, bs, 3800);

    let mut out_scalar = vec![0.0f32; m * n];
    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();

    let mut out_block = vec![0.0f32; m * n];
    simd_quantized_matmul::block_quantized_matmul(
        &act,
        &packed,
        &scales,
        &mut out_block,
        m,
        n,
        k,
        bs,
    )
    .unwrap();

    assert_vec_parity(&out_block, &out_scalar, ABS_TOL, REL_TOL, "stress large BitNet32");
}

// ── 9. DequantWorkspace parity ─────────────────────────────────────────

#[test]
fn parity_dequant_workspace_vs_scalar() {
    let (m, n, k, bs) = (8, 16, 64, 32);
    let w = pseudo_rand_ternary(k * n, 3900);
    let act = pseudo_rand(m * k, 4000);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let mut out_scalar = vec![0.0f32; m * n];
    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_scalar, m, n, k, bs).unwrap();

    let mut out_ws = vec![0.0f32; m * n];
    let mut ws = quantized_matmul::DequantWorkspace::new(k, n);
    quantized_matmul::dequantize_and_matmul_into(
        &act,
        &packed,
        &scales,
        &mut out_ws,
        m,
        n,
        k,
        bs,
        &mut ws,
    )
    .unwrap();

    assert_vec_parity(&out_ws, &out_scalar, ABS_TOL, REL_TOL, "DequantWorkspace vs scalar");
}

// ── 10. i2s_matmul_blocked (scalar) vs i2s_matmul_f32 (scalar) ────────
//
// Both are scalar, but the blocked variant uses a different loop
// structure (outer loop over blocks). They must agree exactly.

#[test]
fn parity_blocked_vs_linear_scalar() {
    let (m, n, k, bs) = (8, 16, 64, 32);
    let w = pseudo_rand_ternary(k * n, 4100);
    let act = pseudo_rand(m * k, 4200);
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);

    let mut out_linear = vec![0.0f32; m * n];
    quantized_matmul::i2s_matmul_f32(&act, &packed, &scales, &mut out_linear, m, n, k, bs).unwrap();

    let mut out_blocked = vec![0.0f32; m * n];
    quantized_matmul::i2s_matmul_blocked(&act, &packed, &scales, &mut out_blocked, m, n, k, bs)
        .unwrap();

    // Both are scalar — should be exact match.
    assert_vec_parity(&out_blocked, &out_linear, 1e-6, 1e-6, "blocked vs linear scalar");
}
