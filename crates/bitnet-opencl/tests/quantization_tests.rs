//! Integration tests for GPU quantization kernels (CPU reference path).
//!
//! Every test is `#[ignore]` because the full OpenCL runtime is required
//! for GPU execution; the CPU reference implementations are exercised here
//! to validate algorithmic correctness.

use bitnet_opencl::quantization::{GpuQuantizer, QuantError};

// ── helpers ────────────────────────────────────────────────────────────

fn make_quantizer() -> GpuQuantizer {
    GpuQuantizer::new()
}

/// Generate a repeating ternary pattern of length `n`.
fn ternary_pattern(n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| match i % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect()
}

// ── I2_S tests ─────────────────────────────────────────────────────────

#[test]
#[ignore = "requires OpenCL runtime"]
fn i2s_pack_unpack_round_trip() {
    let q = make_quantizer();
    let values: Vec<i32> = vec![-1, 0, 1, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 1, 0, -1];
    let packed = q.pack_i2s(&values).unwrap();
    assert_eq!(packed.len(), 1);
    let unpacked = q.unpack_i2s(&packed, 1.0, 16);
    for (i, (&orig, &got)) in values.iter().zip(unpacked.iter()).enumerate() {
        assert_eq!(orig as f32, got, "mismatch at index {i}: expected {orig}, got {got}");
    }
}

#[test]
#[ignore = "requires OpenCL runtime"]
fn i2s_pack_unpack_with_scale() {
    let q = make_quantizer();
    let values = vec![1, -1, 0, 1, 0, 0, -1, 1, 1, 0, -1, 0, 1, -1, 0, 1];
    let packed = q.pack_i2s(&values).unwrap();
    let scale = 2.5_f32;
    let unpacked = q.unpack_i2s(&packed, scale, 16);
    for (i, (&orig, &got)) in values.iter().zip(unpacked.iter()).enumerate() {
        let expected = orig as f32 * scale;
        assert!((expected - got).abs() < 1e-6, "index {i}: expected {expected}, got {got}");
    }
}

#[test]
#[ignore = "requires OpenCL runtime"]
fn i2s_pack_rejects_non_ternary() {
    let q = make_quantizer();
    let bad = vec![0; 15].into_iter().chain(std::iter::once(2)).collect::<Vec<_>>();
    assert!(matches!(q.pack_i2s(&bad), Err(QuantError::InvalidTernary { value: 2 })));
}

#[test]
#[ignore = "requires OpenCL runtime"]
fn i2s_pack_rejects_misaligned_length() {
    let q = make_quantizer();
    assert!(matches!(q.pack_i2s(&[0; 15]), Err(QuantError::Alignment { len: 15, alignment: 16 })));
}

#[test]
#[ignore = "requires OpenCL runtime"]
fn i2s_all_zeros() {
    let q = make_quantizer();
    let values = vec![0_i32; 16];
    let packed = q.pack_i2s(&values).unwrap();
    assert_eq!(packed, vec![0u32]);
    let unpacked = q.unpack_i2s(&packed, 1.0, 16);
    assert!(unpacked.iter().all(|&v| v == 0.0));
}

#[test]
#[ignore = "requires OpenCL runtime"]
fn i2s_all_ones() {
    let q = make_quantizer();
    let values = vec![1_i32; 16];
    let packed = q.pack_i2s(&values).unwrap();
    let unpacked = q.unpack_i2s(&packed, 1.0, 16);
    assert!(unpacked.iter().all(|&v| (v - 1.0).abs() < 1e-6));
}

#[test]
#[ignore = "requires OpenCL runtime"]
fn i2s_all_neg_ones() {
    let q = make_quantizer();
    let values = vec![-1_i32; 16];
    let packed = q.pack_i2s(&values).unwrap();
    let unpacked = q.unpack_i2s(&packed, 1.0, 16);
    assert!(unpacked.iter().all(|&v| (v + 1.0).abs() < 1e-6));
}

#[test]
#[ignore = "requires OpenCL runtime"]
fn i2s_multi_word_round_trip() {
    let q = make_quantizer();
    let values = ternary_pattern(64);
    let packed = q.pack_i2s(&values).unwrap();
    assert_eq!(packed.len(), 4);
    let unpacked = q.unpack_i2s(&packed, 1.0, 64);
    for (i, (&orig, &got)) in values.iter().zip(unpacked.iter()).enumerate() {
        assert_eq!(orig as f32, got, "mismatch at index {i}");
    }
}

// ── QK256 tests ────────────────────────────────────────────────────────

#[test]
#[ignore = "requires OpenCL runtime"]
fn qk256_dequant_matches_cpu_reference() {
    let q = make_quantizer();
    // Build one block of known ternary values
    let values = ternary_pattern(256);
    let packed_from_i2s = {
        let mut words = Vec::with_capacity(16);
        for chunk in values.chunks_exact(16) {
            let mut word: u32 = 0;
            for (i, &v) in chunk.iter().enumerate() {
                let bits: u32 = if v == -1 { 2 } else { v as u32 };
                word |= (bits & 0x3) << (i * 2);
            }
            words.push(word);
        }
        words
    };
    let scale = 0.75_f32;
    let out = q.dequant_qk256(&packed_from_i2s, &[scale], 1).unwrap();
    assert_eq!(out.len(), 256);
    for (i, (&orig, &got)) in values.iter().zip(out.iter()).enumerate() {
        let expected = orig as f32 * scale;
        assert!(
            (expected - got).abs() < 1e-6,
            "block 0, index {i}: expected {expected}, got {got}"
        );
    }
}

#[test]
#[ignore = "requires OpenCL runtime"]
fn qk256_quantize_dequantize_round_trip() {
    let q = make_quantizer();
    // Strong ternary-like input: values close to {-1, 0, 1} × scale
    let scale = 3.0_f32;
    let input: Vec<f32> = (0..256)
        .map(|i| match i % 3 {
            0 => -scale,
            1 => 0.0,
            _ => scale,
        })
        .collect();

    let (packed, scales) = q.quant_qk256(&input).unwrap();
    assert_eq!(scales.len(), 1);
    assert!((scales[0] - scale).abs() < 1e-6);

    let output = q.dequant_qk256(&packed, &scales, 1).unwrap();
    for (i, (&orig, &got)) in input.iter().zip(output.iter()).enumerate() {
        assert!((orig - got).abs() < 1e-4, "index {i}: expected {orig}, got {got}");
    }
}

#[test]
#[ignore = "requires OpenCL runtime"]
fn qk256_block_boundary() {
    let q = make_quantizer();
    // Two blocks: first block all-1, second block all-(-1)
    let mut input = vec![1.0_f32; 256];
    input.extend(vec![-1.0_f32; 256]);
    let (packed, scales) = q.quant_qk256(&input).unwrap();
    assert_eq!(scales.len(), 2);
    let output = q.dequant_qk256(&packed, &scales, 2).unwrap();
    assert_eq!(output.len(), 512);
    assert!(output[0..256].iter().all(|&v| (v - 1.0).abs() < 1e-4));
    assert!(output[256..512].iter().all(|&v| (v + 1.0).abs() < 1e-4));
}

#[test]
#[ignore = "requires OpenCL runtime"]
fn qk256_rejects_misaligned() {
    let q = make_quantizer();
    let bad_input = vec![0.0_f32; 100];
    assert!(matches!(
        q.quant_qk256(&bad_input),
        Err(QuantError::Alignment { len: 100, alignment: 256 })
    ));
}

// ── TL1 tests ──────────────────────────────────────────────────────────

#[test]
#[ignore = "requires OpenCL runtime"]
fn tl1_dequant_matches_lut() {
    let q = make_quantizer();
    let lut: [f32; 3] = [-1.0, 0.0, 1.0];
    // Encode: 4 values per byte → [0b00_01_10_11] won't happen (only 0-2)
    // Byte encodes indices [0, 1, 2, 0] → bits 00_01_10_00 = 0x24
    let packed: Vec<u8> = vec![0b00_10_01_00];
    let out = q.dequant_tl1(&packed, &lut, 4);
    assert_eq!(out, vec![-1.0, 0.0, 1.0, -1.0]);
}

#[test]
#[ignore = "requires OpenCL runtime"]
fn tl1_dequant_multiple_bytes() {
    let q = make_quantizer();
    let lut: [f32; 3] = [0.5, -0.5, 1.5];
    // Two bytes → 8 values
    let packed: Vec<u8> = vec![
        0b00_01_10_00, // indices: 0, 2, 1, 0
        0b10_00_01_10, // indices: 2, 1, 0, 2
    ];
    let out = q.dequant_tl1(&packed, &lut, 8);
    assert_eq!(out, vec![0.5, 1.5, -0.5, 0.5, 1.5, -0.5, 0.5, 1.5]);
}

// ── TL2 tests ──────────────────────────────────────────────────────────

#[test]
#[ignore = "requires OpenCL runtime"]
fn tl2_dequant_matches_paired_lookup() {
    let q = make_quantizer();
    let lut: [[f32; 2]; 3] = [
        [-1.0, -2.0], // index 0
        [0.0, 0.0],   // index 1
        [1.0, 2.0],   // index 2
    ];
    // gid=0: byte_idx=0, shift=0 → bits0=(byte>>0)&3, bits1=(byte>>2)&3
    // gid=1: byte_idx=0, shift=4 → bits0=(byte>>4)&3, bits1=(byte>>6)&3
    // Byte: 0b10_01_00_10 → shift=0: bits0=2,bits1=0; shift=4: bits0=1,bits1=2
    let packed: Vec<u8> = vec![0b10_01_00_10];
    let out = q.dequant_tl2(&packed, &lut, 4);
    // gid=0: bits0=2→lut[2].x=1.0, bits1=0→lut[0].y=-2.0
    // gid=1: bits0=1→lut[1].x=0.0, bits1=2→lut[2].y=2.0
    assert_eq!(out, vec![1.0, -2.0, 0.0, 2.0]);
}

// ── Property-style tests ───────────────────────────────────────────────

#[test]
#[ignore = "requires OpenCL runtime"]
fn property_random_ternary_pack_unpack() {
    let q = make_quantizer();
    // Deterministic pseudo-random ternary vector
    let values: Vec<i32> = (0..256)
        .map(|i| {
            let h = ((i as u64).wrapping_mul(2654435761)) % 3;
            h as i32 - 1 // maps {0,1,2} → {-1,0,1}
        })
        .collect();
    let packed = q.pack_i2s(&values).unwrap();
    let unpacked = q.unpack_i2s(&packed, 1.0, 256);
    for (i, (&orig, &got)) in values.iter().zip(unpacked.iter()).enumerate() {
        assert_eq!(orig as f32, got, "property test mismatch at {i}");
    }
}
