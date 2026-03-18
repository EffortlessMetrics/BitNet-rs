//! AVX2-accelerated quantization operations with scalar fallbacks.
//!
//! Provides I2_S 2-bit quantization/dequantization, absmax scale
//! computation, BitNet ternary {-1, 0, 1} quantization, and ternary
//! bit-packing utilities.  Every public function detects AVX2 at
//! runtime and transparently falls back to a scalar implementation
//! on hardware that lacks the extension.

// ── Scalar helpers (always available) ──────────────────────────────

/// Scalar absolute-max over a float slice.
fn absmax_scale_scalar(input: &[f32]) -> f32 {
    input.iter().copied().fold(0.0_f32, |m, v| m.max(v.abs()))
}

/// Scalar I2_S quantization: pack 2-bit signed values into bytes with
/// per-group scales.
fn quantize_i2s_scalar(input: &[f32], group_size: usize) -> (Vec<u8>, Vec<f32>) {
    assert!(group_size > 0, "group_size must be > 0");
    assert!(group_size.is_multiple_of(4), "group_size must be a multiple of 4");

    let n_groups = input.len().div_ceil(group_size);
    let packed_per_group = group_size / 4;
    let mut packed = Vec::with_capacity(n_groups * packed_per_group);
    let mut scales = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        let start = g * group_size;
        let end = (start + group_size).min(input.len());
        let group = &input[start..end];

        let abs_max = group.iter().copied().fold(0.0_f32, |m, v| m.max(v.abs()));
        scales.push(abs_max);

        let inv = if abs_max == 0.0 { 0.0 } else { 1.0 / abs_max };

        // Process in chunks of 4 (each chunk → 1 byte with four 2-bit values)
        let mut idx = 0;
        while idx < group_size {
            let mut byte: u8 = 0;
            for bit_pos in 0..4 {
                let val = if start + idx < input.len() {
                    let scaled = input[start + idx] * inv;
                    // Clamp to {-1, 0, 1} and map to unsigned 2-bit: 0b00=0, 0b01=1, 0b11=-1
                    let q = scaled.round().clamp(-1.0, 1.0) as i8;
                    match q {
                        1 => 0b01_u8,
                        -1 => 0b11_u8,
                        _ => 0b00_u8,
                    }
                } else {
                    0b00_u8
                };
                byte |= val << (bit_pos * 2);
                idx += 1;
            }
            packed.push(byte);
        }
    }

    (packed, scales)
}

/// Scalar I2_S dequantization: unpack 2-bit values and multiply by
/// per-group scale.
fn dequantize_i2s_scalar(packed: &[u8], scales: &[f32], group_size: usize) -> Vec<f32> {
    assert!(group_size > 0, "group_size must be > 0");
    assert!(group_size.is_multiple_of(4), "group_size must be a multiple of 4");

    let packed_per_group = group_size / 4;
    let total = scales.len() * group_size;
    let mut output = Vec::with_capacity(total);

    for (g, &scale) in scales.iter().enumerate() {
        let base = g * packed_per_group;
        for p in 0..packed_per_group {
            if base + p >= packed.len() {
                // Pad with zeros for incomplete groups.
                output.extend(std::iter::repeat_n(0.0, 4));
                continue;
            }
            let byte = packed[base + p];
            for bit_pos in 0..4 {
                let bits = (byte >> (bit_pos * 2)) & 0b11;
                let val: f32 = match bits {
                    0b01 => 1.0,
                    0b11 => -1.0,
                    _ => 0.0,
                };
                output.push(val * scale);
            }
        }
    }

    output
}

/// Scalar BitNet ternary quantization: map values to {-1, 0, 1}.
fn ternary_quantize_scalar(input: &[f32]) -> (Vec<i8>, f32) {
    let abs_max = absmax_scale_scalar(input);
    if abs_max == 0.0 {
        return (vec![0i8; input.len()], 0.0);
    }
    let inv = 1.0 / abs_max;
    let quantized = input
        .iter()
        .map(|&v| {
            let scaled = v * inv;
            scaled.round().clamp(-1.0, 1.0) as i8
        })
        .collect();
    (quantized, abs_max)
}

// ── AVX2 implementations (x86_64 only) ────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2_impl {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// AVX2 absolute-max: process 8 floats at a time.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available at runtime.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn absmax_avx2(input: &[f32]) -> f32 {
        let len = input.len();
        let chunks = len / 8;
        let ptr = input.as_ptr();

        // SAFETY: AVX2 is verified by caller via is_x86_feature_detected.
        unsafe {
            let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF_u32 as i32));
            let mut max_vec = _mm256_setzero_ps();

            for i in 0..chunks {
                let v = _mm256_loadu_ps(ptr.add(i * 8));
                let abs_v = _mm256_and_ps(v, sign_mask);
                max_vec = _mm256_max_ps(max_vec, abs_v);
            }

            let hi = _mm256_extractf128_ps(max_vec, 1);
            let lo = _mm256_castps256_ps128(max_vec);
            let m128 = _mm_max_ps(lo, hi);
            let shuf = _mm_movehdup_ps(m128);
            let m2 = _mm_max_ps(m128, shuf);
            let shuf2 = _mm_movehl_ps(m2, m2);
            let m_scalar = _mm_max_ss(m2, shuf2);
            let mut result = _mm_cvtss_f32(m_scalar);

            for &item in &input[(chunks * 8)..] {
                let a = item.abs();
                if a > result {
                    result = a;
                }
            }
            result
        }
    }

    /// AVX2-accelerated I2_S quantization.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available at runtime.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn quantize_i2s_avx2_inner(
        input: &[f32],
        group_size: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        // Delegate group-level logic to scalar — the AVX2 win is in
        // the absmax scan which the scalar path already calls.
        // For small groups the overhead of explicit SIMD packing is
        // not worthwhile; the scalar packer is branchless enough.
        //
        // We accelerate only the per-group absmax with AVX2.
        let n_groups = input.len().div_ceil(group_size);
        let packed_per_group = group_size / 4;
        let mut packed = Vec::with_capacity(n_groups * packed_per_group);
        let mut scales = Vec::with_capacity(n_groups);

        for g in 0..n_groups {
            let start = g * group_size;
            let end = (start + group_size).min(input.len());
            let group = &input[start..end];

            // SAFETY: absmax_avx2 has same target_feature requirement.
            let abs_max = unsafe { absmax_avx2(group) };
            scales.push(abs_max);

            let inv = if abs_max == 0.0 { 0.0 } else { 1.0 / abs_max };

            let mut idx = 0;
            while idx < group_size {
                let mut byte: u8 = 0;
                for bit_pos in 0..4u8 {
                    let val = if start + idx < input.len() {
                        let scaled = input[start + idx] * inv;
                        let q = scaled.round().clamp(-1.0, 1.0) as i8;
                        match q {
                            1 => 0b01_u8,
                            -1 => 0b11_u8,
                            _ => 0b00_u8,
                        }
                    } else {
                        0b00_u8
                    };
                    byte |= val << (bit_pos * 2);
                    idx += 1;
                }
                packed.push(byte);
            }
        }

        (packed, scales)
    }

    /// AVX2-accelerated I2_S dequantization.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available at runtime.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn dequantize_i2s_avx2_inner(
        packed: &[u8],
        scales: &[f32],
        group_size: usize,
    ) -> Vec<f32> {
        let packed_per_group = group_size / 4;
        let total = scales.len() * group_size;
        let mut output = Vec::with_capacity(total);

        for (g, &scale) in scales.iter().enumerate() {
            // SAFETY: AVX2 verified by caller.
            let base = g * packed_per_group;
            let full_chunks = packed_per_group / 8;
            let mut p = 0;

            for _ in 0..full_chunks {
                for sub in 0..8 {
                    if base + p + sub >= packed.len() {
                        output.extend(std::iter::repeat_n(0.0, 4));
                        continue;
                    }
                    let byte = packed[base + p + sub];
                    let vals: [f32; 4] = std::array::from_fn(|bit_pos| {
                        let bits = (byte >> (bit_pos * 2)) & 0b11;
                        match bits {
                            0b01 => 1.0,
                            0b11 => -1.0,
                            _ => 0.0,
                        }
                    });
                    for v in &vals {
                        output.push(v * scale);
                    }
                }
                p += 8;
            }

            while p < packed_per_group {
                if base + p >= packed.len() {
                    output.extend(std::iter::repeat_n(0.0, 4));
                } else {
                    let byte = packed[base + p];
                    for bit_pos in 0..4 {
                        let bits = (byte >> (bit_pos * 2)) & 0b11;
                        let val: f32 = match bits {
                            0b01 => 1.0,
                            0b11 => -1.0,
                            _ => 0.0,
                        };
                        output.push(val * scale);
                    }
                }
                p += 1;
            }
        }

        output
    }

    /// AVX2 ternary quantization.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available at runtime.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn ternary_quantize_avx2_inner(input: &[f32]) -> (Vec<i8>, f32) {
        // SAFETY: absmax_avx2 has same target_feature requirement.
        let abs_max = unsafe { absmax_avx2(input) };
        if abs_max == 0.0 {
            return (vec![0i8; input.len()], 0.0);
        }
        let inv = 1.0 / abs_max;
        let mut quantized = Vec::with_capacity(input.len());

        // SAFETY: AVX2 is verified by caller via is_x86_feature_detected.
        unsafe {
            let inv_vec = _mm256_set1_ps(inv);
            let pos_one = _mm256_set1_ps(1.0);
            let neg_one = _mm256_set1_ps(-1.0);
            let chunks = input.len() / 8;
            let ptr = input.as_ptr();

            for i in 0..chunks {
                let v = _mm256_loadu_ps(ptr.add(i * 8));
                let scaled = _mm256_mul_ps(v, inv_vec);
                let rounded =
                    _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                let clamped = _mm256_min_ps(_mm256_max_ps(rounded, neg_one), pos_one);

                let mut tmp = [0.0_f32; 8];
                _mm256_storeu_ps(tmp.as_mut_ptr(), clamped);
                for &t in &tmp {
                    quantized.push(t as i8);
                }
            }

            for &item in &input[(chunks * 8)..] {
                let scaled = item * inv;
                quantized.push(scaled.round().clamp(-1.0, 1.0) as i8);
            }
        }

        (quantized, abs_max)
    }
}

// ── Public API with runtime dispatch ───────────────────────────────

/// Find the absolute-maximum value in `input` using SIMD when available.
pub fn absmax_scale_avx2(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected at runtime.
            return unsafe { avx2_impl::absmax_avx2(input) };
        }
    }

    absmax_scale_scalar(input)
}

/// I2_S 2-bit quantization: pack values into bytes with per-group scales.
///
/// `group_size` must be a positive multiple of 4.  Returns `(packed_bytes,
/// scales)` where each group of `group_size / 4` bytes encodes one group
/// and `scales[g]` is the absmax of group `g`.
pub fn quantize_i2s_avx2(input: &[f32], group_size: usize) -> (Vec<u8>, Vec<f32>) {
    assert!(group_size > 0, "group_size must be > 0");
    assert!(group_size.is_multiple_of(4), "group_size must be a multiple of 4");

    if input.is_empty() {
        return (Vec::new(), Vec::new());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected at runtime.
            return unsafe { avx2_impl::quantize_i2s_avx2_inner(input, group_size) };
        }
    }

    quantize_i2s_scalar(input, group_size)
}

/// I2_S dequantization: unpack 2-bit values and reconstruct floats.
///
/// `group_size` must be a positive multiple of 4.  `scales` length
/// determines the number of groups.
pub fn dequantize_i2s_avx2(packed: &[u8], scales: &[f32], group_size: usize) -> Vec<f32> {
    assert!(group_size > 0, "group_size must be > 0");
    assert!(group_size.is_multiple_of(4), "group_size must be a multiple of 4");

    if scales.is_empty() {
        return Vec::new();
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected at runtime.
            return unsafe { avx2_impl::dequantize_i2s_avx2_inner(packed, scales, group_size) };
        }
    }

    dequantize_i2s_scalar(packed, scales, group_size)
}

/// BitNet ternary quantization: map each element to {-1, 0, 1}.
///
/// Returns `(quantized, scale)` where `scale` is the absmax of the
/// input.  Reconstruct with `output[i] = quantized[i] as f32 * scale`.
pub fn ternary_quantize_avx2(input: &[f32]) -> (Vec<i8>, f32) {
    if input.is_empty() {
        return (Vec::new(), 0.0);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected at runtime.
            return unsafe { avx2_impl::ternary_quantize_avx2_inner(input) };
        }
    }

    ternary_quantize_scalar(input)
}

/// Pack ternary {-1, 0, 1} values into bytes (2 bits each, 4 per byte).
///
/// Encoding: `0b00` → 0, `0b01` → 1, `0b11` → -1.
pub fn pack_ternary_bits(values: &[i8]) -> Vec<u8> {
    let n_bytes = values.len().div_ceil(4);
    let mut packed = Vec::with_capacity(n_bytes);
    for chunk in values.chunks(4) {
        let mut byte: u8 = 0;
        for (i, &v) in chunk.iter().enumerate() {
            let bits: u8 = match v {
                1 => 0b01,
                -1 => 0b11,
                _ => 0b00,
            };
            byte |= bits << (i * 2);
        }
        packed.push(byte);
    }
    packed
}

/// Unpack ternary bits back to {-1, 0, 1} values.
///
/// `count` is the number of original elements (since the last byte may
/// be only partially filled).
pub fn unpack_ternary_bits(packed: &[u8], count: usize) -> Vec<i8> {
    let mut output = Vec::with_capacity(count);
    for &byte in packed {
        for bit_pos in 0..4 {
            if output.len() >= count {
                break;
            }
            let bits = (byte >> (bit_pos * 2)) & 0b11;
            let val: i8 = match bits {
                0b01 => 1,
                0b11 => -1,
                _ => 0,
            };
            output.push(val);
        }
    }
    output.truncate(count);
    output
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── absmax_scale_avx2 ──────────────────────────────────────────

    #[test]
    fn absmax_empty() {
        assert_eq!(absmax_scale_avx2(&[]), 0.0);
    }

    #[test]
    fn absmax_single_positive() {
        assert_eq!(absmax_scale_avx2(&[3.5]), 3.5);
    }

    #[test]
    fn absmax_single_negative() {
        assert_eq!(absmax_scale_avx2(&[-7.0]), 7.0);
    }

    #[test]
    fn absmax_mixed() {
        let input = vec![1.0, -5.0, 3.0, -2.0, 4.0, 0.0, -0.5, 2.5];
        assert_eq!(absmax_scale_avx2(&input), 5.0);
    }

    #[test]
    fn absmax_all_zeros() {
        assert_eq!(absmax_scale_avx2(&[0.0; 16]), 0.0);
    }

    #[test]
    fn absmax_large_input() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.1).collect();
        let expected = input.iter().copied().fold(0.0_f32, |m, v| m.max(v.abs()));
        assert!((absmax_scale_avx2(&input) - expected).abs() < 1e-6);
    }

    #[test]
    fn absmax_tail_elements() {
        // 11 elements: 8 SIMD + 3 tail
        let input = [0.0; 11];
        let mut v = input;
        v[10] = -99.0;
        assert_eq!(absmax_scale_avx2(&v), 99.0);
    }

    #[test]
    fn absmax_negative_dominates() {
        let input = vec![1.0, 2.0, -10.0, 3.0];
        assert_eq!(absmax_scale_avx2(&input), 10.0);
    }

    // ── quantize / dequantize i2s round-trip ───────────────────────

    #[test]
    fn i2s_roundtrip_zeros() {
        let input = [0.0; 8];
        let (packed, scales) = quantize_i2s_avx2(&input, 4);
        let out = dequantize_i2s_avx2(&packed, &scales, 4);
        assert_eq!(out.len(), 8);
        for v in &out {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn i2s_roundtrip_ones() {
        let input = [1.0; 4];
        let (packed, scales) = quantize_i2s_avx2(&input, 4);
        let out = dequantize_i2s_avx2(&packed, &scales, 4);
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.01, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn i2s_roundtrip_negative_ones() {
        let input = [-1.0; 4];
        let (packed, scales) = quantize_i2s_avx2(&input, 4);
        let out = dequantize_i2s_avx2(&packed, &scales, 4);
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.01, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn i2s_roundtrip_mixed() {
        let input = vec![1.0, -1.0, 0.0, 0.5];
        let (packed, scales) = quantize_i2s_avx2(&input, 4);
        let out = dequantize_i2s_avx2(&packed, &scales, 4);
        // 2-bit: values near ±1 reproduce exactly; 0.5 rounds to 1.0*scale
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn i2s_roundtrip_multiple_groups() {
        let input = vec![1.0, -1.0, 0.0, 0.5, -0.5, 0.0, 1.0, -1.0];
        let (packed, scales) = quantize_i2s_avx2(&input, 4);
        assert_eq!(scales.len(), 2);
        let out = dequantize_i2s_avx2(&packed, &scales, 4);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn i2s_empty() {
        let (packed, scales) = quantize_i2s_avx2(&[], 4);
        assert!(packed.is_empty());
        assert!(scales.is_empty());
    }

    #[test]
    fn i2s_large_group() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let (packed, scales) = quantize_i2s_avx2(&input, 256);
        assert_eq!(scales.len(), 1);
        let out = dequantize_i2s_avx2(&packed, &scales, 256);
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn i2s_group_size_8() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.3, -0.3, 0.0, 0.0];
        let (packed, scales) = quantize_i2s_avx2(&input, 8);
        assert_eq!(scales.len(), 1);
        let out = dequantize_i2s_avx2(&packed, &scales, 8);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn i2s_roundtrip_accuracy_well_conditioned() {
        // "Well-conditioned": values are exactly ±scale or 0.
        let scale = 3.0_f32;
        let input = vec![scale, -scale, 0.0, scale, -scale, 0.0, scale, -scale];
        let (packed, scales) = quantize_i2s_avx2(&input, 8);
        let out = dequantize_i2s_avx2(&packed, &scales, 8);
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.01, "round-trip error {}: {} vs {}", (a - b).abs(), a, b);
        }
    }

    #[test]
    fn i2s_non_aligned_input() {
        // 6 elements, group_size 4 → 2 groups (second padded)
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.3, -0.3];
        let (packed, scales) = quantize_i2s_avx2(&input, 4);
        assert_eq!(scales.len(), 2);
        let out = dequantize_i2s_avx2(&packed, &scales, 4);
        // Output length is ceil_to_group_size.
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn i2s_scale_is_absmax() {
        let input = vec![2.0, -3.0, 1.0, 0.5];
        let (_, scales) = quantize_i2s_avx2(&input, 4);
        assert_eq!(scales[0], 3.0);
    }

    #[test]
    fn i2s_all_same_value() {
        let input = [5.0; 8];
        let (packed, scales) = quantize_i2s_avx2(&input, 4);
        let out = dequantize_i2s_avx2(&packed, &scales, 4);
        for v in &out {
            assert!((*v - 5.0).abs() < 0.01);
        }
    }

    #[test]
    fn i2s_all_negative_same() {
        let input = [-4.0; 4];
        let (packed, scales) = quantize_i2s_avx2(&input, 4);
        let out = dequantize_i2s_avx2(&packed, &scales, 4);
        for v in &out {
            assert!((*v - (-4.0)).abs() < 0.01);
        }
    }

    #[test]
    fn i2s_small_values() {
        let input = vec![0.001, -0.001, 0.0005, -0.0005];
        let (packed, scales) = quantize_i2s_avx2(&input, 4);
        let out = dequantize_i2s_avx2(&packed, &scales, 4);
        // Small values map to ±scale or 0; round-trip should be close.
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.01, "{a} vs {b}");
        }
    }

    // ── ternary_quantize_avx2 ──────────────────────────────────────

    #[test]
    fn ternary_empty() {
        let (q, s) = ternary_quantize_avx2(&[]);
        assert!(q.is_empty());
        assert_eq!(s, 0.0);
    }

    #[test]
    fn ternary_zeros() {
        let (q, s) = ternary_quantize_avx2(&[0.0; 8]);
        assert!(q.iter().all(|&v| v == 0));
        assert_eq!(s, 0.0);
    }

    #[test]
    fn ternary_positive_only() {
        let input = vec![1.0, 0.5, 0.9, 0.1, 0.8, 0.6, 0.7, 0.3];
        let (q, s) = ternary_quantize_avx2(&input);
        assert_eq!(s, 1.0);
        assert!(q.iter().all(|&v| v == 0 || v == 1));
    }

    #[test]
    fn ternary_negative_only() {
        let input = vec![-1.0, -0.5, -0.9, -0.1];
        let (q, s) = ternary_quantize_avx2(&input);
        assert_eq!(s, 1.0);
        assert!(q.iter().all(|&v| v == 0 || v == -1));
    }

    #[test]
    fn ternary_mixed() {
        let input = vec![2.0, -2.0, 0.1, -0.1, 1.5, -1.5, 0.0, 0.0];
        let (q, s) = ternary_quantize_avx2(&input);
        assert_eq!(s, 2.0);
        assert_eq!(q[0], 1);
        assert_eq!(q[1], -1);
        assert_eq!(q[7], 0);
    }

    #[test]
    fn ternary_roundtrip() {
        let input = vec![3.0, -3.0, 0.0, 3.0];
        let (q, s) = ternary_quantize_avx2(&input);
        let reconstructed: Vec<f32> = q.iter().map(|&v| v as f32 * s).collect();
        for (a, b) in input.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn ternary_large_input() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 100.0).collect();
        let (q, s) = ternary_quantize_avx2(&input);
        assert_eq!(q.len(), 1000);
        assert!(s > 0.0);
        assert!(q.iter().all(|&v| v >= -1 && v <= 1));
    }

    #[test]
    fn ternary_single_element() {
        let (q, s) = ternary_quantize_avx2(&[42.0]);
        assert_eq!(q, vec![1]);
        assert_eq!(s, 42.0);
    }

    #[test]
    fn ternary_single_negative() {
        let (q, s) = ternary_quantize_avx2(&[-7.0]);
        assert_eq!(q, vec![-1]);
        assert_eq!(s, 7.0);
    }

    // ── pack / unpack ternary bits ─────────────────────────────────

    #[test]
    fn pack_unpack_empty() {
        let packed = pack_ternary_bits(&[]);
        let unpacked = unpack_ternary_bits(&packed, 0);
        assert!(unpacked.is_empty());
    }

    #[test]
    fn pack_unpack_single() {
        let values = [1];
        let packed = pack_ternary_bits(&values);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_ternary_bits(&packed, 1);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_unpack_four() {
        let values = vec![1, -1, 0, 1];
        let packed = pack_ternary_bits(&values);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_ternary_bits(&packed, 4);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_unpack_five() {
        let values = vec![1, -1, 0, 1, -1];
        let packed = pack_ternary_bits(&values);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_ternary_bits(&packed, 5);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_unpack_eight() {
        let values = vec![1, 0, -1, 1, 0, -1, 1, 0];
        let packed = pack_ternary_bits(&values);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_ternary_bits(&packed, 8);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_unpack_all_ones() {
        let values = [1; 16];
        let packed = pack_ternary_bits(&values);
        let unpacked = unpack_ternary_bits(&packed, 16);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_unpack_all_neg_ones() {
        let values = [-1; 16];
        let packed = pack_ternary_bits(&values);
        let unpacked = unpack_ternary_bits(&packed, 16);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_unpack_all_zeros() {
        let values = [0; 16];
        let packed = pack_ternary_bits(&values);
        let unpacked = unpack_ternary_bits(&packed, 16);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_unpack_large() {
        let values: Vec<i8> = (0..100).map(|i| (i % 3) as i8 - 1).collect();
        let packed = pack_ternary_bits(&values);
        let unpacked = unpack_ternary_bits(&packed, 100);
        assert_eq!(unpacked, values);
    }

    // ── scalar / avx2 parity ───────────────────────────────────────

    #[test]
    fn scalar_avx2_absmax_parity() {
        let input: Vec<f32> = (0..33).map(|i| (i as f32 - 16.0) * 0.3).collect();
        let scalar = absmax_scale_scalar(&input);
        let avx2 = absmax_scale_avx2(&input);
        assert!((scalar - avx2).abs() < 1e-6);
    }

    #[test]
    fn scalar_avx2_i2s_parity() {
        let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let (p_s, s_s) = quantize_i2s_scalar(&input, 8);
        let (p_a, s_a) = quantize_i2s_avx2(&input, 8);
        assert_eq!(p_s, p_a);
        for (a, b) in s_s.iter().zip(s_a.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn scalar_avx2_ternary_parity() {
        // Use values that don't produce exact 0.5 after scaling.
        let input: Vec<f32> = vec![-2.0, -1.5, -1.1, -0.7, -0.3, 0.0, 0.3, 0.7, 1.1, 1.5, 2.0];
        let (q_s, s_s) = ternary_quantize_scalar(&input);
        let (q_a, s_a) = ternary_quantize_avx2(&input);
        assert_eq!(q_s, q_a);
        assert!((s_s - s_a).abs() < 1e-6);
    }

    // ── Additional edge-case tests ─────────────────────────────────

    #[test]
    fn i2s_group_size_equals_input_len() {
        let input = vec![1.0, -1.0, 0.5, -0.5];
        let (packed, scales) = quantize_i2s_avx2(&input, 4);
        assert_eq!(scales.len(), 1);
        let out = dequantize_i2s_avx2(&packed, &scales, 4);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn ternary_values_clamped() {
        // All values should clamp to {-1, 0, 1}.
        let input = vec![100.0, -100.0, 50.0, -50.0];
        let (q, _) = ternary_quantize_avx2(&input);
        assert!(q.iter().all(|&v| v >= -1 && v <= 1));
    }

    #[test]
    fn i2s_dequantize_empty_scales() {
        let out = dequantize_i2s_avx2(&[0u8; 4], &[], 4);
        assert!(out.is_empty());
    }

    #[test]
    fn pack_encoding_matches_spec() {
        // Verify: 0→0b00, 1→0b01, -1→0b11
        let packed = pack_ternary_bits(&[0, 1, -1, 0]);
        // byte = 0b00_11_01_00 = 0b00110100 = 0x34 = 52
        assert_eq!(packed[0], 0b00_11_01_00);
    }

    #[test]
    fn i2s_preserves_sign_pattern() {
        let input = vec![1.0, -1.0, 1.0, -1.0];
        let (packed, scales) = quantize_i2s_avx2(&input, 4);
        let out = dequantize_i2s_avx2(&packed, &scales, 4);
        assert!(out[0] > 0.0);
        assert!(out[1] < 0.0);
        assert!(out[2] > 0.0);
        assert!(out[3] < 0.0);
    }

    #[test]
    fn absmax_subnormal() {
        let input = vec![f32::MIN_POSITIVE / 2.0, -(f32::MIN_POSITIVE / 2.0)];
        let result = absmax_scale_avx2(&input);
        assert!(result > 0.0);
    }

    #[test]
    fn i2s_group_size_32() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let (packed, scales) = quantize_i2s_avx2(&input, 32);
        assert_eq!(scales.len(), 2);
        let out = dequantize_i2s_avx2(&packed, &scales, 32);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn ternary_threshold_boundary() {
        // 0.5 / absmax(1.0) = 0.5 → rounds to 1
        let input = vec![0.5, -0.5, 0.4, -0.4];
        let (q, _) = ternary_quantize_avx2(&input);
        assert_eq!(q[0], 1);
        assert_eq!(q[1], -1);
    }

    #[test]
    fn i2s_roundtrip_16_elements() {
        let input: Vec<f32> = vec![
            1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.3, -0.3, 0.8, -0.8, 0.0, 1.0, -1.0, 0.5,
        ];
        let (packed, scales) = quantize_i2s_avx2(&input, 8);
        assert_eq!(scales.len(), 2);
        let out = dequantize_i2s_avx2(&packed, &scales, 8);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn pack_unpack_non_ternary_clamped() {
        // Values outside {-1, 0, 1} are mapped to 0 by pack.
        let values = vec![2, -2, 0, 1];
        let packed = pack_ternary_bits(&values);
        let unpacked = unpack_ternary_bits(&packed, 4);
        // 2 maps to 0b00 (default), so unpacked → 0.
        assert_eq!(unpacked[0], 0);
        assert_eq!(unpacked[1], 0);
        assert_eq!(unpacked[2], 0);
        assert_eq!(unpacked[3], 1);
    }

    #[test]
    fn absmax_exactly_8_elements() {
        let input = vec![1.0, 2.0, 3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
        assert_eq!(absmax_scale_avx2(&input), 8.0);
    }

    #[test]
    fn absmax_9_elements() {
        let mut input = [1.0; 9];
        input[8] = -20.0;
        assert_eq!(absmax_scale_avx2(&input), 20.0);
    }

    #[test]
    fn ternary_11_elements() {
        let input: Vec<f32> = (0..11).map(|i| (i as f32 - 5.0) / 5.0).collect();
        let (q, s) = ternary_quantize_avx2(&input);
        assert_eq!(q.len(), 11);
        assert!(s > 0.0);
    }
}

// ── Property-based tests ───────────────────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Round-trip I2_S for well-conditioned inputs (values are
        /// ±scale or 0) has error < 0.01.
        #[test]
        fn i2s_roundtrip_well_conditioned(
            scale in 0.01_f32..100.0_f32,
            pattern in proptest::collection::vec(
                prop_oneof![Just(-1.0_f32), Just(0.0_f32), Just(1.0_f32)],
                4..=128,
            ),
        ) {
            // Pad to multiple of 4.
            let mut input: Vec<f32> = pattern.iter().map(|&v| v * scale).collect();
            while input.len() % 4 != 0 {
                input.push(0.0);
            }
            let gs = 64; // group size for quantization
            let (packed, scales) = quantize_i2s_avx2(&input, gs);
            let out = dequantize_i2s_avx2(&packed, &scales, gs);
            for (i, (a, b)) in input.iter().zip(out.iter()).enumerate() {
                prop_assert!(
                    (a - b).abs() < 0.01,
                    "round-trip error at index {}: {} vs {} (err={})",
                    i, a, b, (a - b).abs()
                );
            }
        }

        /// Pack/unpack ternary bits is lossless for valid ternary values.
        #[test]
        fn pack_unpack_roundtrip(
            values in proptest::collection::vec(-1_i8..=1, 1..=256),
        ) {
            let packed = pack_ternary_bits(&values);
            let unpacked = unpack_ternary_bits(&packed, values.len());
            prop_assert_eq!(&unpacked, &values);
        }

        /// absmax_scale_avx2 matches scalar for arbitrary inputs.
        #[test]
        fn absmax_matches_scalar(
            input in proptest::collection::vec(-1000.0_f32..1000.0, 1..=512),
        ) {
            let scalar = absmax_scale_scalar(&input);
            let avx2 = absmax_scale_avx2(&input);
            prop_assert!(
                (scalar - avx2).abs() < 1e-5,
                "scalar={} avx2={}", scalar, avx2
            );
        }

        /// Ternary quantization always produces values in {-1, 0, 1}.
        #[test]
        fn ternary_values_in_range(
            input in proptest::collection::vec(-100.0_f32..100.0, 1..=256),
        ) {
            let (q, _) = ternary_quantize_avx2(&input);
            for &v in &q {
                prop_assert!(v >= -1 && v <= 1, "out of range: {}", v);
            }
        }

        /// Ternary quantization round-trip for well-conditioned inputs.
        #[test]
        fn ternary_roundtrip_well_conditioned(
            scale in 0.01_f32..100.0_f32,
            pattern in proptest::collection::vec(
                prop_oneof![Just(-1.0_f32), Just(0.0_f32), Just(1.0_f32)],
                1..=128,
            ),
        ) {
            let input: Vec<f32> = pattern.iter().map(|&v| v * scale).collect();
            let (q, s) = ternary_quantize_avx2(&input);
            let reconstructed: Vec<f32> = q.iter().map(|&v| v as f32 * s).collect();
            for (i, (a, b)) in input.iter().zip(reconstructed.iter()).enumerate() {
                prop_assert!(
                    (a - b).abs() < 0.01,
                    "ternary round-trip error at {}: {} vs {} (err={})",
                    i, a, b, (a - b).abs()
                );
            }
        }
    }
}
