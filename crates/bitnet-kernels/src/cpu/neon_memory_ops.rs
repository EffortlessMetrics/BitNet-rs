//! ARM NEON-optimized memory operations for Apple Silicon.
//!
//! Provides high-performance memory operations using NEON intrinsics:
//! - Aligned memory copy and fill
//! - Interleave/deinterleave f32 arrays
//! - Gather by index
//! - 4×4 matrix transpose using NEON intrinsics

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON-accelerated aligned memory copy using vld1q_f32/vst1q_f32.
///
/// Copies `src` to `dst` using 128-bit (4×f32) NEON loads and stores for aligned data.
/// Handles remaining elements with scalar copy.
///
/// # Panics
///
/// Panics if `dst` is smaller than `src`.
#[cfg(target_arch = "aarch64")]
pub fn neon_memcpy_aligned(src: &[f32], dst: &mut [f32]) {
    assert!(dst.len() >= src.len(), "destination too small: {} < {}", dst.len(), src.len());

    if src.is_empty() {
        return;
    }

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let len = src.len();

    // Process 4×f32 blocks (16 bytes) using NEON.
    let block_count = len / 4;
    for i in 0..block_count {
        unsafe {
            let v = vld1q_f32(src_ptr.add(i * 4));
            vst1q_f32(dst_ptr.add(i * 4), v);
        }
    }

    // Handle remaining elements with scalar copy.
    let remaining = len % 4;
    for i in 0..remaining {
        dst[block_count * 4 + i] = src[block_count * 4 + i];
    }
}

/// NEON-accelerated memory copy for non-aarch64 targets (scalar fallback).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_memcpy_aligned(src: &[f32], dst: &mut [f32]) {
    assert!(dst.len() >= src.len(), "destination too small: {} < {}", dst.len(), src.len());
    dst[..src.len()].copy_from_slice(src);
}

/// Fill f32 buffer with a scalar value using NEON vdupq_n_f32.
///
/// Fills `dst` with `val` using 128-bit NEON stores for 4×f32 at a time.
/// Handles remaining elements with scalar assignment.
///
/// # Panics
///
/// None; handles empty slices gracefully.
#[cfg(target_arch = "aarch64")]
pub fn neon_memset_f32(dst: &mut [f32], val: f32) {
    if dst.is_empty() {
        return;
    }

    let dst_ptr = dst.as_mut_ptr();
    let len = dst.len();

    // Create a 128-bit NEON register filled with `val`.
    let v_fill;
    unsafe {
        v_fill = vdupq_n_f32(val);
    }

    // Store 4×f32 blocks using NEON.
    let block_count = len / 4;
    for i in 0..block_count {
        unsafe {
            vst1q_f32(dst_ptr.add(i * 4), v_fill);
        }
    }

    // Handle remaining elements with scalar assignment.
    let remaining = len % 4;
    for i in 0..remaining {
        dst[block_count * 4 + i] = val;
    }
}

/// Fill f32 buffer with a scalar value (scalar fallback for non-aarch64).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_memset_f32(dst: &mut [f32], val: f32) {
    for elem in dst.iter_mut() {
        *elem = val;
    }
}

/// Interleave two f32 arrays into a single output: [a0, b0, a1, b1, ...].
///
/// Merges two arrays `a` and `b` of equal length into `output` by alternating elements.
/// Output must have length >= 2 * a.len().
///
/// # Panics
///
/// Panics if:
/// - `a` and `b` have different lengths
/// - `output` is smaller than `2 * a.len()`
#[cfg(target_arch = "aarch64")]
pub fn neon_interleave_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "arrays must have equal length");
    let out_len = 2 * a.len();
    assert!(output.len() >= out_len, "output too small: {} < {}", output.len(), out_len);

    if a.is_empty() {
        return;
    }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let len = a.len();

    // Process pairs of elements using vzip (interleave).
    let pair_count = len / 4;
    for i in 0..pair_count {
        unsafe {
            let a_v = vld1q_f32(a_ptr.add(i * 4));
            let b_v = vld1q_f32(b_ptr.add(i * 4));

            // vzip1_f32 and vzip2_f32 interleave within 64-bit halves.
            let low = vzip1q_f32(a_v, b_v);
            let high = vzip2q_f32(a_v, b_v);

            vst1q_f32(out_ptr.add(i * 8), low);
            vst1q_f32(out_ptr.add(i * 8 + 4), high);
        }
    }

    // Handle remaining elements with scalar interleave.
    let remaining = len % 4;
    let base_idx = pair_count * 4;
    for i in 0..remaining {
        output[2 * (base_idx + i)] = a[base_idx + i];
        output[2 * (base_idx + i) + 1] = b[base_idx + i];
    }
}

/// Interleave two f32 arrays (scalar fallback for non-aarch64).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_interleave_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "arrays must have equal length");
    let out_len = 2 * a.len();
    assert!(output.len() >= out_len, "output too small: {} < {}", output.len(), out_len);

    for i in 0..a.len() {
        output[2 * i] = a[i];
        output[2 * i + 1] = b[i];
    }
}

/// Deinterleave f32 array into two halves: [a0, b0, a1, b1, ...] → a=[a0, a1, ...], b=[b0, b1, ...].
///
/// Splits an interleaved array into two separate arrays `a` and `b`.
/// Both `a` and `b` must have length >= input.len() / 2.
///
/// # Panics
///
/// Panics if:
/// - `input` has odd length
/// - `a` or `b` is too small
#[cfg(target_arch = "aarch64")]
pub fn neon_deinterleave_f32(input: &[f32], a: &mut [f32], b: &mut [f32]) {
    assert!(input.len() % 2 == 0, "input length must be even, got {}", input.len());
    let half_len = input.len() / 2;
    assert!(a.len() >= half_len, "array a too small: {} < {}", a.len(), half_len);
    assert!(b.len() >= half_len, "array b too small: {} < {}", b.len(), half_len);

    if input.is_empty() {
        return;
    }

    let in_ptr = input.as_ptr();
    let a_ptr = a.as_mut_ptr();
    let b_ptr = b.as_mut_ptr();
    let len = input.len();

    // Process pairs of 4-element blocks using vuzp (deinterleave).
    let pair_count = len / 8;
    for i in 0..pair_count {
        unsafe {
            let v0 = vld1q_f32(in_ptr.add(i * 8));
            let v1 = vld1q_f32(in_ptr.add(i * 8 + 4));

            let a_low = vuzp1q_f32(v0, v1);
            let b_low = vuzp2q_f32(v0, v1);

            vst1q_f32(a_ptr.add(i * 4), a_low);
            vst1q_f32(b_ptr.add(i * 4), b_low);
        }
    }

    // Handle remaining elements with scalar deinterleave.
    let remaining = len % 8;
    let base_idx = pair_count * 4;
    for i in 0..remaining / 2 {
        a[base_idx + i] = input[pair_count * 8 + 2 * i];
        b[base_idx + i] = input[pair_count * 8 + 2 * i + 1];
    }
}

/// Deinterleave f32 array (scalar fallback for non-aarch64).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_deinterleave_f32(input: &[f32], a: &mut [f32], b: &mut [f32]) {
    assert!(input.len() % 2 == 0, "input length must be even, got {}", input.len());
    let half_len = input.len() / 2;
    assert!(a.len() >= half_len, "array a too small: {} < {}", a.len(), half_len);
    assert!(b.len() >= half_len, "array b too small: {} < {}", b.len(), half_len);

    for i in 0..half_len {
        a[i] = input[2 * i];
        b[i] = input[2 * i + 1];
    }
}

/// Gather elements from `src` by indices, writing to `dst`.
///
/// For each index in `indices`, gathers the element from `src[index]`
/// and writes it to the corresponding position in `dst`.
///
/// # Panics
///
/// Panics if:
/// - `indices` and `dst` have different lengths
/// - Any index is out of bounds for `src`
#[cfg(target_arch = "aarch64")]
pub fn neon_gather_f32(src: &[f32], indices: &[usize], dst: &mut [f32]) {
    assert_eq!(indices.len(), dst.len(), "indices and destination must have equal length");

    if indices.is_empty() {
        return;
    }

    // Validate all indices are in bounds.
    for &idx in indices {
        assert!(idx < src.len(), "index {} out of bounds for src.len() = {}", idx, src.len());
    }

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    // NEON gather with 4-element blocks.
    let block_count = indices.len() / 4;
    for i in 0..block_count {
        unsafe {
            let i0 = indices[i * 4];
            let i1 = indices[i * 4 + 1];
            let i2 = indices[i * 4 + 2];
            let i3 = indices[i * 4 + 3];

            let e0 = *src_ptr.add(i0);
            let e1 = *src_ptr.add(i1);
            let e2 = *src_ptr.add(i2);
            let e3 = *src_ptr.add(i3);

            *dst_ptr.add(i * 4) = e0;
            *dst_ptr.add(i * 4 + 1) = e1;
            *dst_ptr.add(i * 4 + 2) = e2;
            *dst_ptr.add(i * 4 + 3) = e3;
        }
    }

    // Handle remaining elements with scalar gather.
    let remaining = indices.len() % 4;
    for i in 0..remaining {
        let idx = indices[block_count * 4 + i];
        dst[block_count * 4 + i] = src[idx];
    }
}

/// Gather elements from `src` by indices (scalar fallback for non-aarch64).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_gather_f32(src: &[f32], indices: &[usize], dst: &mut [f32]) {
    assert_eq!(indices.len(), dst.len(), "indices and destination must have equal length");

    for (i, &idx) in indices.iter().enumerate() {
        assert!(idx < src.len(), "index {} out of bounds for src.len() = {}", idx, src.len());
        dst[i] = src[idx];
    }
}

/// Transpose a 4×4 matrix in-place using NEON vtrn/vzip intrinsics.
///
/// Performs a 4×4 transpose on row-major data. Input/output arrays must be exactly 16 elements.
/// Uses vtrn1q/vtrn2q for element-level transpose, then vzip for 64-bit swaps.
///
/// # Panics
///
/// Panics if input or output is not exactly 16 elements.
#[cfg(target_arch = "aarch64")]
pub fn neon_transpose_4x4(input: &[f32; 16], output: &mut [f32; 16]) {
    // Load four rows of 4 elements each.
    let r0;
    let r1;
    let r2;
    let r3;
    unsafe {
        r0 = vld1q_f32(&input[0]);
        r1 = vld1q_f32(&input[4]);
        r2 = vld1q_f32(&input[8]);
        r3 = vld1q_f32(&input[12]);
    }

    // Stage 1: Transpose within 32-bit halves using vtrn1q/vtrn2q.
    let t0;
    let t1;
    let t2;
    let t3;
    unsafe {
        t0 = vtrn1q_f32(r0, r1); // [r0[0], r1[0], r0[2], r1[2]]
        t1 = vtrn2q_f32(r0, r1); // [r0[1], r1[1], r0[3], r1[3]]
        t2 = vtrn1q_f32(r2, r3); // [r2[0], r3[0], r2[2], r3[2]]
        t3 = vtrn2q_f32(r2, r3); // [r2[1], r3[1], r2[3], r3[3]]
    }

    // Stage 2: Swap 64-bit halves via f64 reinterpret + vtrn.
    let o0;
    let o1;
    let o2;
    let o3;
    unsafe {
        let t0_64 = vreinterpretq_f64_f32(t0);
        let t1_64 = vreinterpretq_f64_f32(t1);
        let t2_64 = vreinterpretq_f64_f32(t2);
        let t3_64 = vreinterpretq_f64_f32(t3);

        o0 = vreinterpretq_f32_f64(vtrn1q_f64(t0_64, t2_64));
        o1 = vreinterpretq_f32_f64(vtrn1q_f64(t1_64, t3_64));
        o2 = vreinterpretq_f32_f64(vtrn2q_f64(t0_64, t2_64));
        o3 = vreinterpretq_f32_f64(vtrn2q_f64(t1_64, t3_64));
    }

    // Store transposed rows.
    unsafe {
        vst1q_f32(&mut output[0], o0);
        vst1q_f32(&mut output[4], o1);
        vst1q_f32(&mut output[8], o2);
        vst1q_f32(&mut output[12], o3);
    }
}

/// Transpose a 4×4 matrix (scalar fallback for non-aarch64).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_transpose_4x4(input: &[f32; 16], output: &mut [f32; 16]) {
    for i in 0..4 {
        for j in 0..4 {
            output[j * 4 + i] = input[i * 4 + j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_memcpy_aligned_basic() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dst = vec![0.0; 8];
        neon_memcpy_aligned(&src, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_neon_memcpy_aligned_non_aligned() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut dst = vec![0.0; 7];
        neon_memcpy_aligned(&src, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_neon_memcpy_aligned_empty() {
        let src: Vec<f32> = vec![];
        let mut dst = vec![0.0; 5];
        neon_memcpy_aligned(&src, &mut dst);
        // dst[0..0] should be empty, rest unchanged
        assert_eq!(dst[0..0].len(), 0);
    }

    #[test]
    #[should_panic(expected = "destination too small")]
    fn test_neon_memcpy_aligned_too_small() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let mut dst = vec![0.0; 2];
        neon_memcpy_aligned(&src, &mut dst);
    }

    #[test]
    fn test_neon_memset_f32_basic() {
        let mut dst = vec![0.0; 8];
        neon_memset_f32(&mut dst, 5.0);
        assert!(dst.iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_neon_memset_f32_non_aligned() {
        let mut dst = vec![0.0; 7];
        neon_memset_f32(&mut dst, 3.14);
        assert!(dst.iter().all(|&x| (x - 3.14).abs() < 1e-6));
    }

    #[test]
    fn test_neon_memset_f32_empty() {
        let mut dst: Vec<f32> = vec![];
        neon_memset_f32(&mut dst, 1.0);
        assert_eq!(dst.len(), 0);
    }

    #[test]
    fn test_neon_interleave_f32_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 8];
        neon_interleave_f32(&a, &b, &mut output);
        let expected = vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_neon_interleave_f32_single_element() {
        let a = vec![1.5];
        let b = vec![2.5];
        let mut output = vec![0.0; 2];
        neon_interleave_f32(&a, &b, &mut output);
        assert_eq!(output, vec![1.5, 2.5]);
    }

    #[test]
    fn test_neon_interleave_f32_non_aligned() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let mut output = vec![0.0; 10];
        neon_interleave_f32(&a, &b, &mut output);
        let expected = vec![1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 5.0, 10.0];
        assert_eq!(output, expected);
    }

    #[test]
    #[should_panic(expected = "arrays must have equal length")]
    fn test_neon_interleave_f32_unequal_length() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0];
        let mut output = vec![0.0; 4];
        neon_interleave_f32(&a, &b, &mut output);
    }

    #[test]
    fn test_neon_deinterleave_f32_basic() {
        let input = vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];
        let mut a = vec![0.0; 4];
        let mut b = vec![0.0; 4];
        neon_deinterleave_f32(&input, &mut a, &mut b);
        assert_eq!(a, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(b, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_neon_deinterleave_f32_single_pair() {
        let input = vec![1.5, 2.5];
        let mut a = vec![0.0; 1];
        let mut b = vec![0.0; 1];
        neon_deinterleave_f32(&input, &mut a, &mut b);
        assert_eq!(a, vec![1.5]);
        assert_eq!(b, vec![2.5]);
    }

    #[test]
    fn test_neon_deinterleave_f32_non_aligned() {
        let input = vec![1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 5.0, 10.0];
        let mut a = vec![0.0; 5];
        let mut b = vec![0.0; 5];
        neon_deinterleave_f32(&input, &mut a, &mut b);
        assert_eq!(a, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(b, vec![6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    #[should_panic(expected = "input length must be even")]
    fn test_neon_deinterleave_f32_odd_length() {
        let input = vec![1.0, 2.0, 3.0];
        let mut a = vec![0.0; 2];
        let mut b = vec![0.0; 2];
        neon_deinterleave_f32(&input, &mut a, &mut b);
    }

    #[test]
    fn test_neon_gather_f32_basic() {
        let src = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let indices = vec![0, 2, 4, 1, 3];
        let mut dst = vec![0.0; 5];
        neon_gather_f32(&src, &indices, &mut dst);
        assert_eq!(dst, vec![10.0, 30.0, 50.0, 20.0, 40.0]);
    }

    #[test]
    fn test_neon_gather_f32_non_aligned() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let indices = vec![0, 2, 4, 6, 1];
        let mut dst = vec![0.0; 5];
        neon_gather_f32(&src, &indices, &mut dst);
        assert_eq!(dst, vec![1.0, 3.0, 5.0, 7.0, 2.0]);
    }

    #[test]
    fn test_neon_gather_f32_single() {
        let src = vec![42.0];
        let indices = vec![0];
        let mut dst = vec![0.0];
        neon_gather_f32(&src, &indices, &mut dst);
        assert_eq!(dst, vec![42.0]);
    }

    #[test]
    #[should_panic(expected = "index")]
    fn test_neon_gather_f32_out_of_bounds() {
        let src = vec![1.0, 2.0];
        let indices = vec![5];
        let mut dst = vec![0.0];
        neon_gather_f32(&src, &indices, &mut dst);
    }

    #[test]
    fn test_neon_transpose_4x4_basic() {
        // Row-major 4×4:
        // 1  2  3  4
        // 5  6  7  8
        // 9  10 11 12
        // 13 14 15 16
        let input =
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

        // Expected transposed (column-major becomes row-major):
        // 1  5  9  13
        // 2  6  10 14
        // 3  7  11 15
        // 4  8  12 16
        let expected =
            [1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0];

        let mut output = [0.0; 16];
        neon_transpose_4x4(&input, &mut output);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_neon_transpose_4x4_identity() {
        let input =
            [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0];
        let mut output = [0.0; 16];
        neon_transpose_4x4(&input, &mut output);
        assert_eq!(output, input);
    }

    #[test]
    fn test_neon_transpose_4x4_double_transpose() {
        let input =
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let mut transposed = [0.0; 16];
        neon_transpose_4x4(&input, &mut transposed);

        let mut output = [0.0; 16];
        neon_transpose_4x4(&transposed, &mut output);
        assert_eq!(output, input);
    }

    #[test]
    fn test_neon_transpose_4x4_float_values() {
        let input =
            [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5];
        let mut output = [0.0; 16];
        neon_transpose_4x4(&input, &mut output);

        // Verify transpose property: output[i][j] == input[j][i]
        for i in 0..4 {
            for j in 0..4 {
                assert!((output[i * 4 + j] - input[j * 4 + i]).abs() < 1e-6);
            }
        }
    }
}
