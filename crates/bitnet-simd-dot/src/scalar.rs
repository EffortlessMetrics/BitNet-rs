//! Scalar (non-SIMD) fallback implementations.

/// Scalar f32 dot product.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Scalar i8 dot product → i32.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| i32::from(x) * i32::from(y)).sum()
}

/// Scalar binary dot product via XOR + popcount.
///
/// Counts the number of matching bits: `len_bits - popcount(a ^ b)`.
#[inline]
pub fn binary_dot(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    #[allow(clippy::cast_possible_truncation)]
    let total_bits = (a.len() as u32) * 64;
    let diff: u32 = a.iter().zip(b).map(|(&x, &y)| (x ^ y).count_ones()).sum();
    total_bits - diff
}

/// Scalar fused multiply-accumulate: `a · b + c · d`.
#[inline]
pub fn fma_dot_f32(a: &[f32], b: &[f32], c: &[f32], d: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(c.len(), d.len());
    let ab: f32 = a.iter().zip(b).map(|(&x, &y)| x.mul_add(y, 0.0)).sum();
    let cd: f32 = c.iter().zip(d).map(|(&x, &y)| x.mul_add(y, 0.0)).sum();
    ab + cd
}

/// Scalar strided f32 dot product.
#[inline]
pub fn strided_dot_f32(a: &[f32], b: &[f32], stride: usize) -> f32 {
    debug_assert!(stride > 0);
    a.iter().step_by(stride).zip(b.iter().step_by(stride)).map(|(&x, &y)| x * y).sum()
}

/// Scalar batched f32 dot product.
///
/// Each row is `cols` elements; computes `dot(a_row, b_row)` for each of `rows` rows.
#[inline]
#[allow(dead_code)]
pub fn batched_dot_f32(a: &[f32], b: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), rows * cols);
    debug_assert_eq!(b.len(), rows * cols);
    (0..rows)
        .map(|r| {
            let off = r * cols;
            dot_f32(&a[off..off + cols], &b[off..off + cols])
        })
        .collect()
}
