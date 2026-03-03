//! Runtime SIMD dispatch — detects the best instruction set at call-time
//! and routes to the matching kernel.

use crate::scalar;

/// The SIMD instruction set selected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimdLevel {
    /// Pure scalar (fallback on all platforms).
    Scalar,
    /// `x86_64` SSE 4.1 (128-bit).
    Sse41,
    /// `x86_64` AVX2 + FMA (256-bit).
    Avx2,
    /// `x86_64` AVX-512F (512-bit).
    Avx512,
    /// `aarch64` NEON (128-bit).
    Neon,
}

impl SimdLevel {
    /// Detect the highest supported SIMD level on the current CPU.
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                return Self::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Self::Avx2;
            }
            if is_x86_feature_detected!("sse4.1") {
                return Self::Sse41;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is mandatory on aarch64.
            return Self::Neon;
        }
        #[allow(unreachable_code)]
        Self::Scalar
    }
}

impl std::fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scalar => write!(f, "scalar"),
            Self::Sse41 => write!(f, "sse4.1"),
            Self::Avx2 => write!(f, "avx2"),
            Self::Avx512 => write!(f, "avx512"),
            Self::Neon => write!(f, "neon"),
        }
    }
}

// ── Public dispatch functions ───────────────────────────────────────

/// Compute the f32 dot product of `a` and `b`.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[must_use]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_f32: length mismatch");
    if a.is_empty() {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { crate::x86::dot_f32_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { crate::x86::dot_f32_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { crate::x86::dot_f32_sse41(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { crate::neon::dot_f32_neon(a, b) };
    }
    #[allow(unreachable_code)]
    scalar::dot_f32(a, b)
}

/// Compute the i8 dot product of `a` and `b`, returning an `i32`.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[must_use]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "dot_i8: length mismatch");
    if a.is_empty() {
        return 0;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return unsafe { crate::x86::dot_i8_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { crate::x86::dot_i8_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { crate::x86::dot_i8_sse41(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { crate::neon::dot_i8_neon(a, b) };
    }
    #[allow(unreachable_code)]
    scalar::dot_i8(a, b)
}

/// Binary (popcount-based) dot product.
///
/// Returns the count of **matching** bits across `a` and `b`.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[must_use]
pub fn binary_dot(a: &[u64], b: &[u64]) -> u32 {
    assert_eq!(a.len(), b.len(), "binary_dot: length mismatch");
    scalar::binary_dot(a, b)
}

/// Fused multiply-accumulate: `a · b + c · d` in a single pass.
///
/// # Panics
///
/// Panics if `a.len() != b.len()` or `c.len() != d.len()`.
#[must_use]
pub fn fma_dot_f32(a: &[f32], b: &[f32], c: &[f32], d: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "fma_dot_f32: a/b length mismatch");
    assert_eq!(c.len(), d.len(), "fma_dot_f32: c/d length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { crate::x86::fma_dot_f32_avx2(a, b, c, d) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { crate::neon::fma_dot_f32_neon(a, b, c, d) };
    }
    #[allow(unreachable_code)]
    scalar::fma_dot_f32(a, b, c, d)
}

/// Strided f32 dot product — every `stride`-th element.
///
/// # Panics
///
/// * Panics if `a.len() != b.len()`.
/// * Panics if `stride == 0`.
#[must_use]
pub fn strided_dot_f32(a: &[f32], b: &[f32], stride: usize) -> f32 {
    assert_eq!(a.len(), b.len(), "strided_dot_f32: length mismatch");
    assert!(stride > 0, "strided_dot_f32: stride must be > 0");
    scalar::strided_dot_f32(a, b, stride)
}

/// Batched f32 dot product — `rows` independent dot products of length `cols`.
///
/// Returns a `Vec<f32>` with one result per row.
///
/// # Panics
///
/// * Panics if `a.len() != rows * cols`.
/// * Panics if `b.len() != rows * cols`.
#[must_use]
pub fn batched_dot_f32(a: &[f32], b: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(a.len(), rows * cols, "batched_dot_f32: a length mismatch");
    assert_eq!(b.len(), rows * cols, "batched_dot_f32: b length mismatch");
    (0..rows)
        .map(|r| {
            let off = r * cols;
            dot_f32(&a[off..off + cols], &b[off..off + cols])
        })
        .collect()
}
