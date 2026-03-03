//! Runtime CPU feature detection and dispatch.
//!
//! Each public `*_dispatch` function checks for AVX-512F at runtime and
//! delegates to either the AVX-512 or scalar implementation.

use crate::scalar;

/// Returns `true` when the CPU supports AVX-512F.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_avx512f() -> bool {
    is_x86_feature_detected!("avx512f")
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn has_avx512f() -> bool {
    false
}

// ---- softmax in-place -------------------------------------------------------

pub fn softmax_inplace_dispatch(xs: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() {
            // SAFETY: guarded by runtime CPUID check above.
            unsafe {
                crate::avx512::softmax_inplace_avx512(xs);
            }
            return;
        }
    }
    scalar::softmax_inplace(xs);
}

// ---- online softmax ---------------------------------------------------------

pub fn online_softmax_dispatch(logits: &[f32]) -> Vec<f32> {
    // Online softmax is inherently sequential; scalar is used for all backends.
    scalar::online_softmax(logits)
}

// ---- log softmax ------------------------------------------------------------

pub fn log_softmax_dispatch(logits: &[f32]) -> Vec<f32> {
    scalar::log_softmax(logits)
}

// ---- temperature softmax ----------------------------------------------------

pub fn temperature_softmax_dispatch(logits: &[f32], temperature: f32) -> Vec<f32> {
    scalar::temperature_softmax(logits, temperature)
}

// ---- masked softmax ---------------------------------------------------------

pub fn masked_softmax_dispatch(logits: &[f32], mask: &[bool]) -> Vec<f32> {
    scalar::masked_softmax(logits, mask)
}

// ---- batch softmax in-place -------------------------------------------------

pub fn batch_softmax_inplace_dispatch(logits: &mut [f32], row_len: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() {
            for row in logits.chunks_exact_mut(row_len) {
                // SAFETY: guarded by runtime CPUID check above.
                unsafe {
                    crate::avx512::softmax_inplace_avx512(row);
                }
            }
            return;
        }
    }
    scalar::batch_softmax_inplace(logits, row_len);
}
