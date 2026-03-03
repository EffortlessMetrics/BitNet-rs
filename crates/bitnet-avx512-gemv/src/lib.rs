//! AVX-512 optimized GEMV (General Matrix-Vector multiply) kernels.
//!
//! This crate provides high-performance matrix-vector multiplication routines
//! optimised for `BitNet` inference workloads.  Three kernel families are offered:
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_precision_loss)]
//!
//! | Family | Module | Description |
//! |--------|--------|-------------|
//! | **f32** | [`f32_gemv`] | Dense single-precision GEMV |
//! | **i8**  | [`i8_gemv`]  | Quantised 8-bit integer GEMV |
//! | **binary** | [`binary_gemv`] | 1-bit packed binary GEMV |
//!
//! Each module exposes an AVX-512 fast path (gated on `target_feature = "avx512f"`)
//! and a portable scalar fallback.  Use [`detect::avx512_available`] to query
//! runtime support and [`gemv`] / [`gemv_i8`] / [`gemv_binary`] for
//! auto-dispatched entry points.
//!
//! # Example
//!
//! ```
//! use bitnet_avx512_gemv::{gemv, GemvParams};
//!
//! let matrix = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let vector = vec![1.0_f32, 2.0, 3.0];
//! let params = GemvParams::new(2, 3, &matrix, &vector);
//! let result = gemv(&params);
//! assert_eq!(result, vec![14.0, 32.0]);
//! ```

pub mod binary_gemv;
pub mod detect;
pub mod dot;
pub mod f32_gemv;
pub mod i8_gemv;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use binary_gemv::{BinaryGemvParams, gemv_binary};
pub use detect::avx512_available;
pub use f32_gemv::{GemvParams, gemv};
pub use i8_gemv::{I8GemvParams, gemv_i8};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_f32_gemv() {
        // 2×3 matrix, 3-element vector
        let m = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = vec![1.0, 2.0, 3.0];
        let p = GemvParams::new(2, 3, &m, &v);
        let r = gemv(&p);
        assert_eq!(r, vec![14.0, 32.0]);
    }

    #[test]
    fn smoke_i8_gemv() {
        let m: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let v: Vec<i8> = vec![1, 2, 3];
        let p = I8GemvParams::new(2, 3, &m, &v);
        let r = gemv_i8(&p);
        assert_eq!(r, vec![14, 32]);
    }

    #[test]
    fn smoke_binary_gemv() {
        // 2 rows, 8 cols (packed into 1 byte each)
        // row0 = 0xFF (all 1s), row1 = 0x00 (all 0s)
        let packed = vec![0xFF_u8, 0x00];
        let v = vec![1.0_f32; 8];
        let p = BinaryGemvParams::new(2, 8, &packed, &v);
        let r = gemv_binary(&p);
        assert!((r[0] - 8.0).abs() < 1e-6);
        assert!((r[1] - (-8.0)).abs() < 1e-6);
    }
}
