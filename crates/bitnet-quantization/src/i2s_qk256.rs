//! GGML I2_S (QK=256) dispatch wrappers.
//!
//! Scalar QK256 primitives live in the `bitnet-qk256-scalar` microcrate.
//! This module keeps the runtime AVX2/scalar dispatch API stable for callers.

use anyhow::Result;

pub use bitnet_qk256_scalar::{
    I2SQk256NoScale, QK256_BLOCK, QK256_PACKED_BYTES, code_to_f32, gemv_qk256_row,
    gemv_qk256_scalar, unpack_qk256_block,
};

/// Multi-row GEMV with runtime dispatch: y = Ax where A is quantized QK256, x is dense.
///
/// This function automatically selects the best available implementation:
/// - **AVX2**: x86_64 with AVX2 support
/// - **Scalar**: Fallback for all other cases
pub fn gemv_qk256(
    qs_data: &[u8],
    x: &[f32],
    y_out: &mut [f32],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return super::i2s_qk256_avx2::gemv_qk256_avx2(
                qs_data,
                x,
                y_out,
                rows,
                cols,
                row_stride_bytes,
            );
        }
    }

    gemv_qk256_scalar(qs_data, x, y_out, rows, cols, row_stride_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemv_dispatch_smoke() {
        let rows = 2usize;
        let cols = 256usize;
        let row_stride_bytes = QK256_PACKED_BYTES;
        let qs_data = vec![0xAAu8; rows * row_stride_bytes];
        let x = vec![1.0f32; cols];
        let mut out = vec![0.0f32; rows];

        gemv_qk256(&qs_data, &x, &mut out, rows, cols, row_stride_bytes).unwrap();
        assert_eq!(out, vec![256.0f32, 256.0f32]);
    }
}
