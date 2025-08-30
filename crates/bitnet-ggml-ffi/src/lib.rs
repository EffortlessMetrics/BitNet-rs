#![allow(clippy::missing_safety_doc)]

use core::ffi::c_void;

#[cfg(feature = "iq2s-ffi")]
use libc::{c_int, size_t};

#[cfg(feature = "iq2s-ffi")]
unsafe extern "C" {
    // Exposed by our shim; wraps GGML's reference quantizer/dequantizer for IQ2_S.
    fn bitnet_dequantize_row_iq2_s(src: *const c_void, dst: *mut f32, n: c_int);
    fn bitnet_quantize_iq2_s(
        src: *const f32,
        dst: *mut c_void,
        nrow: i64,
        n_per_row: i64,
    ) -> size_t;

    // Constants extraction functions
    fn bitnet_iq2s_qk() -> c_int;
    fn bitnet_iq2s_block_size_bytes() -> c_int;
    fn bitnet_iq2s_requires_qk_multiple() -> c_int;
}

/// Returns true if IQ2_S quantization support is compiled in via the `iq2s-ffi` feature.
#[inline]
pub const fn has_iq2s() -> bool {
    cfg!(feature = "iq2s-ffi")
}

/// Returns the quantization block size for IQ2_S format (256 elements per block).
/// This constant defines how many float values are packed into a single IQ2_S block.
#[cfg(feature = "iq2s-ffi")]
pub fn iq2s_qk() -> usize {
    unsafe { bitnet_iq2s_qk() as usize }
}

/// Returns the quantization block size for IQ2_S format (256 elements per block).
/// This constant defines how many float values are packed into a single IQ2_S block.
#[cfg(not(feature = "iq2s-ffi"))]
pub fn iq2s_qk() -> usize {
    256 // Default fallback, should match QK_IQ2_S
}

/// Returns the size in bytes of a single IQ2_S quantized block structure.
/// This includes the fp16 scale factor, quantized values, and padding.
#[cfg(feature = "iq2s-ffi")]
pub fn iq2s_bytes_per_block() -> usize {
    unsafe { bitnet_iq2s_block_size_bytes() as usize }
}

/// Returns the size in bytes of a single IQ2_S quantized block structure.
/// This includes the fp16 scale factor, quantized values, and padding.
#[cfg(not(feature = "iq2s-ffi"))]
pub fn iq2s_bytes_per_block() -> usize {
    82 // Default fallback for when FFI is not available
}

/// Returns whether the IQ2_S dequantizer requires the element count to be a multiple of QK.
/// Most GGML quantizers have this requirement for correctness.
#[cfg(feature = "iq2s-ffi")]
pub fn iq2s_requires_qk_multiple() -> bool {
    unsafe { bitnet_iq2s_requires_qk_multiple() != 0 }
}

/// Returns whether the IQ2_S dequantizer requires the element count to be a multiple of QK.
/// Most GGML quantizers have this requirement for correctness.
#[cfg(not(feature = "iq2s-ffi"))]
pub fn iq2s_requires_qk_multiple() -> bool {
    true // Safer assumption
}

/// The GGML commit hash from which the IQ2_S implementation was vendored.
/// This is set at build time from the VENDORED_GGML_COMMIT file.
#[cfg(feature = "iq2s-ffi")]
pub const GGML_COMMIT: &str = env!("BITNET_GGML_COMMIT");

/// The GGML commit hash from which the IQ2_S implementation was vendored.
/// Returns "not-compiled" when the `iq2s-ffi` feature is not enabled.
#[cfg(not(feature = "iq2s-ffi"))]
pub const GGML_COMMIT: &str = "not-compiled";

/// Dequantizes a row of IQ2_S quantized data back to f32 values.
///
/// # Safety
/// - `src` must point to valid IQ2_S block data of at least `n / QK_IQ2_S` blocks
/// - `dst` must point to an array of at least `n` f32 elements
/// - `n` must be a multiple of `iq2s_qk()` (256) for correctness
#[cfg(feature = "iq2s-ffi")]
pub unsafe fn dequantize_row_iq2_s(src: *const c_void, dst: *mut f32, n: usize) {
    unsafe { bitnet_dequantize_row_iq2_s(src, dst, n as c_int) };
}

/// Quantizes multiple rows of f32 data into IQ2_S format.
///
/// # Safety
/// - `src` must point to `nrow * n_per_row` valid f32 values
/// - `dst` must have space for `nrow * (n_per_row / QK_IQ2_S) * sizeof(block_iq2_s)` bytes
/// - `n_per_row` must be a multiple of `iq2s_qk()` (256)
///
/// # Returns
/// The number of bytes written to `dst`.
#[cfg(feature = "iq2s-ffi")]
pub unsafe fn quantize_iq2_s(
    src: *const f32,
    dst: *mut c_void,
    nrow: usize,
    n_per_row: usize,
) -> usize {
    unsafe { bitnet_quantize_iq2_s(src, dst, nrow as i64, n_per_row as i64) as usize }
}

/// Dequantizes a row of IQ2_S quantized data back to f32 values.
///
/// # Panics
/// Always panics when the `iq2s-ffi` feature is not enabled.
#[cfg(not(feature = "iq2s-ffi"))]
pub unsafe fn dequantize_row_iq2_s(_src: *const c_void, _dst: *mut f32, _n: usize) {
    panic!("IQ2_S support not compiled: enable feature `iq2s-ffi`");
}

/// Quantizes multiple rows of f32 data into IQ2_S format.
///
/// # Panics
/// Always panics when the `iq2s-ffi` feature is not enabled.
#[cfg(not(feature = "iq2s-ffi"))]
pub unsafe fn quantize_iq2_s(
    _src: *const f32,
    _dst: *mut c_void,
    _nrow: usize,
    _n_per_row: usize,
) -> usize {
    panic!("IQ2_S support not compiled: enable feature `iq2s-ffi`");
}
