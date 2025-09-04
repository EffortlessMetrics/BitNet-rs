#![allow(clippy::missing_safety_doc)]

#[cfg(feature = "iq2s-ffi")]
use libc::c_int;
use libc::c_void;

#[cfg(feature = "iq2s-ffi")]
unsafe extern "C" {
    // Exposed by our shim; wraps GGML's reference dequantizer for IQ2_S.
    fn bitnet_dequantize_row_iq2_s(src: *const c_void, dst: *mut f32, n: c_int);

    // Constants extraction functions
    fn bitnet_iq2s_qk() -> c_int;
    fn bitnet_iq2s_block_size_bytes() -> c_int;
    fn bitnet_iq2s_requires_qk_multiple() -> c_int;
}

#[inline]
pub const fn has_iq2s() -> bool {
    cfg!(feature = "iq2s-ffi")
}

#[cfg(feature = "iq2s-ffi")]
pub fn iq2s_qk() -> usize {
    unsafe { bitnet_iq2s_qk() as usize }
}

#[cfg(not(feature = "iq2s-ffi"))]
pub fn iq2s_qk() -> usize {
    256 // Default fallback, should match QK_IQ2_S
}

#[cfg(feature = "iq2s-ffi")]
pub fn iq2s_bytes_per_block() -> usize {
    unsafe { bitnet_iq2s_block_size_bytes() as usize }
}

#[cfg(not(feature = "iq2s-ffi"))]
pub fn iq2s_bytes_per_block() -> usize {
    66 // Default fallback for when FFI is not available
}

#[cfg(feature = "iq2s-ffi")]
pub fn iq2s_requires_qk_multiple() -> bool {
    unsafe { bitnet_iq2s_requires_qk_multiple() != 0 }
}

#[cfg(not(feature = "iq2s-ffi"))]
pub fn iq2s_requires_qk_multiple() -> bool {
    true // Safer assumption
}

#[cfg(feature = "iq2s-ffi")]
pub const GGML_COMMIT: &str = env!("BITNET_GGML_COMMIT");

#[cfg(not(feature = "iq2s-ffi"))]
pub const GGML_COMMIT: &str = "not-compiled";

#[cfg(feature = "iq2s-ffi")]
pub unsafe fn dequantize_row_iq2_s(src: *const c_void, dst: *mut f32, n: usize) {
    unsafe { bitnet_dequantize_row_iq2_s(src, dst, n as c_int) }
}

#[cfg(not(feature = "iq2s-ffi"))]
pub unsafe fn dequantize_row_iq2_s(_src: *const c_void, _dst: *mut f32, _n: usize) {
    panic!("IQ2_S support not compiled: enable feature `iq2s-ffi`");
}
