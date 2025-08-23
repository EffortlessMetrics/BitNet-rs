#![allow(clippy::missing_safety_doc)]

use libc::{c_int, c_void};

#[cfg(feature = "iq2s-ffi")]
extern "C" {
    // Exposed by our shim; wraps GGML's reference dequantizer for IQ2_S.
    fn bitnet_dequantize_row_iq2_s(src: *const c_void, dst: *mut f32, n: c_int);
}

#[inline]
pub fn has_iq2s() -> bool {
    cfg!(feature = "iq2s-ffi")
}

#[cfg(feature = "iq2s-ffi")]
pub unsafe fn dequantize_row_iq2_s(src: *const c_void, dst: *mut f32, n: usize) {
    bitnet_dequantize_row_iq2_s(src, dst, n as c_int)
}

#[cfg(not(feature = "iq2s-ffi"))]
pub unsafe fn dequantize_row_iq2_s(_src: *const c_void, _dst: *mut f32, _n: usize) {
    panic!("IQ2_S support not compiled: enable feature `iq2s-ffi`");
}