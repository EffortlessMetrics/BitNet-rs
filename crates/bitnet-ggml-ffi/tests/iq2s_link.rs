#![cfg(feature = "integration-tests")]
#[cfg(feature = "iq2s-ffi")]
#[test]
fn iq2s_symbol_is_linked() {
    // Just verify we can call it without UB; we can't fabricate valid blocks here.
    // We pass n=0 which should be a no-op in ggml.
    unsafe {
        let src = std::ptr::null();
        let mut dst_val = 0f32;
        bitnet_ggml_ffi::dequantize_row_iq2_s(src, &mut dst_val as *mut _, 0);
    }
    assert!(bitnet_ggml_ffi::has_iq2s());
}

#[cfg(not(feature = "iq2s-ffi"))]
#[test]
fn iq2s_not_available() {
    assert!(!bitnet_ggml_ffi::has_iq2s());
}
