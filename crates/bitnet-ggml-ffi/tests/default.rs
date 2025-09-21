#![cfg(not(feature = "iq2s-ffi"))]

#[test]
fn iq2s_disabled_by_default() {
    assert!(!bitnet_ggml_ffi::has_iq2s());
}
