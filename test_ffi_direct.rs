// Test to check FFI constants directly
fn main() {
    #[cfg(feature = "iq2s-ffi")]
    {
        println!("iq2s_qk: {}", bitnet_ggml_ffi::iq2s_qk());
        println!("iq2s_bytes_per_block: {}", bitnet_ggml_ffi::iq2s_bytes_per_block());
        println!("iq2s_requires_qk_multiple: {}", bitnet_ggml_ffi::iq2s_requires_qk_multiple());
    }
    #[cfg(not(feature = "iq2s-ffi"))]
    {
        println!("FFI feature not enabled");
    }
}