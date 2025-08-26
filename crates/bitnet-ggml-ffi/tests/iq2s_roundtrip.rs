#[cfg(feature = "iq2s-ffi")]
use bitnet_ggml_ffi::{dequantize_row_iq2_s, iq2s_bytes_per_block, iq2s_qk, quantize_iq2_s};

#[cfg(feature = "iq2s-ffi")]
#[test]
fn roundtrip_quant_dequant() {
    let n = iq2s_qk();
    let mut src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut q = vec![0u8; iq2s_bytes_per_block() * (n / iq2s_qk())];
    let mut dst = vec![0f32; n];
    unsafe {
        quantize_iq2_s(src.as_ptr(), q.as_mut_ptr() as *mut _, 1, n);
        dequantize_row_iq2_s(q.as_ptr() as *const _, dst.as_mut_ptr(), n);
    }
    for (a, b) in src.iter().zip(dst.iter()) {
        assert!((a - b).abs() < 1.5, "{} vs {}", a, b);
    }
}
