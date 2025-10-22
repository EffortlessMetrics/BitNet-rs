pub mod backend;

#[cfg(feature = "iq2s-ffi")]
pub mod iq2s;

pub mod i2s; // Native I2_S dequantization (always available)
pub mod i2s_qk256; // GGML I2_S (QK=256) scalar kernels
pub mod i2s_qk256_avx2; // GGML I2_S (QK=256) AVX2 SIMD kernels
