//! Re-exports of AVX2-focused SIMD math primitives.
//!
//! The implementation lives in the `bitnet-simd-avx2` SRP microcrate.

pub use bitnet_simd_avx2::{
    fast_exp_f32, fast_sigmoid_f32, fast_tanh_f32, simd_dot_product, simd_l2_norm, simd_vector_add,
    simd_vector_mul, simd_vector_scale,
};
