//! SIMD-optimized dot product operations for `BitNet` inference kernels.
//!
//! Provides high-performance dot products across multiple data types (`f32`, `i8`,
//! binary/popcount) with **runtime dispatch** to the best available SIMD level:
//!
//! | Level    | `x86_64`      | `aarch64` |
//! |----------|---------------|-----------|
//! | Scalar   | ✓ (fallback)  | ✓         |
//! | SSE 4.1  | ✓             | —         |
//! | AVX2     | ✓             | —         |
//! | AVX-512  | ✓             | —         |
//! | NEON     | —             | ✓         |
//!
//! # Features
//!
//! * **`f32` dot product** — vectorised FMA accumulation
//! * **`i8` dot product** — widening multiply-add into `i32`
//! * **Binary dot product** — XOR + popcount for Hamming-style similarity
//! * **Fused multiply-accumulate** — `a · b + c · d` in a single pass
//! * **Strided dot product** — skip elements at a fixed stride
//! * **Batched dot product** — `N` independent dot products in one call

mod dispatch;
mod scalar;

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "aarch64")]
mod neon;

pub use dispatch::{
    SimdLevel, batched_dot_f32, binary_dot, dot_f32, dot_i8, fma_dot_f32, strided_dot_f32,
};

#[cfg(test)]
mod tests;
