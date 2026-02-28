//! CPU kernel implementations

pub mod embedding;
pub mod fallback;
pub mod fusion;
pub mod simd_math;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

pub use fallback::*;
pub use simd_math::*;

#[cfg(target_arch = "x86_64")]
pub use x86::*;

#[cfg(target_arch = "aarch64")]
pub use arm::*;
