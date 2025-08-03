//! CPU kernel implementations

// Re-export kernel implementations

pub mod fallback;
#[cfg(target_arch = "x86_64")]
pub mod x86;
#[cfg(target_arch = "aarch64")]
pub mod arm;

pub use fallback::FallbackKernel;
#[cfg(target_arch = "x86_64")]
pub use x86::Avx2Kernel;
#[cfg(target_arch = "aarch64")]
pub use arm::NeonKernel;

