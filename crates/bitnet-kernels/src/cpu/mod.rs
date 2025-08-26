//! CPU kernel implementations

pub mod arm;
pub mod fallback;

#[cfg(target_arch = "x86_64")]
pub mod x86;

pub use arm::*;
pub use fallback::*;

#[cfg(target_arch = "x86_64")]
pub use x86::*;
