//! CPU kernel implementations

pub mod x86;
pub mod arm;
pub mod fallback;

pub use x86::*;
pub use arm::*;
pub use fallback::*;