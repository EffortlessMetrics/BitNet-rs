//! GPU kernel implementations

pub mod auto_tune;
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod benchmark;
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod cuda;
pub mod memory_optimization;
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod mixed_precision;
#[cfg(feature = "oneapi")]
pub mod opencl;
pub mod opencl_error;
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod validation;

#[cfg(all(test, feature = "gpu"))]
mod tests;

pub use auto_tune::*;
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use benchmark::*;
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use cuda::*;
pub use memory_optimization::*;
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use mixed_precision::*;
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use validation::*;
