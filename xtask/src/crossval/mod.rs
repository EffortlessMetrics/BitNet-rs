//! Cross-validation support modules

pub mod backend;
pub mod preflight;

pub use backend::CppBackend;

// Export preflight_backend_libs for all crossval-related features (including inference)
#[cfg(any(feature = "crossval", feature = "crossval-all", feature = "inference"))]
pub use preflight::preflight_backend_libs;

// print_backend_status is only used by crossval/crossval-all commands
#[cfg(any(feature = "crossval", feature = "crossval-all"))]
pub use preflight::print_backend_status;
