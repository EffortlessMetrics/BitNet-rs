//! Cross-validation support modules

pub mod backend;
pub mod preflight;

#[cfg(any(feature = "crossval", feature = "crossval-all", feature = "inference"))]
pub mod parity_both;

pub use backend::CppBackend;

// Export preflight_backend_libs for all crossval-related features (including inference)
#[cfg(any(feature = "crossval", feature = "crossval-all", feature = "inference"))]
pub use preflight::preflight_backend_libs;

// print_backend_status is only used by crossval/crossval-all commands
#[cfg(any(feature = "crossval", feature = "crossval-all"))]
pub use preflight::print_backend_status;

// Export parity_both summary functions
#[cfg(any(feature = "crossval", feature = "crossval-all", feature = "inference"))]
pub use parity_both::{
    LaneResult, ParityBothArgs, both_passed, determine_exit_code, overall_status,
    print_unified_summary,
};

// Export parity_both orchestration function (requires FFI)
#[cfg(all(
    feature = "ffi",
    any(feature = "crossval", feature = "crossval-all", feature = "inference")
))]
pub use parity_both::run_dual_lanes_and_summarize;
