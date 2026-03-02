//! `bitnet-backend-core` — backend capability and selection primitives.

pub mod backend_selection;
pub mod kernel_registry;

pub use backend_selection::{
    BackendRequest, BackendSelectionError, BackendSelectionResult, BackendStartupSummary,
    select_backend,
};
pub use kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};
