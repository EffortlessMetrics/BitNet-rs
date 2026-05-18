//! Model/kernel compatibility claim policy.
//!
//! This module records upstream model/kernel support boundaries that are
//! independent from local Rust loader or kernel correctness. It prevents known
//! unsupported combinations from becoming answer, reference, parity, or
//! benchmark authorities while still allowing diagnostic receipts.

mod claims;
mod labels;
mod support;
mod types;

pub use claims::{CompatibilityClaim, CompatibilityDecision, evaluate_model_kernel_claim};
pub use support::model_kernel_support;
pub use types::{BitnetKernel, HostArch, ModelKernelSupport};

#[cfg(test)]
mod tests;
