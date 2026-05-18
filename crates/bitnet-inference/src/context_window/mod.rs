//! Context window management.
//!
//! Track and manage the token context window for inference.

mod budgets;
mod report;
mod window;

#[cfg(test)]
mod tests;

pub use budgets::{AllocationStrategy, compute_budgets};
pub use report::ContextReport;
pub use window::ContextWindow;
