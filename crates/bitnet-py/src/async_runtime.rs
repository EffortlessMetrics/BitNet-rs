//! Async Runtime Management for Python Bindings

use pyo3::prelude::*;
use std::sync::OnceLock;
use tokio::runtime::Runtime;

/// Global async runtime for Python bindings
static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Get or create the global async runtime
pub fn get_runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| Runtime::new().expect("Failed to create async runtime"))
}

/// Execute a future on the global runtime
pub fn block_on<F: std::future::Future>(future: F) -> F::Output {
    get_runtime().block_on(future)
}

/// Spawn a task on the global runtime
pub fn spawn<F>(future: F) -> tokio::task::JoinHandle<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    get_runtime().spawn(future)
}
