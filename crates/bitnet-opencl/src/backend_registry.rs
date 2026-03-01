//! Backend registry for discovering and managing available compute backends.

use std::collections::HashMap;

use crate::backend_dispatcher::{BackendStatus, Operation};

// ── BackendProvider trait ────────────────────────────────────────────────────

/// Trait implemented by each compute backend (CUDA, `OpenCL`, Vulkan, CPU).
pub trait BackendProvider: Send + Sync {
    /// Unique name of this backend (e.g. `"cuda"`, `"opencl"`).
    fn name(&self) -> &str;

    /// Current runtime status of the backend.
    fn status(&self) -> BackendStatus;

    /// List of operations this backend can execute.
    fn capabilities(&self) -> Vec<Operation>;

    /// Whether this backend supports a specific operation.
    fn supports(&self, op: Operation) -> bool {
        self.capabilities().contains(&op)
    }

    /// Numeric priority score (higher = preferred). Used by
    /// [`Priority`](crate::backend_dispatcher::DispatchStrategy::Priority)
    /// dispatch.
    fn priority_score(&self) -> u32;
}

// ── BackendInfo ──────────────────────────────────────────────────────────────

/// Snapshot of a registered backend's metadata.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub name: String,
    pub status: BackendStatus,
    pub capabilities: Vec<Operation>,
    pub priority_score: u32,
}

// ── BackendRegistry ──────────────────────────────────────────────────────────

/// Central registry of compute backends.
///
/// Backends register themselves at startup; the dispatcher queries the
/// registry to decide where to route each operation.
pub struct BackendRegistry {
    backends: HashMap<String, Box<dyn BackendProvider>>,
}

impl BackendRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { backends: HashMap::new() }
    }

    /// Register a backend. Replaces any previous backend with the same
    /// name.
    pub fn register(&mut self, name: &str, backend: Box<dyn BackendProvider>) {
        self.backends.insert(name.to_owned(), backend);
    }

    /// Remove a previously-registered backend. Returns `true` if found.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.backends.remove(name).is_some()
    }

    /// Look up a backend by name.
    pub fn get(&self, name: &str) -> Option<&dyn BackendProvider> {
        self.backends.get(name).map(std::convert::AsRef::as_ref)
    }

    /// Return metadata snapshots for every registered backend.
    pub fn discover_available(&self) -> Vec<BackendInfo> {
        self.backends
            .values()
            .map(|b| BackendInfo {
                name: b.name().to_owned(),
                status: b.status(),
                capabilities: b.capabilities(),
                priority_score: b.priority_score(),
            })
            .collect()
    }

    /// Number of registered backends.
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}
