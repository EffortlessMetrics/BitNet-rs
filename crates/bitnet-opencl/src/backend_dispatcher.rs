//! Multi-backend GPU dispatcher with automatic backend selection.
//!
//! Routes operations to the best available backend using configurable
//! strategies: priority-based, round-robin, load-based, or explicit
//! backend override.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::backend_registry::BackendRegistry;

// ── Core enums ───────────────────────────────────────────────────────────────

/// Operations that can be dispatched to a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    MatMul,
    Quantize,
    Dequantize,
    Softmax,
    LayerNorm,
    Attention,
    RoPE,
    Sampling,
}

/// Runtime status of a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendStatus {
    /// Backend is ready to accept work.
    Available,
    /// Backend cannot be used.
    Unavailable(String),
    /// Backend is usable but not at full capacity.
    Degraded(String),
}

impl BackendStatus {
    /// Returns `true` when the backend can execute operations.
    pub const fn is_usable(&self) -> bool {
        matches!(self, Self::Available | Self::Degraded(_))
    }
}

/// Strategy the dispatcher uses to pick a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchStrategy {
    /// Select the backend with the highest priority score.
    Priority,
    /// Rotate evenly across all usable backends.
    RoundRobin,
    /// Select the backend reporting the lowest load.
    LoadBased,
    /// Always route to the named backend.
    SpecificBackend(String),
}

// ── DispatchDecision ─────────────────────────────────────────────────────────

/// Record of a single dispatch decision.
#[derive(Debug, Clone)]
pub struct DispatchDecision {
    /// Name of the backend that was selected.
    pub chosen_backend: String,
    /// Human-readable reason for the choice.
    pub reason: String,
    /// Other backends that could have handled the operation.
    pub alternatives_available: Vec<String>,
    /// The operation that was dispatched.
    pub operation: Operation,
}

// ── BackendCapabilityMatrix ──────────────────────────────────────────────────

/// Query helper: which operations each backend supports.
pub struct BackendCapabilityMatrix<'a> {
    registry: &'a BackendRegistry,
}

impl<'a> BackendCapabilityMatrix<'a> {
    pub const fn new(registry: &'a BackendRegistry) -> Self {
        Self { registry }
    }

    /// Backends that support the given operation **and** are usable.
    pub fn backends_for(&self, op: Operation) -> Vec<String> {
        self.registry
            .discover_available()
            .into_iter()
            .filter(|info| info.status.is_usable() && info.capabilities.contains(&op))
            .map(|info| info.name)
            .collect()
    }

    /// Whether *any* usable backend supports the operation.
    pub fn is_supported(&self, op: Operation) -> bool {
        !self.backends_for(op).is_empty()
    }
}

// ── DispatchLog ──────────────────────────────────────────────────────────────

/// Append-only log of dispatch decisions for debugging.
pub struct DispatchLog {
    entries: Mutex<Vec<DispatchDecision>>,
}

impl DispatchLog {
    pub const fn new() -> Self {
        Self { entries: Mutex::new(Vec::new()) }
    }

    pub fn record(&self, decision: DispatchDecision) {
        self.entries.lock().expect("dispatch log lock poisoned").push(decision);
    }

    pub fn entries(&self) -> Vec<DispatchDecision> {
        self.entries.lock().expect("dispatch log lock poisoned").clone()
    }

    pub fn len(&self) -> usize {
        self.entries.lock().expect("dispatch log lock poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        self.entries.lock().expect("dispatch log lock poisoned").clear();
    }
}

impl Default for DispatchLog {
    fn default() -> Self {
        Self::new()
    }
}

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors produced by the dispatcher.
#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    #[error("no backend available for operation {op:?}")]
    NoBackendAvailable { op: Operation },

    #[error("specified backend '{name}' not found")]
    BackendNotFound { name: String },

    #[error("specified backend '{name}' does not support operation {op:?}")]
    OperationNotSupported { name: String, op: Operation },

    #[error("specified backend '{name}' is not usable: {status:?}")]
    BackendNotUsable { name: String, status: BackendStatus },

    #[error(
        "all backends failed for operation {op:?}; \
         last error on '{last_backend}': {last_reason}"
    )]
    AllBackendsFailed { op: Operation, last_backend: String, last_reason: String },
}

// ── BackendDispatcher ────────────────────────────────────────────────────────

/// Routes operations to the best available backend.
pub struct BackendDispatcher {
    registry: BackendRegistry,
    strategy: DispatchStrategy,
    log: DispatchLog,
    round_robin_counter: AtomicUsize,
}

impl BackendDispatcher {
    pub const fn new(registry: BackendRegistry, strategy: DispatchStrategy) -> Self {
        Self {
            registry,
            strategy,
            log: DispatchLog::new(),
            round_robin_counter: AtomicUsize::new(0),
        }
    }

    /// Read-only access to the backing registry.
    pub const fn registry(&self) -> &BackendRegistry {
        &self.registry
    }

    /// Mutable access to the backing registry (add/remove backends).
    pub const fn registry_mut(&mut self) -> &mut BackendRegistry {
        &mut self.registry
    }

    /// Current dispatch strategy.
    pub const fn strategy(&self) -> &DispatchStrategy {
        &self.strategy
    }

    /// Change the strategy at runtime.
    pub fn set_strategy(&mut self, strategy: DispatchStrategy) {
        self.strategy = strategy;
    }

    /// Reference to the dispatch log.
    pub const fn log(&self) -> &DispatchLog {
        &self.log
    }

    /// Build a capability-matrix view over the current registry.
    pub const fn capability_matrix(&self) -> BackendCapabilityMatrix<'_> {
        BackendCapabilityMatrix::new(&self.registry)
    }

    // ── dispatch ─────────────────────────────────────────────────────────

    /// Select a backend for `op` according to the current strategy.
    pub fn dispatch(&self, op: Operation) -> Result<DispatchDecision, DispatchError> {
        let decision = match &self.strategy {
            DispatchStrategy::Priority => self.dispatch_by_priority(op)?,
            DispatchStrategy::RoundRobin => self.dispatch_round_robin(op)?,
            DispatchStrategy::LoadBased => {
                // Load-based falls back to priority for now.
                self.dispatch_by_priority(op)?
            }
            DispatchStrategy::SpecificBackend(name) => self.dispatch_specific(op, name)?,
        };

        self.log.record(decision.clone());
        Ok(decision)
    }

    /// Try `op` on the preferred backend; on failure fall back through
    /// the priority chain until a backend succeeds or all fail.
    pub fn dispatch_with_fallback(&self, op: Operation) -> Result<DispatchDecision, DispatchError> {
        let mut candidates = self.usable_candidates(op);
        candidates.sort_by(|a, b| b.2.cmp(&a.2)); // highest priority first

        if candidates.is_empty() {
            return Err(DispatchError::NoBackendAvailable { op });
        }

        let mut last_backend = String::new();
        let mut last_reason = String::new();

        for (name, _, _) in &candidates {
            let backend = self.registry.get(name).expect("candidate from registry");

            if backend.status().is_usable() && backend.supports(op) {
                let alternatives: Vec<String> = candidates
                    .iter()
                    .filter(|(n, _, _)| n != name)
                    .map(|(n, _, _)| n.clone())
                    .collect();

                let decision = DispatchDecision {
                    chosen_backend: name.clone(),
                    reason: format!(
                        "fallback chain selected '{name}' \
                         (priority {})",
                        backend.priority_score()
                    ),
                    alternatives_available: alternatives,
                    operation: op,
                };
                self.log.record(decision.clone());
                return Ok(decision);
            }

            last_backend.clone_from(name);
            last_reason = format!("backend '{name}' not usable");
        }

        Err(DispatchError::AllBackendsFailed { op, last_backend, last_reason })
    }

    // ── private helpers ──────────────────────────────────────────────────

    /// Candidates that are usable and support `op`: (name, status, priority).
    fn usable_candidates(&self, op: Operation) -> Vec<(String, BackendStatus, u32)> {
        self.registry
            .discover_available()
            .into_iter()
            .filter(|info| info.status.is_usable() && info.capabilities.contains(&op))
            .map(|info| (info.name, info.status, info.priority_score))
            .collect()
    }

    fn dispatch_by_priority(&self, op: Operation) -> Result<DispatchDecision, DispatchError> {
        let mut candidates = self.usable_candidates(op);
        if candidates.is_empty() {
            return Err(DispatchError::NoBackendAvailable { op });
        }

        candidates.sort_by(|a, b| b.2.cmp(&a.2));
        let (name, _, prio) = &candidates[0];

        let alternatives: Vec<String> = candidates[1..].iter().map(|(n, _, _)| n.clone()).collect();

        Ok(DispatchDecision {
            chosen_backend: name.clone(),
            reason: format!("highest priority ({prio}) for {op:?}"),
            alternatives_available: alternatives,
            operation: op,
        })
    }

    fn dispatch_round_robin(&self, op: Operation) -> Result<DispatchDecision, DispatchError> {
        let mut candidates = self.usable_candidates(op);
        if candidates.is_empty() {
            return Err(DispatchError::NoBackendAvailable { op });
        }

        // Stable ordering so round-robin is deterministic.
        candidates.sort_by(|a, b| a.0.cmp(&b.0));

        let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % candidates.len();

        let (name, _, _) = &candidates[idx];
        let alternatives: Vec<String> = candidates
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, (n, _, _))| n.clone())
            .collect();

        Ok(DispatchDecision {
            chosen_backend: name.clone(),
            reason: format!("round-robin index {idx} for {op:?}"),
            alternatives_available: alternatives,
            operation: op,
        })
    }

    fn dispatch_specific(
        &self,
        op: Operation,
        name: &str,
    ) -> Result<DispatchDecision, DispatchError> {
        let backend = self
            .registry
            .get(name)
            .ok_or_else(|| DispatchError::BackendNotFound { name: name.to_owned() })?;

        let status = backend.status();
        if !status.is_usable() {
            return Err(DispatchError::BackendNotUsable { name: name.to_owned(), status });
        }

        if !backend.supports(op) {
            return Err(DispatchError::OperationNotSupported { name: name.to_owned(), op });
        }

        let alternatives: Vec<String> = self
            .usable_candidates(op)
            .into_iter()
            .filter(|(n, _, _)| n != name)
            .map(|(n, _, _)| n)
            .collect();

        Ok(DispatchDecision {
            chosen_backend: name.to_owned(),
            reason: format!("explicitly requested backend '{name}'"),
            alternatives_available: alternatives,
            operation: op,
        })
    }
}
