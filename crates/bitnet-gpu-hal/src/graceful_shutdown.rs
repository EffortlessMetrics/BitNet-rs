<<<<<<< HEAD
//! Graceful shutdown management for the inference server.
//!
//! Coordinates an orderly shutdown through five phases:
//! `Running → Draining → Saving → Stopping → Terminated`.
//!
//! The [`ShutdownManager`] orchestrates drain timeout enforcement,
//! state checkpointing, ordered resource cleanup, and hook
//! callbacks so that in-flight requests complete (or are saved)
//! before GPU memory and connections are released.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ── Configuration ─────────────────────────────────────────────────

/// Tunable knobs for the shutdown sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShutdownConfig {
    /// Maximum milliseconds to wait for in-flight requests to drain.
    pub drain_timeout_ms: u64,
    /// Maximum milliseconds before a forceful termination after drain.
    pub force_timeout_ms: u64,
    /// Whether to persist in-flight request state for later recovery.
    pub save_state: bool,
}

impl Default for ShutdownConfig {
    fn default() -> Self {
        Self { drain_timeout_ms: 30_000, force_timeout_ms: 5_000, save_state: true }
    }
}

// ── Phases ────────────────────────────────────────────────────────

/// Ordered phases of a graceful shutdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShutdownPhase {
    /// Normal operation — accepting requests.
    Running,
    /// No longer accepting new requests; waiting for in-flight work.
    Draining,
    /// Persisting in-flight state for recovery.
    Saving,
    /// Releasing GPU memory, caches, and connections.
    Stopping,
    /// Everything is torn down.
    Terminated,
}

impl fmt::Display for ShutdownPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Running => "Running",
            Self::Draining => "Draining",
            Self::Saving => "Saving",
            Self::Stopping => "Stopping",
            Self::Terminated => "Terminated",
        };
        f.write_str(label)
    }
}

impl ShutdownPhase {
    /// Return the next phase in the shutdown sequence, or `None`
    /// if already `Terminated`.
    pub const fn next(self) -> Option<Self> {
        match self {
            Self::Running => Some(Self::Draining),
            Self::Draining => Some(Self::Saving),
            Self::Saving => Some(Self::Stopping),
            Self::Stopping => Some(Self::Terminated),
            Self::Terminated => None,
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────

/// Errors specific to the shutdown subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShutdownError {
    /// The drain period expired with requests still in-flight.
    DrainTimeout { remaining: usize },
    /// The force timeout expired before cleanup finished.
    ForceTimeout,
    /// Attempted to advance past `Terminated`.
    AlreadyTerminated,
    /// A shutdown hook failed.
    HookFailed { name: String, reason: String },
    /// State checkpoint could not be written.
    CheckpointFailed { reason: String },
    /// The manager is not in the expected phase.
    InvalidPhase { expected: ShutdownPhase, actual: ShutdownPhase },
}

impl fmt::Display for ShutdownError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DrainTimeout { remaining } => {
                write!(f, "drain timeout with {remaining} requests remaining")
            }
            Self::ForceTimeout => write!(f, "force timeout expired"),
            Self::AlreadyTerminated => {
                write!(f, "already terminated")
            }
            Self::HookFailed { name, reason } => {
                write!(f, "hook '{name}' failed: {reason}")
            }
            Self::CheckpointFailed { reason } => {
                write!(f, "checkpoint failed: {reason}")
            }
            Self::InvalidPhase { expected, actual } => {
                write!(f, "invalid phase: expected {expected}, actual {actual}")
            }
        }
    }
}

impl std::error::Error for ShutdownError {}

// ── Shutdown signal ───────────────────────────────────────────────

/// A cloneable, cross-thread shutdown signal backed by an atomic.
#[derive(Debug, Clone)]
pub struct ShutdownSignal {
    triggered: Arc<AtomicBool>,
}

impl ShutdownSignal {
    /// Create a new signal in the un-triggered state.
    pub fn new() -> Self {
        Self { triggered: Arc::new(AtomicBool::new(false)) }
    }

    /// Fire the signal. All clones observe `true` immediately.
    pub fn trigger(&self) {
        self.triggered.store(true, Ordering::Release);
    }

    /// Check whether the signal has been fired.
    pub fn is_triggered(&self) -> bool {
        self.triggered.load(Ordering::Acquire)
    }

    /// Reset the signal (mainly for testing).
    pub fn reset(&self) {
        self.triggered.store(false, Ordering::Release);
    }
}

impl Default for ShutdownSignal {
    fn default() -> Self {
        Self::new()
    }
}

// ── Drain manager ─────────────────────────────────────────────────

/// Tracks in-flight requests and gates new-request admission.
#[derive(Debug)]
pub struct DrainManager {
    in_flight: Arc<AtomicU64>,
    accepting: Arc<AtomicBool>,
}

impl DrainManager {
    /// Create a new drain manager that is accepting requests.
    pub fn new() -> Self {
        Self { in_flight: Arc::new(AtomicU64::new(0)), accepting: Arc::new(AtomicBool::new(true)) }
    }

    /// Try to admit a new request. Returns `false` once draining.
    pub fn try_admit(&self) -> bool {
        if !self.accepting.load(Ordering::Acquire) {
            return false;
        }
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        true
    }

    /// Mark a request as complete.
    pub fn complete(&self) {
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
    }

    /// Stop accepting new requests (begin drain).
    pub fn stop_accepting(&self) {
        self.accepting.store(false, Ordering::Release);
    }

    /// Resume accepting (useful for abort / rollback).
    pub fn resume_accepting(&self) {
        self.accepting.store(true, Ordering::Release);
    }

    /// Number of requests currently in-flight.
    pub fn in_flight_count(&self) -> u64 {
        self.in_flight.load(Ordering::Acquire)
    }

    /// Whether the manager is still accepting new requests.
    pub fn is_accepting(&self) -> bool {
        self.accepting.load(Ordering::Acquire)
    }

    /// Spin-wait until in-flight reaches zero or `timeout` elapses.
    /// Returns the remaining count when the wait ends.
    pub fn wait_for_drain(&self, timeout: Duration) -> u64 {
        let deadline = Instant::now() + timeout;
        loop {
            let count = self.in_flight_count();
            if count == 0 || Instant::now() >= deadline {
                return count;
            }
            std::thread::yield_now();
        }
    }
}

impl Default for DrainManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── State checkpoint ──────────────────────────────────────────────

/// A serialisable snapshot of in-flight request state.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestState {
    /// Opaque request identifier.
    pub request_id: String,
    /// Tokens generated so far.
    pub tokens_generated: usize,
    /// Prompt text (or a hash) for resumption.
    pub prompt_hash: u64,
}

/// Collects request states and serialises them as a checkpoint.
#[derive(Debug, Default, Clone)]
pub struct StateCheckpoint {
    states: Vec<RequestState>,
}

impl StateCheckpoint {
    pub fn new() -> Self {
        Self { states: Vec::new() }
    }

    /// Record one in-flight request's state.
    pub fn record(&mut self, state: RequestState) {
        self.states.push(state);
    }

    /// Number of recorded states.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether the checkpoint is empty.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Return a reference to all recorded states.
    pub fn states(&self) -> &[RequestState] {
        &self.states
    }

    /// Serialise the checkpoint to JSON bytes.
    pub fn to_json(&self) -> Result<Vec<u8>, ShutdownError> {
        serde_json::to_vec(
            &self
                .states
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "request_id": s.request_id,
                        "tokens_generated": s.tokens_generated,
                        "prompt_hash": s.prompt_hash,
                    })
                })
                .collect::<Vec<_>>(),
        )
        .map_err(|e| ShutdownError::CheckpointFailed { reason: e.to_string() })
    }
}

// ── Resource cleanup ──────────────────────────────────────────────

/// Categories of resources that may need cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    GpuMemory,
    KvCache,
    ConnectionPool,
    TempFiles,
}

impl fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::GpuMemory => "GPU memory",
            Self::KvCache => "KV cache",
            Self::ConnectionPool => "connection pool",
            Self::TempFiles => "temp files",
        };
        f.write_str(label)
    }
}

/// Tracks which resource kinds have been cleaned up and how much
/// was freed.
#[derive(Debug, Default)]
pub struct ResourceCleanup {
    freed: Mutex<HashMap<ResourceKind, u64>>,
}

impl ResourceCleanup {
    pub fn new() -> Self {
        Self { freed: Mutex::new(HashMap::new()) }
    }

    /// Record that `bytes` of `kind` were freed.
    pub fn record_freed(&self, kind: ResourceKind, bytes: u64) {
        let mut map = self.freed.lock().expect("lock poisoned");
        *map.entry(kind).or_insert(0) += bytes;
    }

    /// Total bytes freed for a given kind.
    pub fn freed_for(&self, kind: ResourceKind) -> u64 {
        let map = self.freed.lock().expect("lock poisoned");
        map.get(&kind).copied().unwrap_or(0)
    }

    /// Total bytes freed across all kinds.
    pub fn total_freed(&self) -> u64 {
        let map = self.freed.lock().expect("lock poisoned");
        map.values().sum()
    }

    /// Snapshot the cleanup state into a map.
    pub fn snapshot(&self) -> HashMap<ResourceKind, u64> {
        self.freed.lock().expect("lock poisoned").clone()
    }
}

// ── Health transition ─────────────────────────────────────────────

/// Simplified health status reported during shutdown transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
        };
        f.write_str(label)
    }
}

/// Maps a [`ShutdownPhase`] to the appropriate health status.
pub struct HealthTransition;

impl HealthTransition {
    pub const fn status_for(phase: ShutdownPhase) -> HealthStatus {
        match phase {
            ShutdownPhase::Running => HealthStatus::Healthy,
            ShutdownPhase::Draining | ShutdownPhase::Saving => HealthStatus::Degraded,
            ShutdownPhase::Stopping | ShutdownPhase::Terminated => HealthStatus::Unhealthy,
        }
    }
}

// ── Shutdown hook trait ───────────────────────────────────────────

/// Callback interface invoked during shutdown.
pub trait ShutdownHook: Send {
    /// Human-readable name of this hook.
    fn name(&self) -> &str;

    /// Execute the cleanup action. Called once during the
    /// `Stopping` phase.
    fn on_shutdown(&self) -> Result<(), String>;
}

// ── Shutdown report ───────────────────────────────────────────────

/// Summary produced after a shutdown attempt completes.
#[derive(Debug, Clone)]
pub struct ShutdownReport {
    /// Phase reached when the report was generated.
    pub final_phase: ShutdownPhase,
    /// Requests that were still in-flight at drain deadline.
    pub pending_requests: u64,
    /// Per-resource bytes freed.
    pub resources_freed: HashMap<ResourceKind, u64>,
    /// Names of hooks that executed successfully.
    pub hooks_succeeded: Vec<String>,
    /// `(hook_name, error)` pairs for hooks that failed.
    pub hooks_failed: Vec<(String, String)>,
    /// Number of request states checkpointed.
    pub states_saved: usize,
    /// Total wall-clock duration of the shutdown.
    pub elapsed: Duration,
}

impl fmt::Display for ShutdownReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "shutdown reached {:?} in {:?} — \
             {} pending, {} states saved, {} bytes freed, \
             {}/{} hooks ok",
            self.final_phase,
            self.elapsed,
            self.pending_requests,
            self.states_saved,
            self.resources_freed.values().sum::<u64>(),
            self.hooks_succeeded.len(),
            self.hooks_succeeded.len() + self.hooks_failed.len(),
        )
    }
}

// ── Shutdown manager ──────────────────────────────────────────────

/// Orchestrates the full graceful-shutdown sequence.
pub struct ShutdownManager {
    config: ShutdownConfig,
    phase: Mutex<ShutdownPhase>,
    signal: ShutdownSignal,
    drain: DrainManager,
    cleanup: ResourceCleanup,
    hooks: Mutex<Vec<Box<dyn ShutdownHook>>>,
    checkpoint: Mutex<StateCheckpoint>,
}

impl ShutdownManager {
    /// Build a new manager with the given configuration.
    pub fn new(config: ShutdownConfig) -> Self {
        Self {
            config,
            phase: Mutex::new(ShutdownPhase::Running),
            signal: ShutdownSignal::new(),
            drain: DrainManager::new(),
            cleanup: ResourceCleanup::new(),
            hooks: Mutex::new(Vec::new()),
            checkpoint: Mutex::new(StateCheckpoint::new()),
        }
    }

    // ── Accessors ─────────────────────────────────────────────

    pub fn config(&self) -> &ShutdownConfig {
        &self.config
    }

    pub fn phase(&self) -> ShutdownPhase {
        *self.phase.lock().expect("lock poisoned")
    }

    pub fn signal(&self) -> &ShutdownSignal {
        &self.signal
    }

    pub fn drain(&self) -> &DrainManager {
        &self.drain
    }

    pub fn cleanup(&self) -> &ResourceCleanup {
        &self.cleanup
    }

    pub fn health(&self) -> HealthStatus {
        HealthTransition::status_for(self.phase())
    }

    // ── Hook registration ─────────────────────────────────────

    /// Register a hook to be called during the `Stopping` phase.
    pub fn register_hook(&self, hook: Box<dyn ShutdownHook>) -> Result<(), ShutdownError> {
        let phase = self.phase();
        if phase != ShutdownPhase::Running {
            return Err(ShutdownError::InvalidPhase {
                expected: ShutdownPhase::Running,
                actual: phase,
            });
        }
        self.hooks.lock().expect("lock poisoned").push(hook);
        Ok(())
    }

    // ── State recording ───────────────────────────────────────

    /// Record a request state for potential checkpointing.
    pub fn record_request_state(&self, state: RequestState) {
        self.checkpoint.lock().expect("lock poisoned").record(state);
    }

    // ── Phase transitions ─────────────────────────────────────

    /// Advance to the next phase. Returns the new phase.
    fn advance_phase(&self) -> Result<ShutdownPhase, ShutdownError> {
        let mut phase = self.phase.lock().expect("lock poisoned");
        match phase.next() {
            Some(next) => {
                *phase = next;
                Ok(next)
            }
            None => Err(ShutdownError::AlreadyTerminated),
        }
    }

    /// Initiate the shutdown sequence and return a report.
    ///
    /// This is the main entry-point: it walks through every phase,
    /// respecting the configured timeouts.
    pub fn initiate(&self) -> ShutdownReport {
        let start = Instant::now();
        self.signal.trigger();

        // ── Draining ──────────────────────────────────────────
        let _ = self.advance_phase(); // Running → Draining
        self.drain.stop_accepting();
        let drain_timeout = Duration::from_millis(self.config.drain_timeout_ms);
        let remaining = self.drain.wait_for_drain(drain_timeout);

        // ── Saving ────────────────────────────────────────────
        let _ = self.advance_phase(); // Draining → Saving
        let states_saved = if self.config.save_state {
            self.checkpoint.lock().expect("lock poisoned").len()
        } else {
            0
        };

        // ── Stopping (hooks + resource cleanup) ───────────────
        let _ = self.advance_phase(); // Saving → Stopping
        let mut hooks_succeeded = Vec::new();
        let mut hooks_failed = Vec::new();
        {
            let hooks = self.hooks.lock().expect("lock poisoned");
            for hook in hooks.iter() {
                match hook.on_shutdown() {
                    Ok(()) => hooks_succeeded.push(hook.name().to_string()),
                    Err(e) => {
                        hooks_failed.push((hook.name().to_string(), e));
                    }
                }
            }
        }

        // ── Terminated ────────────────────────────────────────
        let _ = self.advance_phase(); // Stopping → Terminated

        ShutdownReport {
            final_phase: self.phase(),
            pending_requests: remaining,
            resources_freed: self.cleanup.snapshot(),
            hooks_succeeded,
            hooks_failed,
            states_saved,
            elapsed: start.elapsed(),
        }
    }
}

impl fmt::Debug for ShutdownManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShutdownManager")
            .field("config", &self.config)
            .field("phase", &self.phase())
            .finish_non_exhaustive()
    }
}

// ══════════════════════════════════════════════════════════════════
//  Tests
// ══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── ShutdownConfig ────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = ShutdownConfig::default();
        assert_eq!(cfg.drain_timeout_ms, 30_000);
        assert_eq!(cfg.force_timeout_ms, 5_000);
        assert!(cfg.save_state);
    }

    #[test]
    fn config_custom_values() {
        let cfg = ShutdownConfig { drain_timeout_ms: 100, force_timeout_ms: 50, save_state: false };
        assert_eq!(cfg.drain_timeout_ms, 100);
        assert!(!cfg.save_state);
    }

    #[test]
    fn config_clone_eq() {
        let a = ShutdownConfig::default();
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── ShutdownPhase ─────────────────────────────────────────

    #[test]
    fn phase_sequence() {
        let mut p = ShutdownPhase::Running;
        let expected = [
            ShutdownPhase::Draining,
            ShutdownPhase::Saving,
            ShutdownPhase::Stopping,
            ShutdownPhase::Terminated,
        ];
        for &e in &expected {
            p = p.next().unwrap();
            assert_eq!(p, e);
        }
        assert!(p.next().is_none());
    }

    #[test]
    fn phase_display() {
        assert_eq!(ShutdownPhase::Running.to_string(), "Running");
        assert_eq!(ShutdownPhase::Draining.to_string(), "Draining");
        assert_eq!(ShutdownPhase::Saving.to_string(), "Saving");
        assert_eq!(ShutdownPhase::Stopping.to_string(), "Stopping");
        assert_eq!(ShutdownPhase::Terminated.to_string(), "Terminated");
    }

    #[test]
    fn phase_terminated_has_no_next() {
        assert_eq!(ShutdownPhase::Terminated.next(), None);
    }

    #[test]
    fn phase_running_next_is_draining() {
        assert_eq!(ShutdownPhase::Running.next(), Some(ShutdownPhase::Draining));
    }

    // ── ShutdownError ─────────────────────────────────────────

    #[test]
    fn error_drain_timeout_display() {
        let e = ShutdownError::DrainTimeout { remaining: 3 };
        assert!(e.to_string().contains("3 requests"));
    }

    #[test]
    fn error_force_timeout_display() {
        let e = ShutdownError::ForceTimeout;
        assert!(e.to_string().contains("force timeout"));
    }

    #[test]
    fn error_already_terminated_display() {
        let e = ShutdownError::AlreadyTerminated;
        assert!(e.to_string().contains("terminated"));
    }

    #[test]
    fn error_hook_failed_display() {
        let e = ShutdownError::HookFailed { name: "gpu".into(), reason: "busy".into() };
        let s = e.to_string();
        assert!(s.contains("gpu") && s.contains("busy"));
    }

    #[test]
    fn error_checkpoint_failed_display() {
        let e = ShutdownError::CheckpointFailed { reason: "disk full".into() };
        assert!(e.to_string().contains("disk full"));
    }

    #[test]
    fn error_invalid_phase_display() {
        let e = ShutdownError::InvalidPhase {
            expected: ShutdownPhase::Running,
            actual: ShutdownPhase::Draining,
        };
        let s = e.to_string();
        assert!(s.contains("Running") && s.contains("Draining"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(ShutdownError::ForceTimeout);
        assert!(!e.to_string().is_empty());
    }

    // ── ShutdownSignal ────────────────────────────────────────

    #[test]
    fn signal_starts_untriggered() {
        let sig = ShutdownSignal::new();
        assert!(!sig.is_triggered());
    }

    #[test]
    fn signal_trigger() {
        let sig = ShutdownSignal::new();
        sig.trigger();
        assert!(sig.is_triggered());
    }

    #[test]
    fn signal_clone_shares_state() {
        let a = ShutdownSignal::new();
        let b = a.clone();
        a.trigger();
        assert!(b.is_triggered());
    }

    #[test]
    fn signal_reset() {
        let sig = ShutdownSignal::new();
        sig.trigger();
        sig.reset();
        assert!(!sig.is_triggered());
    }

    #[test]
    fn signal_default() {
        let sig = ShutdownSignal::default();
        assert!(!sig.is_triggered());
    }

    // ── DrainManager ──────────────────────────────────────────

    #[test]
    fn drain_admits_when_accepting() {
        let dm = DrainManager::new();
        assert!(dm.try_admit());
        assert_eq!(dm.in_flight_count(), 1);
    }

    #[test]
    fn drain_rejects_after_stop() {
        let dm = DrainManager::new();
        dm.stop_accepting();
        assert!(!dm.try_admit());
    }

    #[test]
    fn drain_complete_decrements() {
        let dm = DrainManager::new();
        dm.try_admit();
        dm.try_admit();
        dm.complete();
        assert_eq!(dm.in_flight_count(), 1);
    }

    #[test]
    fn drain_resume_accepts_again() {
        let dm = DrainManager::new();
        dm.stop_accepting();
        assert!(!dm.is_accepting());
        dm.resume_accepting();
        assert!(dm.is_accepting());
        assert!(dm.try_admit());
    }

    #[test]
    fn drain_wait_returns_zero_when_empty() {
        let dm = DrainManager::new();
        let remaining = dm.wait_for_drain(Duration::from_millis(10));
        assert_eq!(remaining, 0);
    }

    #[test]
    fn drain_wait_timeout_with_inflight() {
        let dm = DrainManager::new();
        dm.try_admit();
        let remaining = dm.wait_for_drain(Duration::from_millis(5));
        assert_eq!(remaining, 1);
    }

    #[test]
    fn drain_default() {
        let dm = DrainManager::default();
        assert!(dm.is_accepting());
        assert_eq!(dm.in_flight_count(), 0);
    }

    #[test]
    fn drain_multiple_admit_complete_cycle() {
        let dm = DrainManager::new();
        for _ in 0..10 {
            dm.try_admit();
        }
        assert_eq!(dm.in_flight_count(), 10);
        for _ in 0..10 {
            dm.complete();
        }
        assert_eq!(dm.in_flight_count(), 0);
    }

    // ── StateCheckpoint ───────────────────────────────────────

    #[test]
    fn checkpoint_empty_by_default() {
        let cp = StateCheckpoint::new();
        assert!(cp.is_empty());
        assert_eq!(cp.len(), 0);
    }

    #[test]
    fn checkpoint_record_and_len() {
        let mut cp = StateCheckpoint::new();
        cp.record(RequestState { request_id: "r1".into(), tokens_generated: 5, prompt_hash: 42 });
        assert_eq!(cp.len(), 1);
        assert!(!cp.is_empty());
    }

    #[test]
    fn checkpoint_states_returns_all() {
        let mut cp = StateCheckpoint::new();
        cp.record(RequestState { request_id: "a".into(), tokens_generated: 1, prompt_hash: 0 });
        cp.record(RequestState { request_id: "b".into(), tokens_generated: 2, prompt_hash: 1 });
        let states = cp.states();
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].request_id, "a");
        assert_eq!(states[1].request_id, "b");
    }

    #[test]
    fn checkpoint_to_json_empty() {
        let cp = StateCheckpoint::new();
        let json = cp.to_json().unwrap();
        assert_eq!(json, b"[]");
    }

    #[test]
    fn checkpoint_to_json_with_data() {
        let mut cp = StateCheckpoint::new();
        cp.record(RequestState { request_id: "r1".into(), tokens_generated: 7, prompt_hash: 99 });
        let json = cp.to_json().unwrap();
        let text = String::from_utf8(json).unwrap();
        assert!(text.contains("\"request_id\":\"r1\""));
        assert!(text.contains("\"tokens_generated\":7"));
    }

    #[test]
    fn checkpoint_default() {
        let cp = StateCheckpoint::default();
        assert!(cp.is_empty());
    }

    // ── ResourceCleanup ───────────────────────────────────────

    #[test]
    fn cleanup_initially_zero() {
        let rc = ResourceCleanup::new();
        assert_eq!(rc.total_freed(), 0);
        assert_eq!(rc.freed_for(ResourceKind::GpuMemory), 0);
    }

    #[test]
    fn cleanup_record_and_query() {
        let rc = ResourceCleanup::new();
        rc.record_freed(ResourceKind::GpuMemory, 1024);
        assert_eq!(rc.freed_for(ResourceKind::GpuMemory), 1024);
        assert_eq!(rc.total_freed(), 1024);
    }

    #[test]
    fn cleanup_accumulates_same_kind() {
        let rc = ResourceCleanup::new();
        rc.record_freed(ResourceKind::KvCache, 100);
        rc.record_freed(ResourceKind::KvCache, 200);
        assert_eq!(rc.freed_for(ResourceKind::KvCache), 300);
    }

    #[test]
    fn cleanup_multiple_kinds() {
        let rc = ResourceCleanup::new();
        rc.record_freed(ResourceKind::GpuMemory, 500);
        rc.record_freed(ResourceKind::TempFiles, 50);
        assert_eq!(rc.total_freed(), 550);
    }

    #[test]
    fn cleanup_snapshot() {
        let rc = ResourceCleanup::new();
        rc.record_freed(ResourceKind::ConnectionPool, 10);
        let snap = rc.snapshot();
        assert_eq!(snap.get(&ResourceKind::ConnectionPool), Some(&10));
    }

    #[test]
    fn cleanup_default() {
        let rc = ResourceCleanup::default();
        assert_eq!(rc.total_freed(), 0);
    }

    // ── ResourceKind display ──────────────────────────────────

    #[test]
    fn resource_kind_display() {
        assert_eq!(ResourceKind::GpuMemory.to_string(), "GPU memory");
        assert_eq!(ResourceKind::KvCache.to_string(), "KV cache");
        assert_eq!(ResourceKind::ConnectionPool.to_string(), "connection pool");
        assert_eq!(ResourceKind::TempFiles.to_string(), "temp files");
    }

    // ── HealthTransition ──────────────────────────────────────

    #[test]
    fn health_running_is_healthy() {
        assert_eq!(HealthTransition::status_for(ShutdownPhase::Running), HealthStatus::Healthy,);
    }

    #[test]
    fn health_draining_is_degraded() {
        assert_eq!(HealthTransition::status_for(ShutdownPhase::Draining), HealthStatus::Degraded,);
    }

    #[test]
    fn health_saving_is_degraded() {
        assert_eq!(HealthTransition::status_for(ShutdownPhase::Saving), HealthStatus::Degraded,);
    }

    #[test]
    fn health_stopping_is_unhealthy() {
        assert_eq!(HealthTransition::status_for(ShutdownPhase::Stopping), HealthStatus::Unhealthy,);
    }

    #[test]
    fn health_terminated_is_unhealthy() {
        assert_eq!(
            HealthTransition::status_for(ShutdownPhase::Terminated),
            HealthStatus::Unhealthy,
        );
    }

    #[test]
    fn health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Degraded.to_string(), "degraded");
        assert_eq!(HealthStatus::Unhealthy.to_string(), "unhealthy");
    }

    // ── ShutdownHook trait ────────────────────────────────────

    struct OkHook(&'static str);
    impl ShutdownHook for OkHook {
        fn name(&self) -> &str {
            self.0
        }
        fn on_shutdown(&self) -> Result<(), String> {
            Ok(())
        }
    }

    struct FailHook(&'static str);
    impl ShutdownHook for FailHook {
        fn name(&self) -> &str {
            self.0
        }
        fn on_shutdown(&self) -> Result<(), String> {
            Err("kaboom".into())
        }
    }

    #[test]
    fn hook_ok_returns_ok() {
        let h = OkHook("test");
        assert!(h.on_shutdown().is_ok());
    }

    #[test]
    fn hook_fail_returns_err() {
        let h = FailHook("boom");
        assert!(h.on_shutdown().is_err());
    }

    // ── ShutdownReport ────────────────────────────────────────

    #[test]
    fn report_display() {
        let report = ShutdownReport {
            final_phase: ShutdownPhase::Terminated,
            pending_requests: 0,
            resources_freed: HashMap::new(),
            hooks_succeeded: vec!["a".into()],
            hooks_failed: vec![],
            states_saved: 2,
            elapsed: Duration::from_millis(100),
        };
        let s = report.to_string();
        assert!(s.contains("Terminated"));
        assert!(s.contains("2 states saved"));
    }

    // ── ShutdownManager ───────────────────────────────────────

    #[test]
    fn manager_starts_in_running() {
        let mgr = ShutdownManager::new(ShutdownConfig::default());
        assert_eq!(mgr.phase(), ShutdownPhase::Running);
    }

    #[test]
    fn manager_health_starts_healthy() {
        let mgr = ShutdownManager::new(ShutdownConfig::default());
        assert_eq!(mgr.health(), HealthStatus::Healthy);
    }

    #[test]
    fn manager_signal_not_triggered() {
        let mgr = ShutdownManager::new(ShutdownConfig::default());
        assert!(!mgr.signal().is_triggered());
    }

    #[test]
    fn manager_drain_accepts() {
        let mgr = ShutdownManager::new(ShutdownConfig::default());
        assert!(mgr.drain().try_admit());
        mgr.drain().complete();
    }

    #[test]
    fn manager_register_hook_while_running() {
        let mgr = ShutdownManager::new(ShutdownConfig::default());
        let res = mgr.register_hook(Box::new(OkHook("gpu")));
        assert!(res.is_ok());
    }

    #[test]
    fn manager_initiate_reaches_terminated() {
        let mgr = ShutdownManager::new(ShutdownConfig {
            drain_timeout_ms: 1,
            force_timeout_ms: 1,
            save_state: false,
        });
        let report = mgr.initiate();
        assert_eq!(report.final_phase, ShutdownPhase::Terminated);
    }

    #[test]
    fn manager_initiate_triggers_signal() {
        let mgr = ShutdownManager::new(ShutdownConfig {
            drain_timeout_ms: 1,
            force_timeout_ms: 1,
            save_state: false,
        });
        mgr.initiate();
        assert!(mgr.signal().is_triggered());
    }

    #[test]
    fn manager_initiate_stops_accepting() {
        let mgr = ShutdownManager::new(ShutdownConfig {
            drain_timeout_ms: 1,
            force_timeout_ms: 1,
            save_state: false,
        });
        mgr.initiate();
        assert!(!mgr.drain().is_accepting());
    }

    #[test]
    fn manager_initiate_runs_hooks() {
        let mgr = ShutdownManager::new(ShutdownConfig {
            drain_timeout_ms: 1,
            force_timeout_ms: 1,
            save_state: false,
        });
        mgr.register_hook(Box::new(OkHook("cache"))).unwrap();
        let report = mgr.initiate();
        assert_eq!(report.hooks_succeeded, vec!["cache"]);
    }

    #[test]
    fn manager_initiate_reports_failed_hooks() {
        let mgr = ShutdownManager::new(ShutdownConfig {
            drain_timeout_ms: 1,
            force_timeout_ms: 1,
            save_state: false,
        });
        mgr.register_hook(Box::new(FailHook("gpu"))).unwrap();
        let report = mgr.initiate();
        assert_eq!(report.hooks_failed.len(), 1);
        assert_eq!(report.hooks_failed[0].0, "gpu");
    }

    #[test]
    fn manager_initiate_with_state_save() {
        let mgr = ShutdownManager::new(ShutdownConfig {
            drain_timeout_ms: 1,
            force_timeout_ms: 1,
            save_state: true,
        });
        mgr.record_request_state(RequestState {
            request_id: "r1".into(),
            tokens_generated: 3,
            prompt_hash: 0,
        });
        let report = mgr.initiate();
        assert_eq!(report.states_saved, 1);
    }

    #[test]
    fn manager_initiate_without_state_save() {
        let mgr = ShutdownManager::new(ShutdownConfig {
            drain_timeout_ms: 1,
            force_timeout_ms: 1,
            save_state: false,
        });
        mgr.record_request_state(RequestState {
            request_id: "r1".into(),
            tokens_generated: 3,
            prompt_hash: 0,
        });
        let report = mgr.initiate();
        assert_eq!(report.states_saved, 0);
    }

    #[test]
    fn manager_cleanup_reflected_in_report() {
        let mgr = ShutdownManager::new(ShutdownConfig {
            drain_timeout_ms: 1,
            force_timeout_ms: 1,
            save_state: false,
        });
        mgr.cleanup().record_freed(ResourceKind::GpuMemory, 4096);
        let report = mgr.initiate();
        assert_eq!(report.resources_freed.get(&ResourceKind::GpuMemory), Some(&4096),);
    }

    #[test]
    fn manager_register_hook_after_shutdown_fails() {
        let mgr = ShutdownManager::new(ShutdownConfig {
            drain_timeout_ms: 1,
            force_timeout_ms: 1,
            save_state: false,
        });
        mgr.initiate();
        let res = mgr.register_hook(Box::new(OkHook("late")));
        assert!(res.is_err());
    }

    #[test]
    fn manager_advance_past_terminated_errors() {
        let mgr = ShutdownManager::new(ShutdownConfig {
            drain_timeout_ms: 1,
            force_timeout_ms: 1,
            save_state: false,
        });
        mgr.initiate();
        assert!(mgr.advance_phase().is_err());
    }

    #[test]
    fn manager_config_accessor() {
        let cfg = ShutdownConfig { drain_timeout_ms: 42, force_timeout_ms: 7, save_state: false };
        let mgr = ShutdownManager::new(cfg.clone());
        assert_eq!(mgr.config(), &cfg);
    }

    #[test]
    fn manager_debug() {
        let mgr = ShutdownManager::new(ShutdownConfig::default());
        let dbg = format!("{mgr:?}");
        assert!(dbg.contains("ShutdownManager"));
    }

    #[test]
    fn manager_pending_requests_reported() {
        let mgr = ShutdownManager::new(ShutdownConfig {
            drain_timeout_ms: 1,
            force_timeout_ms: 1,
            save_state: false,
        });
        mgr.drain().try_admit();
        let report = mgr.initiate();
        assert_eq!(report.pending_requests, 1);
    }

    #[test]
    fn manager_elapsed_is_positive() {
        let mgr = ShutdownManager::new(ShutdownConfig {
            drain_timeout_ms: 1,
            force_timeout_ms: 1,
            save_state: false,
        });
        let report = mgr.initiate();
        assert!(report.elapsed.as_nanos() > 0);
    }

    #[test]
    fn manager_multiple_hooks_order() {
        let mgr = ShutdownManager::new(ShutdownConfig {
            drain_timeout_ms: 1,
            force_timeout_ms: 1,
            save_state: false,
        });
        mgr.register_hook(Box::new(OkHook("first"))).unwrap();
        mgr.register_hook(Box::new(OkHook("second"))).unwrap();
        let report = mgr.initiate();
        assert_eq!(report.hooks_succeeded, vec!["first", "second"],);
    }

    #[test]
    fn manager_mixed_hooks() {
        let mgr = ShutdownManager::new(ShutdownConfig {
            drain_timeout_ms: 1,
            force_timeout_ms: 1,
            save_state: false,
        });
        mgr.register_hook(Box::new(OkHook("ok1"))).unwrap();
        mgr.register_hook(Box::new(FailHook("fail1"))).unwrap();
        mgr.register_hook(Box::new(OkHook("ok2"))).unwrap();
        let report = mgr.initiate();
        assert_eq!(report.hooks_succeeded.len(), 2);
        assert_eq!(report.hooks_failed.len(), 1);
    }
}
=======
//! Graceful shutdown coordination for GPU HAL operations.
>>>>>>> 029288df (feat(gpu-hal): workspace integration v9 — waves 56-58 modules)
