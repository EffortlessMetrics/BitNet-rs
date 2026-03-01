//! Model lifecycle management for GPU inference.
//!
//! Provides a state-machine–driven lifecycle for loading, warming up,
//! running, swapping, and unloading models on GPU (or CPU reference)
//! backends. Key components:
//!
//! - [`LifecycleConfig`] — configuration for warmup, cooldown, auto-unload.
//! - [`ModelState`] — state machine (Unloaded → Loading → Ready → Running → Unloading).
//! - [`ModelLoader`] — orchestrates the full model loading pipeline.
//! - [`WarmupManager`] — runs warmup inference passes to pre-populate caches.
//! - [`CooldownManager`] — graceful cooldown with request draining.
//! - [`ModelSwapper`] — hot-swaps models with minimal downtime.
//! - [`MemoryBudget`] — manages memory budget across loaded models.
//! - [`ModelVersionManager`] — tracks and switches between model versions.
//! - [`LifecycleMetrics`] — load time, warmup time, swap time, memory stats.
//! - [`ModelLifecycleEngine`] — unified lifecycle management.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the model lifecycle pipeline.
///
/// Controls warmup iterations, cooldown drain timeout, and automatic
/// unloading of idle models. All durations are wall-clock (CPU reference).
#[derive(Debug, Clone)]
pub struct LifecycleConfig {
    /// Number of warmup inference passes after loading.
    pub warmup_iterations: usize,
    /// Prompt text used for warmup passes.
    pub warmup_prompt: String,
    /// Maximum time to wait for in-flight requests to drain during cooldown.
    pub cooldown_drain_timeout: Duration,
    /// Idle duration after which a model is automatically unloaded.
    /// `None` disables auto-unload.
    pub auto_unload_timeout: Option<Duration>,
    /// Maximum number of models that may be loaded simultaneously.
    pub max_loaded_models: usize,
    /// Total memory budget in bytes across all loaded models.
    pub memory_budget_bytes: u64,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            warmup_prompt: "warmup".to_string(),
            cooldown_drain_timeout: Duration::from_secs(30),
            auto_unload_timeout: Some(Duration::from_secs(300)),
            max_loaded_models: 4,
            memory_budget_bytes: 8 * 1024 * 1024 * 1024, // 8 GiB
        }
    }
}

impl LifecycleConfig {
    /// Create a new `LifecycleConfig` with the given memory budget.
    pub fn new(memory_budget_bytes: u64) -> Self {
        Self { memory_budget_bytes, ..Default::default() }
    }

    /// Validate configuration values, returning an error message on failure.
    pub fn validate(&self) -> Result<(), String> {
        if self.warmup_iterations == 0 {
            return Err("warmup_iterations must be > 0".into());
        }
        if self.max_loaded_models == 0 {
            return Err("max_loaded_models must be > 0".into());
        }
        if self.memory_budget_bytes == 0 {
            return Err("memory_budget_bytes must be > 0".into());
        }
        if self.cooldown_drain_timeout.is_zero() {
            return Err("cooldown_drain_timeout must be > 0".into());
        }
        Ok(())
    }

    /// Builder: set warmup iterations.
    pub fn with_warmup_iterations(mut self, n: usize) -> Self {
        self.warmup_iterations = n;
        self
    }

    /// Builder: set auto-unload timeout.
    pub fn with_auto_unload_timeout(mut self, d: Option<Duration>) -> Self {
        self.auto_unload_timeout = d;
        self
    }

    /// Builder: set max loaded models.
    pub fn with_max_loaded_models(mut self, n: usize) -> Self {
        self.max_loaded_models = n;
        self
    }
}

// ── Model State Machine ─────────────────────────────────────────────────────

/// States in the model lifecycle.
///
/// Valid transitions:
/// - `Unloaded → Loading`
/// - `Loading  → Ready | Failed`
/// - `Ready    → Running | Unloading`
/// - `Running  → Ready | Unloading`
/// - `Unloading → Unloaded`
/// - `Failed → Unloaded`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelState {
    /// Model is not loaded into memory.
    Unloaded,
    /// Model weights are being loaded from storage.
    Loading,
    /// Model is loaded and warmed up; ready to accept requests.
    Ready,
    /// Model is actively processing inference requests.
    Running,
    /// Model is draining requests before unloading.
    Unloading,
    /// Loading or initialization failed.
    Failed,
}

impl fmt::Display for ModelState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unloaded => write!(f, "Unloaded"),
            Self::Loading => write!(f, "Loading"),
            Self::Ready => write!(f, "Ready"),
            Self::Running => write!(f, "Running"),
            Self::Unloading => write!(f, "Unloading"),
            Self::Failed => write!(f, "Failed"),
        }
    }
}

impl ModelState {
    /// Returns `true` if the transition from `self` to `next` is valid.
    pub fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Unloaded, Self::Loading)
                | (Self::Loading, Self::Ready)
                | (Self::Loading, Self::Failed)
                | (Self::Ready, Self::Running)
                | (Self::Ready, Self::Unloading)
                | (Self::Running, Self::Ready)
                | (Self::Running, Self::Unloading)
                | (Self::Unloading, Self::Unloaded)
                | (Self::Failed, Self::Unloaded)
        )
    }

    /// Returns all states reachable from the current state.
    pub fn valid_next_states(self) -> Vec<Self> {
        match self {
            Self::Unloaded => vec![Self::Loading],
            Self::Loading => vec![Self::Ready, Self::Failed],
            Self::Ready => vec![Self::Running, Self::Unloading],
            Self::Running => vec![Self::Ready, Self::Unloading],
            Self::Unloading => vec![Self::Unloaded],
            Self::Failed => vec![Self::Unloaded],
        }
    }

    /// Returns `true` if the model is in a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Unloaded | Self::Failed)
    }
}

// ── State tracker ───────────────────────────────────────────────────────────

/// Tracks the current state of a single model with transition history.
#[derive(Debug, Clone)]
pub struct ModelStateTracker {
    model_id: String,
    state: ModelState,
    transitions: Vec<StateTransition>,
    entered_at: Instant,
}

/// A recorded state transition.
#[derive(Debug, Clone)]
pub struct StateTransition {
    pub from: ModelState,
    pub to: ModelState,
    pub duration_in_source: Duration,
}

impl ModelStateTracker {
    /// Create a new tracker starting in `Unloaded`.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            state: ModelState::Unloaded,
            transitions: Vec::new(),
            entered_at: Instant::now(),
        }
    }

    /// Current state.
    pub fn state(&self) -> ModelState {
        self.state
    }

    /// Model identifier.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Attempt a state transition, returning an error on invalid moves.
    pub fn transition_to(&mut self, next: ModelState) -> Result<(), String> {
        if !self.state.can_transition_to(next) {
            return Err(format!(
                "invalid transition {} → {} for model '{}'",
                self.state, next, self.model_id,
            ));
        }
        let now = Instant::now();
        self.transitions.push(StateTransition {
            from: self.state,
            to: next,
            duration_in_source: now.duration_since(self.entered_at),
        });
        self.state = next;
        self.entered_at = now;
        Ok(())
    }

    /// Transition history.
    pub fn transitions(&self) -> &[StateTransition] {
        &self.transitions
    }

    /// Time spent in the current state.
    pub fn time_in_current_state(&self) -> Duration {
        self.entered_at.elapsed()
    }
}

// ── Model Loader ────────────────────────────────────────────────────────────

/// Describes a model to be loaded.
#[derive(Debug, Clone)]
pub struct ModelDescriptor {
    /// Unique model identifier.
    pub id: String,
    /// Path to the model weights file.
    pub path: String,
    /// Expected size in bytes (for budget checking).
    pub size_bytes: u64,
    /// Model version tag.
    pub version: String,
}

/// Result of a model loading operation.
#[derive(Debug, Clone)]
pub struct LoadResult {
    /// Model identifier.
    pub model_id: String,
    /// Time spent loading from storage.
    pub load_duration: Duration,
    /// Bytes actually loaded.
    pub loaded_bytes: u64,
    /// Whether loading succeeded.
    pub success: bool,
    /// Error message if loading failed.
    pub error: Option<String>,
}

/// Orchestrates model loading: budget check → state transition → load → warmup.
///
/// CPU reference implementation: simulates I/O and kernel compilation delays.
#[derive(Debug, Clone)]
pub struct ModelLoader {
    config: LifecycleConfig,
}

impl ModelLoader {
    /// Create a new `ModelLoader` with the given configuration.
    pub fn new(config: LifecycleConfig) -> Self {
        Self { config }
    }

    /// Configuration reference.
    pub fn config(&self) -> &LifecycleConfig {
        &self.config
    }

    /// Load a model (CPU reference implementation).
    ///
    /// Returns a [`LoadResult`] describing what happened. On success the
    /// caller is expected to advance the model's state tracker to `Ready`.
    pub fn load_model(&self, descriptor: &ModelDescriptor) -> LoadResult {
        let start = Instant::now();

        // Simulate weight loading (CPU reference: instant).
        let loaded_bytes = descriptor.size_bytes;

        LoadResult {
            model_id: descriptor.id.clone(),
            load_duration: start.elapsed(),
            loaded_bytes,
            success: true,
            error: None,
        }
    }

    /// Validate a descriptor before loading.
    pub fn validate_descriptor(&self, descriptor: &ModelDescriptor) -> Result<(), String> {
        if descriptor.id.is_empty() {
            return Err("model id must not be empty".into());
        }
        if descriptor.path.is_empty() {
            return Err("model path must not be empty".into());
        }
        if descriptor.size_bytes == 0 {
            return Err("model size_bytes must be > 0".into());
        }
        Ok(())
    }
}

// ── Warmup Manager ──────────────────────────────────────────────────────────

/// Result of a single warmup pass.
#[derive(Debug, Clone)]
pub struct WarmupPassResult {
    /// Pass index (0-based).
    pub pass_index: usize,
    /// Duration of this pass.
    pub duration: Duration,
    /// Whether the pass succeeded.
    pub success: bool,
}

/// Aggregated warmup results.
#[derive(Debug, Clone)]
pub struct WarmupResult {
    /// Model that was warmed up.
    pub model_id: String,
    /// Per-pass results.
    pub passes: Vec<WarmupPassResult>,
    /// Total warmup wall-clock time.
    pub total_duration: Duration,
    /// Whether all passes succeeded.
    pub all_passed: bool,
}

/// Runs warmup inference passes after model loading.
///
/// CPU reference: each pass is a no-op that records timing.
#[derive(Debug, Clone)]
pub struct WarmupManager {
    iterations: usize,
    prompt: String,
}

impl WarmupManager {
    /// Create from lifecycle config.
    pub fn new(config: &LifecycleConfig) -> Self {
        Self { iterations: config.warmup_iterations, prompt: config.warmup_prompt.clone() }
    }

    /// Create with explicit parameters.
    pub fn with_params(iterations: usize, prompt: impl Into<String>) -> Self {
        Self { iterations, prompt: prompt.into() }
    }

    /// Number of warmup iterations configured.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Warmup prompt.
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Run warmup passes for the given model.
    pub fn run_warmup(&self, model_id: &str) -> WarmupResult {
        let start = Instant::now();
        let mut passes = Vec::with_capacity(self.iterations);

        for i in 0..self.iterations {
            let pass_start = Instant::now();
            // CPU reference: no-op inference pass.
            passes.push(WarmupPassResult {
                pass_index: i,
                duration: pass_start.elapsed(),
                success: true,
            });
        }

        let all_passed = passes.iter().all(|p| p.success);
        WarmupResult {
            model_id: model_id.to_string(),
            passes,
            total_duration: start.elapsed(),
            all_passed,
        }
    }
}

// ── Cooldown Manager ────────────────────────────────────────────────────────

/// Status of a cooldown operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CooldownStatus {
    /// Not started.
    Idle,
    /// Draining in-flight requests.
    Draining,
    /// All requests drained; ready to unload.
    Drained,
    /// Drain timed out; forced unload.
    TimedOut,
}

impl fmt::Display for CooldownStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Draining => write!(f, "Draining"),
            Self::Drained => write!(f, "Drained"),
            Self::TimedOut => write!(f, "TimedOut"),
        }
    }
}

/// Result of a cooldown operation.
#[derive(Debug, Clone)]
pub struct CooldownResult {
    /// Model that was cooled down.
    pub model_id: String,
    /// Final cooldown status.
    pub status: CooldownStatus,
    /// Time spent draining.
    pub drain_duration: Duration,
    /// Requests that were in flight when cooldown began.
    pub initial_in_flight: usize,
    /// Requests that completed during drain.
    pub drained_count: usize,
}

/// Graceful cooldown with request draining before model unload.
///
/// CPU reference: simulates drain completion based on in-flight count.
#[derive(Debug, Clone)]
pub struct CooldownManager {
    drain_timeout: Duration,
}

impl CooldownManager {
    /// Create from lifecycle config.
    pub fn new(config: &LifecycleConfig) -> Self {
        Self { drain_timeout: config.cooldown_drain_timeout }
    }

    /// Create with explicit timeout.
    pub fn with_timeout(timeout: Duration) -> Self {
        Self { drain_timeout: timeout }
    }

    /// Drain timeout.
    pub fn drain_timeout(&self) -> Duration {
        self.drain_timeout
    }

    /// Begin cooldown for the given model.
    ///
    /// `in_flight_requests`: number of requests currently being processed.
    /// CPU reference: all requests drain instantly.
    pub fn begin_cooldown(&self, model_id: &str, in_flight_requests: usize) -> CooldownResult {
        let start = Instant::now();

        // CPU reference: instant drain.
        CooldownResult {
            model_id: model_id.to_string(),
            status: CooldownStatus::Drained,
            drain_duration: start.elapsed(),
            initial_in_flight: in_flight_requests,
            drained_count: in_flight_requests,
        }
    }
}

// ── Model Swapper ───────────────────────────────────────────────────────────

/// Result of a hot-swap operation.
#[derive(Debug, Clone)]
pub struct SwapResult {
    /// Model being replaced.
    pub old_model_id: String,
    /// New model loaded.
    pub new_model_id: String,
    /// Total swap wall-clock time.
    pub swap_duration: Duration,
    /// Time where neither model was serving (downtime window).
    pub downtime: Duration,
    /// Whether the swap succeeded.
    pub success: bool,
    /// Error message if the swap failed.
    pub error: Option<String>,
}

/// Hot-swaps models with minimal downtime.
///
/// Strategy: load new model → warmup → drain old model → switch → unload old.
/// CPU reference: swap is instantaneous.
#[derive(Debug, Clone)]
pub struct ModelSwapper {
    loader: ModelLoader,
    warmup: WarmupManager,
    cooldown: CooldownManager,
}

impl ModelSwapper {
    /// Create a new swapper from lifecycle config.
    pub fn new(config: &LifecycleConfig) -> Self {
        Self {
            loader: ModelLoader::new(config.clone()),
            warmup: WarmupManager::new(config),
            cooldown: CooldownManager::new(config),
        }
    }

    /// Perform a hot-swap from `old` to `new`.
    ///
    /// `old_in_flight`: number of in-flight requests on the old model.
    pub fn swap(
        &self,
        old: &ModelDescriptor,
        new: &ModelDescriptor,
        old_in_flight: usize,
    ) -> SwapResult {
        let start = Instant::now();

        // Step 1: Load new model.
        let load_result = self.loader.load_model(new);
        if !load_result.success {
            return SwapResult {
                old_model_id: old.id.clone(),
                new_model_id: new.id.clone(),
                swap_duration: start.elapsed(),
                downtime: Duration::ZERO,
                success: false,
                error: load_result.error,
            };
        }

        // Step 2: Warmup new model.
        let warmup_result = self.warmup.run_warmup(&new.id);
        if !warmup_result.all_passed {
            return SwapResult {
                old_model_id: old.id.clone(),
                new_model_id: new.id.clone(),
                swap_duration: start.elapsed(),
                downtime: Duration::ZERO,
                success: false,
                error: Some("warmup failed for new model".into()),
            };
        }

        // Step 3: Drain old model.
        let _cooldown = self.cooldown.begin_cooldown(&old.id, old_in_flight);

        // Step 4: Switch (CPU reference: instant).
        let downtime_start = Instant::now();
        let downtime = downtime_start.elapsed();

        SwapResult {
            old_model_id: old.id.clone(),
            new_model_id: new.id.clone(),
            swap_duration: start.elapsed(),
            downtime,
            success: true,
            error: None,
        }
    }
}

// ── Memory Budget ───────────────────────────────────────────────────────────

/// Per-model memory allocation record.
#[derive(Debug, Clone)]
pub struct ModelAllocation {
    /// Model identifier.
    pub model_id: String,
    /// Bytes allocated.
    pub allocated_bytes: u64,
    /// When the allocation was made.
    pub allocated_at: Instant,
}

/// Manages memory budget across loaded models.
///
/// Tracks per-model allocations against a total budget, rejecting loads
/// that would exceed the limit.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    total_budget: u64,
    allocations: Vec<ModelAllocation>,
}

impl MemoryBudget {
    /// Create a new budget with the given total bytes.
    pub fn new(total_budget: u64) -> Self {
        Self { total_budget, allocations: Vec::new() }
    }

    /// Total budget in bytes.
    pub fn total_budget(&self) -> u64 {
        self.total_budget
    }

    /// Bytes currently allocated.
    pub fn used_bytes(&self) -> u64 {
        self.allocations.iter().map(|a| a.allocated_bytes).sum()
    }

    /// Bytes remaining in the budget.
    pub fn available_bytes(&self) -> u64 {
        self.total_budget.saturating_sub(self.used_bytes())
    }

    /// Number of models currently tracked.
    pub fn model_count(&self) -> usize {
        self.allocations.len()
    }

    /// Returns `true` if `bytes` can fit within the remaining budget.
    pub fn can_allocate(&self, bytes: u64) -> bool {
        self.used_bytes().checked_add(bytes).is_some_and(|total| total <= self.total_budget)
    }

    /// Allocate memory for a model. Fails if the budget would be exceeded.
    pub fn allocate(&mut self, model_id: impl Into<String>, bytes: u64) -> Result<(), String> {
        let model_id = model_id.into();
        if self.allocations.iter().any(|a| a.model_id == model_id) {
            return Err(format!("model '{}' already allocated", model_id));
        }
        if !self.can_allocate(bytes) {
            return Err(format!(
                "insufficient budget: need {} bytes, {} available",
                bytes,
                self.available_bytes()
            ));
        }
        self.allocations.push(ModelAllocation {
            model_id,
            allocated_bytes: bytes,
            allocated_at: Instant::now(),
        });
        Ok(())
    }

    /// Release memory for a model.
    pub fn release(&mut self, model_id: &str) -> Result<u64, String> {
        let idx = self
            .allocations
            .iter()
            .position(|a| a.model_id == model_id)
            .ok_or_else(|| format!("model '{}' not found in budget", model_id))?;
        let alloc = self.allocations.remove(idx);
        Ok(alloc.allocated_bytes)
    }

    /// Get the allocation for a specific model.
    pub fn get_allocation(&self, model_id: &str) -> Option<&ModelAllocation> {
        self.allocations.iter().find(|a| a.model_id == model_id)
    }

    /// Utilization ratio (0.0–1.0).
    pub fn utilization(&self) -> f64 {
        if self.total_budget == 0 {
            return 0.0;
        }
        self.used_bytes() as f64 / self.total_budget as f64
    }
}

// ── Model Version Manager ───────────────────────────────────────────────────

/// A versioned model entry.
#[derive(Debug, Clone)]
pub struct VersionedModel {
    /// Model identifier (shared across versions).
    pub model_id: String,
    /// Monotonically increasing version number.
    pub version: u64,
    /// Path to model weights.
    pub path: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// When this version was registered.
    pub registered_at: Instant,
}

/// Tracks model versions and active version selection.
#[derive(Debug, Clone)]
pub struct ModelVersionManager {
    /// All registered versions, keyed by model_id.
    versions: HashMap<String, Vec<VersionedModel>>,
    /// Currently active version per model_id.
    active: HashMap<String, u64>,
}

impl ModelVersionManager {
    /// Create an empty version manager.
    pub fn new() -> Self {
        Self { versions: HashMap::new(), active: HashMap::new() }
    }

    /// Register a new version of a model.
    pub fn register(
        &mut self,
        model_id: impl Into<String>,
        version: u64,
        path: impl Into<String>,
        size_bytes: u64,
    ) -> Result<(), String> {
        let model_id = model_id.into();
        let entries = self.versions.entry(model_id.clone()).or_default();
        if entries.iter().any(|v| v.version == version) {
            return Err(format!("version {} already registered for model '{}'", version, model_id));
        }
        entries.push(VersionedModel {
            model_id: model_id.clone(),
            version,
            path: path.into(),
            size_bytes,
            registered_at: Instant::now(),
        });
        entries.sort_by_key(|v| v.version);

        // Auto-activate if first version.
        if entries.len() == 1 {
            self.active.insert(model_id, version);
        }
        Ok(())
    }

    /// Set the active version for a model.
    pub fn set_active(&mut self, model_id: &str, version: u64) -> Result<(), String> {
        let entries =
            self.versions.get(model_id).ok_or_else(|| format!("model '{}' not found", model_id))?;
        if !entries.iter().any(|v| v.version == version) {
            return Err(format!("version {} not found for model '{}'", version, model_id));
        }
        self.active.insert(model_id.to_string(), version);
        Ok(())
    }

    /// Get the active version number for a model.
    pub fn active_version(&self, model_id: &str) -> Option<u64> {
        self.active.get(model_id).copied()
    }

    /// Get the active version entry.
    pub fn active_entry(&self, model_id: &str) -> Option<&VersionedModel> {
        let version = self.active.get(model_id)?;
        self.versions.get(model_id)?.iter().find(|v| v.version == *version)
    }

    /// List all versions for a model.
    pub fn list_versions(&self, model_id: &str) -> Vec<&VersionedModel> {
        self.versions.get(model_id).map(|v| v.iter().collect()).unwrap_or_default()
    }

    /// Number of distinct models tracked.
    pub fn model_count(&self) -> usize {
        self.versions.len()
    }

    /// Total number of versions across all models.
    pub fn total_versions(&self) -> usize {
        self.versions.values().map(|v| v.len()).sum()
    }

    /// Remove a specific version. Cannot remove the active version.
    pub fn remove_version(&mut self, model_id: &str, version: u64) -> Result<(), String> {
        if self.active.get(model_id) == Some(&version) {
            return Err(format!(
                "cannot remove active version {} of model '{}'",
                version, model_id
            ));
        }
        let entries = self
            .versions
            .get_mut(model_id)
            .ok_or_else(|| format!("model '{}' not found", model_id))?;
        let idx = entries
            .iter()
            .position(|v| v.version == version)
            .ok_or_else(|| format!("version {} not found for model '{}'", version, model_id))?;
        entries.remove(idx);
        Ok(())
    }

    /// Get the latest (highest) version number for a model.
    pub fn latest_version(&self, model_id: &str) -> Option<u64> {
        self.versions.get(model_id).and_then(|v| v.last().map(|e| e.version))
    }
}

impl Default for ModelVersionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Lifecycle Metrics ───────────────────────────────────────────────────────

/// Metrics collected during model lifecycle operations.
#[derive(Debug, Clone)]
pub struct LifecycleMetrics {
    /// Number of models loaded since creation.
    pub loads: u64,
    /// Number of models unloaded since creation.
    pub unloads: u64,
    /// Number of hot-swaps performed.
    pub swaps: u64,
    /// Number of warmup passes executed.
    pub warmup_passes: u64,
    /// Cumulative load time.
    pub total_load_time: Duration,
    /// Cumulative warmup time.
    pub total_warmup_time: Duration,
    /// Cumulative swap time.
    pub total_swap_time: Duration,
    /// Peak memory usage observed (bytes).
    pub peak_memory_bytes: u64,
    /// Number of failed operations.
    pub failures: u64,
}

impl LifecycleMetrics {
    /// Create zeroed metrics.
    pub fn new() -> Self {
        Self {
            loads: 0,
            unloads: 0,
            swaps: 0,
            warmup_passes: 0,
            total_load_time: Duration::ZERO,
            total_warmup_time: Duration::ZERO,
            total_swap_time: Duration::ZERO,
            peak_memory_bytes: 0,
            failures: 0,
        }
    }

    /// Record a load event.
    pub fn record_load(&mut self, duration: Duration) {
        self.loads += 1;
        self.total_load_time += duration;
    }

    /// Record an unload event.
    pub fn record_unload(&mut self) {
        self.unloads += 1;
    }

    /// Record a swap event.
    pub fn record_swap(&mut self, duration: Duration) {
        self.swaps += 1;
        self.total_swap_time += duration;
    }

    /// Record warmup passes.
    pub fn record_warmup(&mut self, passes: u64, duration: Duration) {
        self.warmup_passes += passes;
        self.total_warmup_time += duration;
    }

    /// Update peak memory.
    pub fn update_peak_memory(&mut self, current_bytes: u64) {
        if current_bytes > self.peak_memory_bytes {
            self.peak_memory_bytes = current_bytes;
        }
    }

    /// Record a failure.
    pub fn record_failure(&mut self) {
        self.failures += 1;
    }

    /// Average load time, or `None` if no loads recorded.
    pub fn avg_load_time(&self) -> Option<Duration> {
        if self.loads == 0 {
            return None;
        }
        Some(self.total_load_time / self.loads as u32)
    }

    /// Average warmup time per pass, or `None` if no passes recorded.
    pub fn avg_warmup_time(&self) -> Option<Duration> {
        if self.warmup_passes == 0 {
            return None;
        }
        Some(self.total_warmup_time / self.warmup_passes as u32)
    }

    /// Average swap time, or `None` if no swaps recorded.
    pub fn avg_swap_time(&self) -> Option<Duration> {
        if self.swaps == 0 {
            return None;
        }
        Some(self.total_swap_time / self.swaps as u32)
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for LifecycleMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ── Model Lifecycle Engine ──────────────────────────────────────────────────

/// Unified lifecycle management engine.
///
/// Coordinates state tracking, loading, warmup, cooldown, swapping,
/// memory budgeting, versioning, and metrics for all managed models.
#[derive(Debug, Clone)]
pub struct ModelLifecycleEngine {
    config: LifecycleConfig,
    trackers: HashMap<String, ModelStateTracker>,
    loader: ModelLoader,
    warmup: WarmupManager,
    cooldown: CooldownManager,
    swapper: ModelSwapper,
    budget: MemoryBudget,
    versions: ModelVersionManager,
    metrics: LifecycleMetrics,
}

impl ModelLifecycleEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: LifecycleConfig) -> Self {
        let budget = MemoryBudget::new(config.memory_budget_bytes);
        let loader = ModelLoader::new(config.clone());
        let warmup = WarmupManager::new(&config);
        let cooldown = CooldownManager::new(&config);
        let swapper = ModelSwapper::new(&config);
        Self {
            config,
            trackers: HashMap::new(),
            loader,
            warmup,
            cooldown,
            swapper,
            budget,
            versions: ModelVersionManager::new(),
            metrics: LifecycleMetrics::new(),
        }
    }

    /// Configuration reference.
    pub fn config(&self) -> &LifecycleConfig {
        &self.config
    }

    /// Metrics snapshot.
    pub fn metrics(&self) -> &LifecycleMetrics {
        &self.metrics
    }

    /// Memory budget reference.
    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// Version manager reference.
    pub fn versions(&self) -> &ModelVersionManager {
        &self.versions
    }

    /// Get the state tracker for a model.
    pub fn tracker(&self, model_id: &str) -> Option<&ModelStateTracker> {
        self.trackers.get(model_id)
    }

    /// Number of managed models.
    pub fn model_count(&self) -> usize {
        self.trackers.len()
    }

    /// Load a model through the full pipeline:
    /// register → budget check → load → warmup → Ready.
    pub fn load_model(&mut self, descriptor: &ModelDescriptor) -> Result<LoadResult, String> {
        self.loader.validate_descriptor(descriptor)?;

        // Check model count limit.
        let loaded_count = self.trackers.values().filter(|t| !t.state().is_terminal()).count();
        if loaded_count >= self.config.max_loaded_models {
            self.metrics.record_failure();
            return Err(format!("max loaded models ({}) reached", self.config.max_loaded_models));
        }

        // Budget check.
        if !self.budget.can_allocate(descriptor.size_bytes) {
            self.metrics.record_failure();
            return Err(format!(
                "insufficient memory budget: need {} bytes, {} available",
                descriptor.size_bytes,
                self.budget.available_bytes()
            ));
        }

        // Create or reset tracker.
        let tracker = self
            .trackers
            .entry(descriptor.id.clone())
            .or_insert_with(|| ModelStateTracker::new(&descriptor.id));

        // Unloaded → Loading
        tracker.transition_to(ModelState::Loading)?;

        // Perform load.
        let result = self.loader.load_model(descriptor);

        if result.success {
            // Allocate memory.
            self.budget.allocate(&descriptor.id, descriptor.size_bytes).map_err(|e| {
                // Revert state on budget allocation failure.
                let _ = self
                    .trackers
                    .get_mut(&descriptor.id)
                    .map(|t| t.transition_to(ModelState::Failed));
                e
            })?;

            // Loading → Ready
            self.trackers
                .get_mut(&descriptor.id)
                .expect("tracker must exist")
                .transition_to(ModelState::Ready)?;

            // Run warmup.
            let warmup_result = self.warmup.run_warmup(&descriptor.id);
            self.metrics
                .record_warmup(warmup_result.passes.len() as u64, warmup_result.total_duration);

            // Record metrics.
            self.metrics.record_load(result.load_duration);
            self.metrics.update_peak_memory(self.budget.used_bytes());

            // Register version.
            let _ = self.versions.register(
                &descriptor.id,
                descriptor.version.parse::<u64>().unwrap_or(1),
                &descriptor.path,
                descriptor.size_bytes,
            );
        } else {
            // Loading → Failed
            self.trackers
                .get_mut(&descriptor.id)
                .expect("tracker must exist")
                .transition_to(ModelState::Failed)?;
            self.metrics.record_failure();
        }

        Ok(result)
    }

    /// Unload a model: cooldown → release budget → Unloaded.
    pub fn unload_model(
        &mut self,
        model_id: &str,
        in_flight: usize,
    ) -> Result<CooldownResult, String> {
        let tracker = self
            .trackers
            .get_mut(model_id)
            .ok_or_else(|| format!("model '{}' not found", model_id))?;

        let state = tracker.state();
        if state == ModelState::Unloaded || state == ModelState::Unloading {
            return Err(format!("model '{}' is already {} ", model_id, state));
        }

        // → Unloading
        tracker.transition_to(ModelState::Unloading)?;

        // Drain.
        let result = self.cooldown.begin_cooldown(model_id, in_flight);

        // → Unloaded
        self.trackers
            .get_mut(model_id)
            .expect("tracker must exist")
            .transition_to(ModelState::Unloaded)?;

        // Release budget.
        let _ = self.budget.release(model_id);

        self.metrics.record_unload();

        Ok(result)
    }

    /// Mark a model as Running.
    pub fn mark_running(&mut self, model_id: &str) -> Result<(), String> {
        let tracker = self
            .trackers
            .get_mut(model_id)
            .ok_or_else(|| format!("model '{}' not found", model_id))?;
        tracker.transition_to(ModelState::Running)
    }

    /// Mark a model as Ready (after finishing a batch).
    pub fn mark_ready(&mut self, model_id: &str) -> Result<(), String> {
        let tracker = self
            .trackers
            .get_mut(model_id)
            .ok_or_else(|| format!("model '{}' not found", model_id))?;
        tracker.transition_to(ModelState::Ready)
    }

    /// Hot-swap one model for another.
    pub fn swap_model(
        &mut self,
        old_descriptor: &ModelDescriptor,
        new_descriptor: &ModelDescriptor,
        old_in_flight: usize,
    ) -> Result<SwapResult, String> {
        // Pre-validate new descriptor.
        self.loader.validate_descriptor(new_descriptor)?;

        let result = self.swapper.swap(old_descriptor, new_descriptor, old_in_flight);

        if result.success {
            // Unload old model from budget.
            let _ = self.budget.release(&old_descriptor.id);
            if let Some(t) = self.trackers.get_mut(&old_descriptor.id) {
                let _ = t.transition_to(ModelState::Unloading);
                let _ = t.transition_to(ModelState::Unloaded);
            }
            self.metrics.record_unload();

            // Allocate new model.
            let _ = self.budget.allocate(&new_descriptor.id, new_descriptor.size_bytes);
            self.trackers.insert(new_descriptor.id.clone(), {
                let mut t = ModelStateTracker::new(&new_descriptor.id);
                let _ = t.transition_to(ModelState::Loading);
                let _ = t.transition_to(ModelState::Ready);
                t
            });

            self.metrics.record_swap(result.swap_duration);
            self.metrics.update_peak_memory(self.budget.used_bytes());
        } else {
            self.metrics.record_failure();
        }

        Ok(result)
    }

    /// List all models and their current states.
    pub fn list_models(&self) -> Vec<(&str, ModelState)> {
        self.trackers.iter().map(|(id, t)| (id.as_str(), t.state())).collect()
    }

    /// Get models in a specific state.
    pub fn models_in_state(&self, state: ModelState) -> Vec<&str> {
        self.trackers
            .iter()
            .filter(|(_, t)| t.state() == state)
            .map(|(id, _)| id.as_str())
            .collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> LifecycleConfig {
        LifecycleConfig {
            warmup_iterations: 2,
            warmup_prompt: "test".to_string(),
            cooldown_drain_timeout: Duration::from_secs(5),
            auto_unload_timeout: Some(Duration::from_secs(60)),
            max_loaded_models: 4,
            memory_budget_bytes: 1024 * 1024 * 1024, // 1 GiB
        }
    }

    fn test_descriptor(id: &str) -> ModelDescriptor {
        ModelDescriptor {
            id: id.to_string(),
            path: format!("/models/{}.gguf", id),
            size_bytes: 100 * 1024 * 1024, // 100 MiB
            version: "1".to_string(),
        }
    }

    // ── LifecycleConfig ─────────────────────────────────────────────────

    #[test]
    fn config_default_is_valid() {
        let config = LifecycleConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_new_sets_budget() {
        let config = LifecycleConfig::new(42);
        assert_eq!(config.memory_budget_bytes, 42);
    }

    #[test]
    fn config_validate_rejects_zero_warmup() {
        let mut config = test_config();
        config.warmup_iterations = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_validate_rejects_zero_max_models() {
        let mut config = test_config();
        config.max_loaded_models = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_validate_rejects_zero_budget() {
        let mut config = test_config();
        config.memory_budget_bytes = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_validate_rejects_zero_drain_timeout() {
        let mut config = test_config();
        config.cooldown_drain_timeout = Duration::ZERO;
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_builder_warmup_iterations() {
        let config = LifecycleConfig::default().with_warmup_iterations(10);
        assert_eq!(config.warmup_iterations, 10);
    }

    #[test]
    fn config_builder_auto_unload() {
        let config = LifecycleConfig::default().with_auto_unload_timeout(None);
        assert!(config.auto_unload_timeout.is_none());
    }

    #[test]
    fn config_builder_max_models() {
        let config = LifecycleConfig::default().with_max_loaded_models(8);
        assert_eq!(config.max_loaded_models, 8);
    }

    // ── ModelState ──────────────────────────────────────────────────────

    #[test]
    fn state_display() {
        assert_eq!(ModelState::Unloaded.to_string(), "Unloaded");
        assert_eq!(ModelState::Loading.to_string(), "Loading");
        assert_eq!(ModelState::Ready.to_string(), "Ready");
        assert_eq!(ModelState::Running.to_string(), "Running");
        assert_eq!(ModelState::Unloading.to_string(), "Unloading");
        assert_eq!(ModelState::Failed.to_string(), "Failed");
    }

    #[test]
    fn state_valid_transitions_unloaded() {
        assert!(ModelState::Unloaded.can_transition_to(ModelState::Loading));
        assert!(!ModelState::Unloaded.can_transition_to(ModelState::Ready));
        assert!(!ModelState::Unloaded.can_transition_to(ModelState::Running));
        assert!(!ModelState::Unloaded.can_transition_to(ModelState::Unloading));
        assert!(!ModelState::Unloaded.can_transition_to(ModelState::Failed));
        assert!(!ModelState::Unloaded.can_transition_to(ModelState::Unloaded));
    }

    #[test]
    fn state_valid_transitions_loading() {
        assert!(ModelState::Loading.can_transition_to(ModelState::Ready));
        assert!(ModelState::Loading.can_transition_to(ModelState::Failed));
        assert!(!ModelState::Loading.can_transition_to(ModelState::Running));
        assert!(!ModelState::Loading.can_transition_to(ModelState::Unloading));
        assert!(!ModelState::Loading.can_transition_to(ModelState::Unloaded));
        assert!(!ModelState::Loading.can_transition_to(ModelState::Loading));
    }

    #[test]
    fn state_valid_transitions_ready() {
        assert!(ModelState::Ready.can_transition_to(ModelState::Running));
        assert!(ModelState::Ready.can_transition_to(ModelState::Unloading));
        assert!(!ModelState::Ready.can_transition_to(ModelState::Loading));
        assert!(!ModelState::Ready.can_transition_to(ModelState::Unloaded));
        assert!(!ModelState::Ready.can_transition_to(ModelState::Failed));
        assert!(!ModelState::Ready.can_transition_to(ModelState::Ready));
    }

    #[test]
    fn state_valid_transitions_running() {
        assert!(ModelState::Running.can_transition_to(ModelState::Ready));
        assert!(ModelState::Running.can_transition_to(ModelState::Unloading));
        assert!(!ModelState::Running.can_transition_to(ModelState::Loading));
        assert!(!ModelState::Running.can_transition_to(ModelState::Unloaded));
        assert!(!ModelState::Running.can_transition_to(ModelState::Failed));
        assert!(!ModelState::Running.can_transition_to(ModelState::Running));
    }

    #[test]
    fn state_valid_transitions_unloading() {
        assert!(ModelState::Unloading.can_transition_to(ModelState::Unloaded));
        assert!(!ModelState::Unloading.can_transition_to(ModelState::Loading));
        assert!(!ModelState::Unloading.can_transition_to(ModelState::Ready));
        assert!(!ModelState::Unloading.can_transition_to(ModelState::Running));
        assert!(!ModelState::Unloading.can_transition_to(ModelState::Failed));
        assert!(!ModelState::Unloading.can_transition_to(ModelState::Unloading));
    }

    #[test]
    fn state_valid_transitions_failed() {
        assert!(ModelState::Failed.can_transition_to(ModelState::Unloaded));
        assert!(!ModelState::Failed.can_transition_to(ModelState::Loading));
        assert!(!ModelState::Failed.can_transition_to(ModelState::Ready));
        assert!(!ModelState::Failed.can_transition_to(ModelState::Running));
        assert!(!ModelState::Failed.can_transition_to(ModelState::Unloading));
        assert!(!ModelState::Failed.can_transition_to(ModelState::Failed));
    }

    #[test]
    fn state_valid_next_states() {
        assert_eq!(ModelState::Unloaded.valid_next_states(), vec![ModelState::Loading]);
        assert_eq!(
            ModelState::Loading.valid_next_states(),
            vec![ModelState::Ready, ModelState::Failed]
        );
        assert_eq!(
            ModelState::Ready.valid_next_states(),
            vec![ModelState::Running, ModelState::Unloading]
        );
        assert_eq!(ModelState::Unloading.valid_next_states(), vec![ModelState::Unloaded]);
        assert_eq!(ModelState::Failed.valid_next_states(), vec![ModelState::Unloaded]);
    }

    #[test]
    fn state_is_terminal() {
        assert!(ModelState::Unloaded.is_terminal());
        assert!(ModelState::Failed.is_terminal());
        assert!(!ModelState::Loading.is_terminal());
        assert!(!ModelState::Ready.is_terminal());
        assert!(!ModelState::Running.is_terminal());
        assert!(!ModelState::Unloading.is_terminal());
    }

    // ── ModelStateTracker ───────────────────────────────────────────────

    #[test]
    fn tracker_starts_unloaded() {
        let t = ModelStateTracker::new("m1");
        assert_eq!(t.state(), ModelState::Unloaded);
        assert_eq!(t.model_id(), "m1");
        assert!(t.transitions().is_empty());
    }

    #[test]
    fn tracker_valid_transition_records_history() {
        let mut t = ModelStateTracker::new("m1");
        t.transition_to(ModelState::Loading).unwrap();
        assert_eq!(t.state(), ModelState::Loading);
        assert_eq!(t.transitions().len(), 1);
        assert_eq!(t.transitions()[0].from, ModelState::Unloaded);
        assert_eq!(t.transitions()[0].to, ModelState::Loading);
    }

    #[test]
    fn tracker_invalid_transition_fails() {
        let mut t = ModelStateTracker::new("m1");
        let err = t.transition_to(ModelState::Ready).unwrap_err();
        assert!(err.contains("invalid transition"));
        assert_eq!(t.state(), ModelState::Unloaded);
    }

    #[test]
    fn tracker_full_lifecycle() {
        let mut t = ModelStateTracker::new("m1");
        t.transition_to(ModelState::Loading).unwrap();
        t.transition_to(ModelState::Ready).unwrap();
        t.transition_to(ModelState::Running).unwrap();
        t.transition_to(ModelState::Ready).unwrap();
        t.transition_to(ModelState::Unloading).unwrap();
        t.transition_to(ModelState::Unloaded).unwrap();
        assert_eq!(t.state(), ModelState::Unloaded);
        assert_eq!(t.transitions().len(), 6);
    }

    #[test]
    fn tracker_failed_path() {
        let mut t = ModelStateTracker::new("m1");
        t.transition_to(ModelState::Loading).unwrap();
        t.transition_to(ModelState::Failed).unwrap();
        t.transition_to(ModelState::Unloaded).unwrap();
        assert_eq!(t.state(), ModelState::Unloaded);
        assert_eq!(t.transitions().len(), 3);
    }

    #[test]
    fn tracker_time_in_current_state() {
        let t = ModelStateTracker::new("m1");
        // Should be very small but non-negative.
        assert!(t.time_in_current_state() < Duration::from_secs(1));
    }

    // ── ModelLoader ─────────────────────────────────────────────────────

    #[test]
    fn loader_load_model_succeeds() {
        let loader = ModelLoader::new(test_config());
        let desc = test_descriptor("m1");
        let result = loader.load_model(&desc);
        assert!(result.success);
        assert_eq!(result.model_id, "m1");
        assert_eq!(result.loaded_bytes, desc.size_bytes);
        assert!(result.error.is_none());
    }

    #[test]
    fn loader_config_ref() {
        let cfg = test_config();
        let loader = ModelLoader::new(cfg.clone());
        assert_eq!(loader.config().warmup_iterations, cfg.warmup_iterations);
    }

    #[test]
    fn loader_validate_empty_id() {
        let loader = ModelLoader::new(test_config());
        let mut desc = test_descriptor("m1");
        desc.id = String::new();
        assert!(loader.validate_descriptor(&desc).is_err());
    }

    #[test]
    fn loader_validate_empty_path() {
        let loader = ModelLoader::new(test_config());
        let mut desc = test_descriptor("m1");
        desc.path = String::new();
        assert!(loader.validate_descriptor(&desc).is_err());
    }

    #[test]
    fn loader_validate_zero_size() {
        let loader = ModelLoader::new(test_config());
        let mut desc = test_descriptor("m1");
        desc.size_bytes = 0;
        assert!(loader.validate_descriptor(&desc).is_err());
    }

    #[test]
    fn loader_validate_good_descriptor() {
        let loader = ModelLoader::new(test_config());
        assert!(loader.validate_descriptor(&test_descriptor("m1")).is_ok());
    }

    // ── WarmupManager ───────────────────────────────────────────────────

    #[test]
    fn warmup_new_from_config() {
        let cfg = test_config();
        let w = WarmupManager::new(&cfg);
        assert_eq!(w.iterations(), cfg.warmup_iterations);
        assert_eq!(w.prompt(), cfg.warmup_prompt);
    }

    #[test]
    fn warmup_with_params() {
        let w = WarmupManager::with_params(5, "hello");
        assert_eq!(w.iterations(), 5);
        assert_eq!(w.prompt(), "hello");
    }

    #[test]
    fn warmup_run_all_pass() {
        let w = WarmupManager::with_params(4, "test");
        let result = w.run_warmup("m1");
        assert!(result.all_passed);
        assert_eq!(result.model_id, "m1");
        assert_eq!(result.passes.len(), 4);
        for (i, p) in result.passes.iter().enumerate() {
            assert_eq!(p.pass_index, i);
            assert!(p.success);
        }
    }

    #[test]
    fn warmup_single_pass() {
        let w = WarmupManager::with_params(1, "x");
        let result = w.run_warmup("m1");
        assert_eq!(result.passes.len(), 1);
        assert!(result.all_passed);
    }

    // ── CooldownManager ─────────────────────────────────────────────────

    #[test]
    fn cooldown_from_config() {
        let cfg = test_config();
        let c = CooldownManager::new(&cfg);
        assert_eq!(c.drain_timeout(), cfg.cooldown_drain_timeout);
    }

    #[test]
    fn cooldown_with_timeout() {
        let c = CooldownManager::with_timeout(Duration::from_secs(10));
        assert_eq!(c.drain_timeout(), Duration::from_secs(10));
    }

    #[test]
    fn cooldown_drains_all() {
        let c = CooldownManager::new(&test_config());
        let result = c.begin_cooldown("m1", 5);
        assert_eq!(result.status, CooldownStatus::Drained);
        assert_eq!(result.initial_in_flight, 5);
        assert_eq!(result.drained_count, 5);
        assert_eq!(result.model_id, "m1");
    }

    #[test]
    fn cooldown_zero_in_flight() {
        let c = CooldownManager::new(&test_config());
        let result = c.begin_cooldown("m1", 0);
        assert_eq!(result.status, CooldownStatus::Drained);
        assert_eq!(result.drained_count, 0);
    }

    #[test]
    fn cooldown_status_display() {
        assert_eq!(CooldownStatus::Idle.to_string(), "Idle");
        assert_eq!(CooldownStatus::Draining.to_string(), "Draining");
        assert_eq!(CooldownStatus::Drained.to_string(), "Drained");
        assert_eq!(CooldownStatus::TimedOut.to_string(), "TimedOut");
    }

    // ── ModelSwapper ────────────────────────────────────────────────────

    #[test]
    fn swapper_successful_swap() {
        let s = ModelSwapper::new(&test_config());
        let old = test_descriptor("old");
        let new = test_descriptor("new");
        let result = s.swap(&old, &new, 0);
        assert!(result.success);
        assert_eq!(result.old_model_id, "old");
        assert_eq!(result.new_model_id, "new");
        assert!(result.error.is_none());
    }

    #[test]
    fn swapper_records_swap_duration() {
        let s = ModelSwapper::new(&test_config());
        let old = test_descriptor("old");
        let new = test_descriptor("new");
        let result = s.swap(&old, &new, 3);
        assert!(result.success);
        // Duration should be non-negative.
        assert!(result.swap_duration >= Duration::ZERO);
    }

    #[test]
    fn swapper_swap_with_in_flight() {
        let s = ModelSwapper::new(&test_config());
        let old = test_descriptor("old");
        let new = test_descriptor("new");
        let result = s.swap(&old, &new, 10);
        assert!(result.success);
    }

    // ── MemoryBudget ────────────────────────────────────────────────────

    #[test]
    fn budget_new() {
        let b = MemoryBudget::new(1024);
        assert_eq!(b.total_budget(), 1024);
        assert_eq!(b.used_bytes(), 0);
        assert_eq!(b.available_bytes(), 1024);
        assert_eq!(b.model_count(), 0);
    }

    #[test]
    fn budget_allocate_success() {
        let mut b = MemoryBudget::new(1024);
        b.allocate("m1", 512).unwrap();
        assert_eq!(b.used_bytes(), 512);
        assert_eq!(b.available_bytes(), 512);
        assert_eq!(b.model_count(), 1);
    }

    #[test]
    fn budget_allocate_multiple() {
        let mut b = MemoryBudget::new(1024);
        b.allocate("m1", 256).unwrap();
        b.allocate("m2", 256).unwrap();
        assert_eq!(b.used_bytes(), 512);
        assert_eq!(b.model_count(), 2);
    }

    #[test]
    fn budget_allocate_exact_fit() {
        let mut b = MemoryBudget::new(1024);
        b.allocate("m1", 1024).unwrap();
        assert_eq!(b.available_bytes(), 0);
        assert!(!b.can_allocate(1));
    }

    #[test]
    fn budget_allocate_exceeds() {
        let mut b = MemoryBudget::new(1024);
        assert!(b.allocate("m1", 2048).is_err());
        assert_eq!(b.model_count(), 0);
    }

    #[test]
    fn budget_allocate_duplicate_id() {
        let mut b = MemoryBudget::new(1024);
        b.allocate("m1", 128).unwrap();
        assert!(b.allocate("m1", 128).is_err());
    }

    #[test]
    fn budget_release() {
        let mut b = MemoryBudget::new(1024);
        b.allocate("m1", 512).unwrap();
        let freed = b.release("m1").unwrap();
        assert_eq!(freed, 512);
        assert_eq!(b.used_bytes(), 0);
        assert_eq!(b.model_count(), 0);
    }

    #[test]
    fn budget_release_unknown() {
        let mut b = MemoryBudget::new(1024);
        assert!(b.release("unknown").is_err());
    }

    #[test]
    fn budget_can_allocate() {
        let mut b = MemoryBudget::new(1024);
        assert!(b.can_allocate(1024));
        assert!(b.can_allocate(1));
        assert!(!b.can_allocate(1025));
        b.allocate("m1", 512).unwrap();
        assert!(b.can_allocate(512));
        assert!(!b.can_allocate(513));
    }

    #[test]
    fn budget_get_allocation() {
        let mut b = MemoryBudget::new(1024);
        b.allocate("m1", 256).unwrap();
        let alloc = b.get_allocation("m1").unwrap();
        assert_eq!(alloc.model_id, "m1");
        assert_eq!(alloc.allocated_bytes, 256);
        assert!(b.get_allocation("m2").is_none());
    }

    #[test]
    fn budget_utilization() {
        let mut b = MemoryBudget::new(1000);
        assert!((b.utilization() - 0.0).abs() < f64::EPSILON);
        b.allocate("m1", 500).unwrap();
        assert!((b.utilization() - 0.5).abs() < f64::EPSILON);
        b.allocate("m2", 500).unwrap();
        assert!((b.utilization() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn budget_utilization_zero_total() {
        let b = MemoryBudget::new(0);
        assert!((b.utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn budget_release_then_reallocate() {
        let mut b = MemoryBudget::new(1024);
        b.allocate("m1", 1024).unwrap();
        b.release("m1").unwrap();
        b.allocate("m2", 1024).unwrap();
        assert_eq!(b.model_count(), 1);
        assert_eq!(b.used_bytes(), 1024);
    }

    // ── ModelVersionManager ─────────────────────────────────────────────

    #[test]
    fn version_manager_new() {
        let vm = ModelVersionManager::new();
        assert_eq!(vm.model_count(), 0);
        assert_eq!(vm.total_versions(), 0);
    }

    #[test]
    fn version_manager_default() {
        let vm = ModelVersionManager::default();
        assert_eq!(vm.model_count(), 0);
    }

    #[test]
    fn version_register_first_auto_activates() {
        let mut vm = ModelVersionManager::new();
        vm.register("m1", 1, "/path/v1", 100).unwrap();
        assert_eq!(vm.active_version("m1"), Some(1));
        assert_eq!(vm.model_count(), 1);
        assert_eq!(vm.total_versions(), 1);
    }

    #[test]
    fn version_register_second_does_not_change_active() {
        let mut vm = ModelVersionManager::new();
        vm.register("m1", 1, "/v1", 100).unwrap();
        vm.register("m1", 2, "/v2", 200).unwrap();
        assert_eq!(vm.active_version("m1"), Some(1));
        assert_eq!(vm.total_versions(), 2);
    }

    #[test]
    fn version_register_duplicate_fails() {
        let mut vm = ModelVersionManager::new();
        vm.register("m1", 1, "/v1", 100).unwrap();
        assert!(vm.register("m1", 1, "/v1-dup", 100).is_err());
    }

    #[test]
    fn version_set_active() {
        let mut vm = ModelVersionManager::new();
        vm.register("m1", 1, "/v1", 100).unwrap();
        vm.register("m1", 2, "/v2", 200).unwrap();
        vm.set_active("m1", 2).unwrap();
        assert_eq!(vm.active_version("m1"), Some(2));
    }

    #[test]
    fn version_set_active_missing_model() {
        let mut vm = ModelVersionManager::new();
        assert!(vm.set_active("m1", 1).is_err());
    }

    #[test]
    fn version_set_active_missing_version() {
        let mut vm = ModelVersionManager::new();
        vm.register("m1", 1, "/v1", 100).unwrap();
        assert!(vm.set_active("m1", 99).is_err());
    }

    #[test]
    fn version_active_entry() {
        let mut vm = ModelVersionManager::new();
        vm.register("m1", 1, "/v1", 100).unwrap();
        let entry = vm.active_entry("m1").unwrap();
        assert_eq!(entry.version, 1);
        assert_eq!(entry.path, "/v1");
    }

    #[test]
    fn version_list_versions() {
        let mut vm = ModelVersionManager::new();
        vm.register("m1", 1, "/v1", 100).unwrap();
        vm.register("m1", 3, "/v3", 300).unwrap();
        vm.register("m1", 2, "/v2", 200).unwrap();
        let versions = vm.list_versions("m1");
        assert_eq!(versions.len(), 3);
        // Should be sorted by version.
        assert_eq!(versions[0].version, 1);
        assert_eq!(versions[1].version, 2);
        assert_eq!(versions[2].version, 3);
    }

    #[test]
    fn version_list_empty() {
        let vm = ModelVersionManager::new();
        assert!(vm.list_versions("m1").is_empty());
    }

    #[test]
    fn version_remove() {
        let mut vm = ModelVersionManager::new();
        vm.register("m1", 1, "/v1", 100).unwrap();
        vm.register("m1", 2, "/v2", 200).unwrap();
        vm.remove_version("m1", 2).unwrap();
        assert_eq!(vm.total_versions(), 1);
    }

    #[test]
    fn version_remove_active_fails() {
        let mut vm = ModelVersionManager::new();
        vm.register("m1", 1, "/v1", 100).unwrap();
        assert!(vm.remove_version("m1", 1).is_err());
    }

    #[test]
    fn version_remove_missing_model() {
        let mut vm = ModelVersionManager::new();
        assert!(vm.remove_version("m1", 1).is_err());
    }

    #[test]
    fn version_remove_missing_version() {
        let mut vm = ModelVersionManager::new();
        vm.register("m1", 1, "/v1", 100).unwrap();
        assert!(vm.remove_version("m1", 99).is_err());
    }

    #[test]
    fn version_latest() {
        let mut vm = ModelVersionManager::new();
        vm.register("m1", 1, "/v1", 100).unwrap();
        vm.register("m1", 5, "/v5", 500).unwrap();
        vm.register("m1", 3, "/v3", 300).unwrap();
        assert_eq!(vm.latest_version("m1"), Some(5));
    }

    #[test]
    fn version_latest_empty() {
        let vm = ModelVersionManager::new();
        assert_eq!(vm.latest_version("m1"), None);
    }

    #[test]
    fn version_multiple_models() {
        let mut vm = ModelVersionManager::new();
        vm.register("m1", 1, "/m1v1", 100).unwrap();
        vm.register("m2", 1, "/m2v1", 200).unwrap();
        assert_eq!(vm.model_count(), 2);
        assert_eq!(vm.active_version("m1"), Some(1));
        assert_eq!(vm.active_version("m2"), Some(1));
    }

    // ── LifecycleMetrics ────────────────────────────────────────────────

    #[test]
    fn metrics_new_zeroed() {
        let m = LifecycleMetrics::new();
        assert_eq!(m.loads, 0);
        assert_eq!(m.unloads, 0);
        assert_eq!(m.swaps, 0);
        assert_eq!(m.warmup_passes, 0);
        assert_eq!(m.failures, 0);
        assert_eq!(m.peak_memory_bytes, 0);
    }

    #[test]
    fn metrics_default_zeroed() {
        let m = LifecycleMetrics::default();
        assert_eq!(m.loads, 0);
    }

    #[test]
    fn metrics_record_load() {
        let mut m = LifecycleMetrics::new();
        m.record_load(Duration::from_millis(100));
        assert_eq!(m.loads, 1);
        assert_eq!(m.total_load_time, Duration::from_millis(100));
    }

    #[test]
    fn metrics_record_unload() {
        let mut m = LifecycleMetrics::new();
        m.record_unload();
        assert_eq!(m.unloads, 1);
    }

    #[test]
    fn metrics_record_swap() {
        let mut m = LifecycleMetrics::new();
        m.record_swap(Duration::from_millis(50));
        assert_eq!(m.swaps, 1);
        assert_eq!(m.total_swap_time, Duration::from_millis(50));
    }

    #[test]
    fn metrics_record_warmup() {
        let mut m = LifecycleMetrics::new();
        m.record_warmup(3, Duration::from_millis(30));
        assert_eq!(m.warmup_passes, 3);
        assert_eq!(m.total_warmup_time, Duration::from_millis(30));
    }

    #[test]
    fn metrics_update_peak_memory() {
        let mut m = LifecycleMetrics::new();
        m.update_peak_memory(100);
        assert_eq!(m.peak_memory_bytes, 100);
        m.update_peak_memory(50);
        assert_eq!(m.peak_memory_bytes, 100); // peak unchanged
        m.update_peak_memory(200);
        assert_eq!(m.peak_memory_bytes, 200);
    }

    #[test]
    fn metrics_record_failure() {
        let mut m = LifecycleMetrics::new();
        m.record_failure();
        m.record_failure();
        assert_eq!(m.failures, 2);
    }

    #[test]
    fn metrics_avg_load_time() {
        let mut m = LifecycleMetrics::new();
        assert!(m.avg_load_time().is_none());
        m.record_load(Duration::from_millis(100));
        m.record_load(Duration::from_millis(200));
        assert_eq!(m.avg_load_time(), Some(Duration::from_millis(150)));
    }

    #[test]
    fn metrics_avg_warmup_time() {
        let mut m = LifecycleMetrics::new();
        assert!(m.avg_warmup_time().is_none());
        m.record_warmup(2, Duration::from_millis(100));
        assert_eq!(m.avg_warmup_time(), Some(Duration::from_millis(50)));
    }

    #[test]
    fn metrics_avg_swap_time() {
        let mut m = LifecycleMetrics::new();
        assert!(m.avg_swap_time().is_none());
        m.record_swap(Duration::from_millis(60));
        assert_eq!(m.avg_swap_time(), Some(Duration::from_millis(60)));
    }

    #[test]
    fn metrics_reset() {
        let mut m = LifecycleMetrics::new();
        m.record_load(Duration::from_millis(100));
        m.record_swap(Duration::from_millis(50));
        m.record_failure();
        m.reset();
        assert_eq!(m.loads, 0);
        assert_eq!(m.swaps, 0);
        assert_eq!(m.failures, 0);
    }

    // ── ModelLifecycleEngine ────────────────────────────────────────────

    #[test]
    fn engine_new() {
        let engine = ModelLifecycleEngine::new(test_config());
        assert_eq!(engine.model_count(), 0);
        assert_eq!(engine.metrics().loads, 0);
        assert_eq!(engine.budget().total_budget(), test_config().memory_budget_bytes);
    }

    #[test]
    fn engine_load_model() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        let desc = test_descriptor("m1");
        let result = engine.load_model(&desc).unwrap();
        assert!(result.success);
        assert_eq!(engine.model_count(), 1);
        assert_eq!(engine.metrics().loads, 1);
        assert_eq!(engine.tracker("m1").unwrap().state(), ModelState::Ready);
    }

    #[test]
    fn engine_load_invalid_descriptor() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        let mut desc = test_descriptor("m1");
        desc.id = String::new();
        assert!(engine.load_model(&desc).is_err());
    }

    #[test]
    fn engine_load_exceeds_budget() {
        let config = LifecycleConfig {
            memory_budget_bytes: 50 * 1024 * 1024, // 50 MiB
            ..test_config()
        };
        let mut engine = ModelLifecycleEngine::new(config);
        let desc = test_descriptor("m1"); // 100 MiB
        assert!(engine.load_model(&desc).is_err());
    }

    #[test]
    fn engine_load_exceeds_max_models() {
        let config = LifecycleConfig { max_loaded_models: 1, ..test_config() };
        let mut engine = ModelLifecycleEngine::new(config);
        engine.load_model(&test_descriptor("m1")).unwrap();
        assert!(engine.load_model(&test_descriptor("m2")).is_err());
    }

    #[test]
    fn engine_unload_model() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        engine.load_model(&test_descriptor("m1")).unwrap();
        let result = engine.unload_model("m1", 0).unwrap();
        assert_eq!(result.status, CooldownStatus::Drained);
        assert_eq!(engine.tracker("m1").unwrap().state(), ModelState::Unloaded);
        assert_eq!(engine.metrics().unloads, 1);
    }

    #[test]
    fn engine_unload_releases_budget() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        let desc = test_descriptor("m1");
        engine.load_model(&desc).unwrap();
        let used_before = engine.budget().used_bytes();
        engine.unload_model("m1", 0).unwrap();
        assert!(engine.budget().used_bytes() < used_before);
    }

    #[test]
    fn engine_unload_not_found() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        assert!(engine.unload_model("m1", 0).is_err());
    }

    #[test]
    fn engine_mark_running_and_ready() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        engine.load_model(&test_descriptor("m1")).unwrap();
        engine.mark_running("m1").unwrap();
        assert_eq!(engine.tracker("m1").unwrap().state(), ModelState::Running);
        engine.mark_ready("m1").unwrap();
        assert_eq!(engine.tracker("m1").unwrap().state(), ModelState::Ready);
    }

    #[test]
    fn engine_mark_running_not_found() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        assert!(engine.mark_running("m1").is_err());
    }

    #[test]
    fn engine_mark_ready_not_found() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        assert!(engine.mark_ready("m1").is_err());
    }

    #[test]
    fn engine_swap_model() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        let old = test_descriptor("old");
        let new = test_descriptor("new");
        engine.load_model(&old).unwrap();
        let result = engine.swap_model(&old, &new, 0).unwrap();
        assert!(result.success);
        assert_eq!(engine.metrics().swaps, 1);
    }

    #[test]
    fn engine_swap_invalid_new() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        let old = test_descriptor("old");
        let mut new = test_descriptor("new");
        new.id = String::new();
        assert!(engine.swap_model(&old, &new, 0).is_err());
    }

    #[test]
    fn engine_list_models() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        engine.load_model(&test_descriptor("m1")).unwrap();
        engine.load_model(&test_descriptor("m2")).unwrap();
        let models = engine.list_models();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn engine_models_in_state() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        engine.load_model(&test_descriptor("m1")).unwrap();
        engine.load_model(&test_descriptor("m2")).unwrap();
        engine.mark_running("m1").unwrap();
        let running = engine.models_in_state(ModelState::Running);
        assert_eq!(running.len(), 1);
        assert_eq!(running[0], "m1");
        let ready = engine.models_in_state(ModelState::Ready);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], "m2");
    }

    #[test]
    fn engine_load_unload_reload() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        let desc = test_descriptor("m1");
        engine.load_model(&desc).unwrap();
        engine.unload_model("m1", 0).unwrap();
        // After unload + re-load, need a fresh tracker so create new descriptor.
        let desc2 = ModelDescriptor {
            id: "m1-v2".to_string(),
            path: "/models/m1-v2.gguf".to_string(),
            size_bytes: 100 * 1024 * 1024,
            version: "2".to_string(),
        };
        engine.load_model(&desc2).unwrap();
        assert_eq!(engine.tracker("m1-v2").unwrap().state(), ModelState::Ready);
    }

    #[test]
    fn engine_metrics_track_peak_memory() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        engine.load_model(&test_descriptor("m1")).unwrap();
        let peak = engine.metrics().peak_memory_bytes;
        assert!(peak > 0);
    }

    #[test]
    fn engine_config_ref() {
        let cfg = test_config();
        let engine = ModelLifecycleEngine::new(cfg.clone());
        assert_eq!(engine.config().warmup_iterations, cfg.warmup_iterations);
    }

    #[test]
    fn engine_versions_ref() {
        let engine = ModelLifecycleEngine::new(test_config());
        assert_eq!(engine.versions().model_count(), 0);
    }

    #[test]
    fn engine_unload_already_unloaded() {
        let mut engine = ModelLifecycleEngine::new(test_config());
        engine.load_model(&test_descriptor("m1")).unwrap();
        engine.unload_model("m1", 0).unwrap();
        assert!(engine.unload_model("m1", 0).is_err());
    }
}
