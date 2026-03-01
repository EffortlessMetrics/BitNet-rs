//! Unified backend selector with automatic detection and fallback.
//!
//! Probes the system for available GPU/compute backends, scores each against
//! workload requirements, and selects the best match. Provides fallback chains,
//! user overrides, and lifecycle management.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ── Backend type ────────────────────────────────────────────────────────────

/// Supported compute backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    CUDA,
    OpenCL,
    Vulkan,
    Metal,
    ROCm,
    WebGPU,
    LevelZero,
    CPU,
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CUDA => write!(f, "CUDA"),
            Self::OpenCL => write!(f, "OpenCL"),
            Self::Vulkan => write!(f, "Vulkan"),
            Self::Metal => write!(f, "Metal"),
            Self::ROCm => write!(f, "ROCm"),
            Self::WebGPU => write!(f, "WebGPU"),
            Self::LevelZero => write!(f, "LevelZero"),
            Self::CPU => write!(f, "CPU"),
        }
    }
}

impl BackendType {
    /// All backend types in default priority order.
    pub const fn all() -> &'static [Self] {
        &[
            Self::CUDA,
            Self::ROCm,
            Self::Metal,
            Self::LevelZero,
            Self::Vulkan,
            Self::OpenCL,
            Self::WebGPU,
            Self::CPU,
        ]
    }

    /// Whether this backend targets a GPU device.
    pub const fn is_gpu(&self) -> bool {
        !matches!(self, Self::CPU)
    }
}

// ── Backend priority ────────────────────────────────────────────────────────

/// Ordered list of preferred backends, highest priority first.
#[derive(Debug, Clone)]
pub struct BackendPriority {
    order: Vec<BackendType>,
}

impl Default for BackendPriority {
    fn default() -> Self {
        Self { order: BackendType::all().to_vec() }
    }
}

impl BackendPriority {
    /// Create a priority list from the given order.
    pub const fn new(order: Vec<BackendType>) -> Self {
        Self { order }
    }

    /// Return the ordered slice.
    pub fn order(&self) -> &[BackendType] {
        &self.order
    }

    /// Priority rank of a backend (0 = highest). `None` if absent.
    pub fn rank(&self, backend: BackendType) -> Option<usize> {
        self.order.iter().position(|&b| b == backend)
    }

    /// Returns the highest-priority backend, if any.
    pub fn first(&self) -> Option<BackendType> {
        self.order.first().copied()
    }
}

// ── Data-type support ───────────────────────────────────────────────────────

/// Data types a backend may support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I2,
}

// ── Backend capability ──────────────────────────────────────────────────────

/// What a specific backend instance supports.
#[derive(Debug, Clone)]
pub struct BackendCapability {
    pub backend: BackendType,
    pub supported_dtypes: Vec<DType>,
    pub max_buffer_bytes: u64,
    pub shared_memory_bytes: u64,
    pub max_workgroup_size: u32,
    pub supports_unified_memory: bool,
    pub compute_units: u32,
    pub driver_version: String,
    pub device_name: String,
}

impl BackendCapability {
    /// Whether the backend supports a particular data type.
    pub fn supports_dtype(&self, dtype: DType) -> bool {
        self.supported_dtypes.contains(&dtype)
    }

    /// Rough throughput score based on compute units and buffer size.
    #[allow(clippy::cast_precision_loss)]
    pub fn throughput_score(&self) -> f64 {
        f64::from(self.compute_units) * (self.max_buffer_bytes as f64).log2().max(1.0)
    }
}

// ── Probe result ────────────────────────────────────────────────────────────

/// Outcome of probing a single backend.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub backend: BackendType,
    pub available: bool,
    pub capability: Option<BackendCapability>,
    pub probe_duration: Duration,
    pub error: Option<String>,
}

// ── Backend probe ───────────────────────────────────────────────────────────

/// Probes the system for available compute backends.
#[derive(Debug)]
pub struct BackendProbe {
    timeout: Duration,
    results: Vec<ProbeResult>,
}

impl Default for BackendProbe {
    fn default() -> Self {
        Self { timeout: Duration::from_secs(5), results: Vec::new() }
    }
}

impl BackendProbe {
    /// Create a probe with the given timeout per backend.
    pub const fn with_timeout(timeout: Duration) -> Self {
        Self { timeout, results: Vec::new() }
    }

    /// Probe all backends in the given priority order.
    pub fn probe_all(&mut self, priority: &BackendPriority) {
        self.results.clear();
        for &backend in priority.order() {
            self.results.push(self.probe_one(backend));
        }
    }

    /// Probe a single backend.
    pub fn probe_one(&self, backend: BackendType) -> ProbeResult {
        let start = Instant::now();
        // CPU is always available; GPU backends require runtime detection.
        if backend == BackendType::CPU {
            return ProbeResult {
                backend,
                available: true,
                capability: Some(Self::cpu_capability()),
                probe_duration: start.elapsed(),
                error: None,
            };
        }
        // In this scaffold, GPU backends report unavailable.
        ProbeResult {
            backend,
            available: false,
            capability: None,
            probe_duration: start.elapsed(),
            error: Some(format!(
                "{backend} runtime not detected (probe timeout={:?})",
                self.timeout,
            )),
        }
    }

    /// All probe results collected so far.
    pub fn results(&self) -> &[ProbeResult] {
        &self.results
    }

    /// Only the available backends.
    pub fn available(&self) -> Vec<&ProbeResult> {
        self.results.iter().filter(|r| r.available).collect()
    }

    /// The configured probe timeout.
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    fn cpu_capability() -> BackendCapability {
        BackendCapability {
            backend: BackendType::CPU,
            supported_dtypes: vec![DType::F32, DType::F16, DType::BF16, DType::I8, DType::I2],
            max_buffer_bytes: u64::MAX,
            shared_memory_bytes: 0,
            max_workgroup_size: 1,
            supports_unified_memory: true,
            compute_units: 1,
            driver_version: "host".into(),
            device_name: "CPU".into(),
        }
    }
}

// ── Workload requirements ───────────────────────────────────────────────────

/// Describes what the workload needs from a backend.
#[derive(Debug, Clone)]
pub struct WorkloadRequirements {
    pub required_dtypes: Vec<DType>,
    pub min_buffer_bytes: u64,
    pub min_shared_memory_bytes: u64,
    pub prefer_gpu: bool,
}

impl Default for WorkloadRequirements {
    fn default() -> Self {
        Self {
            required_dtypes: vec![DType::F32],
            min_buffer_bytes: 0,
            min_shared_memory_bytes: 0,
            prefer_gpu: true,
        }
    }
}

// ── Backend scorer ──────────────────────────────────────────────────────────

/// Score assigned to a backend after evaluating its capability match.
#[derive(Debug, Clone)]
pub struct BackendScore {
    pub backend: BackendType,
    pub total: f64,
    pub dtype_match: f64,
    pub memory_match: f64,
    pub throughput: f64,
    pub priority_bonus: f64,
    pub gpu_bonus: f64,
    pub meets_requirements: bool,
}

/// Scores backends based on capability match to workload requirements.
#[derive(Debug)]
pub struct BackendScorer {
    pub dtype_weight: f64,
    pub memory_weight: f64,
    pub throughput_weight: f64,
    pub priority_weight: f64,
    pub gpu_weight: f64,
}

impl Default for BackendScorer {
    fn default() -> Self {
        Self {
            dtype_weight: 30.0,
            memory_weight: 20.0,
            throughput_weight: 25.0,
            priority_weight: 15.0,
            gpu_weight: 10.0,
        }
    }
}

impl BackendScorer {
    /// Score a single backend against the workload requirements.
    #[allow(clippy::cast_precision_loss)]
    pub fn score(
        &self,
        cap: &BackendCapability,
        reqs: &WorkloadRequirements,
        priority: &BackendPriority,
    ) -> BackendScore {
        let total_backends = priority.order().len().max(1) as f64;
        let rank = priority.rank(cap.backend).unwrap_or_else(|| priority.order().len()) as f64;

        // dtype coverage
        let required_count = reqs.required_dtypes.len().max(1);
        let matched_count = reqs.required_dtypes.iter().filter(|d| cap.supports_dtype(**d)).count();
        let dtype_match = matched_count as f64 / required_count as f64;
        let meets_dtypes = matched_count == reqs.required_dtypes.len();

        // memory fit
        let memory_match = if cap.max_buffer_bytes >= reqs.min_buffer_bytes {
            1.0
        } else {
            cap.max_buffer_bytes as f64 / reqs.min_buffer_bytes as f64
        };
        let meets_memory = cap.max_buffer_bytes >= reqs.min_buffer_bytes;

        // shared memory
        let meets_shared = cap.shared_memory_bytes >= reqs.min_shared_memory_bytes;

        // throughput (normalised 0–1 assuming max ~10 000)
        let throughput = (cap.throughput_score() / 10_000.0).min(1.0);

        // priority bonus: highest-priority backend gets 1.0
        let priority_bonus = 1.0 - rank / total_backends;

        // GPU bonus
        let gpu_bonus = if reqs.prefer_gpu && cap.backend.is_gpu() { 1.0 } else { 0.0 };

        let total = dtype_match * self.dtype_weight
            + memory_match * self.memory_weight
            + throughput * self.throughput_weight
            + priority_bonus * self.priority_weight
            + gpu_bonus * self.gpu_weight;

        let meets_requirements = meets_dtypes && meets_memory && meets_shared;

        BackendScore {
            backend: cap.backend,
            total,
            dtype_match,
            memory_match,
            throughput,
            priority_bonus,
            gpu_bonus,
            meets_requirements,
        }
    }

    /// Score all capabilities and return sorted (best first).
    pub fn score_all(
        &self,
        caps: &[BackendCapability],
        reqs: &WorkloadRequirements,
        priority: &BackendPriority,
    ) -> Vec<BackendScore> {
        let mut scores: Vec<BackendScore> =
            caps.iter().map(|c| self.score(c, reqs, priority)).collect();
        scores.sort_by(|a, b| b.total.partial_cmp(&a.total).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }
}

// ── Selection result ────────────────────────────────────────────────────────

/// Outcome of backend selection.
#[derive(Debug, Clone)]
pub struct SelectionResult {
    pub selected: BackendType,
    pub score: f64,
    pub reason: String,
    pub alternatives: Vec<(BackendType, f64)>,
}

// ── Backend selector ────────────────────────────────────────────────────────

/// Selects the best backend: probe → score → validate → select.
#[derive(Debug, Default)]
pub struct BackendSelector {
    priority: BackendPriority,
    scorer: BackendScorer,
    config: BackendConfig,
}

impl BackendSelector {
    /// Create a selector with custom priority, scorer, and config.
    pub const fn new(
        priority: BackendPriority,
        scorer: BackendScorer,
        config: BackendConfig,
    ) -> Self {
        Self { priority, scorer, config }
    }

    /// Run selection: probe the system, score, pick the best.
    pub fn select(&self, reqs: &WorkloadRequirements) -> Result<SelectionResult, String> {
        // Honour forced backend override.
        if let Some(forced) = self.config.forced_backend {
            return Ok(SelectionResult {
                selected: forced,
                score: 0.0,
                reason: format!("forced by config: {forced}"),
                alternatives: vec![],
            });
        }

        let mut probe = BackendProbe::with_timeout(self.config.probe_timeout);
        probe.probe_all(&self.priority);

        let caps: Vec<BackendCapability> =
            probe.available().into_iter().filter_map(|r| r.capability.clone()).collect();

        if caps.is_empty() {
            return Err("no available backends detected".into());
        }

        let scores = self.scorer.score_all(&caps, reqs, &self.priority);

        // Prefer backends that meet all requirements.
        let best = scores
            .iter()
            .filter(|s| s.meets_requirements)
            .max_by(|a, b| a.total.partial_cmp(&b.total).unwrap_or(std::cmp::Ordering::Equal))
            .or_else(|| scores.first());

        best.map_or_else(
            || Err("scoring produced no candidates".into()),
            |b| {
                let alternatives = scores
                    .iter()
                    .filter(|s| s.backend != b.backend)
                    .map(|s| (s.backend, s.total))
                    .collect();
                Ok(SelectionResult {
                    selected: b.backend,
                    score: b.total,
                    reason: format!(
                        "highest score ({:.2}) among {} candidate(s)",
                        b.total,
                        scores.len(),
                    ),
                    alternatives,
                })
            },
        )
    }

    /// Access the current priority list.
    pub const fn priority(&self) -> &BackendPriority {
        &self.priority
    }

    /// Access the current config.
    pub const fn config(&self) -> &BackendConfig {
        &self.config
    }
}

// ── Backend fallback ────────────────────────────────────────────────────────

/// Fallback chain when the primary backend fails.
#[derive(Debug, Clone)]
pub struct BackendFallback {
    chain: Vec<BackendType>,
    max_retries: u32,
}

impl Default for BackendFallback {
    fn default() -> Self {
        Self { chain: vec![BackendType::CPU], max_retries: 2 }
    }
}

impl BackendFallback {
    /// Create a fallback chain.
    pub const fn new(chain: Vec<BackendType>, max_retries: u32) -> Self {
        Self { chain, max_retries }
    }

    /// The ordered fallback chain.
    pub fn chain(&self) -> &[BackendType] {
        &self.chain
    }

    /// Maximum retries per backend before moving to the next.
    pub const fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Walk the chain, returning the first backend accepted by `f`.
    pub fn try_fallback<F>(&self, mut f: F) -> Option<BackendType>
    where
        F: FnMut(BackendType) -> bool,
    {
        for &backend in &self.chain {
            for _ in 0..=self.max_retries {
                if f(backend) {
                    return Some(backend);
                }
            }
        }
        None
    }

    /// Whether the chain is empty.
    pub const fn is_empty(&self) -> bool {
        self.chain.is_empty()
    }

    /// Length of the chain.
    pub const fn len(&self) -> usize {
        self.chain.len()
    }
}

// ── Backend config ──────────────────────────────────────────────────────────

/// User overrides for backend selection.
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Force a specific backend, skipping detection.
    pub forced_backend: Option<BackendType>,
    /// Backends to exclude from consideration.
    pub excluded_backends: Vec<BackendType>,
    /// Probe timeout per backend.
    pub probe_timeout: Duration,
    /// Fallback configuration.
    pub fallback: BackendFallback,
    /// Whether to allow CPU as a last resort.
    pub allow_cpu_fallback: bool,
    /// Extra key–value overrides forwarded to backend init.
    pub extra: HashMap<String, String>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            forced_backend: None,
            excluded_backends: Vec::new(),
            probe_timeout: Duration::from_secs(5),
            fallback: BackendFallback::default(),
            allow_cpu_fallback: true,
            extra: HashMap::new(),
        }
    }
}

impl BackendConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.probe_timeout.is_zero() {
            return Err("probe_timeout must be > 0".into());
        }
        if let Some(forced) = self.forced_backend
            && self.excluded_backends.contains(&forced)
        {
            return Err(format!("forced backend {forced} is also excluded"));
        }
        Ok(())
    }
}

// ── Backend handle ──────────────────────────────────────────────────────────

/// Opaque handle returned after a backend is initialised.
#[derive(Debug)]
pub struct BackendHandle {
    pub backend: BackendType,
    pub device_name: String,
    pub init_duration: Duration,
    active: bool,
}

impl BackendHandle {
    /// Whether the handle is still active.
    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// Shut down the handle.
    pub const fn shutdown(&mut self) {
        self.active = false;
    }
}

// ── Backend initializer ─────────────────────────────────────────────────────

/// Initialises a selected backend and returns a handle.
#[derive(Debug, Default)]
pub struct BackendInitializer {
    config: BackendConfig,
}

impl BackendInitializer {
    /// Create an initializer with the given config.
    pub const fn new(config: BackendConfig) -> Self {
        Self { config }
    }

    /// Initialise the given backend. CPU always succeeds; GPU backends
    /// return an error in this scaffold.
    pub fn initialize(&self, backend: BackendType) -> Result<BackendHandle, String> {
        let start = Instant::now();
        match backend {
            BackendType::CPU => Ok(BackendHandle {
                backend,
                device_name: "CPU".into(),
                init_duration: start.elapsed(),
                active: true,
            }),
            other => Err(format!("{other} initialisation not yet implemented")),
        }
    }

    /// Access the config.
    pub const fn config(&self) -> &BackendConfig {
        &self.config
    }
}

// ── Manager state ───────────────────────────────────────────────────────────

/// Lifecycle phase of the backend manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagerState {
    Idle,
    Selecting,
    Initializing,
    Running,
    FallingBack,
    ShuttingDown,
    Stopped,
}

impl fmt::Display for ManagerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Selecting => write!(f, "Selecting"),
            Self::Initializing => write!(f, "Initializing"),
            Self::Running => write!(f, "Running"),
            Self::FallingBack => write!(f, "FallingBack"),
            Self::ShuttingDown => write!(f, "ShuttingDown"),
            Self::Stopped => write!(f, "Stopped"),
        }
    }
}

// ── Manager event ───────────────────────────────────────────────────────────

/// Events emitted by the backend manager.
#[derive(Debug, Clone)]
pub struct ManagerEvent {
    pub kind: ManagerEventKind,
    pub timestamp: Instant,
    pub message: String,
}

/// Event kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagerEventKind {
    ProbeStarted,
    ProbeCompleted,
    SelectionMade,
    InitStarted,
    InitCompleted,
    FallbackTriggered,
    Shutdown,
    Error,
}

// ── Backend manager ─────────────────────────────────────────────────────────

/// Manages the full lifecycle: select → init → monitor → fallback → shutdown.
#[derive(Debug)]
pub struct BackendManager {
    selector: BackendSelector,
    initializer: BackendInitializer,
    state: ManagerState,
    handle: Option<BackendHandle>,
    events: Vec<ManagerEvent>,
    start_time: Option<Instant>,
}

impl Default for BackendManager {
    fn default() -> Self {
        Self {
            selector: BackendSelector::default(),
            initializer: BackendInitializer::default(),
            state: ManagerState::Idle,
            handle: None,
            events: Vec::new(),
            start_time: None,
        }
    }
}

impl BackendManager {
    /// Create a manager with custom selector and initializer.
    pub const fn new(selector: BackendSelector, initializer: BackendInitializer) -> Self {
        Self {
            selector,
            initializer,
            state: ManagerState::Idle,
            handle: None,
            events: Vec::new(),
            start_time: None,
        }
    }

    /// Current lifecycle state.
    pub const fn state(&self) -> ManagerState {
        self.state
    }

    /// The active backend handle, if any.
    pub const fn handle(&self) -> Option<&BackendHandle> {
        self.handle.as_ref()
    }

    /// All recorded events.
    pub fn events(&self) -> &[ManagerEvent] {
        &self.events
    }

    /// Start the manager: select, init, and run.
    pub fn start(&mut self, reqs: &WorkloadRequirements) -> Result<BackendType, String> {
        if self.state != ManagerState::Idle && self.state != ManagerState::Stopped {
            return Err(format!("cannot start from state {}", self.state));
        }
        self.start_time = Some(Instant::now());

        // Selection phase.
        self.transition(ManagerState::Selecting);
        self.push_event(ManagerEventKind::ProbeStarted, "probing backends");
        let selection = self.selector.select(reqs);
        self.push_event(ManagerEventKind::ProbeCompleted, "probe complete");

        let result = match selection {
            Ok(sel) => sel,
            Err(e) => {
                self.push_event(ManagerEventKind::Error, &e);
                return self.try_fallback(reqs);
            }
        };

        self.push_event(ManagerEventKind::SelectionMade, &format!("selected {}", result.selected));

        // Init phase.
        self.transition(ManagerState::Initializing);
        self.push_event(
            ManagerEventKind::InitStarted,
            &format!("initialising {}", result.selected),
        );

        match self.initializer.initialize(result.selected) {
            Ok(handle) => {
                self.push_event(ManagerEventKind::InitCompleted, "init ok");
                let backend = handle.backend;
                self.handle = Some(handle);
                self.transition(ManagerState::Running);
                Ok(backend)
            }
            Err(e) => {
                self.push_event(ManagerEventKind::Error, &e);
                self.try_fallback(reqs)
            }
        }
    }

    /// Attempt the fallback chain.
    fn try_fallback(&mut self, _reqs: &WorkloadRequirements) -> Result<BackendType, String> {
        self.transition(ManagerState::FallingBack);
        self.push_event(ManagerEventKind::FallbackTriggered, "entering fallback chain");

        let chain = self.selector.config().fallback.chain().to_vec();
        for &backend in &chain {
            if let Ok(handle) = self.initializer.initialize(backend) {
                let bt = handle.backend;
                self.handle = Some(handle);
                self.transition(ManagerState::Running);
                return Ok(bt);
            }
        }

        // Last resort: CPU if allowed.
        if self.selector.config().allow_cpu_fallback
            && let Ok(handle) = self.initializer.initialize(BackendType::CPU)
        {
            let bt = handle.backend;
            self.handle = Some(handle);
            self.transition(ManagerState::Running);
            return Ok(bt);
        }

        self.transition(ManagerState::Stopped);
        Err("all backends (including fallback) failed".into())
    }

    /// Shutdown the manager.
    pub fn shutdown(&mut self) {
        self.transition(ManagerState::ShuttingDown);
        if let Some(ref mut h) = self.handle {
            h.shutdown();
        }
        self.push_event(ManagerEventKind::Shutdown, "shutdown complete");
        self.transition(ManagerState::Stopped);
    }

    /// Whether a backend is currently running.
    pub fn is_running(&self) -> bool {
        self.state == ManagerState::Running
    }

    /// Elapsed time since `start()` was called.
    pub fn uptime(&self) -> Duration {
        self.start_time.map_or(Duration::ZERO, |t| t.elapsed())
    }

    fn transition(&mut self, new: ManagerState) {
        log::debug!("BackendManager: {} -> {}", self.state, new);
        self.state = new;
    }

    fn push_event(&mut self, kind: ManagerEventKind, msg: &str) {
        self.events.push(ManagerEvent {
            kind,
            timestamp: Instant::now(),
            message: msg.to_string(),
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── BackendType ─────────────────────────────────────────────────────

    #[test]
    fn backend_type_display() {
        assert_eq!(BackendType::CUDA.to_string(), "CUDA");
        assert_eq!(BackendType::OpenCL.to_string(), "OpenCL");
        assert_eq!(BackendType::Vulkan.to_string(), "Vulkan");
        assert_eq!(BackendType::Metal.to_string(), "Metal");
        assert_eq!(BackendType::ROCm.to_string(), "ROCm");
        assert_eq!(BackendType::WebGPU.to_string(), "WebGPU");
        assert_eq!(BackendType::LevelZero.to_string(), "LevelZero");
        assert_eq!(BackendType::CPU.to_string(), "CPU");
    }

    #[test]
    fn backend_type_all_contains_every_variant() {
        let all = BackendType::all();
        assert!(all.contains(&BackendType::CUDA));
        assert!(all.contains(&BackendType::OpenCL));
        assert!(all.contains(&BackendType::Vulkan));
        assert!(all.contains(&BackendType::Metal));
        assert!(all.contains(&BackendType::ROCm));
        assert!(all.contains(&BackendType::WebGPU));
        assert!(all.contains(&BackendType::LevelZero));
        assert!(all.contains(&BackendType::CPU));
        assert_eq!(all.len(), 8);
    }

    #[test]
    fn backend_type_is_gpu() {
        assert!(BackendType::CUDA.is_gpu());
        assert!(BackendType::Vulkan.is_gpu());
        assert!(!BackendType::CPU.is_gpu());
    }

    #[test]
    fn backend_type_eq_and_hash() {
        let mut set = std::collections::HashSet::new();
        set.insert(BackendType::CUDA);
        set.insert(BackendType::CUDA);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn backend_type_clone() {
        let a = BackendType::Metal;
        let b = a;
        assert_eq!(a, b);
    }

    // ── BackendPriority ─────────────────────────────────────────────────

    #[test]
    fn priority_default_starts_with_cuda() {
        let p = BackendPriority::default();
        assert_eq!(p.first(), Some(BackendType::CUDA));
    }

    #[test]
    fn priority_custom_order() {
        let p = BackendPriority::new(vec![BackendType::Vulkan, BackendType::CPU]);
        assert_eq!(p.order().len(), 2);
        assert_eq!(p.first(), Some(BackendType::Vulkan));
    }

    #[test]
    fn priority_rank() {
        let p = BackendPriority::default();
        assert_eq!(p.rank(BackendType::CUDA), Some(0));
        assert!(p.rank(BackendType::CPU).unwrap() > 0);
    }

    #[test]
    fn priority_rank_missing() {
        let p = BackendPriority::new(vec![BackendType::CPU]);
        assert_eq!(p.rank(BackendType::CUDA), None);
    }

    #[test]
    fn priority_empty() {
        let p = BackendPriority::new(vec![]);
        assert_eq!(p.first(), None);
        assert!(p.order().is_empty());
    }

    // ── BackendCapability ───────────────────────────────────────────────

    fn make_cap(backend: BackendType, cu: u32, buf: u64) -> BackendCapability {
        BackendCapability {
            backend,
            supported_dtypes: vec![DType::F32, DType::F16],
            max_buffer_bytes: buf,
            shared_memory_bytes: 48 * 1024,
            max_workgroup_size: 1024,
            supports_unified_memory: false,
            compute_units: cu,
            driver_version: "1.0".into(),
            device_name: format!("{backend} device"),
        }
    }

    #[test]
    fn capability_supports_dtype() {
        let c = make_cap(BackendType::CUDA, 80, 1 << 30);
        assert!(c.supports_dtype(DType::F32));
        assert!(c.supports_dtype(DType::F16));
        assert!(!c.supports_dtype(DType::I2));
    }

    #[test]
    fn capability_throughput_score_positive() {
        let c = make_cap(BackendType::CUDA, 80, 1 << 30);
        assert!(c.throughput_score() > 0.0);
    }

    #[test]
    fn capability_throughput_score_scales_with_cu() {
        let c1 = make_cap(BackendType::CUDA, 40, 1 << 30);
        let c2 = make_cap(BackendType::CUDA, 80, 1 << 30);
        assert!(c2.throughput_score() > c1.throughput_score());
    }

    #[test]
    fn capability_throughput_score_scales_with_buffer() {
        let c1 = make_cap(BackendType::CUDA, 80, 1 << 20);
        let c2 = make_cap(BackendType::CUDA, 80, 1 << 30);
        assert!(c2.throughput_score() > c1.throughput_score());
    }

    #[test]
    fn capability_device_name() {
        let c = make_cap(BackendType::Vulkan, 1, 1024);
        assert_eq!(c.device_name, "Vulkan device");
    }

    // ── BackendProbe ────────────────────────────────────────────────────

    #[test]
    fn probe_cpu_always_available() {
        let probe = BackendProbe::default();
        let r = probe.probe_one(BackendType::CPU);
        assert!(r.available);
        assert!(r.capability.is_some());
        assert!(r.error.is_none());
    }

    #[test]
    fn probe_gpu_unavailable_in_scaffold() {
        let probe = BackendProbe::default();
        let r = probe.probe_one(BackendType::CUDA);
        assert!(!r.available);
        assert!(r.error.is_some());
    }

    #[test]
    fn probe_all_includes_cpu() {
        let mut probe = BackendProbe::default();
        probe.probe_all(&BackendPriority::default());
        let avail = probe.available();
        assert!(!avail.is_empty());
        assert!(avail.iter().any(|r| r.backend == BackendType::CPU));
    }

    #[test]
    fn probe_results_match_priority_length() {
        let prio = BackendPriority::default();
        let mut probe = BackendProbe::default();
        probe.probe_all(&prio);
        assert_eq!(probe.results().len(), prio.order().len());
    }

    #[test]
    fn probe_with_timeout() {
        let probe = BackendProbe::with_timeout(Duration::from_millis(100));
        assert_eq!(probe.timeout(), Duration::from_millis(100));
    }

    #[test]
    fn probe_cpu_capability_dtypes() {
        let probe = BackendProbe::default();
        let r = probe.probe_one(BackendType::CPU);
        let cap = r.capability.unwrap();
        assert!(cap.supports_dtype(DType::F32));
        assert!(cap.supports_dtype(DType::I2));
    }

    #[test]
    fn probe_duration_is_short() {
        let probe = BackendProbe::default();
        let r = probe.probe_one(BackendType::CPU);
        assert!(r.probe_duration < Duration::from_secs(1));
    }

    // ── WorkloadRequirements ────────────────────────────────────────────

    #[test]
    fn workload_default_prefers_gpu() {
        let w = WorkloadRequirements::default();
        assert!(w.prefer_gpu);
    }

    #[test]
    fn workload_default_requires_f32() {
        let w = WorkloadRequirements::default();
        assert!(w.required_dtypes.contains(&DType::F32));
    }

    // ── BackendScorer ───────────────────────────────────────────────────

    #[test]
    fn scorer_cpu_meets_default_requirements() {
        let scorer = BackendScorer::default();
        let cap = BackendProbe::cpu_capability();
        let reqs = WorkloadRequirements::default();
        let prio = BackendPriority::default();
        let score = scorer.score(&cap, &reqs, &prio);
        assert!(score.meets_requirements);
        assert!(score.total > 0.0);
    }

    #[test]
    fn scorer_higher_cu_scores_higher() {
        let scorer = BackendScorer::default();
        let reqs = WorkloadRequirements::default();
        let prio = BackendPriority::new(vec![BackendType::CUDA, BackendType::CPU]);
        let c1 = make_cap(BackendType::CUDA, 40, 1 << 30);
        let c2 = make_cap(BackendType::CUDA, 80, 1 << 30);
        let s1 = scorer.score(&c1, &reqs, &prio);
        let s2 = scorer.score(&c2, &reqs, &prio);
        assert!(s2.throughput >= s1.throughput);
    }

    #[test]
    fn scorer_gpu_bonus_when_preferred() {
        let scorer = BackendScorer::default();
        let reqs = WorkloadRequirements { prefer_gpu: true, ..Default::default() };
        let prio = BackendPriority::default();
        let gpu_cap = make_cap(BackendType::CUDA, 80, 1 << 30);
        let cpu_cap = BackendProbe::cpu_capability();
        let gs = scorer.score(&gpu_cap, &reqs, &prio);
        let cs = scorer.score(&cpu_cap, &reqs, &prio);
        assert!(gs.gpu_bonus > cs.gpu_bonus);
    }

    #[test]
    fn scorer_no_gpu_bonus_when_not_preferred() {
        let scorer = BackendScorer::default();
        let reqs = WorkloadRequirements { prefer_gpu: false, ..Default::default() };
        let prio = BackendPriority::default();
        let gpu_cap = make_cap(BackendType::CUDA, 80, 1 << 30);
        let s = scorer.score(&gpu_cap, &reqs, &prio);
        assert!(s.gpu_bonus.abs() < f64::EPSILON);
    }

    #[test]
    fn scorer_dtype_mismatch_lowers_score() {
        let scorer = BackendScorer::default();
        let reqs = WorkloadRequirements { required_dtypes: vec![DType::I2], ..Default::default() };
        let prio = BackendPriority::default();
        let cap = make_cap(BackendType::CUDA, 80, 1 << 30);
        let s = scorer.score(&cap, &reqs, &prio);
        assert!(!s.meets_requirements);
        assert!(s.dtype_match.abs() < f64::EPSILON);
    }

    #[test]
    fn scorer_score_all_sorted_descending() {
        let sc = BackendScorer::default();
        let reqs = WorkloadRequirements::default();
        let prio = BackendPriority::default();
        let caps = vec![
            make_cap(BackendType::CUDA, 80, 1 << 30),
            make_cap(BackendType::Vulkan, 20, 1 << 28),
        ];
        let ranked = sc.score_all(&caps, &reqs, &prio);
        assert!(ranked[0].total >= ranked[1].total);
    }

    #[test]
    fn scorer_empty_caps_returns_empty() {
        let sc = BackendScorer::default();
        let reqs = WorkloadRequirements::default();
        let prio = BackendPriority::default();
        let ranked = sc.score_all(&[], &reqs, &prio);
        assert!(ranked.is_empty());
    }

    #[test]
    fn scorer_priority_bonus_first_is_highest() {
        let scorer = BackendScorer::default();
        let reqs = WorkloadRequirements::default();
        let prio =
            BackendPriority::new(vec![BackendType::CUDA, BackendType::OpenCL, BackendType::CPU]);
        let cap_first = make_cap(BackendType::CUDA, 1, 1024);
        let cap_last = BackendProbe::cpu_capability();
        let s1 = scorer.score(&cap_first, &reqs, &prio);
        let s2 = scorer.score(&cap_last, &reqs, &prio);
        assert!(s1.priority_bonus > s2.priority_bonus);
    }

    #[test]
    fn scorer_memory_mismatch_fails_requirements() {
        let scorer = BackendScorer::default();
        let reqs = WorkloadRequirements { min_buffer_bytes: 1 << 40, ..Default::default() };
        let prio = BackendPriority::default();
        let cap = make_cap(BackendType::CUDA, 80, 1 << 30);
        let s = scorer.score(&cap, &reqs, &prio);
        assert!(!s.meets_requirements);
    }

    #[test]
    fn scorer_memory_match_exactly() {
        let scorer = BackendScorer::default();
        let reqs = WorkloadRequirements { min_buffer_bytes: 1 << 30, ..Default::default() };
        let prio = BackendPriority::default();
        let cap = make_cap(BackendType::CUDA, 80, 1 << 30);
        let s = scorer.score(&cap, &reqs, &prio);
        assert!((s.memory_match - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn scorer_shared_memory_requirement() {
        let scorer = BackendScorer::default();
        let reqs = WorkloadRequirements { min_shared_memory_bytes: 1 << 30, ..Default::default() };
        let prio = BackendPriority::default();
        let cap = make_cap(BackendType::CUDA, 80, 1 << 30);
        let s = scorer.score(&cap, &reqs, &prio);
        assert!(!s.meets_requirements);
    }

    // ── BackendSelector ─────────────────────────────────────────────────

    #[test]
    fn selector_default_selects_cpu() {
        let sel = BackendSelector::default();
        let reqs = WorkloadRequirements::default();
        let result = sel.select(&reqs).unwrap();
        assert_eq!(result.selected, BackendType::CPU);
    }

    #[test]
    fn selector_forced_backend() {
        let cfg = BackendConfig { forced_backend: Some(BackendType::Vulkan), ..Default::default() };
        let sel = BackendSelector::new(BackendPriority::default(), BackendScorer::default(), cfg);
        let reqs = WorkloadRequirements::default();
        let result = sel.select(&reqs).unwrap();
        assert_eq!(result.selected, BackendType::Vulkan);
        assert!(result.reason.contains("forced"));
    }

    #[test]
    fn selector_result_has_alternatives() {
        let sel = BackendSelector::default();
        let reqs = WorkloadRequirements::default();
        let result = sel.select(&reqs).unwrap();
        // CPU is the only available backend, so no alternatives.
        assert!(result.alternatives.is_empty());
    }

    #[test]
    fn selector_priority_accessible() {
        let sel = BackendSelector::default();
        assert!(!sel.priority().order().is_empty());
    }

    #[test]
    fn selector_config_accessible() {
        let sel = BackendSelector::default();
        assert!(sel.config().forced_backend.is_none());
    }

    #[test]
    fn selector_result_score_positive() {
        let sel = BackendSelector::default();
        let reqs = WorkloadRequirements::default();
        let result = sel.select(&reqs).unwrap();
        assert!(result.score > 0.0);
    }

    // ── BackendFallback ─────────────────────────────────────────────────

    #[test]
    fn fallback_default_contains_cpu() {
        let fb = BackendFallback::default();
        assert!(fb.chain().contains(&BackendType::CPU));
    }

    #[test]
    fn fallback_try_first_success() {
        let fb = BackendFallback::new(vec![BackendType::CUDA, BackendType::CPU], 1);
        let result = fb.try_fallback(|b| b == BackendType::CUDA);
        assert_eq!(result, Some(BackendType::CUDA));
    }

    #[test]
    fn fallback_try_skip_to_second() {
        let fb = BackendFallback::new(vec![BackendType::CUDA, BackendType::CPU], 0);
        let result = fb.try_fallback(|b| b == BackendType::CPU);
        assert_eq!(result, Some(BackendType::CPU));
    }

    #[test]
    fn fallback_try_all_fail() {
        let fb = BackendFallback::new(vec![BackendType::CUDA, BackendType::Vulkan], 1);
        let result = fb.try_fallback(|_| false);
        assert_eq!(result, None);
    }

    #[test]
    fn fallback_empty_chain() {
        let fb = BackendFallback::new(vec![], 0);
        assert!(fb.is_empty());
        assert_eq!(fb.len(), 0);
        assert_eq!(fb.try_fallback(|_| true), None);
    }

    #[test]
    fn fallback_max_retries() {
        let fb = BackendFallback::new(vec![BackendType::CUDA], 3);
        assert_eq!(fb.max_retries(), 3);
    }

    #[test]
    fn fallback_retries_exhausted() {
        let fb = BackendFallback::new(vec![BackendType::CUDA], 2);
        let mut attempts = 0u32;
        let result = fb.try_fallback(|_| {
            attempts += 1;
            false
        });
        assert_eq!(result, None);
        // 0..=max_retries = 3 attempts
        assert_eq!(attempts, 3);
    }

    // ── BackendConfig ───────────────────────────────────────────────────

    #[test]
    fn config_default_valid() {
        let c = BackendConfig::default();
        assert!(c.validate().is_ok());
    }

    #[test]
    fn config_zero_timeout_invalid() {
        let c = BackendConfig { probe_timeout: Duration::ZERO, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_forced_and_excluded_invalid() {
        let c = BackendConfig {
            forced_backend: Some(BackendType::CUDA),
            excluded_backends: vec![BackendType::CUDA],
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_extra_overrides() {
        let mut c = BackendConfig::default();
        c.extra.insert("device_id".into(), "0".into());
        assert_eq!(c.extra.get("device_id").unwrap(), "0");
    }

    #[test]
    fn config_allow_cpu_fallback_default_true() {
        let c = BackendConfig::default();
        assert!(c.allow_cpu_fallback);
    }

    // ── BackendInitializer ──────────────────────────────────────────────

    #[test]
    fn initializer_cpu_succeeds() {
        let init = BackendInitializer::default();
        let h = init.initialize(BackendType::CPU).unwrap();
        assert!(h.is_active());
        assert_eq!(h.backend, BackendType::CPU);
    }

    #[test]
    fn initializer_gpu_fails_in_scaffold() {
        let init = BackendInitializer::default();
        assert!(init.initialize(BackendType::CUDA).is_err());
    }

    #[test]
    fn initializer_handle_shutdown() {
        let init = BackendInitializer::default();
        let mut h = init.initialize(BackendType::CPU).unwrap();
        assert!(h.is_active());
        h.shutdown();
        assert!(!h.is_active());
    }

    #[test]
    fn initializer_config_accessible() {
        let init = BackendInitializer::default();
        assert!(init.config().forced_backend.is_none());
    }

    #[test]
    fn initializer_custom_config() {
        let cfg = BackendConfig { allow_cpu_fallback: false, ..Default::default() };
        let init = BackendInitializer::new(cfg);
        assert!(!init.config().allow_cpu_fallback);
    }

    #[test]
    fn initializer_handle_init_duration() {
        let init = BackendInitializer::default();
        let h = init.initialize(BackendType::CPU).unwrap();
        assert!(h.init_duration < Duration::from_secs(1));
    }

    // ── ManagerState ────────────────────────────────────────────────────

    #[test]
    fn manager_state_display() {
        assert_eq!(ManagerState::Idle.to_string(), "Idle");
        assert_eq!(ManagerState::Running.to_string(), "Running");
        assert_eq!(ManagerState::Stopped.to_string(), "Stopped");
    }

    #[test]
    fn manager_state_eq() {
        assert_eq!(ManagerState::Idle, ManagerState::Idle);
        assert_ne!(ManagerState::Idle, ManagerState::Running);
    }

    // ── BackendManager ──────────────────────────────────────────────────

    #[test]
    fn manager_default_idle() {
        let mgr = BackendManager::default();
        assert_eq!(mgr.state(), ManagerState::Idle);
        assert!(mgr.handle().is_none());
    }

    #[test]
    fn manager_start_succeeds_with_cpu() {
        let mut mgr = BackendManager::default();
        let reqs = WorkloadRequirements::default();
        let backend = mgr.start(&reqs).unwrap();
        assert_eq!(backend, BackendType::CPU);
        assert!(mgr.is_running());
    }

    #[test]
    fn manager_start_produces_events() {
        let mut mgr = BackendManager::default();
        let reqs = WorkloadRequirements::default();
        mgr.start(&reqs).unwrap();
        assert!(!mgr.events().is_empty());
    }

    #[test]
    fn manager_shutdown() {
        let mut mgr = BackendManager::default();
        let reqs = WorkloadRequirements::default();
        mgr.start(&reqs).unwrap();
        mgr.shutdown();
        assert_eq!(mgr.state(), ManagerState::Stopped);
        assert!(!mgr.handle().unwrap().is_active());
    }

    #[test]
    fn manager_cannot_start_while_running() {
        let mut mgr = BackendManager::default();
        let reqs = WorkloadRequirements::default();
        mgr.start(&reqs).unwrap();
        assert!(mgr.start(&reqs).is_err());
    }

    #[test]
    fn manager_restart_after_shutdown() {
        let mut mgr = BackendManager::default();
        let reqs = WorkloadRequirements::default();
        mgr.start(&reqs).unwrap();
        mgr.shutdown();
        let backend = mgr.start(&reqs).unwrap();
        assert_eq!(backend, BackendType::CPU);
        assert!(mgr.is_running());
    }

    #[test]
    fn manager_handle_present_when_running() {
        let mut mgr = BackendManager::default();
        let reqs = WorkloadRequirements::default();
        mgr.start(&reqs).unwrap();
        assert!(mgr.handle().is_some());
        assert!(mgr.handle().unwrap().is_active());
    }

    #[test]
    fn manager_uptime_increases() {
        let mut mgr = BackendManager::default();
        let reqs = WorkloadRequirements::default();
        mgr.start(&reqs).unwrap();
        let u1 = mgr.uptime();
        std::thread::sleep(Duration::from_millis(5));
        let u2 = mgr.uptime();
        assert!(u2 >= u1);
    }

    #[test]
    fn manager_uptime_zero_before_start() {
        let mgr = BackendManager::default();
        assert_eq!(mgr.uptime(), Duration::ZERO);
    }

    #[test]
    fn manager_events_contain_probe_and_init() {
        let mut mgr = BackendManager::default();
        let reqs = WorkloadRequirements::default();
        mgr.start(&reqs).unwrap();
        let kinds: Vec<_> = mgr.events().iter().map(|e| e.kind).collect();
        assert!(kinds.contains(&ManagerEventKind::ProbeStarted));
        assert!(kinds.contains(&ManagerEventKind::ProbeCompleted));
        assert!(kinds.contains(&ManagerEventKind::InitCompleted));
    }

    #[test]
    fn manager_events_contain_shutdown() {
        let mut mgr = BackendManager::default();
        let reqs = WorkloadRequirements::default();
        mgr.start(&reqs).unwrap();
        mgr.shutdown();
        assert!(mgr.events().iter().any(|e| e.kind == ManagerEventKind::Shutdown));
    }

    #[test]
    fn manager_custom_selector() {
        let prio = BackendPriority::new(vec![BackendType::CPU]);
        let scorer = BackendScorer::default();
        let cfg = BackendConfig::default();
        let sel = BackendSelector::new(prio, scorer, cfg);
        let init = BackendInitializer::default();
        let mut mgr = BackendManager::new(sel, init);
        let reqs = WorkloadRequirements::default();
        let backend = mgr.start(&reqs).unwrap();
        assert_eq!(backend, BackendType::CPU);
    }

    // ── SelectionResult ─────────────────────────────────────────────────

    #[test]
    fn selection_result_fields() {
        let r = SelectionResult {
            selected: BackendType::CPU,
            score: 42.0,
            reason: "test".into(),
            alternatives: vec![(BackendType::CUDA, 10.0)],
        };
        assert_eq!(r.selected, BackendType::CPU);
        assert!((r.score - 42.0).abs() < f64::EPSILON);
        assert_eq!(r.alternatives.len(), 1);
    }

    // ── ProbeResult ─────────────────────────────────────────────────────

    #[test]
    fn probe_result_fields() {
        let r = ProbeResult {
            backend: BackendType::Metal,
            available: false,
            capability: None,
            probe_duration: Duration::from_millis(1),
            error: Some("nope".into()),
        };
        assert_eq!(r.backend, BackendType::Metal);
        assert!(!r.available);
        assert!(r.capability.is_none());
    }

    // ── BackendScore ────────────────────────────────────────────────────

    #[test]
    fn backend_score_fields() {
        let s = BackendScore {
            backend: BackendType::ROCm,
            total: 55.0,
            dtype_match: 1.0,
            memory_match: 1.0,
            throughput: 0.5,
            priority_bonus: 0.8,
            gpu_bonus: 1.0,
            meets_requirements: true,
        };
        assert_eq!(s.backend, BackendType::ROCm);
        assert!(s.meets_requirements);
    }

    // ── ManagerEvent ────────────────────────────────────────────────────

    #[test]
    fn manager_event_kind_eq() {
        assert_eq!(ManagerEventKind::ProbeStarted, ManagerEventKind::ProbeStarted,);
        assert_ne!(ManagerEventKind::ProbeStarted, ManagerEventKind::Shutdown,);
    }

    // ── DType ───────────────────────────────────────────────────────────

    #[test]
    fn dtype_eq_and_hash() {
        let mut set = std::collections::HashSet::new();
        set.insert(DType::F32);
        set.insert(DType::F32);
        assert_eq!(set.len(), 1);
    }

    // ── Integration / edge cases ────────────────────────────────────────

    #[test]
    fn full_lifecycle() {
        let mut mgr = BackendManager::default();
        let reqs = WorkloadRequirements::default();
        assert_eq!(mgr.state(), ManagerState::Idle);
        let b = mgr.start(&reqs).unwrap();
        assert_eq!(b, BackendType::CPU);
        assert_eq!(mgr.state(), ManagerState::Running);
        mgr.shutdown();
        assert_eq!(mgr.state(), ManagerState::Stopped);
    }

    #[test]
    fn forced_backend_bypasses_scoring() {
        let cfg = BackendConfig { forced_backend: Some(BackendType::WebGPU), ..Default::default() };
        let sel = BackendSelector::new(BackendPriority::default(), BackendScorer::default(), cfg);
        let reqs = WorkloadRequirements::default();
        let result = sel.select(&reqs).unwrap();
        assert_eq!(result.selected, BackendType::WebGPU);
        assert!(result.alternatives.is_empty());
    }

    #[test]
    fn scorer_weights_all_positive() {
        let s = BackendScorer::default();
        assert!(s.dtype_weight > 0.0);
        assert!(s.memory_weight > 0.0);
        assert!(s.throughput_weight > 0.0);
        assert!(s.priority_weight > 0.0);
        assert!(s.gpu_weight > 0.0);
    }

    #[test]
    fn backend_type_all_no_duplicates() {
        let all = BackendType::all();
        let set: std::collections::HashSet<_> = all.iter().collect();
        assert_eq!(set.len(), all.len());
    }

    #[test]
    fn fallback_chain_custom() {
        let fb =
            BackendFallback::new(vec![BackendType::ROCm, BackendType::Vulkan, BackendType::CPU], 1);
        assert_eq!(fb.len(), 3);
        assert!(!fb.is_empty());
    }

    #[test]
    fn manager_fallback_to_cpu() {
        // Force CUDA but CUDA init fails → should fallback to CPU.
        let cfg = BackendConfig {
            forced_backend: Some(BackendType::CUDA),
            allow_cpu_fallback: true,
            ..Default::default()
        };
        let sel = BackendSelector::new(BackendPriority::default(), BackendScorer::default(), cfg);
        let init = BackendInitializer::default();
        let mut mgr = BackendManager::new(sel, init);
        let reqs = WorkloadRequirements::default();
        // select returns CUDA (forced) but init fails → fallback chain
        let backend = mgr.start(&reqs).unwrap();
        assert_eq!(backend, BackendType::CPU);
    }

    #[test]
    fn manager_events_have_messages() {
        let mut mgr = BackendManager::default();
        let reqs = WorkloadRequirements::default();
        mgr.start(&reqs).unwrap();
        for event in mgr.events() {
            assert!(!event.message.is_empty());
        }
    }

    #[test]
    fn probe_results_empty_before_probe() {
        let probe = BackendProbe::default();
        assert!(probe.results().is_empty());
        assert!(probe.available().is_empty());
    }

    #[test]
    fn config_excluded_backends_default_empty() {
        let c = BackendConfig::default();
        assert!(c.excluded_backends.is_empty());
    }

    #[test]
    fn capability_unified_memory_flag() {
        let cap = BackendProbe::cpu_capability();
        assert!(cap.supports_unified_memory);
    }

    #[test]
    fn capability_max_workgroup_size() {
        let cap = make_cap(BackendType::CUDA, 80, 1 << 30);
        assert_eq!(cap.max_workgroup_size, 1024);
    }

    #[test]
    fn workload_custom_requirements() {
        let reqs = WorkloadRequirements {
            required_dtypes: vec![DType::F16, DType::BF16],
            min_buffer_bytes: 1 << 32,
            min_shared_memory_bytes: 48 * 1024,
            prefer_gpu: true,
        };
        assert_eq!(reqs.required_dtypes.len(), 2);
        assert_eq!(reqs.min_buffer_bytes, 1 << 32);
    }
}
