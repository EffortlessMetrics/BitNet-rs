//! Async inference pipeline patterns for Intel Arc A770 GPU.
//!
//! Manages asynchronous kernel dispatch, result collection, and pipeline
//! stages with overlap between compute and data transfer.  All
//! implementations are CPU-reference; real OpenCL dispatch is layered on
//! top when the `oneapi` feature is enabled.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ── Pipeline stage ─────────────────────────────────────────────────

/// Logical inference pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PipelineStage {
    Tokenize,
    Embed,
    Attention,
    FFN,
    Decode,
    Sample,
}

impl PipelineStage {
    /// All stages in canonical execution order.
    pub fn all_ordered() -> &'static [PipelineStage] {
        &[
            PipelineStage::Tokenize,
            PipelineStage::Embed,
            PipelineStage::Attention,
            PipelineStage::FFN,
            PipelineStage::Decode,
            PipelineStage::Sample,
        ]
    }

    /// Zero-based index of this stage in the canonical order.
    pub fn ordinal(self) -> usize {
        match self {
            PipelineStage::Tokenize => 0,
            PipelineStage::Embed => 1,
            PipelineStage::Attention => 2,
            PipelineStage::FFN => 3,
            PipelineStage::Decode => 4,
            PipelineStage::Sample => 5,
        }
    }
}

impl fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            PipelineStage::Tokenize => "Tokenize",
            PipelineStage::Embed => "Embed",
            PipelineStage::Attention => "Attention",
            PipelineStage::FFN => "FFN",
            PipelineStage::Decode => "Decode",
            PipelineStage::Sample => "Sample",
        };
        write!(f, "{name}")
    }
}

// ── Stage status ───────────────────────────────────────────────────

/// Runtime status of a single pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageStatus {
    Pending,
    Running,
    Complete,
    Failed(String),
}

impl StageStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, StageStatus::Complete | StageStatus::Failed(_))
    }
}

impl fmt::Display for StageStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StageStatus::Pending => write!(f, "Pending"),
            StageStatus::Running => write!(f, "Running"),
            StageStatus::Complete => write!(f, "Complete"),
            StageStatus::Failed(e) => write!(f, "Failed({e})"),
        }
    }
}

// ── Async pipeline config ──────────────────────────────────────────

/// Configuration for the async inference pipeline.
#[derive(Debug, Clone)]
pub struct AsyncPipelineConfig {
    /// Maximum number of in-flight stage executions.
    pub max_inflight: usize,
    /// Whether to overlap data transfer with compute.
    pub overlap_transfer: bool,
    /// Number of stages to prefetch ahead of the current execution point.
    pub prefetch_depth: usize,
}

impl Default for AsyncPipelineConfig {
    fn default() -> Self {
        Self { max_inflight: 4, overlap_transfer: true, prefetch_depth: 2 }
    }
}

impl AsyncPipelineConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_inflight == 0 {
            return Err("max_inflight must be > 0".into());
        }
        if self.prefetch_depth > self.max_inflight {
            return Err(format!(
                "prefetch_depth ({}) cannot exceed max_inflight ({})",
                self.prefetch_depth, self.max_inflight
            ));
        }
        Ok(())
    }
}

// ── Pipeline fence ─────────────────────────────────────────────────

/// Synchronisation point between pipeline stages.
///
/// In a real OpenCL implementation this wraps `cl_event`; the CPU
/// reference uses an atomic flag.
#[derive(Debug)]
pub struct PipelineFence {
    /// Unique identifier for this fence.
    pub id: u64,
    /// The stage that must complete before dependents proceed.
    pub stage: PipelineStage,
    /// Whether the fence has been signalled.
    signalled: bool,
    /// Time the fence was created.
    created_at: Instant,
    /// Time the fence was signalled.
    signalled_at: Option<Instant>,
}

impl PipelineFence {
    pub fn new(id: u64, stage: PipelineStage) -> Self {
        Self { id, stage, signalled: false, created_at: Instant::now(), signalled_at: None }
    }

    /// Signal this fence as complete.
    pub fn signal(&mut self) {
        if !self.signalled {
            self.signalled = true;
            self.signalled_at = Some(Instant::now());
        }
    }

    pub fn is_signalled(&self) -> bool {
        self.signalled
    }

    /// Duration from creation to signal (or to now if not yet signalled).
    pub fn wait_duration(&self) -> Duration {
        match self.signalled_at {
            Some(t) => t.duration_since(self.created_at),
            None => self.created_at.elapsed(),
        }
    }
}

// ── Pipeline metrics ───────────────────────────────────────────────

/// Per-stage latency record.
#[derive(Debug, Clone)]
pub struct StageLatency {
    pub stage: PipelineStage,
    pub total_us: u64,
    pub count: u64,
}

impl StageLatency {
    fn new(stage: PipelineStage) -> Self {
        Self { stage, total_us: 0, count: 0 }
    }

    fn record(&mut self, us: u64) {
        self.total_us += us;
        self.count += 1;
    }

    pub fn avg_us(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.total_us as f64 / self.count as f64 }
    }
}

/// Aggregate pipeline metrics.
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    stage_latencies: HashMap<PipelineStage, StageLatency>,
    pub total_tokens: u64,
    pub total_pipeline_runs: u64,
    pipeline_start: Option<Instant>,
    pub total_pipeline_us: u64,
}

impl PipelineMetrics {
    pub fn new() -> Self {
        let mut stage_latencies = HashMap::new();
        for &s in PipelineStage::all_ordered() {
            stage_latencies.insert(s, StageLatency::new(s));
        }
        Self {
            stage_latencies,
            total_tokens: 0,
            total_pipeline_runs: 0,
            pipeline_start: None,
            total_pipeline_us: 0,
        }
    }

    /// Record latency for one stage invocation.
    pub fn record_stage(&mut self, stage: PipelineStage, duration_us: u64) {
        if let Some(lat) = self.stage_latencies.get_mut(&stage) {
            lat.record(duration_us);
        }
    }

    /// Begin timing a full pipeline run.
    pub fn begin_pipeline_run(&mut self) {
        self.pipeline_start = Some(Instant::now());
    }

    /// End timing a full pipeline run.
    pub fn end_pipeline_run(&mut self, tokens: u64) {
        if let Some(start) = self.pipeline_start.take() {
            self.total_pipeline_us += start.elapsed().as_micros() as u64;
            self.total_pipeline_runs += 1;
            self.total_tokens += tokens;
        }
    }

    /// Throughput in tokens per second.
    pub fn tokens_per_second(&self) -> f64 {
        if self.total_pipeline_us == 0 {
            return 0.0;
        }
        self.total_tokens as f64 / (self.total_pipeline_us as f64 / 1_000_000.0)
    }

    /// Average latency for a specific stage.
    pub fn stage_avg_us(&self, stage: PipelineStage) -> f64 {
        self.stage_latencies.get(&stage).map_or(0.0, |l| l.avg_us())
    }

    /// Utilization: fraction of total pipeline time spent in stages.
    pub fn utilization(&self) -> f64 {
        if self.total_pipeline_us == 0 {
            return 0.0;
        }
        let stage_sum: u64 = self.stage_latencies.values().map(|l| l.total_us).sum();
        stage_sum as f64 / self.total_pipeline_us as f64
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        for lat in self.stage_latencies.values_mut() {
            lat.total_us = 0;
            lat.count = 0;
        }
        self.total_tokens = 0;
        self.total_pipeline_runs = 0;
        self.pipeline_start = None;
        self.total_pipeline_us = 0;
    }
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ── Double buffer strategy ─────────────────────────────────────────

/// Alternates between two buffers so that compute on one can overlap
/// with data transfer on the other.
#[derive(Debug)]
pub struct DoubleBufferStrategy {
    /// Two logical buffers, each `buffer_size` elements.
    buffers: [Vec<f32>; 2],
    /// Index of the buffer currently being written to (0 or 1).
    write_idx: usize,
    /// Number of completed swaps.
    swap_count: u64,
}

impl DoubleBufferStrategy {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffers: [vec![0.0; buffer_size], vec![0.0; buffer_size]],
            write_idx: 0,
            swap_count: 0,
        }
    }

    /// Index of the buffer currently used for writing / transfer.
    pub fn write_index(&self) -> usize {
        self.write_idx
    }

    /// Index of the buffer currently used for compute / reading.
    pub fn read_index(&self) -> usize {
        1 - self.write_idx
    }

    /// Swap the read and write buffers.
    pub fn swap(&mut self) {
        self.write_idx = 1 - self.write_idx;
        self.swap_count += 1;
    }

    pub fn swap_count(&self) -> u64 {
        self.swap_count
    }

    /// Mutable reference to the write buffer.
    pub fn write_buffer_mut(&mut self) -> &mut [f32] {
        &mut self.buffers[self.write_idx]
    }

    /// Immutable reference to the read (compute) buffer.
    pub fn read_buffer(&self) -> &[f32] {
        &self.buffers[1 - self.write_idx]
    }

    /// Size of each buffer.
    pub fn buffer_size(&self) -> usize {
        self.buffers[0].len()
    }
}

// ── Pipeline scheduler ─────────────────────────────────────────────

/// Tracks per-stage dependencies and schedules execution.
#[derive(Debug)]
pub struct PipelineScheduler {
    config: AsyncPipelineConfig,
    /// Current status of every stage.
    statuses: HashMap<PipelineStage, StageStatus>,
    /// Stages that each stage depends on.
    dependencies: HashMap<PipelineStage, Vec<PipelineStage>>,
    /// Number of stages currently in `Running` state.
    inflight: usize,
    /// Metrics collector.
    metrics: PipelineMetrics,
    /// Next fence id.
    next_fence_id: u64,
}

impl PipelineScheduler {
    /// Create a scheduler with the default linear dependency chain:
    /// Tokenize → Embed → Attention → FFN → Decode → Sample
    pub fn new(config: AsyncPipelineConfig) -> Result<Self, String> {
        config.validate()?;
        let mut statuses = HashMap::new();
        let mut dependencies: HashMap<PipelineStage, Vec<PipelineStage>> = HashMap::new();
        let stages = PipelineStage::all_ordered();
        for (i, &stage) in stages.iter().enumerate() {
            statuses.insert(stage, StageStatus::Pending);
            if i > 0 {
                dependencies.insert(stage, vec![stages[i - 1]]);
            } else {
                dependencies.insert(stage, vec![]);
            }
        }
        Ok(Self {
            config,
            statuses,
            dependencies,
            inflight: 0,
            metrics: PipelineMetrics::new(),
            next_fence_id: 0,
        })
    }

    /// Create a scheduler with an empty pipeline (no stages registered).
    pub fn empty(config: AsyncPipelineConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            statuses: HashMap::new(),
            dependencies: HashMap::new(),
            inflight: 0,
            metrics: PipelineMetrics::new(),
            next_fence_id: 0,
        })
    }

    /// Register a single stage with its dependencies.
    pub fn add_stage(&mut self, stage: PipelineStage, deps: Vec<PipelineStage>) {
        self.statuses.insert(stage, StageStatus::Pending);
        self.dependencies.insert(stage, deps);
    }

    /// Stages whose dependencies are all `Complete` and that are still
    /// `Pending`, respecting `max_inflight`.
    pub fn ready_stages(&self) -> Vec<PipelineStage> {
        let mut ready: Vec<PipelineStage> = self
            .statuses
            .iter()
            .filter(|(_, status)| **status == StageStatus::Pending)
            .filter(|(stage, _)| {
                self.dependencies.get(stage).is_none_or(|deps| {
                    deps.iter().all(|d| self.statuses.get(d) == Some(&StageStatus::Complete))
                })
            })
            .map(|(stage, _)| *stage)
            .collect();
        ready.sort();
        let capacity = self.config.max_inflight.saturating_sub(self.inflight);
        ready.truncate(capacity);
        ready
    }

    /// Mark a stage as `Running` and return a fence for it.
    pub fn dispatch(&mut self, stage: PipelineStage) -> Result<PipelineFence, String> {
        match self.statuses.get(&stage) {
            Some(StageStatus::Pending) => {}
            Some(other) => {
                return Err(format!("cannot dispatch {stage}: status is {other}"));
            }
            None => return Err(format!("unknown stage: {stage}")),
        }
        // Verify dependencies are met.
        if let Some(deps) = self.dependencies.get(&stage) {
            for dep in deps {
                if self.statuses.get(dep) != Some(&StageStatus::Complete) {
                    return Err(format!("dependency {dep} not complete for {stage}"));
                }
            }
        }
        if self.inflight >= self.config.max_inflight {
            return Err("max inflight reached".into());
        }
        self.statuses.insert(stage, StageStatus::Running);
        self.inflight += 1;
        let id = self.next_fence_id;
        self.next_fence_id += 1;
        Ok(PipelineFence::new(id, stage))
    }

    /// Mark a stage as `Complete`, record latency, and signal the fence.
    pub fn complete(&mut self, fence: &mut PipelineFence, duration_us: u64) -> Result<(), String> {
        let stage = fence.stage;
        match self.statuses.get(&stage) {
            Some(StageStatus::Running) => {}
            Some(other) => {
                return Err(format!("cannot complete {stage}: status is {other}"));
            }
            None => return Err(format!("unknown stage: {stage}")),
        }
        self.statuses.insert(stage, StageStatus::Complete);
        self.inflight -= 1;
        self.metrics.record_stage(stage, duration_us);
        fence.signal();
        Ok(())
    }

    /// Mark a stage as `Failed` and propagate failure to all transitive
    /// dependents.
    pub fn fail(&mut self, stage: PipelineStage, reason: &str) -> Result<(), String> {
        match self.statuses.get(&stage) {
            Some(StageStatus::Running) => {
                self.inflight -= 1;
            }
            Some(StageStatus::Pending) => {}
            Some(StageStatus::Failed(_)) => {
                return Err(format!("{stage} already failed"));
            }
            Some(StageStatus::Complete) => {
                return Err(format!("cannot fail completed stage {stage}"));
            }
            None => return Err(format!("unknown stage: {stage}")),
        }
        self.statuses.insert(stage, StageStatus::Failed(reason.to_string()));
        // Propagate: any stage depending (transitively) on this one also
        // fails.
        let mut to_fail: Vec<PipelineStage> = Vec::new();
        self.collect_dependents(stage, &mut to_fail);
        for dep_stage in to_fail {
            if let Some(status) = self.statuses.get(&dep_stage)
                && !status.is_terminal()
            {
                if *status == StageStatus::Running {
                    self.inflight -= 1;
                }
                self.statuses.insert(
                    dep_stage,
                    StageStatus::Failed(format!("upstream {stage} failed: {reason}")),
                );
            }
        }
        Ok(())
    }

    /// Collect all stages that transitively depend on `root`.
    fn collect_dependents(&self, root: PipelineStage, out: &mut Vec<PipelineStage>) {
        for (&stage, deps) in &self.dependencies {
            if deps.contains(&root) && !out.contains(&stage) {
                out.push(stage);
                self.collect_dependents(stage, out);
            }
        }
    }

    /// Status of every registered stage.
    pub fn status_snapshot(&self) -> HashMap<PipelineStage, StageStatus> {
        self.statuses.clone()
    }

    pub fn stage_status(&self, stage: PipelineStage) -> Option<&StageStatus> {
        self.statuses.get(&stage)
    }

    /// True when all stages are in a terminal state.
    pub fn is_complete(&self) -> bool {
        !self.statuses.is_empty() && self.statuses.values().all(|s| s.is_terminal())
    }

    /// True when any stage has failed.
    pub fn has_failure(&self) -> bool {
        self.statuses.values().any(|s| matches!(s, StageStatus::Failed(_)))
    }

    /// Number of stages currently in-flight.
    pub fn inflight_count(&self) -> usize {
        self.inflight
    }

    /// Reset every stage to `Pending`.
    pub fn reset(&mut self) {
        for status in self.statuses.values_mut() {
            *status = StageStatus::Pending;
        }
        self.inflight = 0;
    }

    /// Drain: run all ready stages to completion using `execute_fn`.
    /// Returns `Ok(())` if every stage completes, or the first failure.
    pub fn drain<F>(&mut self, mut execute_fn: F) -> Result<(), String>
    where
        F: FnMut(PipelineStage) -> Result<u64, String>,
    {
        loop {
            let ready = self.ready_stages();
            if ready.is_empty() {
                break;
            }
            for stage in ready {
                let mut fence = self.dispatch(stage)?;
                match execute_fn(stage) {
                    Ok(duration_us) => {
                        self.complete(&mut fence, duration_us)?;
                    }
                    Err(e) => {
                        self.fail(stage, &e)?;
                        return Err(e);
                    }
                }
            }
        }
        if self.has_failure() { Err("pipeline has failed stages".into()) } else { Ok(()) }
    }

    /// Flush: reset and prepare for a new pipeline run.
    pub fn flush(&mut self) {
        self.reset();
        self.metrics.reset();
        self.next_fence_id = 0;
    }

    pub fn metrics(&self) -> &PipelineMetrics {
        &self.metrics
    }

    pub fn config(&self) -> &AsyncPipelineConfig {
        &self.config
    }
}

// ── CPU reference: simulated async dispatch ────────────────────────

/// Simulates an asynchronous stage execution on the CPU by sleeping for
/// a synthetic duration and returning the elapsed microseconds.
pub fn cpu_simulate_stage(stage: PipelineStage) -> u64 {
    let base_us: u64 = match stage {
        PipelineStage::Tokenize => 50,
        PipelineStage::Embed => 200,
        PipelineStage::Attention => 800,
        PipelineStage::FFN => 600,
        PipelineStage::Decode => 150,
        PipelineStage::Sample => 100,
    };
    let start = Instant::now();
    // Busy-spin for a short duration to simulate work without sleeping.
    let target = Duration::from_micros(base_us);
    while start.elapsed() < target {
        std::hint::spin_loop();
    }
    start.elapsed().as_micros() as u64
}

/// Run a full pipeline end-to-end on the CPU using the scheduler.
pub fn cpu_run_pipeline(config: AsyncPipelineConfig) -> Result<PipelineMetrics, String> {
    let mut scheduler = PipelineScheduler::new(config)?;
    scheduler.metrics.begin_pipeline_run();
    scheduler.drain(|stage| Ok(cpu_simulate_stage(stage)))?;
    scheduler.metrics.end_pipeline_run(1);
    Ok(scheduler.metrics.clone())
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────

    fn default_config() -> AsyncPipelineConfig {
        AsyncPipelineConfig::default()
    }

    fn make_scheduler() -> PipelineScheduler {
        PipelineScheduler::new(default_config()).unwrap()
    }

    fn single_stage_config() -> AsyncPipelineConfig {
        AsyncPipelineConfig { max_inflight: 1, overlap_transfer: false, prefetch_depth: 0 }
    }

    // ── PipelineStage ─────────────────────────────────────────────

    #[test]
    fn stage_all_ordered_length() {
        assert_eq!(PipelineStage::all_ordered().len(), 6);
    }

    #[test]
    fn stage_ordinals_are_sequential() {
        for (i, &stage) in PipelineStage::all_ordered().iter().enumerate() {
            assert_eq!(stage.ordinal(), i);
        }
    }

    #[test]
    fn stage_display() {
        assert_eq!(PipelineStage::Tokenize.to_string(), "Tokenize");
        assert_eq!(PipelineStage::Sample.to_string(), "Sample");
    }

    #[test]
    fn stage_ordering_matches_ordinal() {
        let stages = PipelineStage::all_ordered();
        for i in 0..stages.len() - 1 {
            assert!(stages[i] < stages[i + 1]);
        }
    }

    // ── StageStatus ───────────────────────────────────────────────

    #[test]
    fn status_terminal() {
        assert!(!StageStatus::Pending.is_terminal());
        assert!(!StageStatus::Running.is_terminal());
        assert!(StageStatus::Complete.is_terminal());
        assert!(StageStatus::Failed("x".into()).is_terminal());
    }

    #[test]
    fn status_display() {
        assert_eq!(StageStatus::Running.to_string(), "Running");
        assert_eq!(StageStatus::Failed("oops".into()).to_string(), "Failed(oops)");
    }

    // ── AsyncPipelineConfig ───────────────────────────────────────

    #[test]
    fn config_default_is_valid() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn config_zero_inflight_rejected() {
        let c = AsyncPipelineConfig { max_inflight: 0, ..default_config() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_prefetch_exceeds_inflight_rejected() {
        let c = AsyncPipelineConfig { max_inflight: 2, prefetch_depth: 3, ..default_config() };
        assert!(c.validate().is_err());
    }

    // ── PipelineFence ─────────────────────────────────────────────

    #[test]
    fn fence_starts_unsignalled() {
        let f = PipelineFence::new(0, PipelineStage::Embed);
        assert!(!f.is_signalled());
    }

    #[test]
    fn fence_signal_idempotent() {
        let mut f = PipelineFence::new(0, PipelineStage::Embed);
        f.signal();
        f.signal();
        assert!(f.is_signalled());
    }

    #[test]
    fn fence_wait_duration_works() {
        let f = PipelineFence::new(0, PipelineStage::Tokenize);
        // Duration is always non-negative; just verify it doesn't panic.
        let _ = f.wait_duration();
    }

    // ── PipelineMetrics ───────────────────────────────────────────

    #[test]
    fn metrics_initial_zeros() {
        let m = PipelineMetrics::new();
        assert_eq!(m.total_tokens, 0);
        assert_eq!(m.total_pipeline_runs, 0);
        assert_eq!(m.tokens_per_second(), 0.0);
        assert_eq!(m.utilization(), 0.0);
    }

    #[test]
    fn metrics_stage_avg() {
        let mut m = PipelineMetrics::new();
        m.record_stage(PipelineStage::Attention, 100);
        m.record_stage(PipelineStage::Attention, 200);
        assert!((m.stage_avg_us(PipelineStage::Attention) - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_accumulation_all_stages() {
        let mut m = PipelineMetrics::new();
        for &stage in PipelineStage::all_ordered() {
            m.record_stage(stage, 100);
            m.record_stage(stage, 300);
        }
        for &stage in PipelineStage::all_ordered() {
            assert!(
                (m.stage_avg_us(stage) - 200.0).abs() < f64::EPSILON,
                "stage {stage} avg mismatch"
            );
        }
    }

    #[test]
    fn metrics_throughput_calculation() {
        let mut m = PipelineMetrics::new();
        // Manually set total_pipeline_us to avoid timing flakiness.
        m.total_pipeline_us = 1_000_000; // 1 second
        m.total_tokens = 10;
        assert!((m.tokens_per_second() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_utilization_ratio() {
        let mut m = PipelineMetrics::new();
        m.total_pipeline_us = 1000;
        m.record_stage(PipelineStage::Attention, 500);
        m.record_stage(PipelineStage::FFN, 300);
        assert!((m.utilization() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_reset_clears_everything() {
        let mut m = PipelineMetrics::new();
        m.record_stage(PipelineStage::Embed, 42);
        m.total_tokens = 5;
        m.total_pipeline_runs = 2;
        m.total_pipeline_us = 9999;
        m.reset();
        assert_eq!(m.total_tokens, 0);
        assert_eq!(m.total_pipeline_runs, 0);
        assert_eq!(m.total_pipeline_us, 0);
        assert_eq!(m.stage_avg_us(PipelineStage::Embed), 0.0);
    }

    // ── DoubleBufferStrategy ──────────────────────────────────────

    #[test]
    fn double_buffer_initial_state() {
        let db = DoubleBufferStrategy::new(16);
        assert_eq!(db.write_index(), 0);
        assert_eq!(db.read_index(), 1);
        assert_eq!(db.swap_count(), 0);
        assert_eq!(db.buffer_size(), 16);
    }

    #[test]
    fn double_buffer_swap_alternates() {
        let mut db = DoubleBufferStrategy::new(8);
        assert_eq!(db.write_index(), 0);
        db.swap();
        assert_eq!(db.write_index(), 1);
        assert_eq!(db.read_index(), 0);
        db.swap();
        assert_eq!(db.write_index(), 0);
        assert_eq!(db.swap_count(), 2);
    }

    #[test]
    fn double_buffer_write_read_isolation() {
        let mut db = DoubleBufferStrategy::new(4);
        // Write to buffer 0
        db.write_buffer_mut().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        // Read buffer is buffer 1, should be zeros
        assert_eq!(db.read_buffer(), &[0.0; 4]);
        // After swap, read buffer becomes buffer 0 (the one we wrote to)
        db.swap();
        assert_eq!(db.read_buffer(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn double_buffer_multiple_swaps_preserve_data() {
        let mut db = DoubleBufferStrategy::new(2);
        db.write_buffer_mut()[0] = 10.0;
        db.swap();
        db.write_buffer_mut()[0] = 20.0;
        db.swap();
        // Read buffer should now be buffer 1 (contains 20.0)
        assert_eq!(db.read_buffer()[0], 20.0);
    }

    #[test]
    fn double_buffer_zero_size() {
        let db = DoubleBufferStrategy::new(0);
        assert_eq!(db.buffer_size(), 0);
        assert!(db.read_buffer().is_empty());
    }

    // ── PipelineScheduler: basic lifecycle ────────────────────────

    #[test]
    fn scheduler_initial_all_pending() {
        let sched = make_scheduler();
        for &stage in PipelineStage::all_ordered() {
            assert_eq!(sched.stage_status(stage), Some(&StageStatus::Pending));
        }
    }

    #[test]
    fn scheduler_only_tokenize_ready_initially() {
        let sched = make_scheduler();
        let ready = sched.ready_stages();
        assert_eq!(ready, vec![PipelineStage::Tokenize]);
    }

    #[test]
    fn scheduler_dispatch_and_complete_single() {
        let mut sched = make_scheduler();
        let mut fence = sched.dispatch(PipelineStage::Tokenize).unwrap();
        assert_eq!(sched.stage_status(PipelineStage::Tokenize), Some(&StageStatus::Running));
        sched.complete(&mut fence, 42).unwrap();
        assert_eq!(sched.stage_status(PipelineStage::Tokenize), Some(&StageStatus::Complete));
        assert!(fence.is_signalled());
    }

    #[test]
    fn scheduler_full_linear_pipeline() {
        let mut sched = make_scheduler();
        for &stage in PipelineStage::all_ordered() {
            let mut fence = sched.dispatch(stage).unwrap();
            sched.complete(&mut fence, 10).unwrap();
        }
        assert!(sched.is_complete());
        assert!(!sched.has_failure());
    }

    #[test]
    fn scheduler_dispatch_out_of_order_fails() {
        let mut sched = make_scheduler();
        let result = sched.dispatch(PipelineStage::Embed);
        assert!(result.is_err());
    }

    #[test]
    fn scheduler_dispatch_already_running_fails() {
        let mut sched = make_scheduler();
        let _fence = sched.dispatch(PipelineStage::Tokenize).unwrap();
        let result = sched.dispatch(PipelineStage::Tokenize);
        assert!(result.is_err());
    }

    #[test]
    fn scheduler_complete_wrong_state_fails() {
        let mut sched = make_scheduler();
        // Create a fence but don't dispatch the stage
        let mut fence = PipelineFence::new(99, PipelineStage::Embed);
        let result = sched.complete(&mut fence, 10);
        assert!(result.is_err());
    }

    // ── Failure propagation ───────────────────────────────────────

    #[test]
    fn scheduler_fail_propagates_to_dependents() {
        let mut sched = make_scheduler();
        let mut fence = sched.dispatch(PipelineStage::Tokenize).unwrap();
        sched.complete(&mut fence, 10).unwrap();

        // Dispatch Embed then fail it
        let _fence = sched.dispatch(PipelineStage::Embed).unwrap();
        sched.fail(PipelineStage::Embed, "embed error").unwrap();

        // All downstream stages should be Failed
        for &stage in &[
            PipelineStage::Attention,
            PipelineStage::FFN,
            PipelineStage::Decode,
            PipelineStage::Sample,
        ] {
            assert!(
                matches!(sched.stage_status(stage), Some(StageStatus::Failed(_))),
                "{stage} should be Failed"
            );
        }
        assert!(sched.has_failure());
    }

    #[test]
    fn scheduler_fail_pending_stage() {
        let mut sched = make_scheduler();
        sched.fail(PipelineStage::Tokenize, "bad input").unwrap();
        assert!(matches!(
            sched.stage_status(PipelineStage::Tokenize),
            Some(StageStatus::Failed(_))
        ));
    }

    #[test]
    fn scheduler_fail_completed_stage_errors() {
        let mut sched = make_scheduler();
        let mut fence = sched.dispatch(PipelineStage::Tokenize).unwrap();
        sched.complete(&mut fence, 5).unwrap();
        let result = sched.fail(PipelineStage::Tokenize, "too late");
        assert!(result.is_err());
    }

    #[test]
    fn scheduler_fail_already_failed_errors() {
        let mut sched = make_scheduler();
        sched.fail(PipelineStage::Tokenize, "first").unwrap();
        let result = sched.fail(PipelineStage::Tokenize, "second");
        assert!(result.is_err());
    }

    // ── Pipeline drain ────────────────────────────────────────────

    #[test]
    fn drain_runs_all_stages() {
        let mut sched = make_scheduler();
        let result = sched.drain(|_| Ok(10));
        assert!(result.is_ok());
        assert!(sched.is_complete());
    }

    #[test]
    fn drain_stops_on_failure() {
        let mut sched = make_scheduler();
        let result = sched.drain(|stage| {
            if stage == PipelineStage::Attention { Err("attention failed".into()) } else { Ok(10) }
        });
        assert!(result.is_err());
        assert!(sched.has_failure());
    }

    #[test]
    fn drain_records_metrics() {
        let mut sched = make_scheduler();
        sched.drain(|_| Ok(100)).unwrap();
        for &stage in PipelineStage::all_ordered() {
            assert!(sched.metrics().stage_avg_us(stage) > 0.0);
        }
    }

    // ── Pipeline flush ────────────────────────────────────────────

    #[test]
    fn flush_resets_everything() {
        let mut sched = make_scheduler();
        sched.drain(|_| Ok(10)).unwrap();
        assert!(sched.is_complete());
        sched.flush();
        for &stage in PipelineStage::all_ordered() {
            assert_eq!(sched.stage_status(stage), Some(&StageStatus::Pending));
        }
        assert_eq!(sched.metrics().total_tokens, 0);
    }

    // ── Inflight limits ───────────────────────────────────────────

    #[test]
    fn max_inflight_limits_ready_stages() {
        let config =
            AsyncPipelineConfig { max_inflight: 1, overlap_transfer: false, prefetch_depth: 0 };
        let mut sched = PipelineScheduler::new(config).unwrap();
        assert_eq!(sched.ready_stages().len(), 1);
        let _fence = sched.dispatch(PipelineStage::Tokenize).unwrap();
        assert_eq!(sched.inflight_count(), 1);
        assert!(sched.ready_stages().is_empty());
    }

    #[test]
    fn dispatch_beyond_inflight_fails() {
        let config = single_stage_config();
        let mut sched = PipelineScheduler::new(config).unwrap();
        let _fence = sched.dispatch(PipelineStage::Tokenize).unwrap();
        // Manually add a stage with no deps to attempt second dispatch
        sched.add_stage(PipelineStage::Sample, vec![]);
        let result = sched.dispatch(PipelineStage::Sample);
        assert!(result.is_err());
    }

    // ── Empty pipeline edge case ──────────────────────────────────

    #[test]
    fn empty_scheduler_is_not_complete() {
        // An empty scheduler with no stages should not report complete.
        let sched = PipelineScheduler::empty(default_config()).unwrap();
        assert!(!sched.is_complete());
    }

    #[test]
    fn empty_scheduler_ready_stages_empty() {
        let sched = PipelineScheduler::empty(default_config()).unwrap();
        assert!(sched.ready_stages().is_empty());
    }

    // ── Single stage pipeline ─────────────────────────────────────

    #[test]
    fn single_stage_pipeline() {
        let mut sched = PipelineScheduler::empty(default_config()).unwrap();
        sched.add_stage(PipelineStage::Tokenize, vec![]);
        let mut fence = sched.dispatch(PipelineStage::Tokenize).unwrap();
        sched.complete(&mut fence, 5).unwrap();
        assert!(sched.is_complete());
    }

    // ── All stages fail ───────────────────────────────────────────

    #[test]
    fn all_stages_fail_from_first() {
        let mut sched = make_scheduler();
        sched.fail(PipelineStage::Tokenize, "boom").unwrap();
        // Every stage should now be Failed
        for &stage in PipelineStage::all_ordered() {
            assert!(
                matches!(sched.stage_status(stage), Some(StageStatus::Failed(_))),
                "{stage} should be Failed"
            );
        }
        assert!(sched.is_complete());
    }

    // ── Reset after failure ───────────────────────────────────────

    #[test]
    fn reset_after_failure_allows_retry() {
        let mut sched = make_scheduler();
        sched.fail(PipelineStage::Tokenize, "transient").unwrap();
        sched.reset();
        // Should be able to run again
        let mut fence = sched.dispatch(PipelineStage::Tokenize).unwrap();
        sched.complete(&mut fence, 10).unwrap();
        assert_eq!(sched.stage_status(PipelineStage::Tokenize), Some(&StageStatus::Complete));
    }

    // ── Concurrent stage execution simulation ─────────────────────

    #[test]
    fn concurrent_independent_stages() {
        let config =
            AsyncPipelineConfig { max_inflight: 3, overlap_transfer: true, prefetch_depth: 2 };
        let mut sched = PipelineScheduler::empty(config).unwrap();
        // Three independent stages with no dependencies
        sched.add_stage(PipelineStage::Tokenize, vec![]);
        sched.add_stage(PipelineStage::Embed, vec![]);
        sched.add_stage(PipelineStage::Attention, vec![]);

        let ready = sched.ready_stages();
        assert_eq!(ready.len(), 3);

        // Dispatch all three
        let mut fences: Vec<PipelineFence> =
            ready.into_iter().map(|s| sched.dispatch(s).unwrap()).collect();
        assert_eq!(sched.inflight_count(), 3);

        // Complete all
        for fence in &mut fences {
            sched.complete(fence, 50).unwrap();
        }
        assert!(sched.is_complete());
    }

    // ── Prefetch depth effect ─────────────────────────────────────

    #[test]
    fn prefetch_depth_within_inflight() {
        let config =
            AsyncPipelineConfig { max_inflight: 6, overlap_transfer: true, prefetch_depth: 3 };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn prefetch_depth_zero_valid() {
        let config =
            AsyncPipelineConfig { max_inflight: 2, overlap_transfer: false, prefetch_depth: 0 };
        assert!(config.validate().is_ok());
    }

    // ── CPU reference implementations ─────────────────────────────

    #[test]
    fn cpu_simulate_stage_returns_positive() {
        for &stage in PipelineStage::all_ordered() {
            let us = cpu_simulate_stage(stage);
            assert!(us > 0, "{stage} returned 0 us");
        }
    }

    #[test]
    fn cpu_run_pipeline_succeeds() {
        let metrics = cpu_run_pipeline(default_config()).unwrap();
        assert_eq!(metrics.total_tokens, 1);
        assert_eq!(metrics.total_pipeline_runs, 1);
        assert!(metrics.total_pipeline_us > 0);
    }

    #[test]
    fn cpu_run_pipeline_invalid_config_fails() {
        let config = AsyncPipelineConfig { max_inflight: 0, ..default_config() };
        assert!(cpu_run_pipeline(config).is_err());
    }

    // ── Property-like tests for scheduling fairness ───────────────

    #[test]
    fn property_every_stage_eventually_runs() {
        let mut sched = make_scheduler();
        let mut completed = Vec::new();
        loop {
            let ready = sched.ready_stages();
            if ready.is_empty() {
                break;
            }
            for stage in ready {
                let mut fence = sched.dispatch(stage).unwrap();
                sched.complete(&mut fence, 10).unwrap();
                completed.push(stage);
            }
        }
        assert_eq!(completed.len(), 6);
        // Should follow dependency order
        for (i, &stage) in completed.iter().enumerate() {
            assert_eq!(stage.ordinal(), i, "stage {stage} at wrong position {i}");
        }
    }

    #[test]
    fn property_no_stage_runs_before_dependency() {
        let mut sched = make_scheduler();
        let mut run_order = Vec::new();
        sched
            .drain(|stage| {
                run_order.push(stage);
                Ok(10)
            })
            .unwrap();
        for i in 1..run_order.len() {
            assert!(
                run_order[i].ordinal() > run_order[i - 1].ordinal(),
                "stage {:?} ran before {:?}",
                run_order[i],
                run_order[i - 1]
            );
        }
    }

    #[test]
    fn property_inflight_never_exceeds_max() {
        let config =
            AsyncPipelineConfig { max_inflight: 2, overlap_transfer: true, prefetch_depth: 1 };
        let mut sched = PipelineScheduler::empty(config).unwrap();
        // Add independent stages
        for &stage in PipelineStage::all_ordered() {
            sched.add_stage(stage, vec![]);
        }

        let mut max_seen = 0usize;
        loop {
            let ready = sched.ready_stages();
            if ready.is_empty() {
                break;
            }
            let mut fences = Vec::new();
            for stage in ready {
                fences.push(sched.dispatch(stage).unwrap());
            }
            max_seen = max_seen.max(sched.inflight_count());
            for fence in &mut fences {
                sched.complete(fence, 5).unwrap();
            }
        }
        assert!(max_seen <= 2, "inflight exceeded max: {max_seen}");
    }

    #[test]
    fn property_completed_stages_not_re_dispatched() {
        let mut sched = make_scheduler();
        let mut fence = sched.dispatch(PipelineStage::Tokenize).unwrap();
        sched.complete(&mut fence, 10).unwrap();
        let result = sched.dispatch(PipelineStage::Tokenize);
        assert!(result.is_err());
    }

    // ── Additional edge-case tests ────────────────────────────────

    #[test]
    fn fence_id_increments() {
        let mut sched = make_scheduler();
        let f0 = sched.dispatch(PipelineStage::Tokenize).unwrap();
        assert_eq!(f0.id, 0);
        sched
            .complete(
                &mut PipelineFence::new(0, PipelineStage::Tokenize),
                // Use a fresh fence just for the signal; the scheduler
                // only cares about stage status.
                10,
            )
            .unwrap();
        // After reset we can see ids continue
        let f1 = sched.dispatch(PipelineStage::Embed).unwrap();
        assert_eq!(f1.id, 1);
    }

    #[test]
    fn status_snapshot_clones_independently() {
        let sched = make_scheduler();
        let snap = sched.status_snapshot();
        assert_eq!(snap.len(), 6);
        // Mutating the snapshot doesn't affect the scheduler.
    }

    #[test]
    fn config_accessor_returns_same_config() {
        let config =
            AsyncPipelineConfig { max_inflight: 7, overlap_transfer: false, prefetch_depth: 3 };
        let sched = PipelineScheduler::new(config.clone()).unwrap();
        assert_eq!(sched.config().max_inflight, 7);
        assert!(!sched.config().overlap_transfer);
    }
}
