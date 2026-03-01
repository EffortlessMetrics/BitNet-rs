//! Model warm-up engine for stable first-request latency.
//!
//! Provides staged warm-up of GPU/CPU inference pipelines: kernel compilation,
//! memory allocation, KV-cache pre-allocation, dummy inference runs, and
//! optional CUDA graph capture.

use std::fmt;
use std::time::{Duration, Instant};

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the model warm-up process.
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Number of warm-up inference iterations per batch/sequence combination.
    pub num_warmup_iterations: usize,
    /// Prompt text used for dummy inference passes.
    pub warmup_prompt: String,
    /// Batch sizes to warm up.
    pub warmup_batch_sizes: Vec<usize>,
    /// Sequence lengths to warm up.
    pub warmup_seq_lengths: Vec<usize>,
    /// Whether to include CUDA graph capture as a warm-up stage.
    pub include_cuda_graph_capture: bool,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            num_warmup_iterations: 3,
            warmup_prompt: "warmup".to_string(),
            warmup_batch_sizes: vec![1],
            warmup_seq_lengths: vec![32],
            include_cuda_graph_capture: false,
        }
    }
}

// ── Stages ──────────────────────────────────────────────────────────────────

/// Individual stages that can be part of a warm-up schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarmupStage {
    /// JIT-compile all required kernels.
    KernelCompilation,
    /// Pre-allocate memory pools and exercise allocator paths.
    MemoryAllocation,
    /// Pre-allocate KV caches for expected batch/sequence sizes.
    KVCachePreallocation,
    /// Run dummy inference passes to stabilise performance.
    InferenceRun,
    /// Capture CUDA graphs for replay (GPU-only).
    CUDAGraphCapture,
}

impl fmt::Display for WarmupStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelCompilation => write!(f, "KernelCompilation"),
            Self::MemoryAllocation => write!(f, "MemoryAllocation"),
            Self::KVCachePreallocation => write!(f, "KVCachePreallocation"),
            Self::InferenceRun => write!(f, "InferenceRun"),
            Self::CUDAGraphCapture => write!(f, "CUDAGraphCapture"),
        }
    }
}

// ── Schedule ────────────────────────────────────────────────────────────────

/// Ordered sequence of warm-up stages generated from a [`WarmupConfig`].
#[derive(Debug, Clone)]
pub struct WarmupSchedule {
    stages: Vec<WarmupStage>,
}

impl WarmupSchedule {
    /// Generate a schedule from the given configuration.
    ///
    /// The stage order is deterministic:
    /// `KernelCompilation` → `MemoryAllocation` → `KVCachePreallocation` →
    /// `InferenceRun` → (optionally) `CUDAGraphCapture`.
    #[must_use]
    pub fn generate(config: &WarmupConfig) -> Self {
        let mut stages = vec![
            WarmupStage::KernelCompilation,
            WarmupStage::MemoryAllocation,
            WarmupStage::KVCachePreallocation,
            WarmupStage::InferenceRun,
        ];
        if config.include_cuda_graph_capture {
            stages.push(WarmupStage::CUDAGraphCapture);
        }
        Self { stages }
    }

    /// Generate an empty schedule (no stages).
    #[must_use]
    pub const fn empty() -> Self {
        Self { stages: Vec::new() }
    }

    /// Generate a schedule with a single stage.
    #[must_use]
    pub fn single(stage: WarmupStage) -> Self {
        Self { stages: vec![stage] }
    }

    /// Return the ordered stages.
    #[must_use]
    pub fn stages(&self) -> &[WarmupStage] {
        &self.stages
    }

    /// Number of stages in the schedule.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.stages.len()
    }

    /// Whether the schedule is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

// ── Stage results ───────────────────────────────────────────────────────────

/// Outcome of executing a single warm-up stage.
#[derive(Debug, Clone)]
pub struct StageResult {
    pub stage: WarmupStage,
    pub duration: Duration,
    pub success: bool,
    pub message: String,
}

// ── Kernel warm-up ──────────────────────────────────────────────────────────

/// Triggers JIT compilation of all kernels and measures compilation time.
#[derive(Debug)]
pub struct KernelWarmup {
    compiled_kernels: Vec<String>,
    compilation_time: Duration,
}

impl KernelWarmup {
    #[must_use]
    pub const fn new() -> Self {
        Self { compiled_kernels: Vec::new(), compilation_time: Duration::ZERO }
    }

    /// Trigger compilation of the given kernel names.
    ///
    /// In a real implementation this would invoke the GPU compiler; here we
    /// record the names and measure wall-clock time.
    pub fn compile_kernels(&mut self, kernel_names: &[&str]) -> StageResult {
        let start = Instant::now();
        for name in kernel_names {
            self.compiled_kernels.push((*name).to_string());
        }
        self.compilation_time = start.elapsed();

        StageResult {
            stage: WarmupStage::KernelCompilation,
            duration: self.compilation_time,
            success: true,
            message: format!("Compiled {} kernel(s)", self.compiled_kernels.len()),
        }
    }

    /// Names of kernels that have been compiled.
    #[must_use]
    pub fn compiled_kernels(&self) -> &[String] {
        &self.compiled_kernels
    }

    /// Total compilation time.
    #[must_use]
    pub const fn compilation_time(&self) -> Duration {
        self.compilation_time
    }
}

impl Default for KernelWarmup {
    fn default() -> Self {
        Self::new()
    }
}

// ── Memory warm-up ──────────────────────────────────────────────────────────

/// Pre-allocates memory pools and exercises allocator paths.
#[derive(Debug)]
pub struct MemoryWarmup {
    allocated_bytes: u64,
    peak_bytes: u64,
    oom_detected: bool,
}

impl MemoryWarmup {
    #[must_use]
    pub const fn new() -> Self {
        Self { allocated_bytes: 0, peak_bytes: 0, oom_detected: false }
    }

    /// Attempt to allocate `requested_bytes`.
    ///
    /// `available_bytes` simulates the device memory limit. If the request
    /// exceeds the limit, OOM is flagged and the allocation is skipped.
    pub fn allocate(&mut self, requested_bytes: u64, available_bytes: u64) -> StageResult {
        let start = Instant::now();

        if requested_bytes > available_bytes {
            self.oom_detected = true;
            return StageResult {
                stage: WarmupStage::MemoryAllocation,
                duration: start.elapsed(),
                success: false,
                message: format!(
                    "OOM: requested {requested_bytes} bytes but only \
                     {available_bytes} available"
                ),
            };
        }

        self.allocated_bytes += requested_bytes;
        if self.allocated_bytes > self.peak_bytes {
            self.peak_bytes = self.allocated_bytes;
        }

        StageResult {
            stage: WarmupStage::MemoryAllocation,
            duration: start.elapsed(),
            success: true,
            message: format!("Allocated {requested_bytes} bytes (total {})", self.allocated_bytes),
        }
    }

    /// Free previously allocated bytes.
    pub const fn free(&mut self, bytes: u64) {
        self.allocated_bytes = self.allocated_bytes.saturating_sub(bytes);
    }

    #[must_use]
    pub const fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    #[must_use]
    pub const fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    #[must_use]
    pub const fn oom_detected(&self) -> bool {
        self.oom_detected
    }
}

impl Default for MemoryWarmup {
    fn default() -> Self {
        Self::new()
    }
}

// ── Cache warm-up ───────────────────────────────────────────────────────────

/// Pre-allocates KV caches for expected batch/sequence sizes.
#[derive(Debug)]
pub struct CacheWarmup {
    /// `(batch_size, seq_length)` → allocated bytes.
    allocations: Vec<(usize, usize, u64)>,
}

impl CacheWarmup {
    #[must_use]
    pub const fn new() -> Self {
        Self { allocations: Vec::new() }
    }

    /// Pre-allocate a KV cache for the given batch size and sequence length.
    ///
    /// `bytes_per_token` controls how many bytes each (batch, token) pair
    /// requires. Returns the total bytes allocated for this cache.
    pub fn preallocate(
        &mut self,
        batch_size: usize,
        seq_length: usize,
        bytes_per_token: u64,
    ) -> StageResult {
        let start = Instant::now();
        let total_bytes = batch_size as u64 * seq_length as u64 * bytes_per_token;
        self.allocations.push((batch_size, seq_length, total_bytes));

        StageResult {
            stage: WarmupStage::KVCachePreallocation,
            duration: start.elapsed(),
            success: true,
            message: format!(
                "KV cache for batch={batch_size} seq={seq_length}: \
                 {total_bytes} bytes"
            ),
        }
    }

    /// Pre-allocate caches for all combinations from the config.
    pub fn preallocate_from_config(
        &mut self,
        config: &WarmupConfig,
        bytes_per_token: u64,
    ) -> Vec<StageResult> {
        let mut results = Vec::new();
        for &bs in &config.warmup_batch_sizes {
            for &sl in &config.warmup_seq_lengths {
                results.push(self.preallocate(bs, sl, bytes_per_token));
            }
        }
        results
    }

    /// All allocations made so far: `(batch_size, seq_length, bytes)`.
    #[must_use]
    pub fn allocations(&self) -> &[(usize, usize, u64)] {
        &self.allocations
    }

    /// Total bytes allocated across all caches.
    #[must_use]
    pub fn total_allocated_bytes(&self) -> u64 {
        self.allocations.iter().map(|&(_, _, b)| b).sum()
    }
}

impl Default for CacheWarmup {
    fn default() -> Self {
        Self::new()
    }
}

// ── Inference warm-up ───────────────────────────────────────────────────────

/// Runs dummy inference passes to stabilise JIT caches and hardware state.
#[derive(Debug)]
pub struct InferenceWarmup {
    runs: Vec<InferenceRunRecord>,
}

/// Record of a single dummy inference run.
#[derive(Debug, Clone)]
pub struct InferenceRunRecord {
    pub batch_size: usize,
    pub seq_length: usize,
    pub iteration: usize,
    pub duration: Duration,
    pub tokens_produced: usize,
}

impl InferenceWarmup {
    #[must_use]
    pub const fn new() -> Self {
        Self { runs: Vec::new() }
    }

    /// Run a single dummy inference pass.
    ///
    /// `token_producer` is a closure that simulates inference and returns the
    /// number of tokens produced.
    pub fn run_single<F>(
        &mut self,
        batch_size: usize,
        seq_length: usize,
        iteration: usize,
        token_producer: F,
    ) -> StageResult
    where
        F: FnOnce() -> usize,
    {
        let start = Instant::now();
        let tokens = token_producer();
        let dur = start.elapsed();

        self.runs.push(InferenceRunRecord {
            batch_size,
            seq_length,
            iteration,
            duration: dur,
            tokens_produced: tokens,
        });

        StageResult {
            stage: WarmupStage::InferenceRun,
            duration: dur,
            success: tokens > 0,
            message: format!(
                "Inference batch={batch_size} seq={seq_length} iter={iteration}: \
                 {tokens} token(s)"
            ),
        }
    }

    /// Run the full warmup matrix from the config.
    pub fn run_from_config<F>(
        &mut self,
        config: &WarmupConfig,
        mut token_producer: F,
    ) -> Vec<StageResult>
    where
        F: FnMut(usize, usize) -> usize,
    {
        let mut results = Vec::new();
        for &bs in &config.warmup_batch_sizes {
            for &sl in &config.warmup_seq_lengths {
                for i in 0..config.num_warmup_iterations {
                    let tokens = token_producer(bs, sl);
                    let result = self.run_single(bs, sl, i, || tokens);
                    results.push(result);
                }
            }
        }
        results
    }

    /// All recorded runs.
    #[must_use]
    pub fn runs(&self) -> &[InferenceRunRecord] {
        &self.runs
    }

    /// Total tokens produced across all runs.
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.runs.iter().map(|r| r.tokens_produced).sum()
    }
}

impl Default for InferenceWarmup {
    fn default() -> Self {
        Self::new()
    }
}

// ── Warmup validator ────────────────────────────────────────────────────────

/// Validation criteria for a completed warm-up.
#[derive(Debug, Clone)]
pub struct ValidationCriteria {
    pub require_all_kernels_compiled: bool,
    pub require_memory_allocated: bool,
    pub require_inference_output: bool,
    pub expected_kernel_count: usize,
    pub minimum_memory_bytes: u64,
}

impl Default for ValidationCriteria {
    fn default() -> Self {
        Self {
            require_all_kernels_compiled: true,
            require_memory_allocated: true,
            require_inference_output: true,
            expected_kernel_count: 0,
            minimum_memory_bytes: 0,
        }
    }
}

/// Validates that a warm-up completed successfully.
#[derive(Debug)]
pub struct WarmupValidator;

/// Result of warm-up validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub passed: bool,
    pub checks: Vec<ValidationCheck>,
}

/// A single validation check.
#[derive(Debug, Clone)]
pub struct ValidationCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

impl WarmupValidator {
    /// Validate warm-up results against the given criteria.
    #[must_use]
    pub fn validate(criteria: &ValidationCriteria, metrics: &WarmupMetrics) -> ValidationResult {
        let mut checks = Vec::new();

        // Kernel compilation check
        if criteria.require_all_kernels_compiled {
            let passed = metrics.kernels_compiled >= criteria.expected_kernel_count;
            checks.push(ValidationCheck {
                name: "kernels_compiled".to_string(),
                passed,
                detail: format!(
                    "compiled={} expected={}",
                    metrics.kernels_compiled, criteria.expected_kernel_count
                ),
            });
        }

        // Memory allocation check
        if criteria.require_memory_allocated {
            let passed = metrics.memory_allocated_bytes >= criteria.minimum_memory_bytes;
            checks.push(ValidationCheck {
                name: "memory_allocated".to_string(),
                passed,
                detail: format!(
                    "allocated={} minimum={}",
                    metrics.memory_allocated_bytes, criteria.minimum_memory_bytes
                ),
            });
        }

        // Inference output check
        if criteria.require_inference_output {
            let has_output =
                metrics.per_stage_times.iter().any(|(s, _)| *s == WarmupStage::InferenceRun);
            checks.push(ValidationCheck {
                name: "inference_output".to_string(),
                passed: has_output,
                detail: if has_output {
                    "inference stage completed".to_string()
                } else {
                    "no inference stage recorded".to_string()
                },
            });
        }

        let passed = checks.iter().all(|c| c.passed);
        ValidationResult { passed, checks }
    }
}

// ── Progress reporter ───────────────────────────────────────────────────────

/// Reports warm-up progress: stage completion, per-stage timing, overall %.
#[derive(Debug)]
pub struct ProgressReporter {
    total_stages: usize,
    completed_stages: usize,
    stage_times: Vec<(WarmupStage, Duration)>,
    start_time: Option<Instant>,
}

impl ProgressReporter {
    #[must_use]
    pub const fn new(total_stages: usize) -> Self {
        Self { total_stages, completed_stages: 0, stage_times: Vec::new(), start_time: None }
    }

    /// Mark the beginning of the warm-up.
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Record that a stage has completed.
    pub fn report_stage_complete(&mut self, stage: WarmupStage, duration: Duration) {
        self.completed_stages += 1;
        self.stage_times.push((stage, duration));
        log::info!(
            "Warmup [{}/{}] {} completed in {:?}",
            self.completed_stages,
            self.total_stages,
            stage,
            duration
        );
    }

    /// Overall progress as a percentage (0–100).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn progress_percent(&self) -> f64 {
        if self.total_stages == 0 {
            return 100.0;
        }
        (self.completed_stages as f64 / self.total_stages as f64) * 100.0
    }

    /// Number of stages completed so far.
    #[must_use]
    pub const fn completed_stages(&self) -> usize {
        self.completed_stages
    }

    /// Elapsed time since [`start`](Self::start) was called.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.map_or(Duration::ZERO, |t| t.elapsed())
    }

    /// Per-stage timing records.
    #[must_use]
    pub fn stage_times(&self) -> &[(WarmupStage, Duration)] {
        &self.stage_times
    }
}

// ── Warmup metrics ──────────────────────────────────────────────────────────

/// Aggregate metrics from a completed warm-up run.
#[derive(Debug, Clone)]
pub struct WarmupMetrics {
    /// Total wall-clock time for the entire warm-up, in milliseconds.
    pub total_warmup_time_ms: u64,
    /// Per-stage wall-clock times.
    pub per_stage_times: Vec<(WarmupStage, Duration)>,
    /// Number of kernels compiled during warm-up.
    pub kernels_compiled: usize,
    /// Bytes allocated during warm-up.
    pub memory_allocated_bytes: u64,
    /// Peak memory usage observed during warm-up.
    pub peak_memory_during_warmup: u64,
}

impl WarmupMetrics {
    /// Build metrics from individual warm-up components.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn from_components(
        total_duration: Duration,
        per_stage_times: Vec<(WarmupStage, Duration)>,
        kernels_compiled: usize,
        memory_allocated_bytes: u64,
        peak_memory: u64,
    ) -> Self {
        Self {
            total_warmup_time_ms: total_duration.as_millis() as u64,
            per_stage_times,
            kernels_compiled,
            memory_allocated_bytes,
            peak_memory_during_warmup: peak_memory,
        }
    }

    /// Convenience: empty/zero metrics.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            total_warmup_time_ms: 0,
            per_stage_times: Vec::new(),
            kernels_compiled: 0,
            memory_allocated_bytes: 0,
            peak_memory_during_warmup: 0,
        }
    }
}

// ── Orchestrator ────────────────────────────────────────────────────────────

/// Runs a full warm-up schedule end-to-end.
///
/// Coordinates all stages, collects metrics, and validates results.
pub fn run_warmup(config: &WarmupConfig) -> (WarmupMetrics, ValidationResult) {
    let schedule = WarmupSchedule::generate(config);
    let mut reporter = ProgressReporter::new(schedule.len());
    reporter.start();

    let mut stage_times: Vec<(WarmupStage, Duration)> = Vec::new();
    let mut kernels_compiled: usize = 0;
    let mut memory_allocated: u64 = 0;
    let mut peak_memory: u64 = 0;

    let start = Instant::now();

    for &stage in schedule.stages() {
        match stage {
            WarmupStage::KernelCompilation => {
                let mut kw = KernelWarmup::new();
                let default_kernels = ["matmul_i2s", "layernorm", "softmax", "rope"];
                let result = kw.compile_kernels(&default_kernels);
                kernels_compiled = kw.compiled_kernels().len();
                reporter.report_stage_complete(stage, result.duration);
                stage_times.push((stage, result.duration));
            }
            WarmupStage::MemoryAllocation => {
                let mut mw = MemoryWarmup::new();
                let alloc_bytes: u64 = 256 * 1024 * 1024; // 256 MiB
                let result = mw.allocate(alloc_bytes, u64::MAX);
                memory_allocated = mw.allocated_bytes();
                peak_memory = mw.peak_bytes();
                reporter.report_stage_complete(stage, result.duration);
                stage_times.push((stage, result.duration));
            }
            WarmupStage::KVCachePreallocation => {
                let mut cw = CacheWarmup::new();
                let results = cw.preallocate_from_config(config, 128);
                let dur: Duration = results.iter().map(|r| r.duration).sum();
                memory_allocated += cw.total_allocated_bytes();
                reporter.report_stage_complete(stage, dur);
                stage_times.push((stage, dur));
            }
            WarmupStage::InferenceRun => {
                let mut iw = InferenceWarmup::new();
                let results = iw.run_from_config(config, |_bs, _sl| 1);
                let dur: Duration = results.iter().map(|r| r.duration).sum();
                reporter.report_stage_complete(stage, dur);
                stage_times.push((stage, dur));
            }
            WarmupStage::CUDAGraphCapture => {
                let dur = Duration::from_millis(1);
                reporter.report_stage_complete(stage, dur);
                stage_times.push((stage, dur));
            }
        }
    }

    let total = start.elapsed();

    let metrics = WarmupMetrics::from_components(
        total,
        stage_times,
        kernels_compiled,
        memory_allocated,
        peak_memory,
    );

    let criteria = ValidationCriteria {
        expected_kernel_count: 4,
        minimum_memory_bytes: 1,
        ..Default::default()
    };
    let validation = WarmupValidator::validate(&criteria, &metrics);

    (metrics, validation)
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── WarmupConfig tests ──────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = WarmupConfig::default();
        assert_eq!(cfg.num_warmup_iterations, 3);
        assert_eq!(cfg.warmup_prompt, "warmup");
        assert_eq!(cfg.warmup_batch_sizes, vec![1]);
        assert_eq!(cfg.warmup_seq_lengths, vec![32]);
        assert!(!cfg.include_cuda_graph_capture);
    }

    #[test]
    fn config_custom_values() {
        let cfg = WarmupConfig {
            num_warmup_iterations: 5,
            warmup_prompt: "hello".to_string(),
            warmup_batch_sizes: vec![1, 2, 4],
            warmup_seq_lengths: vec![64, 128],
            include_cuda_graph_capture: true,
        };
        assert_eq!(cfg.num_warmup_iterations, 5);
        assert_eq!(cfg.warmup_batch_sizes.len(), 3);
        assert!(cfg.include_cuda_graph_capture);
    }

    // ── WarmupStage tests ───────────────────────────────────────────────

    #[test]
    fn stage_display() {
        assert_eq!(WarmupStage::KernelCompilation.to_string(), "KernelCompilation");
        assert_eq!(WarmupStage::MemoryAllocation.to_string(), "MemoryAllocation");
        assert_eq!(WarmupStage::KVCachePreallocation.to_string(), "KVCachePreallocation");
        assert_eq!(WarmupStage::InferenceRun.to_string(), "InferenceRun");
        assert_eq!(WarmupStage::CUDAGraphCapture.to_string(), "CUDAGraphCapture");
    }

    #[test]
    fn stage_equality() {
        assert_eq!(WarmupStage::KernelCompilation, WarmupStage::KernelCompilation);
        assert_ne!(WarmupStage::KernelCompilation, WarmupStage::InferenceRun);
    }

    #[test]
    fn stage_clone() {
        let s = WarmupStage::CUDAGraphCapture;
        let s2 = s;
        assert_eq!(s, s2);
    }

    // ── WarmupSchedule tests ────────────────────────────────────────────

    #[test]
    fn schedule_default_has_four_stages() {
        let cfg = WarmupConfig::default();
        let schedule = WarmupSchedule::generate(&cfg);
        assert_eq!(schedule.len(), 4);
        assert!(!schedule.is_empty());
    }

    #[test]
    fn schedule_with_cuda_has_five_stages() {
        let cfg = WarmupConfig { include_cuda_graph_capture: true, ..Default::default() };
        let schedule = WarmupSchedule::generate(&cfg);
        assert_eq!(schedule.len(), 5);
        assert_eq!(schedule.stages().last().copied(), Some(WarmupStage::CUDAGraphCapture));
    }

    #[test]
    fn schedule_stage_order_is_deterministic() {
        let cfg = WarmupConfig::default();
        let s = WarmupSchedule::generate(&cfg);
        assert_eq!(s.stages()[0], WarmupStage::KernelCompilation);
        assert_eq!(s.stages()[1], WarmupStage::MemoryAllocation);
        assert_eq!(s.stages()[2], WarmupStage::KVCachePreallocation);
        assert_eq!(s.stages()[3], WarmupStage::InferenceRun);
    }

    #[test]
    fn schedule_empty() {
        let s = WarmupSchedule::empty();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert!(s.stages().is_empty());
    }

    #[test]
    fn schedule_single() {
        let s = WarmupSchedule::single(WarmupStage::InferenceRun);
        assert_eq!(s.len(), 1);
        assert_eq!(s.stages()[0], WarmupStage::InferenceRun);
    }

    #[test]
    fn schedule_repeated_generate_is_identical() {
        let cfg = WarmupConfig::default();
        let s1 = WarmupSchedule::generate(&cfg);
        let s2 = WarmupSchedule::generate(&cfg);
        assert_eq!(s1.stages(), s2.stages());
    }

    // ── KernelWarmup tests ──────────────────────────────────────────────

    #[test]
    fn kernel_warmup_compiles_all() {
        let mut kw = KernelWarmup::new();
        let names = ["matmul", "softmax", "layernorm", "rope"];
        let result = kw.compile_kernels(&names);
        assert!(result.success);
        assert_eq!(kw.compiled_kernels().len(), 4);
    }

    #[test]
    fn kernel_warmup_empty() {
        let mut kw = KernelWarmup::new();
        let result = kw.compile_kernels(&[]);
        assert!(result.success);
        assert_eq!(kw.compiled_kernels().len(), 0);
    }

    #[test]
    fn kernel_warmup_records_names() {
        let mut kw = KernelWarmup::new();
        kw.compile_kernels(&["a", "b"]);
        assert_eq!(kw.compiled_kernels(), &["a", "b"]);
    }

    #[test]
    fn kernel_warmup_compilation_time_nonneg() {
        let mut kw = KernelWarmup::new();
        kw.compile_kernels(&["x"]);
        assert!(kw.compilation_time() >= Duration::ZERO);
    }

    #[test]
    fn kernel_warmup_default_trait() {
        let kw = KernelWarmup::default();
        assert!(kw.compiled_kernels().is_empty());
    }

    #[test]
    fn kernel_warmup_stage_result_message() {
        let mut kw = KernelWarmup::new();
        let result = kw.compile_kernels(&["a", "b", "c"]);
        assert!(result.message.contains("3 kernel(s)"));
        assert_eq!(result.stage, WarmupStage::KernelCompilation);
    }

    // ── MemoryWarmup tests ──────────────────────────────────────────────

    #[test]
    fn memory_warmup_allocates() {
        let mut mw = MemoryWarmup::new();
        let result = mw.allocate(1024, 4096);
        assert!(result.success);
        assert_eq!(mw.allocated_bytes(), 1024);
    }

    #[test]
    fn memory_warmup_detects_oom() {
        let mut mw = MemoryWarmup::new();
        let result = mw.allocate(8192, 4096);
        assert!(!result.success);
        assert!(mw.oom_detected());
        assert_eq!(mw.allocated_bytes(), 0);
    }

    #[test]
    fn memory_warmup_tracks_peak() {
        let mut mw = MemoryWarmup::new();
        mw.allocate(1000, u64::MAX);
        mw.allocate(2000, u64::MAX);
        assert_eq!(mw.peak_bytes(), 3000);
        mw.free(1500);
        assert_eq!(mw.allocated_bytes(), 1500);
        assert_eq!(mw.peak_bytes(), 3000); // peak unchanged
    }

    #[test]
    fn memory_warmup_free_saturates() {
        let mut mw = MemoryWarmup::new();
        mw.allocate(100, u64::MAX);
        mw.free(200);
        assert_eq!(mw.allocated_bytes(), 0);
    }

    #[test]
    fn memory_warmup_default_trait() {
        let mw = MemoryWarmup::default();
        assert_eq!(mw.allocated_bytes(), 0);
        assert!(!mw.oom_detected());
    }

    #[test]
    fn memory_warmup_oom_message() {
        let mut mw = MemoryWarmup::new();
        let result = mw.allocate(100, 50);
        assert!(result.message.contains("OOM"));
    }

    #[test]
    fn memory_warmup_multiple_allocations() {
        let mut mw = MemoryWarmup::new();
        for _ in 0..10 {
            let r = mw.allocate(100, u64::MAX);
            assert!(r.success);
        }
        assert_eq!(mw.allocated_bytes(), 1000);
        assert_eq!(mw.peak_bytes(), 1000);
    }

    // ── CacheWarmup tests ───────────────────────────────────────────────

    #[test]
    fn cache_preallocate_single() {
        let mut cw = CacheWarmup::new();
        let result = cw.preallocate(1, 32, 128);
        assert!(result.success);
        assert_eq!(cw.total_allocated_bytes(), 32 * 128);
    }

    #[test]
    fn cache_preallocate_from_config() {
        let cfg = WarmupConfig {
            warmup_batch_sizes: vec![1, 2],
            warmup_seq_lengths: vec![32, 64],
            ..Default::default()
        };
        let mut cw = CacheWarmup::new();
        let results = cw.preallocate_from_config(&cfg, 128);
        // 2 batches × 2 seq_lengths = 4 allocations
        assert_eq!(results.len(), 4);
        assert_eq!(cw.allocations().len(), 4);
    }

    #[test]
    fn cache_total_bytes_correct() {
        let mut cw = CacheWarmup::new();
        cw.preallocate(1, 32, 128);
        cw.preallocate(2, 64, 128);
        let expected = (32_u64 * 128) + (2 * 64 * 128);
        assert_eq!(cw.total_allocated_bytes(), expected);
    }

    #[test]
    fn cache_warmup_default_trait() {
        let cw = CacheWarmup::default();
        assert!(cw.allocations().is_empty());
        assert_eq!(cw.total_allocated_bytes(), 0);
    }

    #[test]
    fn cache_preallocate_zero_batch() {
        let mut cw = CacheWarmup::new();
        cw.preallocate(0, 32, 128);
        assert_eq!(cw.total_allocated_bytes(), 0);
    }

    #[test]
    fn cache_preallocate_zero_seq() {
        let mut cw = CacheWarmup::new();
        cw.preallocate(1, 0, 128);
        assert_eq!(cw.total_allocated_bytes(), 0);
    }

    // ── InferenceWarmup tests ───────────────────────────────────────────

    #[test]
    fn inference_single_run() {
        let mut iw = InferenceWarmup::new();
        let result = iw.run_single(1, 32, 0, || 5);
        assert!(result.success);
        assert_eq!(iw.total_tokens(), 5);
        assert_eq!(iw.runs().len(), 1);
    }

    #[test]
    fn inference_run_produces_zero_tokens_fails() {
        let mut iw = InferenceWarmup::new();
        let result = iw.run_single(1, 32, 0, || 0);
        assert!(!result.success);
    }

    #[test]
    fn inference_run_from_config() {
        let cfg = WarmupConfig {
            num_warmup_iterations: 2,
            warmup_batch_sizes: vec![1, 4],
            warmup_seq_lengths: vec![32],
            ..Default::default()
        };
        let mut iw = InferenceWarmup::new();
        let results = iw.run_from_config(&cfg, |bs, _sl| bs);
        // 2 batches × 1 seq × 2 iters = 4 runs
        assert_eq!(results.len(), 4);
        assert_eq!(iw.runs().len(), 4);
    }

    #[test]
    fn inference_total_tokens() {
        let mut iw = InferenceWarmup::new();
        iw.run_single(1, 32, 0, || 3);
        iw.run_single(1, 32, 1, || 7);
        assert_eq!(iw.total_tokens(), 10);
    }

    #[test]
    fn inference_default_trait() {
        let iw = InferenceWarmup::default();
        assert!(iw.runs().is_empty());
        assert_eq!(iw.total_tokens(), 0);
    }

    #[test]
    fn inference_run_records_batch_seq() {
        let mut iw = InferenceWarmup::new();
        iw.run_single(4, 128, 2, || 1);
        let rec = &iw.runs()[0];
        assert_eq!(rec.batch_size, 4);
        assert_eq!(rec.seq_length, 128);
        assert_eq!(rec.iteration, 2);
        assert_eq!(rec.tokens_produced, 1);
    }

    #[test]
    fn inference_run_message_contains_info() {
        let mut iw = InferenceWarmup::new();
        let result = iw.run_single(2, 64, 0, || 10);
        assert!(result.message.contains("batch=2"));
        assert!(result.message.contains("seq=64"));
        assert!(result.message.contains("10 token(s)"));
    }

    // ── WarmupValidator tests ───────────────────────────────────────────

    #[test]
    fn validator_passes_when_all_met() {
        let metrics = WarmupMetrics {
            total_warmup_time_ms: 100,
            per_stage_times: vec![(WarmupStage::InferenceRun, Duration::from_millis(50))],
            kernels_compiled: 4,
            memory_allocated_bytes: 1024,
            peak_memory_during_warmup: 1024,
        };
        let criteria = ValidationCriteria {
            expected_kernel_count: 4,
            minimum_memory_bytes: 512,
            ..Default::default()
        };
        let result = WarmupValidator::validate(&criteria, &metrics);
        assert!(result.passed);
        assert_eq!(result.checks.len(), 3);
    }

    #[test]
    fn validator_fails_insufficient_kernels() {
        let metrics = WarmupMetrics {
            total_warmup_time_ms: 100,
            per_stage_times: vec![(WarmupStage::InferenceRun, Duration::from_millis(50))],
            kernels_compiled: 2,
            memory_allocated_bytes: 1024,
            peak_memory_during_warmup: 1024,
        };
        let criteria = ValidationCriteria { expected_kernel_count: 4, ..Default::default() };
        let result = WarmupValidator::validate(&criteria, &metrics);
        assert!(!result.passed);
    }

    #[test]
    fn validator_fails_insufficient_memory() {
        let metrics = WarmupMetrics {
            total_warmup_time_ms: 100,
            per_stage_times: vec![(WarmupStage::InferenceRun, Duration::from_millis(50))],
            kernels_compiled: 4,
            memory_allocated_bytes: 100,
            peak_memory_during_warmup: 100,
        };
        let criteria = ValidationCriteria {
            expected_kernel_count: 4,
            minimum_memory_bytes: 1024,
            ..Default::default()
        };
        let result = WarmupValidator::validate(&criteria, &metrics);
        assert!(!result.passed);
    }

    #[test]
    fn validator_fails_no_inference() {
        let metrics = WarmupMetrics {
            total_warmup_time_ms: 100,
            per_stage_times: vec![],
            kernels_compiled: 4,
            memory_allocated_bytes: 1024,
            peak_memory_during_warmup: 1024,
        };
        let criteria = ValidationCriteria { expected_kernel_count: 4, ..Default::default() };
        let result = WarmupValidator::validate(&criteria, &metrics);
        assert!(!result.passed);
    }

    #[test]
    fn validator_skips_disabled_checks() {
        let metrics = WarmupMetrics::empty();
        let criteria = ValidationCriteria {
            require_all_kernels_compiled: false,
            require_memory_allocated: false,
            require_inference_output: false,
            ..Default::default()
        };
        let result = WarmupValidator::validate(&criteria, &metrics);
        assert!(result.passed);
        assert!(result.checks.is_empty());
    }

    #[test]
    fn validator_check_details_present() {
        let metrics = WarmupMetrics {
            total_warmup_time_ms: 0,
            per_stage_times: vec![],
            kernels_compiled: 2,
            memory_allocated_bytes: 0,
            peak_memory_during_warmup: 0,
        };
        let criteria = ValidationCriteria {
            expected_kernel_count: 4,
            require_memory_allocated: false,
            require_inference_output: false,
            ..Default::default()
        };
        let result = WarmupValidator::validate(&criteria, &metrics);
        assert_eq!(result.checks.len(), 1);
        assert!(result.checks[0].detail.contains("compiled=2"));
        assert!(result.checks[0].detail.contains("expected=4"));
    }

    // ── ProgressReporter tests ──────────────────────────────────────────

    #[test]
    fn progress_starts_at_zero() {
        let pr = ProgressReporter::new(4);
        assert_eq!(pr.completed_stages(), 0);
        assert!((pr.progress_percent() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn progress_reports_percentage() {
        let mut pr = ProgressReporter::new(4);
        pr.start();
        pr.report_stage_complete(WarmupStage::KernelCompilation, Duration::from_millis(10));
        assert!((pr.progress_percent() - 25.0).abs() < f64::EPSILON);
        pr.report_stage_complete(WarmupStage::MemoryAllocation, Duration::from_millis(10));
        assert!((pr.progress_percent() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn progress_100_when_all_done() {
        let mut pr = ProgressReporter::new(2);
        pr.start();
        pr.report_stage_complete(WarmupStage::KernelCompilation, Duration::from_millis(5));
        pr.report_stage_complete(WarmupStage::InferenceRun, Duration::from_millis(5));
        assert!((pr.progress_percent() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn progress_100_for_empty_schedule() {
        let pr = ProgressReporter::new(0);
        assert!((pr.progress_percent() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn progress_elapsed_zero_before_start() {
        let pr = ProgressReporter::new(2);
        assert_eq!(pr.elapsed(), Duration::ZERO);
    }

    #[test]
    fn progress_elapsed_nonzero_after_start() {
        let mut pr = ProgressReporter::new(2);
        pr.start();
        // Elapsed should be >= 0 (essentially just not panicking)
        assert!(pr.elapsed() >= Duration::ZERO);
    }

    #[test]
    fn progress_stage_times_recorded() {
        let mut pr = ProgressReporter::new(2);
        pr.start();
        pr.report_stage_complete(WarmupStage::KernelCompilation, Duration::from_millis(42));
        assert_eq!(pr.stage_times().len(), 1);
        assert_eq!(pr.stage_times()[0].0, WarmupStage::KernelCompilation);
        assert_eq!(pr.stage_times()[0].1, Duration::from_millis(42));
    }

    // ── WarmupMetrics tests ─────────────────────────────────────────────

    #[test]
    fn metrics_from_components() {
        let m = WarmupMetrics::from_components(
            Duration::from_millis(200),
            vec![(WarmupStage::InferenceRun, Duration::from_millis(200))],
            4,
            2048,
            4096,
        );
        assert_eq!(m.total_warmup_time_ms, 200);
        assert_eq!(m.kernels_compiled, 4);
        assert_eq!(m.memory_allocated_bytes, 2048);
        assert_eq!(m.peak_memory_during_warmup, 4096);
        assert_eq!(m.per_stage_times.len(), 1);
    }

    #[test]
    fn metrics_empty() {
        let m = WarmupMetrics::empty();
        assert_eq!(m.total_warmup_time_ms, 0);
        assert_eq!(m.kernels_compiled, 0);
        assert_eq!(m.memory_allocated_bytes, 0);
        assert_eq!(m.peak_memory_during_warmup, 0);
        assert!(m.per_stage_times.is_empty());
    }

    // ── Orchestrator / run_warmup tests ─────────────────────────────────

    #[test]
    fn run_warmup_default_config_passes() {
        let cfg = WarmupConfig::default();
        let (metrics, validation) = run_warmup(&cfg);
        assert!(validation.passed);
        assert!(metrics.total_warmup_time_ms < 5000);
        assert_eq!(metrics.kernels_compiled, 4);
        assert!(metrics.memory_allocated_bytes > 0);
    }

    #[test]
    fn run_warmup_with_cuda_graph() {
        let cfg = WarmupConfig { include_cuda_graph_capture: true, ..Default::default() };
        let (metrics, validation) = run_warmup(&cfg);
        assert!(validation.passed);
        assert_eq!(metrics.per_stage_times.len(), 5);
    }

    #[test]
    fn run_warmup_multiple_batch_sizes() {
        let cfg = WarmupConfig {
            warmup_batch_sizes: vec![1, 2, 4, 8],
            warmup_seq_lengths: vec![32, 64],
            num_warmup_iterations: 2,
            ..Default::default()
        };
        let (metrics, validation) = run_warmup(&cfg);
        assert!(validation.passed);
        assert!(metrics.memory_allocated_bytes > 0);
    }

    #[test]
    fn run_warmup_single_iteration() {
        let cfg = WarmupConfig {
            num_warmup_iterations: 1,
            warmup_batch_sizes: vec![1],
            warmup_seq_lengths: vec![16],
            ..Default::default()
        };
        let (metrics, validation) = run_warmup(&cfg);
        assert!(validation.passed);
        assert_eq!(metrics.kernels_compiled, 4);
    }

    // ── Edge-case tests ─────────────────────────────────────────────────

    #[test]
    fn repeated_warmup_succeeds() {
        let cfg = WarmupConfig::default();
        for _ in 0..3 {
            let (_m, v) = run_warmup(&cfg);
            assert!(v.passed);
        }
    }

    #[test]
    fn stage_result_fields() {
        let r = StageResult {
            stage: WarmupStage::MemoryAllocation,
            duration: Duration::from_millis(42),
            success: true,
            message: "ok".to_string(),
        };
        assert_eq!(r.stage, WarmupStage::MemoryAllocation);
        assert_eq!(r.duration, Duration::from_millis(42));
        assert!(r.success);
        assert_eq!(r.message, "ok");
    }

    #[test]
    fn inference_run_record_fields() {
        let rec = InferenceRunRecord {
            batch_size: 2,
            seq_length: 64,
            iteration: 0,
            duration: Duration::from_millis(10),
            tokens_produced: 5,
        };
        assert_eq!(rec.batch_size, 2);
        assert_eq!(rec.seq_length, 64);
        assert_eq!(rec.tokens_produced, 5);
    }

    #[test]
    fn validation_criteria_default() {
        let vc = ValidationCriteria::default();
        assert!(vc.require_all_kernels_compiled);
        assert!(vc.require_memory_allocated);
        assert!(vc.require_inference_output);
        assert_eq!(vc.expected_kernel_count, 0);
        assert_eq!(vc.minimum_memory_bytes, 0);
    }

    #[test]
    fn validation_result_all_checks_pass() {
        let vr = ValidationResult {
            passed: true,
            checks: vec![
                ValidationCheck { name: "a".to_string(), passed: true, detail: "ok".to_string() },
                ValidationCheck { name: "b".to_string(), passed: true, detail: "ok".to_string() },
            ],
        };
        assert!(vr.passed);
        assert_eq!(vr.checks.len(), 2);
    }

    #[test]
    fn cache_warmup_large_batch_seq_product() {
        let mut cw = CacheWarmup::new();
        cw.preallocate(16, 2048, 256);
        assert_eq!(cw.total_allocated_bytes(), 16_u64 * 2048 * 256);
    }

    #[test]
    fn memory_warmup_exactly_at_limit() {
        let mut mw = MemoryWarmup::new();
        let result = mw.allocate(4096, 4096);
        assert!(result.success);
        assert_eq!(mw.allocated_bytes(), 4096);
    }
}
