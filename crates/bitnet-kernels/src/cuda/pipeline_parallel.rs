//! CUDA pipeline parallelism for multi-stage inference across GPU streams.
//!
//! Splits a model's forward pass into sequential *stages* — each mapped to a
//! dedicated CUDA stream — and overlaps execution of different micro-batches
//! across stages using configurable scheduling policies (GPipe, 1F1B, etc.).
//!
//! # Key components
//!
//! - [`GpuPipelineSchedule`] — scheduling strategy (Sequential, GPipe, 1F1B, Interleaved).
//! - [`GpuPipelineStage`] — single stage with stream assignment and resource tracking.
//! - [`GpuPipelineConfig`] — full pipeline configuration.
//! - [`StageResources`] — per-stage memory and compute budget.
//! - [`InterStageBuffer`] — device-memory communication buffer between stages.
//! - [`PipelineMetrics`] — latency and throughput estimation.
//! - [`GradAccumulator`] — gradient accumulation across micro-batches for training.
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallback implementations execute operations sequentially.

use std::fmt;

use bitnet_common::{BitNetError, KernelError, Result};

// ── Schedule ───────────────────────────────────────────────────────

/// GPU pipeline scheduling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GpuPipelineSchedule {
    /// Each micro-batch completes all stages before the next begins.
    Sequential,
    /// All micro-batches run the same stage before advancing (GPipe).
    #[default]
    GPipe,
    /// 1F1B steady-state schedule — alternates forward and backward passes
    /// to minimise pipeline bubble and peak activation memory.
    OneF1B,
    /// Interleaved 1F1B with virtual stages for further bubble reduction.
    Interleaved,
}

impl fmt::Display for GpuPipelineSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sequential => write!(f, "Sequential"),
            Self::GPipe => write!(f, "GPipe"),
            Self::OneF1B => write!(f, "1F1B"),
            Self::Interleaved => write!(f, "Interleaved"),
        }
    }
}

// ── Per-stage resource tracking ────────────────────────────────────

/// Resource budget for a single pipeline stage.
#[derive(Debug, Clone)]
pub struct StageResources {
    /// Peak memory usage in bytes.
    pub memory_bytes: u64,
    /// Estimated compute cost (FLOPs or arbitrary units).
    pub compute_cost: f64,
    /// Number of active tensors held across the stage boundary.
    pub active_tensors: usize,
}

impl StageResources {
    /// Create a new resource record.
    pub fn new(memory_bytes: u64, compute_cost: f64, active_tensors: usize) -> Self {
        Self { memory_bytes, compute_cost, active_tensors }
    }

    /// Create a zero-cost resource record.
    pub fn zero() -> Self {
        Self { memory_bytes: 0, compute_cost: 0.0, active_tensors: 0 }
    }

    /// Validate the resource record.
    pub fn validate(&self) -> Result<()> {
        if self.compute_cost < 0.0 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: "compute_cost must be non-negative".into(),
            }));
        }
        Ok(())
    }
}

impl Default for StageResources {
    fn default() -> Self {
        Self::zero()
    }
}

// ── Stage configuration ────────────────────────────────────────────

/// Configuration for a single GPU pipeline stage.
#[derive(Debug, Clone)]
pub struct GpuPipelineStage {
    /// Index of the first layer (inclusive).
    pub start_layer: usize,
    /// Index of the last layer (exclusive).
    pub end_layer: usize,
    /// GPU stream index this stage is assigned to (`None` = auto-assign).
    pub stream_id: Option<usize>,
    /// Resource budget for this stage.
    pub resources: StageResources,
}

impl GpuPipelineStage {
    /// Create a new stage spanning `[start_layer, end_layer)`.
    pub fn new(start_layer: usize, end_layer: usize) -> Self {
        Self { start_layer, end_layer, stream_id: None, resources: StageResources::zero() }
    }

    /// Builder: assign a specific GPU stream.
    pub fn with_stream(mut self, stream_id: usize) -> Self {
        self.stream_id = Some(stream_id);
        self
    }

    /// Builder: attach a resource budget.
    pub fn with_resources(mut self, resources: StageResources) -> Self {
        self.resources = resources;
        self
    }

    /// Number of layers handled by this stage.
    pub fn num_layers(&self) -> usize {
        self.end_layer.saturating_sub(self.start_layer)
    }

    /// Validate the stage configuration.
    pub fn validate(&self) -> Result<()> {
        if self.end_layer <= self.start_layer {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!(
                    "stage end_layer ({}) must be greater than start_layer ({})",
                    self.end_layer, self.start_layer,
                ),
            }));
        }
        self.resources.validate()?;
        Ok(())
    }
}

// ── Inter-stage communication buffer ───────────────────────────────

/// Device-memory buffer for communication between adjacent pipeline stages.
///
/// In a real GPU pipeline each buffer represents a `CUdeviceptr` allocation;
/// the CPU fallback uses a `Vec<f32>` for functional testing.
#[derive(Debug, Clone)]
pub struct InterStageBuffer {
    /// Unique buffer identifier.
    pub id: usize,
    /// Producer stage index.
    pub producer_stage: usize,
    /// Consumer stage index.
    pub consumer_stage: usize,
    /// Buffer capacity in number of f32 elements.
    pub capacity: usize,
    /// Current data (CPU fallback).
    pub data: Vec<f32>,
    /// Whether this buffer has been written by the producer and is ready.
    pub ready: bool,
}

impl InterStageBuffer {
    /// Create a new inter-stage buffer between `producer` and `consumer`.
    pub fn new(id: usize, producer_stage: usize, consumer_stage: usize, capacity: usize) -> Self {
        Self {
            id,
            producer_stage,
            consumer_stage,
            capacity,
            data: Vec::with_capacity(capacity),
            ready: false,
        }
    }

    /// Write data into the buffer (producer side).
    pub fn write(&mut self, data: &[f32]) -> Result<()> {
        if data.len() > self.capacity {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!(
                    "data length {} exceeds buffer capacity {}",
                    data.len(),
                    self.capacity,
                ),
            }));
        }
        self.data = data.to_vec();
        self.ready = true;
        Ok(())
    }

    /// Read data from the buffer (consumer side). Marks buffer as consumed.
    pub fn read(&mut self) -> Result<Vec<f32>> {
        if !self.ready {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: "buffer not ready — producer has not written data".into(),
            }));
        }
        self.ready = false;
        Ok(std::mem::take(&mut self.data))
    }

    /// Reset the buffer to its initial state.
    pub fn reset(&mut self) {
        self.data.clear();
        self.ready = false;
    }
}

// ── Pipeline configuration ─────────────────────────────────────────

/// Full GPU pipeline configuration.
#[derive(Debug, Clone)]
pub struct GpuPipelineConfig {
    /// Ordered list of stages (stage 0 feeds into stage 1, etc.).
    pub stages: Vec<GpuPipelineStage>,
    /// Number of elements per micro-batch (along the batch dimension).
    pub micro_batch_size: usize,
    /// Scheduling strategy.
    pub schedule: GpuPipelineSchedule,
    /// Number of GPU streams available for pipeline execution.
    pub num_streams: usize,
    /// Whether gradient accumulation is enabled (training mode).
    pub grad_accumulation: bool,
}

impl GpuPipelineConfig {
    /// Create a pipeline with the given stages and micro-batch size.
    pub fn new(
        stages: Vec<GpuPipelineStage>,
        micro_batch_size: usize,
        schedule: GpuPipelineSchedule,
    ) -> Self {
        let num_streams = stages.len();
        Self { stages, micro_batch_size, schedule, num_streams, grad_accumulation: false }
    }

    /// Builder: set the number of available GPU streams.
    pub fn with_num_streams(mut self, num_streams: usize) -> Self {
        self.num_streams = num_streams;
        self
    }

    /// Builder: enable gradient accumulation for training.
    pub fn with_grad_accumulation(mut self, enabled: bool) -> Self {
        self.grad_accumulation = enabled;
        self
    }

    /// Number of pipeline stages.
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Validate the entire pipeline configuration.
    pub fn validate(&self) -> Result<()> {
        if self.stages.is_empty() {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: "pipeline must have at least one stage".into(),
            }));
        }
        if self.micro_batch_size == 0 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: "micro_batch_size must be > 0".into(),
            }));
        }
        if self.num_streams == 0 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: "num_streams must be > 0".into(),
            }));
        }
        for (i, stage) in self.stages.iter().enumerate() {
            stage.validate().map_err(|_| {
                BitNetError::Kernel(KernelError::InvalidArguments {
                    reason: format!("invalid stage {i}"),
                })
            })?;
        }
        // Check contiguity: stage[i].end == stage[i+1].start
        for i in 0..self.stages.len() - 1 {
            if self.stages[i].end_layer != self.stages[i + 1].start_layer {
                return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                    reason: format!(
                        "stages must be contiguous: stage {} ends at {} but stage {} starts at {}",
                        i,
                        self.stages[i].end_layer,
                        i + 1,
                        self.stages[i + 1].start_layer,
                    ),
                }));
            }
        }
        // Validate stream assignments are within bounds.
        for (i, stage) in self.stages.iter().enumerate() {
            if let Some(sid) = stage.stream_id
                && sid >= self.num_streams
            {
                return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                    reason: format!(
                        "stage {} stream_id {} exceeds num_streams {}",
                        i, sid, self.num_streams,
                    ),
                }));
            }
        }
        Ok(())
    }

    /// Compute the effective stream assignment for each stage.
    /// Stages with explicit `stream_id` use it; others are round-robin assigned.
    pub fn effective_stream_assignment(&self) -> Vec<usize> {
        let mut assignments = Vec::with_capacity(self.stages.len());
        let mut next_auto = 0usize;
        for stage in &self.stages {
            if let Some(sid) = stage.stream_id {
                assignments.push(sid);
            } else {
                assignments.push(next_auto % self.num_streams);
                next_auto += 1;
            }
        }
        assignments
    }

    /// Total memory across all stages.
    pub fn total_memory(&self) -> u64 {
        self.stages.iter().map(|s| s.resources.memory_bytes).sum()
    }

    /// Total compute cost across all stages.
    pub fn total_compute(&self) -> f64 {
        self.stages.iter().map(|s| s.resources.compute_cost).sum()
    }

    /// Peak memory — the maximum memory required by any single stage.
    pub fn peak_stage_memory(&self) -> u64 {
        self.stages.iter().map(|s| s.resources.memory_bytes).max().unwrap_or(0)
    }
}

// ── Micro-batch helpers ────────────────────────────────────────────

/// Split `input` (shape `[batch, dim]`) into micro-batches of
/// `micro_batch_size` rows each.  The last chunk may be smaller.
pub fn gpu_micro_batch_split(
    input: &[f32],
    batch: usize,
    dim: usize,
    micro_batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    if input.is_empty() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "input must not be empty".into(),
        }));
    }
    if dim == 0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "dim must be > 0".into(),
        }));
    }
    if micro_batch_size == 0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "micro_batch_size must be > 0".into(),
        }));
    }
    if input.len() != batch * dim {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "input length {} does not match batch ({}) * dim ({})",
                input.len(),
                batch,
                dim,
            ),
        }));
    }

    let mut batches = Vec::new();
    let mut offset = 0;
    let mut remaining = batch;
    while remaining > 0 {
        let chunk = remaining.min(micro_batch_size);
        let elems = chunk * dim;
        batches.push(input[offset..offset + elems].to_vec());
        offset += elems;
        remaining -= chunk;
    }
    Ok(batches)
}

/// Merge micro-batch outputs back into a single contiguous buffer.
pub fn gpu_micro_batch_merge(batches: &[Vec<f32>]) -> Result<Vec<f32>> {
    if batches.is_empty() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "no micro-batches to merge".into(),
        }));
    }
    let total_len: usize = batches.iter().map(|b| b.len()).sum();
    let mut out = Vec::with_capacity(total_len);
    for b in batches {
        out.extend_from_slice(b);
    }
    Ok(out)
}

// ── Stage forward ──────────────────────────────────────────────────

/// Execute a single pipeline stage on a micro-batch (CPU fallback).
///
/// The stage function simulates GPU kernel execution by scaling each
/// element by the number of layers in the stage — a lightweight placeholder
/// for real CUDA kernel dispatch that exercises the pipeline mechanics.
pub fn gpu_stage_forward(input: &[f32], stage: &GpuPipelineStage) -> Result<Vec<f32>> {
    stage.validate()?;
    if input.is_empty() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "stage input must not be empty".into(),
        }));
    }
    let num_layers = stage.num_layers() as f32;
    let out: Vec<f32> = input.iter().map(|&x| x * num_layers).collect();
    Ok(out)
}

// ── 1F1B Schedule generation ───────────────────────────────────────

/// A single action in the 1F1B pipeline schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleAction {
    /// Forward pass for micro-batch `micro_batch_id` on `stage_id`.
    Forward { stage_id: usize, micro_batch_id: usize },
    /// Backward pass for micro-batch `micro_batch_id` on `stage_id`.
    Backward { stage_id: usize, micro_batch_id: usize },
    /// Idle bubble slot on `stage_id`.
    Bubble { stage_id: usize },
}

/// Generate a 1F1B schedule for a given number of stages and micro-batches.
///
/// The schedule has three phases:
/// 1. **Warmup** — fill the pipeline with forward passes.
/// 2. **Steady-state** — alternate 1 forward + 1 backward per time-step.
/// 3. **Drain** — flush remaining backward passes.
///
/// Returns a `Vec<Vec<ScheduleAction>>` where the outer index is the
/// time-step and each inner `Vec` contains one action per stage.
pub fn generate_1f1b_schedule(
    num_stages: usize,
    num_micro_batches: usize,
) -> Result<Vec<Vec<ScheduleAction>>> {
    if num_stages == 0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "num_stages must be > 0".into(),
        }));
    }
    if num_micro_batches == 0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "num_micro_batches must be > 0".into(),
        }));
    }

    let mut schedule: Vec<Vec<ScheduleAction>> = Vec::new();

    // Per-stage counters for which forward / backward micro-batch is next.
    let mut fwd_counter = vec![0usize; num_stages];
    let mut bwd_counter = vec![0usize; num_stages];
    // Track how many forwards have completed per stage (ready for backward).
    let mut fwd_completed = vec![0usize; num_stages];

    let total_steps = 2 * num_micro_batches + num_stages - 1;

    for _t in 0..total_steps {
        let mut step = Vec::with_capacity(num_stages);
        for s in 0..num_stages {
            let can_fwd = fwd_counter[s] < num_micro_batches;
            let can_bwd = bwd_counter[s] < fwd_completed[s];

            // In 1F1B: warmup fills forward, then alternate fwd/bwd.
            let in_warmup = fwd_counter[s] < (num_stages - s);

            if can_fwd && (in_warmup || !can_bwd) {
                let mb = fwd_counter[s];
                fwd_counter[s] += 1;
                step.push(ScheduleAction::Forward { stage_id: s, micro_batch_id: mb });
            } else if can_bwd {
                let mb = bwd_counter[s];
                bwd_counter[s] += 1;
                step.push(ScheduleAction::Backward { stage_id: s, micro_batch_id: mb });
            } else if can_fwd {
                let mb = fwd_counter[s];
                fwd_counter[s] += 1;
                step.push(ScheduleAction::Forward { stage_id: s, micro_batch_id: mb });
            } else {
                step.push(ScheduleAction::Bubble { stage_id: s });
            }
        }
        // After each step, a completed forward on stage s makes its
        // micro-batch available for backward.
        for action in &step {
            if let ScheduleAction::Forward { stage_id, .. } = action {
                fwd_completed[*stage_id] += 1;
            }
        }
        schedule.push(step);
    }

    Ok(schedule)
}

// ── Pipeline forward ───────────────────────────────────────────────

/// Execute the full GPU pipeline on `input` (shape `[batch, dim]`).
///
/// The function:
/// 1. Validates the configuration.
/// 2. Splits the input into micro-batches.
/// 3. Runs each micro-batch through all stages using the configured schedule.
/// 4. Merges the results back into a single output buffer.
///
/// On CPU this executes sequentially; on GPU each stage would be dispatched
/// to its assigned CUDA stream with inter-stream synchronisation events.
pub fn gpu_pipeline_forward(
    input: &[f32],
    batch: usize,
    dim: usize,
    config: &GpuPipelineConfig,
) -> Result<Vec<f32>> {
    config.validate()?;

    if input.is_empty() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "input must not be empty".into(),
        }));
    }
    if input.len() != batch * dim {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("input length {} != batch ({}) * dim ({})", input.len(), batch, dim),
        }));
    }

    let mut micro_batches = gpu_micro_batch_split(input, batch, dim, config.micro_batch_size)?;

    match config.schedule {
        GpuPipelineSchedule::Sequential => {
            for mb in &mut micro_batches {
                for stage in &config.stages {
                    *mb = gpu_stage_forward(mb, stage)?;
                }
            }
        }
        GpuPipelineSchedule::GPipe => {
            for stage in &config.stages {
                for mb in &mut micro_batches {
                    *mb = gpu_stage_forward(mb, stage)?;
                }
            }
        }
        GpuPipelineSchedule::OneF1B => {
            // Functional equivalence: 1F1B produces the same output as GPipe
            // for inference (backward passes are no-ops in inference mode).
            for stage in &config.stages {
                for mb in &mut micro_batches {
                    *mb = gpu_stage_forward(mb, stage)?;
                }
            }
        }
        GpuPipelineSchedule::Interleaved => {
            for stage in &config.stages {
                for mb in &mut micro_batches {
                    *mb = gpu_stage_forward(mb, stage)?;
                }
            }
        }
    }

    gpu_micro_batch_merge(&micro_batches)
}

// ── Pipeline metrics ───────────────────────────────────────────────

/// Latency and throughput estimates for a pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    /// Estimated time for a single stage in the same units as `stage_latency`.
    pub stage_latency: f64,
    /// Number of pipeline stages.
    pub num_stages: usize,
    /// Number of micro-batches.
    pub num_micro_batches: usize,
    /// Schedule type.
    pub schedule: GpuPipelineSchedule,
}

impl PipelineMetrics {
    /// Create metrics from pipeline parameters.
    pub fn new(
        stage_latency: f64,
        num_stages: usize,
        num_micro_batches: usize,
        schedule: GpuPipelineSchedule,
    ) -> Result<Self> {
        if stage_latency < 0.0 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: "stage_latency must be non-negative".into(),
            }));
        }
        if num_stages == 0 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: "num_stages must be > 0".into(),
            }));
        }
        Ok(Self { stage_latency, num_stages, num_micro_batches, schedule })
    }

    /// Pipeline bubble fraction: `(p-1) / (m+p-1)`.
    pub fn bubble_fraction(&self) -> f64 {
        gpu_pipeline_bubble_fraction(self.num_stages, self.num_micro_batches)
    }

    /// Total pipeline latency for all micro-batches.
    ///
    /// `latency = stage_latency × (m + p - 1)` for GPipe/1F1B.
    pub fn total_latency(&self) -> f64 {
        if self.num_micro_batches == 0 {
            return 0.0;
        }
        let p = self.num_stages as f64;
        let m = self.num_micro_batches as f64;
        self.stage_latency * (m + p - 1.0)
    }

    /// Throughput in micro-batches per unit time.
    pub fn throughput(&self) -> f64 {
        let lat = self.total_latency();
        if lat <= 0.0 {
            return 0.0;
        }
        self.num_micro_batches as f64 / lat
    }

    /// Pipeline efficiency (1.0 - bubble_fraction).
    pub fn efficiency(&self) -> f64 {
        1.0 - self.bubble_fraction()
    }
}

// ── Bubble-time estimation ─────────────────────────────────────────

/// Compute the pipeline bubble fraction for a GPU pipeline.
///
/// For `p` stages and `m` micro-batches: `(p-1) / (m+p-1)`.
/// Returns 0.0 when the pipeline is degenerate (≤ 1 stage or 0 micro-batches).
pub fn gpu_pipeline_bubble_fraction(num_stages: usize, num_micro_batches: usize) -> f64 {
    if num_stages <= 1 || num_micro_batches == 0 {
        return 0.0;
    }
    let p = num_stages as f64;
    let m = num_micro_batches as f64;
    (p - 1.0) / (m + p - 1.0)
}

/// Compute the optimal number of micro-batches to keep the bubble
/// fraction below `max_bubble_fraction`.
///
/// Derived from `(p-1)/(m+p-1) ≤ f  =>  m ≥ (p-1)*(1-f)/f`.
pub fn gpu_optimal_micro_batch_count(num_stages: usize, max_bubble_fraction: f64) -> usize {
    if num_stages <= 1 {
        return 1;
    }
    if max_bubble_fraction <= 0.0 || max_bubble_fraction > 1.0 {
        return num_stages;
    }
    let p = (num_stages - 1) as f64;
    let m = (p * (1.0 - max_bubble_fraction) / max_bubble_fraction).ceil() as usize;
    m.max(1)
}

// ── Gradient accumulation ──────────────────────────────────────────

/// Gradient accumulator for training pipeline micro-batches.
///
/// Accumulates gradients from multiple micro-batches before applying an
/// optimizer step.  In a real GPU implementation the accumulation would
/// happen in device memory via fused CUDA kernels; this CPU fallback uses
/// `Vec<f32>`.
#[derive(Debug, Clone)]
pub struct GradAccumulator {
    /// Accumulated gradient buffer (same shape as parameters).
    pub buffer: Vec<f32>,
    /// Number of micro-batches accumulated so far.
    pub accumulated_count: usize,
    /// Target number of micro-batches before an optimizer step.
    pub target_count: usize,
}

impl GradAccumulator {
    /// Create a new accumulator for `param_count` parameters.
    pub fn new(param_count: usize, target_count: usize) -> Result<Self> {
        if param_count == 0 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: "param_count must be > 0".into(),
            }));
        }
        if target_count == 0 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: "target_count must be > 0".into(),
            }));
        }
        Ok(Self { buffer: vec![0.0; param_count], accumulated_count: 0, target_count })
    }

    /// Accumulate a gradient from a micro-batch.
    pub fn accumulate(&mut self, gradients: &[f32]) -> Result<()> {
        if gradients.len() != self.buffer.len() {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!(
                    "gradient length {} != accumulator size {}",
                    gradients.len(),
                    self.buffer.len(),
                ),
            }));
        }
        for (acc, &g) in self.buffer.iter_mut().zip(gradients.iter()) {
            *acc += g;
        }
        self.accumulated_count += 1;
        Ok(())
    }

    /// Whether enough micro-batches have been accumulated for an optimizer step.
    pub fn is_ready(&self) -> bool {
        self.accumulated_count >= self.target_count
    }

    /// Return the mean gradient (accumulated / count) and reset the accumulator.
    pub fn finalize(&mut self) -> Result<Vec<f32>> {
        if self.accumulated_count == 0 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: "no gradients have been accumulated".into(),
            }));
        }
        let scale = 1.0 / self.accumulated_count as f32;
        let mean: Vec<f32> = self.buffer.iter().map(|&v| v * scale).collect();
        self.reset();
        Ok(mean)
    }

    /// Reset the accumulator to zero.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.accumulated_count = 0;
    }

    /// Number of parameters tracked.
    pub fn param_count(&self) -> usize {
        self.buffer.len()
    }
}

// ── Pipeline CUDA kernel source (GPU-only) ─────────────────────────

/// Placeholder CUDA kernel source for pipeline stage synchronisation.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const PIPELINE_SYNC_KERNEL_SRC: &str = r#"
extern "C" __global__
void pipeline_sync_barrier(float* __restrict__ dst,
                           const float* __restrict__ src,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}
"#;

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── GpuPipelineSchedule ────────────────────────────────────────

    #[test]
    fn test_schedule_display() {
        assert_eq!(GpuPipelineSchedule::Sequential.to_string(), "Sequential");
        assert_eq!(GpuPipelineSchedule::GPipe.to_string(), "GPipe");
        assert_eq!(GpuPipelineSchedule::OneF1B.to_string(), "1F1B");
        assert_eq!(GpuPipelineSchedule::Interleaved.to_string(), "Interleaved");
    }

    #[test]
    fn test_schedule_default() {
        assert_eq!(GpuPipelineSchedule::default(), GpuPipelineSchedule::GPipe);
    }

    #[test]
    fn test_schedule_eq() {
        assert_eq!(GpuPipelineSchedule::GPipe, GpuPipelineSchedule::GPipe);
        assert_ne!(GpuPipelineSchedule::GPipe, GpuPipelineSchedule::Sequential);
    }

    #[test]
    fn test_schedule_clone() {
        let s = GpuPipelineSchedule::OneF1B;
        let s2 = s;
        assert_eq!(s, s2);
    }

    // ── StageResources ─────────────────────────────────────────────

    #[test]
    fn test_resources_zero() {
        let r = StageResources::zero();
        assert_eq!(r.memory_bytes, 0);
        assert_eq!(r.compute_cost, 0.0);
        assert_eq!(r.active_tensors, 0);
    }

    #[test]
    fn test_resources_new() {
        let r = StageResources::new(1024, 100.0, 5);
        assert_eq!(r.memory_bytes, 1024);
        assert_eq!(r.compute_cost, 100.0);
        assert_eq!(r.active_tensors, 5);
    }

    #[test]
    fn test_resources_validate_ok() {
        StageResources::new(1024, 0.0, 0).validate().unwrap();
    }

    #[test]
    fn test_resources_validate_negative_compute() {
        assert!(StageResources::new(0, -1.0, 0).validate().is_err());
    }

    #[test]
    fn test_resources_default() {
        let r = StageResources::default();
        assert_eq!(r.memory_bytes, 0);
    }

    // ── GpuPipelineStage ───────────────────────────────────────────

    #[test]
    fn test_stage_new() {
        let s = GpuPipelineStage::new(0, 4);
        assert_eq!(s.start_layer, 0);
        assert_eq!(s.end_layer, 4);
        assert_eq!(s.stream_id, None);
    }

    #[test]
    fn test_stage_with_stream() {
        let s = GpuPipelineStage::new(0, 4).with_stream(2);
        assert_eq!(s.stream_id, Some(2));
    }

    #[test]
    fn test_stage_with_resources() {
        let res = StageResources::new(2048, 50.0, 3);
        let s = GpuPipelineStage::new(0, 4).with_resources(res);
        assert_eq!(s.resources.memory_bytes, 2048);
    }

    #[test]
    fn test_stage_num_layers() {
        assert_eq!(GpuPipelineStage::new(0, 4).num_layers(), 4);
        assert_eq!(GpuPipelineStage::new(4, 8).num_layers(), 4);
        assert_eq!(GpuPipelineStage::new(0, 1).num_layers(), 1);
    }

    #[test]
    fn test_stage_validate_ok() {
        GpuPipelineStage::new(0, 4).validate().unwrap();
    }

    #[test]
    fn test_stage_validate_empty_range() {
        assert!(GpuPipelineStage::new(4, 4).validate().is_err());
    }

    #[test]
    fn test_stage_validate_inverted_range() {
        assert!(GpuPipelineStage::new(8, 4).validate().is_err());
    }

    #[test]
    fn test_stage_validate_bad_resources() {
        let s = GpuPipelineStage::new(0, 4).with_resources(StageResources::new(0, -10.0, 0));
        assert!(s.validate().is_err());
    }

    // ── InterStageBuffer ───────────────────────────────────────────

    #[test]
    fn test_buffer_new() {
        let buf = InterStageBuffer::new(0, 0, 1, 16);
        assert_eq!(buf.id, 0);
        assert_eq!(buf.producer_stage, 0);
        assert_eq!(buf.consumer_stage, 1);
        assert_eq!(buf.capacity, 16);
        assert!(!buf.ready);
    }

    #[test]
    fn test_buffer_write_read() {
        let mut buf = InterStageBuffer::new(0, 0, 1, 4);
        buf.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(buf.ready);
        let data = buf.read().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(!buf.ready);
    }

    #[test]
    fn test_buffer_write_exceeds_capacity() {
        let mut buf = InterStageBuffer::new(0, 0, 1, 2);
        assert!(buf.write(&[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn test_buffer_read_not_ready() {
        let mut buf = InterStageBuffer::new(0, 0, 1, 4);
        assert!(buf.read().is_err());
    }

    #[test]
    fn test_buffer_reset() {
        let mut buf = InterStageBuffer::new(0, 0, 1, 4);
        buf.write(&[1.0, 2.0]).unwrap();
        buf.reset();
        assert!(!buf.ready);
        assert!(buf.data.is_empty());
    }

    #[test]
    fn test_buffer_write_read_cycle() {
        let mut buf = InterStageBuffer::new(0, 0, 1, 8);
        for i in 0..3 {
            let data: Vec<f32> = (0..4).map(|x| (x + i) as f32).collect();
            buf.write(&data).unwrap();
            let out = buf.read().unwrap();
            assert_eq!(out, data);
        }
    }

    #[test]
    fn test_buffer_empty_write() {
        let mut buf = InterStageBuffer::new(0, 0, 1, 4);
        buf.write(&[]).unwrap();
        assert!(buf.ready);
        let data = buf.read().unwrap();
        assert!(data.is_empty());
    }

    // ── GpuPipelineConfig ──────────────────────────────────────────

    #[test]
    fn test_config_num_stages() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 4), GpuPipelineStage::new(4, 8)],
            2,
            GpuPipelineSchedule::GPipe,
        );
        assert_eq!(cfg.num_stages(), 2);
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 4), GpuPipelineStage::new(4, 8)],
            2,
            GpuPipelineSchedule::GPipe,
        );
        cfg.validate().unwrap();
    }

    #[test]
    fn test_config_validate_empty_stages() {
        let cfg = GpuPipelineConfig::new(vec![], 2, GpuPipelineSchedule::GPipe);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_micro_batch() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 4)],
            0,
            GpuPipelineSchedule::GPipe,
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_non_contiguous() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 4), GpuPipelineStage::new(5, 8)],
            2,
            GpuPipelineSchedule::GPipe,
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_stage() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(4, 4)],
            1,
            GpuPipelineSchedule::GPipe,
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_streams() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 4)],
            1,
            GpuPipelineSchedule::GPipe,
        )
        .with_num_streams(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_stream_out_of_bounds() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 4).with_stream(5)],
            1,
            GpuPipelineSchedule::GPipe,
        )
        .with_num_streams(2);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_with_grad_accumulation() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 4)],
            1,
            GpuPipelineSchedule::OneF1B,
        )
        .with_grad_accumulation(true);
        assert!(cfg.grad_accumulation);
    }

    #[test]
    fn test_config_stream_assignment_auto() {
        let cfg = GpuPipelineConfig::new(
            vec![
                GpuPipelineStage::new(0, 4),
                GpuPipelineStage::new(4, 8),
                GpuPipelineStage::new(8, 12),
            ],
            2,
            GpuPipelineSchedule::GPipe,
        )
        .with_num_streams(2);
        let a = cfg.effective_stream_assignment();
        assert_eq!(a, vec![0, 1, 0]);
    }

    #[test]
    fn test_config_stream_assignment_explicit() {
        let cfg = GpuPipelineConfig::new(
            vec![
                GpuPipelineStage::new(0, 4).with_stream(1),
                GpuPipelineStage::new(4, 8).with_stream(0),
            ],
            2,
            GpuPipelineSchedule::GPipe,
        )
        .with_num_streams(2);
        let a = cfg.effective_stream_assignment();
        assert_eq!(a, vec![1, 0]);
    }

    #[test]
    fn test_config_total_memory() {
        let cfg = GpuPipelineConfig::new(
            vec![
                GpuPipelineStage::new(0, 4).with_resources(StageResources::new(100, 0.0, 0)),
                GpuPipelineStage::new(4, 8).with_resources(StageResources::new(200, 0.0, 0)),
            ],
            2,
            GpuPipelineSchedule::GPipe,
        );
        assert_eq!(cfg.total_memory(), 300);
    }

    #[test]
    fn test_config_total_compute() {
        let cfg = GpuPipelineConfig::new(
            vec![
                GpuPipelineStage::new(0, 4).with_resources(StageResources::new(0, 10.0, 0)),
                GpuPipelineStage::new(4, 8).with_resources(StageResources::new(0, 20.5, 0)),
            ],
            2,
            GpuPipelineSchedule::GPipe,
        );
        assert!((cfg.total_compute() - 30.5).abs() < 1e-9);
    }

    #[test]
    fn test_config_peak_stage_memory() {
        let cfg = GpuPipelineConfig::new(
            vec![
                GpuPipelineStage::new(0, 4).with_resources(StageResources::new(100, 0.0, 0)),
                GpuPipelineStage::new(4, 8).with_resources(StageResources::new(500, 0.0, 0)),
                GpuPipelineStage::new(8, 12).with_resources(StageResources::new(200, 0.0, 0)),
            ],
            2,
            GpuPipelineSchedule::GPipe,
        );
        assert_eq!(cfg.peak_stage_memory(), 500);
    }

    // ── gpu_micro_batch_split ──────────────────────────────────────

    #[test]
    fn test_split_exact() {
        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let batches = gpu_micro_batch_split(&input, 4, 3, 2).unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(batches[1], &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_split_remainder() {
        let input: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let batches = gpu_micro_batch_split(&input, 5, 3, 2).unwrap();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[2].len(), 3); // last chunk: 1 row
    }

    #[test]
    fn test_split_single_row() {
        let input = vec![1.0, 2.0, 3.0];
        let batches = gpu_micro_batch_split(&input, 1, 3, 1).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_split_micro_batch_larger_than_batch() {
        let input = vec![1.0, 2.0];
        let batches = gpu_micro_batch_split(&input, 2, 1, 10).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], &[1.0, 2.0]);
    }

    #[test]
    fn test_split_empty_input() {
        assert!(gpu_micro_batch_split(&[], 0, 4, 2).is_err());
    }

    #[test]
    fn test_split_zero_dim() {
        assert!(gpu_micro_batch_split(&[1.0], 1, 0, 1).is_err());
    }

    #[test]
    fn test_split_zero_micro_batch() {
        assert!(gpu_micro_batch_split(&[1.0], 1, 1, 0).is_err());
    }

    #[test]
    fn test_split_mismatched_len() {
        assert!(gpu_micro_batch_split(&[1.0, 2.0], 1, 3, 1).is_err());
    }

    // ── gpu_micro_batch_merge ──────────────────────────────────────

    #[test]
    fn test_merge_basic() {
        let batches = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let merged = gpu_micro_batch_merge(&batches).unwrap();
        assert_eq!(merged, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_merge_single() {
        let batches = vec![vec![5.0, 6.0, 7.0]];
        let merged = gpu_micro_batch_merge(&batches).unwrap();
        assert_eq!(merged, vec![5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_merge_empty() {
        let batches: Vec<Vec<f32>> = vec![];
        assert!(gpu_micro_batch_merge(&batches).is_err());
    }

    #[test]
    fn test_split_merge_roundtrip() {
        let input: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let batches = gpu_micro_batch_split(&input, 8, 3, 3).unwrap();
        let merged = gpu_micro_batch_merge(&batches).unwrap();
        assert_eq!(merged, input);
    }

    #[test]
    fn test_split_merge_roundtrip_remainder() {
        let input: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let batches = gpu_micro_batch_split(&input, 5, 4, 2).unwrap();
        let merged = gpu_micro_batch_merge(&batches).unwrap();
        assert_eq!(merged, input);
    }

    // ── gpu_stage_forward ──────────────────────────────────────────

    #[test]
    fn test_stage_forward_basic() {
        let stage = GpuPipelineStage::new(0, 3);
        let input = vec![1.0, 2.0, 3.0];
        let out = gpu_stage_forward(&input, &stage).unwrap();
        assert_eq!(out, vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_stage_forward_single_layer() {
        let stage = GpuPipelineStage::new(0, 1);
        let input = vec![5.0, 10.0];
        let out = gpu_stage_forward(&input, &stage).unwrap();
        assert_eq!(out, vec![5.0, 10.0]);
    }

    #[test]
    fn test_stage_forward_empty_input() {
        let stage = GpuPipelineStage::new(0, 2);
        assert!(gpu_stage_forward(&[], &stage).is_err());
    }

    #[test]
    fn test_stage_forward_invalid_stage() {
        assert!(gpu_stage_forward(&[1.0], &GpuPipelineStage::new(4, 4)).is_err());
    }

    // ── gpu_pipeline_forward (single stage) ────────────────────────

    #[test]
    fn test_forward_single_stage_sequential() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 2)],
            4,
            GpuPipelineSchedule::Sequential,
        );
        let input = vec![1.0; 8]; // 4 rows × 2 dims
        let out = gpu_pipeline_forward(&input, 4, 2, &cfg).unwrap();
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_single_stage_gpipe() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 3)],
            2,
            GpuPipelineSchedule::GPipe,
        );
        let input = vec![2.0; 6];
        let out = gpu_pipeline_forward(&input, 3, 2, &cfg).unwrap();
        assert!(out.iter().all(|&v| (v - 6.0).abs() < 1e-6));
    }

    // ── gpu_pipeline_forward (multi-stage) ─────────────────────────

    fn two_stage_config(schedule: GpuPipelineSchedule) -> GpuPipelineConfig {
        GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 2), GpuPipelineStage::new(2, 5)],
            2,
            schedule,
        )
    }

    #[test]
    fn test_forward_two_stage_sequential() {
        let cfg = two_stage_config(GpuPipelineSchedule::Sequential);
        let input = vec![1.0; 8];
        let out = gpu_pipeline_forward(&input, 4, 2, &cfg).unwrap();
        assert!(out.iter().all(|&v| (v - 6.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_two_stage_gpipe() {
        let cfg = two_stage_config(GpuPipelineSchedule::GPipe);
        let input = vec![1.0; 8];
        let out = gpu_pipeline_forward(&input, 4, 2, &cfg).unwrap();
        assert!(out.iter().all(|&v| (v - 6.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_two_stage_1f1b() {
        let cfg = two_stage_config(GpuPipelineSchedule::OneF1B);
        let input = vec![1.0; 8];
        let out = gpu_pipeline_forward(&input, 4, 2, &cfg).unwrap();
        assert!(out.iter().all(|&v| (v - 6.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_two_stage_interleaved() {
        let cfg = two_stage_config(GpuPipelineSchedule::Interleaved);
        let input = vec![1.0; 8];
        let out = gpu_pipeline_forward(&input, 4, 2, &cfg).unwrap();
        assert!(out.iter().all(|&v| (v - 6.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_three_stages() {
        let cfg = GpuPipelineConfig::new(
            vec![
                GpuPipelineStage::new(0, 2),
                GpuPipelineStage::new(2, 4),
                GpuPipelineStage::new(4, 7),
            ],
            2,
            GpuPipelineSchedule::GPipe,
        );
        let input = vec![1.0; 6]; // 3×2
        let out = gpu_pipeline_forward(&input, 3, 2, &cfg).unwrap();
        // 1 * 2 * 2 * 3 = 12
        assert!(out.iter().all(|&v| (v - 12.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_four_stages() {
        let cfg = GpuPipelineConfig::new(
            vec![
                GpuPipelineStage::new(0, 1),
                GpuPipelineStage::new(1, 2),
                GpuPipelineStage::new(2, 3),
                GpuPipelineStage::new(3, 4),
            ],
            1,
            GpuPipelineSchedule::Sequential,
        );
        let input = vec![2.0; 4]; // 2×2
        let out = gpu_pipeline_forward(&input, 2, 2, &cfg).unwrap();
        assert!(out.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    // ── gpu_pipeline_forward (error cases) ─────────────────────────

    #[test]
    fn test_forward_empty_input() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 2)],
            1,
            GpuPipelineSchedule::GPipe,
        );
        assert!(gpu_pipeline_forward(&[], 0, 2, &cfg).is_err());
    }

    #[test]
    fn test_forward_mismatched_dims() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 2)],
            1,
            GpuPipelineSchedule::GPipe,
        );
        assert!(gpu_pipeline_forward(&[1.0, 2.0], 1, 3, &cfg).is_err());
    }

    #[test]
    fn test_forward_zero_stages() {
        let cfg = GpuPipelineConfig::new(vec![], 1, GpuPipelineSchedule::GPipe);
        assert!(gpu_pipeline_forward(&[1.0], 1, 1, &cfg).is_err());
    }

    // ── gpu_pipeline_forward (various sizes) ───────────────────────

    #[test]
    fn test_forward_single_element() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 2)],
            1,
            GpuPipelineSchedule::GPipe,
        );
        let out = gpu_pipeline_forward(&[3.0], 1, 1, &cfg).unwrap();
        assert_eq!(out, vec![6.0]);
    }

    #[test]
    fn test_forward_large_batch() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 1), GpuPipelineStage::new(1, 2)],
            4,
            GpuPipelineSchedule::GPipe,
        );
        let input = vec![1.0; 128]; // 64×2
        let out = gpu_pipeline_forward(&input, 64, 2, &cfg).unwrap();
        assert_eq!(out.len(), 128);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_micro_batch_one() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 2), GpuPipelineStage::new(2, 4)],
            1,
            GpuPipelineSchedule::OneF1B,
        );
        let input = vec![1.0; 6]; // 3×2
        let out = gpu_pipeline_forward(&input, 3, 2, &cfg).unwrap();
        assert!(out.iter().all(|&v| (v - 4.0).abs() < 1e-6));
    }

    // ── all schedules produce same result ──────────────────────────

    #[test]
    fn test_all_schedules_same_result() {
        let stages = vec![GpuPipelineStage::new(0, 2), GpuPipelineStage::new(2, 5)];
        let input = vec![1.0; 12]; // 6×2
        let mut results = Vec::new();
        for sched in [
            GpuPipelineSchedule::Sequential,
            GpuPipelineSchedule::GPipe,
            GpuPipelineSchedule::OneF1B,
            GpuPipelineSchedule::Interleaved,
        ] {
            let cfg = GpuPipelineConfig::new(stages.clone(), 2, sched);
            results.push(gpu_pipeline_forward(&input, 6, 2, &cfg).unwrap());
        }
        for r in &results[1..] {
            assert_eq!(r, &results[0]);
        }
    }

    #[test]
    fn test_forward_preserves_output_length() {
        let cfg = GpuPipelineConfig::new(
            vec![GpuPipelineStage::new(0, 3), GpuPipelineStage::new(3, 6)],
            3,
            GpuPipelineSchedule::GPipe,
        );
        let input = vec![1.0; 30]; // 10×3
        let out = gpu_pipeline_forward(&input, 10, 3, &cfg).unwrap();
        assert_eq!(out.len(), 30);
    }

    #[test]
    fn test_forward_non_uniform_stage_sizes() {
        let cfg = GpuPipelineConfig::new(
            vec![
                GpuPipelineStage::new(0, 1),  // 1 layer
                GpuPipelineStage::new(1, 10), // 9 layers
            ],
            2,
            GpuPipelineSchedule::GPipe,
        );
        let input = vec![1.0; 6]; // 3×2
        let out = gpu_pipeline_forward(&input, 3, 2, &cfg).unwrap();
        assert!(out.iter().all(|&v| (v - 9.0).abs() < 1e-6));
    }

    // ── gpu_pipeline_bubble_fraction ───────────────────────────────

    #[test]
    fn test_bubble_single_stage() {
        assert_eq!(gpu_pipeline_bubble_fraction(1, 4), 0.0);
    }

    #[test]
    fn test_bubble_zero_micro_batches() {
        assert_eq!(gpu_pipeline_bubble_fraction(4, 0), 0.0);
    }

    #[test]
    fn test_bubble_two_stages_four_micro() {
        let b = gpu_pipeline_bubble_fraction(2, 4);
        assert!((b - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_bubble_four_stages_four_micro() {
        let b = gpu_pipeline_bubble_fraction(4, 4);
        assert!((b - 3.0 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_bubble_many_micro_batches() {
        let b = gpu_pipeline_bubble_fraction(4, 100);
        assert!((b - 3.0 / 103.0).abs() < 1e-6);
    }

    #[test]
    fn test_bubble_one_micro_batch() {
        let b = gpu_pipeline_bubble_fraction(4, 1);
        assert!((b - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_bubble_zero_stages() {
        assert_eq!(gpu_pipeline_bubble_fraction(0, 10), 0.0);
    }

    #[test]
    fn test_bubble_decreases_with_more_micro_batches() {
        let b1 = gpu_pipeline_bubble_fraction(4, 2);
        let b2 = gpu_pipeline_bubble_fraction(4, 8);
        let b3 = gpu_pipeline_bubble_fraction(4, 32);
        assert!(b1 > b2);
        assert!(b2 > b3);
    }

    #[test]
    fn test_bubble_increases_with_more_stages() {
        let b1 = gpu_pipeline_bubble_fraction(2, 8);
        let b2 = gpu_pipeline_bubble_fraction(4, 8);
        let b3 = gpu_pipeline_bubble_fraction(8, 8);
        assert!(b1 < b2);
        assert!(b2 < b3);
    }

    // ── gpu_optimal_micro_batch_count ──────────────────────────────

    #[test]
    fn test_optimal_single_stage() {
        assert_eq!(gpu_optimal_micro_batch_count(1, 0.1), 1);
    }

    #[test]
    fn test_optimal_two_stages_10pct() {
        assert_eq!(gpu_optimal_micro_batch_count(2, 0.1), 9);
    }

    #[test]
    fn test_optimal_four_stages_10pct() {
        assert_eq!(gpu_optimal_micro_batch_count(4, 0.1), 27);
    }

    #[test]
    fn test_optimal_four_stages_50pct() {
        assert_eq!(gpu_optimal_micro_batch_count(4, 0.5), 3);
    }

    #[test]
    fn test_optimal_zero_fraction() {
        assert_eq!(gpu_optimal_micro_batch_count(4, 0.0), 4);
    }

    #[test]
    fn test_optimal_negative_fraction() {
        assert_eq!(gpu_optimal_micro_batch_count(4, -0.5), 4);
    }

    #[test]
    fn test_optimal_fraction_above_one() {
        assert_eq!(gpu_optimal_micro_batch_count(4, 1.5), 4);
    }

    #[test]
    fn test_optimal_fraction_exactly_one() {
        assert_eq!(gpu_optimal_micro_batch_count(4, 1.0), 1);
    }

    #[test]
    fn test_optimal_returns_at_least_one() {
        assert!(gpu_optimal_micro_batch_count(8, 0.5) >= 1);
        assert!(gpu_optimal_micro_batch_count(1, 0.5) >= 1);
    }

    #[test]
    fn test_optimal_satisfies_bubble_constraint() {
        for stages in 2..=8 {
            let frac = 0.15_f64;
            let m = gpu_optimal_micro_batch_count(stages, frac);
            let actual = gpu_pipeline_bubble_fraction(stages, m);
            assert!(
                actual <= frac + 1e-6,
                "stages={stages}, m={m}, actual bubble={actual}, limit={frac}"
            );
        }
    }

    // ── 1F1B schedule generation ───────────────────────────────────

    #[test]
    fn test_1f1b_schedule_basic() {
        let schedule = generate_1f1b_schedule(2, 4).unwrap();
        assert!(!schedule.is_empty());
        // Every step has one action per stage.
        for step in &schedule {
            assert_eq!(step.len(), 2);
        }
    }

    #[test]
    fn test_1f1b_schedule_zero_stages() {
        assert!(generate_1f1b_schedule(0, 4).is_err());
    }

    #[test]
    fn test_1f1b_schedule_zero_micro_batches() {
        assert!(generate_1f1b_schedule(2, 0).is_err());
    }

    #[test]
    fn test_1f1b_schedule_single_stage() {
        let schedule = generate_1f1b_schedule(1, 3).unwrap();
        for step in &schedule {
            assert_eq!(step.len(), 1);
        }
    }

    #[test]
    fn test_1f1b_all_forwards_present() {
        let schedule = generate_1f1b_schedule(3, 4).unwrap();
        for stage_id in 0..3 {
            for mb in 0..4 {
                let found = schedule.iter().any(|step| {
                    step.iter().any(|a| {
                        matches!(a, ScheduleAction::Forward { stage_id: s, micro_batch_id: m }
                            if *s == stage_id && *m == mb)
                    })
                });
                assert!(found, "missing Forward(stage={stage_id}, mb={mb})");
            }
        }
    }

    #[test]
    fn test_1f1b_all_backwards_present() {
        let schedule = generate_1f1b_schedule(3, 4).unwrap();
        for stage_id in 0..3 {
            for mb in 0..4 {
                let found = schedule.iter().any(|step| {
                    step.iter().any(|a| {
                        matches!(a, ScheduleAction::Backward { stage_id: s, micro_batch_id: m }
                            if *s == stage_id && *m == mb)
                    })
                });
                assert!(found, "missing Backward(stage={stage_id}, mb={mb})");
            }
        }
    }

    #[test]
    fn test_1f1b_forward_before_backward() {
        let schedule = generate_1f1b_schedule(2, 3).unwrap();
        for stage_id in 0..2 {
            for mb in 0..3 {
                let fwd_step = schedule.iter().position(|step| {
                    step.iter().any(|a| {
                        matches!(a, ScheduleAction::Forward { stage_id: s, micro_batch_id: m }
                            if *s == stage_id && *m == mb)
                    })
                });
                let bwd_step = schedule.iter().position(|step| {
                    step.iter().any(|a| {
                        matches!(a, ScheduleAction::Backward { stage_id: s, micro_batch_id: m }
                            if *s == stage_id && *m == mb)
                    })
                });
                assert!(
                    fwd_step.unwrap() < bwd_step.unwrap(),
                    "Forward must precede Backward for stage={stage_id}, mb={mb}"
                );
            }
        }
    }

    #[test]
    fn test_1f1b_bubble_count() {
        let schedule = generate_1f1b_schedule(2, 4).unwrap();
        let bubbles: usize = schedule
            .iter()
            .flat_map(|step| step.iter())
            .filter(|a| matches!(a, ScheduleAction::Bubble { .. }))
            .count();
        // Bubbles should be minimal for 2 stages / 4 micro-batches.
        assert!(bubbles < 2 * 4, "too many bubbles: {bubbles}");
    }

    // ── PipelineMetrics ────────────────────────────────────────────

    #[test]
    fn test_metrics_bubble_fraction() {
        let m = PipelineMetrics::new(1.0, 4, 8, GpuPipelineSchedule::GPipe).unwrap();
        let expected = 3.0 / 11.0;
        assert!((m.bubble_fraction() - expected).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_total_latency() {
        let m = PipelineMetrics::new(2.0, 3, 5, GpuPipelineSchedule::OneF1B).unwrap();
        // 2.0 * (5 + 3 - 1) = 2.0 * 7 = 14.0
        assert!((m.total_latency() - 14.0).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_throughput() {
        let m = PipelineMetrics::new(1.0, 2, 10, GpuPipelineSchedule::GPipe).unwrap();
        // lat = 1.0 * (10 + 2 - 1) = 11.0, tp = 10 / 11
        assert!((m.throughput() - 10.0 / 11.0).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_efficiency() {
        let m = PipelineMetrics::new(1.0, 4, 8, GpuPipelineSchedule::GPipe).unwrap();
        assert!((m.efficiency() - (1.0 - 3.0 / 11.0)).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_zero_micro_batches() {
        let m = PipelineMetrics::new(1.0, 2, 0, GpuPipelineSchedule::GPipe).unwrap();
        assert_eq!(m.total_latency(), 0.0);
        assert_eq!(m.throughput(), 0.0);
    }

    #[test]
    fn test_metrics_negative_latency() {
        assert!(PipelineMetrics::new(-1.0, 2, 4, GpuPipelineSchedule::GPipe).is_err());
    }

    #[test]
    fn test_metrics_zero_stages() {
        assert!(PipelineMetrics::new(1.0, 0, 4, GpuPipelineSchedule::GPipe).is_err());
    }

    // ── GradAccumulator ────────────────────────────────────────────

    #[test]
    fn test_grad_new() {
        let acc = GradAccumulator::new(10, 4).unwrap();
        assert_eq!(acc.param_count(), 10);
        assert_eq!(acc.accumulated_count, 0);
        assert!(!acc.is_ready());
    }

    #[test]
    fn test_grad_new_zero_params() {
        assert!(GradAccumulator::new(0, 4).is_err());
    }

    #[test]
    fn test_grad_new_zero_target() {
        assert!(GradAccumulator::new(10, 0).is_err());
    }

    #[test]
    fn test_grad_accumulate_single() {
        let mut acc = GradAccumulator::new(3, 2).unwrap();
        acc.accumulate(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(acc.accumulated_count, 1);
        assert_eq!(acc.buffer, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_grad_accumulate_multiple() {
        let mut acc = GradAccumulator::new(2, 3).unwrap();
        acc.accumulate(&[1.0, 2.0]).unwrap();
        acc.accumulate(&[3.0, 4.0]).unwrap();
        assert_eq!(acc.accumulated_count, 2);
        assert_eq!(acc.buffer, vec![4.0, 6.0]);
    }

    #[test]
    fn test_grad_accumulate_wrong_size() {
        let mut acc = GradAccumulator::new(3, 2).unwrap();
        assert!(acc.accumulate(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_grad_is_ready() {
        let mut acc = GradAccumulator::new(2, 2).unwrap();
        assert!(!acc.is_ready());
        acc.accumulate(&[1.0, 1.0]).unwrap();
        assert!(!acc.is_ready());
        acc.accumulate(&[1.0, 1.0]).unwrap();
        assert!(acc.is_ready());
    }

    #[test]
    fn test_grad_finalize() {
        let mut acc = GradAccumulator::new(2, 2).unwrap();
        acc.accumulate(&[2.0, 4.0]).unwrap();
        acc.accumulate(&[6.0, 8.0]).unwrap();
        let mean = acc.finalize().unwrap();
        assert!((mean[0] - 4.0).abs() < 1e-6);
        assert!((mean[1] - 6.0).abs() < 1e-6);
        assert_eq!(acc.accumulated_count, 0);
    }

    #[test]
    fn test_grad_finalize_empty() {
        let mut acc = GradAccumulator::new(2, 2).unwrap();
        assert!(acc.finalize().is_err());
    }

    #[test]
    fn test_grad_reset() {
        let mut acc = GradAccumulator::new(3, 2).unwrap();
        acc.accumulate(&[1.0, 2.0, 3.0]).unwrap();
        acc.reset();
        assert_eq!(acc.accumulated_count, 0);
        assert!(acc.buffer.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_grad_accumulate_finalize_cycle() {
        let mut acc = GradAccumulator::new(2, 2).unwrap();
        // First cycle
        acc.accumulate(&[1.0, 1.0]).unwrap();
        acc.accumulate(&[3.0, 3.0]).unwrap();
        let m1 = acc.finalize().unwrap();
        assert!((m1[0] - 2.0).abs() < 1e-6);
        // Second cycle
        acc.accumulate(&[10.0, 20.0]).unwrap();
        acc.accumulate(&[30.0, 40.0]).unwrap();
        let m2 = acc.finalize().unwrap();
        assert!((m2[0] - 20.0).abs() < 1e-6);
        assert!((m2[1] - 30.0).abs() < 1e-6);
    }
}
