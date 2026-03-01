//! Module stub - implementation pending merge from feature branch
//! Gradient checkpointing for memory-efficient training and fine-tuning.
//!
//! Trades compute for memory by selectively storing activations during the
//! forward pass and recomputing the rest during the backward pass. Supports
//! multiple checkpoint strategies (every-N, √N, optimal, custom), optional
//! activation compression, and profiling of the recompute-vs-memory tradeoff.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_const_for_fn,
    clippy::return_self_not_must_use
)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Top-level configuration for gradient checkpointing.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Which layers to checkpoint.
    pub strategy: CheckpointStrategy,
    /// Number of layers grouped into a single segment.
    pub segment_size: usize,
    /// Maximum memory budget for stored activations (bytes).
    pub memory_budget: usize,
    /// Whether to compress stored activations.
    pub enable_compression: bool,
    /// Compression level when compression is enabled (1–9).
    pub compression_level: u32,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            strategy: CheckpointStrategy::Sqrt,
            segment_size: 4,
            memory_budget: 1024 * 1024 * 512, // 512 MiB
            enable_compression: false,
            compression_level: 3,
        }
    }
}

impl CheckpointConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), String> {
        if self.segment_size == 0 {
            return Err("segment_size must be > 0".into());
        }
        if self.memory_budget == 0 {
            return Err("memory_budget must be > 0".into());
        }
        if self.compression_level == 0 || self.compression_level > 9 {
            return Err("compression_level must be in [1, 9]".into());
        }
        self.strategy.validate()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Checkpoint strategy
// ---------------------------------------------------------------------------

/// Strategy that selects which layer boundaries are checkpoint boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointStrategy {
    /// No checkpointing – store all activations.
    None,
    /// Checkpoint every `n` layers.
    Every(usize),
    /// Checkpoint roughly every √N layers (balanced compute/memory).
    Sqrt,
    /// Automatically find the optimal set of checkpoints for budget.
    Optimal,
    /// User-supplied list of layer indices to checkpoint.
    Custom(Vec<usize>),
}

impl CheckpointStrategy {
    /// Validate strategy-specific invariants.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Every(n) if *n == 0 => Err("Every(n) requires n > 0".into()),
            Self::Custom(layers) if layers.is_empty() => {
                Err("Custom strategy requires at least one layer".into())
            }
            _ => Ok(()),
        }
    }

    /// Compute the checkpoint layer indices for a model with `num_layers`.
    pub fn compute_indices(&self, num_layers: usize) -> Vec<usize> {
        match self {
            Self::None => Vec::new(),
            Self::Every(n) => (0..num_layers).filter(|i| i % n == 0).collect(),
            Self::Sqrt => {
                if num_layers == 0 {
                    return Vec::new();
                }
                let step = (num_layers as f64).sqrt().ceil().max(1.0) as usize;
                (0..num_layers).step_by(step).collect()
            }
            Self::Optimal => {
                // Heuristic: use √N but also include the last layer.
                if num_layers == 0 {
                    return Vec::new();
                }
                let step = (num_layers as f64).sqrt().ceil().max(1.0) as usize;
                let mut indices: Vec<usize> = (0..num_layers).step_by(step).collect();
                let last = num_layers - 1;
                if indices.last() != Some(&last) {
                    indices.push(last);
                }
                indices
            }
            Self::Custom(layers) => {
                let mut out: Vec<usize> =
                    layers.iter().copied().filter(|&i| i < num_layers).collect();
                out.sort_unstable();
                out.dedup();
                out
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Activation store
// ---------------------------------------------------------------------------

/// Stores and retrieves activation tensors for recomputation.
#[derive(Debug)]
pub struct ActivationStore {
    /// Stored activations keyed by layer index.
    activations: HashMap<usize, ActivationData>,
    /// Total bytes currently stored.
    total_bytes: usize,
    /// Maximum bytes allowed.
    capacity: usize,
}

/// A single stored activation blob.
#[derive(Debug, Clone)]
pub struct ActivationData {
    /// Raw data (possibly compressed).
    pub data: Vec<u8>,
    /// Original (uncompressed) size in bytes.
    pub original_size: usize,
    /// Whether `data` is compressed.
    pub compressed: bool,
    /// Layer index this activation belongs to.
    pub layer_index: usize,
}

impl ActivationStore {
    /// Create a new store with the given byte capacity.
    pub fn new(capacity: usize) -> Self {
        Self { activations: HashMap::new(), total_bytes: 0, capacity }
    }

    /// Store an activation, returning `Err` if capacity would be exceeded.
    pub fn store(&mut self, layer: usize, data: ActivationData) -> Result<(), String> {
        let size = data.data.len();
        if self.total_bytes + size > self.capacity {
            return Err(format!(
                "capacity exceeded: {} + {} > {}",
                self.total_bytes, size, self.capacity,
            ));
        }
        if let Some(old) = self.activations.insert(layer, data) {
            self.total_bytes -= old.data.len();
        }
        self.total_bytes += size;
        Ok(())
    }

    /// Retrieve a stored activation (non-destructive).
    pub fn get(&self, layer: usize) -> Option<&ActivationData> {
        self.activations.get(&layer)
    }

    /// Remove and return a stored activation.
    pub fn take(&mut self, layer: usize) -> Option<ActivationData> {
        if let Some(data) = self.activations.remove(&layer) {
            self.total_bytes -= data.data.len();
            Some(data)
        } else {
            Option::None
        }
    }

    /// Number of activations currently stored.
    pub fn len(&self) -> usize {
        self.activations.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.activations.is_empty()
    }

    /// Total bytes currently stored.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Remaining capacity in bytes.
    pub fn remaining_capacity(&self) -> usize {
        self.capacity.saturating_sub(self.total_bytes)
    }

    /// Clear all stored activations.
    pub fn clear(&mut self) {
        self.activations.clear();
        self.total_bytes = 0;
    }
}

// ---------------------------------------------------------------------------
// Segment planner
// ---------------------------------------------------------------------------

/// Plans which layers to checkpoint based on the memory budget.
#[derive(Debug)]
pub struct SegmentPlanner {
    /// Total number of layers in the model.
    pub num_layers: usize,
    /// Configuration to use for planning.
    pub config: CheckpointConfig,
}

impl SegmentPlanner {
    pub fn new(num_layers: usize, config: CheckpointConfig) -> Self {
        Self { num_layers, config }
    }

    /// Produce the list of [`CheckpointSegment`]s for the model.
    pub fn plan(&self) -> Vec<CheckpointSegment> {
        let indices = self.config.strategy.compute_indices(self.num_layers);
        if indices.is_empty() {
            // Single segment covering all layers – no checkpointing.
            if self.num_layers == 0 {
                return Vec::new();
            }
            return vec![CheckpointSegment {
                start_layer: 0,
                end_layer: self.num_layers - 1,
                checkpoint_at_start: false,
                estimated_memory: 0,
            }];
        }

        let mut segments = Vec::new();
        for (i, &start) in indices.iter().enumerate() {
            let end = if i + 1 < indices.len() {
                indices[i + 1].saturating_sub(1)
            } else {
                self.num_layers.saturating_sub(1)
            };
            segments.push(CheckpointSegment {
                start_layer: start,
                end_layer: end,
                checkpoint_at_start: true,
                estimated_memory: 0,
            });
        }
        segments
    }

    /// Return only the layer indices that will be checkpointed.
    pub fn checkpoint_layers(&self) -> Vec<usize> {
        self.config.strategy.compute_indices(self.num_layers)
    }
}

// ---------------------------------------------------------------------------
// Checkpoint segment
// ---------------------------------------------------------------------------

/// A contiguous group of layers sharing a checkpoint boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointSegment {
    /// First layer in this segment (inclusive).
    pub start_layer: usize,
    /// Last layer in this segment (inclusive).
    pub end_layer: usize,
    /// Whether activations at `start_layer` are persisted.
    pub checkpoint_at_start: bool,
    /// Estimated memory for activations in this segment (bytes).
    pub estimated_memory: usize,
}

impl CheckpointSegment {
    /// Number of layers in this segment.
    pub fn num_layers(&self) -> usize {
        self.end_layer - self.start_layer + 1
    }

    /// Returns `true` when the layer is inside this segment.
    pub fn contains(&self, layer: usize) -> bool {
        layer >= self.start_layer && layer <= self.end_layer
    }
}

// ---------------------------------------------------------------------------
// Recompute scheduler
// ---------------------------------------------------------------------------

/// Schedules forward recomputation during the backward pass.
#[derive(Debug)]
pub struct RecomputeScheduler {
    /// Ordered list of segments.
    segments: Vec<CheckpointSegment>,
    /// Set of layers whose activations are already stored.
    stored_layers: Vec<usize>,
}

/// A single recomputation task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecomputeTask {
    /// Segment that must be recomputed.
    pub segment_index: usize,
    /// Range of layers to recompute (start inclusive, end inclusive).
    pub start_layer: usize,
    pub end_layer: usize,
    /// The stored checkpoint layer used as the starting point.
    pub from_checkpoint: usize,
}

impl RecomputeScheduler {
    pub fn new(segments: Vec<CheckpointSegment>, stored_layers: Vec<usize>) -> Self {
        Self { segments, stored_layers }
    }

    /// Build the recomputation schedule for the backward pass.
    ///
    /// The backward pass iterates layers in reverse; for each segment we
    /// need to recompute forward from the segment's checkpoint.
    pub fn schedule(&self) -> Vec<RecomputeTask> {
        let mut tasks = Vec::new();
        for (idx, seg) in self.segments.iter().enumerate().rev() {
            if !seg.checkpoint_at_start {
                continue;
            }
            // Find nearest stored checkpoint at or before start.
            let checkpoint = self
                .stored_layers
                .iter()
                .copied()
                .filter(|&l| l <= seg.start_layer)
                .max()
                .unwrap_or(seg.start_layer);

            tasks.push(RecomputeTask {
                segment_index: idx,
                start_layer: seg.start_layer,
                end_layer: seg.end_layer,
                from_checkpoint: checkpoint,
            });
        }
        tasks
    }

    /// Number of segments in the schedule.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Total layers that must be recomputed.
    pub fn total_recompute_layers(&self) -> usize {
        self.schedule().iter().map(|t| t.end_layer - t.start_layer + 1).sum()
    }
}

// ---------------------------------------------------------------------------
// Memory estimator
// ---------------------------------------------------------------------------

/// Estimates memory savings from a checkpointing strategy.
#[derive(Debug)]
pub struct MemoryEstimator {
    /// Per-layer activation size in bytes.
    pub activation_size_per_layer: usize,
    /// Total number of layers.
    pub num_layers: usize,
}

/// Result of a memory estimation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEstimate {
    /// Bytes required without checkpointing.
    pub baseline_bytes: usize,
    /// Bytes required with checkpointing.
    pub checkpointed_bytes: usize,
    /// Bytes saved.
    pub savings_bytes: usize,
    /// Number of checkpointed layers.
    pub num_checkpoints: usize,
    /// Number of layers that will be recomputed.
    pub recompute_layers: usize,
}

impl MemoryEstimator {
    pub fn new(activation_size_per_layer: usize, num_layers: usize) -> Self {
        Self { activation_size_per_layer, num_layers }
    }

    /// Estimate memory for the given strategy.
    pub fn estimate(&self, strategy: &CheckpointStrategy) -> MemoryEstimate {
        let baseline = self.activation_size_per_layer * self.num_layers;
        let indices = strategy.compute_indices(self.num_layers);
        let num_checkpoints = indices.len();
        let checkpointed = self.activation_size_per_layer * num_checkpoints;
        let recompute_layers = self.num_layers.saturating_sub(num_checkpoints);
        MemoryEstimate {
            baseline_bytes: baseline,
            checkpointed_bytes: checkpointed,
            savings_bytes: baseline.saturating_sub(checkpointed),
            num_checkpoints,
            recompute_layers,
        }
    }

    /// Return the compression-adjusted estimate.
    pub fn estimate_compressed(
        &self,
        strategy: &CheckpointStrategy,
        compression_ratio: f64,
    ) -> MemoryEstimate {
        let mut est = self.estimate(strategy);
        let compressed = (est.checkpointed_bytes as f64 * compression_ratio) as usize;
        est.savings_bytes = est.baseline_bytes.saturating_sub(compressed);
        est.checkpointed_bytes = compressed;
        est
    }
}

// ---------------------------------------------------------------------------
// Activation compressor
// ---------------------------------------------------------------------------

/// Optionally compresses stored activations to save memory.
#[derive(Debug)]
pub struct ActivationCompressor {
    /// Compression level (1 = fastest, 9 = best ratio).
    level: u32,
    /// Running stats: total bytes before compression.
    total_original: usize,
    /// Running stats: total bytes after compression.
    total_compressed: usize,
}

impl ActivationCompressor {
    pub fn new(level: u32) -> Self {
        Self { level: level.clamp(1, 9), total_original: 0, total_compressed: 0 }
    }

    /// Compress activation data using a lightweight RLE scheme.
    ///
    /// Production implementations would use LZ4/zstd; this placeholder
    /// demonstrates the API contract with a trivial run-length encoder.
    pub fn compress(&mut self, data: &[u8]) -> Vec<u8> {
        self.total_original += data.len();
        let compressed = Self::rle_encode(data);
        self.total_compressed += compressed.len();
        compressed
    }

    /// Decompress previously compressed data.
    pub fn decompress(&self, data: &[u8]) -> Vec<u8> {
        Self::rle_decode(data)
    }

    /// Current compression ratio (compressed / original). Lower is better.
    pub fn compression_ratio(&self) -> f64 {
        if self.total_original == 0 {
            return 1.0;
        }
        self.total_compressed as f64 / self.total_original as f64
    }

    /// Compression level.
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Reset running statistics.
    pub fn reset_stats(&mut self) {
        self.total_original = 0;
        self.total_compressed = 0;
    }

    // -- trivial RLE helpers ------------------------------------------------

    fn rle_encode(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut i = 0;
        while i < data.len() {
            let val = data[i];
            let mut run: u8 = 1;
            while (run as usize) < 255
                && i + (run as usize) < data.len()
                && data[i + run as usize] == val
            {
                run += 1;
            }
            out.push(run);
            out.push(val);
            i += run as usize;
        }
        out
    }

    fn rle_decode(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut i = 0;
        while i + 1 < data.len() {
            let run = data[i] as usize;
            let val = data[i + 1];
            out.extend(std::iter::repeat_n(val, run));
            i += 2;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Checkpoint profiler
// ---------------------------------------------------------------------------

/// Profiles recompute time vs memory savings tradeoff.
#[derive(Debug)]
pub struct CheckpointProfiler {
    /// Estimated per-layer forward time in microseconds.
    pub forward_time_us_per_layer: f64,
    /// Memory estimator for activation sizes.
    estimator: MemoryEstimator,
}

/// A single profiler report for one strategy.
#[derive(Debug, Clone)]
pub struct ProfileReport {
    /// Strategy that was profiled.
    pub strategy: CheckpointStrategy,
    /// Memory estimate.
    pub memory: MemoryEstimate,
    /// Estimated extra recompute time in microseconds.
    pub recompute_time_us: f64,
    /// Memory savings as a fraction (0.0–1.0).
    pub memory_savings_ratio: f64,
    /// Compute overhead as a fraction of one full forward pass.
    pub compute_overhead_ratio: f64,
}

impl CheckpointProfiler {
    pub fn new(
        forward_time_us_per_layer: f64,
        activation_size_per_layer: usize,
        num_layers: usize,
    ) -> Self {
        Self {
            forward_time_us_per_layer,
            estimator: MemoryEstimator::new(activation_size_per_layer, num_layers),
        }
    }

    /// Profile a single strategy.
    pub fn profile(&self, strategy: &CheckpointStrategy) -> ProfileReport {
        let memory = self.estimator.estimate(strategy);
        let recompute_time_us = memory.recompute_layers as f64 * self.forward_time_us_per_layer;
        let full_forward_us = self.estimator.num_layers as f64 * self.forward_time_us_per_layer;
        let compute_overhead_ratio =
            if full_forward_us > 0.0 { recompute_time_us / full_forward_us } else { 0.0 };
        let memory_savings_ratio = if memory.baseline_bytes > 0 {
            memory.savings_bytes as f64 / memory.baseline_bytes as f64
        } else {
            0.0
        };
        ProfileReport {
            strategy: strategy.clone(),
            memory,
            recompute_time_us,
            memory_savings_ratio,
            compute_overhead_ratio,
        }
    }

    /// Profile multiple strategies and return sorted by savings ratio.
    pub fn compare(&self, strategies: &[CheckpointStrategy]) -> Vec<ProfileReport> {
        let mut reports: Vec<ProfileReport> = strategies.iter().map(|s| self.profile(s)).collect();
        reports.sort_by(|a, b| {
            b.memory_savings_ratio
                .partial_cmp(&a.memory_savings_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reports
    }
}

// ---------------------------------------------------------------------------
// Gradient checkpointer (orchestrator)
// ---------------------------------------------------------------------------

/// Orchestrates the full gradient-checkpointing workflow:
/// plan segments → forward pass → store activations → backward pass →
/// recompute as needed.
#[derive(Debug)]
pub struct GradientCheckpointer {
    config: CheckpointConfig,
    store: ActivationStore,
    compressor: Option<ActivationCompressor>,
    segments: Vec<CheckpointSegment>,
    state: CheckpointerState,
    stats: CheckpointerStats,
}

/// Internal state of the checkpointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointerState {
    /// Not yet initialised.
    Idle,
    /// Forward pass in progress.
    Forward,
    /// Backward pass in progress.
    Backward,
    /// Run complete.
    Complete,
}

/// Cumulative statistics.
#[derive(Debug, Clone, Default)]
pub struct CheckpointerStats {
    pub layers_stored: usize,
    pub layers_recomputed: usize,
    pub bytes_stored: usize,
    pub bytes_saved_by_compression: usize,
}

impl GradientCheckpointer {
    /// Create a new checkpointer from config.
    pub fn new(config: CheckpointConfig) -> Self {
        let compressor = if config.enable_compression {
            Some(ActivationCompressor::new(config.compression_level))
        } else {
            Option::None
        };
        let store = ActivationStore::new(config.memory_budget);
        Self {
            config,
            store,
            compressor,
            segments: Vec::new(),
            state: CheckpointerState::Idle,
            stats: CheckpointerStats::default(),
        }
    }

    /// Plan segments for a model with `num_layers`.
    pub fn plan(&mut self, num_layers: usize) {
        let planner = SegmentPlanner::new(num_layers, self.config.clone());
        self.segments = planner.plan();
        self.state = CheckpointerState::Idle;
    }

    /// Begin the forward pass.
    pub fn begin_forward(&mut self) -> Result<(), String> {
        if self.segments.is_empty() {
            return Err("must call plan() before begin_forward()".into());
        }
        self.state = CheckpointerState::Forward;
        Ok(())
    }

    /// Store an activation during the forward pass.
    pub fn store_activation(&mut self, layer: usize, data: Vec<u8>) -> Result<(), String> {
        if self.state != CheckpointerState::Forward {
            return Err("not in forward pass".into());
        }
        let original_size = data.len();
        let (blob, compressed) = if let Some(ref mut comp) = self.compressor {
            let c = comp.compress(&data);
            let saved = original_size.saturating_sub(c.len());
            self.stats.bytes_saved_by_compression += saved;
            (c, true)
        } else {
            (data, false)
        };

        let activation =
            ActivationData { data: blob, original_size, compressed, layer_index: layer };
        self.store.store(layer, activation)?;
        self.stats.layers_stored += 1;
        self.stats.bytes_stored += original_size;
        Ok(())
    }

    /// Transition to the backward pass.
    pub fn begin_backward(&mut self) -> Result<(), String> {
        if self.state != CheckpointerState::Forward {
            return Err("must be in forward pass to begin backward".into());
        }
        self.state = CheckpointerState::Backward;
        Ok(())
    }

    /// Build the recompute schedule for the backward pass.
    pub fn recompute_schedule(&self) -> Vec<RecomputeTask> {
        let stored: Vec<usize> =
            self.segments.iter().filter(|s| s.checkpoint_at_start).map(|s| s.start_layer).collect();
        let scheduler = RecomputeScheduler::new(self.segments.clone(), stored);
        scheduler.schedule()
    }

    /// Retrieve a stored activation (decompressing if needed).
    pub fn retrieve_activation(&self, layer: usize) -> Option<Vec<u8>> {
        let act = self.store.get(layer)?;
        if act.compressed {
            Some(
                self.compressor
                    .as_ref()
                    .map_or_else(|| act.data.clone(), |comp| comp.decompress(&act.data)),
            )
        } else {
            Some(act.data.clone())
        }
    }

    /// Mark a set of layers as recomputed (updates stats).
    pub fn mark_recomputed(&mut self, count: usize) {
        self.stats.layers_recomputed += count;
    }

    /// Finish the backward pass.
    pub fn finish(&mut self) {
        self.state = CheckpointerState::Complete;
        self.store.clear();
    }

    // -- accessors ----------------------------------------------------------

    pub fn state(&self) -> CheckpointerState {
        self.state
    }

    pub fn stats(&self) -> &CheckpointerStats {
        &self.stats
    }

    pub fn segments(&self) -> &[CheckpointSegment] {
        &self.segments
    }

    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- CheckpointConfig ---------------------------------------------------

    #[test]
    fn config_default_is_valid() {
        let cfg = CheckpointConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_zero_segment_size() {
        let cfg = CheckpointConfig { segment_size: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_zero_memory_budget() {
        let cfg = CheckpointConfig { memory_budget: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_compression_level_zero() {
        let cfg = CheckpointConfig { compression_level: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_compression_level_ten() {
        let cfg = CheckpointConfig { compression_level: 10, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_valid_compression_levels() {
        for lvl in 1..=9 {
            let cfg = CheckpointConfig { compression_level: lvl, ..Default::default() };
            assert!(cfg.validate().is_ok(), "level {lvl} should be valid");
        }
    }

    // -- CheckpointStrategy -------------------------------------------------

    #[test]
    fn strategy_none_indices() {
        let indices = CheckpointStrategy::None.compute_indices(10);
        assert!(indices.is_empty());
    }

    #[test]
    fn strategy_every_2_indices() {
        let indices = CheckpointStrategy::Every(2).compute_indices(8);
        assert_eq!(indices, vec![0, 2, 4, 6]);
    }

    #[test]
    fn strategy_every_1_indices() {
        let indices = CheckpointStrategy::Every(1).compute_indices(4);
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn strategy_sqrt_indices_16_layers() {
        let indices = CheckpointStrategy::Sqrt.compute_indices(16);
        // sqrt(16) = 4, step_by(4) → [0, 4, 8, 12]
        assert_eq!(indices, vec![0, 4, 8, 12]);
    }

    #[test]
    fn strategy_sqrt_indices_0_layers() {
        let indices = CheckpointStrategy::Sqrt.compute_indices(0);
        assert!(indices.is_empty());
    }

    #[test]
    fn strategy_sqrt_indices_1_layer() {
        let indices = CheckpointStrategy::Sqrt.compute_indices(1);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn strategy_optimal_includes_last_layer() {
        let indices = CheckpointStrategy::Optimal.compute_indices(10);
        assert_eq!(*indices.last().unwrap(), 9);
    }

    #[test]
    fn strategy_optimal_0_layers() {
        let indices = CheckpointStrategy::Optimal.compute_indices(0);
        assert!(indices.is_empty());
    }

    #[test]
    fn strategy_custom_filters_out_of_range() {
        let indices = CheckpointStrategy::Custom(vec![0, 5, 100]).compute_indices(10);
        assert_eq!(indices, vec![0, 5]);
    }

    #[test]
    fn strategy_custom_deduplicates() {
        let indices = CheckpointStrategy::Custom(vec![3, 3, 1, 1]).compute_indices(10);
        assert_eq!(indices, vec![1, 3]);
    }

    #[test]
    fn strategy_validate_every_zero() {
        assert!(CheckpointStrategy::Every(0).validate().is_err());
    }

    #[test]
    fn strategy_validate_custom_empty() {
        assert!(CheckpointStrategy::Custom(vec![]).validate().is_err());
    }

    #[test]
    fn strategy_validate_none_ok() {
        assert!(CheckpointStrategy::None.validate().is_ok());
    }

    #[test]
    fn strategy_validate_sqrt_ok() {
        assert!(CheckpointStrategy::Sqrt.validate().is_ok());
    }

    #[test]
    fn strategy_validate_optimal_ok() {
        assert!(CheckpointStrategy::Optimal.validate().is_ok());
    }

    // -- ActivationStore ----------------------------------------------------

    #[test]
    fn store_new_is_empty() {
        let store = ActivationStore::new(1024);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.total_bytes(), 0);
    }

    #[test]
    fn store_insert_and_get() {
        let mut store = ActivationStore::new(1024);
        let data = ActivationData {
            data: vec![1, 2, 3],
            original_size: 3,
            compressed: false,
            layer_index: 0,
        };
        assert!(store.store(0, data).is_ok());
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(0).unwrap().data, vec![1, 2, 3]);
    }

    #[test]
    fn store_capacity_exceeded() {
        let mut store = ActivationStore::new(4);
        let data = ActivationData {
            data: vec![0; 5],
            original_size: 5,
            compressed: false,
            layer_index: 0,
        };
        assert!(store.store(0, data).is_err());
    }

    #[test]
    fn store_take_removes() {
        let mut store = ActivationStore::new(1024);
        let data =
            ActivationData { data: vec![42], original_size: 1, compressed: false, layer_index: 0 };
        store.store(0, data).unwrap();
        let taken = store.take(0);
        assert!(taken.is_some());
        assert!(store.is_empty());
        assert_eq!(store.total_bytes(), 0);
    }

    #[test]
    fn store_take_nonexistent() {
        let mut store = ActivationStore::new(1024);
        assert!(store.take(99).is_none());
    }

    #[test]
    fn store_remaining_capacity() {
        let mut store = ActivationStore::new(100);
        let data = ActivationData {
            data: vec![0; 30],
            original_size: 30,
            compressed: false,
            layer_index: 0,
        };
        store.store(0, data).unwrap();
        assert_eq!(store.remaining_capacity(), 70);
    }

    #[test]
    fn store_clear() {
        let mut store = ActivationStore::new(1024);
        for i in 0..5 {
            let data = ActivationData {
                data: vec![i as u8; 10],
                original_size: 10,
                compressed: false,
                layer_index: i,
            };
            store.store(i, data).unwrap();
        }
        assert_eq!(store.len(), 5);
        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.total_bytes(), 0);
    }

    #[test]
    fn store_overwrite_same_layer() {
        let mut store = ActivationStore::new(1024);
        let d1 = ActivationData {
            data: vec![0; 10],
            original_size: 10,
            compressed: false,
            layer_index: 0,
        };
        let d2 = ActivationData {
            data: vec![1; 20],
            original_size: 20,
            compressed: false,
            layer_index: 0,
        };
        store.store(0, d1).unwrap();
        assert_eq!(store.total_bytes(), 10);
        store.store(0, d2).unwrap();
        assert_eq!(store.total_bytes(), 20);
        assert_eq!(store.len(), 1);
    }

    // -- SegmentPlanner -----------------------------------------------------

    #[test]
    fn planner_no_strategy_single_segment() {
        let cfg = CheckpointConfig { strategy: CheckpointStrategy::None, ..Default::default() };
        let planner = SegmentPlanner::new(10, cfg);
        let segments = planner.plan();
        assert_eq!(segments.len(), 1);
        assert!(!segments[0].checkpoint_at_start);
    }

    #[test]
    fn planner_every_2_on_8_layers() {
        let cfg = CheckpointConfig { strategy: CheckpointStrategy::Every(2), ..Default::default() };
        let planner = SegmentPlanner::new(8, cfg);
        let segments = planner.plan();
        assert_eq!(segments.len(), 4); // [0,1] [2,3] [4,5] [6,7]
        assert!(segments[0].checkpoint_at_start);
    }

    #[test]
    fn planner_zero_layers() {
        let cfg = CheckpointConfig::default();
        let planner = SegmentPlanner::new(0, cfg);
        let segments = planner.plan();
        assert!(segments.is_empty());
    }

    #[test]
    fn planner_checkpoint_layers() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::Custom(vec![0, 5, 9]),
            ..Default::default()
        };
        let planner = SegmentPlanner::new(10, cfg);
        assert_eq!(planner.checkpoint_layers(), vec![0, 5, 9]);
    }

    #[test]
    fn planner_segments_cover_all_layers() {
        let cfg = CheckpointConfig { strategy: CheckpointStrategy::Every(3), ..Default::default() };
        let planner = SegmentPlanner::new(10, cfg);
        let segs = planner.plan();
        // Every layer 0..9 should be in exactly one segment.
        for layer in 0..10 {
            let count = segs.iter().filter(|s| s.contains(layer)).count();
            assert_eq!(count, 1, "layer {layer} coverage");
        }
    }

    // -- CheckpointSegment --------------------------------------------------

    #[test]
    fn segment_num_layers() {
        let seg = CheckpointSegment {
            start_layer: 3,
            end_layer: 7,
            checkpoint_at_start: true,
            estimated_memory: 0,
        };
        assert_eq!(seg.num_layers(), 5);
    }

    #[test]
    fn segment_contains() {
        let seg = CheckpointSegment {
            start_layer: 2,
            end_layer: 5,
            checkpoint_at_start: true,
            estimated_memory: 0,
        };
        assert!(!seg.contains(1));
        assert!(seg.contains(2));
        assert!(seg.contains(5));
        assert!(!seg.contains(6));
    }

    // -- RecomputeScheduler -------------------------------------------------

    #[test]
    fn scheduler_empty_segments() {
        let sched = RecomputeScheduler::new(vec![], vec![]);
        assert!(sched.schedule().is_empty());
    }

    #[test]
    fn scheduler_single_segment() {
        let seg = CheckpointSegment {
            start_layer: 0,
            end_layer: 3,
            checkpoint_at_start: true,
            estimated_memory: 0,
        };
        let sched = RecomputeScheduler::new(vec![seg], vec![0]);
        let tasks = sched.schedule();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].from_checkpoint, 0);
        assert_eq!(tasks[0].start_layer, 0);
        assert_eq!(tasks[0].end_layer, 3);
    }

    #[test]
    fn scheduler_multiple_segments_reverse_order() {
        let segs = vec![
            CheckpointSegment {
                start_layer: 0,
                end_layer: 3,
                checkpoint_at_start: true,
                estimated_memory: 0,
            },
            CheckpointSegment {
                start_layer: 4,
                end_layer: 7,
                checkpoint_at_start: true,
                estimated_memory: 0,
            },
        ];
        let sched = RecomputeScheduler::new(segs, vec![0, 4]);
        let tasks = sched.schedule();
        // Backward order: segment 1 first, then segment 0.
        assert_eq!(tasks[0].segment_index, 1);
        assert_eq!(tasks[1].segment_index, 0);
    }

    #[test]
    fn scheduler_skips_non_checkpoint_segments() {
        let segs = vec![
            CheckpointSegment {
                start_layer: 0,
                end_layer: 3,
                checkpoint_at_start: false,
                estimated_memory: 0,
            },
            CheckpointSegment {
                start_layer: 4,
                end_layer: 7,
                checkpoint_at_start: true,
                estimated_memory: 0,
            },
        ];
        let sched = RecomputeScheduler::new(segs, vec![4]);
        let tasks = sched.schedule();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].segment_index, 1);
    }

    #[test]
    fn scheduler_total_recompute_layers() {
        let segs = vec![
            CheckpointSegment {
                start_layer: 0,
                end_layer: 3,
                checkpoint_at_start: true,
                estimated_memory: 0,
            },
            CheckpointSegment {
                start_layer: 4,
                end_layer: 7,
                checkpoint_at_start: true,
                estimated_memory: 0,
            },
        ];
        let sched = RecomputeScheduler::new(segs, vec![0, 4]);
        assert_eq!(sched.total_recompute_layers(), 8);
    }

    #[test]
    fn scheduler_num_segments() {
        let segs = vec![
            CheckpointSegment {
                start_layer: 0,
                end_layer: 1,
                checkpoint_at_start: true,
                estimated_memory: 0,
            },
            CheckpointSegment {
                start_layer: 2,
                end_layer: 3,
                checkpoint_at_start: true,
                estimated_memory: 0,
            },
        ];
        let sched = RecomputeScheduler::new(segs, vec![0, 2]);
        assert_eq!(sched.num_segments(), 2);
    }

    // -- MemoryEstimator ----------------------------------------------------

    #[test]
    fn estimator_no_checkpointing() {
        let est = MemoryEstimator::new(1000, 10);
        let result = est.estimate(&CheckpointStrategy::None);
        assert_eq!(result.baseline_bytes, 10_000);
        assert_eq!(result.checkpointed_bytes, 0);
        assert_eq!(result.savings_bytes, 10_000);
        assert_eq!(result.num_checkpoints, 0);
    }

    #[test]
    fn estimator_every_2() {
        let est = MemoryEstimator::new(1000, 10);
        let result = est.estimate(&CheckpointStrategy::Every(2));
        // Layers 0,2,4,6,8 → 5 checkpoints
        assert_eq!(result.num_checkpoints, 5);
        assert_eq!(result.checkpointed_bytes, 5000);
        assert_eq!(result.savings_bytes, 5000);
        assert_eq!(result.recompute_layers, 5);
    }

    #[test]
    fn estimator_sqrt() {
        let est = MemoryEstimator::new(1000, 16);
        let result = est.estimate(&CheckpointStrategy::Sqrt);
        // sqrt(16) = 4, → [0,4,8,12] → 4 checkpoints
        assert_eq!(result.num_checkpoints, 4);
        assert_eq!(result.checkpointed_bytes, 4000);
    }

    #[test]
    fn estimator_compressed() {
        let est = MemoryEstimator::new(1000, 10);
        let result = est.estimate_compressed(&CheckpointStrategy::Every(1), 0.5);
        // All 10 layers checkpointed, 10000 * 0.5 = 5000
        assert_eq!(result.checkpointed_bytes, 5000);
        assert_eq!(result.savings_bytes, 5000);
    }

    #[test]
    fn estimator_zero_layers() {
        let est = MemoryEstimator::new(1000, 0);
        let result = est.estimate(&CheckpointStrategy::Sqrt);
        assert_eq!(result.baseline_bytes, 0);
        assert_eq!(result.num_checkpoints, 0);
    }

    // -- ActivationCompressor -----------------------------------------------

    #[test]
    fn compressor_round_trip_uniform() {
        let mut comp = ActivationCompressor::new(3);
        let data = vec![42; 100];
        let compressed = comp.compress(&data);
        let decompressed = comp.decompress(&compressed);
        assert_eq!(decompressed, data);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn compressor_round_trip_empty() {
        let mut comp = ActivationCompressor::new(1);
        let data: Vec<u8> = vec![];
        let compressed = comp.compress(&data);
        let decompressed = comp.decompress(&compressed);
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compressor_round_trip_single_byte() {
        let mut comp = ActivationCompressor::new(5);
        let data = vec![7];
        let compressed = comp.compress(&data);
        let decompressed = comp.decompress(&compressed);
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compressor_round_trip_varied() {
        let mut comp = ActivationCompressor::new(5);
        let data: Vec<u8> = (0..=255).collect();
        let compressed = comp.compress(&data);
        let decompressed = comp.decompress(&compressed);
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compressor_ratio_initial() {
        let comp = ActivationCompressor::new(3);
        assert!((comp.compression_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compressor_ratio_after_compress() {
        let mut comp = ActivationCompressor::new(3);
        comp.compress(&[0; 100]);
        assert!(comp.compression_ratio() < 1.0);
    }

    #[test]
    fn compressor_reset_stats() {
        let mut comp = ActivationCompressor::new(3);
        comp.compress(&[0; 100]);
        comp.reset_stats();
        assert!((comp.compression_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compressor_level_clamped() {
        let comp = ActivationCompressor::new(99);
        assert_eq!(comp.level(), 9);
        let comp = ActivationCompressor::new(0);
        assert_eq!(comp.level(), 1);
    }

    // -- CheckpointProfiler -------------------------------------------------

    #[test]
    fn profiler_no_checkpointing() {
        let profiler = CheckpointProfiler::new(100.0, 1000, 10);
        let report = profiler.profile(&CheckpointStrategy::None);
        // All layers recomputed, no memory used.
        assert_eq!(report.memory.num_checkpoints, 0);
        assert!(report.memory_savings_ratio > 0.99);
    }

    #[test]
    fn profiler_full_checkpointing() {
        let profiler = CheckpointProfiler::new(100.0, 1000, 10);
        let report = profiler.profile(&CheckpointStrategy::Every(1));
        assert_eq!(report.memory.num_checkpoints, 10);
        assert!(report.compute_overhead_ratio.abs() < f64::EPSILON);
    }

    #[test]
    fn profiler_compute_overhead_sqrt() {
        let profiler = CheckpointProfiler::new(100.0, 1000, 16);
        let report = profiler.profile(&CheckpointStrategy::Sqrt);
        assert!(report.compute_overhead_ratio > 0.0);
        assert!(report.compute_overhead_ratio < 1.0);
    }

    #[test]
    fn profiler_compare_sorts_by_savings() {
        let profiler = CheckpointProfiler::new(100.0, 1000, 16);
        let reports = profiler.compare(&[
            CheckpointStrategy::Every(1),
            CheckpointStrategy::None,
            CheckpointStrategy::Sqrt,
        ]);
        // None saves the most memory.
        assert_eq!(reports[0].strategy, CheckpointStrategy::None);
    }

    #[test]
    fn profiler_zero_layers() {
        let profiler = CheckpointProfiler::new(100.0, 1000, 0);
        let report = profiler.profile(&CheckpointStrategy::Sqrt);
        assert_eq!(report.memory.baseline_bytes, 0);
    }

    // -- GradientCheckpointer -----------------------------------------------

    #[test]
    fn checkpointer_lifecycle() {
        let cfg = CheckpointConfig { strategy: CheckpointStrategy::Every(2), ..Default::default() };
        let mut ckpt = GradientCheckpointer::new(cfg);
        assert_eq!(ckpt.state(), CheckpointerState::Idle);

        ckpt.plan(8);
        assert!(!ckpt.segments().is_empty());

        ckpt.begin_forward().unwrap();
        assert_eq!(ckpt.state(), CheckpointerState::Forward);

        ckpt.store_activation(0, vec![1; 100]).unwrap();
        ckpt.store_activation(2, vec![2; 100]).unwrap();

        ckpt.begin_backward().unwrap();
        assert_eq!(ckpt.state(), CheckpointerState::Backward);

        let tasks = ckpt.recompute_schedule();
        assert!(!tasks.is_empty());

        ckpt.mark_recomputed(4);
        ckpt.finish();
        assert_eq!(ckpt.state(), CheckpointerState::Complete);
        assert_eq!(ckpt.stats().layers_stored, 2);
        assert_eq!(ckpt.stats().layers_recomputed, 4);
    }

    #[test]
    fn checkpointer_forward_before_plan() {
        let cfg = CheckpointConfig::default();
        let mut ckpt = GradientCheckpointer::new(cfg);
        assert!(ckpt.begin_forward().is_err());
    }

    #[test]
    fn checkpointer_store_outside_forward() {
        let cfg = CheckpointConfig::default();
        let mut ckpt = GradientCheckpointer::new(cfg);
        ckpt.plan(4);
        assert!(ckpt.store_activation(0, vec![1]).is_err());
    }

    #[test]
    fn checkpointer_backward_before_forward() {
        let cfg = CheckpointConfig::default();
        let mut ckpt = GradientCheckpointer::new(cfg);
        ckpt.plan(4);
        assert!(ckpt.begin_backward().is_err());
    }

    #[test]
    fn checkpointer_with_compression() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::Every(1),
            enable_compression: true,
            compression_level: 5,
            ..Default::default()
        };
        let mut ckpt = GradientCheckpointer::new(cfg);
        ckpt.plan(4);
        ckpt.begin_forward().unwrap();

        let data = vec![0u8; 200];
        ckpt.store_activation(0, data.clone()).unwrap();

        let retrieved = ckpt.retrieve_activation(0).unwrap();
        assert_eq!(retrieved, data);
        assert!(ckpt.stats().bytes_saved_by_compression > 0);
    }

    #[test]
    fn checkpointer_retrieve_nonexistent() {
        let cfg = CheckpointConfig::default();
        let ckpt = GradientCheckpointer::new(cfg);
        assert!(ckpt.retrieve_activation(99).is_none());
    }

    #[test]
    fn checkpointer_stats_bytes_stored() {
        let cfg = CheckpointConfig { strategy: CheckpointStrategy::Every(1), ..Default::default() };
        let mut ckpt = GradientCheckpointer::new(cfg);
        ckpt.plan(2);
        ckpt.begin_forward().unwrap();
        ckpt.store_activation(0, vec![0; 50]).unwrap();
        ckpt.store_activation(1, vec![0; 70]).unwrap();
        assert_eq!(ckpt.stats().bytes_stored, 120);
    }

    #[test]
    fn checkpointer_config_accessor() {
        let cfg = CheckpointConfig { segment_size: 8, ..Default::default() };
        let ckpt = GradientCheckpointer::new(cfg);
        assert_eq!(ckpt.config().segment_size, 8);
    }

    #[test]
    fn checkpointer_finish_clears_store() {
        let cfg = CheckpointConfig { strategy: CheckpointStrategy::Every(1), ..Default::default() };
        let mut ckpt = GradientCheckpointer::new(cfg);
        ckpt.plan(2);
        ckpt.begin_forward().unwrap();
        ckpt.store_activation(0, vec![1; 10]).unwrap();
        ckpt.begin_backward().unwrap();
        ckpt.finish();
        assert!(ckpt.retrieve_activation(0).is_none());
    }

    // -- Integration-style tests --------------------------------------------

    #[test]
    fn end_to_end_no_checkpointing() {
        let cfg = CheckpointConfig { strategy: CheckpointStrategy::None, ..Default::default() };
        let mut ckpt = GradientCheckpointer::new(cfg);
        ckpt.plan(4);
        // Single segment, no checkpoint.
        assert_eq!(ckpt.segments().len(), 1);
        assert!(!ckpt.segments()[0].checkpoint_at_start);
    }

    #[test]
    fn end_to_end_every_1() {
        let cfg = CheckpointConfig { strategy: CheckpointStrategy::Every(1), ..Default::default() };
        let mut ckpt = GradientCheckpointer::new(cfg);
        ckpt.plan(4);
        ckpt.begin_forward().unwrap();
        for i in 0..4 {
            ckpt.store_activation(i, vec![i as u8; 32]).unwrap();
        }
        ckpt.begin_backward().unwrap();
        let tasks = ckpt.recompute_schedule();
        // Every layer checkpointed, but schedule still produced.
        assert!(!tasks.is_empty());
        ckpt.finish();
    }

    #[test]
    fn end_to_end_custom_strategy() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::Custom(vec![0, 5, 9]),
            ..Default::default()
        };
        let mut ckpt = GradientCheckpointer::new(cfg);
        ckpt.plan(10);
        assert_eq!(ckpt.segments().len(), 3);
    }

    #[test]
    fn memory_estimator_savings_monotonic() {
        let est = MemoryEstimator::new(1000, 20);
        let s1 = est.estimate(&CheckpointStrategy::Every(1));
        let s5 = est.estimate(&CheckpointStrategy::Every(5));
        // Fewer checkpoints → more savings.
        assert!(s5.savings_bytes > s1.savings_bytes);
    }

    #[test]
    fn profiler_overhead_increases_with_fewer_checkpoints() {
        let profiler = CheckpointProfiler::new(100.0, 1000, 20);
        let r1 = profiler.profile(&CheckpointStrategy::Every(1));
        let r5 = profiler.profile(&CheckpointStrategy::Every(5));
        assert!(r5.compute_overhead_ratio > r1.compute_overhead_ratio);
    }

    #[test]
    fn segment_coverage_every_3_on_10() {
        let cfg = CheckpointConfig { strategy: CheckpointStrategy::Every(3), ..Default::default() };
        let planner = SegmentPlanner::new(10, cfg);
        let segs = planner.plan();
        // All layers 0..9 covered.
        let covered: Vec<usize> = segs.iter().flat_map(|s| s.start_layer..=s.end_layer).collect();
        for l in 0..10 {
            assert!(covered.contains(&l), "layer {l} not covered");
        }
    }

    #[test]
    fn compressor_varied_data_round_trip() {
        let mut comp = ActivationCompressor::new(5);
        let data: Vec<u8> = (0..200).map(|i| (i % 13) as u8).collect();
        let c = comp.compress(&data);
        assert_eq!(comp.decompress(&c), data);
    }

    #[test]
    fn compressor_long_run_round_trip() {
        let mut comp = ActivationCompressor::new(3);
        // 500 zeros → many runs of 255 + remainder
        let data = vec![0u8; 500];
        let c = comp.compress(&data);
        assert_eq!(comp.decompress(&c), data);
    }

    #[test]
    fn strategy_every_large_step() {
        let indices = CheckpointStrategy::Every(100).compute_indices(10);
        // Only layer 0 qualifies.
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn strategy_optimal_small() {
        let indices = CheckpointStrategy::Optimal.compute_indices(2);
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
    }

    #[test]
    fn store_multiple_layers() {
        let mut store = ActivationStore::new(10_000);
        for i in 0..10 {
            let data = ActivationData {
                data: vec![i as u8; 100],
                original_size: 100,
                compressed: false,
                layer_index: i,
            };
            store.store(i, data).unwrap();
        }
        assert_eq!(store.len(), 10);
        assert_eq!(store.total_bytes(), 1000);
    }

    #[test]
    fn checkpointer_plan_resets_segments() {
        let cfg = CheckpointConfig { strategy: CheckpointStrategy::Every(2), ..Default::default() };
        let mut ckpt = GradientCheckpointer::new(cfg);
        ckpt.plan(8);
        let n1 = ckpt.segments().len();
        ckpt.plan(4);
        let n2 = ckpt.segments().len();
        assert!(n2 <= n1);
    }

    #[test]
    fn strategy_sqrt_indices_9_layers() {
        // sqrt(9) = 3, step_by(3) → [0, 3, 6]
        let indices = CheckpointStrategy::Sqrt.compute_indices(9);
        assert_eq!(indices, vec![0, 3, 6]);
    }

    #[test]
    fn strategy_custom_sorted_output() {
        let indices = CheckpointStrategy::Custom(vec![9, 1, 5]).compute_indices(10);
        assert_eq!(indices, vec![1, 5, 9]);
    }

    #[test]
    fn estimator_baseline_proportional() {
        let e1 = MemoryEstimator::new(1000, 10);
        let e2 = MemoryEstimator::new(1000, 20);
        let r1 = e1.estimate(&CheckpointStrategy::Every(1));
        let r2 = e2.estimate(&CheckpointStrategy::Every(1));
        assert_eq!(r2.baseline_bytes, r1.baseline_bytes * 2);
    }

    #[test]
    fn segment_single_layer() {
        let seg = CheckpointSegment {
            start_layer: 5,
            end_layer: 5,
            checkpoint_at_start: true,
            estimated_memory: 0,
        };
        assert_eq!(seg.num_layers(), 1);
        assert!(seg.contains(5));
        assert!(!seg.contains(4));
    }

    #[test]
    fn checkpointer_multiple_recompute_marks() {
        let cfg = CheckpointConfig { strategy: CheckpointStrategy::Every(2), ..Default::default() };
        let mut ckpt = GradientCheckpointer::new(cfg);
        ckpt.plan(8);
        ckpt.begin_forward().unwrap();
        ckpt.begin_backward().unwrap();
        ckpt.mark_recomputed(3);
        ckpt.mark_recomputed(2);
        assert_eq!(ckpt.stats().layers_recomputed, 5);
    }

    #[test]
    fn profiler_recompute_time_proportional() {
        let p = CheckpointProfiler::new(200.0, 1000, 10);
        let r = p.profile(&CheckpointStrategy::Every(5));
        // 2 checkpoints (0,5), 8 recomputed layers → 8 * 200 = 1600 us
        assert!((r.recompute_time_us - 1600.0).abs() < f64::EPSILON);
    }

    #[test]
    fn activation_data_fields() {
        let d = ActivationData {
            data: vec![1, 2, 3],
            original_size: 100,
            compressed: true,
            layer_index: 7,
        };
        assert!(d.compressed);
        assert_eq!(d.layer_index, 7);
        assert_eq!(d.original_size, 100);
    }

    #[test]
    fn checkpointer_state_is_idle_initially() {
        let ckpt = GradientCheckpointer::new(CheckpointConfig::default());
        assert_eq!(ckpt.state(), CheckpointerState::Idle);
        assert!(ckpt.segments().is_empty());
    }
}
