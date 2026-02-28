//! GPU memory defragmentation and OOM recovery.
//!
//! Provides [`MemoryDefragmenter`] for managing defragmentation passes,
//! [`FragmentationAnalyzer`] for computing waste metrics,
//! [`PressureMonitor`] for tracking memory usage trends, and
//! [`OomRecovery`] for surviving out-of-memory conditions.

use std::collections::VecDeque;

use crate::HalError;

// ── Configuration ─────────────────────────────────────────────────────────

/// Defragmentation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefragStrategy {
    /// Move all live blocks to one end, leaving a single contiguous free
    /// region.
    Compaction,
    /// Rebalance blocks across power-of-two buddy allocator bins.
    BuddyRebalance,
    /// Promote or evict blocks based on their generation age.
    GenerationalCollect,
    /// Spread work across multiple small passes to limit latency.
    Incremental,
}

/// Tuning knobs for the defragmenter.
#[derive(Debug, Clone)]
pub struct DefragConfig {
    /// Fragmentation ratio above which a defrag pass is considered.
    pub fragmentation_threshold: f64,
    /// Number of allocation failures before compaction is forced.
    pub compaction_trigger: u32,
    /// Largest single move (bytes) the defragmenter may perform.
    pub max_move_size: usize,
    /// Strategy used by the defragmenter.
    pub strategy: DefragStrategy,
}

impl Default for DefragConfig {
    fn default() -> Self {
        Self {
            fragmentation_threshold: 0.3,
            compaction_trigger: 8,
            max_move_size: 64 * 1024 * 1024, // 64 MiB
            strategy: DefragStrategy::Compaction,
        }
    }
}

// ── Memory blocks ─────────────────────────────────────────────────────────

/// A contiguous region of GPU memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryBlock {
    /// Byte address of the block start.
    pub address: usize,
    /// Size in bytes.
    pub size: usize,
    /// `true` if the block is unused.
    pub is_free: bool,
    /// Generational age (0 = youngest).
    pub generation: u32,
    /// Number of active pins preventing relocation.
    pub pin_count: u32,
}

impl MemoryBlock {
    /// Whether the block can be relocated.
    pub const fn is_movable(&self) -> bool {
        !self.is_free && self.pin_count == 0
    }
}

// ── Compaction plan ───────────────────────────────────────────────────────

/// A single block-move instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockMove {
    /// Source address.
    pub src: usize,
    /// Destination address.
    pub dst: usize,
    /// Bytes to copy.
    pub size: usize,
}

/// An ordered list of moves that would defragment the heap.
#[derive(Debug, Clone, Default)]
pub struct CompactionPlan {
    pub moves: Vec<BlockMove>,
}

impl CompactionPlan {
    /// Total bytes that would be moved.
    pub fn total_bytes_moved(&self) -> usize {
        self.moves.iter().map(|m| m.size).sum()
    }
}

// ── Fragmentation analyser ────────────────────────────────────────────────

/// Snapshot of fragmentation metrics.
#[derive(Debug, Clone)]
pub struct FragmentationMetrics {
    /// Ratio of fragmented free space to total free space (0.0–1.0).
    pub fragmentation_ratio: f64,
    /// Largest contiguous free region (bytes).
    pub largest_free_block: usize,
    /// Sum of all free regions too small to satisfy a typical allocation.
    pub total_waste: usize,
    /// Number of separate free regions.
    pub free_region_count: usize,
}

/// Analyses a block list to produce [`FragmentationMetrics`].
pub struct FragmentationAnalyzer;

impl FragmentationAnalyzer {
    /// Compute metrics for the given block list.
    ///
    /// Blocks **must** be sorted by ascending address.
    pub fn analyze(blocks: &[MemoryBlock]) -> FragmentationMetrics {
        let free_blocks: Vec<&MemoryBlock> =
            blocks.iter().filter(|b| b.is_free).collect();

        let total_free: usize = free_blocks.iter().map(|b| b.size).sum();
        let largest_free =
            free_blocks.iter().map(|b| b.size).max().unwrap_or(0);
        let free_region_count = free_blocks.len();

        let fragmentation_ratio = if total_free == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                1.0 - (largest_free as f64 / total_free as f64)
            }
        };

        // "Waste" = sum of free blocks smaller than the largest.
        let total_waste: usize = free_blocks
            .iter()
            .filter(|b| b.size < largest_free)
            .map(|b| b.size)
            .sum();

        FragmentationMetrics {
            fragmentation_ratio,
            largest_free_block: largest_free,
            total_waste,
            free_region_count,
        }
    }
}

// ── Memory watermarks ─────────────────────────────────────────────────────

/// Pressure level thresholds (fraction of total memory used).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WatermarkLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Configurable watermark thresholds.
#[derive(Debug, Clone)]
pub struct WatermarkConfig {
    /// Usage fraction for Medium (e.g. 0.50).
    pub medium: f64,
    /// Usage fraction for High (e.g. 0.75).
    pub high: f64,
    /// Usage fraction for Critical (e.g. 0.90).
    pub critical: f64,
}

impl Default for WatermarkConfig {
    fn default() -> Self {
        Self { medium: 0.50, high: 0.75, critical: 0.90 }
    }
}

impl WatermarkConfig {
    /// Classify a usage ratio into a watermark level.
    pub fn classify(&self, usage_ratio: f64) -> WatermarkLevel {
        if usage_ratio >= self.critical {
            WatermarkLevel::Critical
        } else if usage_ratio >= self.high {
            WatermarkLevel::High
        } else if usage_ratio >= self.medium {
            WatermarkLevel::Medium
        } else {
            WatermarkLevel::Low
        }
    }
}

// ── Pressure monitor ──────────────────────────────────────────────────────

/// Tracks memory usage over time and predicts OOM.
#[derive(Debug, Clone)]
pub struct PressureMonitor {
    /// Total capacity being monitored.
    total: usize,
    /// Rolling window of usage samples (bytes used).
    samples: VecDeque<usize>,
    /// Maximum samples to retain.
    window_size: usize,
    /// Watermark thresholds.
    watermarks: WatermarkConfig,
}

impl PressureMonitor {
    /// Create a monitor for `total` bytes with default watermarks.
    pub fn new(total: usize) -> Self {
        Self::with_watermarks(total, WatermarkConfig::default())
    }

    /// Create a monitor with custom watermarks and a 64-sample window.
    pub fn with_watermarks(total: usize, watermarks: WatermarkConfig) -> Self {
        Self { total, samples: VecDeque::new(), window_size: 64, watermarks }
    }

    /// Record a new usage sample.
    pub fn record(&mut self, used: usize) {
        if self.samples.len() >= self.window_size {
            self.samples.pop_front();
        }
        self.samples.push_back(used);
    }

    /// Current watermark level based on the most recent sample.
    pub fn current_level(&self) -> WatermarkLevel {
        let used = self.samples.back().copied().unwrap_or(0);
        let ratio = if self.total == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                used as f64 / self.total as f64
            }
        };
        self.watermarks.classify(ratio)
    }

    /// Simple linear trend: positive means usage is increasing.
    pub fn trend(&self) -> f64 {
        let n = self.samples.len();
        if n < 2 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            let first = *self.samples.front().unwrap() as f64;
            let last = *self.samples.back().unwrap() as f64;
            (last - first) / (n - 1) as f64
        }
    }

    /// Predict whether OOM is likely within `steps` additional samples,
    /// assuming the current linear trend continues.
    pub fn predict_oom(&self, steps: usize) -> bool {
        let used = self.samples.back().copied().unwrap_or(0);
        let delta = self.trend();
        if delta <= 0.0 {
            return false;
        }
        #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
        let predicted = used as f64 + delta * steps as f64;
        #[allow(clippy::cast_precision_loss)]
        let total_f = self.total as f64;
        predicted >= total_f
    }

    /// Number of recorded samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

// ── OOM recovery ──────────────────────────────────────────────────────────

/// Strategy for recovering from an OOM condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OomAction {
    /// Evict cached (unpinned) blocks to free memory.
    EvictCaches,
    /// Reduce the inference batch size.
    ReduceBatchSize,
    /// Offload least-recently-used blocks to CPU.
    OffloadToCpu,
}

/// Decides recovery actions for OOM situations.
pub struct OomRecovery;

impl OomRecovery {
    /// Suggest an ordered list of recovery actions for the given shortfall.
    pub fn suggest(
        shortfall: usize,
        blocks: &[MemoryBlock],
    ) -> Vec<OomAction> {
        let evictable: usize = blocks
            .iter()
            .filter(|b| !b.is_free && b.pin_count == 0)
            .map(|b| b.size)
            .sum();

        let mut actions = Vec::new();

        // Always try cache eviction first.
        if evictable > 0 {
            actions.push(OomAction::EvictCaches);
        }

        // If eviction alone won't cover it, reduce batch size.
        if evictable < shortfall {
            actions.push(OomAction::ReduceBatchSize);
        }

        // Last resort: offload to host.
        if evictable < shortfall / 2 {
            actions.push(OomAction::OffloadToCpu);
        }

        actions
    }
}

// ── Allocation history ────────────────────────────────────────────────────

/// A single allocation / deallocation event for profiling.
#[derive(Debug, Clone)]
pub struct AllocEvent {
    /// Monotonic sequence number.
    pub seq: u64,
    /// `true` = allocate, `false` = deallocate.
    pub is_alloc: bool,
    /// Size in bytes.
    pub size: usize,
    /// Address involved.
    pub address: usize,
}

/// Ring-buffer of recent allocation events.
#[derive(Debug, Clone)]
pub struct AllocationHistory {
    events: VecDeque<AllocEvent>,
    capacity: usize,
    next_seq: u64,
}

impl AllocationHistory {
    /// Create a history that retains at most `capacity` events.
    pub fn new(capacity: usize) -> Self {
        Self { events: VecDeque::with_capacity(capacity), capacity, next_seq: 0 }
    }

    /// Record an event.
    pub fn record(&mut self, is_alloc: bool, size: usize, address: usize) {
        if self.events.len() >= self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(AllocEvent {
            seq: self.next_seq,
            is_alloc,
            size,
            address,
        });
        self.next_seq += 1;
    }

    /// Number of recorded events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Iterate over recorded events oldest-first.
    pub fn iter(&self) -> impl Iterator<Item = &AllocEvent> {
        self.events.iter()
    }

    /// Total bytes allocated (net) across the recorded window.
    pub fn net_allocated(&self) -> i64 {
        self.events.iter().fold(0i64, |acc, e| {
            #[allow(clippy::cast_possible_wrap)]
            if e.is_alloc {
                acc + e.size as i64
            } else {
                acc - e.size as i64
            }
        })
    }
}

// ── Defragmenter ──────────────────────────────────────────────────────────

/// Manages defragmentation passes over a block list.
pub struct MemoryDefragmenter {
    config: DefragConfig,
    /// Running count of allocation failures since last defrag.
    failure_count: u32,
}

impl MemoryDefragmenter {
    /// Create a defragmenter with the given config.
    pub fn new(config: DefragConfig) -> Self {
        Self { config, failure_count: 0 }
    }

    /// Report an allocation failure.
    pub fn report_failure(&mut self) {
        self.failure_count = self.failure_count.saturating_add(1);
    }

    /// Reset the failure counter (e.g. after a successful defrag).
    pub fn reset_failures(&mut self) {
        self.failure_count = 0;
    }

    /// Number of allocation failures since the last reset.
    pub const fn failure_count(&self) -> u32 {
        self.failure_count
    }

    /// Active strategy.
    pub const fn strategy(&self) -> DefragStrategy {
        self.config.strategy
    }

    /// Whether a defrag pass should be triggered now.
    pub fn should_defrag(&self, blocks: &[MemoryBlock]) -> bool {
        if self.failure_count >= self.config.compaction_trigger {
            return true;
        }
        let metrics = FragmentationAnalyzer::analyze(blocks);
        metrics.fragmentation_ratio >= self.config.fragmentation_threshold
    }

    /// Build a compaction plan that moves live, unpinned blocks toward
    /// address 0.
    pub fn plan_compaction(
        &self,
        blocks: &[MemoryBlock],
    ) -> Result<CompactionPlan, HalError> {
        if blocks.is_empty() {
            return Ok(CompactionPlan::default());
        }

        let mut moves = Vec::new();
        let mut cursor: usize = 0;

        for blk in blocks {
            if blk.is_free {
                continue;
            }
            if blk.pin_count > 0 {
                // Pinned block cannot move; advance cursor past it.
                cursor = cursor.max(blk.address + blk.size);
                continue;
            }
            if blk.size > self.config.max_move_size {
                cursor = cursor.max(blk.address + blk.size);
                continue;
            }
            if blk.address != cursor {
                moves.push(BlockMove {
                    src: blk.address,
                    dst: cursor,
                    size: blk.size,
                });
            }
            cursor += blk.size;
        }

        Ok(CompactionPlan { moves })
    }

    /// Execute a defrag pass: analyse → plan → return the plan.
    pub fn run(
        &mut self,
        blocks: &[MemoryBlock],
    ) -> Result<CompactionPlan, HalError> {
        let plan = match self.config.strategy {
            DefragStrategy::Compaction | DefragStrategy::Incremental => {
                self.plan_compaction(blocks)?
            }
            DefragStrategy::BuddyRebalance
            | DefragStrategy::GenerationalCollect => {
                // Buddy / generational strategies fall back to compaction
                // in this implementation.
                self.plan_compaction(blocks)?
            }
        };
        self.reset_failures();
        Ok(plan)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ───────────────────────────────────────────────────────────

    fn block(
        address: usize,
        size: usize,
        is_free: bool,
    ) -> MemoryBlock {
        MemoryBlock { address, size, is_free, generation: 0, pin_count: 0 }
    }

    fn pinned_block(
        address: usize,
        size: usize,
        pin_count: u32,
    ) -> MemoryBlock {
        MemoryBlock {
            address,
            size,
            is_free: false,
            generation: 0,
            pin_count,
        }
    }

    fn gen_block(
        address: usize,
        size: usize,
        generation: u32,
    ) -> MemoryBlock {
        MemoryBlock {
            address,
            size,
            is_free: false,
            generation,
            pin_count: 0,
        }
    }

    // ── DefragConfig ──────────────────────────────────────────────────

    #[test]
    fn default_config_values() {
        let cfg = DefragConfig::default();
        assert!((cfg.fragmentation_threshold - 0.3).abs() < f64::EPSILON);
        assert_eq!(cfg.compaction_trigger, 8);
        assert_eq!(cfg.max_move_size, 64 * 1024 * 1024);
        assert_eq!(cfg.strategy, DefragStrategy::Compaction);
    }

    #[test]
    fn custom_config() {
        let cfg = DefragConfig {
            fragmentation_threshold: 0.5,
            compaction_trigger: 4,
            max_move_size: 1024,
            strategy: DefragStrategy::Incremental,
        };
        assert!((cfg.fragmentation_threshold - 0.5).abs() < f64::EPSILON);
        assert_eq!(cfg.strategy, DefragStrategy::Incremental);
    }

    // ── MemoryBlock ───────────────────────────────────────────────────

    #[test]
    fn free_block_is_not_movable() {
        let b = block(0, 100, true);
        assert!(!b.is_movable());
    }

    #[test]
    fn pinned_block_is_not_movable() {
        let b = pinned_block(0, 100, 1);
        assert!(!b.is_movable());
    }

    #[test]
    fn unpinned_live_block_is_movable() {
        let b = block(0, 100, false);
        assert!(b.is_movable());
    }

    #[test]
    fn block_equality() {
        let a = block(0, 64, false);
        let b = block(0, 64, false);
        assert_eq!(a, b);
    }

    // ── FragmentationAnalyzer ─────────────────────────────────────────

    #[test]
    fn no_fragmentation_when_single_free_block() {
        let blocks = vec![
            block(0, 256, false),
            block(256, 256, true),
        ];
        let m = FragmentationAnalyzer::analyze(&blocks);
        assert!((m.fragmentation_ratio - 0.0).abs() < f64::EPSILON);
        assert_eq!(m.largest_free_block, 256);
        assert_eq!(m.total_waste, 0);
        assert_eq!(m.free_region_count, 1);
    }

    #[test]
    fn fragmentation_with_two_free_blocks() {
        let blocks = vec![
            block(0, 64, true),   // free
            block(64, 128, false),
            block(192, 192, true), // free
        ];
        let m = FragmentationAnalyzer::analyze(&blocks);
        // total free = 256, largest = 192 → ratio ≈ 0.25
        let expected = 1.0 - 192.0 / 256.0;
        assert!((m.fragmentation_ratio - expected).abs() < 1e-9);
        assert_eq!(m.largest_free_block, 192);
        assert_eq!(m.total_waste, 64); // the 64-byte block
        assert_eq!(m.free_region_count, 2);
    }

    #[test]
    fn fragmentation_all_free() {
        let blocks = vec![block(0, 100, true), block(100, 100, true)];
        let m = FragmentationAnalyzer::analyze(&blocks);
        assert_eq!(m.free_region_count, 2);
        assert_eq!(m.largest_free_block, 100);
        assert!((m.fragmentation_ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn fragmentation_no_free_blocks() {
        let blocks = vec![block(0, 256, false)];
        let m = FragmentationAnalyzer::analyze(&blocks);
        assert!((m.fragmentation_ratio - 0.0).abs() < f64::EPSILON);
        assert_eq!(m.largest_free_block, 0);
        assert_eq!(m.total_waste, 0);
        assert_eq!(m.free_region_count, 0);
    }

    #[test]
    fn fragmentation_empty_block_list() {
        let m = FragmentationAnalyzer::analyze(&[]);
        assert!((m.fragmentation_ratio - 0.0).abs() < f64::EPSILON);
        assert_eq!(m.largest_free_block, 0);
    }

    #[test]
    fn many_small_free_blocks_high_fragmentation() {
        let blocks: Vec<MemoryBlock> = (0..10)
            .flat_map(|i| {
                vec![
                    block(i * 200, 100, false),
                    block(i * 200 + 100, 100, true),
                ]
            })
            .collect();
        let m = FragmentationAnalyzer::analyze(&blocks);
        // 10 equal-size free blocks → ratio = 1 - 1/10 = 0.9
        assert!((m.fragmentation_ratio - 0.9).abs() < f64::EPSILON);
        assert_eq!(m.free_region_count, 10);
    }

    #[test]
    fn single_element_free_block() {
        let blocks = vec![block(0, 1, true)];
        let m = FragmentationAnalyzer::analyze(&blocks);
        assert_eq!(m.free_region_count, 1);
        assert_eq!(m.largest_free_block, 1);
        assert!((m.fragmentation_ratio - 0.0).abs() < f64::EPSILON);
    }

    // ── CompactionPlan ────────────────────────────────────────────────

    #[test]
    fn compaction_plan_total_bytes() {
        let plan = CompactionPlan {
            moves: vec![
                BlockMove { src: 100, dst: 0, size: 50 },
                BlockMove { src: 200, dst: 50, size: 30 },
            ],
        };
        assert_eq!(plan.total_bytes_moved(), 80);
    }

    #[test]
    fn empty_compaction_plan() {
        let plan = CompactionPlan::default();
        assert_eq!(plan.total_bytes_moved(), 0);
        assert!(plan.moves.is_empty());
    }

    // ── Watermarks ────────────────────────────────────────────────────

    #[test]
    fn watermark_default_thresholds() {
        let wm = WatermarkConfig::default();
        assert!((wm.medium - 0.50).abs() < f64::EPSILON);
        assert!((wm.high - 0.75).abs() < f64::EPSILON);
        assert!((wm.critical - 0.90).abs() < f64::EPSILON);
    }

    #[test]
    fn watermark_classify_low() {
        let wm = WatermarkConfig::default();
        assert_eq!(wm.classify(0.0), WatermarkLevel::Low);
        assert_eq!(wm.classify(0.49), WatermarkLevel::Low);
    }

    #[test]
    fn watermark_classify_medium() {
        let wm = WatermarkConfig::default();
        assert_eq!(wm.classify(0.50), WatermarkLevel::Medium);
        assert_eq!(wm.classify(0.74), WatermarkLevel::Medium);
    }

    #[test]
    fn watermark_classify_high() {
        let wm = WatermarkConfig::default();
        assert_eq!(wm.classify(0.75), WatermarkLevel::High);
        assert_eq!(wm.classify(0.89), WatermarkLevel::High);
    }

    #[test]
    fn watermark_classify_critical() {
        let wm = WatermarkConfig::default();
        assert_eq!(wm.classify(0.90), WatermarkLevel::Critical);
        assert_eq!(wm.classify(1.0), WatermarkLevel::Critical);
    }

    #[test]
    fn watermark_custom_thresholds() {
        let wm = WatermarkConfig { medium: 0.3, high: 0.6, critical: 0.8 };
        assert_eq!(wm.classify(0.29), WatermarkLevel::Low);
        assert_eq!(wm.classify(0.30), WatermarkLevel::Medium);
        assert_eq!(wm.classify(0.60), WatermarkLevel::High);
        assert_eq!(wm.classify(0.80), WatermarkLevel::Critical);
    }

    #[test]
    fn watermark_levels_are_ordered() {
        assert!(WatermarkLevel::Low < WatermarkLevel::Medium);
        assert!(WatermarkLevel::Medium < WatermarkLevel::High);
        assert!(WatermarkLevel::High < WatermarkLevel::Critical);
    }

    // ── PressureMonitor ───────────────────────────────────────────────

    #[test]
    fn pressure_monitor_empty() {
        let pm = PressureMonitor::new(1000);
        assert_eq!(pm.current_level(), WatermarkLevel::Low);
        assert_eq!(pm.sample_count(), 0);
    }

    #[test]
    fn pressure_monitor_records_samples() {
        let mut pm = PressureMonitor::new(1000);
        pm.record(500);
        assert_eq!(pm.sample_count(), 1);
        assert_eq!(pm.current_level(), WatermarkLevel::Medium);
    }

    #[test]
    fn pressure_monitor_trend_increasing() {
        let mut pm = PressureMonitor::new(10_000);
        pm.record(1000);
        pm.record(2000);
        pm.record(3000);
        assert!(pm.trend() > 0.0);
    }

    #[test]
    fn pressure_monitor_trend_decreasing() {
        let mut pm = PressureMonitor::new(10_000);
        pm.record(3000);
        pm.record(2000);
        pm.record(1000);
        assert!(pm.trend() < 0.0);
    }

    #[test]
    fn pressure_monitor_trend_flat() {
        let mut pm = PressureMonitor::new(10_000);
        pm.record(5000);
        pm.record(5000);
        pm.record(5000);
        assert!((pm.trend()).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_monitor_trend_single_sample() {
        let mut pm = PressureMonitor::new(10_000);
        pm.record(5000);
        assert!((pm.trend()).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_monitor_predict_oom_rising() {
        let mut pm = PressureMonitor::new(1000);
        pm.record(800);
        pm.record(900);
        // trend = +100 per step → 2 more steps = 1100 → OOM
        assert!(pm.predict_oom(2));
    }

    #[test]
    fn pressure_monitor_predict_no_oom_stable() {
        let mut pm = PressureMonitor::new(1000);
        pm.record(400);
        pm.record(400);
        assert!(!pm.predict_oom(100));
    }

    #[test]
    fn pressure_monitor_predict_no_oom_decreasing() {
        let mut pm = PressureMonitor::new(1000);
        pm.record(900);
        pm.record(800);
        assert!(!pm.predict_oom(10));
    }

    #[test]
    fn pressure_monitor_window_eviction() {
        let mut pm = PressureMonitor::with_watermarks(
            1000,
            WatermarkConfig::default(),
        );
        // Reduce window for testing.
        pm.window_size = 4;
        for i in 0..10 {
            pm.record(i * 100);
        }
        assert_eq!(pm.sample_count(), 4);
    }

    #[test]
    fn pressure_monitor_zero_total() {
        let pm = PressureMonitor::new(0);
        assert_eq!(pm.current_level(), WatermarkLevel::Low);
    }

    #[test]
    fn pressure_monitor_custom_watermarks() {
        let wm = WatermarkConfig { medium: 0.2, high: 0.4, critical: 0.6 };
        let mut pm = PressureMonitor::with_watermarks(1000, wm);
        pm.record(500); // 50% → High under custom thresholds
        assert_eq!(pm.current_level(), WatermarkLevel::High);
    }

    // ── OomRecovery ───────────────────────────────────────────────────

    #[test]
    fn oom_recovery_suggest_evict_only() {
        let blocks = vec![block(0, 1024, false)]; // 1 KiB evictable
        let actions = OomRecovery::suggest(512, &blocks);
        assert_eq!(actions, vec![OomAction::EvictCaches]);
    }

    #[test]
    fn oom_recovery_suggest_evict_and_reduce() {
        let blocks = vec![block(0, 256, false)];
        let actions = OomRecovery::suggest(512, &blocks);
        assert!(actions.contains(&OomAction::EvictCaches));
        assert!(actions.contains(&OomAction::ReduceBatchSize));
    }

    #[test]
    fn oom_recovery_suggest_all_three() {
        let blocks = vec![block(0, 64, false)];
        let actions = OomRecovery::suggest(1024, &blocks);
        assert_eq!(actions.len(), 3);
        assert_eq!(actions[0], OomAction::EvictCaches);
        assert_eq!(actions[1], OomAction::ReduceBatchSize);
        assert_eq!(actions[2], OomAction::OffloadToCpu);
    }

    #[test]
    fn oom_recovery_no_evictable_blocks() {
        let blocks = vec![pinned_block(0, 256, 1)];
        let actions = OomRecovery::suggest(512, &blocks);
        // No evictable memory → reduce + offload
        assert!(!actions.contains(&OomAction::EvictCaches));
        assert!(actions.contains(&OomAction::ReduceBatchSize));
    }

    #[test]
    fn oom_recovery_empty_blocks() {
        let actions = OomRecovery::suggest(100, &[]);
        assert!(actions.contains(&OomAction::ReduceBatchSize));
        assert!(actions.contains(&OomAction::OffloadToCpu));
    }

    #[test]
    fn oom_recovery_zero_shortfall() {
        let blocks = vec![block(0, 1024, false)];
        let actions = OomRecovery::suggest(0, &blocks);
        assert_eq!(actions, vec![OomAction::EvictCaches]);
    }

    // ── AllocationHistory ─────────────────────────────────────────────

    #[test]
    fn history_empty() {
        let h = AllocationHistory::new(10);
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert_eq!(h.net_allocated(), 0);
    }

    #[test]
    fn history_record_and_iterate() {
        let mut h = AllocationHistory::new(10);
        h.record(true, 128, 0);
        h.record(false, 64, 0);
        assert_eq!(h.len(), 2);
        let events: Vec<_> = h.iter().collect();
        assert!(events[0].is_alloc);
        assert!(!events[1].is_alloc);
    }

    #[test]
    fn history_net_allocated() {
        let mut h = AllocationHistory::new(10);
        h.record(true, 256, 0);
        h.record(true, 128, 256);
        h.record(false, 256, 0);
        assert_eq!(h.net_allocated(), 128);
    }

    #[test]
    fn history_capacity_eviction() {
        let mut h = AllocationHistory::new(3);
        h.record(true, 10, 0);
        h.record(true, 20, 10);
        h.record(true, 30, 30);
        h.record(true, 40, 60); // evicts first
        assert_eq!(h.len(), 3);
        let first = h.iter().next().unwrap();
        assert_eq!(first.size, 20);
    }

    #[test]
    fn history_sequence_numbers() {
        let mut h = AllocationHistory::new(10);
        h.record(true, 1, 0);
        h.record(true, 2, 1);
        h.record(true, 3, 3);
        let seqs: Vec<u64> = h.iter().map(|e| e.seq).collect();
        assert_eq!(seqs, vec![0, 1, 2]);
    }

    // ── MemoryDefragmenter ────────────────────────────────────────────

    #[test]
    fn defrag_default_strategy() {
        let d = MemoryDefragmenter::new(DefragConfig::default());
        assert_eq!(d.strategy(), DefragStrategy::Compaction);
    }

    #[test]
    fn defrag_failure_counting() {
        let mut d = MemoryDefragmenter::new(DefragConfig::default());
        assert_eq!(d.failure_count(), 0);
        d.report_failure();
        d.report_failure();
        assert_eq!(d.failure_count(), 2);
        d.reset_failures();
        assert_eq!(d.failure_count(), 0);
    }

    #[test]
    fn defrag_should_defrag_on_failure_count() {
        let cfg = DefragConfig { compaction_trigger: 2, ..Default::default() };
        let mut d = MemoryDefragmenter::new(cfg);
        let blocks = vec![block(0, 1024, false)];
        assert!(!d.should_defrag(&blocks));
        d.report_failure();
        d.report_failure();
        assert!(d.should_defrag(&blocks));
    }

    #[test]
    fn defrag_should_defrag_on_high_fragmentation() {
        let cfg = DefragConfig {
            fragmentation_threshold: 0.2,
            ..Default::default()
        };
        let d = MemoryDefragmenter::new(cfg);
        // Two equal free blocks → ratio = 0.5 > 0.2
        let blocks = vec![
            block(0, 100, true),
            block(100, 100, false),
            block(200, 100, true),
        ];
        assert!(d.should_defrag(&blocks));
    }

    #[test]
    fn defrag_no_defrag_low_fragmentation() {
        let cfg = DefragConfig {
            fragmentation_threshold: 0.9,
            ..Default::default()
        };
        let d = MemoryDefragmenter::new(cfg);
        let blocks = vec![
            block(0, 100, false),
            block(100, 900, true),
        ];
        assert!(!d.should_defrag(&blocks));
    }

    #[test]
    fn plan_compaction_empty() {
        let d = MemoryDefragmenter::new(DefragConfig::default());
        let plan = d.plan_compaction(&[]).unwrap();
        assert!(plan.moves.is_empty());
    }

    #[test]
    fn plan_compaction_no_fragmentation() {
        let d = MemoryDefragmenter::new(DefragConfig::default());
        let blocks = vec![
            block(0, 128, false),
            block(128, 128, false),
            block(256, 256, true),
        ];
        let plan = d.plan_compaction(&blocks).unwrap();
        assert!(plan.moves.is_empty());
    }

    #[test]
    fn plan_compaction_single_gap() {
        let d = MemoryDefragmenter::new(DefragConfig::default());
        let blocks = vec![
            block(0, 64, true),   // gap
            block(64, 128, false), // should move to 0
        ];
        let plan = d.plan_compaction(&blocks).unwrap();
        assert_eq!(plan.moves.len(), 1);
        assert_eq!(plan.moves[0].src, 64);
        assert_eq!(plan.moves[0].dst, 0);
        assert_eq!(plan.moves[0].size, 128);
    }

    #[test]
    fn plan_compaction_multiple_gaps() {
        let d = MemoryDefragmenter::new(DefragConfig::default());
        let blocks = vec![
            block(0, 32, true),
            block(32, 64, false),
            block(96, 32, true),
            block(128, 64, false),
        ];
        let plan = d.plan_compaction(&blocks).unwrap();
        assert_eq!(plan.moves.len(), 2);
        assert_eq!(plan.moves[0].dst, 0);
        assert_eq!(plan.moves[1].dst, 64);
    }

    #[test]
    fn plan_compaction_skips_pinned_blocks() {
        let d = MemoryDefragmenter::new(DefragConfig::default());
        let blocks = vec![
            block(0, 64, true),
            pinned_block(64, 128, 1), // pinned, stays put
            block(192, 64, true),
            block(256, 64, false), // movable
        ];
        let plan = d.plan_compaction(&blocks).unwrap();
        // Pinned block at 64 keeps cursor at 192; movable block at
        // 256 should move to 192.
        assert_eq!(plan.moves.len(), 1);
        assert_eq!(plan.moves[0].src, 256);
        assert_eq!(plan.moves[0].dst, 192);
    }

    #[test]
    fn plan_compaction_respects_max_move_size() {
        let cfg = DefragConfig { max_move_size: 32, ..Default::default() };
        let d = MemoryDefragmenter::new(cfg);
        let blocks = vec![
            block(0, 16, true),
            block(16, 64, false), // too large to move
        ];
        let plan = d.plan_compaction(&blocks).unwrap();
        assert!(plan.moves.is_empty());
    }

    #[test]
    fn run_compaction_resets_failures() {
        let cfg = DefragConfig::default();
        let mut d = MemoryDefragmenter::new(cfg);
        d.report_failure();
        d.report_failure();
        assert_eq!(d.failure_count(), 2);

        let blocks = vec![
            block(0, 64, true),
            block(64, 128, false),
        ];
        let _plan = d.run(&blocks).unwrap();
        assert_eq!(d.failure_count(), 0);
    }

    #[test]
    fn run_buddy_rebalance_falls_back_to_compaction() {
        let cfg = DefragConfig {
            strategy: DefragStrategy::BuddyRebalance,
            ..Default::default()
        };
        let mut d = MemoryDefragmenter::new(cfg);
        let blocks = vec![
            block(0, 32, true),
            block(32, 64, false),
        ];
        let plan = d.run(&blocks).unwrap();
        assert_eq!(plan.moves.len(), 1);
    }

    #[test]
    fn run_generational_collect_falls_back() {
        let cfg = DefragConfig {
            strategy: DefragStrategy::GenerationalCollect,
            ..Default::default()
        };
        let mut d = MemoryDefragmenter::new(cfg);
        let blocks = vec![
            block(0, 16, true),
            block(16, 32, false),
        ];
        let plan = d.run(&blocks).unwrap();
        assert_eq!(plan.moves.len(), 1);
    }

    #[test]
    fn run_incremental_uses_compaction() {
        let cfg = DefragConfig {
            strategy: DefragStrategy::Incremental,
            ..Default::default()
        };
        let mut d = MemoryDefragmenter::new(cfg);
        let blocks = vec![
            block(0, 8, true),
            block(8, 16, false),
        ];
        let plan = d.run(&blocks).unwrap();
        assert_eq!(plan.moves.len(), 1);
    }

    // ── Integration / scenario tests ──────────────────────────────────

    #[test]
    fn scenario_full_defrag_cycle() {
        let cfg = DefragConfig {
            fragmentation_threshold: 0.2,
            compaction_trigger: 2,
            max_move_size: 4096,
            strategy: DefragStrategy::Compaction,
        };
        let mut defrag = MemoryDefragmenter::new(cfg);

        let blocks = vec![
            block(0, 128, true),
            block(128, 256, false),
            block(384, 64, true),
            block(448, 128, false),
        ];

        // Step 1: check fragmentation triggers defrag
        assert!(defrag.should_defrag(&blocks));

        // Step 2: run defrag
        let plan = defrag.run(&blocks).unwrap();
        assert!(!plan.moves.is_empty());
        assert_eq!(defrag.failure_count(), 0);
    }

    #[test]
    fn scenario_pressure_to_oom_recovery() {
        let mut pm = PressureMonitor::new(1000);
        pm.record(600);
        pm.record(800);
        pm.record(950);
        assert_eq!(pm.current_level(), WatermarkLevel::Critical);
        assert!(pm.predict_oom(1));

        let blocks = vec![
            block(0, 200, false),
            pinned_block(200, 300, 1),
            block(500, 500, true),
        ];
        let actions = OomRecovery::suggest(256, &blocks);
        assert!(actions.contains(&OomAction::EvictCaches));
    }

    #[test]
    fn scenario_history_tracks_allocations() {
        let mut hist = AllocationHistory::new(100);
        hist.record(true, 1024, 0);
        hist.record(true, 512, 1024);
        hist.record(false, 1024, 0);
        assert_eq!(hist.net_allocated(), 512);
        assert_eq!(hist.len(), 3);
    }

    #[test]
    fn generation_metadata_preserved() {
        let b = gen_block(0, 256, 5);
        assert_eq!(b.generation, 5);
        assert!(b.is_movable());
    }

    #[test]
    fn compaction_plan_preserves_order() {
        let d = MemoryDefragmenter::new(DefragConfig::default());
        let blocks = vec![
            block(0, 16, true),
            block(16, 32, false),  // → 0
            block(48, 16, true),
            block(64, 32, false),  // → 32
            block(96, 16, true),
            block(112, 32, false), // → 64
        ];
        let plan = d.plan_compaction(&blocks).unwrap();
        assert_eq!(plan.moves.len(), 3);
        // Destinations must be strictly increasing.
        for w in plan.moves.windows(2) {
            assert!(w[0].dst < w[1].dst);
        }
    }

    #[test]
    fn defrag_strategy_enum_variants() {
        let strategies = [
            DefragStrategy::Compaction,
            DefragStrategy::BuddyRebalance,
            DefragStrategy::GenerationalCollect,
            DefragStrategy::Incremental,
        ];
        // Each variant is distinct.
        for (i, a) in strategies.iter().enumerate() {
            for (j, b) in strategies.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn pressure_monitor_oom_exact_boundary() {
        let mut pm = PressureMonitor::new(100);
        pm.record(90);
        pm.record(95);
        // trend = 5, 1 step → 100 → exactly at limit
        assert!(pm.predict_oom(1));
    }

    #[test]
    fn block_move_debug() {
        let m = BlockMove { src: 0, dst: 100, size: 50 };
        let dbg = format!("{m:?}");
        assert!(dbg.contains("BlockMove"));
    }

    #[test]
    fn fragmentation_metrics_debug() {
        let m = FragmentationAnalyzer::analyze(&[]);
        let dbg = format!("{m:?}");
        assert!(dbg.contains("FragmentationMetrics"));
    }
}
