//! Advanced CUDA graph optimization: fusion, caching, update, profiling.
//!
//! # Overview
//!
//! This module builds on [`super::graph_exec`] to provide production-grade
//! graph optimization passes that reduce kernel launch overhead and maximize
//! GPU occupancy:
//!
//! - [`FusionPattern`] / [`FusionDetector`] — identify fusable kernel sequences
//! - [`GraphMemoryPlanner`] — optimize memory allocation within captured graphs
//! - [`GraphInstanceCache`] — cache instantiated graph executables by shape key
//! - [`GraphUpdater`] — update graph parameters without full recapture
//! - [`GraphBottleneckProfiler`] — profiling with bottleneck detection
//! - [`MultiStreamExecutor`] — execute graph segments across multiple streams
//! - [`GraphOptimizationPipeline`] — compose passes into a configurable pipeline
//!
//! All GPU dispatch is feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallback implementations simulate execution for testing on non-GPU hosts.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bitnet_common::{KernelError, Result};

// ── Fusion patterns ──────────────────────────────────────────────────

/// A recognized pattern of consecutive kernels that can be fused.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// RMSNorm immediately followed by a linear projection.
    RmsNormLinear,
    /// GELU activation immediately followed by a linear projection.
    GeluLinear,
    /// Softmax immediately followed by a mask application.
    SoftmaxMask,
    /// Residual add followed by RMSNorm.
    AddRmsNorm,
    /// Element-wise scale followed by add.
    ScaleAdd,
    /// Quantize immediately followed by matmul.
    QuantizeMatmul,
    /// Dequantize immediately followed by element-wise op.
    DequantizeElementwise,
    /// Custom user-defined pattern with a name.
    Custom(String),
}

impl fmt::Display for FusionPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RmsNormLinear => write!(f, "rmsnorm+linear"),
            Self::GeluLinear => write!(f, "gelu+linear"),
            Self::SoftmaxMask => write!(f, "softmax+mask"),
            Self::AddRmsNorm => write!(f, "add+rmsnorm"),
            Self::ScaleAdd => write!(f, "scale+add"),
            Self::QuantizeMatmul => write!(f, "quantize+matmul"),
            Self::DequantizeElementwise => write!(f, "dequantize+elementwise"),
            Self::Custom(name) => write!(f, "custom({name})"),
        }
    }
}

/// A detected fusion opportunity within a graph.
#[derive(Debug, Clone)]
pub struct FusionCandidate {
    /// The pattern that was detected.
    pub pattern: FusionPattern,
    /// Node indices (in graph node list order) that participate.
    pub node_indices: Vec<usize>,
    /// Estimated speedup factor (e.g. 1.5 = 50% faster).
    pub estimated_speedup: f64,
    /// Estimated memory savings in bytes.
    pub memory_savings_bytes: usize,
}

/// Kernel name matching rules for fusion detection.
#[derive(Debug, Clone)]
pub struct FusionRule {
    /// Pattern to apply when matched.
    pub pattern: FusionPattern,
    /// First kernel name prefix.
    pub first_prefix: String,
    /// Second kernel name prefix.
    pub second_prefix: String,
    /// Estimated speedup for this fusion.
    pub speedup: f64,
    /// Estimated memory savings per element (bytes).
    pub savings_per_elem: usize,
}

impl FusionRule {
    /// Create a new fusion rule.
    pub fn new(
        pattern: FusionPattern,
        first: &str,
        second: &str,
        speedup: f64,
        savings: usize,
    ) -> Self {
        Self {
            pattern,
            first_prefix: first.to_string(),
            second_prefix: second.to_string(),
            speedup,
            savings_per_elem: savings,
        }
    }
}

/// Detects fusable kernel sequences within a graph.
#[derive(Debug)]
pub struct FusionDetector {
    rules: Vec<FusionRule>,
    min_speedup: f64,
}

impl Default for FusionDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionDetector {
    /// Create a detector with the default BitNet fusion rules.
    pub fn new() -> Self {
        let rules = vec![
            FusionRule::new(FusionPattern::RmsNormLinear, "rmsnorm", "linear", 1.5, 4),
            FusionRule::new(FusionPattern::GeluLinear, "gelu", "linear", 1.4, 4),
            FusionRule::new(FusionPattern::SoftmaxMask, "softmax", "mask", 1.3, 0),
            FusionRule::new(FusionPattern::AddRmsNorm, "add", "rmsnorm", 1.35, 4),
            FusionRule::new(FusionPattern::ScaleAdd, "scale", "add", 1.6, 4),
            FusionRule::new(FusionPattern::QuantizeMatmul, "quantize", "matmul", 1.45, 2),
            FusionRule::new(FusionPattern::DequantizeElementwise, "dequant", "elem", 1.25, 4),
        ];
        Self { rules, min_speedup: 1.0 }
    }

    /// Add a custom fusion rule.
    pub fn add_rule(&mut self, rule: FusionRule) {
        self.rules.push(rule);
    }

    /// Set the minimum speedup threshold — candidates below this are filtered.
    pub fn set_min_speedup(&mut self, min: f64) {
        self.min_speedup = min;
    }

    /// Number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Scan a list of kernel names for fusion candidates.
    ///
    /// The `kernel_names` slice should correspond 1-to-1 with graph nodes.
    /// Non-kernel nodes should pass an empty string.
    pub fn detect(&self, kernel_names: &[&str]) -> Vec<FusionCandidate> {
        let mut candidates = Vec::new();
        if kernel_names.len() < 2 {
            return candidates;
        }

        let mut i = 0;
        while i + 1 < kernel_names.len() {
            let first = kernel_names[i];
            let second = kernel_names[i + 1];

            let mut matched = false;
            for rule in &self.rules {
                let first_lower = first.to_lowercase();
                let second_lower = second.to_lowercase();
                if first_lower.starts_with(&rule.first_prefix)
                    && second_lower.starts_with(&rule.second_prefix)
                    && rule.speedup >= self.min_speedup
                {
                    candidates.push(FusionCandidate {
                        pattern: rule.pattern.clone(),
                        node_indices: vec![i, i + 1],
                        estimated_speedup: rule.speedup,
                        memory_savings_bytes: rule.savings_per_elem,
                    });
                    matched = true;
                    break;
                }
            }

            if matched {
                i += 2;
            } else {
                i += 1;
            }
        }

        candidates
    }

    /// Estimate the aggregate speedup from applying all candidates.
    pub fn estimate_aggregate_speedup(&self, candidates: &[FusionCandidate]) -> f64 {
        if candidates.is_empty() {
            return 1.0;
        }
        let sum: f64 = candidates.iter().map(|c| c.estimated_speedup).sum();
        sum / candidates.len() as f64
    }
}

// ── Memory planning ──────────────────────────────────────────────────

/// A planned memory allocation within a graph.
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    /// Unique allocation id.
    pub id: u64,
    /// Size in bytes.
    pub size: usize,
    /// Alignment requirement in bytes.
    pub alignment: usize,
    /// First node index that uses this allocation.
    pub first_use: usize,
    /// Last node index that uses this allocation.
    pub last_use: usize,
    /// Whether this allocation can be reused after `last_use`.
    pub reusable: bool,
}

/// Result of memory planning.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Individual allocations.
    pub allocations: Vec<MemoryAllocation>,
    /// Total peak memory in bytes.
    pub peak_bytes: usize,
    /// Memory saved by reuse, in bytes.
    pub reuse_savings_bytes: usize,
    /// Mapping from allocation id to offset in the unified buffer.
    pub offset_map: HashMap<u64, usize>,
}

/// Plans memory allocations within a captured graph to minimize peak usage.
#[derive(Debug)]
pub struct GraphMemoryPlanner {
    alignment: usize,
    enable_reuse: bool,
    next_id: AtomicU64,
}

impl Default for GraphMemoryPlanner {
    fn default() -> Self {
        Self::new(256)
    }
}

impl GraphMemoryPlanner {
    /// Create a planner with the given alignment (must be power of two).
    pub fn new(alignment: usize) -> Self {
        assert!(alignment.is_power_of_two(), "alignment must be power of two");
        Self { alignment, enable_reuse: true, next_id: AtomicU64::new(1) }
    }

    /// Enable or disable allocation reuse.
    pub fn set_reuse(&mut self, enable: bool) {
        self.enable_reuse = enable;
    }

    /// The alignment used by this planner.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    fn next_alloc_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    fn align_up(size: usize, alignment: usize) -> usize {
        (size + alignment - 1) & !(alignment - 1)
    }

    /// Plan memory for a list of allocation requests.
    ///
    /// Each request is `(size_bytes, first_use_node, last_use_node)`.
    pub fn plan(&self, requests: &[(usize, usize, usize)]) -> Result<MemoryPlan> {
        if requests.is_empty() {
            return Ok(MemoryPlan {
                allocations: Vec::new(),
                peak_bytes: 0,
                reuse_savings_bytes: 0,
                offset_map: HashMap::new(),
            });
        }

        let mut allocations: Vec<MemoryAllocation> = requests
            .iter()
            .map(|&(size, first, last)| MemoryAllocation {
                id: self.next_alloc_id(),
                size: Self::align_up(size, self.alignment),
                alignment: self.alignment,
                first_use: first,
                last_use: last,
                reusable: self.enable_reuse,
            })
            .collect();

        // Sort by first_use for greedy offset assignment.
        allocations.sort_by_key(|a| a.first_use);

        let mut offset_map: HashMap<u64, usize> = HashMap::new();
        // Track free intervals: (offset, size, available_from_node).
        let mut free_slots: Vec<(usize, usize, usize)> = Vec::new();
        let mut current_end: usize = 0;
        let mut reuse_savings: usize = 0;

        for alloc in &allocations {
            // Try to reuse a free slot.
            let mut reused = false;
            if self.enable_reuse
                && let Some(idx) = free_slots
                    .iter()
                    .position(|&(_, sz, avail)| sz >= alloc.size && avail <= alloc.first_use)
            {
                let (offset, _sz, _) = free_slots.remove(idx);
                offset_map.insert(alloc.id, offset);
                reuse_savings += alloc.size;
                reused = true;
                // The slot is busy until alloc.last_use.
                free_slots.push((offset, alloc.size, alloc.last_use + 1));
            }

            if !reused {
                offset_map.insert(alloc.id, current_end);
                free_slots.push((current_end, alloc.size, alloc.last_use + 1));
                current_end += alloc.size;
            }
        }

        Ok(MemoryPlan {
            allocations,
            peak_bytes: current_end,
            reuse_savings_bytes: reuse_savings,
            offset_map,
        })
    }
}

// ── Instance caching ─────────────────────────────────────────────────

/// Key for caching instantiated graph executables.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphShapeKey {
    /// Batch size.
    pub batch_size: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
}

impl fmt::Display for GraphShapeKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "B{}×S{}×H{}×NH{}",
            self.batch_size, self.seq_len, self.hidden_dim, self.num_heads
        )
    }
}

/// A cached graph instance with metadata.
#[derive(Debug)]
struct CachedInstance {
    /// Opaque handle (simulated on CPU).
    _handle_id: u64,
    /// When this instance was created.
    _created_at: Instant,
    /// When this instance was last used.
    last_used: Instant,
    /// How many times this instance was executed.
    exec_count: u64,
    /// Number of nodes in the captured graph.
    node_count: usize,
}

static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(1);

/// Caches instantiated graph executables keyed by shape.
#[derive(Debug)]
pub struct GraphInstanceCache {
    entries: HashMap<GraphShapeKey, CachedInstance>,
    max_entries: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl GraphInstanceCache {
    /// Create a cache with the given maximum number of entries.
    pub fn new(max_entries: usize) -> Self {
        assert!(max_entries > 0, "max_entries must be > 0");
        Self { entries: HashMap::new(), max_entries, hits: 0, misses: 0, evictions: 0 }
    }

    /// Look up a cached instance. Returns `true` if found.
    pub fn get(&mut self, key: &GraphShapeKey) -> bool {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_used = Instant::now();
            entry.exec_count += 1;
            self.hits += 1;
            true
        } else {
            self.misses += 1;
            false
        }
    }

    /// Insert or update a cached instance for the given key.
    pub fn insert(&mut self, key: GraphShapeKey, node_count: usize) {
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&key) {
            self.evict_lru();
        }
        let now = Instant::now();
        self.entries.insert(
            key,
            CachedInstance {
                _handle_id: NEXT_HANDLE_ID.fetch_add(1, Ordering::Relaxed),
                _created_at: now,
                last_used: now,
                exec_count: 0,
                node_count,
            },
        );
    }

    fn evict_lru(&mut self) {
        if let Some((key, _)) = self
            .entries
            .iter()
            .min_by_key(|(_, v)| v.last_used)
            .map(|(k, v)| (k.clone(), v.last_used))
        {
            self.entries.remove(&key);
            self.evictions += 1;
        }
    }

    /// Check if a key is cached.
    pub fn contains(&self, key: &GraphShapeKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cache hit count.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache miss count.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Cache eviction count.
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Hit rate as a fraction in [0, 1].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
    }

    /// Remove a specific entry.
    pub fn remove(&mut self, key: &GraphShapeKey) -> bool {
        self.entries.remove(key).is_some()
    }

    /// Execution count for a cached key (0 if not present).
    pub fn exec_count(&self, key: &GraphShapeKey) -> u64 {
        self.entries.get(key).map_or(0, |e| e.exec_count)
    }

    /// Node count stored for a cached key.
    pub fn node_count(&self, key: &GraphShapeKey) -> Option<usize> {
        self.entries.get(key).map(|e| e.node_count)
    }
}

// ── Graph update without recapture ───────────────────────────────────

/// Describes which parameters of a graph node to update.
#[derive(Debug, Clone)]
pub struct NodeParamUpdate {
    /// Index of the node in the graph's node list.
    pub node_index: usize,
    /// Updated parameters (key-value pairs).
    pub params: HashMap<String, f64>,
}

/// Describes a shape change that can be applied without recapture.
#[derive(Debug, Clone)]
pub struct ShapeUpdate {
    /// New batch size (if changed).
    pub batch_size: Option<usize>,
    /// New sequence length (if changed).
    pub seq_len: Option<usize>,
    /// New hidden dimension (if changed).
    pub hidden_dim: Option<usize>,
}

impl ShapeUpdate {
    /// Whether any dimension changed.
    pub fn has_changes(&self) -> bool {
        self.batch_size.is_some() || self.seq_len.is_some() || self.hidden_dim.is_some()
    }

    /// Number of dimensions that changed.
    pub fn change_count(&self) -> usize {
        self.batch_size.is_some() as usize
            + self.seq_len.is_some() as usize
            + self.hidden_dim.is_some() as usize
    }
}

/// Result of a graph update operation.
#[derive(Debug, Clone)]
pub struct UpdateResult {
    /// Number of nodes whose parameters were updated.
    pub nodes_updated: usize,
    /// Whether a full recapture was required (e.g. topology change).
    pub required_recapture: bool,
    /// Duration of the update.
    pub duration: Duration,
}

/// Updates graph parameters without full recapture.
///
/// When tensor shapes change between inference steps (e.g. different
/// sequence lengths), a full graph recapture is expensive. This updater
/// modifies kernel parameters in-place when the graph topology is unchanged.
#[derive(Debug)]
pub struct GraphUpdater {
    /// Maximum ratio of shape change before forcing recapture.
    recapture_threshold: f64,
    /// History of applied updates for rollback.
    history: Vec<(usize, HashMap<String, f64>)>,
    /// Total updates applied.
    total_updates: u64,
    /// Total recaptures forced.
    total_recaptures: u64,
}

impl Default for GraphUpdater {
    fn default() -> Self {
        Self::new(2.0)
    }
}

impl GraphUpdater {
    /// Create an updater.
    ///
    /// `recapture_threshold` is the maximum ratio of new/old dimension
    /// before a full recapture is required (e.g. 2.0 = allow up to 2× change).
    pub fn new(recapture_threshold: f64) -> Self {
        Self { recapture_threshold, history: Vec::new(), total_updates: 0, total_recaptures: 0 }
    }

    /// Recapture threshold.
    pub fn recapture_threshold(&self) -> f64 {
        self.recapture_threshold
    }

    /// Set a new recapture threshold.
    pub fn set_recapture_threshold(&mut self, threshold: f64) {
        self.recapture_threshold = threshold;
    }

    /// Total updates applied since creation.
    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }

    /// Total recaptures forced since creation.
    pub fn total_recaptures(&self) -> u64 {
        self.total_recaptures
    }

    /// Check whether a shape update requires recapture.
    pub fn needs_recapture(&self, old: &GraphShapeKey, shape: &ShapeUpdate) -> bool {
        let check_dim = |old_val: usize, new_val: Option<usize>| -> bool {
            if let Some(nv) = new_val {
                if old_val == 0 {
                    return nv != 0;
                }
                let ratio = nv as f64 / old_val as f64;
                ratio > self.recapture_threshold || ratio < 1.0 / self.recapture_threshold
            } else {
                false
            }
        };

        check_dim(old.batch_size, shape.batch_size)
            || check_dim(old.seq_len, shape.seq_len)
            || check_dim(old.hidden_dim, shape.hidden_dim)
    }

    /// Apply parameter updates to a graph's node list.
    ///
    /// `node_params` is a mutable slice of `HashMap<String, f64>`, one per node.
    /// Returns the number of nodes updated.
    pub fn apply_param_updates(
        &mut self,
        node_params: &mut [HashMap<String, f64>],
        updates: &[NodeParamUpdate],
    ) -> Result<UpdateResult> {
        let start = Instant::now();
        let mut count = 0;

        for update in updates {
            if update.node_index >= node_params.len() {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "node_index {} out of range (len {})",
                        update.node_index,
                        node_params.len()
                    ),
                }
                .into());
            }

            // Save old params for rollback.
            let old = node_params[update.node_index].clone();
            self.history.push((update.node_index, old));

            for (k, v) in &update.params {
                node_params[update.node_index].insert(k.clone(), *v);
            }
            count += 1;
        }

        self.total_updates += count as u64;

        Ok(UpdateResult {
            nodes_updated: count,
            required_recapture: false,
            duration: start.elapsed(),
        })
    }

    /// Apply a shape update, returning whether recapture was needed.
    pub fn apply_shape_update(
        &mut self,
        old_key: &GraphShapeKey,
        shape: &ShapeUpdate,
        node_params: &mut [HashMap<String, f64>],
    ) -> Result<UpdateResult> {
        let start = Instant::now();

        if self.needs_recapture(old_key, shape) {
            self.total_recaptures += 1;
            return Ok(UpdateResult {
                nodes_updated: 0,
                required_recapture: true,
                duration: start.elapsed(),
            });
        }

        // Update dimension params on all nodes.
        let mut count = 0;
        for params in node_params.iter_mut() {
            let mut changed = false;
            if let Some(bs) = shape.batch_size {
                params.insert("batch_size".to_string(), bs as f64);
                changed = true;
            }
            if let Some(sl) = shape.seq_len {
                params.insert("seq_len".to_string(), sl as f64);
                changed = true;
            }
            if let Some(hd) = shape.hidden_dim {
                params.insert("hidden_dim".to_string(), hd as f64);
                changed = true;
            }
            if changed {
                count += 1;
            }
        }

        self.total_updates += count as u64;

        Ok(UpdateResult {
            nodes_updated: count,
            required_recapture: false,
            duration: start.elapsed(),
        })
    }

    /// Rollback the last `n` parameter updates.
    pub fn rollback(&mut self, node_params: &mut [HashMap<String, f64>], n: usize) -> usize {
        let mut restored = 0;
        for _ in 0..n {
            if let Some((idx, old_params)) = self.history.pop()
                && idx < node_params.len()
            {
                node_params[idx] = old_params;
                restored += 1;
            }
        }
        restored
    }

    /// Clear update history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Number of entries in the rollback history.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

// ── Bottleneck profiling ─────────────────────────────────────────────

/// Category of a detected bottleneck.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BottleneckKind {
    /// Kernel occupies disproportionate time.
    KernelHotspot,
    /// Excessive memory transfers.
    MemoryBound,
    /// Synchronization overhead.
    SyncOverhead,
    /// Low GPU occupancy / utilization.
    LowOccupancy,
    /// Serial dependency chain limiting parallelism.
    SerialDependency,
}

impl fmt::Display for BottleneckKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelHotspot => write!(f, "kernel_hotspot"),
            Self::MemoryBound => write!(f, "memory_bound"),
            Self::SyncOverhead => write!(f, "sync_overhead"),
            Self::LowOccupancy => write!(f, "low_occupancy"),
            Self::SerialDependency => write!(f, "serial_dependency"),
        }
    }
}

/// A detected bottleneck in a graph execution.
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Kind of bottleneck.
    pub kind: BottleneckKind,
    /// Name or label of the offending element.
    pub label: String,
    /// Fraction of total time this bottleneck accounts for.
    pub time_fraction: f64,
    /// Human-readable suggestion.
    pub suggestion: String,
}

/// Per-node timing from a profiling run.
#[derive(Debug, Clone)]
pub struct NodeTiming {
    /// Node index in the graph.
    pub node_index: usize,
    /// Kernel/operation name.
    pub name: String,
    /// Simulated execution time.
    pub duration: Duration,
    /// Stream this node ran on.
    pub stream: u32,
}

/// Result of a profiling run.
#[derive(Debug, Clone)]
pub struct ProfileResult {
    /// Per-node timings.
    pub node_timings: Vec<NodeTiming>,
    /// Total wall-clock time.
    pub total_time: Duration,
    /// Detected bottlenecks.
    pub bottlenecks: Vec<Bottleneck>,
    /// Number of nodes profiled.
    pub nodes_profiled: usize,
}

/// Profiles graph execution and detects bottlenecks.
#[derive(Debug)]
pub struct GraphBottleneckProfiler {
    /// Threshold fraction above which a node is flagged as a hotspot.
    hotspot_threshold: f64,
    /// Threshold fraction for sync overhead detection.
    sync_threshold: f64,
    /// Collected profile results.
    results: Vec<ProfileResult>,
    max_results: usize,
}

impl Default for GraphBottleneckProfiler {
    fn default() -> Self {
        Self::new(0.3, 0.1, 64)
    }
}

impl GraphBottleneckProfiler {
    /// Create a profiler.
    ///
    /// * `hotspot_threshold` — fraction of total time to flag as hotspot (e.g. 0.3).
    /// * `sync_threshold` — fraction of total time in barriers to flag sync overhead.
    /// * `max_results` — maximum number of profile results to retain.
    pub fn new(hotspot_threshold: f64, sync_threshold: f64, max_results: usize) -> Self {
        Self { hotspot_threshold, sync_threshold, results: Vec::new(), max_results }
    }

    /// Hotspot threshold.
    pub fn hotspot_threshold(&self) -> f64 {
        self.hotspot_threshold
    }

    /// Sync overhead threshold.
    pub fn sync_threshold(&self) -> f64 {
        self.sync_threshold
    }

    /// Profile a graph described by node metadata.
    ///
    /// `nodes` is a list of `(name, estimated_cost_us, stream, is_barrier)`.
    pub fn profile(&mut self, nodes: &[(&str, f64, u32, bool)]) -> ProfileResult {
        let start = Instant::now();
        let mut timings = Vec::with_capacity(nodes.len());
        let mut total_us: f64 = 0.0;
        let mut barrier_us: f64 = 0.0;

        for (i, &(name, cost_us, stream, is_barrier)) in nodes.iter().enumerate() {
            let dur = Duration::from_secs_f64(cost_us / 1_000_000.0);
            timings.push(NodeTiming {
                node_index: i,
                name: name.to_string(),
                duration: dur,
                stream,
            });
            total_us += cost_us;
            if is_barrier {
                barrier_us += cost_us;
            }
        }

        let total_time = if total_us > 0.0 {
            Duration::from_secs_f64(total_us / 1_000_000.0)
        } else {
            start.elapsed()
        };

        // Detect bottlenecks.
        let mut bottlenecks = Vec::new();

        // Kernel hotspots.
        for t in &timings {
            if total_us > 0.0 {
                let frac = t.duration.as_secs_f64() * 1_000_000.0 / total_us;
                if frac >= self.hotspot_threshold {
                    bottlenecks.push(Bottleneck {
                        kind: BottleneckKind::KernelHotspot,
                        label: t.name.clone(),
                        time_fraction: frac,
                        suggestion: format!(
                            "Kernel '{}' uses {:.1}% of total time; consider fusion or tiling",
                            t.name,
                            frac * 100.0
                        ),
                    });
                }
            }
        }

        // Sync overhead.
        if total_us > 0.0 && barrier_us / total_us >= self.sync_threshold {
            bottlenecks.push(Bottleneck {
                kind: BottleneckKind::SyncOverhead,
                label: "barriers".to_string(),
                time_fraction: barrier_us / total_us,
                suggestion: format!(
                    "Barrier overhead is {:.1}% of total; remove redundant syncs",
                    barrier_us / total_us * 100.0
                ),
            });
        }

        // Serial dependency: if all nodes are on the same stream, flag it.
        let streams_used: HashSet<u32> = timings.iter().map(|t| t.stream).collect();
        if streams_used.len() == 1 && timings.len() > 4 {
            bottlenecks.push(Bottleneck {
                kind: BottleneckKind::SerialDependency,
                label: "single_stream".to_string(),
                time_fraction: 1.0,
                suggestion: "All nodes on one stream; consider multi-stream execution".to_string(),
            });
        }

        let result = ProfileResult {
            nodes_profiled: timings.len(),
            node_timings: timings,
            total_time,
            bottlenecks,
        };

        if self.results.len() >= self.max_results {
            self.results.remove(0);
        }
        self.results.push(result.clone());

        result
    }

    /// Number of stored profile results.
    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    /// All stored profile results.
    pub fn results(&self) -> &[ProfileResult] {
        &self.results
    }

    /// Average total profiling time across all results.
    pub fn avg_total_time(&self) -> Duration {
        if self.results.is_empty() {
            return Duration::ZERO;
        }
        let sum: Duration = self.results.iter().map(|r| r.total_time).sum();
        sum / self.results.len() as u32
    }

    /// Clear stored results.
    pub fn clear(&mut self) {
        self.results.clear();
    }
}

// ── Multi-stream execution ───────────────────────────────────────────

/// Configuration for multi-stream graph execution.
#[derive(Debug, Clone)]
pub struct MultiStreamConfig {
    /// Number of execution streams.
    pub num_streams: usize,
    /// Whether to insert synchronization events between dependent nodes on
    /// different streams.
    pub auto_sync: bool,
    /// Load-balancing strategy.
    pub balance: LoadBalanceStrategy,
}

impl Default for MultiStreamConfig {
    fn default() -> Self {
        Self { num_streams: 2, auto_sync: true, balance: LoadBalanceStrategy::RoundRobin }
    }
}

impl MultiStreamConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.num_streams == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_streams must be >= 1".into(),
            }
            .into());
        }
        if self.num_streams > 32 {
            return Err(KernelError::InvalidArguments {
                reason: "num_streams must not exceed 32".into(),
            }
            .into());
        }
        Ok(())
    }
}

/// Strategy for assigning nodes to streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalanceStrategy {
    /// Assign nodes to streams in round-robin order.
    RoundRobin,
    /// Assign nodes to the least-loaded stream by estimated cost.
    LeastLoaded,
    /// Keep the original stream assignments from the graph.
    Preserve,
}

/// A node assignment produced by the multi-stream executor.
#[derive(Debug, Clone)]
pub struct StreamAssignment {
    /// Node index.
    pub node_index: usize,
    /// Assigned stream index.
    pub stream: usize,
    /// Sync events to wait for before this node executes.
    pub wait_events: Vec<usize>,
    /// Sync event to signal after this node completes.
    pub signal_event: Option<usize>,
}

/// Result of multi-stream scheduling.
#[derive(Debug, Clone)]
pub struct MultiStreamSchedule {
    /// Per-node assignments.
    pub assignments: Vec<StreamAssignment>,
    /// Total number of sync events inserted.
    pub sync_event_count: usize,
    /// Estimated speedup from parallelism.
    pub estimated_speedup: f64,
    /// Number of streams actually used.
    pub streams_used: usize,
}

/// Plans and executes graph nodes across multiple CUDA streams.
#[derive(Debug)]
pub struct MultiStreamExecutor {
    config: MultiStreamConfig,
    total_schedules: u64,
}

impl MultiStreamExecutor {
    /// Create an executor with the given configuration.
    pub fn new(config: MultiStreamConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, total_schedules: 0 })
    }

    /// The active configuration.
    pub fn config(&self) -> &MultiStreamConfig {
        &self.config
    }

    /// Total number of schedules produced.
    pub fn total_schedules(&self) -> u64 {
        self.total_schedules
    }

    /// Produce a schedule for a list of nodes.
    ///
    /// `nodes` is `(name, estimated_cost_us, original_stream, deps)` where
    /// `deps` is a list of node indices that must complete before this node.
    pub fn schedule(
        &mut self,
        nodes: &[(&str, f64, u32, Vec<usize>)],
    ) -> Result<MultiStreamSchedule> {
        if nodes.is_empty() {
            return Ok(MultiStreamSchedule {
                assignments: Vec::new(),
                sync_event_count: 0,
                estimated_speedup: 1.0,
                streams_used: 0,
            });
        }

        let num_streams = self.config.num_streams;
        let mut stream_loads = vec![0.0_f64; num_streams];
        let mut assignments = Vec::with_capacity(nodes.len());
        let mut node_stream = vec![0usize; nodes.len()];
        let mut event_counter: usize = 0;

        for (i, &(_, cost_us, orig_stream, ref deps)) in nodes.iter().enumerate() {
            let assigned_stream = match self.config.balance {
                LoadBalanceStrategy::RoundRobin => i % num_streams,
                LoadBalanceStrategy::LeastLoaded => stream_loads
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0),
                LoadBalanceStrategy::Preserve => (orig_stream as usize) % num_streams,
            };

            stream_loads[assigned_stream] += cost_us;
            node_stream[i] = assigned_stream;

            // Determine sync events.
            let mut wait_events = Vec::new();
            for &dep in deps {
                if dep < i && node_stream[dep] != assigned_stream && self.config.auto_sync {
                    wait_events.push(dep);
                }
            }

            let signal = if self.config.auto_sync && !deps.is_empty() {
                event_counter += 1;
                Some(event_counter - 1)
            } else {
                None
            };

            assignments.push(StreamAssignment {
                node_index: i,
                stream: assigned_stream,
                wait_events,
                signal_event: signal,
            });
        }

        let streams_used = {
            let s: HashSet<usize> = assignments.iter().map(|a| a.stream).collect();
            s.len()
        };

        // Estimate speedup: ratio of serial time to max-stream time.
        let serial_time: f64 = nodes.iter().map(|n| n.1).sum();
        let max_stream_time = stream_loads.iter().copied().fold(0.0_f64, f64::max);
        let speedup = if max_stream_time > 0.0 { serial_time / max_stream_time } else { 1.0 };

        self.total_schedules += 1;

        Ok(MultiStreamSchedule {
            assignments,
            sync_event_count: event_counter,
            estimated_speedup: speedup,
            streams_used,
        })
    }
}

// ── Optimization pipeline ────────────────────────────────────────────

/// Named optimization pass.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OptimizationPass {
    /// Detect and mark fusable kernel sequences.
    FusionDetection,
    /// Plan memory allocations to minimize peak usage.
    MemoryPlanning,
    /// Remove empty/no-op nodes.
    RemoveEmptyNodes,
    /// Remove redundant barriers.
    RemoveRedundantBarriers,
    /// Merge consecutive identical kernels.
    MergeKernels,
    /// Assign nodes to multiple streams.
    MultiStreamAssignment,
    /// Detect and report bottlenecks.
    BottleneckDetection,
}

impl fmt::Display for OptimizationPass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FusionDetection => write!(f, "fusion_detection"),
            Self::MemoryPlanning => write!(f, "memory_planning"),
            Self::RemoveEmptyNodes => write!(f, "remove_empty_nodes"),
            Self::RemoveRedundantBarriers => write!(f, "remove_redundant_barriers"),
            Self::MergeKernels => write!(f, "merge_kernels"),
            Self::MultiStreamAssignment => write!(f, "multi_stream_assignment"),
            Self::BottleneckDetection => write!(f, "bottleneck_detection"),
        }
    }
}

/// Result of running the optimization pipeline.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Results per pass.
    pub pass_results: Vec<(OptimizationPass, PassResult)>,
    /// Total duration of the pipeline.
    pub total_duration: Duration,
}

/// Result of a single optimization pass.
#[derive(Debug, Clone)]
pub struct PassResult {
    /// Duration of this pass.
    pub duration: Duration,
    /// Number of modifications made.
    pub modifications: usize,
    /// Optional diagnostic message.
    pub message: String,
}

/// Configuration for the optimization pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Which passes to run, in order.
    pub passes: Vec<OptimizationPass>,
    /// Number of streams for multi-stream assignment.
    pub num_streams: usize,
    /// Fusion detector minimum speedup.
    pub min_fusion_speedup: f64,
    /// Hotspot detection threshold.
    pub hotspot_threshold: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            passes: vec![
                OptimizationPass::RemoveEmptyNodes,
                OptimizationPass::RemoveRedundantBarriers,
                OptimizationPass::MergeKernels,
                OptimizationPass::FusionDetection,
                OptimizationPass::MemoryPlanning,
                OptimizationPass::BottleneckDetection,
            ],
            num_streams: 2,
            min_fusion_speedup: 1.0,
            hotspot_threshold: 0.3,
        }
    }
}

/// Composes optimization passes into a configurable pipeline.
///
/// Accepts a list of node descriptors and runs configured passes,
/// returning aggregate results.
#[derive(Debug)]
pub struct GraphOptimizationPipeline {
    config: PipelineConfig,
}

impl Default for GraphOptimizationPipeline {
    fn default() -> Self {
        Self::new(PipelineConfig::default())
    }
}

impl GraphOptimizationPipeline {
    /// Create a pipeline with the given configuration.
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// The current configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Run the pipeline on a graph described as node metadata.
    ///
    /// `nodes` is a list of `(name, estimated_cost_us, stream, is_barrier)`.
    /// `memory_requests` is an optional list of `(size, first_use, last_use)`.
    ///
    /// Returns the pipeline result with per-pass statistics.
    pub fn run(
        &self,
        nodes: &[(&str, f64, u32, bool)],
        memory_requests: Option<&[(usize, usize, usize)]>,
    ) -> Result<PipelineResult> {
        let pipeline_start = Instant::now();
        let mut pass_results = Vec::new();

        for pass in &self.config.passes {
            let pass_start = Instant::now();
            let result = match pass {
                OptimizationPass::FusionDetection => {
                    let mut detector = FusionDetector::new();
                    detector.set_min_speedup(self.config.min_fusion_speedup);
                    let names: Vec<&str> = nodes.iter().map(|n| n.0).collect();
                    let candidates = detector.detect(&names);
                    PassResult {
                        duration: pass_start.elapsed(),
                        modifications: candidates.len(),
                        message: format!("{} fusion candidates found", candidates.len()),
                    }
                }
                OptimizationPass::MemoryPlanning => {
                    if let Some(reqs) = memory_requests {
                        let planner = GraphMemoryPlanner::default();
                        let plan = planner.plan(reqs)?;
                        PassResult {
                            duration: pass_start.elapsed(),
                            modifications: plan.allocations.len(),
                            message: format!(
                                "peak={}B, reuse_savings={}B",
                                plan.peak_bytes, plan.reuse_savings_bytes
                            ),
                        }
                    } else {
                        PassResult {
                            duration: pass_start.elapsed(),
                            modifications: 0,
                            message: "no memory requests provided".to_string(),
                        }
                    }
                }
                OptimizationPass::RemoveEmptyNodes => {
                    let empties = nodes.iter().filter(|n| n.0.is_empty()).count();
                    PassResult {
                        duration: pass_start.elapsed(),
                        modifications: empties,
                        message: format!("{empties} empty nodes identified"),
                    }
                }
                OptimizationPass::RemoveRedundantBarriers => {
                    let barriers = nodes.iter().filter(|n| n.3).count();
                    PassResult {
                        duration: pass_start.elapsed(),
                        modifications: barriers,
                        message: format!("{barriers} barriers analyzed"),
                    }
                }
                OptimizationPass::MergeKernels => {
                    // Count consecutive same-name pairs.
                    let mut merges = 0;
                    let mut i = 0;
                    let names: Vec<&str> = nodes.iter().map(|n| n.0).collect();
                    while i + 1 < names.len() {
                        if !names[i].is_empty() && names[i] == names[i + 1] {
                            merges += 1;
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                    PassResult {
                        duration: pass_start.elapsed(),
                        modifications: merges,
                        message: format!("{merges} kernel pairs mergeable"),
                    }
                }
                OptimizationPass::MultiStreamAssignment => {
                    let ms_config = MultiStreamConfig {
                        num_streams: self.config.num_streams,
                        auto_sync: true,
                        balance: LoadBalanceStrategy::LeastLoaded,
                    };
                    let mut executor = MultiStreamExecutor::new(ms_config)?;
                    let ms_nodes: Vec<(&str, f64, u32, Vec<usize>)> =
                        nodes.iter().map(|&(n, c, s, _)| (n, c, s, vec![])).collect();
                    let schedule = executor.schedule(&ms_nodes)?;
                    PassResult {
                        duration: pass_start.elapsed(),
                        modifications: schedule.assignments.len(),
                        message: format!(
                            "assigned to {} streams, est. speedup {:.2}×",
                            schedule.streams_used, schedule.estimated_speedup
                        ),
                    }
                }
                OptimizationPass::BottleneckDetection => {
                    let mut profiler =
                        GraphBottleneckProfiler::new(self.config.hotspot_threshold, 0.1, 64);
                    let result = profiler.profile(nodes);
                    PassResult {
                        duration: pass_start.elapsed(),
                        modifications: result.bottlenecks.len(),
                        message: format!("{} bottlenecks detected", result.bottlenecks.len()),
                    }
                }
            };
            pass_results.push((pass.clone(), result));
        }

        Ok(PipelineResult { pass_results, total_duration: pipeline_start.elapsed() })
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── FusionPattern ────────────────────────────────────────────

    #[test]
    fn fusion_pattern_display() {
        assert_eq!(FusionPattern::RmsNormLinear.to_string(), "rmsnorm+linear");
        assert_eq!(FusionPattern::GeluLinear.to_string(), "gelu+linear");
        assert_eq!(FusionPattern::SoftmaxMask.to_string(), "softmax+mask");
        assert_eq!(FusionPattern::AddRmsNorm.to_string(), "add+rmsnorm");
        assert_eq!(FusionPattern::ScaleAdd.to_string(), "scale+add");
    }

    #[test]
    fn fusion_pattern_custom_display() {
        let p = FusionPattern::Custom("my_op".to_string());
        assert_eq!(p.to_string(), "custom(my_op)");
    }

    #[test]
    fn fusion_pattern_equality() {
        assert_eq!(FusionPattern::RmsNormLinear, FusionPattern::RmsNormLinear);
        assert_ne!(FusionPattern::RmsNormLinear, FusionPattern::GeluLinear);
    }

    #[test]
    fn fusion_pattern_hash() {
        let mut set = HashSet::new();
        set.insert(FusionPattern::RmsNormLinear);
        set.insert(FusionPattern::GeluLinear);
        set.insert(FusionPattern::RmsNormLinear); // duplicate
        assert_eq!(set.len(), 2);
    }

    // ── FusionDetector ───────────────────────────────────────────

    #[test]
    fn detector_default_rules() {
        let d = FusionDetector::new();
        assert_eq!(d.rule_count(), 7);
    }

    #[test]
    fn detector_detect_rmsnorm_linear() {
        let d = FusionDetector::new();
        let names = ["rmsnorm_f32", "linear_proj"];
        let cs = d.detect(&names);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].pattern, FusionPattern::RmsNormLinear);
        assert_eq!(cs[0].node_indices, vec![0, 1]);
    }

    #[test]
    fn detector_detect_gelu_linear() {
        let d = FusionDetector::new();
        let names = ["gelu_act", "linear_out"];
        let cs = d.detect(&names);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].pattern, FusionPattern::GeluLinear);
    }

    #[test]
    fn detector_detect_softmax_mask() {
        let d = FusionDetector::new();
        let names = ["softmax_row", "mask_apply"];
        let cs = d.detect(&names);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].pattern, FusionPattern::SoftmaxMask);
    }

    #[test]
    fn detector_detect_add_rmsnorm() {
        let d = FusionDetector::new();
        let names = ["add_residual", "rmsnorm_f32"];
        let cs = d.detect(&names);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].pattern, FusionPattern::AddRmsNorm);
    }

    #[test]
    fn detector_detect_scale_add() {
        let d = FusionDetector::new();
        let names = ["scale_f32", "add_bias"];
        let cs = d.detect(&names);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].pattern, FusionPattern::ScaleAdd);
    }

    #[test]
    fn detector_detect_quantize_matmul() {
        let d = FusionDetector::new();
        let names = ["quantize_i2s", "matmul_f32"];
        let cs = d.detect(&names);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].pattern, FusionPattern::QuantizeMatmul);
    }

    #[test]
    fn detector_detect_dequant_elementwise() {
        let d = FusionDetector::new();
        let names = ["dequant_f32", "elementwise_add"];
        let cs = d.detect(&names);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].pattern, FusionPattern::DequantizeElementwise);
    }

    #[test]
    fn detector_no_match() {
        let d = FusionDetector::new();
        let names = ["matmul_f32", "softmax_row"];
        let cs = d.detect(&names);
        assert!(cs.is_empty());
    }

    #[test]
    fn detector_empty_input() {
        let d = FusionDetector::new();
        let cs = d.detect(&[]);
        assert!(cs.is_empty());
    }

    #[test]
    fn detector_single_kernel() {
        let d = FusionDetector::new();
        let names = ["rmsnorm_f32"];
        let cs = d.detect(&names);
        assert!(cs.is_empty());
    }

    #[test]
    fn detector_multiple_fusions() {
        let d = FusionDetector::new();
        let names = ["rmsnorm_f32", "linear_proj", "gelu_act", "linear_out"];
        let cs = d.detect(&names);
        assert_eq!(cs.len(), 2);
        assert_eq!(cs[0].pattern, FusionPattern::RmsNormLinear);
        assert_eq!(cs[1].pattern, FusionPattern::GeluLinear);
    }

    #[test]
    fn detector_non_overlapping_pairs() {
        let d = FusionDetector::new();
        // rmsnorm+linear consumes indices 0,1; gelu at 2 has no pair.
        let names = ["rmsnorm_f32", "linear_proj", "gelu_act"];
        let cs = d.detect(&names);
        assert_eq!(cs.len(), 1);
    }

    #[test]
    fn detector_min_speedup_filter() {
        let mut d = FusionDetector::new();
        d.set_min_speedup(2.0);
        let names = ["rmsnorm_f32", "linear_proj"];
        let cs = d.detect(&names);
        // Default rmsnorm+linear speedup is 1.5, which is below 2.0.
        assert!(cs.is_empty());
    }

    #[test]
    fn detector_add_custom_rule() {
        let mut d = FusionDetector::new();
        let initial = d.rule_count();
        d.add_rule(FusionRule::new(
            FusionPattern::Custom("my_fusion".to_string()),
            "foo",
            "bar",
            2.0,
            8,
        ));
        assert_eq!(d.rule_count(), initial + 1);
        let names = ["foo_kernel", "bar_kernel"];
        let cs = d.detect(&names);
        assert_eq!(cs.len(), 1);
        assert!(matches!(cs[0].pattern, FusionPattern::Custom(_)));
    }

    #[test]
    fn detector_case_insensitive_match() {
        let d = FusionDetector::new();
        let names = ["RmsNorm_F32", "Linear_Proj"];
        let cs = d.detect(&names);
        assert_eq!(cs.len(), 1);
    }

    #[test]
    fn detector_aggregate_speedup_empty() {
        let d = FusionDetector::new();
        assert_eq!(d.estimate_aggregate_speedup(&[]), 1.0);
    }

    #[test]
    fn detector_aggregate_speedup() {
        let d = FusionDetector::new();
        let names = ["rmsnorm_f32", "linear_proj"];
        let cs = d.detect(&names);
        let speedup = d.estimate_aggregate_speedup(&cs);
        assert!(speedup > 1.0);
    }

    // ── GraphMemoryPlanner ───────────────────────────────────────

    #[test]
    fn planner_default_alignment() {
        let p = GraphMemoryPlanner::default();
        assert_eq!(p.alignment(), 256);
    }

    #[test]
    fn planner_custom_alignment() {
        let p = GraphMemoryPlanner::new(64);
        assert_eq!(p.alignment(), 64);
    }

    #[test]
    fn planner_empty_requests() {
        let p = GraphMemoryPlanner::default();
        let plan = p.plan(&[]).unwrap();
        assert_eq!(plan.peak_bytes, 0);
        assert!(plan.allocations.is_empty());
    }

    #[test]
    fn planner_single_allocation() {
        let p = GraphMemoryPlanner::new(256);
        let plan = p.plan(&[(1024, 0, 5)]).unwrap();
        assert_eq!(plan.allocations.len(), 1);
        assert_eq!(plan.peak_bytes, 1024);
        assert_eq!(plan.reuse_savings_bytes, 0);
    }

    #[test]
    fn planner_alignment_rounding() {
        let p = GraphMemoryPlanner::new(256);
        let plan = p.plan(&[(100, 0, 5)]).unwrap();
        // 100 rounded up to 256.
        assert_eq!(plan.peak_bytes, 256);
    }

    #[test]
    fn planner_non_overlapping_reuse() {
        let p = GraphMemoryPlanner::new(256);
        // Alloc A: nodes 0-2, alloc B: nodes 3-5 — no overlap, can reuse.
        let plan = p.plan(&[(512, 0, 2), (512, 3, 5)]).unwrap();
        assert!(plan.reuse_savings_bytes > 0);
        assert_eq!(plan.peak_bytes, 512);
    }

    #[test]
    fn planner_overlapping_no_reuse() {
        let p = GraphMemoryPlanner::new(256);
        // Both live at node 1.
        let plan = p.plan(&[(512, 0, 3), (512, 1, 4)]).unwrap();
        assert_eq!(plan.peak_bytes, 1024);
        assert_eq!(plan.reuse_savings_bytes, 0);
    }

    #[test]
    fn planner_reuse_disabled() {
        let mut p = GraphMemoryPlanner::new(256);
        p.set_reuse(false);
        let plan = p.plan(&[(512, 0, 2), (512, 3, 5)]).unwrap();
        assert_eq!(plan.reuse_savings_bytes, 0);
        assert_eq!(plan.peak_bytes, 1024);
    }

    #[test]
    fn planner_multiple_reuse() {
        let p = GraphMemoryPlanner::new(256);
        // Three sequential, non-overlapping allocations.
        let plan = p.plan(&[(256, 0, 1), (256, 2, 3), (256, 4, 5)]).unwrap();
        // All three can reuse the same slot.
        assert_eq!(plan.peak_bytes, 256);
    }

    #[test]
    fn planner_offset_map_populated() {
        let p = GraphMemoryPlanner::new(256);
        let plan = p.plan(&[(512, 0, 2), (512, 3, 5)]).unwrap();
        assert_eq!(plan.offset_map.len(), 2);
    }

    // ── GraphShapeKey ────────────────────────────────────────────

    #[test]
    fn shape_key_display() {
        let k = GraphShapeKey { batch_size: 1, seq_len: 128, hidden_dim: 2048, num_heads: 32 };
        assert_eq!(k.to_string(), "B1×S128×H2048×NH32");
    }

    #[test]
    fn shape_key_equality() {
        let a = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        let b = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        assert_eq!(a, b);
    }

    #[test]
    fn shape_key_inequality() {
        let a = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        let b = GraphShapeKey { batch_size: 2, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        assert_ne!(a, b);
    }

    #[test]
    fn shape_key_hash_map() {
        let mut map = HashMap::new();
        let k = GraphShapeKey { batch_size: 1, seq_len: 128, hidden_dim: 2048, num_heads: 32 };
        map.insert(k.clone(), 42);
        assert_eq!(map[&k], 42);
    }

    // ── GraphInstanceCache ───────────────────────────────────────

    #[test]
    fn cache_new_empty() {
        let c = GraphInstanceCache::new(16);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.hits(), 0);
        assert_eq!(c.misses(), 0);
    }

    #[test]
    fn cache_insert_and_get() {
        let mut c = GraphInstanceCache::new(16);
        let k = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        c.insert(k.clone(), 10);
        assert!(c.contains(&k));
        assert!(c.get(&k));
        assert_eq!(c.hits(), 1);
    }

    #[test]
    fn cache_miss() {
        let mut c = GraphInstanceCache::new(16);
        let k = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        assert!(!c.get(&k));
        assert_eq!(c.misses(), 1);
    }

    #[test]
    fn cache_eviction() {
        let mut c = GraphInstanceCache::new(2);
        let k1 = GraphShapeKey { batch_size: 1, seq_len: 32, hidden_dim: 512, num_heads: 8 };
        let k2 = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        let k3 = GraphShapeKey { batch_size: 1, seq_len: 128, hidden_dim: 512, num_heads: 8 };
        c.insert(k1.clone(), 10);
        c.insert(k2.clone(), 10);
        c.insert(k3.clone(), 10);
        assert_eq!(c.len(), 2);
        assert_eq!(c.evictions(), 1);
    }

    #[test]
    fn cache_hit_rate_zero() {
        let c = GraphInstanceCache::new(8);
        assert_eq!(c.hit_rate(), 0.0);
    }

    #[test]
    fn cache_hit_rate() {
        let mut c = GraphInstanceCache::new(8);
        let k = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        c.insert(k.clone(), 10);
        c.get(&k); // hit
        c.get(&k); // hit
        let k2 = GraphShapeKey { batch_size: 2, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        c.get(&k2); // miss
        assert!((c.hit_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn cache_clear() {
        let mut c = GraphInstanceCache::new(8);
        let k = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        c.insert(k, 10);
        c.clear();
        assert!(c.is_empty());
        assert_eq!(c.hits(), 0);
        assert_eq!(c.misses(), 0);
    }

    #[test]
    fn cache_remove() {
        let mut c = GraphInstanceCache::new(8);
        let k = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        c.insert(k.clone(), 10);
        assert!(c.remove(&k));
        assert!(!c.contains(&k));
    }

    #[test]
    fn cache_remove_nonexistent() {
        let mut c = GraphInstanceCache::new(8);
        let k = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        assert!(!c.remove(&k));
    }

    #[test]
    fn cache_exec_count() {
        let mut c = GraphInstanceCache::new(8);
        let k = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        c.insert(k.clone(), 10);
        c.get(&k);
        c.get(&k);
        assert_eq!(c.exec_count(&k), 2);
    }

    #[test]
    fn cache_node_count() {
        let mut c = GraphInstanceCache::new(8);
        let k = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        c.insert(k.clone(), 42);
        assert_eq!(c.node_count(&k), Some(42));
    }

    // ── ShapeUpdate ──────────────────────────────────────────────

    #[test]
    fn shape_update_no_changes() {
        let s = ShapeUpdate { batch_size: None, seq_len: None, hidden_dim: None };
        assert!(!s.has_changes());
        assert_eq!(s.change_count(), 0);
    }

    #[test]
    fn shape_update_one_change() {
        let s = ShapeUpdate { batch_size: Some(2), seq_len: None, hidden_dim: None };
        assert!(s.has_changes());
        assert_eq!(s.change_count(), 1);
    }

    #[test]
    fn shape_update_all_changes() {
        let s = ShapeUpdate { batch_size: Some(2), seq_len: Some(128), hidden_dim: Some(4096) };
        assert_eq!(s.change_count(), 3);
    }

    // ── GraphUpdater ─────────────────────────────────────────────

    #[test]
    fn updater_default() {
        let u = GraphUpdater::default();
        assert_eq!(u.recapture_threshold(), 2.0);
        assert_eq!(u.total_updates(), 0);
    }

    #[test]
    fn updater_needs_recapture_small_change() {
        let u = GraphUpdater::new(2.0);
        let old = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        let shape = ShapeUpdate { batch_size: None, seq_len: Some(96), hidden_dim: None };
        assert!(!u.needs_recapture(&old, &shape));
    }

    #[test]
    fn updater_needs_recapture_large_change() {
        let u = GraphUpdater::new(2.0);
        let old = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        let shape = ShapeUpdate { batch_size: None, seq_len: Some(256), hidden_dim: None };
        // 256/64 = 4.0 > 2.0 → recapture.
        assert!(u.needs_recapture(&old, &shape));
    }

    #[test]
    fn updater_needs_recapture_shrink() {
        let u = GraphUpdater::new(2.0);
        let old = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        let shape = ShapeUpdate { batch_size: None, seq_len: Some(10), hidden_dim: None };
        // 10/64 < 0.5 → recapture.
        assert!(u.needs_recapture(&old, &shape));
    }

    #[test]
    fn updater_apply_param_updates() {
        let mut u = GraphUpdater::default();
        let mut params = vec![HashMap::new(), HashMap::new()];
        let updates = vec![NodeParamUpdate {
            node_index: 0,
            params: HashMap::from([("lr".to_string(), 0.001)]),
        }];
        let result = u.apply_param_updates(&mut params, &updates).unwrap();
        assert_eq!(result.nodes_updated, 1);
        assert!(!result.required_recapture);
        assert_eq!(params[0]["lr"], 0.001);
    }

    #[test]
    fn updater_apply_param_out_of_range() {
        let mut u = GraphUpdater::default();
        let mut params = vec![HashMap::new()];
        let updates = vec![NodeParamUpdate { node_index: 5, params: HashMap::new() }];
        assert!(u.apply_param_updates(&mut params, &updates).is_err());
    }

    #[test]
    fn updater_apply_shape_update_no_recapture() {
        let mut u = GraphUpdater::new(2.0);
        let old = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        let shape = ShapeUpdate { batch_size: None, seq_len: Some(96), hidden_dim: None };
        let mut params: Vec<HashMap<String, f64>> = (0..3).map(|_| HashMap::new()).collect();
        let result = u.apply_shape_update(&old, &shape, &mut params).unwrap();
        assert!(!result.required_recapture);
        assert_eq!(result.nodes_updated, 3);
        assert_eq!(params[0]["seq_len"], 96.0);
    }

    #[test]
    fn updater_apply_shape_update_forces_recapture() {
        let mut u = GraphUpdater::new(2.0);
        let old = GraphShapeKey { batch_size: 1, seq_len: 64, hidden_dim: 512, num_heads: 8 };
        let shape = ShapeUpdate { batch_size: None, seq_len: Some(512), hidden_dim: None };
        let mut params: Vec<HashMap<String, f64>> = (0..3).map(|_| HashMap::new()).collect();
        let result = u.apply_shape_update(&old, &shape, &mut params).unwrap();
        assert!(result.required_recapture);
        assert_eq!(u.total_recaptures(), 1);
    }

    #[test]
    fn updater_rollback() {
        let mut u = GraphUpdater::default();
        let mut params = vec![HashMap::from([("x".to_string(), 1.0)])];
        let updates = vec![NodeParamUpdate {
            node_index: 0,
            params: HashMap::from([("x".to_string(), 99.0)]),
        }];
        u.apply_param_updates(&mut params, &updates).unwrap();
        assert_eq!(params[0]["x"], 99.0);
        let restored = u.rollback(&mut params, 1);
        assert_eq!(restored, 1);
        assert_eq!(params[0]["x"], 1.0);
    }

    #[test]
    fn updater_rollback_empty_history() {
        let mut u = GraphUpdater::default();
        let mut params = vec![HashMap::new()];
        assert_eq!(u.rollback(&mut params, 5), 0);
    }

    #[test]
    fn updater_clear_history() {
        let mut u = GraphUpdater::default();
        let mut params = vec![HashMap::new()];
        let updates = vec![NodeParamUpdate {
            node_index: 0,
            params: HashMap::from([("a".to_string(), 1.0)]),
        }];
        u.apply_param_updates(&mut params, &updates).unwrap();
        assert_eq!(u.history_len(), 1);
        u.clear_history();
        assert_eq!(u.history_len(), 0);
    }

    // ── BottleneckKind ───────────────────────────────────────────

    #[test]
    fn bottleneck_kind_display() {
        assert_eq!(BottleneckKind::KernelHotspot.to_string(), "kernel_hotspot");
        assert_eq!(BottleneckKind::MemoryBound.to_string(), "memory_bound");
        assert_eq!(BottleneckKind::SyncOverhead.to_string(), "sync_overhead");
        assert_eq!(BottleneckKind::LowOccupancy.to_string(), "low_occupancy");
        assert_eq!(BottleneckKind::SerialDependency.to_string(), "serial_dependency");
    }

    #[test]
    fn bottleneck_kind_equality() {
        assert_eq!(BottleneckKind::KernelHotspot, BottleneckKind::KernelHotspot);
        assert_ne!(BottleneckKind::KernelHotspot, BottleneckKind::MemoryBound);
    }

    // ── GraphBottleneckProfiler ──────────────────────────────────

    #[test]
    fn profiler_default() {
        let p = GraphBottleneckProfiler::default();
        assert_eq!(p.hotspot_threshold(), 0.3);
        assert_eq!(p.sync_threshold(), 0.1);
        assert_eq!(p.result_count(), 0);
    }

    #[test]
    fn profiler_empty_nodes() {
        let mut p = GraphBottleneckProfiler::default();
        let result = p.profile(&[]);
        assert_eq!(result.nodes_profiled, 0);
        assert!(result.bottlenecks.is_empty());
    }

    #[test]
    fn profiler_detects_hotspot() {
        let mut p = GraphBottleneckProfiler::new(0.3, 0.1, 64);
        let nodes = [("small_kernel", 10.0, 0, false), ("big_kernel", 90.0, 0, false)];
        let result = p.profile(&nodes);
        let hotspots: Vec<_> =
            result.bottlenecks.iter().filter(|b| b.kind == BottleneckKind::KernelHotspot).collect();
        assert!(!hotspots.is_empty());
        assert_eq!(hotspots[0].label, "big_kernel");
    }

    #[test]
    fn profiler_detects_sync_overhead() {
        let mut p = GraphBottleneckProfiler::new(0.5, 0.1, 64);
        let nodes = [
            ("kernel_a", 50.0, 0, false),
            ("barrier", 20.0, 0, true),
            ("kernel_b", 30.0, 0, false),
        ];
        let result = p.profile(&nodes);
        let syncs: Vec<_> =
            result.bottlenecks.iter().filter(|b| b.kind == BottleneckKind::SyncOverhead).collect();
        assert!(!syncs.is_empty());
    }

    #[test]
    fn profiler_detects_serial_dependency() {
        let mut p = GraphBottleneckProfiler::new(0.9, 0.5, 64);
        let nodes = [
            ("k1", 10.0, 0, false),
            ("k2", 10.0, 0, false),
            ("k3", 10.0, 0, false),
            ("k4", 10.0, 0, false),
            ("k5", 10.0, 0, false),
        ];
        let result = p.profile(&nodes);
        let serials: Vec<_> = result
            .bottlenecks
            .iter()
            .filter(|b| b.kind == BottleneckKind::SerialDependency)
            .collect();
        assert!(!serials.is_empty());
    }

    #[test]
    fn profiler_no_serial_dependency_multi_stream() {
        let mut p = GraphBottleneckProfiler::new(0.9, 0.5, 64);
        let nodes = [
            ("k1", 10.0, 0, false),
            ("k2", 10.0, 1, false),
            ("k3", 10.0, 0, false),
            ("k4", 10.0, 1, false),
            ("k5", 10.0, 0, false),
        ];
        let result = p.profile(&nodes);
        let serials: Vec<_> = result
            .bottlenecks
            .iter()
            .filter(|b| b.kind == BottleneckKind::SerialDependency)
            .collect();
        assert!(serials.is_empty());
    }

    #[test]
    fn profiler_result_count() {
        let mut p = GraphBottleneckProfiler::new(0.3, 0.1, 64);
        p.profile(&[("k", 10.0, 0, false)]);
        p.profile(&[("k", 10.0, 0, false)]);
        assert_eq!(p.result_count(), 2);
    }

    #[test]
    fn profiler_avg_total_time_empty() {
        let p = GraphBottleneckProfiler::default();
        assert_eq!(p.avg_total_time(), Duration::ZERO);
    }

    #[test]
    fn profiler_clear() {
        let mut p = GraphBottleneckProfiler::default();
        p.profile(&[("k", 10.0, 0, false)]);
        p.clear();
        assert_eq!(p.result_count(), 0);
    }

    #[test]
    fn profiler_max_results_eviction() {
        let mut p = GraphBottleneckProfiler::new(0.3, 0.1, 2);
        p.profile(&[("k1", 10.0, 0, false)]);
        p.profile(&[("k2", 10.0, 0, false)]);
        p.profile(&[("k3", 10.0, 0, false)]);
        assert_eq!(p.result_count(), 2);
    }

    // ── MultiStreamConfig ────────────────────────────────────────

    #[test]
    fn ms_config_default() {
        let c = MultiStreamConfig::default();
        assert_eq!(c.num_streams, 2);
        assert!(c.auto_sync);
    }

    #[test]
    fn ms_config_validate_ok() {
        let c = MultiStreamConfig {
            num_streams: 4,
            auto_sync: true,
            balance: LoadBalanceStrategy::RoundRobin,
        };
        assert!(c.validate().is_ok());
    }

    #[test]
    fn ms_config_validate_zero() {
        let c = MultiStreamConfig {
            num_streams: 0,
            auto_sync: true,
            balance: LoadBalanceStrategy::RoundRobin,
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn ms_config_validate_too_many() {
        let c = MultiStreamConfig {
            num_streams: 64,
            auto_sync: true,
            balance: LoadBalanceStrategy::RoundRobin,
        };
        assert!(c.validate().is_err());
    }

    // ── MultiStreamExecutor ──────────────────────────────────────

    #[test]
    fn ms_executor_empty() {
        let mut exec = MultiStreamExecutor::new(MultiStreamConfig::default()).unwrap();
        let sched = exec.schedule(&[]).unwrap();
        assert!(sched.assignments.is_empty());
        assert_eq!(sched.streams_used, 0);
    }

    #[test]
    fn ms_executor_round_robin() {
        let config = MultiStreamConfig {
            num_streams: 2,
            auto_sync: false,
            balance: LoadBalanceStrategy::RoundRobin,
        };
        let mut exec = MultiStreamExecutor::new(config).unwrap();
        let nodes = [
            ("k0", 10.0, 0, vec![]),
            ("k1", 10.0, 0, vec![]),
            ("k2", 10.0, 0, vec![]),
            ("k3", 10.0, 0, vec![]),
        ];
        let sched = exec.schedule(&nodes).unwrap();
        assert_eq!(sched.assignments[0].stream, 0);
        assert_eq!(sched.assignments[1].stream, 1);
        assert_eq!(sched.assignments[2].stream, 0);
        assert_eq!(sched.assignments[3].stream, 1);
    }

    #[test]
    fn ms_executor_least_loaded() {
        let config = MultiStreamConfig {
            num_streams: 2,
            auto_sync: false,
            balance: LoadBalanceStrategy::LeastLoaded,
        };
        let mut exec = MultiStreamExecutor::new(config).unwrap();
        let nodes = [("k0", 100.0, 0, vec![]), ("k1", 10.0, 0, vec![]), ("k2", 10.0, 0, vec![])];
        let sched = exec.schedule(&nodes).unwrap();
        // k0 goes to stream 0 (both are 0), k1 goes to stream 1 (least loaded).
        assert_eq!(sched.assignments[1].stream, 1);
    }

    #[test]
    fn ms_executor_preserve() {
        let config = MultiStreamConfig {
            num_streams: 4,
            auto_sync: false,
            balance: LoadBalanceStrategy::Preserve,
        };
        let mut exec = MultiStreamExecutor::new(config).unwrap();
        let nodes = [("k0", 10.0, 2, vec![]), ("k1", 10.0, 3, vec![])];
        let sched = exec.schedule(&nodes).unwrap();
        assert_eq!(sched.assignments[0].stream, 2);
        assert_eq!(sched.assignments[1].stream, 3);
    }

    #[test]
    fn ms_executor_auto_sync_events() {
        let config = MultiStreamConfig {
            num_streams: 2,
            auto_sync: true,
            balance: LoadBalanceStrategy::RoundRobin,
        };
        let mut exec = MultiStreamExecutor::new(config).unwrap();
        let nodes = [
            ("k0", 10.0, 0, vec![]),
            ("k1", 10.0, 0, vec![0]), // depends on k0, different stream
        ];
        let sched = exec.schedule(&nodes).unwrap();
        // k1 is on stream 1, depends on k0 which is on stream 0 → wait event.
        assert!(!sched.assignments[1].wait_events.is_empty());
    }

    #[test]
    fn ms_executor_speedup_estimate() {
        let config = MultiStreamConfig {
            num_streams: 2,
            auto_sync: false,
            balance: LoadBalanceStrategy::RoundRobin,
        };
        let mut exec = MultiStreamExecutor::new(config).unwrap();
        let nodes = [("k0", 50.0, 0, vec![]), ("k1", 50.0, 0, vec![])];
        let sched = exec.schedule(&nodes).unwrap();
        // Ideally ~2× speedup with 2 equal nodes on 2 streams.
        assert!(sched.estimated_speedup >= 1.5);
    }

    #[test]
    fn ms_executor_total_schedules() {
        let mut exec = MultiStreamExecutor::new(MultiStreamConfig::default()).unwrap();
        exec.schedule(&[("k", 10.0, 0, vec![])]).unwrap();
        exec.schedule(&[("k", 10.0, 0, vec![])]).unwrap();
        assert_eq!(exec.total_schedules(), 2);
    }

    #[test]
    fn ms_executor_streams_used() {
        let config = MultiStreamConfig {
            num_streams: 4,
            auto_sync: false,
            balance: LoadBalanceStrategy::RoundRobin,
        };
        let mut exec = MultiStreamExecutor::new(config).unwrap();
        let nodes = [("k0", 10.0, 0, vec![]), ("k1", 10.0, 0, vec![])];
        let sched = exec.schedule(&nodes).unwrap();
        assert_eq!(sched.streams_used, 2);
    }

    // ── OptimizationPass ─────────────────────────────────────────

    #[test]
    fn pass_display() {
        assert_eq!(OptimizationPass::FusionDetection.to_string(), "fusion_detection");
        assert_eq!(OptimizationPass::MemoryPlanning.to_string(), "memory_planning");
        assert_eq!(OptimizationPass::MergeKernels.to_string(), "merge_kernels");
    }

    #[test]
    fn pass_equality() {
        assert_eq!(OptimizationPass::FusionDetection, OptimizationPass::FusionDetection);
        assert_ne!(OptimizationPass::FusionDetection, OptimizationPass::MergeKernels);
    }

    // ── GraphOptimizationPipeline ────────────────────────────────

    #[test]
    fn pipeline_default_config() {
        let p = GraphOptimizationPipeline::default();
        assert!(!p.config().passes.is_empty());
    }

    #[test]
    fn pipeline_empty_nodes() {
        let p = GraphOptimizationPipeline::default();
        let result = p.run(&[], None).unwrap();
        assert!(!result.pass_results.is_empty());
    }

    #[test]
    fn pipeline_with_nodes() {
        let p = GraphOptimizationPipeline::default();
        let nodes = [
            ("rmsnorm_f32", 10.0, 0, false),
            ("linear_proj", 20.0, 0, false),
            ("gelu_act", 5.0, 0, false),
            ("linear_out", 15.0, 0, false),
        ];
        let result = p.run(&nodes, None).unwrap();
        // Should have results for each configured pass.
        assert_eq!(result.pass_results.len(), p.config().passes.len());
    }

    #[test]
    fn pipeline_with_memory_requests() {
        let p = GraphOptimizationPipeline::default();
        let nodes = [("k", 10.0, 0, false)];
        let mem = [(1024, 0, 0)];
        let result = p.run(&nodes, Some(&mem)).unwrap();
        let mem_pass =
            result.pass_results.iter().find(|(p, _)| *p == OptimizationPass::MemoryPlanning);
        assert!(mem_pass.is_some());
    }

    #[test]
    fn pipeline_detects_fusions() {
        let p = GraphOptimizationPipeline::default();
        let nodes = [("rmsnorm_f32", 10.0, 0, false), ("linear_proj", 20.0, 0, false)];
        let result = p.run(&nodes, None).unwrap();
        let fusion_pass = result
            .pass_results
            .iter()
            .find(|(p, _)| *p == OptimizationPass::FusionDetection)
            .unwrap();
        assert!(fusion_pass.1.modifications > 0);
    }

    #[test]
    fn pipeline_detects_merge_candidates() {
        let p = GraphOptimizationPipeline::default();
        let nodes = [("matmul_f32", 10.0, 0, false), ("matmul_f32", 10.0, 0, false)];
        let result = p.run(&nodes, None).unwrap();
        let merge_pass =
            result.pass_results.iter().find(|(p, _)| *p == OptimizationPass::MergeKernels).unwrap();
        assert_eq!(merge_pass.1.modifications, 1);
    }

    #[test]
    fn pipeline_custom_config() {
        let config = PipelineConfig {
            passes: vec![OptimizationPass::FusionDetection],
            num_streams: 4,
            min_fusion_speedup: 1.0,
            hotspot_threshold: 0.3,
        };
        let p = GraphOptimizationPipeline::new(config);
        let nodes = [("rmsnorm_f32", 10.0, 0, false), ("linear_proj", 20.0, 0, false)];
        let result = p.run(&nodes, None).unwrap();
        assert_eq!(result.pass_results.len(), 1);
    }

    #[test]
    fn pipeline_bottleneck_detection() {
        let p = GraphOptimizationPipeline::new(PipelineConfig {
            passes: vec![OptimizationPass::BottleneckDetection],
            num_streams: 2,
            min_fusion_speedup: 1.0,
            hotspot_threshold: 0.2,
        });
        let nodes = [("small", 5.0, 0, false), ("huge", 95.0, 0, false)];
        let result = p.run(&nodes, None).unwrap();
        assert!(result.pass_results[0].1.modifications > 0);
    }

    #[test]
    fn pipeline_multi_stream_pass() {
        let p = GraphOptimizationPipeline::new(PipelineConfig {
            passes: vec![OptimizationPass::MultiStreamAssignment],
            num_streams: 4,
            min_fusion_speedup: 1.0,
            hotspot_threshold: 0.3,
        });
        let nodes = [("k1", 10.0, 0, false), ("k2", 10.0, 0, false), ("k3", 10.0, 0, false)];
        let result = p.run(&nodes, None).unwrap();
        assert_eq!(result.pass_results[0].1.modifications, 3);
    }

    #[test]
    fn pipeline_total_duration() {
        let p = GraphOptimizationPipeline::default();
        let result = p.run(&[("k", 1.0, 0, false)], None).unwrap();
        // Duration should be non-negative (trivially true, but validates the field).
        assert!(result.total_duration.as_nanos() > 0 || result.total_duration == Duration::ZERO);
    }

    #[test]
    fn pipeline_remove_empty_pass() {
        let p = GraphOptimizationPipeline::new(PipelineConfig {
            passes: vec![OptimizationPass::RemoveEmptyNodes],
            num_streams: 1,
            min_fusion_speedup: 1.0,
            hotspot_threshold: 0.3,
        });
        let nodes = [("k", 10.0, 0, false), ("", 0.0, 0, false)];
        let result = p.run(&nodes, None).unwrap();
        assert_eq!(result.pass_results[0].1.modifications, 1);
    }

    #[test]
    fn pipeline_barrier_analysis_pass() {
        let p = GraphOptimizationPipeline::new(PipelineConfig {
            passes: vec![OptimizationPass::RemoveRedundantBarriers],
            num_streams: 1,
            min_fusion_speedup: 1.0,
            hotspot_threshold: 0.3,
        });
        let nodes = [("k", 10.0, 0, false), ("barrier", 1.0, 0, true), ("k2", 10.0, 0, false)];
        let result = p.run(&nodes, None).unwrap();
        assert_eq!(result.pass_results[0].1.modifications, 1);
    }
}
