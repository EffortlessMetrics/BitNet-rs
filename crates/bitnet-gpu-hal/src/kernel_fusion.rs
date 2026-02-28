//! Module stub - implementation pending merge from feature branch
//! Kernel fusion optimizer for reducing memory traffic and kernel launch overhead.
//!
//! Scans a computation graph for fusable operation patterns (e.g. MatMul+Bias,
//! Norm+Activation), estimates the benefit of fusing them, and produces
//! [`FusedKernel`] instances that are cached for reuse.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ── Error type ───────────────────────────────────────────────────────────────

/// Errors produced by the kernel fusion subsystem.
#[derive(Debug, thiserror::Error)]
pub enum FusionError {
    /// Two operations have incompatible shapes and cannot be fused.
    #[error("shape mismatch: op {src} ({src_shape:?}) → op {dst} ({dst_shape:?})")]
    ShapeMismatch {
        src: usize,
        dst: usize,
        src_shape: Vec<usize>,
        dst_shape: Vec<usize>,
    },
    /// Fusing would exceed the configured memory budget.
    #[error("memory budget exceeded: need {required} bytes, limit {limit} bytes")]
    MemoryBudgetExceeded { required: usize, limit: usize },
    /// The candidate pair refers to an operation index that does not exist.
    #[error("invalid op index {0}")]
    InvalidOpIndex(usize),
    /// Fusion is disabled by configuration.
    #[error("auto-fusion is disabled")]
    AutoFuseDisabled,
    /// Generic fusion failure.
    #[error("fusion failed: {0}")]
    Other(String),
}

/// Convenience alias.
pub type FusionResult<T> = Result<T, FusionError>;

// ── Primitive operation model ────────────────────────────────────────────────

/// Kind of primitive compute operation in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    MatMul,
    BiasAdd,
    LayerNorm,
    RmsNorm,
    Activation,
    Softmax,
    QkvProjection,
    GatedFfn,
    Reshape,
    Transpose,
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// A single operation node in the computation graph.
#[derive(Debug, Clone)]
pub struct OpNode {
    /// Unique index within the graph.
    pub id: usize,
    /// What this operation does.
    pub kind: OpKind,
    /// Output tensor shape.
    pub shape: Vec<usize>,
    /// Device identifier (e.g. `"cuda:0"`).
    pub device: String,
    /// Estimated execution time in microseconds.
    pub estimated_us: f64,
    /// Indices of predecessor operations whose outputs this op consumes.
    pub inputs: Vec<usize>,
}

/// A lightweight computation graph – an ordered list of operations.
#[derive(Debug, Clone)]
pub struct ComputeGraph {
    pub ops: Vec<OpNode>,
}

impl ComputeGraph {
    #[must_use]
    pub const fn new(ops: Vec<OpNode>) -> Self {
        Self { ops }
    }

    /// Number of operations in the graph.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns `true` when the graph has no operations.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Look up an operation by index.
    #[must_use]
    pub fn get(&self, idx: usize) -> Option<&OpNode> {
        self.ops.get(idx)
    }
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Governs how aggressively the optimizer fuses operations.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Maximum number of primitive ops that may be merged into one fused kernel.
    pub max_fused_ops: usize,
    /// Hard memory ceiling (bytes) for any single fused kernel's scratch space.
    pub memory_limit: usize,
    /// When `false`, `KernelFusionEngine::optimize` returns the graph unchanged.
    pub auto_fuse_enabled: bool,
    /// Additional user-supplied rules evaluated during candidate screening.
    pub fusion_rules: Vec<FusionRule>,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            max_fused_ops: 4,
            memory_limit: 256 * 1024 * 1024, // 256 MiB
            auto_fuse_enabled: true,
            fusion_rules: Vec::new(),
        }
    }
}

// ── Fusion rules ─────────────────────────────────────────────────────────────

/// Type alias for the fusion predicate closure.
type FusionPredicate = Arc<dyn Fn(&OpNode, &OpNode) -> bool + Send + Sync>;

/// A predicate that decides whether two operations may be fused.
#[derive(Clone)]
pub struct FusionRule {
    /// Human-readable name shown in diagnostics.
    pub name: String,
    /// The actual predicate.  Receives the two candidate ops and returns
    /// `true` when fusion is permitted.
    predicate: FusionPredicate,
}

impl FusionRule {
    /// Create a new rule from a closure.
    pub fn new(
        name: impl Into<String>,
        predicate: impl Fn(&OpNode, &OpNode) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            predicate: Arc::new(predicate),
        }
    }

    /// Evaluate the rule for two ops.
    #[must_use]
    pub fn allows(&self, a: &OpNode, b: &OpNode) -> bool {
        (self.predicate)(a, b)
    }
}

impl fmt::Debug for FusionRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FusionRule")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

// ── Fusion candidate ─────────────────────────────────────────────────────────

/// A pair of operation indices identified as a fusion opportunity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionCandidate {
    /// First (earlier) operation index.
    pub src: usize,
    /// Second (later) operation index.
    pub dst: usize,
    /// Matched pattern.
    pub pattern: FusionPattern,
}

// ── Patterns ─────────────────────────────────────────────────────────────────

/// Recognised fusable patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// Matrix multiply immediately followed by a bias addition.
    MatMulBias,
    /// Layer/RMS normalisation followed by an element-wise activation.
    NormActivation,
    /// Full attention block (Q·K^T / √d → softmax → ·V) in one kernel.
    AttentionFused,
    /// Three parallel projections for Q, K, V from the same input.
    QKVProjection,
    /// Gated feed-forward: gate(x) ⊙ up(x) fused with the down projection.
    GatedFFN,
}

impl fmt::Display for FusionPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatMulBias => write!(f, "MatMulBias"),
            Self::NormActivation => write!(f, "NormActivation"),
            Self::AttentionFused => write!(f, "AttentionFused"),
            Self::QKVProjection => write!(f, "QKVProjection"),
            Self::GatedFFN => write!(f, "GatedFFN"),
        }
    }
}

// ── Pattern matcher ──────────────────────────────────────────────────────────

/// Scans a [`ComputeGraph`] for adjacent operation pairs that match a known
/// [`FusionPattern`].
#[derive(Debug)]
pub struct PatternMatcher {
    /// Enabled patterns (empty = all enabled).
    enabled: Vec<FusionPattern>,
}

impl PatternMatcher {
    /// Create a matcher that recognises every pattern.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            enabled: Vec::new(),
        }
    }

    /// Restrict the matcher to an explicit set of patterns.
    #[must_use]
    pub const fn with_patterns(patterns: Vec<FusionPattern>) -> Self {
        Self { enabled: patterns }
    }

    fn is_enabled(&self, pat: FusionPattern) -> bool {
        self.enabled.is_empty() || self.enabled.contains(&pat)
    }

    /// Return all fusion candidates found in `graph`.
    #[must_use]
    pub fn scan(&self, graph: &ComputeGraph) -> Vec<FusionCandidate> {
        let mut candidates = Vec::new();
        for (i, a) in graph.ops.iter().enumerate() {
            for (j, b) in graph.ops.iter().enumerate().skip(i + 1) {
                if a.device != b.device {
                    continue;
                }
                if let Some(pat) = self.match_pair(a, b) {
                    candidates.push(FusionCandidate {
                        src: i,
                        dst: j,
                        pattern: pat,
                    });
                }
            }
        }
        candidates
    }

    fn match_pair(&self, a: &OpNode, b: &OpNode) -> Option<FusionPattern> {
        // MatMul + BiasAdd
        if a.kind == OpKind::MatMul && b.kind == OpKind::BiasAdd && self.is_enabled(FusionPattern::MatMulBias) {
            return Some(FusionPattern::MatMulBias);
        }
        // LayerNorm/RmsNorm + Activation
        if (a.kind == OpKind::LayerNorm || a.kind == OpKind::RmsNorm)
            && b.kind == OpKind::Activation
            && self.is_enabled(FusionPattern::NormActivation)
        {
            return Some(FusionPattern::NormActivation);
        }
        // QKV projection (two consecutive QkvProjection ops → single fused)
        if a.kind == OpKind::QkvProjection
            && b.kind == OpKind::QkvProjection
            && self.is_enabled(FusionPattern::QKVProjection)
        {
            return Some(FusionPattern::QKVProjection);
        }
        // GatedFfn
        if a.kind == OpKind::GatedFfn
            && b.kind == OpKind::MatMul
            && self.is_enabled(FusionPattern::GatedFFN)
        {
            return Some(FusionPattern::GatedFFN);
        }
        // Attention: Softmax following MatMul (simplified detection)
        if a.kind == OpKind::MatMul
            && b.kind == OpKind::Softmax
            && self.is_enabled(FusionPattern::AttentionFused)
        {
            return Some(FusionPattern::AttentionFused);
        }
        None
    }
}

impl Default for PatternMatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ── Fusion benefit estimation ────────────────────────────────────────────────

/// Quantitative estimate of the benefit of applying a given fusion.
#[derive(Debug, Clone)]
pub struct FusionBenefit {
    /// Estimated speedup ratio (e.g. 1.3 = 30 % faster).
    pub speedup: f64,
    /// Bytes of global-memory traffic saved per invocation.
    pub memory_traffic_saved: usize,
    /// Number of kernel launches eliminated.
    pub kernel_launches_saved: usize,
    /// Pattern that produced this estimate.
    pub pattern: FusionPattern,
}

impl FusionBenefit {
    /// Heuristic benefit estimator.
    #[must_use]
    pub fn estimate(candidate: &FusionCandidate, graph: &ComputeGraph) -> Self {
        let src = &graph.ops[candidate.src];
        let dst = &graph.ops[candidate.dst];

        let element_bytes: usize = 2; // assume fp16
        let traffic_saved = src.shape.iter().product::<usize>() * element_bytes;

        let (speedup, launches_saved) = match candidate.pattern {
            FusionPattern::MatMulBias => (1.15, 1),
            FusionPattern::NormActivation => (1.25, 1),
            FusionPattern::AttentionFused => (1.50, 2),
            FusionPattern::QKVProjection => (1.35, 2),
            FusionPattern::GatedFFN => (1.20, 1),
        };

        // Adjust speedup by relative execution weight.
        let total_us = src.estimated_us + dst.estimated_us;
        let adjusted = if total_us > 0.0 {
            (speedup - 1.0_f64).mul_add(total_us / (total_us + 10.0), 1.0)
        } else {
            speedup
        };

        Self {
            speedup: adjusted,
            memory_traffic_saved: traffic_saved,
            kernel_launches_saved: launches_saved,
            pattern: candidate.pattern,
        }
    }

    /// Returns `true` when the estimated benefit exceeds a minimum threshold.
    #[must_use]
    pub fn is_worthwhile(&self, min_speedup: f64) -> bool {
        self.speedup >= min_speedup
    }
}

// ── Fused kernel ─────────────────────────────────────────────────────────────

/// A compiled fused kernel that replaces two or more primitive operations.
#[derive(Debug, Clone)]
pub struct FusedKernel {
    /// Unique identifier derived from the fused ops.
    pub id: String,
    /// Which pattern this kernel implements.
    pub pattern: FusionPattern,
    /// Indices of the primitive ops this kernel replaces.
    pub fused_ops: Vec<usize>,
    /// Output tensor shape.
    pub output_shape: Vec<usize>,
    /// Estimated execution time of the fused kernel (µs).
    pub estimated_us: f64,
    /// Scratch memory required (bytes).
    pub scratch_bytes: usize,
}

impl FusedKernel {
    /// Compile a fused kernel from a candidate and its source graph.
    pub fn compile(
        candidate: &FusionCandidate,
        graph: &ComputeGraph,
        config: &FusionConfig,
    ) -> FusionResult<Self> {
        let src = graph.get(candidate.src).ok_or(FusionError::InvalidOpIndex(candidate.src))?;
        let dst = graph.get(candidate.dst).ok_or(FusionError::InvalidOpIndex(candidate.dst))?;

        if src.device != dst.device {
            return Err(FusionError::ShapeMismatch {
                src: candidate.src,
                dst: candidate.dst,
                src_shape: src.shape.clone(),
                dst_shape: dst.shape.clone(),
            });
        }

        let element_bytes: usize = 2;
        let scratch = src.shape.iter().product::<usize>() * element_bytes;
        if scratch > config.memory_limit {
            return Err(FusionError::MemoryBudgetExceeded {
                required: scratch,
                limit: config.memory_limit,
            });
        }

        let fused_us = (src.estimated_us + dst.estimated_us) * 0.8;

        let id = format!(
            "fused_{}_{}_{}_{}",
            candidate.pattern, candidate.src, candidate.dst, scratch
        );

        Ok(Self {
            id,
            pattern: candidate.pattern,
            fused_ops: vec![candidate.src, candidate.dst],
            output_shape: dst.shape.clone(),
            estimated_us: fused_us,
            scratch_bytes: scratch,
        })
    }
}

// ── Fused kernel cache ───────────────────────────────────────────────────────

/// Thread-safe cache of compiled [`FusedKernel`]s keyed by their `id`.
#[derive(Debug, Clone)]
pub struct FusedKernelCache {
    inner: Arc<Mutex<HashMap<String, FusedKernel>>>,
    capacity: usize,
}

impl FusedKernelCache {
    /// Create an empty cache with the given maximum capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            capacity,
        }
    }

    /// Try to retrieve a cached kernel.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<FusedKernel> {
        self.inner.lock().ok()?.get(id).cloned()
    }

    /// Insert a kernel.  Returns `false` if the cache is full.
    pub fn insert(&self, kernel: FusedKernel) -> bool {
        if let Ok(mut map) = self.inner.lock() {
            if map.len() >= self.capacity && !map.contains_key(&kernel.id) {
                return false;
            }
            map.insert(kernel.id.clone(), kernel);
            true
        } else {
            false
        }
    }

    /// Number of entries currently cached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().map_or(0, |m| m.len())
    }

    /// Returns `true` when the cache holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all entries.
    pub fn clear(&self) {
        if let Ok(mut map) = self.inner.lock() {
            map.clear();
        }
    }

    /// Returns `true` if the cache contains a kernel with the given id.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.inner.lock().is_ok_and(|m| m.contains_key(id))
    }
}

impl Default for FusedKernelCache {
    fn default() -> Self {
        Self::new(128)
    }
}

// ── Fusion optimizer ─────────────────────────────────────────────────────────

/// Applies one or more fusion passes to a [`ComputeGraph`], producing
/// [`FusedKernel`]s for every profitable candidate.
#[derive(Debug)]
pub struct FusionOptimizer {
    config: FusionConfig,
    matcher: PatternMatcher,
    min_speedup: f64,
}

impl FusionOptimizer {
    /// Build a new optimizer with the given configuration.
    #[must_use]
    pub const fn new(config: FusionConfig) -> Self {
        Self {
            matcher: PatternMatcher::new(),
            config,
            min_speedup: 1.05,
        }
    }

    /// Override the minimum speedup threshold (default 1.05).
    #[must_use]
    pub const fn with_min_speedup(mut self, min: f64) -> Self {
        self.min_speedup = min;
        self
    }

    /// Override the pattern matcher.
    #[must_use]
    pub fn with_matcher(mut self, matcher: PatternMatcher) -> Self {
        self.matcher = matcher;
        self
    }

    /// Run a fusion pass.  Returns fused kernels for every candidate whose
    /// estimated benefit exceeds the threshold.
    pub fn optimize(&self, graph: &ComputeGraph) -> FusionResult<Vec<FusedKernel>> {
        if !self.config.auto_fuse_enabled {
            return Err(FusionError::AutoFuseDisabled);
        }

        let candidates = self.matcher.scan(graph);
        let mut kernels = Vec::new();

        for c in &candidates {
            // Check user-supplied rules.
            let src = graph.get(c.src).ok_or(FusionError::InvalidOpIndex(c.src))?;
            let dst = graph.get(c.dst).ok_or(FusionError::InvalidOpIndex(c.dst))?;
            let rules_ok = self
                .config
                .fusion_rules
                .iter()
                .all(|r| r.allows(src, dst));
            if !rules_ok {
                continue;
            }

            let benefit = FusionBenefit::estimate(c, graph);
            if !benefit.is_worthwhile(self.min_speedup) {
                continue;
            }

            let kernel = FusedKernel::compile(c, graph, &self.config)?;
            kernels.push(kernel);
        }

        Ok(kernels)
    }
}

// ── Kernel fusion engine (orchestrator) ──────────────────────────────────────

/// Top-level orchestrator: analyse graph → match patterns → estimate benefit →
/// fuse → cache.
#[derive(Debug)]
pub struct KernelFusionEngine {
    optimizer: FusionOptimizer,
    cache: FusedKernelCache,
}

/// Summary returned by [`KernelFusionEngine::run`].
#[derive(Debug, Clone)]
pub struct FusionReport {
    /// All fused kernels produced (newly compiled + cache hits).
    pub kernels: Vec<FusedKernel>,
    /// Number of candidates that were found in the cache.
    pub cache_hits: usize,
    /// Number of candidates that were freshly compiled.
    pub cache_misses: usize,
    /// Wall-clock time of the entire fusion pass.
    pub elapsed_us: u128,
}

impl KernelFusionEngine {
    /// Create a new engine with the provided config and cache capacity.
    #[must_use]
    pub fn new(config: FusionConfig, cache_capacity: usize) -> Self {
        Self {
            optimizer: FusionOptimizer::new(config),
            cache: FusedKernelCache::new(cache_capacity),
        }
    }

    /// Create an engine from an existing optimizer and cache.
    #[must_use]
    pub const fn with_parts(optimizer: FusionOptimizer, cache: FusedKernelCache) -> Self {
        Self { optimizer, cache }
    }

    /// Access the underlying cache.
    #[must_use]
    pub const fn cache(&self) -> &FusedKernelCache {
        &self.cache
    }

    /// Run the full fusion pipeline on `graph`.
    pub fn run(&self, graph: &ComputeGraph) -> FusionResult<FusionReport> {
        let start = Instant::now();

        let compiled = self.optimizer.optimize(graph)?;

        let mut kernels = Vec::new();
        let mut hits: usize = 0;
        let mut misses: usize = 0;

        for k in compiled {
            if let Some(cached) = self.cache.get(&k.id) {
                kernels.push(cached);
                hits += 1;
            } else {
                self.cache.insert(k.clone());
                kernels.push(k);
                misses += 1;
            }
        }

        Ok(FusionReport {
            kernels,
            cache_hits: hits,
            cache_misses: misses,
            elapsed_us: start.elapsed().as_micros(),
        })
    }

    /// Clear the compiled-kernel cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

// ── Helper: build tiny test graphs ──────────────────────────────────────────

#[cfg(test)]
fn op(id: usize, kind: OpKind, shape: Vec<usize>) -> OpNode {
    OpNode {
        id,
        kind,
        shape,
        device: "cuda:0".into(),
        estimated_us: 100.0,
        inputs: if id == 0 { vec![] } else { vec![id - 1] },
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    fn matmul_bias_graph() -> ComputeGraph {
        ComputeGraph::new(vec![
            op(0, OpKind::MatMul, vec![1, 128, 256]),
            op(1, OpKind::BiasAdd, vec![1, 128, 256]),
        ])
    }

    fn norm_act_graph() -> ComputeGraph {
        ComputeGraph::new(vec![
            op(0, OpKind::LayerNorm, vec![1, 128]),
            op(1, OpKind::Activation, vec![1, 128]),
        ])
    }

    fn rms_act_graph() -> ComputeGraph {
        ComputeGraph::new(vec![
            op(0, OpKind::RmsNorm, vec![1, 64]),
            op(1, OpKind::Activation, vec![1, 64]),
        ])
    }

    fn attention_graph() -> ComputeGraph {
        ComputeGraph::new(vec![
            op(0, OpKind::MatMul, vec![4, 32, 32]),
            op(1, OpKind::Softmax, vec![4, 32, 32]),
        ])
    }

    fn qkv_graph() -> ComputeGraph {
        ComputeGraph::new(vec![
            op(0, OpKind::QkvProjection, vec![1, 128, 256]),
            op(1, OpKind::QkvProjection, vec![1, 128, 256]),
        ])
    }

    fn gated_ffn_graph() -> ComputeGraph {
        ComputeGraph::new(vec![
            op(0, OpKind::GatedFfn, vec![1, 128, 512]),
            op(1, OpKind::MatMul, vec![1, 128, 256]),
        ])
    }

    fn mixed_graph() -> ComputeGraph {
        ComputeGraph::new(vec![
            op(0, OpKind::MatMul, vec![1, 128, 256]),
            op(1, OpKind::BiasAdd, vec![1, 128, 256]),
            op(2, OpKind::LayerNorm, vec![1, 128, 256]),
            op(3, OpKind::Activation, vec![1, 128, 256]),
            op(4, OpKind::MatMul, vec![4, 32, 32]),
            op(5, OpKind::Softmax, vec![4, 32, 32]),
        ])
    }

    fn empty_graph() -> ComputeGraph {
        ComputeGraph::new(vec![])
    }

    fn single_op_graph() -> ComputeGraph {
        ComputeGraph::new(vec![op(0, OpKind::MatMul, vec![1, 64, 64])])
    }

    fn cross_device_graph() -> ComputeGraph {
        let mut a = op(0, OpKind::MatMul, vec![1, 64, 64]);
        a.device = "cuda:0".into();
        let mut b = op(1, OpKind::BiasAdd, vec![1, 64, 64]);
        b.device = "cuda:1".into();
        ComputeGraph::new(vec![a, b])
    }

    // ── FusionConfig tests ──────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = FusionConfig::default();
        assert_eq!(cfg.max_fused_ops, 4);
        assert_eq!(cfg.memory_limit, 256 * 1024 * 1024);
        assert!(cfg.auto_fuse_enabled);
        assert!(cfg.fusion_rules.is_empty());
    }

    #[test]
    fn config_custom_values() {
        let cfg = FusionConfig {
            max_fused_ops: 8,
            memory_limit: 512 * 1024,
            auto_fuse_enabled: false,
            fusion_rules: vec![],
        };
        assert_eq!(cfg.max_fused_ops, 8);
        assert!(!cfg.auto_fuse_enabled);
    }

    #[test]
    fn config_with_rules() {
        let rule = FusionRule::new("test", |_, _| true);
        let cfg = FusionConfig {
            fusion_rules: vec![rule],
            ..FusionConfig::default()
        };
        assert_eq!(cfg.fusion_rules.len(), 1);
    }

    // ── FusionRule tests ────────────────────────────────────────────────

    #[test]
    fn rule_allows_pair() {
        let rule = FusionRule::new("always-yes", |_, _| true);
        let a = op(0, OpKind::MatMul, vec![1]);
        let b = op(1, OpKind::BiasAdd, vec![1]);
        assert!(rule.allows(&a, &b));
    }

    #[test]
    fn rule_rejects_pair() {
        let rule = FusionRule::new("always-no", |_, _| false);
        let a = op(0, OpKind::MatMul, vec![1]);
        let b = op(1, OpKind::BiasAdd, vec![1]);
        assert!(!rule.allows(&a, &b));
    }

    #[test]
    fn rule_shape_predicate() {
        let rule = FusionRule::new("same-shape", |a, b| a.shape == b.shape);
        let a = op(0, OpKind::MatMul, vec![1, 128]);
        let b = op(1, OpKind::BiasAdd, vec![1, 128]);
        let c = op(2, OpKind::BiasAdd, vec![1, 64]);
        assert!(rule.allows(&a, &b));
        assert!(!rule.allows(&a, &c));
    }

    #[test]
    fn rule_device_predicate() {
        let rule = FusionRule::new("same-device", |a, b| a.device == b.device);
        let a = op(0, OpKind::MatMul, vec![1]);
        let mut b = op(1, OpKind::BiasAdd, vec![1]);
        assert!(rule.allows(&a, &b));
        b.device = "cpu".into();
        assert!(!rule.allows(&a, &b));
    }

    #[test]
    fn rule_debug_format() {
        let rule = FusionRule::new("my-rule", |_, _| true);
        let dbg = format!("{rule:?}");
        assert!(dbg.contains("my-rule"));
    }

    // ── ComputeGraph tests ──────────────────────────────────────────────

    #[test]
    fn graph_len_and_empty() {
        let g = empty_graph();
        assert_eq!(g.len(), 0);
        assert!(g.is_empty());

        let g = single_op_graph();
        assert_eq!(g.len(), 1);
        assert!(!g.is_empty());
    }

    #[test]
    fn graph_get_valid() {
        let g = matmul_bias_graph();
        assert!(g.get(0).is_some());
        assert!(g.get(1).is_some());
        assert!(g.get(2).is_none());
    }

    #[test]
    fn graph_op_properties() {
        let g = matmul_bias_graph();
        let first = g.get(0).unwrap();
        assert_eq!(first.kind, OpKind::MatMul);
        assert_eq!(first.shape, vec![1, 128, 256]);
    }

    // ── FusionCandidate / PatternMatcher tests ──────────────────────────

    #[test]
    fn matcher_finds_matmul_bias() {
        let m = PatternMatcher::new();
        let candidates = m.scan(&matmul_bias_graph());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].pattern, FusionPattern::MatMulBias);
    }

    #[test]
    fn matcher_finds_norm_activation() {
        let m = PatternMatcher::new();
        let candidates = m.scan(&norm_act_graph());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].pattern, FusionPattern::NormActivation);
    }

    #[test]
    fn matcher_finds_rms_norm_activation() {
        let m = PatternMatcher::new();
        let candidates = m.scan(&rms_act_graph());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].pattern, FusionPattern::NormActivation);
    }

    #[test]
    fn matcher_finds_attention() {
        let m = PatternMatcher::new();
        let candidates = m.scan(&attention_graph());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].pattern, FusionPattern::AttentionFused);
    }

    #[test]
    fn matcher_finds_qkv_projection() {
        let m = PatternMatcher::new();
        let candidates = m.scan(&qkv_graph());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].pattern, FusionPattern::QKVProjection);
    }

    #[test]
    fn matcher_finds_gated_ffn() {
        let m = PatternMatcher::new();
        let candidates = m.scan(&gated_ffn_graph());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].pattern, FusionPattern::GatedFFN);
    }

    #[test]
    fn matcher_mixed_graph_multiple_candidates() {
        let m = PatternMatcher::new();
        let candidates = m.scan(&mixed_graph());
        assert!(candidates.len() >= 3);
        let patterns: Vec<_> = candidates.iter().map(|c| c.pattern).collect();
        assert!(patterns.contains(&FusionPattern::MatMulBias));
        assert!(patterns.contains(&FusionPattern::NormActivation));
        assert!(patterns.contains(&FusionPattern::AttentionFused));
    }

    #[test]
    fn matcher_empty_graph_no_candidates() {
        let m = PatternMatcher::new();
        assert!(m.scan(&empty_graph()).is_empty());
    }

    #[test]
    fn matcher_single_op_no_candidates() {
        let m = PatternMatcher::new();
        assert!(m.scan(&single_op_graph()).is_empty());
    }

    #[test]
    fn matcher_cross_device_no_candidates() {
        let m = PatternMatcher::new();
        assert!(m.scan(&cross_device_graph()).is_empty());
    }

    #[test]
    fn matcher_with_restricted_patterns() {
        let m = PatternMatcher::with_patterns(vec![FusionPattern::MatMulBias]);
        let candidates = m.scan(&mixed_graph());
        assert!(candidates.iter().all(|c| c.pattern == FusionPattern::MatMulBias));
    }

    #[test]
    fn matcher_disabled_pattern_not_found() {
        let m = PatternMatcher::with_patterns(vec![FusionPattern::GatedFFN]);
        let candidates = m.scan(&matmul_bias_graph());
        assert!(candidates.is_empty());
    }

    #[test]
    fn matcher_default_trait() {
        let m = PatternMatcher::default();
        let c = m.scan(&matmul_bias_graph());
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn candidate_src_dst_indices() {
        let m = PatternMatcher::new();
        let c = m.scan(&matmul_bias_graph());
        assert_eq!(c[0].src, 0);
        assert_eq!(c[0].dst, 1);
    }

    // ── FusionPattern display ───────────────────────────────────────────

    #[test]
    fn pattern_display() {
        assert_eq!(FusionPattern::MatMulBias.to_string(), "MatMulBias");
        assert_eq!(FusionPattern::NormActivation.to_string(), "NormActivation");
        assert_eq!(FusionPattern::AttentionFused.to_string(), "AttentionFused");
        assert_eq!(FusionPattern::QKVProjection.to_string(), "QKVProjection");
        assert_eq!(FusionPattern::GatedFFN.to_string(), "GatedFFN");
    }

    #[test]
    fn op_kind_display() {
        assert_eq!(OpKind::MatMul.to_string(), "MatMul");
        assert_eq!(OpKind::BiasAdd.to_string(), "BiasAdd");
        assert_eq!(OpKind::LayerNorm.to_string(), "LayerNorm");
    }

    // ── FusionBenefit tests ─────────────────────────────────────────────

    #[test]
    fn benefit_matmul_bias() {
        let g = matmul_bias_graph();
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::MatMulBias };
        let b = FusionBenefit::estimate(&c, &g);
        assert!(b.speedup > 1.0);
        assert_eq!(b.kernel_launches_saved, 1);
        assert!(b.memory_traffic_saved > 0);
        assert_eq!(b.pattern, FusionPattern::MatMulBias);
    }

    #[test]
    fn benefit_norm_activation() {
        let g = norm_act_graph();
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::NormActivation };
        let b = FusionBenefit::estimate(&c, &g);
        assert!(b.speedup > 1.0);
        assert_eq!(b.kernel_launches_saved, 1);
    }

    #[test]
    fn benefit_attention_fused() {
        let g = attention_graph();
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::AttentionFused };
        let b = FusionBenefit::estimate(&c, &g);
        assert!(b.speedup > 1.0);
        assert_eq!(b.kernel_launches_saved, 2);
    }

    #[test]
    fn benefit_qkv_projection() {
        let g = qkv_graph();
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::QKVProjection };
        let b = FusionBenefit::estimate(&c, &g);
        assert!(b.speedup > 1.0);
        assert_eq!(b.kernel_launches_saved, 2);
    }

    #[test]
    fn benefit_gated_ffn() {
        let g = gated_ffn_graph();
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::GatedFFN };
        let b = FusionBenefit::estimate(&c, &g);
        assert!(b.speedup > 1.0);
        assert_eq!(b.kernel_launches_saved, 1);
    }

    #[test]
    fn benefit_is_worthwhile_threshold() {
        let g = matmul_bias_graph();
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::MatMulBias };
        let b = FusionBenefit::estimate(&c, &g);
        assert!(b.is_worthwhile(1.01));
        assert!(!b.is_worthwhile(999.0));
    }

    #[test]
    fn benefit_zero_estimated_time() {
        let mut g = matmul_bias_graph();
        g.ops[0].estimated_us = 0.0;
        g.ops[1].estimated_us = 0.0;
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::MatMulBias };
        let b = FusionBenefit::estimate(&c, &g);
        assert!(b.speedup > 1.0);
    }

    #[test]
    fn benefit_large_execution_time() {
        let mut g = matmul_bias_graph();
        g.ops[0].estimated_us = 50_000.0;
        g.ops[1].estimated_us = 50_000.0;
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::MatMulBias };
        let b = FusionBenefit::estimate(&c, &g);
        // With large time, adjustment should bring speedup closer to base
        assert!(b.speedup > 1.1);
    }

    // ── FusedKernel tests ───────────────────────────────────────────────

    #[test]
    fn fused_kernel_compile_matmul_bias() {
        let g = matmul_bias_graph();
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::MatMulBias };
        let k = FusedKernel::compile(&c, &g, &FusionConfig::default()).unwrap();
        assert_eq!(k.pattern, FusionPattern::MatMulBias);
        assert_eq!(k.fused_ops, vec![0, 1]);
        assert_eq!(k.output_shape, vec![1, 128, 256]);
        assert!(k.estimated_us > 0.0);
        assert!(k.scratch_bytes > 0);
    }

    #[test]
    fn fused_kernel_id_contains_pattern() {
        let g = matmul_bias_graph();
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::MatMulBias };
        let k = FusedKernel::compile(&c, &g, &FusionConfig::default()).unwrap();
        assert!(k.id.contains("MatMulBias"));
    }

    #[test]
    fn fused_kernel_compile_norm_act() {
        let g = norm_act_graph();
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::NormActivation };
        let k = FusedKernel::compile(&c, &g, &FusionConfig::default()).unwrap();
        assert_eq!(k.pattern, FusionPattern::NormActivation);
    }

    #[test]
    fn fused_kernel_memory_budget_exceeded() {
        let g = matmul_bias_graph();
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::MatMulBias };
        let cfg = FusionConfig { memory_limit: 1, ..FusionConfig::default() };
        let err = FusedKernel::compile(&c, &g, &cfg).unwrap_err();
        assert!(matches!(err, FusionError::MemoryBudgetExceeded { .. }));
    }

    #[test]
    fn fused_kernel_invalid_op_index() {
        let g = single_op_graph();
        let c = FusionCandidate { src: 0, dst: 99, pattern: FusionPattern::MatMulBias };
        let err = FusedKernel::compile(&c, &g, &FusionConfig::default()).unwrap_err();
        assert!(matches!(err, FusionError::InvalidOpIndex(99)));
    }

    #[test]
    fn fused_kernel_cross_device_error() {
        let g = cross_device_graph();
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::MatMulBias };
        let err = FusedKernel::compile(&c, &g, &FusionConfig::default()).unwrap_err();
        assert!(matches!(err, FusionError::ShapeMismatch { .. }));
    }

    #[test]
    fn fused_kernel_estimated_time_reduced() {
        let g = matmul_bias_graph();
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::MatMulBias };
        let k = FusedKernel::compile(&c, &g, &FusionConfig::default()).unwrap();
        let total_unfused = g.ops[0].estimated_us + g.ops[1].estimated_us;
        assert!(k.estimated_us < total_unfused);
    }

    // ── FusedKernelCache tests ──────────────────────────────────────────

    #[test]
    fn cache_new_empty() {
        let cache = FusedKernelCache::new(16);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_insert_and_get() {
        let cache = FusedKernelCache::new(16);
        let g = matmul_bias_graph();
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::MatMulBias };
        let k = FusedKernel::compile(&c, &g, &FusionConfig::default()).unwrap();
        let id = k.id.clone();
        assert!(cache.insert(k));
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&id).is_some());
    }

    #[test]
    fn cache_get_missing() {
        let cache = FusedKernelCache::new(16);
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn cache_capacity_limit() {
        let cache = FusedKernelCache::new(1);
        let k1 = FusedKernel {
            id: "k1".into(),
            pattern: FusionPattern::MatMulBias,
            fused_ops: vec![0, 1],
            output_shape: vec![1],
            estimated_us: 10.0,
            scratch_bytes: 64,
        };
        let k2 = FusedKernel {
            id: "k2".into(),
            pattern: FusionPattern::NormActivation,
            fused_ops: vec![2, 3],
            output_shape: vec![1],
            estimated_us: 20.0,
            scratch_bytes: 64,
        };
        assert!(cache.insert(k1));
        assert!(!cache.insert(k2));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_duplicate_insert_updates() {
        let cache = FusedKernelCache::new(1);
        let k1 = FusedKernel {
            id: "k1".into(),
            pattern: FusionPattern::MatMulBias,
            fused_ops: vec![0, 1],
            output_shape: vec![1],
            estimated_us: 10.0,
            scratch_bytes: 64,
        };
        let k1b = FusedKernel {
            id: "k1".into(),
            pattern: FusionPattern::MatMulBias,
            fused_ops: vec![0, 1],
            output_shape: vec![1],
            estimated_us: 20.0,
            scratch_bytes: 128,
        };
        assert!(cache.insert(k1));
        // Same key → still succeeds even at capacity
        assert!(cache.insert(k1b));
        assert_eq!(cache.len(), 1);
        assert!((cache.get("k1").unwrap().estimated_us - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cache_clear() {
        let cache = FusedKernelCache::new(16);
        let k = FusedKernel {
            id: "k".into(),
            pattern: FusionPattern::MatMulBias,
            fused_ops: vec![],
            output_shape: vec![],
            estimated_us: 0.0,
            scratch_bytes: 0,
        };
        cache.insert(k);
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_contains() {
        let cache = FusedKernelCache::new(16);
        let k = FusedKernel {
            id: "abc".into(),
            pattern: FusionPattern::MatMulBias,
            fused_ops: vec![],
            output_shape: vec![],
            estimated_us: 0.0,
            scratch_bytes: 0,
        };
        assert!(!cache.contains("abc"));
        cache.insert(k);
        assert!(cache.contains("abc"));
    }

    #[test]
    fn cache_default_capacity() {
        let cache = FusedKernelCache::default();
        // Default capacity should be 128 - we can insert many items
        for i in 0..128 {
            cache.insert(FusedKernel {
                id: format!("k{i}"),
                pattern: FusionPattern::MatMulBias,
                fused_ops: vec![],
                output_shape: vec![],
                estimated_us: 0.0,
                scratch_bytes: 0,
            });
        }
        assert_eq!(cache.len(), 128);
    }

    // ── FusionOptimizer tests ───────────────────────────────────────────

    #[test]
    fn optimizer_basic_matmul_bias() {
        let opt = FusionOptimizer::new(FusionConfig::default());
        let kernels = opt.optimize(&matmul_bias_graph()).unwrap();
        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].pattern, FusionPattern::MatMulBias);
    }

    #[test]
    fn optimizer_auto_fuse_disabled() {
        let cfg = FusionConfig { auto_fuse_enabled: false, ..FusionConfig::default() };
        let opt = FusionOptimizer::new(cfg);
        let err = opt.optimize(&matmul_bias_graph()).unwrap_err();
        assert!(matches!(err, FusionError::AutoFuseDisabled));
    }

    #[test]
    fn optimizer_empty_graph() {
        let opt = FusionOptimizer::new(FusionConfig::default());
        let kernels = opt.optimize(&empty_graph()).unwrap();
        assert!(kernels.is_empty());
    }

    #[test]
    fn optimizer_respects_rules() {
        let cfg = FusionConfig {
            fusion_rules: vec![FusionRule::new("block-all", |_, _| false)],
            ..FusionConfig::default()
        };
        let opt = FusionOptimizer::new(cfg);
        let kernels = opt.optimize(&matmul_bias_graph()).unwrap();
        assert!(kernels.is_empty());
    }

    #[test]
    fn optimizer_high_min_speedup_filters() {
        let opt = FusionOptimizer::new(FusionConfig::default()).with_min_speedup(999.0);
        let kernels = opt.optimize(&matmul_bias_graph()).unwrap();
        assert!(kernels.is_empty());
    }

    #[test]
    fn optimizer_with_custom_matcher() {
        let matcher = PatternMatcher::with_patterns(vec![FusionPattern::NormActivation]);
        let opt = FusionOptimizer::new(FusionConfig::default()).with_matcher(matcher);
        let kernels = opt.optimize(&matmul_bias_graph()).unwrap();
        assert!(kernels.is_empty());
    }

    #[test]
    fn optimizer_mixed_graph() {
        let opt = FusionOptimizer::new(FusionConfig::default());
        let kernels = opt.optimize(&mixed_graph()).unwrap();
        assert!(kernels.len() >= 3);
    }

    #[test]
    fn optimizer_cross_device_skipped() {
        let opt = FusionOptimizer::new(FusionConfig::default());
        let kernels = opt.optimize(&cross_device_graph()).unwrap();
        assert!(kernels.is_empty());
    }

    #[test]
    fn optimizer_memory_limit_error_propagates() {
        let cfg = FusionConfig { memory_limit: 1, ..FusionConfig::default() };
        let opt = FusionOptimizer::new(cfg);
        let result = opt.optimize(&matmul_bias_graph());
        assert!(result.is_err());
    }

    // ── KernelFusionEngine tests ────────────────────────────────────────

    #[test]
    fn engine_run_basic() {
        let engine = KernelFusionEngine::new(FusionConfig::default(), 64);
        let report = engine.run(&matmul_bias_graph()).unwrap();
        assert_eq!(report.kernels.len(), 1);
        assert_eq!(report.cache_misses, 1);
        assert_eq!(report.cache_hits, 0);
    }

    #[test]
    fn engine_cache_hit_on_rerun() {
        let engine = KernelFusionEngine::new(FusionConfig::default(), 64);
        let _first = engine.run(&matmul_bias_graph()).unwrap();
        let second = engine.run(&matmul_bias_graph()).unwrap();
        assert_eq!(second.cache_hits, 1);
        assert_eq!(second.cache_misses, 0);
    }

    #[test]
    fn engine_clear_cache() {
        let engine = KernelFusionEngine::new(FusionConfig::default(), 64);
        let _ = engine.run(&matmul_bias_graph()).unwrap();
        assert!(!engine.cache().is_empty());
        engine.clear_cache();
        assert!(engine.cache().is_empty());
    }

    #[test]
    fn engine_with_parts() {
        let opt = FusionOptimizer::new(FusionConfig::default());
        let cache = FusedKernelCache::new(32);
        let engine = KernelFusionEngine::with_parts(opt, cache);
        let report = engine.run(&matmul_bias_graph()).unwrap();
        assert!(!report.kernels.is_empty());
    }

    #[test]
    fn engine_empty_graph() {
        let engine = KernelFusionEngine::new(FusionConfig::default(), 64);
        let report = engine.run(&empty_graph()).unwrap();
        assert!(report.kernels.is_empty());
        assert_eq!(report.cache_hits, 0);
        assert_eq!(report.cache_misses, 0);
    }

    #[test]
    fn engine_auto_fuse_disabled() {
        let cfg = FusionConfig { auto_fuse_enabled: false, ..FusionConfig::default() };
        let engine = KernelFusionEngine::new(cfg, 64);
        let err = engine.run(&matmul_bias_graph()).unwrap_err();
        assert!(matches!(err, FusionError::AutoFuseDisabled));
    }

    #[test]
    fn engine_mixed_graph_multiple_kernels() {
        let engine = KernelFusionEngine::new(FusionConfig::default(), 64);
        let report = engine.run(&mixed_graph()).unwrap();
        assert!(report.kernels.len() >= 3);
    }

    #[test]
    fn engine_elapsed_recorded() {
        let engine = KernelFusionEngine::new(FusionConfig::default(), 64);
        let report = engine.run(&matmul_bias_graph()).unwrap();
        // elapsed_us is set (may be 0 on very fast machines, but shouldn't panic)
        let _ = report.elapsed_us;
    }

    #[test]
    fn engine_cache_accessor() {
        let engine = KernelFusionEngine::new(FusionConfig::default(), 64);
        assert!(engine.cache().is_empty());
    }

    // ── Error display tests ─────────────────────────────────────────────

    #[test]
    fn error_shape_mismatch_display() {
        let e = FusionError::ShapeMismatch {
            src: 0,
            dst: 1,
            src_shape: vec![1, 2],
            dst_shape: vec![3, 4],
        };
        let msg = e.to_string();
        assert!(msg.contains("shape mismatch"));
        assert!(msg.contains("[1, 2]"));
        assert!(msg.contains("[3, 4]"));
    }

    #[test]
    fn error_memory_budget_display() {
        let e = FusionError::MemoryBudgetExceeded {
            required: 1024,
            limit: 512,
        };
        let msg = e.to_string();
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn error_invalid_op_index_display() {
        let e = FusionError::InvalidOpIndex(42);
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn error_auto_fuse_disabled_display() {
        let e = FusionError::AutoFuseDisabled;
        assert!(e.to_string().contains("disabled"));
    }

    #[test]
    fn error_other_display() {
        let e = FusionError::Other("custom".into());
        assert!(e.to_string().contains("custom"));
    }

    // ── OpNode tests ────────────────────────────────────────────────────

    #[test]
    fn op_node_clone() {
        let a = op(0, OpKind::MatMul, vec![1, 2, 3]);
        let b = a.clone();
        assert_eq!(a.id, b.id);
        assert_eq!(a.shape, b.shape);
    }

    #[test]
    fn op_node_inputs_chain() {
        let g = ComputeGraph::new(vec![
            op(0, OpKind::MatMul, vec![1]),
            op(1, OpKind::BiasAdd, vec![1]),
            op(2, OpKind::Activation, vec![1]),
        ]);
        assert!(g.ops[0].inputs.is_empty());
        assert_eq!(g.ops[1].inputs, vec![0]);
        assert_eq!(g.ops[2].inputs, vec![1]);
    }

    // ── Integration / end-to-end tests ──────────────────────────────────

    #[test]
    fn e2e_full_transformer_block_fusion() {
        // Simulate a mini transformer block:
        // MatMul → Bias → LayerNorm → Activation → MatMul → Softmax
        let graph = mixed_graph();
        let engine = KernelFusionEngine::new(FusionConfig::default(), 128);
        let report = engine.run(&graph).unwrap();
        assert!(report.kernels.len() >= 3);
        let patterns: Vec<_> = report.kernels.iter().map(|k| k.pattern).collect();
        assert!(patterns.contains(&FusionPattern::MatMulBias));
        assert!(patterns.contains(&FusionPattern::NormActivation));
        assert!(patterns.contains(&FusionPattern::AttentionFused));
    }

    #[test]
    fn e2e_repeated_runs_populate_cache() {
        let engine = KernelFusionEngine::new(FusionConfig::default(), 128);
        let g = matmul_bias_graph();

        let r1 = engine.run(&g).unwrap();
        assert_eq!(r1.cache_misses, 1);

        let r2 = engine.run(&g).unwrap();
        assert_eq!(r2.cache_hits, 1);
        assert_eq!(r2.cache_misses, 0);

        let r3 = engine.run(&g).unwrap();
        assert_eq!(r3.cache_hits, 1);
    }

    #[test]
    fn e2e_different_graphs_different_kernels() {
        let engine = KernelFusionEngine::new(FusionConfig::default(), 128);
        let r1 = engine.run(&matmul_bias_graph()).unwrap();
        let r2 = engine.run(&norm_act_graph()).unwrap();
        assert_ne!(r1.kernels[0].id, r2.kernels[0].id);
        assert_eq!(engine.cache().len(), 2);
    }

    #[test]
    fn e2e_custom_rule_filters_candidates() {
        let cfg = FusionConfig {
            fusion_rules: vec![FusionRule::new("only-norm", |a, _| {
                a.kind == OpKind::LayerNorm || a.kind == OpKind::RmsNorm
            })],
            ..FusionConfig::default()
        };
        let engine = KernelFusionEngine::new(cfg, 128);
        let g = mixed_graph();
        let report = engine.run(&g).unwrap();
        // Only NormActivation should survive the rule
        assert!(report.kernels.iter().all(|k| k.pattern == FusionPattern::NormActivation));
    }

    #[test]
    fn e2e_qkv_projection_fusion() {
        let engine = KernelFusionEngine::new(FusionConfig::default(), 64);
        let report = engine.run(&qkv_graph()).unwrap();
        assert_eq!(report.kernels.len(), 1);
        assert_eq!(report.kernels[0].pattern, FusionPattern::QKVProjection);
    }

    #[test]
    fn e2e_gated_ffn_fusion() {
        let engine = KernelFusionEngine::new(FusionConfig::default(), 64);
        let report = engine.run(&gated_ffn_graph()).unwrap();
        assert_eq!(report.kernels.len(), 1);
        assert_eq!(report.kernels[0].pattern, FusionPattern::GatedFFN);
    }

    #[test]
    fn e2e_attention_fusion() {
        let engine = KernelFusionEngine::new(FusionConfig::default(), 64);
        let report = engine.run(&attention_graph()).unwrap();
        assert_eq!(report.kernels.len(), 1);
        assert_eq!(report.kernels[0].pattern, FusionPattern::AttentionFused);
    }

    #[test]
    fn e2e_all_patterns_covered() {
        // Build a graph with every pattern
        let graph = ComputeGraph::new(vec![
            op(0, OpKind::MatMul, vec![1, 64]),
            op(1, OpKind::BiasAdd, vec![1, 64]),
            op(2, OpKind::LayerNorm, vec![1, 64]),
            op(3, OpKind::Activation, vec![1, 64]),
            op(4, OpKind::MatMul, vec![4, 32, 32]),
            op(5, OpKind::Softmax, vec![4, 32, 32]),
            op(6, OpKind::QkvProjection, vec![1, 64]),
            op(7, OpKind::QkvProjection, vec![1, 64]),
            op(8, OpKind::GatedFfn, vec![1, 128]),
            op(9, OpKind::MatMul, vec![1, 64]),
        ]);
        let engine = KernelFusionEngine::new(FusionConfig::default(), 128);
        let report = engine.run(&graph).unwrap();
        let patterns: Vec<_> = report.kernels.iter().map(|k| k.pattern).collect();
        assert!(patterns.contains(&FusionPattern::MatMulBias));
        assert!(patterns.contains(&FusionPattern::NormActivation));
        assert!(patterns.contains(&FusionPattern::AttentionFused));
        assert!(patterns.contains(&FusionPattern::QKVProjection));
        assert!(patterns.contains(&FusionPattern::GatedFFN));
    }

    #[test]
    fn e2e_rms_norm_activation_fusion() {
        let engine = KernelFusionEngine::new(FusionConfig::default(), 64);
        let report = engine.run(&rms_act_graph()).unwrap();
        assert_eq!(report.kernels.len(), 1);
        assert_eq!(report.kernels[0].pattern, FusionPattern::NormActivation);
    }

    #[test]
    fn fusion_report_totals_correct() {
        let engine = KernelFusionEngine::new(FusionConfig::default(), 128);
        let report = engine.run(&mixed_graph()).unwrap();
        assert_eq!(
            report.kernels.len(),
            report.cache_hits + report.cache_misses,
        );
    }

    #[test]
    fn fused_kernel_scratch_bytes_proportional_to_shape() {
        let small = ComputeGraph::new(vec![
            op(0, OpKind::MatMul, vec![1, 16]),
            op(1, OpKind::BiasAdd, vec![1, 16]),
        ]);
        let large = ComputeGraph::new(vec![
            op(0, OpKind::MatMul, vec![1, 1024]),
            op(1, OpKind::BiasAdd, vec![1, 1024]),
        ]);
        let c = FusionCandidate { src: 0, dst: 1, pattern: FusionPattern::MatMulBias };
        let ks = FusedKernel::compile(&c, &small, &FusionConfig::default()).unwrap();
        let kl = FusedKernel::compile(&c, &large, &FusionConfig::default()).unwrap();
        assert!(kl.scratch_bytes > ks.scratch_bytes);
    }
}
