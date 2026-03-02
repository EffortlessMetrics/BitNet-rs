//! Comprehensive CUDA kernel fusion framework for BitNet inference.
//!
//! This module builds on the pairwise fusion in [`super::fusion`] by providing
//! a **planner → optimizer → registry** pipeline that identifies, selects, and
//! dispatches multi-kernel fusion opportunities across an entire operation
//! sequence.
//!
//! # Architecture
//!
//! ```text
//! Op sequence → FusionPlanner → FusionOptimizer → FusionRegistry → dispatch
//! ```
//!
//! 1. **[`FusionPlanner`]** scans a sequence of [`OpDescriptor`]s and emits
//!    [`FusionOpportunity`]s using pattern matching against [`FusionPattern`].
//! 2. **[`FusionOptimizer`]** selects the best non-overlapping subset of
//!    opportunities given hardware constraints ([`HardwareConstraints`]).
//! 3. **[`FusionRegistry`]** maps each selected [`FusionPattern`] to a
//!    concrete [`FusedKernel`] implementation (GPU launch or CPU fallback).
//!
//! # CPU fallback
//!
//! Every fused operation has a sequential CPU fallback that executes the
//! constituent operations in order.  The CUDA launch paths are gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use bitnet_common::{KernelError, Result};

// ───────────────────────────────────────────────────────────────────
// Activation helper
// ───────────────────────────────────────────────────────────────────

/// SiLU activation: x * sigmoid(x).
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU (tanh approximation).
#[inline]
fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + COEFF * x * x * x)).tanh())
}

/// ReLU activation.
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

// ───────────────────────────────────────────────────────────────────
// Primitive operation descriptors
// ───────────────────────────────────────────────────────────────────

/// A single primitive operation in the computation graph.
#[derive(Debug, Clone, PartialEq)]
pub enum OpType {
    MatMul { m: usize, n: usize, k: usize },
    BiasAdd { len: usize },
    Activation(ActivationKind),
    LayerNorm { len: usize },
    RmsNorm { len: usize },
    ResidualAdd { len: usize },
    Softmax { len: usize },
    AttentionMask { len: usize },
    Linear { in_features: usize, out_features: usize },
    EmbeddingLookup { vocab_size: usize, embed_dim: usize },
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatMul { m, n, k } => write!(f, "MatMul({m}×{k}×{n})"),
            Self::BiasAdd { len } => write!(f, "BiasAdd({len})"),
            Self::Activation(kind) => write!(f, "Activation({kind})"),
            Self::LayerNorm { len } => write!(f, "LayerNorm({len})"),
            Self::RmsNorm { len } => write!(f, "RmsNorm({len})"),
            Self::ResidualAdd { len } => write!(f, "ResidualAdd({len})"),
            Self::Softmax { len } => write!(f, "Softmax({len})"),
            Self::AttentionMask { len } => write!(f, "AttentionMask({len})"),
            Self::Linear { in_features, out_features } => {
                write!(f, "Linear({in_features}→{out_features})")
            }
            Self::EmbeddingLookup { vocab_size, embed_dim } => {
                write!(f, "Embedding({vocab_size},{embed_dim})")
            }
        }
    }
}

/// Activation function kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationKind {
    ReLU,
    SiLU,
    GELU,
}

impl fmt::Display for ActivationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReLU => write!(f, "ReLU"),
            Self::SiLU => write!(f, "SiLU"),
            Self::GELU => write!(f, "GELU"),
        }
    }
}

/// An annotated operation in the sequence with an opaque id.
#[derive(Debug, Clone)]
pub struct OpDescriptor {
    /// Unique index in the operation sequence.
    pub id: usize,
    /// The primitive operation.
    pub op: OpType,
    /// Estimated element count (for cost modelling).
    pub element_count: usize,
}

// ───────────────────────────────────────────────────────────────────
// Fusion patterns
// ───────────────────────────────────────────────────────────────────

/// Recognised multi-kernel fusion patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// MatMul → BiasAdd → Activation.
    MatMulBiasActivation,
    /// LayerNorm → ResidualAdd.
    LayerNormResidual,
    /// Attention QK^T → AttentionMask → Softmax.
    AttentionSoftmax,
    /// Linear → Activation → Linear (full FFN).
    FeedForwardNetwork,
    /// EmbeddingLookup → LayerNorm.
    EmbeddingLayerNorm,
    /// RmsNorm → Linear.
    RmsNormLinear,
    /// ResidualAdd → RmsNorm.
    ResidualRmsNorm,
}

impl FusionPattern {
    /// Number of constituent operations in this pattern.
    pub fn op_count(&self) -> usize {
        match self {
            Self::MatMulBiasActivation => 3,
            Self::LayerNormResidual => 2,
            Self::AttentionSoftmax => 3,
            Self::FeedForwardNetwork => 3,
            Self::EmbeddingLayerNorm => 2,
            Self::RmsNormLinear => 2,
            Self::ResidualRmsNorm => 2,
        }
    }

    /// Estimated launch-overhead savings (number of eliminated kernel launches).
    pub fn saved_launches(&self) -> usize {
        self.op_count().saturating_sub(1)
    }

    /// Human-readable name of this pattern.
    pub fn name(&self) -> &'static str {
        match self {
            Self::MatMulBiasActivation => "MatMul+Bias+Activation",
            Self::LayerNormResidual => "LayerNorm+Residual",
            Self::AttentionSoftmax => "Attention+Mask+Softmax",
            Self::FeedForwardNetwork => "FFN(Linear+Act+Linear)",
            Self::EmbeddingLayerNorm => "Embedding+LayerNorm",
            Self::RmsNormLinear => "RmsNorm+Linear",
            Self::ResidualRmsNorm => "Residual+RmsNorm",
        }
    }
}

impl fmt::Display for FusionPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ───────────────────────────────────────────────────────────────────
// Fusion rules
// ───────────────────────────────────────────────────────────────────

/// A rule that determines when a specific fusion pattern is profitable.
#[derive(Debug, Clone)]
pub struct FusionRule {
    /// The pattern this rule applies to.
    pub pattern: FusionPattern,
    /// Minimum total element count for fusion to be worthwhile.
    pub min_elements: usize,
    /// Maximum total element count (very large tensors may not benefit).
    pub max_elements: usize,
    /// Whether this rule is currently enabled.
    pub enabled: bool,
    /// Priority when two rules overlap (higher wins).
    pub priority: u32,
}

impl FusionRule {
    /// Create a new enabled rule with default element bounds.
    pub fn new(pattern: FusionPattern) -> Self {
        Self {
            pattern,
            min_elements: 32,
            max_elements: usize::MAX,
            enabled: true,
            priority: pattern.saved_launches() as u32,
        }
    }

    /// Check whether the rule is satisfied for the given element count.
    pub fn is_applicable(&self, element_count: usize) -> bool {
        self.enabled && element_count >= self.min_elements && element_count <= self.max_elements
    }

    /// Builder: set minimum elements.
    pub fn with_min_elements(mut self, min: usize) -> Self {
        self.min_elements = min;
        self
    }

    /// Builder: set maximum elements.
    pub fn with_max_elements(mut self, max: usize) -> Self {
        self.max_elements = max;
        self
    }

    /// Builder: set enabled flag.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Builder: set priority.
    pub fn with_priority(mut self, p: u32) -> Self {
        self.priority = p;
        self
    }
}

// ───────────────────────────────────────────────────────────────────
// Fusion opportunity
// ───────────────────────────────────────────────────────────────────

/// A detected fusion opportunity in an operation sequence.
#[derive(Debug, Clone)]
pub struct FusionOpportunity {
    /// The matched fusion pattern.
    pub pattern: FusionPattern,
    /// Indices (into the original op sequence) of fused operations.
    pub op_indices: Vec<usize>,
    /// Estimated speedup factor (1.0 = neutral).
    pub estimated_speedup: f64,
    /// Priority used during non-overlapping selection.
    pub priority: u32,
}

// ───────────────────────────────────────────────────────────────────
// Hardware constraints
// ───────────────────────────────────────────────────────────────────

/// Hardware constraints that influence fusion profitability.
#[derive(Debug, Clone)]
pub struct HardwareConstraints {
    /// Maximum shared memory per block (bytes).
    pub max_shared_mem_bytes: usize,
    /// Maximum threads per block.
    pub max_threads_per_block: usize,
    /// Maximum registers per thread.
    pub max_registers_per_thread: usize,
    /// Whether GPU is available at runtime.
    pub gpu_available: bool,
    /// GPU compute capability (e.g. 80 for sm_80).
    pub compute_capability: u32,
}

impl Default for HardwareConstraints {
    fn default() -> Self {
        Self {
            max_shared_mem_bytes: 48 * 1024,
            max_threads_per_block: 1024,
            max_registers_per_thread: 255,
            gpu_available: false,
            compute_capability: 0,
        }
    }
}

impl HardwareConstraints {
    /// CPU-only constraint set.
    pub fn cpu_only() -> Self {
        Self::default()
    }

    /// Typical NVIDIA GPU constraints.
    pub fn gpu(compute_capability: u32, shared_mem: usize) -> Self {
        Self {
            max_shared_mem_bytes: shared_mem,
            max_threads_per_block: 1024,
            max_registers_per_thread: 255,
            gpu_available: true,
            compute_capability,
        }
    }

    /// Check whether a fused kernel's resource requirements fit.
    pub fn can_launch(&self, shared_mem: usize, threads: usize) -> bool {
        shared_mem <= self.max_shared_mem_bytes && threads <= self.max_threads_per_block
    }
}

// ───────────────────────────────────────────────────────────────────
// FusionPlanner
// ───────────────────────────────────────────────────────────────────

/// Scans an operation sequence and identifies fusion opportunities.
pub struct FusionPlanner {
    rules: Vec<FusionRule>,
}

impl FusionPlanner {
    /// Create a planner with the default rule set (all patterns enabled).
    pub fn new() -> Self {
        Self {
            rules: vec![
                FusionRule::new(FusionPattern::MatMulBiasActivation),
                FusionRule::new(FusionPattern::LayerNormResidual),
                FusionRule::new(FusionPattern::AttentionSoftmax),
                FusionRule::new(FusionPattern::FeedForwardNetwork),
                FusionRule::new(FusionPattern::EmbeddingLayerNorm),
                FusionRule::new(FusionPattern::RmsNormLinear),
                FusionRule::new(FusionPattern::ResidualRmsNorm),
            ],
        }
    }

    /// Create a planner with custom rules.
    pub fn with_rules(rules: Vec<FusionRule>) -> Self {
        Self { rules }
    }

    /// Return a reference to the current rule set.
    pub fn rules(&self) -> &[FusionRule] {
        &self.rules
    }

    /// Scan the operation sequence and return all fusion opportunities.
    pub fn find_opportunities(&self, ops: &[OpDescriptor]) -> Vec<FusionOpportunity> {
        let mut opportunities = Vec::new();
        let len = ops.len();

        for rule in &self.rules {
            if !rule.enabled {
                continue;
            }
            match rule.pattern {
                FusionPattern::MatMulBiasActivation => {
                    self.match_matmul_bias_act(ops, len, rule, &mut opportunities);
                }
                FusionPattern::LayerNormResidual => {
                    self.match_pair(
                        ops,
                        len,
                        rule,
                        |a| matches!(a, OpType::LayerNorm { .. }),
                        |b| matches!(b, OpType::ResidualAdd { .. }),
                        &mut opportunities,
                    );
                }
                FusionPattern::AttentionSoftmax => {
                    self.match_attention_softmax(ops, len, rule, &mut opportunities);
                }
                FusionPattern::FeedForwardNetwork => {
                    self.match_ffn(ops, len, rule, &mut opportunities);
                }
                FusionPattern::EmbeddingLayerNorm => {
                    self.match_pair(
                        ops,
                        len,
                        rule,
                        |a| matches!(a, OpType::EmbeddingLookup { .. }),
                        |b| matches!(b, OpType::LayerNorm { .. }),
                        &mut opportunities,
                    );
                }
                FusionPattern::RmsNormLinear => {
                    self.match_pair(
                        ops,
                        len,
                        rule,
                        |a| matches!(a, OpType::RmsNorm { .. }),
                        |b| matches!(b, OpType::Linear { .. }),
                        &mut opportunities,
                    );
                }
                FusionPattern::ResidualRmsNorm => {
                    self.match_pair(
                        ops,
                        len,
                        rule,
                        |a| matches!(a, OpType::ResidualAdd { .. }),
                        |b| matches!(b, OpType::RmsNorm { .. }),
                        &mut opportunities,
                    );
                }
            }
        }
        opportunities
    }

    fn match_matmul_bias_act(
        &self,
        ops: &[OpDescriptor],
        len: usize,
        rule: &FusionRule,
        out: &mut Vec<FusionOpportunity>,
    ) {
        if len < 3 {
            return;
        }
        for i in 0..len - 2 {
            let total = ops[i].element_count + ops[i + 1].element_count + ops[i + 2].element_count;
            if matches!(ops[i].op, OpType::MatMul { .. })
                && matches!(ops[i + 1].op, OpType::BiasAdd { .. })
                && matches!(ops[i + 2].op, OpType::Activation(_))
                && rule.is_applicable(total)
            {
                out.push(FusionOpportunity {
                    pattern: rule.pattern,
                    op_indices: vec![ops[i].id, ops[i + 1].id, ops[i + 2].id],
                    estimated_speedup: 1.3,
                    priority: rule.priority,
                });
            }
        }
    }

    fn match_attention_softmax(
        &self,
        ops: &[OpDescriptor],
        len: usize,
        rule: &FusionRule,
        out: &mut Vec<FusionOpportunity>,
    ) {
        if len < 3 {
            return;
        }
        for i in 0..len - 2 {
            let total = ops[i].element_count + ops[i + 1].element_count + ops[i + 2].element_count;
            if matches!(ops[i].op, OpType::MatMul { .. })
                && matches!(ops[i + 1].op, OpType::AttentionMask { .. })
                && matches!(ops[i + 2].op, OpType::Softmax { .. })
                && rule.is_applicable(total)
            {
                out.push(FusionOpportunity {
                    pattern: rule.pattern,
                    op_indices: vec![ops[i].id, ops[i + 1].id, ops[i + 2].id],
                    estimated_speedup: 1.5,
                    priority: rule.priority,
                });
            }
        }
    }

    fn match_ffn(
        &self,
        ops: &[OpDescriptor],
        len: usize,
        rule: &FusionRule,
        out: &mut Vec<FusionOpportunity>,
    ) {
        if len < 3 {
            return;
        }
        for i in 0..len - 2 {
            let total = ops[i].element_count + ops[i + 1].element_count + ops[i + 2].element_count;
            if matches!(ops[i].op, OpType::Linear { .. })
                && matches!(ops[i + 1].op, OpType::Activation(_))
                && matches!(ops[i + 2].op, OpType::Linear { .. })
                && rule.is_applicable(total)
            {
                out.push(FusionOpportunity {
                    pattern: rule.pattern,
                    op_indices: vec![ops[i].id, ops[i + 1].id, ops[i + 2].id],
                    estimated_speedup: 1.4,
                    priority: rule.priority,
                });
            }
        }
    }

    fn match_pair(
        &self,
        ops: &[OpDescriptor],
        len: usize,
        rule: &FusionRule,
        pred_a: impl Fn(&OpType) -> bool,
        pred_b: impl Fn(&OpType) -> bool,
        out: &mut Vec<FusionOpportunity>,
    ) {
        if len < 2 {
            return;
        }
        for i in 0..len - 1 {
            let total = ops[i].element_count + ops[i + 1].element_count;
            if pred_a(&ops[i].op) && pred_b(&ops[i + 1].op) && rule.is_applicable(total) {
                out.push(FusionOpportunity {
                    pattern: rule.pattern,
                    op_indices: vec![ops[i].id, ops[i + 1].id],
                    estimated_speedup: 1.2,
                    priority: rule.priority,
                });
            }
        }
    }
}

impl Default for FusionPlanner {
    fn default() -> Self {
        Self::new()
    }
}

// ───────────────────────────────────────────────────────────────────
// FusedKernel
// ───────────────────────────────────────────────────────────────────

/// Represents a concrete fused kernel that can be dispatched.
#[derive(Debug, Clone)]
pub struct FusedKernel {
    /// The pattern this kernel implements.
    pub pattern: FusionPattern,
    /// CUDA kernel name (empty for CPU-only).
    pub kernel_name: String,
    /// Required shared memory (bytes).
    pub shared_mem_bytes: usize,
    /// Threads per block.
    pub threads_per_block: usize,
    /// Whether this kernel can run on CPU.
    pub has_cpu_fallback: bool,
}

impl FusedKernel {
    /// Create a new fused kernel descriptor.
    pub fn new(pattern: FusionPattern) -> Self {
        Self {
            pattern,
            kernel_name: format!("fused_{}", pattern.name().to_lowercase().replace('+', "_")),
            shared_mem_bytes: 0,
            threads_per_block: 256,
            has_cpu_fallback: true,
        }
    }

    /// Builder: set shared memory requirement.
    pub fn with_shared_mem(mut self, bytes: usize) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Builder: set threads per block.
    pub fn with_threads(mut self, n: usize) -> Self {
        self.threads_per_block = n;
        self
    }

    /// Check whether this kernel can run on the given hardware.
    pub fn fits_hardware(&self, hw: &HardwareConstraints) -> bool {
        hw.can_launch(self.shared_mem_bytes, self.threads_per_block)
    }
}

// ───────────────────────────────────────────────────────────────────
// FusionRegistry
// ───────────────────────────────────────────────────────────────────

/// Registry of available fused kernel implementations.
pub struct FusionRegistry {
    kernels: HashMap<FusionPattern, FusedKernel>,
}

impl FusionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { kernels: HashMap::new() }
    }

    /// Create a registry pre-populated with all built-in fused kernels.
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();
        reg.register(FusedKernel::new(FusionPattern::MatMulBiasActivation).with_shared_mem(1024));
        reg.register(FusedKernel::new(FusionPattern::LayerNormResidual).with_shared_mem(1024));
        reg.register(FusedKernel::new(FusionPattern::AttentionSoftmax).with_shared_mem(2048));
        reg.register(FusedKernel::new(FusionPattern::FeedForwardNetwork).with_shared_mem(1024));
        reg.register(FusedKernel::new(FusionPattern::EmbeddingLayerNorm).with_shared_mem(1024));
        reg.register(FusedKernel::new(FusionPattern::RmsNormLinear).with_shared_mem(1024));
        reg.register(FusedKernel::new(FusionPattern::ResidualRmsNorm).with_shared_mem(1024));
        reg
    }

    /// Register a fused kernel, replacing any previous registration for the
    /// same pattern.
    pub fn register(&mut self, kernel: FusedKernel) {
        self.kernels.insert(kernel.pattern, kernel);
    }

    /// Look up a fused kernel by pattern.
    pub fn get(&self, pattern: &FusionPattern) -> Option<&FusedKernel> {
        self.kernels.get(pattern)
    }

    /// All registered patterns.
    pub fn patterns(&self) -> Vec<FusionPattern> {
        self.kernels.keys().copied().collect()
    }

    /// Number of registered kernels.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }
}

impl Default for FusionRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

// ───────────────────────────────────────────────────────────────────
// FusionOptimizer
// ───────────────────────────────────────────────────────────────────

/// Selects the optimal non-overlapping set of fusion opportunities.
pub struct FusionOptimizer {
    hw: HardwareConstraints,
    registry: FusionRegistry,
}

impl FusionOptimizer {
    /// Create an optimizer with the given hardware constraints and registry.
    pub fn new(hw: HardwareConstraints, registry: FusionRegistry) -> Self {
        Self { hw, registry }
    }

    /// Create a CPU-only optimizer with the builtin registry.
    pub fn cpu_only() -> Self {
        Self::new(HardwareConstraints::cpu_only(), FusionRegistry::with_builtins())
    }

    /// Return a reference to the hardware constraints.
    pub fn hardware(&self) -> &HardwareConstraints {
        &self.hw
    }

    /// Return a reference to the registry.
    pub fn registry(&self) -> &FusionRegistry {
        &self.registry
    }

    /// Select the best non-overlapping fusion opportunities.
    ///
    /// Greedy algorithm: sort by priority (descending), then pick each
    /// opportunity whose op indices do not overlap with already-selected ones.
    pub fn select(&self, opportunities: &[FusionOpportunity]) -> Vec<FusionOpportunity> {
        let mut sorted: Vec<_> = opportunities
            .iter()
            .filter(|opp| {
                if let Some(k) = self.registry.get(&opp.pattern) {
                    k.fits_hardware(&self.hw)
                } else {
                    false
                }
            })
            .cloned()
            .collect();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));

        let mut used = std::collections::HashSet::new();
        let mut selected = Vec::new();
        for opp in sorted {
            if opp.op_indices.iter().all(|idx| !used.contains(idx)) {
                for &idx in &opp.op_indices {
                    used.insert(idx);
                }
                selected.push(opp);
            }
        }
        selected
    }
}

// ───────────────────────────────────────────────────────────────────
// FusionBenchmark
// ───────────────────────────────────────────────────────────────────

/// Result of benchmarking fused vs unfused execution.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Pattern that was benchmarked.
    pub pattern: FusionPattern,
    /// Unfused wall-clock time.
    pub unfused_duration: Duration,
    /// Fused wall-clock time.
    pub fused_duration: Duration,
    /// Speedup ratio (unfused / fused).
    pub speedup: f64,
}

/// Benchmarks fused vs unfused kernel execution.
pub struct FusionBenchmark {
    warmup_iters: usize,
    bench_iters: usize,
}

impl FusionBenchmark {
    /// Create a benchmark runner.
    pub fn new(warmup_iters: usize, bench_iters: usize) -> Self {
        Self { warmup_iters, bench_iters }
    }

    /// Benchmark a pair of closures (unfused, fused).
    pub fn run(
        &self,
        pattern: FusionPattern,
        mut unfused: impl FnMut(),
        mut fused: impl FnMut(),
    ) -> BenchmarkResult {
        // Warmup
        for _ in 0..self.warmup_iters {
            unfused();
            fused();
        }

        let start = Instant::now();
        for _ in 0..self.bench_iters {
            unfused();
        }
        let unfused_duration = start.elapsed();

        let start = Instant::now();
        for _ in 0..self.bench_iters {
            fused();
        }
        let fused_duration = start.elapsed();

        let speedup = if fused_duration.as_nanos() > 0 {
            unfused_duration.as_nanos() as f64 / fused_duration.as_nanos() as f64
        } else {
            f64::INFINITY
        };

        BenchmarkResult { pattern, unfused_duration, fused_duration, speedup }
    }
}

impl Default for FusionBenchmark {
    fn default() -> Self {
        Self::new(2, 10)
    }
}

// ───────────────────────────────────────────────────────────────────
// Fused kernel CPU fallback implementations
// ───────────────────────────────────────────────────────────────────

/// Fused MatMul + Bias + Activation (CPU fallback).
///
/// Computes `activation(A @ B + bias)` in a single pass over the output.
///
/// * `a`      — `[m × k]` row-major
/// * `b`      — `[k × n]` row-major
/// * `bias`   — `[n]` (may be empty for no bias)
/// * `output` — `[m × n]` (written)
/// * `act`    — activation function to apply
pub fn fuse_matmul_bias_activation(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    act: ActivationKind,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "dimensions must be non-zero".into() }.into()
        );
    }
    if a.len() < m * k {
        return Err(KernelError::InvalidArguments {
            reason: format!("a length {} < m*k={}", a.len(), m * k),
        }
        .into());
    }
    if b.len() < k * n {
        return Err(KernelError::InvalidArguments {
            reason: format!("b length {} < k*n={}", b.len(), k * n),
        }
        .into());
    }
    if !bias.is_empty() && bias.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("bias length {} < n={n}", bias.len()),
        }
        .into());
    }
    if output.len() < m * n {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < m*n={}", output.len(), m * n),
        }
        .into());
    }

    let activate = match act {
        ActivationKind::ReLU => relu as fn(f32) -> f32,
        ActivationKind::SiLU => silu,
        ActivationKind::GELU => gelu,
    };

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            if !bias.is_empty() {
                acc += bias[j];
            }
            output[i * n + j] = activate(acc);
        }
    }
    Ok(())
}

/// Fused LayerNorm + Residual Add (CPU fallback).
///
/// Computes `LayerNorm(input, gamma, beta) + residual`.
///
/// * `input`    — `[n]`
/// * `gamma`    — `[n]` scale
/// * `beta`     — `[n]` bias (may be empty)
/// * `residual` — `[n]`
/// * `output`   — `[n]` (written)
/// * `eps`      — normalisation epsilon
pub fn fuse_layernorm_residual(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    residual: &[f32],
    output: &mut [f32],
    eps: f32,
) -> Result<()> {
    let n = input.len();
    if n == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "input must be non-empty".into() }.into()
        );
    }
    if gamma.len() != n {
        return Err(KernelError::InvalidArguments {
            reason: format!("gamma length {} != n={n}", gamma.len()),
        }
        .into());
    }
    if !beta.is_empty() && beta.len() != n {
        return Err(KernelError::InvalidArguments {
            reason: format!("beta length {} != n={n}", beta.len()),
        }
        .into());
    }
    if residual.len() != n {
        return Err(KernelError::InvalidArguments {
            reason: format!("residual length {} != n={n}", residual.len()),
        }
        .into());
    }
    if output.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < n={n}", output.len()),
        }
        .into());
    }

    let mean: f32 = input.iter().sum::<f32>() / n as f32;
    let var: f32 = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();

    for i in 0..n {
        let normed = (input[i] - mean) * inv_std * gamma[i];
        let normed = if !beta.is_empty() { normed + beta[i] } else { normed };
        output[i] = normed + residual[i];
    }
    Ok(())
}

/// Fused Attention QK^T + Mask + Softmax (CPU fallback).
///
/// * `scores` — `[seq_len]` pre-computed attention scores (one row)
/// * `mask`   — `[seq_len]` additive mask (0 = keep, large negative = mask)
/// * `output` — `[seq_len]` (written)
/// * `scale`  — scalar multiplier (typically 1/sqrt(d_k))
pub fn fuse_attention_softmax(
    scores: &[f32],
    mask: &[f32],
    output: &mut [f32],
    scale: f32,
) -> Result<()> {
    let n = scores.len();
    if n == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "scores must be non-empty".into() }.into()
        );
    }
    if mask.len() != n {
        return Err(KernelError::InvalidArguments {
            reason: format!("mask length {} != n={n}", mask.len()),
        }
        .into());
    }
    if output.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < n={n}", output.len()),
        }
        .into());
    }

    // Scale + mask + max
    let mut max_val = f32::NEG_INFINITY;
    for (&s, &m) in scores.iter().zip(mask) {
        let v = s * scale + m;
        if v > max_val {
            max_val = v;
        }
    }

    // Exp + sum
    let mut sum = 0.0f32;
    for ((&s, &m), o) in scores.iter().zip(mask).zip(output.iter_mut()) {
        let v = (s * scale + m - max_val).exp();
        *o = v;
        sum += v;
    }

    // Normalise
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for o in output[..n].iter_mut() {
            *o *= inv;
        }
    }
    Ok(())
}

/// Fused FFN: Linear → Activation → Linear (CPU fallback).
///
/// * `input`   — `[in_dim]`
/// * `w1`      — `[hidden × in_dim]` first linear weight
/// * `b1`      — `[hidden]` first linear bias (may be empty)
/// * `w2`      — `[out_dim × hidden]` second linear weight
/// * `b2`      — `[out_dim]` second linear bias (may be empty)
/// * `output`  — `[out_dim]` (written)
/// * `act`     — activation between the two linears
#[allow(clippy::too_many_arguments)]
pub fn fuse_ffn(
    input: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    output: &mut [f32],
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
    act: ActivationKind,
) -> Result<()> {
    if in_dim == 0 || hidden_dim == 0 || out_dim == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "dimensions must be non-zero".into() }.into()
        );
    }
    if input.len() < in_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!("input length {} < in_dim={in_dim}", input.len()),
        }
        .into());
    }
    if w1.len() < hidden_dim * in_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!("w1 length {} < hidden*in={}", w1.len(), hidden_dim * in_dim),
        }
        .into());
    }
    if !b1.is_empty() && b1.len() < hidden_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!("b1 length {} < hidden={hidden_dim}", b1.len()),
        }
        .into());
    }
    if w2.len() < out_dim * hidden_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!("w2 length {} < out*hidden={}", w2.len(), out_dim * hidden_dim),
        }
        .into());
    }
    if !b2.is_empty() && b2.len() < out_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!("b2 length {} < out_dim={out_dim}", b2.len()),
        }
        .into());
    }
    if output.len() < out_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < out_dim={out_dim}", output.len()),
        }
        .into());
    }

    let activate = match act {
        ActivationKind::ReLU => relu as fn(f32) -> f32,
        ActivationKind::SiLU => silu,
        ActivationKind::GELU => gelu,
    };

    // First linear + activation
    let mut hidden = vec![0.0f32; hidden_dim];
    for (i, row) in w1.chunks_exact(in_dim).enumerate().take(hidden_dim) {
        let mut acc = 0.0f32;
        for (&w, &x) in row.iter().zip(input) {
            acc += w * x;
        }
        if !b1.is_empty() {
            acc += b1[i];
        }
        hidden[i] = activate(acc);
    }

    // Second linear
    for (i, row) in w2.chunks_exact(hidden_dim).enumerate().take(out_dim) {
        let mut acc = 0.0f32;
        for (&w, &x) in row.iter().zip(hidden.iter()) {
            acc += w * x;
        }
        if !b2.is_empty() {
            acc += b2[i];
        }
        output[i] = acc;
    }
    Ok(())
}

/// Fused Embedding Lookup + LayerNorm (CPU fallback).
///
/// * `embeddings` — `[vocab_size × embed_dim]` embedding table
/// * `token_ids`  — `[seq_len]` token indices
/// * `gamma`      — `[embed_dim]` LayerNorm scale
/// * `beta`       — `[embed_dim]` LayerNorm bias (may be empty)
/// * `output`     — `[seq_len × embed_dim]` (written)
/// * `eps`        — normalisation epsilon
pub fn fuse_embedding_layernorm(
    embeddings: &[f32],
    token_ids: &[usize],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    vocab_size: usize,
    embed_dim: usize,
    eps: f32,
) -> Result<()> {
    let seq_len = token_ids.len();
    if seq_len == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "token_ids must be non-empty".into() }.into()
        );
    }
    if embed_dim == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "embed_dim must be non-zero".into() }.into()
        );
    }
    if embeddings.len() < vocab_size * embed_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "embeddings length {} < vocab*dim={}",
                embeddings.len(),
                vocab_size * embed_dim
            ),
        }
        .into());
    }
    if gamma.len() != embed_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!("gamma length {} != embed_dim={embed_dim}", gamma.len()),
        }
        .into());
    }
    if !beta.is_empty() && beta.len() != embed_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!("beta length {} != embed_dim={embed_dim}", beta.len()),
        }
        .into());
    }
    if output.len() < seq_len * embed_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < seq_len*dim={}", output.len(), seq_len * embed_dim),
        }
        .into());
    }
    for &tid in token_ids {
        if tid >= vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!("token id {tid} >= vocab_size={vocab_size}"),
            }
            .into());
        }
    }

    for (s, &tid) in token_ids.iter().enumerate() {
        let emb = &embeddings[tid * embed_dim..(tid + 1) * embed_dim];
        let out_row = &mut output[s * embed_dim..(s + 1) * embed_dim];

        // LayerNorm on the looked-up embedding
        let mean: f32 = emb.iter().sum::<f32>() / embed_dim as f32;
        let var: f32 = emb.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / embed_dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..embed_dim {
            let normed = (emb[i] - mean) * inv_std * gamma[i];
            out_row[i] = if !beta.is_empty() { normed + beta[i] } else { normed };
        }
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// CUDA launch stubs (scaffold)
// ───────────────────────────────────────────────────────────────────

/// Launch fused MatMul+Bias+Activation on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_matmul_bias_activation(
    _a: &[f32],
    _b: &[f32],
    _bias: &[f32],
    _output: &mut [f32],
    _m: usize,
    _n: usize,
    _k: usize,
    _act: ActivationKind,
) -> Result<()> {
    Err(KernelError::GpuError {
        reason: "fused MatMul+Bias+Activation CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch fused LayerNorm+Residual on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_layernorm_residual(
    _input: &[f32],
    _gamma: &[f32],
    _beta: &[f32],
    _residual: &[f32],
    _output: &mut [f32],
    _eps: f32,
) -> Result<()> {
    Err(KernelError::GpuError {
        reason: "fused LayerNorm+Residual CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch fused Attention+Softmax on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_attention_softmax(
    _scores: &[f32],
    _mask: &[f32],
    _output: &mut [f32],
    _scale: f32,
) -> Result<()> {
    Err(KernelError::GpuError {
        reason: "fused Attention+Softmax CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch fused FFN on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_ffn(
    _input: &[f32],
    _w1: &[f32],
    _b1: &[f32],
    _w2: &[f32],
    _b2: &[f32],
    _output: &mut [f32],
    _in_dim: usize,
    _hidden_dim: usize,
    _out_dim: usize,
    _act: ActivationKind,
) -> Result<()> {
    Err(KernelError::GpuError {
        reason: "fused FFN CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch fused Embedding+LayerNorm on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_embedding_layernorm(
    _embeddings: &[f32],
    _token_ids: &[usize],
    _gamma: &[f32],
    _beta: &[f32],
    _output: &mut [f32],
    _vocab_size: usize,
    _embed_dim: usize,
    _eps: f32,
) -> Result<()> {
    Err(KernelError::GpuError {
        reason: "fused Embedding+LayerNorm CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ───────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-4;

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    // ── FusionPattern tests ──────────────────────────────────────

    #[test]
    fn pattern_op_count() {
        assert_eq!(FusionPattern::MatMulBiasActivation.op_count(), 3);
        assert_eq!(FusionPattern::LayerNormResidual.op_count(), 2);
        assert_eq!(FusionPattern::AttentionSoftmax.op_count(), 3);
        assert_eq!(FusionPattern::FeedForwardNetwork.op_count(), 3);
        assert_eq!(FusionPattern::EmbeddingLayerNorm.op_count(), 2);
        assert_eq!(FusionPattern::RmsNormLinear.op_count(), 2);
        assert_eq!(FusionPattern::ResidualRmsNorm.op_count(), 2);
    }

    #[test]
    fn pattern_saved_launches() {
        assert_eq!(FusionPattern::MatMulBiasActivation.saved_launches(), 2);
        assert_eq!(FusionPattern::LayerNormResidual.saved_launches(), 1);
        assert_eq!(FusionPattern::AttentionSoftmax.saved_launches(), 2);
    }

    #[test]
    fn pattern_names_non_empty() {
        let patterns = [
            FusionPattern::MatMulBiasActivation,
            FusionPattern::LayerNormResidual,
            FusionPattern::AttentionSoftmax,
            FusionPattern::FeedForwardNetwork,
            FusionPattern::EmbeddingLayerNorm,
            FusionPattern::RmsNormLinear,
            FusionPattern::ResidualRmsNorm,
        ];
        for p in &patterns {
            assert!(!p.name().is_empty());
            assert!(!p.to_string().is_empty());
        }
    }

    #[test]
    fn pattern_display() {
        assert_eq!(FusionPattern::MatMulBiasActivation.to_string(), "MatMul+Bias+Activation");
        assert_eq!(FusionPattern::LayerNormResidual.to_string(), "LayerNorm+Residual");
    }

    // ── FusionRule tests ─────────────────────────────────────────

    #[test]
    fn rule_default_is_enabled() {
        let rule = FusionRule::new(FusionPattern::MatMulBiasActivation);
        assert!(rule.enabled);
        assert!(rule.is_applicable(100));
    }

    #[test]
    fn rule_min_elements_check() {
        let rule = FusionRule::new(FusionPattern::MatMulBiasActivation).with_min_elements(64);
        assert!(!rule.is_applicable(32));
        assert!(rule.is_applicable(64));
        assert!(rule.is_applicable(128));
    }

    #[test]
    fn rule_max_elements_check() {
        let rule = FusionRule::new(FusionPattern::LayerNormResidual).with_max_elements(1024);
        assert!(rule.is_applicable(512));
        assert!(rule.is_applicable(1024));
        assert!(!rule.is_applicable(1025));
    }

    #[test]
    fn rule_disabled() {
        let rule = FusionRule::new(FusionPattern::LayerNormResidual).with_enabled(false);
        assert!(!rule.is_applicable(100));
    }

    #[test]
    fn rule_priority_builder() {
        let rule = FusionRule::new(FusionPattern::MatMulBiasActivation).with_priority(10);
        assert_eq!(rule.priority, 10);
    }

    // ── OpDescriptor / OpType tests ──────────────────────────────

    #[test]
    fn op_type_display() {
        let op = OpType::MatMul { m: 4, n: 8, k: 16 };
        assert!(op.to_string().contains("MatMul"));
        let op = OpType::Activation(ActivationKind::ReLU);
        assert!(op.to_string().contains("ReLU"));
    }

    #[test]
    fn activation_kind_display() {
        assert_eq!(ActivationKind::ReLU.to_string(), "ReLU");
        assert_eq!(ActivationKind::SiLU.to_string(), "SiLU");
        assert_eq!(ActivationKind::GELU.to_string(), "GELU");
    }

    // ── FusionPlanner tests ──────────────────────────────────────

    fn make_matmul_bias_act_seq() -> Vec<OpDescriptor> {
        vec![
            OpDescriptor { id: 0, op: OpType::MatMul { m: 4, n: 8, k: 16 }, element_count: 512 },
            OpDescriptor { id: 1, op: OpType::BiasAdd { len: 32 }, element_count: 32 },
            OpDescriptor { id: 2, op: OpType::Activation(ActivationKind::ReLU), element_count: 32 },
        ]
    }

    #[test]
    fn planner_finds_matmul_bias_act() {
        let planner = FusionPlanner::new();
        let ops = make_matmul_bias_act_seq();
        let opps = planner.find_opportunities(&ops);
        assert!(opps.iter().any(|o| o.pattern == FusionPattern::MatMulBiasActivation));
    }

    #[test]
    fn planner_finds_layernorm_residual() {
        let planner = FusionPlanner::new();
        let ops = vec![
            OpDescriptor { id: 0, op: OpType::LayerNorm { len: 64 }, element_count: 64 },
            OpDescriptor { id: 1, op: OpType::ResidualAdd { len: 64 }, element_count: 64 },
        ];
        let opps = planner.find_opportunities(&ops);
        assert!(opps.iter().any(|o| o.pattern == FusionPattern::LayerNormResidual));
    }

    #[test]
    fn planner_finds_attention_softmax() {
        let planner = FusionPlanner::new();
        let ops = vec![
            OpDescriptor { id: 0, op: OpType::MatMul { m: 1, n: 64, k: 64 }, element_count: 64 },
            OpDescriptor { id: 1, op: OpType::AttentionMask { len: 64 }, element_count: 64 },
            OpDescriptor { id: 2, op: OpType::Softmax { len: 64 }, element_count: 64 },
        ];
        let opps = planner.find_opportunities(&ops);
        assert!(opps.iter().any(|o| o.pattern == FusionPattern::AttentionSoftmax));
    }

    #[test]
    fn planner_finds_ffn() {
        let planner = FusionPlanner::new();
        let ops = vec![
            OpDescriptor {
                id: 0,
                op: OpType::Linear { in_features: 64, out_features: 256 },
                element_count: 256,
            },
            OpDescriptor {
                id: 1,
                op: OpType::Activation(ActivationKind::GELU),
                element_count: 256,
            },
            OpDescriptor {
                id: 2,
                op: OpType::Linear { in_features: 256, out_features: 64 },
                element_count: 64,
            },
        ];
        let opps = planner.find_opportunities(&ops);
        assert!(opps.iter().any(|o| o.pattern == FusionPattern::FeedForwardNetwork));
    }

    #[test]
    fn planner_finds_embedding_layernorm() {
        let planner = FusionPlanner::new();
        let ops = vec![
            OpDescriptor {
                id: 0,
                op: OpType::EmbeddingLookup { vocab_size: 1000, embed_dim: 64 },
                element_count: 64,
            },
            OpDescriptor { id: 1, op: OpType::LayerNorm { len: 64 }, element_count: 64 },
        ];
        let opps = planner.find_opportunities(&ops);
        assert!(opps.iter().any(|o| o.pattern == FusionPattern::EmbeddingLayerNorm));
    }

    #[test]
    fn planner_finds_rmsnorm_linear() {
        let planner = FusionPlanner::new();
        let ops = vec![
            OpDescriptor { id: 0, op: OpType::RmsNorm { len: 64 }, element_count: 64 },
            OpDescriptor {
                id: 1,
                op: OpType::Linear { in_features: 64, out_features: 128 },
                element_count: 128,
            },
        ];
        let opps = planner.find_opportunities(&ops);
        assert!(opps.iter().any(|o| o.pattern == FusionPattern::RmsNormLinear));
    }

    #[test]
    fn planner_finds_residual_rmsnorm() {
        let planner = FusionPlanner::new();
        let ops = vec![
            OpDescriptor { id: 0, op: OpType::ResidualAdd { len: 64 }, element_count: 64 },
            OpDescriptor { id: 1, op: OpType::RmsNorm { len: 64 }, element_count: 64 },
        ];
        let opps = planner.find_opportunities(&ops);
        assert!(opps.iter().any(|o| o.pattern == FusionPattern::ResidualRmsNorm));
    }

    #[test]
    fn planner_no_match_on_wrong_order() {
        let planner = FusionPlanner::new();
        let ops = vec![
            OpDescriptor { id: 0, op: OpType::Activation(ActivationKind::ReLU), element_count: 32 },
            OpDescriptor { id: 1, op: OpType::MatMul { m: 4, n: 8, k: 16 }, element_count: 512 },
            OpDescriptor { id: 2, op: OpType::BiasAdd { len: 32 }, element_count: 32 },
        ];
        let opps = planner.find_opportunities(&ops);
        assert!(!opps.iter().any(|o| o.pattern == FusionPattern::MatMulBiasActivation));
    }

    #[test]
    fn planner_empty_sequence() {
        let planner = FusionPlanner::new();
        let opps = planner.find_opportunities(&[]);
        assert!(opps.is_empty());
    }

    #[test]
    fn planner_single_op_no_match() {
        let planner = FusionPlanner::new();
        let ops = vec![OpDescriptor {
            id: 0,
            op: OpType::MatMul { m: 4, n: 8, k: 16 },
            element_count: 512,
        }];
        let opps = planner.find_opportunities(&ops);
        assert!(opps.is_empty());
    }

    #[test]
    fn planner_respects_min_elements() {
        let rules = vec![FusionRule::new(FusionPattern::LayerNormResidual).with_min_elements(1000)];
        let planner = FusionPlanner::with_rules(rules);
        let ops = vec![
            OpDescriptor { id: 0, op: OpType::LayerNorm { len: 64 }, element_count: 64 },
            OpDescriptor { id: 1, op: OpType::ResidualAdd { len: 64 }, element_count: 64 },
        ];
        let opps = planner.find_opportunities(&ops);
        assert!(opps.is_empty());
    }

    #[test]
    fn planner_disabled_rule_skipped() {
        let rules = vec![FusionRule::new(FusionPattern::LayerNormResidual).with_enabled(false)];
        let planner = FusionPlanner::with_rules(rules);
        let ops = vec![
            OpDescriptor { id: 0, op: OpType::LayerNorm { len: 64 }, element_count: 64 },
            OpDescriptor { id: 1, op: OpType::ResidualAdd { len: 64 }, element_count: 64 },
        ];
        let opps = planner.find_opportunities(&ops);
        assert!(opps.is_empty());
    }

    #[test]
    fn planner_default_has_all_rules() {
        let planner = FusionPlanner::new();
        assert_eq!(planner.rules().len(), 7);
    }

    #[test]
    fn planner_multiple_opportunities_in_long_sequence() {
        let planner = FusionPlanner::new();
        let ops = vec![
            OpDescriptor { id: 0, op: OpType::RmsNorm { len: 64 }, element_count: 64 },
            OpDescriptor {
                id: 1,
                op: OpType::Linear { in_features: 64, out_features: 128 },
                element_count: 128,
            },
            OpDescriptor { id: 2, op: OpType::LayerNorm { len: 128 }, element_count: 128 },
            OpDescriptor { id: 3, op: OpType::ResidualAdd { len: 128 }, element_count: 128 },
        ];
        let opps = planner.find_opportunities(&ops);
        assert!(opps.len() >= 2);
    }

    // ── HardwareConstraints tests ────────────────────────────────

    #[test]
    fn hw_cpu_only_no_gpu() {
        let hw = HardwareConstraints::cpu_only();
        assert!(!hw.gpu_available);
    }

    #[test]
    fn hw_gpu_available() {
        let hw = HardwareConstraints::gpu(80, 48 * 1024);
        assert!(hw.gpu_available);
        assert_eq!(hw.compute_capability, 80);
    }

    #[test]
    fn hw_can_launch_within_limits() {
        let hw = HardwareConstraints::gpu(80, 48 * 1024);
        assert!(hw.can_launch(1024, 256));
        assert!(hw.can_launch(48 * 1024, 1024));
    }

    #[test]
    fn hw_cannot_launch_exceeding_shared_mem() {
        let hw = HardwareConstraints::gpu(80, 48 * 1024);
        assert!(!hw.can_launch(48 * 1024 + 1, 256));
    }

    #[test]
    fn hw_cannot_launch_exceeding_threads() {
        let hw = HardwareConstraints::gpu(80, 48 * 1024);
        assert!(!hw.can_launch(1024, 2048));
    }

    // ── FusedKernel tests ────────────────────────────────────────

    #[test]
    fn fused_kernel_new_defaults() {
        let k = FusedKernel::new(FusionPattern::MatMulBiasActivation);
        assert_eq!(k.pattern, FusionPattern::MatMulBiasActivation);
        assert!(k.has_cpu_fallback);
        assert_eq!(k.threads_per_block, 256);
    }

    #[test]
    fn fused_kernel_with_shared_mem() {
        let k = FusedKernel::new(FusionPattern::LayerNormResidual).with_shared_mem(2048);
        assert_eq!(k.shared_mem_bytes, 2048);
    }

    #[test]
    fn fused_kernel_with_threads() {
        let k = FusedKernel::new(FusionPattern::AttentionSoftmax).with_threads(512);
        assert_eq!(k.threads_per_block, 512);
    }

    #[test]
    fn fused_kernel_fits_hardware() {
        let k = FusedKernel::new(FusionPattern::MatMulBiasActivation).with_shared_mem(1024);
        let hw = HardwareConstraints::gpu(80, 48 * 1024);
        assert!(k.fits_hardware(&hw));
    }

    #[test]
    fn fused_kernel_exceeds_hardware() {
        let k = FusedKernel::new(FusionPattern::MatMulBiasActivation).with_shared_mem(128 * 1024);
        let hw = HardwareConstraints::gpu(80, 48 * 1024);
        assert!(!k.fits_hardware(&hw));
    }

    // ── FusionRegistry tests ─────────────────────────────────────

    #[test]
    fn registry_empty() {
        let reg = FusionRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn registry_builtins_populated() {
        let reg = FusionRegistry::with_builtins();
        assert_eq!(reg.len(), 7);
        assert!(!reg.is_empty());
    }

    #[test]
    fn registry_get_existing() {
        let reg = FusionRegistry::with_builtins();
        assert!(reg.get(&FusionPattern::MatMulBiasActivation).is_some());
        assert!(reg.get(&FusionPattern::LayerNormResidual).is_some());
    }

    #[test]
    fn registry_register_custom() {
        let mut reg = FusionRegistry::new();
        reg.register(FusedKernel::new(FusionPattern::MatMulBiasActivation));
        assert_eq!(reg.len(), 1);
        assert!(reg.get(&FusionPattern::MatMulBiasActivation).is_some());
    }

    #[test]
    fn registry_overwrite_existing() {
        let mut reg = FusionRegistry::with_builtins();
        let before = reg.len();
        reg.register(FusedKernel::new(FusionPattern::MatMulBiasActivation).with_shared_mem(9999));
        assert_eq!(reg.len(), before);
        assert_eq!(reg.get(&FusionPattern::MatMulBiasActivation).unwrap().shared_mem_bytes, 9999);
    }

    #[test]
    fn registry_patterns_returns_all() {
        let reg = FusionRegistry::with_builtins();
        let patterns = reg.patterns();
        assert_eq!(patterns.len(), 7);
    }

    // ── FusionOptimizer tests ────────────────────────────────────

    #[test]
    fn optimizer_selects_non_overlapping() {
        let opt = FusionOptimizer::cpu_only();
        let opps = vec![
            FusionOpportunity {
                pattern: FusionPattern::MatMulBiasActivation,
                op_indices: vec![0, 1, 2],
                estimated_speedup: 1.3,
                priority: 2,
            },
            FusionOpportunity {
                pattern: FusionPattern::LayerNormResidual,
                op_indices: vec![3, 4],
                estimated_speedup: 1.2,
                priority: 1,
            },
        ];
        let selected = opt.select(&opps);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn optimizer_rejects_overlapping() {
        let opt = FusionOptimizer::cpu_only();
        let opps = vec![
            FusionOpportunity {
                pattern: FusionPattern::MatMulBiasActivation,
                op_indices: vec![0, 1, 2],
                estimated_speedup: 1.3,
                priority: 2,
            },
            FusionOpportunity {
                pattern: FusionPattern::LayerNormResidual,
                op_indices: vec![1, 3],
                estimated_speedup: 1.2,
                priority: 1,
            },
        ];
        let selected = opt.select(&opps);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].pattern, FusionPattern::MatMulBiasActivation);
    }

    #[test]
    fn optimizer_prefers_higher_priority() {
        let opt = FusionOptimizer::cpu_only();
        let opps = vec![
            FusionOpportunity {
                pattern: FusionPattern::LayerNormResidual,
                op_indices: vec![0, 1],
                estimated_speedup: 1.2,
                priority: 1,
            },
            FusionOpportunity {
                pattern: FusionPattern::MatMulBiasActivation,
                op_indices: vec![0, 1, 2],
                estimated_speedup: 1.3,
                priority: 5,
            },
        ];
        let selected = opt.select(&opps);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].pattern, FusionPattern::MatMulBiasActivation);
    }

    #[test]
    fn optimizer_empty_input() {
        let opt = FusionOptimizer::cpu_only();
        let selected = opt.select(&[]);
        assert!(selected.is_empty());
    }

    #[test]
    fn optimizer_filters_unregistered_patterns() {
        let opt = FusionOptimizer::new(HardwareConstraints::cpu_only(), FusionRegistry::new());
        let opps = vec![FusionOpportunity {
            pattern: FusionPattern::MatMulBiasActivation,
            op_indices: vec![0, 1, 2],
            estimated_speedup: 1.3,
            priority: 2,
        }];
        let selected = opt.select(&opps);
        assert!(selected.is_empty());
    }

    #[test]
    fn optimizer_hardware_ref() {
        let opt = FusionOptimizer::cpu_only();
        assert!(!opt.hardware().gpu_available);
    }

    #[test]
    fn optimizer_registry_ref() {
        let opt = FusionOptimizer::cpu_only();
        assert_eq!(opt.registry().len(), 7);
    }

    // ── FusionBenchmark tests ────────────────────────────────────

    #[test]
    fn benchmark_runs() {
        let bench = FusionBenchmark::new(1, 5);
        let mut x = 0u32;
        let mut y = 0u32;
        let result = bench.run(
            FusionPattern::MatMulBiasActivation,
            || x = x.wrapping_add(1),
            || y = y.wrapping_add(1),
        );
        assert_eq!(result.pattern, FusionPattern::MatMulBiasActivation);
        assert!(result.speedup > 0.0);
        assert!(x > 0);
        assert!(y > 0);
    }

    #[test]
    fn benchmark_default() {
        let bench = FusionBenchmark::default();
        assert_eq!(bench.warmup_iters, 2);
        assert_eq!(bench.bench_iters, 10);
    }

    // ── fuse_matmul_bias_activation tests ────────────────────────

    #[test]
    fn matmul_bias_relu_identity_weight() {
        // Identity-like: 1x1x1 matmul
        let a = [2.0f32];
        let b = [3.0f32];
        let bias = [1.0f32];
        let mut out = [0.0f32];
        fuse_matmul_bias_activation(&a, &b, &bias, &mut out, 1, 1, 1, ActivationKind::ReLU)
            .unwrap();
        // 2*3 + 1 = 7, relu(7) = 7
        assert!((out[0] - 7.0).abs() < TOL);
    }

    #[test]
    fn matmul_bias_relu_negative() {
        let a = [-2.0f32];
        let b = [3.0f32];
        let bias = [1.0f32];
        let mut out = [0.0f32];
        fuse_matmul_bias_activation(&a, &b, &bias, &mut out, 1, 1, 1, ActivationKind::ReLU)
            .unwrap();
        // -2*3 + 1 = -5, relu(-5) = 0
        assert!((out[0]).abs() < TOL);
    }

    #[test]
    fn matmul_bias_silu() {
        let a = [1.0f32];
        let b = [1.0f32];
        let bias = [0.0f32];
        let mut out = [0.0f32];
        fuse_matmul_bias_activation(&a, &b, &bias, &mut out, 1, 1, 1, ActivationKind::SiLU)
            .unwrap();
        let expected = silu(1.0);
        assert!((out[0] - expected).abs() < TOL);
    }

    #[test]
    fn matmul_bias_gelu() {
        let a = [1.0f32];
        let b = [1.0f32];
        let bias = [0.5f32];
        let mut out = [0.0f32];
        fuse_matmul_bias_activation(&a, &b, &bias, &mut out, 1, 1, 1, ActivationKind::GELU)
            .unwrap();
        let expected = gelu(1.5);
        assert!((out[0] - expected).abs() < TOL);
    }

    #[test]
    fn matmul_bias_act_no_bias() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = [5.0, 6.0, 7.0, 8.0]; // 2x2
        let bias: &[f32] = &[];
        let mut out = [0.0f32; 4];
        fuse_matmul_bias_activation(&a, &b, bias, &mut out, 2, 2, 2, ActivationKind::ReLU).unwrap();
        // Row 0: [1*5+2*7, 1*6+2*8] = [19, 22]
        // Row 1: [3*5+4*7, 3*6+4*8] = [43, 50]
        assert!((out[0] - 19.0).abs() < TOL);
        assert!((out[1] - 22.0).abs() < TOL);
        assert!((out[2] - 43.0).abs() < TOL);
        assert!((out[3] - 50.0).abs() < TOL);
    }

    #[test]
    fn matmul_bias_act_zero_dim_errors() {
        let mut out = [0.0f32];
        assert!(
            fuse_matmul_bias_activation(&[], &[], &[], &mut out, 0, 1, 1, ActivationKind::ReLU)
                .is_err()
        );
    }

    #[test]
    fn matmul_bias_act_a_too_short() {
        let mut out = [0.0f32; 4];
        assert!(
            fuse_matmul_bias_activation(
                &[1.0],
                &[1.0; 4],
                &[],
                &mut out,
                2,
                2,
                2,
                ActivationKind::ReLU
            )
            .is_err()
        );
    }

    #[test]
    fn matmul_bias_act_output_too_short() {
        let mut out = [0.0f32; 1];
        assert!(
            fuse_matmul_bias_activation(
                &[1.0; 4],
                &[1.0; 4],
                &[],
                &mut out,
                2,
                2,
                2,
                ActivationKind::ReLU
            )
            .is_err()
        );
    }

    // ── fuse_layernorm_residual tests ────────────────────────────

    #[test]
    fn layernorm_residual_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta: &[f32] = &[];
        let residual = [0.1, 0.2, 0.3, 0.4];
        let mut out = [0.0f32; 4];
        fuse_layernorm_residual(&input, &gamma, beta, &residual, &mut out, EPS).unwrap();
        // After LN, values should be normalised, then residual added
        let sum: f32 = out.iter().sum();
        // Sum of normed + residual = sum_residual + sum_normed
        // Sum of normed ≈ 0 (mean-subtracted), so sum ≈ 0 + 1.0 = 1.0
        assert!((sum - 1.0).abs() < 0.1);
    }

    #[test]
    fn layernorm_residual_with_beta() {
        let input = [1.0; 4];
        let gamma = [1.0; 4];
        let beta = [0.5; 4];
        let residual = [0.0; 4];
        let mut out = [0.0f32; 4];
        fuse_layernorm_residual(&input, &gamma, &beta, &residual, &mut out, EPS).unwrap();
        // Input is constant → variance = 0 → norm = 0 → output = beta
        for &v in &out {
            assert!((v - 0.5).abs() < TOL);
        }
    }

    #[test]
    fn layernorm_residual_empty_errors() {
        let mut out = [0.0f32; 4];
        assert!(fuse_layernorm_residual(&[], &[], &[], &[], &mut out, EPS).is_err());
    }

    #[test]
    fn layernorm_residual_gamma_mismatch() {
        let mut out = [0.0f32; 4];
        assert!(
            fuse_layernorm_residual(&[1.0; 4], &[1.0; 3], &[], &[0.0; 4], &mut out, EPS).is_err()
        );
    }

    #[test]
    fn layernorm_residual_residual_mismatch() {
        let mut out = [0.0f32; 4];
        assert!(
            fuse_layernorm_residual(&[1.0; 4], &[1.0; 4], &[], &[0.0; 3], &mut out, EPS).is_err()
        );
    }

    #[test]
    fn layernorm_residual_output_too_short() {
        let mut out = [0.0f32; 2];
        assert!(
            fuse_layernorm_residual(&[1.0; 4], &[1.0; 4], &[], &[0.0; 4], &mut out, EPS).is_err()
        );
    }

    // ── fuse_attention_softmax tests ─────────────────────────────

    #[test]
    fn attention_softmax_uniform() {
        let scores = [1.0, 1.0, 1.0, 1.0];
        let mask = [0.0; 4];
        let mut out = [0.0f32; 4];
        fuse_attention_softmax(&scores, &mask, &mut out, 1.0).unwrap();
        for &v in &out {
            assert!((v - 0.25).abs() < TOL);
        }
    }

    #[test]
    fn attention_softmax_with_mask() {
        let scores = [1.0, 1.0, 1.0, 1.0];
        let mask = [0.0, 0.0, -1e9, -1e9];
        let mut out = [0.0f32; 4];
        fuse_attention_softmax(&scores, &mask, &mut out, 1.0).unwrap();
        // Masked positions should be ~0
        assert!(out[2] < 1e-6);
        assert!(out[3] < 1e-6);
        assert!((out[0] + out[1] - 1.0).abs() < TOL);
    }

    #[test]
    fn attention_softmax_with_scale() {
        let scores = [2.0, 1.0];
        let mask = [0.0, 0.0];
        let mut out = [0.0f32; 2];
        fuse_attention_softmax(&scores, &mask, &mut out, 0.5).unwrap();
        // scaled: [1.0, 0.5]
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
        assert!(out[0] > out[1]);
    }

    #[test]
    fn attention_softmax_normalisation() {
        let scores = [3.0, 1.0, 0.5, 2.0];
        let mask = [0.0; 4];
        let mut out = [0.0f32; 4];
        fuse_attention_softmax(&scores, &mask, &mut out, 1.0).unwrap();
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
    }

    #[test]
    fn attention_softmax_empty_errors() {
        let mut out = [0.0f32; 4];
        assert!(fuse_attention_softmax(&[], &[], &mut out, 1.0).is_err());
    }

    #[test]
    fn attention_softmax_mask_mismatch() {
        let mut out = [0.0f32; 4];
        assert!(fuse_attention_softmax(&[1.0; 4], &[0.0; 3], &mut out, 1.0).is_err());
    }

    #[test]
    fn attention_softmax_output_too_short() {
        let mut out = [0.0f32; 2];
        assert!(fuse_attention_softmax(&[1.0; 4], &[0.0; 4], &mut out, 1.0).is_err());
    }

    // ── fuse_ffn tests ───────────────────────────────────────────

    #[test]
    fn ffn_simple_relu() {
        // in=2, hidden=2, out=1
        let input = [1.0, 2.0];
        let w1 = [1.0, 0.0, 0.0, 1.0]; // identity 2x2
        let b1: &[f32] = &[];
        let w2 = [1.0, 1.0]; // 1x2
        let b2: &[f32] = &[];
        let mut out = [0.0f32; 1];
        fuse_ffn(&input, &w1, b1, &w2, b2, &mut out, 2, 2, 1, ActivationKind::ReLU).unwrap();
        // hidden = relu([1,2]) = [1,2], out = 1*1 + 1*2 = 3
        assert!((out[0] - 3.0).abs() < TOL);
    }

    #[test]
    fn ffn_with_biases() {
        let input = [1.0];
        let w1 = [2.0]; // 1x1
        let b1 = [0.5];
        let w2 = [1.0]; // 1x1
        let b2 = [0.1];
        let mut out = [0.0f32; 1];
        fuse_ffn(&input, &w1, &b1, &w2, &b2, &mut out, 1, 1, 1, ActivationKind::ReLU).unwrap();
        // hidden = relu(2*1 + 0.5) = 2.5, out = 1*2.5 + 0.1 = 2.6
        assert!((out[0] - 2.6).abs() < TOL);
    }

    #[test]
    fn ffn_zero_dim_errors() {
        let mut out = [0.0f32; 1];
        assert!(
            fuse_ffn(&[], &[], &[], &[], &[], &mut out, 0, 1, 1, ActivationKind::ReLU).is_err()
        );
    }

    #[test]
    fn ffn_input_too_short() {
        let mut out = [0.0f32; 1];
        assert!(
            fuse_ffn(
                &[1.0],
                &[1.0; 4],
                &[],
                &[1.0; 2],
                &[],
                &mut out,
                2,
                2,
                1,
                ActivationKind::ReLU
            )
            .is_err()
        );
    }

    #[test]
    fn ffn_output_too_short() {
        let mut out = [0.0f32; 0];
        assert!(
            fuse_ffn(
                &[1.0; 2],
                &[1.0; 4],
                &[],
                &[1.0; 2],
                &[],
                &mut out,
                2,
                2,
                1,
                ActivationKind::ReLU
            )
            .is_err()
        );
    }

    #[test]
    fn ffn_silu_activation() {
        let input = [1.0];
        let w1 = [1.0];
        let w2 = [1.0];
        let mut out = [0.0f32; 1];
        fuse_ffn(&input, &w1, &[], &w2, &[], &mut out, 1, 1, 1, ActivationKind::SiLU).unwrap();
        let expected = silu(1.0);
        assert!((out[0] - expected).abs() < TOL);
    }

    #[test]
    fn ffn_gelu_activation() {
        let input = [1.0];
        let w1 = [1.0];
        let w2 = [1.0];
        let mut out = [0.0f32; 1];
        fuse_ffn(&input, &w1, &[], &w2, &[], &mut out, 1, 1, 1, ActivationKind::GELU).unwrap();
        let expected = gelu(1.0);
        assert!((out[0] - expected).abs() < TOL);
    }

    // ── fuse_embedding_layernorm tests ───────────────────────────

    #[test]
    fn embedding_ln_basic() {
        let embeddings = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // vocab=3, dim=2
        let token_ids = [0usize, 2];
        let gamma = [1.0, 1.0];
        let beta: &[f32] = &[];
        let mut out = [0.0f32; 4]; // seq=2, dim=2
        fuse_embedding_layernorm(&embeddings, &token_ids, &gamma, beta, &mut out, 3, 2, EPS)
            .unwrap();
        // Token 0: [1,2] → normed
        // Token 2: [5,6] → normed
        // Each row should sum to ~0 (normalised)
        assert!((out[0] + out[1]).abs() < 0.1);
        assert!((out[2] + out[3]).abs() < 0.1);
    }

    #[test]
    fn embedding_ln_with_beta() {
        let embeddings = [1.0; 4]; // vocab=2, dim=2, constant
        let token_ids = [0usize];
        let gamma = [1.0, 1.0];
        let beta = [0.5, 0.5];
        let mut out = [0.0f32; 2];
        fuse_embedding_layernorm(&embeddings, &token_ids, &gamma, &beta, &mut out, 2, 2, EPS)
            .unwrap();
        // Constant input → var=0 → normed=0 → output=beta=0.5
        for &v in &out {
            assert!((v - 0.5).abs() < TOL);
        }
    }

    #[test]
    fn embedding_ln_empty_tokens_errors() {
        let mut out = [0.0f32; 4];
        assert!(
            fuse_embedding_layernorm(&[1.0; 4], &[], &[1.0; 2], &[], &mut out, 2, 2, EPS).is_err()
        );
    }

    #[test]
    fn embedding_ln_token_oob() {
        let mut out = [0.0f32; 2];
        assert!(
            fuse_embedding_layernorm(&[1.0; 4], &[5], &[1.0; 2], &[], &mut out, 2, 2, EPS).is_err()
        );
    }

    #[test]
    fn embedding_ln_gamma_mismatch() {
        let mut out = [0.0f32; 2];
        assert!(
            fuse_embedding_layernorm(&[1.0; 4], &[0], &[1.0; 3], &[], &mut out, 2, 2, EPS).is_err()
        );
    }

    #[test]
    fn embedding_ln_output_too_short() {
        let mut out = [0.0f32; 1];
        assert!(
            fuse_embedding_layernorm(&[1.0; 4], &[0], &[1.0; 2], &[], &mut out, 2, 2, EPS).is_err()
        );
    }

    #[test]
    fn embedding_ln_zero_embed_dim() {
        let mut out = [0.0f32; 1];
        assert!(fuse_embedding_layernorm(&[], &[0], &[], &[], &mut out, 2, 0, EPS).is_err());
    }

    // ── Activation helper tests ──────────────────────────────────

    #[test]
    fn relu_positive() {
        assert!((relu(3.0) - 3.0).abs() < TOL);
    }

    #[test]
    fn relu_negative() {
        assert!(relu(-1.0).abs() < TOL);
    }

    #[test]
    fn silu_zero() {
        assert!(silu(0.0).abs() < TOL);
    }

    #[test]
    fn silu_positive() {
        let v = silu(1.0);
        assert!(v > 0.5 && v < 1.0);
    }

    #[test]
    fn gelu_zero() {
        assert!(gelu(0.0).abs() < TOL);
    }

    #[test]
    fn gelu_positive() {
        let v = gelu(1.0);
        assert!(v > 0.5 && v < 1.0);
    }

    // ── Integration: planner + optimizer pipeline ────────────────

    #[test]
    fn end_to_end_planner_optimizer() {
        let planner = FusionPlanner::new();
        let optimizer = FusionOptimizer::cpu_only();
        let ops = make_matmul_bias_act_seq();
        let opps = planner.find_opportunities(&ops);
        let selected = optimizer.select(&opps);
        assert!(!selected.is_empty());
    }

    #[test]
    fn end_to_end_no_fusion_on_small() {
        let rules =
            vec![FusionRule::new(FusionPattern::MatMulBiasActivation).with_min_elements(999_999)];
        let planner = FusionPlanner::with_rules(rules);
        let optimizer = FusionOptimizer::cpu_only();
        let ops = make_matmul_bias_act_seq();
        let opps = planner.find_opportunities(&ops);
        let selected = optimizer.select(&opps);
        assert!(selected.is_empty());
    }

    #[test]
    fn end_to_end_transformer_layer_sequence() {
        let planner = FusionPlanner::new();
        let optimizer = FusionOptimizer::cpu_only();
        let ops = vec![
            OpDescriptor { id: 0, op: OpType::RmsNorm { len: 64 }, element_count: 64 },
            OpDescriptor {
                id: 1,
                op: OpType::Linear { in_features: 64, out_features: 256 },
                element_count: 256,
            },
            OpDescriptor { id: 2, op: OpType::MatMul { m: 1, n: 64, k: 64 }, element_count: 64 },
            OpDescriptor { id: 3, op: OpType::AttentionMask { len: 64 }, element_count: 64 },
            OpDescriptor { id: 4, op: OpType::Softmax { len: 64 }, element_count: 64 },
            OpDescriptor {
                id: 5,
                op: OpType::Linear { in_features: 64, out_features: 256 },
                element_count: 256,
            },
            OpDescriptor {
                id: 6,
                op: OpType::Activation(ActivationKind::SiLU),
                element_count: 256,
            },
            OpDescriptor {
                id: 7,
                op: OpType::Linear { in_features: 256, out_features: 64 },
                element_count: 64,
            },
            OpDescriptor { id: 8, op: OpType::ResidualAdd { len: 64 }, element_count: 64 },
        ];
        let opps = planner.find_opportunities(&ops);
        assert!(!opps.is_empty());
        let selected = optimizer.select(&opps);
        // Should find at least RmsNorm+Linear, Attention+Softmax, FFN
        assert!(!selected.is_empty());
    }

    // ── Bias validation edge case ────────────────────────────────

    #[test]
    fn matmul_bias_act_bias_too_short() {
        let mut out = [0.0f32; 4];
        assert!(
            fuse_matmul_bias_activation(
                &[1.0; 4],
                &[1.0; 4],
                &[1.0], // bias len 1 < n=2
                &mut out,
                2,
                2,
                2,
                ActivationKind::ReLU
            )
            .is_err()
        );
    }

    #[test]
    fn layernorm_residual_beta_mismatch() {
        let mut out = [0.0f32; 4];
        assert!(
            fuse_layernorm_residual(
                &[1.0; 4], &[1.0; 4], &[1.0; 3], // beta len 3 != 4
                &[0.0; 4], &mut out, EPS,
            )
            .is_err()
        );
    }

    #[test]
    fn embedding_ln_beta_mismatch() {
        let mut out = [0.0f32; 2];
        assert!(
            fuse_embedding_layernorm(
                &[1.0; 4],
                &[0],
                &[1.0; 2],
                &[1.0; 3], // beta len 3 != 2
                &mut out,
                2,
                2,
                EPS,
            )
            .is_err()
        );
    }

    // ── b length validation in fuse_matmul_bias_activation ───────

    #[test]
    fn matmul_bias_act_b_too_short() {
        let mut out = [0.0f32; 4];
        assert!(
            fuse_matmul_bias_activation(
                &[1.0; 4],
                &[1.0; 2], // b len 2 < k*n=4
                &[],
                &mut out,
                2,
                2,
                2,
                ActivationKind::ReLU
            )
            .is_err()
        );
    }
}
