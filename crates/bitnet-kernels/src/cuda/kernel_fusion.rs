//! CUDA kernel fusion framework with pattern detection and fused operations.
//!
//! This module provides a graph-based fusion framework that detects
//! opportunities to combine multiple consecutive GPU operations into
//! single fused kernels, eliminating intermediate global-memory
//! round-trips and kernel launch overhead.
//!
//! # Fusion patterns
//!
//! | Pattern | Components | Benefit |
//! |---|---|---|
//! | **MatMulBias** | GEMM + bias add | eliminate bias buffer |
//! | **MatMulBiasReLU** | GEMM + bias + ReLU | skip activation write |
//! | **LayerNormResidual** | LayerNorm + residual add | 1 read of input |
//! | **AttentionScoreSoftmax** | QK^T scoring + softmax | single-pass row reduction |
//! | **QKVProjection** | 3× linear projections | shared input read |
//! | **GatedLinearUnit** | linear + gate linear + mul | shared input, skip gate buffer |
//! | **RMSNormLinear** | RMSNorm + linear projection | 1 read of input |
//!
//! # Architecture
//!
//! The fusion pipeline is:
//!
//! 1. Build a [`FusionGraph`] describing the operation DAG
//! 2. Call [`detect_fusion_opportunities`] to find applicable patterns
//! 3. Call [`apply_fusion`] to generate a [`FusedKernel`] descriptor
//! 4. Execute the fused kernel (CPU fallback always available)
//!
//! # CPU fallback
//!
//! Every fused operation has a pure-Rust scalar fallback that is always
//! compiled.  GPU launch stubs are gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use std::fmt;

use bitnet_common::{KernelError, Result};

// ───────────────────────────────────────────────────────────────────
// Fusion patterns
// ───────────────────────────────────────────────────────────────────

/// Recognised fusion patterns that can replace sequences of operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// GEMM followed by bias addition.
    MatMulBias,
    /// GEMM followed by bias addition and ReLU activation.
    MatMulBiasReLU,
    /// LayerNorm fused with residual addition.
    LayerNormResidual,
    /// Attention score computation fused with softmax.
    AttentionScoreSoftmax,
    /// Three linear projections (Q, K, V) sharing a single input read.
    QKVProjection,
    /// Gated linear unit: two linear projections with element-wise gate.
    GatedLinearUnit,
    /// RMSNorm followed by linear projection.
    RMSNormLinear,
}

impl fmt::Display for FusionPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatMulBias => write!(f, "MatMulBias"),
            Self::MatMulBiasReLU => write!(f, "MatMulBiasReLU"),
            Self::LayerNormResidual => write!(f, "LayerNormResidual"),
            Self::AttentionScoreSoftmax => write!(f, "AttentionScoreSoftmax"),
            Self::QKVProjection => write!(f, "QKVProjection"),
            Self::GatedLinearUnit => write!(f, "GatedLinearUnit"),
            Self::RMSNormLinear => write!(f, "RMSNormLinear"),
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Configuration
// ───────────────────────────────────────────────────────────────────

/// Resource limits that govern when fusion is profitable.
#[derive(Debug, Clone)]
pub struct KernelFusionConfig {
    /// Maximum number of operations that may be fused into one kernel.
    pub max_fused_ops: usize,
    /// Shared memory budget in bytes (per-SM limit).
    pub shared_memory_limit: usize,
    /// Maximum register pressure (registers per thread).
    pub register_pressure_limit: usize,
    /// Minimum element count before fusion is attempted.
    pub min_elements: usize,
}

impl Default for KernelFusionConfig {
    fn default() -> Self {
        Self {
            max_fused_ops: 4,
            shared_memory_limit: 48 * 1024, // 48 KiB
            register_pressure_limit: 64,
            min_elements: 32,
        }
    }
}

impl KernelFusionConfig {
    /// Validate configuration invariants.
    pub fn validate(&self) -> std::result::Result<(), KernelFusionError> {
        if self.max_fused_ops == 0 {
            return Err(KernelFusionError::InvalidConfig("max_fused_ops must be > 0".into()));
        }
        if self.shared_memory_limit == 0 {
            return Err(KernelFusionError::InvalidConfig("shared_memory_limit must be > 0".into()));
        }
        if self.register_pressure_limit == 0 {
            return Err(KernelFusionError::InvalidConfig(
                "register_pressure_limit must be > 0".into(),
            ));
        }
        if self.min_elements == 0 {
            return Err(KernelFusionError::InvalidConfig("min_elements must be > 0".into()));
        }
        Ok(())
    }
}

// ───────────────────────────────────────────────────────────────────
// Operation types for the fusion graph
// ───────────────────────────────────────────────────────────────────

/// Primitive operation types that appear as graph nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    /// Dense matrix multiplication.
    MatMul,
    /// Element-wise bias addition.
    BiasAdd,
    /// ReLU activation.
    ReLU,
    /// Layer normalization.
    LayerNorm,
    /// RMS normalization.
    RMSNorm,
    /// Residual (element-wise) addition.
    ResidualAdd,
    /// Attention score computation (Q·K^T / √d).
    AttentionScore,
    /// Softmax.
    Softmax,
    /// Linear projection (y = xW^T + b).
    Linear,
    /// Element-wise gating multiplication.
    GateMul,
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatMul => write!(f, "MatMul"),
            Self::BiasAdd => write!(f, "BiasAdd"),
            Self::ReLU => write!(f, "ReLU"),
            Self::LayerNorm => write!(f, "LayerNorm"),
            Self::RMSNorm => write!(f, "RMSNorm"),
            Self::ResidualAdd => write!(f, "ResidualAdd"),
            Self::AttentionScore => write!(f, "AttentionScore"),
            Self::Softmax => write!(f, "Softmax"),
            Self::Linear => write!(f, "Linear"),
            Self::GateMul => write!(f, "GateMul"),
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Fusion graph
// ───────────────────────────────────────────────────────────────────

/// A single node in the fusion graph.
#[derive(Debug, Clone)]
pub struct FusionNode {
    /// Unique node identifier within the graph.
    pub id: usize,
    /// The primitive operation this node represents.
    pub op_type: OpType,
    /// Input tensor shapes (each shape is a `Vec<usize>`).
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor shapes.
    pub output_shapes: Vec<Vec<usize>>,
    /// Indices of predecessor nodes (data dependencies).
    pub inputs: Vec<usize>,
}

/// Directed acyclic graph of operations eligible for fusion analysis.
#[derive(Debug, Clone)]
pub struct FusionGraph {
    /// Ordered list of nodes (topologically sorted).
    pub nodes: Vec<FusionNode>,
}

impl FusionGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a node and return its id.
    pub fn add_node(
        &mut self,
        op_type: OpType,
        input_shapes: Vec<Vec<usize>>,
        output_shapes: Vec<Vec<usize>>,
        inputs: Vec<usize>,
    ) -> std::result::Result<usize, KernelFusionError> {
        for &dep in &inputs {
            if dep >= self.nodes.len() {
                return Err(KernelFusionError::InvalidGraph(format!(
                    "dependency index {dep} out of range (graph has {} nodes)",
                    self.nodes.len()
                )));
            }
        }
        let id = self.nodes.len();
        self.nodes.push(FusionNode { id, op_type, input_shapes, output_shapes, inputs });
        Ok(id)
    }

    /// Number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Validate the graph (no dangling edges, DAG property).
    pub fn validate(&self) -> std::result::Result<(), KernelFusionError> {
        for node in &self.nodes {
            for &dep in &node.inputs {
                if dep >= node.id {
                    return Err(KernelFusionError::InvalidGraph(format!(
                        "node {} depends on node {} which is not a predecessor",
                        node.id, dep
                    )));
                }
            }
        }
        Ok(())
    }
}

impl Default for FusionGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ───────────────────────────────────────────────────────────────────
// Fused kernel descriptor
// ───────────────────────────────────────────────────────────────────

/// Describes a fused CUDA kernel ready for dispatch.
#[derive(Debug, Clone)]
pub struct FusedKernel {
    /// CUDA C source for the fused kernel.
    pub cuda_source: String,
    /// Launch configuration: (grid, block).
    pub launch_config: LaunchConfig,
    /// Bytes of dynamic shared memory required.
    pub shared_mem_bytes: u32,
    /// The fusion pattern that produced this kernel.
    pub pattern: FusionPattern,
    /// Estimated speedup over unfused execution.
    pub estimated_speedup: f32,
}

/// CUDA launch dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Grid dimensions (x, y, z).
    pub grid: (u32, u32, u32),
    /// Block dimensions (x, y, z).
    pub block: (u32, u32, u32),
}

// ───────────────────────────────────────────────────────────────────
// Error type
// ───────────────────────────────────────────────────────────────────

/// Errors from the kernel fusion framework.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelFusionError {
    /// The fusion graph is structurally invalid.
    InvalidGraph(String),
    /// Configuration is invalid.
    InvalidConfig(String),
    /// The requested pattern cannot be applied.
    PatternNotApplicable(String),
    /// Tensor dimensions do not match.
    DimensionMismatch { expected: usize, got: usize },
    /// An input tensor is empty.
    EmptyInput,
}

impl fmt::Display for KernelFusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(msg) => write!(f, "invalid fusion graph: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "invalid fusion config: {msg}"),
            Self::PatternNotApplicable(msg) => write!(f, "pattern not applicable: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for KernelFusionError {}

// ───────────────────────────────────────────────────────────────────
// Pattern detection
// ───────────────────────────────────────────────────────────────────

/// Scan a fusion graph and return all applicable fusion patterns.
///
/// The detector walks consecutive node pairs / triples and matches
/// against known fusible sequences.
pub fn detect_fusion_opportunities(
    graph: &FusionGraph,
    config: &KernelFusionConfig,
) -> Vec<FusionPattern> {
    if graph.is_empty() {
        return Vec::new();
    }
    if config.max_fused_ops < 2 {
        return Vec::new();
    }

    let mut patterns = Vec::new();
    let nodes = &graph.nodes;

    // Two-node patterns
    for i in 0..nodes.len().saturating_sub(1) {
        let a = &nodes[i];
        let b = &nodes[i + 1];
        if !b.inputs.contains(&a.id) {
            continue;
        }
        match (a.op_type, b.op_type) {
            (OpType::MatMul, OpType::BiasAdd) => {
                patterns.push(FusionPattern::MatMulBias);
            }
            (OpType::LayerNorm, OpType::ResidualAdd) | (OpType::ResidualAdd, OpType::LayerNorm) => {
                patterns.push(FusionPattern::LayerNormResidual);
            }
            (OpType::AttentionScore, OpType::Softmax) => {
                patterns.push(FusionPattern::AttentionScoreSoftmax);
            }
            (OpType::RMSNorm, OpType::Linear) => {
                patterns.push(FusionPattern::RMSNormLinear);
            }
            _ => {}
        }
    }

    // Three-node patterns
    if config.max_fused_ops >= 3 {
        for i in 0..nodes.len().saturating_sub(2) {
            let a = &nodes[i];
            let b = &nodes[i + 1];
            let c = &nodes[i + 2];
            if !b.inputs.contains(&a.id) || !c.inputs.contains(&b.id) {
                continue;
            }
            if a.op_type == OpType::MatMul
                && b.op_type == OpType::BiasAdd
                && c.op_type == OpType::ReLU
            {
                patterns.push(FusionPattern::MatMulBiasReLU);
            }
            // QKV: three consecutive linear projections sharing an input
            if a.op_type == OpType::Linear
                && b.op_type == OpType::Linear
                && c.op_type == OpType::Linear
            {
                // Check they share a common input shape
                if !a.input_shapes.is_empty()
                    && !b.input_shapes.is_empty()
                    && a.input_shapes[0] == b.input_shapes[0]
                    && b.input_shapes[0] == c.input_shapes[0]
                {
                    patterns.push(FusionPattern::QKVProjection);
                }
            }
        }
    }

    // GLU: two linear projections feeding a gate multiply
    if config.max_fused_ops >= 3 {
        for i in 0..nodes.len().saturating_sub(2) {
            let a = &nodes[i];
            let b = &nodes[i + 1];
            let c = &nodes[i + 2];
            if a.op_type == OpType::Linear
                && b.op_type == OpType::Linear
                && c.op_type == OpType::GateMul
                && c.inputs.contains(&a.id)
                && c.inputs.contains(&b.id)
            {
                patterns.push(FusionPattern::GatedLinearUnit);
            }
        }
    }

    patterns
}

// ───────────────────────────────────────────────────────────────────
// Shared memory estimation
// ───────────────────────────────────────────────────────────────────

/// Estimate shared memory bytes required for a fusion pattern.
pub fn estimate_shared_memory(pattern: FusionPattern, n: usize) -> u32 {
    let threads_per_block = (n as u32).min(256);
    match pattern {
        FusionPattern::MatMulBias | FusionPattern::MatMulBiasReLU => {
            // Tile-based: 2 tiles of 32×32 floats
            2 * 32 * 32 * 4
        }
        FusionPattern::LayerNormResidual => {
            // One float per thread for partial sums
            threads_per_block * 4
        }
        FusionPattern::AttentionScoreSoftmax => {
            // Two passes: max + exp-sum, one float per thread each
            threads_per_block * 4 * 2
        }
        FusionPattern::QKVProjection => {
            // Shared input tile: threads × 4 bytes × 3 accumulators
            threads_per_block * 4 * 3
        }
        FusionPattern::GatedLinearUnit => {
            // Two partial sums per thread
            threads_per_block * 4 * 2
        }
        FusionPattern::RMSNormLinear => {
            // Partial sum-of-squares + dot product accumulator
            threads_per_block * 4
        }
    }
}

/// Estimate register pressure (registers per thread) for a pattern.
pub fn estimate_register_pressure(pattern: FusionPattern) -> usize {
    match pattern {
        FusionPattern::MatMulBias => 16,
        FusionPattern::MatMulBiasReLU => 18,
        FusionPattern::LayerNormResidual => 12,
        FusionPattern::AttentionScoreSoftmax => 20,
        FusionPattern::QKVProjection => 24,
        FusionPattern::GatedLinearUnit => 22,
        FusionPattern::RMSNormLinear => 14,
    }
}

// ───────────────────────────────────────────────────────────────────
// Fusion application
// ───────────────────────────────────────────────────────────────────

/// Apply a fusion pattern to generate a fused kernel descriptor.
pub fn apply_fusion(
    graph: &FusionGraph,
    pattern: FusionPattern,
    config: &KernelFusionConfig,
) -> std::result::Result<FusedKernel, KernelFusionError> {
    config.validate()?;
    if graph.is_empty() {
        return Err(KernelFusionError::InvalidGraph("empty graph".into()));
    }

    let n = graph.nodes[0].input_shapes.first().and_then(|s| s.last().copied()).unwrap_or(256);

    let shared_mem = estimate_shared_memory(pattern, n);
    if shared_mem as usize > config.shared_memory_limit {
        return Err(KernelFusionError::PatternNotApplicable(format!(
            "shared memory {shared_mem} exceeds limit {}",
            config.shared_memory_limit
        )));
    }

    let reg_pressure = estimate_register_pressure(pattern);
    if reg_pressure > config.register_pressure_limit {
        return Err(KernelFusionError::PatternNotApplicable(format!(
            "register pressure {reg_pressure} exceeds limit {}",
            config.register_pressure_limit
        )));
    }

    let threads_per_block = (n as u32).min(256);
    let grid_x = match pattern {
        FusionPattern::MatMulBias
        | FusionPattern::MatMulBiasReLU
        | FusionPattern::RMSNormLinear => {
            graph.nodes.last().and_then(|n| n.output_shapes.first()?.first().copied()).unwrap_or(1)
                as u32
        }
        FusionPattern::QKVProjection => 3, // Q, K, V output rows
        _ => 1,
    };

    let cuda_source = generate_cuda_source(pattern);
    let speedup = estimate_fusion_speedup_for_pattern(pattern);

    Ok(FusedKernel {
        cuda_source,
        launch_config: LaunchConfig { grid: (grid_x, 1, 1), block: (threads_per_block, 1, 1) },
        shared_mem_bytes: shared_mem,
        pattern,
        estimated_speedup: speedup,
    })
}

/// Generate CUDA C source for a fusion pattern.
fn generate_cuda_source(pattern: FusionPattern) -> String {
    match pattern {
        FusionPattern::MatMulBias => FUSED_MATMUL_BIAS_SRC.to_string(),
        FusionPattern::MatMulBiasReLU => FUSED_MATMUL_BIAS_RELU_SRC.to_string(),
        FusionPattern::LayerNormResidual => FUSED_LAYER_NORM_RESIDUAL_SRC.to_string(),
        FusionPattern::AttentionScoreSoftmax => FUSED_ATTENTION_SCORE_SOFTMAX_SRC.to_string(),
        FusionPattern::QKVProjection => FUSED_QKV_PROJECTION_SRC.to_string(),
        FusionPattern::GatedLinearUnit => FUSED_GLU_SRC.to_string(),
        FusionPattern::RMSNormLinear => FUSED_RMSNORM_LINEAR_SRC.to_string(),
    }
}

/// Estimate speedup of a fused pattern relative to unfused execution.
fn estimate_fusion_speedup_for_pattern(pattern: FusionPattern) -> f32 {
    match pattern {
        FusionPattern::MatMulBias => 1.1,
        FusionPattern::MatMulBiasReLU => 1.25,
        FusionPattern::LayerNormResidual => 1.4,
        FusionPattern::AttentionScoreSoftmax => 1.5,
        FusionPattern::QKVProjection => 1.8,
        FusionPattern::GatedLinearUnit => 1.6,
        FusionPattern::RMSNormLinear => 1.35,
    }
}

/// Estimate speedup given explicit unfused op count and fused kernel.
///
/// Returns estimated wall-clock ratio (`unfused_time / fused_time`).
pub fn estimate_fusion_speedup(unfused_op_count: usize, fused_kernel: &FusedKernel) -> f32 {
    if unfused_op_count <= 1 {
        return 1.0;
    }
    // Base speedup from pattern + bonus for additional fused ops
    let base = fused_kernel.estimated_speedup;
    let bonus = (unfused_op_count as f32 - 2.0).max(0.0) * 0.05;
    base + bonus
}

// ───────────────────────────────────────────────────────────────────
// CUDA kernel sources (inline C)
// ───────────────────────────────────────────────────────────────────

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const KERNEL_FUSION_CUDA_SRC: &str = r#""#;

const FUSED_MATMUL_BIAS_SRC: &str = r#"
extern "C" __global__ void fused_matmul_bias_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int m, int n, int k)
{
    int row = blockIdx.x;
    if (row >= m) return;
    extern __shared__ float sdata[];
    const float* a_row = a + (long long)row * k;
    float acc = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float sum = 0.0f;
        for (int p = 0; p < k; p++) {
            sum += a_row[p] * b[(long long)p * n + j];
        }
        output[(long long)row * n + j] = sum + bias[j];
    }
}
"#;

const FUSED_MATMUL_BIAS_RELU_SRC: &str = r#"
extern "C" __global__ void fused_matmul_bias_relu_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int m, int n, int k)
{
    int row = blockIdx.x;
    if (row >= m) return;
    const float* a_row = a + (long long)row * k;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float sum = 0.0f;
        for (int p = 0; p < k; p++) {
            sum += a_row[p] * b[(long long)p * n + j];
        }
        float val = sum + bias[j];
        output[(long long)row * n + j] = (val > 0.0f) ? val : 0.0f;
    }
}
"#;

const FUSED_LAYER_NORM_RESIDUAL_SRC: &str = r#"
extern "C" __global__ void fused_layer_norm_residual_f32(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int n, float eps)
{
    extern __shared__ float sdata[];
    // compute mean of (input + residual)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += input[i] + residual[i];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)n;
    __syncthreads();

    // compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float d = (input[i] + residual[i]) - mean;
        local_var += d * d;
    }
    sdata[threadIdx.x] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / (float)n + eps);

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float normed = ((input[i] + residual[i]) - mean) * inv_std;
        output[i] = normed * gamma[i] + beta[i];
    }
}
"#;

const FUSED_ATTENTION_SCORE_SOFTMAX_SRC: &str = r#"
extern "C" __global__ void fused_attention_score_softmax_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ output,
    int seq_len, int head_dim, float scale,
    const float* __restrict__ mask)
{
    extern __shared__ float sdata[];
    // simplified: single-head, q[head_dim], k[seq_len * head_dim]
    // compute scores and softmax in one pass
    int i = threadIdx.x;
    if (i >= seq_len) return;
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        score += q[d] * k[(long long)i * head_dim + d];
    }
    score *= scale;
    if (mask) score += mask[i];
    sdata[i] = score;
    __syncthreads();

    // find max
    float max_val = -1e30f;
    for (int j = 0; j < seq_len; j++) {
        if (sdata[j] > max_val) max_val = sdata[j];
    }
    float exp_val = expf(score - max_val);
    sdata[i] = exp_val;
    __syncthreads();

    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) sum += sdata[j];
    output[i] = (sum > 0.0f) ? (exp_val / sum) : 0.0f;
}
"#;

const FUSED_QKV_PROJECTION_SRC: &str = r#"
extern "C" __global__ void fused_qkv_projection_f32(
    const float* __restrict__ input,
    const float* __restrict__ wq,
    const float* __restrict__ wk,
    const float* __restrict__ wv,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    int n, int out_dim)
{
    int row = blockIdx.x; // 0=Q, 1=K, 2=V
    if (row >= 3) return;
    const float* w = (row == 0) ? wq : ((row == 1) ? wk : wv);
    float* out = (row == 0) ? q_out : ((row == 1) ? k_out : v_out);
    for (int j = threadIdx.x; j < out_dim; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < n; i++) {
            acc += input[i] * w[(long long)j * n + i];
        }
        out[j] = acc;
    }
}
"#;

const FUSED_GLU_SRC: &str = r#"
extern "C" __global__ void fused_glu_f32(
    const float* __restrict__ input,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    float* __restrict__ output,
    int n, int out_dim)
{
    for (int j = threadIdx.x; j < out_dim; j += blockDim.x) {
        float gate_val = 0.0f;
        float up_val = 0.0f;
        for (int i = 0; i < n; i++) {
            gate_val += input[i] * w_gate[(long long)j * n + i];
            up_val   += input[i] * w_up[(long long)j * n + i];
        }
        // SiLU gate
        float sigmoid_gate = 1.0f / (1.0f + expf(-gate_val));
        output[j] = (gate_val * sigmoid_gate) * up_val;
    }
}
"#;

const FUSED_RMSNORM_LINEAR_SRC: &str = r#"
extern "C" __global__ void fused_rmsnorm_linear_kf_f32(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int n, int out_dim, float eps)
{
    int row = blockIdx.x;
    if (row >= out_dim) return;
    extern __shared__ float sdata[];
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = input[i];
        local_ss += v * v;
    }
    sdata[threadIdx.x] = local_ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)n + eps);
    const float* w = weight + (long long)row * n;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        acc += w[i] * (input[i] * gamma[i] * inv_rms);
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = sdata[0];
}
"#;

// ───────────────────────────────────────────────────────────────────
// CPU fallback: fused GEMM + bias
// ───────────────────────────────────────────────────────────────────

/// Fused GEMM + bias addition (CPU fallback).
///
/// Computes `output[i][j] = sum_p(a[i][p] * b[p][j]) + bias[j]`
/// for `i in 0..m`, `j in 0..n`.
///
/// * `a` — `[m × k]` row-major
/// * `b` — `[k × n]` row-major
/// * `bias` — `[n]`
/// * `output` — `[m × n]` (written)
pub fn fused_matmul_bias(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    validate_matmul_args(a, b, bias, output, m, n, k)?;
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            output[i * n + j] = sum + bias[j];
        }
    }
    Ok(())
}

/// Fused GEMM + bias + ReLU (CPU fallback).
///
/// Same as [`fused_matmul_bias`] but applies `max(0, x)` to each output.
pub fn fused_matmul_bias_relu(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    validate_matmul_args(a, b, bias, output, m, n, k)?;
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            let val = sum + bias[j];
            output[i * n + j] = val.max(0.0);
        }
    }
    Ok(())
}

fn validate_matmul_args(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    output: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "dimensions must be non-zero".into() }.into()
        );
    }
    if a.len() < m * k {
        return Err(KernelError::InvalidArguments {
            reason: format!("a length {} < m*k = {}", a.len(), m * k),
        }
        .into());
    }
    if b.len() < k * n {
        return Err(KernelError::InvalidArguments {
            reason: format!("b length {} < k*n = {}", b.len(), k * n),
        }
        .into());
    }
    if bias.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("bias length {} < n = {n}", bias.len()),
        }
        .into());
    }
    if output.len() < m * n {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < m*n = {}", output.len(), m * n),
        }
        .into());
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// CPU fallback: fused LayerNorm + residual
// ───────────────────────────────────────────────────────────────────

/// Fused LayerNorm + residual addition (CPU fallback).
///
/// Computes `output[i] = gamma[i] * ((input[i] + residual[i] - mean) / std) + beta[i]`
///
/// * `input`, `residual` — `[n]`
/// * `gamma`, `beta` — `[n]`
/// * `output` — `[n]` (written)
/// * `eps` — normalisation epsilon
pub fn fused_layer_norm_residual(
    input: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    eps: f32,
) -> Result<()> {
    let n = input.len();
    if n == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "input must be non-empty".into() }.into()
        );
    }
    if residual.len() != n {
        return Err(KernelError::InvalidArguments {
            reason: format!("residual length {} != input length {n}", residual.len()),
        }
        .into());
    }
    if gamma.len() != n {
        return Err(KernelError::InvalidArguments {
            reason: format!("gamma length {} != input length {n}", gamma.len()),
        }
        .into());
    }
    if beta.len() != n {
        return Err(KernelError::InvalidArguments {
            reason: format!("beta length {} != input length {n}", beta.len()),
        }
        .into());
    }
    if output.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < n={n}", output.len()),
        }
        .into());
    }

    // Compute sum of (input + residual)
    let sum: f32 = input.iter().zip(residual).map(|(&a, &b)| a + b).sum();
    let mean = sum / n as f32;

    // Compute variance
    let var: f32 = input
        .iter()
        .zip(residual)
        .map(|(&a, &b)| {
            let d = (a + b) - mean;
            d * d
        })
        .sum();
    let inv_std = 1.0 / (var / n as f32 + eps).sqrt();

    for i in 0..n {
        let normed = ((input[i] + residual[i]) - mean) * inv_std;
        output[i] = normed * gamma[i] + beta[i];
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// CPU fallback: fused attention score + softmax
// ───────────────────────────────────────────────────────────────────

/// Fused attention score computation + softmax (CPU fallback).
///
/// Computes `softmax(Q · K^T * scale + mask)` for a single head.
///
/// * `q` — `[head_dim]` query vector
/// * `k` — `[seq_len × head_dim]` key matrix (row-major)
/// * `scale` — attention scale factor (typically `1 / sqrt(head_dim)`)
/// * `mask` — `[seq_len]` additive mask (0=keep, large-neg=mask out), or empty for no mask
/// * `output` — `[seq_len]` (written)
pub fn fused_attention_score_softmax(
    q: &[f32],
    k: &[f32],
    scale: f32,
    mask: &[f32],
    output: &mut [f32],
) -> Result<()> {
    let head_dim = q.len();
    if head_dim == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "query must be non-empty".into() }.into()
        );
    }
    if k.is_empty() || !k.len().is_multiple_of(head_dim) {
        return Err(KernelError::InvalidArguments {
            reason: format!("key length {} not a multiple of head_dim={head_dim}", k.len()),
        }
        .into());
    }
    let seq_len = k.len() / head_dim;
    if !mask.is_empty() && mask.len() != seq_len {
        return Err(KernelError::InvalidArguments {
            reason: format!("mask length {} != seq_len={seq_len}", mask.len()),
        }
        .into());
    }
    if output.len() < seq_len {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < seq_len={seq_len}", output.len()),
        }
        .into());
    }

    // Compute scores
    let mut max_score = f32::NEG_INFINITY;
    for i in 0..seq_len {
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q[d] * k[i * head_dim + d];
        }
        let mut score = dot * scale;
        if !mask.is_empty() {
            score += mask[i];
        }
        output[i] = score;
        if score > max_score {
            max_score = score;
        }
    }

    // Softmax: exp and sum
    let mut sum = 0.0f32;
    for o in output[..seq_len].iter_mut() {
        let e = (*o - max_score).exp();
        *o = e;
        sum += e;
    }

    // Normalize
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for o in output[..seq_len].iter_mut() {
            *o *= inv;
        }
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// CUDA launch stubs
// ───────────────────────────────────────────────────────────────────

/// Launch fused matmul + bias CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_matmul_bias_cuda(
    _a: &[f32],
    _b: &[f32],
    _bias: &[f32],
    _output: &mut [f32],
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<()> {
    log::debug!("fused matmul+bias CUDA: m={_m}, n={_n}, k={_k}");
    Err(KernelError::GpuError {
        reason: "fused matmul+bias CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch fused matmul + bias + ReLU CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_matmul_bias_relu_cuda(
    _a: &[f32],
    _b: &[f32],
    _bias: &[f32],
    _output: &mut [f32],
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<()> {
    log::debug!("fused matmul+bias+relu CUDA: m={_m}, n={_n}, k={_k}");
    Err(KernelError::GpuError {
        reason: "fused matmul+bias+relu CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch fused LayerNorm + residual CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_layer_norm_residual_cuda(
    _input: &[f32],
    _residual: &[f32],
    _gamma: &[f32],
    _beta: &[f32],
    _output: &mut [f32],
    _n: usize,
    _eps: f32,
) -> Result<()> {
    log::debug!("fused LayerNorm+residual CUDA: n={_n}");
    Err(KernelError::GpuError {
        reason: "fused LayerNorm+residual CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch fused attention score + softmax CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_attention_score_softmax_cuda(
    _q: &[f32],
    _k: &[f32],
    _scale: f32,
    _mask: &[f32],
    _output: &mut [f32],
) -> Result<()> {
    log::debug!("fused attention+softmax CUDA");
    Err(KernelError::GpuError {
        reason: "fused attention+softmax CUDA kernel not yet compiled — scaffold only".into(),
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

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && max_abs_err(a, b) < tol
    }

    // ───────────────── FusionPattern Display ─────────────────

    #[test]
    fn test_fusion_pattern_display() {
        assert_eq!(FusionPattern::MatMulBias.to_string(), "MatMulBias");
        assert_eq!(FusionPattern::MatMulBiasReLU.to_string(), "MatMulBiasReLU");
        assert_eq!(FusionPattern::LayerNormResidual.to_string(), "LayerNormResidual");
        assert_eq!(FusionPattern::AttentionScoreSoftmax.to_string(), "AttentionScoreSoftmax");
        assert_eq!(FusionPattern::QKVProjection.to_string(), "QKVProjection");
        assert_eq!(FusionPattern::GatedLinearUnit.to_string(), "GatedLinearUnit");
        assert_eq!(FusionPattern::RMSNormLinear.to_string(), "RMSNormLinear");
    }

    #[test]
    fn test_fusion_pattern_eq_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FusionPattern::MatMulBias);
        set.insert(FusionPattern::MatMulBiasReLU);
        set.insert(FusionPattern::MatMulBias); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_fusion_pattern_clone() {
        let p = FusionPattern::QKVProjection;
        let p2 = p;
        assert_eq!(p, p2);
    }

    // ───────────────── OpType Display ─────────────────

    #[test]
    fn test_op_type_display() {
        assert_eq!(OpType::MatMul.to_string(), "MatMul");
        assert_eq!(OpType::BiasAdd.to_string(), "BiasAdd");
        assert_eq!(OpType::ReLU.to_string(), "ReLU");
        assert_eq!(OpType::LayerNorm.to_string(), "LayerNorm");
        assert_eq!(OpType::RMSNorm.to_string(), "RMSNorm");
        assert_eq!(OpType::ResidualAdd.to_string(), "ResidualAdd");
        assert_eq!(OpType::AttentionScore.to_string(), "AttentionScore");
        assert_eq!(OpType::Softmax.to_string(), "Softmax");
        assert_eq!(OpType::Linear.to_string(), "Linear");
        assert_eq!(OpType::GateMul.to_string(), "GateMul");
    }

    // ───────────────── KernelFusionConfig ─────────────────

    #[test]
    fn test_config_default() {
        let cfg = KernelFusionConfig::default();
        assert_eq!(cfg.max_fused_ops, 4);
        assert_eq!(cfg.shared_memory_limit, 48 * 1024);
        assert_eq!(cfg.register_pressure_limit, 64);
        assert_eq!(cfg.min_elements, 32);
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = KernelFusionConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_max_fused_ops() {
        let cfg = KernelFusionConfig { max_fused_ops: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_shared_mem() {
        let cfg = KernelFusionConfig { shared_memory_limit: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_register_pressure() {
        let cfg = KernelFusionConfig { register_pressure_limit: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_min_elements() {
        let cfg = KernelFusionConfig { min_elements: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    // ───────────────── FusionGraph construction ─────────────────

    #[test]
    fn test_graph_new_is_empty() {
        let g = FusionGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn test_graph_default() {
        let g = FusionGraph::default();
        assert!(g.is_empty());
    }

    #[test]
    fn test_graph_add_node() {
        let mut g = FusionGraph::new();
        let id = g.add_node(OpType::MatMul, vec![vec![4, 8]], vec![vec![4, 16]], vec![]).unwrap();
        assert_eq!(id, 0);
        assert_eq!(g.len(), 1);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_graph_add_multiple_nodes() {
        let mut g = FusionGraph::new();
        let a = g.add_node(OpType::MatMul, vec![vec![4, 8]], vec![vec![4, 16]], vec![]).unwrap();
        let b = g.add_node(OpType::BiasAdd, vec![vec![4, 16]], vec![vec![4, 16]], vec![a]).unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn test_graph_add_node_invalid_dep() {
        let mut g = FusionGraph::new();
        let result = g.add_node(OpType::MatMul, vec![], vec![], vec![42]);
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_validate_ok() {
        let mut g = FusionGraph::new();
        g.add_node(OpType::MatMul, vec![], vec![], vec![]).unwrap();
        g.add_node(OpType::BiasAdd, vec![], vec![], vec![0]).unwrap();
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_graph_validate_empty() {
        let g = FusionGraph::new();
        assert!(g.validate().is_ok());
    }

    // ───────────────── Pattern detection: positive cases ─────────────────

    fn make_matmul_bias_graph() -> FusionGraph {
        let mut g = FusionGraph::new();
        g.add_node(OpType::MatMul, vec![vec![4, 8]], vec![vec![4, 16]], vec![]).unwrap();
        g.add_node(OpType::BiasAdd, vec![vec![4, 16]], vec![vec![4, 16]], vec![0]).unwrap();
        g
    }

    #[test]
    fn test_detect_matmul_bias() {
        let g = make_matmul_bias_graph();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.contains(&FusionPattern::MatMulBias));
    }

    #[test]
    fn test_detect_matmul_bias_relu() {
        let mut g = FusionGraph::new();
        g.add_node(OpType::MatMul, vec![vec![4, 8]], vec![vec![4, 16]], vec![]).unwrap();
        g.add_node(OpType::BiasAdd, vec![vec![4, 16]], vec![vec![4, 16]], vec![0]).unwrap();
        g.add_node(OpType::ReLU, vec![vec![4, 16]], vec![vec![4, 16]], vec![1]).unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.contains(&FusionPattern::MatMulBias));
        assert!(patterns.contains(&FusionPattern::MatMulBiasReLU));
    }

    #[test]
    fn test_detect_layer_norm_residual() {
        let mut g = FusionGraph::new();
        g.add_node(OpType::LayerNorm, vec![vec![256]], vec![vec![256]], vec![]).unwrap();
        g.add_node(OpType::ResidualAdd, vec![vec![256]], vec![vec![256]], vec![0]).unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.contains(&FusionPattern::LayerNormResidual));
    }

    #[test]
    fn test_detect_residual_layer_norm_reversed() {
        let mut g = FusionGraph::new();
        g.add_node(OpType::ResidualAdd, vec![vec![256]], vec![vec![256]], vec![]).unwrap();
        g.add_node(OpType::LayerNorm, vec![vec![256]], vec![vec![256]], vec![0]).unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.contains(&FusionPattern::LayerNormResidual));
    }

    #[test]
    fn test_detect_attention_score_softmax() {
        let mut g = FusionGraph::new();
        g.add_node(OpType::AttentionScore, vec![vec![64]], vec![vec![128]], vec![]).unwrap();
        g.add_node(OpType::Softmax, vec![vec![128]], vec![vec![128]], vec![0]).unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.contains(&FusionPattern::AttentionScoreSoftmax));
    }

    #[test]
    fn test_detect_rmsnorm_linear() {
        let mut g = FusionGraph::new();
        g.add_node(OpType::RMSNorm, vec![vec![256]], vec![vec![256]], vec![]).unwrap();
        g.add_node(OpType::Linear, vec![vec![256]], vec![vec![512]], vec![0]).unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.contains(&FusionPattern::RMSNormLinear));
    }

    #[test]
    fn test_detect_qkv_projection() {
        let mut g = FusionGraph::new();
        // Three linear projections sharing the same input shape
        g.add_node(OpType::Linear, vec![vec![256]], vec![vec![64]], vec![]).unwrap();
        g.add_node(OpType::Linear, vec![vec![256]], vec![vec![64]], vec![0]).unwrap();
        g.add_node(OpType::Linear, vec![vec![256]], vec![vec![64]], vec![1]).unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.contains(&FusionPattern::QKVProjection));
    }

    #[test]
    fn test_detect_gated_linear_unit() {
        let mut g = FusionGraph::new();
        g.add_node(OpType::Linear, vec![vec![256]], vec![vec![512]], vec![]).unwrap();
        g.add_node(OpType::Linear, vec![vec![256]], vec![vec![512]], vec![0]).unwrap();
        g.add_node(OpType::GateMul, vec![vec![512], vec![512]], vec![vec![512]], vec![0, 1])
            .unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.contains(&FusionPattern::GatedLinearUnit));
    }

    // ───────────────── Pattern detection: negative cases ─────────────────

    #[test]
    fn test_detect_empty_graph() {
        let g = FusionGraph::new();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_detect_single_node_no_patterns() {
        let mut g = FusionGraph::new();
        g.add_node(OpType::MatMul, vec![vec![4, 8]], vec![vec![4, 16]], vec![]).unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_detect_disconnected_nodes() {
        let mut g = FusionGraph::new();
        g.add_node(OpType::MatMul, vec![vec![4, 8]], vec![vec![4, 16]], vec![]).unwrap();
        // BiasAdd does NOT depend on MatMul
        g.add_node(OpType::BiasAdd, vec![vec![4, 16]], vec![vec![4, 16]], vec![]).unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        // No MatMulBias because there's no edge
        assert!(!patterns.contains(&FusionPattern::MatMulBias));
    }

    #[test]
    fn test_detect_wrong_order_no_match() {
        let mut g = FusionGraph::new();
        g.add_node(OpType::BiasAdd, vec![vec![4, 16]], vec![vec![4, 16]], vec![]).unwrap();
        g.add_node(OpType::MatMul, vec![vec![4, 8]], vec![vec![4, 16]], vec![0]).unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(!patterns.contains(&FusionPattern::MatMulBias));
    }

    #[test]
    fn test_detect_max_fused_ops_1_no_patterns() {
        let g = make_matmul_bias_graph();
        let cfg = KernelFusionConfig { max_fused_ops: 1, ..Default::default() };
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_detect_max_fused_ops_2_no_triple() {
        let mut g = FusionGraph::new();
        g.add_node(OpType::MatMul, vec![vec![4, 8]], vec![vec![4, 16]], vec![]).unwrap();
        g.add_node(OpType::BiasAdd, vec![vec![4, 16]], vec![vec![4, 16]], vec![0]).unwrap();
        g.add_node(OpType::ReLU, vec![vec![4, 16]], vec![vec![4, 16]], vec![1]).unwrap();
        let cfg = KernelFusionConfig { max_fused_ops: 2, ..Default::default() };
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.contains(&FusionPattern::MatMulBias));
        assert!(!patterns.contains(&FusionPattern::MatMulBiasReLU));
    }

    #[test]
    fn test_detect_non_fusible_sequence() {
        let mut g = FusionGraph::new();
        g.add_node(OpType::ReLU, vec![vec![256]], vec![vec![256]], vec![]).unwrap();
        g.add_node(OpType::ReLU, vec![vec![256]], vec![vec![256]], vec![0]).unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_detect_qkv_different_shapes_no_match() {
        let mut g = FusionGraph::new();
        g.add_node(OpType::Linear, vec![vec![256]], vec![vec![64]], vec![]).unwrap();
        g.add_node(OpType::Linear, vec![vec![128]], vec![vec![64]], vec![0]).unwrap(); // different input
        g.add_node(OpType::Linear, vec![vec![256]], vec![vec![64]], vec![1]).unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(!patterns.contains(&FusionPattern::QKVProjection));
    }

    // ───────────────── Shared memory estimation ─────────────────

    #[test]
    fn test_shared_mem_matmul_bias() {
        let smem = estimate_shared_memory(FusionPattern::MatMulBias, 256);
        assert_eq!(smem, 2 * 32 * 32 * 4);
    }

    #[test]
    fn test_shared_mem_layer_norm_residual() {
        let smem = estimate_shared_memory(FusionPattern::LayerNormResidual, 256);
        assert_eq!(smem, 256 * 4);
    }

    #[test]
    fn test_shared_mem_attention_score_softmax() {
        let smem = estimate_shared_memory(FusionPattern::AttentionScoreSoftmax, 128);
        assert_eq!(smem, 128 * 4 * 2);
    }

    #[test]
    fn test_shared_mem_qkv() {
        let smem = estimate_shared_memory(FusionPattern::QKVProjection, 256);
        assert_eq!(smem, 256 * 4 * 3);
    }

    #[test]
    fn test_shared_mem_glu() {
        let smem = estimate_shared_memory(FusionPattern::GatedLinearUnit, 256);
        assert_eq!(smem, 256 * 4 * 2);
    }

    #[test]
    fn test_shared_mem_rmsnorm_linear() {
        let smem = estimate_shared_memory(FusionPattern::RMSNormLinear, 256);
        assert_eq!(smem, 256 * 4);
    }

    #[test]
    fn test_shared_mem_small_n_clamped() {
        // n=16 → threads_per_block=16
        let smem = estimate_shared_memory(FusionPattern::LayerNormResidual, 16);
        assert_eq!(smem, 16 * 4);
    }

    #[test]
    fn test_shared_mem_large_n_capped_at_256_threads() {
        let smem = estimate_shared_memory(FusionPattern::LayerNormResidual, 4096);
        assert_eq!(smem, 256 * 4); // capped at 256 threads
    }

    // ───────────────── Register pressure estimation ─────────────────

    #[test]
    fn test_register_pressure_values() {
        assert_eq!(estimate_register_pressure(FusionPattern::MatMulBias), 16);
        assert_eq!(estimate_register_pressure(FusionPattern::MatMulBiasReLU), 18);
        assert_eq!(estimate_register_pressure(FusionPattern::LayerNormResidual), 12);
        assert_eq!(estimate_register_pressure(FusionPattern::AttentionScoreSoftmax), 20);
        assert_eq!(estimate_register_pressure(FusionPattern::QKVProjection), 24);
        assert_eq!(estimate_register_pressure(FusionPattern::GatedLinearUnit), 22);
        assert_eq!(estimate_register_pressure(FusionPattern::RMSNormLinear), 14);
    }

    #[test]
    fn test_all_register_pressures_within_default_limit() {
        let cfg = KernelFusionConfig::default();
        for pattern in [
            FusionPattern::MatMulBias,
            FusionPattern::MatMulBiasReLU,
            FusionPattern::LayerNormResidual,
            FusionPattern::AttentionScoreSoftmax,
            FusionPattern::QKVProjection,
            FusionPattern::GatedLinearUnit,
            FusionPattern::RMSNormLinear,
        ] {
            assert!(estimate_register_pressure(pattern) <= cfg.register_pressure_limit);
        }
    }

    // ───────────────── apply_fusion ─────────────────

    #[test]
    fn test_apply_fusion_matmul_bias() {
        let g = make_matmul_bias_graph();
        let cfg = KernelFusionConfig::default();
        let fk = apply_fusion(&g, FusionPattern::MatMulBias, &cfg).unwrap();
        assert_eq!(fk.pattern, FusionPattern::MatMulBias);
        assert!(fk.estimated_speedup > 1.0);
        assert!(!fk.cuda_source.is_empty());
        assert!(fk.shared_mem_bytes > 0);
    }

    #[test]
    fn test_apply_fusion_all_patterns() {
        let g = make_matmul_bias_graph();
        let cfg = KernelFusionConfig::default();
        for pattern in [
            FusionPattern::MatMulBias,
            FusionPattern::MatMulBiasReLU,
            FusionPattern::LayerNormResidual,
            FusionPattern::AttentionScoreSoftmax,
            FusionPattern::QKVProjection,
            FusionPattern::GatedLinearUnit,
            FusionPattern::RMSNormLinear,
        ] {
            let fk = apply_fusion(&g, pattern, &cfg).unwrap();
            assert_eq!(fk.pattern, pattern);
            assert!(fk.estimated_speedup >= 1.0);
        }
    }

    #[test]
    fn test_apply_fusion_empty_graph_fails() {
        let g = FusionGraph::new();
        let cfg = KernelFusionConfig::default();
        assert!(apply_fusion(&g, FusionPattern::MatMulBias, &cfg).is_err());
    }

    #[test]
    fn test_apply_fusion_invalid_config_fails() {
        let g = make_matmul_bias_graph();
        let cfg = KernelFusionConfig { max_fused_ops: 0, ..Default::default() };
        assert!(apply_fusion(&g, FusionPattern::MatMulBias, &cfg).is_err());
    }

    #[test]
    fn test_apply_fusion_shared_mem_exceeded() {
        let g = make_matmul_bias_graph();
        let cfg = KernelFusionConfig { shared_memory_limit: 1, ..Default::default() };
        assert!(apply_fusion(&g, FusionPattern::MatMulBias, &cfg).is_err());
    }

    #[test]
    fn test_apply_fusion_register_pressure_exceeded() {
        let g = make_matmul_bias_graph();
        let cfg = KernelFusionConfig { register_pressure_limit: 1, ..Default::default() };
        assert!(apply_fusion(&g, FusionPattern::MatMulBias, &cfg).is_err());
    }

    #[test]
    fn test_apply_fusion_launch_config_block_capped() {
        let g = make_matmul_bias_graph();
        let cfg = KernelFusionConfig::default();
        let fk = apply_fusion(&g, FusionPattern::MatMulBias, &cfg).unwrap();
        assert!(fk.launch_config.block.0 <= 256);
    }

    // ───────────────── Speedup estimation ─────────────────

    #[test]
    fn test_estimate_speedup_single_op() {
        let fk = FusedKernel {
            cuda_source: String::new(),
            launch_config: LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1) },
            shared_mem_bytes: 1024,
            pattern: FusionPattern::MatMulBias,
            estimated_speedup: 1.1,
        };
        assert_eq!(estimate_fusion_speedup(1, &fk), 1.0);
    }

    #[test]
    fn test_estimate_speedup_two_ops() {
        let fk = FusedKernel {
            cuda_source: String::new(),
            launch_config: LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1) },
            shared_mem_bytes: 1024,
            pattern: FusionPattern::MatMulBias,
            estimated_speedup: 1.1,
        };
        let s = estimate_fusion_speedup(2, &fk);
        assert!((s - 1.1).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_speedup_three_ops_bonus() {
        let fk = FusedKernel {
            cuda_source: String::new(),
            launch_config: LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1) },
            shared_mem_bytes: 1024,
            pattern: FusionPattern::MatMulBiasReLU,
            estimated_speedup: 1.25,
        };
        let s = estimate_fusion_speedup(3, &fk);
        assert!(s > 1.25);
    }

    #[test]
    fn test_estimate_speedup_zero_ops() {
        let fk = FusedKernel {
            cuda_source: String::new(),
            launch_config: LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1) },
            shared_mem_bytes: 0,
            pattern: FusionPattern::MatMulBias,
            estimated_speedup: 1.1,
        };
        assert_eq!(estimate_fusion_speedup(0, &fk), 1.0);
    }

    // ───────────────── fused_matmul_bias correctness ─────────────────

    #[test]
    fn test_fused_matmul_bias_identity() {
        // A = I (2×2), B = I (2×2), bias = [1, 2]
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let bias = [1.0, 2.0];
        let mut out = [0.0f32; 4];
        fused_matmul_bias(&a, &b, &bias, &mut out, 2, 2, 2).unwrap();
        assert!(approx_eq(&out, &[2.0, 2.0, 1.0, 3.0], TOL));
    }

    #[test]
    fn test_fused_matmul_bias_1x1() {
        let a = [3.0];
        let b = [2.0];
        let bias = [0.5];
        let mut out = [0.0f32; 1];
        fused_matmul_bias(&a, &b, &bias, &mut out, 1, 1, 1).unwrap();
        assert!((out[0] - 6.5).abs() < TOL);
    }

    #[test]
    fn test_fused_matmul_bias_rectangular() {
        // A: 1×3, B: 3×2, bias: [0.1, 0.2]
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // 3×2
        let bias = [0.1, 0.2];
        let mut out = [0.0f32; 2];
        fused_matmul_bias(&a, &b, &bias, &mut out, 1, 2, 3).unwrap();
        // 1*1+2*2+3*3 = 14; 1*4+2*5+3*6 = 32
        assert!((out[0] - 14.1).abs() < TOL);
        assert!((out[1] - 32.2).abs() < TOL);
    }

    #[test]
    fn test_fused_matmul_bias_zero_dims() {
        let mut out = [0.0f32; 1];
        assert!(fused_matmul_bias(&[], &[], &[], &mut out, 0, 1, 1).is_err());
    }

    #[test]
    fn test_fused_matmul_bias_short_a() {
        let a = [1.0]; // too short for 2×2
        let b = [1.0, 0.0, 0.0, 1.0];
        let bias = [0.0, 0.0];
        let mut out = [0.0f32; 4];
        assert!(fused_matmul_bias(&a, &b, &bias, &mut out, 2, 2, 2).is_err());
    }

    #[test]
    fn test_fused_matmul_bias_short_b() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0]; // too short
        let bias = [0.0, 0.0];
        let mut out = [0.0f32; 4];
        assert!(fused_matmul_bias(&a, &b, &bias, &mut out, 2, 2, 2).is_err());
    }

    #[test]
    fn test_fused_matmul_bias_short_bias() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let bias = [0.0]; // too short
        let mut out = [0.0f32; 4];
        assert!(fused_matmul_bias(&a, &b, &bias, &mut out, 2, 2, 2).is_err());
    }

    #[test]
    fn test_fused_matmul_bias_short_output() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let bias = [0.0, 0.0];
        let mut out = [0.0f32; 2]; // too short for 2×2
        assert!(fused_matmul_bias(&a, &b, &bias, &mut out, 2, 2, 2).is_err());
    }

    // ───── fused_matmul_bias vs sequential ─────

    #[test]
    fn test_fused_matmul_bias_matches_sequential() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3×2
        let bias = [0.5, -0.5];

        // Sequential: matmul then add bias
        let mut seq_out = [0.0f32; 4]; // 2×2
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0f32;
                for p in 0..3 {
                    sum += a[i * 3 + p] * b[p * 2 + j];
                }
                seq_out[i * 2 + j] = sum + bias[j];
            }
        }

        // Fused
        let mut fused_out = [0.0f32; 4];
        fused_matmul_bias(&a, &b, &bias, &mut fused_out, 2, 2, 3).unwrap();

        assert!(approx_eq(&fused_out, &seq_out, TOL));
    }

    // ───────────────── fused_matmul_bias_relu correctness ─────────────────

    #[test]
    fn test_fused_matmul_bias_relu_positive() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let bias = [1.0, 2.0];
        let mut out = [0.0f32; 4];
        fused_matmul_bias_relu(&a, &b, &bias, &mut out, 2, 2, 2).unwrap();
        // All positive, so ReLU is identity
        assert!(approx_eq(&out, &[2.0, 2.0, 1.0, 3.0], TOL));
    }

    #[test]
    fn test_fused_matmul_bias_relu_clamps_negative() {
        let a = [1.0];
        let b = [1.0];
        let bias = [-5.0]; // result = 1*1 + (-5) = -4
        let mut out = [0.0f32; 1];
        fused_matmul_bias_relu(&a, &b, &bias, &mut out, 1, 1, 1).unwrap();
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn test_fused_matmul_bias_relu_vs_sequential() {
        let a = [1.0, -1.0, 2.0, -2.0]; // 2×2
        let b = [3.0, -3.0, 1.0, -1.0]; // 2×2
        let bias = [0.5, -10.0];

        // Sequential
        let mut seq_out = [0.0f32; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0f32;
                for p in 0..2 {
                    sum += a[i * 2 + p] * b[p * 2 + j];
                }
                seq_out[i * 2 + j] = (sum + bias[j]).max(0.0);
            }
        }

        let mut fused_out = [0.0f32; 4];
        fused_matmul_bias_relu(&a, &b, &bias, &mut fused_out, 2, 2, 2).unwrap();
        assert!(approx_eq(&fused_out, &seq_out, TOL));
    }

    #[test]
    fn test_fused_matmul_bias_relu_zero_dims() {
        let mut out = [0.0f32; 1];
        assert!(fused_matmul_bias_relu(&[], &[], &[], &mut out, 0, 1, 1).is_err());
    }

    // ───────────────── fused_layer_norm_residual ─────────────────

    #[test]
    fn test_fused_layer_norm_residual_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let residual = [0.0, 0.0, 0.0, 0.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut out = [0.0f32; 4];
        fused_layer_norm_residual(&input, &residual, &gamma, &beta, &mut out, EPS).unwrap();
        // Mean = 2.5, var = 1.25, inv_std ≈ 0.894
        // Normalized should sum to ~0
        let sum: f32 = out.iter().sum();
        assert!(sum.abs() < 1e-3);
    }

    #[test]
    fn test_fused_layer_norm_residual_with_residual() {
        let input = [1.0, 2.0];
        let residual = [1.0, 0.0];
        let gamma = [1.0; 2];
        let beta = [0.0; 2];
        let mut out = [0.0f32; 2];
        fused_layer_norm_residual(&input, &residual, &gamma, &beta, &mut out, EPS).unwrap();
        // input+residual = [2, 2], mean=2, var=0 → normalized = [0, 0]
        assert!(out[0].abs() < 1e-3);
        assert!(out[1].abs() < 1e-3);
    }

    #[test]
    fn test_fused_layer_norm_residual_with_gamma_beta() {
        let input = [1.0, 3.0];
        let residual = [0.0, 0.0];
        let gamma = [2.0, 2.0];
        let beta = [1.0, 1.0];
        let mut out = [0.0f32; 2];
        fused_layer_norm_residual(&input, &residual, &gamma, &beta, &mut out, EPS).unwrap();
        // Normalized = [-1, 1] (approx), scaled by 2 → [-2, 2], +1 → [-1, 3]
        assert!((out[0] - (-1.0)).abs() < 0.1);
        assert!((out[1] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_fused_layer_norm_residual_empty_input() {
        let mut out = [0.0f32; 1];
        assert!(fused_layer_norm_residual(&[], &[], &[], &[], &mut out, EPS).is_err());
    }

    #[test]
    fn test_fused_layer_norm_residual_mismatched_residual() {
        let input = [1.0, 2.0];
        let residual = [1.0];
        let gamma = [1.0; 2];
        let beta = [0.0; 2];
        let mut out = [0.0f32; 2];
        assert!(
            fused_layer_norm_residual(&input, &residual, &gamma, &beta, &mut out, EPS).is_err()
        );
    }

    #[test]
    fn test_fused_layer_norm_residual_mismatched_gamma() {
        let input = [1.0, 2.0];
        let residual = [0.0, 0.0];
        let gamma = [1.0];
        let beta = [0.0; 2];
        let mut out = [0.0f32; 2];
        assert!(
            fused_layer_norm_residual(&input, &residual, &gamma, &beta, &mut out, EPS).is_err()
        );
    }

    #[test]
    fn test_fused_layer_norm_residual_mismatched_beta() {
        let input = [1.0, 2.0];
        let residual = [0.0, 0.0];
        let gamma = [1.0; 2];
        let beta = [0.0];
        let mut out = [0.0f32; 2];
        assert!(
            fused_layer_norm_residual(&input, &residual, &gamma, &beta, &mut out, EPS).is_err()
        );
    }

    #[test]
    fn test_fused_layer_norm_residual_short_output() {
        let input = [1.0, 2.0];
        let residual = [0.0, 0.0];
        let gamma = [1.0; 2];
        let beta = [0.0; 2];
        let mut out = [0.0f32; 1];
        assert!(
            fused_layer_norm_residual(&input, &residual, &gamma, &beta, &mut out, EPS).is_err()
        );
    }

    #[test]
    fn test_fused_layer_norm_residual_vs_sequential() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let residual = [0.5, -0.5, 0.5, -0.5];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];

        // Sequential: add residual, then layer norm
        let combined: Vec<f32> = input.iter().zip(&residual).map(|(a, b)| a + b).collect();
        let mean: f32 = combined.iter().sum::<f32>() / 4.0;
        let var: f32 = combined.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 4.0;
        let inv_std = 1.0 / (var + EPS).sqrt();
        let seq_out: Vec<f32> = combined
            .iter()
            .enumerate()
            .map(|(i, x)| (x - mean) * inv_std * gamma[i] + beta[i])
            .collect();

        let mut fused_out = [0.0f32; 4];
        fused_layer_norm_residual(&input, &residual, &gamma, &beta, &mut fused_out, EPS).unwrap();

        assert!(approx_eq(&fused_out, &seq_out, TOL));
    }

    // ───────────────── fused_attention_score_softmax ─────────────────

    #[test]
    fn test_fused_attention_score_softmax_basic() {
        let q = [1.0, 0.0]; // head_dim = 2
        let k = [1.0, 0.0, 0.0, 1.0]; // seq_len = 2
        let mut out = [0.0f32; 2];
        fused_attention_score_softmax(&q, &k, 1.0, &[], &mut out).unwrap();
        // scores: [1.0, 0.0] → softmax
        assert!(out[0] > out[1]); // first should be larger
        assert!((out[0] + out[1] - 1.0).abs() < TOL); // sums to 1
    }

    #[test]
    fn test_fused_attention_score_softmax_uniform() {
        let q = [1.0, 1.0];
        let k = [1.0, 1.0, 1.0, 1.0]; // both keys same
        let mut out = [0.0f32; 2];
        fused_attention_score_softmax(&q, &k, 0.5, &[], &mut out).unwrap();
        // Equal scores → uniform softmax
        assert!((out[0] - 0.5).abs() < TOL);
        assert!((out[1] - 0.5).abs() < TOL);
    }

    #[test]
    fn test_fused_attention_score_softmax_with_mask() {
        let q = [1.0, 0.0];
        let k = [1.0, 0.0, 0.0, 1.0];
        let mask = [0.0, -1e9]; // mask out second position
        let mut out = [0.0f32; 2];
        fused_attention_score_softmax(&q, &k, 1.0, &mask, &mut out).unwrap();
        assert!(out[0] > 0.99); // nearly all attention on first
        assert!(out[1] < 0.01);
    }

    #[test]
    fn test_fused_attention_score_softmax_sums_to_one() {
        let q = [0.5, 0.3, 0.1];
        let k = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // 3×3, identity
        let mut out = [0.0f32; 3];
        fused_attention_score_softmax(&q, &k, 1.0, &[], &mut out).unwrap();
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
    }

    #[test]
    fn test_fused_attention_score_softmax_empty_q() {
        let mut out = [0.0f32; 1];
        assert!(fused_attention_score_softmax(&[], &[1.0], 1.0, &[], &mut out).is_err());
    }

    #[test]
    fn test_fused_attention_score_softmax_empty_k() {
        let q = [1.0];
        let mut out = [0.0f32; 1];
        assert!(fused_attention_score_softmax(&q, &[], 1.0, &[], &mut out).is_err());
    }

    #[test]
    fn test_fused_attention_score_softmax_k_not_multiple() {
        let q = [1.0, 0.0]; // head_dim=2
        let k = [1.0, 0.0, 0.0]; // not a multiple of 2
        let mut out = [0.0f32; 1];
        assert!(fused_attention_score_softmax(&q, &k, 1.0, &[], &mut out).is_err());
    }

    #[test]
    fn test_fused_attention_score_softmax_mask_wrong_len() {
        let q = [1.0, 0.0];
        let k = [1.0, 0.0, 0.0, 1.0];
        let mask = [0.0]; // wrong length
        let mut out = [0.0f32; 2];
        assert!(fused_attention_score_softmax(&q, &k, 1.0, &mask, &mut out).is_err());
    }

    #[test]
    fn test_fused_attention_score_softmax_short_output() {
        let q = [1.0, 0.0];
        let k = [1.0, 0.0, 0.0, 1.0];
        let mut out = [0.0f32; 1]; // too short
        assert!(fused_attention_score_softmax(&q, &k, 1.0, &[], &mut out).is_err());
    }

    #[test]
    fn test_fused_attention_score_softmax_vs_sequential() {
        let q = [0.5, 0.3];
        let k = [1.0, 0.0, 0.0, 1.0, 0.5, 0.5]; // 3×2
        let scale = 0.707;

        // Sequential: compute scores
        let seq_len = 3;
        let head_dim = 2;
        let mut scores = vec![0.0f32; seq_len];
        for i in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[d] * k[i * head_dim + d];
            }
            scores[i] = dot * scale;
        }
        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let seq_out: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let mut fused_out = [0.0f32; 3];
        fused_attention_score_softmax(&q, &k, scale, &[], &mut fused_out).unwrap();
        assert!(approx_eq(&fused_out, &seq_out, TOL));
    }

    // ───────────────── FusedKernel struct ─────────────────

    #[test]
    fn test_fused_kernel_debug() {
        let fk = FusedKernel {
            cuda_source: "test".into(),
            launch_config: LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1) },
            shared_mem_bytes: 1024,
            pattern: FusionPattern::MatMulBias,
            estimated_speedup: 1.1,
        };
        let dbg = format!("{fk:?}");
        assert!(dbg.contains("MatMulBias"));
    }

    #[test]
    fn test_launch_config_eq() {
        let a = LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1) };
        let b = LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1) };
        assert_eq!(a, b);
    }

    #[test]
    fn test_launch_config_ne() {
        let a = LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1) };
        let b = LaunchConfig { grid: (2, 1, 1), block: (256, 1, 1) };
        assert_ne!(a, b);
    }

    // ───────────────── KernelFusionError ─────────────────

    #[test]
    fn test_error_display_invalid_graph() {
        let e = KernelFusionError::InvalidGraph("cycle detected".into());
        assert!(e.to_string().contains("cycle detected"));
    }

    #[test]
    fn test_error_display_invalid_config() {
        let e = KernelFusionError::InvalidConfig("bad value".into());
        assert!(e.to_string().contains("bad value"));
    }

    #[test]
    fn test_error_display_pattern_not_applicable() {
        let e = KernelFusionError::PatternNotApplicable("too much smem".into());
        assert!(e.to_string().contains("too much smem"));
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let e = KernelFusionError::DimensionMismatch { expected: 10, got: 5 };
        let s = e.to_string();
        assert!(s.contains("10") && s.contains("5"));
    }

    #[test]
    fn test_error_display_empty_input() {
        let e = KernelFusionError::EmptyInput;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn test_error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(KernelFusionError::EmptyInput);
        assert!(!e.to_string().is_empty());
    }

    // ───────────────── CUDA source generation ─────────────────

    #[test]
    fn test_generate_cuda_source_matmul_bias() {
        let src = generate_cuda_source(FusionPattern::MatMulBias);
        assert!(src.contains("fused_matmul_bias"));
    }

    #[test]
    fn test_generate_cuda_source_matmul_bias_relu() {
        let src = generate_cuda_source(FusionPattern::MatMulBiasReLU);
        assert!(src.contains("relu"));
    }

    #[test]
    fn test_generate_cuda_source_layer_norm_residual() {
        let src = generate_cuda_source(FusionPattern::LayerNormResidual);
        assert!(src.contains("layer_norm_residual"));
    }

    #[test]
    fn test_generate_cuda_source_attention_softmax() {
        let src = generate_cuda_source(FusionPattern::AttentionScoreSoftmax);
        assert!(src.contains("attention_score_softmax"));
    }

    #[test]
    fn test_generate_cuda_source_qkv() {
        let src = generate_cuda_source(FusionPattern::QKVProjection);
        assert!(src.contains("qkv_projection"));
    }

    #[test]
    fn test_generate_cuda_source_glu() {
        let src = generate_cuda_source(FusionPattern::GatedLinearUnit);
        assert!(src.contains("glu"));
    }

    #[test]
    fn test_generate_cuda_source_rmsnorm_linear() {
        let src = generate_cuda_source(FusionPattern::RMSNormLinear);
        assert!(src.contains("rmsnorm_linear"));
    }

    #[test]
    fn test_all_patterns_produce_nonempty_source() {
        for pattern in [
            FusionPattern::MatMulBias,
            FusionPattern::MatMulBiasReLU,
            FusionPattern::LayerNormResidual,
            FusionPattern::AttentionScoreSoftmax,
            FusionPattern::QKVProjection,
            FusionPattern::GatedLinearUnit,
            FusionPattern::RMSNormLinear,
        ] {
            assert!(!generate_cuda_source(pattern).is_empty(), "empty source for {pattern}");
        }
    }

    // ───────────────── FusionNode ─────────────────

    #[test]
    fn test_fusion_node_fields() {
        let node = FusionNode {
            id: 0,
            op_type: OpType::MatMul,
            input_shapes: vec![vec![4, 8], vec![8, 16]],
            output_shapes: vec![vec![4, 16]],
            inputs: vec![],
        };
        assert_eq!(node.id, 0);
        assert_eq!(node.op_type, OpType::MatMul);
        assert_eq!(node.input_shapes.len(), 2);
        assert_eq!(node.output_shapes.len(), 1);
    }

    #[test]
    fn test_fusion_node_clone() {
        let node = FusionNode {
            id: 1,
            op_type: OpType::ReLU,
            input_shapes: vec![vec![256]],
            output_shapes: vec![vec![256]],
            inputs: vec![0],
        };
        let cloned = node.clone();
        assert_eq!(cloned.id, node.id);
        assert_eq!(cloned.op_type, node.op_type);
    }

    // ───────────────── Already-fused / single-op edge cases ─────────────────

    #[test]
    fn test_detect_already_fused_no_double_match() {
        // A single MatMul node: no patterns
        let mut g = FusionGraph::new();
        g.add_node(OpType::MatMul, vec![vec![4, 8]], vec![vec![4, 16]], vec![]).unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_detect_long_chain() {
        // MatMul → BiasAdd → ReLU → Softmax (should find MatMulBias + MatMulBiasReLU)
        let mut g = FusionGraph::new();
        g.add_node(OpType::MatMul, vec![vec![4, 8]], vec![vec![4, 16]], vec![]).unwrap();
        g.add_node(OpType::BiasAdd, vec![vec![4, 16]], vec![vec![4, 16]], vec![0]).unwrap();
        g.add_node(OpType::ReLU, vec![vec![4, 16]], vec![vec![4, 16]], vec![1]).unwrap();
        g.add_node(OpType::Softmax, vec![vec![4, 16]], vec![vec![4, 16]], vec![2]).unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.contains(&FusionPattern::MatMulBias));
        assert!(patterns.contains(&FusionPattern::MatMulBiasReLU));
        // ReLU → Softmax is NOT a recognized fusion
        assert!(!patterns.contains(&FusionPattern::AttentionScoreSoftmax));
    }

    // ───────────────── End-to-end: detect + apply ─────────────────

    #[test]
    fn test_end_to_end_detect_and_apply() {
        let g = make_matmul_bias_graph();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(!patterns.is_empty());
        for p in &patterns {
            let fk = apply_fusion(&g, *p, &cfg).unwrap();
            assert!(fk.estimated_speedup >= 1.0);
        }
    }

    #[test]
    fn test_end_to_end_complex_graph() {
        let mut g = FusionGraph::new();
        // RMSNorm → Linear chain
        g.add_node(OpType::RMSNorm, vec![vec![256]], vec![vec![256]], vec![]).unwrap();
        g.add_node(OpType::Linear, vec![vec![256]], vec![vec![512]], vec![0]).unwrap();
        let cfg = KernelFusionConfig::default();
        let patterns = detect_fusion_opportunities(&g, &cfg);
        assert!(patterns.contains(&FusionPattern::RMSNormLinear));
        let fk = apply_fusion(&g, FusionPattern::RMSNormLinear, &cfg).unwrap();
        assert_eq!(fk.pattern, FusionPattern::RMSNormLinear);
    }
}
