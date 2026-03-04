//! NEON-optimized pipeline scheduler for Apple Silicon inference.
//!
//! Provides an efficient multi-stage pipeline that dispatches compute kernels
//! (matmul, layer-norm, activations, softmax, element-wise ops) with
//! prefetch-friendly batching tuned for ARM NEON `float32x4` lanes.

use std::fmt;

// ── Configuration ───────────────────────────────────────────────────────

/// Tunable knobs for the NEON pipeline scheduler.
#[derive(Debug, Clone)]
pub struct NeonPipelineConfig {
    pub num_stages: usize,
    pub prefetch_distance: usize,
    pub batch_size: usize,
    pub enable_fma: bool,
}

impl Default for NeonPipelineConfig {
    fn default() -> Self {
        Self { num_stages: 4, prefetch_distance: 2, batch_size: 32, enable_fma: true }
    }
}

// ── Operations ──────────────────────────────────────────────────────────

/// Activation function variants supported by the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    ReLU,
    GeLU,
    SiLU,
    Tanh,
    Sigmoid,
}

impl fmt::Display for ActivationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReLU => write!(f, "ReLU"),
            Self::GeLU => write!(f, "GeLU"),
            Self::SiLU => write!(f, "SiLU"),
            Self::Tanh => write!(f, "Tanh"),
            Self::Sigmoid => write!(f, "Sigmoid"),
        }
    }
}

/// A single compute operation that a pipeline stage will execute.
#[derive(Debug, Clone)]
pub enum PipelineOp {
    MatMul { m: usize, n: usize, k: usize },
    LayerNorm { dim: usize },
    Activation { kind: ActivationKind },
    Softmax { dim: usize },
    Add,
    Scale { factor: f32 },
}

// ── Pipeline stage ──────────────────────────────────────────────────────

/// Named wrapper around a [`PipelineOp`].
#[derive(Debug, Clone)]
pub struct NeonPipelineStage {
    pub name: String,
    pub op: PipelineOp,
}

// ── Scalar op helpers (portable – no intrinsics) ────────────────────────

fn execute_matmul(input: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    // Interpret `input` as an (m × k) matrix, multiply by an identity-like
    // (k × n) weight placeholder → output (m × n).  Real kernels would use
    // NEON intrinsics; here we keep it deterministic for testing.
    let mut out = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0_f32;
            for i in 0..k {
                let a = input.get(row * k + i).copied().unwrap_or(0.0);
                // Identity-like weight: 1.0 on the diagonal, 0 elsewhere.
                let w = if i == col { 1.0 } else { 0.0 };
                acc += a * w;
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn execute_layer_norm(input: &[f32], dim: usize) -> Vec<f32> {
    let eps = 1e-5_f32;
    let num_groups = input.len() / dim;
    let mut out = vec![0.0_f32; input.len()];
    for g in 0..num_groups {
        let start = g * dim;
        let end = start + dim;
        let slice = &input[start..end];
        let mean = slice.iter().sum::<f32>() / dim as f32;
        let var = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for (i, &v) in slice.iter().enumerate() {
            out[start + i] = (v - mean) * inv_std;
        }
    }
    out
}

fn apply_activation(input: &[f32], kind: ActivationKind) -> Vec<f32> {
    input
        .iter()
        .map(|&x| match kind {
            ActivationKind::ReLU => x.max(0.0),
            ActivationKind::GeLU => {
                0.5 * x
                    * (1.0
                        + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3)))
                            .tanh())
            }
            ActivationKind::SiLU => x * (1.0 / (1.0 + (-x).exp())),
            ActivationKind::Tanh => x.tanh(),
            ActivationKind::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        })
        .collect()
}

fn execute_softmax(input: &[f32], dim: usize) -> Vec<f32> {
    let num_groups = input.len() / dim;
    let mut out = vec![0.0_f32; input.len()];
    for g in 0..num_groups {
        let start = g * dim;
        let end = start + dim;
        let slice = &input[start..end];
        let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = slice.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (i, &e) in exps.iter().enumerate() {
            out[start + i] = e / sum;
        }
    }
    out
}

fn execute_add(input: &[f32]) -> Vec<f32> {
    // Element-wise add with self (double).
    input.iter().map(|&x| x + x).collect()
}

fn execute_scale(input: &[f32], factor: f32) -> Vec<f32> {
    input.iter().map(|&x| x * factor).collect()
}

// ── Pipeline ────────────────────────────────────────────────────────────

/// Multi-stage NEON pipeline scheduler.
#[derive(Debug)]
pub struct NeonPipeline {
    config: NeonPipelineConfig,
    stages: Vec<NeonPipelineStage>,
}

impl NeonPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: NeonPipelineConfig) -> Self {
        Self { config, stages: Vec::new() }
    }

    /// Append a stage to the pipeline.
    pub fn add_stage(&mut self, stage: NeonPipelineStage) {
        self.stages.push(stage);
    }

    /// Execute all stages sequentially, feeding each stage's output into the
    /// next.
    pub fn execute(&self, input: &[f32]) -> Vec<f32> {
        let mut buf = input.to_vec();
        for stage in &self.stages {
            buf = match &stage.op {
                PipelineOp::MatMul { m, n, k } => execute_matmul(&buf, *m, *n, *k),
                PipelineOp::LayerNorm { dim } => execute_layer_norm(&buf, *dim),
                PipelineOp::Activation { kind } => apply_activation(&buf, *kind),
                PipelineOp::Softmax { dim } => execute_softmax(&buf, *dim),
                PipelineOp::Add => execute_add(&buf),
                PipelineOp::Scale { factor } => execute_scale(&buf, *factor),
            };
        }
        buf
    }

    /// Validate that the pipeline configuration is sane.
    pub fn validate(&self) -> Result<(), String> {
        if self.config.batch_size == 0 {
            return Err("batch_size must be > 0".into());
        }
        if self.config.num_stages == 0 {
            return Err("num_stages must be > 0".into());
        }
        if self.stages.is_empty() {
            return Err("pipeline has no stages".into());
        }
        for stage in &self.stages {
            match &stage.op {
                PipelineOp::MatMul { m, n, k } => {
                    if *m == 0 || *n == 0 || *k == 0 {
                        return Err(format!(
                            "stage '{}': MatMul dimensions must be > 0",
                            stage.name
                        ));
                    }
                }
                PipelineOp::LayerNorm { dim } | PipelineOp::Softmax { dim } => {
                    if *dim == 0 {
                        return Err(format!("stage '{}': dim must be > 0", stage.name));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Estimate the total FLOPs for one forward pass with the given
    /// `input_size`.
    pub fn estimate_flops(&self, input_size: usize) -> u64 {
        let mut flops: u64 = 0;
        for stage in &self.stages {
            flops += match &stage.op {
                // 2·m·n·k multiply-adds
                PipelineOp::MatMul { m, n, k } => 2 * (*m as u64) * (*n as u64) * (*k as u64),
                // mean + var + normalise  ≈ 5·dim per group
                PipelineOp::LayerNorm { dim: _ } => 5 * input_size as u64,
                // One op per element (approximate)
                PipelineOp::Activation { kind } => match kind {
                    ActivationKind::ReLU => input_size as u64,
                    _ => 5 * input_size as u64,
                },
                // exp + sum + div  ≈ 5 per element
                PipelineOp::Softmax { dim: _ } => 5 * input_size as u64,
                PipelineOp::Add => input_size as u64,
                PipelineOp::Scale { .. } => input_size as u64,
            };
        }
        flops
    }

    /// Number of stages currently in the pipeline.
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Remove all stages from the pipeline.
    pub fn clear(&mut self) {
        self.stages.clear();
    }

    /// Borrow the pipeline configuration.
    pub fn config(&self) -> &NeonPipelineConfig {
        &self.config
    }
}

// ── Free functions ──────────────────────────────────────────────────────

/// Suggest an optimised execution order by sorting cheaper ops (element-wise)
/// before expensive ones (matmul), returning index permutation into
/// `stages`.
pub fn optimize_stage_order(stages: &[NeonPipelineStage]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..stages.len()).collect();
    indices.sort_by_key(|&i| match &stages[i].op {
        PipelineOp::Scale { .. } | PipelineOp::Add => 0_u8,
        PipelineOp::Activation { .. } => 1,
        PipelineOp::LayerNorm { .. } => 2,
        PipelineOp::Softmax { .. } => 3,
        PipelineOp::MatMul { .. } => 4,
    });
    indices
}

/// Estimate peak memory (bytes) required to execute `stages` given an
/// `input_size` element buffer.
pub fn estimate_memory_bytes(stages: &[NeonPipelineStage], input_size: usize) -> usize {
    let elem = std::mem::size_of::<f32>();
    // We always keep the current buffer plus the output buffer.
    let mut max_bytes = input_size * elem;
    for stage in stages {
        let stage_out = match &stage.op {
            PipelineOp::MatMul { m, n, .. } => m * n,
            _ => input_size,
        };
        let needed = input_size * elem + stage_out * elem;
        if needed > max_bytes {
            max_bytes = needed;
        }
    }
    max_bytes
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a default pipeline with one stage.
    fn pipeline_with(op: PipelineOp) -> NeonPipeline {
        let mut p = NeonPipeline::new(NeonPipelineConfig::default());
        p.add_stage(NeonPipelineStage { name: "s0".into(), op });
        p
    }

    // ── Config defaults ────────────────────────────────────────────

    #[test]
    fn test_default_config_values() {
        let cfg = NeonPipelineConfig::default();
        assert_eq!(cfg.num_stages, 4);
        assert_eq!(cfg.prefetch_distance, 2);
        assert_eq!(cfg.batch_size, 32);
        assert!(cfg.enable_fma);
    }

    #[test]
    fn test_config_clone() {
        let cfg = NeonPipelineConfig::default();
        let cfg2 = cfg.clone();
        assert_eq!(cfg2.batch_size, cfg.batch_size);
    }

    // ── Construction / stage management ────────────────────────────

    #[test]
    fn test_new_pipeline_empty() {
        let p = NeonPipeline::new(NeonPipelineConfig::default());
        assert_eq!(p.num_stages(), 0);
    }

    #[test]
    fn test_add_single_stage() {
        let p = pipeline_with(PipelineOp::Add);
        assert_eq!(p.num_stages(), 1);
    }

    #[test]
    fn test_add_multiple_stages() {
        let mut p = NeonPipeline::new(NeonPipelineConfig::default());
        for i in 0..5 {
            p.add_stage(NeonPipelineStage { name: format!("s{i}"), op: PipelineOp::Add });
        }
        assert_eq!(p.num_stages(), 5);
    }

    #[test]
    fn test_clear_stages() {
        let mut p = pipeline_with(PipelineOp::Add);
        p.clear();
        assert_eq!(p.num_stages(), 0);
    }

    #[test]
    fn test_clear_then_rebuild() {
        let mut p = pipeline_with(PipelineOp::Add);
        p.clear();
        p.add_stage(NeonPipelineStage {
            name: "new".into(),
            op: PipelineOp::Scale { factor: 2.0 },
        });
        assert_eq!(p.num_stages(), 1);
        let out = p.execute(&[3.0]);
        assert!((out[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_accessor() {
        let cfg = NeonPipelineConfig { batch_size: 64, ..Default::default() };
        let p = NeonPipeline::new(cfg);
        assert_eq!(p.config().batch_size, 64);
    }

    // ── Validation ─────────────────────────────────────────────────

    #[test]
    fn test_validate_empty_pipeline() {
        let p = NeonPipeline::new(NeonPipelineConfig::default());
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_zero_batch_size() {
        let mut p = NeonPipeline::new(NeonPipelineConfig { batch_size: 0, ..Default::default() });
        p.add_stage(NeonPipelineStage { name: "s".into(), op: PipelineOp::Add });
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_zero_num_stages() {
        let mut p = NeonPipeline::new(NeonPipelineConfig { num_stages: 0, ..Default::default() });
        p.add_stage(NeonPipelineStage { name: "s".into(), op: PipelineOp::Add });
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_matmul_zero_dim() {
        let p = pipeline_with(PipelineOp::MatMul { m: 0, n: 4, k: 4 });
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_layernorm_zero_dim() {
        let p = pipeline_with(PipelineOp::LayerNorm { dim: 0 });
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_softmax_zero_dim() {
        let p = pipeline_with(PipelineOp::Softmax { dim: 0 });
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_good_pipeline() {
        let p = pipeline_with(PipelineOp::Scale { factor: 1.0 });
        assert!(p.validate().is_ok());
    }

    // ── Single-stage execution ─────────────────────────────────────

    #[test]
    fn test_execute_scale() {
        let p = pipeline_with(PipelineOp::Scale { factor: 3.0 });
        let out = p.execute(&[1.0, 2.0, 4.0]);
        assert_eq!(out, vec![3.0, 6.0, 12.0]);
    }

    #[test]
    fn test_execute_scale_zero() {
        let p = pipeline_with(PipelineOp::Scale { factor: 0.0 });
        let out = p.execute(&[5.0, -3.0]);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_execute_add() {
        let p = pipeline_with(PipelineOp::Add);
        let out = p.execute(&[1.0, 2.0]);
        assert_eq!(out, vec![2.0, 4.0]);
    }

    #[test]
    fn test_execute_relu() {
        let p = pipeline_with(PipelineOp::Activation { kind: ActivationKind::ReLU });
        let out = p.execute(&[-1.0, 0.0, 1.0, 5.0]);
        assert_eq!(out, vec![0.0, 0.0, 1.0, 5.0]);
    }

    #[test]
    fn test_execute_sigmoid() {
        let p = pipeline_with(PipelineOp::Activation { kind: ActivationKind::Sigmoid });
        let out = p.execute(&[0.0]);
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_execute_tanh() {
        let p = pipeline_with(PipelineOp::Activation { kind: ActivationKind::Tanh });
        let out = p.execute(&[0.0]);
        assert!(out[0].abs() < 1e-6);
    }

    #[test]
    fn test_execute_silu() {
        let p = pipeline_with(PipelineOp::Activation { kind: ActivationKind::SiLU });
        let out = p.execute(&[0.0]);
        assert!(out[0].abs() < 1e-6);
    }

    #[test]
    fn test_execute_gelu() {
        let p = pipeline_with(PipelineOp::Activation { kind: ActivationKind::GeLU });
        let out = p.execute(&[0.0]);
        assert!(out[0].abs() < 1e-6);
    }

    #[test]
    fn test_execute_layer_norm() {
        let p = pipeline_with(PipelineOp::LayerNorm { dim: 4 });
        let out = p.execute(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out.len(), 4);
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "layer-norm output should be zero-mean");
    }

    #[test]
    fn test_execute_softmax() {
        let p = pipeline_with(PipelineOp::Softmax { dim: 3 });
        let out = p.execute(&[1.0, 2.0, 3.0]);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1");
    }

    #[test]
    fn test_execute_softmax_uniform() {
        let p = pipeline_with(PipelineOp::Softmax { dim: 4 });
        let out = p.execute(&[0.0, 0.0, 0.0, 0.0]);
        for &v in &out {
            assert!((v - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_execute_matmul_identity() {
        // 2×2 input times identity-like 2×2 weight → same first columns
        let p = pipeline_with(PipelineOp::MatMul { m: 2, n: 2, k: 2 });
        let out = p.execute(&[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    // ── Multi-stage execution ──────────────────────────────────────

    #[test]
    fn test_two_stage_scale_add() {
        let mut p = NeonPipeline::new(NeonPipelineConfig::default());
        p.add_stage(NeonPipelineStage {
            name: "scale".into(),
            op: PipelineOp::Scale { factor: 2.0 },
        });
        p.add_stage(NeonPipelineStage { name: "add".into(), op: PipelineOp::Add });
        // 3.0 * 2.0 = 6.0 → 6.0 + 6.0 = 12.0
        let out = p.execute(&[3.0]);
        assert!((out[0] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_three_stage_chain() {
        let mut p = NeonPipeline::new(NeonPipelineConfig::default());
        p.add_stage(NeonPipelineStage {
            name: "scale".into(),
            op: PipelineOp::Scale { factor: 0.5 },
        });
        p.add_stage(NeonPipelineStage {
            name: "relu".into(),
            op: PipelineOp::Activation { kind: ActivationKind::ReLU },
        });
        p.add_stage(NeonPipelineStage { name: "add".into(), op: PipelineOp::Add });
        // -4 * 0.5 = -2 → relu → 0 → add → 0
        let out = p.execute(&[-4.0]);
        assert!(out[0].abs() < 1e-6);
    }

    #[test]
    fn test_multi_stage_with_layernorm_and_softmax() {
        let mut p = NeonPipeline::new(NeonPipelineConfig::default());
        p.add_stage(NeonPipelineStage { name: "ln".into(), op: PipelineOp::LayerNorm { dim: 4 } });
        p.add_stage(NeonPipelineStage { name: "sm".into(), op: PipelineOp::Softmax { dim: 4 } });
        let out = p.execute(&[1.0, 2.0, 3.0, 4.0]);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ── Empty pipeline / edge cases ────────────────────────────────

    #[test]
    fn test_execute_empty_pipeline() {
        let p = NeonPipeline::new(NeonPipelineConfig::default());
        let out = p.execute(&[1.0, 2.0, 3.0]);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_execute_empty_input() {
        let p = pipeline_with(PipelineOp::Scale { factor: 2.0 });
        let out = p.execute(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_execute_single_element() {
        let p = pipeline_with(PipelineOp::Scale { factor: -1.0 });
        let out = p.execute(&[7.0]);
        assert!((out[0] - (-7.0)).abs() < 1e-6);
    }

    #[test]
    fn test_execute_large_input() {
        let p = pipeline_with(PipelineOp::Scale { factor: 1.0 });
        let input: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let out = p.execute(&input);
        assert_eq!(out.len(), 1024);
        assert!((out[1023] - 1023.0).abs() < 1e-3);
    }

    // ── FLOPs estimation ───────────────────────────────────────────

    #[test]
    fn test_flops_matmul() {
        let p = pipeline_with(PipelineOp::MatMul { m: 4, n: 4, k: 4 });
        assert_eq!(p.estimate_flops(16), 2 * 4 * 4 * 4);
    }

    #[test]
    fn test_flops_add() {
        let p = pipeline_with(PipelineOp::Add);
        assert_eq!(p.estimate_flops(100), 100);
    }

    #[test]
    fn test_flops_scale() {
        let p = pipeline_with(PipelineOp::Scale { factor: 2.0 });
        assert_eq!(p.estimate_flops(50), 50);
    }

    #[test]
    fn test_flops_relu() {
        let p = pipeline_with(PipelineOp::Activation { kind: ActivationKind::ReLU });
        assert_eq!(p.estimate_flops(64), 64);
    }

    #[test]
    fn test_flops_gelu() {
        let p = pipeline_with(PipelineOp::Activation { kind: ActivationKind::GeLU });
        assert_eq!(p.estimate_flops(64), 5 * 64);
    }

    #[test]
    fn test_flops_layernorm() {
        let p = pipeline_with(PipelineOp::LayerNorm { dim: 32 });
        assert_eq!(p.estimate_flops(128), 5 * 128);
    }

    #[test]
    fn test_flops_softmax() {
        let p = pipeline_with(PipelineOp::Softmax { dim: 8 });
        assert_eq!(p.estimate_flops(8), 5 * 8);
    }

    #[test]
    fn test_flops_multistage_cumulative() {
        let mut p = NeonPipeline::new(NeonPipelineConfig::default());
        p.add_stage(NeonPipelineStage { name: "add".into(), op: PipelineOp::Add });
        p.add_stage(NeonPipelineStage {
            name: "scale".into(),
            op: PipelineOp::Scale { factor: 1.0 },
        });
        assert_eq!(p.estimate_flops(10), 20);
    }

    #[test]
    fn test_flops_empty_pipeline_zero() {
        let p = NeonPipeline::new(NeonPipelineConfig::default());
        assert_eq!(p.estimate_flops(256), 0);
    }

    // ── Memory estimation ──────────────────────────────────────────

    #[test]
    fn test_memory_no_stages() {
        let bytes = estimate_memory_bytes(&[], 16);
        assert_eq!(bytes, 16 * 4);
    }

    #[test]
    fn test_memory_scale_same_size() {
        let stages =
            vec![NeonPipelineStage { name: "s".into(), op: PipelineOp::Scale { factor: 1.0 } }];
        let bytes = estimate_memory_bytes(&stages, 8);
        // input(8*4) + output(8*4) = 64
        assert_eq!(bytes, 64);
    }

    #[test]
    fn test_memory_matmul_larger_output() {
        let stages = vec![NeonPipelineStage {
            name: "mm".into(),
            op: PipelineOp::MatMul { m: 4, n: 8, k: 4 },
        }];
        // input 16 elems * 4 + output (4*8=32) * 4 = 64 + 128 = 192
        let bytes = estimate_memory_bytes(&stages, 16);
        assert_eq!(bytes, 192);
    }

    // ── Stage ordering optimization ────────────────────────────────

    #[test]
    fn test_optimize_order_empty() {
        let order = optimize_stage_order(&[]);
        assert!(order.is_empty());
    }

    #[test]
    fn test_optimize_order_single() {
        let stages = vec![NeonPipelineStage {
            name: "mm".into(),
            op: PipelineOp::MatMul { m: 4, n: 4, k: 4 },
        }];
        assert_eq!(optimize_stage_order(&stages), vec![0]);
    }

    #[test]
    fn test_optimize_order_cheapest_first() {
        let stages = vec![
            NeonPipelineStage { name: "mm".into(), op: PipelineOp::MatMul { m: 4, n: 4, k: 4 } },
            NeonPipelineStage { name: "add".into(), op: PipelineOp::Add },
            NeonPipelineStage {
                name: "relu".into(),
                op: PipelineOp::Activation { kind: ActivationKind::ReLU },
            },
        ];
        let order = optimize_stage_order(&stages);
        // Add(0) < Activation(1) < MatMul(4) → indices [1, 2, 0]
        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn test_optimize_order_preserves_equal_priority() {
        let stages = vec![
            NeonPipelineStage { name: "a".into(), op: PipelineOp::Add },
            NeonPipelineStage { name: "s".into(), op: PipelineOp::Scale { factor: 1.0 } },
        ];
        let order = optimize_stage_order(&stages);
        // Both priority 0 → stable sort keeps original order.
        assert_eq!(order, vec![0, 1]);
    }

    // ── ActivationKind Display ─────────────────────────────────────

    #[test]
    fn test_activation_kind_display() {
        assert_eq!(format!("{}", ActivationKind::ReLU), "ReLU");
        assert_eq!(format!("{}", ActivationKind::GeLU), "GeLU");
        assert_eq!(format!("{}", ActivationKind::SiLU), "SiLU");
        assert_eq!(format!("{}", ActivationKind::Tanh), "Tanh");
        assert_eq!(format!("{}", ActivationKind::Sigmoid), "Sigmoid");
    }

    // ── Misc edge-case / regression tests ──────────────────────────

    #[test]
    fn test_layer_norm_multi_group() {
        let p = pipeline_with(PipelineOp::LayerNorm { dim: 2 });
        let out = p.execute(&[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(out.len(), 4);
        // Each group of 2 should be zero-mean.
        let m1 = (out[0] + out[1]) / 2.0;
        let m2 = (out[2] + out[3]) / 2.0;
        assert!(m1.abs() < 1e-5);
        assert!(m2.abs() < 1e-5);
    }

    #[test]
    fn test_softmax_two_groups() {
        let p = pipeline_with(PipelineOp::Softmax { dim: 2 });
        let out = p.execute(&[0.0, 0.0, 0.0, 0.0]);
        for chunk in out.chunks(2) {
            let s: f32 = chunk.iter().sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_scale_negative_factor() {
        let p = pipeline_with(PipelineOp::Scale { factor: -2.0 });
        let out = p.execute(&[1.0, -1.0]);
        assert!((out[0] - (-2.0)).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_debug_impl() {
        let p = pipeline_with(PipelineOp::Add);
        let dbg = format!("{p:?}");
        assert!(dbg.contains("NeonPipeline"));
    }
}
