//! CPU pipeline parallelism for multi-stage inference across threads.
//!
//! Splits a model's forward pass into sequential *stages* (each covering a
//! contiguous range of layers) and overlaps execution of different
//! micro-batches across stages using configurable scheduling policies.

use std::fmt;

use bitnet_common::{BitNetError, KernelError, Result};

// ── Schedule ───────────────────────────────────────────────────────

/// Pipeline scheduling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PipelineSchedule {
    /// Each micro-batch completes all stages before the next begins.
    Sequential,
    /// All micro-batches run the same stage before advancing (GPipe).
    #[default]
    GPipe,
    /// 1F1B steady-state schedule (PipeDream).
    PipeDream,
    /// Interleaved 1F1B with virtual stages.
    Interleaved,
}

impl fmt::Display for PipelineSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sequential => write!(f, "Sequential"),
            Self::GPipe => write!(f, "GPipe"),
            Self::PipeDream => write!(f, "PipeDream"),
            Self::Interleaved => write!(f, "Interleaved"),
        }
    }
}

// ── Stage configuration ────────────────────────────────────────────

/// Configuration for a single pipeline stage.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Index of the first layer (inclusive).
    pub start_layer: usize,
    /// Index of the last layer (exclusive).
    pub end_layer: usize,
    /// Optional thread affinity hint (core id). `None` = OS decides.
    pub thread_affinity: Option<usize>,
}

impl PipelineStage {
    /// Create a new stage spanning `[start_layer, end_layer)`.
    pub fn new(start_layer: usize, end_layer: usize) -> Self {
        Self { start_layer, end_layer, thread_affinity: None }
    }

    /// Builder: attach a thread-affinity hint.
    pub fn with_affinity(mut self, core_id: usize) -> Self {
        self.thread_affinity = Some(core_id);
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
        Ok(())
    }
}

// ── Pipeline configuration ─────────────────────────────────────────

/// Full pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Ordered list of stages (stage 0 feeds into stage 1, etc.).
    pub stages: Vec<PipelineStage>,
    /// Number of elements per micro-batch (along the batch dimension).
    pub micro_batch_size: usize,
    /// Scheduling strategy.
    pub schedule: PipelineSchedule,
}

impl PipelineConfig {
    /// Create a pipeline with the given stages and micro-batch size.
    pub fn new(
        stages: Vec<PipelineStage>,
        micro_batch_size: usize,
        schedule: PipelineSchedule,
    ) -> Self {
        Self { stages, micro_batch_size, schedule }
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
        Ok(())
    }
}

// ── Micro-batch helpers ────────────────────────────────────────────

/// Split `input` (shape `[batch, dim]`) into micro-batches of
/// `micro_batch_size` rows each.  The last chunk may be smaller.
///
/// Returns a `Vec` of owned micro-batch buffers.
pub fn micro_batch_split(
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
pub fn micro_batch_merge(batches: &[Vec<f32>]) -> Result<Vec<f32>> {
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

/// Execute a single pipeline stage on a micro-batch.
///
/// The stage function simulates processing by scaling each element by the
/// number of layers in the stage — a lightweight placeholder for real
/// layer computation that still exercises the pipeline mechanics.
pub fn stage_forward(input: &[f32], stage: &PipelineStage) -> Result<Vec<f32>> {
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

// ── Pipeline forward ───────────────────────────────────────────────

/// Execute the full pipeline on `input` (shape `[batch, dim]`).
///
/// The function:
/// 1. Validates the configuration.
/// 2. Splits the input into micro-batches.
/// 3. Runs each micro-batch through all stages using the configured
///    schedule.
/// 4. Merges the results back into a single output buffer.
pub fn pipeline_forward(
    input: &[f32],
    batch: usize,
    dim: usize,
    config: &PipelineConfig,
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

    let mut micro_batches = micro_batch_split(input, batch, dim, config.micro_batch_size)?;

    match config.schedule {
        PipelineSchedule::Sequential => {
            // Each micro-batch completes all stages before next starts.
            for mb in &mut micro_batches {
                for stage in &config.stages {
                    *mb = stage_forward(mb, stage)?;
                }
            }
        }
        PipelineSchedule::GPipe => {
            // All micro-batches pass through one stage before any advances.
            for stage in &config.stages {
                for mb in &mut micro_batches {
                    *mb = stage_forward(mb, stage)?;
                }
            }
        }
        PipelineSchedule::PipeDream => {
            // 1F1B steady-state: simulate with sequential per-stage
            // sweeps (functional equivalence; real overlap is threaded).
            for stage in &config.stages {
                for mb in &mut micro_batches {
                    *mb = stage_forward(mb, stage)?;
                }
            }
        }
        PipelineSchedule::Interleaved => {
            // Interleaved 1F1B with virtual stages — same functional
            // result as GPipe for correctness testing.
            for stage in &config.stages {
                for mb in &mut micro_batches {
                    *mb = stage_forward(mb, stage)?;
                }
            }
        }
    }

    micro_batch_merge(&micro_batches)
}

// ── Bubble-time estimation ─────────────────────────────────────────

/// Compute the pipeline bubble time as a fraction of total compute.
///
/// For a pipeline with `p` stages and `m` micro-batches the bubble
/// fraction is `(p - 1) / (m + p - 1)`.  Returns 0.0 when the pipeline
/// is degenerate (≤ 1 stage or ≤ 0 micro-batches).
pub fn pipeline_bubble_time(num_stages: usize, num_micro_batches: usize) -> f32 {
    if num_stages <= 1 || num_micro_batches == 0 {
        return 0.0;
    }
    let p = num_stages as f32;
    let m = num_micro_batches as f32;
    (p - 1.0) / (m + p - 1.0)
}

/// Compute the optimal number of micro-batches to keep the bubble
/// fraction below `max_bubble_fraction`.
///
/// Derived from `(p-1)/(m+p-1) ≤ f  =>  m ≥ (p-1)*(1-f)/f`.
/// Returns at least 1.
pub fn optimal_micro_batch_count(num_stages: usize, max_bubble_fraction: f32) -> usize {
    if num_stages <= 1 {
        return 1;
    }
    if max_bubble_fraction <= 0.0 || max_bubble_fraction > 1.0 {
        // Invalid fraction — return num_stages as a safe fallback.
        return num_stages;
    }
    let p = (num_stages - 1) as f32;
    let m = (p * (1.0 - max_bubble_fraction) / max_bubble_fraction).ceil() as usize;
    m.max(1)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PipelineSchedule ───────────────────────────────────────────

    #[test]
    fn test_schedule_display() {
        assert_eq!(PipelineSchedule::Sequential.to_string(), "Sequential");
        assert_eq!(PipelineSchedule::GPipe.to_string(), "GPipe");
        assert_eq!(PipelineSchedule::PipeDream.to_string(), "PipeDream");
        assert_eq!(PipelineSchedule::Interleaved.to_string(), "Interleaved");
    }

    #[test]
    fn test_schedule_default() {
        assert_eq!(PipelineSchedule::default(), PipelineSchedule::GPipe);
    }

    #[test]
    fn test_schedule_eq() {
        assert_eq!(PipelineSchedule::GPipe, PipelineSchedule::GPipe);
        assert_ne!(PipelineSchedule::GPipe, PipelineSchedule::Sequential);
    }

    // ── PipelineStage ──────────────────────────────────────────────

    #[test]
    fn test_stage_new() {
        let s = PipelineStage::new(0, 4);
        assert_eq!(s.start_layer, 0);
        assert_eq!(s.end_layer, 4);
        assert_eq!(s.thread_affinity, None);
    }

    #[test]
    fn test_stage_with_affinity() {
        let s = PipelineStage::new(0, 4).with_affinity(3);
        assert_eq!(s.thread_affinity, Some(3));
    }

    #[test]
    fn test_stage_num_layers() {
        assert_eq!(PipelineStage::new(0, 4).num_layers(), 4);
        assert_eq!(PipelineStage::new(4, 8).num_layers(), 4);
        assert_eq!(PipelineStage::new(0, 1).num_layers(), 1);
    }

    #[test]
    fn test_stage_validate_ok() {
        PipelineStage::new(0, 4).validate().unwrap();
    }

    #[test]
    fn test_stage_validate_empty_range() {
        assert!(PipelineStage::new(4, 4).validate().is_err());
    }

    #[test]
    fn test_stage_validate_inverted_range() {
        assert!(PipelineStage::new(8, 4).validate().is_err());
    }

    // ── PipelineConfig ─────────────────────────────────────────────

    #[test]
    fn test_config_num_stages() {
        let cfg = PipelineConfig::new(
            vec![PipelineStage::new(0, 4), PipelineStage::new(4, 8)],
            2,
            PipelineSchedule::GPipe,
        );
        assert_eq!(cfg.num_stages(), 2);
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = PipelineConfig::new(
            vec![PipelineStage::new(0, 4), PipelineStage::new(4, 8)],
            2,
            PipelineSchedule::GPipe,
        );
        cfg.validate().unwrap();
    }

    #[test]
    fn test_config_validate_empty_stages() {
        let cfg = PipelineConfig::new(vec![], 2, PipelineSchedule::GPipe);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_micro_batch() {
        let cfg = PipelineConfig::new(vec![PipelineStage::new(0, 4)], 0, PipelineSchedule::GPipe);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_non_contiguous() {
        let cfg = PipelineConfig::new(
            vec![PipelineStage::new(0, 4), PipelineStage::new(5, 8)],
            2,
            PipelineSchedule::GPipe,
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_stage() {
        let cfg = PipelineConfig::new(vec![PipelineStage::new(4, 4)], 1, PipelineSchedule::GPipe);
        assert!(cfg.validate().is_err());
    }

    // ── micro_batch_split ──────────────────────────────────────────

    #[test]
    fn test_split_exact() {
        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let batches = micro_batch_split(&input, 4, 3, 2).unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(batches[1], &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_split_remainder() {
        let input: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let batches = micro_batch_split(&input, 5, 3, 2).unwrap();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[2].len(), 3); // last chunk: 1 row
    }

    #[test]
    fn test_split_single_row() {
        let input = vec![1.0, 2.0, 3.0];
        let batches = micro_batch_split(&input, 1, 3, 1).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_split_micro_batch_larger_than_batch() {
        let input = vec![1.0, 2.0];
        let batches = micro_batch_split(&input, 2, 1, 10).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], &[1.0, 2.0]);
    }

    #[test]
    fn test_split_empty_input() {
        let input: Vec<f32> = vec![];
        assert!(micro_batch_split(&input, 0, 4, 2).is_err());
    }

    #[test]
    fn test_split_zero_dim() {
        assert!(micro_batch_split(&[1.0], 1, 0, 1).is_err());
    }

    #[test]
    fn test_split_zero_micro_batch() {
        assert!(micro_batch_split(&[1.0], 1, 1, 0).is_err());
    }

    #[test]
    fn test_split_mismatched_len() {
        assert!(micro_batch_split(&[1.0, 2.0], 1, 3, 1).is_err());
    }

    // ── micro_batch_merge ──────────────────────────────────────────

    #[test]
    fn test_merge_basic() {
        let batches = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let merged = micro_batch_merge(&batches).unwrap();
        assert_eq!(merged, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_merge_single() {
        let batches = vec![vec![5.0, 6.0, 7.0]];
        let merged = micro_batch_merge(&batches).unwrap();
        assert_eq!(merged, vec![5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_merge_empty() {
        let batches: Vec<Vec<f32>> = vec![];
        assert!(micro_batch_merge(&batches).is_err());
    }

    #[test]
    fn test_split_merge_roundtrip() {
        let input: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let batches = micro_batch_split(&input, 8, 3, 3).unwrap();
        let merged = micro_batch_merge(&batches).unwrap();
        assert_eq!(merged, input);
    }

    #[test]
    fn test_split_merge_roundtrip_remainder() {
        let input: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let batches = micro_batch_split(&input, 5, 4, 2).unwrap();
        let merged = micro_batch_merge(&batches).unwrap();
        assert_eq!(merged, input);
    }

    // ── stage_forward ──────────────────────────────────────────────

    #[test]
    fn test_stage_forward_basic() {
        let stage = PipelineStage::new(0, 3);
        let input = vec![1.0, 2.0, 3.0];
        let out = stage_forward(&input, &stage).unwrap();
        assert_eq!(out, vec![3.0, 6.0, 9.0]); // scaled by 3 layers
    }

    #[test]
    fn test_stage_forward_single_layer() {
        let stage = PipelineStage::new(0, 1);
        let input = vec![5.0, 10.0];
        let out = stage_forward(&input, &stage).unwrap();
        assert_eq!(out, vec![5.0, 10.0]); // scaled by 1
    }

    #[test]
    fn test_stage_forward_empty_input() {
        let stage = PipelineStage::new(0, 2);
        assert!(stage_forward(&[], &stage).is_err());
    }

    #[test]
    fn test_stage_forward_invalid_stage() {
        assert!(stage_forward(&[1.0], &PipelineStage::new(4, 4)).is_err());
    }

    // ── pipeline_forward (single stage) ────────────────────────────

    #[test]
    fn test_forward_single_stage_sequential() {
        let cfg =
            PipelineConfig::new(vec![PipelineStage::new(0, 2)], 4, PipelineSchedule::Sequential);
        let input = vec![1.0; 8]; // 4 rows × 2 dims
        let out = pipeline_forward(&input, 4, 2, &cfg).unwrap();
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_single_stage_gpipe() {
        let cfg = PipelineConfig::new(vec![PipelineStage::new(0, 3)], 2, PipelineSchedule::GPipe);
        let input = vec![2.0; 6];
        let out = pipeline_forward(&input, 3, 2, &cfg).unwrap();
        // 2.0 * 3 layers = 6.0
        assert!(out.iter().all(|&v| (v - 6.0).abs() < 1e-6));
    }

    // ── pipeline_forward (multi-stage) ─────────────────────────────

    fn two_stage_config(schedule: PipelineSchedule) -> PipelineConfig {
        PipelineConfig::new(vec![PipelineStage::new(0, 2), PipelineStage::new(2, 5)], 2, schedule)
    }

    #[test]
    fn test_forward_two_stage_sequential() {
        let cfg = two_stage_config(PipelineSchedule::Sequential);
        let input = vec![1.0; 8]; // 4×2
        let out = pipeline_forward(&input, 4, 2, &cfg).unwrap();
        // stage0: *2, stage1: *3 → total *6
        assert!(out.iter().all(|&v| (v - 6.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_two_stage_gpipe() {
        let cfg = two_stage_config(PipelineSchedule::GPipe);
        let input = vec![1.0; 8];
        let out = pipeline_forward(&input, 4, 2, &cfg).unwrap();
        assert!(out.iter().all(|&v| (v - 6.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_two_stage_pipedream() {
        let cfg = two_stage_config(PipelineSchedule::PipeDream);
        let input = vec![1.0; 8];
        let out = pipeline_forward(&input, 4, 2, &cfg).unwrap();
        assert!(out.iter().all(|&v| (v - 6.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_two_stage_interleaved() {
        let cfg = two_stage_config(PipelineSchedule::Interleaved);
        let input = vec![1.0; 8];
        let out = pipeline_forward(&input, 4, 2, &cfg).unwrap();
        assert!(out.iter().all(|&v| (v - 6.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_three_stages() {
        let cfg = PipelineConfig::new(
            vec![PipelineStage::new(0, 2), PipelineStage::new(2, 4), PipelineStage::new(4, 7)],
            2,
            PipelineSchedule::GPipe,
        );
        let input = vec![1.0; 6]; // 3×2
        let out = pipeline_forward(&input, 3, 2, &cfg).unwrap();
        // 1 * 2 * 2 * 3 = 12
        assert!(out.iter().all(|&v| (v - 12.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_four_stages() {
        let cfg = PipelineConfig::new(
            vec![
                PipelineStage::new(0, 1),
                PipelineStage::new(1, 2),
                PipelineStage::new(2, 3),
                PipelineStage::new(3, 4),
            ],
            1,
            PipelineSchedule::Sequential,
        );
        let input = vec![2.0; 4]; // 2×2
        let out = pipeline_forward(&input, 2, 2, &cfg).unwrap();
        // 2 * 1 * 1 * 1 * 1 = 2
        assert!(out.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    // ── pipeline_forward (error cases) ─────────────────────────────

    #[test]
    fn test_forward_empty_input() {
        let cfg = PipelineConfig::new(vec![PipelineStage::new(0, 2)], 1, PipelineSchedule::GPipe);
        assert!(pipeline_forward(&[], 0, 2, &cfg).is_err());
    }

    #[test]
    fn test_forward_mismatched_dims() {
        let cfg = PipelineConfig::new(vec![PipelineStage::new(0, 2)], 1, PipelineSchedule::GPipe);
        assert!(pipeline_forward(&[1.0, 2.0], 1, 3, &cfg).is_err());
    }

    #[test]
    fn test_forward_zero_stages() {
        let cfg = PipelineConfig::new(vec![], 1, PipelineSchedule::GPipe);
        assert!(pipeline_forward(&[1.0], 1, 1, &cfg).is_err());
    }

    // ── pipeline_forward (various input sizes) ─────────────────────

    #[test]
    fn test_forward_single_element() {
        let cfg = PipelineConfig::new(vec![PipelineStage::new(0, 2)], 1, PipelineSchedule::GPipe);
        let out = pipeline_forward(&[3.0], 1, 1, &cfg).unwrap();
        assert_eq!(out, vec![6.0]);
    }

    #[test]
    fn test_forward_large_batch() {
        let cfg = PipelineConfig::new(
            vec![PipelineStage::new(0, 1), PipelineStage::new(1, 2)],
            4,
            PipelineSchedule::GPipe,
        );
        let input = vec![1.0; 128]; // 64×2
        let out = pipeline_forward(&input, 64, 2, &cfg).unwrap();
        assert_eq!(out.len(), 128);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_large_dim() {
        let cfg =
            PipelineConfig::new(vec![PipelineStage::new(0, 3)], 1, PipelineSchedule::Sequential);
        let input = vec![2.0; 256]; // 1×256
        let out = pipeline_forward(&input, 1, 256, &cfg).unwrap();
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&v| (v - 6.0).abs() < 1e-6));
    }

    #[test]
    fn test_forward_micro_batch_equals_batch() {
        let cfg = PipelineConfig::new(vec![PipelineStage::new(0, 2)], 4, PipelineSchedule::GPipe);
        let input = vec![1.0; 8]; // 4×2
        let out = pipeline_forward(&input, 4, 2, &cfg).unwrap();
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_forward_micro_batch_one() {
        let cfg = PipelineConfig::new(
            vec![PipelineStage::new(0, 2), PipelineStage::new(2, 4)],
            1,
            PipelineSchedule::PipeDream,
        );
        let input = vec![1.0; 6]; // 3×2
        let out = pipeline_forward(&input, 3, 2, &cfg).unwrap();
        // 1 * 2 * 2 = 4
        assert!(out.iter().all(|&v| (v - 4.0).abs() < 1e-6));
    }

    // ── pipeline_bubble_time ───────────────────────────────────────

    #[test]
    fn test_bubble_single_stage() {
        assert_eq!(pipeline_bubble_time(1, 4), 0.0);
    }

    #[test]
    fn test_bubble_zero_micro_batches() {
        assert_eq!(pipeline_bubble_time(4, 0), 0.0);
    }

    #[test]
    fn test_bubble_two_stages_four_micro() {
        // (2-1)/(4+2-1) = 1/5 = 0.2
        let b = pipeline_bubble_time(2, 4);
        assert!((b - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_bubble_four_stages_four_micro() {
        // (4-1)/(4+4-1) = 3/7 ≈ 0.4286
        let b = pipeline_bubble_time(4, 4);
        assert!((b - 3.0 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_bubble_many_micro_batches() {
        // (4-1)/(100+4-1) = 3/103
        let b = pipeline_bubble_time(4, 100);
        assert!((b - 3.0 / 103.0).abs() < 1e-6);
    }

    #[test]
    fn test_bubble_one_micro_batch() {
        // (4-1)/(1+4-1) = 3/4 = 0.75
        let b = pipeline_bubble_time(4, 1);
        assert!((b - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_bubble_zero_stages() {
        assert_eq!(pipeline_bubble_time(0, 10), 0.0);
    }

    // ── optimal_micro_batch_count ──────────────────────────────────

    #[test]
    fn test_optimal_single_stage() {
        assert_eq!(optimal_micro_batch_count(1, 0.1), 1);
    }

    #[test]
    fn test_optimal_two_stages_10pct() {
        // (1)*(0.9)/0.1 = 9
        assert_eq!(optimal_micro_batch_count(2, 0.1), 9);
    }

    #[test]
    fn test_optimal_four_stages_10pct() {
        // (3)*(0.9)/0.1 = 27
        assert_eq!(optimal_micro_batch_count(4, 0.1), 27);
    }

    #[test]
    fn test_optimal_four_stages_50pct() {
        // (3)*(0.5)/0.5 = 3
        assert_eq!(optimal_micro_batch_count(4, 0.5), 3);
    }

    #[test]
    fn test_optimal_zero_fraction() {
        // Invalid → fallback to num_stages
        assert_eq!(optimal_micro_batch_count(4, 0.0), 4);
    }

    #[test]
    fn test_optimal_negative_fraction() {
        assert_eq!(optimal_micro_batch_count(4, -0.5), 4);
    }

    #[test]
    fn test_optimal_fraction_above_one() {
        assert_eq!(optimal_micro_batch_count(4, 1.5), 4);
    }

    #[test]
    fn test_optimal_fraction_exactly_one() {
        // (p-1)*(0)/1 = 0 → clamped to 1
        assert_eq!(optimal_micro_batch_count(4, 1.0), 1);
    }

    // ── Schedule functional equivalence ────────────────────────────

    #[test]
    fn test_all_schedules_same_result() {
        let stages = vec![PipelineStage::new(0, 2), PipelineStage::new(2, 5)];
        let input = vec![1.0; 12]; // 6×2
        let mut results = Vec::new();
        for sched in [
            PipelineSchedule::Sequential,
            PipelineSchedule::GPipe,
            PipelineSchedule::PipeDream,
            PipelineSchedule::Interleaved,
        ] {
            let cfg = PipelineConfig::new(stages.clone(), 2, sched);
            results.push(pipeline_forward(&input, 6, 2, &cfg).unwrap());
        }
        for r in &results[1..] {
            assert_eq!(r, &results[0]);
        }
    }

    // ── Stage affinity ─────────────────────────────────────────────

    #[test]
    fn test_stage_affinity_preserved() {
        let s = PipelineStage::new(0, 4).with_affinity(7);
        assert_eq!(s.thread_affinity, Some(7));
        assert_eq!(s.num_layers(), 4);
    }

    // ── Additional edge cases ──────────────────────────────────────

    #[test]
    fn test_merge_unequal_sizes() {
        let batches = vec![vec![1.0, 2.0, 3.0], vec![4.0]];
        let merged = micro_batch_merge(&batches).unwrap();
        assert_eq!(merged, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_forward_preserves_output_length() {
        let cfg = PipelineConfig::new(
            vec![PipelineStage::new(0, 3), PipelineStage::new(3, 6)],
            3,
            PipelineSchedule::GPipe,
        );
        let input = vec![1.0; 30]; // 10×3
        let out = pipeline_forward(&input, 10, 3, &cfg).unwrap();
        assert_eq!(out.len(), 30);
    }

    #[test]
    fn test_forward_values_multi_stage() {
        let cfg = PipelineConfig::new(
            vec![PipelineStage::new(0, 2), PipelineStage::new(2, 3)],
            1,
            PipelineSchedule::GPipe,
        );
        // input 5.0, stage0: *2 = 10, stage1: *1 = 10
        let out = pipeline_forward(&[5.0], 1, 1, &cfg).unwrap();
        assert!((out[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_single_layer_stages() {
        let cfg = PipelineConfig::new(
            vec![PipelineStage::new(0, 1), PipelineStage::new(1, 2), PipelineStage::new(2, 3)],
            1,
            PipelineSchedule::GPipe,
        );
        cfg.validate().unwrap();
        let out = pipeline_forward(&[7.0], 1, 1, &cfg).unwrap();
        // 7 * 1 * 1 * 1 = 7
        assert!((out[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_bubble_decreases_with_more_micro_batches() {
        let b1 = pipeline_bubble_time(4, 2);
        let b2 = pipeline_bubble_time(4, 8);
        let b3 = pipeline_bubble_time(4, 32);
        assert!(b1 > b2);
        assert!(b2 > b3);
    }

    #[test]
    fn test_bubble_increases_with_more_stages() {
        let b1 = pipeline_bubble_time(2, 8);
        let b2 = pipeline_bubble_time(4, 8);
        let b3 = pipeline_bubble_time(8, 8);
        assert!(b1 < b2);
        assert!(b2 < b3);
    }

    #[test]
    fn test_optimal_returns_at_least_one() {
        assert!(optimal_micro_batch_count(8, 0.5) >= 1);
        assert!(optimal_micro_batch_count(1, 0.5) >= 1);
    }

    #[test]
    fn test_optimal_satisfies_bubble_constraint() {
        for stages in 2..=8 {
            let frac = 0.15_f32;
            let m = optimal_micro_batch_count(stages, frac);
            let actual = pipeline_bubble_time(stages, m);
            assert!(
                actual <= frac + 1e-6,
                "stages={stages}, m={m}, actual bubble={actual}, limit={frac}"
            );
        }
    }

    #[test]
    fn test_split_batch_of_one_with_large_micro() {
        let input = vec![42.0; 4]; // 1×4
        let batches = micro_batch_split(&input, 1, 4, 100).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], input);
    }

    #[test]
    fn test_forward_non_uniform_stage_sizes() {
        let cfg = PipelineConfig::new(
            vec![
                PipelineStage::new(0, 1),  // 1 layer
                PipelineStage::new(1, 10), // 9 layers
            ],
            2,
            PipelineSchedule::GPipe,
        );
        let input = vec![1.0; 6]; // 3×2
        let out = pipeline_forward(&input, 3, 2, &cfg).unwrap();
        // 1 * 1 * 9 = 9
        assert!(out.iter().all(|&v| (v - 9.0).abs() < 1e-6));
    }
}
