//! Builder pattern for constructing [`GpuInferencePipeline`] instances.
//!
//! Allows incremental configuration, optional optimisation flags, and
//! explicit stage overrides before producing a validated pipeline.

use crate::pipeline::{
    GpuInferencePipeline, PipelineConfig, PipelineError, PipelineStage, QuantFormat,
};

// ---------------------------------------------------------------------------
// Optimisation flags
// ---------------------------------------------------------------------------

/// Optimisation toggles applied when building a pipeline.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct OptimizationFlags {
    /// Fuse adjacent RMS-norm + linear stages into a single kernel.
    pub kernel_fusion: bool,
    /// Reuse GPU buffers whose lifetimes do not overlap.
    pub memory_reuse: bool,
    /// Enable asynchronous kernel execution (overlap compute and
    /// transfer).
    pub async_execution: bool,
    /// Record per-kernel timing metrics.
    pub profiling: bool,
}

impl Default for OptimizationFlags {
    fn default() -> Self {
        Self { kernel_fusion: false, memory_reuse: true, async_execution: false, profiling: false }
    }
}

// ---------------------------------------------------------------------------
// `PipelineBuilder`
// ---------------------------------------------------------------------------

/// Fluent builder for [`GpuInferencePipeline`].
pub struct PipelineBuilder {
    config: PipelineConfig,
    stages: Vec<PipelineStage>,
    optimizations: OptimizationFlags,
    extra_layers: usize,
}

impl PipelineBuilder {
    /// Start with a default (zero-sized) config — caller must set
    /// fields before calling [`build`](Self::build).
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: PipelineConfig {
                num_layers: 0,
                hidden_dim: 0,
                num_heads: 0,
                head_dim: 0,
                vocab_size: 0,
                max_seq_len: 0,
                use_flash_attention: false,
                fuse_layernorm: false,
            },
            stages: Vec::new(),
            optimizations: OptimizationFlags::default(),
            extra_layers: 0,
        }
    }

    /// Start from an existing [`PipelineConfig`].
    #[must_use]
    pub fn with_config(config: PipelineConfig) -> Self {
        Self {
            config,
            stages: Vec::new(),
            optimizations: OptimizationFlags::default(),
            extra_layers: 0,
        }
    }

    /// Append one transformer layer (norm -> attn -> norm -> ffn).
    #[must_use]
    pub const fn add_layer(mut self) -> Self {
        self.extra_layers += 1;
        self
    }

    /// Append a custom stage to the pipeline.
    #[must_use]
    pub fn add_stage(mut self, stage: PipelineStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Append a dequantization stage.
    #[must_use]
    pub fn add_dequantize(mut self, format: QuantFormat, block_size: usize) -> Self {
        self.stages.push(PipelineStage::Dequantize { format, block_size });
        self
    }

    /// Enable kernel fusion optimisations.
    #[must_use]
    pub const fn enable_fusion(mut self) -> Self {
        self.optimizations.kernel_fusion = true;
        self.config.fuse_layernorm = true;
        self
    }

    /// Enable per-kernel profiling.
    #[must_use]
    pub const fn enable_profiling(mut self) -> Self {
        self.optimizations.profiling = true;
        self
    }

    /// Enable async kernel execution.
    #[must_use]
    pub const fn enable_async(mut self) -> Self {
        self.optimizations.async_execution = true;
        self
    }

    /// Enable memory buffer reuse.
    #[must_use]
    pub const fn enable_memory_reuse(mut self) -> Self {
        self.optimizations.memory_reuse = true;
        self
    }

    /// Replace the optimisation flags wholesale.
    #[must_use]
    pub const fn with_optimizations(mut self, flags: OptimizationFlags) -> Self {
        self.optimizations = flags;
        self
    }

    /// Current optimisation flags.
    #[must_use]
    pub const fn optimizations(&self) -> &OptimizationFlags {
        &self.optimizations
    }

    /// Consume the builder and produce a validated pipeline.
    pub fn build(mut self) -> Result<GpuInferencePipeline, PipelineError> {
        self.config.num_layers += self.extra_layers;
        self.config.validate()?;

        if self.stages.is_empty() {
            return GpuInferencePipeline::new(self.config);
        }

        let mut full_stages = Vec::new();
        full_stages.push(PipelineStage::Embedding {
            vocab_size: self.config.vocab_size,
            dim: self.config.hidden_dim,
        });
        full_stages.append(&mut self.stages);
        full_stages.push(PipelineStage::Linear {
            in_features: self.config.hidden_dim,
            out_features: self.config.vocab_size,
        });
        full_stages.push(PipelineStage::Softmax { dim: self.config.vocab_size });

        GpuInferencePipeline::from_parts(self.config, full_stages)
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
