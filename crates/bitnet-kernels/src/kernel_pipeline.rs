//! Kernel pipeline for chaining multiple kernel operations into an optimized
//! execution pipeline (e.g., quantize → matmul → dequantize → layernorm).
//!
//! Supports sequential, pipelined, and fused execution modes with automatic
//! memory planning, shape validation, and per-stage profiling.

use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Shape descriptor
// ---------------------------------------------------------------------------

/// Describes the shape and element type flowing between pipeline stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    pub dims: Vec<usize>,
    pub elem_bytes: usize,
}

impl TensorShape {
    pub fn new(dims: Vec<usize>, elem_bytes: usize) -> Self {
        Self { dims, elem_bytes }
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Total byte size.
    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.elem_bytes
    }
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "] x {}B", self.elem_bytes)
    }
}

// ---------------------------------------------------------------------------
// Pipeline errors
// ---------------------------------------------------------------------------

/// Errors that can occur during pipeline construction or execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    /// Pipeline contains no stages.
    Empty,
    /// Shape mismatch between consecutive stages.
    ShapeMismatch { stage_index: usize, expected: TensorShape, got: TensorShape },
    /// A stage's input validation failed.
    ValidationFailed { stage_index: usize, reason: String },
    /// A stage execution failed.
    ExecutionFailed { stage_index: usize, reason: String },
    /// Fused execution is not supported for the given stage pair.
    FusionUnsupported { first: usize, second: usize },
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "pipeline has no stages"),
            Self::ShapeMismatch { stage_index, expected, got } => {
                write!(f, "shape mismatch at stage {stage_index}: expected {expected}, got {got}")
            }
            Self::ValidationFailed { stage_index, reason } => {
                write!(f, "validation failed at stage {stage_index}: {reason}")
            }
            Self::ExecutionFailed { stage_index, reason } => {
                write!(f, "execution failed at stage {stage_index}: {reason}")
            }
            Self::FusionUnsupported { first, second } => {
                write!(f, "fusion unsupported for stages {first} and {second}")
            }
        }
    }
}

impl std::error::Error for PipelineError {}

// ---------------------------------------------------------------------------
// PipelineStage trait
// ---------------------------------------------------------------------------

/// A single stage in the kernel pipeline.
pub trait PipelineStage: Send + Sync {
    /// Human-readable name of this stage.
    fn name(&self) -> &str;

    /// Validate that `input` is acceptable for this stage.
    fn validate_input(&self, input: &TensorShape) -> Result<(), String>;

    /// Given an input shape, return the shape this stage will produce.
    fn expected_output_shape(&self, input: &TensorShape) -> TensorShape;

    /// Execute the stage on `data`, returning (possibly new) output buffer.
    /// `data` is the intermediate buffer; stages may modify it in-place when
    /// input and output shapes match.
    fn execute(&self, data: &mut Vec<u8>, input_shape: &TensorShape)
    -> Result<TensorShape, String>;

    /// Whether this stage can be fused with the *following* stage.
    fn supports_fusion_with(&self, _next: &str) -> bool {
        false
    }

    /// Whether this stage can operate in-place (no extra allocation).
    fn supports_in_place(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Execution mode
// ---------------------------------------------------------------------------

/// How the pipeline executes its stages.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Execute stages one after another.
    #[default]
    Sequential,
    /// Overlap memory transfers with compute where possible.
    Pipelined,
    /// Combine compatible adjacent stages into a single fused kernel.
    Fused,
}

// ---------------------------------------------------------------------------
// Per-stage profiling
// ---------------------------------------------------------------------------

/// Timing and memory statistics for a single stage execution.
#[derive(Debug, Clone)]
pub struct StageProfile {
    pub stage_name: String,
    pub stage_index: usize,
    pub elapsed: Duration,
    pub input_bytes: usize,
    pub output_bytes: usize,
}

/// Aggregate profiling for a full pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineProfile {
    pub stages: Vec<StageProfile>,
    pub total_elapsed: Duration,
    pub memory_high_water_mark: usize,
}

impl PipelineProfile {
    /// Sum of all individual stage durations (wall-clock overlap ignored).
    pub fn stage_time_sum(&self) -> Duration {
        self.stages.iter().map(|s| s.elapsed).sum()
    }
}

// ---------------------------------------------------------------------------
// Memory plan
// ---------------------------------------------------------------------------

/// Pre-computed buffer sizes for every intermediate result in the pipeline.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Byte size required for each intermediate buffer (len == stages + 1,
    /// element 0 is the pipeline input, element N is stage N's output).
    pub buffer_sizes: Vec<usize>,
    /// Stages where the operation can happen in-place (no new allocation).
    pub in_place_stages: Vec<bool>,
    /// Peak memory that will be live at any point during sequential execution.
    pub high_water_mark: usize,
}

// ---------------------------------------------------------------------------
// KernelPipeline
// ---------------------------------------------------------------------------

/// A chain of [`PipelineStage`]s with memory planning and profiling.
pub struct KernelPipeline {
    stages: Vec<Box<dyn PipelineStage>>,
    mode: ExecutionMode,
    enable_profiling: bool,
}

impl fmt::Debug for KernelPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelPipeline")
            .field("num_stages", &self.stages.len())
            .field("mode", &self.mode)
            .field("enable_profiling", &self.enable_profiling)
            .finish()
    }
}

impl KernelPipeline {
    /// Number of stages.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Whether the pipeline has no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Current execution mode.
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// Whether profiling is enabled.
    pub fn profiling_enabled(&self) -> bool {
        self.enable_profiling
    }

    /// Names of all stages in order.
    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.name()).collect()
    }

    // -- validation ---------------------------------------------------------

    /// Validate that the shapes are compatible across all adjacent stages,
    /// given the pipeline's initial `input_shape`.
    pub fn validate(&self, input_shape: &TensorShape) -> Result<(), PipelineError> {
        if self.stages.is_empty() {
            return Err(PipelineError::Empty);
        }

        let mut shape = input_shape.clone();
        for (i, stage) in self.stages.iter().enumerate() {
            stage
                .validate_input(&shape)
                .map_err(|reason| PipelineError::ValidationFailed { stage_index: i, reason })?;
            shape = stage.expected_output_shape(&shape);
        }
        Ok(())
    }

    /// Compute the output shape of the entire pipeline without executing it.
    pub fn output_shape(&self, input_shape: &TensorShape) -> Result<TensorShape, PipelineError> {
        self.validate(input_shape)?;
        let mut shape = input_shape.clone();
        for stage in &self.stages {
            shape = stage.expected_output_shape(&shape);
        }
        Ok(shape)
    }

    // -- memory planning ----------------------------------------------------

    /// Compute a [`MemoryPlan`] for the pipeline given `input_shape`.
    pub fn plan_memory(&self, input_shape: &TensorShape) -> Result<MemoryPlan, PipelineError> {
        self.validate(input_shape)?;

        let mut buffer_sizes = Vec::with_capacity(self.stages.len() + 1);
        let mut in_place = Vec::with_capacity(self.stages.len());
        buffer_sizes.push(input_shape.byte_size());

        let mut shape = input_shape.clone();
        for stage in &self.stages {
            let out_shape = stage.expected_output_shape(&shape);
            let can_in_place =
                stage.supports_in_place() && out_shape.byte_size() <= shape.byte_size();
            in_place.push(can_in_place);
            buffer_sizes.push(out_shape.byte_size());
            shape = out_shape;
        }

        // High-water mark: in sequential mode at most two buffers are live at
        // once (current input + current output), unless in-place.
        let mut hwm: usize = buffer_sizes[0];
        for i in 0..self.stages.len() {
            let live =
                if in_place[i] { buffer_sizes[i] } else { buffer_sizes[i] + buffer_sizes[i + 1] };
            hwm = hwm.max(live);
        }

        Ok(MemoryPlan { buffer_sizes, in_place_stages: in_place, high_water_mark: hwm })
    }

    // -- execution ----------------------------------------------------------

    /// Execute the pipeline on `data` with the declared `input_shape`.
    ///
    /// Returns the final output shape and, if profiling is enabled, a
    /// [`PipelineProfile`].
    pub fn execute(
        &self,
        data: &mut Vec<u8>,
        input_shape: &TensorShape,
    ) -> Result<(TensorShape, Option<PipelineProfile>), PipelineError> {
        self.validate(input_shape)?;

        match self.mode {
            ExecutionMode::Sequential | ExecutionMode::Pipelined => {
                self.execute_sequential(data, input_shape)
            }
            ExecutionMode::Fused => self.execute_fused(data, input_shape),
        }
    }

    fn execute_sequential(
        &self,
        data: &mut Vec<u8>,
        input_shape: &TensorShape,
    ) -> Result<(TensorShape, Option<PipelineProfile>), PipelineError> {
        let pipeline_start = Instant::now();
        let mut profiles: Vec<StageProfile> = Vec::new();
        let mut hwm = data.len();

        let mut shape = input_shape.clone();
        for (i, stage) in self.stages.iter().enumerate() {
            let input_bytes = shape.byte_size();
            let start = Instant::now();
            shape = stage
                .execute(data, &shape)
                .map_err(|reason| PipelineError::ExecutionFailed { stage_index: i, reason })?;
            let elapsed = start.elapsed();
            hwm = hwm.max(data.len());

            if self.enable_profiling {
                profiles.push(StageProfile {
                    stage_name: stage.name().to_string(),
                    stage_index: i,
                    elapsed,
                    input_bytes,
                    output_bytes: shape.byte_size(),
                });
            }
        }

        let profile = if self.enable_profiling {
            Some(PipelineProfile {
                stages: profiles,
                total_elapsed: pipeline_start.elapsed(),
                memory_high_water_mark: hwm,
            })
        } else {
            None
        };

        Ok((shape, profile))
    }

    fn execute_fused(
        &self,
        data: &mut Vec<u8>,
        input_shape: &TensorShape,
    ) -> Result<(TensorShape, Option<PipelineProfile>), PipelineError> {
        // Verify fusion compatibility first; fall back to sequential for
        // pairs that cannot be fused.
        for i in 0..self.stages.len().saturating_sub(1) {
            let next_name = self.stages[i + 1].name();
            if !self.stages[i].supports_fusion_with(next_name) {
                // Fall back to sequential – fusion isn't required to succeed
                // for the whole pipeline.
                return self.execute_sequential(data, input_shape);
            }
        }
        // If all pairs support fusion, we still execute sequentially (actual
        // kernel fusion would require backend-specific codegen).
        self.execute_sequential(data, input_shape)
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Fluent builder for [`KernelPipeline`].
pub struct PipelineBuilder {
    stages: Vec<Box<dyn PipelineStage>>,
    mode: ExecutionMode,
    enable_profiling: bool,
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self { stages: Vec::new(), mode: ExecutionMode::Sequential, enable_profiling: false }
    }

    /// Append a stage to the pipeline.
    pub fn stage(mut self, stage: Box<dyn PipelineStage>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Set the execution mode.
    pub fn execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enable or disable per-stage profiling.
    pub fn profiling(mut self, enabled: bool) -> Self {
        self.enable_profiling = enabled;
        self
    }

    /// Build the pipeline. Returns an error if the pipeline would be empty.
    pub fn build(self) -> Result<KernelPipeline, PipelineError> {
        if self.stages.is_empty() {
            return Err(PipelineError::Empty);
        }
        Ok(KernelPipeline {
            stages: self.stages,
            mode: self.mode,
            enable_profiling: self.enable_profiling,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers: concrete stages -------------------------------------------

    /// Identity pass-through (same shape in and out).
    struct IdentityStage;

    impl PipelineStage for IdentityStage {
        fn name(&self) -> &str {
            "identity"
        }
        fn validate_input(&self, _input: &TensorShape) -> Result<(), String> {
            Ok(())
        }
        fn expected_output_shape(&self, input: &TensorShape) -> TensorShape {
            input.clone()
        }
        fn execute(
            &self,
            _data: &mut Vec<u8>,
            input_shape: &TensorShape,
        ) -> Result<TensorShape, String> {
            Ok(input_shape.clone())
        }
        fn supports_in_place(&self) -> bool {
            true
        }
    }

    /// Doubles the last dimension.
    struct ExpandStage;

    impl PipelineStage for ExpandStage {
        fn name(&self) -> &str {
            "expand"
        }
        fn validate_input(&self, input: &TensorShape) -> Result<(), String> {
            if input.dims.is_empty() {
                return Err("need at least 1 dim".into());
            }
            Ok(())
        }
        fn expected_output_shape(&self, input: &TensorShape) -> TensorShape {
            let mut dims = input.dims.clone();
            if let Some(last) = dims.last_mut() {
                *last *= 2;
            }
            TensorShape::new(dims, input.elem_bytes)
        }
        fn execute(
            &self,
            data: &mut Vec<u8>,
            input_shape: &TensorShape,
        ) -> Result<TensorShape, String> {
            let out = self.expected_output_shape(input_shape);
            data.resize(out.byte_size(), 0);
            Ok(out)
        }
    }

    /// Halves the last dimension (requires even last dim).
    struct ShrinkStage;

    impl PipelineStage for ShrinkStage {
        fn name(&self) -> &str {
            "shrink"
        }
        fn validate_input(&self, input: &TensorShape) -> Result<(), String> {
            match input.dims.last() {
                Some(d) if *d >= 2 && d % 2 == 0 => Ok(()),
                _ => Err("last dim must be even and >= 2".into()),
            }
        }
        fn expected_output_shape(&self, input: &TensorShape) -> TensorShape {
            let mut dims = input.dims.clone();
            if let Some(last) = dims.last_mut() {
                *last /= 2;
            }
            TensorShape::new(dims, input.elem_bytes)
        }
        fn execute(
            &self,
            data: &mut Vec<u8>,
            input_shape: &TensorShape,
        ) -> Result<TensorShape, String> {
            let out = self.expected_output_shape(input_shape);
            data.truncate(out.byte_size());
            Ok(out)
        }
        fn supports_in_place(&self) -> bool {
            true
        }
    }

    /// Changes elem_bytes (simulates quantize/dequantize).
    struct ReinterpretStage {
        target_elem_bytes: usize,
    }

    impl PipelineStage for ReinterpretStage {
        fn name(&self) -> &str {
            "reinterpret"
        }
        fn validate_input(&self, _input: &TensorShape) -> Result<(), String> {
            Ok(())
        }
        fn expected_output_shape(&self, input: &TensorShape) -> TensorShape {
            TensorShape::new(input.dims.clone(), self.target_elem_bytes)
        }
        fn execute(
            &self,
            data: &mut Vec<u8>,
            input_shape: &TensorShape,
        ) -> Result<TensorShape, String> {
            let out = self.expected_output_shape(input_shape);
            data.resize(out.byte_size(), 0);
            Ok(out)
        }
    }

    /// Always fails execution.
    struct FailStage;

    impl PipelineStage for FailStage {
        fn name(&self) -> &str {
            "fail"
        }
        fn validate_input(&self, _input: &TensorShape) -> Result<(), String> {
            Ok(())
        }
        fn expected_output_shape(&self, input: &TensorShape) -> TensorShape {
            input.clone()
        }
        fn execute(
            &self,
            _data: &mut Vec<u8>,
            _input_shape: &TensorShape,
        ) -> Result<TensorShape, String> {
            Err("stage always fails".into())
        }
    }

    /// Requires a specific number of dimensions.
    struct RequireDimsStage {
        required_ndim: usize,
    }

    impl PipelineStage for RequireDimsStage {
        fn name(&self) -> &str {
            "require_dims"
        }
        fn validate_input(&self, input: &TensorShape) -> Result<(), String> {
            if input.dims.len() != self.required_ndim {
                return Err(format!(
                    "expected {} dims, got {}",
                    self.required_ndim,
                    input.dims.len()
                ));
            }
            Ok(())
        }
        fn expected_output_shape(&self, input: &TensorShape) -> TensorShape {
            input.clone()
        }
        fn execute(
            &self,
            _data: &mut Vec<u8>,
            input_shape: &TensorShape,
        ) -> Result<TensorShape, String> {
            self.validate_input(input_shape)?;
            Ok(input_shape.clone())
        }
        fn supports_in_place(&self) -> bool {
            true
        }
    }

    /// Fusable stage that reports compatibility with a named partner.
    struct FusableStage {
        stage_name: String,
        fuse_with: String,
    }

    impl PipelineStage for FusableStage {
        fn name(&self) -> &str {
            &self.stage_name
        }
        fn validate_input(&self, _input: &TensorShape) -> Result<(), String> {
            Ok(())
        }
        fn expected_output_shape(&self, input: &TensorShape) -> TensorShape {
            input.clone()
        }
        fn execute(
            &self,
            _data: &mut Vec<u8>,
            input_shape: &TensorShape,
        ) -> Result<TensorShape, String> {
            Ok(input_shape.clone())
        }
        fn supports_fusion_with(&self, next: &str) -> bool {
            next == self.fuse_with
        }
        fn supports_in_place(&self) -> bool {
            true
        }
    }

    fn shape_1d(n: usize) -> TensorShape {
        TensorShape::new(vec![n], 4)
    }

    fn shape_2d(rows: usize, cols: usize) -> TensorShape {
        TensorShape::new(vec![rows, cols], 4)
    }

    // ======================================================================
    // TensorShape tests
    // ======================================================================

    #[test]
    fn tensor_shape_num_elements() {
        assert_eq!(shape_2d(3, 4).num_elements(), 12);
    }

    #[test]
    fn tensor_shape_byte_size() {
        assert_eq!(shape_2d(3, 4).byte_size(), 48); // 12 * 4
    }

    #[test]
    fn tensor_shape_display() {
        let s = shape_2d(2, 8);
        assert_eq!(format!("{s}"), "[2, 8] x 4B");
    }

    #[test]
    fn tensor_shape_scalar() {
        let s = TensorShape::new(vec![1], 1);
        assert_eq!(s.num_elements(), 1);
        assert_eq!(s.byte_size(), 1);
    }

    // ======================================================================
    // Builder tests
    // ======================================================================

    #[test]
    fn builder_empty_pipeline_errors() {
        let res = PipelineBuilder::new().build();
        assert_eq!(res.unwrap_err(), PipelineError::Empty);
    }

    #[test]
    fn builder_single_stage() {
        let pipe = PipelineBuilder::new().stage(Box::new(IdentityStage)).build().unwrap();
        assert_eq!(pipe.len(), 1);
        assert!(!pipe.is_empty());
    }

    #[test]
    fn builder_multiple_stages() {
        let pipe = PipelineBuilder::new()
            .stage(Box::new(IdentityStage))
            .stage(Box::new(ExpandStage))
            .stage(Box::new(ShrinkStage))
            .build()
            .unwrap();
        assert_eq!(pipe.len(), 3);
    }

    #[test]
    fn builder_sets_mode() {
        let pipe = PipelineBuilder::new()
            .stage(Box::new(IdentityStage))
            .execution_mode(ExecutionMode::Pipelined)
            .build()
            .unwrap();
        assert_eq!(pipe.mode(), ExecutionMode::Pipelined);
    }

    #[test]
    fn builder_sets_profiling() {
        let pipe =
            PipelineBuilder::new().stage(Box::new(IdentityStage)).profiling(true).build().unwrap();
        assert!(pipe.profiling_enabled());
    }

    #[test]
    fn builder_default_mode_is_sequential() {
        let pipe = PipelineBuilder::new().stage(Box::new(IdentityStage)).build().unwrap();
        assert_eq!(pipe.mode(), ExecutionMode::Sequential);
    }

    #[test]
    fn builder_default_profiling_disabled() {
        let pipe = PipelineBuilder::new().stage(Box::new(IdentityStage)).build().unwrap();
        assert!(!pipe.profiling_enabled());
    }

    #[test]
    fn builder_default_impl() {
        // `PipelineBuilder` implements Default.
        let _b = PipelineBuilder::default();
    }

    // ======================================================================
    // Validation tests
    // ======================================================================

    #[test]
    fn validate_identity_passes() {
        let pipe = PipelineBuilder::new().stage(Box::new(IdentityStage)).build().unwrap();
        assert!(pipe.validate(&shape_1d(8)).is_ok());
    }

    #[test]
    fn validate_expand_then_shrink_passes() {
        let pipe = PipelineBuilder::new()
            .stage(Box::new(ExpandStage))
            .stage(Box::new(ShrinkStage))
            .build()
            .unwrap();
        // 8 → expand → 16 → shrink → 8
        assert!(pipe.validate(&shape_1d(8)).is_ok());
    }

    #[test]
    fn validate_shrink_rejects_odd_dim() {
        let pipe = PipelineBuilder::new().stage(Box::new(ShrinkStage)).build().unwrap();
        let err = pipe.validate(&shape_1d(7)).unwrap_err();
        assert!(matches!(err, PipelineError::ValidationFailed { stage_index: 0, .. }));
    }

    #[test]
    fn validate_chained_shape_mismatch_caught() {
        // expand produces even dim, but feeding a scalar-like 1-element into
        // shrink after expand: expand(1) → 2, then shrink(2) → 1, OK.
        // Instead test: require_dims(3) after a 2-d input.
        let pipe = PipelineBuilder::new()
            .stage(Box::new(RequireDimsStage { required_ndim: 3 }))
            .build()
            .unwrap();
        let err = pipe.validate(&shape_2d(4, 4)).unwrap_err();
        assert!(matches!(err, PipelineError::ValidationFailed { stage_index: 0, .. }));
    }

    #[test]
    fn validate_empty_pipeline_errors() {
        // Force an empty pipeline via builder then manual clear isn't possible,
        // but `build()` itself returns an error – tested in builder_empty.
        // Construct one through a back-door for coverage:
        let pipe = KernelPipeline {
            stages: vec![],
            mode: ExecutionMode::Sequential,
            enable_profiling: false,
        };
        assert_eq!(pipe.validate(&shape_1d(4)).unwrap_err(), PipelineError::Empty);
    }

    // ======================================================================
    // Output shape tests
    // ======================================================================

    #[test]
    fn output_shape_identity() {
        let pipe = PipelineBuilder::new().stage(Box::new(IdentityStage)).build().unwrap();
        assert_eq!(pipe.output_shape(&shape_1d(8)).unwrap(), shape_1d(8));
    }

    #[test]
    fn output_shape_expand() {
        let pipe = PipelineBuilder::new().stage(Box::new(ExpandStage)).build().unwrap();
        assert_eq!(pipe.output_shape(&shape_1d(8)).unwrap(), shape_1d(16));
    }

    #[test]
    fn output_shape_expand_shrink_roundtrip() {
        let pipe = PipelineBuilder::new()
            .stage(Box::new(ExpandStage))
            .stage(Box::new(ShrinkStage))
            .build()
            .unwrap();
        assert_eq!(pipe.output_shape(&shape_1d(8)).unwrap(), shape_1d(8));
    }

    #[test]
    fn output_shape_reinterpret() {
        let pipe = PipelineBuilder::new()
            .stage(Box::new(ReinterpretStage { target_elem_bytes: 1 }))
            .build()
            .unwrap();
        let out = pipe.output_shape(&shape_1d(16)).unwrap();
        assert_eq!(out.elem_bytes, 1);
        assert_eq!(out.byte_size(), 16); // 16 elements * 1 byte
    }

    // ======================================================================
    // Memory planning tests
    // ======================================================================

    #[test]
    fn memory_plan_identity_in_place() {
        let pipe = PipelineBuilder::new().stage(Box::new(IdentityStage)).build().unwrap();
        let plan = pipe.plan_memory(&shape_1d(8)).unwrap();
        assert!(plan.in_place_stages[0]);
        assert_eq!(plan.buffer_sizes.len(), 2);
        assert_eq!(plan.high_water_mark, 32); // 8 * 4
    }

    #[test]
    fn memory_plan_expand_not_in_place() {
        let pipe = PipelineBuilder::new().stage(Box::new(ExpandStage)).build().unwrap();
        let plan = pipe.plan_memory(&shape_1d(8)).unwrap();
        assert!(!plan.in_place_stages[0]);
        // hwm = input + output = 32 + 64 = 96
        assert_eq!(plan.high_water_mark, 96);
    }

    #[test]
    fn memory_plan_shrink_in_place() {
        let pipe = PipelineBuilder::new().stage(Box::new(ShrinkStage)).build().unwrap();
        let plan = pipe.plan_memory(&shape_1d(8)).unwrap();
        assert!(plan.in_place_stages[0]);
        // in-place: hwm = input = 32
        assert_eq!(plan.high_water_mark, 32);
    }

    #[test]
    fn memory_plan_multi_stage_hwm() {
        let pipe = PipelineBuilder::new()
            .stage(Box::new(ExpandStage))  // 32 → 64
            .stage(Box::new(ExpandStage))  // 64 → 128
            .build()
            .unwrap();
        let plan = pipe.plan_memory(&shape_1d(8)).unwrap();
        // Stage 0: 32 + 64 = 96; Stage 1: 64 + 128 = 192
        assert_eq!(plan.high_water_mark, 192);
    }

    // ======================================================================
    // Execution tests
    // ======================================================================

    #[test]
    fn execute_identity() {
        let pipe = PipelineBuilder::new().stage(Box::new(IdentityStage)).build().unwrap();
        let input = shape_1d(4);
        let mut data = vec![0u8; input.byte_size()];
        let (out, _) = pipe.execute(&mut data, &input).unwrap();
        assert_eq!(out, shape_1d(4));
    }

    #[test]
    fn execute_expand() {
        let pipe = PipelineBuilder::new().stage(Box::new(ExpandStage)).build().unwrap();
        let input = shape_1d(4);
        let mut data = vec![1u8; input.byte_size()];
        let (out, _) = pipe.execute(&mut data, &input).unwrap();
        assert_eq!(out, shape_1d(8));
        assert_eq!(data.len(), 32); // 8 * 4
    }

    #[test]
    fn execute_chain_three_stages() {
        let pipe = PipelineBuilder::new()
            .stage(Box::new(ExpandStage))
            .stage(Box::new(IdentityStage))
            .stage(Box::new(ShrinkStage))
            .build()
            .unwrap();
        let input = shape_1d(4);
        let mut data = vec![0u8; input.byte_size()];
        let (out, _) = pipe.execute(&mut data, &input).unwrap();
        assert_eq!(out, shape_1d(4));
    }

    #[test]
    fn execute_fail_stage_propagates() {
        let pipe = PipelineBuilder::new()
            .stage(Box::new(IdentityStage))
            .stage(Box::new(FailStage))
            .build()
            .unwrap();
        let input = shape_1d(4);
        let mut data = vec![0u8; input.byte_size()];
        let err = pipe.execute(&mut data, &input).unwrap_err();
        assert!(matches!(err, PipelineError::ExecutionFailed { stage_index: 1, .. }));
    }

    #[test]
    fn execute_reinterpret_quantize_dequantize() {
        // Simulate: f32 → u8 (quantize) → f32 (dequantize)
        let pipe = PipelineBuilder::new()
            .stage(Box::new(ReinterpretStage { target_elem_bytes: 1 }))
            .stage(Box::new(ReinterpretStage { target_elem_bytes: 4 }))
            .build()
            .unwrap();
        let input = shape_1d(16);
        let mut data = vec![0u8; input.byte_size()];
        let (out, _) = pipe.execute(&mut data, &input).unwrap();
        assert_eq!(out.elem_bytes, 4);
        assert_eq!(out, shape_1d(16));
    }

    // ======================================================================
    // Profiling tests
    // ======================================================================

    #[test]
    fn profiling_disabled_returns_none() {
        let pipe =
            PipelineBuilder::new().stage(Box::new(IdentityStage)).profiling(false).build().unwrap();
        let input = shape_1d(4);
        let mut data = vec![0u8; input.byte_size()];
        let (_, profile) = pipe.execute(&mut data, &input).unwrap();
        assert!(profile.is_none());
    }

    #[test]
    fn profiling_enabled_returns_some() {
        let pipe =
            PipelineBuilder::new().stage(Box::new(IdentityStage)).profiling(true).build().unwrap();
        let input = shape_1d(4);
        let mut data = vec![0u8; input.byte_size()];
        let (_, profile) = pipe.execute(&mut data, &input).unwrap();
        assert!(profile.is_some());
    }

    #[test]
    fn profiling_stage_count_matches() {
        let pipe = PipelineBuilder::new()
            .stage(Box::new(ExpandStage))
            .stage(Box::new(ShrinkStage))
            .profiling(true)
            .build()
            .unwrap();
        let input = shape_1d(4);
        let mut data = vec![0u8; input.byte_size()];
        let (_, profile) = pipe.execute(&mut data, &input).unwrap();
        let p = profile.unwrap();
        assert_eq!(p.stages.len(), 2);
        assert_eq!(p.stages[0].stage_name, "expand");
        assert_eq!(p.stages[1].stage_name, "shrink");
    }

    #[test]
    fn profiling_records_byte_sizes() {
        let pipe =
            PipelineBuilder::new().stage(Box::new(ExpandStage)).profiling(true).build().unwrap();
        let input = shape_1d(4);
        let mut data = vec![0u8; input.byte_size()];
        let (_, profile) = pipe.execute(&mut data, &input).unwrap();
        let p = profile.unwrap();
        assert_eq!(p.stages[0].input_bytes, 16); // 4 * 4
        assert_eq!(p.stages[0].output_bytes, 32); // 8 * 4
    }

    #[test]
    fn profiling_hwm_tracked() {
        let pipe =
            PipelineBuilder::new().stage(Box::new(ExpandStage)).profiling(true).build().unwrap();
        let input = shape_1d(4);
        let mut data = vec![0u8; input.byte_size()];
        let (_, profile) = pipe.execute(&mut data, &input).unwrap();
        let p = profile.unwrap();
        assert!(p.memory_high_water_mark >= 32);
    }

    #[test]
    fn profiling_stage_time_sum() {
        let pipe = PipelineBuilder::new()
            .stage(Box::new(IdentityStage))
            .stage(Box::new(IdentityStage))
            .profiling(true)
            .build()
            .unwrap();
        let input = shape_1d(4);
        let mut data = vec![0u8; input.byte_size()];
        let (_, profile) = pipe.execute(&mut data, &input).unwrap();
        let p = profile.unwrap();
        // Sum of individual stages should not exceed total (modulo measurement noise).
        assert!(p.stage_time_sum() <= p.total_elapsed + Duration::from_micros(100));
    }

    // ======================================================================
    // Execution mode tests
    // ======================================================================

    #[test]
    fn pipelined_mode_executes() {
        let pipe = PipelineBuilder::new()
            .stage(Box::new(IdentityStage))
            .execution_mode(ExecutionMode::Pipelined)
            .build()
            .unwrap();
        let input = shape_1d(4);
        let mut data = vec![0u8; input.byte_size()];
        let (out, _) = pipe.execute(&mut data, &input).unwrap();
        assert_eq!(out, shape_1d(4));
    }

    #[test]
    fn fused_mode_falls_back_when_unsupported() {
        let pipe = PipelineBuilder::new()
            .stage(Box::new(IdentityStage))
            .stage(Box::new(ExpandStage))
            .execution_mode(ExecutionMode::Fused)
            .build()
            .unwrap();
        let input = shape_1d(4);
        let mut data = vec![0u8; input.byte_size()];
        // IdentityStage does not support fusion → falls back to sequential.
        let (out, _) = pipe.execute(&mut data, &input).unwrap();
        assert_eq!(out, shape_1d(8));
    }

    #[test]
    fn fused_mode_with_compatible_stages() {
        let pipe = PipelineBuilder::new()
            .stage(Box::new(FusableStage {
                stage_name: "stage_a".into(),
                fuse_with: "stage_b".into(),
            }))
            .stage(Box::new(FusableStage {
                stage_name: "stage_b".into(),
                fuse_with: "stage_a".into(),
            }))
            .execution_mode(ExecutionMode::Fused)
            .build()
            .unwrap();
        let input = shape_1d(4);
        let mut data = vec![0u8; input.byte_size()];
        let (out, _) = pipe.execute(&mut data, &input).unwrap();
        assert_eq!(out, shape_1d(4));
    }

    // ======================================================================
    // stage_names / Debug
    // ======================================================================

    #[test]
    fn stage_names_in_order() {
        let pipe = PipelineBuilder::new()
            .stage(Box::new(ExpandStage))
            .stage(Box::new(ShrinkStage))
            .build()
            .unwrap();
        assert_eq!(pipe.stage_names(), vec!["expand", "shrink"]);
    }

    #[test]
    fn pipeline_debug_output() {
        let pipe = PipelineBuilder::new().stage(Box::new(IdentityStage)).build().unwrap();
        let dbg = format!("{pipe:?}");
        assert!(dbg.contains("KernelPipeline"));
        assert!(dbg.contains("num_stages: 1"));
    }

    // ======================================================================
    // Error display
    // ======================================================================

    #[test]
    fn pipeline_error_display_empty() {
        assert_eq!(PipelineError::Empty.to_string(), "pipeline has no stages");
    }

    #[test]
    fn pipeline_error_display_shape_mismatch() {
        let err = PipelineError::ShapeMismatch {
            stage_index: 1,
            expected: shape_1d(8),
            got: shape_1d(4),
        };
        let msg = err.to_string();
        assert!(msg.contains("shape mismatch at stage 1"));
    }

    #[test]
    fn pipeline_error_display_fusion() {
        let err = PipelineError::FusionUnsupported { first: 0, second: 1 };
        assert!(err.to_string().contains("fusion unsupported"));
    }
}
