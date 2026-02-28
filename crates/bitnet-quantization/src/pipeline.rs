//! Quantization pipeline orchestrating multi-stage model precision conversion.
//!
//! This module provides [`QuantizationPipeline`] which drives a calibrate → quantize →
//! verify → optimize-packing workflow, collecting per-layer error metrics and enforcing
//! a configurable error threshold.

use crate::{Quantize, QuantizedTensor, QuantizerFactory};
use bitnet_common::{BitNetError, BitNetTensor, QuantizationType, Result, Tensor};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Precision descriptor used in [`PipelineConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// 32-bit floating point (source-only).
    F32,
    /// 2-bit signed quantization.
    I2S,
    /// Table-lookup 1 (ARM NEON).
    TL1,
    /// Table-lookup 2 (x86 AVX).
    TL2,
}

impl Precision {
    /// Convert to a [`QuantizationType`] for quantization targets.
    ///
    /// Returns `None` for [`Precision::F32`] which is source-only.
    fn as_quantization_type(self) -> Option<QuantizationType> {
        match self {
            Precision::F32 => None,
            Precision::I2S => Some(QuantizationType::I2S),
            Precision::TL1 => Some(QuantizationType::TL1),
            Precision::TL2 => Some(QuantizationType::TL2),
        }
    }
}

/// Configuration for the quantization pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Precision of the incoming tensors.
    pub source_precision: Precision,
    /// Target quantization precision.
    pub target_precision: Precision,
    /// Number of calibration samples to use for activation statistics.
    pub calibration_samples: usize,
    /// Maximum acceptable per-layer MSE before the pipeline reports a threshold
    /// violation.
    pub error_threshold: f64,
}

impl PipelineConfig {
    /// Validate the configuration, returning an error on invalid combinations.
    pub fn validate(&self) -> Result<()> {
        if self.target_precision == Precision::F32 {
            return Err(BitNetError::Quantization(
                bitnet_common::QuantizationError::InvalidInput {
                    reason: "target precision cannot be F32".into(),
                },
            ));
        }
        if self.calibration_samples == 0 {
            return Err(BitNetError::Quantization(
                bitnet_common::QuantizationError::InvalidInput {
                    reason: "calibration_samples must be > 0".into(),
                },
            ));
        }
        if self.error_threshold <= 0.0 {
            return Err(BitNetError::Quantization(
                bitnet_common::QuantizationError::InvalidInput {
                    reason: "error_threshold must be positive".into(),
                },
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pipeline stages
// ---------------------------------------------------------------------------

/// Discrete stages the pipeline passes through **in order**.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QuantizationStage {
    /// Collect activation statistics from representative data.
    Calibration = 0,
    /// Quantize tensors to the target precision.
    Quantization = 1,
    /// Round-trip verify each layer and collect error metrics.
    Verification = 2,
    /// Optimise the packed representation (e.g. bit-packing compaction).
    PackingOptimization = 3,
}

// ---------------------------------------------------------------------------
// Calibration data
// ---------------------------------------------------------------------------

/// Activation statistics gathered during the calibration stage.
#[derive(Debug, Clone)]
pub struct CalibrationData {
    /// Per-layer minimum observed activation value.
    pub min_values: Vec<f32>,
    /// Per-layer maximum observed activation value.
    pub max_values: Vec<f32>,
    /// Per-layer mean activation value.
    pub mean_values: Vec<f32>,
    /// Number of samples used to compute these statistics.
    pub num_samples: usize,
}

impl CalibrationData {
    /// Create empty calibration data for `num_layers` layers.
    fn new(num_layers: usize) -> Self {
        Self {
            min_values: vec![f32::MAX; num_layers],
            max_values: vec![f32::MIN; num_layers],
            mean_values: vec![0.0; num_layers],
            num_samples: 0,
        }
    }

    /// Incorporate one sample (one value per layer) into the running stats.
    fn update(&mut self, layer_values: &[f32]) {
        for (i, &v) in layer_values.iter().enumerate() {
            if i >= self.min_values.len() {
                break;
            }
            self.min_values[i] = self.min_values[i].min(v);
            self.max_values[i] = self.max_values[i].max(v);
            // Incremental mean via Welford-style update.
            let n = self.num_samples as f32 + 1.0;
            self.mean_values[i] += (v - self.mean_values[i]) / n;
        }
        self.num_samples += 1;
    }
}

// ---------------------------------------------------------------------------
// Pipeline result
// ---------------------------------------------------------------------------

/// Metrics collected after the pipeline completes.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// MSE per layer (quantization → dequantization round-trip).
    pub per_layer_errors: Vec<f64>,
    /// Maximum single-layer MSE across all layers.
    pub max_error: f64,
    /// Mean MSE across all layers.
    pub mean_error: f64,
    /// Total size of the quantized representation in bytes.
    pub quantized_size_bytes: u64,
    /// `original_fp32_bytes / quantized_bytes`.
    pub compression_ratio: f64,
    /// `true` when **any** layer MSE exceeds the configured threshold.
    pub threshold_violated: bool,
    /// Calibration statistics gathered from representative data.
    pub calibration: CalibrationData,
    /// Quantized tensors produced by the pipeline.
    pub quantized_tensors: Vec<QuantizedTensor>,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Orchestrates a multi-stage quantization workflow.
pub struct QuantizationPipeline {
    config: PipelineConfig,
    current_stage: Option<QuantizationStage>,
}

impl QuantizationPipeline {
    /// Create a new pipeline from validated configuration.
    pub fn new(config: PipelineConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, current_stage: None })
    }

    /// Return the current pipeline stage (if any).
    pub fn current_stage(&self) -> Option<QuantizationStage> {
        self.current_stage
    }

    /// Execute the full pipeline over a slice of layers represented as
    /// [`BitNetTensor`]s.
    ///
    /// The pipeline proceeds through calibration → quantization →
    /// verification → packing optimisation, returning aggregated
    /// [`PipelineResult`] metrics.
    pub fn execute(&mut self, layers: &[BitNetTensor]) -> Result<PipelineResult> {
        if layers.is_empty() {
            return Err(BitNetError::Quantization(
                bitnet_common::QuantizationError::InvalidInput {
                    reason: "pipeline requires at least one layer".into(),
                },
            ));
        }

        let qtype = self
            .config
            .target_precision
            .as_quantization_type()
            .expect("validated in PipelineConfig::validate");

        // ---- Stage 1: Calibration ------------------------------------------
        let calibration = self.calibrate(layers)?;

        // ---- Stage 2: Quantization -----------------------------------------
        let quantized = self.quantize(layers, qtype)?;

        // ---- Stage 3: Verification -----------------------------------------
        let per_layer_errors = self.verify(layers, &quantized)?;

        // ---- Stage 4: Packing Optimisation ---------------------------------
        let quantized = self.optimize_packing(quantized)?;

        // ---- Aggregate metrics ---------------------------------------------
        let max_error = per_layer_errors.iter().copied().fold(0.0_f64, f64::max);
        let mean_error = if per_layer_errors.is_empty() {
            0.0
        } else {
            per_layer_errors.iter().sum::<f64>() / per_layer_errors.len() as f64
        };

        let quantized_size_bytes: u64 = quantized.iter().map(|q| q.data.len() as u64).sum();
        let original_bytes: u64 =
            layers.iter().map(|t| (t.shape().iter().product::<usize>() * 4) as u64).sum();
        let compression_ratio = if quantized_size_bytes == 0 {
            1.0
        } else {
            original_bytes as f64 / quantized_size_bytes as f64
        };

        let threshold_violated = per_layer_errors.iter().any(|&e| e > self.config.error_threshold);

        Ok(PipelineResult {
            per_layer_errors,
            max_error,
            mean_error,
            quantized_size_bytes,
            compression_ratio,
            threshold_violated,
            calibration,
            quantized_tensors: quantized,
        })
    }

    // -- internal stage helpers ---------------------------------------------

    fn advance_stage(&mut self, next: QuantizationStage) -> Result<()> {
        if let Some(cur) = self.current_stage
            && next <= cur
        {
            return Err(BitNetError::Quantization(
                bitnet_common::QuantizationError::InvalidInput {
                    reason: format!(
                        "stage ordering violation: cannot move from {cur:?} to {next:?}"
                    ),
                },
            ));
        }
        self.current_stage = Some(next);
        Ok(())
    }

    fn calibrate(&mut self, layers: &[BitNetTensor]) -> Result<CalibrationData> {
        self.advance_stage(QuantizationStage::Calibration)?;

        let num_layers = layers.len();
        let mut cal = CalibrationData::new(num_layers);

        for _ in 0..self.config.calibration_samples {
            let mut sample = Vec::with_capacity(num_layers);
            for layer in layers {
                let data = layer.to_vec()?;
                // Representative statistic: mean absolute value of the layer.
                let mean_abs = if data.is_empty() {
                    0.0
                } else {
                    data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32
                };
                sample.push(mean_abs);
            }
            cal.update(&sample);
        }

        Ok(cal)
    }

    fn quantize(
        &mut self,
        layers: &[BitNetTensor],
        qtype: QuantizationType,
    ) -> Result<Vec<QuantizedTensor>> {
        self.advance_stage(QuantizationStage::Quantization)?;

        let quantizer = QuantizerFactory::create(qtype);
        layers.iter().map(|t| quantizer.quantize_tensor(t)).collect()
    }

    fn verify(
        &mut self,
        originals: &[BitNetTensor],
        quantized: &[QuantizedTensor],
    ) -> Result<Vec<f64>> {
        self.advance_stage(QuantizationStage::Verification)?;

        originals
            .iter()
            .zip(quantized.iter())
            .map(|(orig, q)| {
                let deq = q.dequantize()?;
                let orig_data = orig.to_vec()?;
                let deq_data = deq.to_vec()?;

                let mse: f64 = orig_data
                    .iter()
                    .zip(deq_data.iter())
                    .map(|(a, b)| {
                        let diff = (*a as f64) - (*b as f64);
                        diff * diff
                    })
                    .sum::<f64>()
                    / orig_data.len().max(1) as f64;

                Ok(mse)
            })
            .collect()
    }

    fn optimize_packing(
        &mut self,
        quantized: Vec<QuantizedTensor>,
    ) -> Result<Vec<QuantizedTensor>> {
        self.advance_stage(QuantizationStage::PackingOptimization)?;
        // Future: re-pack for cache-line alignment, strip trailing padding,
        // etc.  For now, return as-is.
        Ok(quantized)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a single-layer f32 tensor of `n` elements.
    fn make_layer(values: &[f32]) -> BitNetTensor {
        BitNetTensor::from_slice(values, &[values.len()], &bitnet_common::Device::Cpu)
            .expect("tensor creation")
    }

    fn default_config() -> PipelineConfig {
        PipelineConfig {
            source_precision: Precision::F32,
            target_precision: Precision::I2S,
            calibration_samples: 4,
            error_threshold: 1.0,
        }
    }

    // -- full pipeline execution --------------------------------------------

    #[test]
    fn test_full_pipeline_execution() {
        let layers = vec![
            make_layer(&[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]),
            make_layer(&[1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1]),
        ];
        let mut pipeline = QuantizationPipeline::new(default_config()).expect("valid config");
        let result = pipeline.execute(&layers).expect("pipeline succeeds");

        assert_eq!(result.per_layer_errors.len(), 2);
        assert!(result.max_error >= 0.0);
        assert!(result.mean_error >= 0.0);
        assert!(result.quantized_size_bytes > 0);
        assert!(result.compression_ratio > 0.0);
        assert_eq!(result.quantized_tensors.len(), 2);
        assert_eq!(pipeline.current_stage(), Some(QuantizationStage::PackingOptimization));
    }

    // -- per-layer error tracking -------------------------------------------

    #[test]
    fn test_per_layer_error_tracking() {
        // Layer with large values → larger quantization error.
        let layers =
            vec![make_layer(&[0.01, -0.01, 0.01, -0.01]), make_layer(&[10.0, -10.0, 10.0, -10.0])];
        let mut pipeline = QuantizationPipeline::new(default_config()).expect("valid config");
        let result = pipeline.execute(&layers).expect("pipeline succeeds");

        assert_eq!(result.per_layer_errors.len(), 2);
        // The high-magnitude layer should have a higher MSE.
        assert!(
            result.per_layer_errors[1] >= result.per_layer_errors[0],
            "large-value layer should have equal or higher error"
        );
    }

    // -- compression ratio --------------------------------------------------

    #[test]
    fn test_compression_ratio_calculation() {
        let layers = vec![make_layer(&(0..128).map(|i| i as f32 * 0.01).collect::<Vec<_>>())];
        let mut pipeline = QuantizationPipeline::new(default_config()).expect("valid config");
        let result = pipeline.execute(&layers).expect("pipeline succeeds");

        // I2S packs 4 values per byte; with scale overhead, compression > 1.
        assert!(
            result.compression_ratio > 1.0,
            "compression ratio {} should exceed 1.0",
            result.compression_ratio
        );
    }

    // -- threshold violation ------------------------------------------------

    #[test]
    fn test_threshold_violation_detected() {
        let layers = vec![make_layer(&[100.0, -100.0, 50.0, -50.0])];
        let cfg = PipelineConfig {
            error_threshold: 1e-12, // impossibly tight
            ..default_config()
        };
        let mut pipeline = QuantizationPipeline::new(cfg).expect("valid config");
        let result = pipeline.execute(&layers).expect("pipeline succeeds");

        assert!(result.threshold_violated, "tight threshold should trigger violation");
    }

    #[test]
    fn test_no_threshold_violation_with_generous_threshold() {
        let layers = vec![make_layer(&[0.1, -0.1, 0.2, -0.2])];
        let cfg = PipelineConfig {
            error_threshold: 1e6, // very generous
            ..default_config()
        };
        let mut pipeline = QuantizationPipeline::new(cfg).expect("valid config");
        let result = pipeline.execute(&layers).expect("pipeline succeeds");

        assert!(!result.threshold_violated, "generous threshold should not trigger violation");
    }

    // -- config validation --------------------------------------------------

    #[test]
    fn test_config_validation_f32_target() {
        let cfg = PipelineConfig { target_precision: Precision::F32, ..default_config() };
        assert!(QuantizationPipeline::new(cfg).is_err());
    }

    #[test]
    fn test_config_validation_zero_calibration_samples() {
        let cfg = PipelineConfig { calibration_samples: 0, ..default_config() };
        assert!(QuantizationPipeline::new(cfg).is_err());
    }

    #[test]
    fn test_config_validation_negative_threshold() {
        let cfg = PipelineConfig { error_threshold: -0.5, ..default_config() };
        assert!(QuantizationPipeline::new(cfg).is_err());
    }

    #[test]
    fn test_config_validation_zero_threshold() {
        let cfg = PipelineConfig { error_threshold: 0.0, ..default_config() };
        assert!(QuantizationPipeline::new(cfg).is_err());
    }

    // -- stage ordering enforcement -----------------------------------------

    #[test]
    fn test_stage_ordering_enforced() {
        // Running execute twice on the same pipeline should fail on the second
        // call because stages cannot regress.
        let layers = vec![make_layer(&[0.1, -0.1, 0.2, -0.2])];
        let mut pipeline = QuantizationPipeline::new(default_config()).expect("valid config");
        pipeline.execute(&layers).expect("first run succeeds");

        let err = pipeline.execute(&layers);
        assert!(err.is_err(), "second execute should fail (stage regression)");
    }

    // -- empty input --------------------------------------------------------

    #[test]
    fn test_empty_input_rejected() {
        let mut pipeline = QuantizationPipeline::new(default_config()).expect("valid config");
        let err = pipeline.execute(&[]);
        assert!(err.is_err(), "empty layers should be rejected");
    }

    // -- statistics accuracy ------------------------------------------------

    #[test]
    fn test_mse_statistics_accuracy() {
        // Uniform small values → small MSE.
        let layers = vec![make_layer(&[0.0; 64])];
        let mut pipeline = QuantizationPipeline::new(default_config()).expect("valid config");
        let result = pipeline.execute(&layers).expect("pipeline succeeds");

        assert!(
            result.per_layer_errors[0] < 1e-6,
            "all-zero layer should have near-zero MSE, got {}",
            result.per_layer_errors[0]
        );
        assert!((result.max_error - result.mean_error).abs() < 1e-9);
    }

    #[test]
    fn test_max_error_is_worst_layer() {
        let layers = vec![
            make_layer(&[0.0; 32]),                  // near-zero error
            make_layer(&[50.0, -50.0, 25.0, -25.0]), // larger error
        ];
        let mut pipeline = QuantizationPipeline::new(default_config()).expect("valid config");
        let result = pipeline.execute(&layers).expect("pipeline succeeds");

        assert_eq!(
            result.max_error,
            result.per_layer_errors.iter().copied().fold(0.0_f64, f64::max)
        );
    }

    // -- target precision variants ------------------------------------------

    #[test]
    fn test_pipeline_with_tl1_target() {
        let layers = vec![make_layer(&[0.5, -0.5, 0.3, -0.3])];
        let cfg = PipelineConfig { target_precision: Precision::TL1, ..default_config() };
        let mut pipeline = QuantizationPipeline::new(cfg).expect("valid config");
        let result = pipeline.execute(&layers).expect("TL1 pipeline succeeds");
        assert!(!result.quantized_tensors.is_empty());
    }

    #[test]
    fn test_pipeline_with_tl2_target() {
        let layers = vec![make_layer(&[0.5, -0.5, 0.3, -0.3])];
        let cfg = PipelineConfig { target_precision: Precision::TL2, ..default_config() };
        let mut pipeline = QuantizationPipeline::new(cfg).expect("valid config");
        let result = pipeline.execute(&layers).expect("TL2 pipeline succeeds");
        assert!(!result.quantized_tensors.is_empty());
    }

    // -- calibration data accuracy ------------------------------------------

    #[test]
    fn test_calibration_collects_statistics() {
        let layers = vec![make_layer(&[1.0, -2.0, 3.0, -4.0]), make_layer(&[0.5, 0.5, 0.5, 0.5])];
        let cfg = PipelineConfig { calibration_samples: 8, ..default_config() };
        let mut pipeline = QuantizationPipeline::new(cfg).expect("valid config");
        let result = pipeline.execute(&layers).expect("pipeline succeeds");

        assert_eq!(result.calibration.min_values.len(), 2);
        assert_eq!(result.calibration.max_values.len(), 2);
        assert_eq!(result.calibration.num_samples, 8);
    }
}
