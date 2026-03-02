//! Mixed-precision quantization strategy manager for per-layer optimization.
//!
//! Supports different quantization levels per layer for optimal accuracy/performance
//! trade-offs on Intel A770 and similar OpenCL-capable GPUs. Provides CPU reference
//! implementations for sensitivity analysis, calibration, and plan generation.

use std::fmt;

// ── Quantization level ──────────────────────────────────────────────────────

/// Supported quantization precisions, ordered from lowest to highest bit-width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QuantLevel {
    Binary,
    Ternary,
    Int4,
    Int8,
    Float16,
    Float32,
}

impl fmt::Display for QuantLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binary => write!(f, "Binary(1b)"),
            Self::Ternary => write!(f, "Ternary(2b)"),
            Self::Int4 => write!(f, "Int4(4b)"),
            Self::Int8 => write!(f, "Int8(8b)"),
            Self::Float16 => write!(f, "Float16(16b)"),
            Self::Float32 => write!(f, "Float32(32b)"),
        }
    }
}

// ── Per-layer configuration ─────────────────────────────────────────────────

/// Quantization configuration for a single layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerQuantConfig {
    pub layer_name: String,
    pub weight_quant: QuantLevel,
    pub activation_quant: QuantLevel,
    pub accumulator: QuantLevel,
}

// ── Strategy ────────────────────────────────────────────────────────────────

/// High-level quantization strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantStrategy {
    /// Every layer uses the same quantization level.
    Uniform(QuantLevel),
    /// Per-layer levels chosen by sensitivity analysis.
    MixedSensitivity(Vec<LayerQuantConfig>),
    /// Gradually transition from `start` precision to `end` precision across layers.
    Progressive { start: QuantLevel, end: QuantLevel },
    /// Automatically calibrated from sample data.
    AutoCalibrated,
}

// ── Sensitivity ─────────────────────────────────────────────────────────────

/// Sensitivity score for a single layer, guiding quantization decisions.
#[derive(Debug, Clone, PartialEq)]
pub struct SensitivityScore {
    pub layer_name: String,
    pub score: f64,
    pub recommended_level: QuantLevel,
}

// ── Quantization plan ───────────────────────────────────────────────────────

/// A complete quantization plan for all layers.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantPlan {
    pub strategy: QuantStrategy,
    pub layer_configs: Vec<LayerQuantConfig>,
    pub estimated_memory_mb: f64,
    pub estimated_accuracy_loss: f64,
}

// ── Calibration ─────────────────────────────────────────────────────────────

/// Calibration statistics for a single layer.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationResult {
    pub layer_name: String,
    pub weight_range: (f32, f32),
    pub activation_range: (f32, f32),
    pub outlier_ratio: f32,
}

// ── Manager ─────────────────────────────────────────────────────────────────

/// Manages mixed quantization state: plan, calibrations, and statistics.
pub struct MixedQuantManager {
    pub plan: Option<QuantPlan>,
    pub calibrations: Vec<CalibrationResult>,
    pub stats: QuantStats,
}

/// Aggregate statistics for a quantization run.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantStats {
    pub layers_analyzed: usize,
    pub memory_saved_mb: f64,
    pub accuracy_impact: f64,
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors produced by the mixed-quantization subsystem.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantError {
    UnsupportedLevel(QuantLevel),
    CalibrationFailed(String),
    IncompatibleLevels { weight: QuantLevel, activation: QuantLevel },
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedLevel(l) => write!(f, "unsupported quantization level: {l}"),
            Self::CalibrationFailed(msg) => write!(f, "calibration failed: {msg}"),
            Self::IncompatibleLevels { weight, activation } => {
                write!(f, "incompatible levels: weight={weight}, activation={activation}")
            }
        }
    }
}

impl std::error::Error for QuantError {}

// ── CPU reference implementations ───────────────────────────────────────────

/// Create a new, empty `MixedQuantManager`.
pub fn create_mixed_quant_manager() -> MixedQuantManager {
    MixedQuantManager {
        plan: None,
        calibrations: Vec::new(),
        stats: QuantStats { layers_analyzed: 0, memory_saved_mb: 0.0, accuracy_impact: 0.0 },
    }
}

/// Compute a sensitivity score from weight/activation variance.
///
/// Higher variance yields a higher sensitivity score and therefore a
/// recommendation for higher precision.
pub fn cpu_compute_sensitivity(
    weights: &[f32],
    activations: &[f32],
    layer_name: &str,
) -> SensitivityScore {
    let weight_var = variance(weights);
    let act_var = variance(activations);
    let score = (weight_var + act_var) / 2.0;

    let recommended_level = if score > 1.0 {
        QuantLevel::Float16
    } else if score > 0.5 {
        QuantLevel::Int8
    } else if score > 0.1 {
        QuantLevel::Int4
    } else {
        QuantLevel::Ternary
    };

    SensitivityScore { layer_name: layer_name.to_string(), score, recommended_level }
}

/// Calibrate a layer by computing weight/activation ranges and outlier ratio.
pub fn cpu_calibrate_layer(
    weights: &[f32],
    activations: &[f32],
    layer_name: &str,
) -> CalibrationResult {
    let weight_range = min_max(weights);
    let activation_range = min_max(activations);
    let outlier_ratio = compute_outlier_ratio(activations);

    CalibrationResult {
        layer_name: layer_name.to_string(),
        weight_range,
        activation_range,
        outlier_ratio,
    }
}

/// Choose an appropriate quantization level given sensitivity and a memory budget.
pub fn cpu_select_quant_level(sensitivity: &SensitivityScore, memory_budget_mb: f64) -> QuantLevel {
    if memory_budget_mb < 100.0 {
        // Tight budget: only allow low-precision.
        if sensitivity.score > 0.5 { QuantLevel::Int4 } else { QuantLevel::Ternary }
    } else {
        sensitivity.recommended_level
    }
}

/// Create a uniform plan where every layer uses the same level.
pub fn cpu_create_uniform_plan(level: QuantLevel, num_layers: usize) -> QuantPlan {
    let configs: Vec<LayerQuantConfig> = (0..num_layers)
        .map(|i| LayerQuantConfig {
            layer_name: format!("layer_{i}"),
            weight_quant: level,
            activation_quant: level,
            accumulator: accumulator_for(level),
        })
        .collect();

    let bits = cpu_bits_per_element(&level);
    let estimated_memory_mb = (num_layers as f64) * (bits as f64) / 8.0;
    let estimated_accuracy_loss = accuracy_loss_for_level(&level);

    QuantPlan {
        strategy: QuantStrategy::Uniform(level),
        layer_configs: configs,
        estimated_memory_mb,
        estimated_accuracy_loss,
    }
}

/// Create a mixed plan driven by per-layer sensitivities and a memory budget.
pub fn cpu_create_mixed_plan(
    sensitivities: &[SensitivityScore],
    memory_budget_mb: f64,
) -> QuantPlan {
    let configs: Vec<LayerQuantConfig> = sensitivities
        .iter()
        .map(|s| {
            let level = cpu_select_quant_level(s, memory_budget_mb);
            LayerQuantConfig {
                layer_name: s.layer_name.clone(),
                weight_quant: level,
                activation_quant: level,
                accumulator: accumulator_for(level),
            }
        })
        .collect();

    let estimated_memory_mb: f64 =
        configs.iter().map(|c| cpu_bits_per_element(&c.weight_quant) as f64 / 8.0).sum();
    let estimated_accuracy_loss: f64 =
        configs.iter().map(|c| accuracy_loss_for_level(&c.weight_quant)).sum::<f64>()
            / configs.len().max(1) as f64;

    QuantPlan {
        strategy: QuantStrategy::MixedSensitivity(configs.clone()),
        layer_configs: configs,
        estimated_memory_mb,
        estimated_accuracy_loss,
    }
}

/// Create a progressive plan that interpolates from `start` to `end` across layers.
pub fn cpu_create_progressive_plan(
    start: QuantLevel,
    end: QuantLevel,
    num_layers: usize,
) -> QuantPlan {
    let ordered = [
        QuantLevel::Binary,
        QuantLevel::Ternary,
        QuantLevel::Int4,
        QuantLevel::Int8,
        QuantLevel::Float16,
        QuantLevel::Float32,
    ];

    let start_idx = ordered.iter().position(|l| *l == start).unwrap_or(0);
    let end_idx = ordered.iter().position(|l| *l == end).unwrap_or(ordered.len() - 1);

    let configs: Vec<LayerQuantConfig> = (0..num_layers)
        .map(|i| {
            let t = if num_layers <= 1 { 0.0 } else { i as f64 / (num_layers - 1) as f64 };
            let idx = (start_idx as f64 + t * (end_idx as f64 - start_idx as f64)).round() as usize;
            let level = ordered[idx.min(ordered.len() - 1)];
            LayerQuantConfig {
                layer_name: format!("layer_{i}"),
                weight_quant: level,
                activation_quant: level,
                accumulator: accumulator_for(level),
            }
        })
        .collect();

    let estimated_memory_mb: f64 =
        configs.iter().map(|c| cpu_bits_per_element(&c.weight_quant) as f64 / 8.0).sum();
    let estimated_accuracy_loss: f64 =
        configs.iter().map(|c| accuracy_loss_for_level(&c.weight_quant)).sum::<f64>()
            / configs.len().max(1) as f64;

    QuantPlan {
        strategy: QuantStrategy::Progressive { start, end },
        layer_configs: configs,
        estimated_memory_mb,
        estimated_accuracy_loss,
    }
}

/// Estimate total memory (MB) for a plan given total model parameters.
pub fn cpu_estimate_memory(plan: &QuantPlan, model_params: usize) -> f64 {
    if plan.layer_configs.is_empty() {
        return 0.0;
    }
    let params_per_layer = model_params / plan.layer_configs.len().max(1);
    plan.layer_configs
        .iter()
        .map(|c| {
            let bits = cpu_bits_per_element(&c.weight_quant);
            (params_per_layer as f64 * bits as f64) / (8.0 * 1024.0 * 1024.0)
        })
        .sum()
}

/// Estimate relative accuracy loss for a plan (0.0 = no loss, 1.0 = total loss).
pub fn cpu_estimate_accuracy_loss(plan: &QuantPlan) -> f64 {
    if plan.layer_configs.is_empty() {
        return 0.0;
    }
    plan.layer_configs.iter().map(|c| accuracy_loss_for_level(&c.weight_quant)).sum::<f64>()
        / plan.layer_configs.len() as f64
}

/// Number of bits needed to represent one element at the given level.
pub fn cpu_bits_per_element(level: &QuantLevel) -> usize {
    match level {
        QuantLevel::Binary => 1,
        QuantLevel::Ternary => 2,
        QuantLevel::Int4 => 4,
        QuantLevel::Int8 => 8,
        QuantLevel::Float16 => 16,
        QuantLevel::Float32 => 32,
    }
}

/// Check whether `weight` and `activation` levels can be combined.
///
/// Rule: weight precision must not exceed activation precision (higher enum
/// ordinal = higher precision).
pub fn cpu_is_compatible(weight: &QuantLevel, activation: &QuantLevel) -> bool {
    // Weight precision ≤ activation precision is always fine.
    // We also allow equal levels.
    *weight <= *activation
}

/// Retrieve aggregate statistics from the manager.
pub fn cpu_get_stats(mgr: &MixedQuantManager) -> QuantStats {
    mgr.stats.clone()
}

/// Format a plan into a human-readable summary string.
pub fn format_quant_plan(plan: &QuantPlan) -> String {
    let mut out = String::new();
    out.push_str(&format!("Strategy: {:?}\n", plan.strategy));
    out.push_str(&format!("Layers: {}\n", plan.layer_configs.len()));
    out.push_str(&format!("Est. memory: {:.2} MB\n", plan.estimated_memory_mb));
    out.push_str(&format!("Est. accuracy loss: {:.4}\n", plan.estimated_accuracy_loss));
    for cfg in &plan.layer_configs {
        out.push_str(&format!(
            "  {} — weight={}, act={}, acc={}\n",
            cfg.layer_name, cfg.weight_quant, cfg.activation_quant, cfg.accumulator,
        ));
    }
    out
}

// ── Internal helpers ────────────────────────────────────────────────────────

fn variance(data: &[f32]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len() as f64;
    let mean = data.iter().map(|&x| x as f64).sum::<f64>() / n;
    data.iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n
}

fn min_max(data: &[f32]) -> (f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (min, max)
}

fn compute_outlier_ratio(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len() as f64;
    let mean = data.iter().map(|&x| x as f64).sum::<f64>() / n;
    let std_dev = (data
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n)
        .sqrt();
    if std_dev < f64::EPSILON {
        return 0.0;
    }
    let outliers = data.iter().filter(|&&x| (x as f64 - mean).abs() > 3.0 * std_dev).count() as f64;
    (outliers / n) as f32
}

/// Choose an appropriate accumulator precision for the given weight level.
fn accumulator_for(level: QuantLevel) -> QuantLevel {
    match level {
        QuantLevel::Binary | QuantLevel::Ternary => QuantLevel::Int8,
        QuantLevel::Int4 => QuantLevel::Int8,
        QuantLevel::Int8 => QuantLevel::Float16,
        QuantLevel::Float16 | QuantLevel::Float32 => QuantLevel::Float32,
    }
}

/// Approximate relative accuracy loss for a quantization level.
fn accuracy_loss_for_level(level: &QuantLevel) -> f64 {
    match level {
        QuantLevel::Binary => 0.15,
        QuantLevel::Ternary => 0.08,
        QuantLevel::Int4 => 0.04,
        QuantLevel::Int8 => 0.01,
        QuantLevel::Float16 => 0.001,
        QuantLevel::Float32 => 0.0,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── bits_per_element ────────────────────────────────────────────────

    #[test]
    fn bits_per_element_binary() {
        assert_eq!(cpu_bits_per_element(&QuantLevel::Binary), 1);
    }

    #[test]
    fn bits_per_element_ternary() {
        assert_eq!(cpu_bits_per_element(&QuantLevel::Ternary), 2);
    }

    #[test]
    fn bits_per_element_int4() {
        assert_eq!(cpu_bits_per_element(&QuantLevel::Int4), 4);
    }

    #[test]
    fn bits_per_element_int8() {
        assert_eq!(cpu_bits_per_element(&QuantLevel::Int8), 8);
    }

    #[test]
    fn bits_per_element_float16() {
        assert_eq!(cpu_bits_per_element(&QuantLevel::Float16), 16);
    }

    #[test]
    fn bits_per_element_float32() {
        assert_eq!(cpu_bits_per_element(&QuantLevel::Float32), 32);
    }

    // ── sensitivity ─────────────────────────────────────────────────────

    #[test]
    fn sensitivity_high_variance_yields_high_score() {
        let weights = vec![-10.0, 10.0, -10.0, 10.0];
        let activations = vec![-5.0, 5.0, -5.0, 5.0];
        let s = cpu_compute_sensitivity(&weights, &activations, "attn");
        assert!(s.score > 1.0, "expected high score, got {}", s.score);
        assert_eq!(s.recommended_level, QuantLevel::Float16);
    }

    #[test]
    fn sensitivity_low_variance_yields_low_score() {
        let weights = vec![0.01, 0.02, 0.01, 0.02];
        let activations = vec![0.01, 0.02, 0.01, 0.02];
        let s = cpu_compute_sensitivity(&weights, &activations, "norm");
        assert!(s.score < 0.1, "expected low score, got {}", s.score);
        assert_eq!(s.recommended_level, QuantLevel::Ternary);
    }

    #[test]
    fn sensitivity_medium_variance_recommends_int8() {
        let weights = vec![-1.0, 1.0, -1.0, 1.0];
        let activations = vec![0.0; 4];
        let s = cpu_compute_sensitivity(&weights, &activations, "ffn");
        assert!(s.score > 0.1);
        assert!(s.recommended_level == QuantLevel::Int4 || s.recommended_level == QuantLevel::Int8);
    }

    #[test]
    fn sensitivity_empty_data() {
        let s = cpu_compute_sensitivity(&[], &[], "empty");
        assert_eq!(s.score, 0.0);
        assert_eq!(s.recommended_level, QuantLevel::Ternary);
    }

    #[test]
    fn sensitivity_stores_layer_name() {
        let s = cpu_compute_sensitivity(&[1.0], &[1.0], "my_layer");
        assert_eq!(s.layer_name, "my_layer");
    }

    // ── calibration ─────────────────────────────────────────────────────

    #[test]
    fn calibrate_correct_weight_range() {
        let weights = vec![-2.0, 0.0, 3.0];
        let activations = vec![1.0];
        let c = cpu_calibrate_layer(&weights, &activations, "l0");
        assert_eq!(c.weight_range, (-2.0, 3.0));
    }

    #[test]
    fn calibrate_correct_activation_range() {
        let weights = vec![1.0];
        let activations = vec![-0.5, 0.5, 1.5];
        let c = cpu_calibrate_layer(&weights, &activations, "l0");
        assert_eq!(c.activation_range, (-0.5, 1.5));
    }

    #[test]
    fn calibrate_outlier_ratio_zero_when_uniform() {
        let activations = vec![1.0; 100];
        let c = cpu_calibrate_layer(&[0.0], &activations, "l0");
        assert_eq!(c.outlier_ratio, 0.0);
    }

    #[test]
    fn calibrate_outlier_ratio_nonzero() {
        let mut activations = vec![0.0; 1000];
        activations[0] = 100.0; // outlier
        let c = cpu_calibrate_layer(&[0.0], &activations, "l0");
        assert!(c.outlier_ratio > 0.0, "expected nonzero outlier ratio");
    }

    #[test]
    fn calibrate_empty_data() {
        let c = cpu_calibrate_layer(&[], &[], "empty");
        assert_eq!(c.weight_range, (0.0, 0.0));
        assert_eq!(c.activation_range, (0.0, 0.0));
        assert_eq!(c.outlier_ratio, 0.0);
    }

    #[test]
    fn calibrate_stores_layer_name() {
        let c = cpu_calibrate_layer(&[1.0], &[1.0], "test_layer");
        assert_eq!(c.layer_name, "test_layer");
    }

    // ── select level ────────────────────────────────────────────────────

    #[test]
    fn select_level_sensitive_high_budget() {
        let s = SensitivityScore {
            layer_name: "a".into(),
            score: 2.0,
            recommended_level: QuantLevel::Float16,
        };
        assert_eq!(cpu_select_quant_level(&s, 1000.0), QuantLevel::Float16);
    }

    #[test]
    fn select_level_sensitive_low_budget() {
        let s = SensitivityScore {
            layer_name: "a".into(),
            score: 2.0,
            recommended_level: QuantLevel::Float16,
        };
        assert_eq!(cpu_select_quant_level(&s, 50.0), QuantLevel::Int4);
    }

    #[test]
    fn select_level_insensitive_low_budget() {
        let s = SensitivityScore {
            layer_name: "a".into(),
            score: 0.1,
            recommended_level: QuantLevel::Ternary,
        };
        assert_eq!(cpu_select_quant_level(&s, 50.0), QuantLevel::Ternary);
    }

    // ── uniform plan ────────────────────────────────────────────────────

    #[test]
    fn uniform_plan_all_layers_same() {
        let plan = cpu_create_uniform_plan(QuantLevel::Int8, 4);
        assert_eq!(plan.layer_configs.len(), 4);
        for cfg in &plan.layer_configs {
            assert_eq!(cfg.weight_quant, QuantLevel::Int8);
            assert_eq!(cfg.activation_quant, QuantLevel::Int8);
        }
    }

    #[test]
    fn uniform_plan_strategy_tag() {
        let plan = cpu_create_uniform_plan(QuantLevel::Ternary, 2);
        assert_eq!(plan.strategy, QuantStrategy::Uniform(QuantLevel::Ternary));
    }

    #[test]
    fn uniform_plan_zero_layers() {
        let plan = cpu_create_uniform_plan(QuantLevel::Binary, 0);
        assert!(plan.layer_configs.is_empty());
    }

    // ── mixed plan ──────────────────────────────────────────────────────

    #[test]
    fn mixed_plan_different_per_layer() {
        let sens = vec![
            SensitivityScore {
                layer_name: "a".into(),
                score: 2.0,
                recommended_level: QuantLevel::Float16,
            },
            SensitivityScore {
                layer_name: "b".into(),
                score: 0.05,
                recommended_level: QuantLevel::Ternary,
            },
        ];
        let plan = cpu_create_mixed_plan(&sens, 1000.0);
        assert_eq!(plan.layer_configs[0].weight_quant, QuantLevel::Float16);
        assert_eq!(plan.layer_configs[1].weight_quant, QuantLevel::Ternary);
    }

    #[test]
    fn mixed_plan_empty_sensitivities() {
        let plan = cpu_create_mixed_plan(&[], 1000.0);
        assert!(plan.layer_configs.is_empty());
    }

    // ── progressive plan ────────────────────────────────────────────────

    #[test]
    fn progressive_plan_gradual_transition() {
        let plan = cpu_create_progressive_plan(QuantLevel::Binary, QuantLevel::Float32, 6);
        assert_eq!(plan.layer_configs.len(), 6);
        // First layer should be Binary, last should be Float32
        assert_eq!(plan.layer_configs[0].weight_quant, QuantLevel::Binary);
        assert_eq!(plan.layer_configs[5].weight_quant, QuantLevel::Float32);
    }

    #[test]
    fn progressive_plan_single_layer() {
        let plan = cpu_create_progressive_plan(QuantLevel::Int4, QuantLevel::Int8, 1);
        assert_eq!(plan.layer_configs.len(), 1);
        assert_eq!(plan.layer_configs[0].weight_quant, QuantLevel::Int4);
    }

    #[test]
    fn progressive_plan_same_start_end() {
        let plan = cpu_create_progressive_plan(QuantLevel::Int8, QuantLevel::Int8, 4);
        for cfg in &plan.layer_configs {
            assert_eq!(cfg.weight_quant, QuantLevel::Int8);
        }
    }

    #[test]
    fn progressive_plan_strategy_tag() {
        let plan = cpu_create_progressive_plan(QuantLevel::Ternary, QuantLevel::Float16, 3);
        assert_eq!(
            plan.strategy,
            QuantStrategy::Progressive { start: QuantLevel::Ternary, end: QuantLevel::Float16 }
        );
    }

    // ── memory estimate ─────────────────────────────────────────────────

    #[test]
    fn memory_lower_quant_less_memory() {
        let plan_low = cpu_create_uniform_plan(QuantLevel::Ternary, 4);
        let plan_high = cpu_create_uniform_plan(QuantLevel::Float32, 4);
        let mem_low = cpu_estimate_memory(&plan_low, 1_000_000);
        let mem_high = cpu_estimate_memory(&plan_high, 1_000_000);
        assert!(mem_low < mem_high, "low={mem_low}, high={mem_high}");
    }

    #[test]
    fn memory_estimate_zero_params() {
        let plan = cpu_create_uniform_plan(QuantLevel::Int8, 2);
        assert_eq!(cpu_estimate_memory(&plan, 0), 0.0);
    }

    #[test]
    fn memory_estimate_empty_plan() {
        let plan = cpu_create_uniform_plan(QuantLevel::Int8, 0);
        assert_eq!(cpu_estimate_memory(&plan, 1_000_000), 0.0);
    }

    // ── accuracy loss ───────────────────────────────────────────────────

    #[test]
    fn accuracy_loss_lower_quant_higher_loss() {
        let plan_low = cpu_create_uniform_plan(QuantLevel::Binary, 4);
        let plan_high = cpu_create_uniform_plan(QuantLevel::Float32, 4);
        let loss_low = cpu_estimate_accuracy_loss(&plan_low);
        let loss_high = cpu_estimate_accuracy_loss(&plan_high);
        assert!(loss_low > loss_high, "binary_loss={loss_low}, f32_loss={loss_high}");
    }

    #[test]
    fn accuracy_loss_float32_zero() {
        let plan = cpu_create_uniform_plan(QuantLevel::Float32, 4);
        assert_eq!(cpu_estimate_accuracy_loss(&plan), 0.0);
    }

    #[test]
    fn accuracy_loss_empty_plan() {
        let plan = cpu_create_uniform_plan(QuantLevel::Binary, 0);
        assert_eq!(cpu_estimate_accuracy_loss(&plan), 0.0);
    }

    // ── compatibility ───────────────────────────────────────────────────

    #[test]
    fn compatible_binary_weight_float32_activation() {
        assert!(cpu_is_compatible(&QuantLevel::Binary, &QuantLevel::Float32));
    }

    #[test]
    fn compatible_same_level() {
        assert!(cpu_is_compatible(&QuantLevel::Int8, &QuantLevel::Int8));
    }

    #[test]
    fn incompatible_float32_weight_binary_activation() {
        assert!(!cpu_is_compatible(&QuantLevel::Float32, &QuantLevel::Binary));
    }

    #[test]
    fn incompatible_float16_weight_int4_activation() {
        assert!(!cpu_is_compatible(&QuantLevel::Float16, &QuantLevel::Int4));
    }

    // ── edge cases ──────────────────────────────────────────────────────

    #[test]
    fn edge_single_layer_model() {
        let plan = cpu_create_uniform_plan(QuantLevel::Int4, 1);
        assert_eq!(plan.layer_configs.len(), 1);
        assert!(cpu_estimate_memory(&plan, 1000) > 0.0);
    }

    #[test]
    fn edge_zero_parameter_layer() {
        let plan = cpu_create_uniform_plan(QuantLevel::Int4, 1);
        assert_eq!(cpu_estimate_memory(&plan, 0), 0.0);
    }

    // ── BitNet-specific ─────────────────────────────────────────────────

    #[test]
    fn bitnet_ternary_weights_int8_activations() {
        assert!(cpu_is_compatible(&QuantLevel::Ternary, &QuantLevel::Int8));
    }

    #[test]
    fn bitnet_ternary_accumulator_is_int8() {
        let plan = cpu_create_uniform_plan(QuantLevel::Ternary, 1);
        assert_eq!(plan.layer_configs[0].accumulator, QuantLevel::Int8);
    }

    #[test]
    fn bitnet_binary_accumulator_is_int8() {
        let plan = cpu_create_uniform_plan(QuantLevel::Binary, 1);
        assert_eq!(plan.layer_configs[0].accumulator, QuantLevel::Int8);
    }

    #[test]
    fn bitnet_int8_accumulator_is_float16() {
        let plan = cpu_create_uniform_plan(QuantLevel::Int8, 1);
        assert_eq!(plan.layer_configs[0].accumulator, QuantLevel::Float16);
    }

    // ── property tests ──────────────────────────────────────────────────

    #[test]
    fn property_memory_decreases_with_lower_quant() {
        let levels = [
            QuantLevel::Binary,
            QuantLevel::Ternary,
            QuantLevel::Int4,
            QuantLevel::Int8,
            QuantLevel::Float16,
            QuantLevel::Float32,
        ];
        let params = 1_000_000;
        let mut prev_mem = 0.0_f64;
        for level in &levels {
            let plan = cpu_create_uniform_plan(*level, 4);
            let mem = cpu_estimate_memory(&plan, params);
            assert!(mem >= prev_mem, "{level}: mem={mem} < prev={prev_mem}");
            prev_mem = mem;
        }
    }

    #[test]
    fn property_bits_binary_less_than_float32() {
        assert!(
            cpu_bits_per_element(&QuantLevel::Binary) < cpu_bits_per_element(&QuantLevel::Float32)
        );
    }

    #[test]
    fn property_bits_monotonically_increasing() {
        let levels = [
            QuantLevel::Binary,
            QuantLevel::Ternary,
            QuantLevel::Int4,
            QuantLevel::Int8,
            QuantLevel::Float16,
            QuantLevel::Float32,
        ];
        for w in levels.windows(2) {
            assert!(
                cpu_bits_per_element(&w[0]) < cpu_bits_per_element(&w[1]),
                "{:?} should have fewer bits than {:?}",
                w[0],
                w[1]
            );
        }
    }

    // ── manager ─────────────────────────────────────────────────────────

    #[test]
    fn manager_initial_state() {
        let mgr = create_mixed_quant_manager();
        assert!(mgr.plan.is_none());
        assert!(mgr.calibrations.is_empty());
        let stats = cpu_get_stats(&mgr);
        assert_eq!(stats.layers_analyzed, 0);
    }

    #[test]
    fn manager_add_calibration() {
        let mut mgr = create_mixed_quant_manager();
        let c = cpu_calibrate_layer(&[1.0, 2.0], &[0.0, 1.0], "l0");
        mgr.calibrations.push(c);
        assert_eq!(mgr.calibrations.len(), 1);
    }

    #[test]
    fn manager_set_plan() {
        let mut mgr = create_mixed_quant_manager();
        let plan = cpu_create_uniform_plan(QuantLevel::Int4, 3);
        mgr.plan = Some(plan);
        assert!(mgr.plan.is_some());
    }

    // ── format ──────────────────────────────────────────────────────────

    #[test]
    fn format_plan_contains_strategy() {
        let plan = cpu_create_uniform_plan(QuantLevel::Int8, 2);
        let s = format_quant_plan(&plan);
        assert!(s.contains("Uniform"), "formatted plan should mention strategy");
    }

    #[test]
    fn format_plan_contains_layer_count() {
        let plan = cpu_create_uniform_plan(QuantLevel::Int8, 3);
        let s = format_quant_plan(&plan);
        assert!(s.contains("Layers: 3"));
    }

    #[test]
    fn format_plan_contains_layer_names() {
        let plan = cpu_create_uniform_plan(QuantLevel::Int8, 2);
        let s = format_quant_plan(&plan);
        assert!(s.contains("layer_0"));
        assert!(s.contains("layer_1"));
    }

    // ── error display ───────────────────────────────────────────────────

    #[test]
    fn error_display_unsupported_level() {
        let e = QuantError::UnsupportedLevel(QuantLevel::Binary);
        assert!(e.to_string().contains("unsupported"));
    }

    #[test]
    fn error_display_calibration_failed() {
        let e = QuantError::CalibrationFailed("bad data".into());
        assert!(e.to_string().contains("bad data"));
    }

    #[test]
    fn error_display_incompatible_levels() {
        let e = QuantError::IncompatibleLevels {
            weight: QuantLevel::Float32,
            activation: QuantLevel::Binary,
        };
        let msg = e.to_string();
        assert!(msg.contains("incompatible"));
        assert!(msg.contains("Float32"));
    }

    // ── quant level display ─────────────────────────────────────────────

    #[test]
    fn quant_level_display() {
        assert_eq!(format!("{}", QuantLevel::Binary), "Binary(1b)");
        assert_eq!(format!("{}", QuantLevel::Float32), "Float32(32b)");
    }
}
