//! Weight quantization planning utilities.
//!
//! Estimates quantization impact and selects optimal configurations.

/// Quantization format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    F32,
    F16,
    BF16,
    Int8,
    Int4,
    I2S,
    QK256,
}

impl QuantFormat {
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::F32 => 32.0,
            Self::F16 | Self::BF16 => 16.0,
            Self::Int8 => 8.0,
            Self::Int4 => 4.5, // with scales overhead
            Self::I2S => 2.0,
            Self::QK256 => 2.3, // with block scales
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::I2S => "i2s",
            Self::QK256 => "qk256",
        }
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, Self::Int8 | Self::Int4 | Self::I2S | Self::QK256)
    }
}

/// Quantization plan for a model.
#[derive(Debug, Clone)]
pub struct QuantPlan {
    pub source_format: QuantFormat,
    pub target_format: QuantFormat,
    pub param_count: u64,
    pub source_bytes: u64,
    pub target_bytes: u64,
    pub compression_ratio: f32,
    pub estimated_quality_loss: f32,
}

/// Plan quantization from source to target format.
pub fn plan_quantization(param_count: u64, source: QuantFormat, target: QuantFormat) -> QuantPlan {
    let source_bytes = (param_count as f64 * source.bits_per_weight() as f64 / 8.0) as u64;
    let target_bytes = (param_count as f64 * target.bits_per_weight() as f64 / 8.0) as u64;
    let compression =
        if target_bytes > 0 { source_bytes as f32 / target_bytes as f32 } else { 0.0 };

    // Rough quality loss estimates based on literature
    let quality_loss = match (&source, &target) {
        (_, QuantFormat::F32) => 0.0,
        (QuantFormat::F32, QuantFormat::F16) | (QuantFormat::F32, QuantFormat::BF16) => 0.001,
        (_, QuantFormat::F16) | (_, QuantFormat::BF16) => 0.001,
        (_, QuantFormat::Int8) => 0.01,
        (_, QuantFormat::Int4) => 0.03,
        (_, QuantFormat::I2S) => 0.05,
        (_, QuantFormat::QK256) => 0.04,
    };

    QuantPlan {
        source_format: source,
        target_format: target,
        param_count,
        source_bytes,
        target_bytes,
        compression_ratio: compression,
        estimated_quality_loss: quality_loss,
    }
}

/// Compare multiple quantization options.
pub fn compare_formats(
    param_count: u64,
    source: QuantFormat,
    targets: &[QuantFormat],
) -> Vec<QuantPlan> {
    targets.iter().map(|&t| plan_quantization(param_count, source, t)).collect()
}

/// Recommend the best quantization for a given memory budget.
pub fn recommend_for_budget(
    param_count: u64,
    source: QuantFormat,
    max_bytes: u64,
) -> Option<QuantFormat> {
    let candidates = [
        QuantFormat::F16,
        QuantFormat::BF16,
        QuantFormat::Int8,
        QuantFormat::Int4,
        QuantFormat::I2S,
        QuantFormat::QK256,
    ];

    // Pick the highest-quality format that fits
    candidates
        .iter()
        .map(|&fmt| (fmt, plan_quantization(param_count, source, fmt)))
        .filter(|(_, plan)| plan.target_bytes <= max_bytes)
        .min_by(|(_, a), (_, b)| {
            a.estimated_quality_loss.partial_cmp(&b.estimated_quality_loss).unwrap()
        })
        .map(|(fmt, _)| fmt)
}

/// Estimated memory savings in bytes.
pub fn memory_savings(plan: &QuantPlan) -> i64 {
    plan.source_bytes as i64 - plan.target_bytes as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits_per_weight() {
        assert_eq!(QuantFormat::F32.bits_per_weight(), 32.0);
        assert_eq!(QuantFormat::F16.bits_per_weight(), 16.0);
        assert_eq!(QuantFormat::Int8.bits_per_weight(), 8.0);
        assert_eq!(QuantFormat::I2S.bits_per_weight(), 2.0);
    }

    #[test]
    fn test_format_name() {
        assert_eq!(QuantFormat::Int4.name(), "int4");
        assert_eq!(QuantFormat::QK256.name(), "qk256");
    }

    #[test]
    fn test_is_integer() {
        assert!(QuantFormat::Int8.is_integer());
        assert!(QuantFormat::I2S.is_integer());
        assert!(!QuantFormat::F32.is_integer());
        assert!(!QuantFormat::BF16.is_integer());
    }

    #[test]
    fn test_plan_f32_to_f16() {
        let p = plan_quantization(1_000_000, QuantFormat::F32, QuantFormat::F16);
        assert_eq!(p.source_bytes, 4_000_000);
        assert_eq!(p.target_bytes, 2_000_000);
        assert!((p.compression_ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_plan_f32_to_int4() {
        let p = plan_quantization(1_000_000, QuantFormat::F32, QuantFormat::Int4);
        assert!(p.target_bytes < p.source_bytes);
        assert!(p.compression_ratio > 5.0);
    }

    #[test]
    fn test_plan_quality_loss() {
        let p16 = plan_quantization(1000, QuantFormat::F32, QuantFormat::F16);
        let p8 = plan_quantization(1000, QuantFormat::F32, QuantFormat::Int8);
        let p4 = plan_quantization(1000, QuantFormat::F32, QuantFormat::Int4);
        assert!(p16.estimated_quality_loss < p8.estimated_quality_loss);
        assert!(p8.estimated_quality_loss < p4.estimated_quality_loss);
    }

    #[test]
    fn test_compare_formats() {
        let plans = compare_formats(
            1_000_000,
            QuantFormat::F32,
            &[QuantFormat::F16, QuantFormat::Int8, QuantFormat::Int4],
        );
        assert_eq!(plans.len(), 3);
    }

    #[test]
    fn test_recommend_large_budget() {
        let rec = recommend_for_budget(1_000_000, QuantFormat::F32, 10_000_000);
        // Should pick F16/BF16 — highest quality that fits
        assert!(rec.is_some());
        let r = rec.unwrap();
        assert!(r == QuantFormat::F16 || r == QuantFormat::BF16);
    }

    #[test]
    fn test_recommend_small_budget() {
        let rec = recommend_for_budget(1_000_000, QuantFormat::F32, 500_000);
        // Only I2S and QK256 fit under 500KB for 1M params
        assert!(rec.is_some());
    }

    #[test]
    fn test_recommend_zero_budget() {
        let rec = recommend_for_budget(1_000_000, QuantFormat::F32, 0);
        assert!(rec.is_none());
    }

    #[test]
    fn test_memory_savings() {
        let p = plan_quantization(1_000_000, QuantFormat::F32, QuantFormat::F16);
        assert_eq!(memory_savings(&p), 2_000_000);
    }

    #[test]
    fn test_memory_savings_same() {
        let p = plan_quantization(1_000_000, QuantFormat::F32, QuantFormat::F32);
        assert_eq!(memory_savings(&p), 0);
    }

    #[test]
    fn test_phi4_plan() {
        // Phi-4: ~14B params, BF16 → Int4
        let p = plan_quantization(14_000_000_000, QuantFormat::BF16, QuantFormat::Int4);
        // 14B * 16bits = 28GB → 14B * 4.5bits ≈ 7.9GB
        assert!(p.target_bytes < 10_000_000_000);
        assert!(p.compression_ratio > 3.0);
    }
}
