//! Quantization strategy planner.
//!
//! Given model characteristics and hardware constraints, recommend
//! a quantization strategy that balances accuracy and performance.

/// Quantization precision level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QuantLevel {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
    INT2,
}

impl QuantLevel {
    pub fn bits_per_weight(&self) -> u32 {
        match self {
            Self::FP32 => 32,
            Self::FP16 | Self::BF16 => 16,
            Self::INT8 => 8,
            Self::INT4 => 4,
            Self::INT2 => 2,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::FP32 => "FP32",
            Self::FP16 => "FP16",
            Self::BF16 => "BF16",
            Self::INT8 => "INT8",
            Self::INT4 => "INT4",
            Self::INT2 => "INT2",
        }
    }

    pub fn compression_ratio_vs_fp32(&self) -> f64 {
        32.0 / self.bits_per_weight() as f64
    }
}

/// Layer type classification for mixed-precision planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Embedding,
    Attention,
    FeedForward,
    Normalization,
    OutputHead,
}

/// Per-layer quantization decision.
#[derive(Debug, Clone)]
pub struct LayerPlan {
    pub layer_name: String,
    pub kind: LayerKind,
    pub level: QuantLevel,
    pub param_count: usize,
    pub rationale: String,
}

impl LayerPlan {
    pub fn size_bytes(&self) -> usize {
        (self.param_count as f64 * self.level.bits_per_weight() as f64 / 8.0) as usize
    }
}

/// Hardware constraints.
#[derive(Debug, Clone)]
pub struct HardwareConstraints {
    pub available_memory_bytes: usize,
    pub supports_int8: bool,
    pub supports_int4: bool,
    pub supports_bf16: bool,
    pub prefer_speed: bool,
}

impl Default for HardwareConstraints {
    fn default() -> Self {
        Self {
            available_memory_bytes: 16 * 1024 * 1024 * 1024, // 16 GB
            supports_int8: true,
            supports_int4: false,
            supports_bf16: false,
            prefer_speed: false,
        }
    }
}

/// Model profile for planning.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    pub total_params: usize,
    pub layers: Vec<(String, LayerKind, usize)>, // (name, kind, param_count)
    pub original_dtype: QuantLevel,
}

/// Complete quantization plan.
#[derive(Debug)]
pub struct QuantPlan {
    pub layer_plans: Vec<LayerPlan>,
    pub total_original_bytes: usize,
    pub total_quantized_bytes: usize,
}

impl QuantPlan {
    pub fn compression_ratio(&self) -> f64 {
        if self.total_quantized_bytes == 0 {
            return 0.0;
        }
        self.total_original_bytes as f64 / self.total_quantized_bytes as f64
    }

    pub fn layer_count(&self) -> usize {
        self.layer_plans.len()
    }

    pub fn memory_saved_bytes(&self) -> usize {
        self.total_original_bytes.saturating_sub(self.total_quantized_bytes)
    }

    pub fn unique_levels(&self) -> Vec<QuantLevel> {
        let mut levels: Vec<_> = self.layer_plans.iter().map(|p| p.level).collect();
        levels.sort();
        levels.dedup();
        levels
    }
}

/// Plan quantization for a model.
pub fn plan_quantization(profile: &ModelProfile, constraints: &HardwareConstraints) -> QuantPlan {
    let original_bytes_per_param = profile.original_dtype.bits_per_weight() as f64 / 8.0;
    let total_original_bytes = (profile.total_params as f64 * original_bytes_per_param) as usize;

    let mut layer_plans = Vec::new();

    for (name, kind, params) in &profile.layers {
        let (level, rationale) = select_level(*kind, constraints);
        layer_plans.push(LayerPlan {
            layer_name: name.clone(),
            kind: *kind,
            level,
            param_count: *params,
            rationale,
        });
    }

    let total_quantized_bytes: usize = layer_plans.iter().map(|p| p.size_bytes()).sum();

    QuantPlan { layer_plans, total_original_bytes, total_quantized_bytes }
}

fn select_level(kind: LayerKind, constraints: &HardwareConstraints) -> (QuantLevel, String) {
    match kind {
        LayerKind::Normalization => {
            (QuantLevel::FP16, "normalization kept at FP16 for numerical stability".into())
        }
        LayerKind::Embedding | LayerKind::OutputHead => {
            if constraints.supports_int8 {
                (QuantLevel::INT8, "embedding/head quantized to INT8 for memory savings".into())
            } else {
                (QuantLevel::FP16, "embedding/head kept at FP16 (INT8 not supported)".into())
            }
        }
        LayerKind::Attention | LayerKind::FeedForward => {
            if constraints.supports_int4 && constraints.prefer_speed {
                (QuantLevel::INT4, "aggressive INT4 for speed".into())
            } else if constraints.supports_int8 {
                (QuantLevel::INT8, "INT8 for balanced accuracy/speed".into())
            } else if constraints.supports_bf16 {
                (QuantLevel::BF16, "BF16 for hardware-native computation".into())
            } else {
                (QuantLevel::FP16, "FP16 default (no lower precision supported)".into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_profile() -> ModelProfile {
        ModelProfile {
            total_params: 1_000_000,
            layers: vec![
                ("embed".into(), LayerKind::Embedding, 500_000),
                ("attn.0".into(), LayerKind::Attention, 200_000),
                ("ffn.0".into(), LayerKind::FeedForward, 200_000),
                ("norm".into(), LayerKind::Normalization, 50_000),
                ("head".into(), LayerKind::OutputHead, 50_000),
            ],
            original_dtype: QuantLevel::FP16,
        }
    }

    #[test]
    fn test_quant_level_bits() {
        assert_eq!(QuantLevel::INT4.bits_per_weight(), 4);
        assert_eq!(QuantLevel::FP32.bits_per_weight(), 32);
    }

    #[test]
    fn test_compression_ratio() {
        assert!((QuantLevel::INT8.compression_ratio_vs_fp32() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_plan_default_constraints() {
        let plan = plan_quantization(&sample_profile(), &HardwareConstraints::default());
        assert_eq!(plan.layer_count(), 5);
        assert!(plan.compression_ratio() > 1.0);
    }

    #[test]
    fn test_norm_stays_fp16() {
        let plan = plan_quantization(&sample_profile(), &HardwareConstraints::default());
        let norm = plan.layer_plans.iter().find(|p| p.kind == LayerKind::Normalization).unwrap();
        assert_eq!(norm.level, QuantLevel::FP16);
    }

    #[test]
    fn test_int4_aggressive() {
        let hw =
            HardwareConstraints { supports_int4: true, prefer_speed: true, ..Default::default() };
        let plan = plan_quantization(&sample_profile(), &hw);
        let attn = plan.layer_plans.iter().find(|p| p.kind == LayerKind::Attention).unwrap();
        assert_eq!(attn.level, QuantLevel::INT4);
    }

    #[test]
    fn test_bf16_fallback() {
        let hw =
            HardwareConstraints { supports_int8: false, supports_bf16: true, ..Default::default() };
        let plan = plan_quantization(&sample_profile(), &hw);
        let ffn = plan.layer_plans.iter().find(|p| p.kind == LayerKind::FeedForward).unwrap();
        assert_eq!(ffn.level, QuantLevel::BF16);
    }

    #[test]
    fn test_memory_saved() {
        let plan = plan_quantization(&sample_profile(), &HardwareConstraints::default());
        assert!(plan.memory_saved_bytes() > 0);
    }

    #[test]
    fn test_unique_levels() {
        let plan = plan_quantization(&sample_profile(), &HardwareConstraints::default());
        let levels = plan.unique_levels();
        assert!(levels.contains(&QuantLevel::FP16));
        assert!(levels.contains(&QuantLevel::INT8));
    }

    #[test]
    fn test_layer_plan_size() {
        let lp = LayerPlan {
            layer_name: "test".into(),
            kind: LayerKind::Attention,
            level: QuantLevel::INT8,
            param_count: 1000,
            rationale: "test".into(),
        };
        assert_eq!(lp.size_bytes(), 1000);
    }

    #[test]
    fn test_fp16_only_hw() {
        let hw = HardwareConstraints {
            supports_int8: false,
            supports_int4: false,
            supports_bf16: false,
            ..Default::default()
        };
        let plan = plan_quantization(&sample_profile(), &hw);
        assert!(plan.layer_plans.iter().all(|p| p.level == QuantLevel::FP16));
    }

    #[test]
    fn test_empty_model() {
        let profile =
            ModelProfile { total_params: 0, layers: vec![], original_dtype: QuantLevel::FP16 };
        let plan = plan_quantization(&profile, &HardwareConstraints::default());
        assert_eq!(plan.layer_count(), 0);
    }

    #[test]
    fn test_quant_level_name() {
        assert_eq!(QuantLevel::INT4.name(), "INT4");
        assert_eq!(QuantLevel::BF16.name(), "BF16");
    }
}
