//! Mixed-precision quantization support.
//!
//! Allows different quantization strategies per layer, enabling
//! accuracy-sensitive layers (e.g. first/last transformer blocks,
//! LayerNorm projections) to use higher precision while keeping
//! most layers at aggressive 2-bit quantization.

use bitnet_common::{QuantizationType, Result};
use std::collections::HashMap;

/// Per-layer quantization configuration.
#[derive(Debug, Clone)]
pub struct LayerQuantConfig {
    /// Quantization type for this layer.
    pub qtype: QuantizationType,
    /// Block size override (None = use quantizer default).
    pub block_size: Option<usize>,
    /// Whether to use asymmetric quantization.
    pub asymmetric: bool,
}

impl Default for LayerQuantConfig {
    fn default() -> Self {
        Self { qtype: QuantizationType::I2S, block_size: None, asymmetric: false }
    }
}

/// Policy for assigning quantization parameters to layers.
#[derive(Debug, Clone, Default)]
pub struct MixedPrecisionPolicy {
    /// Default config applied when no specific rule matches.
    pub default_config: LayerQuantConfig,
    /// Exact layer-name overrides (highest priority).
    layer_overrides: HashMap<String, LayerQuantConfig>,
    /// Pattern-based overrides (checked in insertion order).
    pattern_rules: Vec<(String, LayerQuantConfig)>,
}

impl MixedPrecisionPolicy {
    /// Create a uniform policy (same config for every layer).
    pub fn uniform(qtype: QuantizationType) -> Self {
        Self {
            default_config: LayerQuantConfig { qtype, block_size: None, asymmetric: false },
            layer_overrides: HashMap::new(),
            pattern_rules: Vec::new(),
        }
    }

    /// Set an exact layer-name override.
    pub fn set_layer_override(&mut self, layer_name: &str, config: LayerQuantConfig) {
        self.layer_overrides.insert(layer_name.to_string(), config);
    }

    /// Add a pattern-based rule. Patterns use simple substring matching
    /// against layer names (e.g. `"ln"` matches `"transformer.layer_0.ln"`).
    pub fn add_pattern_rule(&mut self, pattern: &str, config: LayerQuantConfig) {
        self.pattern_rules.push((pattern.to_string(), config));
    }

    /// Resolve the quantization config for a given layer name.
    ///
    /// Resolution order:
    /// 1. Exact layer override
    /// 2. First matching pattern rule
    /// 3. Default config
    pub fn resolve(&self, layer_name: &str) -> &LayerQuantConfig {
        if let Some(cfg) = self.layer_overrides.get(layer_name) {
            return cfg;
        }
        for (pattern, cfg) in &self.pattern_rules {
            if layer_name.contains(pattern.as_str()) {
                return cfg;
            }
        }
        &self.default_config
    }

    /// Return the total number of explicit overrides (layer + pattern).
    pub fn num_overrides(&self) -> usize {
        self.layer_overrides.len() + self.pattern_rules.len()
    }

    /// Validate that all referenced quantization types are supported.
    pub fn validate(&self) -> Result<()> {
        // QuantizationType is an enum — all variants are supported.
        // This hook exists for future extensibility (e.g. 4-bit, 8-bit).
        Ok(())
    }
}

/// Summary of which quantization configs are used across a model.
#[derive(Debug, Clone, Default)]
pub struct MixedPrecisionSummary {
    /// Count of layers per quantization type.
    pub type_counts: HashMap<String, usize>,
    /// Names of layers with non-default overrides.
    pub overridden_layers: Vec<String>,
}

impl MixedPrecisionSummary {
    /// Build a summary by resolving every layer against a policy.
    pub fn from_policy(layer_names: &[&str], policy: &MixedPrecisionPolicy) -> Self {
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        let mut overridden = Vec::new();

        for &name in layer_names {
            let cfg = policy.resolve(name);
            *type_counts.entry(format!("{:?}", cfg.qtype)).or_default() += 1;
            if !std::ptr::eq(cfg, &policy.default_config) {
                overridden.push(name.to_string());
            }
        }

        Self { type_counts, overridden_layers: overridden }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_policy_resolves_same_for_all() {
        let policy = MixedPrecisionPolicy::uniform(QuantizationType::TL2);
        assert_eq!(policy.resolve("layer_0").qtype, QuantizationType::TL2);
        assert_eq!(policy.resolve("anything").qtype, QuantizationType::TL2);
    }

    #[test]
    fn exact_override_takes_priority() {
        let mut policy = MixedPrecisionPolicy::default();
        policy.set_layer_override(
            "output_proj",
            LayerQuantConfig {
                qtype: QuantizationType::TL1,
                block_size: Some(64),
                asymmetric: false,
            },
        );
        assert_eq!(policy.resolve("output_proj").qtype, QuantizationType::TL1);
        assert_eq!(policy.resolve("hidden").qtype, QuantizationType::I2S);
    }

    #[test]
    fn pattern_rule_matches_substring() {
        let mut policy = MixedPrecisionPolicy::default();
        policy.add_pattern_rule(
            "ln",
            LayerQuantConfig { qtype: QuantizationType::TL1, block_size: None, asymmetric: false },
        );
        assert_eq!(policy.resolve("transformer.layer_0.ln_1").qtype, QuantizationType::TL1);
        assert_eq!(policy.resolve("transformer.layer_0.attn").qtype, QuantizationType::I2S);
    }

    #[test]
    fn exact_override_beats_pattern() {
        let mut policy = MixedPrecisionPolicy::default();
        policy.add_pattern_rule(
            "ln",
            LayerQuantConfig { qtype: QuantizationType::TL1, block_size: None, asymmetric: false },
        );
        policy.set_layer_override(
            "model.ln_final",
            LayerQuantConfig {
                qtype: QuantizationType::TL2,
                block_size: Some(128),
                asymmetric: true,
            },
        );
        let cfg = policy.resolve("model.ln_final");
        assert_eq!(cfg.qtype, QuantizationType::TL2);
        assert!(cfg.asymmetric);
    }

    #[test]
    fn num_overrides_counts_both() {
        let mut policy = MixedPrecisionPolicy::default();
        policy.set_layer_override("a", LayerQuantConfig::default());
        policy.set_layer_override("b", LayerQuantConfig::default());
        policy.add_pattern_rule("ln", LayerQuantConfig::default());
        assert_eq!(policy.num_overrides(), 3);
    }

    #[test]
    fn validate_accepts_all_builtin_types() {
        let mut policy = MixedPrecisionPolicy::uniform(QuantizationType::I2S);
        policy.set_layer_override(
            "x",
            LayerQuantConfig { qtype: QuantizationType::TL1, block_size: None, asymmetric: false },
        );
        policy.add_pattern_rule(
            "y",
            LayerQuantConfig { qtype: QuantizationType::TL2, block_size: None, asymmetric: false },
        );
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn summary_counts_types() {
        let mut policy = MixedPrecisionPolicy::default(); // I2S default
        policy.add_pattern_rule(
            "ln",
            LayerQuantConfig { qtype: QuantizationType::TL1, block_size: None, asymmetric: false },
        );
        let layers = &["attn_0", "ln_0", "attn_1", "ln_1", "output"];
        let summary = MixedPrecisionSummary::from_policy(layers, &policy);
        assert_eq!(summary.type_counts.get("I2S"), Some(&3));
        assert_eq!(summary.type_counts.get("TL1"), Some(&2));
        assert_eq!(summary.overridden_layers.len(), 2);
    }

    #[test]
    fn default_layer_config_is_i2s() {
        let cfg = LayerQuantConfig::default();
        assert_eq!(cfg.qtype, QuantizationType::I2S);
        assert!(cfg.block_size.is_none());
        assert!(!cfg.asymmetric);
    }
}
