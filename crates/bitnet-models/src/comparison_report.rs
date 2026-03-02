//! Model comparison report generation.
//!
//! Side-by-side comparison of model specifications.

use std::collections::BTreeMap;

/// A single model's specs for comparison.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: String,
    pub family: String,
    pub params_millions: u64,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub max_context: usize,
    pub activation: String,
    pub norm_type: String,
    pub weight_dtype: String,
}

impl ModelSpec {
    pub fn gqa_ratio(&self) -> f32 {
        if self.num_kv_heads == 0 {
            return 0.0;
        }
        self.num_heads as f32 / self.num_kv_heads as f32
    }

    pub fn head_dim(&self) -> usize {
        if self.num_heads == 0 {
            return 0;
        }
        self.hidden_size / self.num_heads
    }

    pub fn intermediate_size_estimate(&self) -> usize {
        // Common: 4x hidden or 8/3 * hidden
        (self.hidden_size * 8) / 3
    }
}

/// Difference between two model specs.
#[derive(Debug, Clone)]
pub struct SpecDiff {
    pub field: String,
    pub model_a: String,
    pub model_b: String,
}

/// Compare two model specs and return differences.
pub fn compare_specs(a: &ModelSpec, b: &ModelSpec) -> Vec<SpecDiff> {
    let mut diffs = Vec::new();
    let mut check = |field: &str, va: &str, vb: &str| {
        if va != vb {
            diffs.push(SpecDiff {
                field: field.to_string(),
                model_a: va.to_string(),
                model_b: vb.to_string(),
            });
        }
    };

    check("family", &a.family, &b.family);
    check("params_millions", &a.params_millions.to_string(), &b.params_millions.to_string());
    check("hidden_size", &a.hidden_size.to_string(), &b.hidden_size.to_string());
    check("num_layers", &a.num_layers.to_string(), &b.num_layers.to_string());
    check("num_heads", &a.num_heads.to_string(), &b.num_heads.to_string());
    check("num_kv_heads", &a.num_kv_heads.to_string(), &b.num_kv_heads.to_string());
    check("vocab_size", &a.vocab_size.to_string(), &b.vocab_size.to_string());
    check("max_context", &a.max_context.to_string(), &b.max_context.to_string());
    check("activation", &a.activation, &b.activation);
    check("norm_type", &a.norm_type, &b.norm_type);
    check("weight_dtype", &a.weight_dtype, &b.weight_dtype);

    diffs
}

/// A comparison report for multiple models.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    pub models: Vec<ModelSpec>,
    pub diffs: BTreeMap<String, Vec<String>>,
}

impl ComparisonReport {
    /// Build a comparison report from a list of model specs.
    pub fn build(models: Vec<ModelSpec>) -> Self {
        let mut diffs: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let fields = [
            "family",
            "params_millions",
            "hidden_size",
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "vocab_size",
            "max_context",
            "activation",
            "norm_type",
            "weight_dtype",
        ];

        for field in &fields {
            let values: Vec<String> = models
                .iter()
                .map(|m| match *field {
                    "family" => m.family.clone(),
                    "params_millions" => m.params_millions.to_string(),
                    "hidden_size" => m.hidden_size.to_string(),
                    "num_layers" => m.num_layers.to_string(),
                    "num_heads" => m.num_heads.to_string(),
                    "num_kv_heads" => m.num_kv_heads.to_string(),
                    "vocab_size" => m.vocab_size.to_string(),
                    "max_context" => m.max_context.to_string(),
                    "activation" => m.activation.clone(),
                    "norm_type" => m.norm_type.clone(),
                    "weight_dtype" => m.weight_dtype.clone(),
                    _ => String::new(),
                })
                .collect();
            diffs.insert(field.to_string(), values);
        }

        Self { models, diffs }
    }

    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Fields where all models differ.
    pub fn differing_fields(&self) -> Vec<String> {
        self.diffs
            .iter()
            .filter(|(_, vals)| {
                if vals.len() <= 1 {
                    return false;
                }
                let first = &vals[0];
                vals.iter().any(|v| v != first)
            })
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Fields where all models are identical.
    pub fn common_fields(&self) -> Vec<String> {
        self.diffs
            .iter()
            .filter(|(_, vals)| {
                if vals.len() <= 1 {
                    return true;
                }
                let first = &vals[0];
                vals.iter().all(|v| v == first)
            })
            .map(|(k, _)| k.clone())
            .collect()
    }
}

/// Well-known model specs for quick comparison.
pub fn phi4_spec() -> ModelSpec {
    ModelSpec {
        name: "Phi-4".into(),
        family: "phi".into(),
        params_millions: 14000,
        hidden_size: 5120,
        num_layers: 40,
        num_heads: 40,
        num_kv_heads: 10,
        vocab_size: 100352,
        max_context: 16384,
        activation: "silu".into(),
        norm_type: "rmsnorm".into(),
        weight_dtype: "bf16".into(),
    }
}

pub fn llama3_8b_spec() -> ModelSpec {
    ModelSpec {
        name: "LLaMA-3-8B".into(),
        family: "llama".into(),
        params_millions: 8000,
        hidden_size: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 128256,
        max_context: 8192,
        activation: "silu".into(),
        norm_type: "rmsnorm".into(),
        weight_dtype: "bf16".into(),
    }
}

pub fn bitnet_2b_spec() -> ModelSpec {
    ModelSpec {
        name: "BitNet-2B".into(),
        family: "bitnet".into(),
        params_millions: 2000,
        hidden_size: 2560,
        num_layers: 30,
        num_heads: 20,
        num_kv_heads: 5,
        vocab_size: 32000,
        max_context: 4096,
        activation: "relu2".into(),
        norm_type: "subnorm".into(),
        weight_dtype: "i2s".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gqa_ratio() {
        let s = phi4_spec();
        assert!((s.gqa_ratio() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_head_dim() {
        let s = phi4_spec();
        assert_eq!(s.head_dim(), 128);
    }

    #[test]
    fn test_compare_same() {
        let a = phi4_spec();
        let b = phi4_spec();
        assert!(compare_specs(&a, &b).is_empty());
    }

    #[test]
    fn test_compare_different() {
        let a = phi4_spec();
        let b = llama3_8b_spec();
        let diffs = compare_specs(&a, &b);
        assert!(!diffs.is_empty());
    }

    #[test]
    fn test_compare_activation() {
        let a = phi4_spec();
        let b = bitnet_2b_spec();
        let diffs = compare_specs(&a, &b);
        assert!(diffs.iter().any(|d| d.field == "activation"));
    }

    #[test]
    fn test_report_build() {
        let r = ComparisonReport::build(vec![phi4_spec(), llama3_8b_spec()]);
        assert_eq!(r.model_count(), 2);
    }

    #[test]
    fn test_report_differing() {
        let r = ComparisonReport::build(vec![phi4_spec(), llama3_8b_spec()]);
        let diff = r.differing_fields();
        assert!(diff.contains(&"hidden_size".to_string()));
    }

    #[test]
    fn test_report_common() {
        let r = ComparisonReport::build(vec![phi4_spec(), llama3_8b_spec()]);
        let common = r.common_fields();
        assert!(common.contains(&"activation".to_string())); // both silu
    }

    #[test]
    fn test_bitnet_vs_phi() {
        let a = bitnet_2b_spec();
        let b = phi4_spec();
        let diffs = compare_specs(&a, &b);
        // Should differ on activation, norm, weight_dtype
        let fields: Vec<&str> = diffs.iter().map(|d| d.field.as_str()).collect();
        assert!(fields.contains(&"activation"));
        assert!(fields.contains(&"norm_type"));
        assert!(fields.contains(&"weight_dtype"));
    }

    #[test]
    fn test_three_model_report() {
        let r = ComparisonReport::build(vec![phi4_spec(), llama3_8b_spec(), bitnet_2b_spec()]);
        assert_eq!(r.model_count(), 3);
        let diff = r.differing_fields();
        assert!(diff.len() > 3); // many fields differ
    }

    #[test]
    fn test_gqa_ratio_zero() {
        let mut s = phi4_spec();
        s.num_kv_heads = 0;
        assert_eq!(s.gqa_ratio(), 0.0);
    }

    #[test]
    fn test_head_dim_zero() {
        let mut s = phi4_spec();
        s.num_heads = 0;
        assert_eq!(s.head_dim(), 0);
    }

    #[test]
    fn test_intermediate_estimate() {
        let s = phi4_spec();
        let est = s.intermediate_size_estimate();
        assert!(est > s.hidden_size);
    }

    #[test]
    fn test_spec_names() {
        assert_eq!(phi4_spec().name, "Phi-4");
        assert_eq!(llama3_8b_spec().name, "LLaMA-3-8B");
        assert_eq!(bitnet_2b_spec().name, "BitNet-2B");
    }
}
