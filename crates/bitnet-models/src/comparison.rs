//! Model comparison utilities for side-by-side analysis of SLM architectures.

use std::fmt;

/// Summary of a model's key architectural properties.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSummary {
    pub name: String,
    pub architecture: String,
    pub num_parameters: u64,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub max_context: usize,
    pub dtype: String,
}

impl fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}, {:.1}B params, {} layers)",
            self.name,
            self.architecture,
            self.num_parameters as f64 / 1e9,
            self.num_layers,
        )
    }
}

/// Significance level for a difference between two models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Significance {
    Critical,
    Major,
    Minor,
    Informational,
}

impl fmt::Display for Significance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "CRITICAL"),
            Self::Major => write!(f, "MAJOR"),
            Self::Minor => write!(f, "MINOR"),
            Self::Informational => write!(f, "INFO"),
        }
    }
}

/// A single difference between two models.
#[derive(Debug, Clone, PartialEq)]
pub struct Difference {
    pub field: String,
    pub value_a: String,
    pub value_b: String,
    pub significance: Significance,
}

/// Result of comparing two models.
#[derive(Debug, Clone)]
pub struct ModelComparison {
    pub model_a: ModelSummary,
    pub model_b: ModelSummary,
    pub differences: Vec<Difference>,
    pub similarity_score: f32,
}

/// Compare two model summaries and produce a detailed comparison.
pub fn compare_models(a: &ModelSummary, b: &ModelSummary) -> ModelComparison {
    let mut differences = Vec::new();
    let total_fields = 9u32;
    let mut matching = 0u32;

    // Architecture
    if a.architecture != b.architecture {
        differences.push(Difference {
            field: "architecture".into(),
            value_a: a.architecture.clone(),
            value_b: b.architecture.clone(),
            significance: Significance::Critical,
        });
    } else {
        matching += 1;
    }

    // Parameters
    if a.num_parameters != b.num_parameters {
        let ratio = param_ratio(a.num_parameters, b.num_parameters);
        let significance = if ratio > 5.0 {
            Significance::Critical
        } else if ratio > 2.0 {
            Significance::Major
        } else {
            Significance::Minor
        };
        differences.push(Difference {
            field: "num_parameters".into(),
            value_a: a.num_parameters.to_string(),
            value_b: b.num_parameters.to_string(),
            significance,
        });
    } else {
        matching += 1;
    }

    // Layers
    if a.num_layers != b.num_layers {
        let sig = usize_diff_significance(a.num_layers, b.num_layers);
        differences.push(Difference {
            field: "num_layers".into(),
            value_a: a.num_layers.to_string(),
            value_b: b.num_layers.to_string(),
            significance: sig,
        });
    } else {
        matching += 1;
    }

    // Hidden size
    if a.hidden_size != b.hidden_size {
        let sig = usize_diff_significance(a.hidden_size, b.hidden_size);
        differences.push(Difference {
            field: "hidden_size".into(),
            value_a: a.hidden_size.to_string(),
            value_b: b.hidden_size.to_string(),
            significance: sig,
        });
    } else {
        matching += 1;
    }

    // Heads
    if a.num_heads != b.num_heads {
        differences.push(Difference {
            field: "num_heads".into(),
            value_a: a.num_heads.to_string(),
            value_b: b.num_heads.to_string(),
            significance: Significance::Major,
        });
    } else {
        matching += 1;
    }

    // KV heads
    if a.num_kv_heads != b.num_kv_heads {
        differences.push(Difference {
            field: "num_kv_heads".into(),
            value_a: a.num_kv_heads.to_string(),
            value_b: b.num_kv_heads.to_string(),
            significance: Significance::Minor,
        });
    } else {
        matching += 1;
    }

    // Vocab size
    if a.vocab_size != b.vocab_size {
        differences.push(Difference {
            field: "vocab_size".into(),
            value_a: a.vocab_size.to_string(),
            value_b: b.vocab_size.to_string(),
            significance: Significance::Minor,
        });
    } else {
        matching += 1;
    }

    // Max context
    if a.max_context != b.max_context {
        differences.push(Difference {
            field: "max_context".into(),
            value_a: a.max_context.to_string(),
            value_b: b.max_context.to_string(),
            significance: Significance::Minor,
        });
    } else {
        matching += 1;
    }

    // Dtype
    if a.dtype != b.dtype {
        differences.push(Difference {
            field: "dtype".into(),
            value_a: a.dtype.clone(),
            value_b: b.dtype.clone(),
            significance: Significance::Informational,
        });
    } else {
        matching += 1;
    }

    let similarity_score = matching as f32 / total_fields as f32;

    ModelComparison { model_a: a.clone(), model_b: b.clone(), differences, similarity_score }
}

/// Format a pairwise comparison as a human-readable table.
pub fn format_comparison(comparison: &ModelComparison) -> String {
    let a = &comparison.model_a;
    let b = &comparison.model_b;

    let mut out = String::new();
    out.push_str(&format!("Model Comparison: {} vs {}\n", a.name, b.name));
    out.push_str(&format!("Similarity: {:.0}%\n\n", comparison.similarity_score * 100.0));

    let header = format!("{:<16} {:<24} {:<24}\n", "Field", a.name, b.name);
    out.push_str(&header);
    out.push_str(&"-".repeat(64));
    out.push('\n');

    let rows: Vec<(&str, String, String)> = vec![
        ("architecture", a.architecture.clone(), b.architecture.clone()),
        (
            "num_parameters",
            format_param_count(a.num_parameters),
            format_param_count(b.num_parameters),
        ),
        ("num_layers", a.num_layers.to_string(), b.num_layers.to_string()),
        ("hidden_size", a.hidden_size.to_string(), b.hidden_size.to_string()),
        ("num_heads", a.num_heads.to_string(), b.num_heads.to_string()),
        ("num_kv_heads", a.num_kv_heads.to_string(), b.num_kv_heads.to_string()),
        ("vocab_size", a.vocab_size.to_string(), b.vocab_size.to_string()),
        ("max_context", a.max_context.to_string(), b.max_context.to_string()),
        ("dtype", a.dtype.clone(), b.dtype.clone()),
    ];

    for (field, va, vb) in rows {
        let marker = if va != vb { " *" } else { "" };
        out.push_str(&format!("{:<16} {:<24} {:<24}{}\n", field, va, vb, marker));
    }

    if !comparison.differences.is_empty() {
        out.push_str("\nDifferences:\n");
        for d in &comparison.differences {
            out.push_str(&format!(
                "  [{}] {}: {} vs {}\n",
                d.significance, d.field, d.value_a, d.value_b
            ));
        }
    }

    out
}

/// Format a multi-model comparison table.
pub fn format_comparison_table(models: &[ModelSummary]) -> String {
    if models.is_empty() {
        return String::from("No models to compare.\n");
    }

    let name_width = 24;
    let col_width = 16;

    let mut out = String::new();

    // Header row
    out.push_str(&format!("{:<width$}", "Model", width = name_width));
    let column_headers =
        ["Arch", "Params", "Layers", "Hidden", "Heads", "KV Heads", "Vocab", "Context", "Dtype"];
    for h in &column_headers {
        out.push_str(&format!("{:<width$}", h, width = col_width));
    }
    out.push('\n');
    out.push_str(&"-".repeat(name_width + col_width * column_headers.len()));
    out.push('\n');

    // Data rows
    for m in models {
        out.push_str(&format!("{:<width$}", m.name, width = name_width));
        out.push_str(&format!("{:<width$}", m.architecture, width = col_width));
        out.push_str(&format!(
            "{:<width$}",
            format_param_count(m.num_parameters),
            width = col_width
        ));
        out.push_str(&format!("{:<width$}", m.num_layers, width = col_width));
        out.push_str(&format!("{:<width$}", m.hidden_size, width = col_width));
        out.push_str(&format!("{:<width$}", m.num_heads, width = col_width));
        out.push_str(&format!("{:<width$}", m.num_kv_heads, width = col_width));
        out.push_str(&format!("{:<width$}", m.vocab_size, width = col_width));
        out.push_str(&format!("{:<width$}", m.max_context, width = col_width));
        out.push_str(&format!("{:<width$}", m.dtype, width = col_width));
        out.push('\n');
    }

    out
}

/// Return reference summaries for well-known SLMs.
pub fn known_model_summaries() -> Vec<ModelSummary> {
    vec![
        ModelSummary {
            name: "Phi-4".into(),
            architecture: "PhiForCausalLM".into(),
            num_parameters: 14_000_000_000,
            num_layers: 40,
            hidden_size: 5120,
            num_heads: 40,
            num_kv_heads: 10,
            vocab_size: 100352,
            max_context: 16384,
            dtype: "BF16".into(),
        },
        ModelSummary {
            name: "LLaMA-3.2-1B".into(),
            architecture: "LlamaForCausalLM".into(),
            num_parameters: 1_240_000_000,
            num_layers: 16,
            hidden_size: 2048,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 128256,
            max_context: 131072,
            dtype: "BF16".into(),
        },
        ModelSummary {
            name: "Qwen2.5-7B".into(),
            architecture: "Qwen2ForCausalLM".into(),
            num_parameters: 7_600_000_000,
            num_layers: 28,
            hidden_size: 3584,
            num_heads: 28,
            num_kv_heads: 4,
            vocab_size: 152064,
            max_context: 131072,
            dtype: "BF16".into(),
        },
        ModelSummary {
            name: "Gemma-2-2B".into(),
            architecture: "Gemma2ForCausalLM".into(),
            num_parameters: 2_600_000_000,
            num_layers: 26,
            hidden_size: 2304,
            num_heads: 8,
            num_kv_heads: 4,
            vocab_size: 256000,
            max_context: 8192,
            dtype: "BF16".into(),
        },
        ModelSummary {
            name: "Mistral-7B-v0.3".into(),
            architecture: "MistralForCausalLM".into(),
            num_parameters: 7_200_000_000,
            num_layers: 32,
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32768,
            max_context: 32768,
            dtype: "BF16".into(),
        },
        ModelSummary {
            name: "SmolLM2-1.7B".into(),
            architecture: "LlamaForCausalLM".into(),
            num_parameters: 1_700_000_000,
            num_layers: 24,
            hidden_size: 2048,
            num_heads: 32,
            num_kv_heads: 32,
            vocab_size: 49152,
            max_context: 8192,
            dtype: "BF16".into(),
        },
        ModelSummary {
            name: "BitNet-b1.58-2B".into(),
            architecture: "BitnetForCausalLM".into(),
            num_parameters: 2_000_000_000,
            num_layers: 26,
            hidden_size: 2560,
            num_heads: 32,
            num_kv_heads: 32,
            vocab_size: 32000,
            max_context: 4096,
            dtype: "I2_S".into(),
        },
    ]
}

// --- helpers ---

fn param_ratio(a: u64, b: u64) -> f64 {
    let (big, small) = if a > b { (a, b) } else { (b, a) };
    if small == 0 {
        return f64::INFINITY;
    }
    big as f64 / small as f64
}

fn usize_diff_significance(a: usize, b: usize) -> Significance {
    let ratio = param_ratio(a as u64, b as u64);
    if ratio > 3.0 {
        Significance::Major
    } else if ratio > 1.5 {
        Significance::Minor
    } else {
        Significance::Informational
    }
}

fn format_param_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.0}M", n as f64 / 1e6)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_a() -> ModelSummary {
        ModelSummary {
            name: "ModelA".into(),
            architecture: "LlamaForCausalLM".into(),
            num_parameters: 1_000_000_000,
            num_layers: 24,
            hidden_size: 2048,
            num_heads: 16,
            num_kv_heads: 4,
            vocab_size: 32000,
            max_context: 4096,
            dtype: "BF16".into(),
        }
    }

    fn sample_b() -> ModelSummary {
        ModelSummary {
            name: "ModelB".into(),
            architecture: "MistralForCausalLM".into(),
            num_parameters: 7_000_000_000,
            num_layers: 32,
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            max_context: 32768,
            dtype: "BF16".into(),
        }
    }

    #[test]
    fn test_compare_identical_models() {
        let a = sample_a();
        let cmp = compare_models(&a, &a);
        assert!(cmp.differences.is_empty());
        assert!((cmp.similarity_score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compare_different_models() {
        let cmp = compare_models(&sample_a(), &sample_b());
        assert!(!cmp.differences.is_empty());
        assert!(cmp.similarity_score < 1.0);
    }

    #[test]
    fn test_similarity_score_range() {
        let cmp = compare_models(&sample_a(), &sample_b());
        assert!(cmp.similarity_score >= 0.0);
        assert!(cmp.similarity_score <= 1.0);
    }

    #[test]
    fn test_architecture_difference_is_critical() {
        let cmp = compare_models(&sample_a(), &sample_b());
        let arch_diff = cmp.differences.iter().find(|d| d.field == "architecture").unwrap();
        assert_eq!(arch_diff.significance, Significance::Critical);
    }

    #[test]
    fn test_large_param_difference_is_critical() {
        let mut a = sample_a();
        let mut b = sample_a();
        a.num_parameters = 1_000_000_000;
        b.num_parameters = 14_000_000_000;
        let cmp = compare_models(&a, &b);
        let diff = cmp.differences.iter().find(|d| d.field == "num_parameters").unwrap();
        assert_eq!(diff.significance, Significance::Critical);
    }

    #[test]
    fn test_moderate_param_difference_is_major() {
        let mut a = sample_a();
        let mut b = sample_a();
        a.num_parameters = 3_000_000_000;
        b.num_parameters = 7_000_000_000;
        let cmp = compare_models(&a, &b);
        let diff = cmp.differences.iter().find(|d| d.field == "num_parameters").unwrap();
        assert_eq!(diff.significance, Significance::Major);
    }

    #[test]
    fn test_small_param_difference_is_minor() {
        let mut a = sample_a();
        let mut b = sample_a();
        a.num_parameters = 1_000_000_000;
        b.num_parameters = 1_200_000_000;
        let cmp = compare_models(&a, &b);
        let diff = cmp.differences.iter().find(|d| d.field == "num_parameters").unwrap();
        assert_eq!(diff.significance, Significance::Minor);
    }

    #[test]
    fn test_dtype_difference_is_informational() {
        let mut a = sample_a();
        let mut b = sample_a();
        a.dtype = "BF16".into();
        b.dtype = "I2_S".into();
        let cmp = compare_models(&a, &b);
        let diff = cmp.differences.iter().find(|d| d.field == "dtype").unwrap();
        assert_eq!(diff.significance, Significance::Informational);
    }

    #[test]
    fn test_difference_count_matches_changed_fields() {
        let mut b = sample_a();
        b.num_layers = 32;
        b.hidden_size = 4096;
        let cmp = compare_models(&sample_a(), &b);
        assert_eq!(cmp.differences.len(), 2);
    }

    #[test]
    fn test_format_comparison_contains_model_names() {
        let cmp = compare_models(&sample_a(), &sample_b());
        let text = format_comparison(&cmp);
        assert!(text.contains("ModelA"));
        assert!(text.contains("ModelB"));
    }

    #[test]
    fn test_format_comparison_contains_similarity() {
        let cmp = compare_models(&sample_a(), &sample_a());
        let text = format_comparison(&cmp);
        assert!(text.contains("100%"));
    }

    #[test]
    fn test_format_comparison_marks_differences() {
        let cmp = compare_models(&sample_a(), &sample_b());
        let text = format_comparison(&cmp);
        assert!(text.contains(" *"));
    }

    #[test]
    fn test_format_comparison_table_header() {
        let models = known_model_summaries();
        let table = format_comparison_table(&models);
        assert!(table.contains("Model"));
        assert!(table.contains("Arch"));
        assert!(table.contains("Params"));
    }

    #[test]
    fn test_format_comparison_table_rows() {
        let models = known_model_summaries();
        let table = format_comparison_table(&models);
        assert!(table.contains("Phi-4"));
        assert!(table.contains("BitNet-b1.58-2B"));
    }

    #[test]
    fn test_format_comparison_table_empty() {
        let table = format_comparison_table(&[]);
        assert_eq!(table, "No models to compare.\n");
    }

    #[test]
    fn test_known_model_summaries_count() {
        let models = known_model_summaries();
        assert_eq!(models.len(), 7);
    }

    #[test]
    fn test_known_model_summaries_names() {
        let models = known_model_summaries();
        let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"Phi-4"));
        assert!(names.contains(&"LLaMA-3.2-1B"));
        assert!(names.contains(&"Qwen2.5-7B"));
        assert!(names.contains(&"Gemma-2-2B"));
        assert!(names.contains(&"Mistral-7B-v0.3"));
        assert!(names.contains(&"SmolLM2-1.7B"));
        assert!(names.contains(&"BitNet-b1.58-2B"));
    }

    #[test]
    fn test_bitnet_model_dtype() {
        let models = known_model_summaries();
        let bitnet = models.iter().find(|m| m.name == "BitNet-b1.58-2B").unwrap();
        assert_eq!(bitnet.dtype, "I2_S");
    }

    #[test]
    fn test_model_summary_display() {
        let m = sample_a();
        let s = format!("{}", m);
        assert!(s.contains("ModelA"));
        assert!(s.contains("1.0B"));
    }

    #[test]
    fn test_significance_ordering() {
        assert!(Significance::Critical < Significance::Major);
        assert!(Significance::Major < Significance::Minor);
        assert!(Significance::Minor < Significance::Informational);
    }

    #[test]
    fn test_significance_display() {
        assert_eq!(format!("{}", Significance::Critical), "CRITICAL");
        assert_eq!(format!("{}", Significance::Informational), "INFO");
    }

    #[test]
    fn test_format_param_count_billions() {
        assert_eq!(format_param_count(14_000_000_000), "14.0B");
        assert_eq!(format_param_count(1_240_000_000), "1.2B");
    }

    #[test]
    fn test_format_param_count_millions() {
        assert_eq!(format_param_count(350_000_000), "350M");
    }

    #[test]
    fn test_format_param_count_small() {
        assert_eq!(format_param_count(999), "999");
    }

    #[test]
    fn test_compare_models_symmetric_score() {
        let a = sample_a();
        let b = sample_b();
        let ab = compare_models(&a, &b);
        let ba = compare_models(&b, &a);
        assert!((ab.similarity_score - ba.similarity_score).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compare_models_symmetric_diff_count() {
        let a = sample_a();
        let b = sample_b();
        assert_eq!(
            compare_models(&a, &b).differences.len(),
            compare_models(&b, &a).differences.len()
        );
    }

    #[test]
    fn test_param_ratio_zero() {
        assert!(param_ratio(100, 0).is_infinite());
    }
}
