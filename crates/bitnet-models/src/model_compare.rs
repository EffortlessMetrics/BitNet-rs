//! Model comparison utility.
//!
//! Compare two model configurations or weight sets to detect differences.

/// A difference between two models.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelDiff {
    pub field: String,
    pub left: String,
    pub right: String,
    pub severity: DiffSeverity,
}

/// Severity of a difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffSeverity {
    Info,
    Warning,
    Error,
}

impl DiffSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

/// Comparable model config.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub architecture: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub params: Vec<(String, String)>,
}

impl ModelConfig {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            architecture: String::new(),
            num_layers: 0,
            hidden_size: 0,
            num_heads: 0,
            num_kv_heads: 0,
            vocab_size: 0,
            context_length: 0,
            params: Vec::new(),
        }
    }

    pub fn with_architecture(mut self, arch: &str) -> Self {
        self.architecture = arch.to_string();
        self
    }

    pub fn with_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }
    pub fn with_hidden(mut self, n: usize) -> Self {
        self.hidden_size = n;
        self
    }
    pub fn with_heads(mut self, heads: usize, kv: usize) -> Self {
        self.num_heads = heads;
        self.num_kv_heads = kv;
        self
    }
    pub fn with_vocab(mut self, n: usize) -> Self {
        self.vocab_size = n;
        self
    }
    pub fn with_context(mut self, n: usize) -> Self {
        self.context_length = n;
        self
    }
    pub fn with_param(mut self, key: &str, value: &str) -> Self {
        self.params.push((key.to_string(), value.to_string()));
        self
    }

    pub fn gqa_ratio(&self) -> f64 {
        if self.num_kv_heads == 0 {
            return 0.0;
        }
        self.num_heads as f64 / self.num_kv_heads as f64
    }

    pub fn estimated_params(&self) -> u64 {
        let h = self.hidden_size as u64;
        let v = self.vocab_size as u64;
        let l = self.num_layers as u64;
        // Rough: 2*v*h (embed+output) + l*(12*h^2)
        2 * v * h + l * 12 * h * h
    }
}

/// Compare two model configs.
pub fn compare_configs(left: &ModelConfig, right: &ModelConfig) -> Vec<ModelDiff> {
    let mut diffs = Vec::new();

    if left.architecture != right.architecture {
        diffs.push(ModelDiff {
            field: "architecture".into(),
            left: left.architecture.clone(),
            right: right.architecture.clone(),
            severity: DiffSeverity::Error,
        });
    }

    macro_rules! cmp_field {
        ($field:ident, $sev:expr) => {
            if left.$field != right.$field {
                diffs.push(ModelDiff {
                    field: stringify!($field).into(),
                    left: format!("{}", left.$field),
                    right: format!("{}", right.$field),
                    severity: $sev,
                });
            }
        };
    }

    cmp_field!(num_layers, DiffSeverity::Error);
    cmp_field!(hidden_size, DiffSeverity::Error);
    cmp_field!(num_heads, DiffSeverity::Warning);
    cmp_field!(num_kv_heads, DiffSeverity::Warning);
    cmp_field!(vocab_size, DiffSeverity::Warning);
    cmp_field!(context_length, DiffSeverity::Info);

    diffs
}

/// Comparison result.
#[derive(Debug)]
pub struct ComparisonResult {
    pub left_name: String,
    pub right_name: String,
    pub diffs: Vec<ModelDiff>,
}

impl ComparisonResult {
    pub fn is_compatible(&self) -> bool {
        !self.diffs.iter().any(|d| d.severity == DiffSeverity::Error)
    }

    pub fn error_count(&self) -> usize {
        self.diffs.iter().filter(|d| d.severity == DiffSeverity::Error).count()
    }

    pub fn warning_count(&self) -> usize {
        self.diffs.iter().filter(|d| d.severity == DiffSeverity::Warning).count()
    }

    pub fn has_diffs(&self) -> bool {
        !self.diffs.is_empty()
    }
}

pub fn compare(left: &ModelConfig, right: &ModelConfig) -> ComparisonResult {
    ComparisonResult {
        left_name: left.name.clone(),
        right_name: right.name.clone(),
        diffs: compare_configs(left, right),
    }
}

/// Tensor shape comparison.
#[derive(Debug, Clone)]
pub struct ShapeDiff {
    pub tensor_name: String,
    pub left_shape: Vec<usize>,
    pub right_shape: Vec<usize>,
}

pub fn compare_shapes(
    left: &[(String, Vec<usize>)],
    right: &[(String, Vec<usize>)],
) -> Vec<ShapeDiff> {
    let right_map: std::collections::HashMap<_, _> = right.iter().cloned().collect();
    let mut diffs = Vec::new();
    for (name, lshape) in left {
        if let Some(rshape) = right_map.get(name) {
            if lshape != rshape {
                diffs.push(ShapeDiff {
                    tensor_name: name.clone(),
                    left_shape: lshape.clone(),
                    right_shape: rshape.clone(),
                });
            }
        }
    }
    diffs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phi4() -> ModelConfig {
        ModelConfig::new("phi-4")
            .with_architecture("PhiForCausalLM")
            .with_layers(40)
            .with_hidden(5120)
            .with_heads(40, 10)
            .with_vocab(100352)
            .with_context(16384)
    }

    fn bitnet() -> ModelConfig {
        ModelConfig::new("bitnet-2b")
            .with_architecture("BitnetForCausalLM")
            .with_layers(30)
            .with_hidden(2560)
            .with_heads(20, 5)
            .with_vocab(32000)
            .with_context(4096)
    }

    #[test]
    fn test_identical_no_diffs() {
        let a = phi4();
        let b = phi4();
        let diffs = compare_configs(&a, &b);
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_different_arch() {
        let diffs = compare_configs(&phi4(), &bitnet());
        let arch_diff = diffs.iter().find(|d| d.field == "architecture");
        assert!(arch_diff.is_some());
        assert_eq!(arch_diff.unwrap().severity, DiffSeverity::Error);
    }

    #[test]
    fn test_layer_diff() {
        let diffs = compare_configs(&phi4(), &bitnet());
        assert!(diffs.iter().any(|d| d.field == "num_layers"));
    }

    #[test]
    fn test_comparison_result() {
        let result = compare(&phi4(), &bitnet());
        assert!(!result.is_compatible());
        assert!(result.error_count() > 0);
        assert!(result.has_diffs());
    }

    #[test]
    fn test_compatible_models() {
        let a = phi4();
        let mut b = phi4();
        b.context_length = 8192; // only info-level diff
        let result = compare(&a, &b);
        assert!(result.is_compatible());
    }

    #[test]
    fn test_gqa_ratio() {
        assert!((phi4().gqa_ratio() - 4.0).abs() < 0.01);
        assert!((bitnet().gqa_ratio() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_estimated_params() {
        let p = phi4().estimated_params();
        assert!(p > 1_000_000_000); // > 1B
    }

    #[test]
    fn test_builder() {
        let cfg = ModelConfig::new("test")
            .with_architecture("LlamaForCausalLM")
            .with_param("rope_theta", "500000.0");
        assert_eq!(cfg.params.len(), 1);
    }

    #[test]
    fn test_severity_str() {
        assert_eq!(DiffSeverity::Error.as_str(), "error");
        assert_eq!(DiffSeverity::Info.as_str(), "info");
    }

    #[test]
    fn test_compare_shapes_identical() {
        let a = vec![("w".into(), vec![3, 4])];
        let b = vec![("w".into(), vec![3, 4])];
        assert!(compare_shapes(&a, &b).is_empty());
    }

    #[test]
    fn test_compare_shapes_different() {
        let a = vec![("w".into(), vec![3, 4])];
        let b = vec![("w".into(), vec![4, 4])];
        let diffs = compare_shapes(&a, &b);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].tensor_name, "w");
    }

    #[test]
    fn test_warning_count() {
        let result = compare(&phi4(), &bitnet());
        assert!(result.warning_count() > 0);
    }
}
