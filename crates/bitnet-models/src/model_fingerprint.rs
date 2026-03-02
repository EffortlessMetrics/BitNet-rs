//! Model fingerprinting for identity and integrity verification.
//!
//! Generates a compact fingerprint from model metadata without
//! reading all weight data (fast, O(1) in model size).

use std::collections::BTreeMap;
use std::fmt;

/// A compact model fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelFingerprint {
    /// Architecture family (e.g., "phi", "llama").
    pub architecture: String,
    /// Number of parameters (approximate).
    pub param_count: u64,
    /// Number of layers.
    pub num_layers: u32,
    /// Hidden dimension.
    pub hidden_size: u32,
    /// Number of attention heads.
    pub num_heads: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Quantization type (e.g., "f16", "i2s", "q4_0").
    pub quant_type: String,
    /// Extra metadata tags.
    pub tags: BTreeMap<String, String>,
}

impl ModelFingerprint {
    pub fn new(architecture: &str) -> Self {
        Self {
            architecture: architecture.to_string(),
            param_count: 0,
            num_layers: 0,
            hidden_size: 0,
            num_heads: 0,
            vocab_size: 0,
            quant_type: String::new(),
            tags: BTreeMap::new(),
        }
    }

    pub fn with_param_count(mut self, count: u64) -> Self {
        self.param_count = count;
        self
    }

    pub fn with_layers(mut self, n: u32) -> Self {
        self.num_layers = n;
        self
    }

    pub fn with_hidden_size(mut self, size: u32) -> Self {
        self.hidden_size = size;
        self
    }

    pub fn with_heads(mut self, n: u32) -> Self {
        self.num_heads = n;
        self
    }

    pub fn with_vocab_size(mut self, size: u32) -> Self {
        self.vocab_size = size;
        self
    }

    pub fn with_quant_type(mut self, qt: &str) -> Self {
        self.quant_type = qt.to_string();
        self
    }

    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Compact string identifier. Format: `arch-layers-hidden-heads-vocab-quant`.
    pub fn compact_id(&self) -> String {
        format!(
            "{}-L{}-H{}-A{}-V{}-{}",
            self.architecture, self.num_layers, self.hidden_size,
            self.num_heads, self.vocab_size, self.quant_type
        )
    }

    /// Estimated memory for weights (bytes) based on quant type.
    pub fn estimated_weight_bytes(&self) -> u64 {
        let bits_per_param = match self.quant_type.as_str() {
            "f32" => 32,
            "f16" | "bf16" => 16,
            "i8" | "q8_0" => 8,
            "i4" | "q4_0" | "q4_1" => 4,
            "i2s" | "i2_s" => 2,
            _ => 16, // default to f16
        };
        self.param_count * bits_per_param / 8
    }

    /// Whether this is a quantized model (less than 16 bits per param).
    pub fn is_quantized(&self) -> bool {
        matches!(
            self.quant_type.as_str(),
            "i2s" | "i2_s" | "i4" | "q4_0" | "q4_1" | "i8" | "q8_0"
        )
    }

    /// Whether two fingerprints represent the same architecture.
    pub fn same_architecture(&self, other: &Self) -> bool {
        self.architecture == other.architecture
            && self.num_layers == other.num_layers
            && self.hidden_size == other.hidden_size
            && self.num_heads == other.num_heads
    }

    /// Whether two fingerprints are the same model at different quantization.
    pub fn same_model_different_quant(&self, other: &Self) -> bool {
        self.same_architecture(other)
            && self.vocab_size == other.vocab_size
            && self.quant_type != other.quant_type
    }

    /// Human-readable size label.
    pub fn size_label(&self) -> String {
        let billions = self.param_count as f64 / 1e9;
        if billions >= 1.0 {
            format!("{:.1}B", billions)
        } else {
            let millions = self.param_count as f64 / 1e6;
            format!("{:.0}M", millions)
        }
    }
}

impl fmt::Display for ModelFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}, {} layers, hidden={}, heads={}, vocab={}, quant={})",
            self.architecture,
            self.size_label(),
            self.num_layers,
            self.hidden_size,
            self.num_heads,
            self.vocab_size,
            self.quant_type,
        )
    }
}

/// Well-known model fingerprints for quick identification.
pub fn known_fingerprints() -> Vec<ModelFingerprint> {
    vec![
        ModelFingerprint::new("bitnet")
            .with_param_count(2_000_000_000)
            .with_layers(30)
            .with_hidden_size(2560)
            .with_heads(20)
            .with_vocab_size(32000)
            .with_quant_type("i2s")
            .with_tag("name", "bitnet-b1.58-2B-4T"),
        ModelFingerprint::new("phi")
            .with_param_count(14_000_000_000)
            .with_layers(40)
            .with_hidden_size(5120)
            .with_heads(40)
            .with_vocab_size(100352)
            .with_quant_type("bf16")
            .with_tag("name", "phi-4"),
        ModelFingerprint::new("llama")
            .with_param_count(8_000_000_000)
            .with_layers(32)
            .with_hidden_size(4096)
            .with_heads(32)
            .with_vocab_size(128256)
            .with_quant_type("bf16")
            .with_tag("name", "llama-3.1-8B"),
        ModelFingerprint::new("qwen2")
            .with_param_count(7_000_000_000)
            .with_layers(28)
            .with_hidden_size(3584)
            .with_heads(28)
            .with_vocab_size(152064)
            .with_quant_type("bf16")
            .with_tag("name", "qwen2.5-7B"),
        ModelFingerprint::new("gemma")
            .with_param_count(2_000_000_000)
            .with_layers(18)
            .with_hidden_size(2048)
            .with_heads(8)
            .with_vocab_size(256000)
            .with_quant_type("bf16")
            .with_tag("name", "gemma-2-2B"),
        ModelFingerprint::new("mistral")
            .with_param_count(7_000_000_000)
            .with_layers(32)
            .with_hidden_size(4096)
            .with_heads(32)
            .with_vocab_size(32000)
            .with_quant_type("bf16")
            .with_tag("name", "mistral-7B-v0.3"),
        ModelFingerprint::new("smollm2")
            .with_param_count(1_700_000_000)
            .with_layers(24)
            .with_hidden_size(2048)
            .with_heads(32)
            .with_vocab_size(49152)
            .with_quant_type("bf16")
            .with_tag("name", "SmolLM2-1.7B"),
    ]
}

/// Attempt to match a fingerprint against known models.
pub fn identify_model(fp: &ModelFingerprint) -> Option<&'static str> {
    let known = [
        ("bitnet-b1.58-2B-4T", "bitnet", 30u32, 2560u32),
        ("phi-4", "phi", 40, 5120),
        ("llama-3.1-8B", "llama", 32, 4096),
        ("qwen2.5-7B", "qwen2", 28, 3584),
        ("gemma-2-2B", "gemma", 18, 2048),
        ("mistral-7B-v0.3", "mistral", 32, 4096),
        ("SmolLM2-1.7B", "smollm2", 24, 2048),
    ];
    for (name, arch, layers, hidden) in known {
        if fp.architecture == arch
            && fp.num_layers == layers
            && fp.hidden_size == hidden
        {
            return Some(name);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_fingerprint() {
        let fp = ModelFingerprint::new("llama");
        assert_eq!(fp.architecture, "llama");
        assert_eq!(fp.param_count, 0);
    }

    #[test]
    fn test_builder_chain() {
        let fp = ModelFingerprint::new("phi")
            .with_param_count(14_000_000_000)
            .with_layers(40)
            .with_hidden_size(5120)
            .with_heads(40)
            .with_vocab_size(100352)
            .with_quant_type("bf16");
        assert_eq!(fp.num_layers, 40);
        assert_eq!(fp.hidden_size, 5120);
        assert_eq!(fp.quant_type, "bf16");
    }

    #[test]
    fn test_compact_id() {
        let fp = ModelFingerprint::new("phi")
            .with_layers(40)
            .with_hidden_size(5120)
            .with_heads(40)
            .with_vocab_size(100352)
            .with_quant_type("bf16");
        assert_eq!(fp.compact_id(), "phi-L40-H5120-A40-V100352-bf16");
    }

    #[test]
    fn test_estimated_weight_bytes_f16() {
        let fp = ModelFingerprint::new("test")
            .with_param_count(1_000_000_000)
            .with_quant_type("f16");
        assert_eq!(fp.estimated_weight_bytes(), 2_000_000_000);
    }

    #[test]
    fn test_estimated_weight_bytes_i2s() {
        let fp = ModelFingerprint::new("test")
            .with_param_count(2_000_000_000)
            .with_quant_type("i2s");
        assert_eq!(fp.estimated_weight_bytes(), 500_000_000);
    }

    #[test]
    fn test_estimated_weight_bytes_q4() {
        let fp = ModelFingerprint::new("test")
            .with_param_count(7_000_000_000)
            .with_quant_type("q4_0");
        assert_eq!(fp.estimated_weight_bytes(), 3_500_000_000);
    }

    #[test]
    fn test_is_quantized() {
        assert!(ModelFingerprint::new("t").with_quant_type("i2s").is_quantized());
        assert!(ModelFingerprint::new("t").with_quant_type("q4_0").is_quantized());
        assert!(!ModelFingerprint::new("t").with_quant_type("f16").is_quantized());
        assert!(!ModelFingerprint::new("t").with_quant_type("bf16").is_quantized());
    }

    #[test]
    fn test_same_architecture() {
        let a = ModelFingerprint::new("llama")
            .with_layers(32)
            .with_hidden_size(4096)
            .with_heads(32);
        let b = ModelFingerprint::new("llama")
            .with_layers(32)
            .with_hidden_size(4096)
            .with_heads(32)
            .with_quant_type("q4_0");
        assert!(a.same_architecture(&b));
    }

    #[test]
    fn test_different_architecture() {
        let a = ModelFingerprint::new("llama")
            .with_layers(32)
            .with_hidden_size(4096)
            .with_heads(32);
        let b = ModelFingerprint::new("phi")
            .with_layers(40)
            .with_hidden_size(5120)
            .with_heads(40);
        assert!(!a.same_architecture(&b));
    }

    #[test]
    fn test_same_model_different_quant() {
        let a = ModelFingerprint::new("llama")
            .with_layers(32)
            .with_hidden_size(4096)
            .with_heads(32)
            .with_vocab_size(32000)
            .with_quant_type("f16");
        let b = ModelFingerprint::new("llama")
            .with_layers(32)
            .with_hidden_size(4096)
            .with_heads(32)
            .with_vocab_size(32000)
            .with_quant_type("q4_0");
        assert!(a.same_model_different_quant(&b));
    }

    #[test]
    fn test_not_same_model_different_quant_same_quant() {
        let a = ModelFingerprint::new("llama")
            .with_layers(32)
            .with_hidden_size(4096)
            .with_heads(32)
            .with_vocab_size(32000)
            .with_quant_type("f16");
        let b = a.clone();
        assert!(!a.same_model_different_quant(&b));
    }

    #[test]
    fn test_size_label_billions() {
        let fp = ModelFingerprint::new("t").with_param_count(14_000_000_000);
        assert_eq!(fp.size_label(), "14.0B");
    }

    #[test]
    fn test_size_label_small() {
        let fp = ModelFingerprint::new("t").with_param_count(125_000_000);
        assert_eq!(fp.size_label(), "125M");
    }

    #[test]
    fn test_display() {
        let fp = ModelFingerprint::new("phi")
            .with_param_count(14_000_000_000)
            .with_layers(40)
            .with_hidden_size(5120)
            .with_heads(40)
            .with_vocab_size(100352)
            .with_quant_type("bf16");
        let s = format!("{fp}");
        assert!(s.contains("phi"));
        assert!(s.contains("14.0B"));
        assert!(s.contains("40 layers"));
    }

    #[test]
    fn test_with_tag() {
        let fp = ModelFingerprint::new("test")
            .with_tag("name", "my-model")
            .with_tag("source", "hf");
        assert_eq!(fp.tags["name"], "my-model");
        assert_eq!(fp.tags["source"], "hf");
    }

    #[test]
    fn test_known_fingerprints() {
        let known = known_fingerprints();
        assert!(known.len() >= 7);
        assert_eq!(known[0].architecture, "bitnet");
        assert_eq!(known[1].architecture, "phi");
    }

    #[test]
    fn test_identify_model_phi4() {
        let fp = ModelFingerprint::new("phi")
            .with_layers(40)
            .with_hidden_size(5120);
        assert_eq!(identify_model(&fp), Some("phi-4"));
    }

    #[test]
    fn test_identify_model_llama() {
        let fp = ModelFingerprint::new("llama")
            .with_layers(32)
            .with_hidden_size(4096);
        assert_eq!(identify_model(&fp), Some("llama-3.1-8B"));
    }

    #[test]
    fn test_identify_model_unknown() {
        let fp = ModelFingerprint::new("unknown")
            .with_layers(99)
            .with_hidden_size(9999);
        assert_eq!(identify_model(&fp), None);
    }

    #[test]
    fn test_eq_and_hash() {
        let a = ModelFingerprint::new("phi").with_layers(40);
        let b = ModelFingerprint::new("phi").with_layers(40);
        assert_eq!(a, b);
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
    }

    #[test]
    fn test_estimated_weight_bytes_f32() {
        let fp = ModelFingerprint::new("test")
            .with_param_count(1_000_000_000)
            .with_quant_type("f32");
        assert_eq!(fp.estimated_weight_bytes(), 4_000_000_000);
    }

    #[test]
    fn test_size_label_zero() {
        let fp = ModelFingerprint::new("t");
        assert_eq!(fp.size_label(), "0M");
    }
}
