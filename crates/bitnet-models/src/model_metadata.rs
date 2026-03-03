//! Model metadata extraction and representation.
//!
//! Extract and store key model properties: architecture, sizes,
//! quantization type, license, and provenance.

use std::collections::HashMap;

/// Model license type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum License {
    Mit,
    Apache2,
    Llama,
    Gemma,
    Custom(String),
    Unknown,
}

impl License {
    pub fn from_str_relaxed(s: &str) -> Self {
        let lower = s.to_lowercase();
        if lower.contains("mit") {
            Self::Mit
        } else if lower.contains("apache") {
            Self::Apache2
        } else if lower.contains("llama") {
            Self::Llama
        } else if lower.contains("gemma") {
            Self::Gemma
        } else if lower.is_empty() {
            Self::Unknown
        } else {
            Self::Custom(s.to_string())
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Mit => "MIT",
            Self::Apache2 => "Apache-2.0",
            Self::Llama => "Llama",
            Self::Gemma => "Gemma",
            Self::Custom(s) => s,
            Self::Unknown => "Unknown",
        }
    }

    pub fn is_open_source(&self) -> bool {
        matches!(self, Self::Mit | Self::Apache2)
    }
}

/// Quantization information.
#[derive(Debug, Clone)]
pub struct QuantInfo {
    pub method: String,
    pub bits: u32,
    pub group_size: Option<usize>,
}

impl QuantInfo {
    pub fn new(method: impl Into<String>, bits: u32) -> Self {
        Self { method: method.into(), bits, group_size: None }
    }

    pub fn with_group_size(mut self, size: usize) -> Self {
        self.group_size = Some(size);
        self
    }
}

/// Complete model metadata.
#[derive(Debug, Clone)]
pub struct ModelCard {
    pub name: String,
    pub architecture: String,
    pub param_count: Option<u64>,
    pub hidden_size: Option<usize>,
    pub num_layers: Option<usize>,
    pub num_heads: Option<usize>,
    pub num_kv_heads: Option<usize>,
    pub vocab_size: Option<usize>,
    pub context_length: Option<usize>,
    pub license: License,
    pub quant_info: Option<QuantInfo>,
    pub source_url: Option<String>,
    pub extra: HashMap<String, String>,
}

impl ModelCard {
    pub fn new(name: impl Into<String>, architecture: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            architecture: architecture.into(),
            param_count: None,
            hidden_size: None,
            num_layers: None,
            num_heads: None,
            num_kv_heads: None,
            vocab_size: None,
            context_length: None,
            license: License::Unknown,
            quant_info: None,
            source_url: None,
            extra: HashMap::new(),
        }
    }

    pub fn with_params(mut self, count: u64) -> Self {
        self.param_count = Some(count);
        self
    }
    pub fn with_hidden_size(mut self, size: usize) -> Self {
        self.hidden_size = Some(size);
        self
    }
    pub fn with_layers(mut self, n: usize) -> Self {
        self.num_layers = Some(n);
        self
    }
    pub fn with_heads(mut self, n: usize) -> Self {
        self.num_heads = Some(n);
        self
    }
    pub fn with_kv_heads(mut self, n: usize) -> Self {
        self.num_kv_heads = Some(n);
        self
    }
    pub fn with_vocab(mut self, n: usize) -> Self {
        self.vocab_size = Some(n);
        self
    }
    pub fn with_context_length(mut self, n: usize) -> Self {
        self.context_length = Some(n);
        self
    }
    pub fn with_license(mut self, license: License) -> Self {
        self.license = license;
        self
    }
    pub fn with_quant(mut self, info: QuantInfo) -> Self {
        self.quant_info = Some(info);
        self
    }
    pub fn with_source(mut self, url: impl Into<String>) -> Self {
        self.source_url = Some(url.into());
        self
    }

    pub fn set_extra(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.extra.insert(key.into(), value.into());
    }

    pub fn is_quantized(&self) -> bool {
        self.quant_info.is_some()
    }

    pub fn gqa_ratio(&self) -> Option<usize> {
        match (self.num_heads, self.num_kv_heads) {
            (Some(h), Some(kv)) if kv > 0 => Some(h / kv),
            _ => None,
        }
    }

    pub fn head_dim(&self) -> Option<usize> {
        match (self.hidden_size, self.num_heads) {
            (Some(h), Some(n)) if n > 0 => Some(h / n),
            _ => None,
        }
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let params = self
            .param_count
            .map(|p| {
                if p >= 1_000_000_000 {
                    format!("{:.1}B", p as f64 / 1e9)
                } else if p >= 1_000_000 {
                    format!("{:.1}M", p as f64 / 1e6)
                } else {
                    format!("{p}")
                }
            })
            .unwrap_or_else(|| "?".into());

        format!("{} ({}, {} params)", self.name, self.architecture, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_metadata() {
        let md = ModelCard::new("phi-4", "PhiForCausalLM")
            .with_params(14_000_000_000)
            .with_layers(40)
            .with_heads(40)
            .with_kv_heads(10);
        assert_eq!(md.gqa_ratio(), Some(4));
    }

    #[test]
    fn test_head_dim() {
        let md = ModelCard::new("m", "a").with_hidden_size(5120).with_heads(40);
        assert_eq!(md.head_dim(), Some(128));
    }

    #[test]
    fn test_summary() {
        let md = ModelCard::new("phi-4", "Phi").with_params(14_000_000_000);
        let s = md.summary();
        assert!(s.contains("14.0B"));
        assert!(s.contains("phi-4"));
    }

    #[test]
    fn test_summary_millions() {
        let md = ModelCard::new("small", "A").with_params(350_000_000);
        assert!(md.summary().contains("350.0M"));
    }

    #[test]
    fn test_license_parsing() {
        assert_eq!(License::from_str_relaxed("MIT License"), License::Mit);
        assert_eq!(License::from_str_relaxed("Apache-2.0"), License::Apache2);
        assert!(License::from_str_relaxed("MIT License").is_open_source());
    }

    #[test]
    fn test_license_custom() {
        let lic = License::from_str_relaxed("proprietary-v2");
        assert_eq!(lic, License::Custom("proprietary-v2".into()));
        assert!(!lic.is_open_source());
    }

    #[test]
    fn test_quant_info() {
        let qi = QuantInfo::new("GPTQ", 4).with_group_size(128);
        assert_eq!(qi.bits, 4);
        assert_eq!(qi.group_size, Some(128));
    }

    #[test]
    fn test_is_quantized() {
        let md = ModelCard::new("m", "a");
        assert!(!md.is_quantized());
        let md2 = md.with_quant(QuantInfo::new("AWQ", 4));
        assert!(md2.is_quantized());
    }

    #[test]
    fn test_extra() {
        let mut md = ModelCard::new("m", "a");
        md.set_extra("creator", "microsoft");
        assert_eq!(md.extra.get("creator").unwrap(), "microsoft");
    }

    #[test]
    fn test_gqa_no_kv() {
        let md = ModelCard::new("m", "a").with_heads(32);
        assert!(md.gqa_ratio().is_none());
    }

    #[test]
    fn test_builder_chain() {
        let md = ModelCard::new("model", "arch")
            .with_params(1_000_000)
            .with_hidden_size(768)
            .with_layers(12)
            .with_heads(12)
            .with_kv_heads(4)
            .with_vocab(32000)
            .with_context_length(4096)
            .with_license(License::Apache2)
            .with_source("https://example.com");
        assert_eq!(md.context_length, Some(4096));
        assert_eq!(md.license, License::Apache2);
    }

    #[test]
    fn test_license_unknown() {
        let lic = License::from_str_relaxed("");
        assert_eq!(lic, License::Unknown);
        assert_eq!(lic.name(), "Unknown");
    }
}
