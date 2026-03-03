//! Query interface for the model registry.
//!
//! Filter and search models by architecture, size, quantization, etc.

/// Model registry entry.
#[derive(Debug, Clone)]
pub struct RegistryEntry {
    pub id: String,
    pub name: String,
    pub architecture: String,
    pub param_count: u64,
    pub quant_type: String,
    pub format: String,
    pub tags: Vec<String>,
}

impl RegistryEntry {
    pub fn size_label(&self) -> String {
        let b = self.param_count as f64 / 1e9;
        if b >= 1.0 { format!("{b:.1}B") } else { format!("{:.0}M", self.param_count as f64 / 1e6) }
    }

    pub fn matches_query(&self, query: &str) -> bool {
        let q = query.to_lowercase();
        self.id.to_lowercase().contains(&q)
            || self.name.to_lowercase().contains(&q)
            || self.architecture.to_lowercase().contains(&q)
            || self.tags.iter().any(|t| t.to_lowercase().contains(&q))
    }
}

/// Filter criteria for registry queries.
#[derive(Debug, Clone, Default)]
pub struct RegistryFilter {
    pub architecture: Option<String>,
    pub max_params: Option<u64>,
    pub min_params: Option<u64>,
    pub quant_type: Option<String>,
    pub format: Option<String>,
    pub tag: Option<String>,
}

impl RegistryFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_architecture(mut self, arch: &str) -> Self {
        self.architecture = Some(arch.to_string());
        self
    }

    pub fn with_max_params(mut self, max: u64) -> Self {
        self.max_params = Some(max);
        self
    }

    pub fn with_min_params(mut self, min: u64) -> Self {
        self.min_params = Some(min);
        self
    }

    pub fn with_quant(mut self, qt: &str) -> Self {
        self.quant_type = Some(qt.to_string());
        self
    }

    pub fn with_format(mut self, fmt: &str) -> Self {
        self.format = Some(fmt.to_string());
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tag = Some(tag.to_string());
        self
    }

    pub fn matches(&self, entry: &RegistryEntry) -> bool {
        if let Some(ref arch) = self.architecture {
            if !entry.architecture.eq_ignore_ascii_case(arch) {
                return false;
            }
        }
        if let Some(max) = self.max_params {
            if entry.param_count > max {
                return false;
            }
        }
        if let Some(min) = self.min_params {
            if entry.param_count < min {
                return false;
            }
        }
        if let Some(ref qt) = self.quant_type {
            if !entry.quant_type.eq_ignore_ascii_case(qt) {
                return false;
            }
        }
        if let Some(ref fmt) = self.format {
            if !entry.format.eq_ignore_ascii_case(fmt) {
                return false;
            }
        }
        if let Some(ref tag) = self.tag {
            if !entry.tags.iter().any(|t| t.eq_ignore_ascii_case(tag)) {
                return false;
            }
        }
        true
    }
}

/// Query a registry.
pub fn query<'a>(entries: &'a [RegistryEntry], filter: &RegistryFilter) -> Vec<&'a RegistryEntry> {
    entries.iter().filter(|e| filter.matches(e)).collect()
}

/// Search by text query.
pub fn search<'a>(entries: &'a [RegistryEntry], query_text: &str) -> Vec<&'a RegistryEntry> {
    entries.iter().filter(|e| e.matches_query(query_text)).collect()
}

/// Built-in registry entries.
pub fn builtin_registry() -> Vec<RegistryEntry> {
    vec![
        RegistryEntry {
            id: "bitnet-2b".into(),
            name: "BitNet b1.58 2B".into(),
            architecture: "bitnet".into(),
            param_count: 2_000_000_000,
            quant_type: "i2s".into(),
            format: "gguf".into(),
            tags: vec!["bitnet".into(), "1-bit".into()],
        },
        RegistryEntry {
            id: "phi-4".into(),
            name: "Phi-4 14B".into(),
            architecture: "phi".into(),
            param_count: 14_000_000_000,
            quant_type: "bf16".into(),
            format: "safetensors".into(),
            tags: vec!["slm".into(), "microsoft".into()],
        },
        RegistryEntry {
            id: "llama3-8b".into(),
            name: "LLaMA 3.1 8B".into(),
            architecture: "llama".into(),
            param_count: 8_000_000_000,
            quant_type: "bf16".into(),
            format: "safetensors".into(),
            tags: vec!["slm".into(), "meta".into()],
        },
        RegistryEntry {
            id: "qwen25-7b".into(),
            name: "Qwen 2.5 7B".into(),
            architecture: "qwen2".into(),
            param_count: 7_000_000_000,
            quant_type: "bf16".into(),
            format: "safetensors".into(),
            tags: vec!["slm".into(), "alibaba".into()],
        },
        RegistryEntry {
            id: "smollm2-1.7b".into(),
            name: "SmolLM2 1.7B".into(),
            architecture: "llama".into(),
            param_count: 1_700_000_000,
            quant_type: "bf16".into(),
            format: "safetensors".into(),
            tags: vec!["slm".into(), "small".into()],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> Vec<RegistryEntry> {
        builtin_registry()
    }

    #[test]
    fn test_builtin_count() {
        assert!(builtin_registry().len() >= 5);
    }

    #[test]
    fn test_size_label() {
        let e = &registry()[1]; // phi-4
        assert_eq!(e.size_label(), "14.0B");
    }

    #[test]
    fn test_size_label_small() {
        let mut e = registry()[0].clone();
        e.param_count = 125_000_000;
        assert_eq!(e.size_label(), "125M");
    }

    #[test]
    fn test_matches_query() {
        let e = &registry()[0];
        assert!(e.matches_query("bitnet"));
        assert!(e.matches_query("1-bit"));
        assert!(!e.matches_query("phi"));
    }

    #[test]
    fn test_filter_arch() {
        let r = registry();
        let f = RegistryFilter::new().with_architecture("phi");
        let results = query(&r, &f);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "phi-4");
    }

    #[test]
    fn test_filter_max_params() {
        let r = registry();
        let f = RegistryFilter::new().with_max_params(3_000_000_000);
        let results = query(&r, &f);
        assert!(results.len() >= 2); // bitnet + smollm2
    }

    #[test]
    fn test_filter_quant() {
        let r = registry();
        let f = RegistryFilter::new().with_quant("i2s");
        let results = query(&r, &f);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_filter_format() {
        let r = registry();
        let f = RegistryFilter::new().with_format("safetensors");
        let results = query(&r, &f);
        assert!(results.len() >= 4);
    }

    #[test]
    fn test_filter_tag() {
        let r = registry();
        let f = RegistryFilter::new().with_tag("slm");
        let results = query(&r, &f);
        assert!(results.len() >= 4);
    }

    #[test]
    fn test_filter_combined() {
        let r = registry();
        let f = RegistryFilter::new().with_architecture("llama").with_max_params(5_000_000_000);
        let results = query(&r, &f);
        assert_eq!(results.len(), 1); // smollm2
    }

    #[test]
    fn test_search() {
        let r = registry();
        let results = search(&r, "phi");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_empty() {
        let r = registry();
        let results = search(&r, "nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn test_filter_no_match() {
        let r = registry();
        let f = RegistryFilter::new().with_architecture("gpt4");
        assert!(query(&r, &f).is_empty());
    }

    #[test]
    fn test_filter_min_params() {
        let r = registry();
        let f = RegistryFilter::new().with_min_params(10_000_000_000);
        let results = query(&r, &f);
        assert_eq!(results.len(), 1); // phi-4
    }

    #[test]
    fn test_default_filter_matches_all() {
        let r = registry();
        let f = RegistryFilter::default();
        assert_eq!(query(&r, &f).len(), r.len());
    }
}
