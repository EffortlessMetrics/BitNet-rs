//! Model catalog — registry of known models with metadata.
//!
//! Provides a searchable catalog of supported SLM models.

/// Model size category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    Tiny,   // < 500M
    Small,  // 500M - 3B
    Medium, // 3B - 10B
    Large,  // 10B - 30B
    XLarge, // > 30B
}

impl ModelSize {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
            Self::XLarge => "xlarge",
        }
    }

    pub fn from_params(params_b: f64) -> Self {
        if params_b < 0.5 {
            Self::Tiny
        } else if params_b < 3.0 {
            Self::Small
        } else if params_b < 10.0 {
            Self::Medium
        } else if params_b < 30.0 {
            Self::Large
        } else {
            Self::XLarge
        }
    }
}

/// Catalog entry for a known model.
#[derive(Debug, Clone)]
pub struct CatalogEntry {
    pub id: String,
    pub name: String,
    pub family: String,
    pub params_b: f64,
    pub size: ModelSize,
    pub architecture: String,
    pub context_length: usize,
    pub vocab_size: usize,
    pub license: String,
    pub hf_repo: String,
}

/// Model catalog.
#[derive(Debug, Clone, Default)]
pub struct ModelCatalog {
    entries: Vec<CatalogEntry>,
}

impl ModelCatalog {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Build catalog with known SLM models.
    pub fn builtin() -> Self {
        let mut cat = Self::new();

        cat.add(CatalogEntry {
            id: "bitnet-2b".into(),
            name: "BitNet b1.58 2B".into(),
            family: "bitnet".into(),
            params_b: 2.0,
            size: ModelSize::Small,
            architecture: "BitnetForCausalLM".into(),
            context_length: 4096,
            vocab_size: 32000,
            license: "MIT".into(),
            hf_repo: "microsoft/bitnet-b1.58-2B-4T-gguf".into(),
        });

        cat.add(CatalogEntry {
            id: "phi-4".into(),
            name: "Phi-4 14B".into(),
            family: "phi".into(),
            params_b: 14.0,
            size: ModelSize::Large,
            architecture: "PhiForCausalLM".into(),
            context_length: 16384,
            vocab_size: 100352,
            license: "MIT".into(),
            hf_repo: "microsoft/phi-4".into(),
        });

        cat.add(CatalogEntry {
            id: "phi-4-mini".into(),
            name: "Phi-4 Mini 3.8B".into(),
            family: "phi".into(),
            params_b: 3.8,
            size: ModelSize::Medium,
            architecture: "PhiForCausalLM".into(),
            context_length: 16384,
            vocab_size: 100352,
            license: "MIT".into(),
            hf_repo: "microsoft/phi-4-mini".into(),
        });

        cat.add(CatalogEntry {
            id: "qwen2.5-3b".into(),
            name: "Qwen 2.5 3B".into(),
            family: "qwen".into(),
            params_b: 3.0,
            size: ModelSize::Medium,
            architecture: "Qwen2ForCausalLM".into(),
            context_length: 32768,
            vocab_size: 151936,
            license: "Apache-2.0".into(),
            hf_repo: "Qwen/Qwen2.5-3B-Instruct".into(),
        });

        cat.add(CatalogEntry {
            id: "llama3-8b".into(),
            name: "LLaMA 3 8B".into(),
            family: "llama".into(),
            params_b: 8.0,
            size: ModelSize::Medium,
            architecture: "LlamaForCausalLM".into(),
            context_length: 8192,
            vocab_size: 128256,
            license: "Llama3".into(),
            hf_repo: "meta-llama/Meta-Llama-3-8B-Instruct".into(),
        });

        cat.add(CatalogEntry {
            id: "gemma2-2b".into(),
            name: "Gemma 2 2B".into(),
            family: "gemma".into(),
            params_b: 2.0,
            size: ModelSize::Small,
            architecture: "Gemma2ForCausalLM".into(),
            context_length: 8192,
            vocab_size: 256128,
            license: "Gemma".into(),
            hf_repo: "google/gemma-2-2b-it".into(),
        });

        cat.add(CatalogEntry {
            id: "mistral-7b".into(),
            name: "Mistral 7B".into(),
            family: "mistral".into(),
            params_b: 7.0,
            size: ModelSize::Medium,
            architecture: "MistralForCausalLM".into(),
            context_length: 32768,
            vocab_size: 32000,
            license: "Apache-2.0".into(),
            hf_repo: "mistralai/Mistral-7B-Instruct-v0.3".into(),
        });

        cat.add(CatalogEntry {
            id: "smollm2-1.7b".into(),
            name: "SmolLM2 1.7B".into(),
            family: "smollm".into(),
            params_b: 1.7,
            size: ModelSize::Small,
            architecture: "LlamaForCausalLM".into(),
            context_length: 8192,
            vocab_size: 49152,
            license: "Apache-2.0".into(),
            hf_repo: "HuggingFaceTB/SmolLM2-1.7B-Instruct".into(),
        });

        cat
    }

    pub fn add(&mut self, entry: CatalogEntry) {
        self.entries.push(entry);
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }

    pub fn all(&self) -> &[CatalogEntry] {
        &self.entries
    }

    pub fn get(&self, id: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn by_family(&self, family: &str) -> Vec<&CatalogEntry> {
        self.entries.iter().filter(|e| e.family == family).collect()
    }

    pub fn by_size(&self, size: ModelSize) -> Vec<&CatalogEntry> {
        self.entries.iter().filter(|e| e.size == size).collect()
    }

    pub fn search(&self, query: &str) -> Vec<&CatalogEntry> {
        let q = query.to_lowercase();
        self.entries
            .iter()
            .filter(|e| {
                e.name.to_lowercase().contains(&q)
                    || e.id.contains(&q)
                    || e.family.contains(&q)
            })
            .collect()
    }

    pub fn families(&self) -> Vec<String> {
        let mut fams: Vec<_> =
            self.entries.iter().map(|e| e.family.clone()).collect();
        fams.sort();
        fams.dedup();
        fams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_count() {
        let cat = ModelCatalog::builtin();
        assert!(cat.count() >= 8);
    }

    #[test]
    fn test_get_by_id() {
        let cat = ModelCatalog::builtin();
        let phi = cat.get("phi-4").unwrap();
        assert_eq!(phi.params_b, 14.0);
    }

    #[test]
    fn test_get_missing() {
        let cat = ModelCatalog::builtin();
        assert!(cat.get("nonexistent").is_none());
    }

    #[test]
    fn test_by_family() {
        let cat = ModelCatalog::builtin();
        let phi = cat.by_family("phi");
        assert!(phi.len() >= 2);
    }

    #[test]
    fn test_by_size() {
        let cat = ModelCatalog::builtin();
        let small = cat.by_size(ModelSize::Small);
        assert!(small.len() >= 2);
    }

    #[test]
    fn test_search() {
        let cat = ModelCatalog::builtin();
        let results = cat.search("llama");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_families() {
        let cat = ModelCatalog::builtin();
        let fams = cat.families();
        assert!(fams.contains(&"phi".to_string()));
        assert!(fams.contains(&"llama".to_string()));
    }

    #[test]
    fn test_size_from_params() {
        assert_eq!(ModelSize::from_params(0.3), ModelSize::Tiny);
        assert_eq!(ModelSize::from_params(2.0), ModelSize::Small);
        assert_eq!(ModelSize::from_params(7.0), ModelSize::Medium);
        assert_eq!(ModelSize::from_params(14.0), ModelSize::Large);
        assert_eq!(ModelSize::from_params(70.0), ModelSize::XLarge);
    }

    #[test]
    fn test_size_str() {
        assert_eq!(ModelSize::Tiny.as_str(), "tiny");
        assert_eq!(ModelSize::Large.as_str(), "large");
    }

    #[test]
    fn test_bitnet_in_catalog() {
        let cat = ModelCatalog::builtin();
        let bitnet = cat.get("bitnet-2b").unwrap();
        assert_eq!(bitnet.vocab_size, 32000);
    }

    #[test]
    fn test_add_custom() {
        let mut cat = ModelCatalog::new();
        cat.add(CatalogEntry {
            id: "custom".into(),
            name: "Custom".into(),
            family: "test".into(),
            params_b: 1.0,
            size: ModelSize::Small,
            architecture: "Test".into(),
            context_length: 2048,
            vocab_size: 1000,
            license: "MIT".into(),
            hf_repo: "test/test".into(),
        });
        assert_eq!(cat.count(), 1);
    }

    #[test]
    fn test_search_case_insensitive() {
        let cat = ModelCatalog::builtin();
        let results = cat.search("PHI");
        assert!(!results.is_empty());
    }
}
