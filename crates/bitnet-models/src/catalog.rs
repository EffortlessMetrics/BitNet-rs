//! SLM model catalog for discovering and comparing small language models.
//!
//! Provides a built-in catalog of well-known SLM models with hardware
//! recommendations, task classifications, and metadata for model selection.

use std::fmt;

/// Supported inference tasks for a model.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelTask {
    TextGeneration,
    CodeGeneration,
    Chat,
    InstructionFollowing,
    Summarization,
    Translation,
    QuestionAnswering,
    FunctionCalling,
}

impl fmt::Display for ModelTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TextGeneration => write!(f, "Text Generation"),
            Self::CodeGeneration => write!(f, "Code Generation"),
            Self::Chat => write!(f, "Chat"),
            Self::InstructionFollowing => write!(f, "Instruction Following"),
            Self::Summarization => write!(f, "Summarization"),
            Self::Translation => write!(f, "Translation"),
            Self::QuestionAnswering => write!(f, "Question Answering"),
            Self::FunctionCalling => write!(f, "Function Calling"),
        }
    }
}

/// Hardware requirements and recommendations for running a model.
#[derive(Debug, Clone)]
pub struct HardwareRecommendation {
    pub min_ram_gb: f32,
    pub recommended_ram_gb: f32,
    pub min_vram_gb: Option<f32>,
    pub supports_cpu_inference: bool,
    pub supports_quantization: bool,
    pub recommended_quantization: Option<String>,
}

/// A single entry in the model catalog.
#[derive(Debug, Clone)]
pub struct ModelCatalogEntry {
    pub id: String,
    pub name: String,
    pub publisher: String,
    pub hf_repo: String,
    pub architecture: String,
    pub parameter_count: u64,
    pub default_dtype: String,
    pub license: String,
    pub release_date: String,
    pub description: String,
    pub tags: Vec<String>,
    pub supported_tasks: Vec<ModelTask>,
    pub recommended_hardware: HardwareRecommendation,
}

/// A catalog of known SLM models with metadata and hardware recommendations.
#[derive(Debug)]
pub struct ModelCatalog {
    entries: Vec<ModelCatalogEntry>,
}

impl ModelCatalog {
    /// Create a new catalog populated with built-in model entries.
    pub fn new() -> Self {
        Self { entries: builtin_entries() }
    }

    /// Look up a catalog entry by its unique ID.
    pub fn get(&self, id: &str) -> Option<&ModelCatalogEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Search entries by substring match on name, publisher, description, or tags.
    pub fn search(&self, query: &str) -> Vec<&ModelCatalogEntry> {
        let q = query.to_lowercase();
        self.entries
            .iter()
            .filter(|e| {
                e.name.to_lowercase().contains(&q)
                    || e.publisher.to_lowercase().contains(&q)
                    || e.description.to_lowercase().contains(&q)
                    || e.id.to_lowercase().contains(&q)
                    || e.tags.iter().any(|t| t.to_lowercase().contains(&q))
            })
            .collect()
    }

    /// Return all entries that support the given task.
    pub fn filter_by_task(&self, task: &ModelTask) -> Vec<&ModelCatalogEntry> {
        self.entries.iter().filter(|e| e.supported_tasks.contains(task)).collect()
    }

    /// Return entries whose parameter count is at most `max_params`.
    pub fn filter_by_max_params(&self, max_params: u64) -> Vec<&ModelCatalogEntry> {
        self.entries.iter().filter(|e| e.parameter_count <= max_params).collect()
    }

    /// Return a slice of all catalog entries.
    pub fn list_all(&self) -> &[ModelCatalogEntry] {
        &self.entries
    }
}

impl Default for ModelCatalog {
    fn default() -> Self {
        Self::new()
    }
}

/// Render a human-readable table of every model in the catalog.
pub fn format_catalog(catalog: &ModelCatalog) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<25} {:<12} {:<14} {:<10} {}\n",
        "ID", "Publisher", "Params", "License", "Tasks"
    ));
    out.push_str(&"-".repeat(85));
    out.push('\n');
    for e in catalog.list_all() {
        let params = format_params(e.parameter_count);
        let tasks: Vec<String> = e.supported_tasks.iter().map(|t| t.to_string()).collect();
        out.push_str(&format!(
            "{:<25} {:<12} {:<14} {:<10} {}\n",
            e.id,
            e.publisher,
            params,
            e.license,
            tasks.join(", ")
        ));
    }
    out
}

fn format_params(count: u64) -> String {
    let billions = count as f64 / 1_000_000_000.0;
    format!("{billions:.2}B")
}

// ---------------------------------------------------------------------------
// Built-in catalog entries
// ---------------------------------------------------------------------------

fn builtin_entries() -> Vec<ModelCatalogEntry> {
    vec![
        ModelCatalogEntry {
            id: "phi-4-14b".into(),
            name: "Phi-4".into(),
            publisher: "Microsoft".into(),
            hf_repo: "microsoft/phi-4".into(),
            architecture: "PhiForCausalLM".into(),
            parameter_count: 14_000_000_000,
            default_dtype: "bf16".into(),
            license: "MIT".into(),
            release_date: "2024-12-12".into(),
            description: "Microsoft Phi-4 14B reasoning model with \
                          strong math and code performance"
                .into(),
            tags: vec!["reasoning".into(), "math".into(), "code".into(), "slm".into()],
            supported_tasks: vec![
                ModelTask::TextGeneration,
                ModelTask::CodeGeneration,
                ModelTask::Chat,
                ModelTask::InstructionFollowing,
                ModelTask::QuestionAnswering,
            ],
            recommended_hardware: HardwareRecommendation {
                min_ram_gb: 16.0,
                recommended_ram_gb: 32.0,
                min_vram_gb: Some(12.0),
                supports_cpu_inference: true,
                supports_quantization: true,
                recommended_quantization: Some("Q4_K_M".into()),
            },
        },
        ModelCatalogEntry {
            id: "phi-4-mini-3.8b".into(),
            name: "Phi-4-mini".into(),
            publisher: "Microsoft".into(),
            hf_repo: "microsoft/Phi-4-mini-instruct".into(),
            architecture: "PhiForCausalLM".into(),
            parameter_count: 3_800_000_000,
            default_dtype: "bf16".into(),
            license: "MIT".into(),
            release_date: "2025-02-27".into(),
            description: "Compact Phi-4 variant optimized for \
                          instruction following and chat"
                .into(),
            tags: vec!["instruct".into(), "compact".into(), "slm".into()],
            supported_tasks: vec![
                ModelTask::TextGeneration,
                ModelTask::Chat,
                ModelTask::InstructionFollowing,
                ModelTask::QuestionAnswering,
            ],
            recommended_hardware: HardwareRecommendation {
                min_ram_gb: 8.0,
                recommended_ram_gb: 16.0,
                min_vram_gb: Some(4.0),
                supports_cpu_inference: true,
                supports_quantization: true,
                recommended_quantization: Some("Q4_K_M".into()),
            },
        },
        ModelCatalogEntry {
            id: "llama-3.2-1b".into(),
            name: "LLaMA-3.2-1B".into(),
            publisher: "Meta".into(),
            hf_repo: "meta-llama/Llama-3.2-1B".into(),
            architecture: "LlamaForCausalLM".into(),
            parameter_count: 1_240_000_000,
            default_dtype: "bf16".into(),
            license: "Llama-3.2".into(),
            release_date: "2024-09-25".into(),
            description: "Meta LLaMA 3.2 1B lightweight model for \
                          on-device and edge deployment"
                .into(),
            tags: vec!["edge".into(), "lightweight".into(), "slm".into()],
            supported_tasks: vec![
                ModelTask::TextGeneration,
                ModelTask::Summarization,
                ModelTask::QuestionAnswering,
            ],
            recommended_hardware: HardwareRecommendation {
                min_ram_gb: 4.0,
                recommended_ram_gb: 8.0,
                min_vram_gb: Some(2.0),
                supports_cpu_inference: true,
                supports_quantization: true,
                recommended_quantization: Some("Q4_K_M".into()),
            },
        },
        ModelCatalogEntry {
            id: "llama-3.2-3b".into(),
            name: "LLaMA-3.2-3B".into(),
            publisher: "Meta".into(),
            hf_repo: "meta-llama/Llama-3.2-3B".into(),
            architecture: "LlamaForCausalLM".into(),
            parameter_count: 3_210_000_000,
            default_dtype: "bf16".into(),
            license: "Llama-3.2".into(),
            release_date: "2024-09-25".into(),
            description: "Meta LLaMA 3.2 3B balanced model for \
                          general-purpose text tasks"
                .into(),
            tags: vec!["general-purpose".into(), "multilingual".into(), "slm".into()],
            supported_tasks: vec![
                ModelTask::TextGeneration,
                ModelTask::Chat,
                ModelTask::Summarization,
                ModelTask::Translation,
                ModelTask::QuestionAnswering,
            ],
            recommended_hardware: HardwareRecommendation {
                min_ram_gb: 8.0,
                recommended_ram_gb: 16.0,
                min_vram_gb: Some(4.0),
                supports_cpu_inference: true,
                supports_quantization: true,
                recommended_quantization: Some("Q4_K_M".into()),
            },
        },
        ModelCatalogEntry {
            id: "qwen2.5-7b".into(),
            name: "Qwen2.5-7B".into(),
            publisher: "Alibaba".into(),
            hf_repo: "Qwen/Qwen2.5-7B".into(),
            architecture: "Qwen2ForCausalLM".into(),
            parameter_count: 7_620_000_000,
            default_dtype: "bf16".into(),
            license: "Apache-2.0".into(),
            release_date: "2024-09-19".into(),
            description: "Alibaba Qwen2.5 7B with strong multilingual \
                          and coding capabilities"
                .into(),
            tags: vec!["multilingual".into(), "code".into(), "slm".into()],
            supported_tasks: vec![
                ModelTask::TextGeneration,
                ModelTask::CodeGeneration,
                ModelTask::Chat,
                ModelTask::InstructionFollowing,
                ModelTask::Translation,
                ModelTask::FunctionCalling,
            ],
            recommended_hardware: HardwareRecommendation {
                min_ram_gb: 16.0,
                recommended_ram_gb: 32.0,
                min_vram_gb: Some(8.0),
                supports_cpu_inference: true,
                supports_quantization: true,
                recommended_quantization: Some("Q4_K_M".into()),
            },
        },
        ModelCatalogEntry {
            id: "qwen2.5-1.5b".into(),
            name: "Qwen2.5-1.5B".into(),
            publisher: "Alibaba".into(),
            hf_repo: "Qwen/Qwen2.5-1.5B".into(),
            architecture: "Qwen2ForCausalLM".into(),
            parameter_count: 1_540_000_000,
            default_dtype: "bf16".into(),
            license: "Apache-2.0".into(),
            release_date: "2024-09-19".into(),
            description: "Alibaba Qwen2.5 1.5B compact model suitable \
                          for edge and mobile deployment"
                .into(),
            tags: vec!["edge".into(), "compact".into(), "multilingual".into(), "slm".into()],
            supported_tasks: vec![
                ModelTask::TextGeneration,
                ModelTask::Chat,
                ModelTask::Translation,
                ModelTask::QuestionAnswering,
            ],
            recommended_hardware: HardwareRecommendation {
                min_ram_gb: 4.0,
                recommended_ram_gb: 8.0,
                min_vram_gb: Some(2.0),
                supports_cpu_inference: true,
                supports_quantization: true,
                recommended_quantization: Some("Q4_K_S".into()),
            },
        },
        ModelCatalogEntry {
            id: "gemma-2-2b".into(),
            name: "Gemma-2-2B".into(),
            publisher: "Google".into(),
            hf_repo: "google/gemma-2-2b".into(),
            architecture: "Gemma2ForCausalLM".into(),
            parameter_count: 2_610_000_000,
            default_dtype: "bf16".into(),
            license: "Gemma".into(),
            release_date: "2024-06-27".into(),
            description: "Google Gemma 2 2B lightweight model with \
                          strong benchmark performance for its size"
                .into(),
            tags: vec!["lightweight".into(), "efficient".into(), "slm".into()],
            supported_tasks: vec![
                ModelTask::TextGeneration,
                ModelTask::Chat,
                ModelTask::QuestionAnswering,
            ],
            recommended_hardware: HardwareRecommendation {
                min_ram_gb: 8.0,
                recommended_ram_gb: 16.0,
                min_vram_gb: Some(4.0),
                supports_cpu_inference: true,
                supports_quantization: true,
                recommended_quantization: Some("Q4_K_M".into()),
            },
        },
        ModelCatalogEntry {
            id: "mistral-7b-v0.3".into(),
            name: "Mistral-7B-v0.3".into(),
            publisher: "Mistral AI".into(),
            hf_repo: "mistralai/Mistral-7B-v0.3".into(),
            architecture: "MistralForCausalLM".into(),
            parameter_count: 7_250_000_000,
            default_dtype: "bf16".into(),
            license: "Apache-2.0".into(),
            release_date: "2024-05-22".into(),
            description: "Mistral 7B v0.3 with extended vocabulary and \
                          function-calling support"
                .into(),
            tags: vec!["general-purpose".into(), "function-calling".into(), "slm".into()],
            supported_tasks: vec![
                ModelTask::TextGeneration,
                ModelTask::CodeGeneration,
                ModelTask::Chat,
                ModelTask::InstructionFollowing,
                ModelTask::FunctionCalling,
            ],
            recommended_hardware: HardwareRecommendation {
                min_ram_gb: 16.0,
                recommended_ram_gb: 32.0,
                min_vram_gb: Some(8.0),
                supports_cpu_inference: true,
                supports_quantization: true,
                recommended_quantization: Some("Q4_K_M".into()),
            },
        },
        ModelCatalogEntry {
            id: "smollm2-1.7b".into(),
            name: "SmolLM2-1.7B".into(),
            publisher: "HuggingFace".into(),
            hf_repo: "HuggingFaceTB/SmolLM2-1.7B".into(),
            architecture: "LlamaForCausalLM".into(),
            parameter_count: 1_710_000_000,
            default_dtype: "bf16".into(),
            license: "Apache-2.0".into(),
            release_date: "2024-11-01".into(),
            description: "HuggingFace SmolLM2 1.7B efficient model \
                          designed for on-device inference"
                .into(),
            tags: vec!["edge".into(), "on-device".into(), "efficient".into(), "slm".into()],
            supported_tasks: vec![
                ModelTask::TextGeneration,
                ModelTask::Chat,
                ModelTask::Summarization,
            ],
            recommended_hardware: HardwareRecommendation {
                min_ram_gb: 4.0,
                recommended_ram_gb: 8.0,
                min_vram_gb: Some(2.0),
                supports_cpu_inference: true,
                supports_quantization: true,
                recommended_quantization: Some("Q4_K_S".into()),
            },
        },
        ModelCatalogEntry {
            id: "bitnet-b1.58-2b".into(),
            name: "BitNet-b1.58-2B".into(),
            publisher: "Microsoft".into(),
            hf_repo: "microsoft/bitnet-b1.58-2B-4T-gguf".into(),
            architecture: "BitNetForCausalLM".into(),
            parameter_count: 2_400_000_000,
            default_dtype: "i2_s".into(),
            license: "MIT".into(),
            release_date: "2025-02-18".into(),
            description: "Microsoft BitNet b1.58 2B ternary-weight \
                          model with extreme efficiency"
                .into(),
            tags: vec![
                "1-bit".into(),
                "ternary".into(),
                "efficient".into(),
                "bitnet".into(),
                "slm".into(),
            ],
            supported_tasks: vec![
                ModelTask::TextGeneration,
                ModelTask::Chat,
                ModelTask::QuestionAnswering,
            ],
            recommended_hardware: HardwareRecommendation {
                min_ram_gb: 4.0,
                recommended_ram_gb: 8.0,
                min_vram_gb: None,
                supports_cpu_inference: true,
                supports_quantization: false,
                recommended_quantization: None,
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_catalog_has_at_least_10_entries() {
        let catalog = ModelCatalog::new();
        assert!(
            catalog.list_all().len() >= 10,
            "catalog should contain at least 10 built-in entries"
        );
    }

    #[test]
    fn test_get_existing_entry() {
        let catalog = ModelCatalog::new();
        let entry = catalog.get("phi-4-14b").expect("phi-4-14b should exist");
        assert_eq!(entry.name, "Phi-4");
        assert_eq!(entry.publisher, "Microsoft");
    }

    #[test]
    fn test_get_missing_entry() {
        let catalog = ModelCatalog::new();
        assert!(catalog.get("nonexistent-model").is_none());
    }

    #[test]
    fn test_search_by_name() {
        let catalog = ModelCatalog::new();
        let results = catalog.search("Phi-4");
        assert!(
            results.iter().any(|e| e.id == "phi-4-14b"),
            "search for 'Phi-4' should find phi-4-14b"
        );
    }

    #[test]
    fn test_search_by_publisher() {
        let catalog = ModelCatalog::new();
        let results = catalog.search("Microsoft");
        assert!(results.len() >= 2, "Microsoft should publish multiple models");
    }

    #[test]
    fn test_search_partial_match() {
        let catalog = ModelCatalog::new();
        let results = catalog.search("llama");
        assert!(
            results.iter().any(|e| e.id == "llama-3.2-1b"),
            "partial search for 'llama' should find LLaMA entries"
        );
        assert!(
            results.iter().any(|e| e.id == "llama-3.2-3b"),
            "partial search for 'llama' should find both LLaMA entries"
        );
    }

    #[test]
    fn test_search_empty_query() {
        let catalog = ModelCatalog::new();
        let results = catalog.search("");
        assert_eq!(
            results.len(),
            catalog.list_all().len(),
            "empty search should return all entries"
        );
    }

    #[test]
    fn test_search_no_results() {
        let catalog = ModelCatalog::new();
        let results = catalog.search("zzz_no_such_model_zzz");
        assert!(results.is_empty(), "nonsense query should return no results");
    }

    #[test]
    fn test_filter_by_task_text_generation() {
        let catalog = ModelCatalog::new();
        let results = catalog.filter_by_task(&ModelTask::TextGeneration);
        assert!(results.len() >= 5, "most models should support TextGeneration");
    }

    #[test]
    fn test_filter_by_task_code_generation() {
        let catalog = ModelCatalog::new();
        let results = catalog.filter_by_task(&ModelTask::CodeGeneration);
        assert!(!results.is_empty(), "at least one model should support CodeGeneration");
        assert!(
            results.iter().any(|e| e.id == "phi-4-14b"),
            "Phi-4 should support code generation"
        );
    }

    #[test]
    fn test_filter_by_task_chat() {
        let catalog = ModelCatalog::new();
        let results = catalog.filter_by_task(&ModelTask::Chat);
        assert!(results.len() >= 5, "many models should support Chat");
    }

    #[test]
    fn test_filter_by_task_no_match() {
        // All entries support at least one task, but we can check a
        // specific task that few support.
        let catalog = ModelCatalog::new();
        let results = catalog.filter_by_task(&ModelTask::FunctionCalling);
        // Only Qwen2.5-7B and Mistral-7B-v0.3 support FunctionCalling
        assert!(results.len() >= 2);
        for entry in &results {
            assert!(entry.supported_tasks.contains(&ModelTask::FunctionCalling));
        }
    }

    #[test]
    fn test_filter_by_max_params_2b() {
        let catalog = ModelCatalog::new();
        let results = catalog.filter_by_max_params(2_000_000_000);
        assert!(!results.is_empty(), "should have models with <= 2B params");
        for entry in &results {
            assert!(entry.parameter_count <= 2_000_000_000);
        }
    }

    #[test]
    fn test_filter_by_max_params_5b() {
        let catalog = ModelCatalog::new();
        let results = catalog.filter_by_max_params(5_000_000_000);
        // Should include 1.24B, 1.54B, 1.71B, 2.4B, 2.61B, 3.21B, 3.8B
        assert!(results.len() >= 7);
        for entry in &results {
            assert!(entry.parameter_count <= 5_000_000_000);
        }
    }

    #[test]
    fn test_filter_by_max_params_10b() {
        let catalog = ModelCatalog::new();
        let results = catalog.filter_by_max_params(10_000_000_000);
        // All except Phi-4 14B
        assert!(results.len() >= 9);
    }

    #[test]
    fn test_filter_by_max_params_20b() {
        let catalog = ModelCatalog::new();
        let results = catalog.filter_by_max_params(20_000_000_000);
        assert_eq!(
            results.len(),
            catalog.list_all().len(),
            "20B threshold should include all entries"
        );
    }

    #[test]
    fn test_model_catalog_entry_construction() {
        let entry = ModelCatalogEntry {
            id: "test-model".into(),
            name: "Test Model".into(),
            publisher: "Test Org".into(),
            hf_repo: "test-org/test-model".into(),
            architecture: "TestArch".into(),
            parameter_count: 100_000_000,
            default_dtype: "f32".into(),
            license: "MIT".into(),
            release_date: "2025-01-01".into(),
            description: "A test model".into(),
            tags: vec!["test".into()],
            supported_tasks: vec![ModelTask::TextGeneration],
            recommended_hardware: HardwareRecommendation {
                min_ram_gb: 2.0,
                recommended_ram_gb: 4.0,
                min_vram_gb: None,
                supports_cpu_inference: true,
                supports_quantization: false,
                recommended_quantization: None,
            },
        };
        assert_eq!(entry.id, "test-model");
        assert_eq!(entry.parameter_count, 100_000_000);
    }

    #[test]
    fn test_model_task_display() {
        assert_eq!(ModelTask::TextGeneration.to_string(), "Text Generation");
        assert_eq!(ModelTask::CodeGeneration.to_string(), "Code Generation");
        assert_eq!(ModelTask::Chat.to_string(), "Chat");
        assert_eq!(ModelTask::InstructionFollowing.to_string(), "Instruction Following");
        assert_eq!(ModelTask::Summarization.to_string(), "Summarization");
        assert_eq!(ModelTask::Translation.to_string(), "Translation");
        assert_eq!(ModelTask::QuestionAnswering.to_string(), "Question Answering");
        assert_eq!(ModelTask::FunctionCalling.to_string(), "Function Calling");
    }

    #[test]
    fn test_hardware_recommendation_cpu_only() {
        let hw = HardwareRecommendation {
            min_ram_gb: 4.0,
            recommended_ram_gb: 8.0,
            min_vram_gb: None,
            supports_cpu_inference: true,
            supports_quantization: false,
            recommended_quantization: None,
        };
        assert!(hw.supports_cpu_inference);
        assert!(hw.min_vram_gb.is_none());
        assert!(!hw.supports_quantization);
    }

    #[test]
    fn test_hardware_recommendation_gpu() {
        let hw = HardwareRecommendation {
            min_ram_gb: 16.0,
            recommended_ram_gb: 32.0,
            min_vram_gb: Some(12.0),
            supports_cpu_inference: true,
            supports_quantization: true,
            recommended_quantization: Some("Q4_K_M".into()),
        };
        assert_eq!(hw.min_vram_gb, Some(12.0));
        assert_eq!(hw.recommended_quantization.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn test_format_catalog_contains_header() {
        let catalog = ModelCatalog::new();
        let output = format_catalog(&catalog);
        assert!(output.contains("ID"), "table should have ID header");
        assert!(output.contains("Publisher"), "table should have Publisher header");
    }

    #[test]
    fn test_format_catalog_contains_entries() {
        let catalog = ModelCatalog::new();
        let output = format_catalog(&catalog);
        assert!(output.contains("phi-4-14b"));
        assert!(output.contains("bitnet-b1.58-2b"));
    }

    #[test]
    fn test_catalog_ids_are_unique() {
        let catalog = ModelCatalog::new();
        let ids: HashSet<&str> = catalog.list_all().iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids.len(), catalog.list_all().len(), "all catalog entry IDs must be unique");
    }

    #[test]
    fn test_all_entries_have_valid_hf_repo() {
        let catalog = ModelCatalog::new();
        for entry in catalog.list_all() {
            assert!(
                entry.hf_repo.contains('/'),
                "hf_repo '{}' should contain owner/repo format",
                entry.hf_repo
            );
            let parts: Vec<&str> = entry.hf_repo.splitn(2, '/').collect();
            assert_eq!(parts.len(), 2);
            assert!(
                !parts[0].is_empty() && !parts[1].is_empty(),
                "hf_repo '{}' has empty owner or repo",
                entry.hf_repo
            );
        }
    }

    #[test]
    fn test_default_trait() {
        let catalog = ModelCatalog::default();
        assert!(catalog.list_all().len() >= 10);
    }

    #[test]
    fn test_bitnet_entry_has_no_vram_requirement() {
        let catalog = ModelCatalog::new();
        let entry = catalog.get("bitnet-b1.58-2b").expect("bitnet entry should exist");
        assert!(
            entry.recommended_hardware.min_vram_gb.is_none(),
            "BitNet model is CPU-optimised; no VRAM requirement"
        );
        assert!(!entry.recommended_hardware.supports_quantization);
    }
}
