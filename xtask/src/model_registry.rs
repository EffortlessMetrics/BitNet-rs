//! Registry of known HuggingFace model entries for `download-model`.
//!
//! Each entry describes a model's repo ID, expected format, approximate size,
//! whether an auth token is required, and the files to download.

use std::fmt;

/// Format of the model weights on HuggingFace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gguf => write!(f, "GGUF"),
            Self::SafeTensors => write!(f, "SafeTensors"),
        }
    }
}

/// A known model entry in the registry.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// HuggingFace repository ID (e.g. `microsoft/phi-4`).
    pub repo_id: &'static str,
    /// Human-readable display name.
    pub display_name: &'static str,
    /// Weight format.
    pub format: ModelFormat,
    /// Approximate total download size.
    pub approx_size: &'static str,
    /// Whether `HF_TOKEN` is required.
    pub auth_required: bool,
    /// Files to download (relative to repo root).
    pub files: &'static [&'static str],
}

/// All known models. The first entry is the default BitNet model.
pub static KNOWN_MODELS: &[ModelEntry] = &[
    // ── BitNet (default) ────────────────────────────────────────────
    ModelEntry {
        repo_id: "microsoft/bitnet-b1.58-2B-4T-gguf",
        display_name: "BitNet b1.58 2B-4T (GGUF)",
        format: ModelFormat::Gguf,
        approx_size: "~0.5 GB",
        auth_required: false,
        files: &["ggml-model-i2_s.gguf"],
    },
    // ── Phi-4 family ────────────────────────────────────────────────
    ModelEntry {
        repo_id: "microsoft/phi-4",
        display_name: "Phi-4 14B",
        format: ModelFormat::SafeTensors,
        approx_size: "~29 GB",
        auth_required: true,
        files: &[
            "model-00001-of-00006.safetensors",
            "model-00002-of-00006.safetensors",
            "model-00003-of-00006.safetensors",
            "model-00004-of-00006.safetensors",
            "model-00005-of-00006.safetensors",
            "model-00006-of-00006.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    },
    ModelEntry {
        repo_id: "microsoft/Phi-4-mini-instruct",
        display_name: "Phi-4-mini Instruct 3.8B",
        format: ModelFormat::SafeTensors,
        approx_size: "~7.6 GB",
        auth_required: false,
        files: &[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    },
    // ── Qwen 2.5 family ─────────────────────────────────────────────
    ModelEntry {
        repo_id: "Qwen/Qwen2.5-7B-Instruct",
        display_name: "Qwen 2.5 7B Instruct",
        format: ModelFormat::SafeTensors,
        approx_size: "~15 GB",
        auth_required: false,
        files: &[
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    },
    ModelEntry {
        repo_id: "Qwen/Qwen2.5-1.5B-Instruct",
        display_name: "Qwen 2.5 1.5B Instruct",
        format: ModelFormat::SafeTensors,
        approx_size: "~3 GB",
        auth_required: false,
        files: &[
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    },
    // ── Google Gemma ─────────────────────────────────────────────────
    ModelEntry {
        repo_id: "google/gemma-2-2b-it",
        display_name: "Gemma 2 2B Instruct",
        format: ModelFormat::SafeTensors,
        approx_size: "~5 GB",
        auth_required: true,
        files: &[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    },
    // ── Mistral ──────────────────────────────────────────────────────
    ModelEntry {
        repo_id: "mistralai/Mistral-7B-Instruct-v0.3",
        display_name: "Mistral 7B Instruct v0.3",
        format: ModelFormat::SafeTensors,
        approx_size: "~15 GB",
        auth_required: false,
        files: &[
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    },
    // ── Meta LLaMA 3.2 ──────────────────────────────────────────────
    ModelEntry {
        repo_id: "meta-llama/Llama-3.2-1B-Instruct",
        display_name: "LLaMA 3.2 1B Instruct",
        format: ModelFormat::SafeTensors,
        approx_size: "~2.5 GB",
        auth_required: true,
        files: &[
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    },
    // ── HuggingFace SmolLM2 ─────────────────────────────────────────
    ModelEntry {
        repo_id: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        display_name: "SmolLM2 1.7B Instruct",
        format: ModelFormat::SafeTensors,
        approx_size: "~3.4 GB",
        auth_required: false,
        files: &[
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    },
];

/// Look up a model by repo ID (case-insensitive).
pub fn lookup(repo_id: &str) -> Option<&'static ModelEntry> {
    KNOWN_MODELS.iter().find(|m| m.repo_id.eq_ignore_ascii_case(repo_id))
}

/// Format the registry as a human-readable table for `--list`.
pub fn format_table() -> String {
    let mut out = String::from("Known models:\n\n");
    out.push_str(&format!(
        "  {:<50} {:<14} {:<10} {}\n",
        "REPO ID", "FORMAT", "SIZE", "AUTH"
    ));
    out.push_str(&format!("  {}\n", "-".repeat(90)));
    for m in KNOWN_MODELS {
        out.push_str(&format!(
            "  {:<50} {:<14} {:<10} {}\n",
            m.repo_id,
            m.format,
            m.approx_size,
            if m.auth_required { "HF_TOKEN" } else { "none" }
        ));
    }
    out.push_str(&format!(
        "\nUse --id <REPO_ID> to select a model. \
         Files are listed with --list --id <REPO_ID>.\n"
    ));
    out
}

/// Format detailed info for a single model entry.
pub fn format_detail(entry: &ModelEntry) -> String {
    let mut out = format!("Model: {} ({})\n", entry.display_name, entry.repo_id);
    out.push_str(&format!("  Format:  {}\n", entry.format));
    out.push_str(&format!("  Size:    {}\n", entry.approx_size));
    out.push_str(&format!(
        "  Auth:    {}\n",
        if entry.auth_required { "HF_TOKEN required" } else { "none" }
    ));
    out.push_str(&format!("  Files ({}):\n", entry.files.len()));
    for f in entry.files {
        out.push_str(&format!("    - {f}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_default_model() {
        let entry = lookup("microsoft/bitnet-b1.58-2B-4T-gguf").unwrap();
        assert_eq!(entry.format, ModelFormat::Gguf);
        assert!(!entry.auth_required);
        assert!(entry.files.contains(&"ggml-model-i2_s.gguf"));
    }

    #[test]
    fn test_lookup_phi4() {
        let entry = lookup("microsoft/phi-4").unwrap();
        assert_eq!(entry.format, ModelFormat::SafeTensors);
        assert!(entry.auth_required);
        assert!(entry.files.contains(&"config.json"));
        assert!(entry.files.contains(&"tokenizer.json"));
        assert_eq!(entry.files.iter().filter(|f| f.ends_with(".safetensors")).count(), 6);
    }

    #[test]
    fn test_lookup_phi4_mini() {
        let entry = lookup("microsoft/Phi-4-mini-instruct").unwrap();
        assert_eq!(entry.approx_size, "~7.6 GB");
        assert!(!entry.auth_required);
    }

    #[test]
    fn test_lookup_qwen_7b() {
        let entry = lookup("Qwen/Qwen2.5-7B-Instruct").unwrap();
        assert_eq!(entry.format, ModelFormat::SafeTensors);
        assert!(!entry.auth_required);
    }

    #[test]
    fn test_lookup_qwen_1_5b() {
        let entry = lookup("Qwen/Qwen2.5-1.5B-Instruct").unwrap();
        assert_eq!(entry.approx_size, "~3 GB");
        assert!(entry.files.contains(&"model.safetensors"));
    }

    #[test]
    fn test_lookup_gemma() {
        let entry = lookup("google/gemma-2-2b-it").unwrap();
        assert!(entry.auth_required);
        assert_eq!(entry.approx_size, "~5 GB");
    }

    #[test]
    fn test_lookup_mistral() {
        let entry = lookup("mistralai/Mistral-7B-Instruct-v0.3").unwrap();
        assert!(!entry.auth_required);
        assert_eq!(entry.files.iter().filter(|f| f.ends_with(".safetensors")).count(), 3);
    }

    #[test]
    fn test_lookup_llama() {
        let entry = lookup("meta-llama/Llama-3.2-1B-Instruct").unwrap();
        assert!(entry.auth_required);
        assert_eq!(entry.approx_size, "~2.5 GB");
    }

    #[test]
    fn test_lookup_smollm2() {
        let entry = lookup("HuggingFaceTB/SmolLM2-1.7B-Instruct").unwrap();
        assert!(!entry.auth_required);
        assert_eq!(entry.approx_size, "~3.4 GB");
    }

    #[test]
    fn test_lookup_case_insensitive() {
        assert!(lookup("MICROSOFT/PHI-4").is_some());
        assert!(lookup("qwen/qwen2.5-7b-instruct").is_some());
    }

    #[test]
    fn test_lookup_unknown_returns_none() {
        assert!(lookup("unknown/nonexistent-model").is_none());
    }

    #[test]
    fn test_all_entries_have_config_and_tokenizer_or_gguf() {
        for m in KNOWN_MODELS {
            match m.format {
                ModelFormat::SafeTensors => {
                    assert!(
                        m.files.contains(&"config.json"),
                        "{} missing config.json",
                        m.repo_id
                    );
                    assert!(
                        m.files.contains(&"tokenizer.json"),
                        "{} missing tokenizer.json",
                        m.repo_id
                    );
                }
                ModelFormat::Gguf => {
                    assert!(
                        m.files.iter().any(|f| f.ends_with(".gguf")),
                        "{} has no .gguf file",
                        m.repo_id
                    );
                }
            }
        }
    }

    #[test]
    fn test_format_table_contains_all_models() {
        let table = format_table();
        for m in KNOWN_MODELS {
            assert!(table.contains(m.repo_id), "table missing {}", m.repo_id);
        }
    }

    #[test]
    fn test_list_flag_shows_all_known_models() {
        let table = format_table();
        assert_eq!(
            KNOWN_MODELS.len(),
            table.lines().filter(|l| l.contains('/')).count(),
            "--list output should contain one line per model"
        );
    }

    #[test]
    fn test_format_detail_includes_files() {
        let entry = lookup("microsoft/phi-4").unwrap();
        let detail = format_detail(entry);
        for f in entry.files {
            assert!(detail.contains(f), "detail missing file {f}");
        }
    }

    #[test]
    fn test_url_construction() {
        let entry = lookup("microsoft/phi-4").unwrap();
        let base = "https://huggingface.co";
        let rev = "main";
        for f in entry.files {
            let url = format!("{base}/{}/resolve/{rev}/{f}", entry.repo_id);
            assert!(
                url.starts_with("https://huggingface.co/microsoft/phi-4/resolve/main/"),
                "bad URL: {url}"
            );
        }
    }

    #[test]
    fn test_no_duplicate_repo_ids() {
        let mut seen = std::collections::HashSet::new();
        for m in KNOWN_MODELS {
            assert!(
                seen.insert(m.repo_id.to_ascii_lowercase()),
                "duplicate repo_id: {}",
                m.repo_id
            );
        }
    }
}
