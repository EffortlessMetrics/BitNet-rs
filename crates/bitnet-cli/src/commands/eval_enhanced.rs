use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Enhanced evaluation results with complete metadata
#[derive(Debug, Serialize, Deserialize)]
pub struct EnhancedEvalResults {
    // Core evaluation results
    pub model_path: String,
    pub text_file: String,
    pub lines_evaluated: usize,
    pub total_tokens: usize,
    pub mean_nll: f64,
    pub std_nll: f64,
    pub perplexity: f64,
    pub timing_ms: EvalTiming,
    pub tokens_per_second: f64,

    // Optional fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits_dump: Option<Vec<LogitStep>>,

    // Complete metadata
    pub metadata: EvalMetadata,

    // Scoring policy
    pub scoring_policy: ScoringPolicy,

    // Tokenizer information
    pub tokenizer: TokenizerInfo,

    // Model format and configuration
    pub model_info: ModelInfo,

    // Totals for comprehensive tracking
    pub totals: EvalTotals,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvalMetadata {
    pub timestamp: String,
    pub platform: PlatformInfo,
    pub environment: EnvironmentInfo,
    pub deterministic: bool,
    pub seed: Option<u64>,
    pub threads: usize,
    pub bitnet_version: String,
    pub rust_version: String,
    pub command_line: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PlatformInfo {
    pub os: String,
    pub arch: String,
    pub cpu: String,
    pub is_wsl: bool,
    pub hostname: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub bitnet_deterministic: bool,
    pub rayon_num_threads: usize,
    pub omp_num_threads: usize,
    pub cuda_device: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScoringPolicy {
    pub add_bos: bool,
    pub append_eos: bool,
    pub mask_pad: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenizerInfo {
    pub source: String,  // "embedded-gguf", "external-json", "safetensors"
    pub tokenizer_type: String,  // "HF", "SPM", "BPE"
    pub vocab_size: usize,
    pub special_tokens: HashMap<String, u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub format: String,  // "gguf", "safetensors"
    pub quantization: Option<String>,  // "1-bit", "FP32", etc.
    pub parameters: Option<String>,
    pub size_mb: f64,
    pub unmapped_tensors: Vec<String>,
    pub ignored_tensors: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvalTotals {
    pub lines: usize,
    pub predicted_tokens: usize,
    pub total_chars: usize,
    pub effective_batch_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvalTiming {
    pub total: f64,
    pub per_line: f64,
    pub per_token: f64,
    pub warmup_ms: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogitStep {
    pub step: usize,
    pub input_token: u32,
    pub top_k: Vec<LogitEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogitEntry {
    pub token_id: u32,
    pub logit: f32,
    pub prob: f32,
}

impl EnhancedEvalResults {
    /// Create enhanced results from basic results
    pub fn from_basic(basic: &super::EvalResults) -> Result<Self> {
        // Get platform information
        let platform = PlatformInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu: get_cpu_info(),
            is_wsl: is_wsl2(),
            hostname: hostname::get()
                .map(|h| h.to_string_lossy().to_string())
                .unwrap_or_else(|_| "unknown".to_string()),
        };

        // Get environment info
        let environment = EnvironmentInfo {
            bitnet_deterministic: std::env::var("BITNET_DETERMINISTIC")
                .map(|v| v == "1")
                .unwrap_or(false),
            rayon_num_threads: std::env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(num_cpus::get()),
            omp_num_threads: std::env::var("OMP_NUM_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
            cuda_device: std::env::var("CUDA_VISIBLE_DEVICES").ok(),
        };

        // Get version information
        let bitnet_version = env!("CARGO_PKG_VERSION").to_string();
        let rust_version = rustc_version::version()
            .map(|v| v.to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        // Build metadata
        let metadata = EvalMetadata {
            timestamp: chrono::Utc::now().to_rfc3339(),
            platform,
            environment,
            deterministic: std::env::var("BITNET_DETERMINISTIC")
                .map(|v| v == "1")
                .unwrap_or(false),
            seed: basic.seed,
            threads: basic.threads.unwrap_or(1),
            bitnet_version,
            rust_version,
            command_line: std::env::args().collect(),
        };

        // Create enhanced results
        Ok(Self {
            model_path: basic.model_path.clone(),
            text_file: basic.text_file.clone(),
            lines_evaluated: basic.lines_evaluated,
            total_tokens: basic.total_tokens,
            mean_nll: basic.mean_nll,
            std_nll: basic.std_nll,
            perplexity: basic.perplexity,
            timing_ms: basic.timing_ms.clone(),
            tokens_per_second: basic.tokens_per_second,
            logits_dump: basic.logits_dump.clone(),
            metadata,
            scoring_policy: basic.scoring_policy.clone().unwrap_or(ScoringPolicy {
                add_bos: true,
                append_eos: false,
                mask_pad: true,
            }),
            tokenizer: TokenizerInfo {
                source: detect_tokenizer_source(&basic.model_path),
                tokenizer_type: "HF".to_string(),  // Would be detected from actual tokenizer
                vocab_size: 50257,  // Would be obtained from tokenizer
                special_tokens: HashMap::new(),  // Would be populated from tokenizer
            },
            model_info: ModelInfo {
                format: detect_model_format(&basic.model_path),
                quantization: None,  // Would be detected from model
                parameters: None,  // Would be obtained from model
                size_mb: get_file_size_mb(&basic.model_path),
                unmapped_tensors: Vec::new(),  // Would be tracked during loading
                ignored_tensors: Vec::new(),  // Would be tracked during loading
            },
            totals: basic.totals.clone().unwrap_or(EvalTotals {
                lines: basic.lines_evaluated,
                predicted_tokens: basic.total_tokens,
                total_chars: 0,  // Would be calculated
                effective_batch_size: 1,
            }),
        })
    }
}

// Helper functions
fn get_cpu_info() -> String {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|line| line.starts_with("model name"))
                    .map(|line| {
                        line.split(':')
                            .nth(1)
                            .unwrap_or("unknown")
                            .trim()
                            .to_string()
                    })
            })
            .unwrap_or_else(|| "unknown".to_string())
    }

    #[cfg(not(target_os = "linux"))]
    {
        "unknown".to_string()
    }
}

fn is_wsl2() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/version")
            .map(|content| content.to_lowercase().contains("microsoft"))
            .unwrap_or(false)
    }

    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

fn detect_tokenizer_source(model_path: &str) -> String {
    if model_path.ends_with(".gguf") {
        "embedded-gguf".to_string()
    } else if model_path.ends_with(".safetensors") {
        "external-json".to_string()
    } else {
        "unknown".to_string()
    }
}

fn detect_model_format(model_path: &str) -> String {
    if model_path.ends_with(".gguf") {
        "gguf".to_string()
    } else if model_path.ends_with(".safetensors") {
        "safetensors".to_string()
    } else {
        "unknown".to_string()
    }
}

fn get_file_size_mb(path: &str) -> f64 {
    std::fs::metadata(path)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0)
}
