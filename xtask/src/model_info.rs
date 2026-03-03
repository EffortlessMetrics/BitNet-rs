//! Model info subcommand for inspecting SLM model metadata.
//!
//! Displays detailed information about a model file including architecture,
//! parameters, layers, memory estimates, and format detection.

use std::path::Path;

/// CLI arguments for the `model-info` subcommand.
#[derive(Debug, Clone, Default)]
pub struct ModelInfoCommand {
    /// Path to model file or HuggingFace repo ID.
    pub model_path: String,
    /// Model format hint: "gguf", "safetensors", or "auto".
    pub format: Option<String>,
    /// Output as JSON instead of human-readable table.
    pub json: bool,
    /// Show extra details (special tokens, quantization info).
    pub verbose: bool,
}

/// Comprehensive model metadata output.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelInfoOutput {
    pub name: String,
    pub architecture: String,
    pub format: String,
    pub num_parameters: u64,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub max_context_length: usize,
    pub dtype: String,
    pub file_size_bytes: u64,
    pub estimated_memory_gb: f32,
    pub quantization: Option<String>,
    pub special_tokens: Vec<(String, u32)>,
}

/// Format model info as a human-readable table.
pub fn format_model_info(info: &ModelInfoOutput) -> String {
    let gqa_ratio =
        if info.num_kv_heads > 0 { info.num_attention_heads / info.num_kv_heads } else { 0 };

    let mut out = String::new();
    out.push_str(&format!("Model: {}\n", info.name));
    out.push_str(&format!("Architecture: {}\n", info.architecture));
    out.push_str(&format!("Parameters: {}\n", format_parameter_count(info.num_parameters)));
    out.push_str(&format!("Layers: {}\n", info.num_layers));
    out.push_str(&format!("Hidden Size: {}\n", info.hidden_size));

    if gqa_ratio > 1 {
        out.push_str(&format!(
            "Attention: {} heads, {} KV heads (GQA {}:1)\n",
            info.num_attention_heads, info.num_kv_heads, gqa_ratio
        ));
    } else {
        out.push_str(&format!(
            "Attention: {} heads, {} KV heads\n",
            info.num_attention_heads, info.num_kv_heads
        ));
    }

    out.push_str(&format!("Vocab Size: {}\n", format_number_with_commas(info.vocab_size)));
    out.push_str(&format!(
        "Context: {} tokens\n",
        format_number_with_commas(info.max_context_length)
    ));
    out.push_str(&format!("Dtype: {}\n", info.dtype));
    out.push_str(&format!("File Size: {}\n", format_file_size(info.file_size_bytes)));
    out.push_str(&format!("Memory (est.): {:.1} GB\n", info.estimated_memory_gb));

    if let Some(ref q) = info.quantization {
        out.push_str(&format!("Quantization: {}\n", q));
    }

    if !info.special_tokens.is_empty() {
        out.push_str("Special Tokens:\n");
        for (name, id) in &info.special_tokens {
            out.push_str(&format!("  {}: {}\n", name, id));
        }
    }

    out
}

/// Format a parameter count in human-readable form.
///
/// Examples: "125M", "355M", "1.2B", "7.0B", "14.0B", "70.0B"
pub fn format_parameter_count(count: u64) -> String {
    if count == 0 {
        return "0".to_string();
    }
    let billions = count as f64 / 1_000_000_000.0;
    let millions = count as f64 / 1_000_000.0;

    if count >= 1_000_000_000 {
        format!("{:.1}B", billions)
    } else if count >= 1_000_000 {
        format!("{:.0}M", millions)
    } else if count >= 1_000 {
        format!("{:.0}K", count as f64 / 1_000.0)
    } else {
        format!("{}", count)
    }
}

/// Format a byte count as human-readable file size.
///
/// Examples: "256 MB", "3.2 GB", "27.5 GB"
pub fn format_file_size(bytes: u64) -> String {
    if bytes == 0 {
        return "0 B".to_string();
    }

    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    const TB: u64 = 1024 * GB;

    if bytes >= TB {
        format!("{:.1} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Detect model format from file path or directory structure.
///
/// Returns "GGUF", "SafeTensors", or "SafeTensors (sharded)".
pub fn detect_model_format(path: &str) -> String {
    let p = Path::new(path);

    if path.ends_with(".gguf") {
        return "GGUF".to_string();
    }

    if path.ends_with(".safetensors") {
        return "SafeTensors".to_string();
    }

    // Check if directory contains sharded safetensors index
    if p.is_dir() {
        let index_path = p.join("model.safetensors.index.json");
        if index_path.exists() {
            return "SafeTensors (sharded)".to_string();
        }
    }

    "Unknown".to_string()
}

/// Format a number with comma separators.
fn format_number_with_commas(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Format model info as JSON.
pub fn format_model_info_json(info: &ModelInfoOutput) -> String {
    serde_json::to_string_pretty(info).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
#[allow(clippy::all, clippy::pedantic, clippy::nursery)]
mod tests {
    use super::*;

    // ── format_parameter_count tests ────────────────────────────────

    #[test]
    fn test_format_parameter_count_125m() {
        assert_eq!(format_parameter_count(125_000_000), "125M");
    }

    #[test]
    fn test_format_parameter_count_355m() {
        assert_eq!(format_parameter_count(355_000_000), "355M");
    }

    #[test]
    fn test_format_parameter_count_1_2b() {
        assert_eq!(format_parameter_count(1_200_000_000), "1.2B");
    }

    #[test]
    fn test_format_parameter_count_7b() {
        assert_eq!(format_parameter_count(7_000_000_000), "7.0B");
    }

    #[test]
    fn test_format_parameter_count_14b() {
        assert_eq!(format_parameter_count(14_000_000_000), "14.0B");
    }

    #[test]
    fn test_format_parameter_count_70b() {
        assert_eq!(format_parameter_count(70_000_000_000), "70.0B");
    }

    #[test]
    fn test_format_parameter_count_zero() {
        assert_eq!(format_parameter_count(0), "0");
    }

    #[test]
    fn test_format_parameter_count_small() {
        assert_eq!(format_parameter_count(500), "500");
    }

    #[test]
    fn test_format_parameter_count_thousands() {
        assert_eq!(format_parameter_count(50_000), "50K");
    }

    // ── format_file_size tests ──────────────────────────────────────

    #[test]
    fn test_format_file_size_bytes() {
        assert_eq!(format_file_size(512), "512 B");
    }

    #[test]
    fn test_format_file_size_kb() {
        assert_eq!(format_file_size(2048), "2.0 KB");
    }

    #[test]
    fn test_format_file_size_mb() {
        // 256 MB = 256 * 1024 * 1024
        assert_eq!(format_file_size(268_435_456), "256.0 MB");
    }

    #[test]
    fn test_format_file_size_gb() {
        // ~3.2 GB
        let bytes = (3.2 * 1024.0 * 1024.0 * 1024.0) as u64;
        assert_eq!(format_file_size(bytes), "3.2 GB");
    }

    #[test]
    fn test_format_file_size_large_gb() {
        // ~27.5 GB
        let bytes = (27.5 * 1024.0 * 1024.0 * 1024.0) as u64;
        assert_eq!(format_file_size(bytes), "27.5 GB");
    }

    #[test]
    fn test_format_file_size_tb() {
        // 1.5 TB
        let bytes = (1.5 * 1024.0 * 1024.0 * 1024.0 * 1024.0) as u64;
        assert_eq!(format_file_size(bytes), "1.5 TB");
    }

    #[test]
    fn test_format_file_size_zero() {
        assert_eq!(format_file_size(0), "0 B");
    }

    // ── detect_model_format tests ───────────────────────────────────

    #[test]
    fn test_detect_format_gguf() {
        assert_eq!(detect_model_format("model.gguf"), "GGUF");
    }

    #[test]
    fn test_detect_format_gguf_with_path() {
        assert_eq!(detect_model_format("models/ggml-model-i2_s.gguf"), "GGUF");
    }

    #[test]
    fn test_detect_format_safetensors() {
        assert_eq!(detect_model_format("model.safetensors"), "SafeTensors");
    }

    #[test]
    fn test_detect_format_safetensors_with_path() {
        assert_eq!(detect_model_format("models/model-00001-of-00006.safetensors"), "SafeTensors");
    }

    #[test]
    fn test_detect_format_unknown() {
        assert_eq!(detect_model_format("model.bin"), "Unknown");
    }

    // ── ModelInfoOutput construction tests ───────────────────────────

    fn make_phi4_info() -> ModelInfoOutput {
        ModelInfoOutput {
            name: "microsoft/phi-4".to_string(),
            architecture: "Phi4".to_string(),
            format: "SafeTensors".to_string(),
            num_parameters: 14_000_000_000,
            num_layers: 40,
            hidden_size: 5120,
            num_attention_heads: 40,
            num_kv_heads: 10,
            vocab_size: 100_352,
            max_context_length: 16_384,
            dtype: "BF16".to_string(),
            file_size_bytes: (27.5 * 1024.0 * 1024.0 * 1024.0) as u64,
            estimated_memory_gb: 30.1,
            quantization: None,
            special_tokens: vec![("bos".to_string(), 1), ("eos".to_string(), 2)],
        }
    }

    fn make_llama3_info() -> ModelInfoOutput {
        ModelInfoOutput {
            name: "meta-llama/Llama-3.2-1B-Instruct".to_string(),
            architecture: "LLaMA".to_string(),
            format: "SafeTensors".to_string(),
            num_parameters: 1_240_000_000,
            num_layers: 16,
            hidden_size: 2048,
            num_attention_heads: 32,
            num_kv_heads: 8,
            vocab_size: 128_256,
            max_context_length: 131_072,
            dtype: "BF16".to_string(),
            file_size_bytes: (2.5 * 1024.0 * 1024.0 * 1024.0) as u64,
            estimated_memory_gb: 2.8,
            quantization: None,
            special_tokens: vec![],
        }
    }

    fn make_qwen25_info() -> ModelInfoOutput {
        ModelInfoOutput {
            name: "Qwen/Qwen2.5-7B-Instruct".to_string(),
            architecture: "Qwen2".to_string(),
            format: "SafeTensors".to_string(),
            num_parameters: 7_600_000_000,
            num_layers: 28,
            hidden_size: 3584,
            num_attention_heads: 28,
            num_kv_heads: 4,
            vocab_size: 152_064,
            max_context_length: 131_072,
            dtype: "BF16".to_string(),
            file_size_bytes: (15.0 * 1024.0 * 1024.0 * 1024.0) as u64,
            estimated_memory_gb: 16.5,
            quantization: None,
            special_tokens: vec![],
        }
    }

    #[test]
    fn test_phi4_construction() {
        let info = make_phi4_info();
        assert_eq!(info.name, "microsoft/phi-4");
        assert_eq!(info.architecture, "Phi4");
        assert_eq!(info.num_parameters, 14_000_000_000);
        assert_eq!(info.num_layers, 40);
        assert_eq!(info.hidden_size, 5120);
        assert_eq!(info.num_attention_heads, 40);
        assert_eq!(info.num_kv_heads, 10);
    }

    #[test]
    fn test_llama3_construction() {
        let info = make_llama3_info();
        assert_eq!(info.name, "meta-llama/Llama-3.2-1B-Instruct");
        assert_eq!(info.architecture, "LLaMA");
        assert_eq!(info.num_parameters, 1_240_000_000);
        assert_eq!(info.num_layers, 16);
        assert_eq!(info.num_kv_heads, 8);
    }

    #[test]
    fn test_qwen25_construction() {
        let info = make_qwen25_info();
        assert_eq!(info.name, "Qwen/Qwen2.5-7B-Instruct");
        assert_eq!(info.architecture, "Qwen2");
        assert_eq!(info.num_parameters, 7_600_000_000);
        assert_eq!(info.num_kv_heads, 4);
        assert_eq!(info.vocab_size, 152_064);
    }

    // ── format_model_info tests ─────────────────────────────────────

    #[test]
    fn test_format_model_info_phi4_table() {
        let info = make_phi4_info();
        let output = format_model_info(&info);
        assert!(output.contains("Model: microsoft/phi-4"));
        assert!(output.contains("Architecture: Phi4"));
        assert!(output.contains("Parameters: 14.0B"));
        assert!(output.contains("Layers: 40"));
        assert!(output.contains("Hidden Size: 5120"));
        assert!(output.contains("GQA 4:1"));
        assert!(output.contains("Vocab Size: 100,352"));
        assert!(output.contains("Context: 16,384 tokens"));
        assert!(output.contains("Dtype: BF16"));
        assert!(output.contains("Memory (est.): 30.1 GB"));
    }

    #[test]
    fn test_format_model_info_llama3_no_gqa_label_when_ratio_gt_1() {
        let info = make_llama3_info();
        let output = format_model_info(&info);
        // LLaMA-3.2-1B has 32 heads / 8 KV = GQA 4:1
        assert!(output.contains("GQA 4:1"));
        assert!(output.contains("32 heads, 8 KV heads"));
    }

    #[test]
    fn test_format_model_info_shows_special_tokens() {
        let info = make_phi4_info();
        let output = format_model_info(&info);
        assert!(output.contains("Special Tokens:"));
        assert!(output.contains("bos: 1"));
        assert!(output.contains("eos: 2"));
    }

    #[test]
    fn test_format_model_info_hides_special_tokens_when_empty() {
        let info = make_llama3_info();
        let output = format_model_info(&info);
        assert!(!output.contains("Special Tokens:"));
    }

    #[test]
    fn test_format_model_info_shows_quantization() {
        let mut info = make_phi4_info();
        info.quantization = Some("Q4_K_M".to_string());
        let output = format_model_info(&info);
        assert!(output.contains("Quantization: Q4_K_M"));
    }

    // ── ModelInfoCommand defaults ───────────────────────────────────

    #[test]
    fn test_command_defaults() {
        let cmd = ModelInfoCommand::default();
        assert_eq!(cmd.model_path, "");
        assert!(cmd.format.is_none());
        assert!(!cmd.json);
        assert!(!cmd.verbose);
    }

    // ── JSON output mode ────────────────────────────────────────────

    #[test]
    fn test_json_output() {
        let info = make_phi4_info();
        let json_str = format_model_info_json(&info);
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["name"], "microsoft/phi-4");
        assert_eq!(parsed["architecture"], "Phi4");
        assert_eq!(parsed["num_parameters"], 14_000_000_000u64);
        assert_eq!(parsed["num_layers"], 40);
    }

    #[test]
    fn test_json_roundtrip() {
        let info = make_phi4_info();
        let json_str = format_model_info_json(&info);
        let roundtripped: ModelInfoOutput = serde_json::from_str(&json_str).unwrap();
        assert_eq!(roundtripped.name, info.name);
        assert_eq!(roundtripped.num_parameters, info.num_parameters);
        assert_eq!(roundtripped.num_layers, info.num_layers);
    }

    // ── Verbose mode ────────────────────────────────────────────────

    #[test]
    fn test_verbose_includes_special_tokens_in_json() {
        let info = make_phi4_info();
        let json_str = format_model_info_json(&info);
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let tokens = parsed["special_tokens"].as_array().unwrap();
        assert!(!tokens.is_empty());
        // First token: ["bos", 1]
        assert_eq!(tokens[0][0], "bos");
        assert_eq!(tokens[0][1], 1);
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_zero_parameters() {
        let info = ModelInfoOutput {
            name: "empty".to_string(),
            architecture: "Unknown".to_string(),
            format: "Unknown".to_string(),
            num_parameters: 0,
            num_layers: 0,
            hidden_size: 0,
            num_attention_heads: 0,
            num_kv_heads: 0,
            vocab_size: 0,
            max_context_length: 0,
            dtype: "Unknown".to_string(),
            file_size_bytes: 0,
            estimated_memory_gb: 0.0,
            quantization: None,
            special_tokens: vec![],
        };
        let output = format_model_info(&info);
        assert!(output.contains("Parameters: 0"));
        assert!(output.contains("Layers: 0"));
    }

    #[test]
    fn test_very_large_model() {
        let info = ModelInfoOutput {
            name: "hypothetical/405b".to_string(),
            architecture: "LLaMA".to_string(),
            format: "SafeTensors".to_string(),
            num_parameters: 405_000_000_000,
            num_layers: 126,
            hidden_size: 16384,
            num_attention_heads: 128,
            num_kv_heads: 8,
            vocab_size: 128_256,
            max_context_length: 131_072,
            dtype: "BF16".to_string(),
            file_size_bytes: (800.0 * 1024.0 * 1024.0 * 1024.0) as u64,
            estimated_memory_gb: 810.0,
            quantization: None,
            special_tokens: vec![],
        };
        let output = format_model_info(&info);
        assert!(output.contains("Parameters: 405.0B"));
        assert!(output.contains("Layers: 126"));
        assert!(output.contains("GQA 16:1"));
    }

    #[test]
    fn test_unknown_architecture() {
        let info = ModelInfoOutput {
            name: "custom/model".to_string(),
            architecture: "Unknown".to_string(),
            format: "GGUF".to_string(),
            num_parameters: 500_000_000,
            num_layers: 24,
            hidden_size: 1024,
            num_attention_heads: 16,
            num_kv_heads: 16,
            vocab_size: 32_000,
            max_context_length: 4096,
            dtype: "F16".to_string(),
            file_size_bytes: 1_000_000_000,
            estimated_memory_gb: 1.1,
            quantization: Some("I2_S".to_string()),
            special_tokens: vec![],
        };
        let output = format_model_info(&info);
        assert!(output.contains("Architecture: Unknown"));
        assert!(output.contains("Quantization: I2_S"));
        // MHA (equal heads) → no GQA label
        assert!(!output.contains("GQA"));
    }

    #[test]
    fn test_number_formatting_with_commas() {
        assert_eq!(format_number_with_commas(0), "0");
        assert_eq!(format_number_with_commas(999), "999");
        assert_eq!(format_number_with_commas(1_000), "1,000");
        assert_eq!(format_number_with_commas(100_352), "100,352");
        assert_eq!(format_number_with_commas(1_000_000), "1,000,000");
    }
}
