//! Model configuration and metadata management for BitNet and standard
//! transformer models. Provides CPU reference implementations for parameter
//! estimation, memory budgeting, and GPU auto-configuration.

use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Supported model architectures.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelArchitecture {
    BitNetB158,
    Llama,
    Mistral,
    Phi,
    Qwen,
    Custom(String),
}

impl fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitNetB158 => write!(f, "BitNet b1.58"),
            Self::Llama => write!(f, "Llama"),
            Self::Mistral => write!(f, "Mistral"),
            Self::Phi => write!(f, "Phi"),
            Self::Qwen => write!(f, "Qwen"),
            Self::Custom(name) => write!(f, "Custom({name})"),
        }
    }
}

/// Quantization parameters for a model.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationConfig {
    pub weight_bits: u8,
    pub activation_bits: u8,
    pub group_size: usize,
    pub method: String,
}

/// Core model hyper-parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
    pub quantization: Option<QuantizationConfig>,
}

/// GPU-specific execution configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuConfig {
    pub preferred_batch_size: usize,
    pub max_batch_size: usize,
    pub kv_cache_dtype: String,
    pub use_flash_attention: bool,
    pub workgroup_size_hint: Option<usize>,
}

/// Combined inference configuration.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub model: ModelConfig,
    pub gpu: GpuConfig,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_tokens: usize,
}

/// Human-readable metadata about a model artefact.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub author: String,
    pub license: String,
    pub parameters_b: f64,
    pub file_size_gb: f64,
}

/// Manages model + GPU configuration together.
#[derive(Debug, Clone)]
pub struct ConfigManager {
    pub model_config: Option<ModelConfig>,
    pub gpu_config: GpuConfig,
    pub metadata: Option<ModelMetadata>,
}

/// Configuration errors.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    InvalidConfig(String),
    MissingField(String),
    IncompatibleArchitecture(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::MissingField(field) => write!(f, "missing field: {field}"),
            Self::IncompatibleArchitecture(msg) => {
                write!(f, "incompatible architecture: {msg}")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

// ---------------------------------------------------------------------------
// CPU Reference Implementations
// ---------------------------------------------------------------------------

/// Create a new [`ConfigManager`] with the given GPU configuration.
pub fn create_config_manager(gpu: GpuConfig) -> ConfigManager {
    ConfigManager { model_config: None, gpu_config: gpu, metadata: None }
}

/// Default configuration for BitNet b1.58 2B.
pub fn cpu_bitnet_2b_config() -> ModelConfig {
    ModelConfig {
        architecture: ModelArchitecture::BitNetB158,
        vocab_size: 32000,
        hidden_size: 2048,
        num_layers: 24,
        num_heads: 32,
        num_kv_heads: 32,
        head_dim: 64,
        intermediate_size: 5504,
        max_seq_len: 2048,
        rope_base: 10000.0,
        norm_eps: 1e-5,
        quantization: Some(QuantizationConfig {
            weight_bits: 2,
            activation_bits: 8,
            group_size: 64,
            method: "i2s".to_string(),
        }),
    }
}

/// Default configuration for BitNet 3B.
pub fn cpu_bitnet_3b_config() -> ModelConfig {
    ModelConfig {
        architecture: ModelArchitecture::BitNetB158,
        vocab_size: 32000,
        hidden_size: 3200,
        num_layers: 26,
        num_heads: 32,
        num_kv_heads: 32,
        head_dim: 100,
        intermediate_size: 8640,
        max_seq_len: 2048,
        rope_base: 10000.0,
        norm_eps: 1e-5,
        quantization: Some(QuantizationConfig {
            weight_bits: 2,
            activation_bits: 8,
            group_size: 64,
            method: "i2s".to_string(),
        }),
    }
}

/// Default configuration for Llama 2 7B.
pub fn cpu_llama_7b_config() -> ModelConfig {
    ModelConfig {
        architecture: ModelArchitecture::Llama,
        vocab_size: 32000,
        hidden_size: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 32,
        head_dim: 128,
        intermediate_size: 11008,
        max_seq_len: 4096,
        rope_base: 10000.0,
        norm_eps: 1e-5,
        quantization: None,
    }
}

/// Validate that a [`ModelConfig`] has sensible values.
pub fn cpu_validate_config(config: &ModelConfig) -> Result<(), ConfigError> {
    if config.hidden_size == 0 {
        return Err(ConfigError::InvalidConfig(
            "hidden_size must be > 0".to_string(),
        ));
    }
    if config.num_layers == 0 {
        return Err(ConfigError::InvalidConfig(
            "num_layers must be > 0".to_string(),
        ));
    }
    if config.num_heads == 0 {
        return Err(ConfigError::InvalidConfig(
            "num_heads must be > 0".to_string(),
        ));
    }
    if config.vocab_size == 0 {
        return Err(ConfigError::InvalidConfig(
            "vocab_size must be > 0".to_string(),
        ));
    }
    if config.head_dim == 0 {
        return Err(ConfigError::InvalidConfig(
            "head_dim must be > 0".to_string(),
        ));
    }
    if config.intermediate_size == 0 {
        return Err(ConfigError::InvalidConfig(
            "intermediate_size must be > 0".to_string(),
        ));
    }
    if config.max_seq_len == 0 {
        return Err(ConfigError::InvalidConfig(
            "max_seq_len must be > 0".to_string(),
        ));
    }
    if config.num_kv_heads > config.num_heads {
        return Err(ConfigError::InvalidConfig(
            "num_kv_heads must be <= num_heads".to_string(),
        ));
    }
    Ok(())
}

/// Estimate the total number of parameters (in billions).
pub fn cpu_estimate_parameters(config: &ModelConfig) -> f64 {
    let h = config.hidden_size as f64;
    let l = config.num_layers as f64;
    let v = config.vocab_size as f64;
    let ff = config.intermediate_size as f64;
    let kv_heads = config.num_kv_heads as f64;
    let heads = config.num_heads as f64;
    let hd = config.head_dim as f64;

    // Embedding
    let embed = v * h;
    // Per-layer attention: Q + K + V projections + output projection
    let kv_dim = kv_heads * hd;
    let q_dim = heads * hd;
    let attn_per_layer = h * q_dim + h * kv_dim + h * kv_dim + q_dim * h;
    // Per-layer FFN (gate + up + down for SwiGLU-style)
    let ffn_per_layer = 3.0 * h * ff;
    // LayerNorm (2 per layer) + final norm
    let norm_per_layer = 4.0 * h;

    let total =
        embed + l * (attn_per_layer + ffn_per_layer + norm_per_layer) + v * h;
    total / 1e9
}

/// Estimate VRAM required in MB.
pub fn cpu_estimate_memory_mb(config: &ModelConfig) -> f64 {
    let params = cpu_estimate_parameters(config) * 1e9;
    let bytes_per_param = match &config.quantization {
        Some(q) => q.weight_bits as f64 / 8.0,
        None => 2.0, // fp16
    };
    let weight_mb = params * bytes_per_param / (1024.0 * 1024.0);

    // KV cache estimate: 2 * num_layers * 2 * num_kv_heads * head_dim * max_seq_len * 2 bytes
    let kv_cache_mb = 2.0
        * config.num_layers as f64
        * config.num_kv_heads as f64
        * config.head_dim as f64
        * config.max_seq_len as f64
        * 2.0
        / (1024.0 * 1024.0);

    // ~10 % overhead for activations / workspace
    (weight_mb + kv_cache_mb) * 1.1
}

/// Estimate FLOPs per forward pass for a single token.
pub fn cpu_estimate_flops_per_token(config: &ModelConfig) -> f64 {
    let h = config.hidden_size as f64;
    let l = config.num_layers as f64;
    let ff = config.intermediate_size as f64;
    let s = config.max_seq_len as f64;

    // Attention: Q*K^T + softmax + attn*V ≈ 4 * h * s per layer
    let attn_flops = 4.0 * h * s;
    // FFN: 2 matmuls of h×ff  → 4 * h * ff per layer
    let ffn_flops = 4.0 * h * ff;

    l * (attn_flops + ffn_flops)
}

/// Check whether the model fits in the given GPU memory budget.
pub fn cpu_fits_in_gpu(config: &ModelConfig, gpu_memory_mb: usize) -> bool {
    cpu_estimate_memory_mb(config) <= gpu_memory_mb as f64
}

/// Auto-configure GPU settings based on model size.
pub fn cpu_auto_configure_gpu(config: &ModelConfig, gpu: &mut GpuConfig) {
    let mem_mb = cpu_estimate_memory_mb(config);

    // Scale batch size down for larger models.
    if mem_mb > 8192.0 {
        gpu.preferred_batch_size = 1;
        gpu.max_batch_size = gpu.max_batch_size.min(4);
    } else if mem_mb > 4096.0 {
        gpu.preferred_batch_size = gpu.preferred_batch_size.min(4);
        gpu.max_batch_size = gpu.max_batch_size.min(16);
    }

    // Disable flash attention for very small models (overhead > benefit).
    if config.num_layers <= 4 {
        gpu.use_flash_attention = false;
    }

    // Use fp16 KV cache for quantized models to save memory.
    if config.quantization.is_some() {
        gpu.kv_cache_dtype = "f16".to_string();
    }
}

/// Build an [`InferenceConfig`] from model and GPU configs with sensible
/// sampling defaults.
pub fn cpu_create_inference_config(
    model: ModelConfig,
    gpu: GpuConfig,
) -> InferenceConfig {
    InferenceConfig {
        model,
        gpu,
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        max_tokens: 512,
    }
}

/// Serialise an [`InferenceConfig`] to a JSON string.
pub fn cpu_config_to_json(config: &InferenceConfig) -> String {
    let quant_json = match &config.model.quantization {
        Some(q) => format!(
            r#"{{
      "weight_bits": {},
      "activation_bits": {},
      "group_size": {},
      "method": "{}"
    }}"#,
            q.weight_bits, q.activation_bits, q.group_size, q.method,
        ),
        None => "null".to_string(),
    };

    format!(
        r#"{{
  "model": {{
    "architecture": "{}",
    "vocab_size": {},
    "hidden_size": {},
    "num_layers": {},
    "num_heads": {},
    "num_kv_heads": {},
    "head_dim": {},
    "intermediate_size": {},
    "max_seq_len": {},
    "rope_base": {},
    "norm_eps": {},
    "quantization": {}
  }},
  "gpu": {{
    "preferred_batch_size": {},
    "max_batch_size": {},
    "kv_cache_dtype": "{}",
    "use_flash_attention": {},
    "workgroup_size_hint": {}
  }},
  "temperature": {},
  "top_k": {},
  "top_p": {},
  "max_tokens": {}
}}"#,
        config.model.architecture,
        config.model.vocab_size,
        config.model.hidden_size,
        config.model.num_layers,
        config.model.num_heads,
        config.model.num_kv_heads,
        config.model.head_dim,
        config.model.intermediate_size,
        config.model.max_seq_len,
        config.model.rope_base,
        config.model.norm_eps,
        quant_json,
        config.gpu.preferred_batch_size,
        config.gpu.max_batch_size,
        config.gpu.kv_cache_dtype,
        config.gpu.use_flash_attention,
        config
            .gpu
            .workgroup_size_hint
            .map_or("null".to_string(), |v| v.to_string()),
        config.temperature,
        config.top_k,
        config.top_p,
        config.max_tokens,
    )
}

/// Format a human-readable model summary.
pub fn format_model_summary(
    config: &ModelConfig,
    meta: Option<&ModelMetadata>,
) -> String {
    let params = cpu_estimate_parameters(config);
    let mem = cpu_estimate_memory_mb(config);
    let mut s = format!(
        "Architecture: {}\n\
         Layers: {}, Heads: {}, Hidden: {}\n\
         Vocab: {}, Seq Len: {}\n\
         Est. Parameters: {:.2}B\n\
         Est. VRAM: {:.0} MB",
        config.architecture,
        config.num_layers,
        config.num_heads,
        config.hidden_size,
        config.vocab_size,
        config.max_seq_len,
        params,
        mem,
    );
    if let Some(q) = &config.quantization {
        s.push_str(&format!(
            "\nQuantization: {}b weights, {}b activations ({})",
            q.weight_bits, q.activation_bits, q.method
        ));
    }
    if let Some(m) = meta {
        s.push_str(&format!(
            "\nModel: {} v{} by {} [{}]",
            m.name, m.version, m.author, m.license
        ));
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_gpu() -> GpuConfig {
        GpuConfig {
            preferred_batch_size: 8,
            max_batch_size: 32,
            kv_cache_dtype: "f16".to_string(),
            use_flash_attention: true,
            workgroup_size_hint: None,
        }
    }

    // -- preset configs ---------------------------------------------------

    #[test]
    fn bitnet_2b_config_has_correct_hidden_size() {
        let c = cpu_bitnet_2b_config();
        assert_eq!(c.hidden_size, 2048);
    }

    #[test]
    fn bitnet_2b_config_has_correct_layers() {
        let c = cpu_bitnet_2b_config();
        assert_eq!(c.num_layers, 24);
    }

    #[test]
    fn bitnet_2b_config_is_quantized() {
        let c = cpu_bitnet_2b_config();
        assert!(c.quantization.is_some());
        assert_eq!(c.quantization.as_ref().unwrap().weight_bits, 2);
    }

    #[test]
    fn bitnet_3b_config_has_correct_hidden_size() {
        let c = cpu_bitnet_3b_config();
        assert_eq!(c.hidden_size, 3200);
    }

    #[test]
    fn llama_7b_config_has_correct_hidden_size() {
        let c = cpu_llama_7b_config();
        assert_eq!(c.hidden_size, 4096);
    }

    #[test]
    fn llama_7b_config_has_no_quantization() {
        let c = cpu_llama_7b_config();
        assert!(c.quantization.is_none());
    }

    #[test]
    fn bitnet_2b_architecture_is_bitnet() {
        let c = cpu_bitnet_2b_config();
        assert_eq!(c.architecture, ModelArchitecture::BitNetB158);
    }

    // -- validation -------------------------------------------------------

    #[test]
    fn validate_valid_config_passes() {
        let c = cpu_bitnet_2b_config();
        assert!(cpu_validate_config(&c).is_ok());
    }

    #[test]
    fn validate_llama_config_passes() {
        let c = cpu_llama_7b_config();
        assert!(cpu_validate_config(&c).is_ok());
    }

    #[test]
    fn validate_rejects_zero_hidden_size() {
        let mut c = cpu_bitnet_2b_config();
        c.hidden_size = 0;
        assert!(matches!(
            cpu_validate_config(&c),
            Err(ConfigError::InvalidConfig(_))
        ));
    }

    #[test]
    fn validate_rejects_zero_layers() {
        let mut c = cpu_bitnet_2b_config();
        c.num_layers = 0;
        assert!(matches!(
            cpu_validate_config(&c),
            Err(ConfigError::InvalidConfig(_))
        ));
    }

    #[test]
    fn validate_rejects_zero_heads() {
        let mut c = cpu_bitnet_2b_config();
        c.num_heads = 0;
        assert!(matches!(
            cpu_validate_config(&c),
            Err(ConfigError::InvalidConfig(_))
        ));
    }

    #[test]
    fn validate_rejects_zero_vocab() {
        let mut c = cpu_bitnet_2b_config();
        c.vocab_size = 0;
        assert!(matches!(
            cpu_validate_config(&c),
            Err(ConfigError::InvalidConfig(_))
        ));
    }

    #[test]
    fn validate_rejects_kv_heads_greater_than_heads() {
        let mut c = cpu_bitnet_2b_config();
        c.num_kv_heads = c.num_heads + 1;
        assert!(matches!(
            cpu_validate_config(&c),
            Err(ConfigError::InvalidConfig(_))
        ));
    }

    // -- parameter estimation ---------------------------------------------

    #[test]
    fn estimate_parameters_2b_reasonable() {
        let params = cpu_estimate_parameters(&cpu_bitnet_2b_config());
        assert!(params > 1.0 && params < 4.0, "got {params}B");
    }

    #[test]
    fn estimate_parameters_7b_reasonable() {
        let params = cpu_estimate_parameters(&cpu_llama_7b_config());
        assert!(params > 4.0 && params < 12.0, "got {params}B");
    }

    #[test]
    fn estimate_parameters_3b_between_2b_and_7b() {
        let p2 = cpu_estimate_parameters(&cpu_bitnet_2b_config());
        let p3 = cpu_estimate_parameters(&cpu_bitnet_3b_config());
        let p7 = cpu_estimate_parameters(&cpu_llama_7b_config());
        assert!(p2 < p3 && p3 < p7);
    }

    // -- memory estimation ------------------------------------------------

    #[test]
    fn memory_estimate_quantized_less_than_fp16() {
        let q = cpu_bitnet_2b_config();
        let mut fp = q.clone();
        fp.quantization = None;
        assert!(cpu_estimate_memory_mb(&q) < cpu_estimate_memory_mb(&fp));
    }

    #[test]
    fn memory_estimate_2b_reasonable() {
        let mem = cpu_estimate_memory_mb(&cpu_bitnet_2b_config());
        assert!(mem > 100.0 && mem < 8192.0, "got {mem} MB");
    }

    #[test]
    fn memory_estimate_7b_larger_than_2b() {
        let m2 = cpu_estimate_memory_mb(&cpu_bitnet_2b_config());
        let m7 = cpu_estimate_memory_mb(&cpu_llama_7b_config());
        assert!(m7 > m2);
    }

    // -- FLOPs estimation -------------------------------------------------

    #[test]
    fn flops_estimate_positive() {
        let f = cpu_estimate_flops_per_token(&cpu_bitnet_2b_config());
        assert!(f > 0.0);
    }

    #[test]
    fn flops_scales_with_layers() {
        let c1 = cpu_bitnet_2b_config();
        let mut c2 = c1.clone();
        c2.num_layers *= 2;
        let f1 = cpu_estimate_flops_per_token(&c1);
        let f2 = cpu_estimate_flops_per_token(&c2);
        let ratio = f2 / f1;
        assert!((ratio - 2.0).abs() < 0.01, "ratio = {ratio}");
    }

    #[test]
    fn flops_7b_greater_than_2b() {
        let f2 = cpu_estimate_flops_per_token(&cpu_bitnet_2b_config());
        let f7 = cpu_estimate_flops_per_token(&cpu_llama_7b_config());
        assert!(f7 > f2);
    }

    // -- fits in GPU ------------------------------------------------------

    #[test]
    fn bitnet_2b_fits_in_16gb() {
        assert!(cpu_fits_in_gpu(&cpu_bitnet_2b_config(), 16384));
    }

    #[test]
    fn large_model_does_not_fit_in_16gb() {
        let mut big = cpu_llama_7b_config();
        big.num_layers = 80;
        big.hidden_size = 8192;
        big.intermediate_size = 28672;
        big.num_heads = 64;
        big.num_kv_heads = 8;
        big.head_dim = 128;
        assert!(!cpu_fits_in_gpu(&big, 16384));
    }

    #[test]
    fn fits_in_gpu_with_exact_budget() {
        let mem = cpu_estimate_memory_mb(&cpu_bitnet_2b_config());
        assert!(cpu_fits_in_gpu(&cpu_bitnet_2b_config(), mem.ceil() as usize));
    }

    // -- auto configure GPU -----------------------------------------------

    #[test]
    fn auto_configure_reduces_batch_for_large_model() {
        let mut gpu = default_gpu();
        let mut big = cpu_llama_7b_config();
        big.num_layers = 80;
        big.hidden_size = 8192;
        big.intermediate_size = 28672;
        big.num_heads = 64;
        big.num_kv_heads = 8;
        big.head_dim = 128;
        cpu_auto_configure_gpu(&big, &mut gpu);
        assert!(gpu.preferred_batch_size <= 4);
    }

    #[test]
    fn auto_configure_disables_flash_attn_for_tiny_model() {
        let mut gpu = default_gpu();
        let mut tiny = cpu_bitnet_2b_config();
        tiny.num_layers = 2;
        cpu_auto_configure_gpu(&tiny, &mut gpu);
        assert!(!gpu.use_flash_attention);
    }

    #[test]
    fn auto_configure_sets_f16_kv_for_quantized() {
        let mut gpu = default_gpu();
        gpu.kv_cache_dtype = "f32".to_string();
        let c = cpu_bitnet_2b_config();
        cpu_auto_configure_gpu(&c, &mut gpu);
        assert_eq!(gpu.kv_cache_dtype, "f16");
    }

    // -- JSON serialization -----------------------------------------------

    #[test]
    fn json_contains_architecture() {
        let cfg = cpu_create_inference_config(
            cpu_bitnet_2b_config(),
            default_gpu(),
        );
        let json = cpu_config_to_json(&cfg);
        assert!(json.contains("BitNet b1.58"));
    }

    #[test]
    fn json_contains_temperature() {
        let cfg = cpu_create_inference_config(
            cpu_bitnet_2b_config(),
            default_gpu(),
        );
        let json = cpu_config_to_json(&cfg);
        assert!(json.contains("temperature"));
    }

    #[test]
    fn json_contains_quantization() {
        let cfg = cpu_create_inference_config(
            cpu_bitnet_2b_config(),
            default_gpu(),
        );
        let json = cpu_config_to_json(&cfg);
        assert!(json.contains("weight_bits"));
    }

    #[test]
    fn json_null_quantization_for_fp_model() {
        let cfg =
            cpu_create_inference_config(cpu_llama_7b_config(), default_gpu());
        let json = cpu_config_to_json(&cfg);
        assert!(json.contains("\"quantization\": null"));
    }

    #[test]
    fn json_contains_gpu_fields() {
        let cfg = cpu_create_inference_config(
            cpu_bitnet_2b_config(),
            default_gpu(),
        );
        let json = cpu_config_to_json(&cfg);
        assert!(json.contains("preferred_batch_size"));
        assert!(json.contains("max_batch_size"));
        assert!(json.contains("use_flash_attention"));
    }

    // -- inference config -------------------------------------------------

    #[test]
    fn inference_config_default_sampling() {
        let cfg = cpu_create_inference_config(
            cpu_bitnet_2b_config(),
            default_gpu(),
        );
        assert!((cfg.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(cfg.top_k, 40);
    }

    // -- edge cases -------------------------------------------------------

    #[test]
    fn single_layer_model_validates() {
        let mut c = cpu_bitnet_2b_config();
        c.num_layers = 1;
        assert!(cpu_validate_config(&c).is_ok());
    }

    #[test]
    fn single_head_model_validates() {
        let mut c = cpu_bitnet_2b_config();
        c.num_heads = 1;
        c.num_kv_heads = 1;
        assert!(cpu_validate_config(&c).is_ok());
    }

    #[test]
    fn single_layer_model_parameter_estimate_positive() {
        let mut c = cpu_bitnet_2b_config();
        c.num_layers = 1;
        assert!(cpu_estimate_parameters(&c) > 0.0);
    }

    // -- property tests ---------------------------------------------------

    #[test]
    fn more_layers_means_more_parameters() {
        let c1 = cpu_bitnet_2b_config();
        let mut c2 = c1.clone();
        c2.num_layers += 8;
        assert!(cpu_estimate_parameters(&c2) > cpu_estimate_parameters(&c1));
    }

    #[test]
    fn quantized_uses_less_memory() {
        let q = cpu_bitnet_2b_config();
        let mut fp = q.clone();
        fp.quantization = None;
        assert!(cpu_estimate_memory_mb(&q) < cpu_estimate_memory_mb(&fp));
    }

    // -- ConfigManager ----------------------------------------------------

    #[test]
    fn config_manager_creation() {
        let mgr = create_config_manager(default_gpu());
        assert!(mgr.model_config.is_none());
        assert!(mgr.metadata.is_none());
    }

    #[test]
    fn config_manager_with_model() {
        let mut mgr = create_config_manager(default_gpu());
        mgr.model_config = Some(cpu_bitnet_2b_config());
        assert!(mgr.model_config.is_some());
    }

    // -- format_model_summary --------------------------------------------

    #[test]
    fn summary_contains_architecture() {
        let s = format_model_summary(&cpu_bitnet_2b_config(), None);
        assert!(s.contains("BitNet b1.58"));
    }

    #[test]
    fn summary_contains_metadata_when_provided() {
        let meta = ModelMetadata {
            name: "test-model".to_string(),
            version: "1.0".to_string(),
            author: "tester".to_string(),
            license: "MIT".to_string(),
            parameters_b: 2.0,
            file_size_gb: 0.5,
        };
        let s = format_model_summary(&cpu_bitnet_2b_config(), Some(&meta));
        assert!(s.contains("test-model"));
        assert!(s.contains("tester"));
    }

    #[test]
    fn summary_shows_quantization_info() {
        let s = format_model_summary(&cpu_bitnet_2b_config(), None);
        assert!(s.contains("2b weights"));
    }

    // -- Display / Error --------------------------------------------------

    #[test]
    fn config_error_display() {
        let e = ConfigError::InvalidConfig("bad".to_string());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn missing_field_error_display() {
        let e = ConfigError::MissingField("hidden_size".to_string());
        assert!(e.to_string().contains("hidden_size"));
    }

    #[test]
    fn architecture_display() {
        assert_eq!(
            ModelArchitecture::BitNetB158.to_string(),
            "BitNet b1.58"
        );
        assert_eq!(
            ModelArchitecture::Custom("foo".into()).to_string(),
            "Custom(foo)"
        );
    }
}
