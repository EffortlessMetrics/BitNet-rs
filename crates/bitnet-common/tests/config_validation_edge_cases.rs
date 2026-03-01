//! Edge-case tests for BitNetConfig: validation rules, merge semantics,
//! sub-config defaults, serde roundtrips, and multi-SLM model presets.

use bitnet_common::config::{
    ActivationType, BitNetConfig, InferenceConfig, ModelConfig, ModelFormat, NormType,
    PerformanceConfig, QuantizationConfig, RopeScaling, TokenizerConfig,
};

// ---------------------------------------------------------------------------
// BitNetConfig default validation passes
// ---------------------------------------------------------------------------

#[test]
fn default_config_validates() {
    let cfg = BitNetConfig::default();
    assert!(cfg.validate().is_ok());
}

// ---------------------------------------------------------------------------
// ModelConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn model_config_defaults() {
    let m = ModelConfig::default();
    assert_eq!(m.vocab_size, 32000);
    assert_eq!(m.hidden_size, 4096);
    assert_eq!(m.num_layers, 32);
    assert_eq!(m.num_heads, 32);
    assert_eq!(m.num_key_value_heads, 0);
    assert_eq!(m.intermediate_size, 11008);
    assert_eq!(m.max_position_embeddings, 2048);
    assert_eq!(m.format, ModelFormat::Gguf);
    assert_eq!(m.norm_type, NormType::LayerNorm);
    assert_eq!(m.activation_type, ActivationType::Silu);
    assert!(m.path.is_none());
    assert!(m.rope_theta.is_none());
    assert!(m.rope_scaling.is_none());
    assert!(m.rms_norm_eps.is_none());
}

// ---------------------------------------------------------------------------
// InferenceConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn inference_config_defaults() {
    let i = InferenceConfig::default();
    assert_eq!(i.max_length, 2048);
    assert_eq!(i.max_new_tokens, 512);
    assert!((i.temperature - 1.0).abs() < f32::EPSILON);
    assert_eq!(i.top_k, Some(50));
    assert_eq!(i.top_p, Some(0.9));
    assert!((i.repetition_penalty - 1.1).abs() < f32::EPSILON);
    assert!(i.seed.is_none());
    assert!(i.add_bos);
    assert!(!i.append_eos);
    assert!(i.mask_pad);
}

// ---------------------------------------------------------------------------
// QuantizationConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn quantization_config_defaults() {
    let q = QuantizationConfig::default();
    assert_eq!(q.block_size, 64);
    assert!((q.precision - 1e-4).abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// PerformanceConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn performance_config_defaults() {
    let p = PerformanceConfig::default();
    assert!(p.num_threads.is_none());
    assert!(!p.use_gpu);
    assert_eq!(p.batch_size, 1);
    assert!(p.memory_limit.is_none());
}

// ---------------------------------------------------------------------------
// Validation — model errors
// ---------------------------------------------------------------------------

#[test]
fn validate_rejects_zero_vocab_size() {
    let mut cfg = BitNetConfig::default();
    cfg.model.vocab_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_zero_hidden_size() {
    let mut cfg = BitNetConfig::default();
    cfg.model.hidden_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_zero_num_layers() {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_layers = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_zero_num_heads() {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_heads = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_hidden_not_divisible_by_heads() {
    let mut cfg = BitNetConfig::default();
    cfg.model.hidden_size = 4097; // not divisible by 32
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_kv_heads_greater_than_heads() {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_key_value_heads = 64; // > 32
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_heads_not_divisible_by_kv_heads() {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_key_value_heads = 3; // 32 not divisible by 3
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_accepts_valid_gqa_config() {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_key_value_heads = 8; // 32 / 8 = 4 groups
    assert!(cfg.validate().is_ok());
}

#[test]
fn validate_rejects_zero_intermediate_size() {
    let mut cfg = BitNetConfig::default();
    cfg.model.intermediate_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_zero_max_position_embeddings() {
    let mut cfg = BitNetConfig::default();
    cfg.model.max_position_embeddings = 0;
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Validation — inference errors
// ---------------------------------------------------------------------------

#[test]
fn validate_rejects_zero_max_length() {
    let mut cfg = BitNetConfig::default();
    cfg.inference.max_length = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_zero_max_new_tokens() {
    let mut cfg = BitNetConfig::default();
    cfg.inference.max_new_tokens = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_zero_temperature() {
    let mut cfg = BitNetConfig::default();
    cfg.inference.temperature = 0.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_negative_temperature() {
    let mut cfg = BitNetConfig::default();
    cfg.inference.temperature = -1.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_zero_top_k() {
    let mut cfg = BitNetConfig::default();
    cfg.inference.top_k = Some(0);
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_accepts_none_top_k() {
    let mut cfg = BitNetConfig::default();
    cfg.inference.top_k = None;
    assert!(cfg.validate().is_ok());
}

#[test]
fn validate_rejects_zero_top_p() {
    let mut cfg = BitNetConfig::default();
    cfg.inference.top_p = Some(0.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_top_p_above_one() {
    let mut cfg = BitNetConfig::default();
    cfg.inference.top_p = Some(1.1);
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_accepts_top_p_exactly_one() {
    let mut cfg = BitNetConfig::default();
    cfg.inference.top_p = Some(1.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn validate_accepts_none_top_p() {
    let mut cfg = BitNetConfig::default();
    cfg.inference.top_p = None;
    assert!(cfg.validate().is_ok());
}

#[test]
fn validate_rejects_zero_repetition_penalty() {
    let mut cfg = BitNetConfig::default();
    cfg.inference.repetition_penalty = 0.0;
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Validation — quantization errors
// ---------------------------------------------------------------------------

#[test]
fn validate_rejects_zero_block_size() {
    let mut cfg = BitNetConfig::default();
    cfg.quantization.block_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_non_power_of_two_block_size() {
    let mut cfg = BitNetConfig::default();
    cfg.quantization.block_size = 65;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_zero_precision() {
    let mut cfg = BitNetConfig::default();
    cfg.quantization.precision = 0.0;
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Validation — performance errors
// ---------------------------------------------------------------------------

#[test]
fn validate_rejects_zero_num_threads() {
    let mut cfg = BitNetConfig::default();
    cfg.performance.num_threads = Some(0);
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_zero_batch_size() {
    let mut cfg = BitNetConfig::default();
    cfg.performance.batch_size = 0;
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Merge semantics
// ---------------------------------------------------------------------------

#[test]
fn merge_overrides_non_default_model_values() {
    let mut base = BitNetConfig::default();
    let mut other = BitNetConfig::default();
    other.model.hidden_size = 5120;
    other.model.num_heads = 40;
    other.model.num_layers = 40;
    base.merge_with(other);
    assert_eq!(base.model.hidden_size, 5120);
    assert_eq!(base.model.num_heads, 40);
    assert_eq!(base.model.num_layers, 40);
}

#[test]
fn merge_preserves_default_values_in_other() {
    let mut base = BitNetConfig::default();
    base.model.hidden_size = 5120;
    let other = BitNetConfig::default(); // all defaults
    base.merge_with(other);
    // other has default hidden_size (4096), so base's 5120 should be preserved
    assert_eq!(base.model.hidden_size, 5120);
}

#[test]
fn merge_overrides_path() {
    let mut base = BitNetConfig::default();
    let mut other = BitNetConfig::default();
    other.model.path = Some("/tmp/model.gguf".into());
    base.merge_with(other);
    assert_eq!(base.model.path.unwrap().to_str().unwrap(), "/tmp/model.gguf");
}

#[test]
fn merge_overrides_inference_settings() {
    let mut base = BitNetConfig::default();
    let mut other = BitNetConfig::default();
    other.inference.max_new_tokens = 1024;
    other.inference.temperature = 0.7;
    base.merge_with(other);
    assert_eq!(base.inference.max_new_tokens, 1024);
    assert!((base.inference.temperature - 0.7).abs() < f32::EPSILON);
}

#[test]
fn merge_overrides_seed() {
    let mut base = BitNetConfig::default();
    let mut other = BitNetConfig::default();
    other.inference.seed = Some(42);
    base.merge_with(other);
    assert_eq!(base.inference.seed, Some(42));
}

#[test]
fn merge_overrides_performance() {
    let mut base = BitNetConfig::default();
    let mut other = BitNetConfig::default();
    other.performance.use_gpu = true;
    other.performance.batch_size = 8;
    other.performance.num_threads = Some(4);
    base.merge_with(other);
    assert!(base.performance.use_gpu);
    assert_eq!(base.performance.batch_size, 8);
    assert_eq!(base.performance.num_threads, Some(4));
}

// ---------------------------------------------------------------------------
// Serde roundtrip
// ---------------------------------------------------------------------------

#[test]
fn bitnet_config_json_roundtrip() {
    let cfg = BitNetConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: BitNetConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg2.model.vocab_size, cfg.model.vocab_size);
    assert_eq!(cfg2.inference.max_new_tokens, cfg.inference.max_new_tokens);
    assert_eq!(cfg2.quantization.block_size, cfg.quantization.block_size);
}

#[test]
fn model_config_json_roundtrip() {
    let mut m = ModelConfig::default();
    m.hidden_size = 5120;
    m.norm_type = NormType::RmsNorm;
    m.activation_type = ActivationType::Gelu;
    let json = serde_json::to_string(&m).unwrap();
    let m2: ModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(m2.hidden_size, 5120);
    assert_eq!(m2.norm_type, NormType::RmsNorm);
    assert_eq!(m2.activation_type, ActivationType::Gelu);
}

#[test]
fn inference_config_json_roundtrip() {
    let mut i = InferenceConfig::default();
    i.seed = Some(42);
    let json = serde_json::to_string(&i).unwrap();
    let i2: InferenceConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(i2.seed, Some(42));
}

// ---------------------------------------------------------------------------
// NormType / ActivationType / ModelFormat enum coverage
// ---------------------------------------------------------------------------

#[test]
fn norm_type_default_is_layernorm() {
    assert_eq!(NormType::default(), NormType::LayerNorm);
}

#[test]
fn activation_type_default_is_silu() {
    assert_eq!(ActivationType::default(), ActivationType::Silu);
}

#[test]
fn model_format_default_is_gguf() {
    assert_eq!(ModelFormat::default(), ModelFormat::Gguf);
}

#[test]
fn norm_type_serde_roundtrip() {
    for norm in [NormType::LayerNorm, NormType::RmsNorm] {
        let json = serde_json::to_string(&norm).unwrap();
        let norm2: NormType = serde_json::from_str(&json).unwrap();
        assert_eq!(norm, norm2);
    }
}

#[test]
fn activation_type_serde_roundtrip() {
    for act in [ActivationType::Silu, ActivationType::Relu2, ActivationType::Gelu] {
        let json = serde_json::to_string(&act).unwrap();
        let act2: ActivationType = serde_json::from_str(&json).unwrap();
        assert_eq!(act, act2);
    }
}

#[test]
fn model_format_serde_roundtrip() {
    for fmt in [ModelFormat::Gguf, ModelFormat::SafeTensors, ModelFormat::HuggingFace] {
        let json = serde_json::to_string(&fmt).unwrap();
        let fmt2: ModelFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(fmt, fmt2);
    }
}

// ---------------------------------------------------------------------------
// RopeScaling
// ---------------------------------------------------------------------------

#[test]
fn rope_scaling_serde_roundtrip() {
    let rs = RopeScaling { scaling_type: "linear".to_string(), factor: 4.0 };
    let json = serde_json::to_string(&rs).unwrap();
    let rs2: RopeScaling = serde_json::from_str(&json).unwrap();
    assert_eq!(rs2.scaling_type, "linear");
    assert!((rs2.factor - 4.0).abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// TokenizerConfig (from config module)
// ---------------------------------------------------------------------------

#[test]
fn tokenizer_config_defaults() {
    let tc = TokenizerConfig::default();
    assert_eq!(tc.bos_id, None);
    assert_eq!(tc.eos_id, None);
    assert_eq!(tc.unk_id, None);
    assert_eq!(tc.pad_id, None);
}

#[test]
fn tokenizer_config_serde_roundtrip() {
    let tc = TokenizerConfig { bos_id: Some(1), eos_id: Some(2), unk_id: Some(0), pad_id: None };
    let json = serde_json::to_string(&tc).unwrap();
    let tc2: TokenizerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(tc2.bos_id, Some(1));
    assert_eq!(tc2.eos_id, Some(2));
}

// ---------------------------------------------------------------------------
// Multi-SLM model presets — validate typical configurations
// ---------------------------------------------------------------------------

#[test]
fn phi4_preset_validates() {
    let mut cfg = BitNetConfig::default();
    cfg.model.hidden_size = 5120;
    cfg.model.num_heads = 40;
    cfg.model.num_key_value_heads = 10;
    cfg.model.num_layers = 40;
    cfg.model.intermediate_size = 17920;
    cfg.model.max_position_embeddings = 16384;
    cfg.model.vocab_size = 100352;
    cfg.model.norm_type = NormType::RmsNorm;
    cfg.model.activation_type = ActivationType::Silu;
    assert!(cfg.validate().is_ok());
}

#[test]
fn llama3_8b_preset_validates() {
    let mut cfg = BitNetConfig::default();
    cfg.model.hidden_size = 4096;
    cfg.model.num_heads = 32;
    cfg.model.num_key_value_heads = 8;
    cfg.model.num_layers = 32;
    cfg.model.intermediate_size = 14336;
    cfg.model.max_position_embeddings = 8192;
    cfg.model.vocab_size = 128256;
    cfg.model.norm_type = NormType::RmsNorm;
    assert!(cfg.validate().is_ok());
}

#[test]
fn gemma_2b_preset_validates() {
    let mut cfg = BitNetConfig::default();
    cfg.model.hidden_size = 2048;
    cfg.model.num_heads = 8;
    cfg.model.num_key_value_heads = 1;
    cfg.model.num_layers = 18;
    cfg.model.intermediate_size = 16384;
    cfg.model.max_position_embeddings = 8192;
    cfg.model.vocab_size = 256000;
    cfg.model.norm_type = NormType::RmsNorm;
    cfg.model.activation_type = ActivationType::Gelu;
    assert!(cfg.validate().is_ok());
}

#[test]
fn qwen25_7b_preset_validates() {
    let mut cfg = BitNetConfig::default();
    cfg.model.hidden_size = 3584;
    cfg.model.num_heads = 28;
    cfg.model.num_key_value_heads = 4;
    cfg.model.num_layers = 28;
    cfg.model.intermediate_size = 18944;
    cfg.model.max_position_embeddings = 32768;
    cfg.model.vocab_size = 152064;
    assert!(cfg.validate().is_ok());
}

#[test]
fn mistral_7b_preset_validates() {
    let mut cfg = BitNetConfig::default();
    cfg.model.hidden_size = 4096;
    cfg.model.num_heads = 32;
    cfg.model.num_key_value_heads = 8;
    cfg.model.num_layers = 32;
    cfg.model.intermediate_size = 14336;
    cfg.model.max_position_embeddings = 32768;
    cfg.model.vocab_size = 32000;
    assert!(cfg.validate().is_ok());
}

// ---------------------------------------------------------------------------
// File loading — error cases
// ---------------------------------------------------------------------------

#[test]
fn from_file_nonexistent_path_errors() {
    let result = BitNetConfig::from_file("/nonexistent/path/config.toml");
    assert!(result.is_err());
}

#[test]
fn from_file_unsupported_extension_errors() {
    // Create a temp file with unsupported extension
    let dir = std::env::temp_dir().join("bitnet_test_config");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("config.yaml");
    std::fs::write(&path, "key: value").unwrap();
    let result = BitNetConfig::from_file(&path);
    assert!(result.is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn from_file_invalid_json_errors() {
    let dir = std::env::temp_dir().join("bitnet_test_config");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("invalid.json");
    std::fs::write(&path, "not valid json{{{").unwrap();
    let result = BitNetConfig::from_file(&path);
    assert!(result.is_err());
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Clone semantics
// ---------------------------------------------------------------------------

#[test]
fn bitnet_config_clone() {
    let mut cfg = BitNetConfig::default();
    cfg.model.hidden_size = 5120;
    cfg.inference.seed = Some(42);
    let cfg2 = cfg.clone();
    assert_eq!(cfg2.model.hidden_size, 5120);
    assert_eq!(cfg2.inference.seed, Some(42));
}
