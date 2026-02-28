//! Snapshot tests for stable bitnet-models configuration defaults.
//! These test key configuration constants that should never silently change.

use bitnet_models::{
    loader::LoadConfig,
    production_loader::{DeviceStrategy, ProductionLoadConfig},
};

#[test]
fn load_config_default_uses_mmap() {
    let cfg = LoadConfig::default();
    insta::assert_snapshot!(format!(
        "use_mmap={} validate_checksums={}",
        cfg.use_mmap, cfg.validate_checksums
    ));
}

#[test]
fn production_load_config_default_strict_validation() {
    let cfg = ProductionLoadConfig::default();
    insta::assert_snapshot!(format!(
        "strict_validation={} validate_tensor_alignment={}",
        cfg.strict_validation, cfg.validate_tensor_alignment
    ));
}

#[test]
fn production_load_config_default_max_model_size() {
    let cfg = ProductionLoadConfig::default();
    let size_gb = cfg.max_model_size_bytes.unwrap_or(0) / (1024 * 1024 * 1024);
    insta::assert_snapshot!(format!("max_size_gb={size_gb}"));
}

#[test]
fn device_strategy_cpu_only_debug() {
    insta::assert_snapshot!(format!("{:?}", DeviceStrategy::CpuOnly));
}

#[test]
fn device_strategy_hybrid_debug() {
    insta::assert_snapshot!(format!(
        "{:?}",
        DeviceStrategy::Hybrid { cpu_layers: 4, gpu_layers: 8 }
    ));
}

// -- Wave 3: model metadata & validation snapshots ---------------------------

use bitnet_models::gguf_parity::GgufMetadata;

#[test]
fn gguf_metadata_populated_debug() {
    let meta = GgufMetadata {
        vocab_size: 32000,
        hidden_size: 2048,
        num_layers: 24,
        num_heads: 16,
        tokenizer_type: Some("BPE".to_string()),
        model_type: "llama".to_string(),
    };
    insta::assert_debug_snapshot!("gguf_metadata_populated", meta);
}

#[test]
fn gguf_metadata_minimal_debug() {
    let meta = GgufMetadata {
        vocab_size: 0,
        hidden_size: 0,
        num_layers: 0,
        num_heads: 0,
        tokenizer_type: None,
        model_type: String::new(),
    };
    insta::assert_debug_snapshot!("gguf_metadata_minimal", meta);
}

#[test]
fn device_strategy_gpu_only_debug() {
    insta::assert_snapshot!("device_strategy_gpu_only", format!("{:?}", DeviceStrategy::GpuOnly));
}

#[test]
fn production_load_config_defaults_debug() {
    let cfg = ProductionLoadConfig::default();
    insta::assert_debug_snapshot!("production_load_config_defaults", cfg);
}

#[test]
fn validation_result_valid_debug() {
    use bitnet_models::ValidationResult;
    let result = ValidationResult {
        passed: true,
        errors: vec![],
        warnings: vec!["Minor: unusual embedding size".to_string()],
        alignment_issues: vec![],
        recommendations: vec![],
    };
    insta::assert_debug_snapshot!("validation_result_valid", result);
}

#[test]
fn validation_result_invalid_debug() {
    use bitnet_models::ValidationResult;
    let result = ValidationResult {
        passed: false,
        errors: vec!["Missing required tensor: token_embd.weight".to_string()],
        warnings: vec![],
        alignment_issues: vec![],
        recommendations: vec!["Consider re-exporting with F16 LayerNorm".to_string()],
    };
    insta::assert_debug_snapshot!("validation_result_invalid", result);
}
