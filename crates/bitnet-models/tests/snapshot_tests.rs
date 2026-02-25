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
