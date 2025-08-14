//! Configuration tests

use super::*;
use std::env;
use std::io::Write;
use std::sync::Mutex;
use tempfile::NamedTempFile;

// Mutex to ensure environment variable tests don't run concurrently
static ENV_TEST_MUTEX: Mutex<()> = Mutex::new(());

// Helper function to safely acquire the mutex
fn acquire_env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_TEST_MUTEX.lock().unwrap_or_else(|poisoned| {
        // If the mutex is poisoned, clear it and continue
        poisoned.into_inner()
    })
}

#[test]
fn test_default_config() {
    let config = BitNetConfig::default();
    assert!(config.validate().is_ok());
    assert_eq!(config.model.vocab_size, 32000);
    assert_eq!(config.inference.temperature, 1.0);
    assert_eq!(config.quantization.quantization_type, QuantizationType::I2S);
    assert!(!config.performance.use_gpu);
}

#[test]
fn test_config_builder() {
    let config = BitNetConfig::builder()
        .vocab_size(50000)
        .temperature(0.8)
        .use_gpu(true)
        .quantization_type(QuantizationType::TL1)
        .build()
        .unwrap();

    assert_eq!(config.model.vocab_size, 50000);
    assert_eq!(config.inference.temperature, 0.8);
    assert!(config.performance.use_gpu);
    assert_eq!(config.quantization.quantization_type, QuantizationType::TL1);
}

#[test]
fn test_config_validation() {
    // Test invalid vocab_size
    let mut config = BitNetConfig::default();
    config.model.vocab_size = 0;
    assert!(config.validate().is_err());

    // Test invalid temperature
    config = BitNetConfig::default();
    config.inference.temperature = 0.0;
    assert!(config.validate().is_err());

    // Test invalid top_p
    config = BitNetConfig::default();
    config.inference.top_p = Some(1.5);
    assert!(config.validate().is_err());

    // Test invalid hidden_size/num_heads ratio
    config = BitNetConfig::default();
    config.model.hidden_size = 100;
    config.model.num_heads = 7;
    assert!(config.validate().is_err());
}

#[test]
fn test_toml_config_loading() {
    // Clean up any existing env vars first
    let env_vars = [
        "BITNET_VOCAB_SIZE",
        "BITNET_TEMPERATURE",
        "BITNET_USE_GPU",
        "BITNET_QUANTIZATION_TYPE",
        "BITNET_MEMORY_LIMIT",
        "BITNET_MODEL_FORMAT",
    ];
    for var in &env_vars {
        env::remove_var(var);
    }

    let toml_content = r#"
[model]
vocab_size = 50000
hidden_size = 2048
num_layers = 24

[inference]
temperature = 0.8
max_new_tokens = 256

[quantization]
quantization_type = "TL1"
block_size = 128

[performance]
use_gpu = true
batch_size = 4
"#;

    let mut temp_file = NamedTempFile::with_suffix(".toml").unwrap();
    temp_file.write_all(toml_content.as_bytes()).unwrap();

    let config = BitNetConfig::from_file(temp_file.path()).unwrap();
    assert_eq!(config.model.vocab_size, 50000);
    assert_eq!(config.model.hidden_size, 2048);
    assert_eq!(config.model.num_layers, 24);
    assert_eq!(config.inference.temperature, 0.8);
    assert_eq!(config.inference.max_new_tokens, 256);
    assert_eq!(config.quantization.quantization_type, QuantizationType::TL1);
    assert_eq!(config.quantization.block_size, 128);
    assert!(config.performance.use_gpu);
    assert_eq!(config.performance.batch_size, 4);
}

#[test]
fn test_json_config_loading() {
    // Clean up any existing env vars first
    let env_vars = [
        "BITNET_VOCAB_SIZE",
        "BITNET_TEMPERATURE",
        "BITNET_USE_GPU",
        "BITNET_QUANTIZATION_TYPE",
        "BITNET_MEMORY_LIMIT",
        "BITNET_MODEL_FORMAT",
    ];
    for var in &env_vars {
        env::remove_var(var);
    }

    let json_content = r#"
{
    "model": {
        "vocab_size": 40000,
        "hidden_size": 3072,
        "format": "SafeTensors"
    },
    "inference": {
        "temperature": 0.9,
        "top_k": 40
    },
    "quantization": {
        "quantization_type": "TL2"
    },
    "performance": {
        "num_threads": 8
    }
}
"#;

    let mut temp_file = NamedTempFile::with_suffix(".json").unwrap();
    temp_file.write_all(json_content.as_bytes()).unwrap();

    let config = BitNetConfig::from_file(temp_file.path()).unwrap();
    assert_eq!(config.model.vocab_size, 40000);
    assert_eq!(config.model.hidden_size, 3072);
    assert!(matches!(config.model.format, ModelFormat::SafeTensors));
    assert_eq!(config.inference.temperature, 0.9);
    assert_eq!(config.inference.top_k, Some(40));
    assert_eq!(config.quantization.quantization_type, QuantizationType::TL2);
    assert_eq!(config.performance.num_threads, Some(8));
}

#[test]
fn test_env_overrides() {
    let _lock = acquire_env_lock();

    // Clean up any existing env vars first
    let env_vars = [
        "BITNET_VOCAB_SIZE",
        "BITNET_TEMPERATURE",
        "BITNET_USE_GPU",
        "BITNET_QUANTIZATION_TYPE",
        "BITNET_MEMORY_LIMIT",
        "BITNET_MODEL_FORMAT",
    ];
    for var in &env_vars {
        env::remove_var(var);
    }

    // Set environment variables
    env::set_var("BITNET_VOCAB_SIZE", "60000");
    env::set_var("BITNET_TEMPERATURE", "0.7");
    env::set_var("BITNET_USE_GPU", "true");
    env::set_var("BITNET_QUANTIZATION_TYPE", "TL2");
    env::set_var("BITNET_MEMORY_LIMIT", "2GB");

    let config = BitNetConfig::from_env().unwrap();
    assert_eq!(config.model.vocab_size, 60000);
    assert_eq!(config.inference.temperature, 0.7);
    assert!(config.performance.use_gpu);
    assert_eq!(config.quantization.quantization_type, QuantizationType::TL2);
    assert_eq!(
        config.performance.memory_limit,
        Some(2 * 1024 * 1024 * 1024)
    );

    // Clean up
    for var in &env_vars {
        env::remove_var(var);
    }
}

#[test]
fn test_config_merging() {
    let mut base_config = BitNetConfig::default();
    base_config.model.vocab_size = 30000;
    base_config.inference.temperature = 0.8;

    let override_config = BitNetConfig::builder()
        .vocab_size(50000)
        .use_gpu(true)
        .build()
        .unwrap();

    base_config.merge_with(override_config);

    assert_eq!(base_config.model.vocab_size, 50000); // Overridden
    assert_eq!(base_config.inference.temperature, 0.8); // Preserved
    assert!(base_config.performance.use_gpu); // New value
}

#[test]
fn test_config_loader_precedence() {
    let _lock = acquire_env_lock();

    // Clean up any existing env vars first
    let env_vars = [
        "BITNET_VOCAB_SIZE",
        "BITNET_TEMPERATURE",
        "BITNET_USE_GPU",
        "BITNET_QUANTIZATION_TYPE",
        "BITNET_MEMORY_LIMIT",
        "BITNET_MODEL_FORMAT",
    ];
    for var in &env_vars {
        env::remove_var(var);
    }

    // Create a config file
    let toml_content = r#"
[model]
vocab_size = 40000

[inference]
temperature = 0.9
"#;

    let mut temp_file = NamedTempFile::with_suffix(".toml").unwrap();
    temp_file.write_all(toml_content.as_bytes()).unwrap();

    // Set environment variable that should override file
    env::set_var("BITNET_VOCAB_SIZE", "50000");

    let config = ConfigLoader::load_with_precedence(Some(temp_file.path())).unwrap();

    // Environment should override file
    assert_eq!(config.model.vocab_size, 50000);
    // File should override default
    assert_eq!(config.inference.temperature, 0.9);

    // Clean up
    for var in &env_vars {
        env::remove_var(var);
    }
}

#[test]
fn test_memory_limit_parsing() {
    let _lock = acquire_env_lock();

    // Clean up any existing env vars first
    let env_vars = [
        "BITNET_VOCAB_SIZE",
        "BITNET_TEMPERATURE",
        "BITNET_USE_GPU",
        "BITNET_QUANTIZATION_TYPE",
        "BITNET_MEMORY_LIMIT",
        "BITNET_MODEL_FORMAT",
    ];
    for var in &env_vars {
        env::remove_var(var);
    }

    env::set_var("BITNET_MEMORY_LIMIT", "1GB");
    let mut config = BitNetConfig::default();
    config.apply_env_overrides().unwrap();
    assert_eq!(config.performance.memory_limit, Some(1024 * 1024 * 1024));

    env::set_var("BITNET_MEMORY_LIMIT", "512MB");
    config.apply_env_overrides().unwrap();
    assert_eq!(config.performance.memory_limit, Some(512 * 1024 * 1024));

    env::set_var("BITNET_MEMORY_LIMIT", "none");
    config.apply_env_overrides().unwrap();
    assert_eq!(config.performance.memory_limit, None);

    for var in &env_vars {
        env::remove_var(var);
    }
}

#[test]
fn test_invalid_env_values() {
    let _lock = acquire_env_lock();

    // Clean up any existing env vars first
    let env_vars = [
        "BITNET_VOCAB_SIZE",
        "BITNET_TEMPERATURE",
        "BITNET_USE_GPU",
        "BITNET_QUANTIZATION_TYPE",
        "BITNET_MEMORY_LIMIT",
        "BITNET_MODEL_FORMAT",
    ];
    for var in &env_vars {
        env::remove_var(var);
    }

    // Test invalid vocab size
    env::set_var("BITNET_VOCAB_SIZE", "invalid");
    let mut config = BitNetConfig::default();
    assert!(config.apply_env_overrides().is_err());
    env::remove_var("BITNET_VOCAB_SIZE");

    // Test invalid model format
    env::set_var("BITNET_MODEL_FORMAT", "invalid");
    config = BitNetConfig::default();
    assert!(config.apply_env_overrides().is_err());
    env::remove_var("BITNET_MODEL_FORMAT");

    // Test invalid use_gpu value
    env::set_var("BITNET_USE_GPU", "maybe");
    config = BitNetConfig::default();
    assert!(config.apply_env_overrides().is_err());
    env::remove_var("BITNET_USE_GPU");

    // Clean up
    for var in &env_vars {
        env::remove_var(var);
    }
}
