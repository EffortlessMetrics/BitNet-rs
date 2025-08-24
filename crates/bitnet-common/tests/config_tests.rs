#![cfg(feature = "integration-tests")]
//! Comprehensive configuration tests for bitnet-common

use bitnet_common::*;
use proptest::prelude::*;
use std::env;
use std::fs;
use std::io::Write;
use std::sync::Mutex;
use tempfile::{NamedTempFile, TempDir};

// Mutex to ensure environment variable tests don't run concurrently
static ENV_TEST_MUTEX: Mutex<()> = Mutex::new(());

fn acquire_env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_TEST_MUTEX.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn test_model_format_variants() {
    assert_ne!(ModelFormat::Gguf, ModelFormat::SafeTensors);
    assert_ne!(ModelFormat::Gguf, ModelFormat::HuggingFace);
    assert_ne!(ModelFormat::SafeTensors, ModelFormat::HuggingFace);
}

#[test]
fn test_model_format_default() {
    assert_eq!(ModelFormat::default(), ModelFormat::Gguf);
}

#[test]
fn test_model_format_serialization() {
    let formats = vec![ModelFormat::Gguf, ModelFormat::SafeTensors, ModelFormat::HuggingFace];

    for format in formats {
        let json = serde_json::to_string(&format).unwrap();
        let deserialized: ModelFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(format, deserialized);
    }
}

#[test]
fn test_config_defaults() {
    let config = BitNetConfig::default();

    // Model defaults
    assert_eq!(config.model.path, None);
    assert_eq!(config.model.format, ModelFormat::Gguf);
    assert_eq!(config.model.vocab_size, 32000);
    assert_eq!(config.model.hidden_size, 4096);
    assert_eq!(config.model.num_layers, 32);
    assert_eq!(config.model.num_heads, 32);
    assert_eq!(config.model.intermediate_size, 11008);
    assert_eq!(config.model.max_position_embeddings, 2048);

    // Inference defaults
    assert_eq!(config.inference.max_length, 2048);
    assert_eq!(config.inference.max_new_tokens, 512);
    assert_eq!(config.inference.temperature, 1.0);
    assert_eq!(config.inference.top_k, Some(50));
    assert_eq!(config.inference.top_p, Some(0.9));
    assert_eq!(config.inference.repetition_penalty, 1.1);
    assert_eq!(config.inference.seed, None);

    // Quantization defaults
    assert_eq!(config.quantization.quantization_type, QuantizationType::I2S);
    assert_eq!(config.quantization.block_size, 64);
    assert_eq!(config.quantization.precision, 1e-4);

    // Performance defaults
    assert_eq!(config.performance.num_threads, None);
    assert!(!config.performance.use_gpu);
    assert_eq!(config.performance.batch_size, 1);
    assert_eq!(config.performance.memory_limit, None);
}

#[test]
fn test_config_validation_success() {
    let config = BitNetConfig::default();
    assert!(config.validate().is_ok());

    // Test valid custom config
    let config = BitNetConfig::builder()
        .vocab_size(50000)
        .hidden_size(8192)
        .num_heads(64)
        .temperature(0.8)
        .top_p(Some(0.95))
        .build()
        .unwrap();

    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validation_failures() {
    // Test zero vocab_size
    let mut config = BitNetConfig::default();
    config.model.vocab_size = 0;
    assert!(config.validate().is_err());

    // Test zero hidden_size
    config = BitNetConfig::default();
    config.model.hidden_size = 0;
    assert!(config.validate().is_err());

    // Test zero num_layers
    config = BitNetConfig::default();
    config.model.num_layers = 0;
    assert!(config.validate().is_err());

    // Test zero num_heads
    config = BitNetConfig::default();
    config.model.num_heads = 0;
    assert!(config.validate().is_err());

    // Test hidden_size not divisible by num_heads
    config = BitNetConfig::default();
    config.model.hidden_size = 100;
    config.model.num_heads = 7;
    assert!(config.validate().is_err());

    // Test zero intermediate_size
    config = BitNetConfig::default();
    config.model.intermediate_size = 0;
    assert!(config.validate().is_err());

    // Test zero max_position_embeddings
    config = BitNetConfig::default();
    config.model.max_position_embeddings = 0;
    assert!(config.validate().is_err());

    // Test zero max_length
    config = BitNetConfig::default();
    config.inference.max_length = 0;
    assert!(config.validate().is_err());

    // Test zero max_new_tokens
    config = BitNetConfig::default();
    config.inference.max_new_tokens = 0;
    assert!(config.validate().is_err());

    // Test zero temperature
    config = BitNetConfig::default();
    config.inference.temperature = 0.0;
    assert!(config.validate().is_err());

    // Test negative temperature
    config = BitNetConfig::default();
    config.inference.temperature = -1.0;
    assert!(config.validate().is_err());

    // Test zero top_k
    config = BitNetConfig::default();
    config.inference.top_k = Some(0);
    assert!(config.validate().is_err());

    // Test invalid top_p (too low)
    config = BitNetConfig::default();
    config.inference.top_p = Some(0.0);
    assert!(config.validate().is_err());

    // Test invalid top_p (too high)
    config = BitNetConfig::default();
    config.inference.top_p = Some(1.5);
    assert!(config.validate().is_err());

    // Test zero repetition_penalty
    config = BitNetConfig::default();
    config.inference.repetition_penalty = 0.0;
    assert!(config.validate().is_err());

    // Test zero block_size
    config = BitNetConfig::default();
    config.quantization.block_size = 0;
    assert!(config.validate().is_err());

    // Test non-power-of-2 block_size
    config = BitNetConfig::default();
    config.quantization.block_size = 63;
    assert!(config.validate().is_err());

    // Test zero precision
    config = BitNetConfig::default();
    config.quantization.precision = 0.0;
    assert!(config.validate().is_err());

    // Test zero num_threads
    config = BitNetConfig::default();
    config.performance.num_threads = Some(0);
    assert!(config.validate().is_err());

    // Test zero batch_size
    config = BitNetConfig::default();
    config.performance.batch_size = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_builder_comprehensive() {
    let config = BitNetConfig::builder()
        .model_path("/path/to/model.gguf")
        .model_format(ModelFormat::SafeTensors)
        .vocab_size(50000)
        .hidden_size(8192)
        .num_layers(48)
        .num_heads(64)
        .max_length(4096)
        .temperature(0.7)
        .top_k(Some(40))
        .top_p(Some(0.95))
        .quantization_type(QuantizationType::TL2)
        .use_gpu(true)
        .num_threads(Some(16))
        .batch_size(8)
        .build()
        .unwrap();

    assert_eq!(config.model.path, Some("/path/to/model.gguf".into()));
    assert_eq!(config.model.format, ModelFormat::SafeTensors);
    assert_eq!(config.model.vocab_size, 50000);
    assert_eq!(config.model.hidden_size, 8192);
    assert_eq!(config.model.num_layers, 48);
    assert_eq!(config.model.num_heads, 64);
    assert_eq!(config.inference.max_length, 4096);
    assert_eq!(config.inference.temperature, 0.7);
    assert_eq!(config.inference.top_k, Some(40));
    assert_eq!(config.inference.top_p, Some(0.95));
    assert_eq!(config.quantization.quantization_type, QuantizationType::TL2);
    assert!(config.performance.use_gpu);
    assert_eq!(config.performance.num_threads, Some(16));
    assert_eq!(config.performance.batch_size, 8);
}

#[test]
fn test_config_builder_validation_failure() {
    let result = BitNetConfig::builder()
        .vocab_size(0) // Invalid
        .build();
    assert!(result.is_err());
}

#[test]
fn test_toml_config_loading() {
    let toml_content = r#"
[model]
vocab_size = 40000
hidden_size = 2048
num_layers = 24
num_heads = 16
format = "SafeTensors"
path = "/test/model.safetensors"

[inference]
temperature = 0.8
max_new_tokens = 256
top_k = 40
top_p = 0.95
repetition_penalty = 1.2
seed = 42

[quantization]
quantization_type = "TL1"
block_size = 128
precision = 1e-5

[performance]
use_gpu = true
batch_size = 4
num_threads = 8
memory_limit = 2048
"#;

    let mut temp_file = NamedTempFile::with_suffix(".toml").unwrap();
    temp_file.write_all(toml_content.as_bytes()).unwrap();
    temp_file.flush().unwrap();

    let config = BitNetConfig::from_file(temp_file.path()).unwrap();

    assert_eq!(config.model.vocab_size, 40000);
    assert_eq!(config.model.hidden_size, 2048);
    assert_eq!(config.model.num_layers, 24);
    assert_eq!(config.model.num_heads, 16);
    assert_eq!(config.model.format, ModelFormat::SafeTensors);
    assert_eq!(config.model.path, Some("/test/model.safetensors".into()));

    assert_eq!(config.inference.temperature, 0.8);
    assert_eq!(config.inference.max_new_tokens, 256);
    assert_eq!(config.inference.top_k, Some(40));
    assert_eq!(config.inference.top_p, Some(0.95));
    assert_eq!(config.inference.repetition_penalty, 1.2);
    assert_eq!(config.inference.seed, Some(42));

    assert_eq!(config.quantization.quantization_type, QuantizationType::TL1);
    assert_eq!(config.quantization.block_size, 128);
    assert_eq!(config.quantization.precision, 1e-5);

    assert!(config.performance.use_gpu);
    assert_eq!(config.performance.batch_size, 4);
    assert_eq!(config.performance.num_threads, Some(8));
    assert_eq!(config.performance.memory_limit, Some(2048));
}

#[test]
fn test_json_config_loading() {
    let json_content = r#"
{
    "model": {
        "vocab_size": 35000,
        "hidden_size": 3072,
        "num_layers": 28,
        "num_heads": 24,
        "format": "HuggingFace",
        "path": "/test/model"
    },
    "inference": {
        "temperature": 0.9,
        "max_length": 1024,
        "top_k": null,
        "top_p": 0.85,
        "seed": null
    },
    "quantization": {
        "quantization_type": "TL2",
        "block_size": 256
    },
    "performance": {
        "use_gpu": false,
        "num_threads": null,
        "memory_limit": null
    }
}
"#;

    let mut temp_file = NamedTempFile::with_suffix(".json").unwrap();
    temp_file.write_all(json_content.as_bytes()).unwrap();
    temp_file.flush().unwrap();

    let config = BitNetConfig::from_file(temp_file.path()).unwrap();

    assert_eq!(config.model.vocab_size, 35000);
    assert_eq!(config.model.hidden_size, 3072);
    assert_eq!(config.model.num_layers, 28);
    assert_eq!(config.model.num_heads, 24);
    assert_eq!(config.model.format, ModelFormat::HuggingFace);
    assert_eq!(config.model.path, Some("/test/model".into()));

    assert_eq!(config.inference.temperature, 0.9);
    assert_eq!(config.inference.max_length, 1024);
    assert_eq!(config.inference.top_k, None);
    assert_eq!(config.inference.top_p, Some(0.85));
    assert_eq!(config.inference.seed, None);

    assert_eq!(config.quantization.quantization_type, QuantizationType::TL2);
    assert_eq!(config.quantization.block_size, 256);

    assert!(!config.performance.use_gpu);
    assert_eq!(config.performance.num_threads, None);
    assert_eq!(config.performance.memory_limit, None);
}

#[test]
fn test_config_file_error_handling() {
    // Test non-existent file
    let result = BitNetConfig::from_file("non_existent_file.toml");
    assert!(result.is_err());

    // Test invalid TOML
    let mut temp_file = NamedTempFile::with_suffix(".toml").unwrap();
    temp_file.write_all(b"invalid toml content [[[").unwrap();
    temp_file.flush().unwrap();

    let result = BitNetConfig::from_file(temp_file.path());
    assert!(result.is_err());

    // Test invalid JSON
    let mut temp_file = NamedTempFile::with_suffix(".json").unwrap();
    temp_file.write_all(b"{ invalid json }").unwrap();
    temp_file.flush().unwrap();

    let result = BitNetConfig::from_file(temp_file.path());
    assert!(result.is_err());

    // Test unsupported file extension
    let mut temp_file = NamedTempFile::with_suffix(".yaml").unwrap();
    temp_file.write_all(b"model:\n  vocab_size: 1000").unwrap();
    temp_file.flush().unwrap();

    let result = BitNetConfig::from_file(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_env_variable_overrides() {
    let _lock = acquire_env_lock();

    // Clean up any existing env vars
    let env_vars = [
        "BITNET_VOCAB_SIZE",
        "BITNET_TEMPERATURE",
        "BITNET_USE_GPU",
        "BITNET_QUANTIZATION_TYPE",
        "BITNET_MEMORY_LIMIT",
        "BITNET_MODEL_FORMAT",
        "BITNET_MODEL_PATH",
        "BITNET_HIDDEN_SIZE",
        "BITNET_NUM_LAYERS",
        "BITNET_NUM_HEADS",
        "BITNET_MAX_LENGTH",
        "BITNET_MAX_NEW_TOKENS",
        "BITNET_TOP_K",
        "BITNET_TOP_P",
        "BITNET_SEED",
        "BITNET_BLOCK_SIZE",
        "BITNET_NUM_THREADS",
        "BITNET_BATCH_SIZE",
    ];
    for var in &env_vars {
        unsafe {
            env::remove_var(var);
        }
    }

    // Set environment variables
    unsafe {
        env::set_var("BITNET_VOCAB_SIZE", "60000");
    }
    unsafe {
        env::set_var("BITNET_TEMPERATURE", "0.7");
    }
    unsafe {
        env::set_var("BITNET_USE_GPU", "true");
    }
    unsafe {
        env::set_var("BITNET_QUANTIZATION_TYPE", "TL2");
    }
    unsafe {
        env::set_var("BITNET_MEMORY_LIMIT", "2GB");
    }
    unsafe {
        env::set_var("BITNET_MODEL_FORMAT", "safetensors");
    }
    unsafe {
        env::set_var("BITNET_MODEL_PATH", "/test/path");
    }
    unsafe {
        env::set_var("BITNET_HIDDEN_SIZE", "8192");
    }
    unsafe {
        env::set_var("BITNET_NUM_LAYERS", "48");
    }
    unsafe {
        env::set_var("BITNET_NUM_HEADS", "64");
    }
    unsafe {
        env::set_var("BITNET_MAX_LENGTH", "4096");
    }
    unsafe {
        env::set_var("BITNET_MAX_NEW_TOKENS", "1024");
    }
    unsafe {
        env::set_var("BITNET_TOP_K", "30");
    }
    unsafe {
        env::set_var("BITNET_TOP_P", "0.85");
    }
    unsafe {
        env::set_var("BITNET_SEED", "123");
    }
    unsafe {
        env::set_var("BITNET_BLOCK_SIZE", "128");
    }
    unsafe {
        env::set_var("BITNET_NUM_THREADS", "16");
    }
    unsafe {
        env::set_var("BITNET_BATCH_SIZE", "8");
    }

    let config = BitNetConfig::from_env().unwrap();

    assert_eq!(config.model.vocab_size, 60000);
    assert_eq!(config.inference.temperature, 0.7);
    assert!(config.performance.use_gpu);
    assert_eq!(config.quantization.quantization_type, QuantizationType::TL2);
    assert_eq!(config.performance.memory_limit, Some(2 * 1024 * 1024 * 1024));
    assert_eq!(config.model.format, ModelFormat::SafeTensors);
    assert_eq!(config.model.path, Some("/test/path".into()));
    assert_eq!(config.model.hidden_size, 8192);
    assert_eq!(config.model.num_layers, 48);
    assert_eq!(config.model.num_heads, 64);
    assert_eq!(config.inference.max_length, 4096);
    assert_eq!(config.inference.max_new_tokens, 1024);
    assert_eq!(config.inference.top_k, Some(30));
    assert_eq!(config.inference.top_p, Some(0.85));
    assert_eq!(config.inference.seed, Some(123));
    assert_eq!(config.quantization.block_size, 128);
    assert_eq!(config.performance.num_threads, Some(16));
    assert_eq!(config.performance.batch_size, 8);

    // Clean up
    for var in &env_vars {
        unsafe {
            env::remove_var(var);
        }
    }
}

#[test]
fn test_env_variable_special_values() {
    let _lock = acquire_env_lock();

    // Test "none" values
    unsafe {
        env::set_var("BITNET_TOP_K", "none");
    }
    unsafe {
        env::set_var("BITNET_TOP_P", "none");
    }
    unsafe {
        env::set_var("BITNET_SEED", "none");
    }
    unsafe {
        env::set_var("BITNET_NUM_THREADS", "auto");
    }
    unsafe {
        env::set_var("BITNET_MEMORY_LIMIT", "none");
    }

    let config = BitNetConfig::from_env().unwrap();

    assert_eq!(config.inference.top_k, None);
    assert_eq!(config.inference.top_p, None);
    assert_eq!(config.inference.seed, None);
    assert_eq!(config.performance.num_threads, None);
    assert_eq!(config.performance.memory_limit, None);

    // Clean up
    unsafe {
        env::remove_var("BITNET_TOP_K");
    }
    unsafe {
        env::remove_var("BITNET_TOP_P");
    }
    unsafe {
        env::remove_var("BITNET_SEED");
    }
    unsafe {
        env::remove_var("BITNET_NUM_THREADS");
    }
    unsafe {
        env::remove_var("BITNET_MEMORY_LIMIT");
    }
}

#[test]
fn test_memory_limit_parsing() {
    let _lock = acquire_env_lock();

    // Test various memory limit formats
    let test_cases = vec![
        ("1GB", Some(1024 * 1024 * 1024)),
        ("512MB", Some(512 * 1024 * 1024)),
        ("1024KB", Some(1024 * 1024)),
        ("1048576", Some(1048576)),
        ("2gb", Some(2 * 1024 * 1024 * 1024)), // lowercase
        ("none", None),
    ];

    for (input, expected) in test_cases {
        unsafe {
            env::set_var("BITNET_MEMORY_LIMIT", input);
        }
        let config = BitNetConfig::from_env().unwrap();
        assert_eq!(config.performance.memory_limit, expected, "Failed for input: {}", input);
    }

    // Test invalid memory limit
    unsafe {
        env::set_var("BITNET_MEMORY_LIMIT", "invalid_size");
    }
    let result = BitNetConfig::from_env();
    assert!(result.is_err());

    unsafe {
        env::remove_var("BITNET_MEMORY_LIMIT");
    }
}

#[test]
fn test_config_merging() {
    let mut base_config = BitNetConfig::default();
    base_config.model.vocab_size = 30000;
    base_config.inference.temperature = 0.8;
    base_config.performance.use_gpu = false;

    let override_config = BitNetConfig::builder()
        .vocab_size(50000)
        .use_gpu(true)
        .quantization_type(QuantizationType::TL2)
        .build()
        .unwrap();

    base_config.merge_with(override_config);

    // Overridden values
    assert_eq!(base_config.model.vocab_size, 50000);
    assert!(base_config.performance.use_gpu);
    assert_eq!(base_config.quantization.quantization_type, QuantizationType::TL2);

    // Preserved values
    assert_eq!(base_config.inference.temperature, 0.8);
}

#[test]
fn test_config_loader_precedence() {
    let _lock = acquire_env_lock();

    // Clean up env vars
    unsafe {
        env::remove_var("BITNET_VOCAB_SIZE");
    }
    unsafe {
        env::remove_var("BITNET_TEMPERATURE");
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
    temp_file.flush().unwrap();

    // Set environment variable that should override file
    unsafe {
        env::set_var("BITNET_VOCAB_SIZE", "50000");
    }

    let config = ConfigLoader::load_with_precedence(Some(temp_file.path())).unwrap();

    // Environment should override file
    assert_eq!(config.model.vocab_size, 50000);
    // File should override default
    assert_eq!(config.inference.temperature, 0.9);

    // Clean up
    unsafe {
        env::remove_var("BITNET_VOCAB_SIZE");
    }
}

#[test]
fn test_config_sources() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test.toml");

    let toml_content = r#"
[model]
vocab_size = 35000

[inference]
temperature = 0.85
"#;
    fs::write(&config_path, toml_content).unwrap();

    let inline_config = BitNetConfig::builder().use_gpu(true).batch_size(4).build().unwrap();

    let sources =
        vec![ConfigSource::File(config_path), ConfigSource::Inline(Box::new(inline_config))];

    let config = ConfigLoader::load_from_sources(&sources).unwrap();

    // From file
    assert_eq!(config.model.vocab_size, 35000);
    assert_eq!(config.inference.temperature, 0.85);

    // From inline
    assert!(config.performance.use_gpu);
    assert_eq!(config.performance.batch_size, 4);
}

// Property-based tests
proptest! {
    #[test]
    fn test_config_builder_with_arbitrary_values(
        vocab_size in 1usize..1000000,
        hidden_size in 64usize..16384,
        num_layers in 1usize..100,
        temperature in 0.1f32..2.0,
        batch_size in 1usize..128
    ) {
        // Ensure hidden_size is divisible by a reasonable number of heads
        let num_heads = if hidden_size >= 64 {
            let mut heads = 8;
            while hidden_size % heads != 0 && heads <= 64 {
                heads += 8;
            }
            if hidden_size % heads == 0 { heads } else { 8 }
        } else { 1 };

        let result = BitNetConfig::builder()
            .vocab_size(vocab_size)
            .hidden_size(hidden_size)
            .num_layers(num_layers)
            .num_heads(num_heads)
            .temperature(temperature)
            .batch_size(batch_size)
            .build();

        if hidden_size % num_heads == 0 {
            let config = result.unwrap();
            assert_eq!(config.model.vocab_size, vocab_size);
            assert_eq!(config.model.hidden_size, hidden_size);
            assert_eq!(config.model.num_layers, num_layers);
            assert_eq!(config.model.num_heads, num_heads);
            assert_eq!(config.inference.temperature, temperature);
            assert_eq!(config.performance.batch_size, batch_size);
        }
    }

    #[test]
    fn test_config_serialization_roundtrip(
        vocab_size in 1usize..100000,
        temperature in 0.1f32..2.0,
        use_gpu in any::<bool>(),
        quantization_type in prop_oneof![
            Just(QuantizationType::I2S),
            Just(QuantizationType::TL1),
            Just(QuantizationType::TL2)
        ]
    ) {
        let config = BitNetConfig::builder()
            .vocab_size(vocab_size)
            .temperature(temperature)
            .use_gpu(use_gpu)
            .quantization_type(quantization_type)
            .build()
            .unwrap();

        // Test JSON roundtrip
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: BitNetConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.model.vocab_size, deserialized.model.vocab_size);
        assert_eq!(config.inference.temperature, deserialized.inference.temperature);
        assert_eq!(config.performance.use_gpu, deserialized.performance.use_gpu);
        assert_eq!(config.quantization.quantization_type, deserialized.quantization.quantization_type);
    }
}

#[test]
fn test_config_debug_formatting() {
    let config = BitNetConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("BitNetConfig"));
    assert!(debug_str.contains("model"));
    assert!(debug_str.contains("inference"));
    assert!(debug_str.contains("quantization"));
    assert!(debug_str.contains("performance"));
}

#[test]
fn test_config_clone() {
    let config = BitNetConfig::default();
    let cloned = config.clone();

    assert_eq!(config.model.vocab_size, cloned.model.vocab_size);
    assert_eq!(config.inference.temperature, cloned.inference.temperature);
    assert_eq!(config.quantization.quantization_type, cloned.quantization.quantization_type);
    assert_eq!(config.performance.use_gpu, cloned.performance.use_gpu);
}

#[test]
fn test_config_send_sync() {
    // Ensure config types are Send + Sync for use in async contexts
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<BitNetConfig>();
    assert_send_sync::<ModelConfig>();
    assert_send_sync::<InferenceConfig>();
    assert_send_sync::<QuantizationConfig>();
    assert_send_sync::<PerformanceConfig>();
    assert_send_sync::<ConfigBuilder>();
}
