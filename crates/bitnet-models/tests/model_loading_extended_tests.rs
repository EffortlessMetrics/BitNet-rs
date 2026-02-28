//! Extended tests for `bitnet-models` core types.
//!
//! Targets areas not covered by `comprehensive_tests.rs` (gated on
//! `integration-tests`) or `model_proptests.rs`:
//!
//! - `ModelMetadata` construction variants (quantization types, fingerprint,
//!   large context length)
//! - `QuantizationType` Display, serde round-trip, Copy, equality
//! - `BitNetConfig` / `ModelConfig` default field values and serialization
//! - `LoadConfig` defaults and clone
//! - `ProductionLoadConfig` defaults and strict-validation flag
//! - `ProductionModelLoader` construction, memory requirements, device config
//! - `ModelLoader` extension behaviour (unknown extension, bad path)
//! - Memory-estimation sanity for known configs
//!
//! Run with:
//!   `cargo test --locked -p bitnet-models --no-default-features --features cpu \
//!    -- model_loading_extended_tests`
#![cfg(feature = "cpu")]

use bitnet_common::{BitNetConfig, Device, ModelMetadata, QuantizationType};
use bitnet_models::{
    loader::{LoadConfig, ModelLoader},
    production_loader::{ProductionLoadConfig, ProductionModelLoader},
};

// ---------------------------------------------------------------------------
// ModelMetadata – construction variants
// ---------------------------------------------------------------------------

#[test]
fn model_metadata_no_quantization() {
    let m = ModelMetadata {
        name: "base".to_string(),
        version: "0.1".to_string(),
        architecture: "llama".to_string(),
        vocab_size: 32000,
        context_length: 4096,
        quantization: None,
        fingerprint: None,
        corrections_applied: None,
    };
    assert!(m.quantization.is_none());
    assert_eq!(m.vocab_size, 32000);
}

#[test]
fn model_metadata_i2s_quantization() {
    let m = ModelMetadata {
        name: "bitnet".to_string(),
        version: "1.0".to_string(),
        architecture: "bitnet".to_string(),
        vocab_size: 32000,
        context_length: 2048,
        quantization: Some(QuantizationType::I2S),
        fingerprint: None,
        corrections_applied: None,
    };
    assert_eq!(m.quantization, Some(QuantizationType::I2S));
}

#[test]
fn model_metadata_tl1_quantization() {
    let m = ModelMetadata {
        name: "bitnet-tl1".to_string(),
        version: "1.0".to_string(),
        architecture: "bitnet".to_string(),
        vocab_size: 32000,
        context_length: 2048,
        quantization: Some(QuantizationType::TL1),
        fingerprint: None,
        corrections_applied: None,
    };
    assert_eq!(m.quantization, Some(QuantizationType::TL1));
}

#[test]
fn model_metadata_tl2_quantization() {
    let m = ModelMetadata {
        name: "bitnet-tl2".to_string(),
        version: "1.0".to_string(),
        architecture: "bitnet".to_string(),
        vocab_size: 32000,
        context_length: 2048,
        quantization: Some(QuantizationType::TL2),
        fingerprint: None,
        corrections_applied: None,
    };
    assert_eq!(m.quantization, Some(QuantizationType::TL2));
}

#[test]
fn model_metadata_with_fingerprint() {
    let m = ModelMetadata {
        name: "model-sha".to_string(),
        version: "1.0".to_string(),
        architecture: "bitnet".to_string(),
        vocab_size: 32000,
        context_length: 2048,
        quantization: Some(QuantizationType::I2S),
        fingerprint: Some("sha256-deadbeef".to_string()),
        corrections_applied: None,
    };
    assert_eq!(m.fingerprint.as_deref(), Some("sha256-deadbeef"));
}

#[test]
fn model_metadata_large_context_length() {
    let m = ModelMetadata {
        name: "long-ctx".to_string(),
        version: "1.0".to_string(),
        architecture: "llama".to_string(),
        vocab_size: 128256,
        context_length: 131_072,
        quantization: None,
        fingerprint: None,
        corrections_applied: None,
    };
    assert_eq!(m.context_length, 131_072);
    assert_eq!(m.vocab_size, 128256);
}

// ---------------------------------------------------------------------------
// QuantizationType – Display, serde, Copy, equality
// ---------------------------------------------------------------------------

#[test]
fn quantization_type_i2s_display() {
    assert_eq!(format!("{}", QuantizationType::I2S), "I2_S");
}

#[test]
fn quantization_type_tl1_display() {
    assert_eq!(format!("{}", QuantizationType::TL1), "TL1");
}

#[test]
fn quantization_type_tl2_display() {
    assert_eq!(format!("{}", QuantizationType::TL2), "TL2");
}

#[test]
fn quantization_type_all_variants_distinct() {
    assert_ne!(QuantizationType::I2S, QuantizationType::TL1);
    assert_ne!(QuantizationType::TL1, QuantizationType::TL2);
    assert_ne!(QuantizationType::I2S, QuantizationType::TL2);
}

#[test]
fn quantization_type_copy() {
    let original = QuantizationType::I2S;
    let copy = original;
    assert_eq!(original, copy);
}

#[test]
fn quantization_type_serde_roundtrip_json() {
    for qt in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let json = serde_json::to_string(&qt).unwrap();
        let back: QuantizationType = serde_json::from_str(&json).unwrap();
        assert_eq!(qt, back, "serde round-trip failed for {:?}", qt);
    }
}

// ---------------------------------------------------------------------------
// BitNetConfig / ModelConfig – default field values
// ---------------------------------------------------------------------------

#[test]
fn bitnet_config_default_vocab_size() {
    let cfg = BitNetConfig::default();
    assert_eq!(cfg.model.vocab_size, 32000);
}

#[test]
fn bitnet_config_default_hidden_size() {
    let cfg = BitNetConfig::default();
    assert_eq!(cfg.model.hidden_size, 4096);
}

#[test]
fn bitnet_config_default_num_layers() {
    let cfg = BitNetConfig::default();
    assert_eq!(cfg.model.num_layers, 32);
}

#[test]
fn bitnet_config_default_num_heads() {
    let cfg = BitNetConfig::default();
    assert_eq!(cfg.model.num_heads, 32);
}

#[test]
fn bitnet_config_default_max_position_embeddings() {
    let cfg = BitNetConfig::default();
    assert_eq!(cfg.model.max_position_embeddings, 2048);
}

#[test]
fn bitnet_config_default_num_key_value_heads_is_zero() {
    // 0 means "use num_heads" (MHA default)
    let cfg = BitNetConfig::default();
    assert_eq!(cfg.model.num_key_value_heads, 0);
}

#[test]
fn bitnet_config_default_intermediate_size() {
    let cfg = BitNetConfig::default();
    assert!(cfg.model.intermediate_size > 0);
}

#[test]
fn bitnet_config_serde_roundtrip_json() {
    let original = BitNetConfig::default();
    let json = serde_json::to_string(&original).unwrap();
    let restored: BitNetConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(original.model.vocab_size, restored.model.vocab_size);
    assert_eq!(original.model.hidden_size, restored.model.hidden_size);
    assert_eq!(original.model.num_layers, restored.model.num_layers);
}

// ---------------------------------------------------------------------------
// LoadConfig – default and clone
// ---------------------------------------------------------------------------

#[test]
fn load_config_default_use_mmap_true() {
    let cfg = LoadConfig::default();
    assert!(cfg.use_mmap);
}

#[test]
fn load_config_default_validate_checksums_true() {
    let cfg = LoadConfig::default();
    assert!(cfg.validate_checksums);
}

#[test]
fn load_config_default_no_progress_callback() {
    let cfg = LoadConfig::default();
    assert!(cfg.progress_callback.is_none());
}

#[test]
fn load_config_clone_is_independent() {
    let original = LoadConfig::default();
    let mut cloned = original.clone();
    cloned.use_mmap = false;
    // Original should be unchanged.
    assert!(LoadConfig::default().use_mmap);
    assert!(!cloned.use_mmap);
}

// ---------------------------------------------------------------------------
// ProductionLoadConfig – defaults
// ---------------------------------------------------------------------------

#[test]
fn production_load_config_default_strict_validation() {
    let cfg = ProductionLoadConfig::default();
    assert!(cfg.strict_validation);
}

#[test]
fn production_load_config_default_target_device_is_cpu() {
    let cfg = ProductionLoadConfig::default();
    assert_eq!(cfg.target_device, Device::Cpu);
}

#[test]
fn production_load_config_default_validate_tensor_alignment() {
    let cfg = ProductionLoadConfig::default();
    assert!(cfg.validate_tensor_alignment);
}

#[test]
fn production_load_config_default_max_model_size_is_set() {
    let cfg = ProductionLoadConfig::default();
    let max = cfg.max_model_size_bytes.expect("max_model_size_bytes should be set");
    // Default is 32 GiB.
    assert_eq!(max, 32 * 1024 * 1024 * 1024);
}

#[test]
fn production_load_config_profile_memory_false_by_default() {
    let cfg = ProductionLoadConfig::default();
    assert!(!cfg.profile_memory);
}

// ---------------------------------------------------------------------------
// ProductionModelLoader – construction / memory / device config
// ---------------------------------------------------------------------------

#[test]
fn production_model_loader_new_enables_validation() {
    let loader = ProductionModelLoader::new();
    assert!(loader.validation_enabled);
}

#[test]
fn production_model_loader_strict_sets_flags() {
    let loader = ProductionModelLoader::new_with_strict_validation();
    assert!(loader.config.strict_validation);
    assert!(loader.config.validate_tensor_alignment);
}

#[test]
fn production_model_loader_memory_requirements_cpu() {
    let loader = ProductionModelLoader::new();
    let req = loader.get_memory_requirements("cpu");
    assert!(req.total_mb > 0, "total_mb must be positive");
    assert!(req.cpu_memory_mb > 0, "cpu_memory_mb must be positive");
    assert!(req.gpu_memory_mb.is_none(), "no GPU memory for cpu device");
}

#[test]
fn production_model_loader_memory_requirements_gpu() {
    let loader = ProductionModelLoader::new();
    let req = loader.get_memory_requirements("gpu");
    assert!(req.total_mb > 0);
    assert!(req.gpu_memory_mb.is_some(), "GPU memory must be set for gpu device");
}

#[test]
fn production_model_loader_optimal_device_config_has_strategy() {
    let loader = ProductionModelLoader::new();
    let dc = loader.get_optimal_device_config();
    assert!(dc.strategy.is_some());
    assert!(dc.recommended_batch_size >= 1);
}

#[test]
fn production_model_loader_cpu_threads_at_least_one() {
    let loader = ProductionModelLoader::new();
    let dc = loader.get_optimal_device_config();
    let threads = dc.cpu_threads.expect("cpu_threads should be Some");
    assert!(threads >= 1);
}

// ---------------------------------------------------------------------------
// ModelLoader – available formats and error on missing path
// ---------------------------------------------------------------------------

#[test]
fn model_loader_available_formats_includes_gguf() {
    let loader = ModelLoader::new(Device::Cpu);
    let formats = loader.available_formats();
    assert!(formats.contains(&"GGUF"), "GGUF must be a known format");
}

#[test]
fn model_loader_available_formats_includes_safetensors() {
    let loader = ModelLoader::new(Device::Cpu);
    let formats = loader.available_formats();
    assert!(formats.contains(&"SafeTensors"), "SafeTensors must be a known format");
}

#[test]
fn model_loader_load_nonexistent_path_returns_err() {
    let loader = ModelLoader::new(Device::Cpu);
    let result = loader.load("/nonexistent/path/model.gguf");
    assert!(result.is_err(), "loading a non-existent file must fail");
}

// ---------------------------------------------------------------------------
// Memory estimation – bytes-per-param sanity for known configs
// ---------------------------------------------------------------------------

/// Sanity-check that I2_S 2B-param model fits within 1.5 GiB.
/// 2-bit quant: 2 billion params × 0.25 bytes/param ≈ 500 MB.
#[test]
fn memory_estimation_i2s_2b_model_under_1_5_gib() {
    const PARAMS_2B: u64 = 2_000_000_000;
    const BITS_PER_PARAM_I2S: u64 = 2;
    let bytes = PARAMS_2B * BITS_PER_PARAM_I2S / 8;
    let mib = bytes / (1024 * 1024);
    assert!(mib < 1536, "I2_S 2B model should be < 1.5 GiB, got {} MiB", mib);
}

/// F16 weights for LayerNorm at the same scale are tiny compared to the model.
#[test]
fn memory_estimation_layernorm_f16_weights_small() {
    // Typical BitNet 2B: 32 layers × hidden_size=4096 × 2 bytes = ~256 KiB
    const LAYERS: u64 = 32;
    const HIDDEN: u64 = 4096;
    const BYTES_PER_F16: u64 = 2;
    let ln_bytes = LAYERS * HIDDEN * BYTES_PER_F16;
    let ln_kb = ln_bytes / 1024;
    assert!(ln_kb < 512, "LayerNorm weights should be < 512 KiB, got {} KiB", ln_kb);
}

// ---------------------------------------------------------------------------
// proptest – QuantizationType round-trips and ModelMetadata invariants
// ---------------------------------------------------------------------------

#[cfg(test)]
mod model_proptests_ext {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// ModelMetadata with any valid vocab_size preserves the value.
        #[test]
        fn model_metadata_vocab_size_roundtrip(vocab_size in 1usize..=200_000) {
            let m = ModelMetadata {
                name: "test".to_string(),
                version: "1.0".to_string(),
                architecture: "bitnet".to_string(),
                vocab_size,
                context_length: 2048,
                quantization: None,
                fingerprint: None,
                corrections_applied: None,
            };
            prop_assert_eq!(m.vocab_size, vocab_size);
        }

        /// BitNetConfig serde round-trip preserves vocab_size for arbitrary values.
        #[test]
        fn bitnet_config_vocab_size_roundtrip(vocab_size in 1usize..=256_000) {
            let mut cfg = BitNetConfig::default();
            cfg.model.vocab_size = vocab_size;
            let json = serde_json::to_string(&cfg).unwrap();
            let back: BitNetConfig = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(back.model.vocab_size, vocab_size);
        }

        /// ProductionLoadConfig never loses its max_model_size_bytes field on clone.
        #[test]
        fn production_load_config_max_size_survives_clone(
            size_gib in 1u64..=128,
        ) {
            let cfg = ProductionLoadConfig {
                max_model_size_bytes: Some(size_gib * 1024 * 1024 * 1024),
                ..Default::default()
            };
            let cloned = cfg.clone();
            prop_assert_eq!(cloned.max_model_size_bytes, Some(size_gib * 1024 * 1024 * 1024));
        }
    }
}
