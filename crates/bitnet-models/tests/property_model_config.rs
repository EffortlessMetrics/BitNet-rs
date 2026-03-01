//! Property-based tests for model configuration validation.
//!
//! Invariants verified:
//! - Default ModelConfig passes validation
//! - Serde round-trip preserves all fields
//! - Zero-valued architecture fields always fail validation
//! - Hidden size divisibility by num_heads is enforced
//! - Valid configs are closed under field clamping to positive values

#![cfg(all(test, feature = "cpu"))]

use bitnet_common::BitNetConfig;
use bitnet_common::config::{ModelConfig, ModelFormat};
use proptest::prelude::*;

// ── Default config validity ─────────────────────────────────────────────────

proptest! {
    /// The default BitNetConfig always passes validation.
    #[test]
    fn default_config_always_valid(_dummy in 0u8..1) {
        let cfg = BitNetConfig::default();
        prop_assert!(
            cfg.validate().is_ok(),
            "Default BitNetConfig must pass validation: {:?}",
            cfg.validate().err()
        );
    }

    /// Default ModelConfig has positive architecture fields.
    #[test]
    fn default_model_config_fields_positive(_dummy in 0u8..1) {
        let m = ModelConfig::default();
        prop_assert!(m.vocab_size > 0, "vocab_size must be > 0");
        prop_assert!(m.hidden_size > 0, "hidden_size must be > 0");
        prop_assert!(m.num_layers > 0, "num_layers must be > 0");
        prop_assert!(m.num_heads > 0, "num_heads must be > 0");
        prop_assert!(m.intermediate_size > 0, "intermediate_size must be > 0");
        prop_assert!(
            m.max_position_embeddings > 0,
            "max_position_embeddings must be > 0"
        );
    }
}

// ── Serde round-trip ────────────────────────────────────────────────────────

proptest! {
    /// ModelConfig survives JSON serialize -> deserialize without data loss
    /// for core architecture fields.
    #[test]
    fn model_config_json_roundtrip(
        vocab_size in 1usize..=100_000,
        hidden_size in 1usize..=8192,
        num_layers in 1usize..=128,
        num_heads in 1usize..=128,
        intermediate_size in 1usize..=32768,
        max_pos in 1usize..=16384,
    ) {
        let mut cfg = ModelConfig::default();
        cfg.vocab_size = vocab_size;
        cfg.hidden_size = hidden_size;
        cfg.num_layers = num_layers;
        cfg.num_heads = num_heads;
        cfg.intermediate_size = intermediate_size;
        cfg.max_position_embeddings = max_pos;

        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: ModelConfig = serde_json::from_str(&json).expect("deserialize");

        prop_assert_eq!(back.vocab_size, vocab_size);
        prop_assert_eq!(back.hidden_size, hidden_size);
        prop_assert_eq!(back.num_layers, num_layers);
        prop_assert_eq!(back.num_heads, num_heads);
        prop_assert_eq!(back.intermediate_size, intermediate_size);
        prop_assert_eq!(back.max_position_embeddings, max_pos);
    }

    /// BitNetConfig survives JSON serialize -> deserialize round-trip.
    #[test]
    fn bitnet_config_json_roundtrip(
        num_heads in 1usize..=8,
        head_dim in 1usize..=64,
        vocab_size in 1usize..=32000,
        num_layers in 1usize..=16,
    ) {
        let hidden_size = num_heads * head_dim;
        let mut cfg = BitNetConfig::default();
        cfg.model.hidden_size = hidden_size;
        cfg.model.num_heads = num_heads;
        cfg.model.vocab_size = vocab_size;
        cfg.model.num_layers = num_layers;

        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: BitNetConfig = serde_json::from_str(&json).expect("deserialize");

        prop_assert_eq!(back.model.hidden_size, hidden_size);
        prop_assert_eq!(back.model.num_heads, num_heads);
        prop_assert_eq!(back.model.vocab_size, vocab_size);
        prop_assert_eq!(back.model.num_layers, num_layers);
    }
}

// ── Validation catches invalid configurations ───────────────────────────────

proptest! {
    /// Any config with hidden_size=0 always fails validation.
    #[test]
    fn zero_hidden_size_fails(
        vocab_size in 1usize..=32000,
        num_layers in 1usize..=16,
    ) {
        let mut cfg = BitNetConfig::default();
        cfg.model.hidden_size = 0;
        cfg.model.vocab_size = vocab_size;
        cfg.model.num_layers = num_layers;
        prop_assert!(cfg.validate().is_err(), "hidden_size=0 must be rejected");
    }

    /// Any config with num_layers=0 always fails validation.
    #[test]
    fn zero_num_layers_fails(
        num_heads in 1usize..=8,
        head_dim in 1usize..=64,
        vocab_size in 1usize..=32000,
    ) {
        let mut cfg = BitNetConfig::default();
        cfg.model.hidden_size = num_heads * head_dim;
        cfg.model.num_heads = num_heads;
        cfg.model.vocab_size = vocab_size;
        cfg.model.num_layers = 0;
        prop_assert!(cfg.validate().is_err(), "num_layers=0 must be rejected");
    }

    /// hidden_size not divisible by num_heads always fails validation.
    #[test]
    fn non_divisible_hidden_by_heads_fails(
        num_heads in 2usize..=16,
        base in 1usize..=64,
    ) {
        let hidden_size = base * num_heads + 1; // guaranteed remainder 1
        let mut cfg = BitNetConfig::default();
        cfg.model.hidden_size = hidden_size;
        cfg.model.num_heads = num_heads;
        cfg.model.num_key_value_heads = 0;
        prop_assert!(
            cfg.validate().is_err(),
            "hidden_size={} % num_heads={} != 0 must be rejected",
            hidden_size,
            num_heads
        );
    }

    /// Any config with vocab_size=0 always fails validation.
    #[test]
    fn zero_vocab_size_fails(
        num_heads in 1usize..=8,
        head_dim in 1usize..=64,
        num_layers in 1usize..=16,
    ) {
        let mut cfg = BitNetConfig::default();
        cfg.model.hidden_size = num_heads * head_dim;
        cfg.model.num_heads = num_heads;
        cfg.model.vocab_size = 0;
        cfg.model.num_layers = num_layers;
        prop_assert!(cfg.validate().is_err(), "vocab_size=0 must be rejected");
    }
}

// ── Valid config construction ────────────────────────────────────────────────

proptest! {
    /// Any config where hidden_size = num_heads * head_dim (exact multiple)
    /// and all fields are positive passes validation.
    #[test]
    fn valid_config_passes_validation(
        num_heads in 1usize..=8,
        head_dim in 1usize..=64,
        vocab_size in 1usize..=32000,
        num_layers in 1usize..=16,
    ) {
        let hidden_size = num_heads * head_dim;
        let mut cfg = BitNetConfig::default();
        cfg.model.hidden_size = hidden_size;
        cfg.model.num_heads = num_heads;
        cfg.model.num_key_value_heads = 0;
        cfg.model.vocab_size = vocab_size;
        cfg.model.num_layers = num_layers;
        prop_assert!(
            cfg.validate().is_ok(),
            "valid config should pass: {:?}",
            cfg.validate().err()
        );
    }

    /// ModelFormat round-trips through serde for all variants.
    #[test]
    fn model_format_serde_roundtrip(
        fmt in prop_oneof![
            Just(ModelFormat::Gguf),
            Just(ModelFormat::SafeTensors),
            Just(ModelFormat::HuggingFace),
        ]
    ) {
        let json = serde_json::to_string(&fmt).expect("serialize");
        let back: ModelFormat = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(fmt, back);
    }
}
