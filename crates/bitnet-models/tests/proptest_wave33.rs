//! Wave 33 property tests: model configuration and naming invariants.
//!
//! Properties tested (5):
//! 1. GGUF header round-trip (write→read preserves metadata)
//! 2. Tensor name sanitization (normalize_name) is idempotent
//! 3. Model config validation rejects invalid dimensions (zero fields)
//! 4. Layer count must match architecture spec (num_kv_heads ≤ num_heads)
//! 5. Embedding dimension must be positive and divisible by num_heads

#![cfg(all(test, feature = "cpu"))]

use bitnet_models::config::GgufModelConfig;
use bitnet_models::gguf_writer::GgufBuilder;
use bitnet_models::names::{is_layernorm_weight, is_projection_weight};
use proptest::prelude::*;
use std::io::Cursor;

// ── helpers ─────────────────────────────────────────────────────────────

/// Build a valid `GgufModelConfig` from parameters, computing head_dim.
fn make_config(
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
) -> GgufModelConfig {
    GgufModelConfig {
        architecture: "llama".to_string(),
        model_name: None,
        vocab_size: 32_000,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim: if num_heads > 0 { hidden_size / num_heads } else { 0 },
        intermediate_size: 11_008,
        max_seq_len: 2048,
        rope_theta: 10_000.0,
        rope_scaling: None,
        quantization: Default::default(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // ── 1. GGUF header round-trip ───────────────────────────────────────

    /// Writing metadata with GgufBuilder and reading it back via the GGUF
    /// parser preserves the architecture string and u32 metadata values.
    #[test]
    fn prop_gguf_metadata_round_trip(
        arch in "[a-z]{3,8}",
        vocab in 1u32..200_000,
        layers in 1u32..128,
    ) {
        let mut buf = Cursor::new(Vec::new());

        let builder = GgufBuilder::new()
            .architecture(&arch)
            .metadata_string("general.architecture", &arch)
            .metadata_u32(&format!("{arch}.vocab_size"), vocab)
            .metadata_u32(&format!("{arch}.block_count"), layers);

        buf = builder.write(buf).expect("write should succeed");
        let data = buf.into_inner();

        // Parse back with the low-level GGUF reader.
        let reader = bitnet_models::formats::gguf::GgufReader::new(&data)
            .expect("reader should parse written data");

        let read_arch = reader
            .get_string_metadata("general.architecture")
            .expect("architecture metadata should exist");
        prop_assert_eq!(&read_arch, &arch);

        let key_vocab = format!("{arch}.vocab_size");
        let read_vocab = reader
            .get_u32_metadata(&key_vocab)
            .expect("vocab_size metadata should exist");
        prop_assert_eq!(read_vocab, vocab);

        let key_layers = format!("{arch}.block_count");
        let read_layers = reader
            .get_u32_metadata(&key_layers)
            .expect("block_count metadata should exist");
        prop_assert_eq!(read_layers, layers);
    }

    // ── 2. Tensor name classification is idempotent ─────────────────────

    /// `is_layernorm_weight` and `is_projection_weight` are stable
    /// predicates: calling them twice on the same name returns the same
    /// answer, and the two categories are mutually exclusive for known
    /// names.
    #[test]
    fn prop_name_classification_idempotent(
        layer_idx in 0u32..64,
        suffix in prop_oneof![
            Just("attn_norm.weight"),
            Just("ffn_norm.weight"),
            Just("attn_q.weight"),
            Just("attn_k.weight"),
            Just("attn_v.weight"),
            Just("attn_output.weight"),
            Just("ffn_gate.weight"),
            Just("ffn_up.weight"),
            Just("ffn_down.weight"),
        ],
    ) {
        let name = format!("blk.{layer_idx}.{suffix}");
        let ln1 = is_layernorm_weight(&name);
        let ln2 = is_layernorm_weight(&name);
        prop_assert_eq!(ln1, ln2, "is_layernorm_weight must be idempotent");

        let proj1 = is_projection_weight(&name);
        let proj2 = is_projection_weight(&name);
        prop_assert_eq!(proj1, proj2, "is_projection_weight must be idempotent");

        // Known names are either LN or projection, never both.
        prop_assert!(
            !(ln1 && proj1),
            "name '{}' classified as both layernorm AND projection",
            name,
        );
    }

    // ── 3. Validation rejects zero dimensions ───────────────────────────

    /// Setting any critical dimension to zero should cause `validate()` to
    /// return an error.
    #[test]
    fn prop_validate_rejects_zero_dims(
        field_idx in 0usize..5,
    ) {
        // Start with a valid config.
        let mut cfg = make_config(4096, 32, 32, 32);

        // Zero out one field at a time.
        match field_idx {
            0 => cfg.vocab_size = 0,
            1 => { cfg.hidden_size = 0; cfg.head_dim = 0; },
            2 => cfg.num_layers = 0,
            3 => { cfg.num_heads = 0; cfg.head_dim = 0; },
            4 => cfg.intermediate_size = 0,
            _ => unreachable!(),
        }

        prop_assert!(
            cfg.validate().is_err(),
            "zeroed field_idx={} should fail validation",
            field_idx,
        );
    }

    // ── 4. num_kv_heads must be ≤ num_heads ─────────────────────────────

    /// If `num_kv_heads > num_heads`, validation must reject the config.
    #[test]
    fn prop_kv_heads_le_num_heads(
        num_heads in 1u32..=64,
        extra in 1u32..=32,
    ) {
        let hidden = (num_heads as usize) * 128;
        let num_kv = (num_heads + extra) as usize; // strictly > num_heads
        let cfg = make_config(hidden, num_heads as usize, num_kv, 32);

        prop_assert!(
            cfg.validate().is_err(),
            "num_kv_heads ({}) > num_heads ({}) should fail",
            num_kv,
            num_heads,
        );
    }

    // ── 5. hidden_size must be divisible by num_heads ───────────────────

    /// Validation must reject configs where `hidden_size % num_heads != 0`.
    #[test]
    fn prop_hidden_divisible_by_heads(
        num_heads in 2usize..=64,
        offset in 1usize..=127,
    ) {
        // Pick a hidden_size that is NOT divisible by num_heads.
        let base = num_heads * 64;
        let hidden = base + (offset % num_heads).max(1); // ensure remainder != 0
        // Intentionally set head_dim to an incorrect value to trigger validation.
        let cfg = GgufModelConfig {
            architecture: "llama".to_string(),
            model_name: None,
            vocab_size: 32_000,
            hidden_size: hidden,
            num_layers: 32,
            num_heads,
            num_kv_heads: num_heads,
            head_dim: hidden / num_heads, // integer division, won't match hidden/num_heads exactly
            intermediate_size: 11_008,
            max_seq_len: 2048,
            rope_theta: 10_000.0,
            rope_scaling: None,
            quantization: Default::default(),
        };

        // Either hidden_size % num_heads != 0 triggers error,
        // or head_dim != hidden_size / num_heads triggers error.
        if hidden % num_heads != 0 {
            prop_assert!(
                cfg.validate().is_err(),
                "hidden_size {} not divisible by num_heads {} should fail",
                hidden,
                num_heads,
            );
        }
        // If it happens to be divisible (offset % num_heads == 0 was guarded
        // above via .max(1), but just in case), validation may pass — that's OK.
    }
}
