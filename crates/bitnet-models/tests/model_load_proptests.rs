//! Property-based tests for GGUF loading invariants in `bitnet-models`.
//!
//! Covers five areas:
//! 1. `BitNetConfig` JSON serialisation round-trip (model-dimension fields).
//! 2. Any config produced by the valid-config strategy passes `validate()`.
//! 3. `normalize_vendor_key` never panics on arbitrary input strings.
//! 4. Every quantised `GgufTensorType` reports a block size ≥ 1.
//! 5. `I2SFlavor` byte-layout invariant: data bytes ≤ total bytes.
//!
//! Negative tests:
//! 6. A config with `hidden_size = 0` always fails `validate()`.
//!
//! Additional unit-level invariants (non-proptest):
//! 7. All known quantised types are positive-block-size.
//! 8. `I2SFlavor::GgmlQk256NoScale` has exactly 256 elements per block.

use bitnet_common::{
    BitNetConfig,
    config::{InferenceConfig, ModelConfig, PerformanceConfig, QuantizationConfig},
};
use bitnet_models::formats::gguf::{GgufTensorType, I2SFlavor};
use bitnet_models::weight_mapper::normalize_vendor_key;
use proptest::prelude::*;

// ── Strategies ────────────────────────────────────────────────────────────────

/// Head counts that divide common hidden sizes evenly.
fn valid_num_heads() -> impl Strategy<Value = usize> {
    prop_oneof![Just(1usize), Just(2), Just(4), Just(8), Just(16), Just(32)]
}

/// Builds a `BitNetConfig` whose model fields satisfy all `validate()` constraints:
/// * all dimension fields are positive
/// * `hidden_size` is an integer multiple of `num_heads`
/// * inference / quantisation / performance fields are at valid defaults
fn arb_valid_config() -> impl Strategy<Value = BitNetConfig> {
    valid_num_heads().prop_flat_map(|num_heads| {
        (
            1usize..=256,   // head multiplier → hidden_size = num_heads * mult
            1usize..=32,    // num_layers
            1usize..=32000, // vocab_size
            1usize..=16384, // intermediate_size
            1usize..=4096,  // max_position_embeddings
        )
            .prop_map(move |(mult, num_layers, vocab_size, intermediate_size, max_pos)| {
                BitNetConfig {
                    model: ModelConfig {
                        vocab_size,
                        hidden_size: num_heads * mult,
                        num_heads,
                        num_key_value_heads: 0, // 0 means "use num_heads"
                        num_layers,
                        intermediate_size,
                        max_position_embeddings: max_pos,
                        ..Default::default()
                    },
                    inference: InferenceConfig::default(),
                    quantization: QuantizationConfig::default(),
                    performance: PerformanceConfig::default(),
                }
            })
    })
}

// ── Property tests ────────────────────────────────────────────────────────────

proptest! {
    /// Property 1 – JSON round-trip preserves model-dimension fields.
    ///
    /// After `serialize → deserialize`, `vocab_size`, `hidden_size`, `num_layers`,
    /// `num_heads`, `intermediate_size`, and `max_position_embeddings` must be
    /// identical to the original values.
    #[test]
    fn config_json_roundtrip_preserves_model_dims(config in arb_valid_config()) {
        let json = serde_json::to_string(&config)
            .expect("BitNetConfig must always be serialisable to JSON");
        let restored: BitNetConfig =
            serde_json::from_str(&json).expect("JSON round-trip must be deserializable");

        prop_assert_eq!(restored.model.vocab_size, config.model.vocab_size);
        prop_assert_eq!(restored.model.hidden_size, config.model.hidden_size);
        prop_assert_eq!(restored.model.num_layers, config.model.num_layers);
        prop_assert_eq!(restored.model.num_heads, config.model.num_heads);
        prop_assert_eq!(restored.model.intermediate_size, config.model.intermediate_size);
        prop_assert_eq!(
            restored.model.max_position_embeddings,
            config.model.max_position_embeddings
        );
    }

    /// Property 2 – Every config from `arb_valid_config` passes `validate()`.
    ///
    /// The strategy is designed so that all dimension invariants hold by
    /// construction; `validate()` must therefore succeed.
    #[test]
    fn valid_config_always_passes_validate(config in arb_valid_config()) {
        prop_assert!(
            config.validate().is_ok(),
            "validate() failed for config: \
             hidden={}, heads={}, layers={}, vocab={}, intermediate={}, max_pos={}",
            config.model.hidden_size,
            config.model.num_heads,
            config.model.num_layers,
            config.model.vocab_size,
            config.model.intermediate_size,
            config.model.max_position_embeddings,
        );
    }

    /// Property 3 – A JSON round-tripped config must still pass `validate()`.
    ///
    /// If a valid config serialises and deserialises without loss, the
    /// restored config must satisfy the same validation constraints.
    #[test]
    fn config_json_roundtrip_stays_valid(config in arb_valid_config()) {
        prop_assume!(config.validate().is_ok());

        let json = serde_json::to_string(&config).unwrap();
        let restored: BitNetConfig = serde_json::from_str(&json).unwrap();

        prop_assert!(
            restored.validate().is_ok(),
            "Round-tripped config failed validate()"
        );
    }

    /// Property 4 – `normalize_vendor_key` must not panic on any input.
    ///
    /// The function applies compiled regexes to the input string.  It must
    /// return `Some(_)` or `None` without panicking for any UTF-8 string,
    /// including empty strings, non-ASCII characters, and very long inputs.
    #[test]
    fn normalize_vendor_key_never_panics(key in ".*") {
        let _ = normalize_vendor_key(&key);
    }

    /// Property 5 – A config with `hidden_size = 0` always fails `validate()`.
    ///
    /// Zero-valued dimension fields are explicitly rejected by the validator.
    /// This is a negative / mutation-killing property.
    #[test]
    fn config_with_zero_hidden_size_fails_validate(
        num_heads in 1usize..=32,
        num_layers in 1usize..=32,
        vocab_size in 1usize..=32000,
    ) {
        let config = BitNetConfig {
            model: ModelConfig {
                vocab_size,
                hidden_size: 0, // deliberately invalid
                num_heads,
                num_key_value_heads: 0,
                num_layers,
                intermediate_size: 1024,
                max_position_embeddings: 512,
                ..Default::default()
            },
            ..Default::default()
        };
        prop_assert!(
            config.validate().is_err(),
            "Config with hidden_size=0 must be rejected by validate()"
        );
    }

    /// Property 6 – `GgufTensorType::from_quant_string` round-trips the canonical
    /// lower-case names for all quantised types that have a string representation.
    ///
    /// Ensures the string → type mapping does not panic and returns `Some` for the
    /// canonical aliases defined in the type implementation.
    #[test]
    fn from_quant_string_accepts_known_aliases(
        name in prop_oneof![
            Just("i2_s"),
            Just("iq2_s"),
            Just("q4_0"),
            Just("q4_1"),
            Just("q5_0"),
            Just("q5_1"),
            Just("q8_0"),
            Just("q8_1"),
            Just("q2_k"),
            Just("q3_k"),
            Just("q4_k"),
            Just("q5_k"),
            Just("q6_k"),
            Just("q8_k"),
            Just("f32"),
            Just("f16"),
            Just("f64"),
        ]
    ) {
        let result = GgufTensorType::from_quant_string(name);
        prop_assert!(result.is_some(), "from_quant_string({name:?}) must return Some(_)");
    }
}

// ── Unit-level invariants ─────────────────────────────────────────────────────

/// Every quantised `GgufTensorType` must report `is_quantized() == true` and
/// a `block_size() >= 1`.
#[test]
fn quantized_tensor_types_have_positive_block_size() {
    use GgufTensorType::*;
    let quantized =
        [Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, I2_S, IQ2_S];
    for ty in &quantized {
        assert!(ty.is_quantized(), "{ty:?} should be identified as quantized");
        assert!(ty.block_size() >= 1, "{ty:?} must have block_size >= 1");
    }
}

/// Non-quantised types (`F32`, `F16`, `F64`) must NOT be classified as quantised.
#[test]
fn float_types_not_classified_as_quantized() {
    use GgufTensorType::*;
    for ty in &[F32, F16, F64] {
        assert!(!ty.is_quantized(), "{ty:?} must not be classified as quantized");
    }
}

/// For every `I2SFlavor`:
/// * `block_size()` is positive
/// * `data_bytes_per_block()` is positive
/// * `data_bytes_per_block() <= total_bytes_per_block()`
///
/// The last constraint expresses that inline metadata (e.g. scale bytes) can
/// only add to, never subtract from, the data portion.
#[test]
fn i2s_flavor_byte_layout_invariants() {
    use I2SFlavor::*;
    for flavor in &[BitNet32F16, Split32WithSibling, GgmlQk256NoScale] {
        let block_sz = flavor.block_size();
        let data_bytes = flavor.data_bytes_per_block();
        let total_bytes = flavor.total_bytes_per_block();

        assert!(block_sz > 0, "{flavor:?}: block_size must be > 0");
        assert!(data_bytes > 0, "{flavor:?}: data_bytes_per_block must be > 0");
        assert!(
            data_bytes <= total_bytes,
            "{flavor:?}: data_bytes_per_block ({data_bytes}) \
             must be <= total_bytes_per_block ({total_bytes})"
        );
    }
}

/// The QK256 flavor (`GgmlQk256NoScale`) must have exactly 256 elements per
/// block, matching the `QK256_BLOCK` constant used in the kernel implementations.
#[test]
fn qk256_flavor_has_exactly_256_elements_per_block() {
    assert_eq!(
        I2SFlavor::GgmlQk256NoScale.block_size(),
        256,
        "QK256 flavor must have exactly 256 elements per block"
    );
}
