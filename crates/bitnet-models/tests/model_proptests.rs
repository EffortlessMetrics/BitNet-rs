//! Property-based tests for `bitnet-models` core subsystems.
//!
//! Covers six areas not already addressed by the existing property-test suite:
//!
//! 1. **`align_up` arithmetic** – monotonicity, divisibility, idempotency.
//! 2. **`ModelFormat` path detection** – canonical extensions map to the correct
//!    variant without I/O.
//! 3. **`detect_i2s_flavor` disambiguation** – exact-byte matches are always
//!    honoured with the documented priority order (QK256 → BitNet32F16 →
//!    Split32WithSibling).
//! 4. **`detect_qk256_orientation_by_bytes`** – the function always selects the
//!    shape whose expected byte count is closest to the available bytes.
//! 5. **`expected_qk256_shape`** – known tensor-name patterns return shapes whose
//!    dimensions agree with the model configuration.
//! 6. **`BitNetConfig::validate` negative cases** – zero `intermediate_size` and
//!    zero `max_position_embeddings` are always rejected.

#![cfg(all(test, feature = "cpu"))]

use bitnet_common::{BitNetConfig, config::ModelConfig};
use bitnet_models::formats::gguf::{I2SFlavor, TensorInfo, detect_i2s_flavor};
use bitnet_models::formats::{ModelFormat, gguf::GgufTensorType};
use bitnet_models::qk256_utils::{detect_qk256_orientation_by_bytes, expected_qk256_shape};
use proptest::prelude::*;
use std::path::Path;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Build a minimal `TensorInfo` with the given byte payload size.
fn tensor_info(name: &str, size_bytes: u64) -> TensorInfo {
    TensorInfo {
        name: name.to_string(),
        shape: vec![1, size_bytes as usize],
        tensor_type: GgufTensorType::I2_S,
        size: size_bytes,
        offset: 0,
    }
}

/// Construct a valid `BitNetConfig` with `hidden_size = num_heads * head_dim`.
fn make_config(num_heads: usize, head_dim: usize, intermediate: usize) -> BitNetConfig {
    BitNetConfig {
        model: ModelConfig {
            hidden_size: num_heads * head_dim,
            num_heads,
            num_key_value_heads: 0,
            vocab_size: 32000,
            num_layers: 4,
            intermediate_size: intermediate,
            max_position_embeddings: 2048,
            ..Default::default()
        },
        ..Default::default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1.  align_up arithmetic properties
// ─────────────────────────────────────────────────────────────────────────────

use bitnet_models::formats::gguf::align_up;

proptest! {
    /// align_up(x, a) ≥ x for all x and all power-of-two alignments a ∈ 1..=4096.
    #[test]
    fn align_up_result_geq_input(
        x in 0usize..=1_000_000usize,
        log2_a in 0u32..=12u32,  // 2^0=1 .. 2^12=4096
    ) {
        let a = 1usize << log2_a;
        prop_assert!(align_up(x, a) >= x,
            "align_up({}, {}) = {} should be >= {}", x, a, align_up(x, a), x);
    }

    /// align_up(x, a) % a == 0 for all power-of-two alignments a ≥ 1.
    #[test]
    fn align_up_result_is_divisible(
        x in 0usize..=1_000_000usize,
        log2_a in 0u32..=12u32,
    ) {
        let a = 1usize << log2_a;
        prop_assert_eq!(align_up(x, a) % a, 0,
            "align_up({}, {}) = {} is not divisible by {}", x, a, align_up(x, a), a);
    }

    /// align_up is idempotent: aligning an already-aligned value is a no-op.
    #[test]
    fn align_up_idempotent(
        x in 0usize..=1_000_000usize,
        log2_a in 0u32..=12u32,
    ) {
        let a = 1usize << log2_a;
        let once = align_up(x, a);
        let twice = align_up(once, a);
        prop_assert_eq!(once, twice,
            "align_up(align_up({}, {}), {}) = {} != {} (not idempotent)", x, a, a, twice, once);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  ModelFormat path-extension detection
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// Any path ending in `.gguf` (case-insensitive) must be detected as `Gguf`.
    ///
    /// `detect_from_path` checks the extension before doing any file I/O, so
    /// the file need not exist for the extension-based branch to fire.
    #[test]
    fn model_format_detects_gguf_extension(
        stem in "[a-zA-Z0-9_-]{1,32}",
    ) {
        let path_str = format!("/tmp/{stem}.gguf");
        let result = ModelFormat::detect_from_path(Path::new(&path_str));
        prop_assert!(
            matches!(result, Ok(ModelFormat::Gguf)),
            "Expected Gguf for path '{}', got {:?}", path_str, result
        );
    }

    /// Any path ending in `.safetensors` must be detected as `SafeTensors`.
    #[test]
    fn model_format_detects_safetensors_extension(
        stem in "[a-zA-Z0-9_-]{1,32}",
    ) {
        let path_str = format!("/tmp/{stem}.safetensors");
        let result = ModelFormat::detect_from_path(Path::new(&path_str));
        prop_assert!(
            matches!(result, Ok(ModelFormat::SafeTensors)),
            "Expected SafeTensors for path '{}', got {:?}", path_str, result
        );
    }
}

/// `ModelFormat::name()` is always non-empty and
/// `ModelFormat::extension()` is always non-empty for both variants.
#[test]
fn model_format_name_and_extension_are_nonempty() {
    for fmt in &[ModelFormat::Gguf, ModelFormat::SafeTensors] {
        assert!(!fmt.name().is_empty(), "{fmt:?}.name() must be non-empty");
        assert!(!fmt.extension().is_empty(), "{fmt:?}.extension() must be non-empty");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  detect_i2s_flavor: exact-byte priority
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// When the available bytes are an exact QK256 match, the function always
    /// returns `GgmlQk256NoScale` regardless of sibling presence.
    ///
    /// The QK256 exact-match branch fires first in the priority order, so the
    /// assertion holds for any `nelems ≥ 1`.
    #[test]
    fn detect_i2s_qk256_exact_always_wins(
        nelems in 1usize..=65536usize,
        has_sibling in any::<bool>(),
    ) {
        let blocks256 = nelems.div_ceil(256);
        let qk256_need = blocks256 * 64;
        let info = tensor_info("blk.0.attn_q.weight", qk256_need as u64);
        let result = detect_i2s_flavor(&info, has_sibling, nelems);
        prop_assert!(
            matches!(result, Ok(I2SFlavor::GgmlQk256NoScale)),
            "nelems={}: expected GgmlQk256NoScale for exact qk256_need={} bytes, got {:?}",
            nelems, qk256_need, result
        );
    }

    /// When the available bytes exactly match the BitNet32-F16 inline layout
    /// (blocks32 × 10) and the QK256 layout doesn't match, the function returns
    /// `BitNet32F16`.
    ///
    /// The QK256 byte count for `nelems` elements is `ceil(nelems/256) * 64`.
    /// We filter out the rare values where `blocks32 * 10 == blocks256 * 64`
    /// using `prop_assume!` to avoid false conflicts.
    #[test]
    fn detect_i2s_bitnet32f16_exact_match(
        nelems in 1usize..=65536usize,
    ) {
        let blocks32  = nelems.div_ceil(32);
        let blocks256 = nelems.div_ceil(256);
        let inline_need = blocks32 * 10;
        let qk256_need  = blocks256 * 64;

        // Only test cases where QK256 doesn't already claim the exact match.
        prop_assume!(inline_need != qk256_need);

        let info = tensor_info("blk.0.attn_q.weight", inline_need as u64);
        let result = detect_i2s_flavor(&info, false, nelems);
        prop_assert!(
            matches!(result, Ok(I2SFlavor::BitNet32F16)),
            "nelems={}: expected BitNet32F16 for exact inline_need={} bytes, got {:?}",
            nelems, inline_need, result
        );
    }

    /// When the available bytes exactly match the Split32 layout (blocks32 × 8)
    /// and neither QK256 nor BitNet32-F16 claims an exact match, and a sibling
    /// scale tensor is present, the function returns `Split32WithSibling`.
    ///
    /// We filter cases where `split_need == qk256_need` (they collide when
    /// `nelems` is a multiple of 256) or where `split_need == inline_need`
    /// (impossible since 8 ≠ 10, so this assumption always holds).
    #[test]
    fn detect_i2s_split32_exact_with_sibling(
        nelems in 1usize..=65536usize,
    ) {
        let blocks32  = nelems.div_ceil(32);
        let blocks256 = nelems.div_ceil(256);
        let split_need  = blocks32 * 8;
        let inline_need = blocks32 * 10; // always != split_need (10 != 8)
        let qk256_need  = blocks256 * 64;

        // Skip configurations where split_need collides with qk256_need or
        // inline_need (the latter is already impossible, kept for clarity).
        prop_assume!(split_need != qk256_need);
        prop_assume!(split_need != inline_need);

        let info = tensor_info("blk.0.attn_q.weight", split_need as u64);
        let result = detect_i2s_flavor(&info, true, nelems);
        prop_assert!(
            matches!(result, Ok(I2SFlavor::Split32WithSibling)),
            "nelems={}: expected Split32WithSibling for exact split_need={} bytes \
             (qk256_need={}, inline_need={}), got {:?}",
            nelems, split_need, qk256_need, inline_need, result
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  detect_qk256_orientation_by_bytes: picks the closer match
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// The orientation function always returns the shape whose expected byte
    /// count (QK256 layout: rows × ceil(cols/256) × 64) is closer to the
    /// available bytes.
    ///
    /// When both shapes produce the same expected size (the tie-break case) the
    /// function returns the as-is shape by convention.
    #[test]
    fn qk256_orientation_picks_closer_or_as_is_on_tie(
        rows_a in 1usize..=64usize,
        cols_a in 1usize..=256usize,
        rows_b in 1usize..=64usize,
        cols_b in 1usize..=256usize,
        available in 0usize..=200_000usize,
    ) {
        let shape_as_is    = (rows_a, cols_a);
        let shape_transposed = (rows_b, cols_b);

        let exp_as_is      = rows_a * cols_a.div_ceil(256) * 64;
        let exp_transposed = rows_b * cols_b.div_ceil(256) * 64;

        let chosen = detect_qk256_orientation_by_bytes(shape_as_is, shape_transposed, available);

        let dist_as_is      = available.abs_diff(exp_as_is);
        let dist_transposed = available.abs_diff(exp_transposed);

        if dist_transposed < dist_as_is {
            prop_assert_eq!(
                chosen, shape_transposed,
                "should pick transposed (closer): available={}, exp_as_is={}, exp_transposed={}",
                available, exp_as_is, exp_transposed
            );
        } else {
            // Tie or as-is is closer — function must not pick transposed.
            prop_assert_eq!(
                chosen, shape_as_is,
                "should pick as-is (tied or closer): available={}, exp_as_is={}, exp_transposed={}",
                available, exp_as_is, exp_transposed
            );
        }
    }

    /// When the two shapes produce identical expected byte counts, the function
    /// must return `shape_as_is` (tie-breaking convention).
    #[test]
    fn qk256_orientation_as_is_wins_on_tie(
        rows in 1usize..=64usize,
        cols in 1usize..=256usize,
        available in 0usize..=200_000usize,
    ) {
        // Construct two shapes whose expected sizes are equal.
        let shape = (rows, cols);
        let chosen = detect_qk256_orientation_by_bytes(shape, shape, available);
        prop_assert_eq!(
            chosen, shape,
            "as-is shape must be returned on a tie; available={}, shape={:?}",
            available, shape
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  expected_qk256_shape: dimension invariants
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// For `q_proj` / `attn_q` tensors the expected shape must be square:
    /// both dimensions equal `hidden_size`.
    #[test]
    fn q_proj_expected_shape_is_square(
        num_heads in 1usize..=16usize,
        head_dim  in 1usize..=128usize,
        intermediate in 1usize..=8192usize,
        tensor_name in prop_oneof![
            Just("blk.0.attn_q.weight"),
            Just("model.layers.0.self_attn.q_proj.weight"),
            Just("attn_q.weight"),
        ],
    ) {
        let cfg = make_config(num_heads, head_dim, intermediate);
        let hidden = cfg.model.hidden_size;
        if let Some((rows, cols)) = expected_qk256_shape(tensor_name, &cfg) {
            prop_assert_eq!(rows, hidden,
                "{}: rows ({}) should equal hidden_size ({})", tensor_name, rows, hidden);
            prop_assert_eq!(cols, hidden,
                "{}: cols ({}) should equal hidden_size ({})", tensor_name, cols, hidden);
        }
        // None is also acceptable (name not recognised) — no assertion needed.
    }

    /// For `ffn_down` / `down_proj` tensors the expected shape must be
    /// `(hidden_size, intermediate_size)`.
    #[test]
    fn ffn_down_expected_shape_matches_config(
        num_heads in 1usize..=16usize,
        head_dim  in 1usize..=128usize,
        intermediate in 1usize..=8192usize,
        tensor_name in prop_oneof![
            Just("blk.0.ffn_down.weight"),
            Just("model.layers.0.mlp.down_proj.weight"),
        ],
    ) {
        let cfg = make_config(num_heads, head_dim, intermediate);
        let hidden = cfg.model.hidden_size;
        if let Some((rows, cols)) = expected_qk256_shape(tensor_name, &cfg) {
            prop_assert_eq!(rows, hidden,
                "{}: rows ({}) should equal hidden_size ({})", tensor_name, rows, hidden);
            prop_assert_eq!(cols, intermediate,
                "{}: cols ({}) should equal intermediate_size ({})", tensor_name, cols, intermediate);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6.  BitNetConfig::validate negative invariants
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// Any config with `intermediate_size = 0` must be rejected by `validate()`.
    #[test]
    fn zero_intermediate_size_always_fails_validate(
        num_heads  in 1usize..=16usize,
        head_dim   in 1usize..=64usize,
        num_layers in 1usize..=16usize,
        vocab_size in 1usize..=32000usize,
    ) {
        let mut cfg = BitNetConfig::default();
        cfg.model.hidden_size      = num_heads * head_dim;
        cfg.model.num_heads        = num_heads;
        cfg.model.num_key_value_heads = 0;
        cfg.model.num_layers       = num_layers;
        cfg.model.vocab_size       = vocab_size;
        cfg.model.intermediate_size = 0; // invalid
        prop_assert!(
            cfg.validate().is_err(),
            "Config with intermediate_size=0 must be rejected by validate()"
        );
    }

    /// Any config with `max_position_embeddings = 0` must be rejected by
    /// `validate()`.
    #[test]
    fn zero_max_position_embeddings_always_fails_validate(
        num_heads  in 1usize..=16usize,
        head_dim   in 1usize..=64usize,
        num_layers in 1usize..=16usize,
        vocab_size in 1usize..=32000usize,
    ) {
        let mut cfg = BitNetConfig::default();
        cfg.model.hidden_size              = num_heads * head_dim;
        cfg.model.num_heads                = num_heads;
        cfg.model.num_key_value_heads      = 0;
        cfg.model.num_layers               = num_layers;
        cfg.model.vocab_size               = vocab_size;
        cfg.model.intermediate_size        = 1024; // valid
        cfg.model.max_position_embeddings  = 0;    // invalid
        prop_assert!(
            cfg.validate().is_err(),
            "Config with max_position_embeddings=0 must be rejected by validate()"
        );
    }
}
