//! Property-based tests — wave 36.
//!
//! Covers bitnet-models: GGUF header parsing determinism, align_up properties,
//! model architecture detection consistency, tensor name predicates,
//! and config validation.

use bitnet_models::architecture::{
    ModelArchitecture, detect_architecture, get_defaults, supported_architectures,
};
use bitnet_models::formats::gguf::{GgufHeader, align_up};
use bitnet_models::names::{is_layernorm_weight, is_projection_weight};
use proptest::prelude::*;

// ── Strategies ──────────────────────────────────────────────────────────────

fn arb_alignment() -> impl Strategy<Value = usize> {
    prop_oneof![Just(1usize), Just(2), Just(4), Just(8), Just(16), Just(32), Just(64), Just(128),]
}

fn arb_layer_index() -> impl Strategy<Value = u32> {
    0u32..128
}

fn arb_known_architecture() -> impl Strategy<Value = &'static str> {
    prop_oneof![
        Just("bitnet"),
        Just("BitNet"),
        Just("phi"),
        Just("Phi-3"),
        Just("phi4"),
        Just("qwen"),
        Just("Qwen2"),
        Just("gemma"),
        Just("Gemma-2"),
        Just("mistral"),
        Just("Mixtral"),
        Just("llama"),
        Just("LLaMA-3"),
        Just("smollm"),
        Just("falcon"),
        Just("mpt"),
        Just("bloom"),
        Just("stablelm"),
        Just("tinyllama"),
        Just("deepseek"),
        Just("codellama"),
        Just("starcoder"),
    ]
}

// ── Property tests ──────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // ════════════════════════════════════════════════════════════════
    // 1. align_up properties
    // ════════════════════════════════════════════════════════════════

    /// align_up result >= input offset.
    #[test]
    fn prop_align_up_ge_input(off in 0usize..100_000, align in arb_alignment()) {
        let result = align_up(off, align);
        prop_assert!(result >= off, "align_up({}, {}) = {} should be >= {}", off, align, result, off);
    }

    /// align_up result is a multiple of alignment.
    #[test]
    fn prop_align_up_multiple(off in 0usize..100_000, align in arb_alignment()) {
        let result = align_up(off, align);
        prop_assert_eq!(result % align, 0, "align_up({}, {}) = {} not aligned", off, align, result);
    }

    /// align_up of already-aligned value is identity.
    #[test]
    fn prop_align_up_idempotent(off in 0usize..100_000, align in arb_alignment()) {
        let once = align_up(off, align);
        let twice = align_up(once, align);
        prop_assert_eq!(once, twice, "align_up should be idempotent");
    }

    /// align_up with alignment=1 is identity.
    #[test]
    fn prop_align_up_one_identity(off in 0usize..100_000) {
        prop_assert_eq!(align_up(off, 1), off);
    }

    /// align_up(0, align) == 0.
    #[test]
    fn prop_align_up_zero(align in arb_alignment()) {
        prop_assert_eq!(align_up(0, align), 0);
    }

    /// align_up overflows are bounded — result <= off + align - 1.
    #[test]
    fn prop_align_up_bounded(off in 0usize..100_000, align in arb_alignment()) {
        let result = align_up(off, align);
        prop_assert!(result < off + align, "align_up should add at most align-1 bytes");
    }

    // ════════════════════════════════════════════════════════════════
    // 2. GGUF header parsing determinism
    // ════════════════════════════════════════════════════════════════

    /// Parsing the same GGUF header bytes twice yields identical results.
    #[test]
    fn prop_gguf_header_deterministic(
        version in prop_oneof![Just(2u32), Just(3u32)],
        tensor_count in 0u64..1000,
        metadata_count in 0u64..1000
    ) {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&metadata_count.to_le_bytes());
        // For v3, add alignment + data_offset + a fake KV pair header
        // to avoid heuristic confusion
        if version >= 3 {
            // Add enough bytes so v3 heuristic can work
            // Pad with zeros (non-ASCII, so heuristic won't think it's a KV key)
            buf.extend_from_slice(&[0u8; 64]);
        }

        let mut off1 = 0;
        let h1 = GgufHeader::read(&buf, &mut off1);
        let mut off2 = 0;
        let h2 = GgufHeader::read(&buf, &mut off2);

        match (h1, h2) {
            (Ok(a), Ok(b)) => {
                prop_assert_eq!(a.version, b.version);
                prop_assert_eq!(a.tensor_count, b.tensor_count);
                prop_assert_eq!(a.metadata_kv_count, b.metadata_kv_count);
                prop_assert_eq!(off1, off2, "offsets should match");
            }
            (Err(_), Err(_)) => { /* both fail — fine */ }
            (Ok(_), Err(e)) => prop_assert!(false, "first succeeded but second failed: {}", e),
            (Err(e), Ok(_)) => prop_assert!(false, "first failed but second succeeded: {}", e),
        }
    }

    /// GGUF header rejects invalid magic bytes.
    #[test]
    fn prop_gguf_rejects_bad_magic(
        byte0 in any::<u8>(),
        byte1 in any::<u8>(),
        byte2 in any::<u8>(),
        byte3 in any::<u8>()
    ) {
        prop_assume!([byte0, byte1, byte2, byte3] != *b"GGUF");
        let mut buf = vec![byte0, byte1, byte2, byte3];
        buf.extend_from_slice(&[0u8; 28]); // enough for header
        let mut off = 0;
        let result = GgufHeader::read(&buf, &mut off);
        prop_assert!(result.is_err(), "should reject non-GGUF magic");
    }

    /// GGUF header rejects too many tensors (>100K).
    #[test]
    fn prop_gguf_rejects_too_many_tensors(
        tensor_count in 100_001u64..200_000
    ) {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&2u32.to_le_bytes()); // version 2
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // metadata count
        buf.extend_from_slice(&[0u8; 32]);

        let mut off = 0;
        let result = GgufHeader::read(&buf, &mut off);
        prop_assert!(result.is_err(), "should reject tensor_count > 100K");
    }

    // ════════════════════════════════════════════════════════════════
    // 3. Model architecture detection
    // ════════════════════════════════════════════════════════════════

    /// detect_architecture is deterministic.
    #[test]
    fn prop_detect_arch_deterministic(name in arb_known_architecture()) {
        let a1 = detect_architecture(name);
        let a2 = detect_architecture(name);
        prop_assert_eq!(a1, a2, "detect_architecture should be deterministic");
    }

    /// detect_architecture for known families never returns Unknown.
    #[test]
    fn prop_detect_arch_known(name in arb_known_architecture()) {
        let arch = detect_architecture(name);
        prop_assert!(
            !matches!(arch, ModelArchitecture::Unknown(_)),
            "known family '{}' should not map to Unknown, got {:?}",
            name, arch
        );
    }

    /// get_defaults returns valid config for all architectures.
    #[test]
    fn prop_defaults_valid(name in arb_known_architecture()) {
        let arch = detect_architecture(name);
        let defaults = get_defaults(&arch);
        prop_assert!(defaults.rope_base > 0.0, "rope_base should be positive");
        prop_assert!(defaults.vocab_size > 0, "vocab_size should be positive");
        prop_assert!(defaults.typical_hidden_size > 0, "hidden_size should be positive");
    }

    /// supported_architectures is non-empty and stable across calls.
    #[test]
    fn prop_supported_arch_stable(_dummy in 0u8..1) {
        let a = supported_architectures();
        let b = supported_architectures();
        prop_assert!(!a.is_empty());
        prop_assert_eq!(a.len(), b.len());
    }

    // ════════════════════════════════════════════════════════════════
    // 4. Tensor name predicates
    // ════════════════════════════════════════════════════════════════

    /// LayerNorm weight names are recognized for any layer index.
    #[test]
    fn prop_layernorm_any_layer(layer in arb_layer_index()) {
        let names = [
            format!("blk.{}.attn_norm.weight", layer),
            format!("blk.{}.ffn_norm.weight", layer),
            format!("blk.{}.input_layernorm.weight", layer),
            format!("blk.{}.post_attention_layernorm.weight", layer),
            format!("blk.{}.rms_norm.weight", layer),
            format!("blk.{}.norm.weight", layer),
        ];
        for name in &names {
            prop_assert!(is_layernorm_weight(name), "'{}' should be LayerNorm", name);
        }
    }

    /// Projection weight names are recognized for any layer index.
    #[test]
    fn prop_projection_any_layer(layer in arb_layer_index()) {
        let names = [
            format!("blk.{}.attn_q.weight", layer),
            format!("blk.{}.attn_k.weight", layer),
            format!("blk.{}.attn_v.weight", layer),
            format!("blk.{}.attn_output.weight", layer),
            format!("blk.{}.ffn_gate.weight", layer),
            format!("blk.{}.ffn_up.weight", layer),
            format!("blk.{}.ffn_down.weight", layer),
        ];
        for name in &names {
            prop_assert!(is_projection_weight(name), "'{}' should be projection", name);
        }
    }

    /// LayerNorm and projection predicates are mutually exclusive.
    #[test]
    fn prop_ln_proj_exclusive(layer in arb_layer_index()) {
        let ln_name = format!("blk.{}.attn_norm.weight", layer);
        let proj_name = format!("blk.{}.attn_q.weight", layer);

        prop_assert!(is_layernorm_weight(&ln_name));
        prop_assert!(!is_projection_weight(&ln_name));

        prop_assert!(is_projection_weight(&proj_name));
        prop_assert!(!is_layernorm_weight(&proj_name));
    }

    /// final_norm.weight is always LayerNorm.
    #[test]
    fn prop_final_norm_is_ln(_dummy in 0u8..1) {
        prop_assert!(is_layernorm_weight("final_norm.weight"));
        prop_assert!(!is_projection_weight("final_norm.weight"));
    }
}
