//! Property-based tests — wave 35.
//!
//! 15 properties covering GGUF header field validation, tensor name parsing
//! invariants, registry filter composition, and model metadata round-trip.

#![cfg(all(test, feature = "cpu"))]

use bitnet_gguf::{GGUF_MAGIC, check_magic, read_version};
use bitnet_models::formats::gguf::{GgufTensorType, I2SFlavor};
use bitnet_models::model_fingerprint::ModelFingerprint;
use bitnet_models::names::{is_layernorm_weight, is_projection_weight};
use bitnet_models::registry_query::{RegistryEntry, RegistryFilter, query};
use proptest::prelude::*;

// -------------------------------------------------------------------
// Strategy helpers
// -------------------------------------------------------------------

fn any_tensor_type() -> impl Strategy<Value = GgufTensorType> {
    prop_oneof![
        Just(GgufTensorType::F32),
        Just(GgufTensorType::F16),
        Just(GgufTensorType::F64),
        Just(GgufTensorType::Q4_0),
        Just(GgufTensorType::Q4_1),
        Just(GgufTensorType::Q5_0),
        Just(GgufTensorType::Q5_1),
        Just(GgufTensorType::Q8_0),
        Just(GgufTensorType::Q8_1),
        Just(GgufTensorType::Q2_K),
        Just(GgufTensorType::Q3_K),
        Just(GgufTensorType::Q4_K),
        Just(GgufTensorType::Q5_K),
        Just(GgufTensorType::Q6_K),
        Just(GgufTensorType::Q8_K),
        Just(GgufTensorType::IQ2_S),
        Just(GgufTensorType::I2_S),
    ]
}

fn sample_registry_entry(arch: &str, params: u64, qt: &str, fmt: &str) -> RegistryEntry {
    RegistryEntry {
        id: format!("{arch}-{qt}"),
        name: format!("Test {arch}"),
        architecture: arch.to_string(),
        param_count: params,
        quant_type: qt.to_string(),
        format: fmt.to_string(),
        tags: vec![arch.to_string()],
    }
}

// ===================================================================
// 1–4. GGUF header field validation
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Valid GGUF magic bytes pass check_magic.
    #[test]
    fn prop_gguf_valid_magic_passes(_dummy in 0..1i32) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 64]);
        prop_assert!(check_magic(&buf));
    }

    /// Mutated magic bytes fail check_magic.
    #[test]
    fn prop_gguf_mutated_magic_fails(byte_idx in 0usize..4, flip in 1u8..=255) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 64]);
        buf[byte_idx] ^= flip;
        // Only assert failure if the magic is no longer valid
        if buf[..4] != GGUF_MAGIC {
            prop_assert!(!check_magic(&buf));
        }
    }

    /// read_version returns version for valid header data.
    #[test]
    fn prop_gguf_version_roundtrip(version in 2u32..=3) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&[0u8; 64]);
        let v = read_version(&buf);
        prop_assert_eq!(v, Some(version));
    }

    /// read_version returns None for undersized buffers.
    #[test]
    fn prop_gguf_version_undersized(len in 0usize..8) {
        let buf = vec![0u8; len];
        prop_assert!(read_version(&buf).is_none());
    }
}

// ===================================================================
// 5–7. Tensor name parsing invariants
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Known LayerNorm names are detected.
    #[test]
    fn prop_known_ln_names_detected(
        prefix in "[a-z]{1,10}",
        suffix in prop_oneof![
            Just(".norm.weight"),
            Just(".input_layernorm.weight"),
            Just(".post_attention_layernorm.weight"),
            Just(".attn_norm.weight"),
            Just(".ffn_norm.weight"),
            Just(".rms_norm.weight"),
        ],
    ) {
        let name = format!("{prefix}{suffix}");
        prop_assert!(
            is_layernorm_weight(&name),
            "'{}' should be detected as LN", name
        );
    }

    /// Known projection names are detected.
    #[test]
    fn prop_known_proj_names_detected(
        prefix in "[a-z]{1,10}",
        suffix in prop_oneof![
            Just(".q_proj.weight"),
            Just(".k_proj.weight"),
            Just(".v_proj.weight"),
            Just(".o_proj.weight"),
            Just(".attn_q.weight"),
            Just(".attn_k.weight"),
        ],
    ) {
        let name = format!("{prefix}{suffix}");
        prop_assert!(
            is_projection_weight(&name),
            "'{}' should be detected as projection", name
        );
    }

    /// LN and projection predicates are mutually exclusive.
    #[test]
    fn prop_ln_proj_exclusive(name in "\\PC{0,60}") {
        let is_ln = is_layernorm_weight(&name);
        let is_proj = is_projection_weight(&name);
        prop_assert!(
            !(is_ln && is_proj),
            "'{}' classified as both LN and projection", name
        );
    }
}

// ===================================================================
// 8–11. Registry filter composition
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Empty filter matches all entries.
    #[test]
    fn prop_empty_filter_matches_all(
        n_entries in 1usize..=10,
    ) {
        let entries: Vec<RegistryEntry> = (0..n_entries)
            .map(|i| sample_registry_entry("bitnet", (i as u64 + 1) * 1_000_000, "i2s", "gguf"))
            .collect();
        let filter = RegistryFilter::new();
        let result = query(&entries, &filter);
        prop_assert_eq!(result.len(), entries.len());
    }

    /// Architecture filter restricts results correctly.
    #[test]
    fn prop_arch_filter_restricts(arch in "[a-z]{3,8}") {
        let matching = sample_registry_entry(&arch, 1_000_000, "i2s", "gguf");
        let other = sample_registry_entry("different", 1_000_000, "i2s", "gguf");
        let entries = vec![matching, other];
        let filter = RegistryFilter::new().with_architecture(&arch);
        let result = query(&entries, &filter);
        prop_assert!(
            result.len() <= 1,
            "arch filter should restrict to matching entries"
        );
        for e in &result {
            prop_assert!(
                e.architecture.eq_ignore_ascii_case(&arch),
                "filtered entry arch='{}' doesn't match '{}'", e.architecture, arch
            );
        }
    }

    /// Param range filter respects min/max bounds.
    #[test]
    fn prop_param_range_filter(
        min_params in 100_000u64..500_000,
        max_params in 500_001u64..2_000_000,
    ) {
        let entries = vec![
            sample_registry_entry("a", 50_000, "i2s", "gguf"),     // too small
            sample_registry_entry("b", 600_000, "i2s", "gguf"),    // in range
            sample_registry_entry("c", 5_000_000, "i2s", "gguf"),  // too large
        ];
        let filter = RegistryFilter::new()
            .with_min_params(min_params)
            .with_max_params(max_params);
        let result = query(&entries, &filter);
        for e in &result {
            prop_assert!(e.param_count >= min_params, "entry below min");
            prop_assert!(e.param_count <= max_params, "entry above max");
        }
    }

    /// Composing two filters is more restrictive than either alone.
    #[test]
    fn prop_filter_composition_restrictive(
        arch in prop_oneof![Just("bitnet"), Just("llama")],
        qt in prop_oneof![Just("i2s"), Just("f16")],
    ) {
        let entries = vec![
            sample_registry_entry("bitnet", 1_000_000, "i2s", "gguf"),
            sample_registry_entry("bitnet", 2_000_000, "f16", "gguf"),
            sample_registry_entry("llama", 1_000_000, "i2s", "gguf"),
            sample_registry_entry("llama", 2_000_000, "f16", "safetensors"),
        ];
        let f_arch = RegistryFilter::new().with_architecture(&arch);
        let f_both = RegistryFilter::new().with_architecture(&arch).with_quant(&qt);
        let r_arch = query(&entries, &f_arch);
        let r_both = query(&entries, &f_both);
        prop_assert!(
            r_both.len() <= r_arch.len(),
            "composed filter should be <= arch filter: {} > {}", r_both.len(), r_arch.len()
        );
    }
}

// ===================================================================
// 12–15. Model metadata round-trip
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// ModelFingerprint compact_id contains architecture name.
    #[test]
    fn prop_fingerprint_compact_id_contains_arch(arch in "[a-zA-Z]{3,10}") {
        let fp = ModelFingerprint::new(&arch);
        let id = fp.compact_id();
        prop_assert!(
            id.contains(&arch),
            "compact_id '{}' should contain arch '{}'", id, arch
        );
    }

    /// ModelFingerprint same_architecture is reflexive.
    #[test]
    fn prop_fingerprint_same_arch_reflexive(
        arch in "[a-z]{3,8}",
        layers in 1u32..=96,
        hidden in 64u32..=8192,
        heads in 1u32..=64,
    ) {
        let fp = ModelFingerprint::new(&arch)
            .with_layers(layers)
            .with_hidden_size(hidden)
            .with_heads(heads);
        prop_assert!(fp.same_architecture(&fp));
    }

    /// I2SFlavor block_size matches expected values per variant.
    #[test]
    fn prop_i2s_flavor_block_sizes(
        flavor in prop_oneof![
            Just(I2SFlavor::BitNet32F16),
            Just(I2SFlavor::Split32WithSibling),
            Just(I2SFlavor::GgmlQk256NoScale),
        ],
    ) {
        let bs: usize = flavor.block_size();
        match flavor {
            I2SFlavor::BitNet32F16 | I2SFlavor::Split32WithSibling => {
                prop_assert_eq!(bs, 32);
            }
            I2SFlavor::GgmlQk256NoScale => {
                prop_assert_eq!(bs, 256);
            }
        }
        // total_bytes_per_block >= data_bytes_per_block
        let total: usize = flavor.total_bytes_per_block();
        let data: usize = flavor.data_bytes_per_block();
        prop_assert!(
            total >= data,
            "total < data for {:?}", flavor
        );
    }

    /// GgufTensorType element_size is positive for all known types.
    #[test]
    fn prop_tensor_type_positive_size(t in any_tensor_type()) {
        let size: usize = t.element_size();
        prop_assert!(size > 0, "{:?} element_size={}", t, size);
        let bs: usize = t.block_size();
        prop_assert!(bs >= 1, "{:?} block_size={}", t, bs);
    }
}
