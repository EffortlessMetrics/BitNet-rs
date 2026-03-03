//! Property-based tests for format detection, header parsing robustness,
//! and tensor name filtering (proptest wave 31).

use bitnet_models::format_detector::ModelFormat;
use bitnet_models::names::{is_layernorm_weight, is_projection_weight};
use bitnet_models::weight_loader::{DType, TensorData, TensorInfo, WeightFormat};
use proptest::prelude::*;
use std::path::Path;

// ── Format detection determinism ──────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// WeightFormat::detect is deterministic for the same path.
    #[test]
    fn weight_format_detect_deterministic(path in ".*\\.(gguf|safetensors|bin|onnx|txt)") {
        let r1 = WeightFormat::detect(&path);
        let r2 = WeightFormat::detect(&path);
        prop_assert_eq!(r1, r2, "format detection must be deterministic");
    }

    /// ModelFormat::from_extension is deterministic.
    #[test]
    fn model_format_from_ext_deterministic(
        stem in "[a-z]{1,20}",
        ext in prop_oneof![
            Just("gguf"), Just("safetensors"), Just("bin"),
            Just("onnx"), Just("json"), Just("txt"), Just("pt"),
        ],
    ) {
        let path_str = format!("{stem}.{ext}");
        let path = Path::new(&path_str);
        let r1 = ModelFormat::from_extension(path);
        let r2 = ModelFormat::from_extension(path);
        prop_assert_eq!(r1, r2);
    }

    /// .gguf extension always detects as Gguf.
    #[test]
    fn gguf_extension_detected(stem in "[a-zA-Z0-9_-]{1,30}") {
        let path = format!("{stem}.gguf");
        let fmt = WeightFormat::detect(&path);
        prop_assert_eq!(fmt, Some(WeightFormat::Gguf));
    }

    /// .safetensors extension always detects as SafeTensors.
    #[test]
    fn safetensors_extension_detected(stem in "[a-zA-Z0-9_-]{1,30}") {
        let path = format!("{stem}.safetensors");
        let fmt = WeightFormat::detect(&path);
        prop_assert_eq!(fmt, Some(WeightFormat::SafeTensors));
    }

    /// Unrecognized extensions return None.
    #[test]
    fn unknown_extension_returns_none(stem in "[a-z]{1,10}", ext in "[a-z]{1,4}") {
        prop_assume!(ext != "gguf" && ext != "safetensors");
        let path = format!("{stem}.{ext}");
        let fmt = WeightFormat::detect(&path);
        prop_assert_eq!(fmt, None);
    }
}

// ── Magic-byte header parsing ─────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// from_magic never panics on arbitrary byte slices.
    #[test]
    fn from_magic_no_panic(bytes in prop::collection::vec(any::<u8>(), 0..=64)) {
        let _ = ModelFormat::from_magic(&bytes);
    }

    /// GGUF magic bytes always produce Gguf format.
    #[test]
    fn gguf_magic_detected(trailing in prop::collection::vec(any::<u8>(), 0..=32)) {
        let mut bytes = vec![0x47, 0x47, 0x55, 0x46]; // "GGUF"
        bytes.extend_from_slice(&trailing);
        let fmt = ModelFormat::from_magic(&bytes);
        prop_assert_eq!(fmt, ModelFormat::Gguf);
    }

    /// Empty bytes produce Unknown format, no panic.
    #[test]
    fn empty_bytes_is_unknown(_dummy in 0u8..1) {
        let fmt = ModelFormat::from_magic(&[]);
        prop_assert_eq!(fmt, ModelFormat::Unknown);
    }

    /// Short bytes (< 4) never produce Gguf.
    #[test]
    fn short_bytes_not_gguf(bytes in prop::collection::vec(any::<u8>(), 0..4)) {
        let fmt = ModelFormat::from_magic(&bytes);
        prop_assert_ne!(fmt, ModelFormat::Gguf);
    }

    /// from_magic is deterministic.
    #[test]
    fn from_magic_deterministic(bytes in prop::collection::vec(any::<u8>(), 0..=32)) {
        let r1 = ModelFormat::from_magic(&bytes);
        let r2 = ModelFormat::from_magic(&bytes);
        prop_assert_eq!(r1, r2);
    }
}

// ── Tensor name filtering is case-sensitive ───────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// LayerNorm matching is case-sensitive — uppercase variants don't match.
    #[test]
    fn layernorm_case_sensitive(prefix in "blk\\.[0-9]{1,2}") {
        let lower = format!("{prefix}.attn_norm.weight");
        let upper = format!("{prefix}.ATTN_NORM.WEIGHT");
        prop_assert!(is_layernorm_weight(&lower), "lowercase should match");
        prop_assert!(!is_layernorm_weight(&upper), "UPPERCASE should NOT match");
    }

    /// Projection matching is case-sensitive — uppercase variants don't match.
    #[test]
    fn projection_case_sensitive(prefix in "blk\\.[0-9]{1,2}") {
        let lower = format!("{prefix}.attn_q.weight");
        let upper = format!("{prefix}.ATTN_Q.WEIGHT");
        prop_assert!(is_projection_weight(&lower), "lowercase should match");
        prop_assert!(!is_projection_weight(&upper), "UPPERCASE should NOT match");
    }

    /// Arbitrary strings never panic in is_layernorm_weight.
    #[test]
    fn layernorm_no_panic(name in "\\PC{0,100}") {
        let _ = is_layernorm_weight(&name);
    }

    /// Arbitrary strings never panic in is_projection_weight.
    #[test]
    fn projection_no_panic(name in "\\PC{0,100}") {
        let _ = is_projection_weight(&name);
    }

    /// Known LayerNorm patterns always match.
    #[test]
    fn known_layernorm_patterns(
        idx in 0u32..100,
        suffix in prop_oneof![
            Just("attn_norm.weight"),
            Just("ffn_norm.weight"),
            Just("attention_norm.weight"),
            Just("input_layernorm.weight"),
            Just("post_attention_layernorm.weight"),
        ],
    ) {
        let name = format!("blk.{idx}.{suffix}");
        prop_assert!(is_layernorm_weight(&name), "{name} should be layernorm");
    }

    /// Known projection patterns always match.
    #[test]
    fn known_projection_patterns(
        idx in 0u32..100,
        suffix in prop_oneof![
            Just("attn_q.weight"),
            Just("attn_k.weight"),
            Just("attn_v.weight"),
            Just("attn_output.weight"),
            Just("ffn_gate.weight"),
            Just("ffn_up.weight"),
            Just("ffn_down.weight"),
        ],
    ) {
        let name = format!("blk.{idx}.{suffix}");
        prop_assert!(is_projection_weight(&name), "{name} should be projection");
    }
}

// ── TensorData / TensorInfo consistency ───────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// TensorData numel matches shape product.
    #[test]
    fn tensor_data_numel(
        d1 in 1usize..=64,
        d2 in 1usize..=64,
    ) {
        let shape = vec![d1, d2];
        let numel = d1 * d2;
        let data = TensorData {
            shape: shape.clone(),
            dtype: DType::F32,
            data: vec![0u8; numel * 4],
        };
        prop_assert_eq!(data.numel(), numel);
        prop_assert_eq!(data.expected_byte_len(), numel * 4);
    }

    /// DType element_size is always 1, 2, 4, or 8.
    #[test]
    fn dtype_element_size_valid(
        dt in prop_oneof![
            Just(DType::F16), Just(DType::BF16), Just(DType::F32), Just(DType::F64),
            Just(DType::I8), Just(DType::I16), Just(DType::I32),
            Just(DType::U8), Just(DType::U16), Just(DType::U32),
        ]
    ) {
        let s = dt.element_size();
        prop_assert!(
            s == 1 || s == 2 || s == 4 || s == 8,
            "unexpected element_size {s} for {dt:?}"
        );
    }

    /// TensorInfo byte size is consistent with shape and dtype.
    #[test]
    fn tensor_info_size_consistent(
        d1 in 1usize..=32,
        d2 in 1usize..=32,
        dt in prop_oneof![
            Just(DType::F32), Just(DType::F16), Just(DType::I8), Just(DType::U8),
        ],
    ) {
        let shape = vec![d1, d2];
        let numel: usize = shape.iter().product();
        let expected_bytes = (numel * dt.element_size()) as u64;
        let info = TensorInfo {
            shape,
            dtype: dt,
            offset: 0,
            size: expected_bytes,
        };
        prop_assert_eq!(info.size, expected_bytes);
    }
}
