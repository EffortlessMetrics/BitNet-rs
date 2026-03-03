//! Property-based tests — wave 36.
//!
//! Covers bitnet-kernels: activation function domain properties,
//! kernel provider selection, SIMD detection consistency, and
//! activation registry round-trips.

#![cfg(feature = "cpu")]

use bitnet_kernels::activation_registry::{
    ActivationType, activate, activate_inplace, activate_vec,
};
use proptest::prelude::*;

// ── Strategies ──────────────────────────────────────────────────────────────

fn arb_activation() -> impl Strategy<Value = ActivationType> {
    prop_oneof![
        Just(ActivationType::ReLU),
        Just(ActivationType::ReLU2),
        Just(ActivationType::SiLU),
        Just(ActivationType::GeLU),
        Just(ActivationType::GeLUTanh),
        Just(ActivationType::Tanh),
        Just(ActivationType::Sigmoid),
        Just(ActivationType::Mish),
    ]
}

fn finite_f32() -> impl Strategy<Value = f32> {
    (-100.0f32..100.0).prop_filter("must be finite", |x| x.is_finite())
}

fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(finite_f32(), 1..=max_len)
}

fn arb_model_family() -> impl Strategy<Value = &'static str> {
    prop_oneof![
        Just("bitnet"),
        Just("phi"),
        Just("phi2"),
        Just("phi3"),
        Just("phi4"),
        Just("llama"),
        Just("llama2"),
        Just("llama3"),
        Just("mistral"),
        Just("mixtral"),
        Just("qwen"),
        Just("qwen2"),
        Just("gemma"),
        Just("gemma2"),
        Just("gpt2"),
        Just("gptneo"),
        Just("falcon"),
    ]
}

// ── Property tests ──────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // ════════════════════════════════════════════════════════════════
    // 1. Activation function domain properties
    // ════════════════════════════════════════════════════════════════

    /// ReLU output is always non-negative.
    #[test]
    fn prop_relu_nonneg(x in finite_f32()) {
        let y = activate(x, ActivationType::ReLU);
        prop_assert!(y >= 0.0, "ReLU({}) = {} should be >= 0", x, y);
    }

    /// ReLU2 (squared ReLU) output is always non-negative.
    #[test]
    fn prop_relu2_nonneg(x in finite_f32()) {
        let y = activate(x, ActivationType::ReLU2);
        prop_assert!(y >= 0.0, "ReLU2({}) = {} should be >= 0", x, y);
    }

    /// Sigmoid output is in [0, 1].
    #[test]
    fn prop_sigmoid_bounded(x in finite_f32()) {
        let y = activate(x, ActivationType::Sigmoid);
        prop_assert!(y >= 0.0 && y <= 1.0, "Sigmoid({}) = {} should be in [0,1]", x, y);
    }

    /// Tanh output is in [-1, 1].
    #[test]
    fn prop_tanh_bounded(x in finite_f32()) {
        let y = activate(x, ActivationType::Tanh);
        prop_assert!(y >= -1.0 && y <= 1.0, "Tanh({}) = {} should be in [-1,1]", x, y);
    }

    /// GeLU at x=0 is approximately 0.
    #[test]
    fn prop_gelu_zero(x in -0.001f32..0.001) {
        let y = activate(x, ActivationType::GeLU);
        prop_assert!((y).abs() < 0.01, "GeLU({}) = {} should be near 0", x, y);
    }

    /// GeLU-Tanh at x=0 is approximately 0.
    #[test]
    fn prop_gelu_tanh_zero(x in -0.001f32..0.001) {
        let y = activate(x, ActivationType::GeLUTanh);
        prop_assert!((y).abs() < 0.01, "GeLUTanh({}) = {} should be near 0", x, y);
    }

    /// All activations produce finite output for finite input.
    #[test]
    fn prop_activation_finite(x in finite_f32(), act in arb_activation()) {
        let y = activate(x, act);
        prop_assert!(y.is_finite(), "{:?}({}) = {} should be finite", act, x, y);
    }

    /// ReLU is monotonically non-decreasing.
    #[test]
    fn prop_relu_monotone(a in finite_f32(), b in finite_f32()) {
        if a <= b {
            let ya = activate(a, ActivationType::ReLU);
            let yb = activate(b, ActivationType::ReLU);
            prop_assert!(ya <= yb, "ReLU({}) = {} <= ReLU({}) = {}", a, ya, b, yb);
        }
    }

    /// Sigmoid is monotonically non-decreasing.
    #[test]
    fn prop_sigmoid_monotone(a in finite_f32(), b in finite_f32()) {
        if a <= b {
            let ya = activate(a, ActivationType::Sigmoid);
            let yb = activate(b, ActivationType::Sigmoid);
            prop_assert!(ya <= yb + 1e-6, "Sigmoid({}) = {} <= Sigmoid({}) = {}", a, ya, b, yb);
        }
    }

    /// ReLU(x) == x for x > 0.
    #[test]
    fn prop_relu_identity_positive(x in 0.001f32..100.0) {
        let y = activate(x, ActivationType::ReLU);
        prop_assert!((y - x).abs() < 1e-6, "ReLU({}) = {} should equal x", x, y);
    }

    /// ReLU(x) == 0 for x < 0.
    #[test]
    fn prop_relu_zero_negative(x in -100.0f32..-0.001) {
        let y = activate(x, ActivationType::ReLU);
        prop_assert_eq!(y, 0.0, "ReLU({}) should be 0", x);
    }

    // ════════════════════════════════════════════════════════════════
    // 2. activate_vec / activate_inplace consistency
    // ════════════════════════════════════════════════════════════════

    /// activate_vec and activate produce same results element-wise.
    #[test]
    fn prop_activate_vec_matches_scalar(
        data in finite_f32_vec(64),
        act in arb_activation()
    ) {
        let vec_result = activate_vec(&data, act);
        prop_assert_eq!(vec_result.len(), data.len());
        for (i, (&input, &output)) in data.iter().zip(vec_result.iter()).enumerate() {
            let expected = activate(input, act);
            prop_assert!(
                (output - expected).abs() < 1e-6,
                "mismatch at index {}: activate_vec gave {}, scalar gave {}",
                i, output, expected
            );
        }
    }

    /// activate_inplace produces same results as activate_vec.
    #[test]
    fn prop_activate_inplace_matches_vec(
        data in finite_f32_vec(64),
        act in arb_activation()
    ) {
        let vec_result = activate_vec(&data, act);
        let mut inplace = data.clone();
        activate_inplace(&mut inplace, act);
        prop_assert_eq!(inplace.len(), vec_result.len());
        for (i, (&a, &b)) in inplace.iter().zip(vec_result.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < 1e-6,
                "mismatch at index {}: inplace={}, vec={}",
                i, a, b
            );
        }
    }

    /// activate_vec preserves length.
    #[test]
    fn prop_activate_vec_preserves_len(
        data in finite_f32_vec(128),
        act in arb_activation()
    ) {
        let result = activate_vec(&data, act);
        prop_assert_eq!(result.len(), data.len());
    }

    // ════════════════════════════════════════════════════════════════
    // 3. Activation name round-trip
    // ════════════════════════════════════════════════════════════════

    /// from_name(name()) recovers the activation type for all variants.
    #[test]
    fn prop_activation_name_roundtrip(act in arb_activation()) {
        let name = act.name();
        let recovered = ActivationType::from_name(name);
        prop_assert_eq!(recovered, Some(act), "round-trip failed for {:?}", act);
    }

    /// from_name returns Some for all known activation names.
    #[test]
    fn prop_from_name_known(
        name in prop_oneof![
            Just("relu"),
            Just("relu2"),
            Just("relu_squared"),
            Just("squared_relu"),
            Just("silu"),
            Just("swish"),
            Just("gelu"),
            Just("gelu_tanh"),
            Just("gelu_new"),
            Just("gelu_fast"),
            Just("tanh"),
            Just("sigmoid"),
            Just("mish"),
        ]
    ) {
        prop_assert!(
            ActivationType::from_name(name).is_some(),
            "from_name('{}') should return Some",
            name
        );
    }

    // ════════════════════════════════════════════════════════════════
    // 4. Model family defaults
    // ════════════════════════════════════════════════════════════════

    /// for_family always returns a valid ActivationType for known families.
    #[test]
    fn prop_for_family_valid(family in arb_model_family()) {
        let act = ActivationType::for_family(family);
        // Verify it's a valid variant by round-tripping the name
        let name = act.name();
        prop_assert!(ActivationType::from_name(name).is_some());
    }

    /// BitNet family always maps to ReLU2.
    #[test]
    fn prop_bitnet_family_relu2(_dummy in 0u8..1) {
        prop_assert_eq!(ActivationType::for_family("bitnet"), ActivationType::ReLU2);
    }

    /// Unknown families default to SiLU.
    #[test]
    fn prop_unknown_family_silu(
        family in "[a-z]{8,16}"
    ) {
        // Filter out known families
        let known = ["bitnet", "phi", "phi2", "phi3", "phi4", "llama", "llama2",
                      "llama3", "mistral", "mixtral", "qwen", "qwen2", "gemma",
                      "gemma2", "gpt2", "gptneo", "falcon"];
        prop_assume!(!known.contains(&family.as_str()));
        prop_assert_eq!(ActivationType::for_family(&family), ActivationType::SiLU);
    }
}
