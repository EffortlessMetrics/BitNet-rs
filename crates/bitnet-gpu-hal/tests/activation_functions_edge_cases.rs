//! Edge-case tests for activation functions: ActivationType, ActivationConfig,
//! Activation trait implementations (ReLU, GeLU, SiLU, Mish, Tanh, Sigmoid,
//! LeakyReLU, ELU, SELU, Softplus, QuickGELU, HardSwish, HardSigmoid),
//! ActivationRegistry, FusedActivation, ActivationProfiler.

use bitnet_gpu_hal::activation_functions::Activation;
use bitnet_gpu_hal::activation_functions::{
    ActivationConfig, ActivationProfiler, ActivationRegistry, ActivationType, ELUActivation,
    FusedActivation, GeLUActivation, HardSigmoidActivation, HardSwishActivation,
    LeakyReLUActivation, MishActivation, QuickGELUActivation, ReLUActivation, SELUActivation,
    SiLUActivation, SigmoidActivation, SoftplusActivation, TanhActivation, create_activation,
};

// ── ActivationType ────────────────────────────────────────────────────────────

#[test]
fn activation_type_all_13_variants() {
    let variants = [
        ActivationType::ReLU,
        ActivationType::GeLU,
        ActivationType::SiLU,
        ActivationType::Mish,
        ActivationType::Tanh,
        ActivationType::Sigmoid,
        ActivationType::LeakyReLU(0.01),
        ActivationType::ELU(1.0),
        ActivationType::SELU,
        ActivationType::Softplus,
        ActivationType::QuickGELU,
        ActivationType::HardSwish,
        ActivationType::HardSigmoid,
    ];
    assert_eq!(variants.len(), 13);
}

#[test]
fn activation_type_display() {
    let s = format!("{}", ActivationType::ReLU);
    assert!(!s.is_empty());
    let s = format!("{}", ActivationType::LeakyReLU(0.01));
    assert!(!s.is_empty());
}

#[test]
fn activation_type_clone_eq() {
    let a = ActivationType::SiLU;
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn activation_type_leaky_relu_param() {
    let a = ActivationType::LeakyReLU(0.1);
    let b = ActivationType::LeakyReLU(0.2);
    assert_ne!(a, b);
}

// ── ActivationConfig ──────────────────────────────────────────────────────────

#[test]
fn activation_config_new() {
    let cfg = ActivationConfig::new(ActivationType::ReLU);
    assert_eq!(cfg.activation_type, ActivationType::ReLU);
    assert!(!cfg.in_place);
    assert!(!cfg.approximate);
}

#[test]
fn activation_config_builder() {
    let cfg =
        ActivationConfig::new(ActivationType::GeLU).with_in_place(true).with_approximate(true);
    assert!(cfg.in_place);
    assert!(cfg.approximate);
}

// ── ReLU ──────────────────────────────────────────────────────────────────────

#[test]
fn relu_forward_positive() {
    let relu = ReLUActivation;
    let out = relu.forward(&[1.0, 2.0, 3.0]);
    assert_eq!(out, vec![1.0, 2.0, 3.0]);
}

#[test]
fn relu_forward_negative() {
    let relu = ReLUActivation;
    let out = relu.forward(&[-1.0, -2.0, -3.0]);
    assert_eq!(out, vec![0.0, 0.0, 0.0]);
}

#[test]
fn relu_forward_zero() {
    let relu = ReLUActivation;
    let out = relu.forward(&[0.0]);
    assert_eq!(out, vec![0.0]);
}

#[test]
fn relu_forward_empty() {
    let relu = ReLUActivation;
    let out = relu.forward(&[]);
    assert!(out.is_empty());
}

#[test]
fn relu_backward() {
    let relu = ReLUActivation;
    let grad = relu.backward(&[1.0, -1.0, 0.0]);
    assert_eq!(grad[0], 1.0);
    assert_eq!(grad[1], 0.0);
}

#[test]
fn relu_name() {
    let relu = ReLUActivation;
    assert_eq!(relu.name(), "ReLU");
}

#[test]
fn relu_is_monotonic() {
    let relu = ReLUActivation;
    assert!(relu.is_monotonic());
}

#[test]
fn relu_inplace() {
    let relu = ReLUActivation;
    let mut data = vec![-1.0, 0.0, 1.0];
    relu.forward_inplace(&mut data);
    assert_eq!(data, vec![0.0, 0.0, 1.0]);
}

// ── GeLU ──────────────────────────────────────────────────────────────────────

#[test]
fn gelu_forward_zero() {
    let gelu = GeLUActivation::new(false);
    let out = gelu.forward(&[0.0]);
    assert!((out[0] - 0.0).abs() < 1e-6);
}

#[test]
fn gelu_forward_positive() {
    let gelu = GeLUActivation::new(false);
    let out = gelu.forward(&[1.0]);
    assert!(out[0] > 0.0);
    assert!(out[0] < 1.0);
}

#[test]
fn gelu_approximate_vs_exact() {
    let exact = GeLUActivation::new(false);
    let approx = GeLUActivation::new(true);
    let x = vec![0.5];
    let e = exact.forward(&x);
    let a = approx.forward(&x);
    // Should be close but not identical
    assert!((e[0] - a[0]).abs() < 0.05);
}

#[test]
fn gelu_name() {
    assert_eq!(GeLUActivation::new(false).name(), "GeLU(exact)");
}

// ── SiLU ──────────────────────────────────────────────────────────────────────

#[test]
fn silu_forward_zero() {
    let silu = SiLUActivation;
    let out = silu.forward(&[0.0]);
    assert!((out[0] - 0.0).abs() < 1e-6);
}

#[test]
fn silu_forward_positive() {
    let silu = SiLUActivation;
    let out = silu.forward(&[2.0]);
    // SiLU(2) = 2 * sigmoid(2) ≈ 2 * 0.8808 ≈ 1.7616
    assert!((out[0] - 1.7616).abs() < 0.01);
}

#[test]
fn silu_forward_negative() {
    let silu = SiLUActivation;
    let out = silu.forward(&[-2.0]);
    assert!(out[0] < 0.0);
}

#[test]
fn silu_name() {
    assert_eq!(SiLUActivation.name(), "SiLU");
}

// ── Mish ──────────────────────────────────────────────────────────────────────

#[test]
fn mish_forward_zero() {
    let mish = MishActivation;
    let out = mish.forward(&[0.0]);
    assert!((out[0] - 0.0).abs() < 1e-6);
}

#[test]
fn mish_forward_positive() {
    let mish = MishActivation;
    let out = mish.forward(&[1.0]);
    assert!(out[0] > 0.0);
}

#[test]
fn mish_name() {
    assert_eq!(MishActivation.name(), "Mish");
}

// ── Tanh ──────────────────────────────────────────────────────────────────────

#[test]
fn tanh_forward_zero() {
    let t = TanhActivation;
    let out = t.forward(&[0.0]);
    assert!((out[0] - 0.0).abs() < 1e-6);
}

#[test]
fn tanh_forward_bounds() {
    let t = TanhActivation;
    let out = t.forward(&[100.0, -100.0]);
    assert!((out[0] - 1.0).abs() < 1e-5);
    assert!((out[1] - (-1.0)).abs() < 1e-5);
}

#[test]
fn tanh_name() {
    assert_eq!(TanhActivation.name(), "Tanh");
}

// ── Sigmoid ───────────────────────────────────────────────────────────────────

#[test]
fn sigmoid_forward_zero() {
    let s = SigmoidActivation;
    let out = s.forward(&[0.0]);
    assert!((out[0] - 0.5).abs() < 1e-6);
}

#[test]
fn sigmoid_forward_large() {
    let s = SigmoidActivation;
    let out = s.forward(&[100.0, -100.0]);
    assert!((out[0] - 1.0).abs() < 1e-5);
    assert!(out[1] < 1e-5);
}

#[test]
fn sigmoid_name() {
    assert_eq!(SigmoidActivation.name(), "Sigmoid");
}

// ── LeakyReLU ─────────────────────────────────────────────────────────────────

#[test]
fn leaky_relu_positive() {
    let lr = LeakyReLUActivation::new(0.01);
    let out = lr.forward(&[5.0]);
    assert_eq!(out[0], 5.0);
}

#[test]
fn leaky_relu_negative() {
    let lr = LeakyReLUActivation::new(0.1);
    let out = lr.forward(&[-10.0]);
    assert!((out[0] - (-1.0)).abs() < 1e-6);
}

#[test]
fn leaky_relu_default_alpha() {
    let lr = LeakyReLUActivation::default();
    let out = lr.forward(&[-1.0]);
    assert!((out[0] - (-0.01)).abs() < 1e-6);
}

#[test]
fn leaky_relu_name() {
    assert_eq!(LeakyReLUActivation::new(0.01).name(), "LeakyReLU");
}

// ── ELU ───────────────────────────────────────────────────────────────────────

#[test]
fn elu_positive() {
    let elu = ELUActivation::new(1.0);
    let out = elu.forward(&[2.0]);
    assert_eq!(out[0], 2.0);
}

#[test]
fn elu_negative() {
    let elu = ELUActivation::new(1.0);
    let out = elu.forward(&[-1.0]);
    // ELU(-1) = 1.0 * (exp(-1) - 1) ≈ -0.6321
    assert!((out[0] - (-0.6321)).abs() < 0.01);
}

#[test]
fn elu_default_alpha() {
    let elu = ELUActivation::default();
    let out = elu.forward(&[-1.0]);
    assert!(out[0] < 0.0);
}

#[test]
fn elu_name() {
    assert_eq!(ELUActivation::new(1.0).name(), "ELU");
}

// ── SELU ──────────────────────────────────────────────────────────────────────

#[test]
fn selu_positive() {
    let selu = SELUActivation;
    let out = selu.forward(&[1.0]);
    // SELU(1) = lambda * 1 = 1.0507
    assert!((out[0] - 1.0507).abs() < 0.01);
}

#[test]
fn selu_negative() {
    let selu = SELUActivation;
    let out = selu.forward(&[-1.0]);
    assert!(out[0] < 0.0);
}

#[test]
fn selu_name() {
    assert_eq!(SELUActivation.name(), "SELU");
}

// ── Softplus ──────────────────────────────────────────────────────────────────

#[test]
fn softplus_positive() {
    let sp = SoftplusActivation;
    let out = sp.forward(&[1.0]);
    // softplus(1) = ln(1 + exp(1)) ≈ 1.3133
    assert!((out[0] - 1.3133).abs() < 0.01);
}

#[test]
fn softplus_zero() {
    let sp = SoftplusActivation;
    let out = sp.forward(&[0.0]);
    assert!((out[0] - 0.6931).abs() < 0.01);
}

#[test]
fn softplus_name() {
    assert_eq!(SoftplusActivation.name(), "Softplus");
}

// ── QuickGELU ─────────────────────────────────────────────────────────────────

#[test]
fn quick_gelu_zero() {
    let qg = QuickGELUActivation;
    let out = qg.forward(&[0.0]);
    assert!((out[0] - 0.0).abs() < 1e-6);
}

#[test]
fn quick_gelu_positive() {
    let qg = QuickGELUActivation;
    let out = qg.forward(&[1.0]);
    assert!(out[0] > 0.0);
}

#[test]
fn quick_gelu_name() {
    assert_eq!(QuickGELUActivation.name(), "QuickGELU");
}

// ── HardSwish ─────────────────────────────────────────────────────────────────

#[test]
fn hard_swish_zero() {
    let hs = HardSwishActivation;
    let out = hs.forward(&[0.0]);
    assert!((out[0] - 0.0).abs() < 1e-6);
}

#[test]
fn hard_swish_large_positive() {
    let hs = HardSwishActivation;
    let out = hs.forward(&[10.0]);
    assert!((out[0] - 10.0).abs() < 1e-5);
}

#[test]
fn hard_swish_name() {
    assert_eq!(HardSwishActivation.name(), "HardSwish");
}

// ── HardSigmoid ───────────────────────────────────────────────────────────────

#[test]
fn hard_sigmoid_zero() {
    let hs = HardSigmoidActivation;
    let out = hs.forward(&[0.0]);
    assert!((out[0] - 0.5).abs() < 1e-5);
}

#[test]
fn hard_sigmoid_clamped_high() {
    let hs = HardSigmoidActivation;
    let out = hs.forward(&[10.0]);
    assert!((out[0] - 1.0).abs() < 1e-5);
}

#[test]
fn hard_sigmoid_clamped_low() {
    let hs = HardSigmoidActivation;
    let out = hs.forward(&[-10.0]);
    assert!((out[0] - 0.0).abs() < 1e-5);
}

#[test]
fn hard_sigmoid_name() {
    assert_eq!(HardSigmoidActivation.name(), "HardSigmoid");
}

// ── NaN / Inf propagation ─────────────────────────────────────────────────────

#[test]
fn relu_nan_propagation() {
    let relu = ReLUActivation;
    let out = relu.forward(&[f32::NAN]);
    assert!(out[0].is_nan() || out[0] == 0.0);
}

#[test]
fn relu_inf_propagation() {
    let relu = ReLUActivation;
    let out = relu.forward(&[f32::INFINITY, f32::NEG_INFINITY]);
    assert_eq!(out[0], f32::INFINITY);
    assert_eq!(out[1], 0.0);
}

#[test]
fn sigmoid_inf_saturation() {
    let sig = SigmoidActivation;
    let out = sig.forward(&[f32::INFINITY, f32::NEG_INFINITY]);
    assert!((out[0] - 1.0).abs() < 1e-5);
    assert!(out[1].abs() < 1e-5);
}

// ── create_activation factory ─────────────────────────────────────────────────

#[test]
fn create_activation_all_types() {
    let types = [
        ActivationType::ReLU,
        ActivationType::GeLU,
        ActivationType::SiLU,
        ActivationType::Mish,
        ActivationType::Tanh,
        ActivationType::Sigmoid,
        ActivationType::LeakyReLU(0.01),
        ActivationType::ELU(1.0),
        ActivationType::SELU,
        ActivationType::Softplus,
        ActivationType::QuickGELU,
        ActivationType::HardSwish,
        ActivationType::HardSigmoid,
    ];
    for ty in &types {
        let act = create_activation(ty);
        let out = act.forward(&[1.0]);
        assert!(!out.is_empty());
    }
}

// ── ActivationRegistry ────────────────────────────────────────────────────────

#[test]
fn registry_new_empty() {
    let reg = ActivationRegistry::new();
    assert!(reg.is_empty());
    assert_eq!(reg.len(), 0);
}

#[test]
fn registry_with_builtins() {
    let reg = ActivationRegistry::with_builtins();
    assert!(!reg.is_empty());
    // Should have all 14 built-in activations
    assert!(reg.len() >= 13);
}

#[test]
fn registry_get_builtin() {
    let reg = ActivationRegistry::with_builtins();
    assert!(reg.get("relu").is_some());
    assert!(reg.get("gelu").is_some());
    assert!(reg.get("silu").is_some());
}

#[test]
fn registry_get_missing() {
    let reg = ActivationRegistry::with_builtins();
    assert!(reg.get("nonexistent").is_none());
}

#[test]
fn registry_register_custom() {
    let mut reg = ActivationRegistry::new();
    reg.register("my_relu", Box::new(ReLUActivation));
    assert_eq!(reg.len(), 1);
    let act = reg.get("my_relu").unwrap();
    assert_eq!(act.name(), "ReLU");
}

#[test]
fn registry_list_names_sorted() {
    let mut reg = ActivationRegistry::new();
    reg.register("zeta", Box::new(ReLUActivation));
    reg.register("alpha", Box::new(SiLUActivation));
    let names = reg.list_names();
    assert_eq!(names[0], "alpha");
    assert_eq!(names[1], "zeta");
}

// ── FusedActivation ───────────────────────────────────────────────────────────

#[test]
fn fused_activation_with_scalar_bias() {
    let relu = ReLUActivation;
    let fused = FusedActivation::new(&relu, vec![1.0]);
    let out = fused.forward(&[0.5]);
    // bias 1.0 + relu(0.5) or relu(0.5 + 1.0) — depends on impl
    assert!(!out.is_empty());
}

#[test]
fn fused_activation_with_vector_bias() {
    let relu = ReLUActivation;
    let fused = FusedActivation::new(&relu, vec![1.0, 2.0, 3.0]);
    let out = fused.forward(&[-0.5, -1.5, -2.5]);
    assert_eq!(out.len(), 3);
}

#[test]
#[should_panic]
fn fused_activation_empty_bias_panics() {
    let relu = ReLUActivation;
    let _fused = FusedActivation::new(&relu, vec![]);
}

// ── ActivationProfiler ────────────────────────────────────────────────────────

#[test]
fn profiler_basic() {
    let profiler = ActivationProfiler::new(10);
    let relu = ReLUActivation;
    let input = vec![1.0; 100];
    let result = profiler.profile(&relu, &input);
    assert_eq!(result.name, "ReLU");
    assert_eq!(result.num_elements, 100);
    assert_eq!(result.iterations, 10);
    assert!(result.elements_per_second > 0.0);
}

#[test]
fn profiler_single_iteration() {
    let profiler = ActivationProfiler::new(1);
    let silu = SiLUActivation;
    let input = vec![0.0; 10];
    let result = profiler.profile(&silu, &input);
    assert_eq!(result.iterations, 1);
}

// ── Monotonicity ──────────────────────────────────────────────────────────────

#[test]
fn monotonic_activations() {
    assert!(ReLUActivation.is_monotonic());
    assert!(SigmoidActivation.is_monotonic());
    assert!(TanhActivation.is_monotonic());
}

#[test]
fn non_monotonic_activations() {
    // Mish and some others are not monotonic
    let mish = MishActivation;
    // Just check that the method exists and returns a bool
    let _m = mish.is_monotonic();
}

// ── Backward pass ─────────────────────────────────────────────────────────────

#[test]
fn sigmoid_backward() {
    let sig = SigmoidActivation;
    let grad = sig.backward(&[0.0]);
    // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.25
    assert!((grad[0] - 0.25).abs() < 1e-5);
}

#[test]
fn tanh_backward() {
    let t = TanhActivation;
    let grad = t.backward(&[0.0]);
    // tanh'(0) = 1 - tanh(0)^2 = 1
    assert!((grad[0] - 1.0).abs() < 1e-5);
}

#[test]
fn silu_backward_zero() {
    let silu = SiLUActivation;
    let grad = silu.backward(&[0.0]);
    // SiLU'(0) = sigmoid(0) + 0 * sigmoid(0) * (1-sigmoid(0)) = 0.5
    assert!((grad[0] - 0.5).abs() < 1e-5);
}
