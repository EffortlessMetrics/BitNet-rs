//! CPU activation kernels re-exported from `bitnet-cpu-activations`.

pub use bitnet_cpu_activations::*;

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn relu_non_negative(x in -1e6f32..1e6) {
            let y = relu(x);
            prop_assert!(y >= 0.0 || y.is_nan(), "relu({x}) = {y} < 0");
        }

        #[test]
        fn relu_identity_for_positive(x in 0.0f32..1e6) {
            prop_assert_eq!(relu(x), x);
        }

        #[test]
        fn sigmoid_in_unit_interval(x in -1e3f32..1e3) {
            let y = sigmoid(x);
            prop_assert!(y >= 0.0 && y <= 1.0, "sigmoid({x}) = {y} not in [0,1]");
        }

        #[test]
        fn sigmoid_monotonic(a in -100.0f32..100.0, delta in 0.0f32..10.0) {
            let b = a + delta;
            prop_assert!(
                sigmoid(b) >= sigmoid(a),
                "sigmoid not monotonic: sigmoid({}) > sigmoid({})",
                a,
                b
            );
        }

        #[test]
        fn hard_sigmoid_in_unit_interval(x in -1e6f32..1e6) {
            let y = hard_sigmoid(x);
            if !x.is_nan() {
                prop_assert!(
                    y >= 0.0 && y <= 1.0,
                    "hard_sigmoid({x}) = {y} not in [0,1]"
                );
            }
        }

        #[test]
        fn tanh_in_bounds(x in -1e3f32..1e3) {
            let y = tanh_act(x);
            prop_assert!(y >= -1.0 && y <= 1.0, "tanh({x}) = {y} not in [-1,1]");
        }

        #[test]
        fn softplus_non_negative(x in -100.0f32..100.0) {
            let y = softplus(x);
            prop_assert!(y >= 0.0, "softplus({x}) = {y} < 0");
        }

        #[test]
        fn selu_negative_bounded(x in -1e3f32..0.0f32) {
            let y = selu(x);
            prop_assert!(
                y >= -1.6732632 * 1.050_701 - 0.01,
                "selu({x}) = {y} below expected lower bound"
            );
        }

        #[test]
        fn elu_negative_bounded(x in -100.0f32..0.0f32, alpha in 0.1f32..10.0) {
            let y = elu(x, alpha);
            prop_assert!(y >= -alpha, "elu({x}, {alpha}) = {y} < -{alpha}");
        }

        #[test]
        fn relu_vec_all_non_negative(
            xs in prop::collection::vec(-1e6f32..1e6, 1..256)
        ) {
            let ys = apply_activation(&xs, ActivationType::ReLU);
            for (i, &y) in ys.iter().enumerate() {
                prop_assert!(y >= 0.0 || y.is_nan(), "relu_vec[{i}] = {y} < 0");
            }
        }

        #[test]
        fn sigmoid_vec_all_in_unit(
            xs in prop::collection::vec(-1e3f32..1e3, 1..256)
        ) {
            let ys = apply_activation(&xs, ActivationType::Sigmoid);
            for (i, &y) in ys.iter().enumerate() {
                prop_assert!(
                    y >= 0.0 && y <= 1.0,
                    "sigmoid_vec[{i}] = {y} not in [0,1]"
                );
            }
        }

        #[test]
        fn activation_preserves_length(
            xs in prop::collection::vec(-10.0f32..10.0, 1..128)
        ) {
            prop_assert_eq!(apply_activation(&xs, ActivationType::ReLU).len(), xs.len());
            prop_assert_eq!(
                apply_activation(&xs, ActivationType::Sigmoid).len(),
                xs.len()
            );
            prop_assert_eq!(apply_activation(&xs, ActivationType::GELU).len(), xs.len());
            prop_assert_eq!(apply_activation(&xs, ActivationType::SiLU).len(), xs.len());
            prop_assert_eq!(apply_activation(&xs, ActivationType::Tanh).len(), xs.len());
        }
    }
}
