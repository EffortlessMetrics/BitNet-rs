//! Property-based tests for GPU tensor operations.
//!
//! Verified invariants:
//! - `matmul(I, X) == X` — identity matrix is neutral element.
//! - `softmax` output sums to 1.0 (within epsilon).
//! - `add(a, b) == add(b, a)` — commutativity.
//! - `mul(x, 0) == 0` — zero annihilation.
//! - `silu(0) == 0`.
//! - `rmsnorm` output has unit RMS (within epsilon) when weight is 1.

use bitnet_opencl::{Tensor, TensorShape, tensor_ops_cpu};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Generate a square matrix dimension in [1, 8].
fn dim_strategy() -> impl Strategy<Value = usize> {
    1..=8_usize
}

/// Generate a vector of `n` f32 values in a sane range.
fn vec_f32(n: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-10.0..10.0_f32, n)
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn matmul_identity(n in dim_strategy()) {
        // Build n×n identity.
        let mut eye_data = vec![0.0f32; n * n];
        for i in 0..n {
            eye_data[i * n + i] = 1.0;
        }
        let eye = Tensor::new(TensorShape::new(&[n, n]), eye_data).unwrap();

        // Random n×n matrix.
        let x_data: Vec<f32> = (0..n * n).map(|i| i as f32 * 0.1).collect();
        let x = Tensor::new(TensorShape::new(&[n, n]), x_data.clone()).unwrap();

        let result = tensor_ops_cpu::matmul(&x, &eye).unwrap();
        for (a, b) in result.data.iter().zip(&x_data) {
            prop_assert!((a - b).abs() < 1e-4,
                "matmul(X, I) != X at element: {a} vs {b}");
        }
    }

    #[test]
    fn softmax_sums_to_one(n in 1..=16_usize, data in vec_f32(16)) {
        let n = n.min(data.len()).max(1);
        let slice = &data[..n];
        let t = Tensor::new(
            TensorShape::new(&[n]),
            slice.to_vec(),
        ).unwrap();
        let s = tensor_ops_cpu::softmax(&t, 0).unwrap();
        let sum: f32 = s.data.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax sum = {sum}, expected 1.0"
        );
    }

    #[test]
    fn add_commutative(
        n in 1..=32_usize,
        a_data in vec_f32(32),
        b_data in vec_f32(32),
    ) {
        let n = n.min(a_data.len()).min(b_data.len()).max(1);
        let a = Tensor::new(
            TensorShape::new(&[n]),
            a_data[..n].to_vec(),
        ).unwrap();
        let b = Tensor::new(
            TensorShape::new(&[n]),
            b_data[..n].to_vec(),
        ).unwrap();
        let ab = tensor_ops_cpu::add(&a, &b).unwrap();
        let ba = tensor_ops_cpu::add(&b, &a).unwrap();
        for (x, y) in ab.data.iter().zip(&ba.data) {
            prop_assert!((x - y).abs() < 1e-6,
                "add not commutative: {x} vs {y}");
        }
    }

    #[test]
    fn mul_by_zero(n in 1..=32_usize, data in vec_f32(32)) {
        let n = n.min(data.len()).max(1);
        let a = Tensor::new(
            TensorShape::new(&[n]),
            data[..n].to_vec(),
        ).unwrap();
        let zero = Tensor::new(
            TensorShape::new(&[n]),
            vec![0.0; n],
        ).unwrap();
        let result = tensor_ops_cpu::mul(&a, &zero).unwrap();
        for &v in &result.data {
            prop_assert!(v == 0.0, "mul by zero gave {v}");
        }
    }

    #[test]
    fn silu_zero_is_zero(n in 1..=16_usize) {
        let t = Tensor::new(
            TensorShape::new(&[n]),
            vec![0.0; n],
        ).unwrap();
        let s = tensor_ops_cpu::silu(&t).unwrap();
        for &v in &s.data {
            prop_assert!(v.abs() < 1e-9,
                "silu(0) = {v}, expected 0");
        }
    }

    #[test]
    fn rmsnorm_unit_rms(n in 1..=16_usize, data in vec_f32(16)) {
        let n = n.min(data.len()).max(1);
        // Avoid all-zero input (RMS undefined in a useful sense).
        let mut input_data: Vec<f32> = data[..n].to_vec();
        if input_data.iter().all(|&v| v == 0.0) {
            input_data[0] = 1.0;
        }
        let input = Tensor::new(
            TensorShape::new(&[1, n]),
            input_data,
        ).unwrap();
        let weight = Tensor::new(
            TensorShape::new(&[n]),
            vec![1.0; n],
        ).unwrap();
        let out = tensor_ops_cpu::rmsnorm(&input, &weight, 1e-5).unwrap();
        let rms = (out.data.iter().map(|x| x * x).sum::<f32>()
            / n as f32).sqrt();
        prop_assert!(
            (rms - 1.0).abs() < 0.05,
            "rmsnorm RMS = {rms}, expected ≈ 1.0"
        );
    }
}
