//! Reusable test data generators and golden output fixtures.
//!
//! Provides deterministic pseudo-random data and known-good reference
//! outputs for kernel validation.

/// Simple deterministic PRNG (xorshift64) for reproducible test data.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFFFF) as f32 / 0xFFFFFF as f32
    }
}

/// Generate a deterministic vector of f32 values in [-1, 1].
pub fn random_f32_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Xorshift64::new(seed);
    (0..len).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
}

/// Generate a deterministic vector of ternary values {-1, 0, +1}.
pub fn random_ternary_vec(len: usize, seed: u64) -> Vec<i8> {
    let mut rng = Xorshift64::new(seed);
    (0..len)
        .map(|_| match rng.next_u64() % 3 {
            0 => -1i8,
            1 => 0i8,
            _ => 1i8,
        })
        .collect()
}

/// Generate a row-major identity matrix of size n×n.
pub fn identity_matrix(n: usize) -> Vec<f32> {
    let mut mat = vec![0.0f32; n * n];
    for i in 0..n {
        mat[i * n + i] = 1.0;
    }
    mat
}

/// Golden 4×4 matmul test case: A * B = C.
pub fn golden_matmul_4x4() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    #[rustfmt::skip]
    let a = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    #[rustfmt::skip]
    let b = vec![
        1.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 1.0,
    ];
    // C = A * B (hand-computed)
    #[rustfmt::skip]
    let c = vec![
        5.0,  2.0,  3.0,  5.0,
        13.0, 6.0,  7.0,  13.0,
        21.0, 10.0, 11.0, 21.0,
        29.0, 14.0, 15.0, 29.0,
    ];
    (a, b, c)
}

/// Golden softmax test case for 8 elements.
pub fn golden_softmax_8() -> (Vec<f32>, Vec<f32>) {
    let input = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
    // Compute expected softmax
    let max_val = 4.0f32;
    let exps: Vec<f64> = input
        .iter()
        .map(|&x| ((x - max_val) as f64).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    let expected: Vec<f32> = exps.iter().map(|&e| (e / sum) as f32).collect();
    (input, expected)
}

/// Golden RMSNorm test case for 4 elements.
pub fn golden_rmsnorm_4() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-6f32;

    let mean_sq: f64 =
        input.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / 4.0;
    let rms = (mean_sq + eps as f64).sqrt();
    let expected: Vec<f32> = input.iter().map(|&x| (x as f64 / rms) as f32).collect();

    (input, weight, expected)
}

/// Golden SiLU test case for known values.
pub fn golden_silu_values() -> (Vec<f32>, Vec<f32>) {
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| {
            let sigmoid = 1.0 / (1.0 + (-x as f64).exp());
            (x as f64 * sigmoid) as f32
        })
        .collect();
    (input, expected)
}

/// Golden GELU test case for known values.
pub fn golden_gelu_values() -> (Vec<f32>, Vec<f32>) {
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let sqrt_2_over_pi = (2.0f64 / std::f64::consts::PI).sqrt();
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| {
            let x64 = x as f64;
            let inner =
                sqrt_2_over_pi * (x64 + 0.044715 * x64 * x64 * x64);
            (0.5 * x64 * (1.0 + inner.tanh())) as f32
        })
        .collect();
    (input, expected)
}
