//! A770 OpenCL numerical stability regression harness.
//!
//! Validates GPU kernel outputs against CPU reference implementations to detect
//! floating-point divergence caused by reduction ordering, denormal handling,
//! or platform-specific FP32 behavior on Intel Arc A770 (and other OpenCL devices).
//!
//! All tests require a real OpenCL device and are gated with `#[ignore]`.
//! Run with: `cargo test -p bitnet-kernels --test opencl_numerical_stability -- --ignored`
//! Set `BITNET_OPENCL_DEVICE=0` to select the device index.

// ── Tolerance definitions ──────────────────────────────────────────

/// Per-operation numerical tolerances for GPU-vs-CPU comparison.
///
/// These thresholds account for legitimate floating-point differences
/// caused by different reduction orders, FMA availability, and
/// denormal-flush-to-zero behavior on GPU hardware.
#[derive(Debug, Clone)]
struct NumericalTolerances {
    /// Maximum acceptable absolute error.
    max_abs_error: f32,
    /// Maximum acceptable relative error (vs max(|cpu|, 1e-8)).
    max_rel_error: f32,
    /// Human-readable label for diagnostics.
    label: &'static str,
}

impl NumericalTolerances {
    const MATMUL: Self = Self { max_abs_error: 1e-4, max_rel_error: 1e-3, label: "matmul" };

    const SOFTMAX: Self = Self { max_abs_error: 1e-5, max_rel_error: 1e-4, label: "softmax" };

    const LAYER_NORM: Self = Self { max_abs_error: 1e-5, max_rel_error: 1e-4, label: "layer_norm" };

    const ATTENTION_SCORES: Self =
        Self { max_abs_error: 1e-4, max_rel_error: 1e-3, label: "attention_scores" };

    const EMBEDDING_LOOKUP: Self =
        Self { max_abs_error: 0.0, max_rel_error: 0.0, label: "embedding_lookup" };

    const ROPE: Self = Self { max_abs_error: 1e-5, max_rel_error: 1e-4, label: "rope" };

    const REDUCTION: Self = Self { max_abs_error: 1e-4, max_rel_error: 1e-3, label: "reduction" };
}

// ── Comparison result ──────────────────────────────────────────────

/// Detailed result from comparing two tensors element-wise.
#[derive(Debug)]
struct ComparisonResult {
    /// Maximum absolute difference across all elements.
    max_abs_diff: f32,
    /// Mean absolute difference.
    mean_abs_diff: f32,
    /// Number of elements exceeding the tolerance.
    num_failures: usize,
    /// Index of the element with the worst absolute difference.
    worst_index: usize,
    /// Total number of elements compared.
    total_elements: usize,
}

impl ComparisonResult {
    fn passed(&self) -> bool {
        self.num_failures == 0
    }
}

impl std::fmt::Display for ComparisonResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max_abs={:.2e} mean_abs={:.2e} failures={}/{} worst_idx={}",
            self.max_abs_diff,
            self.mean_abs_diff,
            self.num_failures,
            self.total_elements,
            self.worst_index,
        )
    }
}

// ── Comparison helper ──────────────────────────────────────────────

/// Compare two f32 slices element-wise against the given tolerances.
///
/// For each element, both absolute and relative error are checked.
/// An element fails if *both* thresholds are exceeded (allowing large
/// absolute error on tiny values and large relative error near zero).
fn compare_tensors(cpu: &[f32], gpu: &[f32], tol: &NumericalTolerances) -> ComparisonResult {
    assert_eq!(cpu.len(), gpu.len(), "tensor length mismatch");
    let n = cpu.len();
    if n == 0 {
        return ComparisonResult {
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
            num_failures: 0,
            worst_index: 0,
            total_elements: 0,
        };
    }

    let mut max_abs_diff: f32 = 0.0;
    let mut sum_abs_diff: f64 = 0.0;
    let mut num_failures: usize = 0;
    let mut worst_index: usize = 0;

    for i in 0..n {
        let c = cpu[i];
        let g = gpu[i];

        // Both NaN → agree; one NaN → mismatch handled below
        if c.is_nan() && g.is_nan() {
            continue;
        }

        let abs_diff = (c - g).abs();
        let rel_diff = abs_diff / c.abs().max(1e-8);

        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
            worst_index = i;
        }
        sum_abs_diff += abs_diff as f64;

        // Fail only when BOTH thresholds are exceeded.
        // For exact-match (embedding_lookup), both thresholds are 0.0.
        let abs_exceeded = abs_diff > tol.max_abs_error;
        let rel_exceeded = rel_diff > tol.max_rel_error;
        if abs_exceeded && rel_exceeded {
            num_failures += 1;
        }

        // NaN in one but not the other is always a failure.
        if c.is_nan() != g.is_nan() {
            num_failures += 1;
        }
    }

    ComparisonResult {
        max_abs_diff,
        mean_abs_diff: (sum_abs_diff / n as f64) as f32,
        num_failures,
        worst_index,
        total_elements: n,
    }
}

// ── CPU reference implementations ──────────────────────────────────
// Inline reference functions so tests are self-contained and don't
// depend on any specific kernel backend being compiled.

/// Naive f32 matrix multiply: C[m×n] = A[m×k] · B[k×n].
fn cpu_matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for t in 0..k {
            let a_val = a[i * k + t];
            for j in 0..n {
                c[i * n + j] += a_val * b[t * n + j];
            }
        }
    }
    c
}

/// Row-wise numerically-stable softmax.
fn cpu_softmax(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = input.to_vec();
    for r in 0..rows {
        let row = &mut out[r * cols..(r + 1) * cols];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
    out
}

/// Layer normalization over the last `norm_size` elements per instance.
fn cpu_layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    norm_size: usize,
    eps: f32,
) -> Vec<f32> {
    let batch = input.len() / norm_size;
    let mut out = vec![0.0f32; input.len()];
    for b in 0..batch {
        let start = b * norm_size;
        let slice = &input[start..start + norm_size];
        let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;
        let var: f32 =
            slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / norm_size as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..norm_size {
            out[start + i] = (slice[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
    out
}

/// Scaled attention scores: (Q · K^T) / sqrt(head_dim), per head.
fn cpu_attention_scores(
    q: &[f32],
    k: &[f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; num_heads * seq_len * seq_len];
    for h in 0..num_heads {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[h * seq_len * head_dim + i * head_dim + d]
                        * k[h * seq_len * head_dim + j * head_dim + d];
                }
                scores[h * seq_len * seq_len + i * seq_len + j] = dot * scale;
            }
        }
    }
    scores
}

/// RoPE rotation: apply sin/cos rotary embeddings to pairs of elements.
fn cpu_rope(input: &[f32], seq_len: usize, head_dim: usize, base: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    let half = head_dim / 2;
    for pos in 0..seq_len {
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let idx0 = pos * head_dim + i;
            let idx1 = pos * head_dim + i + half;
            let x0 = input[idx0];
            let x1 = input[idx1];
            out[idx0] = x0 * cos_a - x1 * sin_a;
            out[idx1] = x0 * sin_a + x1 * cos_a;
        }
    }
    out
}

/// Kahan-compensated summation (higher accuracy than naive sum).
fn cpu_kahan_sum(data: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for &v in data {
        let y = v as f64 - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum as f32
}

// ── Deterministic seeded RNG (xorshift32) ──────────────────────────
// Avoids pulling in `rand` as a dependency for the test file.

struct Xorshift32(u32);

impl Xorshift32 {
    fn new(seed: u32) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniform f32 in [lo, hi).
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let t = (self.next_u32() as f64) / (u32::MAX as f64);
        lo + (hi - lo) * t as f32
    }

    fn fill_f32(&mut self, buf: &mut [f32], lo: f32, hi: f32) {
        for v in buf.iter_mut() {
            *v = self.next_f32(lo, hi);
        }
    }
}

// ── "GPU" simulation layer ─────────────────────────────────────────
// Until a real OpenCL runtime is available in the test environment,
// the gpu_* helpers call the same CPU reference with a reversed
// accumulation order to simulate reduction-order divergence. This
// lets us validate the harness infrastructure itself. When a real
// OpenCL backend is wired in, replace the body of each gpu_* function
// with the actual kernel dispatch.

fn gpu_matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // Simulate GPU by accumulating inner products in reverse order
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for t in (0..k).rev() {
                sum += a[i * k + t] * b[t * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn gpu_softmax(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    // Same stable algorithm — GPU would use parallel reduction
    cpu_softmax(input, rows, cols)
}

fn gpu_layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    norm_size: usize,
    eps: f32,
) -> Vec<f32> {
    cpu_layer_norm(input, gamma, beta, norm_size, eps)
}

fn gpu_attention_scores(
    q: &[f32],
    k: &[f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    // Simulate reversed dot-product accumulation
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; num_heads * seq_len * seq_len];
    for h in 0..num_heads {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in (0..head_dim).rev() {
                    dot += q[h * seq_len * head_dim + i * head_dim + d]
                        * k[h * seq_len * head_dim + j * head_dim + d];
                }
                scores[h * seq_len * seq_len + i * seq_len + j] = dot * scale;
            }
        }
    }
    scores
}

fn gpu_rope(input: &[f32], seq_len: usize, head_dim: usize, base: f32) -> Vec<f32> {
    cpu_rope(input, seq_len, head_dim, base)
}

fn gpu_reduce_sum(data: &[f32]) -> f32 {
    // Simulate tree-reduction (pairwise) which differs from sequential sum
    if data.is_empty() {
        return 0.0;
    }
    let mut buf: Vec<f32> = data.to_vec();
    while buf.len() > 1 {
        let mut next = Vec::with_capacity((buf.len() + 1) / 2);
        for chunk in buf.chunks(2) {
            if chunk.len() == 2 {
                next.push(chunk[0] + chunk[1]);
            } else {
                next.push(chunk[0]);
            }
        }
        buf = next;
    }
    buf[0]
}

// ── Assertion helper ───────────────────────────────────────────────

fn assert_comparison(result: &ComparisonResult, tol: &NumericalTolerances) {
    assert!(
        result.passed(),
        "[{}] numerical parity FAILED: {}\n  tolerances: max_abs={:.2e} max_rel={:.2e}",
        tol.label,
        result,
        tol.max_abs_error,
        tol.max_rel_error,
    );
    eprintln!("[{}] PASSED: {}", tol.label, result);
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires OpenCL device - run with BITNET_OPENCL_DEVICE=0"]
fn test_matmul_numerical_parity() {
    let sizes: &[(usize, usize, usize)] = &[
        (16, 16, 16),    // tiny square
        (64, 64, 64),    // small square
        (128, 256, 64),  // rectangular
        (33, 65, 17),    // non-power-of-2
        (1, 1024, 1),    // degenerate vector dot product
        (256, 256, 256), // medium square
    ];

    let mut rng = Xorshift32::new(42);
    for &(m, k, n) in sizes {
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        rng.fill_f32(&mut a, -1.0, 1.0);
        rng.fill_f32(&mut b, -1.0, 1.0);

        let cpu = cpu_matmul_f32(&a, &b, m, k, n);
        let gpu = gpu_matmul_f32(&a, &b, m, k, n);

        let result = compare_tensors(&cpu, &gpu, &NumericalTolerances::MATMUL);
        assert_comparison(&result, &NumericalTolerances::MATMUL);
    }
}

#[test]
#[ignore = "requires OpenCL device - run with BITNET_OPENCL_DEVICE=0"]
fn test_softmax_stability() {
    let cols = 128;
    let rows = 4;
    let mut rng = Xorshift32::new(123);

    // Normal values
    let mut normal = vec![0.0f32; rows * cols];
    rng.fill_f32(&mut normal, -5.0, 5.0);
    let cpu = cpu_softmax(&normal, rows, cols);
    let gpu = gpu_softmax(&normal, rows, cols);
    let result = compare_tensors(&cpu, &gpu, &NumericalTolerances::SOFTMAX);
    assert_comparison(&result, &NumericalTolerances::SOFTMAX);

    // Large logit values (overflow risk without max-subtraction)
    let mut large = vec![0.0f32; rows * cols];
    rng.fill_f32(&mut large, 80.0, 90.0);
    let cpu = cpu_softmax(&large, rows, cols);
    let gpu = gpu_softmax(&large, rows, cols);
    let result = compare_tensors(&cpu, &gpu, &NumericalTolerances::SOFTMAX);
    assert_comparison(&result, &NumericalTolerances::SOFTMAX);

    // Very negative values (underflow risk)
    let mut neg = vec![0.0f32; rows * cols];
    rng.fill_f32(&mut neg, -100.0, -80.0);
    let cpu = cpu_softmax(&neg, rows, cols);
    let gpu = gpu_softmax(&neg, rows, cols);
    let result = compare_tensors(&cpu, &gpu, &NumericalTolerances::SOFTMAX);
    assert_comparison(&result, &NumericalTolerances::SOFTMAX);

    // Near-zero logits (uniform distribution expected)
    let near_zero = vec![1e-10_f32; rows * cols];
    let cpu = cpu_softmax(&near_zero, rows, cols);
    let gpu = gpu_softmax(&near_zero, rows, cols);
    let result = compare_tensors(&cpu, &gpu, &NumericalTolerances::SOFTMAX);
    assert_comparison(&result, &NumericalTolerances::SOFTMAX);
}

#[test]
#[ignore = "requires OpenCL device - run with BITNET_OPENCL_DEVICE=0"]
fn test_layer_norm_small_variance() {
    let norm_size = 64;
    let batch = 4;
    let eps = 1e-5;
    let mut rng = Xorshift32::new(777);

    let mut gamma = vec![0.0f32; norm_size];
    let mut beta = vec![0.0f32; norm_size];
    rng.fill_f32(&mut gamma, 0.9, 1.1);
    rng.fill_f32(&mut beta, -0.1, 0.1);

    // Normal variance
    let mut normal_input = vec![0.0f32; batch * norm_size];
    rng.fill_f32(&mut normal_input, -2.0, 2.0);
    let cpu = cpu_layer_norm(&normal_input, &gamma, &beta, norm_size, eps);
    let gpu = gpu_layer_norm(&normal_input, &gamma, &beta, norm_size, eps);
    let result = compare_tensors(&cpu, &gpu, &NumericalTolerances::LAYER_NORM);
    assert_comparison(&result, &NumericalTolerances::LAYER_NORM);

    // Tiny variance (constant + tiny perturbation — stability stress test)
    let mut tiny_var_input = vec![1.0f32; batch * norm_size];
    for v in tiny_var_input.iter_mut() {
        *v += rng.next_f32(-1e-6, 1e-6);
    }
    let cpu = cpu_layer_norm(&tiny_var_input, &gamma, &beta, norm_size, eps);
    let gpu = gpu_layer_norm(&tiny_var_input, &gamma, &beta, norm_size, eps);
    let result = compare_tensors(&cpu, &gpu, &NumericalTolerances::LAYER_NORM);
    assert_comparison(&result, &NumericalTolerances::LAYER_NORM);

    // Large values
    let mut large_input = vec![0.0f32; batch * norm_size];
    rng.fill_f32(&mut large_input, 1e4, 1e5);
    let cpu = cpu_layer_norm(&large_input, &gamma, &beta, norm_size, eps);
    let gpu = gpu_layer_norm(&large_input, &gamma, &beta, norm_size, eps);
    let result = compare_tensors(&cpu, &gpu, &NumericalTolerances::LAYER_NORM);
    assert_comparison(&result, &NumericalTolerances::LAYER_NORM);
}

#[test]
#[ignore = "requires OpenCL device - run with BITNET_OPENCL_DEVICE=0"]
fn test_attention_scores_parity() {
    let configs: &[(usize, usize, usize)] = &[
        // (num_heads, seq_len, head_dim)
        (4, 16, 64),
        (8, 32, 64),
        (1, 128, 32),
        (2, 7, 13), // non-power-of-2 dimensions
    ];
    let mut rng = Xorshift32::new(999);

    for &(num_heads, seq_len, head_dim) in configs {
        let total = num_heads * seq_len * head_dim;
        let mut q = vec![0.0f32; total];
        let mut k = vec![0.0f32; total];
        rng.fill_f32(&mut q, -1.0, 1.0);
        rng.fill_f32(&mut k, -1.0, 1.0);

        let cpu = cpu_attention_scores(&q, &k, num_heads, seq_len, head_dim);
        let gpu = gpu_attention_scores(&q, &k, num_heads, seq_len, head_dim);

        let result = compare_tensors(&cpu, &gpu, &NumericalTolerances::ATTENTION_SCORES);
        assert_comparison(&result, &NumericalTolerances::ATTENTION_SCORES);
    }
}

#[test]
#[ignore = "requires OpenCL device - run with BITNET_OPENCL_DEVICE=0"]
fn test_rope_sincos_accuracy() {
    let configs: &[(usize, usize)] = &[
        (16, 64), // (seq_len, head_dim)
        (128, 128),
        (7, 32), // non-power-of-2 seq_len
    ];
    let base = 10000.0f32;
    let mut rng = Xorshift32::new(2025);

    for &(seq_len, head_dim) in configs {
        let total = seq_len * head_dim;
        let mut input = vec![0.0f32; total];
        rng.fill_f32(&mut input, -1.0, 1.0);

        let cpu = cpu_rope(&input, seq_len, head_dim, base);
        let gpu = gpu_rope(&input, seq_len, head_dim, base);

        let result = compare_tensors(&cpu, &gpu, &NumericalTolerances::ROPE);
        assert_comparison(&result, &NumericalTolerances::ROPE);
    }
}

#[test]
#[ignore = "requires OpenCL device - run with BITNET_OPENCL_DEVICE=0"]
fn test_reduction_sum_large() {
    // 1M floats: sequential sum vs tree-reduction can diverge measurably
    let n = 1_000_000;
    let mut rng = Xorshift32::new(314);
    let mut data = vec![0.0f32; n];
    rng.fill_f32(&mut data, -1.0, 1.0);

    let cpu = cpu_kahan_sum(&data);
    let gpu = gpu_reduce_sum(&data);

    let abs_diff = (cpu - gpu).abs();
    let rel_diff = abs_diff / cpu.abs().max(1e-8);
    eprintln!(
        "[reduction_sum_large] cpu={cpu:.8} gpu={gpu:.8} abs_diff={abs_diff:.2e} rel_diff={rel_diff:.2e}"
    );
    assert!(
        abs_diff <= NumericalTolerances::REDUCTION.max_abs_error
            || rel_diff <= NumericalTolerances::REDUCTION.max_rel_error,
        "reduction sum diverged: abs={abs_diff:.2e} rel={rel_diff:.2e}",
    );
}

#[test]
#[ignore = "requires OpenCL device - run with BITNET_OPENCL_DEVICE=0"]
fn test_denormal_handling() {
    // Subnormal (denormal) floats: GPU may flush to zero
    let denormals: Vec<f32> = (0..256)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            sign * f32::MIN_POSITIVE * (1.0 + i as f32 * 0.001) / (1u32 << 20) as f32
        })
        .collect();

    // Softmax on subnormals should still produce valid probabilities
    let cpu = cpu_softmax(&denormals, 1, denormals.len());
    let gpu = gpu_softmax(&denormals, 1, denormals.len());

    // Verify outputs are valid probabilities
    let cpu_sum: f32 = cpu.iter().sum();
    let gpu_sum: f32 = gpu.iter().sum();
    assert!((cpu_sum - 1.0).abs() < 1e-5, "CPU softmax sum={cpu_sum}, expected ~1.0");
    assert!((gpu_sum - 1.0).abs() < 1e-5, "GPU softmax sum={gpu_sum}, expected ~1.0");

    // Check parity (relaxed tolerance for denormal flushing)
    let tol =
        NumericalTolerances { max_abs_error: 1e-4, max_rel_error: 1e-2, label: "denormal_softmax" };
    let result = compare_tensors(&cpu, &gpu, &tol);
    assert_comparison(&result, &tol);

    // MatMul with denormal inputs
    let m = 16;
    let k = 16;
    let n = 16;
    let a: Vec<f32> = (0..m * k)
        .map(|i| f32::MIN_POSITIVE / (1u32 << 10) as f32 * ((i % 7) as f32 + 1.0))
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| f32::MIN_POSITIVE / (1u32 << 10) as f32 * ((i % 5) as f32 + 1.0))
        .collect();

    let cpu = cpu_matmul_f32(&a, &b, m, k, n);
    let gpu = gpu_matmul_f32(&a, &b, m, k, n);

    // With denormals, GPU might flush to zero — just verify no NaN/Inf
    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(!g.is_nan(), "GPU matmul produced NaN at index {i}");
        assert!(!g.is_infinite(), "GPU matmul produced Inf at index {i}");
        assert!(!c.is_nan(), "CPU matmul produced NaN at index {i}");
    }
    eprintln!("[denormal_matmul] completed: {} elements, no NaN/Inf detected", cpu.len());
}

#[test]
#[ignore = "requires OpenCL device - run with BITNET_OPENCL_DEVICE=0"]
fn test_nan_propagation() {
    let cols = 32;

    // NaN in softmax input → should produce NaN in same row
    let mut input_with_nan = vec![1.0f32; cols];
    input_with_nan[cols / 2] = f32::NAN;
    let cpu = cpu_softmax(&input_with_nan, 1, cols);
    let gpu = gpu_softmax(&input_with_nan, 1, cols);
    // Both should contain NaN
    assert!(cpu.iter().any(|v| v.is_nan()), "CPU softmax should propagate NaN");
    assert!(gpu.iter().any(|v| v.is_nan()), "GPU softmax should propagate NaN");

    // Infinity in input → softmax should still be defined
    let mut input_with_inf = vec![1.0f32; cols];
    input_with_inf[0] = f32::INFINITY;
    let cpu = cpu_softmax(&input_with_inf, 1, cols);
    let gpu = gpu_softmax(&input_with_inf, 1, cols);
    // The inf element should get softmax ≈ 1.0, others ≈ 0.0
    // Just verify no unexpected NaN in output
    let cpu_nan_count = cpu.iter().filter(|v| v.is_nan()).count();
    let gpu_nan_count = gpu.iter().filter(|v| v.is_nan()).count();
    assert_eq!(
        cpu_nan_count, gpu_nan_count,
        "NaN count mismatch: cpu={cpu_nan_count} gpu={gpu_nan_count}"
    );

    // NaN in matmul input → output should contain NaN
    let m = 4;
    let k = 4;
    let n = 4;
    let mut a = vec![1.0f32; m * k];
    a[0] = f32::NAN;
    let b = vec![1.0f32; k * n];
    let cpu = cpu_matmul_f32(&a, &b, m, k, n);
    let gpu = gpu_matmul_f32(&a, &b, m, k, n);
    assert!(cpu[0..n].iter().any(|v| v.is_nan()), "CPU matmul should propagate NaN to first row");
    assert!(gpu[0..n].iter().any(|v| v.is_nan()), "GPU matmul should propagate NaN to first row");
}

#[test]
#[ignore = "requires OpenCL device - run with BITNET_OPENCL_DEVICE=0"]
fn test_zero_input() {
    let n = 128;

    // All-zero matmul → all-zero output
    let a = vec![0.0f32; n * n];
    let b = vec![0.0f32; n * n];
    let cpu = cpu_matmul_f32(&a, &b, n, n, n);
    let gpu = gpu_matmul_f32(&a, &b, n, n, n);
    let result = compare_tensors(&cpu, &gpu, &NumericalTolerances::EMBEDDING_LOOKUP);
    assert_comparison(&result, &NumericalTolerances::EMBEDDING_LOOKUP);
    assert!(cpu.iter().all(|&v| v == 0.0), "CPU matmul of zeros should be all zeros");
    assert!(gpu.iter().all(|&v| v == 0.0), "GPU matmul of zeros should be all zeros");

    // All-zero softmax → uniform distribution (1/n each)
    let zeros = vec![0.0f32; n];
    let cpu = cpu_softmax(&zeros, 1, n);
    let gpu = gpu_softmax(&zeros, 1, n);
    let expected = 1.0 / n as f32;
    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!((c - expected).abs() < 1e-6, "CPU softmax(0)[{i}]={c}, expected {expected}");
        assert!((g - expected).abs() < 1e-6, "GPU softmax(0)[{i}]={g}, expected {expected}");
    }

    // Zero-vector reduction → 0.0
    let zero_data = vec![0.0f32; 1024];
    assert_eq!(cpu_kahan_sum(&zero_data), 0.0);
    assert_eq!(gpu_reduce_sum(&zero_data), 0.0);
}

#[test]
#[ignore = "requires OpenCL device - run with BITNET_OPENCL_DEVICE=0"]
fn test_large_values() {
    let n = 64;
    let mut rng = Xorshift32::new(1337);

    // Near FP32 max (but not overflowing when multiplied)
    let scale = (f32::MAX / (n as f32 * 100.0)).sqrt();
    let mut a = vec![0.0f32; n * n];
    let mut b = vec![0.0f32; n * n];
    rng.fill_f32(&mut a, -scale, scale);
    rng.fill_f32(&mut b, -1.0, 1.0); // keep B small to avoid overflow

    let cpu = cpu_matmul_f32(&a, &b, n, n, n);
    let gpu = gpu_matmul_f32(&a, &b, n, n, n);

    // Check no Inf/NaN in either output
    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(!c.is_nan() && !c.is_infinite(), "CPU overflow at index {i}: {c}");
        assert!(!g.is_nan() && !g.is_infinite(), "GPU overflow at index {i}: {g}");
    }

    // Relaxed tolerance for large-magnitude values
    let tol = NumericalTolerances {
        max_abs_error: scale * 1e-3,
        max_rel_error: 1e-3,
        label: "large_values_matmul",
    };
    let result = compare_tensors(&cpu, &gpu, &tol);
    assert_comparison(&result, &tol);

    // Layer norm should be scale-invariant — output should be the same
    // magnitude regardless of input scale
    let norm_size = n;
    let batch = 4;
    let gamma = vec![1.0f32; norm_size];
    let beta = vec![0.0f32; norm_size];
    let mut large_input = vec![0.0f32; batch * norm_size];
    rng.fill_f32(&mut large_input, 1e6, 1e7);

    let cpu = cpu_layer_norm(&large_input, &gamma, &beta, norm_size, 1e-5);
    let gpu = gpu_layer_norm(&large_input, &gamma, &beta, norm_size, 1e-5);

    let result = compare_tensors(&cpu, &gpu, &NumericalTolerances::LAYER_NORM);
    assert_comparison(&result, &NumericalTolerances::LAYER_NORM);

    // Verify output is normalized (roughly zero mean, unit variance)
    for b_idx in 0..batch {
        let start = b_idx * norm_size;
        let slice = &cpu[start..start + norm_size];
        let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;
        assert!(mean.abs() < 0.1, "layer_norm output mean should be near 0, got {mean}");
    }
}

// ── Unit tests for the comparison infrastructure itself ─────────────

#[cfg(test)]
mod harness_tests {
    use super::*;

    #[test]
    fn compare_identical_tensors() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tol = NumericalTolerances::MATMUL;
        let result = compare_tensors(&data, &data, &tol);
        assert!(result.passed());
        assert_eq!(result.max_abs_diff, 0.0);
        assert_eq!(result.num_failures, 0);
    }

    #[test]
    fn compare_with_small_diff() {
        let cpu = vec![1.0, 2.0, 3.0];
        let gpu = vec![1.0 + 1e-7, 2.0 - 1e-7, 3.0 + 1e-7];
        let tol = NumericalTolerances::MATMUL; // max_abs=1e-4
        let result = compare_tensors(&cpu, &gpu, &tol);
        assert!(result.passed());
    }

    #[test]
    fn compare_detects_large_diff() {
        let cpu = vec![1.0, 2.0, 3.0];
        let gpu = vec![1.0, 2.5, 3.0]; // 0.5 abs error at index 1
        let tol = NumericalTolerances::MATMUL;
        let result = compare_tensors(&cpu, &gpu, &tol);
        assert!(!result.passed());
        assert_eq!(result.worst_index, 1);
        assert!((result.max_abs_diff - 0.5).abs() < 1e-7);
    }

    #[test]
    fn compare_nan_agreement() {
        let cpu = vec![1.0, f32::NAN, 3.0];
        let gpu = vec![1.0, f32::NAN, 3.0];
        let tol = NumericalTolerances::MATMUL;
        let result = compare_tensors(&cpu, &gpu, &tol);
        assert!(result.passed());
    }

    #[test]
    fn compare_nan_disagreement() {
        let cpu = vec![1.0, f32::NAN, 3.0];
        let gpu = vec![1.0, 0.0, 3.0];
        let tol = NumericalTolerances::MATMUL;
        let result = compare_tensors(&cpu, &gpu, &tol);
        assert!(!result.passed());
    }

    #[test]
    fn compare_empty_tensors() {
        let empty: Vec<f32> = vec![];
        let tol = NumericalTolerances::MATMUL;
        let result = compare_tensors(&empty, &empty, &tol);
        assert!(result.passed());
        assert_eq!(result.total_elements, 0);
    }

    #[test]
    fn exact_match_tolerance() {
        let cpu = vec![1.0, 2.0, 3.0];
        let gpu = vec![1.0, 2.0, 3.0];
        let tol = NumericalTolerances::EMBEDDING_LOOKUP; // exact match
        let result = compare_tensors(&cpu, &gpu, &tol);
        assert!(result.passed());
    }

    #[test]
    fn xorshift_deterministic() {
        let mut rng1 = Xorshift32::new(42);
        let mut rng2 = Xorshift32::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn cpu_reference_softmax_sums_to_one() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = cpu_softmax(&input, 1, 4);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum={sum}");
    }

    #[test]
    fn cpu_reference_matmul_identity() {
        // A × I = A for 2×2
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let result = cpu_matmul_f32(&a, &identity, 2, 2, 2);
        assert_eq!(result, a);
    }
}
