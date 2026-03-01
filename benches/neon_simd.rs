//! Criterion micro-benchmarks for ARM NEON SIMD operations on Apple Silicon.
//!
//! Covers softmax, RMS normalization, quantized dot product, matrix-vector
//! multiply, and activation functions (SiLU / GELU).
//!
//! On non-aarch64 targets the benchmark group is empty so the file still
//! compiles without errors.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Helper: deterministic mock data
// ---------------------------------------------------------------------------

fn make_f32_vec(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) / (n as f32) - 0.5).collect()
}

fn make_i8_vec(n: usize) -> Vec<i8> {
    (0..n).map(|i| (i % 251) as i8).collect()
}

fn make_f32_matrix(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect()
}

// ===========================================================================
// aarch64 NEON implementations
// ===========================================================================
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// NEON-vectorised softmax over an f32 slice (in-place).
    ///
    /// 1. Find max via `vmaxq_f32` reduction.
    /// 2. Subtract max and compute exp (fast scalar fallback per lane –
    ///    a full NEON exp is out of scope for a micro-benchmark).
    /// 3. Sum and normalise.
    pub fn softmax_neon(v: &mut [f32]) {
        let n = v.len();
        if n == 0 {
            return;
        }

        // --- max reduction (NEON) ---
        let chunks = n / 4;
        let mut vmax = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
        for i in 0..chunks {
            let a = unsafe { vld1q_f32(v.as_ptr().add(i * 4)) };
            vmax = unsafe { vmaxq_f32(vmax, a) };
        }
        let mut max_val = unsafe { vmaxvq_f32(vmax) };
        for i in (chunks * 4)..n {
            if v[i] > max_val {
                max_val = v[i];
            }
        }

        // --- exp(x - max) ---
        for x in v.iter_mut() {
            *x = (*x - max_val).exp();
        }

        // --- sum reduction (NEON) ---
        let mut vsum = unsafe { vdupq_n_f32(0.0) };
        for i in 0..chunks {
            let a = unsafe { vld1q_f32(v.as_ptr().add(i * 4)) };
            vsum = unsafe { vaddq_f32(vsum, a) };
        }
        let mut sum: f32 = unsafe { vaddvq_f32(vsum) };
        for i in (chunks * 4)..n {
            sum += v[i];
        }

        // --- normalise ---
        let inv = 1.0 / sum;
        let vinv = unsafe { vdupq_n_f32(inv) };
        for i in 0..chunks {
            let a = unsafe { vld1q_f32(v.as_ptr().add(i * 4)) };
            let r = unsafe { vmulq_f32(a, vinv) };
            unsafe { vst1q_f32(v.as_mut_ptr().add(i * 4), r) };
        }
        for i in (chunks * 4)..n {
            v[i] *= inv;
        }
    }

    /// RMS normalisation: y_i = x_i / sqrt(mean(x²) + eps).
    pub fn rms_norm_neon(input: &[f32], output: &mut [f32], eps: f32) {
        let n = input.len();
        assert_eq!(n, output.len());
        if n == 0 {
            return;
        }

        let chunks = n / 4;
        let mut vsum_sq = unsafe { vdupq_n_f32(0.0) };
        for i in 0..chunks {
            let a = unsafe { vld1q_f32(input.as_ptr().add(i * 4)) };
            vsum_sq = unsafe { vfmaq_f32(vsum_sq, a, a) };
        }
        let mut sum_sq: f32 = unsafe { vaddvq_f32(vsum_sq) };
        for i in (chunks * 4)..n {
            sum_sq += input[i] * input[i];
        }

        let rms = (sum_sq / n as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        let vinv = unsafe { vdupq_n_f32(inv_rms) };

        for i in 0..chunks {
            let a = unsafe { vld1q_f32(input.as_ptr().add(i * 4)) };
            let r = unsafe { vmulq_f32(a, vinv) };
            unsafe { vst1q_f32(output.as_mut_ptr().add(i * 4), r) };
        }
        for i in (chunks * 4)..n {
            output[i] = input[i] * inv_rms;
        }
    }

    /// Quantised dot product: Σ (a_i8 * b_i8) accumulated as i32, returned as f32
    /// scaled by `scale`.
    pub fn quantized_dot_neon(a: &[i8], b: &[i8], scale: f32) -> f32 {
        let n = a.len();
        assert_eq!(n, b.len());

        let chunks = n / 16;
        let mut vacc = unsafe { vdupq_n_s32(0) };

        for i in 0..chunks {
            let va = unsafe { vld1q_s8(a.as_ptr().add(i * 16)) };
            let vb = unsafe { vld1q_s8(b.as_ptr().add(i * 16)) };

            // Widening multiply-add: low and high halves.
            let lo_a = unsafe { vget_low_s8(va) };
            let hi_a = unsafe { vget_high_s8(va) };
            let lo_b = unsafe { vget_low_s8(vb) };
            let hi_b = unsafe { vget_high_s8(vb) };

            let prod_lo = unsafe { vmull_s8(lo_a, lo_b) };
            let prod_hi = unsafe { vmull_s8(hi_a, hi_b) };

            vacc = unsafe { vpadalq_s16(vacc, prod_lo) };
            vacc = unsafe { vpadalq_s16(vacc, prod_hi) };
        }

        let mut acc: i32 = unsafe { vaddvq_s32(vacc) };
        for i in (chunks * 16)..n {
            acc += (a[i] as i32) * (b[i] as i32);
        }

        acc as f32 * scale
    }

    /// Small matrix-vector product: y = M·x, M is (rows × cols), x is (cols,).
    pub fn matvec_neon(mat: &[f32], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
        assert_eq!(mat.len(), rows * cols);
        assert_eq!(x.len(), cols);
        assert_eq!(y.len(), rows);

        let chunks = cols / 4;
        for r in 0..rows {
            let row = &mat[r * cols..(r + 1) * cols];
            let mut vsum = unsafe { vdupq_n_f32(0.0) };
            for c in 0..chunks {
                let va = unsafe { vld1q_f32(row.as_ptr().add(c * 4)) };
                let vb = unsafe { vld1q_f32(x.as_ptr().add(c * 4)) };
                vsum = unsafe { vfmaq_f32(vsum, va, vb) };
            }
            let mut s: f32 = unsafe { vaddvq_f32(vsum) };
            for c in (chunks * 4)..cols {
                s += row[c] * x[c];
            }
            y[r] = s;
        }
    }

    /// SiLU activation: x * σ(x), vectorised over f32x4.
    pub fn silu_neon(input: &[f32], output: &mut [f32]) {
        let n = input.len();
        assert_eq!(n, output.len());

        let chunks = n / 4;
        for i in 0..chunks {
            let vx = unsafe { vld1q_f32(input.as_ptr().add(i * 4)) };
            // σ(x) ≈ scalar per lane (intrinsic sigmoid not available).
            let mut buf = [0.0f32; 4];
            unsafe { vst1q_f32(buf.as_mut_ptr(), vx) };
            for b in &mut buf {
                *b = *b * (1.0 / (1.0 + (-*b).exp()));
            }
            let vr = unsafe { vld1q_f32(buf.as_ptr()) };
            unsafe { vst1q_f32(output.as_mut_ptr().add(i * 4), vr) };
        }
        for i in (chunks * 4)..n {
            let x = input[i];
            output[i] = x * (1.0 / (1.0 + (-x).exp()));
        }
    }

    /// GELU activation (tanh approximation), vectorised over f32x4.
    pub fn gelu_neon(input: &[f32], output: &mut [f32]) {
        let n = input.len();
        assert_eq!(n, output.len());
        let sqrt_2_over_pi: f32 = 0.797_884_56;
        let coeff: f32 = 0.044_715;

        let chunks = n / 4;
        for i in 0..chunks {
            let vx = unsafe { vld1q_f32(input.as_ptr().add(i * 4)) };
            let mut buf = [0.0f32; 4];
            unsafe { vst1q_f32(buf.as_mut_ptr(), vx) };
            for b in &mut buf {
                let x = *b;
                let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
                *b = 0.5 * x * (1.0 + inner.tanh());
            }
            let vr = unsafe { vld1q_f32(buf.as_ptr()) };
            unsafe { vst1q_f32(output.as_mut_ptr().add(i * 4), vr) };
        }
        for i in (chunks * 4)..n {
            let x = input[i];
            let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            output[i] = 0.5 * x * (1.0 + inner.tanh());
        }
    }
}

// ===========================================================================
// Benchmark groups
// ===========================================================================

#[cfg(target_arch = "aarch64")]
fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("neon_softmax");
    for &size in &[256, 1024, 4096] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            b.iter_batched(
                || make_f32_vec(n),
                |mut v| {
                    neon::softmax_neon(&mut v);
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

#[cfg(target_arch = "aarch64")]
fn bench_rms_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("neon_rms_norm");
    for &size in &[256, 1024, 4096] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            let input = make_f32_vec(n);
            let mut output = vec![0.0f32; n];
            b.iter(|| {
                neon::rms_norm_neon(&input, &mut output, 1e-5);
                black_box(&output);
            });
        });
    }
    group.finish();
}

#[cfg(target_arch = "aarch64")]
fn bench_quantized_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("neon_quantized_dot");
    for &size in &[256, 1024, 4096] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            let a = make_i8_vec(n);
            let bv = make_i8_vec(n);
            b.iter(|| black_box(neon::quantized_dot_neon(&a, &bv, 0.01)));
        });
    }
    group.finish();
}

#[cfg(target_arch = "aarch64")]
fn bench_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("neon_matvec");
    for &size in &[256, 512, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            let mat = make_f32_matrix(n, n);
            let x = make_f32_vec(n);
            let mut y = vec![0.0f32; n];
            b.iter(|| {
                neon::matvec_neon(&mat, &x, &mut y, n, n);
                black_box(&y);
            });
        });
    }
    group.finish();
}

#[cfg(target_arch = "aarch64")]
fn bench_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("neon_activations");
    for &size in &[256, 1024, 4096] {
        let input = make_f32_vec(size);
        let mut output = vec![0.0f32; size];
        group.bench_with_input(BenchmarkId::new("silu", size), &size, |b, &_n| {
            b.iter(|| {
                neon::silu_neon(&input, &mut output);
                black_box(&output);
            });
        });
        group.bench_with_input(BenchmarkId::new("gelu", size), &size, |b, &_n| {
            b.iter(|| {
                neon::gelu_neon(&input, &mut output);
                black_box(&output);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion wiring
// ---------------------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
criterion_group!(
    benches,
    bench_softmax,
    bench_rms_norm,
    bench_quantized_dot,
    bench_matvec,
    bench_activations,
);

// Stub for non-aarch64 targets so the file still compiles.
#[cfg(not(target_arch = "aarch64"))]
fn _stub(_c: &mut Criterion) {}

#[cfg(not(target_arch = "aarch64"))]
criterion_group!(benches, _stub);

criterion_main!(benches);
