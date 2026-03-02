//! CPU weight initialization kernels.
//!
//! Provides common initialization strategies (zeros, ones, uniform, normal,
//! Xavier, Kaiming, orthogonal, constant) using a dependency-free xorshift64
//! PRNG and Box-Muller transform for normal samples.

use std::fmt;

// ── Error type ────────────────────────────────────────────────────────

/// Errors produced by weight initialization routines.
#[derive(Debug, Clone, PartialEq)]
pub enum InitError {
    /// Fan dimension must be non-zero.
    ZeroFan(&'static str),
    /// Uniform range is invalid (`low >= high`).
    InvalidRange { low: f32, high: f32 },
    /// Standard deviation must be positive and finite.
    InvalidStd(f32),
}

impl fmt::Display for InitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroFan(name) => write!(f, "{name} must be non-zero"),
            Self::InvalidRange { low, high } => {
                write!(f, "invalid uniform range: low ({low}) >= high ({high})")
            }
            Self::InvalidStd(v) => write!(f, "std must be positive and finite, got {v}"),
        }
    }
}

impl std::error::Error for InitError {}

// ── Strategy enum ─────────────────────────────────────────────────────

/// Weight initialization strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum InitStrategy {
    /// Fill with zeros.
    Zeros,
    /// Fill with ones.
    Ones,
    /// Uniform distribution U(low, high).
    Uniform,
    /// Normal distribution N(mean, std).
    Normal,
    /// Xavier/Glorot uniform: U(−a, a) where a = √(6 / (fan_in + fan_out)).
    XavierUniform,
    /// Xavier/Glorot normal: N(0, √(2 / (fan_in + fan_out))).
    XavierNormal,
    /// Kaiming/He uniform: U(−a, a) where a = √(6 / fan_in).
    KaimingUniform,
    /// Kaiming/He normal: N(0, √(2 / fan_in)).
    KaimingNormal,
    /// Orthogonal initialization (via Gram-Schmidt on a random matrix).
    Orthogonal,
    /// Fill with a constant value.
    Constant(f32),
}

// ── Xorshift64 PRNG ──────────────────────────────────────────────────

/// Minimal xorshift64 PRNG — no external dependencies.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state.
        Self(if seed == 0 { 0x5EED_CAFE_BABE_D00D } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Returns a float in (0, 1) — open interval to keep Box-Muller safe.
    fn next_f32(&mut self) -> f32 {
        // Use upper 23 bits for the mantissa + 1 to avoid exact 0.
        let v = ((self.next_u64() >> 40) as f32 + 1.0) / ((1u64 << 24) as f32);
        v.clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON)
    }
}

// ── Box-Muller ────────────────────────────────────────────────────────

fn box_muller(u1: f32, u2: f32) -> (f32, f32) {
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

// ── Public initializers ───────────────────────────────────────────────

/// Fill `output` with zeros.
pub fn initialize_zeros(output: &mut [f32]) {
    output.fill(0.0);
}

/// Fill `output` with ones.
pub fn initialize_ones(output: &mut [f32]) {
    output.fill(1.0);
}

/// Fill `output` with U(`low`, `high`).
pub fn initialize_uniform(
    output: &mut [f32],
    low: f32,
    high: f32,
    seed: u64,
) -> Result<(), InitError> {
    if low >= high {
        return Err(InitError::InvalidRange { low, high });
    }
    let mut rng = Xorshift64::new(seed);
    let range = high - low;
    for v in output.iter_mut() {
        *v = low + rng.next_f32() * range;
    }
    Ok(())
}

/// Fill `output` with N(`mean`, `std`).
pub fn initialize_normal(
    output: &mut [f32],
    mean: f32,
    std: f32,
    seed: u64,
) -> Result<(), InitError> {
    if !std.is_finite() || std <= 0.0 {
        return Err(InitError::InvalidStd(std));
    }
    let mut rng = Xorshift64::new(seed);
    let mut i = 0;
    while i < output.len() {
        let (z0, z1) = box_muller(rng.next_f32(), rng.next_f32());
        output[i] = mean + std * z0;
        i += 1;
        if i < output.len() {
            output[i] = mean + std * z1;
            i += 1;
        }
    }
    Ok(())
}

/// Xavier/Glorot uniform: U(−a, a), a = √(6 / (fan_in + fan_out)).
pub fn initialize_xavier_uniform(
    output: &mut [f32],
    fan_in: usize,
    fan_out: usize,
    seed: u64,
) -> Result<(), InitError> {
    if fan_in == 0 {
        return Err(InitError::ZeroFan("fan_in"));
    }
    if fan_out == 0 {
        return Err(InitError::ZeroFan("fan_out"));
    }
    let a = (6.0 / (fan_in + fan_out) as f32).sqrt();
    initialize_uniform(output, -a, a, seed)
}

/// Xavier/Glorot normal: N(0, √(2 / (fan_in + fan_out))).
pub fn initialize_xavier_normal(
    output: &mut [f32],
    fan_in: usize,
    fan_out: usize,
    seed: u64,
) -> Result<(), InitError> {
    if fan_in == 0 {
        return Err(InitError::ZeroFan("fan_in"));
    }
    if fan_out == 0 {
        return Err(InitError::ZeroFan("fan_out"));
    }
    let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
    initialize_normal(output, 0.0, std, seed)
}

/// Kaiming/He uniform: U(−a, a), a = √(6 / fan_in).
pub fn initialize_kaiming_uniform(
    output: &mut [f32],
    fan_in: usize,
    seed: u64,
) -> Result<(), InitError> {
    if fan_in == 0 {
        return Err(InitError::ZeroFan("fan_in"));
    }
    let a = (6.0 / fan_in as f32).sqrt();
    initialize_uniform(output, -a, a, seed)
}

/// Kaiming/He normal: N(0, √(2 / fan_in)).
pub fn initialize_kaiming_normal(
    output: &mut [f32],
    fan_in: usize,
    seed: u64,
) -> Result<(), InitError> {
    if fan_in == 0 {
        return Err(InitError::ZeroFan("fan_in"));
    }
    let std = (2.0 / fan_in as f32).sqrt();
    initialize_normal(output, 0.0, std, seed)
}

/// Orthogonal initialization via Gram-Schmidt on a random matrix.
///
/// Produces a `rows × cols` matrix (stored row-major in `output`) whose
/// rows (if rows ≤ cols) or columns (if cols < rows) are orthonormal.
pub fn initialize_orthogonal(
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seed: u64,
) -> Result<(), InitError> {
    if rows == 0 {
        return Err(InitError::ZeroFan("rows"));
    }
    if cols == 0 {
        return Err(InitError::ZeroFan("cols"));
    }
    let n = rows * cols;
    if output.len() < n {
        return Ok(());
    }

    // Fill with normal random values first.
    initialize_normal(&mut output[..n], 0.0, 1.0, seed)?;

    // Gram-Schmidt over the shorter dimension.
    let (vectors, vec_len) = if rows <= cols { (rows, cols) } else { (cols, rows) };

    // Helper closures for row-major access.
    let idx = |r: usize, c: usize| -> usize { r * cols + c };

    if rows <= cols {
        // Orthonormalize rows.
        for i in 0..vectors {
            // Subtract projections onto all previous rows.
            for j in 0..i {
                let mut dot = 0.0f32;
                for k in 0..vec_len {
                    dot += output[idx(i, k)] * output[idx(j, k)];
                }
                for k in 0..vec_len {
                    output[idx(i, k)] -= dot * output[idx(j, k)];
                }
            }
            // Normalize.
            let mut norm = 0.0f32;
            for k in 0..vec_len {
                norm += output[idx(i, k)] * output[idx(i, k)];
            }
            let norm = norm.sqrt().max(f32::EPSILON);
            for k in 0..vec_len {
                output[idx(i, k)] /= norm;
            }
        }
    } else {
        // Orthonormalize columns.
        for i in 0..vectors {
            for j in 0..i {
                let mut dot = 0.0f32;
                for r in 0..rows {
                    dot += output[idx(r, i)] * output[idx(r, j)];
                }
                for r in 0..rows {
                    output[idx(r, i)] -= dot * output[idx(r, j)];
                }
            }
            let mut norm = 0.0f32;
            for r in 0..rows {
                norm += output[idx(r, i)] * output[idx(r, i)];
            }
            let norm = norm.sqrt().max(f32::EPSILON);
            for r in 0..rows {
                output[idx(r, i)] /= norm;
            }
        }
    }
    Ok(())
}

/// Dispatch initialization by [`InitStrategy`].
///
/// For `Zeros`, `Ones`, `Constant` the fan/seed parameters are unused.
/// For `Uniform` the range defaults to U(−1, 1).
/// For `Normal` the distribution defaults to N(0, 1).
pub fn initialize(
    output: &mut [f32],
    strategy: &InitStrategy,
    fan_in: usize,
    fan_out: usize,
    seed: u64,
) -> Result<(), InitError> {
    match strategy {
        InitStrategy::Zeros => {
            initialize_zeros(output);
            Ok(())
        }
        InitStrategy::Ones => {
            initialize_ones(output);
            Ok(())
        }
        InitStrategy::Uniform => initialize_uniform(output, -1.0, 1.0, seed),
        InitStrategy::Normal => initialize_normal(output, 0.0, 1.0, seed),
        InitStrategy::XavierUniform => initialize_xavier_uniform(output, fan_in, fan_out, seed),
        InitStrategy::XavierNormal => initialize_xavier_normal(output, fan_in, fan_out, seed),
        InitStrategy::KaimingUniform => initialize_kaiming_uniform(output, fan_in, seed),
        InitStrategy::KaimingNormal => initialize_kaiming_normal(output, fan_in, seed),
        InitStrategy::Orthogonal => initialize_orthogonal(output, fan_in, fan_out, seed),
        InitStrategy::Constant(c) => {
            output.fill(*c);
            Ok(())
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 10_000;
    const SEED: u64 = 42;

    fn variance(data: &[f32]) -> f32 {
        let n = data.len() as f32;
        let mean = data.iter().sum::<f32>() / n;
        data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n
    }

    fn mean(data: &[f32]) -> f32 {
        data.iter().sum::<f32>() / data.len() as f32
    }

    // ── Zeros / Ones / Constant ───────────────────────────────────

    #[test]
    fn test_zeros() {
        let mut buf = vec![1.0; 64];
        initialize_zeros(&mut buf);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ones() {
        let mut buf = vec![0.0; 64];
        initialize_ones(&mut buf);
        assert!(buf.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_constant() {
        let mut buf = vec![0.0; 64];
        initialize(&mut buf, &InitStrategy::Constant(3.14), 1, 1, 0).unwrap();
        assert!(buf.iter().all(|&v| (v - 3.14).abs() < 1e-6));
    }

    #[test]
    fn test_zeros_empty() {
        let mut buf: Vec<f32> = vec![];
        initialize_zeros(&mut buf);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_ones_empty() {
        let mut buf: Vec<f32> = vec![];
        initialize_ones(&mut buf);
        assert!(buf.is_empty());
    }

    // ── Uniform ───────────────────────────────────────────────────

    #[test]
    fn test_uniform_range() {
        let mut buf = vec![0.0; N];
        initialize_uniform(&mut buf, -1.0, 1.0, SEED).unwrap();
        assert!(buf.iter().all(|&v| (-1.0..1.0).contains(&v)));
    }

    #[test]
    fn test_uniform_custom_range() {
        let mut buf = vec![0.0; N];
        initialize_uniform(&mut buf, 2.0, 5.0, SEED).unwrap();
        assert!(buf.iter().all(|&v| (2.0..5.0).contains(&v)));
    }

    #[test]
    fn test_uniform_invalid_range() {
        let mut buf = vec![0.0; 8];
        assert!(initialize_uniform(&mut buf, 1.0, 1.0, SEED).is_err());
        assert!(initialize_uniform(&mut buf, 2.0, 1.0, SEED).is_err());
    }

    #[test]
    fn test_uniform_empty() {
        let mut buf: Vec<f32> = vec![];
        initialize_uniform(&mut buf, -1.0, 1.0, SEED).unwrap();
    }

    // ── Normal ────────────────────────────────────────────────────

    #[test]
    fn test_normal_mean_std() {
        let mut buf = vec![0.0; N];
        initialize_normal(&mut buf, 0.0, 1.0, SEED).unwrap();
        let m = mean(&buf);
        let v = variance(&buf);
        assert!(m.abs() < 0.1, "mean {m} too far from 0");
        assert!((v - 1.0).abs() < 0.15, "variance {v} too far from 1");
    }

    #[test]
    fn test_normal_custom_mean_std() {
        let mut buf = vec![0.0; N];
        initialize_normal(&mut buf, 5.0, 2.0, SEED).unwrap();
        let m = mean(&buf);
        let v = variance(&buf);
        assert!((m - 5.0).abs() < 0.2, "mean {m} too far from 5");
        assert!((v - 4.0).abs() < 0.5, "variance {v} too far from 4");
    }

    #[test]
    fn test_normal_invalid_std() {
        let mut buf = vec![0.0; 8];
        assert!(initialize_normal(&mut buf, 0.0, 0.0, SEED).is_err());
        assert!(initialize_normal(&mut buf, 0.0, -1.0, SEED).is_err());
        assert!(initialize_normal(&mut buf, 0.0, f32::INFINITY, SEED).is_err());
        assert!(initialize_normal(&mut buf, 0.0, f32::NAN, SEED).is_err());
    }

    #[test]
    fn test_normal_empty() {
        let mut buf: Vec<f32> = vec![];
        initialize_normal(&mut buf, 0.0, 1.0, SEED).unwrap();
    }

    // ── Xavier ────────────────────────────────────────────────────

    #[test]
    fn test_xavier_uniform_range() {
        let (fan_in, fan_out) = (256, 512);
        let mut buf = vec![0.0; N];
        initialize_xavier_uniform(&mut buf, fan_in, fan_out, SEED).unwrap();
        let a = (6.0 / (fan_in + fan_out) as f32).sqrt();
        assert!(buf.iter().all(|&v| v >= -a && v < a));
    }

    #[test]
    fn test_xavier_uniform_variance() {
        let (fan_in, fan_out) = (256, 512);
        let mut buf = vec![0.0; N];
        initialize_xavier_uniform(&mut buf, fan_in, fan_out, SEED).unwrap();
        let expected_var = 2.0 / (fan_in + fan_out) as f32;
        let v = variance(&buf);
        let rel_err = (v - expected_var).abs() / expected_var;
        assert!(rel_err < 0.15, "xavier uniform var {v} vs expected {expected_var}");
    }

    #[test]
    fn test_xavier_normal_variance() {
        let (fan_in, fan_out) = (256, 512);
        let mut buf = vec![0.0; N];
        initialize_xavier_normal(&mut buf, fan_in, fan_out, SEED).unwrap();
        let expected_var = 2.0 / (fan_in + fan_out) as f32;
        let v = variance(&buf);
        let rel_err = (v - expected_var).abs() / expected_var;
        assert!(rel_err < 0.15, "xavier normal var {v} vs expected {expected_var}");
    }

    #[test]
    fn test_xavier_zero_fan_in() {
        let mut buf = vec![0.0; 8];
        assert_eq!(
            initialize_xavier_uniform(&mut buf, 0, 128, SEED),
            Err(InitError::ZeroFan("fan_in"))
        );
    }

    #[test]
    fn test_xavier_zero_fan_out() {
        let mut buf = vec![0.0; 8];
        assert_eq!(
            initialize_xavier_normal(&mut buf, 128, 0, SEED),
            Err(InitError::ZeroFan("fan_out"))
        );
    }

    // ── Kaiming ───────────────────────────────────────────────────

    #[test]
    fn test_kaiming_uniform_range() {
        let fan_in = 512;
        let mut buf = vec![0.0; N];
        initialize_kaiming_uniform(&mut buf, fan_in, SEED).unwrap();
        let a = (6.0 / fan_in as f32).sqrt();
        assert!(buf.iter().all(|&v| v >= -a && v < a));
    }

    #[test]
    fn test_kaiming_uniform_variance() {
        let fan_in = 512;
        let mut buf = vec![0.0; N];
        initialize_kaiming_uniform(&mut buf, fan_in, SEED).unwrap();
        let expected_var = 2.0 / fan_in as f32;
        let v = variance(&buf);
        let rel_err = (v - expected_var).abs() / expected_var;
        assert!(rel_err < 0.15, "kaiming uniform var {v} vs expected {expected_var}");
    }

    #[test]
    fn test_kaiming_normal_variance() {
        let fan_in = 512;
        let mut buf = vec![0.0; N];
        initialize_kaiming_normal(&mut buf, fan_in, SEED).unwrap();
        let expected_var = 2.0 / fan_in as f32;
        let v = variance(&buf);
        let rel_err = (v - expected_var).abs() / expected_var;
        assert!(rel_err < 0.15, "kaiming normal var {v} vs expected {expected_var}");
    }

    #[test]
    fn test_kaiming_zero_fan_in() {
        let mut buf = vec![0.0; 8];
        assert_eq!(
            initialize_kaiming_uniform(&mut buf, 0, SEED),
            Err(InitError::ZeroFan("fan_in"))
        );
        assert_eq!(initialize_kaiming_normal(&mut buf, 0, SEED), Err(InitError::ZeroFan("fan_in")));
    }

    // ── Orthogonal ────────────────────────────────────────────────

    #[test]
    fn test_orthogonal_rows_orthonormal() {
        let (rows, cols) = (4, 8);
        let mut buf = vec![0.0; rows * cols];
        initialize_orthogonal(&mut buf, rows, cols, SEED).unwrap();

        for i in 0..rows {
            // Unit norm.
            let norm: f32 = (0..cols).map(|k| buf[i * cols + k].powi(2)).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "row {i} norm = {norm}");
            // Orthogonality.
            for j in (i + 1)..rows {
                let dot: f32 = (0..cols).map(|k| buf[i * cols + k] * buf[j * cols + k]).sum();
                assert!(dot.abs() < 1e-4, "dot(row{i}, row{j}) = {dot}");
            }
        }
    }

    #[test]
    fn test_orthogonal_cols_orthonormal() {
        let (rows, cols) = (8, 4);
        let mut buf = vec![0.0; rows * cols];
        initialize_orthogonal(&mut buf, rows, cols, SEED).unwrap();

        for i in 0..cols {
            let norm: f32 = (0..rows).map(|r| buf[r * cols + i].powi(2)).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "col {i} norm = {norm}");
            for j in (i + 1)..cols {
                let dot: f32 = (0..rows).map(|r| buf[r * cols + i] * buf[r * cols + j]).sum();
                assert!(dot.abs() < 1e-4, "dot(col{i}, col{j}) = {dot}");
            }
        }
    }

    #[test]
    fn test_orthogonal_zero_dims() {
        let mut buf = vec![0.0; 16];
        assert_eq!(initialize_orthogonal(&mut buf, 0, 4, SEED), Err(InitError::ZeroFan("rows")));
        assert_eq!(initialize_orthogonal(&mut buf, 4, 0, SEED), Err(InitError::ZeroFan("cols")));
    }

    // ── Determinism ───────────────────────────────────────────────

    #[test]
    fn test_determinism_uniform() {
        let mut a = vec![0.0; 256];
        let mut b = vec![0.0; 256];
        initialize_uniform(&mut a, -1.0, 1.0, 123).unwrap();
        initialize_uniform(&mut b, -1.0, 1.0, 123).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_determinism_normal() {
        let mut a = vec![0.0; 256];
        let mut b = vec![0.0; 256];
        initialize_normal(&mut a, 0.0, 1.0, 99).unwrap();
        initialize_normal(&mut b, 0.0, 1.0, 99).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_determinism_xavier() {
        let mut a = vec![0.0; 256];
        let mut b = vec![0.0; 256];
        initialize_xavier_uniform(&mut a, 128, 64, 77).unwrap();
        initialize_xavier_uniform(&mut b, 128, 64, 77).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_determinism_kaiming() {
        let mut a = vec![0.0; 256];
        let mut b = vec![0.0; 256];
        initialize_kaiming_normal(&mut a, 256, 55).unwrap();
        initialize_kaiming_normal(&mut b, 256, 55).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_determinism_orthogonal() {
        let mut a = vec![0.0; 32];
        let mut b = vec![0.0; 32];
        initialize_orthogonal(&mut a, 4, 8, 13).unwrap();
        initialize_orthogonal(&mut b, 4, 8, 13).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_different_seeds_differ() {
        let mut a = vec![0.0; 256];
        let mut b = vec![0.0; 256];
        initialize_uniform(&mut a, -1.0, 1.0, 1).unwrap();
        initialize_uniform(&mut b, -1.0, 1.0, 2).unwrap();
        assert_ne!(a, b);
    }

    // ── Dispatch ──────────────────────────────────────────────────

    #[test]
    fn test_dispatch_zeros() {
        let mut buf = vec![1.0; 16];
        initialize(&mut buf, &InitStrategy::Zeros, 1, 1, 0).unwrap();
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dispatch_ones() {
        let mut buf = vec![0.0; 16];
        initialize(&mut buf, &InitStrategy::Ones, 1, 1, 0).unwrap();
        assert!(buf.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_dispatch_uniform() {
        let mut buf = vec![0.0; N];
        initialize(&mut buf, &InitStrategy::Uniform, 1, 1, SEED).unwrap();
        assert!(buf.iter().all(|&v| (-1.0..1.0).contains(&v)));
    }

    #[test]
    fn test_dispatch_xavier_normal() {
        let mut buf = vec![0.0; N];
        initialize(&mut buf, &InitStrategy::XavierNormal, 256, 512, SEED).unwrap();
        let expected_var = 2.0 / (256 + 512) as f32;
        let v = variance(&buf);
        let rel_err = (v - expected_var).abs() / expected_var;
        assert!(rel_err < 0.15);
    }

    #[test]
    fn test_dispatch_kaiming_uniform() {
        let mut buf = vec![0.0; N];
        initialize(&mut buf, &InitStrategy::KaimingUniform, 512, 1, SEED).unwrap();
        let a = (6.0f32 / 512.0).sqrt();
        assert!(buf.iter().all(|&v| v >= -a && v < a));
    }

    // ── Error Display ─────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = InitError::ZeroFan("fan_in");
        assert_eq!(e.to_string(), "fan_in must be non-zero");

        let e = InitError::InvalidRange { low: 2.0, high: 1.0 };
        assert!(e.to_string().contains("invalid uniform range"));

        let e = InitError::InvalidStd(-1.0);
        assert!(e.to_string().contains("positive and finite"));
    }

    #[test]
    fn test_error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(InitError::ZeroFan("x"));
        assert!(e.to_string().contains("must be non-zero"));
    }
}
