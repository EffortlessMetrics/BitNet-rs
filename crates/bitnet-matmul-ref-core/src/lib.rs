//! Deterministic CPU reference matrix multiplication helpers.
//!
//! This crate isolates small, backend-agnostic matmul helpers that are used by
//! GPU validation tests and runtime sanity checks.

/// Error returned when matrix dimensions and flat buffer lengths do not match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulDimensionError {
    /// Human-friendly matrix name (`"matrix A"` or `"matrix B"`).
    pub name: &'static str,
    /// Expected flat element count.
    pub expected: usize,
    /// Actual flat element count.
    pub actual: usize,
}

impl std::fmt::Display for MatmulDimensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "invalid dimensions for {}: expected {}, got {}",
            self.name, self.expected, self.actual
        )
    }
}

impl std::error::Error for MatmulDimensionError {}

/// Validate flattened matrix buffer sizes for `M×K` and `K×N` inputs.
pub fn validate_flat_matmul_inputs(
    a_len: usize,
    b_len: usize,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), MatmulDimensionError> {
    let expected_a = (m * k) as usize;
    if a_len != expected_a {
        return Err(MatmulDimensionError { name: "matrix A", expected: expected_a, actual: a_len });
    }

    let expected_b = (k * n) as usize;
    if b_len != expected_b {
        return Err(MatmulDimensionError { name: "matrix B", expected: expected_b, actual: b_len });
    }

    Ok(())
}

/// Compute `C = A × B` for row-major flattened matrices.
pub fn cpu_matmul(a: &[f32], b: &[f32], m: u32, n: u32, k: u32) -> Vec<f32> {
    validate_flat_matmul_inputs(a.len(), b.len(), m, n, k).unwrap_or_else(|e| panic!("{e}"));

    let (m, n, k) = (m as usize, n as usize, k as usize);
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_matrix_a_size() {
        let err = validate_flat_matmul_inputs(3, 4, 2, 2, 2).expect_err("expected size mismatch");
        assert_eq!(err.name, "matrix A");
        assert_eq!(err.expected, 4);
        assert_eq!(err.actual, 3);
    }

    #[test]
    fn validates_matrix_b_size() {
        let err = validate_flat_matmul_inputs(4, 3, 2, 2, 2).expect_err("expected size mismatch");
        assert_eq!(err.name, "matrix B");
        assert_eq!(err.expected, 4);
        assert_eq!(err.actual, 3);
    }

    #[test]
    fn multiplies_non_square() {
        #[rustfmt::skip]
        let a = vec![1.0, 2.0, 3.0,
                     4.0, 5.0, 6.0];
        #[rustfmt::skip]
        let b = vec![7.0,  8.0,
                     9.0,  10.0,
                     11.0, 12.0];
        let c = cpu_matmul(&a, &b, 2, 2, 3);
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    #[should_panic(expected = "invalid dimensions for matrix A: expected 4, got 2")]
    fn cpu_matmul_panics_on_invalid_a() {
        cpu_matmul(&[1.0, 2.0], &[1.0; 4], 2, 2, 2);
    }
}
