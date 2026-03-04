#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unsafe_op_in_unsafe_fn)]
//! Apple Silicon NEON ARM64 regression tests.
//!
//! Validates NEON-accelerated kernels against scalar reference implementations
//! to catch correctness regressions on AArch64. Covers elementwise ops,
//! reductions, FMA pipelines, and softmax.
#![cfg(all(feature = "cpu", target_arch = "aarch64"))]
#![allow(clippy::undocumented_unsafe_blocks, unused_unsafe)]

#[cfg(test)]
mod tests {
    use bitnet_kernels::cpu::neon_elementwise::*;
    use bitnet_kernels::cpu::neon_fma_ops::*;
    use bitnet_kernels::cpu::neon_reductions::*;
    use bitnet_kernels::cpu::neon_softmax::*;

    const EPS: f32 = 1e-5;

    fn assert_approx(a: f32, b: f32, ctx: &str) {
        assert!((a - b).abs() < EPS, "{ctx}: expected {a} ≈ {b} (diff = {})", (a - b).abs());
    }

    fn assert_slices_approx(a: &[f32], b: &[f32], ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() < EPS,
                "{ctx}[{i}]: expected {x} ≈ {y} (diff = {})",
                (x - y).abs()
            );
        }
    }

    // ── Scalar references ────────────────────────────────────────────

    fn scalar_add(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    fn scalar_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x * y).collect()
    }

    fn scalar_fma(a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
        a.iter().zip(b).zip(c).map(|((x, y), z)| x.mul_add(*y, *z)).collect()
    }

    fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn scalar_sum(a: &[f32]) -> f32 {
        a.iter().sum()
    }

    // ── Elementwise tests ────────────────────────────────────────────

    #[test]
    fn neon_add_matches_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let expected = scalar_add(&a, &b);
        let mut out = vec![0.0; a.len()];
        unsafe { neon_add_f32(&a, &b, &mut out) };
        assert_slices_approx(&out, &expected, "neon_add_f32");
    }

    #[test]
    fn neon_mul_matches_scalar() {
        let a = vec![1.0, -2.0, 3.0, -4.0, 5.5, 6.0, 7.0];
        let b = vec![2.0, 3.0, -1.0, 0.5, 2.0, 0.0, -1.0];
        let expected = scalar_mul(&a, &b);
        let mut out = vec![0.0; a.len()];
        unsafe { neon_mul_f32(&a, &b, &mut out) };
        assert_slices_approx(&out, &expected, "neon_mul_f32");
    }

    #[test]
    fn neon_scale_matches_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = 2.5_f32;
        let expected: Vec<f32> = a.iter().map(|x| x * s).collect();
        let mut out = vec![0.0; a.len()];
        unsafe { neon_scale_f32(&a, s, &mut out) };
        assert_slices_approx(&out, &expected, "neon_scale_f32");
    }

    #[test]
    fn neon_fma_elementwise_matches_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
        let c = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let expected = scalar_fma(&a, &b, &c);
        let mut out = vec![0.0; a.len()];
        unsafe { neon_fma_f32(&a, &b, &c, &mut out) };
        assert_slices_approx(&out, &expected, "neon_fma_f32");
    }

    // ── Reduction tests ──────────────────────────────────────────────

    #[test]
    fn neon_sum_matches_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let expected = scalar_sum(&data);
        let result = unsafe { neon_sum_f32(&data) };
        assert_approx(result, expected, "neon_sum_f32");
    }

    #[test]
    fn neon_sum_empty_returns_zero() {
        let data: Vec<f32> = vec![];
        let result = unsafe { neon_sum_f32(&data) };
        assert_approx(result, 0.0, "neon_sum_f32 empty");
    }

    #[test]
    fn neon_max_finds_correct_value() {
        let data = vec![1.0, 5.0, 3.0, -1.0, 7.0, 2.0, 6.0, 4.0, 0.5];
        let expected = 7.0_f32;
        let result = unsafe { neon_max_f32(&data) };
        assert_approx(result, expected, "neon_max_f32");
    }

    #[test]
    fn neon_argmax_finds_correct_index() {
        let data = vec![1.0, 5.0, 3.0, -1.0, 7.0, 2.0, 6.0, 4.0, 0.5];
        let result = unsafe { neon_argmax_f32(&data) };
        assert_eq!(result, 4, "neon_argmax_f32 should return index 4");
    }

    #[test]
    fn neon_dot_product_matches_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        let expected = scalar_dot(&a, &b);
        let result = unsafe { neon_dot_f32(&a, &b) };
        assert_approx(result, expected, "neon_dot_f32");
    }

    #[test]
    fn neon_l2_norm_matches_scalar() {
        let data = vec![3.0, 4.0]; // classic 3-4-5 triangle
        let result = unsafe { neon_l2_norm_f32(&data) };
        assert_approx(result, 5.0, "neon_l2_norm_f32 (3,4)→5");
    }

    // ── FMA pipeline tests ───────────────────────────────────────────

    #[test]
    fn fma_matches_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let c = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let expected = scalar_fma(&a, &b, &c);
        let mut out = vec![0.0; a.len()];
        unsafe { fma_f32(&a, &b, &c, &mut out) };
        assert_slices_approx(&out, &expected, "fma_f32");
    }

    #[test]
    fn dot_product_fma_matches_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let expected = scalar_dot(&a, &b);
        let result = unsafe { dot_product_fma_f32(&a, &b) };
        assert_approx(result, expected, "dot_product_fma_f32");
    }

    #[test]
    fn matvec_fma_2x4_matches_scalar() {
        // 2×4 matrix, 4-element vector, 2-element bias
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.5, 1.0];
        // row0 dot x + bias0 = (1+2+3+4) + 0.5 = 10.5
        // row1 dot x + bias1 = (5+6+7+8) + 1.0 = 27.0
        let expected = vec![10.5, 27.0];
        let mut out = vec![0.0; 2];
        unsafe { matvec_fma_f32(&matrix, &x, &bias, &mut out, 2, 4) };
        assert_slices_approx(&out, &expected, "matvec_fma_f32 2×4");
    }

    #[test]
    fn scale_bias_scalar_matches_reference() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scale = 2.0_f32;
        let bias = 0.5_f32;
        let expected: Vec<f32> = x.iter().map(|v| v * scale + bias).collect();
        let mut out = vec![0.0; x.len()];
        unsafe { scale_bias_scalar_f32(&x, scale, bias, &mut out) };
        assert_slices_approx(&out, &expected, "scale_bias_scalar_f32");
    }

    // ── Softmax tests ────────────────────────────────────────────────

    #[test]
    fn softmax_neon_matches_scalar_reference() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut neon_out = vec![0.0; input.len()];
        let mut scalar_out = vec![0.0; input.len()];
        unsafe { softmax_neon(&input, &mut neon_out) };
        softmax_scalar(&input, &mut scalar_out);
        // Softmax uses fast exp approximation; allow slightly larger tolerance.
        for (i, (n, s)) in neon_out.iter().zip(&scalar_out).enumerate() {
            assert!(
                (n - s).abs() < 1e-3,
                "softmax[{i}]: neon={n} vs scalar={s} (diff={})",
                (n - s).abs()
            );
        }
    }

    #[test]
    fn softmax_neon_sums_to_one() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut out = vec![0.0; input.len()];
        unsafe { softmax_neon(&input, &mut out) };
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "softmax output should sum to ~1.0, got {sum}");
    }

    #[test]
    fn softmax_inplace_matches_out_of_place() {
        let input = vec![2.0, -1.0, 0.5, 3.0, 1.0];
        let mut oop_out = vec![0.0; input.len()];
        unsafe { softmax_neon(&input, &mut oop_out) };
        let mut inplace = input.clone();
        unsafe { softmax_neon_inplace(&mut inplace) };
        assert_slices_approx(&inplace, &oop_out, "softmax inplace vs out-of-place");
    }
}
