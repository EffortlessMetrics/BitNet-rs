//! Cross-validation: mock GPU execution vs CPU reference implementations.

use bitnet_opencl::reference_kernels;
use bitnet_opencl::test_fixtures;
use bitnet_opencl::testing::{MockGpuContext, NumericalValidator};

/// Helper: write f32 slice to mock GPU buffer and return bytes.
fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Helper: read f32 slice from byte buffer.
fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

// ── Cross-validation: matmul ────────────────────────────────────────

#[test]
fn test_cross_validate_matmul() {
    let n = 4;
    let (a, b, _) = test_fixtures::golden_matmul_4x4();

    // CPU reference
    let mut cpu_result = vec![0.0f32; n * n];
    reference_kernels::ref_matmul(&a, &b, &mut cpu_result, n, n, n);

    // Mock GPU
    let mut ctx = MockGpuContext::new("mock", 65536);
    ctx.register_kernel("matmul_4x4", |inputs, output| {
        let a_f32 = bytes_to_f32_inner(inputs[0]);
        let b_f32 = bytes_to_f32_inner(inputs[1]);
        let mut c = vec![0.0f32; 16];
        reference_kernels::ref_matmul(&a_f32, &b_f32, &mut c, 4, 4, 4);
        let bytes: Vec<u8> = c.iter().flat_map(|f| f.to_le_bytes()).collect();
        output[..bytes.len()].copy_from_slice(&bytes);
        Ok(())
    });

    let a_h = ctx.alloc_buffer(a.len() * 4);
    let b_h = ctx.alloc_buffer(b.len() * 4);
    let c_h = ctx.alloc_buffer(n * n * 4);

    ctx.write_buffer(a_h, &f32_to_bytes(&a)).unwrap();
    ctx.write_buffer(b_h, &f32_to_bytes(&b)).unwrap();
    ctx.execute_kernel("matmul_4x4", &[a_h, b_h], c_h).unwrap();

    let gpu_result = bytes_to_f32(ctx.read_buffer(c_h).unwrap());

    let v = NumericalValidator::strict();
    let result = v.compare_f32(&cpu_result, &gpu_result);
    assert!(result.passed, "matmul cross-validation failed");
}

// ── Cross-validation: softmax ───────────────────────────────────────

#[test]
fn test_cross_validate_softmax() {
    let (input, _) = test_fixtures::golden_softmax_8();
    let n = input.len();

    // CPU reference
    let mut cpu_result = vec![0.0f32; n];
    reference_kernels::ref_softmax(&input, &mut cpu_result, n);

    // Mock GPU
    let mut ctx = MockGpuContext::new("mock", 65536);
    ctx.register_kernel("softmax", |inputs, output| {
        let inp = bytes_to_f32_inner(inputs[0]);
        let n = inp.len();
        let mut out = vec![0.0f32; n];
        reference_kernels::ref_softmax(&inp, &mut out, n);
        let bytes: Vec<u8> = out.iter().flat_map(|f| f.to_le_bytes()).collect();
        output[..bytes.len()].copy_from_slice(&bytes);
        Ok(())
    });

    let in_h = ctx.alloc_buffer(n * 4);
    let out_h = ctx.alloc_buffer(n * 4);
    ctx.write_buffer(in_h, &f32_to_bytes(&input)).unwrap();
    ctx.execute_kernel("softmax", &[in_h], out_h).unwrap();

    let gpu_result = bytes_to_f32(ctx.read_buffer(out_h).unwrap());

    let v = NumericalValidator::strict();
    assert!(v.compare_f32(&cpu_result, &gpu_result).passed);
}

// ── Cross-validation: rmsnorm ───────────────────────────────────────

#[test]
fn test_cross_validate_rmsnorm() {
    let (input, weight, _) = test_fixtures::golden_rmsnorm_4();
    let n = input.len();

    // CPU reference
    let mut cpu_result = vec![0.0f32; n];
    reference_kernels::ref_rmsnorm(&input, &weight, &mut cpu_result, 1e-6);

    // Mock GPU
    let mut ctx = MockGpuContext::new("mock", 65536);
    ctx.register_kernel("rmsnorm", |inputs, output| {
        let inp = bytes_to_f32_inner(inputs[0]);
        let w = bytes_to_f32_inner(inputs[1]);
        let n = inp.len();
        let mut out = vec![0.0f32; n];
        reference_kernels::ref_rmsnorm(&inp, &w, &mut out, 1e-6);
        let bytes: Vec<u8> = out.iter().flat_map(|f| f.to_le_bytes()).collect();
        output[..bytes.len()].copy_from_slice(&bytes);
        Ok(())
    });

    let in_h = ctx.alloc_buffer(n * 4);
    let w_h = ctx.alloc_buffer(n * 4);
    let out_h = ctx.alloc_buffer(n * 4);
    ctx.write_buffer(in_h, &f32_to_bytes(&input)).unwrap();
    ctx.write_buffer(w_h, &f32_to_bytes(&weight)).unwrap();
    ctx.execute_kernel("rmsnorm", &[in_h, w_h], out_h).unwrap();

    let gpu_result = bytes_to_f32(ctx.read_buffer(out_h).unwrap());

    let v = NumericalValidator::strict();
    assert!(v.compare_f32(&cpu_result, &gpu_result).passed);
}

// ── Property tests: random inputs ───────────────────────────────────

#[test]
fn test_property_softmax_sums_to_one_random() {
    for seed in 1..=10 {
        let input = test_fixtures::random_f32_vec(64, seed);
        let mut output = vec![0.0f32; 64];
        reference_kernels::ref_softmax(&input, &mut output, 64);

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "seed {}: softmax sum = {}", seed, sum);
    }
}

#[test]
fn test_property_rmsnorm_output_scale_invariant() {
    for seed in 1..=5 {
        let input = test_fixtures::random_f32_vec(32, seed);
        let weight = vec![1.0f32; 32];

        let scaled: Vec<f32> = input.iter().map(|&x| x * 100.0).collect();

        let mut out1 = vec![0.0f32; 32];
        let mut out2 = vec![0.0f32; 32];
        reference_kernels::ref_rmsnorm(&input, &weight, &mut out1, 1e-6);
        reference_kernels::ref_rmsnorm(&scaled, &weight, &mut out2, 1e-6);

        let v = NumericalValidator::relaxed();
        assert!(
            v.compare_f32(&out1, &out2).passed,
            "rmsnorm should be scale-invariant (seed {})",
            seed
        );
    }
}

#[test]
fn test_property_matmul_identity_random() {
    for seed in 1..=5 {
        let n = 8;
        let a = test_fixtures::random_f32_vec(n * n, seed);
        let eye = test_fixtures::identity_matrix(n);
        let mut c = vec![0.0f32; n * n];
        reference_kernels::ref_matmul(&a, &eye, &mut c, n, n, n);

        let v = NumericalValidator::strict();
        assert!(v.compare_f32(&a, &c).passed, "A * I != A for seed {}", seed);
    }
}

// ── Snapshot tests: deterministic outputs ───────────────────────────

#[test]
fn test_snapshot_silu_deterministic() {
    let input = test_fixtures::random_f32_vec(16, 42);
    let mut out1 = vec![0.0f32; 16];
    let mut out2 = vec![0.0f32; 16];

    reference_kernels::ref_silu(&input, &mut out1);
    reference_kernels::ref_silu(&input, &mut out2);

    assert_eq!(out1, out2, "SiLU must be deterministic");
}

#[test]
fn test_snapshot_gelu_deterministic() {
    let input = test_fixtures::random_f32_vec(16, 42);
    let mut out1 = vec![0.0f32; 16];
    let mut out2 = vec![0.0f32; 16];

    reference_kernels::ref_gelu(&input, &mut out1);
    reference_kernels::ref_gelu(&input, &mut out2);

    assert_eq!(out1, out2, "GELU must be deterministic");
}

#[test]
fn test_snapshot_matmul_deterministic() {
    let n = 8;
    let a = test_fixtures::random_f32_vec(n * n, 123);
    let b = test_fixtures::random_f32_vec(n * n, 456);
    let mut c1 = vec![0.0f32; n * n];
    let mut c2 = vec![0.0f32; n * n];

    reference_kernels::ref_matmul(&a, &b, &mut c1, n, n, n);
    reference_kernels::ref_matmul(&a, &b, &mut c2, n, n, n);

    assert_eq!(c1, c2, "matmul must be deterministic");
}

/// Helper used inside kernel closures (can't capture outer fn).
fn bytes_to_f32_inner(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
