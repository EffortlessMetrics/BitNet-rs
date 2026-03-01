//! CPU-reference tests for the tiled matmul and quantized GEMV OpenCL kernels.
//!
//! All tests exercise CPU reference implementations that mirror the OpenCL
//! kernel logic.  No OpenCL runtime or GPU hardware is required.

const TILE_SIZE: usize = 16;
const TILE_K: usize = 16;

// ---------------------------------------------------------------------------
// CPU reference: tiled GEMM  (C = alpha * A * B + beta * C)
// ---------------------------------------------------------------------------

/// Naive reference matmul for validation.
fn reference_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// CPU emulation of the tiled GEMM kernel with alpha/beta scaling.
fn tiled_matmul_f32_cpu(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    let num_groups_m = (m + TILE_SIZE - 1) / TILE_SIZE;
    let num_groups_n = (n + TILE_SIZE - 1) / TILE_SIZE;
    let num_tiles_k = (k + TILE_K - 1) / TILE_K;

    for gm in 0..num_groups_m {
        for gn in 0..num_groups_n {
            let mut acc = [[0.0f32; TILE_SIZE]; TILE_SIZE];

            for t in 0..num_tiles_k {
                let mut tile_a = [[0.0f32; TILE_K]; TILE_SIZE];
                let mut tile_b = [[0.0f32; TILE_SIZE]; TILE_K];

                for row in 0..TILE_SIZE {
                    for col in 0..TILE_K {
                        let global_row = gm * TILE_SIZE + row;
                        let a_col = t * TILE_K + col;
                        if global_row < m && a_col < k {
                            tile_a[row][col] = a[global_row * k + a_col];
                        }
                    }
                }

                for row in 0..TILE_K {
                    for col in 0..TILE_SIZE {
                        let b_row = t * TILE_K + row;
                        let global_col = gn * TILE_SIZE + col;
                        if b_row < k && global_col < n {
                            tile_b[row][col] = b[b_row * n + global_col];
                        }
                    }
                }

                for row in 0..TILE_SIZE {
                    for col in 0..TILE_SIZE {
                        for kk in 0..TILE_K {
                            acc[row][col] += tile_a[row][kk] * tile_b[kk][col];
                        }
                    }
                }
            }

            for row in 0..TILE_SIZE {
                for col in 0..TILE_SIZE {
                    let global_row = gm * TILE_SIZE + row;
                    let global_col = gn * TILE_SIZE + col;
                    if global_row < m && global_col < n {
                        let idx = global_row * n + global_col;
                        c[idx] = alpha * acc[row][col] + beta * c[idx];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference: quantized GEMV  (y = diag(scales) * W_unpacked * x)
// ---------------------------------------------------------------------------

/// Pack ternary weights (-1, 0, +1) into 2-bit encoding.
/// Encoding: value + 1  ->  -1=0b00, 0=0b01, +1=0b10.
fn pack_i2s(weights: &[i8]) -> Vec<u8> {
    assert!(weights.len() % 4 == 0);
    weights
        .chunks_exact(4)
        .map(|quad| {
            let mut byte = 0u8;
            for (i, &w) in quad.iter().enumerate() {
                let bits = (w + 1) as u8; // -1->0, 0->1, +1->2
                byte |= (bits & 0x3) << (i * 2);
            }
            byte
        })
        .collect()
}

/// CPU emulation of the quantized GEMV kernel.
fn quantized_gemv_i2s_cpu(
    w_packed: &[u8],
    x: &[f32],
    scales: &[f32],
    m: usize,
    k: usize,
) -> Vec<f32> {
    let k_packed = k / 4;
    let mut y = vec![0.0f32; m];

    for row in 0..m {
        let mut acc = 0.0f32;
        for j in 0..k_packed {
            let packed = w_packed[row * k_packed + j];
            let base_k = j * 4;

            let w0 = ((packed >> 0) & 0x3) as i32 - 1;
            let w1 = ((packed >> 2) & 0x3) as i32 - 1;
            let w2 = ((packed >> 4) & 0x3) as i32 - 1;
            let w3 = ((packed >> 6) & 0x3) as i32 - 1;

            acc += w0 as f32 * x[base_k];
            acc += w1 as f32 * x[base_k + 1];
            acc += w2 as f32 * x[base_k + 2];
            acc += w3 as f32 * x[base_k + 3];
        }
        y[row] = acc * scales[row];
    }
    y
}

// ===== Helper =====

fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        assert!((va - vb).abs() <= tol, "mismatch at index {i}: {va} vs {vb} (tol={tol})");
    }
}

// ===== Kernel source validation =====

#[test]
fn tiled_matmul_cl_contains_expected_kernels() {
    let src = bitnet_kernels::kernels::TILED_MATMUL_SRC;
    assert!(src.contains("__kernel void tiled_matmul_f32"));
    assert!(src.contains("__kernel void quantized_gemv_i2s"));
}

#[test]
fn tiled_matmul_cl_uses_local_memory_barriers() {
    let src = bitnet_kernels::kernels::TILED_MATMUL_SRC;
    assert!(src.contains("barrier(CLK_LOCAL_MEM_FENCE)"));
    assert!(src.contains("__local float*"));
}

#[test]
fn tiled_matmul_cl_defines_tile_size() {
    let src = bitnet_kernels::kernels::TILED_MATMUL_SRC;
    assert!(src.contains("#define TILE_SIZE 16"));
    assert!(src.contains("#define TILE_K 16"));
}

// ===== Tiled GEMM correctness =====

#[test]
fn tiled_matmul_basic_square() {
    let m = 4;
    let n = 4;
    let k = 4;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();

    let expected = reference_matmul(&a, &b, m, n, k);
    let mut c = vec![0.0f32; m * n];
    tiled_matmul_f32_cpu(&a, &b, &mut c, m, n, k, 1.0, 0.0);

    assert_approx_eq(&c, &expected, 1e-5);
}

#[test]
fn tiled_matmul_identity() {
    let m = 8;
    let n = 8;
    let k = 8;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5).collect();
    let mut identity = vec![0.0f32; k * n];
    for i in 0..k.min(n) {
        identity[i * n + i] = 1.0;
    }

    let mut c = vec![0.0f32; m * n];
    tiled_matmul_f32_cpu(&a, &identity, &mut c, m, n, k, 1.0, 0.0);

    assert_approx_eq(&c, &a, 1e-5);
}

#[test]
fn tiled_matmul_alpha_beta() {
    let m = 4;
    let n = 4;
    let k = 4;
    let a: Vec<f32> = vec![1.0; m * k];
    let b: Vec<f32> = vec![1.0; k * n];
    let mut c: Vec<f32> = vec![10.0; m * n];

    // C = 2.0 * (A * B) + 0.5 * C_old
    // A*B = k (each element), so C = 2*4 + 0.5*10 = 13.0
    tiled_matmul_f32_cpu(&a, &b, &mut c, m, n, k, 2.0, 0.5);

    let expected = vec![13.0f32; m * n];
    assert_approx_eq(&c, &expected, 1e-5);
}

#[test]
fn tiled_matmul_non_square() {
    let m = 3;
    let n = 7;
    let k = 5;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.2).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.3).collect();

    let expected = reference_matmul(&a, &b, m, n, k);
    let mut c = vec![0.0f32; m * n];
    tiled_matmul_f32_cpu(&a, &b, &mut c, m, n, k, 1.0, 0.0);

    assert_approx_eq(&c, &expected, 1e-4);
}

#[test]
fn tiled_matmul_k_not_divisible_by_tile() {
    let m = 16;
    let n = 16;
    let k = 17;
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) * 0.2).collect();

    let expected = reference_matmul(&a, &b, m, n, k);
    let mut c = vec![0.0f32; m * n];
    tiled_matmul_f32_cpu(&a, &b, &mut c, m, n, k, 1.0, 0.0);

    assert_approx_eq(&c, &expected, 1e-3);
}

#[test]
fn tiled_matmul_m_n_not_divisible_by_tile() {
    let m = 13;
    let n = 11;
    let k = 16;
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 3) as f32) - 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 4) as f32) * 0.5).collect();

    let expected = reference_matmul(&a, &b, m, n, k);
    let mut c = vec![0.0f32; m * n];
    tiled_matmul_f32_cpu(&a, &b, &mut c, m, n, k, 1.0, 0.0);

    assert_approx_eq(&c, &expected, 1e-3);
}

#[test]
fn tiled_matmul_large_tile_aligned() {
    let m = 32;
    let n = 32;
    let k = 32;
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.002).cos()).collect();

    let expected = reference_matmul(&a, &b, m, n, k);
    let mut c = vec![0.0f32; m * n];
    tiled_matmul_f32_cpu(&a, &b, &mut c, m, n, k, 1.0, 0.0);

    assert_approx_eq(&c, &expected, 1e-2);
}

#[test]
fn tiled_matmul_transpose_property() {
    // (A * B)^T == B^T * A^T
    let m = 6;
    let n = 8;
    let k = 5;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.2).collect();

    let ab = reference_matmul(&a, &b, m, n, k);

    let mut at = vec![0.0f32; k * m];
    for i in 0..m {
        for j in 0..k {
            at[j * m + i] = a[i * k + j];
        }
    }
    let mut bt = vec![0.0f32; n * k];
    for i in 0..k {
        for j in 0..n {
            bt[j * k + i] = b[i * n + j];
        }
    }

    let btat = reference_matmul(&bt, &at, n, m, k);

    let mut ab_t = vec![0.0f32; n * m];
    for i in 0..m {
        for j in 0..n {
            ab_t[j * m + i] = ab[i * n + j];
        }
    }

    assert_approx_eq(&ab_t, &btat, 1e-3);
}

// ===== Quantized GEMV correctness =====

#[test]
fn quantized_gemv_basic() {
    let m = 2;
    let k = 8;
    let weights: Vec<i8> = vec![1; m * k];
    let packed = pack_i2s(&weights);
    let x: Vec<f32> = vec![1.0; k];
    let scales = vec![1.0f32; m];

    let y = quantized_gemv_i2s_cpu(&packed, &x, &scales, m, k);
    assert_approx_eq(&y, &[8.0, 8.0], 1e-6);
}

#[test]
fn quantized_gemv_known_weights() {
    let m = 1;
    let k = 4;
    let weights: Vec<i8> = vec![-1, 0, 1, 1];
    let packed = pack_i2s(&weights);
    let x = vec![2.0f32, 3.0, 4.0, 5.0];
    let scales = vec![0.5f32];

    let y = quantized_gemv_i2s_cpu(&packed, &x, &scales, m, k);
    // dot = -1*2 + 0*3 + 1*4 + 1*5 = 7.0; y = 7 * 0.5 = 3.5
    assert_approx_eq(&y, &[3.5], 1e-6);
}

#[test]
fn quantized_gemv_all_zeros() {
    let m = 4;
    let k = 8;
    let weights: Vec<i8> = vec![0; m * k];
    let packed = pack_i2s(&weights);
    let x: Vec<f32> = vec![42.0; k];
    let scales = vec![1.0f32; m];

    let y = quantized_gemv_i2s_cpu(&packed, &x, &scales, m, k);
    assert_approx_eq(&y, &vec![0.0; m], 1e-6);
}

#[test]
fn quantized_gemv_alternating_weights() {
    let m = 1;
    let k = 8;
    let weights: Vec<i8> = (0..k).map(|i| if i % 2 == 0 { -1 } else { 1 }).collect();
    let packed = pack_i2s(&weights);
    let x = vec![1.0f32; k];
    let scales = vec![2.0f32];

    let y = quantized_gemv_i2s_cpu(&packed, &x, &scales, m, k);
    // dot = (-1+1)*4 = 0; y = 0 * 2 = 0
    assert_approx_eq(&y, &[0.0], 1e-6);
}

#[test]
fn quantized_gemv_scale_applied() {
    let m = 2;
    let k = 4;
    let weights: Vec<i8> = vec![1, 1, 1, 1, -1, -1, -1, -1];
    let packed = pack_i2s(&weights);
    let x = vec![1.0f32; k];
    let scales = vec![3.0, 5.0];

    let y = quantized_gemv_i2s_cpu(&packed, &x, &scales, m, k);
    assert_approx_eq(&y, &[12.0, -20.0], 1e-6);
}

#[test]
fn quantized_gemv_output_bounded_by_scale_times_k() {
    let m = 8;
    let k = 64;
    let weights: Vec<i8> = (0..m * k)
        .map(|i| match i % 3 {
            0 => -1i8,
            1 => 0,
            _ => 1,
        })
        .collect();
    let packed = pack_i2s(&weights);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.1).sin()).collect();
    let scales: Vec<f32> = (0..m).map(|i| (i as f32 + 1.0) * 0.5).collect();

    let y = quantized_gemv_i2s_cpu(&packed, &x, &scales, m, k);

    for row in 0..m {
        let bound = scales[row].abs() * k as f32;
        assert!(y[row].abs() <= bound + 1e-6, "row {row}: |y|={} > bound={bound}", y[row].abs());
    }
}

// ===== Pack/unpack round-trip =====

#[test]
fn pack_i2s_round_trip() {
    let original: Vec<i8> = vec![-1, 0, 1, 1, 0, -1, 1, 0];
    let packed = pack_i2s(&original);

    let mut unpacked = Vec::new();
    for &byte in &packed {
        for i in 0..4 {
            let bits = ((byte >> (i * 2)) & 0x3) as i32 - 1;
            unpacked.push(bits as i8);
        }
    }
    assert_eq!(original, unpacked);
}

// ===== Benchmark-like test (CPU reference timing) =====

#[test]
fn tiled_matmul_2048x2048_cpu_reference() {
    let m = 2048;
    let n = 2048;
    let k = 2048;
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.0001).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.0002).cos()).collect();

    let start = std::time::Instant::now();
    let mut c = vec![0.0f32; m * n];
    tiled_matmul_f32_cpu(&a, &b, &mut c, m, n, k, 1.0, 0.0);
    let elapsed = start.elapsed();

    assert!(c.iter().any(|&v| v.abs() > 1e-10), "result should be non-zero");

    let gflops = 2.0 * m as f64 * n as f64 * k as f64 / elapsed.as_secs_f64() / 1e9;
    eprintln!("CPU tiled matmul {m}x{n}x{k}: {:.2?} ({gflops:.2} GFLOP/s)", elapsed);
}

// ===== Hardware tests (require OpenCL device) =====

#[test]
#[ignore = "requires OpenCL device - run with --ignored on Intel Arc A770"]
fn opencl_tiled_matmul_f32_device() {
    let _src = bitnet_kernels::kernels::TILED_MATMUL_SRC;
    todo!("implement OpenCL device test when runtime is available");
}

#[test]
#[ignore = "requires OpenCL device - run with --ignored on Intel Arc A770"]
fn opencl_quantized_gemv_i2s_device() {
    let _src = bitnet_kernels::kernels::TILED_MATMUL_SRC;
    todo!("implement OpenCL device test when runtime is available");
}
