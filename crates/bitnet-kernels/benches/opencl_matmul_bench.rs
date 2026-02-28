//! Criterion benchmarks comparing naive vs tiled OpenCL matmul kernels.
//!
//! Uses CPU reference implementations mirroring the OpenCL kernel logic.
//! When run with `--features oneapi` on Intel Arc hardware, the real
//! OpenCL paths are exercised.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

/// Branchless ternary decode: 0b00→0, 0b01→+1, 0b11→−1.
#[inline]
fn decode_ternary(bits: u8) -> i32 {
    let mag = (bits & 1) as i32;
    let sign = ((bits >> 1) & 1) as i32;
    mag - 2 * mag * sign
}

/// Naive matmul (mirrors `matmul_i2s` kernel).
fn matmul_naive(a: &[i8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    let k_packed = k / 4;
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for kp in 0..k_packed {
                let packed = b[kp * n + col];
                for sub in 0..4u32 {
                    let k_idx = kp * 4 + sub as usize;
                    if k_idx >= k {
                        break;
                    }
                    let bits = (packed >> (sub * 2)) & 0x03;
                    let w = if bits == 0x01 {
                        1
                    } else if bits == 0x03 {
                        -1
                    } else {
                        0
                    };
                    sum += a[row * k + k_idx] as f32 * w as f32;
                }
            }
            c[row * n + col] = sum;
        }
    }
}

/// Tiled matmul (mirrors `matmul_i2s_tiled` kernel logic on CPU).
fn matmul_tiled(a: &[i8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    const TILE: usize = 16;

    for v in c.iter_mut() {
        *v = 0.0;
    }

    let num_tiles = (k + TILE - 1) / TILE;

    for tile_row in (0..m).step_by(TILE) {
        for tile_col in (0..n).step_by(TILE) {
            for t in 0..num_tiles {
                let mut tile_a = [[0.0f32; TILE]; TILE];
                let mut tile_b = [[0.0f32; TILE]; TILE];

                for lr in 0..TILE {
                    for lc in 0..TILE {
                        let gr = tile_row + lr;
                        let ac = t * TILE + lc;
                        tile_a[lr][lc] = if gr < m && ac < k {
                            a[gr * k + ac] as f32
                        } else {
                            0.0
                        };

                        let br = t * TILE + lr;
                        let gc = tile_col + lc;
                        tile_b[lr][lc] = if br < k && gc < n {
                            let byte_idx = (br / 4) * n + gc;
                            let sub = br & 3;
                            let packed = b[byte_idx];
                            let bits = (packed >> (sub as u32 * 2)) & 0x03;
                            decode_ternary(bits) as f32
                        } else {
                            0.0
                        };
                    }
                }

                for lr in 0..TILE {
                    let gr = tile_row + lr;
                    if gr >= m {
                        break;
                    }
                    for lc in 0..TILE {
                        let gc = tile_col + lc;
                        if gc >= n {
                            break;
                        }
                        let mut acc = 0.0f32;
                        for kk in 0..TILE {
                            acc += tile_a[lr][kk] * tile_b[kk][lc];
                        }
                        c[gr * n + gc] += acc;
                    }
                }
            }
        }
    }
}

fn make_activations(m: usize, k: usize) -> Vec<i8> {
    (0..m * k).map(|i| ((i % 256) as i16 - 128) as i8).collect()
}

fn make_weights(k: usize, n: usize) -> Vec<u8> {
    (0..(k / 4) * n).map(|i| ((i * 0x37 + 0x1B) & 0xFF) as u8).collect()
}

fn bench_matmul_naive_vs_tiled(c: &mut Criterion) {
    let mut group = c.benchmark_group("opencl_matmul_comparison");

    let sizes: Vec<(usize, usize, usize)> =
        vec![(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)];

    for &(m, n, k) in &sizes {
        let a = make_activations(m, k);
        let b = make_weights(k, n);
        let ops = (m * n * k) as u64;
        group.throughput(Throughput::Elements(ops));

        group.bench_with_input(
            BenchmarkId::new("naive", format!("{m}x{n}x{k}")),
            &(m, n, k),
            |bench, &(m, n, k)| {
                let mut c_out = vec![0.0f32; m * n];
                bench.iter(|| {
                    matmul_naive(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut c_out),
                        black_box(m),
                        black_box(n),
                        black_box(k),
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tiled", format!("{m}x{n}x{k}")),
            &(m, n, k),
            |bench, &(m, n, k)| {
                let mut c_out = vec![0.0f32; m * n];
                bench.iter(|| {
                    matmul_tiled(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut c_out),
                        black_box(m),
                        black_box(n),
                        black_box(k),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_correctness_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("opencl_matmul_correctness");
    group.sample_size(10);

    let (m, n, k) = (64, 64, 64);
    let a = make_activations(m, k);
    let b = make_weights(k, n);

    group.bench_function("verify_naive_eq_tiled", |bench| {
        bench.iter(|| {
            let mut c_naive = vec![0.0f32; m * n];
            let mut c_tiled = vec![0.0f32; m * n];
            matmul_naive(&a, &b, &mut c_naive, m, n, k);
            matmul_tiled(&a, &b, &mut c_tiled, m, n, k);
            for i in 0..m * n {
                assert!(
                    (c_naive[i] - c_tiled[i]).abs() < 1e-4,
                    "mismatch at {i}: naive={} tiled={}",
                    c_naive[i],
                    c_tiled[i]
                );
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_matmul_naive_vs_tiled, bench_correctness_check);
criterion_main!(benches);
