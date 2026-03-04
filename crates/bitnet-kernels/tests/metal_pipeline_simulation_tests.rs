#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
#![cfg(target_os = "macos")]
#![allow(
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    unused_imports,
    dead_code
)]

// ============================================================================
// Helper Structs and Functions: Metal Pipeline Simulation
// ============================================================================

/// Simulates MTLBuffer behavior: GPU-accessible memory with alignment requirements
#[derive(Clone, Debug)]
struct MetalBufferSim {
    data: Vec<f32>,
}

impl MetalBufferSim {
    fn new(size: usize) -> Self {
        MetalBufferSim { data: vec![0.0; size] }
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Option<f32> {
        self.data.get(index).copied()
    }

    fn set(&mut self, index: usize, value: f32) {
        if index < self.data.len() {
            self.data[index] = value;
        }
    }
}

/// Calculates required buffer size with 4096-byte alignment (Metal requirement on Apple Silicon)
fn align_buffer_size(size: usize) -> usize {
    const ALIGNMENT: usize = 4096;
    size.div_ceil(ALIGNMENT) * ALIGNMENT
}

/// Computes dispatch grid dimensions for threadgroups
/// Returns (num_threadgroups, threads_per_group)
fn dispatch_threadgroups(total_threads: usize, threads_per_group: usize) -> (usize, usize) {
    let num_threadgroups = total_threads.div_ceil(threads_per_group);
    (num_threadgroups, threads_per_group)
}

/// Simulates Metal SIMD group reduction (sum operation)
/// On Apple GPU, SIMD width is 32 (not 64 like NVIDIA)
fn simdgroup_reduce_sum(values: &[f32], simd_width: usize) -> f32 {
    debug_assert!(
        simd_width == 32 || simd_width == 16 || simd_width == 8,
        "Apple GPU SIMD width is typically 32"
    );

    let mut sum = 0.0;
    for i in 0..values.len() {
        sum += values[i];
    }
    sum
}

/// Simulates SIMD group maximum operation
fn simdgroup_reduce_max(values: &[f32], simd_width: usize) -> f32 {
    debug_assert!(simd_width == 32, "Apple GPU SIMD width is 32");

    let mut max_val = values[0];
    for i in 1..values.len() {
        if values[i] > max_val {
            max_val = values[i];
        }
    }
    max_val
}

/// Simulates broadcast: first lane value to all lanes
fn simdgroup_broadcast(value: f32, _simd_width: usize) -> f32 {
    value
}

/// Placeholder for threadgroup barrier synchronization
/// In actual Metal, this would be `threadgroup_barrier(mem_flags::mem_threadgroup)`
fn threadgroup_barrier_sim() {
    // No-op in simulation; in real kernel this synchronizes all threads in threadgroup
}

// ============================================================================
// Test Module: buffer_management
// ============================================================================

#[cfg(test)]
mod buffer_management {
    use super::*;

    #[test]
    fn test_buffer_alignment_4k() {
        // Metal requires 4096-byte alignment on Apple Silicon
        let sizes = [1, 100, 4095, 4096, 4097, 8192, 8193];
        let expected = [4096, 4096, 4096, 4096, 8192, 8192, 12288];

        for (size, exp) in sizes.iter().zip(expected.iter()) {
            let aligned = align_buffer_size(*size);
            assert_eq!(aligned, *exp, "Size {} should align to {}", size, exp);
            assert_eq!(aligned % 4096, 0, "Aligned size {} must be multiple of 4096", aligned);
        }
    }

    #[test]
    fn test_buffer_size_for_matrix() {
        // For M×N matrix of f32, buffer size = M * N * 4 bytes
        // Must then align to 4096 bytes

        let test_cases = vec![
            (32, 32, 4096),        // 32×32 × 4 bytes = 4096 → 4096
            (64, 64, 16384),       // 64×64 × 4 bytes = 16384 → 16384
            (128, 128, 65536),     // 128×128 × 4 bytes = 65536 → 65536
            (1024, 1024, 4194304), // 1024×1024 × 4 bytes = 4194304 → 4194304
        ];

        for (m, n, expected_size) in test_cases {
            let buffer_size = align_buffer_size(m * n * 4) / 4; // Convert back to element count
            assert_eq!(
                buffer_size * 4,
                expected_size,
                "Matrix {}×{} should need {} bytes",
                m,
                n,
                expected_size
            );
        }
    }

    #[test]
    fn test_triple_buffer_rotation() {
        // Simulate triple-buffering: 3 buffers rotating for async compute
        const BUFFER_SIZE: usize = 1024;
        const NUM_BUFFERS: usize = 3;

        let mut buffers: Vec<MetalBufferSim> =
            (0..NUM_BUFFERS).map(|_| MetalBufferSim::new(BUFFER_SIZE)).collect();

        let mut write_idx = 0;
        let mut read_idx = (NUM_BUFFERS - 1) % NUM_BUFFERS;

        // Simulate 6 iterations of triple-buffering
        for iteration in 0..6 {
            // Write to write buffer
            for i in 0..BUFFER_SIZE {
                buffers[write_idx].set(i, iteration as f32 + i as f32);
            }

            // Verify read buffer from previous iteration
            if iteration > 0 {
                for i in 0..BUFFER_SIZE {
                    let expected = (iteration - 1) as f32 + i as f32;
                    let actual = buffers[read_idx].get(i).unwrap();
                    assert_eq!(actual, expected);
                }
            }

            // Rotate indices
            write_idx = (write_idx + 1) % NUM_BUFFERS;
            read_idx = (read_idx + 1) % NUM_BUFFERS;
        }
    }

    #[test]
    fn test_buffer_offset_indexing() {
        // Verify correct index math for sub-buffer access
        let buffer = MetalBufferSim::new(4096);

        let base_offset = 256;
        let _element_size = 4; // f32 in bytes, though we're working in f32 units

        // Access pattern: linear_index = base_offset + local_index
        for local_idx in 0..100 {
            let global_idx = base_offset + local_idx;
            assert!(
                global_idx < buffer.len(),
                "Index {} should be within buffer bounds",
                global_idx
            );
        }

        // Verify 2D indexing: for tiles in a matrix
        let tile_size = 16;
        let matrix_width = 64;
        let tile_row = 2;
        let tile_col = 3;

        let tile_start_row = tile_row * tile_size;
        let tile_start_col = tile_col * tile_size;

        for i in 0..tile_size {
            for j in 0..tile_size {
                let row = tile_start_row + i;
                let col = tile_start_col + j;
                let linear_idx = row * matrix_width + col;
                assert!(
                    linear_idx < buffer.len(),
                    "Tile access at ({}, {}) → linear_idx {} in bounds",
                    row,
                    col,
                    linear_idx
                );
            }
        }
    }
}

// ============================================================================
// Test Module: dispatch_sizing
// ============================================================================

#[cfg(test)]
mod dispatch_sizing {
    use super::*;

    #[test]
    fn test_threadgroup_1d_dispatch() {
        // 1D dispatch: process N elements with threadgroup_size threads per group
        let total_elements = 1024;
        let threadgroup_size = 32;

        let (num_groups, tg_size) = dispatch_threadgroups(total_elements, threadgroup_size);

        assert_eq!(tg_size, 32);
        assert_eq!(num_groups, 32); // 1024 / 32 = 32 groups
        assert!(num_groups * tg_size >= total_elements);
    }

    #[test]
    fn test_threadgroup_2d_dispatch() {
        // 2D dispatch for matmul tiles: M×N matrix, 8×8 tiles
        let matrix_m: usize = 256;
        let matrix_n: usize = 256;
        let tile_size: usize = 8;

        let tiles_m = matrix_m.div_ceil(tile_size);
        let tiles_n = matrix_n.div_ceil(tile_size);

        assert_eq!(tiles_m, 32);
        assert_eq!(tiles_n, 32);

        let (num_groups_m, _) = dispatch_threadgroups(tiles_m, 1);
        let (num_groups_n, _) = dispatch_threadgroups(tiles_n, 1);

        assert_eq!(num_groups_m, 32);
        assert_eq!(num_groups_n, 32);
    }

    #[test]
    fn test_threadgroup_size_limits() {
        // Apple Silicon max threadgroup size is 1024 threads
        const APPLE_SILICON_MAX_THREADGROUP: usize = 1024;

        // Typical 1D threadgroup
        let (_, tg_size_1d) = dispatch_threadgroups(4096, 256);
        assert!(tg_size_1d <= APPLE_SILICON_MAX_THREADGROUP);

        // Typical 2D threadgroup (32x32 = 1024)
        let tg_size_2d = 32 * 32;
        assert_eq!(tg_size_2d, APPLE_SILICON_MAX_THREADGROUP);

        // Cannot exceed max
        assert!(tg_size_1d <= APPLE_SILICON_MAX_THREADGROUP);
        assert!(tg_size_2d <= APPLE_SILICON_MAX_THREADGROUP);
    }

    #[test]
    fn test_dispatch_covers_all_elements() {
        // Verify no element left unprocessed with non-power-of-2 sizes
        let test_cases = vec![(1000, 32), (1023, 32), (1024, 32), (1025, 32), (777, 17), (99, 8)];

        for (total_elements, threadgroup_size) in test_cases {
            let (num_groups, tg_size) = dispatch_threadgroups(total_elements, threadgroup_size);
            let total_threads = num_groups * tg_size;

            assert!(
                total_threads >= total_elements,
                "Dispatch must cover all {} elements; got {} threads",
                total_elements,
                total_threads
            );

            // Each group except possibly the last should be fully used
            let threads_needed = num_groups * tg_size;
            assert!(threads_needed - total_elements < tg_size);
        }
    }

    #[test]
    fn test_warp_size_32() {
        // Apple GPU SIMD width is 32 (not 64 like NVIDIA)
        const APPLE_SIMD_WIDTH: usize = 32;

        // Threadgroup should be multiple of SIMD width
        let threadgroup_sizes = vec![32, 64, 128, 256, 512, 1024];

        for size in threadgroup_sizes {
            assert_eq!(
                size % APPLE_SIMD_WIDTH,
                0,
                "Threadgroup size {} should be multiple of SIMD width 32",
                size
            );
        }
    }
}

// ============================================================================
// Test Module: simd_group_ops
// ============================================================================

#[cfg(test)]
mod simd_group_ops {
    use super::*;

    #[test]
    fn test_simdgroup_sum_power_of_2() {
        // Reduction on 32 elements (one SIMD group on Apple GPU)
        let values: Vec<f32> = (0..32).map(|i| i as f32).collect();

        let sum = simdgroup_reduce_sum(&values, 32);
        let expected: f32 = (0..32).map(|i| i as f32).sum();

        assert!((sum - expected).abs() < 1e-5);
    }

    #[test]
    fn test_simdgroup_sum_single_value() {
        // Single element reduction
        let values = vec![42.5];

        let sum = simdgroup_reduce_sum(&values, 32);
        assert!((sum - 42.5).abs() < 1e-5);
    }

    #[test]
    fn test_simdgroup_max() {
        // SIMD group maximum operation
        let values: Vec<f32> = vec![1.0, 5.0, 3.0, 9.0, 2.0, 8.0];

        let max_val = simdgroup_reduce_max(&values, 32);
        assert!((max_val - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_simdgroup_broadcast() {
        // Broadcast first lane to all
        let broadcast_value = std::f32::consts::PI;

        let result = simdgroup_broadcast(broadcast_value, 32);
        assert!((result - broadcast_value).abs() < 1e-5);
    }
}

// ============================================================================
// Test Module: kernel_patterns
// ============================================================================

#[cfg(test)]
mod kernel_patterns {
    use super::*;

    #[test]
    fn test_tiled_matmul_8x8() {
        // Verify 8×8 tile matmul pattern matching Metal tiling
        // Simulate: C[i][j] += A[i][k] * B[k][j]

        const TILE_SIZE: usize = 8;
        let mut c_tile = vec![vec![0.0; TILE_SIZE]; TILE_SIZE];

        // Initialize A and B tiles
        let a_tile: Vec<Vec<f32>> = (0..TILE_SIZE)
            .map(|i| (0..TILE_SIZE).map(|j| (i * TILE_SIZE + j) as f32).collect())
            .collect();

        let b_tile: Vec<Vec<f32>> =
            (0..TILE_SIZE).map(|i| (0..TILE_SIZE).map(|j| (i + j) as f32).collect()).collect();

        // Tiled matmul
        for i in 0..TILE_SIZE {
            for j in 0..TILE_SIZE {
                for k in 0..TILE_SIZE {
                    c_tile[i][j] += a_tile[i][k] * b_tile[k][j];
                }
            }
        }

        // Verify some results
        let c00_expected: f32 = (0..TILE_SIZE).map(|k| a_tile[0][k] * b_tile[k][0]).sum();
        assert!((c_tile[0][0] - c00_expected).abs() < 1e-5);
    }

    #[test]
    fn test_elementwise_relu() {
        // Simple elementwise kernel pattern: max(x, 0)
        let input: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let expected: Vec<f32> = [0.0, 0.0, 0.0, 1.0, 2.0].to_vec();

        let output: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();

        for (actual, exp) in output.iter().zip(expected.iter()) {
            assert!((actual - exp).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_two_pass() {
        // Metal softmax: pass1 = find max, pass2 = exp + sum + normalize
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];

        // Pass 1: find max
        let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(max_val, 5.0);

        // Pass 2: exp and sum
        let exp_vals: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();

        // Normalize
        let softmax: Vec<f32> = exp_vals.iter().map(|&x| x / sum_exp).collect();

        // Verify: softmax sums to 1
        let softmax_sum: f32 = softmax.iter().sum();
        assert!((softmax_sum - 1.0).abs() < 1e-5);

        // Verify: all values in [0, 1]
        for &val in &softmax {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_reduction_tree() {
        // Tree reduction pattern for sum across threadgroups
        // Simulates multiple passes of reduction

        let mut values: Vec<f32> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_vec();
        let original_sum: f32 = values.iter().sum();

        // Simulate tree reduction with stride doubling
        let mut stride = 1;
        while stride < values.len() {
            for i in (0..values.len()).step_by(stride * 2) {
                if i + stride < values.len() {
                    values[i] += values[i + stride];
                }
            }
            stride *= 2;
        }

        // Final reduced value
        assert!((values[0] - original_sum).abs() < 1e-5);
    }
}
