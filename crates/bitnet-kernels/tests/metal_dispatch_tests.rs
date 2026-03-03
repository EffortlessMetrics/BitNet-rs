//! Metal dispatch sizing and workgroup configuration tests.
//!
//! Validates that compute dispatch parameters are correct for Apple Silicon GPU
//! constraints. All tests are pure computation — no Metal hardware required.

// ── Apple Silicon Metal constants ───────────────────────────────────────────

/// Maximum threads per threadgroup on Apple Silicon (M1/M2/M3/M4).
const MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// SIMD width (thread execution width) on Apple Silicon GPUs.
const SIMD_WIDTH: u32 = 32;

/// Metal buffer alignment requirement (bytes).
const BUFFER_ALIGNMENT: usize = 256;

/// Maximum threadgroup memory (shared memory) per threadgroup (bytes).
const MAX_THREADGROUP_MEMORY: usize = 32 * 1024;

// ── Dispatch helper functions ───────────────────────────────────────────────

/// Ceil-division: returns the number of threadgroups needed to cover `total`
/// elements with `group_size` threads per group.
fn ceil_div(total: u32, group_size: u32) -> u32 {
    assert_ne!(group_size, 0, "group_size must be non-zero");
    total.div_ceil(group_size)
}

/// Align `size` up to the next multiple of [`BUFFER_ALIGNMENT`].
fn align_buffer(size: usize) -> usize {
    (size + BUFFER_ALIGNMENT - 1) & !(BUFFER_ALIGNMENT - 1)
}

/// Returns true if `n` is a power of two (and non-zero).
fn is_power_of_two(n: u32) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Returns true if `n` is a valid Metal workgroup dimension: either a power of
/// two or a multiple of [`SIMD_WIDTH`].
fn is_valid_workgroup_dim(n: u32) -> bool {
    n > 0 && (is_power_of_two(n) || n.is_multiple_of(SIMD_WIDTH))
}

/// Choose a 1-D threadgroup size ≤ [`MAX_THREADS_PER_THREADGROUP`] that is
/// a multiple of [`SIMD_WIDTH`] and does not exceed `total_elements`.
fn optimal_threadgroup_1d(total_elements: u32) -> u32 {
    if total_elements == 0 {
        return 0;
    }
    // Round up to the nearest SIMD_WIDTH, then clamp.
    let rounded = total_elements.div_ceil(SIMD_WIDTH) * SIMD_WIDTH;
    rounded.min(MAX_THREADS_PER_THREADGROUP)
}

/// Compute 2-D threadgroup dimensions that stay within
/// [`MAX_THREADS_PER_THREADGROUP`]. Returns `(width, height)`.
fn optimal_threadgroup_2d(cols: u32, rows: u32) -> (u32, u32) {
    if cols == 0 || rows == 0 {
        return (0, 0);
    }
    // Start with width = SIMD_WIDTH, grow height as much as possible.
    let w = SIMD_WIDTH.min(cols);
    let max_h = MAX_THREADS_PER_THREADGROUP / w;
    let h = max_h.min(rows);
    (w, h)
}

/// Compute 3-D threadgroup dimensions. Returns `(x, y, z)`.
fn optimal_threadgroup_3d(dim_x: u32, dim_y: u32, dim_z: u32) -> (u32, u32, u32) {
    if dim_x == 0 || dim_y == 0 || dim_z == 0 {
        return (0, 0, 0);
    }
    let x = SIMD_WIDTH.min(dim_x);
    let max_yz = MAX_THREADS_PER_THREADGROUP / x;
    let y = max_yz.min(dim_y);
    let z = if y > 0 { (max_yz / y).min(dim_z) } else { 0 };
    (x, y, z.max(1))
}

/// Compute the 1-D dispatch grid (number of threadgroups) needed to cover
/// `total` elements with the given `group_size`.
fn dispatch_grid_1d(total: u32, group_size: u32) -> u32 {
    if total == 0 || group_size == 0 {
        return 0;
    }
    ceil_div(total, group_size)
}

/// Compute the 2-D dispatch grid. Returns `(grid_x, grid_y)`.
fn dispatch_grid_2d(cols: u32, rows: u32, group_w: u32, group_h: u32) -> (u32, u32) {
    if group_w == 0 || group_h == 0 {
        return (0, 0);
    }
    (ceil_div(cols, group_w), ceil_div(rows, group_h))
}

/// Compute the 3-D dispatch grid. Returns `(grid_x, grid_y, grid_z)`.
fn dispatch_grid_3d(
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    gx: u32,
    gy: u32,
    gz: u32,
) -> (u32, u32, u32) {
    if gx == 0 || gy == 0 || gz == 0 {
        return (0, 0, 0);
    }
    (ceil_div(dim_x, gx), ceil_div(dim_y, gy), ceil_div(dim_z, gz))
}

/// Estimate shared memory usage for a tile of `floats` f32 values.
fn shared_memory_for_tile(floats: usize) -> usize {
    floats * std::mem::size_of::<f32>()
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── ceil_div correctness ────────────────────────────────────────────

    mod ceil_div_tests {
        use super::*;

        #[test]
        fn exact_division() {
            assert_eq!(ceil_div(1024, 32), 32);
            assert_eq!(ceil_div(256, 256), 1);
        }

        #[test]
        fn rounds_up() {
            assert_eq!(ceil_div(1025, 32), 33);
            assert_eq!(ceil_div(1, 1024), 1);
            assert_eq!(ceil_div(1023, 1024), 1);
        }

        #[test]
        fn single_element() {
            assert_eq!(ceil_div(1, 1), 1);
            assert_eq!(ceil_div(1, 32), 1);
        }

        #[test]
        #[should_panic(expected = "group_size must be non-zero")]
        fn zero_group_panics() {
            ceil_div(10, 0);
        }
    }

    // ── max threads per threadgroup ─────────────────────────────────────

    mod max_threads_limit {
        use super::*;

        #[test]
        fn threadgroup_1d_never_exceeds_limit() {
            for n in [1, 31, 32, 33, 512, 1024, 2048, 100_000] {
                let tg = optimal_threadgroup_1d(n);
                assert!(
                    tg <= MAX_THREADS_PER_THREADGROUP,
                    "1D threadgroup {tg} exceeds limit for n={n}"
                );
            }
        }

        #[test]
        fn threadgroup_2d_never_exceeds_limit() {
            for (c, r) in [(1, 1), (64, 64), (1024, 1024), (4096, 4096)] {
                let (w, h) = optimal_threadgroup_2d(c, r);
                let total = w * h;
                assert!(
                    total <= MAX_THREADS_PER_THREADGROUP,
                    "2D threadgroup {w}x{h}={total} exceeds limit for {c}x{r}"
                );
            }
        }

        #[test]
        fn threadgroup_3d_never_exceeds_limit() {
            for (x, y, z) in [(8, 8, 8), (32, 32, 32), (256, 256, 256)] {
                let (gx, gy, gz) = optimal_threadgroup_3d(x, y, z);
                let total = gx * gy * gz;
                assert!(
                    total <= MAX_THREADS_PER_THREADGROUP,
                    "3D threadgroup {gx}x{gy}x{gz}={total} exceeds limit for {x}x{y}x{z}"
                );
            }
        }
    }

    // ── workgroup size validation ───────────────────────────────────────

    mod workgroup_validation {
        use super::*;

        #[test]
        fn powers_of_two_are_valid() {
            for p in 0..11 {
                let n = 1u32 << p;
                assert!(is_valid_workgroup_dim(n), "{n} should be valid (power of 2)");
            }
        }

        #[test]
        fn simd_multiples_are_valid() {
            for m in 1..=32 {
                let n = m * SIMD_WIDTH;
                assert!(is_valid_workgroup_dim(n), "{n} should be valid (SIMD multiple)");
            }
        }

        #[test]
        fn zero_is_invalid() {
            assert!(!is_valid_workgroup_dim(0));
        }

        #[test]
        fn non_power_non_simd_invalid() {
            // 3 is neither power-of-2 nor multiple of 32
            assert!(!is_valid_workgroup_dim(3));
            assert!(!is_valid_workgroup_dim(5));
            assert!(!is_valid_workgroup_dim(48)); // 48 is not pow2, not multiple of 32
        }

        #[test]
        fn is_power_of_two_basic() {
            assert!(is_power_of_two(1));
            assert!(is_power_of_two(2));
            assert!(is_power_of_two(1024));
            assert!(!is_power_of_two(0));
            assert!(!is_power_of_two(3));
            assert!(!is_power_of_two(1000));
        }

        #[test]
        fn optimal_1d_is_simd_aligned() {
            for n in [33, 64, 100, 500, 1024, 4096] {
                let tg = optimal_threadgroup_1d(n);
                assert_eq!(tg % SIMD_WIDTH, 0, "1D threadgroup {tg} not SIMD-aligned for n={n}");
            }
        }
    }

    // ── dispatch grid calculations ──────────────────────────────────────

    mod dispatch_grid {
        use super::*;

        #[test]
        fn grid_1d_exact_fit() {
            assert_eq!(dispatch_grid_1d(1024, 256), 4);
            assert_eq!(dispatch_grid_1d(32, 32), 1);
        }

        #[test]
        fn grid_1d_requires_rounding() {
            // 1025 / 256 = 4.003... -> 5 groups
            assert_eq!(dispatch_grid_1d(1025, 256), 5);
            assert_eq!(dispatch_grid_1d(1, 32), 1);
        }

        #[test]
        fn grid_2d_basic() {
            let (gx, gy) = dispatch_grid_2d(1024, 768, 32, 32);
            assert_eq!(gx, 32); // 1024/32
            assert_eq!(gy, 24); // 768/32
        }

        #[test]
        fn grid_2d_non_exact() {
            let (gx, gy) = dispatch_grid_2d(1000, 500, 32, 32);
            assert_eq!(gx, 32); // ceil(1000/32) = 32
            assert_eq!(gy, 16); // ceil(500/32) = 16
        }

        #[test]
        fn grid_3d_basic() {
            let (gx, gy, gz) = dispatch_grid_3d(256, 256, 128, 32, 8, 4);
            assert_eq!(gx, 8);
            assert_eq!(gy, 32);
            assert_eq!(gz, 32);
        }

        #[test]
        fn grid_covers_all_elements_1d() {
            for total in [1, 31, 32, 33, 1023, 1024, 1025, 65535] {
                let group = optimal_threadgroup_1d(total);
                let grid = dispatch_grid_1d(total, group);
                let covered = grid * group;
                assert!(covered >= total, "1D dispatch covers {covered} < {total} elements");
            }
        }

        #[test]
        fn grid_covers_all_elements_2d() {
            for (cols, rows) in [(1, 1), (31, 17), (1024, 768), (4096, 2048)] {
                let (gw, gh) = optimal_threadgroup_2d(cols, rows);
                let (gx, gy) = dispatch_grid_2d(cols, rows, gw, gh);
                let covered_x = gx * gw;
                let covered_y = gy * gh;
                assert!(covered_x >= cols, "2D x: {covered_x} < {cols}");
                assert!(covered_y >= rows, "2D y: {covered_y} < {rows}");
            }
        }

        #[test]
        fn grid_covers_all_elements_3d() {
            for (dx, dy, dz) in [(8, 8, 8), (33, 33, 33), (256, 128, 64)] {
                let (gx, gy, gz) = optimal_threadgroup_3d(dx, dy, dz);
                let (nx, ny, nz) = dispatch_grid_3d(dx, dy, dz, gx, gy, gz);
                assert!(nx * gx >= dx, "3D x under-covered");
                assert!(ny * gy >= dy, "3D y under-covered");
                assert!(nz * gz >= dz, "3D z under-covered");
            }
        }
    }

    // ── buffer alignment ────────────────────────────────────────────────

    mod buffer_alignment {
        use super::*;

        #[test]
        fn already_aligned_unchanged() {
            assert_eq!(align_buffer(256), 256);
            assert_eq!(align_buffer(512), 512);
            assert_eq!(align_buffer(0), 0);
        }

        #[test]
        fn rounds_up_to_256() {
            assert_eq!(align_buffer(1), 256);
            assert_eq!(align_buffer(255), 256);
            assert_eq!(align_buffer(257), 512);
        }

        #[test]
        fn alignment_always_multiple_of_256() {
            for size in [1, 7, 128, 255, 256, 257, 1000, 4096, 65537] {
                let aligned = align_buffer(size);
                assert_eq!(
                    aligned % BUFFER_ALIGNMENT,
                    0,
                    "align_buffer({size}) = {aligned} not a multiple of {BUFFER_ALIGNMENT}"
                );
                assert!(aligned >= size, "aligned {aligned} < original {size}");
            }
        }

        #[test]
        fn typical_tensor_buffer_sizes() {
            // f32 tensor: 1024 elements x 4 bytes = 4096 bytes (already aligned)
            assert_eq!(align_buffer(1024 * 4), 4096);
            // f16 tensor: 1024 elements x 2 bytes = 2048 (already aligned)
            assert_eq!(align_buffer(1024 * 2), 2048);
            // odd size: 1023 x 4 = 4092 -> 4096
            assert_eq!(align_buffer(1023 * 4), 4096);
        }
    }

    // ── edge cases: zero-size dispatches ────────────────────────────────

    mod zero_size_dispatch {
        use super::*;

        #[test]
        fn threadgroup_1d_zero() {
            assert_eq!(optimal_threadgroup_1d(0), 0);
        }

        #[test]
        fn threadgroup_2d_zero_col() {
            assert_eq!(optimal_threadgroup_2d(0, 100), (0, 0));
        }

        #[test]
        fn threadgroup_2d_zero_row() {
            assert_eq!(optimal_threadgroup_2d(100, 0), (0, 0));
        }

        #[test]
        fn threadgroup_3d_any_zero() {
            assert_eq!(optimal_threadgroup_3d(0, 10, 10), (0, 0, 0));
            assert_eq!(optimal_threadgroup_3d(10, 0, 10), (0, 0, 0));
            assert_eq!(optimal_threadgroup_3d(10, 10, 0), (0, 0, 0));
        }

        #[test]
        fn grid_1d_zero_total() {
            assert_eq!(dispatch_grid_1d(0, 32), 0);
        }

        #[test]
        fn grid_1d_zero_group() {
            assert_eq!(dispatch_grid_1d(100, 0), 0);
        }

        #[test]
        fn grid_2d_zero_group() {
            assert_eq!(dispatch_grid_2d(64, 64, 0, 32), (0, 0));
            assert_eq!(dispatch_grid_2d(64, 64, 32, 0), (0, 0));
        }
    }

    // ── single-element tensors ──────────────────────────────────────────

    mod single_element {
        use super::*;

        #[test]
        fn threadgroup_1d_single() {
            let tg = optimal_threadgroup_1d(1);
            // Should round up to SIMD_WIDTH
            assert_eq!(tg, SIMD_WIDTH);
        }

        #[test]
        fn grid_1d_single() {
            let tg = optimal_threadgroup_1d(1);
            let grid = dispatch_grid_1d(1, tg);
            assert_eq!(grid, 1);
        }

        #[test]
        fn threadgroup_2d_single() {
            let (w, h) = optimal_threadgroup_2d(1, 1);
            assert!(w >= 1);
            assert!(h >= 1);
            assert!(w * h <= MAX_THREADS_PER_THREADGROUP);
        }
    }

    // ── non-power-of-2 dimensions ───────────────────────────────────────

    mod non_power_of_two {
        use super::*;

        #[test]
        fn prime_dimensions_covered() {
            for n in [7, 13, 127, 257, 1021, 4099] {
                let tg = optimal_threadgroup_1d(n);
                let grid = dispatch_grid_1d(n, tg);
                let covered = grid * tg;
                assert!(covered >= n, "prime {n}: covered={covered}");
            }
        }

        #[test]
        fn odd_2d_dimensions() {
            let (w, h) = optimal_threadgroup_2d(997, 503);
            assert!(w > 0 && h > 0);
            let (gx, gy) = dispatch_grid_2d(997, 503, w, h);
            assert!(gx * w >= 997);
            assert!(gy * h >= 503);
        }
    }

    // ── thread density ──────────────────────────────────────────────────

    mod thread_density {
        use super::*;

        #[test]
        fn waste_ratio_within_bounds_1d() {
            // For reasonable sizes, waste (extra threads beyond total) should be
            // less than one full threadgroup.
            for total in [32, 1000, 4096, 65536] {
                let tg = optimal_threadgroup_1d(total);
                let grid = dispatch_grid_1d(total, tg);
                let launched = grid * tg;
                let waste = launched - total;
                assert!(waste < tg, "1D waste {waste} >= threadgroup {tg} for total={total}");
            }
        }

        #[test]
        fn no_negative_waste() {
            for total in [1, 32, 1024, 100_000] {
                let tg = optimal_threadgroup_1d(total);
                let grid = dispatch_grid_1d(total, tg);
                assert!(grid * tg >= total);
            }
        }
    }

    // ── shared memory limits ────────────────────────────────────────────

    mod shared_memory {
        use super::*;

        #[test]
        fn small_tile_fits() {
            // 256 f32 values = 1024 bytes
            let mem = shared_memory_for_tile(256);
            assert_eq!(mem, 1024);
            assert!(mem <= MAX_THREADGROUP_MEMORY);
        }

        #[test]
        fn max_f32_tile_in_shared_memory() {
            // 32 KB / 4 bytes = 8192 f32 values
            let max_floats = MAX_THREADGROUP_MEMORY / std::mem::size_of::<f32>();
            assert_eq!(max_floats, 8192);
            assert_eq!(shared_memory_for_tile(max_floats), MAX_THREADGROUP_MEMORY);
        }

        #[test]
        fn exceeds_shared_memory_detected() {
            let mem = shared_memory_for_tile(8193);
            assert!(mem > MAX_THREADGROUP_MEMORY, "8193 f32s should exceed 32 KB shared memory");
        }

        #[test]
        fn typical_matmul_tile_fits() {
            // 16x16 tile = 256 floats x 2 (A + B tiles) = 2048 bytes
            let tile_a = shared_memory_for_tile(16 * 16);
            let tile_b = shared_memory_for_tile(16 * 16);
            assert!(tile_a + tile_b <= MAX_THREADGROUP_MEMORY);
        }

        #[test]
        fn reduction_scratchpad_fits() {
            // Reduction: one f32 per thread in a 1024-thread group = 4096 bytes
            let mem = shared_memory_for_tile(MAX_THREADS_PER_THREADGROUP as usize);
            assert_eq!(mem, 4096);
            assert!(mem <= MAX_THREADGROUP_MEMORY);
        }
    }

    // ── round-up dispatch (ceil division) ───────────────────────────────

    mod round_up_dispatch {
        use super::*;

        #[test]
        fn exact_multiples_no_extra_groups() {
            assert_eq!(ceil_div(1024, 1024), 1);
            assert_eq!(ceil_div(2048, 1024), 2);
            assert_eq!(ceil_div(32, 32), 1);
        }

        #[test]
        fn one_element_over_triggers_extra_group() {
            assert_eq!(ceil_div(1025, 1024), 2);
            assert_eq!(ceil_div(33, 32), 2);
        }

        #[test]
        fn large_element_counts() {
            // 1M elements / 256 threads = 3907 groups (rounds from 3906.25)
            assert_eq!(ceil_div(1_000_000, 256), 3907);
        }
    }

    // ── Metal-specific constraint validation ────────────────────────────

    mod metal_constraints {
        use super::*;

        #[test]
        fn thread_execution_width_is_32() {
            assert_eq!(SIMD_WIDTH, 32, "Apple Silicon threadExecutionWidth must be 32");
        }

        #[test]
        fn max_total_threads_is_1024() {
            assert_eq!(MAX_THREADS_PER_THREADGROUP, 1024);
        }

        #[test]
        fn max_threads_is_simd_multiple() {
            assert_eq!(MAX_THREADS_PER_THREADGROUP % SIMD_WIDTH, 0);
        }

        #[test]
        fn buffer_alignment_is_256() {
            assert_eq!(BUFFER_ALIGNMENT, 256);
        }

        #[test]
        fn threadgroup_memory_is_32kb() {
            assert_eq!(MAX_THREADGROUP_MEMORY, 32 * 1024);
        }

        #[test]
        fn optimal_2d_width_at_least_simd() {
            // For any non-trivial column count, the width should use SIMD_WIDTH.
            for cols in [32, 64, 128, 1024] {
                let (w, _) = optimal_threadgroup_2d(cols, 32);
                assert_eq!(w, SIMD_WIDTH, "2D width should be SIMD_WIDTH for cols={cols}");
            }
        }

        #[test]
        fn optimal_2d_narrow_tensor() {
            // When cols < SIMD_WIDTH, width should clamp to cols.
            let (w, h) = optimal_threadgroup_2d(8, 1024);
            assert_eq!(w, 8);
            assert!(w * h <= MAX_THREADS_PER_THREADGROUP);
        }
    }

    // ── common dispatch scenarios ───────────────────────────────────────

    mod common_scenarios {
        use super::*;

        #[test]
        fn vector_elementwise_4096() {
            let n = 4096u32;
            let tg = optimal_threadgroup_1d(n);
            let grid = dispatch_grid_1d(n, tg);
            assert_eq!(tg, 1024); // max threadgroup
            assert_eq!(grid, 4); // 4096/1024
        }

        #[test]
        fn matmul_1024x1024() {
            let (w, h) = optimal_threadgroup_2d(1024, 1024);
            let (gx, gy) = dispatch_grid_2d(1024, 1024, w, h);
            assert!(gx * w >= 1024);
            assert!(gy * h >= 1024);
            assert!(w * h <= MAX_THREADS_PER_THREADGROUP);
        }

        #[test]
        fn batch_norm_batch_32_channels_256() {
            // Typical BN: reduce over (batch=32, spatial=1024) per channel.
            let total_per_channel = 32u32 * 1024;
            let tg = optimal_threadgroup_1d(total_per_channel);
            let grid = dispatch_grid_1d(total_per_channel, tg);
            assert!(grid * tg >= total_per_channel);
        }

        #[test]
        fn attention_head_64x64() {
            // Single attention head: 64 query positions x 64 key positions
            let (w, h) = optimal_threadgroup_2d(64, 64);
            let (gx, gy) = dispatch_grid_2d(64, 64, w, h);
            assert!(gx * w >= 64);
            assert!(gy * h >= 64);
        }

        #[test]
        fn embedding_lookup_50k_vocab() {
            // Each token selects one embedding vector (just need per-element copy)
            let vocab_dim = 768u32;
            let tg = optimal_threadgroup_1d(vocab_dim);
            let grid = dispatch_grid_1d(vocab_dim, tg);
            assert_eq!(grid, 1); // 768 fits in a single 1024-thread group
        }
    }
}
