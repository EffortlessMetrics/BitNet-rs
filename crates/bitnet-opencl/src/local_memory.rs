//! OpenCL local memory optimization for GPU kernels.
//!
//! Provides tiled matrix multiplication using `__local` memory for
//! work-group shared data, configurable tile sizes based on device
//! local memory capacity, and allocation strategies that respect
//! hardware limits.

use std::fmt;

/// Error types for local memory operations.
#[derive(Debug, thiserror::Error)]
pub enum LocalMemoryError {
    #[error("requested local memory {requested} bytes exceeds device limit {limit} bytes")]
    ExceedsDeviceLimit { requested: usize, limit: usize },

    #[error("tile size {tile_size} exceeds local memory capacity for dtype size {dtype_bytes}")]
    TileTooLarge { tile_size: usize, dtype_bytes: usize },

    #[error("matrix dimensions ({m}×{k}) × ({k2}×{n}) incompatible (k={k} != k2={k2})")]
    DimensionMismatch { m: usize, k: usize, k2: usize, n: usize },

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Device local memory capabilities queried from OpenCL.
#[derive(Debug, Clone, Copy)]
pub struct LocalMemoryInfo {
    /// Total local memory available per work-group in bytes.
    pub total_bytes: usize,
    /// Whether local memory is dedicated (true) or shared with global (false).
    pub is_dedicated: bool,
    /// Maximum work-group size supported by the device.
    pub max_work_group_size: usize,
}

impl LocalMemoryInfo {
    /// Compute the maximum square tile size that fits in local memory.
    ///
    /// A tiled matmul requires 2 tiles of `tile_size × tile_size` elements.
    /// Each element occupies `dtype_bytes`.
    pub fn max_tile_size(&self, dtype_bytes: usize) -> usize {
        // 2 tiles: A_tile + B_tile, each tile_size^2 elements
        // total = 2 * tile_size^2 * dtype_bytes <= total_bytes
        let max_elements = self.total_bytes / (2 * dtype_bytes);
        // tile_size = floor(sqrt(max_elements))
        (max_elements as f64).sqrt() as usize
    }
}

/// Configuration for local-memory-optimized tiled matmul.
#[derive(Debug, Clone, Copy)]
pub struct TiledMatmulConfig {
    /// Tile dimension (tiles are tile_size × tile_size).
    pub tile_size: usize,
    /// Bytes per element of matrix A (e.g., 1 for i8, 4 for f32).
    pub dtype_a_bytes: usize,
    /// Bytes per element of matrix B.
    pub dtype_b_bytes: usize,
    /// Bytes per element of accumulator/output C.
    pub dtype_c_bytes: usize,
}

impl Default for TiledMatmulConfig {
    fn default() -> Self {
        Self {
            tile_size: 16,
            dtype_a_bytes: 1,  // i8 for ternary weights
            dtype_b_bytes: 4,  // f32 activations
            dtype_c_bytes: 4,  // f32 output
        }
    }
}

impl TiledMatmulConfig {
    /// Create a config with the given tile size.
    pub fn with_tile_size(tile_size: usize) -> Self {
        Self { tile_size, ..Default::default() }
    }

    /// Total local memory required for both A and B tiles (in bytes).
    pub fn local_memory_required(&self) -> usize {
        let a_tile = self.tile_size * self.tile_size * self.dtype_a_bytes;
        let b_tile = self.tile_size * self.tile_size * self.dtype_b_bytes;
        a_tile + b_tile
    }

    /// Validate the config against device capabilities.
    pub fn validate(&self, info: &LocalMemoryInfo) -> Result<(), LocalMemoryError> {
        if self.tile_size == 0 {
            return Err(LocalMemoryError::InvalidConfig(
                "tile_size must be > 0".into(),
            ));
        }

        let required = self.local_memory_required();
        if required > info.total_bytes {
            return Err(LocalMemoryError::ExceedsDeviceLimit {
                requested: required,
                limit: info.total_bytes,
            });
        }

        let work_group = self.tile_size * self.tile_size;
        if work_group > info.max_work_group_size {
            return Err(LocalMemoryError::InvalidConfig(format!(
                "work-group size {} (tile {}²) exceeds device max {}",
                work_group, self.tile_size, info.max_work_group_size
            )));
        }

        Ok(())
    }

    /// Select the best tile size for the given device.
    ///
    /// Picks the largest power-of-2 tile size that fits within local memory
    /// and work-group limits.
    pub fn auto_select(info: &LocalMemoryInfo) -> Self {
        let config = Self::default();
        let max_by_memory = info.max_tile_size(
            std::cmp::max(config.dtype_a_bytes, config.dtype_b_bytes),
        );
        let max_by_wg = (info.max_work_group_size as f64).sqrt() as usize;
        let max_tile = std::cmp::min(max_by_memory, max_by_wg);

        // Pick largest power-of-2 ≤ max_tile, minimum 4.
        let tile_size = if max_tile >= 32 {
            32
        } else if max_tile >= 16 {
            16
        } else if max_tile >= 8 {
            8
        } else {
            4
        };

        Self { tile_size, ..config }
    }
}

/// Local memory allocation plan for a single kernel invocation.
#[derive(Debug, Clone)]
pub struct LocalMemoryAllocation {
    /// Name of the allocation (for diagnostics).
    pub name: String,
    /// Size in bytes.
    pub size_bytes: usize,
}

impl fmt::Display for LocalMemoryAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={} bytes", self.name, self.size_bytes)
    }
}

/// Allocation strategy that plans local memory usage for a kernel launch.
#[derive(Debug)]
pub struct LocalMemoryAllocator {
    device_info: LocalMemoryInfo,
    allocations: Vec<LocalMemoryAllocation>,
    total_allocated: usize,
}

impl LocalMemoryAllocator {
    /// Create a new allocator for the given device.
    pub fn new(device_info: LocalMemoryInfo) -> Self {
        Self {
            device_info,
            allocations: Vec::new(),
            total_allocated: 0,
        }
    }

    /// Reserve local memory for a named buffer.
    pub fn allocate(
        &mut self,
        name: &str,
        size_bytes: usize,
    ) -> Result<(), LocalMemoryError> {
        let new_total = self.total_allocated + size_bytes;
        if new_total > self.device_info.total_bytes {
            return Err(LocalMemoryError::ExceedsDeviceLimit {
                requested: new_total,
                limit: self.device_info.total_bytes,
            });
        }
        self.allocations.push(LocalMemoryAllocation {
            name: name.to_string(),
            size_bytes,
        });
        self.total_allocated = new_total;
        Ok(())
    }

    /// Total bytes allocated so far.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Remaining local memory available.
    pub fn remaining(&self) -> usize {
        self.device_info.total_bytes.saturating_sub(self.total_allocated)
    }

    /// Get all allocations.
    pub fn allocations(&self) -> &[LocalMemoryAllocation] {
        &self.allocations
    }

    /// Reset all allocations.
    pub fn reset(&mut self) {
        self.allocations.clear();
        self.total_allocated = 0;
    }
}

/// Generate an OpenCL kernel source string for tiled matmul using local memory.
///
/// The generated kernel performs C = A × B where:
/// - A is M×K (row-major, i8 ternary weights)
/// - B is K×N (row-major, f32 activations)
/// - C is M×N (row-major, f32 output)
///
/// Work-groups of `tile_size × tile_size` threads cooperatively load
/// tiles of A and B into `__local` memory, compute partial sums, and
/// iterate over K in steps of `tile_size`.
pub fn generate_tiled_matmul_kernel(config: &TiledMatmulConfig) -> String {
    let ts = config.tile_size;
    format!(
        r#"// Auto-generated tiled matmul kernel (tile_size={ts})
// Uses __local memory for work-group shared data staging.
__kernel void matmul_tiled(
    __global const char* A,   // M×K, i8 ternary
    __global const float* B,  // K×N, f32
    __global float* C,        // M×N, f32
    const uint M,
    const uint N,
    const uint K)
{{
    const uint TILE = {ts};
    const uint row = get_local_id(0);
    const uint col = get_local_id(1);
    const uint global_row = get_group_id(0) * TILE + row;
    const uint global_col = get_group_id(1) * TILE + col;

    __local char  A_tile[{ts}][{ts}];
    __local float B_tile[{ts}][{ts}];

    float acc = 0.0f;

    const uint num_tiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < num_tiles; t++) {{
        // Load A tile
        uint a_col = t * TILE + col;
        if (global_row < M && a_col < K)
            A_tile[row][col] = A[global_row * K + a_col];
        else
            A_tile[row][col] = 0;

        // Load B tile
        uint b_row = t * TILE + row;
        if (b_row < K && global_col < N)
            B_tile[row][col] = B[b_row * N + global_col];
        else
            B_tile[row][col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot product
        for (uint e = 0; e < TILE; e++) {{
            acc += (float)A_tile[row][e] * B_tile[e][col];
        }}

        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    if (global_row < M && global_col < N) {{
        C[global_row * N + global_col] = acc;
    }}
}}
"#
    )
}

/// Perform reference (CPU) tiled matmul to validate GPU results.
///
/// A: M×K i8, B: K×N f32, C: M×N f32.
pub fn reference_tiled_matmul(
    a: &[i8],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, LocalMemoryError> {
    if a.len() != m * k {
        return Err(LocalMemoryError::DimensionMismatch {
            m,
            k,
            k2: if m > 0 { a.len() / m } else { 0 },
            n,
        });
    }
    if b.len() != k * n {
        return Err(LocalMemoryError::DimensionMismatch {
            m,
            k,
            k2: if n > 0 { b.len() / n } else { 0 },
            n,
        });
    }

    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] as f32 * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    Ok(c)
}

/// OpenCL kernel source for tiled matmul with local memory (tile=16, default).
pub const TILED_MATMUL_CL_SRC: &str = include_str!("kernels/matmul_tiled.cl");

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> LocalMemoryInfo {
        LocalMemoryInfo {
            total_bytes: 65536, // 64 KiB (typical Intel Arc)
            is_dedicated: true,
            max_work_group_size: 1024,
        }
    }

    #[test]
    fn test_max_tile_size_f32() {
        let info = test_device();
        // 2 * tile^2 * 4 <= 65536 → tile^2 <= 8192 → tile <= 90
        let max = info.max_tile_size(4);
        assert!(max >= 64 && max <= 90, "max_tile={max}");
    }

    #[test]
    fn test_max_tile_size_i8() {
        let info = test_device();
        // 2 * tile^2 * 1 <= 65536 → tile^2 <= 32768 → tile <= 181
        let max = info.max_tile_size(1);
        assert!(max >= 128 && max <= 181, "max_tile={max}");
    }

    #[test]
    fn test_config_validate_fits() {
        let info = test_device();
        let config = TiledMatmulConfig::with_tile_size(16);
        // 16^2 * 1 + 16^2 * 4 = 256 + 1024 = 1280 bytes — fits
        assert!(config.validate(&info).is_ok());
    }

    #[test]
    fn test_config_validate_too_large() {
        let info = LocalMemoryInfo {
            total_bytes: 256, // very small
            is_dedicated: true,
            max_work_group_size: 1024,
        };
        let config = TiledMatmulConfig::with_tile_size(16);
        let err = config.validate(&info).unwrap_err();
        assert!(matches!(err, LocalMemoryError::ExceedsDeviceLimit { .. }));
    }

    #[test]
    fn test_config_validate_work_group_too_large() {
        let info = LocalMemoryInfo {
            total_bytes: 65536,
            is_dedicated: true,
            max_work_group_size: 128, // only 128 threads
        };
        // tile=16 → WG=256 > 128
        let config = TiledMatmulConfig::with_tile_size(16);
        let err = config.validate(&info).unwrap_err();
        assert!(matches!(err, LocalMemoryError::InvalidConfig(_)));
    }

    #[test]
    fn test_auto_select_picks_appropriate_tile() {
        let info = test_device();
        let config = TiledMatmulConfig::auto_select(&info);
        assert!(
            config.tile_size == 16 || config.tile_size == 32,
            "tile_size={}",
            config.tile_size
        );
        assert!(config.validate(&info).is_ok());
    }

    #[test]
    fn test_auto_select_small_device() {
        let info = LocalMemoryInfo {
            total_bytes: 512,
            is_dedicated: true,
            max_work_group_size: 64,
        };
        let config = TiledMatmulConfig::auto_select(&info);
        assert!(config.tile_size <= 8, "tile_size={}", config.tile_size);
    }

    #[test]
    fn test_allocator_basic() {
        let info = test_device();
        let mut alloc = LocalMemoryAllocator::new(info);
        alloc.allocate("a_tile", 1024).unwrap();
        alloc.allocate("b_tile", 4096).unwrap();
        assert_eq!(alloc.total_allocated(), 5120);
        assert_eq!(alloc.remaining(), 65536 - 5120);
        assert_eq!(alloc.allocations().len(), 2);
    }

    #[test]
    fn test_allocator_exceeds_limit() {
        let info = test_device();
        let mut alloc = LocalMemoryAllocator::new(info);
        alloc.allocate("big", 60000).unwrap();
        let err = alloc.allocate("overflow", 10000).unwrap_err();
        assert!(matches!(err, LocalMemoryError::ExceedsDeviceLimit { .. }));
    }

    #[test]
    fn test_allocator_reset() {
        let info = test_device();
        let mut alloc = LocalMemoryAllocator::new(info);
        alloc.allocate("buf", 1024).unwrap();
        alloc.reset();
        assert_eq!(alloc.total_allocated(), 0);
        assert!(alloc.allocations().is_empty());
    }

    #[test]
    fn test_reference_matmul_identity() {
        // 2×2 identity × [1,2; 3,4] = [1,2; 3,4]
        let a: Vec<i8> = vec![1, 0, 0, 1]; // identity
        let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let c = reference_tiled_matmul(&a, &b, 2, 2, 2).unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reference_matmul_ternary() {
        // Ternary weights {-1, 0, 1}
        let a: Vec<i8> = vec![1, -1, 0, 1];
        let b: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0];
        // C[0,0] = 1*2 + (-1)*4 = -2
        // C[0,1] = 1*3 + (-1)*5 = -2
        // C[1,0] = 0*2 + 1*4 = 4
        // C[1,1] = 0*3 + 1*5 = 5
        let c = reference_tiled_matmul(&a, &b, 2, 2, 2).unwrap();
        assert_eq!(c, vec![-2.0, -2.0, 4.0, 5.0]);
    }

    #[test]
    fn test_generate_kernel_contains_local_memory() {
        let config = TiledMatmulConfig::with_tile_size(16);
        let src = generate_tiled_matmul_kernel(&config);
        assert!(src.contains("__local char"));
        assert!(src.contains("__local float"));
        assert!(src.contains("barrier(CLK_LOCAL_MEM_FENCE)"));
        assert!(src.contains("A_tile[16][16]"));
        assert!(src.contains("B_tile[16][16]"));
    }

    #[test]
    fn test_generate_kernel_different_tile_sizes() {
        for ts in [4, 8, 16, 32] {
            let config = TiledMatmulConfig::with_tile_size(ts);
            let src = generate_tiled_matmul_kernel(&config);
            let expected = format!("A_tile[{ts}][{ts}]");
            assert!(src.contains(&expected), "tile_size={ts} missing {expected}");
        }
    }

    #[test]
    fn test_local_memory_required_calculation() {
        let config = TiledMatmulConfig {
            tile_size: 16,
            dtype_a_bytes: 1,
            dtype_b_bytes: 4,
            dtype_c_bytes: 4,
        };
        // A tile: 16*16*1 = 256, B tile: 16*16*4 = 1024
        assert_eq!(config.local_memory_required(), 1280);
    }

    #[test]
    fn test_tiled_matmul_cl_src_is_valid() {
        // Verify the included .cl source contains expected keywords.
        assert!(TILED_MATMUL_CL_SRC.contains("__local"));
        assert!(TILED_MATMUL_CL_SRC.contains("barrier"));
        assert!(TILED_MATMUL_CL_SRC.contains("matmul_tiled"));
    }
}
