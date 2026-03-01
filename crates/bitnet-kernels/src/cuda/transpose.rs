//! CUDA transpose and reshape kernels with CPU fallback.
//!
//! # Transpose operations
//!
//! - **2D transpose**: Tiled shared-memory transpose for row-major matrices.
//!   Uses 32×32 tiles with +1 padding to avoid bank conflicts.
//! - **ND transpose**: Generalised permutation-based transpose for
//!   arbitrary-rank tensors, computing output strides from the inverse
//!   permutation.
//!
//! # Reshape
//!
//! [`reshape_cpu`] validates that old and new shapes have the same total
//! element count and returns a reinterpreted (zero-copy logical) copy of
//! the data.
//!
//! # Kernel strategy
//!
//! The 2D CUDA kernel uses 32×32 shared-memory tiles with a +1 column
//! padding to avoid bank conflicts on 32-bit accesses.  Each block loads
//! one tile collaboratively, synchronises, then writes the transposed tile
//! to global memory in coalesced order.
//!
//! The ND kernel maps each output element to a linear thread index, then
//! computes the corresponding input index via the inverse permutation.
//!
//! # CPU fallback
//!
//! CPU implementations live in [`crate::cpu::transpose`] and are
//! re-exported here for convenience.

#[cfg(any(feature = "gpu", feature = "cuda"))]
use bitnet_common::KernelError;
use bitnet_common::Result;

// Re-export CPU fallback functions.
pub use crate::cpu::transpose::{
    TransposeConfig as CudaTransposeConfig, reshape as reshape_cpu,
    transpose_2d as transpose_2d_cpu_fallback, transpose_nd as transpose_nd_cpu_fallback,
};

// ---------------------------------------------------------------------------
// CUDA kernel source (feature-gated)
// ---------------------------------------------------------------------------

/// Tiled 2D transpose CUDA kernel using 32×32 shared-memory tiles.
///
/// Padding (`TILE+1` columns) eliminates shared-memory bank conflicts.
/// Grid dimensions should cover `ceil(cols/TILE) × ceil(rows/TILE)` blocks.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const TRANSPOSE_2D_KERNEL_SRC: &str = r#"
#define TILE 32

extern "C" __global__ void transpose_2d_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    __shared__ float tile[TILE][TILE + 1];

    int bx = blockIdx.x * TILE;
    int by = blockIdx.y * TILE;

    int ix = bx + threadIdx.x;
    int iy = by + threadIdx.y;

    if (ix < cols && iy < rows) {
        tile[threadIdx.y][threadIdx.x] = input[iy * cols + ix];
    }
    __syncthreads();

    int ox = by + threadIdx.x;
    int oy = bx + threadIdx.y;

    if (ox < rows && oy < cols) {
        output[oy * rows + ox] = tile[threadIdx.x][threadIdx.y];
    }
}
"#;

/// ND transpose CUDA kernel using permutation-based index mapping.
///
/// Each thread computes the output-space coordinates from its linear index,
/// applies the inverse permutation to find the input-space coordinates, and
/// copies the element.  `ndim` must be ≤ 8.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const TRANSPOSE_ND_KERNEL_SRC: &str = r#"
#define MAX_DIMS 8

extern "C" __global__ void transpose_nd_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ out_shape,
    const int* __restrict__ in_strides,
    const int* __restrict__ perm,
    int ndim,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int coords[MAX_DIMS];
        int tmp = i;
        for (int d = ndim - 1; d >= 0; d--) {
            coords[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }

        int in_idx = 0;
        for (int d = 0; d < ndim; d++) {
            in_idx += coords[d] * in_strides[perm[d]];
        }
        output[i] = input[in_idx];
    }
}
"#;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Re-export `TransposeConfig` as `CudaTransposeConfig` for cuda-module
// consumers; the canonical implementation lives in `crate::cpu::transpose`.

// ---------------------------------------------------------------------------
// CUDA dispatch (feature-gated)
// ---------------------------------------------------------------------------

/// Launch the tiled 2D transpose kernel on the CUDA device.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_transpose_2d(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<()> {
    use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    let n = rows * cols;
    if input.len() < n || output.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: "buffer too small for transpose".into(),
        }
        .into());
    }

    let ctx = CudaContext::new(0).map_err(|e| KernelError::GpuError {
        reason: format!("failed to acquire CUDA device 0: {e:?}"),
    })?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(TRANSPOSE_2D_KERNEL_SRC).map_err(|e| KernelError::GpuError {
        reason: format!("NVRTC compilation failed: {e:?}"),
    })?;
    let module = ctx.load_module(ptx).map_err(|e| KernelError::GpuError {
        reason: format!("failed to load PTX module: {e:?}"),
    })?;
    let func = module.load_function("transpose_2d_f32").map_err(|e| KernelError::GpuError {
        reason: format!("transpose_2d_f32 not found: {e:?}"),
    })?;

    let input_dev = stream
        .memcpy_stod(input)
        .map_err(|e| KernelError::GpuError { reason: format!("memcpy H→D failed: {e:?}") })?;
    let mut output_dev: CudaSlice<f32> = stream
        .alloc_zeros(n)
        .map_err(|e| KernelError::GpuError { reason: format!("device alloc failed: {e:?}") })?;

    let tile = 32u32;
    let gx = (cols as u32 + tile - 1) / tile;
    let gy = (rows as u32 + tile - 1) / tile;
    let launch_cfg =
        LaunchConfig { grid_dim: (gx, gy, 1), block_dim: (tile, tile, 1), shared_mem_bytes: 0 };
    let rows_i = rows as i32;
    let cols_i = cols as i32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_dev);
    builder.arg(&mut output_dev);
    builder.arg(&rows_i);
    builder.arg(&cols_i);

    // Safety: kernel signature matches CUDA source; buffers validated above.
    unsafe { builder.launch(launch_cfg) }
        .map_err(|e| KernelError::GpuError { reason: format!("CUDA launch failed: {e:?}") })?;

    stream
        .synchronize()
        .map_err(|e| KernelError::GpuError { reason: format!("stream sync failed: {e:?}") })?;

    let host: Vec<f32> = stream
        .memcpy_dtov(&output_dev)
        .map_err(|e| KernelError::GpuError { reason: format!("memcpy D→H failed: {e:?}") })?;
    output[..n].copy_from_slice(&host[..n]);
    Ok(())
}

/// Unified 2D transpose dispatcher: tries GPU, falls back to CPU.
pub fn transpose_2d_forward(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if let Ok(()) = launch_transpose_2d(input, output, rows, cols) {
            return Ok(());
        }
        log::debug!("CUDA transpose_2d unavailable, falling back to CPU");
    }
    let result = transpose_2d_cpu_fallback(input, rows, cols);
    let n = rows * cols;
    output[..n].copy_from_slice(&result);
    Ok(())
}
