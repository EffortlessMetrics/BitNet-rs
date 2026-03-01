//! CUDA linear projection kernel launcher.
//!
//! GPU-specific launch stub and PTX kernel source for linear projection
//! (`y = x · Wᵀ + bias`). The CPU fallback and shared configuration
//! live in [`crate::cpu::linear`].

use crate::cpu::linear::LinearConfig;
use bitnet_common::{KernelError, Result};

// ── CUDA kernel source ────────────────────────────────────────────────

/// CUDA kernel source for linear projection (y = x · Wᵀ + bias).
///
/// Tiled GEMM with transpose-B, plus optional bias addition.
pub const LINEAR_KERNEL_SRC: &str = r#"
extern "C" {

// Linear projection kernel: y = x * W^T + bias
// x:      [batch_size, in_features]  (row-major)
// weight: [out_features, in_features] (row-major, transposed during mul)
// bias:   [out_features] or NULL
// output: [batch_size, out_features]  (row-major)
__global__ void linear_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int has_bias
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output feature

    if (row < batch_size && col < out_features) {
        float acc = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            // x[row, k] * W[col, k]  (W transposed: W^T[k, col])
            acc += x[row * in_features + k] * weight[col * in_features + k];
        }
        if (has_bias) {
            acc += bias[col];
        }
        output[row * out_features + col] = acc;
    }
}

} // extern "C"
"#;

// ── CUDA launch stub ──────────────────────────────────────────────────

/// Launch stub for the linear projection CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled
/// and loaded.
pub fn launch_linear(
    _x: &[f32],
    _weight: &[f32],
    _bias: Option<&[f32]>,
    _output: &mut [f32],
    config: &LinearConfig,
) -> Result<()> {
    log::debug!(
        "linear CUDA stub: batch={}, in={}, out={}, bias={}, grid={:?}",
        config.batch_size,
        config.in_features,
        config.out_features,
        config.has_bias,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "linear projection CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}
