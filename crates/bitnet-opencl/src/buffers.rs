//! Buffer size calculations for OpenCL kernel arguments.
//!
//! Each function computes the number of bytes required for a specific
//! tensor layout so the host can allocate OpenCL buffers of exactly
//! the right size.

use bitnet_common::{KernelError, Result};

/// Bytes required for a QK256-packed weight matrix.
///
/// QK256 packs 256 ternary weights into 64 bytes (2 bits each).
/// An `[n_out, k]` weight matrix requires `n_out * (k / 256) * 64` bytes.
///
/// # Errors
///
/// Returns an error if `k` is not a positive multiple of 256, or if the
/// computed size would overflow `usize`.
pub fn qk256_weight_bytes(n_out: usize, k: usize) -> Result<usize> {
    if k == 0 || !k.is_multiple_of(256) {
        return Err(KernelError::InvalidArguments {
            reason: format!("QK256 inner dimension k={k} must be a positive multiple of 256"),
        }
        .into());
    }

    let blocks_per_row = k / 256;
    let bytes_per_row =
        blocks_per_row.checked_mul(64).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("QK256 buffer overflow: blocks_per_row={blocks_per_row} * 64"),
        })?;

    n_out.checked_mul(bytes_per_row).ok_or_else(|| {
        KernelError::InvalidArguments {
            reason: format!("QK256 buffer overflow: n_out={n_out} * bytes_per_row={bytes_per_row}"),
        }
        .into()
    })
}

/// Bytes required for QK256 per-block scale factors (FP16).
///
/// One `f16` (2 bytes) per 256-element block.
pub fn qk256_scale_bytes(n_out: usize, k: usize) -> Result<usize> {
    if k == 0 || !k.is_multiple_of(256) {
        return Err(KernelError::InvalidArguments {
            reason: format!("QK256 inner dimension k={k} must be a positive multiple of 256"),
        }
        .into());
    }

    let blocks_per_row = k / 256;
    let total_blocks =
        n_out.checked_mul(blocks_per_row).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("QK256 scale overflow: n_out={n_out} * blocks={blocks_per_row}"),
        })?;

    total_blocks.checked_mul(2).ok_or_else(|| {
        KernelError::InvalidArguments {
            reason: format!("QK256 scale overflow: total_blocks={total_blocks} * 2"),
        }
        .into()
    })
}

/// Bytes required for an FP32 tensor of the given element count.
///
/// # Errors
///
/// Returns an error if `count * 4` would overflow `usize`.
pub fn fp32_buffer_bytes(count: usize) -> Result<usize> {
    count.checked_mul(4).ok_or_else(|| {
        KernelError::InvalidArguments { reason: format!("FP32 buffer overflow: count={count} * 4") }
            .into()
    })
}

/// Bytes required for an FP16 tensor of the given element count.
pub fn fp16_buffer_bytes(count: usize) -> Result<usize> {
    count.checked_mul(2).ok_or_else(|| {
        KernelError::InvalidArguments { reason: format!("FP16 buffer overflow: count={count} * 2") }
            .into()
    })
}

/// Shared memory bytes needed by the QK256 GEMV kernel for a given `k`.
///
/// Mirrors the CUDA calculation: `(k / 256) * (64 + 2)` bytes, with a
/// minimum of 4096.
pub fn qk256_gemv_shared_mem(k: usize) -> Result<usize> {
    if k == 0 || !k.is_multiple_of(256) {
        return Err(KernelError::InvalidArguments {
            reason: format!("QK256 inner dimension k={k} must be a positive multiple of 256"),
        }
        .into());
    }
    let blocks = k / 256;
    let raw = blocks.checked_mul(66).ok_or_else(|| KernelError::InvalidArguments {
        reason: format!("shared mem overflow: blocks={blocks} * 66"),
    })?;
    Ok(raw.max(4096))
}
