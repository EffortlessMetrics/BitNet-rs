//! Kernel variant selection based on problem size and hardware capabilities.
//!
//! [`KernelSelector`] inspects the device limits (max work-group size,
//! preferred vector width, sub-group support) and the problem dimensions
//! to pick the most efficient kernel variant for each operation.

use crate::kernels;

/// Kernel implementation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelVariant {
    /// Simple per-element kernel – best for tiny problems.
    Naive,
    /// Local-memory tiled kernel (TILE×TILE work-groups).
    Tiled,
    /// float4 vectorized kernel for high-throughput bulk work.
    Vectorized,
    /// Sub-group (warp/wave) cooperative kernel.
    Subgroup,
}

/// Hardware-aware kernel selector.
///
/// Constructed once per device and reused for every dispatch decision.
#[derive(Debug, Clone)]
pub struct KernelSelector {
    max_work_group_size: usize,
    has_subgroups: bool,
    preferred_vector_width: usize,
}

/// Threshold below which the naive (non-tiled) kernel is faster because
/// the tiling overhead dominates.
const MATMUL_TILE_THRESHOLD: usize = 64;

/// Minimum row length before vectorised softmax is worthwhile.
const SOFTMAX_VEC_THRESHOLD: usize = 256;

/// Minimum seq_len before fused attention beats separate kernels.
const ATTENTION_FUSED_THRESHOLD: usize = 32;

impl KernelSelector {
    /// Create a selector with the given device capabilities.
    pub fn new(
        max_work_group_size: usize,
        has_subgroups: bool,
        preferred_vector_width: usize,
    ) -> Self {
        Self { max_work_group_size, has_subgroups, preferred_vector_width }
    }

    /// Recommended tile size clamped to device limits.
    pub fn tile_size(&self) -> usize {
        let ts = kernels::DEFAULT_TILE_SIZE;
        // tile_size² must fit in max_work_group_size.
        if ts * ts > self.max_work_group_size {
            // Fall back to the largest square that fits.
            let side = (self.max_work_group_size as f64).sqrt() as usize;
            side.max(1)
        } else {
            ts
        }
    }

    /// Select the best matmul kernel for the given dimensions.
    pub fn select_matmul(&self, m: usize, n: usize, k: usize) -> KernelVariant {
        let elements = m.saturating_mul(n);
        if elements < MATMUL_TILE_THRESHOLD * MATMUL_TILE_THRESHOLD {
            return KernelVariant::Naive;
        }
        // Prefer vectorized when N is wide enough and device supports float4.
        if self.preferred_vector_width >= kernels::VECTOR_WIDTH
            && n >= kernels::VECTOR_WIDTH
            && k >= kernels::DEFAULT_TILE_SIZE
        {
            return KernelVariant::Vectorized;
        }
        if self.has_subgroups && m >= 128 && n >= 128 {
            return KernelVariant::Subgroup;
        }
        KernelVariant::Tiled
    }

    /// Select the best softmax kernel.
    pub fn select_softmax(&self, n: usize) -> KernelVariant {
        if n < SOFTMAX_VEC_THRESHOLD {
            return KernelVariant::Naive;
        }
        if self.preferred_vector_width >= kernels::VECTOR_WIDTH
            && n.is_multiple_of(kernels::VECTOR_WIDTH)
        {
            return KernelVariant::Vectorized;
        }
        KernelVariant::Tiled // work-group reduction variant
    }

    /// Select the best attention kernel.
    pub fn select_attention(&self, seq_len: usize, head_dim: usize) -> KernelVariant {
        if seq_len < ATTENTION_FUSED_THRESHOLD {
            return KernelVariant::Naive;
        }
        // Local memory needed: (seq_len + wg_size) * sizeof(float)
        let local_bytes = (seq_len + self.max_work_group_size) * 4;
        // Intel GPUs typically have 64 KB SLM per sub-slice.
        const MAX_LOCAL_BYTES: usize = 48 * 1024;
        if local_bytes > MAX_LOCAL_BYTES {
            // Sequence too long for single-pass fused kernel.
            return KernelVariant::Tiled;
        }
        if self.has_subgroups && head_dim >= 64 {
            return KernelVariant::Subgroup;
        }
        KernelVariant::Vectorized
    }

    /// Maximum work-group size reported by the device.
    pub fn max_work_group_size(&self) -> usize {
        self.max_work_group_size
    }

    /// Whether the device supports sub-group operations.
    pub fn has_subgroups(&self) -> bool {
        self.has_subgroups
    }

    /// Preferred vector width (in floats) for the device.
    pub fn preferred_vector_width(&self) -> usize {
        self.preferred_vector_width
    }
}
