//! Kernel variant selection based on problem size and device capabilities.
//!
//! The [`KernelSelector`] inspects the matrix dimensions and the device
//! capabilities to choose the best kernel variant (e.g. tiled vs. naive,
//! FP16 vs. FP32 accumulation).

use crate::OpenClDeviceInfo;

/// A kernel variant chosen by the selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelVariant {
    /// Small problem — use a simple, low-overhead kernel.
    Small,
    /// Medium problem — use a tiled kernel with local memory.
    Tiled,
    /// Large problem — use a multi-pass reduction kernel.
    LargeReduction,
}

impl std::fmt::Display for KernelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelVariant::Small => write!(f, "small"),
            KernelVariant::Tiled => write!(f, "tiled"),
            KernelVariant::LargeReduction => write!(f, "large-reduction"),
        }
    }
}

/// Selects the optimal kernel variant for a given problem size and device.
pub struct KernelSelector;

impl KernelSelector {
    /// Choose the GEMV kernel variant based on the output dimension and
    /// inner dimension `k`.
    ///
    /// # Panics
    ///
    /// Does not panic. Returns `Small` as the safe default for edge cases.
    pub fn select_gemv(n_out: usize, k: usize, device: &OpenClDeviceInfo) -> KernelVariant {
        if n_out == 0 || k == 0 {
            return KernelVariant::Small;
        }

        let total_work = n_out.saturating_mul(k);

        // Threshold heuristic: if the problem fits in one workgroup worth
        // of work with a single pass, use the simple kernel.
        let workgroup_capacity = device.max_workgroup_size * 256;

        if total_work <= workgroup_capacity {
            KernelVariant::Small
        } else if total_work <= workgroup_capacity * device.max_compute_units as usize {
            KernelVariant::Tiled
        } else {
            KernelVariant::LargeReduction
        }
    }

    /// Choose the RMSNorm kernel variant based on hidden dimension.
    pub fn select_rmsnorm(hidden_dim: usize, device: &OpenClDeviceInfo) -> KernelVariant {
        if hidden_dim == 0 {
            return KernelVariant::Small;
        }
        if hidden_dim <= device.max_workgroup_size {
            KernelVariant::Small
        } else {
            KernelVariant::Tiled
        }
    }

    /// Choose the attention kernel variant based on sequence lengths.
    pub fn select_attention(
        seq_len_q: usize,
        seq_len_kv: usize,
        head_dim: usize,
        device: &OpenClDeviceInfo,
    ) -> KernelVariant {
        if seq_len_q == 0 || seq_len_kv == 0 || head_dim == 0 {
            return KernelVariant::Small;
        }

        let score_elements = seq_len_q.saturating_mul(seq_len_kv);

        // Tiled attention when the score matrix exceeds local memory.
        let score_bytes = score_elements.saturating_mul(4); // FP32
        if score_bytes <= device.local_mem_bytes as usize {
            KernelVariant::Small
        } else if score_bytes <= device.global_mem_bytes as usize / 4 {
            KernelVariant::Tiled
        } else {
            KernelVariant::LargeReduction
        }
    }
}
