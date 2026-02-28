//! Workgroup size computation for OpenCL kernels.
//!
//! OpenCL work-group sizes must satisfy device constraints (max work-group
//! size, preferred multiple) and divide the global NDRange evenly.

use crate::OpenClDeviceInfo;
use bitnet_common::{KernelError, Result};

/// Computed work-group configuration for a 1-D kernel launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkgroupConfig {
    /// Local work-group size (number of work-items per group).
    pub local_size: usize,
    /// Global NDRange size (total work-items; multiple of `local_size`).
    pub global_size: usize,
    /// Number of work-groups (`global_size / local_size`).
    pub num_groups: usize,
}

impl WorkgroupConfig {
    /// Compute a 1-D workgroup configuration for `problem_size` elements.
    ///
    /// The local size is the largest multiple of the device's preferred
    /// work-group multiple that fits within `max_workgroup_size` and does
    /// not exceed `problem_size`.
    ///
    /// # Errors
    ///
    /// Returns `KernelError::InvalidArguments` if `problem_size` is zero.
    pub fn for_1d(problem_size: usize, device: &OpenClDeviceInfo) -> Result<Self> {
        if problem_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "workgroup problem_size must be non-zero".into(),
            }
            .into());
        }

        let pref = device.preferred_workgroup_multiple.max(1);
        let max_wg = device.max_workgroup_size.max(1);

        // Choose local_size as largest multiple of `pref` that is ≤ max_wg
        // and ≤ problem_size.
        let cap = max_wg.min(problem_size);
        let local_size = (cap / pref) * pref;
        // Ensure at least one work-item.
        let local_size = local_size.max(1);

        // Global size must be a multiple of local_size ≥ problem_size.
        let global_size = problem_size.div_ceil(local_size) * local_size;
        let num_groups = global_size / local_size;

        Ok(Self { local_size, global_size, num_groups })
    }

    /// Compute a 1-D workgroup configuration for GEMV where the problem
    /// dimension is the output vector length.
    pub fn for_gemv(n_out: usize, device: &OpenClDeviceInfo) -> Result<Self> {
        Self::for_1d(n_out, device)
    }

    /// Compute a 1-D workgroup configuration for RMSNorm where each
    /// work-group handles one row.
    pub fn for_rmsnorm(
        hidden_dim: usize,
        n_rows: usize,
        device: &OpenClDeviceInfo,
    ) -> Result<Self> {
        if hidden_dim == 0 || n_rows == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "rmsnorm dimensions must be non-zero: \
                     hidden_dim={hidden_dim}, n_rows={n_rows}"
                ),
            }
            .into());
        }

        let pref = device.preferred_workgroup_multiple.max(1);
        let max_wg = device.max_workgroup_size.max(1);

        // Local size handles elements within a row.
        let cap = max_wg.min(hidden_dim);
        let local_size = ((cap / pref) * pref).max(1);

        // One work-group per row.
        let global_size = n_rows * local_size;
        let num_groups = n_rows;

        Ok(Self { local_size, global_size, num_groups })
    }
}
