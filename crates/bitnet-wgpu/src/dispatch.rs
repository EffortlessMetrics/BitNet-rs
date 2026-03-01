//! Workgroup dispatch helpers and NVIDIA-tuned sizing.
//!
//! Re-exported from `bitnet-dispatch-core` so existing `bitnet-wgpu::dispatch`
//! imports remain stable.

pub use bitnet_dispatch_core::{
    DispatchConfig, DispatchEntry, DispatchRecorder, NVIDIA_WARP_SIZE, compute_dispatch_size,
    optimal_workgroup_size_nvidia,
};
