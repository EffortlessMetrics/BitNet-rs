//! Back-compat shim for SPIR-V support in `bitnet-opencl`.
//!
//! The SPIR-V compiler, validator, and cache implementation has been
//! extracted into the SRP microcrate `bitnet-spirv` and is re-exported
//! here to preserve existing `bitnet_opencl::spirv::*` imports.

pub use bitnet_spirv::*;
