//! OpenCL-based GPU kernel provider for Intel Arc GPUs.
//!
//! This module implements the [`KernelProvider`] trait using OpenCL 3.0
//! via the `opencl3` crate, targeting Intel's Compute Runtime for Arc GPUs.

use crate::KernelProvider;
use bitnet_common::{KernelError, QuantizationType, Result};

/// OpenCL kernel provider for Intel Arc GPUs.
#[derive(Debug)]
pub struct OpenClKernel {
    // Will hold OpenCL context, queue, program etc.
    _private: (),
}

impl OpenClKernel {
    /// Attempt to create a new OpenCL kernel provider.
    ///
    /// Returns `Ok(Self)` if an OpenCL-capable Intel GPU is found,
    /// or an error if no suitable device is available.
    pub fn new() -> Result<Self> {
        // TODO: Initialize OpenCL context, find Intel GPU device
        Err(KernelError::GpuError {
            reason: "OpenCL backend not yet implemented".into(),
        }
        .into())
    }
}

impl KernelProvider for OpenClKernel {
    fn name(&self) -> &'static str {
        "opencl-intel"
    }

    fn is_available(&self) -> bool {
        false // TODO: Check OpenCL device availability
    }

    fn matmul_i2s(
        &self,
        _a: &[i8],
        _b: &[u8],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        Err(KernelError::GpuError {
            reason: "OpenCL matmul_i2s not yet implemented".into(),
        }
        .into())
    }

    fn quantize(
        &self,
        _input: &[f32],
        _output: &mut [u8],
        _scales: &mut [f32],
        _qtype: QuantizationType,
    ) -> Result<()> {
        Err(KernelError::GpuError {
            reason: "OpenCL quantize not yet implemented".into(),
        }
        .into())
    }
}
