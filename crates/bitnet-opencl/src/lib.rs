//! `OpenCL` runtime binding layer for `BitNet` GPU inference.
//!
//! Provides safe wrappers around opencl3 types with dynamic library
//! loading and graceful fallback when no `OpenCL` runtime is installed.

pub mod benchmark_utils;
pub mod runtime;

pub use runtime::{
    OpenClDeviceInfo, OpenClPlatformInfo, enumerate_gpu_devices, enumerate_platforms,
    mock_device_intel_arc, mock_device_nvidia, mock_platform, opencl_available,
};
