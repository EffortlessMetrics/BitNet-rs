//! OpenCL runtime visibility wrapper for platform receipts.
#![allow(clippy::doc_markdown)]

use serde::{Deserialize, Serialize};

/// OpenCL device facts needed for hardware-lane identity receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenClRuntimeDevice {
    /// OpenCL platform name.
    pub platform_name: Option<String>,
    /// Device name.
    pub device_name: String,
    /// Device vendor.
    pub vendor: String,
    /// Driver version reported by OpenCL.
    pub driver_version: Option<String>,
    /// Whether the device is a GPU.
    pub is_gpu: bool,
}

/// OpenCL runtime visibility result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenClRuntimeProbe {
    /// Whether an OpenCL runtime was visible.
    pub runtime_available: bool,
    /// Devices reported by the runtime.
    pub devices: Vec<OpenClRuntimeDevice>,
    /// Non-fatal probe error when the runtime was absent or unusable.
    pub error: Option<String>,
}

/// Native OpenCL tiny kernel smoke result for a selected Intel GPU lane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenClTinyKernelSmoke {
    /// Whether kernel dispatch/readback executed and matched CPU expected output.
    pub passed: bool,
    /// Highest proof stage reached by this smoke.
    pub proof_stage: String,
    /// Requested hardware lane.
    pub requested_backend: String,
    /// Selected backend when Arc 140V native OpenCL was used.
    pub selected_backend: Option<String>,
    /// Runtime API used by the smoke.
    pub runtime_api: Option<String>,
    /// Runtime device name selected for execution.
    pub runtime_device: Option<String>,
    /// OpenCL platform index selected for execution.
    pub platform_index: Option<usize>,
    /// OpenCL device index selected for execution within the platform.
    pub device_index: Option<usize>,
    /// OpenCL platform name.
    pub platform_name: Option<String>,
    /// OpenCL device vendor.
    pub vendor: Option<String>,
    /// OpenCL driver version.
    pub driver_version: Option<String>,
    /// Kernel name.
    pub kernel_name: String,
    /// Input element count.
    pub input_len: usize,
    /// Output comparison tolerance.
    pub tolerance: f32,
    /// Maximum absolute error from CPU expected output.
    pub max_abs_error: Option<f32>,
    /// Mean absolute error from CPU expected output.
    pub mean_abs_error: Option<f32>,
    /// Time spent enqueueing writes and kernel work.
    pub enqueue_ms: Option<f64>,
    /// Time spent reading the result back.
    pub readback_ms: Option<f64>,
    /// Whether a native kernel executed.
    pub kernel_execution: bool,
    /// Always false; CPU fallback cannot satisfy selected Intel GPU proof.
    pub fallback_used: bool,
    /// Always false for this smoke.
    pub cpu_fallback_allowed: bool,
    /// Always false; this is not BitNet inference.
    pub bitnet_inference: bool,
    /// Always false; this is not packed QK256 decode.
    pub qk256_decode: bool,
    /// Non-fatal runtime or execution error.
    pub error: Option<String>,
}

impl OpenClRuntimeProbe {
    /// Build an unavailable OpenCL probe result.
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self { runtime_available: false, devices: Vec::new(), error: Some(reason.into()) }
    }
}

/// Probe OpenCL visibility without making an execution claim.
pub fn probe_opencl_runtime() -> OpenClRuntimeProbe {
    #[cfg(feature = "opencl")]
    {
        let result = crate::opencl::probe_opencl();
        let devices = result
            .devices
            .into_iter()
            .map(|device| {
                let is_gpu = device.is_gpu();
                OpenClRuntimeDevice {
                    platform_name: result
                        .platforms
                        .get(device.platform_index)
                        .map(|platform| platform.name.clone()),
                    device_name: device.name,
                    vendor: device.vendor,
                    driver_version: Some(device.driver_version),
                    is_gpu,
                }
            })
            .collect();

        OpenClRuntimeProbe {
            runtime_available: result.runtime_available,
            devices,
            error: result.error,
        }
    }

    #[cfg(not(feature = "opencl"))]
    {
        OpenClRuntimeProbe::unavailable("compiled without opencl feature")
    }
}

/// Run a tiny native OpenCL vector-add kernel on Arc 140V when compiled with
/// OpenCL support.
pub fn run_arc140v_opencl_tiny_kernel_smoke() -> OpenClTinyKernelSmoke {
    #[cfg(feature = "opencl")]
    {
        let raw = crate::opencl::run_intel_arc_140v_tiny_vector_add_smoke();
        let selected_backend = raw.passed.then(|| "intel-arc-140v-opencl".to_owned());
        OpenClTinyKernelSmoke {
            passed: raw.passed,
            proof_stage: raw.proof_stage,
            requested_backend: "intel-arc-140v".to_owned(),
            selected_backend,
            runtime_api: Some("opencl".to_owned()),
            runtime_device: raw.device_name,
            platform_index: raw.platform_index,
            device_index: raw.device_index,
            platform_name: raw.platform_name,
            vendor: raw.vendor,
            driver_version: raw.driver_version,
            kernel_name: raw.kernel_name,
            input_len: raw.input_len,
            tolerance: raw.tolerance,
            max_abs_error: raw.max_abs_error,
            mean_abs_error: raw.mean_abs_error,
            enqueue_ms: raw.enqueue_ms,
            readback_ms: raw.readback_ms,
            kernel_execution: raw.kernel_execution,
            fallback_used: raw.fallback_used,
            cpu_fallback_allowed: false,
            bitnet_inference: false,
            qk256_decode: false,
            error: raw.error,
        }
    }

    #[cfg(not(feature = "opencl"))]
    {
        unavailable_tiny_kernel_smoke("intel-arc-140v")
    }
}

/// Run a tiny native OpenCL vector-add kernel on Arc A770 when compiled with
/// OpenCL support.
pub fn run_a770_opencl_tiny_kernel_smoke() -> OpenClTinyKernelSmoke {
    #[cfg(feature = "opencl")]
    {
        let raw = crate::opencl::run_intel_arc_a770_tiny_vector_add_smoke();
        let selected_backend = raw.passed.then(|| "intel-arc-a770-opencl".to_owned());
        OpenClTinyKernelSmoke {
            passed: raw.passed,
            proof_stage: raw.proof_stage,
            requested_backend: "intel-arc-a770".to_owned(),
            selected_backend,
            runtime_api: Some("opencl".to_owned()),
            runtime_device: raw.device_name,
            platform_index: raw.platform_index,
            device_index: raw.device_index,
            platform_name: raw.platform_name,
            vendor: raw.vendor,
            driver_version: raw.driver_version,
            kernel_name: raw.kernel_name,
            input_len: raw.input_len,
            tolerance: raw.tolerance,
            max_abs_error: raw.max_abs_error,
            mean_abs_error: raw.mean_abs_error,
            enqueue_ms: raw.enqueue_ms,
            readback_ms: raw.readback_ms,
            kernel_execution: raw.kernel_execution,
            fallback_used: raw.fallback_used,
            cpu_fallback_allowed: false,
            bitnet_inference: false,
            qk256_decode: false,
            error: raw.error,
        }
    }

    #[cfg(not(feature = "opencl"))]
    {
        unavailable_tiny_kernel_smoke("intel-arc-a770")
    }
}

#[cfg(not(feature = "opencl"))]
fn unavailable_tiny_kernel_smoke(requested_backend: &str) -> OpenClTinyKernelSmoke {
    OpenClTinyKernelSmoke {
        passed: false,
        proof_stage: "runtime_detected".to_owned(),
        requested_backend: requested_backend.to_owned(),
        selected_backend: None,
        runtime_api: Some("opencl".to_owned()),
        runtime_device: None,
        platform_index: None,
        device_index: None,
        platform_name: None,
        vendor: None,
        driver_version: None,
        kernel_name: "tiny_vector_add".to_owned(),
        input_len: 16,
        tolerance: 1.0e-6,
        max_abs_error: None,
        mean_abs_error: None,
        enqueue_ms: None,
        readback_ms: None,
        kernel_execution: false,
        fallback_used: false,
        cpu_fallback_allowed: false,
        bitnet_inference: false,
        qk256_decode: false,
        error: Some("compiled without opencl feature".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::{run_a770_opencl_tiny_kernel_smoke, run_arc140v_opencl_tiny_kernel_smoke};

    #[test]
    fn opencl_tiny_kernel_smoke_reports_unavailable_without_feature() {
        let smoke = run_arc140v_opencl_tiny_kernel_smoke();
        assert!(!smoke.fallback_used);
        assert!(!smoke.cpu_fallback_allowed);
        assert!(!smoke.bitnet_inference);
        assert!(!smoke.qk256_decode);

        #[cfg(not(feature = "opencl"))]
        {
            assert!(!smoke.passed);
            assert!(!smoke.kernel_execution);
            assert_eq!(smoke.error.as_deref(), Some("compiled without opencl feature"));
        }
    }

    #[test]
    fn a770_opencl_tiny_kernel_smoke_reports_unavailable_without_feature() {
        let smoke = run_a770_opencl_tiny_kernel_smoke();
        assert_eq!(smoke.requested_backend, "intel-arc-a770");
        assert!(!smoke.fallback_used);
        assert!(!smoke.cpu_fallback_allowed);
        assert!(!smoke.bitnet_inference);
        assert!(!smoke.qk256_decode);

        #[cfg(not(feature = "opencl"))]
        {
            assert!(!smoke.passed);
            assert!(!smoke.kernel_execution);
            assert_eq!(smoke.error.as_deref(), Some("compiled without opencl feature"));
        }
    }
}
