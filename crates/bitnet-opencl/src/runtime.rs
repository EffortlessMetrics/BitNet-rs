//! OpenCL runtime: platform/device discovery via the OpenCL ICD loader.

/// Placeholder for OpenCL runtime discovery.
///
/// Requires the `opencl-runtime` feature and a working OpenCL ICD loader.
pub fn discover_devices() -> Vec<String> {
    // TODO: enumerate OpenCL platforms and devices via opencl3
    Vec::new()
}
