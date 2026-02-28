//! Metal device capabilities query.

/// Information about an Apple Silicon GPU.
#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    pub name: String,
    pub registry_id: u64,
    pub max_threads_per_threadgroup: u64,
    pub max_buffer_length: u64,
    pub has_unified_memory: bool,
    pub recommended_max_working_set_size: u64,
}

/// Query Metal device capabilities.
///
/// On non-macOS platforms this always returns `None`.
#[cfg(target_os = "macos")]
pub fn query_device() -> Option<MetalDeviceInfo> {
    let device = metal::Device::system_default()?;
    Some(MetalDeviceInfo {
        name: device.name().to_string(),
        registry_id: device.registry_id(),
        max_threads_per_threadgroup: device.max_threads_per_threadgroup().width,
        max_buffer_length: device.max_buffer_length(),
        has_unified_memory: device.has_unified_memory(),
        recommended_max_working_set_size: device.recommended_max_working_set_size(),
    })
}

#[cfg(not(target_os = "macos"))]
pub fn query_device() -> Option<MetalDeviceInfo> {
    None
}
