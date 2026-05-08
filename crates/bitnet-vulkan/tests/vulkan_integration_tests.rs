use bitnet_vulkan::{VulkanShaderSource, kernels};

#[test]
fn shader_reexport_matches_kernels_module() {
    assert_eq!(VulkanShaderSource::ALL, kernels::VulkanShaderSource::ALL);
    assert_eq!(VulkanShaderSource::Matmul.name(), "matmul");
}

#[test]
fn runtime_discovery_is_non_panicking() {
    let devices = bitnet_vulkan::runtime::discover_devices();

    assert!(devices.iter().all(|name| !name.trim().is_empty()));
}

#[test]
#[cfg(not(feature = "vulkan-runtime"))]
fn runtime_discovery_without_feature_is_empty() {
    let devices = bitnet_vulkan::runtime::discover_devices();

    assert!(devices.is_empty());
}
