use bitnet_vulkan::VulkanShaderSource;

#[test]
fn shader_reexport_matches_kernels_module() {
    // The kernels module was emptied and VulkanShaderSource is re-exported at the crate root.
    assert_eq!(VulkanShaderSource::Matmul.name(), "matmul");
}

#[test]
fn runtime_discovery_placeholder_is_empty() {
    let devices = bitnet_vulkan::runtime::discover_devices();
    assert!(devices.is_empty());
}
