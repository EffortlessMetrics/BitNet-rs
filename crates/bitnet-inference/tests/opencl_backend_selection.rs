//! Integration smoke test for the OpenCL / OneAPI backend wiring.

#[cfg(feature = "oneapi")]
mod oneapi_tests {
    use bitnet_kernels::device_features;

    #[test]
    fn opencl_backend_compiled() {
        // When oneapi feature is enabled, the compile-time flag must be true.
        assert!(
            device_features::oneapi_compiled(),
            "oneapi should be compiled when feature is enabled"
        );
    }

    #[test]
    fn device_capabilities_include_oneapi() {
        let caps = device_features::current_kernel_capabilities();
        assert!(caps.oneapi_compiled, "KernelCapabilities should report oneapi_compiled=true");
    }

    #[test]
    fn device_token_maps_to_opencl() {
        use bitnet_inference::npu::map_device_token;

        let dev = map_device_token("oneapi").expect("oneapi token should map");
        assert!(
            matches!(dev, bitnet_common::Device::OpenCL(_)),
            "oneapi token should produce Device::OpenCL"
        );

        let dev2 = map_device_token("opencl").expect("opencl token should map");
        assert!(
            matches!(dev2, bitnet_common::Device::OpenCL(_)),
            "opencl token should produce Device::OpenCL"
        );
    }
}

#[cfg(not(feature = "oneapi"))]
mod no_oneapi {
    use bitnet_kernels::device_features;

    #[test]
    fn oneapi_not_compiled_without_feature() {
        assert!(
            !device_features::oneapi_compiled(),
            "oneapi should NOT be compiled when feature is disabled"
        );
    }

    #[test]
    fn oneapi_runtime_false_without_feature() {
        assert!(
            !device_features::oneapi_available_runtime(),
            "oneapi runtime should be false when feature is disabled"
        );
    }
}
