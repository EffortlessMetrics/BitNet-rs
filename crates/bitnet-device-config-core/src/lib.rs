//! Core device configuration parsing and runtime resolution.

mod backend;
mod config;
mod parse;
mod profile;
mod resolve;

pub use config::DeviceConfig;
pub use profile::{
    DeviceProfileContract, DeviceProfileLabel, DeviceProfileStoragePolicy,
    DeviceProfileUnsupportedClaim, ThermalPolicy,
};

#[cfg(test)]
mod tests {
    use super::DeviceConfig;

    fn parse_device(input: &str) -> Option<DeviceConfig> {
        input.parse::<DeviceConfig>().ok()
    }

    #[test]
    fn parses_supported_aliases() {
        assert_eq!(parse_device("cpu"), Some(DeviceConfig::Cpu));
        assert_eq!(parse_device("auto"), Some(DeviceConfig::Auto));
        assert_eq!(parse_device("gpu"), Some(DeviceConfig::Gpu(0)));
        assert_eq!(parse_device("cuda:2"), Some(DeviceConfig::Gpu(2)));
        assert_eq!(parse_device("vulkan:3"), Some(DeviceConfig::Gpu(3)));
        assert_eq!(parse_device("npu"), Some(DeviceConfig::IntelNpu(0)));
        assert_eq!(parse_device("intel-npu:1"), Some(DeviceConfig::IntelNpu(1)));
        assert_eq!(parse_device("openvino-npu"), Some(DeviceConfig::OpenVinoNpu));
        assert_eq!(
            parse_device("nvidia-rtx-5070-ti-cuda"),
            Some(DeviceConfig::NvidiaRtx5070TiCuda)
        );
        assert_eq!(
            parse_device("nvidia-rtx-5070-ti-wgpu"),
            Some(DeviceConfig::NvidiaRtx5070TiWgpu)
        );
        assert_eq!(parse_device("intel-a770-opencl"), Some(DeviceConfig::IntelA770OpenCl));
        assert_eq!(parse_device("metal"), Some(DeviceConfig::Metal));
        assert_eq!(parse_device("mpsgraph"), Some(DeviceConfig::MpsGraph));
        assert_eq!(parse_device("apple-m4-metal"), Some(DeviceConfig::AppleM4Metal));
        assert_eq!(parse_device("apple-m4-mpsgraph"), Some(DeviceConfig::AppleM4MpsGraph));
        assert_eq!(parse_device("apple-m4-cpu-neon"), Some(DeviceConfig::AppleM4CpuNeon));
        assert_eq!(parse_device("apple-m3-air-metal"), Some(DeviceConfig::AppleM3AirMetal));
        assert_eq!(parse_device("apple-m3-air-mpsgraph"), Some(DeviceConfig::AppleM3AirMpsGraph));
        assert_eq!(parse_device("apple-m3-air-cpu-neon"), Some(DeviceConfig::AppleM3AirCpuNeon));
    }

    #[test]
    fn rejects_invalid_values() {
        assert!("unknown".parse::<DeviceConfig>().is_err());
        assert!("gpu:".parse::<DeviceConfig>().is_err());
        assert!("gpu:abc".parse::<DeviceConfig>().is_err());
        assert!("npu:".parse::<DeviceConfig>().is_err());
        assert!("intel-npu:abc".parse::<DeviceConfig>().is_err());
    }

    #[test]
    fn apple_backend_labels_do_not_alias() {
        let metal = DeviceConfig::Metal;
        let apple_metal = DeviceConfig::AppleM4Metal;
        let mpsgraph = DeviceConfig::MpsGraph;
        let apple_mpsgraph = DeviceConfig::AppleM4MpsGraph;
        let apple_cpu = DeviceConfig::AppleM4CpuNeon;
        assert_eq!(parse_device("apple-m3-air-cpu-neon"), Some(DeviceConfig::AppleM3AirCpuNeon));
        let apple_m3_air_metal = DeviceConfig::AppleM3AirMetal;
        let apple_m3_air_mpsgraph = DeviceConfig::AppleM3AirMpsGraph;
        let apple_m3_air_cpu = DeviceConfig::AppleM3AirCpuNeon;

        assert_eq!(metal.backend_label(), "metal");
        assert_eq!(apple_metal.backend_label(), "apple-m4-metal");
        assert_eq!(mpsgraph.backend_label(), "mpsgraph");
        assert_eq!(apple_mpsgraph.backend_label(), "apple-m4-mpsgraph");
        assert_eq!(apple_cpu.backend_label(), "apple-m4-cpu-neon");
        assert_eq!(apple_m3_air_metal.backend_label(), "apple-m3-air-metal");
        assert_eq!(apple_m3_air_mpsgraph.backend_label(), "apple-m3-air-mpsgraph");
        assert_eq!(apple_m3_air_cpu.backend_label(), "apple-m3-air-cpu-neon");
        assert_ne!(apple_m3_air_metal.backend_label(), apple_metal.backend_label());
        assert_ne!(apple_m3_air_mpsgraph.backend_label(), apple_mpsgraph.backend_label());
        assert_ne!(apple_m3_air_cpu.backend_label(), apple_cpu.backend_label());
        assert_eq!(apple_m3_air_metal.backend_request().to_string(), "apple-m3-air-metal");
        assert_eq!(apple_m3_air_metal.resolve(), bitnet_common::Device::Cpu);
        assert_eq!(apple_m3_air_mpsgraph.resolve(), bitnet_common::Device::Cpu);
        assert_eq!(apple_m3_air_cpu.resolve(), bitnet_common::Device::Cpu);
    }

    #[test]
    fn apple_m3_air_profile_contract_is_shared_across_lane_labels() {
        let metal = DeviceConfig::AppleM3AirMetal.device_profile_contract();
        let mpsgraph = DeviceConfig::AppleM3AirMpsGraph.device_profile_contract();
        let cpu = DeviceConfig::AppleM3AirCpuNeon.device_profile_contract();
        assert!(metal.is_some());
        assert_eq!(metal, mpsgraph);
        assert_eq!(metal, cpu);

        let Some(contract) = metal else {
            return;
        };
        assert_eq!(contract.profile_id, "apple-m3-macbook-air");
        assert_eq!(contract.soc_family, "Apple M3");
        assert_eq!(contract.thermal_policy, super::ThermalPolicy::FanlessMobile);
        assert_eq!(contract.storage.cache_root_required, true);
        assert_eq!(contract.storage.large_artifact_sweep_allowed, true);
        assert_eq!(contract.storage.model_binaries_committed, false);
        assert_eq!(
            contract.label("apple-m3-air-metal").map(|label| label.execution_available),
            Some(false)
        );
        assert_eq!(
            contract.label("apple-m3-air-mpsgraph").map(|label| label.execution_available),
            Some(false)
        );
        assert_eq!(
            contract.label("apple-m3-air-cpu-neon").map(|label| label.execution_available),
            Some(true)
        );
        assert!(contract.rejects(super::DeviceProfileUnsupportedClaim::MetalModelInference));
        assert!(contract.rejects(super::DeviceProfileUnsupportedClaim::MpsGraphModelInference));
        assert!(contract.rejects(super::DeviceProfileUnsupportedClaim::NeuralEngineExecution));
        assert!(contract.rejects(super::DeviceProfileUnsupportedClaim::Qk256AppleSilicon));
        assert!(DeviceConfig::AppleM4CpuNeon.device_profile_contract().is_none());
    }

    #[test]
    fn rtx_5070_ti_backend_labels_do_not_alias_legacy_gpu_labels() {
        let generic_gpu = DeviceConfig::Gpu(0);
        let generic_cuda = DeviceConfig::Gpu(0);
        let rtx_cuda = DeviceConfig::NvidiaRtx5070TiCuda;
        let rtx_wgpu = DeviceConfig::NvidiaRtx5070TiWgpu;

        assert_eq!(rtx_cuda.backend_label(), "nvidia-rtx-5070-ti-cuda");
        assert_eq!(rtx_wgpu.backend_label(), "nvidia-rtx-5070-ti-wgpu");
        assert_ne!(rtx_cuda.backend_label(), generic_gpu.backend_label());
        assert_ne!(rtx_cuda.backend_label(), generic_cuda.backend_label());
        assert_ne!(rtx_wgpu.backend_label(), generic_gpu.backend_label());
        assert_ne!(rtx_wgpu.backend_label(), generic_cuda.backend_label());
    }

    #[test]
    fn intel_a770_opencl_backend_label_does_not_alias_generic_gpu_or_oneapi() {
        let generic_gpu = DeviceConfig::Gpu(0);
        let generic_opencl_alias = DeviceConfig::Gpu(0);
        let a770 = DeviceConfig::IntelA770OpenCl;

        assert_eq!(a770.backend_label(), "intel-a770-opencl");
        assert_eq!(a770.backend_request().to_string(), "intel-a770-opencl");
        assert_eq!(a770.resolve(), bitnet_common::Device::OpenCL(0));
        assert_ne!(a770.backend_label(), generic_gpu.backend_label());
        assert_ne!(a770.backend_label(), generic_opencl_alias.backend_label());
        assert_ne!(a770.backend_request(), bitnet_common::BackendRequest::OneApi);
    }

    #[test]
    fn intel_npu_backend_labels_do_not_alias_gpu_or_cpu_labels() {
        let generic_gpu = DeviceConfig::Gpu(0);
        let generic_cuda = DeviceConfig::Gpu(0);
        let cpu = DeviceConfig::Cpu;
        let npu = DeviceConfig::IntelNpu(0);
        let indexed_npu = DeviceConfig::IntelNpu(1);
        let openvino_npu = DeviceConfig::OpenVinoNpu;

        assert_eq!(npu.backend_label(), "intel-npu");
        assert_eq!(indexed_npu.backend_label(), "intel-npu:1");
        assert_eq!(openvino_npu.backend_label(), "openvino-npu");
        assert_eq!(npu.resolve(), bitnet_common::Device::Npu);
        assert_eq!(indexed_npu.resolve(), bitnet_common::Device::Npu);
        assert_eq!(openvino_npu.resolve(), bitnet_common::Device::Npu);
        assert_ne!(npu.backend_label(), generic_gpu.backend_label());
        assert_ne!(npu.backend_label(), generic_cuda.backend_label());
        assert_ne!(npu.backend_label(), cpu.backend_label());
    }

    #[test]
    fn default_is_auto() {
        assert_eq!(DeviceConfig::default(), DeviceConfig::Auto);
    }

    #[test]
    fn resolve_returns_cpu_for_cpu_variants() {
        use bitnet_common::Device;
        assert_eq!(DeviceConfig::Cpu.resolve(), Device::Cpu);
        assert_eq!(DeviceConfig::NvidiaRtx5070TiWgpu.resolve(), Device::Cpu);
        assert_eq!(DeviceConfig::MpsGraph.resolve(), Device::Cpu);
        assert_eq!(DeviceConfig::AppleM4MpsGraph.resolve(), Device::Cpu);
        assert_eq!(DeviceConfig::AppleM4CpuNeon.resolve(), Device::Cpu);
        assert_eq!(DeviceConfig::AppleM3AirCpuNeon.resolve(), Device::Cpu);
    }

    #[test]
    fn resolve_returns_cuda_for_gpu_and_rtx_cuda() {
        use bitnet_common::Device;
        assert_eq!(DeviceConfig::Gpu(0).resolve(), Device::Cuda(0));
        assert_eq!(DeviceConfig::Gpu(5).resolve(), Device::Cuda(5));
        assert_eq!(DeviceConfig::NvidiaRtx5070TiCuda.resolve(), Device::Cuda(0));
    }

    #[test]
    fn resolve_returns_metal_for_metal_and_apple_m4_metal_only() {
        use bitnet_common::Device;
        assert_eq!(DeviceConfig::Metal.resolve(), Device::Metal);
        assert_eq!(DeviceConfig::AppleM4Metal.resolve(), Device::Metal);
        // AppleM3AirMetal is intentionally identity-only and resolves to CPU.
        assert_eq!(DeviceConfig::AppleM3AirMetal.resolve(), Device::Cpu);
    }

    #[test]
    fn parsing_is_case_insensitive_on_alias_keys() {
        assert_eq!(parse_device("CPU"), Some(DeviceConfig::Cpu));
        assert_eq!(parse_device("Cuda"), Some(DeviceConfig::Gpu(0)));
        assert_eq!(parse_device("METAL"), Some(DeviceConfig::Metal));
        assert_eq!(parse_device("Apple-M4-Metal"), Some(DeviceConfig::AppleM4Metal));
        assert_eq!(parse_device("GPU:7"), Some(DeviceConfig::Gpu(7)));
    }

    #[test]
    fn parsing_rejects_empty_and_whitespace() {
        assert!("".parse::<DeviceConfig>().is_err());
        // Whitespace inputs are not normalized by FromStr.
        assert!(" cpu".parse::<DeviceConfig>().is_err());
        assert!("cpu ".parse::<DeviceConfig>().is_err());
    }

    #[test]
    fn parsing_rejects_negative_or_overflow_device_ids() {
        for alias in ["gpu", "cuda", "vulkan", "opencl", "ocl", "npu", "intel-npu"] {
            let input = format!("{alias}:-1");
            assert!(parse_device(&input).is_none(), "{input} should be rejected");
        }
        // usize overflow on 64-bit; this is larger than u64::MAX so always parses error.
        assert!("gpu:99999999999999999999999".parse::<DeviceConfig>().is_err());
    }

    #[test]
    fn parsing_preserves_indexed_device_ids() {
        assert_eq!(parse_device("gpu:42"), Some(DeviceConfig::Gpu(42)));
        assert_eq!(parse_device("intel-npu:3"), Some(DeviceConfig::IntelNpu(3)));
        assert_eq!(parse_device("npu:7"), Some(DeviceConfig::IntelNpu(7)));
    }

    #[test]
    fn parsing_gpu_aliases_share_zero_default_index() {
        for alias in ["gpu", "cuda", "vulkan", "opencl", "ocl"] {
            assert_eq!(
                parse_device(alias),
                Some(DeviceConfig::Gpu(0)),
                "alias {alias} should map to Gpu(0)",
            );
        }
    }

    #[test]
    fn backend_label_for_indexed_gpu_delegates_to_backend_request() {
        // Backend label for Gpu(N) is not the indexed string; it falls through
        // to the BackendRequest Display (which represents the kind only).
        let gpu_indexed = DeviceConfig::Gpu(3);
        assert_eq!(gpu_indexed.backend_label(), gpu_indexed.backend_request().to_string());
    }

    #[test]
    fn serde_round_trip_preserves_variants() {
        let cases = [
            DeviceConfig::Auto,
            DeviceConfig::Cpu,
            DeviceConfig::Gpu(0),
            DeviceConfig::Gpu(7),
            DeviceConfig::IntelNpu(0),
            DeviceConfig::IntelNpu(2),
            DeviceConfig::OpenVinoNpu,
            DeviceConfig::NvidiaRtx5070TiCuda,
            DeviceConfig::NvidiaRtx5070TiWgpu,
            DeviceConfig::IntelA770OpenCl,
            DeviceConfig::Metal,
            DeviceConfig::MpsGraph,
            DeviceConfig::AppleM4Metal,
            DeviceConfig::AppleM4MpsGraph,
            DeviceConfig::AppleM4CpuNeon,
            DeviceConfig::AppleM3AirMetal,
            DeviceConfig::AppleM3AirMpsGraph,
            DeviceConfig::AppleM3AirCpuNeon,
        ];
        for cfg in cases {
            let json = serde_json::to_string(&cfg);
            assert!(json.is_ok(), "serialize failed for {cfg:?}: {json:?}");
            if let Ok(json) = json {
                let round = serde_json::from_str::<DeviceConfig>(&json);
                assert_eq!(round.ok(), Some(cfg));
            }
        }
    }

    #[test]
    fn serde_wire_shape_is_stable_for_indexed_variants() {
        assert_eq!(
            serde_json::to_string(&DeviceConfig::Gpu(7)).ok().as_deref(),
            Some(r#"{"Gpu":7}"#)
        );
        assert_eq!(
            serde_json::to_string(&DeviceConfig::IntelNpu(2)).ok().as_deref(),
            Some(r#"{"IntelNpu":2}"#),
        );
    }
}
