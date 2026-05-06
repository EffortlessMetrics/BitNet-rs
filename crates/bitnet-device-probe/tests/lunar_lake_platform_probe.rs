use bitnet_device_probe::intel::lunar_lake::{
    IntelArc140vProbe, IntelNpuProbe, LNL258V_PROOF_STAGE_RUNTIME_DETECTED, Lnl258vCpuProbe,
    Lnl258vPlatformProbe, PlatformMemoryProbe, PlatformPowerProbe,
};
use bitnet_device_probe::runtimes::{OpenVinoDeviceProbe, OpenVinoProbe};

#[test]
#[allow(clippy::too_many_lines)]
fn lunar_lake_platform_probe_serializes_visibility_receipt_shape() {
    let probe = Lnl258vPlatformProbe {
        machine_id: "intel-258v".to_owned(),
        proof_stage: LNL258V_PROOF_STAGE_RUNTIME_DETECTED.to_owned(),
        os: "linux".to_owned(),
        os_build: Some("test-os".to_owned()),
        arch: "x86_64".to_owned(),
        cpu: Lnl258vCpuProbe {
            brand: Some("Intel Core Ultra 7 258V".to_owned()),
            cores: 8,
            threads: 8,
            p_core_count: Some(4),
            lp_e_core_count: Some(4),
            has_avx2: true,
            has_avx512: false,
            has_fma: true,
            has_sse42: true,
            scheduler_hint: Some("record topology context".to_owned()),
        },
        arc140v: IntelArc140vProbe {
            proof_stage: LNL258V_PROOF_STAGE_RUNTIME_DETECTED.to_owned(),
            requested_backend: "intel-arc-140v".to_owned(),
            selected_backend: Some("intel-arc-140v-opencl".to_owned()),
            runtime_api: Some("opencl".to_owned()),
            available: true,
            pci_device_id: Some("0x64A0".to_owned()),
            identity_evidence: vec!["opencl:Intel(R) Arc(TM) 140V Graphics".to_owned()],
            opencl_available: true,
            opencl_platform_name: Some("Intel(R) OpenCL Graphics".to_owned()),
            opencl_device_name: Some("Intel(R) Arc(TM) 140V Graphics".to_owned()),
            opencl_driver_version: Some("test-driver".to_owned()),
            level_zero_available: true,
            level_zero_devices: vec!["Intel(R) Arc(TM) 140V Graphics".to_owned()],
            openvino_gpu_visible: true,
            openvino_gpu_device: Some("GPU.0".to_owned()),
            openvino_gpu_full_name: Some("Intel(R) Arc(TM) 140V Graphics".to_owned()),
            shared_memory_bytes: Some(32 * 1024 * 1024 * 1024),
            power_mode: Some("balanced".to_owned()),
            fallback_used: false,
            failure_reason: None,
        },
        npu: IntelNpuProbe {
            proof_stage: "runtime_detected".to_owned(),
            requested_backend: "intel-npu".to_owned(),
            selected_backend: Some("intel-npu-openvino".to_owned()),
            runtime_api: Some("openvino".to_owned()),
            runtime_device: Some("NPU".to_owned()),
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            available: true,
            accel_device_present: true,
            accel_devices: vec!["/dev/accel/accel0".to_owned()],
            intel_vpu_driver_seen: true,
            driver_hint: Some("intel_vpu/ivpu evidence".to_owned()),
            openvino_runtime_available: true,
            openvino_version: Some("2026.1".to_owned()),
            openvino_available_devices: vec![
                "CPU".to_owned(),
                "GPU.0".to_owned(),
                "NPU".to_owned(),
            ],
            openvino_npu_visible: true,
            openvino_npu_full_name: Some("Intel(R) AI Boost".to_owned()),
            supported_properties: vec!["FULL_DEVICE_NAME".to_owned()],
            driver_version: Some("1.2.3".to_owned()),
            compiler_version: Some("4.5.6".to_owned()),
            total_mem_size: Some(1_048_576),
            alloc_mem_size: Some(524_288),
            max_tiles: Some(2),
            fallback_used: false,
            failure_reason: None,
        },
        openvino: OpenVinoProbe {
            runtime_available: true,
            version: Some("2026.1".to_owned()),
            available_devices: vec!["CPU".to_owned(), "GPU.0".to_owned(), "NPU".to_owned()],
            devices: vec![
                OpenVinoDeviceProbe {
                    device: "GPU.0".to_owned(),
                    full_name: Some("Intel(R) Arc(TM) 140V Graphics".to_owned()),
                    supported_properties: Vec::new(),
                    properties: Vec::new(),
                },
                OpenVinoDeviceProbe {
                    device: "NPU".to_owned(),
                    full_name: Some("Intel(R) AI Boost".to_owned()),
                    supported_properties: vec!["FULL_DEVICE_NAME".to_owned()],
                    properties: Vec::new(),
                },
            ],
            error: None,
        },
        memory: PlatformMemoryProbe {
            total_bytes: Some(32 * 1024 * 1024 * 1024),
            shared_memory_bytes: Some(32 * 1024 * 1024 * 1024),
            shared_memory: true,
        },
        power: PlatformPowerProbe {
            mode: Some("balanced".to_owned()),
            thermal_profile: Some("default".to_owned()),
            ac_power: Some(true),
        },
        fallback_used: false,
        status: LNL258V_PROOF_STAGE_RUNTIME_DETECTED.to_owned(),
        failure_reason: None,
    };

    let value = serde_json::to_value(&probe).expect("probe serializes");
    assert_eq!(value["machine_id"], "intel-258v");
    assert_eq!(value["proof_stage"], "runtime_detected");
    assert_eq!(value["fallback_used"], false);
    assert_eq!(value["cpu"]["has_avx2"], true);
    assert_eq!(value["cpu"]["has_avx512"], false);
    assert_eq!(value["arc140v"]["proof_stage"], "runtime_detected");
    assert_eq!(value["arc140v"]["selected_backend"], "intel-arc-140v-opencl");
    assert_eq!(value["arc140v"]["runtime_api"], "opencl");
    assert_eq!(value["arc140v"]["openvino_gpu_device"], "GPU.0");
    assert_eq!(value["arc140v"]["fallback_used"], false);
    assert_eq!(value["npu"]["requested_backend"], "intel-npu");
    assert_eq!(value["npu"]["selected_backend"], "intel-npu-openvino");
    assert_eq!(value["npu"]["runtime_api"], "openvino");
    assert_eq!(value["npu"]["runtime_device"], "NPU");
    assert_eq!(value["npu"]["openvino_npu_visible"], true);
    assert_eq!(value["npu"]["fallback_used"], false);
}

#[test]
fn unavailable_runtime_fields_are_false_or_null_not_claims() {
    let openvino = OpenVinoProbe::unavailable("not installed");
    assert!(!openvino.runtime_available);
    assert!(openvino.available_devices.is_empty());
    assert_eq!(openvino.version, None);
    assert!(openvino.error.is_some());
    assert!(!openvino.npu_visible());
    assert_eq!(openvino.gpu_device_token(), None);
}
