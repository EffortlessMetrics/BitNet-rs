//! Edge-case integration tests for `bitnet_kernels::gpu_utils` module.
//!
//! Uses `BITNET_GPU_FAKE` env var (via `temp_env`) to exercise GpuInfo
//! construction, summary, preflight, and various backend combinations
//! deterministically, without requiring real GPU hardware.

use bitnet_kernels::gpu_utils::{GpuInfo, get_gpu_info, gpu_available, preflight_check};
use serial_test::serial;

// =========================================================================
// GpuInfo construction and accessors
// =========================================================================

#[test]
fn gpu_info_all_false() {
    let info = GpuInfo {
        cuda: false,
        cuda_version: None,
        metal: false,
        rocm: false,
        rocm_version: None,
        opengl: false,
        wgpu: false,
    };
    assert!(!info.any_available());
    assert_eq!(info.summary(), "No GPU backends available");
}

#[test]
fn gpu_info_cuda_only() {
    let info = GpuInfo {
        cuda: true,
        cuda_version: None,
        metal: false,
        rocm: false,
        rocm_version: None,
        opengl: false,
        wgpu: false,
    };
    assert!(info.any_available());
    assert!(info.summary().contains("CUDA"));
}

#[test]
fn gpu_info_cuda_with_version() {
    let info = GpuInfo {
        cuda: true,
        cuda_version: Some("12.4".into()),
        metal: false,
        rocm: false,
        rocm_version: None,
        opengl: false,
        wgpu: false,
    };
    assert!(info.summary().contains("CUDA 12.4"));
}

#[test]
fn gpu_info_metal_only() {
    let info = GpuInfo {
        cuda: false,
        cuda_version: None,
        metal: true,
        rocm: false,
        rocm_version: None,
        opengl: false,
        wgpu: false,
    };
    assert!(info.any_available());
    assert!(info.summary().contains("Metal"));
}

#[test]
fn gpu_info_rocm_with_version() {
    let info = GpuInfo {
        cuda: false,
        cuda_version: None,
        metal: false,
        rocm: true,
        rocm_version: Some("6.2.1".into()),
        opengl: false,
        wgpu: false,
    };
    assert!(info.any_available());
    assert!(info.summary().contains("ROCm 6.2.1"));
}

#[test]
fn gpu_info_all_backends() {
    let info = GpuInfo {
        cuda: true,
        cuda_version: Some("12.0".into()),
        metal: true,
        rocm: true,
        rocm_version: Some("6.0".into()),
        opengl: true,
        wgpu: true,
    };
    let s = info.summary();
    assert!(s.contains("CUDA 12.0"));
    assert!(s.contains("Metal"));
    assert!(s.contains("ROCm 6.0"));
    assert!(s.contains("OpenGL"));
    assert!(s.contains("WebGPU"));
}

// =========================================================================
// BITNET_GPU_FAKE env var
// =========================================================================

#[test]
#[serial(bitnet_env)]
fn fake_gpu_cuda() {
    temp_env::with_var("BITNET_GPU_FAKE", Some("cuda"), || {
        let info = get_gpu_info();
        assert!(info.cuda);
        assert!(info.wgpu); // wgpu is true when cuda is
        assert!(gpu_available());
    });
}

#[test]
#[serial(bitnet_env)]
fn fake_gpu_metal() {
    temp_env::with_var("BITNET_GPU_FAKE", Some("metal"), || {
        let info = get_gpu_info();
        assert!(info.metal);
        assert!(info.wgpu);
        assert!(gpu_available());
    });
}

#[test]
#[serial(bitnet_env)]
fn fake_gpu_rocm() {
    temp_env::with_var("BITNET_GPU_FAKE", Some("rocm"), || {
        let info = get_gpu_info();
        assert!(info.rocm);
        assert!(info.wgpu);
    });
}

#[test]
#[serial(bitnet_env)]
fn fake_gpu_opengl() {
    temp_env::with_var("BITNET_GPU_FAKE", Some("opengl"), || {
        let info = get_gpu_info();
        assert!(info.opengl);
        assert!(info.wgpu);
    });
}

#[test]
#[serial(bitnet_env)]
fn fake_gpu_gl_alias() {
    temp_env::with_var("BITNET_GPU_FAKE", Some("gl"), || {
        let info = get_gpu_info();
        assert!(info.opengl);
        assert!(info.wgpu);
    });
}

#[test]
#[serial(bitnet_env)]
fn fake_gpu_wgpu() {
    temp_env::with_var("BITNET_GPU_FAKE", Some("wgpu"), || {
        let info = get_gpu_info();
        assert!(info.wgpu);
        assert!(!info.cuda);
        assert!(!info.metal);
        assert!(!info.rocm);
    });
}

#[test]
#[serial(bitnet_env)]
fn fake_gpu_multiple() {
    temp_env::with_var("BITNET_GPU_FAKE", Some("cuda,metal"), || {
        let info = get_gpu_info();
        assert!(info.cuda);
        assert!(info.metal);
        assert!(info.wgpu);
    });
}

#[test]
#[serial(bitnet_env)]
fn fake_gpu_case_insensitive() {
    temp_env::with_var("BITNET_GPU_FAKE", Some("CUDA"), || {
        let info = get_gpu_info();
        assert!(info.cuda);
    });
}

#[test]
#[serial(bitnet_env)]
fn fake_gpu_unrecognized_value() {
    temp_env::with_var("BITNET_GPU_FAKE", Some("unknown"), || {
        let info = get_gpu_info();
        assert!(!info.cuda);
        assert!(!info.metal);
        assert!(!info.rocm);
        assert!(!info.opengl);
        assert!(!info.wgpu);
        assert!(!info.any_available());
    });
}

// =========================================================================
// preflight_check
// =========================================================================

#[test]
#[serial(bitnet_env)]
fn preflight_check_with_fake_cuda() {
    temp_env::with_var("BITNET_GPU_FAKE", Some("cuda"), || {
        assert!(preflight_check().is_ok());
    });
}

#[test]
#[serial(bitnet_env)]
fn preflight_check_with_no_fake_and_no_real_gpu() {
    // On machines without GPU, preflight should fail (or succeed if OpenGL is found).
    // Just verify it doesn't panic.
    temp_env::with_var("BITNET_GPU_FAKE", None::<&str>, || {
        let _ = preflight_check();
    });
}

// =========================================================================
// BITNET_STRICT_NO_FAKE_GPU
// =========================================================================

#[test]
#[serial(bitnet_env)]
#[should_panic(expected = "strict mode forbids fake GPU")]
fn strict_mode_rejects_fake_gpu() {
    temp_env::with_vars(
        [("BITNET_GPU_FAKE", Some("cuda")), ("BITNET_STRICT_NO_FAKE_GPU", Some("1"))],
        || {
            let _ = get_gpu_info();
        },
    );
}

#[test]
#[serial(bitnet_env)]
#[should_panic(expected = "strict mode forbids fake GPU")]
fn strict_mode_rejects_fake_gpu_with_normalized_truthy_value() {
    temp_env::with_vars(
        [("BITNET_GPU_FAKE", Some("cuda")), ("BITNET_STRICT_NO_FAKE_GPU", Some(" TRUE "))],
        || {
            let _ = get_gpu_info();
        },
    );
}
