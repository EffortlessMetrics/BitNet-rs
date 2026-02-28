//! Snapshot tests for bitnet-kernels stable API surface.
//! These tests pin the Display/Debug format of types that must remain stable.

use bitnet_kernels::KernelManager;

#[test]
fn kernel_manager_has_at_least_one_provider() {
    let mgr = KernelManager::new();
    let providers = mgr.list_available_providers();
    insta::assert_snapshot!(format!(
        "count={} has_providers={}",
        providers.len(),
        !providers.is_empty()
    ));
}

#[test]
fn fallback_kernel_name_is_stable() {
    let mgr = KernelManager::new();
    let providers = mgr.list_available_providers();
    insta::assert_snapshot!(format!(
        "has_fallback={}",
        providers.iter().any(|p| p.contains("fallback") || p.contains("cpu"))
    ));
}

#[test]
fn select_best_returns_a_provider() {
    let mgr = KernelManager::new();
    let result = mgr.select_best();
    insta::assert_snapshot!(format!("ok={}", result.is_ok()));
}

#[test]
fn selected_provider_name_non_empty_after_selection() {
    let mgr = KernelManager::new();
    let _ = mgr.select_best();
    let name = mgr.selected_provider_name();
    insta::assert_snapshot!(format!(
        "some={} non_empty={}",
        name.is_some(),
        name.map(|n| !n.is_empty()).unwrap_or(false)
    ));
}

// -- Wave 3: GPU HAL type snapshots ------------------------------------------

use bitnet_kernels::gpu_utils::GpuInfo;

#[test]
fn gpu_info_no_gpu_debug() {
    let info = GpuInfo {
        cuda: false,
        cuda_version: None,
        metal: false,
        rocm: false,
        rocm_version: None,
        opengl: false,
        wgpu: false,
    };
    insta::assert_debug_snapshot!("gpu_info_no_gpu", info);
}

#[test]
fn gpu_info_summary_no_gpu() {
    let info = GpuInfo {
        cuda: false,
        cuda_version: None,
        metal: false,
        rocm: false,
        rocm_version: None,
        opengl: false,
        wgpu: false,
    };
    insta::assert_snapshot!("gpu_info_summary_no_gpu", info.summary());
}

#[test]
fn gpu_info_cuda_only() {
    let info = GpuInfo {
        cuda: true,
        cuda_version: Some("12.4".to_string()),
        metal: false,
        rocm: false,
        rocm_version: None,
        opengl: false,
        wgpu: false,
    };
    insta::assert_debug_snapshot!("gpu_info_cuda_only", info);
}

#[test]
fn gpu_info_full_stack() {
    let info = GpuInfo {
        cuda: true,
        cuda_version: Some("12.4".to_string()),
        metal: true,
        rocm: true,
        rocm_version: Some("6.0".to_string()),
        opengl: true,
        wgpu: true,
    };
    insta::assert_debug_snapshot!("gpu_info_full_stack", info);
}

#[test]
fn gpu_info_any_available_false_when_none() {
    let info = GpuInfo {
        cuda: false,
        cuda_version: None,
        metal: false,
        rocm: false,
        rocm_version: None,
        opengl: false,
        wgpu: false,
    };
    insta::assert_snapshot!("gpu_info_any_available_none", format!("{}", info.any_available()));
}

#[test]
fn gpu_info_any_available_true_when_cuda() {
    let info = GpuInfo {
        cuda: true,
        cuda_version: None,
        metal: false,
        rocm: false,
        rocm_version: None,
        opengl: false,
        wgpu: false,
    };
    insta::assert_snapshot!("gpu_info_any_available_cuda", format!("{}", info.any_available()));
}
