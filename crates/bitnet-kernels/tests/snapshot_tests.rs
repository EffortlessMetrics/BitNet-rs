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
    // The fallback (cpu) provider must always be present
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
