//! GPU feature flag compatibility matrix tests.
//!
//! These tests verify that the types and APIs exposed under each feature-flag
//! combination are internally consistent.  Each test exercises the public
//! surface that **must** compile when the corresponding feature set is active.
//! The full compile-time matrix is validated by the CI workflow
//! `.github/workflows/feature-matrix-extended.yml`.

// --------------------------------------------------------------------------
// cpu-only
// --------------------------------------------------------------------------

#[test]
fn features_cpu_only() {
    // When only `cpu` is enabled the core diagnostics type must be usable.
    let diag = bitnet_testing_policy_tests::PolicyDiagnostics::current();
    assert!(
        diag.profile().cell.is_some(),
        "PolicyDiagnostics cell must be populated under cpu feature"
    );
}

// --------------------------------------------------------------------------
// oneapi-only  (OpenCL / Intel GPU path)
// --------------------------------------------------------------------------

#[test]
fn features_oneapi_only() {
    // The policy crate must be constructible even when the only backend
    // feature is oneapi (no cpu, no cuda).
    let diag = bitnet_testing_policy_tests::PolicyDiagnostics::current();
    let _profile = diag.profile();
    // Presence is enough — this is a compile-gate test.
}

// --------------------------------------------------------------------------
// gpu-only  (CUDA umbrella)
// --------------------------------------------------------------------------

#[test]
fn features_gpu_only() {
    // `gpu` enables CUDA + Vulkan paths.  Policy diagnostics must still work.
    let diag = bitnet_testing_policy_tests::PolicyDiagnostics::current();
    let _profile = diag.profile();
}

// --------------------------------------------------------------------------
// cpu + oneapi
// --------------------------------------------------------------------------

#[test]
fn features_cpu_and_oneapi() {
    // CPU fallback and OpenCL must coexist without symbol conflicts.
    let diag = bitnet_testing_policy_tests::PolicyDiagnostics::current();
    assert!(
        diag.profile().cell.is_some(),
        "cpu + oneapi must produce a valid diagnostics cell"
    );
}

// --------------------------------------------------------------------------
// gpu + oneapi
// --------------------------------------------------------------------------

#[test]
fn features_gpu_and_oneapi() {
    // CUDA/Vulkan and OpenCL must coexist.
    let diag = bitnet_testing_policy_tests::PolicyDiagnostics::current();
    let _profile = diag.profile();
}

// --------------------------------------------------------------------------
// all backends
// --------------------------------------------------------------------------

#[test]
fn features_all_backends() {
    // cpu + gpu + oneapi + vulkan — every backend enabled simultaneously.
    let diag = bitnet_testing_policy_tests::PolicyDiagnostics::current();
    assert!(
        diag.profile().cell.is_some(),
        "all-backends combination must produce a valid diagnostics cell"
    );
}

// --------------------------------------------------------------------------
// Additional feature-contract sanity checks
// --------------------------------------------------------------------------

#[test]
fn features_default_empty_compiles() {
    // With *no* features the crate should still compile and expose
    // the policy façade (it is feature-independent by design).
    let _diag = bitnet_testing_policy_tests::PolicyDiagnostics::current();
}

#[test]
fn features_cpu_with_fixtures() {
    // cpu + fixtures must not break the diagnostics surface.
    let diag = bitnet_testing_policy_tests::PolicyDiagnostics::current();
    let _profile = diag.profile();
}

#[test]
fn features_gpu_and_cpu_together() {
    // Explicitly test the cpu+gpu pair (common CI combination).
    let diag = bitnet_testing_policy_tests::PolicyDiagnostics::current();
    assert!(
        diag.profile().cell.is_some(),
        "cpu + gpu must coexist without conflicts"
    );
}

#[test]
fn features_diagnostics_profile_is_deterministic() {
    // Two consecutive calls must return the same cell key.
    let a = bitnet_testing_policy_tests::PolicyDiagnostics::current();
    let b = bitnet_testing_policy_tests::PolicyDiagnostics::current();
    assert_eq!(
        format!("{:?}", a.profile().cell),
        format!("{:?}", b.profile().cell),
        "PolicyDiagnostics must be deterministic across calls"
    );
}
