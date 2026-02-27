//! Additional invariant tests for `CompileTimeLibCapabilities`.
//!
//! These supplement `sys_tests.rs` with:
//! - `Send + Sync` bounds for the disabled-stub error type
//! - Consistency between `from_compile_time()` fields and `cfg!()` macro values
//! - Exact summary strings for all 8 field combinations (including logically impossible ones)
//! - `PartialEq` symmetry and transitivity
//! - `Debug` output contains the struct name

use bitnet_sys::CompileTimeLibCapabilities;

// ---------------------------------------------------------------------------
// Compile-time Send + Sync assertions
// ---------------------------------------------------------------------------

fn assert_send_sync<T: Send + Sync + 'static>() {}

#[test]
fn compile_time_lib_capabilities_is_send_sync() {
    assert_send_sync::<CompileTimeLibCapabilities>();
}

// ---------------------------------------------------------------------------
// from_compile_time() fields must match the underlying cfg!() macro values
// ---------------------------------------------------------------------------

#[test]
fn from_compile_time_available_matches_cfg_macro() {
    let caps = CompileTimeLibCapabilities::from_compile_time();
    assert_eq!(
        caps.available,
        cfg!(bitnet_cpp_available),
        "available field must match cfg!(bitnet_cpp_available)"
    );
}

#[test]
fn from_compile_time_has_cuda_matches_cfg_macro() {
    let caps = CompileTimeLibCapabilities::from_compile_time();
    assert_eq!(
        caps.has_cuda,
        cfg!(bitnet_cpp_has_cuda),
        "has_cuda field must match cfg!(bitnet_cpp_has_cuda)"
    );
}

#[test]
fn from_compile_time_has_shim_matches_cfg_macro() {
    let caps = CompileTimeLibCapabilities::from_compile_time();
    assert_eq!(
        caps.has_bitnet_shim,
        cfg!(bitnet_cpp_has_bitnet_shim),
        "has_bitnet_shim field must match cfg!(bitnet_cpp_has_bitnet_shim)"
    );
}

// ---------------------------------------------------------------------------
// Exact summary strings for all 8 field combinations
// (sys_tests.rs only covers 4 of 8 exactly; this covers all 8)
// ---------------------------------------------------------------------------

const ALL_SUMMARIES: &[(bool, bool, bool, &str)] = &[
    (false, false, false, "cpp=unavailable cuda=no shim=no"),
    (false, false, true, "cpp=unavailable cuda=no shim=yes"),
    (false, true, false, "cpp=unavailable cuda=yes shim=no"),
    (false, true, true, "cpp=unavailable cuda=yes shim=yes"),
    (true, false, false, "cpp=available cuda=no shim=no"),
    (true, false, true, "cpp=available cuda=no shim=yes"),
    (true, true, false, "cpp=available cuda=yes shim=no"),
    (true, true, true, "cpp=available cuda=yes shim=yes"),
];

#[test]
fn summary_exact_for_all_eight_combinations() {
    for &(available, has_cuda, has_bitnet_shim, expected) in ALL_SUMMARIES {
        let caps = CompileTimeLibCapabilities { available, has_cuda, has_bitnet_shim };
        assert_eq!(
            caps.summary(),
            expected,
            "available={available} has_cuda={has_cuda} has_bitnet_shim={has_bitnet_shim}"
        );
    }
}

// ---------------------------------------------------------------------------
// PartialEq: symmetry and transitivity
// ---------------------------------------------------------------------------

#[test]
fn partial_eq_is_symmetric() {
    let a = CompileTimeLibCapabilities { available: true, has_cuda: false, has_bitnet_shim: true };
    let b = a.clone();
    assert_eq!(a, b, "a == b (symmetric)");
    assert_eq!(b, a, "b == a (symmetric)");
}

#[test]
fn partial_eq_is_transitive() {
    let a = CompileTimeLibCapabilities { available: true, has_cuda: true, has_bitnet_shim: false };
    let b = a.clone();
    let c = a.clone();
    assert_eq!(a, b, "a == b");
    assert_eq!(b, c, "b == c");
    assert_eq!(a, c, "a == c (transitive)");
}

#[test]
fn partial_eq_differs_on_has_bitnet_shim() {
    let a = CompileTimeLibCapabilities { available: true, has_cuda: false, has_bitnet_shim: true };
    let b = CompileTimeLibCapabilities { available: true, has_cuda: false, has_bitnet_shim: false };
    assert_ne!(a, b, "structs differing only in has_bitnet_shim must not be equal");
}

// ---------------------------------------------------------------------------
// Debug output contains struct name
// ---------------------------------------------------------------------------

#[test]
fn debug_output_contains_struct_name() {
    let caps =
        CompileTimeLibCapabilities { available: true, has_cuda: true, has_bitnet_shim: true };
    let debug = format!("{caps:?}");
    assert!(
        debug.contains("CompileTimeLibCapabilities"),
        "Debug output must contain the struct name; got: {debug}"
    );
}

// ---------------------------------------------------------------------------
// Disabled-stub error type — only available when the `ffi` feature is off
// ---------------------------------------------------------------------------

#[cfg(not(feature = "ffi"))]
mod disabled_error_tests {
    use bitnet_sys::disabled::DisabledError;

    fn assert_send_sync<T: Send + Sync + 'static>() {}

    #[test]
    fn disabled_error_is_send_sync() {
        assert_send_sync::<DisabledError>();
    }

    #[test]
    fn disabled_error_debug_contains_type_name() {
        let err = DisabledError;
        let debug = format!("{err:?}");
        assert!(
            debug.contains("DisabledError"),
            "Debug output must contain 'DisabledError'; got: {debug}"
        );
    }

    #[test]
    fn disabled_error_display_equals_to_string() {
        let err = DisabledError;
        assert_eq!(format!("{err}"), err.to_string(), "Display and to_string() must agree");
    }

    #[test]
    fn disabled_error_display_mentions_ffi_or_features() {
        let msg = DisabledError.to_string();
        let lower = msg.to_lowercase();
        assert!(
            lower.contains("ffi") || lower.contains("--features") || lower.contains("feature"),
            "error message should guide the user toward the ffi feature; got: {msg:?}"
        );
    }
}
