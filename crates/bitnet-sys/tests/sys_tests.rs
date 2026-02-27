//! Comprehensive unit tests for `bitnet-sys` crate.
//!
//! Covers `CompileTimeLibCapabilities` (always available) and the disabled-stub
//! module (available when `ffi` feature is **not** enabled).

use bitnet_sys::CompileTimeLibCapabilities;

// ---------------------------------------------------------------------------
// CompileTimeLibCapabilities — field constructor variants
// ---------------------------------------------------------------------------

#[test]
fn caps_all_false_fields() {
    let caps =
        CompileTimeLibCapabilities { available: false, has_cuda: false, has_bitnet_shim: false };
    assert!(!caps.available);
    assert!(!caps.has_cuda);
    assert!(!caps.has_bitnet_shim);
}

#[test]
fn caps_available_only() {
    let caps =
        CompileTimeLibCapabilities { available: true, has_cuda: false, has_bitnet_shim: false };
    assert!(caps.available);
    assert!(!caps.has_cuda);
    assert!(!caps.has_bitnet_shim);
}

#[test]
fn caps_available_and_cuda() {
    let caps =
        CompileTimeLibCapabilities { available: true, has_cuda: true, has_bitnet_shim: false };
    assert!(caps.available);
    assert!(caps.has_cuda);
    assert!(!caps.has_bitnet_shim);
}

#[test]
fn caps_available_and_shim() {
    let caps =
        CompileTimeLibCapabilities { available: true, has_cuda: false, has_bitnet_shim: true };
    assert!(caps.available);
    assert!(!caps.has_cuda);
    assert!(caps.has_bitnet_shim);
}

#[test]
fn caps_all_true_fields() {
    let caps =
        CompileTimeLibCapabilities { available: true, has_cuda: true, has_bitnet_shim: true };
    assert!(caps.available);
    assert!(caps.has_cuda);
    assert!(caps.has_bitnet_shim);
}

// ---------------------------------------------------------------------------
// CompileTimeLibCapabilities — summary() exact format
// ---------------------------------------------------------------------------

#[test]
fn summary_all_false_exact() {
    let caps =
        CompileTimeLibCapabilities { available: false, has_cuda: false, has_bitnet_shim: false };
    assert_eq!(caps.summary(), "cpp=unavailable cuda=no shim=no");
}

#[test]
fn summary_all_true_exact() {
    let caps =
        CompileTimeLibCapabilities { available: true, has_cuda: true, has_bitnet_shim: true };
    assert_eq!(caps.summary(), "cpp=available cuda=yes shim=yes");
}

#[test]
fn summary_available_no_cuda_shim_yes_exact() {
    let caps =
        CompileTimeLibCapabilities { available: true, has_cuda: false, has_bitnet_shim: true };
    assert_eq!(caps.summary(), "cpp=available cuda=no shim=yes");
}

#[test]
fn summary_available_cuda_yes_shim_no_exact() {
    let caps =
        CompileTimeLibCapabilities { available: true, has_cuda: true, has_bitnet_shim: false };
    assert_eq!(caps.summary(), "cpp=available cuda=yes shim=no");
}

#[test]
fn summary_contains_cpp_key() {
    for available in [false, true] {
        let caps =
            CompileTimeLibCapabilities { available, has_cuda: false, has_bitnet_shim: false };
        assert!(caps.summary().contains("cpp="), "summary must contain 'cpp='");
    }
}

#[test]
fn summary_contains_cuda_key() {
    for has_cuda in [false, true] {
        let caps = CompileTimeLibCapabilities { available: true, has_cuda, has_bitnet_shim: false };
        assert!(caps.summary().contains("cuda="), "summary must contain 'cuda='");
    }
}

#[test]
fn summary_contains_shim_key() {
    for has_bitnet_shim in [false, true] {
        let caps = CompileTimeLibCapabilities { available: true, has_cuda: false, has_bitnet_shim };
        assert!(caps.summary().contains("shim="), "summary must contain 'shim='");
    }
}

#[test]
fn summary_cuda_token_reflects_field() {
    let yes_caps =
        CompileTimeLibCapabilities { available: true, has_cuda: true, has_bitnet_shim: false };
    assert!(yes_caps.summary().contains("cuda=yes"), "has_cuda=true must produce cuda=yes");

    let no_caps =
        CompileTimeLibCapabilities { available: false, has_cuda: false, has_bitnet_shim: false };
    assert!(no_caps.summary().contains("cuda=no"), "has_cuda=false must produce cuda=no");
}

#[test]
fn summary_shim_token_reflects_field() {
    let yes_caps =
        CompileTimeLibCapabilities { available: true, has_cuda: false, has_bitnet_shim: true };
    assert!(yes_caps.summary().contains("shim=yes"), "has_bitnet_shim=true must produce shim=yes");

    let no_caps =
        CompileTimeLibCapabilities { available: false, has_cuda: false, has_bitnet_shim: false };
    assert!(no_caps.summary().contains("shim=no"), "has_bitnet_shim=false must produce shim=no");
}

#[test]
fn summary_is_deterministic() {
    let caps =
        CompileTimeLibCapabilities { available: true, has_cuda: false, has_bitnet_shim: true };
    assert_eq!(caps.summary(), caps.summary());
}

#[test]
fn summary_is_nonempty_for_all_combinations() {
    for available in [false, true] {
        for has_cuda in [false, true] {
            for has_bitnet_shim in [false, true] {
                let caps = CompileTimeLibCapabilities { available, has_cuda, has_bitnet_shim };
                assert!(!caps.summary().is_empty(), "summary must never be empty");
            }
        }
    }
}

#[test]
fn summary_length_is_bounded() {
    for available in [false, true] {
        for has_cuda in [false, true] {
            for has_bitnet_shim in [false, true] {
                let caps = CompileTimeLibCapabilities { available, has_cuda, has_bitnet_shim };
                let s = caps.summary();
                assert!(s.len() < 128, "summary must be short; got len={}: {s}", s.len());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Clone and PartialEq
// ---------------------------------------------------------------------------

#[test]
fn clone_produces_equal_struct() {
    let caps =
        CompileTimeLibCapabilities { available: true, has_cuda: false, has_bitnet_shim: true };
    let cloned = caps.clone();
    assert_eq!(caps, cloned, "clone must equal original");
}

#[test]
fn equality_distinguishes_different_structs() {
    let a = CompileTimeLibCapabilities { available: true, has_cuda: true, has_bitnet_shim: false };
    let b = CompileTimeLibCapabilities { available: true, has_cuda: false, has_bitnet_shim: false };
    assert_ne!(a, b, "structs with different has_cuda must not be equal");
}

#[test]
fn equality_is_reflexive() {
    let caps =
        CompileTimeLibCapabilities { available: false, has_cuda: false, has_bitnet_shim: false };
    assert_eq!(caps, caps.clone(), "caps must equal a clone of itself");
}

// ---------------------------------------------------------------------------
// Debug formatting
// ---------------------------------------------------------------------------

#[test]
fn debug_format_is_nonempty() {
    let caps =
        CompileTimeLibCapabilities { available: true, has_cuda: false, has_bitnet_shim: false };
    assert!(!format!("{caps:?}").is_empty(), "Debug output must not be empty");
}

#[test]
fn debug_format_contains_field_names() {
    let caps =
        CompileTimeLibCapabilities { available: false, has_cuda: false, has_bitnet_shim: false };
    let debug = format!("{caps:?}");
    assert!(debug.contains("available"), "Debug must mention 'available': {debug}");
    assert!(debug.contains("has_cuda"), "Debug must mention 'has_cuda': {debug}");
    assert!(debug.contains("has_bitnet_shim"), "Debug must mention 'has_bitnet_shim': {debug}");
}

// ---------------------------------------------------------------------------
// from_compile_time() — structural invariants
// ---------------------------------------------------------------------------

#[test]
fn from_compile_time_has_cuda_implies_available() {
    let caps = CompileTimeLibCapabilities::from_compile_time();
    if caps.has_cuda {
        assert!(caps.available, "has_cuda must imply available at build time");
    }
}

#[test]
fn from_compile_time_has_shim_implies_available() {
    let caps = CompileTimeLibCapabilities::from_compile_time();
    if caps.has_bitnet_shim {
        assert!(caps.available, "has_bitnet_shim must imply available at build time");
    }
}

#[test]
fn from_compile_time_summary_contains_all_keys() {
    let s = CompileTimeLibCapabilities::from_compile_time().summary();
    assert!(s.contains("cpp="), "runtime summary must contain 'cpp=': {s}");
    assert!(s.contains("cuda="), "runtime summary must contain 'cuda=': {s}");
    assert!(s.contains("shim="), "runtime summary must contain 'shim=': {s}");
}

#[test]
fn from_compile_time_is_deterministic() {
    let a = CompileTimeLibCapabilities::from_compile_time();
    let b = CompileTimeLibCapabilities::from_compile_time();
    assert_eq!(a, b, "from_compile_time must be deterministic");
}

// ---------------------------------------------------------------------------
// Disabled stubs — only compiled when `ffi` feature is not enabled
// ---------------------------------------------------------------------------

#[cfg(not(feature = "ffi"))]
mod disabled_stub_tests {
    use bitnet_sys::disabled;

    #[test]
    fn stub_is_available_returns_false() {
        assert!(!disabled::is_available(), "stub is_available() must return false");
    }

    #[test]
    fn stub_version_returns_err() {
        assert!(disabled::version().is_err(), "stub version() must return Err");
    }

    #[test]
    fn stub_initialize_returns_err() {
        assert!(disabled::initialize().is_err(), "stub initialize() must return Err");
    }

    #[test]
    fn stub_load_model_returns_err() {
        assert!(
            disabled::load_model("/any/path.gguf").is_err(),
            "stub load_model() must return Err"
        );
    }

    #[test]
    fn stub_cleanup_returns_err() {
        assert!(disabled::cleanup().is_err(), "stub cleanup() must return Err");
    }

    #[test]
    fn stub_generate_returns_err() {
        let mut handle = disabled::ModelHandle;
        assert!(
            disabled::generate(&mut handle, "hello", 1).is_err(),
            "stub generate() must return Err"
        );
    }

    #[test]
    fn stub_generate_with_zero_tokens_returns_err() {
        let mut handle = disabled::ModelHandle;
        // Even with max_tokens=0, the stub must return Err (not panic).
        assert!(
            disabled::generate(&mut handle, "", 0).is_err(),
            "stub generate() with empty prompt/zero tokens must return Err"
        );
    }

    #[test]
    fn disabled_error_message_is_informative() {
        let err = disabled::version().unwrap_err();
        let msg = err.to_string();
        assert!(!msg.is_empty(), "error message must not be empty");
        // The message should help the user understand what to do.
        assert!(
            msg.to_lowercase().contains("ffi")
                || msg.to_lowercase().contains("bitnet")
                || msg.to_lowercase().contains("bindings"),
            "error message should guide the user; got: {msg}"
        );
    }

    #[test]
    fn stub_version_error_is_displayable() {
        let err = disabled::version().unwrap_err();
        // Ensure Display impl works and doesn't panic.
        let _ = format!("{err}");
        let _ = format!("{err:?}");
    }

    #[test]
    fn stub_load_model_error_is_displayable() {
        let result = disabled::load_model("irrelevant.gguf");
        assert!(result.is_err());
        // Extract error via match since ModelHandle doesn't impl Debug
        if let Err(e) = result {
            let _ = format!("{e}");
        }
    }
}
