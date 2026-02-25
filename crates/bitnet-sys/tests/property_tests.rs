//! Property-based tests for `bitnet-sys` — `CompileTimeLibCapabilities` invariants.
//!
//! These tests exercise the `CompileTimeLibCapabilities` struct, which captures
//! what C++ libraries were detected at build time via symbol analysis.
//!
//! # Key invariants
//! - `has_cuda` ⇒ `available` (CUDA symbols require a library to be present)
//! - `has_bitnet_shim` ⇒ `available` (shim symbols require a library)
//! - `summary()` always contains canonical key-value tokens

use bitnet_sys::CompileTimeLibCapabilities;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Strategy: generate arbitrary CompileTimeLibCapabilities
// ---------------------------------------------------------------------------

/// Generate a logically valid (implication-respecting) capabilities struct.
fn valid_caps_strategy() -> impl Strategy<Value = CompileTimeLibCapabilities> {
    // available: bool; cuda/shim can only be true when available is true
    (any::<bool>(), any::<bool>(), any::<bool>()).prop_map(|(available, maybe_cuda, maybe_shim)| {
        CompileTimeLibCapabilities {
            available,
            has_cuda: available && maybe_cuda,
            has_bitnet_shim: available && maybe_shim,
        }
    })
}

/// Generate all 8 combinations including logically impossible ones (for negative tests).
fn any_caps_strategy() -> impl Strategy<Value = CompileTimeLibCapabilities> {
    (any::<bool>(), any::<bool>(), any::<bool>()).prop_map(
        |(available, has_cuda, has_bitnet_shim)| CompileTimeLibCapabilities {
            available,
            has_cuda,
            has_bitnet_shim,
        },
    )
}

// ---------------------------------------------------------------------------
// Property: implication invariants on valid capabilities
// ---------------------------------------------------------------------------

proptest! {
    /// `has_cuda` implies `available` for any logically valid capabilities.
    #[test]
    fn prop_has_cuda_implies_available(caps in valid_caps_strategy()) {
        if caps.has_cuda {
            prop_assert!(caps.available, "has_cuda={} but available={}", caps.has_cuda, caps.available);
        }
    }

    /// `has_bitnet_shim` implies `available` for any logically valid capabilities.
    #[test]
    fn prop_has_shim_implies_available(caps in valid_caps_strategy()) {
        if caps.has_bitnet_shim {
            prop_assert!(caps.available, "has_bitnet_shim={} but available={}", caps.has_bitnet_shim, caps.available);
        }
    }

    /// `summary()` always contains the three canonical key prefixes.
    #[test]
    fn prop_summary_contains_canonical_keys(caps in any_caps_strategy()) {
        let s = caps.summary();
        prop_assert!(s.contains("cpp="), "summary missing 'cpp=': {s}");
        prop_assert!(s.contains("cuda="), "summary missing 'cuda=': {s}");
        prop_assert!(s.contains("shim="), "summary missing 'shim=': {s}");
    }

    /// `summary()` is deterministic — calling it twice gives identical output.
    #[test]
    fn prop_summary_is_deterministic(caps in any_caps_strategy()) {
        prop_assert_eq!(caps.summary(), caps.summary());
    }

    /// `clone()` produces an identical struct.
    #[test]
    fn prop_clone_equals_original(caps in any_caps_strategy()) {
        let cloned = caps.clone();
        prop_assert_eq!(caps.available, cloned.available);
        prop_assert_eq!(caps.has_cuda, cloned.has_cuda);
        prop_assert_eq!(caps.has_bitnet_shim, cloned.has_bitnet_shim);
    }

    /// When `available=false`, `summary()` must say "cpp=unavailable".
    #[test]
    fn prop_unavailable_has_correct_summary_token(
        has_cuda in any::<bool>(),
        has_bitnet_shim in any::<bool>(),
    ) {
        let caps = CompileTimeLibCapabilities { available: false, has_cuda, has_bitnet_shim };
        let s = caps.summary();
        prop_assert!(s.contains("cpp=unavailable"), "expected 'cpp=unavailable' in: {s}");
    }

    /// When `available=true`, `summary()` must say "cpp=available".
    #[test]
    fn prop_available_has_correct_summary_token(
        has_cuda in any::<bool>(),
        has_bitnet_shim in any::<bool>(),
    ) {
        let caps = CompileTimeLibCapabilities { available: true, has_cuda, has_bitnet_shim };
        let s = caps.summary();
        prop_assert!(s.contains("cpp=available"), "expected 'cpp=available' in: {s}");
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[test]
fn from_compile_time_satisfies_implication_invariants() {
    let caps = CompileTimeLibCapabilities::from_compile_time();
    // In all builds: has_cuda and has_bitnet_shim must imply available.
    if caps.has_cuda {
        assert!(caps.available, "build-time: has_cuda implies available");
    }
    if caps.has_bitnet_shim {
        assert!(caps.available, "build-time: has_bitnet_shim implies available");
    }
}

#[test]
fn summary_all_three_keys_present_at_runtime() {
    let caps = CompileTimeLibCapabilities::from_compile_time();
    let s = caps.summary();
    assert!(s.contains("cpp="), "summary missing 'cpp=': {s}");
    assert!(s.contains("cuda="), "summary missing 'cuda=': {s}");
    assert!(s.contains("shim="), "summary missing 'shim=': {s}");
}

#[test]
fn summary_cuda_yes_only_when_available() {
    // If we have CUDA, available must be set too.
    let caps = CompileTimeLibCapabilities::from_compile_time();
    if caps.summary().contains("cuda=yes") {
        assert!(caps.available);
    }
}
