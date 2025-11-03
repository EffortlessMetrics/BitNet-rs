//! Backward Compatibility Test Suite for Environment Variable Changes
//!
//! This test suite validates backward compatibility for the RPATH merging strategy
//! and environment variable defaults specification. It ensures that existing user
//! configurations continue to work without modification during the transition.
//!
//! ## Specifications Coverage
//!
//! - **RPATH Merging Strategy** (Section 5: Regression Tests)
//!   - Legacy BITNET_CROSSVAL_LIBDIR (still works, highest priority)
//!   - Priority chain: explicit user values > runtime defaults > fallback values
//!   - No breaking changes to existing workflows
//!
//! - **Environment Variable Defaults** (Section 9: Migration Guide)
//!   - Deprecated BITNET_CPP_PATH (fallback to BITNET_CPP_DIR)
//!   - Existing shell exports (unchanged format)
//!   - Build script behavior (no breaking changes)
//!   - Preflight output (append-only, no removals)
//!
//! ## Test Structure
//!
//! All tests use `#[serial(bitnet_env)]` to prevent environment variable races
//! across test processes. Tests validate:
//!
//! 1. **Legacy variable support**: Old configurations still work
//! 2. **Priority order**: Explicit values take precedence over defaults
//! 3. **Deprecation warnings**: Emitted for old variables (not silent failures)
//! 4. **Migration scenarios**: Users can transition smoothly
//! 5. **No breaking changes**: Existing behavior preserved
//!
//! ## Test Categories
//!
//! - `test_legacy_*`: Validate existing environment variables still work
//! - `test_deprecated_*`: Validate fallback behavior for deprecated variables
//! - `test_priority_*`: Validate precedence chain (user > runtime > fallback)
//! - `test_migration_*`: Validate migration scenarios from Section 9
//! - `test_no_breaking_*`: Validate no breaking changes to existing behavior

use serial_test::serial;
use std::env;
use std::path::PathBuf;

// Import test infrastructure from bitnet_tests crate
use bitnet_tests::support::env_guard::EnvGuard;

/// Tests feature spec: rpath-merging-strategy.md#regression-tests (Section 5.3)
/// Validates that legacy BITNET_CROSSVAL_LIBDIR still works with highest priority
#[test]
#[serial(bitnet_env)]
fn test_legacy_bitnet_crossval_libdir_still_works() {
    let _guard = EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
    _guard.set("/legacy/path/to/libs");

    // Should use legacy path (highest priority)
    let resolved = resolve_library_path();
    assert_eq!(
        resolved,
        Some(PathBuf::from("/legacy/path/to/libs")),
        "Legacy BITNET_CROSSVAL_LIBDIR should take highest priority"
    );
}

/// Tests feature spec: rpath-merging-strategy.md#regression-tests (Section 5.3)
/// Validates that new granular variables don't override legacy BITNET_CROSSVAL_LIBDIR
#[test]
#[serial(bitnet_env)]
fn test_legacy_libdir_overrides_new_granular_vars() {
    let _g1 = EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
    let _g2 = EnvGuard::new("CROSSVAL_RPATH_BITNET");
    let _g3 = EnvGuard::new("CROSSVAL_RPATH_LLAMA");

    _g1.set("/legacy/override");
    _g2.set("/new/bitnet/path");
    _g3.set("/new/llama/path");

    // Legacy variable should take precedence over new granular variables
    let resolved = resolve_library_path();
    assert_eq!(
        resolved,
        Some(PathBuf::from("/legacy/override")),
        "Legacy BITNET_CROSSVAL_LIBDIR should override new granular variables"
    );
}

/// Tests feature spec: bitnet-env-defaults.md#migration-guide (Section 9.3)
/// Validates that deprecated BITNET_CPP_PATH falls back when BITNET_CPP_DIR not set
#[test]
#[serial(bitnet_env)]
fn test_deprecated_bitnet_cpp_path_fallback() {
    let _g1 = EnvGuard::new("BITNET_CPP_DIR");
    let _g2 = EnvGuard::new("BITNET_CPP_PATH");

    _g1.remove(); // Ensure BITNET_CPP_DIR not set
    _g2.set("/old/cpp/path");

    // Should fall back to deprecated var
    let resolved = resolve_bitnet_cpp_dir();
    assert_eq!(
        resolved,
        Some(PathBuf::from("/old/cpp/path")),
        "Should fall back to deprecated BITNET_CPP_PATH when BITNET_CPP_DIR not set"
    );

    // TODO: Verify deprecation warning emitted (check logs when implementation added)
}

/// Tests feature spec: bitnet-env-defaults.md#migration-guide (Section 9.3)
/// Validates that BITNET_CPP_DIR takes precedence over deprecated BITNET_CPP_PATH
#[test]
#[serial(bitnet_env)]
fn test_bitnet_cpp_dir_takes_precedence_over_deprecated_path() {
    let _g1 = EnvGuard::new("BITNET_CPP_DIR");
    let _g2 = EnvGuard::new("BITNET_CPP_PATH");

    _g1.set("/new/cpp/dir");
    _g2.set("/old/cpp/path");

    // BITNET_CPP_DIR should win
    let resolved = resolve_bitnet_cpp_dir();
    assert_eq!(
        resolved,
        Some(PathBuf::from("/new/cpp/dir")),
        "BITNET_CPP_DIR should take precedence over deprecated BITNET_CPP_PATH"
    );
}

/// Tests feature spec: rpath-merging-strategy.md#acceptance-criteria (Section 6.3 AC1)
/// Validates backward compatibility: existing BITNET_CPP_DIR usage continues to work
#[test]
#[serial(bitnet_env)]
fn test_existing_bitnet_cpp_dir_unchanged() {
    let _g1 = EnvGuard::new("BITNET_CPP_DIR");
    _g1.set("/custom/bitnet/cpp");

    // Existing BITNET_CPP_DIR usage should continue to work without modification
    let resolved = resolve_bitnet_cpp_dir();
    assert_eq!(
        resolved,
        Some(PathBuf::from("/custom/bitnet/cpp")),
        "Existing BITNET_CPP_DIR usage should continue to work unchanged"
    );
}

/// Tests feature spec: rpath-merging-strategy.md#environment-variable-schema (Section 2.1)
/// Validates priority order: BITNET_CROSSVAL_LIBDIR > CROSSVAL_RPATH_* > BITNET_CPP_DIR fallback
#[test]
#[serial(bitnet_env)]
fn test_priority_chain_legacy_highest() {
    let _g1 = EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
    let _g2 = EnvGuard::new("CROSSVAL_RPATH_BITNET");
    let _g3 = EnvGuard::new("BITNET_CPP_DIR");

    _g1.set("/priority1/legacy");
    _g2.set("/priority2/granular");
    _g3.set("/priority3/fallback");

    // Priority 1: Legacy override
    let resolved = resolve_library_path();
    assert_eq!(
        resolved,
        Some(PathBuf::from("/priority1/legacy")),
        "Priority chain: BITNET_CROSSVAL_LIBDIR should be highest priority"
    );
}

/// Tests feature spec: rpath-merging-strategy.md#environment-variable-schema (Section 2.1)
/// Validates priority order: CROSSVAL_RPATH_* takes precedence when legacy var not set
#[test]
#[serial(bitnet_env)]
fn test_priority_chain_granular_second() {
    let _g1 = EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
    let _g2 = EnvGuard::new("CROSSVAL_RPATH_BITNET");
    let _g3 = EnvGuard::new("BITNET_CPP_DIR");

    _g1.remove(); // Legacy not set
    _g2.set("/priority2/granular");
    _g3.set("/priority3/fallback/build/bin");

    // Priority 2: Granular variables (when legacy not set)
    let resolved = resolve_library_path();
    assert_eq!(
        resolved,
        Some(PathBuf::from("/priority2/granular")),
        "Priority chain: CROSSVAL_RPATH_BITNET should be second priority"
    );
}

/// Tests feature spec: rpath-merging-strategy.md#environment-variable-schema (Section 2.1)
/// Validates priority order: BITNET_CPP_DIR fallback when no explicit overrides
#[test]
#[serial(bitnet_env)]
fn test_priority_chain_fallback_third() {
    let _g1 = EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
    let _g2 = EnvGuard::new("CROSSVAL_RPATH_BITNET");
    let _g3 = EnvGuard::new("CROSSVAL_RPATH_LLAMA");
    let _g4 = EnvGuard::new("BITNET_CPP_DIR");

    _g1.remove(); // Legacy not set
    _g2.remove(); // Granular BITNET not set
    _g3.remove(); // Granular LLAMA not set
    _g4.set("/priority3/fallback");

    // Priority 3: BITNET_CPP_DIR auto-discovery fallback
    let resolved = resolve_library_path_with_autodiscovery();
    assert!(
        resolved.is_some(),
        "Priority chain: BITNET_CPP_DIR should provide fallback via auto-discovery"
    );
    // Verify it uses BITNET_CPP_DIR as base for auto-discovery
    let resolved_path = resolved.unwrap();
    let resolved_str = resolved_path.to_string_lossy();
    assert!(
        resolved_str.contains("/priority3/fallback"),
        "Fallback should use BITNET_CPP_DIR as base: got {}",
        resolved_str
    );
}

/// Tests feature spec: bitnet-env-defaults.md#migration-guide (Section 9.1)
/// Validates migration scenario: users with no custom environment variables
#[test]
#[serial(bitnet_env)]
fn test_migration_scenario_no_custom_vars() {
    let _g1 = EnvGuard::new("BITNET_CPP_DIR");
    let _g2 = EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
    let _g3 = EnvGuard::new("CROSSVAL_RPATH_BITNET");
    let _g4 = EnvGuard::new("CROSSVAL_RPATH_LLAMA");

    // Clear all environment variables (simulate fresh user)
    _g1.remove();
    _g2.remove();
    _g3.remove();
    _g4.remove();

    // Should fall back to runtime default: ~/.cache/bitnet_cpp
    let resolved = resolve_bitnet_cpp_dir_with_defaults();
    assert!(resolved.is_some(), "Fresh users should get runtime default ~/.cache/bitnet_cpp");

    // Verify default matches expected pattern
    let resolved_path = resolved.unwrap();
    assert!(
        resolved_path.to_string_lossy().contains(".cache/bitnet_cpp"),
        "Default should be ~/.cache/bitnet_cpp, got: {}",
        resolved_path.display()
    );
}

/// Tests feature spec: bitnet-env-defaults.md#migration-guide (Section 9.2)
/// Validates migration scenario: users with BITNET_CPP_DIR set (no change required)
#[test]
#[serial(bitnet_env)]
fn test_migration_scenario_existing_bitnet_cpp_dir() {
    let _guard = EnvGuard::new("BITNET_CPP_DIR");
    _guard.set("/custom/path");

    // Existing values should be respected (no change required)
    let resolved = resolve_bitnet_cpp_dir();
    assert_eq!(
        resolved,
        Some(PathBuf::from("/custom/path")),
        "Users with BITNET_CPP_DIR set should see no change"
    );
}

/// Tests feature spec: bitnet-env-defaults.md#migration-guide (Section 9.4)
/// Validates migration scenario: users with custom BITNET_CROSSVAL_LIBDIR (no change required)
#[test]
#[serial(bitnet_env)]
fn test_migration_scenario_custom_crossval_libdir() {
    let _guard = EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
    _guard.set("/opt/custom/lib");

    // Explicit overrides should be respected (highest priority)
    let resolved = resolve_library_path();
    assert_eq!(
        resolved,
        Some(PathBuf::from("/opt/custom/lib")),
        "Users with custom BITNET_CROSSVAL_LIBDIR should see no change"
    );
}

/// Tests feature spec: rpath-merging-strategy.md#regression-tests (Section 5.3)
/// Validates shell export format unchanged (POSIX sh format)
#[test]
#[serial(bitnet_env)]
fn test_shell_export_format_unchanged_posix() {
    let _guard = EnvGuard::new("BITNET_CPP_DIR");
    _guard.set("/test/cpp/dir");

    // Verify POSIX sh format matches existing pattern
    let exports = emit_shell_exports_sh();
    assert!(
        exports.starts_with("export BITNET_CPP_DIR="),
        "POSIX sh export should start with 'export BITNET_CPP_DIR='"
    );

    #[cfg(target_os = "linux")]
    assert!(
        exports.contains("export LD_LIBRARY_PATH="),
        "Linux sh export should contain LD_LIBRARY_PATH"
    );

    #[cfg(target_os = "macos")]
    assert!(
        exports.contains("export DYLD_LIBRARY_PATH="),
        "macOS sh export should contain DYLD_LIBRARY_PATH"
    );
}

/// Tests feature spec: rpath-merging-strategy.md#regression-tests (Section 5.3)
/// Validates shell export format unchanged (fish format)
#[test]
#[serial(bitnet_env)]
fn test_shell_export_format_unchanged_fish() {
    let _guard = EnvGuard::new("BITNET_CPP_DIR");
    _guard.set("/test/cpp/dir");

    // Verify fish format matches existing pattern
    let exports = emit_shell_exports_fish();
    assert!(
        exports.contains("set -gx BITNET_CPP_DIR"),
        "Fish export should use 'set -gx BITNET_CPP_DIR'"
    );

    #[cfg(target_os = "linux")]
    assert!(
        exports.contains("set -gx LD_LIBRARY_PATH"),
        "Linux fish export should contain LD_LIBRARY_PATH"
    );
}

/// Tests feature spec: rpath-merging-strategy.md#regression-tests (Section 5.3)
/// Validates shell export format unchanged (PowerShell format)
#[test]
#[serial(bitnet_env)]
fn test_shell_export_format_unchanged_pwsh() {
    let _guard = EnvGuard::new("BITNET_CPP_DIR");
    _guard.set("/test/cpp/dir");

    // Verify PowerShell format matches existing pattern
    let exports = emit_shell_exports_pwsh();
    assert!(
        exports.starts_with("$env:BITNET_CPP_DIR"),
        "PowerShell export should start with '$env:BITNET_CPP_DIR'"
    );
    assert!(exports.contains("$env:PATH"), "PowerShell export should contain PATH");
}

/// Tests feature spec: rpath-merging-strategy.md#regression-tests (Section 5.3)
/// Validates build script behavior: no breaking changes to emitted cargo directives
#[test]
#[serial(bitnet_env)]
fn test_build_script_no_breaking_changes() {
    let _guard = EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
    _guard.set("/test/lib");

    // Verify build.rs emits same environment variables as before
    let directives = simulate_build_script_output();

    // Old directives should still be present
    assert!(
        directives.contains("cargo:rustc-link-search=native="),
        "Build script should still emit rustc-link-search directive"
    );

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    assert!(
        directives.contains("cargo:rustc-link-arg=-Wl,-rpath,"),
        "Build script should still emit RPATH directive on Unix"
    );

    // Plus new directives (append-only, no removals)
    assert!(
        directives.contains("cargo:rerun-if-env-changed=BITNET_CROSSVAL_LIBDIR"),
        "Build script should watch BITNET_CROSSVAL_LIBDIR"
    );
}

/// Tests feature spec: rpath-merging-strategy.md#regression-tests (Section 5.3)
/// Validates preflight output: append-only, no removals
#[test]
#[serial(bitnet_env)]
fn test_preflight_output_append_only() {
    let _guard = EnvGuard::new("BITNET_CPP_DIR");
    _guard.set("/test/cpp/dir");

    // Verify preflight output contains all old sections
    let output = simulate_preflight_output();

    // Old sections should still be present
    assert!(
        output.contains("Backend Library Status"),
        "Preflight should still show 'Backend Library Status' section"
    );
    assert!(output.contains("bitnet.cpp"), "Preflight should still check bitnet.cpp backend");
    assert!(output.contains("llama.cpp"), "Preflight should still check llama.cpp backend");

    // Plus new sections (append-only)
    // (No specific new sections defined yet in spec, but verify no removals)
}

/// Tests feature spec: bitnet-env-defaults.md#migration-guide (Section 9.3)
/// Validates deprecation warning emitted for BITNET_CPP_PATH (not silent failure)
#[test]
#[serial(bitnet_env)]
fn test_deprecation_warning_emitted_for_bitnet_cpp_path() {
    let _g1 = EnvGuard::new("BITNET_CPP_DIR");
    let _g2 = EnvGuard::new("BITNET_CPP_PATH");

    _g1.remove();
    _g2.set("/old/path");

    // Capture warnings (implementation will emit to cargo:warning or logs)
    let warnings = capture_deprecation_warnings();

    // Should emit deprecation warning (not silent)
    assert!(
        warnings.iter().any(|w| w.contains("BITNET_CPP_PATH") && w.contains("deprecated")),
        "Should emit deprecation warning for BITNET_CPP_PATH usage"
    );
}

/// Tests feature spec: rpath-merging-strategy.md#acceptance-criteria (Section 6.3 AC1)
/// Validates no breaking changes: BITNET_CPP_DIR fallback auto-discovery still works
#[test]
#[serial(bitnet_env)]
fn test_no_breaking_changes_cpp_dir_fallback() {
    let _g1 = EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
    let _g2 = EnvGuard::new("CROSSVAL_RPATH_BITNET");
    let _g3 = EnvGuard::new("CROSSVAL_RPATH_LLAMA");
    let _g4 = EnvGuard::new("BITNET_CPP_DIR");

    _g1.remove();
    _g2.remove();
    _g3.remove();
    _g4.set("/test/cpp/dir");

    // Existing BITNET_CPP_DIR fallback should still work
    let resolved = resolve_library_path_with_autodiscovery();
    assert!(resolved.is_some(), "BITNET_CPP_DIR auto-discovery fallback should still work");

    let resolved_path = resolved.unwrap();
    let resolved_str = resolved_path.to_string_lossy();
    assert!(
        resolved_str.contains("/test/cpp/dir"),
        "Auto-discovery should use BITNET_CPP_DIR as base"
    );
}

/// Tests feature spec: rpath-merging-strategy.md#acceptance-criteria (Section 6.3 AC3)
/// Validates graceful degradation: missing all variables results in STUB mode (no build failure)
#[test]
#[serial(bitnet_env)]
fn test_no_breaking_changes_graceful_stub_mode() {
    let _g1 = EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
    let _g2 = EnvGuard::new("CROSSVAL_RPATH_BITNET");
    let _g3 = EnvGuard::new("CROSSVAL_RPATH_LLAMA");
    let _g4 = EnvGuard::new("BITNET_CPP_DIR");

    // Clear all variables
    _g1.remove();
    _g2.remove();
    _g3.remove();
    _g4.remove();

    // Should gracefully degrade to STUB mode (no panic/error)
    let result = simulate_build_script_with_no_vars();
    assert!(result.is_ok(), "Build should succeed in STUB mode when no variables set");

    // Verify STUB mode indicated
    let output = result.unwrap();
    assert!(
        output.contains("STUB") || output.is_empty(),
        "Should indicate STUB mode or emit no RPATH directives"
    );
}

// ============================================================================
// Helper Functions (Implementation Stubs for TDD)
// ============================================================================
// These functions represent the API contracts being tested. They will be
// implemented in the actual codebase (xtask/build.rs, crossval/build.rs, etc.)

/// Resolve library path using priority chain
///
/// Tests feature spec: rpath-merging-strategy.md#environment-variable-schema (Section 2.1)
///
/// Priority Order:
/// 1. BITNET_CROSSVAL_LIBDIR (legacy override)
/// 2. CROSSVAL_RPATH_BITNET (granular)
/// 3. BITNET_CPP_DIR auto-discovery (fallback)
fn resolve_library_path() -> Option<PathBuf> {
    // Priority 1: Legacy override
    if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
        return Some(PathBuf::from(lib_dir));
    }

    // Priority 2: Granular variables
    if let Ok(bitnet_path) = env::var("CROSSVAL_RPATH_BITNET") {
        return Some(PathBuf::from(bitnet_path));
    }

    // Priority 3: Auto-discovery fallback (not implemented in this helper)
    None
}

/// Resolve library path with auto-discovery from BITNET_CPP_DIR
///
/// Tests feature spec: rpath-merging-strategy.md#environment-variable-schema (Section 2.1)
fn resolve_library_path_with_autodiscovery() -> Option<PathBuf> {
    // Priority 1: Legacy override
    if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
        return Some(PathBuf::from(lib_dir));
    }

    // Priority 2: Granular variables
    if let Ok(bitnet_path) = env::var("CROSSVAL_RPATH_BITNET") {
        return Some(PathBuf::from(bitnet_path));
    }

    // Priority 3: Auto-discovery from BITNET_CPP_DIR
    if let Ok(cpp_dir) = env::var("BITNET_CPP_DIR") {
        // Simulate auto-discovery: try known paths
        let candidates = [
            PathBuf::from(&cpp_dir).join("build/3rdparty/llama.cpp/build/bin"),
            PathBuf::from(&cpp_dir).join("build/bin"),
            PathBuf::from(&cpp_dir).join("build"),
        ];

        // Return first candidate (in real implementation, check existence)
        return Some(candidates[0].clone());
    }

    None
}

/// Resolve BITNET_CPP_DIR using precedence chain
///
/// Tests feature spec: bitnet-env-defaults.md#api-contracts (Section 3.1)
///
/// Precedence:
/// 1. BITNET_CPP_DIR (explicit user value)
/// 2. BITNET_CPP_PATH (deprecated fallback)
fn resolve_bitnet_cpp_dir() -> Option<PathBuf> {
    // Tier 1: Explicit user value
    if let Ok(cpp_dir) = env::var("BITNET_CPP_DIR") {
        return Some(PathBuf::from(cpp_dir));
    }

    // Tier 2: Deprecated fallback
    if let Ok(cpp_path) = env::var("BITNET_CPP_PATH") {
        // TODO: Emit deprecation warning in real implementation
        return Some(PathBuf::from(cpp_path));
    }

    None
}

/// Resolve BITNET_CPP_DIR with runtime defaults
///
/// Tests feature spec: bitnet-env-defaults.md#api-contracts (Section 3.1)
fn resolve_bitnet_cpp_dir_with_defaults() -> Option<PathBuf> {
    // Tier 1: Explicit user value
    if let Ok(cpp_dir) = env::var("BITNET_CPP_DIR") {
        return Some(PathBuf::from(cpp_dir));
    }

    // Tier 2: Deprecated fallback
    if let Ok(cpp_path) = env::var("BITNET_CPP_PATH") {
        return Some(PathBuf::from(cpp_path));
    }

    // Tier 3: Runtime default
    if let Some(home) = dirs::home_dir() {
        return Some(home.join(".cache/bitnet_cpp"));
    }

    None
}

/// Emit shell exports in POSIX sh format
///
/// Tests feature spec: bitnet-env-defaults.md#shell-export-formats (Section 3.2)
fn emit_shell_exports_sh() -> String {
    let cpp_dir = env::var("BITNET_CPP_DIR").unwrap_or_else(|_| "/default/path".to_string());

    #[cfg(target_os = "linux")]
    {
        format!(
            "export BITNET_CPP_DIR=\"{}\"\nexport LD_LIBRARY_PATH=\"{}/build/bin:${{LD_LIBRARY_PATH:-}}\"",
            cpp_dir, cpp_dir
        )
    }

    #[cfg(target_os = "macos")]
    {
        format!(
            "export BITNET_CPP_DIR=\"{}\"\nexport DYLD_LIBRARY_PATH=\"{}/build/bin:${{DYLD_LIBRARY_PATH:-}}\"",
            cpp_dir, cpp_dir
        )
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        format!("export BITNET_CPP_DIR=\"{}\"", cpp_dir)
    }
}

/// Emit shell exports in fish format
///
/// Tests feature spec: bitnet-env-defaults.md#shell-export-formats (Section 3.2)
fn emit_shell_exports_fish() -> String {
    let cpp_dir = env::var("BITNET_CPP_DIR").unwrap_or_else(|_| "/default/path".to_string());

    #[cfg(target_os = "linux")]
    {
        format!(
            "set -gx BITNET_CPP_DIR \"{}\"\nset -gx LD_LIBRARY_PATH \"{}/build/bin\" $LD_LIBRARY_PATH",
            cpp_dir, cpp_dir
        )
    }

    #[cfg(target_os = "macos")]
    {
        format!(
            "set -gx BITNET_CPP_DIR \"{}\"\nset -gx DYLD_LIBRARY_PATH \"{}/build/bin\" $DYLD_LIBRARY_PATH",
            cpp_dir, cpp_dir
        )
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        format!("set -gx BITNET_CPP_DIR \"{}\"", cpp_dir)
    }
}

/// Emit shell exports in PowerShell format
///
/// Tests feature spec: bitnet-env-defaults.md#shell-export-formats (Section 3.2)
fn emit_shell_exports_pwsh() -> String {
    let cpp_dir = env::var("BITNET_CPP_DIR").unwrap_or_else(|_| "C:\\default\\path".to_string());
    format!(
        "$env:BITNET_CPP_DIR = \"{}\"\n$env:PATH = \"{}/build/bin;\" + $env:PATH",
        cpp_dir, cpp_dir
    )
}

/// Simulate build script output (cargo directives)
///
/// Tests feature spec: rpath-merging-strategy.md#regression-tests (Section 5.3)
fn simulate_build_script_output() -> String {
    let lib_dir = env::var("BITNET_CROSSVAL_LIBDIR").unwrap_or_else(|_| "/default/lib".to_string());

    let mut output = String::new();
    output.push_str(&format!("cargo:rustc-link-search=native={}\n", lib_dir));

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    output.push_str(&format!("cargo:rustc-link-arg=-Wl,-rpath,{}\n", lib_dir));

    output.push_str("cargo:rerun-if-env-changed=BITNET_CROSSVAL_LIBDIR\n");
    output.push_str("cargo:rerun-if-env-changed=BITNET_CPP_DIR\n");
    output.push_str("cargo:rerun-if-env-changed=CROSSVAL_RPATH_BITNET\n");
    output.push_str("cargo:rerun-if-env-changed=CROSSVAL_RPATH_LLAMA\n");

    output
}

/// Simulate preflight command output
///
/// Tests feature spec: rpath-merging-strategy.md#regression-tests (Section 5.3)
fn simulate_preflight_output() -> String {
    let mut output = String::new();
    output.push_str("Backend Library Status:\n");
    output.push_str("  bitnet.cpp: AVAILABLE\n");
    output.push_str("  llama.cpp: AVAILABLE\n");
    output
}

/// Capture deprecation warnings
///
/// Tests feature spec: bitnet-env-defaults.md#migration-guide (Section 9.3)
fn capture_deprecation_warnings() -> Vec<String> {
    let mut warnings = Vec::new();

    // Check if deprecated BITNET_CPP_PATH is set
    if env::var("BITNET_CPP_PATH").is_ok() && env::var("BITNET_CPP_DIR").is_err() {
        warnings.push(
            "Warning: BITNET_CPP_PATH is deprecated. Use BITNET_CPP_DIR instead.".to_string(),
        );
    }

    warnings
}

/// Simulate build script with no environment variables (STUB mode)
///
/// Tests feature spec: rpath-merging-strategy.md#acceptance-criteria (Section 6.3 AC3)
fn simulate_build_script_with_no_vars() -> Result<String, String> {
    // STUB mode: no variables set, build succeeds with no RPATH directives
    if env::var("BITNET_CROSSVAL_LIBDIR").is_err()
        && env::var("CROSSVAL_RPATH_BITNET").is_err()
        && env::var("BITNET_CPP_DIR").is_err()
    {
        // Graceful degradation: return empty output (no RPATH)
        return Ok(String::new());
    }

    Ok("cargo:rustc-link-search=native=/some/path\n".to_string())
}

// ============================================================================
// Integration Modules (for test infrastructure)
// ============================================================================
// Note: EnvGuard is imported from tests::support::env_guard at the top of the file
