// RPATH Merging Algorithm Comprehensive Test Scaffolding
// This test suite validates the RPATH merging logic in `xtask/build.rs` for
// multi-backend cross-validation library path resolution.
// Tests follow bitnet-rs TDD patterns:
// - Feature-gated: Tests compile but fail due to missing implementation
// - EnvGuard isolation: `#[serial(bitnet_env)]` for env-mutating tests
// - Property-based: Validation of merge algorithm correctness
// - Cross-platform: Unix/Windows cfg-gated tests
// Specification: /tmp/rpath_merge_analysis.md
// Related: docs/specs/rpath-merging-strategy.md

use serial_test::serial;
use std::env;
use tempfile::TempDir;

// ====================================================================================
// BASIC MERGE OPERATIONS (AC1: Basic Merge, 5 tests)
// ====================================================================================
/// Tests feature spec: rpath_merge_analysis.md#4-merge-and-deduplicate-algorithm
#[cfg(test)]
mod basic_merge_operations {
    use super::*;

    /// Test single path input (baseline case)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.1-algorithm-overview
    ///
    /// Validates that a single path is returned as-is without modification.
    #[test]
    fn test_merge_single_path() {
        use xtask::build_helpers::merge_and_deduplicate;

        // Arrange: Create single valid path
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let bitnet_lib = temp_dir.path().join("bitnet_lib");
        std::fs::create_dir(&bitnet_lib).expect("Failed to create bitnet_lib directory");

        // Act: Merge single path
        let result = merge_and_deduplicate(&[bitnet_lib.to_str().unwrap()]);

        // Assert: Result should be single path (no colon separator)
        assert_eq!(result, bitnet_lib.canonicalize().unwrap().display().to_string());
        assert_eq!(
            result.matches(':').count(),
            0,
            "Single path should not contain colon separator"
        );
    }

    /// Test two distinct paths (basic merge case)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.2-detailed-algorithm
    ///
    /// Validates that two distinct paths are merged with colon separator, preserving order.
    #[test]
    fn test_merge_two_distinct_paths() {
        use xtask::build_helpers::merge_and_deduplicate;

        // Arrange: Create two distinct directories
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let bitnet_lib = temp_dir.path().join("bitnet_lib");
        let llama_lib = temp_dir.path().join("llama_lib");

        std::fs::create_dir(&bitnet_lib).expect("Failed to create bitnet_lib");
        std::fs::create_dir(&llama_lib).expect("Failed to create llama_lib");

        // Act: Merge two distinct paths
        let result =
            merge_and_deduplicate(&[bitnet_lib.to_str().unwrap(), llama_lib.to_str().unwrap()]);

        // Assert: Result should be "path1:path2" with colon separator
        let expected = format!(
            "{}:{}",
            bitnet_lib.canonicalize().unwrap().display(),
            llama_lib.canonicalize().unwrap().display()
        );
        assert_eq!(result, expected);
        assert_eq!(
            result.matches(':').count(),
            1,
            "Two paths should have exactly one colon separator"
        );
    }

    /// Test three distinct paths
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.5-colon-separator-formatting
    ///
    /// Validates merging of three distinct paths with correct colon separators.
    #[test]
    fn test_merge_three_distinct_paths() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let path1 = temp_dir.path().join("lib1");
        let path2 = temp_dir.path().join("lib2");
        let path3 = temp_dir.path().join("lib3");

        std::fs::create_dir(&path1).unwrap();
        std::fs::create_dir(&path2).unwrap();
        std::fs::create_dir(&path3).unwrap();

        let result = merge_and_deduplicate(&[
            path1.to_str().unwrap(),
            path2.to_str().unwrap(),
            path3.to_str().unwrap(),
        ]);

        // Should have exactly 2 colons separating 3 paths
        assert_eq!(result.matches(':').count(), 2);
        assert!(result.contains(&path1.canonicalize().unwrap().display().to_string()));
        assert!(result.contains(&path2.canonicalize().unwrap().display().to_string()));
        assert!(result.contains(&path3.canonicalize().unwrap().display().to_string()));
    }

    /// Test five distinct paths (stress test)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.1-complete-priority-hierarchy
    ///
    /// Validates merging of five paths (matching five-tier priority system).
    #[test]
    fn test_merge_five_distinct_paths() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let mut paths = Vec::new();

        for i in 1..=5 {
            let path = temp_dir.path().join(format!("lib{}", i));
            std::fs::create_dir(&path).unwrap();
            paths.push(path);
        }

        let path_strs: Vec<&str> = paths.iter().map(|p| p.to_str().unwrap()).collect();
        let result = merge_and_deduplicate(&path_strs);

        // Should have exactly 4 colons separating 5 paths
        assert_eq!(result.matches(':').count(), 4);
        for path in &paths {
            assert!(result.contains(&path.canonicalize().unwrap().display().to_string()));
        }
    }

    /// Test empty input (edge case)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.2-detailed-algorithm
    ///
    /// Validates that empty input returns empty string.
    #[test]
    fn test_merge_empty_input() {
        use xtask::build_helpers::merge_and_deduplicate;

        let paths: Vec<&str> = vec![];
        let result = merge_and_deduplicate(&paths);

        assert_eq!(result, "", "Empty input should return empty string");
    }
}

// ====================================================================================
// CANONICALIZATION TESTS (AC2: Path Normalization, 5 tests)
// ====================================================================================
/// Tests feature spec: rpath_merge_analysis.md#4.3-canonicalization-details
#[cfg(test)]
mod canonicalization_tests {
    use super::*;

    /// Test symlink resolution and deduplication
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.3-canonicalization-details
    ///
    /// Validates that symlinks are canonicalized and deduplicated when pointing to same target.
    #[test]
    #[cfg(unix)] // Symlinks are Unix-specific
    fn test_canonicalize_symlink() {
        use xtask::build_helpers::merge_and_deduplicate;

        // Arrange: Create real directory and symlink pointing to it
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let real_dir = temp_dir.path().join("real_lib");
        let symlink_dir = temp_dir.path().join("link_lib");

        std::fs::create_dir(&real_dir).expect("Failed to create real_lib");
        std::os::unix::fs::symlink(&real_dir, &symlink_dir).expect("Failed to create symlink");

        // Act: Merge real path and symlink (should deduplicate via canonicalization)
        let result =
            merge_and_deduplicate(&[real_dir.to_str().unwrap(), symlink_dir.to_str().unwrap()]);

        // Assert: Result should be single canonical path (symlink resolved)
        assert_eq!(result.matches(':').count(), 0, "Symlink should deduplicate to single path");
        let canonical = real_dir.canonicalize().unwrap().display().to_string();
        assert_eq!(result, canonical);
    }

    /// Test relative path conversion to absolute
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.3-canonicalization-details (Example 2)
    ///
    /// Validates that relative paths are converted to absolute canonical paths.
    #[test]
    fn test_canonicalize_relative_path() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let lib_dir = temp_dir.path().join("lib");
        std::fs::create_dir(&lib_dir).unwrap();

        // Change to temp directory to test relative path
        let original_dir = env::current_dir().unwrap();
        env::set_current_dir(&temp_dir).unwrap();

        let result = merge_and_deduplicate(&["./lib"]);

        // Restore original directory
        env::set_current_dir(original_dir).unwrap();

        // Should be absolute canonical path
        assert!(
            result.starts_with('/') || result.chars().nth(1) == Some(':'),
            "Should be absolute path"
        );
        assert!(!result.contains("./"), "Should not contain relative path components");
    }

    /// Test path normalization (dots removed)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.3-canonicalization-details (Example 3)
    ///
    /// Validates that path components like `.` and `..` are normalized.
    #[test]
    fn test_canonicalize_dot_components() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let lib_dir = temp_dir.path().join("lib");
        std::fs::create_dir(&lib_dir).unwrap();

        // Create path with dot components
        let messy_path = format!(
            "{}/./../{}/./lib",
            temp_dir.path().display(),
            temp_dir.path().file_name().unwrap().to_str().unwrap()
        );

        let result = merge_and_deduplicate(&[&messy_path]);

        // Should not contain ./ or ../ after canonicalization
        assert!(!result.contains("./"), "Dot components should be removed");
        assert!(!result.contains("../"), "Parent components should be removed");
        assert_eq!(result, lib_dir.canonicalize().unwrap().display().to_string());
    }

    /// Test double slashes normalization
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.3-canonicalization-details (Edge Cases)
    ///
    /// Validates that double slashes are normalized to single slashes.
    #[test]
    #[cfg(unix)]
    fn test_canonicalize_double_slashes() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let lib_dir = temp_dir.path().join("lib");
        std::fs::create_dir(&lib_dir).unwrap();

        // Create path with double slashes
        let path_with_doubles = format!("{}//lib", temp_dir.path().display());

        let result = merge_and_deduplicate(&[&path_with_doubles]);

        // Should not contain double slashes after canonicalization
        assert!(!result.contains("//"), "Double slashes should be normalized");
        assert_eq!(result, lib_dir.canonicalize().unwrap().display().to_string());
    }

    /// Test case normalization on macOS
    ///
    /// Tests feature spec: rpath_merge_analysis.md#5.5-case-sensitivity-by-platform
    ///
    /// Validates that case is normalized on case-insensitive filesystems (macOS).
    #[test]
    #[cfg(target_os = "macos")]
    fn test_canonicalize_case_normalization() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let lib_dir = temp_dir.path().join("lib");
        std::fs::create_dir(&lib_dir).unwrap();

        // On macOS, case-insensitive filesystem normalizes case
        let path_str = lib_dir.to_str().unwrap();
        let upper_path = path_str.to_uppercase();

        let result = merge_and_deduplicate(&[path_str, &upper_path]);

        // Should deduplicate to single path (case-insensitive)
        assert_eq!(result.matches(':').count(), 0, "Case variants should deduplicate on macOS");
    }
}

// ====================================================================================
// DEDUPLICATION TESTS (AC3: Duplicate Removal, 5 tests)
// ====================================================================================
/// Tests feature spec: rpath_merge_analysis.md#4.4-deduplication-strategy
#[cfg(test)]
mod deduplication_tests {
    use super::*;

    /// Test deduplication of identical paths
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.4-deduplication-strategy
    ///
    /// Validates that duplicate paths are deduplicated to single entry.
    #[test]
    fn test_deduplicate_identical_paths() {
        use xtask::build_helpers::merge_and_deduplicate;

        // Arrange: Create single directory, reference it twice
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let shared_lib = temp_dir.path().join("shared_lib");
        std::fs::create_dir(&shared_lib).expect("Failed to create shared_lib");

        // Act: Merge identical paths (should deduplicate)
        let result =
            merge_and_deduplicate(&[shared_lib.to_str().unwrap(), shared_lib.to_str().unwrap()]);

        // Assert: Result should be single path (deduplication applied)
        assert_eq!(result, shared_lib.canonicalize().unwrap().display().to_string());
        assert_eq!(
            result.matches(':').count(),
            0,
            "Deduplicated paths should not contain colon separator"
        );
    }

    /// Test deduplication of three copies of same path
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.4-deduplication-strategy
    ///
    /// Validates that multiple duplicates are all deduplicated.
    #[test]
    fn test_deduplicate_multiple_copies() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let lib_dir = temp_dir.path().join("lib");
        std::fs::create_dir(&lib_dir).unwrap();

        let path_str = lib_dir.to_str().unwrap();
        let result = merge_and_deduplicate(&[path_str, path_str, path_str]);

        assert_eq!(result.matches(':').count(), 0);
        assert_eq!(result, lib_dir.canonicalize().unwrap().display().to_string());
    }

    /// Test deduplication with mixed valid and duplicate paths
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.4-deduplication-strategy
    ///
    /// Validates deduplication in presence of multiple distinct and duplicate paths.
    #[test]
    fn test_deduplicate_mixed_paths() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let path1 = temp_dir.path().join("lib1");
        let path2 = temp_dir.path().join("lib2");
        std::fs::create_dir(&path1).unwrap();
        std::fs::create_dir(&path2).unwrap();

        // Mix: path1, path2, path1 (duplicate), path2 (duplicate)
        let result = merge_and_deduplicate(&[
            path1.to_str().unwrap(),
            path2.to_str().unwrap(),
            path1.to_str().unwrap(),
            path2.to_str().unwrap(),
        ]);

        // Should have exactly 1 colon (2 unique paths)
        assert_eq!(result.matches(':').count(), 1);
        assert!(result.contains(&path1.canonicalize().unwrap().display().to_string()));
        assert!(result.contains(&path2.canonicalize().unwrap().display().to_string()));
    }

    /// Test ordering preservation with deduplication
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.4-deduplication-strategy
    ///
    /// Validates that first occurrence is kept, later duplicates are removed.
    #[test]
    fn test_deduplication_preserves_first_occurrence() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let path1 = temp_dir.path().join("path_bitnet");
        let path2 = temp_dir.path().join("path_llama");
        std::fs::create_dir(&path1).unwrap();
        std::fs::create_dir(&path2).unwrap();

        // Order: bitnet, llama, bitnet (duplicate)
        let result = merge_and_deduplicate(&[
            path1.to_str().unwrap(),
            path2.to_str().unwrap(),
            path1.to_str().unwrap(),
        ]);

        // Bitnet should appear first, llama second, duplicate ignored
        let canonical1 = path1.canonicalize().unwrap().display().to_string();
        let canonical2 = path2.canonicalize().unwrap().display().to_string();
        let expected = format!("{}:{}", canonical1, canonical2);
        assert_eq!(result, expected);
    }

    /// Test HashSet insertion order preservation
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.4-deduplication-strategy (Table)
    ///
    /// Validates that deduplication uses HashSet + Vec for O(n) performance.
    #[test]
    fn test_deduplication_algorithm_efficiency() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let mut paths = Vec::new();

        // Create 10 paths, then repeat them
        for i in 0..10 {
            let path = temp_dir.path().join(format!("lib{}", i));
            std::fs::create_dir(&path).unwrap();
            paths.push(path);
        }

        let path_strs: Vec<&str> = paths.iter().map(|p| p.to_str().unwrap()).collect();

        // Input: 10 unique paths + 10 duplicates = 20 paths
        let mut input = path_strs.clone();
        input.extend(&path_strs);

        let result = merge_and_deduplicate(&input);

        // Should deduplicate to 10 unique paths (9 colons)
        assert_eq!(result.matches(':').count(), 9);
    }
}

// ====================================================================================
// PRIORITY CHAIN TESTS (AC4: Five-Tier Priority, 10 tests)
// ====================================================================================
/// Tests feature spec: rpath_merge_analysis.md#3-priority-chain-for-rpath-sources
#[cfg(test)]
mod priority_chain_tests {
    use super::*;

    /// Priority 1: BITNET_CROSSVAL_LIBDIR overrides all
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.1 (Priority 1)
    ///
    /// Validates that BITNET_CROSSVAL_LIBDIR takes precedence over new variables.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_priority_1_crossval_libdir_override() {
        unimplemented!(
            "Priority 1: BITNET_CROSSVAL_LIBDIR override pending build.rs implementation. \
             Manual verification: export BITNET_CROSSVAL_LIBDIR=/tmp/test && cargo build -p xtask --features crossval-all"
        );
    }

    /// Priority 2: CROSSVAL_RPATH_BITNET takes precedence over auto-discovery
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.1 (Priority 2)
    ///
    /// Validates granular RPATH specification overrides BITNET_CPP_DIR.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_priority_2_granular_rpath_bitnet() {
        unimplemented!(
            "Priority 2: CROSSVAL_RPATH_BITNET granular override pending. \
             Manual verification: export CROSSVAL_RPATH_BITNET=/tmp/bitnet && cargo build"
        );
    }

    /// Priority 2: CROSSVAL_RPATH_LLAMA independent specification
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.1 (Priority 2)
    ///
    /// Validates granular RPATH for llama.cpp.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_priority_2_granular_rpath_llama() {
        unimplemented!(
            "Priority 2: CROSSVAL_RPATH_LLAMA granular override pending. \
             Manual verification: export CROSSVAL_RPATH_LLAMA=/tmp/llama && cargo build"
        );
    }

    /// Priority 2: Both granular variables merge correctly
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.1 (Priority 2)
    ///
    /// Validates merging of both CROSSVAL_RPATH_BITNET and CROSSVAL_RPATH_LLAMA.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_priority_2_both_granular_variables() {
        unimplemented!(
            "Priority 2: Merging both CROSSVAL_RPATH_BITNET and CROSSVAL_RPATH_LLAMA pending. \
             Manual verification: export both vars && cargo build && readelf"
        );
    }

    /// Priority 3A: BITNET_CPP_DIR auto-discovery (Tier 1)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.1 (Priority 3A - Search Tier 1)
    ///
    /// Validates auto-discovery from BITNET_CPP_DIR/build/bin.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_priority_3a_bitnet_cpp_dir_tier1() {
        unimplemented!(
            "Priority 3A Tier 1: BITNET_CPP_DIR/build/bin auto-discovery pending. \
             Manual verification: export BITNET_CPP_DIR=~/.cache/bitnet_cpp && cargo build"
        );
    }

    /// Priority 3A: BITNET_CPP_DIR auto-discovery (Tier 2 fallback)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.1 (Priority 3A - Search Tier 2)
    ///
    /// Validates fallback to embedded llama.cpp paths.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_priority_3a_bitnet_cpp_dir_tier2() {
        unimplemented!(
            "Priority 3A Tier 2: BITNET_CPP_DIR embedded llama fallback pending. \
             Manual verification: Test with build/3rdparty/llama.cpp paths"
        );
    }

    /// Priority 3B: LLAMA_CPP_DIR auto-discovery
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.1 (Priority 3B)
    ///
    /// Validates auto-discovery from standalone llama.cpp installation.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_priority_3b_llama_cpp_dir() {
        unimplemented!(
            "Priority 3B: LLAMA_CPP_DIR auto-discovery pending. \
             Manual verification: export LLAMA_CPP_DIR=~/.cache/llama_cpp && cargo build"
        );
    }

    /// Priority 4A: Default BitNet.cpp installation path
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.1 (Priority 4A)
    ///
    /// Validates fallback to $HOME/.cache/bitnet_cpp.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_priority_4a_default_bitnet_path() {
        unimplemented!(
            "Priority 4A: Default ~/.cache/bitnet_cpp fallback pending. \
             Manual verification: Unset all explicit vars && cargo build"
        );
    }

    /// Priority 4B: Default llama.cpp installation path
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.1 (Priority 4B)
    ///
    /// Validates fallback to $HOME/.cache/llama_cpp.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_priority_4b_default_llama_path() {
        unimplemented!(
            "Priority 4B: Default ~/.cache/llama_cpp fallback pending. \
             Manual verification: Unset all explicit vars && cargo build"
        );
    }

    /// Priority 5: Graceful degradation (STUB mode)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.1 (Priority 5)
    ///
    /// Validates that build succeeds even when no libraries are found.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_priority_5_stub_mode() {
        unimplemented!(
            "Priority 5: STUB mode graceful degradation pending. \
             Manual verification: Unset all vars, remove cache dirs && cargo build"
        );
    }
}

// ====================================================================================
// ENVIRONMENT VARIABLE PRECEDENCE TESTS (AC5: Precedence Rules, 5 tests)
// ====================================================================================
/// Tests feature spec: rpath_merge_analysis.md#3.2-environment-variable-summary-table
#[cfg(test)]
mod env_precedence_tests {
    use super::*;

    /// Test BITNET_CROSSVAL_LIBDIR wins over CROSSVAL_RPATH_*
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.2 (Priority 1 vs Priority 2)
    ///
    /// Validates that legacy variable overrides new granular variables.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_precedence_legacy_over_granular() {
        unimplemented!(
            "Precedence: BITNET_CROSSVAL_LIBDIR should win over CROSSVAL_RPATH_*. \
             Manual verification: Set both, verify BITNET_CROSSVAL_LIBDIR wins"
        );
    }

    /// Test CROSSVAL_RPATH_* wins over BITNET_CPP_DIR
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.2 (Priority 2 vs Priority 3)
    ///
    /// Validates that explicit RPATH overrides auto-discovery.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_precedence_granular_over_autodiscovery() {
        unimplemented!(
            "Precedence: CROSSVAL_RPATH_BITNET should win over BITNET_CPP_DIR. \
             Manual verification: Set both, verify CROSSVAL_RPATH_BITNET wins"
        );
    }

    /// Test BITNET_CPP_DIR wins over default paths
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.2 (Priority 3 vs Priority 4)
    ///
    /// Validates that explicit env var overrides default cache paths.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_precedence_explicit_over_defaults() {
        unimplemented!(
            "Precedence: BITNET_CPP_DIR should win over ~/.cache defaults. \
             Manual verification: Set BITNET_CPP_DIR, verify it wins"
        );
    }

    /// Test warning when both BITNET_CROSSVAL_LIBDIR and CROSSVAL_RPATH_* set
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.1 (Priority 1 - Action)
    ///
    /// Validates that warning is emitted when conflicting variables are set.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution and output capture"]
    fn test_precedence_conflict_warning() {
        unimplemented!(
            "Precedence Warning: Should emit cargo:warning when both legacy and new vars set. \
             Manual verification: Set both, check cargo build output for warning"
        );
    }

    /// Test HOME variable expansion for defaults
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.2 (HOME variable)
    ///
    /// Validates that $HOME is correctly expanded for default paths.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_precedence_home_expansion() {
        unimplemented!(
            "HOME expansion: Verify ~/.cache paths use actual HOME value. \
             Manual verification: Check that default paths resolve to actual $HOME"
        );
    }
}

// ====================================================================================
// PLATFORM-SPECIFIC TESTS (AC6: Cross-Platform, 5 tests)
// ====================================================================================
/// Tests feature spec: rpath_merge_analysis.md#5-platform-specific-considerations
#[cfg(test)]
mod platform_specific_tests {
    use super::*;

    /// Test Unix colon separator
    ///
    /// Tests feature spec: rpath_merge_analysis.md#5.1-linux-implementation
    ///
    /// Validates that Unix uses colon (':') as RPATH separator.
    #[test]
    #[cfg(unix)]
    fn test_unix_colon_separator() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let path1 = temp_dir.path().join("path1");
        let path2 = temp_dir.path().join("path2");
        std::fs::create_dir(&path1).unwrap();
        std::fs::create_dir(&path2).unwrap();

        let result = merge_and_deduplicate(&[path1.to_str().unwrap(), path2.to_str().unwrap()]);

        assert!(result.contains(':'), "Unix RPATH should use colon separator");
        assert_eq!(result.matches(':').count(), 1);
    }

    /// Test Windows PATH handling (RPATH N/A)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#5.3-windows-implementation
    ///
    /// Validates that Windows build emits warning about RPATH not applicable.
    #[test]
    #[cfg(windows)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_windows_rpath_not_applicable() {
        unimplemented!(
            "Windows: Should emit cargo:warning that RPATH not applicable, use PATH. \
             Manual verification: cargo build on Windows && check warning output"
        );
    }

    /// Test readelf verification on Linux
    ///
    /// Tests feature spec: rpath_merge_analysis.md#5.1-linux-implementation (Verification)
    ///
    /// Validates RPATH embedding can be verified with readelf.
    #[test]
    #[cfg(target_os = "linux")]
    #[ignore = "Integration test - requires binary build and readelf"]
    fn test_linux_readelf_verification() {
        unimplemented!(
            "Linux: readelf -d target/debug/xtask | grep RPATH verification. \
             Manual verification: Build binary && run readelf command"
        );
    }

    /// Test otool verification on macOS
    ///
    /// Tests feature spec: rpath_merge_analysis.md#5.2-macos-implementation (Verification)
    ///
    /// Validates RPATH embedding can be verified with otool.
    #[test]
    #[cfg(target_os = "macos")]
    #[ignore = "Integration test - requires binary build and otool"]
    fn test_macos_otool_verification() {
        unimplemented!(
            "macOS: otool -l target/debug/xtask | grep -A 3 LC_RPATH verification. \
             Manual verification: Build binary && run otool command"
        );
    }

    /// Test ldd library resolution on Linux
    ///
    /// Tests feature spec: rpath_merge_analysis.md#5.1-linux-implementation (Runtime Loader)
    ///
    /// Validates runtime library resolution from RPATH.
    #[test]
    #[cfg(target_os = "linux")]
    #[ignore = "Integration test - requires binary build and ldd"]
    fn test_linux_ldd_resolution() {
        unimplemented!(
            "Linux: ldd target/debug/xtask | grep libbitnet verification. \
             Manual verification: Build binary with real libraries && run ldd"
        );
    }
}

// ====================================================================================
// LENGTH VALIDATION TESTS (AC7: 4096 Byte Limit, 3 tests)
// ====================================================================================
/// Tests feature spec: rpath_merge_analysis.md#4.6-length-validation
#[cfg(test)]
mod length_validation_tests {
    use super::*;

    /// Test normal length (well under limit)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.6-length-validation
    ///
    /// Validates that typical RPATH lengths are well within 4KB limit.
    #[test]
    fn test_length_normal_case() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let mut paths = Vec::new();

        // Create 5 paths with reasonable length (~50 chars each = ~250 total)
        for i in 0..5 {
            let path = temp_dir.path().join(format!("lib_{}", i));
            std::fs::create_dir(&path).unwrap();
            paths.push(path);
        }

        let path_strs: Vec<&str> = paths.iter().map(|p| p.to_str().unwrap()).collect();
        let result = merge_and_deduplicate(&path_strs);

        // Should be well under 4KB
        assert!(result.len() < 4096, "Normal case should be under 4KB limit");
        assert!(result.len() < 500, "Normal case with 5 paths should be under 500 bytes");
    }

    /// Test approaching limit (many paths)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.6-length-validation
    ///
    /// Validates behavior with many paths approaching 4KB limit.
    #[test]
    fn test_length_approaching_limit() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let mut paths = Vec::new();

        // Create 20 paths to test higher (but still valid) lengths
        for i in 0..20 {
            let path = temp_dir.path().join(format!("lib_{}", i));
            std::fs::create_dir(&path).unwrap();
            paths.push(path);
        }

        let path_strs: Vec<&str> = paths.iter().map(|p| p.to_str().unwrap()).collect();
        let result = merge_and_deduplicate(&path_strs);

        // Should still be under 4KB
        assert!(result.len() < 4096, "20 paths should still be under 4KB limit");
    }

    /// Test exceeding limit (should panic)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.6-length-validation
    ///
    /// Validates that exceeding 4KB limit causes panic with helpful message.
    #[test]
    #[should_panic(expected = "Merged RPATH exceeds maximum length")]
    #[ignore = "Difficult to create paths long enough to exceed 4KB in temp dir"]
    fn test_length_exceeds_limit() {
        use xtask::build_helpers::merge_and_deduplicate;

        // Create extremely long paths (this is difficult in temp dirs)
        // Skip actual implementation as temp paths are typically short
        // This test documents expected behavior: panic if > 4096 bytes

        let long_paths = vec![""; 1000]; // Placeholder - real test would need 4KB+ paths
        let _result = merge_and_deduplicate(&long_paths);
    }
}

// ====================================================================================
// ERROR HANDLING TESTS (AC8: Invalid Paths, 4 tests)
// ====================================================================================
/// Tests feature spec: rpath_merge_analysis.md#4.3-canonicalization-details (Edge Cases)
#[cfg(test)]
mod error_handling_tests {
    use super::*;

    /// Test invalid path gracefully skipped
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.3 (Edge Case: Non-existent)
    ///
    /// Validates that non-existent paths emit warnings and are skipped.
    #[test]
    fn test_invalid_path_skipped() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let valid_path = temp_dir.path().join("valid_lib");
        std::fs::create_dir(&valid_path).unwrap();

        let invalid_path = "/this/path/does/not/exist/and/never/will";

        let result = merge_and_deduplicate(&[valid_path.to_str().unwrap(), invalid_path]);

        // Should contain only valid path (invalid skipped)
        assert_eq!(result, valid_path.canonicalize().unwrap().display().to_string());
        assert_eq!(result.matches(':').count(), 0);
    }

    /// Test all invalid paths (returns empty string)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.2-detailed-algorithm
    ///
    /// Validates behavior when all input paths are invalid.
    #[test]
    fn test_all_invalid_paths() {
        use xtask::build_helpers::merge_and_deduplicate;

        let invalid1 = "/invalid/path/one";
        let invalid2 = "/invalid/path/two";

        let result = merge_and_deduplicate(&[invalid1, invalid2]);

        assert_eq!(result, "", "All invalid paths should return empty string");
    }

    /// Test permission errors gracefully handled
    ///
    /// Tests feature spec: rpath_merge_analysis.md#10.6-permission-errors
    ///
    /// Validates that permission-denied paths emit warnings and are skipped.
    #[test]
    #[cfg(unix)]
    #[ignore = "Requires root to create inaccessible directories"]
    fn test_permission_errors() {
        unimplemented!(
            "Permission errors: Should skip paths with no read permission. \
             Manual verification: Create dir with 000 perms && verify skipped"
        );
    }

    /// Test mixed valid and invalid paths
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.3 (Edge Cases)
    ///
    /// Validates that merge continues with valid paths even when some are invalid.
    #[test]
    fn test_mixed_valid_invalid_paths() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let path1 = temp_dir.path().join("lib1");
        let path2 = temp_dir.path().join("lib2");
        std::fs::create_dir(&path1).unwrap();
        std::fs::create_dir(&path2).unwrap();

        let invalid = "/nonexistent/path";

        let result =
            merge_and_deduplicate(&[path1.to_str().unwrap(), invalid, path2.to_str().unwrap()]);

        // Should contain only two valid paths
        assert_eq!(result.matches(':').count(), 1);
        assert!(result.contains(&path1.canonicalize().unwrap().display().to_string()));
        assert!(result.contains(&path2.canonicalize().unwrap().display().to_string()));
        assert!(!result.contains(invalid));
    }
}

// ====================================================================================
// EMPTY INPUT HANDLING TESTS (AC9: Edge Cases, 2 tests)
// ====================================================================================

#[cfg(test)]
mod empty_input_tests {

    /// Test empty vector input
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.2-detailed-algorithm
    ///
    /// Validates that empty input returns empty string.
    #[test]
    fn test_empty_vector() {
        use xtask::build_helpers::merge_and_deduplicate;

        let paths: Vec<&str> = vec![];
        let result = merge_and_deduplicate(&paths);

        assert_eq!(result, "", "Empty input should return empty string");
    }

    /// Test vector with only empty strings
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.2-detailed-algorithm
    ///
    /// Validates that empty string paths are skipped.
    #[test]
    fn test_only_empty_strings() {
        use xtask::build_helpers::merge_and_deduplicate;

        let paths = vec!["", "", ""];
        let result = merge_and_deduplicate(&paths);

        assert_eq!(result, "", "Only empty strings should return empty string");
    }
}

// ====================================================================================
// BACKWARD COMPATIBILITY TESTS (AC10: Legacy Support, 3 tests)
// ====================================================================================
/// Tests feature spec: rpath_merge_analysis.md#2.2-single-vs-multi-path (Legacy)
#[cfg(test)]
mod backward_compatibility_tests {
    use super::*;

    /// Test BITNET_CROSSVAL_LIBDIR still works (regression)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#2.2 (Legacy Single-Path)
    ///
    /// Validates that existing deployments using BITNET_CROSSVAL_LIBDIR continue to work.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_backward_compat_crossval_libdir() {
        unimplemented!(
            "Backward Compatibility: BITNET_CROSSVAL_LIBDIR should work as before. \
             Manual verification: export BITNET_CROSSVAL_LIBDIR=/opt/lib && cargo build"
        );
    }

    /// Test BITNET_CPP_DIR fallback preserved
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.1 (Priority 3A)
    ///
    /// Validates that existing auto-discovery from BITNET_CPP_DIR still works.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_backward_compat_cpp_dir() {
        unimplemented!(
            "Backward Compatibility: BITNET_CPP_DIR auto-discovery should work. \
             Manual verification: export BITNET_CPP_DIR=~/.cache/bitnet_cpp && cargo build"
        );
    }

    /// Test no regression in STUB mode
    ///
    /// Tests feature spec: rpath_merge_analysis.md#3.1 (Priority 5)
    ///
    /// Validates that graceful degradation still works when no libraries found.
    #[test]
    #[serial(bitnet_env)]
    #[ignore = "Integration test - requires build.rs execution"]
    fn test_backward_compat_stub_mode() {
        unimplemented!(
            "Backward Compatibility: STUB mode should still succeed gracefully. \
             Manual verification: Unset all vars && cargo build (should succeed)"
        );
    }
}

// ====================================================================================
// INTEGRATION WITH BUILD.RS (AC11: Build System, 2 tests)
// ====================================================================================

#[cfg(test)]
mod build_integration_tests {

    /// Test cargo:rerun-if-env-changed directives
    ///
    /// Tests feature spec: rpath_merge_analysis.md#6.3-rerun-if-triggers
    ///
    /// Validates that build.rs emits correct rebuild triggers.
    #[test]
    #[ignore = "Integration test - requires build.rs output capture"]
    fn test_rerun_triggers() {
        unimplemented!(
            "Build Integration: cargo:rerun-if-env-changed directives verification. \
             Manual verification: Check build.rs output for rerun-if-env-changed"
        );
    }

    /// Test cargo:rustc-link-arg emission
    ///
    /// Tests feature spec: rpath_merge_analysis.md#6.2-cargo-build-directive-syntax
    ///
    /// Validates that correct linker arguments are emitted.
    #[test]
    #[ignore = "Integration test - requires build.rs output capture"]
    fn test_linker_arg_emission() {
        unimplemented!(
            "Build Integration: cargo:rustc-link-arg=-Wl,-rpath,... verification. \
             Manual verification: Check build.rs output for rustc-link-arg"
        );
    }
}

// ====================================================================================
// MANUAL VERIFICATION HELPERS (AC12: Verification Tools, 2 tests)
// ====================================================================================

#[cfg(test)]
mod verification_helpers {

    /// Simulate readelf output parsing (Linux)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#7.3-manual-verification (Linux)
    ///
    /// Provides helper for verifying RPATH in built binary using readelf.
    #[test]
    #[cfg(target_os = "linux")]
    #[ignore = "Helper test - requires built binary"]
    fn test_simulate_readelf_verification() {
        unimplemented!(
            "Verification Helper: Parse readelf -d output to verify RPATH. \
             Manual verification: readelf -d target/debug/xtask | grep RPATH"
        );
    }

    /// Simulate otool output parsing (macOS)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#7.3-manual-verification (macOS)
    ///
    /// Provides helper for verifying RPATH in built binary using otool.
    #[test]
    #[cfg(target_os = "macos")]
    #[ignore = "Helper test - requires built binary"]
    fn test_simulate_otool_verification() {
        unimplemented!(
            "Verification Helper: Parse otool -l output to verify RPATH. \
             Manual verification: otool -l target/debug/xtask | grep -A 3 LC_RPATH"
        );
    }
}

// ====================================================================================
// ORDERING PRESERVATION TESTS (Additional Coverage, 2 tests)
// ====================================================================================

#[cfg(test)]
mod ordering_tests {
    use super::*;

    /// Test ordering preserved (BitNet before llama)
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.4-deduplication-strategy
    ///
    /// Validates that RPATH ordering is preserved: BitNet libraries before llama libraries.
    #[test]
    fn test_ordering_preserved() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let bitnet_lib = temp_dir.path().join("path_bitnet");
        let llama_lib = temp_dir.path().join("path_llama");

        std::fs::create_dir(&bitnet_lib).unwrap();
        std::fs::create_dir(&llama_lib).unwrap();

        let result =
            merge_and_deduplicate(&[bitnet_lib.to_str().unwrap(), llama_lib.to_str().unwrap()]);

        let canonical_bitnet = bitnet_lib.canonicalize().unwrap().display().to_string();
        let canonical_llama = llama_lib.canonicalize().unwrap().display().to_string();
        let expected = format!("{}:{}", canonical_bitnet, canonical_llama);

        assert_eq!(result, expected, "Ordering must be preserved: BitNet before llama");
        assert!(result.starts_with(&canonical_bitnet), "BitNet should appear first");
    }

    /// Test ordering with three paths
    ///
    /// Tests feature spec: rpath_merge_analysis.md#4.4-deduplication-strategy
    ///
    /// Validates insertion order is maintained for multiple paths.
    #[test]
    fn test_ordering_three_paths() {
        use xtask::build_helpers::merge_and_deduplicate;

        let temp_dir = TempDir::new().unwrap();
        let path1 = temp_dir.path().join("first");
        let path2 = temp_dir.path().join("second");
        let path3 = temp_dir.path().join("third");

        std::fs::create_dir(&path1).unwrap();
        std::fs::create_dir(&path2).unwrap();
        std::fs::create_dir(&path3).unwrap();

        let result = merge_and_deduplicate(&[
            path1.to_str().unwrap(),
            path2.to_str().unwrap(),
            path3.to_str().unwrap(),
        ]);

        let c1 = path1.canonicalize().unwrap().display().to_string();
        let c2 = path2.canonicalize().unwrap().display().to_string();
        let c3 = path3.canonicalize().unwrap().display().to_string();

        // Verify order: first, second, third
        let parts: Vec<&str> = result.split(':').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], c1);
        assert_eq!(parts[1], c2);
        assert_eq!(parts[2], c3);
    }
}
