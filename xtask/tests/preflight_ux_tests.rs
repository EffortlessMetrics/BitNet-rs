//! Unit and integration tests for preflight UX parity enhancements
//!
//! Tests specification: docs/specs/preflight-ux-parity.md
//!
//! These tests validate the four UX improvements:
//! 1. Setup command unification (no --bitnet flag inconsistency)
//! 2. Search path coverage (7 paths including build/bin)
//! 3. Path context labels (embedded vs standalone clarity)
//! 4. Build metadata enhancement (shows required libraries)
//!
//! **Test Strategy**: TDD scaffolding - tests compile but fail due to missing implementation.
//! All tests are marked with specification references for traceability.

#[cfg(test)]
mod preflight_ux_unit_tests {
    #[allow(unused_imports)]
    use std::path::PathBuf;

    /// Tests feature spec: preflight-ux-parity.md#3.1.1-setup_command-unified-return
    ///
    /// Validates that both backends return identical setup commands without the --bitnet flag.
    ///
    /// **Expected behavior**: Both BitNet and LLaMA should emit:
    /// `eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"`
    ///
    /// **Contract**: Returns identical setup command for all backends, relying on auto-detection
    /// logic in `setup-cpp-auto` to determine which backend(s) to install.
    #[test]
    fn test_setup_command_consistency() {
        use xtask::crossval::CppBackend;

        // Both backends should return identical setup command
        let bitnet_cmd = CppBackend::BitNet.setup_command();
        let llama_cmd = CppBackend::Llama.setup_command();

        assert_eq!(bitnet_cmd, llama_cmd, "Setup commands should be identical for both backends");

        // Verify no --bitnet flag in either command
        assert!(
            !bitnet_cmd.contains("--bitnet"),
            "BitNet setup command should not have --bitnet flag (auto-detection handles this)"
        );
        assert!(
            !llama_cmd.contains("--bitnet"),
            "LLaMA setup command should not have --bitnet flag"
        );

        // Verify both use setup-cpp-auto
        assert!(
            bitnet_cmd.contains("setup-cpp-auto"),
            "Setup command should use setup-cpp-auto for auto-detection"
        );
        assert!(
            llama_cmd.contains("setup-cpp-auto"),
            "Setup command should use setup-cpp-auto for auto-detection"
        );

        // Verify emit flag is present
        assert!(
            bitnet_cmd.contains("--emit=sh"),
            "Setup command should have --emit=sh flag for shell output"
        );
    }

    /// Tests feature spec: preflight-ux-parity.md#3.2.1-get_library_search_paths-enhancement
    ///
    /// Validates that the search path list includes build/bin for standalone llama.cpp installations
    /// and that the priority order is correct.
    ///
    /// **Expected behavior**:
    /// - 7 total search paths (6 original + build/bin)
    /// - build/bin appears at index 2 (after build/, before build/lib/)
    /// - All original paths preserved in correct order
    ///
    /// **Contract**: Returns 7 paths in priority order. New path inserted at index 2.
    #[test]
    fn test_search_paths_include_build_bin() {
        use std::path::PathBuf;
        use xtask::crossval::preflight::get_library_search_paths;

        let paths = get_library_search_paths();

        // Should have 7 paths total (6 original + 1 new build/bin)
        // Note: BITNET_CROSSVAL_LIBDIR is only included if set, so we expect 6 paths
        // (paths don't include the override path in the vec, it's checked separately)
        assert_eq!(paths.len(), 6, "Should search 6 paths (excluding override)");

        // build/bin should be present
        let has_build_bin = paths.iter().any(|p| p.to_string_lossy().ends_with("build/bin"));
        assert!(has_build_bin, "Should include build/bin path for standalone llama.cpp");

        // Verify priority order: build comes before build/bin before build/lib
        let build_idx = paths.iter().position(|p: &PathBuf| p.ends_with("build")).unwrap();
        let build_bin_idx = paths.iter().position(|p: &PathBuf| p.ends_with("build/bin")).unwrap();
        let build_lib_idx = paths.iter().position(|p: &PathBuf| p.ends_with("build/lib")).unwrap();

        assert!(build_idx < build_bin_idx, "build/ should come before build/bin");
        assert!(build_bin_idx < build_lib_idx, "build/bin should come before build/lib");
    }

    /// Tests feature spec: preflight-ux-parity.md#3.2.1-get_library_search_paths-enhancement
    ///
    /// Validates the priority order of search paths:
    /// build/ must come before build/bin, which must come before build/lib.
    ///
    /// **Expected behavior**: Priority ordering preserves main output directory first.
    #[test]
    fn test_search_path_priority_order() {
        use std::path::PathBuf;
        use xtask::crossval::preflight::get_library_search_paths;

        let paths = get_library_search_paths();

        // Find indices of build/, build/bin, build/lib
        let build_idx = paths.iter().position(|p: &PathBuf| p.ends_with("build")).unwrap();
        let build_bin_idx = paths.iter().position(|p: &PathBuf| p.ends_with("build/bin")).unwrap();
        let build_lib_idx = paths.iter().position(|p: &PathBuf| p.ends_with("build/lib")).unwrap();

        // Assert build_idx < build_bin_idx < build_lib_idx
        assert!(build_idx < build_bin_idx, "build/ should come before build/bin");
        assert!(build_bin_idx < build_lib_idx, "build/bin should come before build/lib");
    }

    /// Tests feature spec: preflight-ux-parity.md#3.2.2-get_path_context_label-new-helper
    ///
    /// Validates that path context labels are generated correctly for different path types.
    ///
    /// **Expected behavior**:
    /// - Embedded llama.cpp paths get " (embedded llama.cpp)" label
    /// - Embedded ggml paths get " (embedded ggml)" label
    /// - Standalone llama.cpp paths get " (standalone llama.cpp)" label
    /// - CROSSVAL_LIBDIR override gets " (explicit override)" label
    /// - Standard paths (build, build/lib, lib) get empty string
    ///
    /// **Contract**: Returns static string labels for path clarification. Empty string for
    /// paths that don't need context.
    #[test]
    fn test_path_context_label_accuracy() {
        use std::path::PathBuf;
        use xtask::crossval::preflight::get_path_context_label;

        // Test cases from spec:
        let test_cases = vec![
            (
                "/opt/bitnet/build/3rdparty/llama.cpp/src",
                " (embedded llama.cpp)",
                "Embedded llama.cpp paths should be labeled",
            ),
            (
                "/opt/bitnet/build/3rdparty/llama.cpp/ggml/src",
                " (embedded ggml)",
                "Embedded ggml paths should be labeled",
            ),
            (
                "/opt/bitnet/build/bin",
                " (standalone llama.cpp)",
                "Standalone llama.cpp paths (build/bin) should be labeled",
            ),
            ("/opt/bitnet/build", "", "Standard build/ path should have no label"),
            ("/opt/bitnet/build/lib", "", "Standard build/lib path should have no label"),
            ("/opt/bitnet/lib", "", "Standard lib/ path should have no label"),
        ];

        for (path_str, expected_label, description) in test_cases {
            let path = PathBuf::from(path_str);
            let actual_label = get_path_context_label(&path);
            assert_eq!(actual_label, expected_label, "{}", description);
        }
    }

    /// Tests feature spec: preflight-ux-parity.md#3.3.1-format_build_metadata-enhancement
    ///
    /// Validates that build metadata includes the required libraries for each backend.
    ///
    /// **Expected behavior**:
    /// - BitNet metadata shows: "Required libraries: libbitnet"
    /// - LLaMA metadata shows: "Required libraries: libllama, libggml"
    /// - Line appears in Build-Time Detection Metadata section
    ///
    /// **Contract**: Adds "Required libraries" line showing what build system searched for.
    /// Uses `join(", ")` for multi-lib backends (LLaMA).
    #[test]
    fn test_build_metadata_includes_required_libraries() {
        use xtask::crossval::CppBackend;
        use xtask::crossval::preflight::format_build_metadata;

        let bitnet_meta = format_build_metadata(CppBackend::BitNet);
        assert!(
            bitnet_meta.contains("Required libraries: libbitnet"),
            "BitNet metadata should show required library"
        );

        let llama_meta = format_build_metadata(CppBackend::Llama);
        assert!(
            llama_meta.contains("Required libraries: libllama, libggml"),
            "LLaMA metadata should show both required libraries"
        );
    }

    /// Tests feature spec: preflight-ux-parity.md#3.3.1-format_build_metadata-enhancement
    ///
    /// Validates that build metadata preserves existing output structure while adding new line.
    ///
    /// **Expected behavior**: New "Required libraries" line appears between CROSSVAL_HAS_*
    /// and "Last xtask build" timestamp.
    #[test]
    fn test_build_metadata_format_structure() {
        use xtask::crossval::CppBackend;
        use xtask::crossval::preflight::format_build_metadata;

        let bitnet_meta = format_build_metadata(CppBackend::BitNet);

        // Verify section header "Build-Time Detection Metadata" present
        assert!(
            bitnet_meta.contains("Build-Time Detection Metadata"),
            "Should contain metadata section header"
        );

        // Verify CROSSVAL_HAS_* line present
        assert!(
            bitnet_meta.contains("CROSSVAL_HAS_BITNET"),
            "Should contain CROSSVAL_HAS_BITNET line"
        );

        // Verify "Required libraries:" line present
        assert!(
            bitnet_meta.contains("Required libraries:"),
            "Should contain Required libraries line"
        );

        // Verify timestamp line present
        assert!(bitnet_meta.contains("Last xtask build:"), "Should contain timestamp line");

        // Verify feature flags line present
        assert!(
            bitnet_meta.contains("Build feature flags: crossval-all"),
            "Should contain feature flags line"
        );

        // Verify ordering: CROSSVAL_HAS_* comes before Required libraries
        let crossval_pos = bitnet_meta.find("CROSSVAL_HAS_").unwrap();
        let required_pos = bitnet_meta.find("Required libraries:").unwrap();
        let timestamp_pos = bitnet_meta.find("Last xtask build:").unwrap();

        assert!(
            crossval_pos < required_pos,
            "CROSSVAL_HAS_* should come before Required libraries"
        );
        assert!(required_pos < timestamp_pos, "Required libraries should come before timestamp");
    }

    /// Tests feature spec: preflight-ux-parity.md#3.1.2-required_libs-unchanged
    ///
    /// Validates that required_libs() returns correct library stems for each backend.
    ///
    /// **Expected behavior**:
    /// - BitNet requires: ["libbitnet"]
    /// - LLaMA requires: ["libllama", "libggml"]
    ///
    /// **Contract**: Returns library stems (without .so/.dylib extension) for build-time
    /// and runtime detection.
    #[test]
    fn test_backend_required_libs_contract() {
        use xtask::crossval::CppBackend;

        // BitNet should require single library
        let bitnet_libs = CppBackend::BitNet.required_libs();
        assert_eq!(bitnet_libs, &["libbitnet"], "BitNet backend should require libbitnet");
        assert_eq!(bitnet_libs.len(), 1, "BitNet should require exactly one library");

        // LLaMA should require two libraries
        let llama_libs = CppBackend::Llama.required_libs();
        assert_eq!(
            llama_libs,
            &["libllama", "libggml"],
            "LLaMA backend should require libllama and libggml"
        );
        assert_eq!(llama_libs.len(), 2, "LLaMA should require exactly two libraries");
    }

    /// Tests feature spec: preflight-ux-parity.md#verification-criteria
    ///
    /// Validates that backend name() method returns correct strings for diagnostics.
    ///
    /// **Expected behavior**: BitNet → "bitnet.cpp", LLaMA → "llama.cpp"
    #[test]
    fn test_backend_name_consistency() {
        use xtask::crossval::CppBackend;

        assert_eq!(
            CppBackend::BitNet.name(),
            "bitnet.cpp",
            "BitNet backend name should be 'bitnet.cpp'"
        );
        assert_eq!(
            CppBackend::Llama.name(),
            "llama.cpp",
            "LLaMA backend name should be 'llama.cpp'"
        );
    }
}

#[cfg(test)]
mod preflight_ux_integration_tests {
    use std::process::Command;

    /// Helper to run xtask preflight command and capture output
    ///
    /// Returns (stdout, stderr, exit_code)
    fn run_preflight_command(backend: &str, verbose: bool) -> (String, String, i32) {
        let mut cmd = Command::new("cargo");
        cmd.arg("run")
            .arg("-p")
            .arg("xtask")
            .arg("--features")
            .arg("crossval-all")
            .arg("--")
            .arg("preflight")
            .arg("--backend")
            .arg(backend);

        if verbose {
            cmd.arg("--verbose");
        }

        let output = cmd.output().expect("Failed to execute preflight command");

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        (stdout, stderr, exit_code)
    }

    /// Tests feature spec: preflight-ux-parity.md#5.2.1-success-path-with-verbose-output
    ///
    /// Validates success output format when backend libraries are available.
    ///
    /// **Expected behavior**:
    /// - Header: "✓ Backend 'X': AVAILABLE"
    /// - Environment Configuration section
    /// - Library Search Paths showing 7 paths with context labels
    /// - Required Libraries section
    /// - Build Metadata with "Required libraries: X"
    /// - Platform-Specific Configuration
    /// - Summary section
    #[test]
    #[ignore] // TODO: Requires C++ libraries installed for success path testing
    fn test_success_output_validation() {
        // Run with verbose flag to get full diagnostics
        let (stdout, _stderr, exit_code) = run_preflight_command("bitnet", true);

        // Should exit successfully when libraries found
        assert_eq!(exit_code, 0, "Preflight should succeed when libraries are available");

        // Validate section headers (7 sections in verbose success output)
        let expected_sections = vec![
            "Backend 'bitnet.cpp': AVAILABLE",
            "Environment Configuration",
            "Library Search Paths",
            "Required Libraries",
            "Build-Time Detection Metadata",
            "Platform-Specific Configuration",
            "Summary",
        ];

        for section in &expected_sections {
            assert!(stdout.contains(section), "Success output should contain section: {}", section);
        }

        // Validate context labels appear in search paths
        assert!(
            stdout.contains("(standalone llama.cpp)") || stdout.contains("(embedded"),
            "Success output should show path context labels"
        );

        // Validate build metadata shows required libraries
        assert!(
            stdout.contains("Required libraries:"),
            "Success output should show required libraries in build metadata"
        );
    }

    /// Tests feature spec: preflight-ux-parity.md#5.2.1-failure-path-with-recovery-instructions
    ///
    /// Validates failure output format when backend libraries are NOT available.
    ///
    /// **Expected behavior**:
    /// - Error header: "Backend 'X' libraries NOT FOUND"
    /// - CRITICAL explanation about build-time detection
    /// - Required libraries list
    /// - RECOVERY STEPS with unified setup command (no --bitnet flag)
    /// - TROUBLESHOOTING section
    #[test]
    #[ignore] // TODO: Requires environment without C++ libraries (or mock failure)
    fn test_failure_output_validation() {
        // Note: This test requires running in environment without C++ libraries
        // or using BITNET_CPP_DIR="" to simulate missing libraries

        // Run preflight in failure scenario
        let (stdout, stderr, exit_code) = run_preflight_command("llama", false);

        // Should exit with error when libraries not found
        assert_ne!(exit_code, 0, "Preflight should fail when libraries are unavailable");

        // Check error output (could be in stdout or stderr depending on implementation)
        let output = format!("{}{}", stdout, stderr);

        // Validate error header
        assert!(
            output.contains("libraries NOT FOUND"),
            "Error output should indicate libraries not found"
        );

        // Validate CRITICAL explanation
        assert!(
            output.contains("CRITICAL") || output.contains("BUILD time"),
            "Error output should explain build-time vs runtime detection"
        );

        // Validate required libraries listed
        assert!(
            output.contains("libllama") || output.contains("Required libraries"),
            "Error output should list required libraries"
        );

        // Validate unified setup command (no --bitnet flag difference)
        assert!(
            output.contains("setup-cpp-auto"),
            "Error output should show setup-cpp-auto command"
        );
        assert!(
            !output.contains("--bitnet") || output.contains("# Both backends"),
            "Error output should use unified setup command without --bitnet flag"
        );

        // Validate recovery sections
        assert!(
            output.contains("RECOVERY STEPS") || output.contains("Option A"),
            "Error output should provide recovery steps"
        );
        assert!(
            output.contains("TROUBLESHOOTING") || output.contains("--verbose"),
            "Error output should suggest troubleshooting with verbose flag"
        );
    }

    /// Tests feature spec: preflight-ux-parity.md#5.2.1-verbose-mode-section-completeness
    ///
    /// Validates that verbose mode shows all diagnostic sections.
    ///
    /// **Expected behavior**: Verbose output includes search paths with existence status,
    /// found libraries enumerated, platform-specific loader variables.
    #[test]
    #[ignore] // TODO: Requires implementation of verbose diagnostics enhancements
    fn test_verbose_mode_section_completeness() {
        let (stdout_verbose, _, _) = run_preflight_command("bitnet", true);
        let (stdout_normal, _, _) = run_preflight_command("bitnet", false);

        // Verbose output should be longer (more sections)
        assert!(
            stdout_verbose.len() > stdout_normal.len(),
            "Verbose output should contain more information than normal output"
        );

        // Verbose should show search paths with detailed status
        assert!(
            stdout_verbose.contains("build/bin") || stdout_verbose.contains("build/lib"),
            "Verbose output should enumerate search paths"
        );

        // Verbose should show existence status (✓ or ✗)
        assert!(
            stdout_verbose.contains("✓") || stdout_verbose.contains("✗"),
            "Verbose output should show path existence status markers"
        );

        // Verbose should show environment variables
        #[cfg(target_os = "linux")]
        assert!(
            stdout_verbose.contains("LD_LIBRARY_PATH"),
            "Verbose output should show LD_LIBRARY_PATH on Linux"
        );

        #[cfg(target_os = "macos")]
        assert!(
            stdout_verbose.contains("DYLD_LIBRARY_PATH"),
            "Verbose output should show DYLD_LIBRARY_PATH on macOS"
        );
    }

    /// Tests feature spec: preflight-ux-parity.md#5.2.1-search-path-priority-display-order
    ///
    /// Validates that search paths are displayed in correct priority order.
    ///
    /// **Expected behavior**: Paths appear numbered 1-7 in priority order with build/bin
    /// at position 3 (after build/, before build/lib/).
    #[test]
    #[ignore] // TODO: Requires implementation of search path display enhancement
    fn test_search_path_priority_display_order() {
        let (stdout, _, _) = run_preflight_command("llama", true);

        // Extract numbered search path lines
        // Expected order:
        // 1. BITNET_CROSSVAL_LIBDIR override (if set)
        // 2. BITNET_CPP_DIR/build
        // 3. BITNET_CPP_DIR/build/bin (NEW - standalone llama.cpp)
        // 4. BITNET_CPP_DIR/build/lib
        // 5. BITNET_CPP_DIR/build/3rdparty/llama.cpp/src (embedded llama.cpp)
        // 6. BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src (embedded ggml)
        // 7. BITNET_CPP_DIR/lib

        // Find position of build/bin in output
        let build_bin_pattern = "build/bin";
        assert!(stdout.contains(build_bin_pattern), "Verbose output should show build/bin path");

        // Validate build/bin appears before build/lib
        if let (Some(bin_pos), Some(lib_pos)) = (stdout.find("build/bin"), stdout.find("build/lib"))
        {
            assert!(bin_pos < lib_pos, "build/bin should appear before build/lib in search order");
        }

        // Validate context label for build/bin
        assert!(
            stdout.contains("(standalone llama.cpp)"),
            "build/bin should be labeled as standalone llama.cpp"
        );
    }

    /// Tests feature spec: preflight-ux-parity.md#5.3-platform-specific-tests
    ///
    /// Validates platform-specific environment variable handling.
    ///
    /// **Expected behavior**:
    /// - Linux: Shows LD_LIBRARY_PATH
    /// - macOS: Shows DYLD_LIBRARY_PATH
    /// - Windows: Shows PATH
    #[test]
    #[ignore] // TODO: Platform-specific test requiring conditional compilation
    fn test_platform_specific_loader_variables() {
        let (stdout, _, _) = run_preflight_command("bitnet", true);

        #[cfg(target_os = "linux")]
        {
            assert!(
                stdout.contains("LD_LIBRARY_PATH"),
                "Linux output should reference LD_LIBRARY_PATH"
            );
            assert!(
                !stdout.contains("DYLD_LIBRARY_PATH"),
                "Linux output should not reference DYLD_LIBRARY_PATH"
            );
        }

        #[cfg(target_os = "macos")]
        {
            assert!(
                stdout.contains("DYLD_LIBRARY_PATH"),
                "macOS output should reference DYLD_LIBRARY_PATH"
            );
            assert!(
                !stdout.contains("LD_LIBRARY_PATH"),
                "macOS output should not reference LD_LIBRARY_PATH"
            );
        }

        #[cfg(target_os = "windows")]
        {
            assert!(stdout.contains("PATH"), "Windows output should reference PATH variable");
            assert!(
                !stdout.contains("LD_LIBRARY_PATH"),
                "Windows output should not reference LD_LIBRARY_PATH"
            );
        }
    }

    /// Tests feature spec: preflight-ux-parity.md#6.2-backward-compatibility
    ///
    /// Validates that existing scripts with --bitnet flag continue to work.
    ///
    /// **Expected behavior**: Old command syntax with --bitnet flag should still work
    /// (backward compatibility).
    #[test]
    #[ignore] // TODO: Requires setup-cpp-auto command implementation
    fn test_backward_compatibility_bitnet_flag() {
        // Test that old command with --bitnet flag still works
        let mut cmd = Command::new("cargo");
        cmd.arg("run")
            .arg("-p")
            .arg("xtask")
            .arg("--features")
            .arg("crossval-all")
            .arg("--")
            .arg("setup-cpp-auto")
            .arg("--bitnet")
            .arg("--emit=sh");

        let output = cmd.output().expect("Failed to execute setup-cpp-auto");

        // Should exit successfully (even if flag is deprecated/ignored)
        assert_eq!(
            output.status.code().unwrap_or(-1),
            0,
            "Old command with --bitnet flag should still work for backward compatibility"
        );
    }

    /// Tests feature spec: preflight-ux-parity.md#6.3-cross-platform-verification
    ///
    /// Validates that output format is consistent across platforms.
    ///
    /// **Expected behavior**: Section headers, separator lengths, and structure should be
    /// identical across Linux/macOS/Windows (only loader variable names differ).
    #[test]
    #[ignore] // TODO: Requires cross-platform testing infrastructure
    fn test_cross_platform_output_format_consistency() {
        let (stdout, _, _) = run_preflight_command("llama", true);

        // Verify separator lengths (should be 70 characters)
        let _heavy_separator = "━".repeat(70);
        let _light_separator = "─".repeat(70);

        // Check that separators exist (exact match not required due to UTF-8 width issues)
        assert!(
            stdout.contains("━━━━") || stdout.contains("────"),
            "Output should use visual separators for section delineation"
        );

        // Verify section structure is present
        let section_headers = vec![
            "Environment Configuration",
            "Library Search Paths",
            "Build-Time Detection Metadata",
            "Platform-Specific Configuration",
        ];

        for header in section_headers {
            assert!(stdout.contains(header), "Output should contain section header: {}", header);
        }
    }
}

#[cfg(test)]
mod preflight_ux_property_tests {
    /// Tests feature spec: preflight-ux-parity.md#verification-criteria
    ///
    /// Property-based test: Setup commands should never contain backend-specific flags.
    ///
    /// **Property**: For all backends, setup_command() output should not contain
    /// backend-specific flags (--bitnet, --llama, etc.).
    #[test]
    fn property_setup_commands_are_backend_agnostic() {
        use xtask::crossval::CppBackend;

        let backends = vec![CppBackend::BitNet, CppBackend::Llama];

        for backend in backends {
            let cmd = backend.setup_command();

            // Property: No backend-specific flags
            assert!(
                !cmd.contains("--bitnet"),
                "Backend {:?} setup command should not contain --bitnet flag",
                backend
            );
            assert!(
                !cmd.contains("--llama"),
                "Backend {:?} setup command should not contain --llama flag",
                backend
            );

            // Property: Must use setup-cpp-auto for auto-detection
            assert!(
                cmd.contains("setup-cpp-auto"),
                "Backend {:?} setup command must use setup-cpp-auto",
                backend
            );

            // Property: Must specify emit format
            assert!(
                cmd.contains("--emit="),
                "Backend {:?} setup command must specify emit format",
                backend
            );
        }
    }

    /// Tests feature spec: preflight-ux-parity.md#3.1.2-required_libs-unchanged
    ///
    /// Property-based test: required_libs() should return non-empty arrays.
    ///
    /// **Property**: For all backends, required_libs() returns at least one library.
    #[test]
    fn property_required_libs_non_empty() {
        use xtask::crossval::CppBackend;

        let backends = vec![CppBackend::BitNet, CppBackend::Llama];

        for backend in backends {
            let libs = backend.required_libs();

            // Property: Must require at least one library
            assert!(!libs.is_empty(), "Backend {:?} must require at least one library", backend);

            // Property: All library names must start with "lib" prefix
            for lib in libs {
                assert!(
                    lib.starts_with("lib"),
                    "Backend {:?} library '{}' must start with 'lib' prefix",
                    backend,
                    lib
                );
            }

            // Property: Library names should not contain extensions
            for lib in libs {
                assert!(
                    !lib.contains(".so") && !lib.contains(".dylib") && !lib.contains(".dll"),
                    "Backend {:?} library '{}' should not contain file extension (use stems only)",
                    backend,
                    lib
                );
            }
        }
    }

    /// Tests feature spec: preflight-ux-parity.md#6.1-success-metrics
    ///
    /// Property-based test: Backend names should be valid identifier strings.
    ///
    /// **Property**: For all backends, name() returns a valid backend identifier.
    #[test]
    fn property_backend_names_valid() {
        use xtask::crossval::CppBackend;

        let backends = vec![CppBackend::BitNet, CppBackend::Llama];

        for backend in backends {
            let name = backend.name();

            // Property: Name must be non-empty
            assert!(!name.is_empty(), "Backend {:?} name must not be empty", backend);

            // Property: Name should contain ".cpp" suffix (convention for C++ backends)
            assert!(
                name.ends_with(".cpp"),
                "Backend {:?} name '{}' should end with '.cpp' suffix",
                backend,
                name
            );

            // Property: Name should be lowercase (consistent naming)
            assert_eq!(
                name,
                name.to_lowercase(),
                "Backend {:?} name '{}' should be lowercase",
                backend,
                name
            );
        }
    }
}
