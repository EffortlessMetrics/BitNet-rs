//! BitNet.cpp Auto-Setup Parity Test Scaffolding
//!
//! This test suite validates the BitNet.cpp auto-setup functionality with parity
//! to llama.cpp setup, providing one-command clone, build, and environment configuration.
//!
//! **Specification Reference**: docs/specs/bitnet-cpp-auto-setup-parity.md
//!
//! ## Test Coverage (Acceptance Criteria AC1-AC22)
//!
//! ### Clone and Build (AC1-AC7)
//! - AC1: Detects missing BitNet.cpp installation
//! - AC2: Automatically clones from GitHub repository
//! - AC3: Runs build process (setup_env.py or CMake)
//! - AC4: Emits RPATH for both BitNet.cpp and vendored llama.cpp libraries
//! - AC5: Exports BITNET_CPP_DIR and BITNET_CROSSVAL_LIBDIR environment variables
//! - AC6: Multi-platform support (Linux, macOS, Windows)
//! - AC7: Idempotent updates (re-running setup detects existing build)
//!
//! ### Library Discovery (AC8-AC13)
//! - AC8: Rebuild options with --force and --clean flags
//! - AC9: Preserves user modifications unless --clean specified
//! - AC10: Fast-path message for existing builds
//! - AC11: BITNET_CPP_DIR controls installation directory
//! - AC12: BITNET_CROSSVAL_LIBDIR overrides auto-discovered paths
//! - AC13: Precedence: Explicit LIBDIR > Auto-discovery > Default
//!
//! ### RPATH Configuration (AC14-AC17)
//! - AC14: Same environment variable names for both backends
//! - AC15: --backend both option installs both backends
//! - AC16: RPATH merges paths for both BitNet.cpp and llama.cpp libraries
//! - AC17: Auto-discovery prioritizes BitNet.cpp over llama.cpp
//!
//! ### Shell Emission (AC18-AC20)
//! - AC18: Clear documentation on backend usage per command
//! - AC19: Missing dependency errors show installation commands
//! - AC20: Build failures show CMake log excerpt with troubleshooting link
//!
//! ### Error Handling (AC21-AC22)
//! - AC21: Library discovery failures suggest --verbose flag
//! - AC22: Platform-specific errors provide workarounds
//!
//! ## Testing Strategy
//!
//! - Feature-gated: Tests use `#[cfg(feature = "crossval-all")]` where applicable
//! - Environment isolation: `#[serial(bitnet_env)]` for environment variable tests
//! - Mock external commands: Use temporary directories for isolated filesystem ops
//! - Property-based testing: RPATH deduplication, path resolution correctness
//! - All tests marked `#[ignore]` until implementation complete (TDD scaffolding)
//!
//! ## Usage
//!
//! ```bash
//! # Run all tests (currently ignored, awaiting implementation)
//! cargo test -p xtask --test bitnet_cpp_auto_setup_tests
//!
//! # Run specific test category
//! cargo test -p xtask --test bitnet_cpp_auto_setup_tests clone_and_build
//!
//! # Run with ignored tests (during implementation)
//! cargo test -p xtask --test bitnet_cpp_auto_setup_tests -- --ignored --include-ignored
//! ```

use serial_test::serial;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

// ============================================================================
// Test Helpers and Utilities
// ============================================================================

/// RAII guard for environment variable management
///
/// Automatically restores environment variable state on drop, ensuring test isolation.
struct EnvGuard {
    key: String,
    old: Option<String>,
}

impl EnvGuard {
    /// Create a new environment variable guard, capturing current state
    fn new(key: &str) -> Self {
        let old = env::var(key).ok();
        Self { key: key.to_string(), old }
    }

    /// Clear the environment variable and return guard
    fn clear(key: &str) -> Self {
        let guard = Self::new(key);
        unsafe {
            env::remove_var(key);
        }
        guard
    }

    /// Set the environment variable to a new value
    fn set(&self, val: &str) {
        unsafe {
            env::set_var(&self.key, val);
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(ref v) = self.old {
                env::set_var(&self.key, v);
            } else {
                env::remove_var(&self.key);
            }
        }
    }
}

/// Find workspace root by walking up to .git directory
fn workspace_root() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while !path.join(".git").exists() {
        if !path.pop() {
            panic!("Could not find workspace root (no .git directory found)");
        }
    }
    path
}

/// Create a mock library file for testing library discovery
fn create_mock_lib(dir: &Path, name: &str) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let lib_path = dir.join(name);
    fs::write(lib_path, b"mock library content")?;
    Ok(())
}

/// Create a mock BitNet.cpp repository structure for testing
fn create_mock_bitnet_repo(repo: &Path) -> std::io::Result<()> {
    // Create directory structure
    fs::create_dir_all(repo.join("build/bin"))?;
    fs::create_dir_all(repo.join("build/lib"))?;
    fs::create_dir_all(repo.join("build/3rdparty/llama.cpp/build/bin"))?;
    fs::create_dir_all(repo.join("src"))?;

    // Create mock libraries
    #[cfg(target_os = "linux")]
    {
        create_mock_lib(&repo.join("build/bin"), "libbitnet.so")?;
        create_mock_lib(&repo.join("build/3rdparty/llama.cpp/build/bin"), "libllama.so")?;
        create_mock_lib(&repo.join("build/3rdparty/llama.cpp/build/bin"), "libggml.so")?;
    }

    #[cfg(target_os = "macos")]
    {
        create_mock_lib(&repo.join("build/bin"), "libbitnet.dylib")?;
        create_mock_lib(&repo.join("build/3rdparty/llama.cpp/build/bin"), "libllama.dylib")?;
        create_mock_lib(&repo.join("build/3rdparty/llama.cpp/build/bin"), "libggml.dylib")?;
    }

    #[cfg(target_os = "windows")]
    {
        create_mock_lib(&repo.join("build/bin"), "bitnet.dll")?;
        create_mock_lib(&repo.join("build/3rdparty/llama.cpp/build/bin"), "llama.dll")?;
        create_mock_lib(&repo.join("build/3rdparty/llama.cpp/build/bin"), "ggml.dll")?;
    }

    // Create mock setup_env.py
    fs::write(repo.join("setup_env.py"), "#!/usr/bin/env python3\n# Mock setup script\n")?;

    // Create mock CMakeLists.txt
    fs::write(
        repo.join("CMakeLists.txt"),
        "cmake_minimum_required(VERSION 3.18)\nproject(bitnet)\n",
    )?;

    Ok(())
}

// ============================================================================
// AC1-AC7: Clone and Build Tests
// ============================================================================

/// AC1: Command detects missing BitNet.cpp installation
///
/// **Test**: Run `setup-cpp-auto --backend bitnet` with no existing installation
/// **Expected**: Automatically triggers clone and build workflow
/// **Tag**: `// AC:AC1`
#[test]
#[ignore] // AC:AC1
fn test_ac1_detects_missing_installation() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");

    // Ensure directory does not exist
    assert!(!bitnet_dir.exists(), "Bitnet directory should not exist initially");

    // TODO: Implement setup-cpp-auto --backend bitnet logic
    // Expected behavior:
    // 1. Check if bitnet_dir exists
    // 2. If not exists, trigger clone workflow
    // 3. Verify clone was initiated

    unimplemented!(
        "AC1: Detection logic for missing BitNet.cpp installation not yet implemented. \
         Expected: setup-cpp-auto --backend bitnet detects missing repo and triggers clone."
    );
}

/// AC2: Automatically clones from BitNet repository
///
/// **Test**: Verify `git clone https://github.com/microsoft/BitNet` executed
/// **Expected**: Repository cloned to `~/.cache/bitnet_cpp` (or `$BITNET_CPP_DIR`)
/// **Tag**: `// AC:AC2`
#[test]
#[ignore] // AC:AC2
fn test_ac2_clones_from_github() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");

    // TODO: Implement git clone logic in install_or_update_bitnet_cpp()
    // Expected behavior:
    // 1. Execute: git clone --recurse-submodules https://github.com/microsoft/BitNet <bitnet_dir>
    // 2. Verify .git directory exists in bitnet_dir
    // 3. Verify 3rdparty/llama.cpp submodule initialized

    unimplemented!(
        "AC2: Git clone logic not yet implemented. \
         Expected: git clone --recurse-submodules https://github.com/microsoft/BitNet to BITNET_CPP_DIR. \
         Verification: Check .git directory and submodules."
    );
}

/// AC3: Runs build process (setup_env.py or CMake)
///
/// **Test**: Verify `python3 setup_env.py` or CMake build executed
/// **Expected**: Libraries built in `build/` directory
/// **Tag**: `// AC:AC3`
#[test]
#[ignore] // AC:AC3
fn test_ac3_runs_build_process() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");
    create_mock_bitnet_repo(&bitnet_dir).expect("Failed to create mock repo");

    // TODO: Implement build process in install_or_update_bitnet_cpp()
    // Expected behavior:
    // 1. Check for setup_env.py
    // 2. If exists, run: python3 setup_env.py
    // 3. Else, run: mkdir build && cd build && cmake .. && cmake --build .
    // 4. Verify build/bin/ contains libraries

    unimplemented!(
        "AC3: Build process not yet implemented. \
         Expected: Run setup_env.py (preferred) or CMake fallback. \
         Verification: Check build/bin/ for libbitnet.so, build/3rdparty/llama.cpp/build/bin/ for libllama.so."
    );
}

/// AC4: Emits RPATH for both BitNet.cpp and vendored llama.cpp libraries
///
/// **Test**: Inspect emitted shell exports for RPATH entries
/// **Expected**: Both `build/bin` and `build/3rdparty/llama.cpp/build/bin` in RPATH
/// **Tag**: `// AC:AC4`
#[test]
#[ignore] // AC:AC4
fn test_ac4_emits_rpath_for_both_backends() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");
    create_mock_bitnet_repo(&bitnet_dir).expect("Failed to create mock repo");

    // TODO: Implement find_bitnet_lib_dirs() to discover both BitNet and vendored llama libraries
    // Expected behavior:
    // 1. Discover build/bin/ (BitNet.cpp output)
    // 2. Discover build/3rdparty/llama.cpp/build/bin/ (vendored llama.cpp)
    // 3. Merge paths using merge_and_deduplicate()
    // 4. Emit merged RPATH via emit_exports()

    unimplemented!(
        "AC4: RPATH emission for dual libraries not yet implemented. \
         Expected: find_bitnet_lib_dirs() returns [build/bin, build/3rdparty/llama.cpp/build/bin]. \
         Verification: Check emitted LD_LIBRARY_PATH contains both paths."
    );
}

/// AC5: Exports environment variables
///
/// **Test**: Eval emitted shell exports and check vars
/// **Expected**: `BITNET_CPP_DIR` and `BITNET_CROSSVAL_LIBDIR` set correctly
/// **Tag**: `// AC:AC5`
#[test]
#[ignore] // AC:AC5
fn test_ac5_exports_environment_variables() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");
    create_mock_bitnet_repo(&bitnet_dir).expect("Failed to create mock repo");

    // TODO: Implement environment variable emission in emit_exports()
    // Expected behavior:
    // 1. Emit BITNET_CPP_DIR=<repo_path>
    // 2. Emit BITNET_CROSSVAL_LIBDIR=<primary_lib_dir>
    // 3. Emit LD_LIBRARY_PATH (Linux) / DYLD_LIBRARY_PATH (macOS) / PATH (Windows)

    unimplemented!(
        "AC5: Environment variable exports not yet implemented. \
         Expected: emit_exports() outputs BITNET_CPP_DIR, BITNET_CROSSVAL_LIBDIR, and loader path. \
         Verification: Parse emitted shell exports and verify variable presence."
    );
}

/// AC6: Multi-platform support
///
/// **Test**: Run on Linux, macOS, Windows (with Git Bash or PowerShell)
/// **Expected**: Successful build on all platforms with platform-appropriate env vars
/// **Tag**: `// AC:AC6`
#[test]
#[ignore] // AC:AC6
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn test_ac6_multi_platform_support() {
    // TODO: Implement platform-specific logic in emit_exports()
    // Expected behavior:
    // - Linux: LD_LIBRARY_PATH
    // - macOS: DYLD_LIBRARY_PATH
    // - Windows: PATH
    // Verification: Check emitted exports use correct platform variable

    unimplemented!(
        "AC6: Platform-specific support not yet fully tested. \
         Expected: emit_exports() uses LD_LIBRARY_PATH (Linux), DYLD_LIBRARY_PATH (macOS), PATH (Windows). \
         Verification: Run on each platform and verify correct variable emitted."
    );
}

/// AC7: Idempotent updates
///
/// **Test**: Run `setup-cpp-auto --backend bitnet` twice consecutively
/// **Expected**: Second run detects existing build, fast-path success
/// **Tag**: `// AC:AC7`
#[test]
#[ignore] // AC:AC7
fn test_ac7_idempotent_updates() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");
    create_mock_bitnet_repo(&bitnet_dir).expect("Failed to create mock repo");

    // TODO: Implement idempotency check in install_or_update_bitnet_cpp()
    // Expected behavior:
    // 1. First run: Clone and build
    // 2. Second run (no --force): Detect existing build, skip clone/build
    // 3. Verify "already built" message emitted
    // 4. Verify fast-path returns early with success

    unimplemented!(
        "AC7: Idempotency logic not yet implemented. \
         Expected: Second run detects existing build/bin/ and skips clone/build. \
         Verification: Check for 'BitNet.cpp already built at <path>' message."
    );
}

// ============================================================================
// AC8-AC13: Library Discovery and Environment Variable Precedence
// ============================================================================

/// AC8: Rebuild options with --force and --clean flags
///
/// **Test**: Run with `--force` or `--clean` flags
/// **Expected**: Existing build cleaned and rebuilt
/// **Tag**: `// AC:AC8`
#[test]
#[ignore] // AC:AC8
fn test_ac8_rebuild_options() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");
    create_mock_bitnet_repo(&bitnet_dir).expect("Failed to create mock repo");

    // TODO: Implement --force and --clean flag handling
    // Expected behavior:
    // 1. --force: Re-run build even if already built (git pull + rebuild)
    // 2. --clean: Remove build/ directory, then rebuild
    // Verification: Check build directory recreated

    unimplemented!(
        "AC8: --force and --clean flags not yet implemented. \
         Expected: --force triggers rebuild, --clean removes build/ first. \
         Verification: Check build directory state before/after."
    );
}

/// AC9: Preserves user modifications unless --clean specified
///
/// **Test**: Modify CMake cache, re-run setup
/// **Expected**: User flags preserved unless `--clean` specified
/// **Tag**: `// AC:AC9`
#[test]
#[ignore] // AC:AC9
fn test_ac9_preserves_user_modifications() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");
    create_mock_bitnet_repo(&bitnet_dir).expect("Failed to create mock repo");

    // Create mock CMakeCache.txt with user modifications
    let cache_path = bitnet_dir.join("build/CMakeCache.txt");
    fs::write(&cache_path, "USER_CUSTOM_FLAG:STRING=value\n").expect("Failed to write CMakeCache");

    // TODO: Implement preservation logic
    // Expected behavior:
    // 1. Re-run without --clean: CMakeCache.txt preserved
    // 2. Re-run with --clean: CMakeCache.txt removed and regenerated

    unimplemented!(
        "AC9: User modification preservation not yet implemented. \
         Expected: Re-running without --clean preserves CMakeCache.txt. \
         Verification: Check CMakeCache.txt contents before/after."
    );
}

/// AC10: Fast-path message for existing builds
///
/// **Test**: Run on existing build without flags
/// **Expected**: "BitNet.cpp already built at <path>" message, no rebuild
/// **Tag**: `// AC:AC10`
#[test]
#[ignore] // AC:AC10
fn test_ac10_fast_path_message() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");
    create_mock_bitnet_repo(&bitnet_dir).expect("Failed to create mock repo");

    // TODO: Implement fast-path detection and messaging
    // Expected behavior:
    // 1. Check if build/ exists and contains libraries
    // 2. Emit: "[bitnet] BitNet.cpp already built at <path>"
    // 3. Exit early without re-running build

    unimplemented!(
        "AC10: Fast-path messaging not yet implemented. \
         Expected: Emit 'BitNet.cpp already built' message when build exists. \
         Verification: Capture stderr and check for message."
    );
}

/// AC11: BITNET_CPP_DIR controls installation directory
///
/// **Test**: Set BITNET_CPP_DIR to custom path
/// **Expected**: Installation uses specified directory
/// **Tag**: `// AC:AC11`
#[test]
#[serial(bitnet_env)]
#[ignore] // AC:AC11
fn test_ac11_bitnet_cpp_dir_controls_installation() {
    let _guard = EnvGuard::clear("BITNET_CPP_DIR");

    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let custom_dir = temp_dir.path().join("custom_bitnet");

    let _guard_custom = EnvGuard::new("BITNET_CPP_DIR");
    _guard_custom.set(custom_dir.to_str().unwrap());

    // TODO: Verify BITNET_CPP_DIR precedence
    // Expected behavior:
    // 1. Read BITNET_CPP_DIR environment variable
    // 2. Use specified path instead of default ~/.cache/bitnet_cpp
    // 3. Clone and build to custom directory

    unimplemented!(
        "AC11: BITNET_CPP_DIR precedence not yet tested. \
         Expected: setup-cpp-auto uses BITNET_CPP_DIR value instead of default. \
         Verification: Check installation directory matches env var."
    );
}

/// AC12: BITNET_CROSSVAL_LIBDIR overrides auto-discovered paths
///
/// **Test**: Set BITNET_CROSSVAL_LIBDIR explicitly
/// **Expected**: Uses specified path instead of auto-discovery
/// **Tag**: `// AC:AC12`
#[test]
#[serial(bitnet_env)]
#[ignore] // AC:AC12
fn test_ac12_crossval_libdir_override() {
    let _guard = EnvGuard::clear("BITNET_CROSSVAL_LIBDIR");

    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let custom_lib = temp_dir.path().join("custom_lib");
    fs::create_dir_all(&custom_lib).expect("Failed to create custom lib dir");

    let _guard_custom = EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
    _guard_custom.set(custom_lib.to_str().unwrap());

    // TODO: Verify BITNET_CROSSVAL_LIBDIR override logic
    // Expected behavior:
    // 1. Check for BITNET_CROSSVAL_LIBDIR environment variable
    // 2. If set, use specified path instead of auto-discovery
    // 3. Emit specified path in shell exports

    unimplemented!(
        "AC12: BITNET_CROSSVAL_LIBDIR override not yet implemented. \
         Expected: Explicit BITNET_CROSSVAL_LIBDIR takes precedence over auto-discovery. \
         Verification: Check emitted BITNET_CROSSVAL_LIBDIR matches env var."
    );
}

/// AC13: Precedence order validation
///
/// **Test**: Test Explicit LIBDIR > Auto-discovery > Default precedence chain
/// **Expected**: Correct precedence order enforced
/// **Tag**: `// AC:AC13`
#[test]
#[serial(bitnet_env)]
#[ignore] // AC:AC13
fn test_ac13_precedence_order() {
    // TODO: Implement comprehensive precedence tests
    // Precedence chain:
    // 1. Explicit BITNET_CROSSVAL_LIBDIR (highest)
    // 2. Auto-discovery from BITNET_CPP_DIR/build/
    // 3. Default ~/.cache/bitnet_cpp/build/bin

    unimplemented!(
        "AC13: Precedence order validation not yet comprehensive. \
         Expected: Explicit > Auto-discovery > Default. \
         Verification: Test all three tiers and verify correct selection."
    );
}

// ============================================================================
// AC14-AC17: RPATH Configuration and Dual-Backend Support
// ============================================================================

/// AC14: Same environment variable names for both backends
///
/// **Test**: Verify consistent naming between BitNet.cpp and llama.cpp
/// **Expected**: BITNET_CPP_DIR, BITNET_CROSSVAL_LIBDIR used for both
/// **Tag**: `// AC:AC14`
#[test]
#[ignore] // AC:AC14
fn test_ac14_consistent_env_var_names() {
    // TODO: Document and verify environment variable naming consistency
    // Expected behavior:
    // - BITNET_CPP_DIR used for both BitNet.cpp and llama.cpp root
    // - BITNET_CROSSVAL_LIBDIR used for library discovery override
    // - No separate variables for different backends

    unimplemented!(
        "AC14: Environment variable naming consistency documented but not enforced in tests. \
         Expected: Same variables (BITNET_CPP_DIR, BITNET_CROSSVAL_LIBDIR) for both backends. \
         Verification: Check documentation and code uses consistent names."
    );
}

/// AC15: --backend both option installs both backends
///
/// **Test**: Run `setup-cpp-auto --backend both`
/// **Expected**: Both BitNet.cpp and llama.cpp installed
/// **Tag**: `// AC:AC15`
#[test]
#[ignore] // AC:AC15
fn test_ac15_backend_both_option() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let install_dir = temp_dir.path();

    // TODO: Implement --backend both logic
    // Expected behavior:
    // 1. Install BitNet.cpp (via install_or_update_bitnet_cpp)
    // 2. Install llama.cpp (via install_or_update_llama_cpp)
    // 3. Merge library paths from both backends
    // 4. Emit unified RPATH

    unimplemented!(
        "AC15: --backend both option not yet implemented. \
         Expected: Installs both BitNet.cpp and llama.cpp, merges RPATHs. \
         Verification: Check both repositories cloned and libraries discovered."
    );
}

/// AC16: RPATH merges paths for both backends
///
/// **Test**: Verify merged RPATH contains paths from both BitNet.cpp and llama.cpp
/// **Expected**: Merged RPATH with deduplication
/// **Tag**: `// AC:AC16`
#[test]
#[ignore] // AC:AC16
fn test_ac16_rpath_merging() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_lib = temp_dir.path().join("bitnet_lib");
    let llama_lib = temp_dir.path().join("llama_lib");

    fs::create_dir_all(&bitnet_lib).expect("Failed to create bitnet_lib");
    fs::create_dir_all(&llama_lib).expect("Failed to create llama_lib");

    // TODO: Implement RPATH merging logic
    // Expected behavior:
    // 1. Collect library paths from BitNet.cpp backend
    // 2. Collect library paths from llama.cpp backend
    // 3. Merge using merge_and_deduplicate() from build_helpers
    // 4. Emit unified RPATH via emit_exports()

    unimplemented!(
        "AC16: RPATH merging for dual backends not yet implemented. \
         Expected: merge_and_deduplicate() combines paths from both backends. \
         Verification: Check emitted LD_LIBRARY_PATH contains both paths."
    );
}

/// AC17: Auto-discovery prioritizes BitNet.cpp over llama.cpp
///
/// **Test**: When both backends present, verify BitNet.cpp paths come first
/// **Expected**: BitNet.cpp library paths before llama.cpp in RPATH
/// **Tag**: `// AC:AC17`
#[test]
#[ignore] // AC:AC17
fn test_ac17_autodiscovery_priority() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");
    create_mock_bitnet_repo(&bitnet_dir).expect("Failed to create mock bitnet repo");

    // TODO: Implement priority ordering in library discovery
    // Expected behavior:
    // 1. Discover BitNet.cpp libraries first (build/bin)
    // 2. Discover vendored llama.cpp libraries second (build/3rdparty/llama.cpp/build/bin)
    // 3. Preserve ordering in merged RPATH

    unimplemented!(
        "AC17: Priority ordering not yet enforced. \
         Expected: BitNet.cpp paths appear before llama.cpp paths in RPATH. \
         Verification: Parse RPATH and verify ordering."
    );
}

// ============================================================================
// AC18-AC22: Error Handling and User Experience
// ============================================================================

/// AC18: Clear documentation on backend usage
///
/// **Test**: Verify documentation clarity for backend selection
/// **Expected**: Users understand which backend is used for each command
/// **Tag**: `// AC:AC18`
#[test]
#[ignore] // AC:AC18
fn test_ac18_clear_documentation() {
    // TODO: Validate documentation completeness
    // Expected documentation:
    // - setup-cpp-auto --backend bitnet: Install BitNet.cpp
    // - setup-cpp-auto --backend llama: Install llama.cpp
    // - setup-cpp-auto --backend both: Install both
    // - crossval-per-token: Auto-detects backend from model path

    unimplemented!(
        "AC18: Documentation validation not yet automated. \
         Expected: Comprehensive docs in CLAUDE.md and docs/howto/cpp-setup.md. \
         Verification: Manual review of documentation completeness."
    );
}

/// AC19: Missing dependency errors show installation commands
///
/// **Test**: Run setup with missing dependency (git, cmake, python3)
/// **Expected**: Error message shows platform-specific installation commands
/// **Tag**: `// AC:AC19`
#[test]
#[ignore] // AC:AC19
fn test_ac19_missing_dependency_errors() {
    // TODO: Implement check_dependency() with helpful error messages
    // Expected error format:
    // ```
    // Error: git executable not found
    //
    // Installation:
    //   Ubuntu/Debian:  sudo apt install git
    //   macOS:          brew install git
    //   Windows:        choco install git
    // ```

    unimplemented!(
        "AC19: Dependency check error messages not yet implemented. \
         Expected: check_dependency() emits platform-specific installation commands. \
         Verification: Simulate missing git/cmake/python3 and check error message."
    );
}

/// AC20: Build failures show CMake log excerpt
///
/// **Test**: Simulate CMake build failure
/// **Expected**: Error shows last 20 lines of CMake log + troubleshooting link
/// **Tag**: `// AC:AC20`
#[test]
#[ignore] // AC:AC20
fn test_ac20_build_failure_errors() {
    // TODO: Implement build failure error handling
    // Expected error format:
    // ```
    // Error: BitNet.cpp build failed during CMake configuration
    //
    // CMake output (last 20 lines):
    //   <log excerpt>
    //
    // Troubleshooting:
    //   1. For CPU-only build: --cmake-flags "-DGGML_CUDA=OFF"
    //   2. Full log: <path>/build/CMakeFiles/CMakeOutput.log
    //
    // See: docs/howto/cpp-setup.md#troubleshooting-build-failures
    // ```

    unimplemented!(
        "AC20: Build failure error messages not yet comprehensive. \
         Expected: Show CMake log excerpt + troubleshooting steps. \
         Verification: Simulate build failure and check error output."
    );
}

/// AC21: Library discovery failures suggest --verbose
///
/// **Test**: Simulate library discovery failure
/// **Expected**: Error message suggests --verbose flag for debugging
/// **Tag**: `// AC:AC21`
#[test]
#[ignore] // AC:AC21
fn test_ac21_library_discovery_failures() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");

    // Create repo structure but no libraries
    fs::create_dir_all(bitnet_dir.join("build/bin")).expect("Failed to create build dir");

    // TODO: Implement library discovery error handling
    // Expected error format:
    // ```
    // Error: No BitNet.cpp libraries found after build
    //
    // Expected libraries:
    //   - libbitnet.so (or .dylib/.dll)
    //   - libllama.so (vendored llama.cpp)
    //
    // Searched directories:
    //   ✗ build/bin (not found)
    //   ✗ build/3rdparty/llama.cpp/build/bin (not found)
    //
    // Debugging steps:
    //   1. Re-run with --verbose
    //   2. Check build logs
    // ```

    unimplemented!(
        "AC21: Library discovery error messages not yet helpful. \
         Expected: Suggest --verbose flag and show searched directories. \
         Verification: Simulate missing libraries and check error message."
    );
}

/// AC22: Platform-specific errors provide workarounds
///
/// **Test**: Simulate platform-specific error (e.g., CUDA not found on Linux)
/// **Expected**: Error shows platform-appropriate workaround
/// **Tag**: `// AC:AC22`
#[test]
#[ignore] // AC:AC22
#[cfg(target_os = "linux")]
fn test_ac22_platform_specific_errors() {
    // TODO: Implement platform-specific error handling
    // Example: CUDA not found error
    // Expected error format:
    // ```
    // Error: CUDA toolkit not found
    //
    // Workarounds:
    //   1. CPU-only build: --cmake-flags "-DGGML_CUDA=OFF"
    //   2. Install CUDA: https://developer.nvidia.com/cuda-downloads
    //   3. Check PATH: /usr/local/cuda/bin should be in PATH
    // ```

    unimplemented!(
        "AC22: Platform-specific error messages not yet comprehensive. \
         Expected: Provide platform-appropriate workarounds (Linux: apt, macOS: brew, Windows: choco). \
         Verification: Simulate platform-specific failures and check error messages."
    );
}

// ============================================================================
// Property-Based Tests: RPATH Deduplication and Path Resolution
// ============================================================================

/// Property test: RPATH deduplication is idempotent
///
/// Validates that applying deduplication multiple times yields same result.
#[test]
#[ignore] // Property-based test
fn test_property_rpath_deduplication_idempotent() {
    use xtask::build_helpers::merge_and_deduplicate;

    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let path1 = temp_dir.path().join("lib1");
    let path2 = temp_dir.path().join("lib2");

    fs::create_dir_all(&path1).expect("Failed to create lib1");
    fs::create_dir_all(&path2).expect("Failed to create lib2");

    let paths = vec![
        path1.to_str().unwrap(),
        path2.to_str().unwrap(),
        path1.to_str().unwrap(), // Duplicate
    ];

    // First merge
    let first = merge_and_deduplicate(&paths);

    // Extract paths and merge again (simulating idempotency)
    let parts: Vec<&str> = first.split(':').collect();
    let second = merge_and_deduplicate(&parts);

    // Idempotent property: merging already-merged paths yields same result
    assert_eq!(first, second, "RPATH deduplication should be idempotent");
}

/// Property test: Path resolution handles symlinks correctly
///
/// Validates that symlinks are canonicalized and deduplicated.
#[test]
#[ignore] // Property-based test
#[cfg(unix)] // Symlinks are Unix-specific
fn test_property_symlink_resolution() {
    use xtask::build_helpers::merge_and_deduplicate;

    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let real_path = temp_dir.path().join("real");
    let symlink_path = temp_dir.path().join("link");

    fs::create_dir_all(&real_path).expect("Failed to create real path");
    std::os::unix::fs::symlink(&real_path, &symlink_path).expect("Failed to create symlink");

    let paths = vec![real_path.to_str().unwrap(), symlink_path.to_str().unwrap()];

    let result = merge_and_deduplicate(&paths);

    // Property: Both paths resolve to same canonical path, so only one entry
    assert_eq!(
        result.matches(':').count(),
        0,
        "Symlink should deduplicate to single canonical path"
    );
}

/// Property test: RPATH ordering is preserved
///
/// Validates that insertion order is maintained after deduplication.
#[test]
#[ignore] // Property-based test
fn test_property_rpath_ordering_preserved() {
    use xtask::build_helpers::merge_and_deduplicate;

    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_lib = temp_dir.path().join("bitnet");
    let llama_lib = temp_dir.path().join("llama");

    fs::create_dir_all(&bitnet_lib).expect("Failed to create bitnet lib");
    fs::create_dir_all(&llama_lib).expect("Failed to create llama lib");

    let paths = vec![bitnet_lib.to_str().unwrap(), llama_lib.to_str().unwrap()];

    let result = merge_and_deduplicate(&paths);

    // Property: First path appears before second path in result
    let canonical_bitnet = bitnet_lib.canonicalize().unwrap().display().to_string();
    let canonical_llama = llama_lib.canonicalize().unwrap().display().to_string();

    assert!(result.starts_with(&canonical_bitnet), "BitNet path should appear first");
    assert!(result.contains(&format!(":{}", canonical_llama)), "llama path should appear second");
}

// ============================================================================
// Integration Tests: Shell Emission and Environment Variables
// ============================================================================

/// Integration test: Shell emission for sh/bash/zsh
///
/// Validates that emitted shell exports are syntactically correct for POSIX shells.
#[test]
#[ignore] // Integration test
fn test_shell_emission_sh() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");
    create_mock_bitnet_repo(&bitnet_dir).expect("Failed to create mock repo");

    // TODO: Implement emit_exports() for Emit::Sh
    // Expected output format:
    // ```sh
    // export BITNET_CPP_DIR="/path/to/bitnet_cpp"
    // export BITNET_CROSSVAL_LIBDIR="/path/to/bitnet_cpp/build/bin"
    // export LD_LIBRARY_PATH="/path1:/path2:${LD_LIBRARY_PATH:-}"
    // echo "[bitnet] C++ ready at $BITNET_CPP_DIR"
    // ```

    unimplemented!(
        "Shell emission for sh not yet tested. \
         Expected: Valid POSIX shell syntax with export statements. \
         Verification: Parse emitted output and validate syntax."
    );
}

/// Integration test: Shell emission for fish
///
/// Validates that emitted shell exports use fish syntax.
#[test]
#[ignore] // Integration test
fn test_shell_emission_fish() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");
    create_mock_bitnet_repo(&bitnet_dir).expect("Failed to create mock repo");

    // TODO: Implement emit_exports() for Emit::Fish
    // Expected output format:
    // ```fish
    // set -gx BITNET_CPP_DIR "/path/to/bitnet_cpp"
    // set -gx BITNET_CROSSVAL_LIBDIR "/path/to/bitnet_cpp/build/bin"
    // set -gx LD_LIBRARY_PATH "/path1:/path2" $LD_LIBRARY_PATH
    // echo "[bitnet] C++ ready at $BITNET_CPP_DIR"
    // ```

    unimplemented!(
        "Shell emission for fish not yet tested. \
         Expected: Valid fish shell syntax with set -gx statements. \
         Verification: Parse emitted output and validate fish syntax."
    );
}

/// Integration test: Shell emission for PowerShell
///
/// Validates that emitted shell exports use PowerShell syntax.
#[test]
#[ignore] // Integration test
#[cfg(target_os = "windows")]
fn test_shell_emission_pwsh() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");
    create_mock_bitnet_repo(&bitnet_dir).expect("Failed to create mock repo");

    // TODO: Implement emit_exports() for Emit::Pwsh
    // Expected output format:
    // ```powershell
    // $env:BITNET_CPP_DIR = "C:\path\to\bitnet_cpp"
    // $env:BITNET_CROSSVAL_LIBDIR = "C:\path\to\bitnet_cpp\build\bin"
    // $env:PATH = "C:\path1;C:\path2;" + $env:PATH
    // Write-Host "[bitnet] C++ ready at $env:BITNET_CPP_DIR"
    // ```

    unimplemented!(
        "Shell emission for PowerShell not yet tested. \
         Expected: Valid PowerShell syntax with $env: assignments. \
         Verification: Parse emitted output and validate PowerShell syntax."
    );
}

/// Integration test: Shell emission for Windows cmd
///
/// Validates that emitted shell exports use batch script syntax.
#[test]
#[ignore] // Integration test
#[cfg(target_os = "windows")]
fn test_shell_emission_cmd() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let bitnet_dir = temp_dir.path().join("bitnet_cpp");
    create_mock_bitnet_repo(&bitnet_dir).expect("Failed to create mock repo");

    // TODO: Implement emit_exports() for Emit::Cmd
    // Expected output format:
    // ```batch
    // set BITNET_CPP_DIR=C:\path\to\bitnet_cpp
    // set BITNET_CROSSVAL_LIBDIR=C:\path\to\bitnet_cpp\build\bin
    // set PATH=C:\path1;C:\path2;%PATH%
    // echo [bitnet] C++ ready at %BITNET_CPP_DIR%
    // ```

    unimplemented!(
        "Shell emission for cmd not yet tested. \
         Expected: Valid batch script syntax with set statements. \
         Verification: Parse emitted output and validate batch syntax."
    );
}
