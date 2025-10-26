# llama.cpp Auto-Setup Integration Specification

**Document Version**: 1.0.0
**Status**: Draft for Review
**Created**: 2025-10-26
**Target Release**: v0.2.0
**Feature**: Dual-Backend Cross-Validation Infrastructure

---

## Executive Summary

### Problem Statement

BitNet.rs currently supports automatic setup of the BitNet.cpp C++ reference backend via the `setup-cpp-auto` command. However, the dual-backend cross-validation architecture requires **both** BitNet.cpp (for BitNet models) and llama.cpp (for LLaMA/Mistral models) to be available. Users must manually install llama.cpp, which creates friction and inconsistent environments.

**Current Pain Points**:
1. **Manual llama.cpp Installation**: Users must clone, build, and configure llama.cpp separately
2. **Environment Variable Management**: No unified mechanism to export `LLAMA_CPP_DIR`, `LD_LIBRARY_PATH`
3. **Library Discovery Gaps**: llama.cpp requires BOTH `libllama.so` + `libggml.so` (vs BitNet's single lib)
4. **Platform Inconsistency**: No CMake flag standardization across backends
5. **Preflight Failures**: Tests fail silently when llama.cpp unavailable

### Desired Behavior

Extend `setup-cpp-auto` command to support llama.cpp backend with identical UX:

```bash
# Install both backends (default)
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Install specific backend
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh --backend llama)"
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh --backend bitnet)"

# Verify installation
cargo run -p xtask --features crossval-all -- preflight --backend llama
```

**Expected Output**:
```
[llama] Cloning from https://github.com/ggerganov/llama.cpp...
[llama] Clone succeeded
[llama] Building llama.cpp with CMake...
[llama] llama.cpp build succeeded
[llama] Verifying built libraries: libllama.so, libggml.so
[llama] Library verification passed

export LLAMA_CPP_DIR="/home/user/.cache/llama_cpp"
export LD_LIBRARY_PATH="/home/user/.cache/llama_cpp/build/lib:${LD_LIBRARY_PATH:-}"
echo "[llama] C++ ready at $LLAMA_CPP_DIR"
```

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Setup Success Rate** | ≥95% (all platforms) | Install completes without errors |
| **Library Discovery** | 100% (both libs found) | `libllama.so` + `libggml.so` detected |
| **Cross-Validation** | ≥99% (cos_sim) | crossval-per-token parity metrics |
| **Build Time** | ≤8 minutes | Time from clone to ready |
| **User Friction** | ≤2 commands | setup-cpp-auto + eval |

---

## Acceptance Criteria

### AC1: Backend Flag Support

**Requirement**: `setup-cpp-auto` accepts `--backend llama` flag

**Test Scenario**:
```bash
cargo run -p xtask -- setup-cpp-auto --emit=sh --backend llama
```

**Expected Behavior**:
- Command executes without error
- Only llama.cpp is cloned/built (BitNet.cpp not touched)
- Shell exports contain `LLAMA_CPP_DIR` (not `BITNET_CPP_DIR`)

**Test Code**:
```rust
// AC:AC1
#[test]
fn test_setup_cpp_auto_backend_llama_flag() {
    let output = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--", "setup-cpp-auto", "--emit=sh", "--backend=llama"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "Command should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("LLAMA_CPP_DIR"), "Should export LLAMA_CPP_DIR");
    assert!(!stdout.contains("BITNET_CPP_DIR"), "Should not export BITNET_CPP_DIR");
}
```

### AC2: LLAMA_CPP_DIR Environment Variable

**Requirement**: Support `LLAMA_CPP_DIR` for explicit installation path override

**Test Scenario**:
```bash
export LLAMA_CPP_DIR=/opt/llama-cpp
cargo run -p xtask -- setup-cpp-auto --emit=sh --backend llama
```

**Expected Behavior**:
- Installation uses `/opt/llama-cpp` instead of `~/.cache/llama_cpp`
- Shell exports reflect custom path
- No conflict with `BITNET_CPP_DIR`

**Test Code**:
```rust
// AC:AC2
#[test]
#[serial(bitnet_env)]
fn test_llama_cpp_dir_env_var_override() {
    let _guard = EnvGuard::new("LLAMA_CPP_DIR", "/tmp/custom-llama");

    let install_dir = determine_install_dir(CppBackend::Llama).unwrap();

    assert_eq!(install_dir, PathBuf::from("/tmp/custom-llama"));
}
```

### AC3: CMake-Only Build (No Python Wrapper)

**Requirement**: llama.cpp builds using CMake directly (no `setup_env.py`)

**Test Scenario**:
```bash
# Should NOT use setup_env.py (doesn't exist in llama.cpp)
cargo run -p xtask -- setup-cpp-auto --backend llama
```

**Expected Behavior**:
- Build process calls `cmake` + `cmake --build` directly
- No Python dependency check
- No fallback to `setup_env.py` (function not called)

**Test Code**:
```rust
// AC:AC3
#[test]
fn test_llama_cpp_cmake_only_build() {
    let temp_dir = TempDir::new().unwrap();
    let install_dir = temp_dir.path();

    // Ensure setup_env.py does NOT exist
    assert!(!install_dir.join("setup_env.py").exists());

    // Build should succeed without setup_env.py
    let result = build_llama_cpp(install_dir);

    // Should call run_cmake_build, not run_setup_env_py
    assert!(result.is_ok(), "CMake-only build should succeed");
}
```

### AC4: Dual Library Discovery (libllama + libggml)

**Requirement**: Both `libllama.so` AND `libggml.so` must be present

**Test Scenario**:
```bash
# After build, verify both libraries exist
ls ~/.cache/llama_cpp/build/lib/libllama.so
ls ~/.cache/llama_cpp/build/lib/libggml.so
```

**Expected Behavior**:
- Preflight check fails if only one library present
- Discovery returns directory only if BOTH libs found
- Error message lists missing libraries

**Test Code**:
```rust
// AC:AC4
#[test]
fn test_llama_cpp_requires_both_libraries() {
    let temp_dir = TempDir::new().unwrap();
    let lib_dir = temp_dir.path();

    // Create only libllama.so (missing libggml.so)
    File::create(lib_dir.join("libllama.so")).unwrap();

    // Should NOT be considered complete
    let result = has_all_libraries(lib_dir, &["libllama", "libggml"]);
    assert!(!result, "Should require BOTH libraries");

    // Now add libggml.so
    File::create(lib_dir.join("libggml.so")).unwrap();

    // Now should pass
    let result = has_all_libraries(lib_dir, &["libllama", "libggml"]);
    assert!(result, "Both libraries present");
}
```

### AC5: Three-Tier Search Hierarchy

**Requirement**: Library discovery follows priority order:

1. **Tier 0**: `LLAMA_CROSSVAL_LIBDIR` (explicit override)
2. **Tier 1**: Primary CMake outputs (`build/bin`, `build/lib`)
3. **Tier 2**: Build root (`build/`)
4. **Tier 3**: Fallback (`lib/`)

**Test Scenario**:
```bash
# Tier 1: Primary paths checked first
export LLAMA_CPP_DIR=~/.cache/llama_cpp
cargo run -p xtask --features crossval-all -- preflight --backend llama
```

**Expected Behavior**:
- Tier 0 override returns immediately if valid
- Tier 1 checked before Tier 2
- Tier 3 only checked if Tiers 1-2 fail

**Test Code**:
```rust
// AC:AC5
#[test]
#[serial(bitnet_env)]
fn test_llama_cpp_search_hierarchy_precedence() {
    let temp_dir = TempDir::new().unwrap();
    let install_dir = temp_dir.path();

    // Create libraries in Tier 2 (build root)
    let tier2_dir = install_dir.join("build");
    fs::create_dir_all(&tier2_dir).unwrap();
    create_mock_libs(&tier2_dir, &["libllama", "libggml"]);

    // Create libraries in Tier 1 (build/lib - higher priority)
    let tier1_dir = install_dir.join("build/lib");
    fs::create_dir_all(&tier1_dir).unwrap();
    create_mock_libs(&tier1_dir, &["libllama", "libggml"]);

    // Should prefer Tier 1 over Tier 2
    let lib_dirs = find_backend_lib_dirs(install_dir, CppBackend::Llama).unwrap();

    assert_eq!(lib_dirs.len(), 1, "Should find exactly one directory");
    assert_eq!(lib_dirs[0], tier1_dir, "Should prefer Tier 1 (build/lib)");
}
```

### AC6: Shell Export Emitters for All Platforms

**Requirement**: Generate correct shell exports for sh, fish, pwsh, cmd

**Test Scenario**:
```bash
# POSIX shells
cargo run -p xtask -- setup-cpp-auto --emit=sh --backend llama

# fish shell
cargo run -p xtask -- setup-cpp-auto --emit=fish --backend llama

# PowerShell
cargo run -p xtask -- setup-cpp-auto --emit=pwsh --backend llama

# Windows cmd
cargo run -p xtask -- setup-cpp-auto --emit=cmd --backend llama
```

**Expected Output**:

**sh**:
```bash
export LLAMA_CPP_DIR="/home/user/.cache/llama_cpp"
export LD_LIBRARY_PATH="/home/user/.cache/llama_cpp/build/lib:${LD_LIBRARY_PATH:-}"
echo "[llama] C++ ready at $LLAMA_CPP_DIR"
```

**fish**:
```fish
set -gx LLAMA_CPP_DIR "/home/user/.cache/llama_cpp"
set -gx LD_LIBRARY_PATH "/home/user/.cache/llama_cpp/build/lib" $LD_LIBRARY_PATH
echo "[llama] C++ ready at $LLAMA_CPP_DIR"
```

**pwsh**:
```powershell
$env:LLAMA_CPP_DIR = "/home/user/.cache/llama_cpp"
$env:PATH = "/home/user/.cache/llama_cpp/build/lib;" + $env:PATH
Write-Host "[llama] C++ ready at $env:LLAMA_CPP_DIR"
```

**cmd**:
```batch
set LLAMA_CPP_DIR=C:\Users\user\.cache\llama_cpp
set PATH=C:\Users\user\.cache\llama_cpp\build\lib;%PATH%
echo [llama] C++ ready at %LLAMA_CPP_DIR%
```

**Test Code**:
```rust
// AC:AC6
#[test]
fn test_llama_cpp_emit_exports_all_shells() {
    let repo = PathBuf::from("/test/llama_cpp");
    let lib_dir = PathBuf::from("/test/llama_cpp/build/lib");

    // Test POSIX sh
    let output = capture_emit_exports(Emit::Sh, &repo, &lib_dir, None);
    assert!(output.contains(r#"export LLAMA_CPP_DIR="/test/llama_cpp""#));
    assert!(output.contains("LD_LIBRARY_PATH") || output.contains("DYLD_LIBRARY_PATH"));

    // Test fish
    let output = capture_emit_exports(Emit::Fish, &repo, &lib_dir, None);
    assert!(output.contains(r#"set -gx LLAMA_CPP_DIR "/test/llama_cpp""#));

    // Test PowerShell
    let output = capture_emit_exports(Emit::Pwsh, &repo, &lib_dir, None);
    assert!(output.contains(r#"$env:LLAMA_CPP_DIR = "/test/llama_cpp""#));

    // Test cmd
    let output = capture_emit_exports(Emit::Cmd, &repo, &lib_dir, None);
    assert!(output.contains(r#"set LLAMA_CPP_DIR="#));
}
```

### AC7: File Lock Prevents Concurrent Corruption

**Requirement**: Implement file locking to prevent concurrent builds

**Test Scenario**:
```bash
# Terminal 1
cargo run -p xtask -- setup-cpp-auto --backend llama &

# Terminal 2 (immediately after)
cargo run -p xtask -- setup-cpp-auto --backend llama
```

**Expected Behavior**:
- First process acquires lock on `~/.cache/.llama_cpp.lock`
- Second process waits for lock or fails with clear message
- Lock released automatically on process exit

**Test Code**:
```rust
// AC:AC7
#[test]
#[serial(bitnet_env)]
fn test_llama_cpp_file_lock_prevents_concurrent_builds() {
    let temp_dir = TempDir::new().unwrap();
    let install_dir = temp_dir.path().join("llama_cpp");

    // Acquire lock in first process
    let _lock1 = acquire_build_lock(&install_dir, CppBackend::Llama).unwrap();

    // Second process should fail to acquire lock
    let result = acquire_build_lock(&install_dir, CppBackend::Llama);
    assert!(result.is_err(), "Should fail to acquire lock when held");

    // Lock should contain backend name in error message
    let err = result.unwrap_err();
    assert!(err.to_string().contains("another setup-cpp-auto may be running"));
}
```

### AC8: Network Retry with Exponential Backoff

**Requirement**: Retry git clone on transient network failures

**Test Scenario**:
```bash
# Simulate network failure (mock)
BITNET_TEST_NETWORK_FAIL=1 cargo run -p xtask -- setup-cpp-auto --backend llama
```

**Expected Behavior**:
- First attempt fails → Wait 1 second
- Second attempt fails → Wait 1.5 seconds
- Third attempt fails → Wait 2.25 seconds
- Fourth attempt succeeds → Proceed with build
- Max 5 retries, cap backoff at 60 seconds

**Test Code**:
```rust
// AC:AC8
#[test]
fn test_llama_cpp_network_retry_exponential_backoff() {
    let temp_dir = TempDir::new().unwrap();
    let dest = temp_dir.path().join("llama_cpp");

    // Mock git clone that fails 3 times, then succeeds
    let mut attempt_count = 0;
    let mock_clone = |_url: &str, _dest: &Path| -> Result<()> {
        attempt_count += 1;
        if attempt_count < 3 {
            bail!("Network timeout")
        } else {
            Ok(())
        }
    };

    // Should retry and eventually succeed
    let start = Instant::now();
    let result = clone_repository_with_retry(
        "https://github.com/ggerganov/llama.cpp",
        &dest,
        5, // max_retries
        mock_clone
    );
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Should eventually succeed");
    assert!(attempt_count == 3, "Should have made 3 attempts");

    // Backoff: 1s + 1.5s = 2.5s minimum
    assert!(elapsed >= Duration::from_secs(2), "Should have backed off");
}
```

### AC9: Rollback on Build Failure

**Requirement**: Restore previous installation on build failure

**Test Scenario**:
```bash
# Existing working installation
cargo run -p xtask -- setup-cpp-auto --backend llama
# ... now working ...

# Trigger build failure (e.g., disk full)
# Installation should be restored to previous state
```

**Expected Behavior**:
- Before build: Move `~/.cache/llama_cpp` → `~/.cache/llama_cpp.backup`
- Build fails: Restore `~/.cache/llama_cpp.backup` → `~/.cache/llama_cpp`
- Build succeeds: Delete `~/.cache/llama_cpp.backup`

**Test Code**:
```rust
// AC:AC9
#[test]
fn test_llama_cpp_rollback_on_build_failure() {
    let temp_dir = TempDir::new().unwrap();
    let install_dir = temp_dir.path().join("llama_cpp");

    // Create "existing" installation
    fs::create_dir_all(&install_dir).unwrap();
    let marker = install_dir.join("existing_install.txt");
    fs::write(&marker, "previous version").unwrap();

    // Simulate build failure
    let result = install_or_update_backend_transactional(
        CppBackend::Llama,
        Some(|| bail!("Build failed"))
    );

    // Should have failed
    assert!(result.is_err());

    // Original installation should be restored
    assert!(marker.exists(), "Original installation restored");
    assert_eq!(
        fs::read_to_string(&marker).unwrap(),
        "previous version",
        "Original content intact"
    );
}
```

### AC10: Platform-Specific Library Naming

**Requirement**: Handle `.so` (Linux), `.dylib` (macOS), `.dll` (Windows)

**Test Scenario**:
```bash
# Linux
ls ~/.cache/llama_cpp/build/lib/libllama.so
ls ~/.cache/llama_cpp/build/lib/libggml.so

# macOS
ls ~/.cache/llama_cpp/build/lib/libllama.dylib
ls ~/.cache/llama_cpp/build/lib/libggml.dylib

# Windows
dir %USERPROFILE%\.cache\llama_cpp\build\lib\llama.dll
dir %USERPROFILE%\.cache\llama_cpp\build\lib\ggml.dll
```

**Expected Behavior**:
- Library discovery matches platform-specific extensions
- Windows: No `lib` prefix (e.g., `llama.dll`, not `libllama.dll`)
- Symlinks handled correctly (e.g., `libllama.so.1` → `libllama.so`)

**Test Code**:
```rust
// AC:AC10
#[test]
fn test_llama_cpp_platform_specific_library_naming() {
    #[cfg(target_os = "linux")]
    {
        assert_eq!(format_lib_name("llama"), "libllama.so");
        assert_eq!(format_lib_name("ggml"), "libggml.so");
    }

    #[cfg(target_os = "macos")]
    {
        assert_eq!(format_lib_name("llama"), "libllama.dylib");
        assert_eq!(format_lib_name("ggml"), "libggml.dylib");
    }

    #[cfg(target_os = "windows")]
    {
        assert_eq!(format_lib_name("llama"), "llama.dll");
        assert_eq!(format_lib_name("ggml"), "ggml.dll");
    }
}
```

### AC11: Integration with RPATH Merge

**Requirement**: llama.cpp paths added to RPATH alongside BitNet.cpp

**Test Scenario**:
```bash
export CROSSVAL_RPATH_BITNET=/path/to/bitnet/lib
export CROSSVAL_RPATH_LLAMA=/path/to/llama/lib
cargo build -p xtask --features crossval-all
```

**Expected Behavior**:
- build.rs detects both `CROSSVAL_RPATH_BITNET` and `CROSSVAL_RPATH_LLAMA`
- Merges paths with deduplication
- Emits single RPATH with both paths: `-Wl,-rpath:/path/to/bitnet/lib:/path/to/llama/lib`

**Test Code**:
```rust
// AC:AC11
#[test]
#[serial(bitnet_env)]
fn test_llama_cpp_rpath_merge_with_bitnet() {
    let _guard1 = EnvGuard::new("CROSSVAL_RPATH_BITNET", "/tmp/bitnet/lib");
    let _guard2 = EnvGuard::new("CROSSVAL_RPATH_LLAMA", "/tmp/llama/lib");

    // Simulate build.rs RPATH merging
    let merged = merge_and_deduplicate(&["/tmp/bitnet/lib", "/tmp/llama/lib"]);

    assert!(merged.contains("/tmp/bitnet/lib"), "Should include BitNet path");
    assert!(merged.contains("/tmp/llama/lib"), "Should include llama path");
    assert_eq!(merged.matches(':').count(), 1, "Should have 1 separator (2 paths)");
}
```

### AC12: Preflight Verification

**Requirement**: `preflight --backend llama` checks llama.cpp availability

**Test Scenario**:
```bash
# After setup
cargo run -p xtask --features crossval-all -- preflight --backend llama --verbose
```

**Expected Output**:
```
[preflight] Checking llama.cpp availability...
[preflight] Install dir: /home/user/.cache/llama_cpp
[preflight]   OK: libllama found
[preflight]   OK: libggml found
[preflight] llama.cpp is available
```

**Test Code**:
```rust
// AC:AC12
#[test]
fn test_llama_cpp_preflight_verification() {
    let temp_dir = TempDir::new().unwrap();
    let install_dir = temp_dir.path().join("llama_cpp");
    let lib_dir = install_dir.join("build/lib");
    fs::create_dir_all(&lib_dir).unwrap();

    // Create both required libraries
    create_mock_libs(&lib_dir, &["libllama", "libggml"]);

    // Preflight should succeed
    let result = preflight_check(CppBackend::Llama, &install_dir, true);
    assert!(result.is_ok(), "Preflight should pass when both libs present");

    // Remove one library - should fail
    remove_lib(&lib_dir, "libggml");
    let result = preflight_check(CppBackend::Llama, &install_dir, false);
    assert!(result.is_err(), "Preflight should fail with missing library");
}
```

### AC13: Cross-Platform Support (Linux/macOS/Windows)

**Requirement**: All functionality works on Linux, macOS, Windows

**Test Matrix**:

| Platform | Library Discovery | Build | Shell Exports | RPATH |
|----------|-------------------|-------|---------------|-------|
| Linux x86_64 | ✓ | ✓ | sh, fish | ✓ |
| Linux aarch64 | ✓ | ✓ | sh, fish | ✓ |
| macOS Intel | ✓ | ✓ | sh, fish | ✓ |
| macOS ARM64 | ✓ | ✓ | sh, fish | ✓ |
| Windows x86_64 | ✓ | ✓ | pwsh, cmd | N/A (uses PATH) |

**Test Code**:
```rust
// AC:AC13
#[test]
fn test_llama_cpp_cross_platform_compatibility() {
    let temp_dir = TempDir::new().unwrap();
    let lib_dir = temp_dir.path();

    // Create platform-specific library
    #[cfg(target_os = "linux")]
    let lib_name = "libllama.so";

    #[cfg(target_os = "macos")]
    let lib_name = "libllama.dylib";

    #[cfg(target_os = "windows")]
    let lib_name = "llama.dll";

    File::create(lib_dir.join(lib_name)).unwrap();

    // Discovery should work on current platform
    let found = find_libraries_in_dir(lib_dir, "libllama");
    assert!(!found.is_empty(), "Should find platform-specific library");
}
```

### AC14: Documentation and Help Text

**Requirement**: Comprehensive help text and documentation

**Test Scenario**:
```bash
cargo run -p xtask -- setup-cpp-auto --help
```

**Expected Output**:
```
Auto-bootstrap C++ reference backends

Usage: xtask setup-cpp-auto [OPTIONS]

Options:
  --emit <FORMAT>    Shell format for exports (sh, fish, pwsh, cmd) [default: sh]
  --backend <NAME>   Backend to setup (bitnet, llama, both) [default: both]
  -h, --help         Print help
  -V, --version      Print version

Examples:
  # Install both backends (recommended)
  eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

  # Install only llama.cpp
  eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh --backend llama)"

  # fish shell
  cargo run -p xtask -- setup-cpp-auto --emit=fish | source

  # PowerShell
  cargo run -p xtask -- setup-cpp-auto --emit=pwsh | Invoke-Expression

Environment Variables:
  LLAMA_CPP_DIR          - Custom installation path (default: ~/.cache/llama_cpp)
  CROSSVAL_RPATH_LLAMA   - Custom library path for RPATH embedding
```

**Test Code**:
```rust
// AC:AC14
#[test]
fn test_llama_cpp_help_text_comprehensive() {
    let output = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--", "setup-cpp-auto", "--help"])
        .output()
        .unwrap();

    let help = String::from_utf8_lossy(&output.stdout);

    assert!(help.contains("--backend"), "Should document --backend flag");
    assert!(help.contains("llama"), "Should mention llama backend");
    assert!(help.contains("LLAMA_CPP_DIR"), "Should document env vars");
    assert!(help.contains("Examples:"), "Should include usage examples");
}
```

### AC15: Tests for All ACs

**Requirement**: Complete test coverage for AC1-AC14

**Test Coverage**:
```bash
# Run all llama.cpp-specific tests
cargo test -p xtask --test bitnet_cpp_auto_setup_tests llama

# Run integration tests
cargo test -p xtask --test integration_workflow_tests llama_cpp

# Run with coverage
cargo tarpaulin --packages xtask --test-threads 1 -- llama
```

**Expected Coverage**: ≥80% for all llama.cpp code paths

**Test Code**:
```rust
// AC:AC15
#[test]
fn test_all_acceptance_criteria_covered() {
    // This test verifies all AC1-AC14 tests exist and pass
    let test_results = vec![
        run_test("test_setup_cpp_auto_backend_llama_flag"), // AC1
        run_test("test_llama_cpp_dir_env_var_override"), // AC2
        run_test("test_llama_cpp_cmake_only_build"), // AC3
        run_test("test_llama_cpp_requires_both_libraries"), // AC4
        run_test("test_llama_cpp_search_hierarchy_precedence"), // AC5
        run_test("test_llama_cpp_emit_exports_all_shells"), // AC6
        run_test("test_llama_cpp_file_lock_prevents_concurrent_builds"), // AC7
        run_test("test_llama_cpp_network_retry_exponential_backoff"), // AC8
        run_test("test_llama_cpp_rollback_on_build_failure"), // AC9
        run_test("test_llama_cpp_platform_specific_library_naming"), // AC10
        run_test("test_llama_cpp_rpath_merge_with_bitnet"), // AC11
        run_test("test_llama_cpp_preflight_verification"), // AC12
        run_test("test_llama_cpp_cross_platform_compatibility"), // AC13
        run_test("test_llama_cpp_help_text_comprehensive"), // AC14
    ];

    let passed = test_results.iter().filter(|r| *r).count();
    let total = test_results.len();

    assert_eq!(passed, total, "All AC tests must pass ({}/{})", passed, total);
}
```

---

## Architecture

### High-Level Component Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                       xtask/src/main.rs                         │
│                  (CLI Argument Parsing)                         │
│                                                                 │
│  Commands::SetupCppAuto {                                       │
│    emit: Emit,                                                  │
│    backend: String,  ← NEW: "bitnet" | "llama" | "both"       │
│  }                                                              │
└──────────────────────────┬─────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────┐
│            xtask/src/cpp_setup_auto.rs::run()                   │
│                  (Main Entry Point)                             │
│                                                                 │
│  1. Parse backend selection                                     │
│  2. Determine installation directory                            │
│  3. Install or update backend                                   │
│  4. Find library directories                                    │
│  5. Emit shell exports                                          │
└──────────────────────────┬─────────────────────────────────────┘
                           ↓
        ┌──────────────────┴──────────────────┐
        ↓                                      ↓
┌─────────────────────┐            ┌──────────────────────┐
│  BitNet Backend     │            │  llama.cpp Backend   │
│                     │            │                      │
│  - setup_env.py     │            │  - CMake only        │
│  - libbitnet.so     │            │  - libllama.so       │
│  - Vendored llama   │            │  - libggml.so        │
└─────────────────────┘            └──────────────────────┘
        ↓                                      ↓
┌────────────────────────────────────────────────────────────────┐
│          xtask/src/cpp_setup_auto.rs::emit_exports()            │
│                  (Platform-Specific Exports)                    │
│                                                                 │
│  match emit {                                                   │
│    Emit::Sh   → export LLAMA_CPP_DIR=...                        │
│    Emit::Fish → set -gx LLAMA_CPP_DIR ...                       │
│    Emit::Pwsh → $env:LLAMA_CPP_DIR = ...                        │
│    Emit::Cmd  → set LLAMA_CPP_DIR=...                           │
│  }                                                              │
└────────────────────────────────────────────────────────────────┘
```

### Backend Abstraction Layer

**File**: `xtask/src/cpp_setup_auto.rs`

**Enum Definition**:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CppBackend {
    BitNet,
    Llama,
}

impl CppBackend {
    pub fn name(&self) -> &'static str {
        match self {
            Self::BitNet => "bitnet.cpp",
            Self::Llama => "llama.cpp",
        }
    }

    pub fn repo_url(&self) -> &'static str {
        match self {
            Self::BitNet => "https://github.com/microsoft/BitNet",
            Self::Llama => "https://github.com/ggerganov/llama.cpp",
        }
    }

    pub fn install_subdir(&self) -> &'static str {
        match self {
            Self::BitNet => "bitnet_cpp",
            Self::Llama => "llama_cpp",
        }
    }

    pub fn required_libs(&self) -> &[&'static str] {
        match self {
            Self::BitNet => &["libbitnet"],
            Self::Llama => &["libllama", "libggml"],
        }
    }

    pub fn env_var(&self) -> &'static str {
        match self {
            Self::BitNet => "BITNET_CPP_DIR",
            Self::Llama => "LLAMA_CPP_DIR",
        }
    }
}
```

### Function Signatures

**Core Public API**:

```rust
/// Main entry point for setup-cpp-auto command
pub fn run(emit: Emit, backend: Option<CppBackend>) -> Result<()>

/// Install or update specified backend
pub fn install_or_update_backend(backend: CppBackend) -> Result<PathBuf>

/// Find library directories for backend
pub fn find_backend_lib_dirs(
    install_dir: &Path,
    backend: CppBackend,
) -> Result<Vec<PathBuf>>
```

**Backend-Specific Builders**:

```rust
/// Build BitNet.cpp (existing, uses setup_env.py or CMake)
fn build_bitnet_cpp(install_dir: &Path) -> Result<()>

/// Build llama.cpp (NEW, CMake-only)
fn build_llama_cpp(install_dir: &Path) -> Result<()>

/// Dispatcher to backend-specific builder
fn build_backend(backend: CppBackend, install_dir: &Path) -> Result<()> {
    match backend {
        CppBackend::BitNet => build_bitnet_cpp(install_dir),
        CppBackend::Llama => build_llama_cpp(install_dir),
    }
}
```

**Helper Functions**:

```rust
/// Determine installation directory from env var or default
fn determine_install_dir(backend: CppBackend) -> Result<PathBuf>

/// Clone repository with submodules
fn clone_repository(url: &str, dest: &Path) -> Result<()>

/// Clone with retry and exponential backoff (NEW)
fn clone_repository_with_retry(
    url: &str,
    dest: &Path,
    max_retries: u32,
) -> Result<()>

/// Acquire file lock for build operation (NEW)
fn acquire_build_lock(install_dir: &Path, backend: CppBackend) -> Result<File>

/// Update existing repository
fn update_repository(install_dir: &Path) -> Result<()>

/// Run CMake build
fn run_cmake_build(repo_path: &Path) -> Result<()>

/// Run CMake build with custom flags (NEW)
fn run_cmake_build_with_flags(
    repo_path: &Path,
    backend: CppBackend,
) -> Result<()>

/// Check if directory contains required libraries
fn has_all_libraries(dir: &Path, required: &[&str]) -> bool

/// Emit platform-specific shell exports
fn emit_exports(
    emit: Emit,
    backend: CppBackend,
    repo: &Path,
    lib_dir: &Path,
    crossval_libdir: Option<&str>,
)

/// Transactional install with rollback on failure (NEW)
fn install_or_update_backend_transactional(
    backend: CppBackend,
) -> Result<PathBuf>
```

### Data Structures

**Emit Enum** (existing):
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emit {
    Sh,    // POSIX shells (bash, zsh)
    Fish,  // fish shell
    Pwsh,  // PowerShell
    Cmd,   // Windows cmd
}

impl From<&str> for Emit {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "fish" => Emit::Fish,
            "pwsh" | "powershell" => Emit::Pwsh,
            "cmd" | "batch" => Emit::Cmd,
            _ => Emit::Sh, // Default
        }
    }
}
```

**BuildLock Structure** (NEW):
```rust
pub struct BuildLock {
    file: File,
    path: PathBuf,
}

impl BuildLock {
    pub fn acquire(install_dir: &Path, backend: CppBackend) -> Result<Self> {
        let lock_path = install_dir.parent()
            .unwrap_or(install_dir)
            .join(format!(".{}.lock", backend.install_subdir()));

        let file = File::create(&lock_path)
            .with_context(|| format!("Failed to create lock file: {}", lock_path.display()))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            use nix::fcntl::{flock, FlockArg};

            flock(file.as_raw_fd(), FlockArg::LockExclusiveNonblock)
                .with_context(|| format!(
                    "Failed to acquire lock - another setup-cpp-auto may be running for {}",
                    backend.name()
                ))?;
        }

        Ok(Self { file, path: lock_path })
    }
}

impl Drop for BuildLock {
    fn drop(&mut self) {
        // Lock automatically released on file close
        let _ = fs::remove_file(&self.path);
    }
}
```

### Integration Points

**1. CLI Argument Parsing** (`xtask/src/main.rs`):

```rust
#[derive(Subcommand)]
pub enum Commands {
    #[command(name = "setup-cpp-auto")]
    #[command(about = "Auto-bootstrap C++ reference backends")]
    SetupCppAuto {
        /// Shell format for environment variable exports
        #[arg(long, value_parser = ["sh", "fish", "pwsh", "cmd"], default_value = "sh")]
        emit: String,

        /// Which backend(s) to setup: "bitnet", "llama", or "both"
        #[arg(long, value_parser = ["bitnet", "llama", "both"], default_value = "both")]
        backend: String,
    },
    // ... other commands
}
```

**Handler**:
```rust
Commands::SetupCppAuto { emit, backend } => {
    let emit_format = cpp_setup_auto::Emit::from(emit.as_str());

    let backends = match backend.as_str() {
        "bitnet" => vec![CppBackend::BitNet],
        "llama" => vec![CppBackend::Llama],
        "both" => vec![CppBackend::BitNet, CppBackend::Llama],
        _ => bail!("Invalid backend: {}", backend),
    };

    for backend in backends {
        cpp_setup_auto::run(emit_format, Some(backend))?;
    }
}
```

**2. Build.rs RPATH Integration** (`xtask/build.rs`):

```rust
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BITNET_CROSSVAL_LIBDIR");
    println!("cargo:rerun-if-env-changed=BITNET_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=CROSSVAL_RPATH_BITNET");
    println!("cargo:rerun-if-env-changed=CROSSVAL_RPATH_LLAMA");  // NEW

    #[cfg(any(feature = "crossval", feature = "crossval-all", feature = "ffi"))]
    embed_crossval_rpath();
}

fn embed_crossval_rpath() {
    // Priority 1: Legacy single-directory override
    if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
        emit_rpath(lib_dir);
        return;
    }

    // Priority 2: Granular backend-specific paths
    let mut rpath_candidates = Vec::new();

    if let Ok(bitnet_path) = env::var("CROSSVAL_RPATH_BITNET") {
        if Path::new(&bitnet_path).exists() {
            rpath_candidates.push(bitnet_path);
        }
    }

    if let Ok(llama_path) = env::var("CROSSVAL_RPATH_LLAMA") {  // NEW
        if Path::new(&llama_path).exists() {
            rpath_candidates.push(llama_path);
        }
    }

    if !rpath_candidates.is_empty() {
        let merged = merge_and_deduplicate(&rpath_candidates.iter().map(|s| s.as_str()).collect::<Vec<_>>());
        emit_rpath(merged);
        return;
    }

    // Priority 3: Auto-discovery from BITNET_CPP_DIR and LLAMA_CPP_DIR
    // ... (existing logic extended for llama.cpp)
}
```

**3. Preflight Check Integration** (`xtask/src/crossval/preflight.rs`):

```rust
pub fn check_backend_available(backend: CppBackend, verbose: bool) -> Result<()> {
    let install_dir = determine_install_dir(backend)?;

    if verbose {
        eprintln!("[preflight] Checking {} availability...", backend.name());
        eprintln!("[preflight] Install dir: {}", install_dir.display());
    }

    // 1. Check installation directory exists
    if !install_dir.exists() {
        bail!("{} not installed at {}", backend.name(), install_dir.display());
    }

    // 2. Find library directories
    let lib_dirs = find_backend_lib_dirs(&install_dir, backend)?;
    if lib_dirs.is_empty() {
        bail!("No libraries found for {} at {}", backend.name(), install_dir.display());
    }

    // 3. Verify all required libraries present
    for lib_pattern in backend.required_libs() {
        let found = lib_dirs.iter()
            .any(|dir| !find_libraries_in_dir(dir, lib_pattern).is_empty());

        if !found {
            bail!("Missing required library: {} for {}", lib_pattern, backend.name());
        }

        if verbose {
            eprintln!("[preflight]   OK: {} found", lib_pattern);
        }
    }

    if verbose {
        eprintln!("[preflight] {} is available", backend.name());
    }

    Ok(())
}
```

---

## Implementation Details

### Step-by-Step Implementation Guide

#### Step 1: Extend CppBackend Enum

**File**: `xtask/src/cpp_setup_auto.rs`

**Changes**:
1. Add `required_libs()` method to CppBackend
2. Add `env_var()` method to CppBackend
3. Update all match statements to handle Llama variant

**Code**:
```rust
impl CppBackend {
    // ... existing methods ...

    pub fn required_libs(&self) -> &[&'static str] {
        match self {
            Self::BitNet => &["libbitnet"],
            Self::Llama => &["libllama", "libggml"],
        }
    }

    pub fn env_var(&self) -> &'static str {
        match self {
            Self::BitNet => "BITNET_CPP_DIR",
            Self::Llama => "LLAMA_CPP_DIR",
        }
    }
}
```

#### Step 2: Implement llama.cpp Builder

**File**: `xtask/src/cpp_setup_auto.rs`

**New Function**:
```rust
fn build_llama_cpp(install_dir: &Path) -> Result<()> {
    eprintln!("[llama] Building llama.cpp with CMake...");

    // llama.cpp uses CMake-only, no setup_env.py
    run_cmake_build_with_flags(install_dir, CppBackend::Llama)?;

    eprintln!("[llama] llama.cpp build succeeded");
    Ok(())
}
```

**CMake Flags for llama.cpp**:
```rust
fn get_cmake_flags(backend: CppBackend) -> Vec<String> {
    let mut flags = vec!["-DCMAKE_BUILD_TYPE=Release".to_string()];

    // CRITICAL: Build shared libraries for FFI
    flags.push("-DBUILD_SHARED_LIBS=ON".to_string());

    // Enable CPU-specific optimizations
    flags.push("-DLLAMA_NATIVE=ON".to_string());

    // GPU support (configurable)
    let cuda_enabled = env::var("BITNET_ENABLE_CUDA")
        .ok()
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(false);

    if !cuda_enabled {
        flags.push("-DGGML_CUDA=OFF".to_string());
    }

    flags
}
```

#### Step 3: Enhance Library Discovery

**File**: `xtask/src/cpp_setup_auto.rs`

**Replace `find_bitnet_lib_dirs()` with Generic Version**:
```rust
pub fn find_backend_lib_dirs(
    install_dir: &Path,
    backend: CppBackend,
) -> Result<Vec<PathBuf>> {
    let mut lib_dirs = vec![];

    // Priority 0: Explicit override
    let override_env = match backend {
        CppBackend::BitNet => "BITNET_CROSSVAL_LIBDIR",
        CppBackend::Llama => "LLAMA_CROSSVAL_LIBDIR",
    };

    if let Ok(explicit_libdir) = env::var(override_env) {
        let explicit_path = PathBuf::from(explicit_libdir);
        if has_all_libraries(&explicit_path, backend.required_libs()) {
            return Ok(vec![explicit_path]);
        }
    }

    // Tier 1: Primary search paths (backend-specific)
    let tier1_candidates = match backend {
        CppBackend::BitNet => vec![
            install_dir.join("build/bin"),
            install_dir.join("build/lib"),
            install_dir.join("build/3rdparty/llama.cpp/build/bin"),
        ],
        CppBackend::Llama => vec![
            install_dir.join("build/bin"),
            install_dir.join("build/lib"),
            install_dir.join("build"),
        ],
    };

    for candidate in &tier1_candidates {
        if has_all_libraries(candidate, backend.required_libs()) {
            lib_dirs.push(candidate.clone());
        }
    }

    // Tier 2: Fallback
    let fallback = [install_dir.join("lib")];
    for candidate in &fallback {
        if has_all_libraries(candidate, backend.required_libs()) {
            lib_dirs.push(candidate.clone());
        }
    }

    Ok(lib_dirs)
}

fn has_all_libraries(dir: &Path, required: &[&str]) -> bool {
    if !dir.is_dir() {
        return false;
    }

    required.iter().all(|lib| {
        find_libraries_in_dir(dir, lib).len() > 0
    })
}
```

#### Step 4: Add File Locking

**File**: `xtask/src/cpp_setup_auto.rs`

**New Function**:
```rust
fn acquire_build_lock(install_dir: &Path, backend: CppBackend) -> Result<BuildLock> {
    let lock_path = install_dir.parent()
        .unwrap_or(install_dir)
        .join(format!(".{}.lock", backend.install_subdir()));

    let file = File::create(&lock_path)
        .with_context(|| format!("Failed to create lock file: {}", lock_path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        use nix::fcntl::{flock, FlockArg};
        use std::os::unix::io::AsRawFd;

        flock(file.as_raw_fd(), FlockArg::LockExclusiveNonblock)
            .with_context(|| format!(
                "Failed to acquire lock - another setup-cpp-auto may be running for {}\n\
                 If no other instance is running, remove lock file: {}",
                backend.name(),
                lock_path.display()
            ))?;
    }

    #[cfg(windows)]
    {
        // Windows file locking via LockFile API
        use std::os::windows::io::AsRawHandle;
        use winapi::um::fileapi::LockFile;

        let handle = file.as_raw_handle();
        let result = unsafe { LockFile(handle as _, 0, 0, 1, 0) };

        if result == 0 {
            bail!(
                "Failed to acquire lock - another setup-cpp-auto may be running for {}\n\
                 If no other instance is running, remove lock file: {}",
                backend.name(),
                lock_path.display()
            );
        }
    }

    Ok(BuildLock { file, path: lock_path })
}
```

**Usage in `install_or_update_backend()`**:
```rust
pub fn install_or_update_backend(backend: CppBackend) -> Result<PathBuf> {
    let install_dir = determine_install_dir(backend)?;

    // Acquire lock before any filesystem modifications
    let _lock = acquire_build_lock(&install_dir, backend)?;

    // ... rest of install logic ...

    Ok(install_dir)
}
```

#### Step 5: Implement Network Retry

**File**: `xtask/src/cpp_setup_auto.rs`

**New Function**:
```rust
fn clone_repository_with_retry(
    url: &str,
    dest: &Path,
    max_retries: u32,
) -> Result<()> {
    let mut attempt = 0;
    let mut backoff = Duration::from_secs(1);

    loop {
        attempt += 1;

        eprintln!("[bitnet] Clone attempt {} of {}...", attempt, max_retries);

        match clone_repository(url, dest) {
            Ok(_) => return Ok(()),
            Err(e) if attempt < max_retries => {
                eprintln!(
                    "[bitnet] Clone attempt {} failed: {}. Retrying in {:?}...",
                    attempt, e, backoff
                );

                // Clean up partial clone
                if dest.exists() {
                    let _ = fs::remove_dir_all(dest);
                }

                std::thread::sleep(backoff);

                // Exponential backoff: 1s → 1.5s → 2.25s → 3.375s → ...
                backoff = backoff.mul_f32(1.5).min(Duration::from_secs(60));
            }
            Err(e) => {
                return Err(e).with_context(|| format!(
                    "Failed to clone after {} attempts. Last error",
                    max_retries
                ));
            }
        }
    }
}
```

#### Step 6: Add Rollback Mechanism

**File**: `xtask/src/cpp_setup_auto.rs`

**New Function**:
```rust
pub fn install_or_update_backend_transactional(backend: CppBackend) -> Result<PathBuf> {
    let install_dir = determine_install_dir(backend)?;
    let backup_dir = install_dir.with_extension(".backup");

    // Create backup of existing installation
    if install_dir.exists() {
        eprintln!("[{}] Creating backup...", backend.name());

        // Remove old backup if exists
        if backup_dir.exists() {
            fs::remove_dir_all(&backup_dir)
                .with_context(|| format!("Failed to remove old backup: {}", backup_dir.display()))?;
        }

        fs::rename(&install_dir, &backup_dir)
            .with_context(|| format!("Failed to create backup: {}", backup_dir.display()))?;
    }

    // Attempt new installation
    match install_or_update_backend(backend) {
        Ok(dir) => {
            // Success - clean up backup
            if backup_dir.exists() {
                let _ = fs::remove_dir_all(&backup_dir);
            }
            Ok(dir)
        }
        Err(e) => {
            eprintln!("[{}] Build failed, restoring backup...", backend.name());

            // Restore backup on failure
            if backup_dir.exists() {
                // Remove partial installation
                if install_dir.exists() {
                    let _ = fs::remove_dir_all(&install_dir);
                }

                fs::rename(&backup_dir, &install_dir)
                    .with_context(|| format!("Failed to restore backup from {}", backup_dir.display()))?;

                eprintln!("[{}] Previous installation restored", backend.name());
            }

            Err(e)
        }
    }
}
```

#### Step 7: Update Shell Export Emitters

**File**: `xtask/src/cpp_setup_auto.rs`

**Modify `emit_exports()` Signature**:
```rust
fn emit_exports(
    emit: Emit,
    backend: CppBackend,  // NEW: backend parameter
    repo: &Path,
    lib_dir: &Path,
    crossval_libdir: Option<&str>,
) {
    let env_var = backend.env_var();

    match emit {
        Emit::Sh => {
            println!(r#"export {}="{}""#, env_var, repo.display());

            if let Some(libdir) = crossval_libdir {
                let crossval_env = match backend {
                    CppBackend::BitNet => "BITNET_CROSSVAL_LIBDIR",
                    CppBackend::Llama => "LLAMA_CROSSVAL_LIBDIR",
                };
                println!(r#"export {}="{}""#, crossval_env, libdir);
            }

            #[cfg(target_os = "linux")]
            println!(r#"export LD_LIBRARY_PATH="{}:${{LD_LIBRARY_PATH:-}}""#, lib_dir.display());

            #[cfg(target_os = "macos")]
            println!(r#"export DYLD_LIBRARY_PATH="{}:${{DYLD_LIBRARY_PATH:-}}""#, lib_dir.display());

            #[cfg(target_os = "windows")]
            println!(r#"export PATH="{}:${{PATH:-}}""#, lib_dir.display());

            println!(r#"echo "[{}] C++ ready at ${}""#, backend.name(), env_var);
        }

        Emit::Fish => {
            println!(r#"set -gx {} "{}""#, env_var, repo.display());

            if let Some(libdir) = crossval_libdir {
                let crossval_env = match backend {
                    CppBackend::BitNet => "BITNET_CROSSVAL_LIBDIR",
                    CppBackend::Llama => "LLAMA_CROSSVAL_LIBDIR",
                };
                println!(r#"set -gx {} "{}""#, crossval_env, libdir);
            }

            #[cfg(target_os = "linux")]
            println!(r#"set -gx LD_LIBRARY_PATH "{}" $LD_LIBRARY_PATH"#, lib_dir.display());

            #[cfg(target_os = "macos")]
            println!(r#"set -gx DYLD_LIBRARY_PATH "{}" $DYLD_LIBRARY_PATH"#, lib_dir.display());

            #[cfg(target_os = "windows")]
            println!(r#"set -gx PATH "{}" $PATH"#, lib_dir.display());

            println!(r#"echo "[{}] C++ ready at ${}""#, backend.name(), env_var);
        }

        Emit::Pwsh => {
            println!(r#"$env:{} = "{}""#, env_var, repo.display());

            if let Some(libdir) = crossval_libdir {
                let crossval_env = match backend {
                    CppBackend::BitNet => "BITNET_CROSSVAL_LIBDIR",
                    CppBackend::Llama => "LLAMA_CROSSVAL_LIBDIR",
                };
                println!(r#"$env:{} = "{}""#, crossval_env, libdir);
            }

            println!(r#"$env:PATH = "{};" + $env:PATH"#, lib_dir.display());
            println!(r#"Write-Host "[{}] C++ ready at $env:{}""#, backend.name(), env_var);
        }

        Emit::Cmd => {
            println!(r#"set {}={}"#, env_var, repo.display());

            if let Some(libdir) = crossval_libdir {
                let crossval_env = match backend {
                    CppBackend::BitNet => "BITNET_CROSSVAL_LIBDIR",
                    CppBackend::Llama => "LLAMA_CROSSVAL_LIBDIR",
                };
                println!(r#"set {}={}"#, crossval_env, libdir);
            }

            println!(r#"set PATH={};%PATH%"#, lib_dir.display());
            println!(r#"echo [{}] C++ ready at %{}%"#, backend.name(), env_var);
        }
    }
}
```

#### Step 8: Update `run()` Entry Point

**File**: `xtask/src/cpp_setup_auto.rs`

**Modified Function**:
```rust
pub fn run(emit: Emit, backend: Option<CppBackend>) -> Result<()> {
    let backends = match backend {
        Some(b) => vec![b],
        None => vec![CppBackend::BitNet, CppBackend::Llama], // Default: both
    };

    for backend in backends {
        eprintln!("[{}] Setting up {}...", backend.name(), backend.name());

        let repo = install_or_update_backend_transactional(backend)?;

        let lib_dirs = find_backend_lib_dirs(&repo, backend)?;
        if lib_dirs.is_empty() {
            bail!(
                "No libraries found for {} after build.\n\
                 Expected: {}\n\
                 Searched: {:?}",
                backend.name(),
                backend.required_libs().join(", "),
                repo.join("build")
            );
        }

        let lib_dir = &lib_dirs[0];

        // Auto-discover CROSSVAL_LIBDIR if not explicitly set
        let crossval_env = match backend {
            CppBackend::BitNet => "BITNET_CROSSVAL_LIBDIR",
            CppBackend::Llama => "LLAMA_CROSSVAL_LIBDIR",
        };

        let crossval_libdir = env::var(crossval_env).ok().or_else(|| {
            lib_dirs.first().map(|p| p.display().to_string())
        });

        emit_exports(
            emit,
            backend,
            &repo,
            lib_dir,
            crossval_libdir.as_deref()
        );

        eprintln!("[{}] Setup complete", backend.name());
    }

    Ok(())
}
```

---

## Testing Strategy

### Unit Tests

**File**: `xtask/tests/bitnet_cpp_auto_setup_tests.rs`

**Test Organization**:
```rust
mod llama_cpp_tests {
    use super::*;

    #[test]
    fn test_llama_cpp_backend_methods() {
        assert_eq!(CppBackend::Llama.name(), "llama.cpp");
        assert_eq!(
            CppBackend::Llama.repo_url(),
            "https://github.com/ggerganov/llama.cpp"
        );
        assert_eq!(CppBackend::Llama.install_subdir(), "llama_cpp");
        assert_eq!(CppBackend::Llama.required_libs(), &["libllama", "libggml"]);
        assert_eq!(CppBackend::Llama.env_var(), "LLAMA_CPP_DIR");
    }

    #[test]
    fn test_llama_cpp_library_discovery() {
        let temp = TempDir::new().unwrap();
        let lib_dir = temp.path().join("build/lib");
        fs::create_dir_all(&lib_dir).unwrap();

        // Create both required libraries
        create_mock_lib(&lib_dir, "libllama");
        create_mock_lib(&lib_dir, "libggml");

        let result = has_all_libraries(&lib_dir, &["libllama", "libggml"]);
        assert!(result, "Should find both libraries");
    }

    #[test]
    fn test_llama_cpp_cmake_flags() {
        let flags = get_cmake_flags(CppBackend::Llama);

        assert!(flags.contains(&"-DCMAKE_BUILD_TYPE=Release".to_string()));
        assert!(flags.contains(&"-DBUILD_SHARED_LIBS=ON".to_string()));
        assert!(flags.contains(&"-DLLAMA_NATIVE=ON".to_string()));
    }

    // ... more unit tests for AC1-AC15 ...
}
```

### Integration Tests

**File**: `xtask/tests/integration_workflow_tests.rs`

**Test Cases**:
```rust
#[test]
#[ignore] // Run manually with --ignored
#[serial(bitnet_env)]
fn test_llama_cpp_full_setup_workflow() {
    // Full end-to-end test (requires network)
    let output = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--", "setup-cpp-auto", "--backend=llama", "--emit=sh"])
        .output()
        .expect("Failed to run setup-cpp-auto");

    assert!(output.status.success(), "Setup should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("LLAMA_CPP_DIR"));
    assert!(stdout.contains("LD_LIBRARY_PATH") || stdout.contains("DYLD_LIBRARY_PATH"));
}

#[test]
#[serial(bitnet_env)]
fn test_llama_cpp_preflight_after_setup() {
    let _guard = EnvGuard::new("LLAMA_CPP_DIR", "/tmp/test-llama");

    // Mock installation
    let lib_dir = PathBuf::from("/tmp/test-llama/build/lib");
    fs::create_dir_all(&lib_dir).unwrap();
    create_mock_libs(&lib_dir, &["libllama", "libggml"]);

    // Preflight should pass
    let result = check_backend_available(CppBackend::Llama, false);
    assert!(result.is_ok(), "Preflight should pass with both libs");
}
```

### Cross-Platform Tests

**Platform Matrix**:

| Test | Linux | macOS | Windows |
|------|-------|-------|---------|
| Library naming | ✓ .so | ✓ .dylib | ✓ .dll |
| Library discovery | ✓ | ✓ | ✓ |
| Shell exports (sh) | ✓ | ✓ | ✓ (Git Bash) |
| Shell exports (fish) | ✓ | ✓ | ✗ |
| Shell exports (pwsh) | ✓ | ✓ | ✓ |
| RPATH embedding | ✓ | ✓ | N/A |
| File locking | ✓ flock | ✓ flock | ✓ LockFile |

**Conditional Compilation Tests**:
```rust
#[test]
#[cfg(target_os = "linux")]
fn test_llama_cpp_linux_library_extensions() {
    let lib_name = format_lib_name("llama");
    assert_eq!(lib_name, "libllama.so");
}

#[test]
#[cfg(target_os = "macos")]
fn test_llama_cpp_macos_library_extensions() {
    let lib_name = format_lib_name("llama");
    assert_eq!(lib_name, "libllama.dylib");
}

#[test]
#[cfg(target_os = "windows")]
fn test_llama_cpp_windows_library_extensions() {
    let lib_name = format_lib_name("llama");
    assert_eq!(lib_name, "llama.dll");
}
```

### Performance Tests

**Test Scenario**: Measure build time and resource usage

```rust
#[test]
#[ignore] // Manual performance test
fn test_llama_cpp_build_performance() {
    let temp = TempDir::new().unwrap();
    let install_dir = temp.path();

    let start = Instant::now();

    // Clone
    clone_repository_with_retry(
        "https://github.com/ggerganov/llama.cpp",
        install_dir,
        3
    ).unwrap();

    let clone_time = start.elapsed();

    // Build
    let build_start = Instant::now();
    build_llama_cpp(install_dir).unwrap();
    let build_time = build_start.elapsed();

    println!("Clone time: {:?}", clone_time);
    println!("Build time: {:?}", build_time);

    // Targets: Clone ≤2 min, Build ≤8 min
    assert!(clone_time < Duration::from_secs(120), "Clone too slow");
    assert!(build_time < Duration::from_secs(480), "Build too slow");
}
```

---

## Risks and Mitigations

### Risk 1: Concurrent Build Corruption

**Risk**: Multiple processes building llama.cpp simultaneously

**Impact**: High (build artifacts corrupted, hard to diagnose)

**Probability**: Medium (common in CI environments)

**Mitigation**:
- Implement file locking (AC7)
- Use platform-specific lock mechanisms (flock on Unix, LockFile on Windows)
- Lock file naming: `.llama_cpp.lock` (backend-specific)
- Clear error message: "another setup-cpp-auto may be running"

**Validation**:
```rust
#[test]
fn test_concurrent_build_prevention() {
    let lock1 = acquire_build_lock(path, CppBackend::Llama).unwrap();
    let lock2 = acquire_build_lock(path, CppBackend::Llama);
    assert!(lock2.is_err(), "Should prevent concurrent builds");
}
```

### Risk 2: Network Failures During Clone

**Risk**: git clone fails due to transient network issues

**Impact**: High (blocks setup, user frustration)

**Probability**: Medium (GitHub outages, rate limits)

**Mitigation**:
- Retry with exponential backoff (AC8)
- Max 5 retries, cap at 60 seconds
- Clean up partial clones between retries
- Detailed error messages with troubleshooting

**Validation**:
```rust
#[test]
fn test_network_retry_recovers_from_transient_failure() {
    let mock_fail_twice = /* ... mock that fails 2 times ... */;
    let result = clone_repository_with_retry(url, dest, 5, mock_fail_twice);
    assert!(result.is_ok(), "Should recover from transient failures");
}
```

### Risk 3: Incomplete Library Build

**Risk**: Build succeeds but only creates one of libllama.so/libggml.so

**Impact**: High (runtime crashes, confusing errors)

**Probability**: Low (CMake usually builds both or neither)

**Mitigation**:
- Post-build verification (AC4)
- Check for ALL required libraries
- Fail with clear message listing missing libs
- Rollback to previous installation (AC9)

**Validation**:
```rust
#[test]
fn test_incomplete_build_detection() {
    let lib_dir = create_temp_dir();
    create_mock_lib(lib_dir, "libllama"); // Only one lib

    let result = verify_build_completeness(lib_dir, CppBackend::Llama);
    assert!(result.is_err(), "Should detect missing libggml");
}
```

### Risk 4: Platform-Specific Library Naming Mismatch

**Risk**: Windows searches for libllama.dll instead of llama.dll

**Impact**: High (library not found, cross-validation fails)

**Probability**: Medium (Windows naming conventions differ)

**Mitigation**:
- Platform-specific name formatting (AC10)
- Strip "lib" prefix on Windows
- Test on all platforms
- Abstract library naming via `format_lib_name()`

**Validation**:
```rust
#[test]
#[cfg(target_os = "windows")]
fn test_windows_dll_naming_without_lib_prefix() {
    let name = format_lib_name("llama");
    assert_eq!(name, "llama.dll", "Windows should not have lib prefix");
}
```

### Risk 5: RPATH Length Limit Exceeded

**Risk**: Merged RPATH exceeds 4096 bytes (linker limit)

**Impact**: High (build fails with cryptic linker error)

**Probability**: Low (only with many backends or deep paths)

**Mitigation**:
- Validate RPATH length in `merge_and_deduplicate()`
- Prefer canonicalized paths (removes redundancy)
- Deduplication via HashSet
- Fail early with clear error

**Validation**:
```rust
#[test]
#[should_panic(expected = "exceeds maximum length")]
fn test_rpath_length_limit_exceeded() {
    let long_paths: Vec<_> = (0..1000)
        .map(|i| format!("/very/long/path/number/{}", i))
        .collect();

    let refs: Vec<&str> = long_paths.iter().map(|s| s.as_str()).collect();
    merge_and_deduplicate(&refs); // Should panic
}
```

### Risk 6: CMake Version Incompatibility

**Risk**: User has CMake <3.18 (llama.cpp requires ≥3.18)

**Impact**: Medium (build fails, but clear error)

**Probability**: Low (most systems have recent CMake)

**Mitigation**:
- Preflight CMake version check
- Clear error message with installation instructions
- Suggest package manager commands per platform

**Validation**:
```rust
#[test]
fn test_cmake_version_check() {
    let version = check_cmake_version().unwrap();
    assert!(version >= (3, 18), "CMake ≥3.18 required for llama.cpp");
}
```

### Risk 7: Disk Space Exhaustion During Build

**Risk**: Build fails mid-way due to disk full

**Impact**: Medium (partial build, confusing state)

**Probability**: Low (builds are 50-100MB)

**Mitigation**:
- Rollback mechanism (AC9)
- Restore previous installation on failure
- Check disk space before build (optional)

**Validation**:
```rust
#[test]
fn test_rollback_restores_on_disk_full() {
    let mock_disk_full = /* ... simulate ENOSPC ... */;
    let result = install_or_update_backend_transactional(
        CppBackend::Llama,
        Some(mock_disk_full)
    );

    assert!(result.is_err(), "Should fail on disk full");
    assert!(previous_install_exists(), "Should restore previous");
}
```

### Risk 8: Windows PATH Length Limit

**Risk**: Windows PATH exceeds 32,767 characters

**Impact**: Low (PATH truncated, DLLs not found)

**Probability**: Very Low (rare configuration)

**Mitigation**:
- Recommend side-by-side DLL deployment
- Document PATH limit in Windows guide
- Suggest registry modification for long PATH support

**Documentation**:
```markdown
## Windows PATH Considerations

Windows has a 32,767 character limit for PATH. If adding llama.cpp
to PATH would exceed this:

1. Use side-by-side DLL deployment (copy DLLs to exe directory)
2. Enable long PATH support: Computer Configuration → Administrative
   Templates → System → Filesystem → Enable Win32 long paths
```

---

## Documentation Requirements

### User-Facing Documentation

**File**: `docs/howto/cpp-setup.md`

**New Section**:
```markdown
### Setting Up llama.cpp Backend

llama.cpp provides reference inference for LLaMA, Mistral, and other
non-BitNet models.

#### Quick Setup

```bash
# Install llama.cpp only
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh --backend llama)"

# Or install both backends (recommended)
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
```

#### Custom Installation Path

```bash
export LLAMA_CPP_DIR=/opt/llama-cpp
cargo run -p xtask -- setup-cpp-auto --backend llama
```

#### Verification

```bash
cargo run -p xtask --features crossval-all -- preflight --backend llama --verbose
```

Expected output:
```
[preflight] Checking llama.cpp availability...
[preflight]   OK: libllama found
[preflight]   OK: libggml found
[preflight] llama.cpp is available
```

#### Troubleshooting

**Problem**: Libraries not found

```bash
# Check installation directory
ls -la ~/.cache/llama_cpp/build/lib/

# Should see:
# libllama.so (or .dylib on macOS, .dll on Windows)
# libggml.so (or .dylib on macOS, .dll on Windows)
```

**Problem**: Build fails with "CMake not found"

Install CMake ≥3.18:
- Linux: `sudo apt install cmake`
- macOS: `brew install cmake`
- Windows: `choco install cmake`

**Problem**: Preflight fails after successful build

Rebuild xtask to refresh build-time constants:
```bash
cargo clean -p xtask
cargo build -p xtask --features crossval-all
```
```

### Developer Documentation

**File**: `docs/explanation/dual-backend-crossval.md`

**Update Section**:
```markdown
### Backend-Specific Build Requirements

#### BitNet.cpp

- **Build System**: setup_env.py (Python wrapper) OR CMake fallback
- **Libraries**: libbitnet.so (single library)
- **Submodules**: Vendors llama.cpp in 3rdparty/
- **GPU Support**: Bundled in repository

#### llama.cpp

- **Build System**: CMake-only (no Python wrapper)
- **Libraries**: libllama.so + libggml.so (BOTH required)
- **Submodules**: None (standalone)
- **GPU Support**: Optional via `-DGGML_CUDA=ON`

### Library Discovery Priority

Both backends follow three-tier search:

1. **Tier 0**: Explicit override (`BITNET_CROSSVAL_LIBDIR` or `LLAMA_CROSSVAL_LIBDIR`)
2. **Tier 1**: Primary CMake outputs (`build/bin`, `build/lib`)
3. **Tier 2**: Build root fallback (`build/`, `lib/`)

### RPATH Merging

When both backends present:

```bash
export CROSSVAL_RPATH_BITNET=/path/to/bitnet/lib
export CROSSVAL_RPATH_LLAMA=/path/to/llama/lib
cargo build -p xtask --features crossval-all
```

Emits:
```
-Wl,-rpath:/path/to/bitnet/lib:/path/to/llama/lib
```

Paths are deduplicated and canonicalized to prevent RPATH length issues.
```

### Help Text

**File**: `xtask/src/main.rs`

**Update Command Help**:
```rust
#[command(name = "setup-cpp-auto")]
#[command(about = "Auto-bootstrap C++ reference backends")]
#[command(long_about = "\
Auto-bootstrap C++ reference backends for cross-validation.

This command clones, builds, and configures BitNet.cpp and/or llama.cpp,
then emits shell exports for the current session.

EXAMPLES:
  # Install both backends (recommended)
  eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\"

  # Install only llama.cpp
  eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh --backend llama)\"

  # fish shell
  cargo run -p xtask -- setup-cpp-auto --emit=fish | source

  # PowerShell
  cargo run -p xtask -- setup-cpp-auto --emit=pwsh | Invoke-Expression

ENVIRONMENT VARIABLES:
  BITNET_CPP_DIR          Custom BitNet.cpp installation path
  LLAMA_CPP_DIR           Custom llama.cpp installation path
  CROSSVAL_RPATH_BITNET   BitNet.cpp library path for RPATH
  CROSSVAL_RPATH_LLAMA    llama.cpp library path for RPATH

SEE ALSO:
  docs/howto/cpp-setup.md - Complete setup guide
  docs/explanation/dual-backend-crossval.md - Architecture details
")]
SetupCppAuto { /* ... */ }
```

### Error Messages

**Clear Diagnostics for Common Failures**:

```rust
// Example: Missing library error
if lib_dirs.is_empty() {
    let install_dir = determine_install_dir(backend)?;

    bail!(
        "No libraries found for {} after build.\n\
         \n\
         Expected libraries: {}\n\
         \n\
         Searched paths:\n\
         {}\n\
         \n\
         Troubleshooting:\n\
         1. Check build logs for errors: {}/build.log\n\
         2. Verify disk space: df -h\n\
         3. Try manual rebuild: cd {} && cmake --build build\n\
         4. See: docs/howto/cpp-setup.md#troubleshooting",
        backend.name(),
        backend.required_libs().join(", "),
        tier1_candidates.iter()
            .map(|p| format!("  - {}", p.display()))
            .collect::<Vec<_>>()
            .join("\n"),
        install_dir.display(),
        install_dir.display()
    );
}
```

---

## Appendix

### Complete Code Examples

#### Example 1: Full llama.cpp Build Function

```rust
fn build_llama_cpp(install_dir: &Path) -> Result<()> {
    eprintln!("[llama] Building llama.cpp with CMake...");

    // Check CMake version
    let cmake_version = check_cmake_version()
        .context("Failed to detect CMake version")?;

    if cmake_version < (3, 18) {
        bail!(
            "llama.cpp requires CMake ≥3.18, found {:?}\n\
             \n\
             Install instructions:\n\
             - Ubuntu/Debian: sudo apt install cmake\n\
             - macOS: brew install cmake\n\
             - Windows: choco install cmake",
            cmake_version
        );
    }

    // Build with custom flags
    run_cmake_build_with_flags(install_dir, CppBackend::Llama)
        .context("CMake build failed for llama.cpp")?;

    // Verify both libraries were built
    let lib_dirs = find_backend_lib_dirs(install_dir, CppBackend::Llama)?;

    if lib_dirs.is_empty() {
        bail!(
            "Build completed but no libraries found.\n\
             Expected: libllama.so + libggml.so\n\
             Check: {}/build/lib/",
            install_dir.display()
        );
    }

    // Verify BOTH required libraries
    let required = CppBackend::Llama.required_libs();
    for lib in required {
        let found = lib_dirs.iter()
            .any(|dir| !find_libraries_in_dir(dir, lib).is_empty());

        if !found {
            bail!(
                "Missing required library: {}\n\
                 Found libraries:\n\
                 {}\n\
                 This may indicate a partial build failure.",
                lib,
                lib_dirs.iter()
                    .flat_map(|dir| find_all_libraries_in_dir(dir))
                    .map(|p| format!("  - {}", p.display()))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
    }

    eprintln!("[llama] llama.cpp build succeeded");
    Ok(())
}
```

#### Example 2: Complete Library Discovery with Diagnostics

```rust
pub fn find_backend_lib_dirs(
    install_dir: &Path,
    backend: CppBackend,
) -> Result<Vec<PathBuf>> {
    let mut lib_dirs = vec![];

    // Priority 0: Explicit override
    let override_env = match backend {
        CppBackend::BitNet => "BITNET_CROSSVAL_LIBDIR",
        CppBackend::Llama => "LLAMA_CROSSVAL_LIBDIR",
    };

    if let Ok(explicit_libdir) = env::var(override_env) {
        let explicit_path = PathBuf::from(explicit_libdir);

        if !explicit_path.exists() {
            eprintln!(
                "Warning: {} points to non-existent path: {}",
                override_env,
                explicit_path.display()
            );
        } else if has_all_libraries(&explicit_path, backend.required_libs()) {
            return Ok(vec![explicit_path]);
        } else {
            eprintln!(
                "Warning: {} exists but missing required libraries: {:?}",
                explicit_path.display(),
                backend.required_libs()
            );
        }
    }

    // Tier 1: Primary search paths
    let tier1_candidates = match backend {
        CppBackend::BitNet => vec![
            install_dir.join("build/bin"),
            install_dir.join("build/lib"),
            install_dir.join("build/3rdparty/llama.cpp/build/bin"),
        ],
        CppBackend::Llama => vec![
            install_dir.join("build/bin"),
            install_dir.join("build/lib"),
            install_dir.join("build"),
        ],
    };

    for candidate in &tier1_candidates {
        if has_all_libraries(candidate, backend.required_libs()) {
            lib_dirs.push(candidate.clone());
        }
    }

    if !lib_dirs.is_empty() {
        return Ok(lib_dirs);
    }

    // Tier 2: Fallback
    let fallback = [install_dir.join("lib")];

    for candidate in &fallback {
        if has_all_libraries(candidate, backend.required_libs()) {
            lib_dirs.push(candidate.clone());
        }
    }

    Ok(lib_dirs)
}
```

### Reference Implementation Timeline

**Phase 1: Foundation (Week 1)**
- AC1: Backend flag support
- AC2: LLAMA_CPP_DIR environment variable
- AC3: CMake-only build

**Phase 2: Core Functionality (Week 2)**
- AC4: Dual library discovery
- AC5: Three-tier search hierarchy
- AC6: Shell export emitters

**Phase 3: Reliability (Week 3)**
- AC7: File locking
- AC8: Network retry
- AC9: Rollback mechanism

**Phase 4: Polish (Week 4)**
- AC10: Platform-specific naming
- AC11: RPATH integration
- AC12: Preflight verification

**Phase 5: Documentation & Testing (Week 5)**
- AC13: Cross-platform tests
- AC14: Documentation
- AC15: Complete test coverage

---

**Specification Complete**
**Total Length**: 1,800+ lines
**Status**: Ready for Implementation Review
**Next Steps**: Team review → Feature branch → Implementation → PR
