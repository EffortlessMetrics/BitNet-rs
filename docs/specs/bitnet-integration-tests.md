# BitNet.cpp Auto-Configuration Integration Tests Specification

**Date**: 2025-10-25
**Status**: Architectural Blueprint
**Target Audience**: Implementation engineers, CI/CD maintainers
**Version**: 1.0.0

## Purpose

This specification defines comprehensive end-to-end integration tests for validating BitNet.cpp auto-configuration parity with llama.cpp, covering library detection, RPATH embedding, preflight diagnostics, environment variable precedence, and platform-specific behavior.

## Executive Summary

The BitNet.rs cross-validation infrastructure implements dual-backend support (BitNet.cpp + llama.cpp) with sophisticated auto-detection and configuration. However, the current test coverage focuses on unit-level detection logic without validating the complete integration flow. This specification addresses the gap by defining integration tests that validate:

1. **Library detection** across multiple directory layouts
2. **RPATH embedding** in built binaries (Linux/macOS)
3. **Preflight command output** formatting and diagnostics
4. **Environment variable precedence** resolution
5. **Platform-specific behavior** (Linux `.so`, macOS `.dylib`, Windows `.lib`)

**Key insight**: Tests must work in CI without requiring real BitNet.cpp installations by using fixture-based directory structures and mock binaries.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Architecture](#architecture)
3. [Test Scenarios](#test-scenarios)
4. [Implementation Plan](#implementation-plan)
5. [Fixture Requirements](#fixture-requirements)
6. [CI/CD Integration](#cicd-integration)
7. [Success Criteria](#success-criteria)
8. [Non-Goals](#non-goals)

---

## 1. Problem Statement

### 1.1 Current State

**What exists:**
- Unit tests for `crossval/build.rs` library detection (search path scanning)
- Unit tests for `xtask/src/crossval/preflight.rs` diagnostic output
- Manual verification workflows (`cargo run -p xtask -- preflight --verbose`)

**What's missing:**
- Integration tests validating complete build → run → verify flow
- Fixture-based tests for directory structure variants
- RPATH validation in built binaries (runtime correctness)
- Environment variable precedence verification
- Platform-specific library extension handling
- CI-compatible tests (no external dependencies)

### 1.2 Integration Gaps

| Gap | Impact | Current Workaround |
|-----|--------|-------------------|
| **No fixture-based directory tests** | Can't validate detection across layouts | Manual testing with real BitNet.cpp |
| **No RPATH embedding validation** | Silent failures at runtime | Trust build.rs output, hope loader works |
| **No preflight output schema tests** | Format drift breaks tooling | Manual review of preflight output |
| **No env var precedence tests** | User confusion about override order | Documentation only |
| **No platform-specific CI tests** | Linux-only validation, macOS/Windows drift | Cross-platform manual testing |

### 1.3 Requirements

**Must have:**
- Tests executable in CI without external C++ installations
- Deterministic fixtures (no flaky filesystem races)
- Validation of actual binary artifacts (RPATH, linkage)
- Cross-platform support (Linux, macOS, Windows)

**Should have:**
- Parallel execution safety (isolated fixtures per test)
- Clear failure diagnostics (pinpoint configuration issues)
- Incremental fixture generation (reuse across tests)

**Nice to have:**
- Performance benchmarks (detection speed)
- Visual test reports (directory structure diagrams)

---

## 2. Architecture

### 2.1 Test Hierarchy

```
tests/integration/
├── crossval_detection/           # Library detection tests
│   ├── bitnet_standard_layout.rs # Standard BitNet.cpp build
│   ├── bitnet_custom_libdir.rs   # BITNET_CROSSVAL_LIBDIR override
│   ├── llama_standalone.rs       # Standalone llama.cpp (no BitNet)
│   ├── dual_backend.rs           # Both BitNet + llama
│   └── missing_libs.rs           # Graceful failure cases
├── rpath_embedding/              # Binary artifact validation
│   ├── linux_rpath.rs            # ELF RPATH validation (readelf)
│   ├── macos_rpath.rs            # Mach-O RPATH validation (otool)
│   └── windows_linkage.rs        # PE linkage validation (dumpbin)
├── preflight_diagnostics/        # Command output validation
│   ├── success_output.rs         # Backend AVAILABLE formatting
│   ├── failure_output.rs         # Backend NOT FOUND messaging
│   ├── verbose_mode.rs           # Detailed diagnostic sections
│   └── search_path_display.rs    # Library path enumeration
├── env_var_precedence/           # Environment resolution
│   ├── crossval_libdir.rs        # Highest priority override
│   ├── bitnet_cpp_dir.rs         # Standard directory resolution
│   ├── default_cache.rs          # ~/.cache/bitnet_cpp fallback
│   └── legacy_path_var.rs        # BITNET_CPP_PATH backward compat
└── fixtures/                     # Shared test infrastructure
    ├── mod.rs                    # Fixture generation API
    ├── directory_layouts.rs      # Pre-built directory structures
    ├── mock_libraries.rs         # Stub .so/.dylib/.lib files
    └── env_isolation.rs          # Test environment sandboxing
```

### 2.2 Fixture Strategy

**Approach**: Generate temporary directory structures with mock library files that match real BitNet.cpp/llama.cpp layouts.

**Components:**

1. **Directory Layout Generator**
   - Creates temporary directories with expected structure
   - Generates stub library files (empty .so/.dylib/.lib with correct names)
   - Writes minimal header files (llama.h stub for header detection)
   - Configurable layouts: standard, custom, partial, broken

2. **Mock Library Creator**
   - Generates platform-specific stub libraries:
     - Linux: `libllama.so`, `libggml.so` (ELF headers)
     - macOS: `libllama.dylib`, `libggml.dylib` (Mach-O headers)
     - Windows: `llama.lib`, `ggml.lib` (PE headers)
   - No actual symbols/code (build.rs only checks existence)
   - Correct file extensions and naming conventions

3. **Environment Isolation**
   - Uses `EnvGuard` pattern (existing in `tests/helpers/env_guard.rs`)
   - Marks tests with `#[serial(bitnet_env)]` for serial execution
   - Cleans up temp directories on drop
   - Captures build output for assertion

### 2.3 Test Execution Flow

```
Test Setup
  ↓
1. Generate fixture directory layout
  ↓
2. Create mock library files
  ↓
3. Set environment variables (BITNET_CPP_DIR, etc.)
  ↓
4. Invoke build.rs via cargo metadata or direct compilation
  ↓
5. Capture build output (cargo:rustc-link-* directives)
  ↓
6. Verify detection flags (HAS_BITNET, HAS_LLAMA)
  ↓
7. Validate RPATH embedding (readelf/otool)
  ↓
8. Run preflight command against built binary
  ↓
9. Assert output format and content
  ↓
Cleanup (temp dir deletion)
```

### 2.4 Mocking Strategy

**What to mock:**
- Directory structures (temporary filesystem layout)
- Library files (stub binaries with correct extensions)
- Header files (minimal llama.h with sentinel content)

**What NOT to mock:**
- Build.rs execution (real cargo build process)
- RPATH embedding (real linker behavior)
- Preflight command (real xtask binary)
- Environment variable resolution (real OS env)

**Rationale**: Integration tests validate real build and runtime behavior, not unit-level logic.

---

## 3. Test Scenarios

### 3.1 Library Detection Scenarios

#### Scenario 1: BitNet.cpp Standard Layout
**Description**: Validate detection when BitNet.cpp is built in standard CMake layout.

**Fixture Structure**:
```
{temp}/bitnet_cpp/
├── include/ggml-bitnet.h
├── 3rdparty/llama.cpp/include/llama.h
├── build/
│   ├── lib/
│   │   ├── libbitnet.so
│   │   ├── libllama.so
│   │   └── libggml.so
│   └── 3rdparty/llama.cpp/
│       ├── src/libllama.so
│       └── ggml/src/libggml.so
```

**Environment**:
```bash
BITNET_CPP_DIR={temp}/bitnet_cpp
```

**Expected Results**:
- `HAS_BITNET=true`
- `HAS_LLAMA=true`
- RPATH includes: `{temp}/bitnet_cpp/build/lib`
- Preflight: "✓ Backend 'bitnet.cpp': AVAILABLE"

**Test File**: `tests/integration/crossval_detection/bitnet_standard_layout.rs`

---

#### Scenario 2: Custom BITNET_CROSSVAL_LIBDIR
**Description**: Validate explicit library directory override (highest priority).

**Fixture Structure**:
```
{temp}/custom_libs/
├── libbitnet.so
├── libllama.so
└── libggml.so

{temp}/bitnet_cpp/
└── include/ggml-bitnet.h  (headers only, no libs)
```

**Environment**:
```bash
BITNET_CPP_DIR={temp}/bitnet_cpp
BITNET_CROSSVAL_LIBDIR={temp}/custom_libs  # Override
```

**Expected Results**:
- `HAS_BITNET=true` (found in custom_libs)
- `HAS_LLAMA=true`
- RPATH includes: `{temp}/custom_libs`
- Preflight: Shows custom_libs in search path #1

**Test File**: `tests/integration/crossval_detection/bitnet_custom_libdir.rs`

---

#### Scenario 3: Standalone llama.cpp (No BitNet)
**Description**: Validate detection when only llama.cpp is available (fallback mode).

**Fixture Structure**:
```
{temp}/llama_cpp/
├── include/llama.h
└── build/
    ├── bin/
    │   ├── libllama.so
    │   └── libggml.so
    └── src/llama.cpp
```

**Environment**:
```bash
BITNET_CPP_DIR={temp}/llama_cpp
```

**Expected Results**:
- `HAS_BITNET=false`
- `HAS_LLAMA=true`
- RPATH includes: `{temp}/llama_cpp/build/bin`
- Preflight: "✓ Backend 'llama.cpp': AVAILABLE"
- Preflight: "❌ Backend 'bitnet.cpp': NOT FOUND"

**Test File**: `tests/integration/crossval_detection/llama_standalone.rs`

---

#### Scenario 4: Both Backends Present
**Description**: Validate dual-backend detection (BitNet.cpp embedding llama.cpp).

**Fixture Structure**:
```
{temp}/bitnet_cpp/
├── include/ggml-bitnet.h
├── 3rdparty/llama.cpp/include/llama.h
├── build/
│   ├── lib/libbitnet.so
│   └── 3rdparty/llama.cpp/
│       ├── src/libllama.so
│       └── ggml/src/libggml.so
```

**Environment**:
```bash
BITNET_CPP_DIR={temp}/bitnet_cpp
```

**Expected Results**:
- `HAS_BITNET=true`
- `HAS_LLAMA=true`
- RPATH includes: both `build/lib` and `build/3rdparty/llama.cpp/src`
- Preflight (both backends): Both show AVAILABLE

**Test File**: `tests/integration/crossval_detection/dual_backend.rs`

---

#### Scenario 5: Missing Libraries (Graceful Failure)
**Description**: Validate error messaging when libraries not found.

**Fixture Structure**:
```
{temp}/bitnet_cpp/
└── include/ggml-bitnet.h  (headers only, no build/ directory)
```

**Environment**:
```bash
BITNET_CPP_DIR={temp}/bitnet_cpp
```

**Expected Results**:
- `HAS_BITNET=false`
- `HAS_LLAMA=false`
- No RPATH emission
- Preflight: "❌ Backend 'bitnet.cpp': NOT FOUND"
- Preflight: Shows recovery steps with setup-cpp-auto command

**Test File**: `tests/integration/crossval_detection/missing_libs.rs`

---

### 3.2 RPATH Embedding Scenarios

#### Scenario 6: Linux RPATH Validation
**Description**: Verify RPATH is correctly embedded in ELF binary on Linux.

**Approach**:
1. Build xtask binary with fixture libraries
2. Run `readelf -d target/debug/xtask | grep RPATH`
3. Parse RPATH entries
4. Validate expected library directories present

**Expected RPATH**:
```
RPATH: {temp}/bitnet_cpp/build/lib:{temp}/bitnet_cpp/build/3rdparty/llama.cpp/src
```

**Test File**: `tests/integration/rpath_embedding/linux_rpath.rs`

**Dependencies**: Requires `readelf` binary (binutils package)

---

#### Scenario 7: macOS RPATH Validation
**Description**: Verify RPATH is correctly embedded in Mach-O binary on macOS.

**Approach**:
1. Build xtask binary with fixture libraries
2. Run `otool -l target/debug/xtask | grep -A2 LC_RPATH`
3. Parse RPATH entries
4. Validate expected library directories present

**Expected RPATH**:
```
cmd LC_RPATH
cmdsize 56
path {temp}/bitnet_cpp/build/lib
```

**Test File**: `tests/integration/rpath_embedding/macos_rpath.rs`

**Dependencies**: Requires `otool` binary (macOS Developer Tools)

---

#### Scenario 8: Windows Linkage Validation
**Description**: Verify DLL dependencies in PE binary on Windows (no RPATH concept).

**Approach**:
1. Build xtask binary with fixture libraries
2. Run `dumpbin /DEPENDENTS target/debug/xtask.exe`
3. Validate DLL dependencies listed
4. Check PATH variable includes library directories

**Expected Dependencies**:
```
llama.dll
ggml.dll
```

**Test File**: `tests/integration/rpath_embedding/windows_linkage.rs`

**Dependencies**: Requires `dumpbin` (MSVC toolchain)

**Note**: Windows doesn't use RPATH; relies on PATH variable for DLL discovery.

---

### 3.3 Preflight Diagnostics Scenarios

#### Scenario 9: Success Output Format
**Description**: Validate preflight success message formatting for both backends.

**Test Coverage**:
- Non-verbose output: Single-line success
- Verbose output: 7 sections (env config, search paths, required libs, metadata, platform, summary)
- Backend-specific differences (libbitnet vs libllama requirements)

**Expected Output (Non-Verbose)**:
```
✓ Backend 'bitnet.cpp' libraries found
```

**Expected Output (Verbose)**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Backend 'bitnet.cpp': AVAILABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Environment Configuration
─────────────────────────────────────────────────
  BITNET_CPP_DIR         = {temp}/bitnet_cpp
  BITNET_CROSSVAL_LIBDIR = (not set)
  LD_LIBRARY_PATH        = {temp}/bitnet_cpp/build/lib:...

Library Search Paths (Priority Order)
─────────────────────────────────────────────────
  1. BITNET_CROSSVAL_LIBDIR override
     (not set - using default search order)

  2. BITNET_CPP_DIR/build
     ✓ {temp}/bitnet_cpp/build (exists)
     Found libraries:
       - libbitnet.so
       - libllama.so

  [... remaining sections ...]
```

**Test File**: `tests/integration/preflight_diagnostics/success_output.rs`

**Validation**:
- Regex patterns for section headers
- Library counts in each path
- Timestamp format (ISO 8601)
- Platform-specific paths (LD_LIBRARY_PATH vs DYLD_LIBRARY_PATH)

---

#### Scenario 10: Failure Output Format
**Description**: Validate preflight failure message formatting and recovery steps.

**Expected Output (Non-Verbose)**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Backend 'bitnet.cpp' libraries NOT FOUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL: C++ backend libraries not detected at BUILD TIME.
[... recovery steps ...]
```

**Expected Output (Verbose)**:
```
❌ Backend 'bitnet.cpp': NOT AVAILABLE

DIAGNOSIS
─────────────────────────────────────────────────
Build-time detection did not find required libraries.
[... detailed diagnostics ...]

RECOMMENDED FIX
─────────────────────────────────────────────────
Step 1: Run auto-setup command
  eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

Step 2: Rebuild xtask
  cargo clean -p xtask && cargo build -p xtask --features crossval-all
[... remaining steps ...]
```

**Test File**: `tests/integration/preflight_diagnostics/failure_output.rs`

**Validation**:
- Error message structure (sections present)
- Recovery command accuracy (setup-cpp-auto)
- Documentation link presence
- Missing library enumeration

---

#### Scenario 11: Verbose Mode Details
**Description**: Validate verbose output includes all diagnostic sections.

**Required Sections**:
1. Header with backend name
2. Environment Configuration
3. Library Search Paths (with existence status)
4. Required Libraries
5. Build-Time Detection Metadata
6. Platform-Specific Configuration
7. Summary

**Test File**: `tests/integration/preflight_diagnostics/verbose_mode.rs`

**Validation**:
- All 7 sections present
- Separator lines (heavy vs light)
- Library paths with ✓/✗ markers
- Timestamp format validation

---

#### Scenario 12: Search Path Display
**Description**: Validate search path enumeration and priority order.

**Expected Output**:
```
Library Search Paths (Priority Order)
─────────────────────────────────────────────────
  1. BITNET_CROSSVAL_LIBDIR override
     (not set - using default search order)

  2. BITNET_CPP_DIR/build
     ✓ {path}/build (exists)

  3. BITNET_CPP_DIR/build/lib
     ✓ {path}/build/lib (exists)

  4. BITNET_CPP_DIR/build/3rdparty/llama.cpp/src
     ✗ {path}/build/3rdparty/llama.cpp/src (not found)

  5. BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src
     ✗ {path}/build/3rdparty/llama.cpp/ggml/src (not found)

  6. BITNET_CPP_DIR/lib
     ✗ {path}/lib (not found)
```

**Test File**: `tests/integration/preflight_diagnostics/search_path_display.rs`

**Validation**:
- Priority numbers (1-6)
- Path existence markers (✓/✗)
- Library listings for existing paths
- Correct path resolution (not just env var interpolation)

---

### 3.4 Environment Variable Precedence Scenarios

#### Scenario 13: BITNET_CROSSVAL_LIBDIR Override
**Description**: Validate highest priority override takes precedence.

**Setup**:
```bash
BITNET_CPP_DIR={temp}/default
BITNET_CROSSVAL_LIBDIR={temp}/override  # Should win
```

**Expected Behavior**:
- Libraries found in `{temp}/override`
- `{temp}/default` paths NOT checked
- Preflight shows override in priority #1

**Test File**: `tests/integration/env_var_precedence/crossval_libdir.rs`

---

#### Scenario 14: BITNET_CPP_DIR Resolution
**Description**: Validate standard directory resolution (priority 2).

**Setup**:
```bash
BITNET_CPP_DIR={temp}/custom  # No CROSSVAL_LIBDIR set
```

**Expected Behavior**:
- Libraries found in `{temp}/custom/build/*`
- Search paths derived from BITNET_CPP_DIR
- Preflight shows custom path

**Test File**: `tests/integration/env_var_precedence/bitnet_cpp_dir.rs`

---

#### Scenario 15: Default Cache Directory
**Description**: Validate fallback to `~/.cache/bitnet_cpp` when no env vars set.

**Setup**:
```bash
# Unset BITNET_CPP_DIR, BITNET_CROSSVAL_LIBDIR
HOME={temp}/home
```

**Expected Behavior**:
- Libraries searched in `{temp}/home/.cache/bitnet_cpp`
- Preflight shows default cache path
- No user environment variables displayed (not set)

**Test File**: `tests/integration/env_var_precedence/default_cache.rs`

---

#### Scenario 16: Legacy BITNET_CPP_PATH
**Description**: Validate backward compatibility with BITNET_CPP_PATH.

**Setup**:
```bash
BITNET_CPP_PATH={temp}/legacy  # Old variable name
# BITNET_CPP_DIR not set
```

**Expected Behavior**:
- Libraries found via BITNET_CPP_PATH
- Preflight shows deprecation notice
- Same search paths as BITNET_CPP_DIR

**Test File**: `tests/integration/env_var_precedence/legacy_path_var.rs`

---

### 3.5 Platform-Specific Scenarios

#### Scenario 17: Linux .so Libraries
**Description**: Validate Linux shared library detection.

**Fixture Libraries**:
```
libbitnet.so
libllama.so
libggml.so
```

**Expected Behavior**:
- Build.rs matches `*.so` extension
- RPATH embedded via `-Wl,-rpath,{path}`
- Runtime uses LD_LIBRARY_PATH

**Test File**: Platform-specific test in `crossval_detection/` suite

---

#### Scenario 18: macOS .dylib Libraries
**Description**: Validate macOS dynamic library detection.

**Fixture Libraries**:
```
libbitnet.dylib
libllama.dylib
libggml.dylib
```

**Expected Behavior**:
- Build.rs matches `*.dylib` extension
- RPATH embedded via `-Wl,-rpath,{path}`
- Runtime uses DYLD_LIBRARY_PATH

**Test File**: Platform-specific test in `crossval_detection/` suite

---

#### Scenario 19: Windows .lib Static Libraries
**Description**: Validate Windows static library detection (no RPATH concept).

**Fixture Libraries**:
```
bitnet.lib
llama.lib
ggml.lib
```

**Expected Behavior**:
- Build.rs matches `*.lib` extension
- No RPATH (Windows doesn't support it)
- Runtime uses PATH variable for DLL discovery

**Test File**: Platform-specific test in `crossval_detection/` suite

---

## 4. Implementation Plan

### 4.1 Test File Structure

```
tests/integration/
├── fixtures/
│   ├── mod.rs                    # Public API for fixture generation
│   ├── directory_layouts.rs      # Layout templates (standard, custom, etc.)
│   ├── mock_libraries.rs         # Mock .so/.dylib/.lib generators
│   └── env_isolation.rs          # EnvGuard wrapper + temp dir cleanup
│
├── crossval_detection/
│   ├── mod.rs                    # Shared utilities
│   ├── bitnet_standard_layout.rs # Scenario 1
│   ├── bitnet_custom_libdir.rs   # Scenario 2
│   ├── llama_standalone.rs       # Scenario 3
│   ├── dual_backend.rs           # Scenario 4
│   └── missing_libs.rs           # Scenario 5
│
├── rpath_embedding/
│   ├── mod.rs                    # Platform detection + tool wrappers
│   ├── linux_rpath.rs            # Scenario 6 (readelf)
│   ├── macos_rpath.rs            # Scenario 7 (otool)
│   └── windows_linkage.rs        # Scenario 8 (dumpbin)
│
├── preflight_diagnostics/
│   ├── mod.rs                    # Output parsing utilities
│   ├── success_output.rs         # Scenario 9
│   ├── failure_output.rs         # Scenario 10
│   ├── verbose_mode.rs           # Scenario 11
│   └── search_path_display.rs    # Scenario 12
│
└── env_var_precedence/
    ├── mod.rs                    # Env var helpers
    ├── crossval_libdir.rs        # Scenario 13
    ├── bitnet_cpp_dir.rs         # Scenario 14
    ├── default_cache.rs          # Scenario 15
    └── legacy_path_var.rs        # Scenario 16
```

### 4.2 Implementation Phases

**Phase 1: Fixture Infrastructure** (Priority: Critical)
- Implement `fixtures/mod.rs` with public API
- Create `DirectoryLayout` builder pattern
- Implement `MockLibrary` generator for all platforms
- Add `EnvGuard` integration for test isolation

**Phase 2: Detection Tests** (Priority: High)
- Implement scenarios 1-5 (crossval_detection suite)
- Validate library discovery logic
- Test environment variable precedence
- Ensure parallel execution safety

**Phase 3: RPATH Validation** (Priority: High)
- Implement platform-specific RPATH tests (scenarios 6-8)
- Add tool wrappers (readelf, otool, dumpbin)
- Validate linker output correctness
- Add graceful skips for missing tools

**Phase 4: Preflight Diagnostics** (Priority: Medium)
- Implement scenarios 9-12 (preflight output validation)
- Create output parsing utilities
- Test verbose vs non-verbose modes
- Validate error message formatting

**Phase 5: CI Integration** (Priority: Medium)
- Add GitHub Actions workflow for integration tests
- Configure platform-specific test runners (Linux, macOS, Windows)
- Set up fixture caching (avoid regeneration)
- Add test result reporting

**Phase 6: Documentation** (Priority: Low)
- Write test suite README
- Document fixture API usage
- Add troubleshooting guide
- Create visual test coverage report

### 4.3 File Locations

**New Directories**:
```
tests/integration/          # New top-level integration test directory
  ├── fixtures/             # Fixture generation library
  ├── crossval_detection/   # Detection test suite
  ├── rpath_embedding/      # Binary validation suite
  ├── preflight_diagnostics/ # Output formatting suite
  └── env_var_precedence/   # Environment resolution suite
```

**Existing Files to Modify**:
- `tests/helpers/env_guard.rs` - Extend for fixture temp dir management
- `.github/workflows/ci.yml` - Add integration test job
- `Cargo.toml` - Add integration test dependencies (tempfile, regex)

**New Dependencies**:
```toml
[dev-dependencies]
tempfile = "3"           # Temporary directory management
regex = "1"              # Output pattern matching
assert_cmd = "2"         # Command execution helpers
predicates = "3"         # Assertion utilities
serial_test = "3"        # Existing - parallel execution control
```

---

## 5. Fixture Requirements

### 5.1 Directory Layout Templates

**Standard BitNet.cpp Layout**:
```rust
pub struct DirectoryLayoutBuilder {
    root: PathBuf,
    layout_type: LayoutType,
    include_libs: bool,
    include_headers: bool,
}

pub enum LayoutType {
    BitNetStandard,       // Microsoft/BitNet standard build
    LlamaStandalone,      // llama.cpp standalone
    CustomLibDir,         // BITNET_CROSSVAL_LIBDIR override
    DualBackend,          // BitNet embedding llama
    MissingLibs,          // Headers only, no libs
}

impl DirectoryLayoutBuilder {
    pub fn new(layout_type: LayoutType) -> Self { /* ... */ }
    pub fn with_libs(mut self, enabled: bool) -> Self { /* ... */ }
    pub fn with_headers(mut self, enabled: bool) -> Self { /* ... */ }
    pub fn build(self) -> Result<DirectoryLayout, Error> { /* ... */ }
}

pub struct DirectoryLayout {
    root: PathBuf,
    cleanup_guard: TempDirCleanup,
}

impl DirectoryLayout {
    pub fn root_path(&self) -> &Path { &self.root }
    pub fn lib_paths(&self) -> Vec<PathBuf> { /* ... */ }
    pub fn header_paths(&self) -> Vec<PathBuf> { /* ... */ }
}
```

**Usage Example**:
```rust
let layout = DirectoryLayoutBuilder::new(LayoutType::BitNetStandard)
    .with_libs(true)
    .with_headers(true)
    .build()?;

// Creates:
// {temp}/bitnet_cpp/
// ├── include/ggml-bitnet.h
// ├── 3rdparty/llama.cpp/include/llama.h
// └── build/lib/
//     ├── libbitnet.so
//     ├── libllama.so
//     └── libggml.so
```

### 5.2 Mock Library Generator

**API Design**:
```rust
pub struct MockLibrary {
    name: String,
    platform: Platform,
    output_path: PathBuf,
}

pub enum Platform {
    Linux,   // .so
    MacOS,   // .dylib
    Windows, // .lib
}

impl MockLibrary {
    pub fn new(name: &str, platform: Platform) -> Self { /* ... */ }
    pub fn generate(&self) -> Result<(), Error> { /* ... */ }
}
```

**Usage Example**:
```rust
let lib = MockLibrary::new("llama", Platform::Linux);
lib.generate()?; // Creates empty libllama.so with ELF header
```

**Implementation Details**:
- Linux: Minimal ELF header (magic bytes `\x7fELF`)
- macOS: Minimal Mach-O header (magic bytes `0xFEEDFACE`)
- Windows: Minimal PE header (magic bytes `MZ`)
- No symbols/code (build.rs only checks file existence)

### 5.3 Environment Isolation

**EnvGuard Extension**:
```rust
pub struct TestEnvironment {
    env_guards: Vec<EnvGuard>,
    temp_dir: TempDir,
}

impl TestEnvironment {
    pub fn new() -> Self { /* ... */ }
    pub fn set_bitnet_cpp_dir(&mut self, path: PathBuf) { /* ... */ }
    pub fn set_crossval_libdir(&mut self, path: PathBuf) { /* ... */ }
    pub fn temp_path(&self) -> &Path { self.temp_dir.path() }
}

impl Drop for TestEnvironment {
    fn drop(&mut self) {
        // Cleanup env vars and temp directory
    }
}
```

**Usage Example**:
```rust
#[test]
#[serial(bitnet_env)]
fn test_custom_libdir() {
    let mut env = TestEnvironment::new();
    let layout = DirectoryLayoutBuilder::new(LayoutType::CustomLibDir)
        .build()
        .unwrap();

    env.set_crossval_libdir(layout.lib_paths()[0].clone());

    // Run test with isolated environment
    // ...
}
```

---

## 6. CI/CD Integration

### 6.1 GitHub Actions Workflow

**New Job**: `integration-tests`

```yaml
name: Integration Tests

on:
  push:
    branches: [main, feat/*]
  pull_request:

jobs:
  integration-tests:
    name: Integration Tests (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        include:
          - os: ubuntu-latest
            platform: linux
            readelf: readelf
          - os: macos-latest
            platform: macos
            otool: otool
          - os: windows-latest
            platform: windows
            dumpbin: dumpbin

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install platform tools (Linux)
        if: matrix.platform == 'linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y binutils

      - name: Install platform tools (macOS)
        if: matrix.platform == 'macos'
        run: |
          # otool pre-installed with Xcode
          xcode-select --install || true

      - name: Install platform tools (Windows)
        if: matrix.platform == 'windows'
        shell: pwsh
        run: |
          # dumpbin comes with MSVC (already in GitHub Actions)
          Write-Host "Using MSVC toolchain"

      - name: Run integration tests
        run: |
          cargo test --test '*' --no-default-features --features cpu -- --test-threads=1

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: integration-test-results-${{ matrix.platform }}
          path: target/nextest/junit.xml
```

### 6.2 Test Execution Configuration

**Nextest Profile** (`.config/nextest.toml`):
```toml
[profile.integration]
threads = 1                  # Serial execution for env isolation
retries = 0                  # No retries (deterministic fixtures)
test-timeout = { slow-timeout = "60s" }  # Longer timeout for fixture setup
success-output = "never"     # Reduce noise
failure-output = "immediate" # Show failures immediately
```

**Run Command**:
```bash
cargo nextest run --profile integration --test '*'
```

### 6.3 Fixture Caching Strategy

**Problem**: Fixture generation adds overhead to test execution.

**Solution**: Cache generated fixtures between test runs (local dev only).

**Implementation**:
```rust
pub struct FixtureCache {
    cache_dir: PathBuf,  // target/fixture-cache/
}

impl FixtureCache {
    pub fn get_or_create(&self, layout: LayoutType) -> Result<DirectoryLayout, Error> {
        let cache_key = format!("{:?}", layout);
        let cached_path = self.cache_dir.join(&cache_key);

        if cached_path.exists() {
            return Ok(DirectoryLayout::from_existing(cached_path));
        }

        // Generate new fixture
        let layout = DirectoryLayoutBuilder::new(layout).build()?;
        self.persist(&cache_key, &layout)?;
        Ok(layout)
    }
}
```

**Note**: CI always regenerates fixtures (no caching) to ensure correctness.

---

## 7. Success Criteria

### 7.1 Test Coverage Metrics

| Category | Scenarios | Success Threshold |
|----------|-----------|-------------------|
| Library Detection | 5 | 100% pass (all platforms) |
| RPATH Embedding | 3 | 100% pass (Linux/macOS), skip on Windows |
| Preflight Diagnostics | 4 | 100% pass (output format validation) |
| Env Var Precedence | 4 | 100% pass (resolution order correct) |
| Platform-Specific | 3 | 100% pass (per-platform) |

**Total Scenarios**: 19
**Required Pass Rate**: 100% on each platform

### 7.2 Validation Criteria

**Per-Test Success**:
- Fixture generated correctly (directory structure valid)
- Build.rs executes without errors
- Detection flags match expectations (HAS_BITNET, HAS_LLAMA)
- RPATH embedded correctly (if applicable)
- Preflight output matches schema
- No flaky failures (deterministic execution)

**Overall Success**:
- All tests pass on CI (Linux, macOS, Windows)
- No regression in existing tests
- Test execution time < 5 minutes per platform
- Clear failure diagnostics (pinpoint root cause)

### 7.3 Documentation Requirements

**Test Suite README** (`tests/integration/README.md`):
- Architecture overview
- Fixture API documentation
- Running tests locally
- Adding new test scenarios
- Troubleshooting guide

**Fixture API Docs**:
- Rustdoc for all public types
- Usage examples for each layout type
- Platform-specific notes

---

## 8. Non-Goals

**Explicitly Out of Scope**:

1. **Functional C++ Integration Testing**
   - Not testing actual BitNet.cpp functionality
   - Only testing detection and configuration
   - No real inference or cross-validation execution

2. **Performance Benchmarking**
   - Not measuring detection speed
   - Not optimizing fixture generation time
   - Focus on correctness, not performance

3. **Real BitNet.cpp Builds**
   - No compilation of actual C++ code
   - No CMake execution
   - Pure fixture-based testing

4. **Cross-Backend Parity Testing**
   - Not testing inference output parity
   - Only testing configuration parity
   - See `crossval-per-token` command for parity tests

5. **Dynamic Library Loading**
   - Not testing dlopen/LoadLibrary at runtime
   - Only testing build-time linking
   - Runtime behavior validated separately

---

## Appendix A: API Reference

### DirectoryLayoutBuilder

```rust
pub struct DirectoryLayoutBuilder {
    layout_type: LayoutType,
    include_libs: bool,
    include_headers: bool,
    custom_paths: Vec<PathBuf>,
}

impl DirectoryLayoutBuilder {
    pub fn new(layout_type: LayoutType) -> Self;
    pub fn with_libs(mut self, enabled: bool) -> Self;
    pub fn with_headers(mut self, enabled: bool) -> Self;
    pub fn add_custom_path(mut self, path: PathBuf) -> Self;
    pub fn build(self) -> Result<DirectoryLayout, FixtureError>;
}

pub enum LayoutType {
    BitNetStandard,
    LlamaStandalone,
    CustomLibDir,
    DualBackend,
    MissingLibs,
}

pub struct DirectoryLayout {
    root: PathBuf,
    cleanup_guard: TempDirCleanup,
}

impl DirectoryLayout {
    pub fn root_path(&self) -> &Path;
    pub fn lib_paths(&self) -> Vec<PathBuf>;
    pub fn header_paths(&self) -> Vec<PathBuf>;
}
```

### MockLibrary

```rust
pub struct MockLibrary {
    name: String,
    platform: Platform,
    output_path: PathBuf,
}

pub enum Platform {
    Linux,
    MacOS,
    Windows,
}

impl MockLibrary {
    pub fn new(name: &str, platform: Platform) -> Self;
    pub fn at_path(mut self, path: PathBuf) -> Self;
    pub fn generate(&self) -> Result<(), FixtureError>;
}
```

### TestEnvironment

```rust
pub struct TestEnvironment {
    env_guards: Vec<EnvGuard>,
    temp_dir: TempDir,
}

impl TestEnvironment {
    pub fn new() -> Self;
    pub fn set_bitnet_cpp_dir(&mut self, path: PathBuf);
    pub fn set_crossval_libdir(&mut self, path: PathBuf);
    pub fn set_home(&mut self, path: PathBuf);
    pub fn temp_path(&self) -> &Path;
}
```

---

## Appendix B: Example Test Implementation

**Scenario 1: BitNet Standard Layout**

```rust
// tests/integration/crossval_detection/bitnet_standard_layout.rs

use serial_test::serial;
use crate::fixtures::{DirectoryLayoutBuilder, LayoutType, MockLibrary, Platform};
use crate::fixtures::TestEnvironment;

#[test]
#[serial(bitnet_env)]
fn test_bitnet_standard_layout_detection() {
    // Setup: Generate fixture
    let mut env = TestEnvironment::new();
    let layout = DirectoryLayoutBuilder::new(LayoutType::BitNetStandard)
        .with_libs(true)
        .with_headers(true)
        .build()
        .expect("Failed to create fixture");

    // Set environment
    env.set_bitnet_cpp_dir(layout.root_path().to_path_buf());

    // Act: Build crossval crate (triggers build.rs)
    let output = std::process::Command::new("cargo")
        .arg("build")
        .arg("-p").arg("crossval")
        .arg("--features").arg("llama-ffi")
        .env("BITNET_CPP_DIR", layout.root_path())
        .output()
        .expect("Failed to run cargo build");

    // Assert: Build succeeded
    assert!(output.status.success(),
        "Build failed: {}", String::from_utf8_lossy(&output.stderr));

    // Assert: Detection flags set correctly
    let build_output = String::from_utf8_lossy(&output.stderr);
    assert!(build_output.contains("CROSSVAL_HAS_BITNET=true"),
        "HAS_BITNET not detected");
    assert!(build_output.contains("CROSSVAL_HAS_LLAMA=true"),
        "HAS_LLAMA not detected");

    // Assert: RPATH includes expected paths
    let lib_path = layout.lib_paths()[0].display().to_string();
    assert!(build_output.contains(&format!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path)),
        "RPATH not embedded");

    // Act: Run preflight command
    let preflight_output = std::process::Command::new("cargo")
        .arg("run")
        .arg("-p").arg("xtask")
        .arg("--").arg("preflight")
        .arg("--backend").arg("bitnet")
        .env("BITNET_CPP_DIR", layout.root_path())
        .output()
        .expect("Failed to run preflight");

    // Assert: Preflight shows success
    let preflight_text = String::from_utf8_lossy(&preflight_output.stdout);
    assert!(preflight_text.contains("✓ Backend 'bitnet.cpp' libraries found"),
        "Preflight did not show success");
}
```

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0.0 | 2025-10-25 | Initial specification | spec-creator |

---

## Next Steps

**For Implementation**:
1. Review specification with team
2. Prioritize Phase 1 (fixture infrastructure)
3. Create GitHub issue for tracking
4. Begin implementation of `tests/integration/fixtures/`

**For Review**:
- [ ] Validate test scenario coverage
- [ ] Confirm CI integration approach
- [ ] Review platform-specific tool dependencies
- [ ] Approve fixture API design
