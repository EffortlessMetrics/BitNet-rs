# Integration Test Fixture Infrastructure

Comprehensive test scaffolding for BitNet.cpp auto-configuration integration tests.

## Test Specification

- **Specification**: `docs/specs/bitnet-integration-tests.md`
- **Phase**: Phase 1 - Fixture Infrastructure (Section 4.1)
- **Status**: Test scaffolding complete, compilation verified

## Components

### 1. DirectoryLayoutBuilder (`directory_layouts.rs`)

Creates temporary directory structures matching real BitNet.cpp/llama.cpp layouts.

**Supported Layouts:**
- `BitNetStandard`: Standard BitNet.cpp CMake build layout
- `LlamaStandalone`: Standalone llama.cpp (no BitNet)
- `CustomLibDir`: Custom library directory override (BITNET_CROSSVAL_LIBDIR)
- `DualBackend`: BitNet embedding llama.cpp
- `MissingLibs`: Headers only (graceful failure testing)

**Tests Created:** 10 tests validating all 5 layout types

### 2. MockLibrary (`mock_libraries.rs`)

Generates platform-specific stub library files with correct file headers.

**Platform Support:**
- Linux: `.so` files with ELF headers (`\x7fELF`)
- macOS: `.dylib` files with Mach-O headers (`0xFEEDFACF`)
- Windows: `.lib` files with PE headers (`MZ`)

**Library Types:**
- `BitNet`: libbitnet.{so,dylib,lib}
- `Llama`: libllama.{so,dylib,lib}
- `Ggml`: libggml.{so,dylib,lib}

**Tests Created:** 6 tests (unit tests + platform-specific header validation)

### 3. TestEnvironment (`env_isolation.rs`)

Provides environment isolation using EnvGuard pattern with automatic cleanup.

**Features:**
- Automatic environment variable restoration
- Temporary directory management with cleanup
- Serial execution safety with `#[serial(bitnet_env)]`
- Support for BITNET_CPP_DIR, BITNET_CROSSVAL_LIBDIR, HOME, LD_LIBRARY_PATH

**Tests Created:** 6 tests validating environment isolation and cleanup

### 4. Integration Tests (`tests.rs`)

Comprehensive tests combining all fixture components.

**Test Coverage:**
- DirectoryLayoutBuilder API (5 layout types)
- MockLibrary generation (Linux .so, macOS .dylib, Windows .lib)
- TestEnvironment isolation (temp dir cleanup, env restoration)
- Layout validation (headers present, libs present)
- Complete workflow integration tests

**Tests Created:** 24 tests

## Total Test Count

**36 tests** created across 4 modules:
- `directory_layouts.rs`: 10 tests
- `mock_libraries.rs`: 6 tests
- `env_isolation.rs`: 6 tests
- `tests.rs`: 24 tests (including integration scenarios)

## Compilation Verification

```bash
# Verified with default features
cargo test --features integration-tests --no-run -p bitnet-tests
# Status: ✓ PASSED

# Verified with library tests only
cargo test --lib --features integration-tests --no-run -p bitnet-tests
# Status: ✓ PASSED
```

## Usage Example

```rust
use serial_test::serial;
use fixtures::{DirectoryLayoutBuilder, LayoutType, TestEnvironment};

#[test]
#[serial(bitnet_env)]
fn test_bitnet_standard_layout() {
    let mut env = TestEnvironment::new().unwrap();
    let layout = DirectoryLayoutBuilder::new(LayoutType::BitNetStandard)
        .with_libs(true)
        .with_headers(true)
        .build()
        .unwrap();

    env.set_bitnet_cpp_dir(layout.root_path().to_path_buf());

    // Verify structure
    assert!(layout.has_library(LibraryType::BitNet));
    assert!(layout.has_library(LibraryType::Llama));
    assert!(layout.has_header("ggml-bitnet.h"));
}
```

## Next Steps (Future Phases)

Phase 2-5 from specification:
- [ ] Phase 2: Detection Tests (crossval_detection suite)
- [ ] Phase 3: RPATH Validation (rpath_embedding suite)
- [ ] Phase 4: Preflight Diagnostics (preflight_diagnostics suite)
- [ ] Phase 5: CI Integration

## Traceability

All tests are linked to specification sections via doc comments:
- Tests feature spec: `bitnet-integration-tests.md#implementation-plan-phase-1`
- Scenario references: `bitnet-integration-tests.md#scenario-{1-19}`

## Feature Gate

Tests are compiled when the `integration-tests` feature is enabled:

```toml
[features]
integration-tests = []
```

## Test Status

✅ **All tests compile successfully**
- No syntax errors
- No missing dependencies
- Proper feature gating with `#[cfg(feature = "integration-tests")]`
- Platform-specific tests use appropriate `#[cfg(target_os = "...")]`
- Environment isolation tests use `#[serial(bitnet_env)]`
