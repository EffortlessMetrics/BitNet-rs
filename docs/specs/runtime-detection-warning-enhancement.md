# Runtime Detection Warning Enhancement Specification

**Status**: Draft
**Author**: BitNet.rs Specification Agent
**Date**: 2025-10-27
**Related Issues**: N/A (Enhancement)
**Related Documents**:
- `/tmp/explore_runtime_detection.md` (Architectural Analysis)
- `docs/howto/cpp-setup.md` (C++ Reference Setup)
- `docs/explanation/dual-backend-crossval.md` (Dual-Backend Architecture)

---

## Executive Summary

The BitNet.rs test infrastructure uses a dual-detection system for C++ backend libraries:

1. **Build-time constants**: `HAS_BITNET`, `HAS_LLAMA` baked into xtask binary during compilation
2. **Runtime fallback**: `detect_backend_runtime()` searches filesystem for libraries after xtask is built

**Problem**: When users install C++ libraries after building xtask, runtime detection succeeds but build-time constants remain stale. Users don't receive clear warnings that they need to rebuild xtask to update the embedded constants.

**Solution**: Implement prominent, actionable warnings when runtime detection differs from build-time detection, with environment-aware behavior (dev vs CI modes).

---

## Requirements Analysis

### Functional Requirements

**FR1: Runtime vs Build-Time Mismatch Detection**
- Detect when `detect_backend_runtime()` finds libraries but build-time constants are `false`
- Track which backend was detected (bitnet vs llama)
- Record which search path matched

**FR2: User-Facing Warning Messages**
- Print single-line, prominent warning when mismatch detected
- Include exact rebuild command with correct feature flags
- Show backend name and matched path
- Provide verbose mode with full search diagnostics

**FR3: Environment-Aware Behavior**
- **Dev mode** (default): Allow tests to proceed with warning
- **CI mode**: Respect build-time constants only, no runtime override
- Detect CI environments via standard environment variables

**FR4: Verbose Diagnostics**
- Show all search paths attempted
- Indicate which paths exist vs missing
- List libraries found in each path
- Display environment variable configuration

---

## Architecture Approach

### Component Integration

**Target Crate**: `xtask/src/crossval/preflight.rs`

**Integration Points**:
1. `ensure_backend_or_skip()` (tests/support/backend_helpers.rs)
2. `detect_backend_runtime()` (tests/support/backend_helpers.rs)
3. `preflight_backend_libs()` (xtask/src/crossval/preflight.rs)

### Warning Placement Strategy

```
┌─────────────────────────────────────────────────────────┐
│  Test Execution Flow                                    │
│                                                          │
│  ensure_backend_or_skip(backend)                        │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────┼────────┐
        │                 │
   ┌────▼──────┐    ┌────▼──────────┐
   │ Priority1 │    │   Priority2   │
   │ Build-Time│    │   Runtime     │
   │ Constants │    │   Detection   │
   │ (Fastest) │    │   (Fallback)  │
   └────┬──────┘    └────┬──────────┘
        │                │
        │ Available      │ Found
        │                │ (warn)
        │           ┌────▼──────────────┐
        │           │ WARNING EMISSION  │
        │           │ ⚠️  Stale Build   │
        │           │ Dev: Continue     │
        │           │ CI: Skip/Fail     │
        │           └────┬──────────────┘
   ┌────▼────────────────▼─────────┐
   │  Test Continues (dev mode)    │
   │  OR Skip (CI mode)            │
   └───────────────────────────────┘
```

---

## Warning Message Specification

### 1. Standard Warning (Dev Mode)

**Format**: Single-line, prefixed with warning symbol, actionable rebuild command

```
⚠️  STALE BUILD: bitnet.cpp found at runtime but not at build time. Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all
```

**Location**: Emitted in `ensure_backend_or_skip()` after runtime detection succeeds

**Output Stream**: `stderr` (warnings convention)

**Timing**: Emitted once per test run, deduplicated via static flag

### 2. Verbose Warning (--verbose flag)

**Format**: Multi-line with context, search diagnostics, environment state

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  STALE BUILD DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Backend 'bitnet.cpp' found at runtime but not at xtask build time.

This happens when:
  1. You built xtask
  2. Then installed BitNet.cpp libraries later
  3. xtask binary still contains old detection constants

Why rebuild is needed:
  • Library detection runs at BUILD time (not runtime)
  • Results are baked into the xtask binary as constants
  • Runtime detection is a fallback for developer convenience
  • Rebuild refreshes the constants to match filesystem reality

Runtime Detection Results:
  Matched path: /home/user/.cache/bitnet_cpp/build
  Libraries found: libbitnet.so

Build-Time Detection State:
  HAS_BITNET = false (stale)
  Last xtask build: 2025-10-27 10:30:15 UTC

Fix:
  cargo clean -p crossval && cargo build -p xtask --features crossval-all

Then re-run your test.
```

**Trigger**: Set `VERBOSE=1` or use `--verbose` flag in test harness

**Output Stream**: `stderr`

### 3. CI Mode Behavior

**No Warning**: CI mode respects build-time constants only, runtime detection is ignored

**Skip Message**: Standard skip diagnostic with clear explanation

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⊘ Test skipped: bitnet.cpp not available (CI mode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CI mode detected (CI=1 or BITNET_TEST_NO_REPAIR=1).
Runtime detection found libraries but build-time constants are stale.

In CI mode:
  • Build-time detection is the source of truth
  • Runtime fallback is DISABLED for determinism
  • xtask must be rebuilt to detect libraries

Setup Instructions:
  1. Install backend:
     eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
  2. Rebuild xtask:
     cargo clean -p crossval && cargo build -p xtask --features crossval-all
  3. Re-run CI job
```

---

## Dev vs CI Behavior Logic

### Environment Detection

**CI Detection Function** (already exists in `preflight.rs:411-418`):

```rust
pub fn is_ci() -> bool {
    std::env::var_os("CI").is_some()
        || std::env::var_os("GITHUB_ACTIONS").is_some()
        || std::env::var_os("JENKINS_HOME").is_some()
        || std::env::var_os("GITLAB_CI").is_some()
        || std::env::var_os("CIRCLECI").is_some()
        || std::env::var_os("BITNET_TEST_NO_REPAIR").is_some()
}
```

### Behavior Matrix

| Scenario | Build-Time | Runtime | Dev Mode | CI Mode |
|----------|-----------|---------|----------|---------|
| Libraries found at build | `true` | `true` | ✅ Continue | ✅ Continue |
| Libraries not found at build | `false` | `false` | ⊘ Skip/Repair | ⊘ Skip |
| **Stale build (core scenario)** | `false` | `true` | ⚠️ Continue + Warning | ⊘ Skip (no runtime override) |
| Libraries removed after build | `true` | `false` | ⚠️ Error + Rebuild hint | ❌ Error |

### Decision Tree

```rust
// Pseudo-code for logic flow
fn ensure_backend_or_skip(backend: CppBackend) {
    // Priority 1: Build-time constant check
    let has_at_build_time = match backend {
        CppBackend::BitNet => HAS_BITNET,
        CppBackend::Llama => HAS_LLAMA,
    };

    if has_at_build_time {
        return; // Backend available, proceed
    }

    // Priority 2: Runtime detection fallback
    let runtime_available = detect_backend_runtime(backend).unwrap_or(false);

    if runtime_available {
        // STALE BUILD SCENARIO
        if is_ci() {
            // CI mode: respect build-time constants only
            skip_test_with_ci_message(backend);
        } else {
            // Dev mode: allow test to proceed with warning
            emit_stale_build_warning(backend);
            return; // Continue execution
        }
    } else {
        // Priority 3: Backend not found anywhere
        if is_ci() {
            skip_test_with_setup_instructions(backend);
        } else {
            attempt_auto_repair_or_skip(backend);
        }
    }
}
```

---

## Implementation Approach

### 1. Warning Emission Function

**Location**: `tests/support/backend_helpers.rs`

**Function Signature**:

```rust
/// Emit stale build warning when runtime detection succeeds but build-time constants are false
///
/// # Arguments
///
/// * `backend` - The C++ backend detected at runtime
/// * `matched_path` - The directory path where libraries were found
/// * `verbose` - If true, emit detailed diagnostic output
fn emit_stale_build_warning(
    backend: CppBackend,
    matched_path: &Path,
    verbose: bool,
) {
    static WARNING_EMITTED: std::sync::Once = std::sync::Once::new();

    WARNING_EMITTED.call_once(|| {
        if verbose {
            emit_verbose_stale_warning(backend, matched_path);
        } else {
            emit_standard_stale_warning(backend);
        }
    });
}
```

**Standard Warning Implementation**:

```rust
fn emit_standard_stale_warning(backend: CppBackend) {
    eprintln!(
        "⚠️  STALE BUILD: {} found at runtime but not at build time. \
         Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all",
        backend.name()
    );
}
```

**Verbose Warning Implementation**:

```rust
fn emit_verbose_stale_warning(backend: CppBackend, matched_path: &Path) {
    const SEPARATOR: &str = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

    eprintln!("{}", SEPARATOR);
    eprintln!("⚠️  STALE BUILD DETECTION");
    eprintln!("{}", SEPARATOR);
    eprintln!();
    eprintln!("Backend '{}' found at runtime but not at xtask build time.", backend.name());
    eprintln!();
    eprintln!("This happens when:");
    eprintln!("  1. You built xtask");
    eprintln!("  2. Then installed {} libraries later", backend.name());
    eprintln!("  3. xtask binary still contains old detection constants");
    eprintln!();
    eprintln!("Why rebuild is needed:");
    eprintln!("  • Library detection runs at BUILD time (not runtime)");
    eprintln!("  • Results are baked into the xtask binary as constants");
    eprintln!("  • Runtime detection is a fallback for developer convenience");
    eprintln!("  • Rebuild refreshes the constants to match filesystem reality");
    eprintln!();
    eprintln!("Runtime Detection Results:");
    eprintln!("  Matched path: {}", matched_path.display());

    if let Some(libs) = find_libs_in_path(matched_path, backend) {
        eprintln!("  Libraries found: {}", libs.join(", "));
    }

    eprintln!();
    eprintln!("Build-Time Detection State:");
    eprintln!("  HAS_{} = false (stale)",
        match backend {
            CppBackend::BitNet => "BITNET",
            CppBackend::Llama => "LLAMA",
        }
    );

    if let Some(timestamp) = get_xtask_build_timestamp() {
        eprintln!("  Last xtask build: {}", timestamp);
    }

    eprintln!();
    eprintln!("Fix:");
    eprintln!("  cargo clean -p crossval && cargo build -p xtask --features crossval-all");
    eprintln!();
    eprintln!("Then re-run your test.");
}
```

### 2. Enhanced Runtime Detection

**Location**: `tests/support/backend_helpers.rs`

**Modification**: Return matched path along with availability status

```rust
/// Detect backend at runtime and return matched path
///
/// # Returns
///
/// * `Ok((true, Some(path)))` - Backend found, path where libraries located
/// * `Ok((false, None))` - Backend not found
/// * `Err(String)` - Error during detection
fn detect_backend_runtime(backend: CppBackend) -> Result<(bool, Option<PathBuf>), String> {
    let mut candidates = Vec::new();

    // Priority 1: BITNET_CROSSVAL_LIBDIR override
    if let Ok(p) = std::env::var("BITNET_CROSSVAL_LIBDIR") {
        candidates.push(p.into());
    }

    // Priority 2: Granular backend-specific overrides
    match backend {
        CppBackend::BitNet => {
            if let Ok(p) = std::env::var("CROSSVAL_RPATH_BITNET") {
                candidates.push(p.into());
            }
        }
        CppBackend::Llama => {
            if let Ok(p) = std::env::var("CROSSVAL_RPATH_LLAMA") {
                candidates.push(p.into());
            }
        }
    }

    // Priority 3: Home directory subdirectories
    let home_var = match backend {
        CppBackend::BitNet => "BITNET_CPP_DIR",
        CppBackend::Llama => "LLAMA_CPP_DIR",
    };

    if let Ok(root) = std::env::var(home_var) {
        let root_path = Path::new(&root);
        for sub in ["build", "build/bin", "build/lib"] {
            candidates.push(root_path.join(sub));
        }
    }

    // Platform-specific library extensions
    let exts = if cfg!(target_os = "windows") {
        vec!["dll"]
    } else if cfg!(target_os = "macos") {
        vec!["dylib"]
    } else {
        vec!["so"]
    };

    let needs: &[&str] = match backend {
        CppBackend::BitNet => &["bitnet"],
        CppBackend::Llama => &["llama", "ggml"],
    };

    // Search candidates and return first match
    for dir in &candidates {
        if !dir.exists() {
            continue;
        }

        let all_found = needs.iter().all(|stem| {
            exts.iter().any(|ext| {
                let lib_name = format_lib_name_ext(stem, ext);
                dir.join(&lib_name).exists()
            })
        });

        if all_found {
            return Ok((true, Some(dir.clone())));
        }
    }

    Ok((false, None))
}
```

### 3. Integration in `ensure_backend_or_skip()`

**Location**: `tests/support/backend_helpers.rs`

**Current Code** (lines 85-104):

```rust
// Check runtime detection as fallback (Priority 2)
if let Ok(runtime_available) = detect_backend_runtime(backend) {
    if runtime_available {
        eprintln!(
            "⚠️  {} detected at runtime but not build-time. Rebuild xtask.",
            backend_name(backend)
        );
        eprintln!("    cargo clean -p crossval && cargo build --features crossval-all");
        return; // Backend available at runtime, warn about rebuild but continue
    }
}
```

**Enhanced Code**:

```rust
// Check runtime detection as fallback (Priority 2)
if let Ok((runtime_available, matched_path)) = detect_backend_runtime(backend) {
    if runtime_available {
        // STALE BUILD SCENARIO: Runtime found libs but build-time constant is false

        if is_ci() {
            // CI mode: respect build-time constants only (no runtime override)
            let skip_msg = format_ci_stale_skip_diagnostic(backend, matched_path.as_deref());
            eprintln!("{}", skip_msg);
            std::process::exit(0); // Skip test in CI
        } else {
            // Dev mode: allow test to proceed with warning
            let verbose = std::env::var("VERBOSE").is_ok();
            emit_stale_build_warning(backend, matched_path.as_ref().unwrap(), verbose);
            return; // Continue execution
        }
    }
}
```

### 4. CI Skip Diagnostic

**Location**: `tests/support/backend_helpers.rs`

**Function Signature**:

```rust
/// Format CI-mode skip message when runtime detects libraries but build-time constants are stale
fn format_ci_stale_skip_diagnostic(backend: CppBackend, matched_path: Option<&Path>) -> String {
    const SEPARATOR: &str = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

    let mut msg = String::new();
    msg.push_str(&format!("{}\n", SEPARATOR));
    msg.push_str(&format!("⊘ Test skipped: {} not available (CI mode)\n", backend.name()));
    msg.push_str(&format!("{}\n\n", SEPARATOR));

    msg.push_str("CI mode detected (CI=1 or BITNET_TEST_NO_REPAIR=1).\n");
    msg.push_str("Runtime detection found libraries but build-time constants are stale.\n\n");

    if let Some(path) = matched_path {
        msg.push_str(&format!("Runtime found libraries at: {}\n", path.display()));
        msg.push_str("But xtask was built before libraries were installed.\n\n");
    }

    msg.push_str("In CI mode:\n");
    msg.push_str("  • Build-time detection is the source of truth\n");
    msg.push_str("  • Runtime fallback is DISABLED for determinism\n");
    msg.push_str("  • xtask must be rebuilt to detect libraries\n\n");

    msg.push_str("Setup Instructions:\n");
    msg.push_str("  1. Install backend:\n");
    msg.push_str(&format!("     {}\n", backend.setup_command()));
    msg.push_str("  2. Rebuild xtask:\n");
    msg.push_str("     cargo clean -p crossval && cargo build -p xtask --features crossval-all\n");
    msg.push_str("  3. Re-run CI job\n");

    msg
}
```

---

## Environment Variable Contracts

### Detection Environment Variables

| Variable | Purpose | Used By | Priority |
|----------|---------|---------|----------|
| `BITNET_CROSSVAL_LIBDIR` | Explicit global library directory override | Runtime detection, build-time search | 1 (highest) |
| `CROSSVAL_RPATH_BITNET` | BitNet-specific library path override | Runtime detection | 2 |
| `CROSSVAL_RPATH_LLAMA` | Llama-specific library path override | Runtime detection | 2 |
| `BITNET_CPP_DIR` | BitNet.cpp installation root | Runtime detection, build-time search | 3 |
| `LLAMA_CPP_DIR` | llama.cpp installation root | Runtime detection, build-time search | 3 |

### Control Environment Variables

| Variable | Purpose | Values | Default |
|----------|---------|--------|---------|
| `CI` | CI environment detection | `1` or any value | unset (local dev) |
| `GITHUB_ACTIONS` | GitHub Actions CI detection | `true` | unset |
| `BITNET_TEST_NO_REPAIR` | Disable auto-repair (force CI mode) | `1` or any value | unset |
| `VERBOSE` | Enable verbose diagnostics | `1` or any value | unset |

### Warning Control Variables (New)

| Variable | Purpose | Values | Default |
|----------|---------|--------|---------|
| `BITNET_SUPPRESS_STALE_WARNING` | Suppress stale build warnings (testing only) | `1` | unset |
| `BITNET_FORCE_VERBOSE_WARNING` | Force verbose warning even without `VERBOSE=1` | `1` | unset |

---

## Testing Strategy

### Unit Tests

**Test File**: `tests/support/backend_helpers_tests.rs` (new file)

**Test Coverage**:

1. **Test: Warning emission deduplication**
   - Verify `std::sync::Once` ensures single warning per test run
   - Expected: Second call to `emit_stale_build_warning()` is no-op

2. **Test: Standard vs verbose warning format**
   - Mock `VERBOSE=1`, verify verbose output contains diagnostics
   - Mock `VERBOSE=unset`, verify single-line output
   - Expected: Format matches specification

3. **Test: CI mode skip logic**
   - Mock `CI=1`, verify `ensure_backend_or_skip()` exits with 0
   - Expected: No test execution, CI diagnostic message emitted

4. **Test: Dev mode continue logic**
   - Mock `CI=unset`, verify test continues after warning
   - Expected: Warning emitted, function returns (no exit)

5. **Test: Matched path extraction**
   - Mock library files in temp directory
   - Verify `detect_backend_runtime()` returns correct path
   - Expected: Path where libraries found is returned

### Integration Tests

**Test File**: `xtask/tests/stale_build_detection_tests.rs` (new file)

**Test Scenarios**:

1. **Scenario: Fresh build (no warning)**
   - Setup: Build xtask with libraries already installed
   - Action: Run test requiring backend
   - Expected: No warning, test proceeds

2. **Scenario: Stale build (dev mode warning)**
   - Setup: Build xtask, then install libraries
   - Action: Run test requiring backend (CI=unset)
   - Expected: Warning emitted, test proceeds

3. **Scenario: Stale build (CI mode skip)**
   - Setup: Build xtask, then install libraries
   - Action: Run test requiring backend (CI=1)
   - Expected: Skip diagnostic, test exits 0

4. **Scenario: Verbose mode diagnostics**
   - Setup: Stale build scenario
   - Action: Run test with VERBOSE=1
   - Expected: Multi-line diagnostic with search paths

5. **Scenario: Multiple backends (dual stale)**
   - Setup: Build xtask, install both BitNet and Llama
   - Action: Run tests requiring both backends
   - Expected: Warning deduplicated per backend

---

## Success Criteria

### Acceptance Criteria

**AC1: Runtime vs Build-Time Mismatch Detection**
- ✅ Warning emitted when `detect_backend_runtime()` succeeds but `HAS_BITNET/HAS_LLAMA` is `false`
- ✅ Backend name (bitnet.cpp or llama.cpp) shown in warning
- ✅ Matched path displayed in verbose mode

**AC2: Exact Rebuild Command**
- ✅ Warning includes: `cargo clean -p crossval && cargo build -p xtask --features crossval-all`
- ✅ Command is copy-pasteable (no line breaks in standard warning)

**AC3: Dev Mode Allows Continuation**
- ✅ Test proceeds after warning in local dev environment
- ✅ No `std::process::exit()` called in dev mode
- ✅ Warning appears once per test run (deduplication)

**AC4: CI Mode Respects Build-Time Only**
- ✅ Test skipped when `CI=1` and runtime detection finds libs but build-time is stale
- ✅ Skip diagnostic explains why (determinism, build-time source of truth)
- ✅ Exit code 0 (skip, not failure)

**AC5: Backend Name Display**
- ✅ Warning shows "bitnet.cpp" or "llama.cpp" explicitly
- ✅ Verbose mode shows backend name and required libraries

**AC6: Matched Path Display**
- ✅ Verbose mode shows which directory path contained libraries
- ✅ Path displayed: absolute canonical path

**AC7: Verbose Mode Search Diagnostics**
- ✅ Show all search paths attempted (Priority 1-3)
- ✅ Indicate which paths exist vs missing
- ✅ List libraries found in matched path
- ✅ Display environment variable configuration (BITNET_CPP_DIR, etc.)

### Performance Requirements

**Latency**:
- Warning emission: < 5ms (formatting overhead)
- Runtime detection: < 50ms (filesystem checks)
- No performance impact on CI skip path (early exit)

**Memory**:
- Static warning deduplication: 1 byte per backend (2 bytes total)
- Matched path storage: stack-allocated PathBuf

---

## Risk Assessment

### Technical Risks

**Risk 1: Warning Fatigue**
- **Impact**: Users ignore warnings after seeing repeatedly
- **Mitigation**: Deduplicate per test run via `std::sync::Once`
- **Mitigation**: Provide `BITNET_SUPPRESS_STALE_WARNING=1` escape hatch for CI

**Risk 2: CI Determinism Breakage**
- **Impact**: Runtime override causes non-deterministic CI behavior
- **Mitigation**: Strict CI mode disables runtime override entirely
- **Mitigation**: Exit code 0 (skip) prevents spurious CI failures

**Risk 3: Confusing Error Messages**
- **Impact**: Users don't understand "stale build" concept
- **Mitigation**: Verbose mode explains WHY rebuild is needed
- **Mitigation**: Single-line warning focuses on action (rebuild command)

**Risk 4: Multiple Backend Warnings**
- **Impact**: Dual-backend tests emit warnings for both backends
- **Mitigation**: Per-backend deduplication via `Once` flag
- **Mitigation**: Consider unified "stale build" message for multiple backends

---

## Implementation Timeline

### Phase 1: Core Warning Logic (Week 1)

**Deliverables**:
- `emit_stale_build_warning()` function
- `format_ci_stale_skip_diagnostic()` function
- `detect_backend_runtime()` enhanced to return matched path
- Unit tests for warning emission

**Validation**:
- Unit tests pass
- Warning format matches specification
- Deduplication verified

### Phase 2: Integration (Week 2)

**Deliverables**:
- Integration in `ensure_backend_or_skip()`
- CI mode skip logic
- Verbose mode diagnostics
- Integration tests

**Validation**:
- Stale build scenario reproduces warning
- CI mode skip behavior verified
- Verbose output contains all diagnostic fields

### Phase 3: Documentation (Week 3)

**Deliverables**:
- Update `docs/howto/cpp-setup.md` with warning explanation
- Add troubleshooting section for stale build warnings
- Update `CLAUDE.md` with environment variable contracts

**Validation**:
- Documentation reviewed
- User workflow tested end-to-end

---

## Open Questions

1. **Q: Should we auto-rebuild xtask on stale detection?**
   - **A**: No. Rebuilding during test execution breaks incremental builds and test isolation. User-initiated rebuild is safer.

2. **Q: Should we provide a `--force-rebuild` flag to auto-trigger rebuild?**
   - **A**: Future enhancement. Current spec focuses on warning-only approach.

3. **Q: How to handle dual-backend stale builds?**
   - **A**: Emit separate warnings per backend. Future enhancement: unified warning.

4. **Q: Should we cache matched paths to avoid repeated filesystem searches?**
   - **A**: No. Detection runs once per test process (deduplication via `Once`), caching adds complexity for minimal benefit.

5. **Q: Should we emit warning in CI mode before skipping?**
   - **A**: Yes, but as part of skip diagnostic (not standalone warning). CI gets full context for debugging.

---

## References

### Internal Documentation

- **Exploration Document**: `/tmp/explore_runtime_detection.md`
- **C++ Setup Guide**: `docs/howto/cpp-setup.md`
- **Dual-Backend Architecture**: `docs/explanation/dual-backend-crossval.md`

### Source Code References

- **Preflight Module**: `xtask/src/crossval/preflight.rs`
- **Backend Helpers**: `tests/support/backend_helpers.rs`
- **Build Script**: `crossval/build.rs`

### Related Issues

- **Issue #469**: Tokenizer parity and FFI build hygiene
- **PR #475**: Feature gate unification (GPU/CPU predicates)

---

## Appendix A: Example Warning Scenarios

### Scenario 1: Fresh Install (No Warning)

**Setup**:
```bash
# Install libraries first
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Build xtask (detects libraries)
cargo build -p xtask --features crossval-all
```

**Result**: `HAS_BITNET=true`, no warning

**Test Output**:
```
✓ Backend 'bitnet.cpp' libraries found
test preflight_bitnet_backend ... ok
```

---

### Scenario 2: Stale Build (Dev Mode)

**Setup**:
```bash
# Build xtask first (no libraries)
cargo build -p xtask --features crossval-all

# Install libraries after build
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
```

**Result**: `HAS_BITNET=false`, runtime finds libraries

**Test Output** (standard):
```
⚠️  STALE BUILD: bitnet.cpp found at runtime but not at build time. Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all
test preflight_bitnet_backend ... ok
```

**Test Output** (verbose with `VERBOSE=1`):
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  STALE BUILD DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Backend 'bitnet.cpp' found at runtime but not at xtask build time.

This happens when:
  1. You built xtask
  2. Then installed BitNet.cpp libraries later
  3. xtask binary still contains old detection constants

Why rebuild is needed:
  • Library detection runs at BUILD time (not runtime)
  • Results are baked into the xtask binary as constants
  • Runtime detection is a fallback for developer convenience
  • Rebuild refreshes the constants to match filesystem reality

Runtime Detection Results:
  Matched path: /home/user/.cache/bitnet_cpp/build
  Libraries found: libbitnet.so

Build-Time Detection State:
  HAS_BITNET = false (stale)
  Last xtask build: 2025-10-27 10:30:15 UTC

Fix:
  cargo clean -p crossval && cargo build -p xtask --features crossval-all

Then re-run your test.

test preflight_bitnet_backend ... ok
```

---

### Scenario 3: Stale Build (CI Mode)

**Setup**: Same as Scenario 2, but `CI=1` set

**Test Output**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⊘ Test skipped: bitnet.cpp not available (CI mode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CI mode detected (CI=1 or BITNET_TEST_NO_REPAIR=1).
Runtime detection found libraries but build-time constants are stale.

Runtime found libraries at: /home/user/.cache/bitnet_cpp/build
But xtask was built before libraries were installed.

In CI mode:
  • Build-time detection is the source of truth
  • Runtime fallback is DISABLED for determinism
  • xtask must be rebuilt to detect libraries

Setup Instructions:
  1. Install backend:
     eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
  2. Rebuild xtask:
     cargo clean -p crossval && cargo build -p xtask --features crossval-all
  3. Re-run CI job

test preflight_bitnet_backend ... SKIPPED (exit 0)
```

---

## Appendix B: Environment Variable Decision Tree

```
User Question: "Why isn't my backend detected?"

├─ Did you install libraries AFTER building xtask?
│  ├─ YES → See STALE BUILD warning → Rebuild xtask
│  └─ NO → Continue to next check
│
├─ Is CI=1 set?
│  ├─ YES → Runtime detection DISABLED → Must rebuild xtask
│  └─ NO → Runtime detection enabled → Check paths
│
├─ Which environment variable should I use?
│  ├─ BITNET_CROSSVAL_LIBDIR → Unified library directory (Priority 1)
│  ├─ CROSSVAL_RPATH_BITNET → BitNet-specific path (Priority 2)
│  ├─ CROSSVAL_RPATH_LLAMA → Llama-specific path (Priority 2)
│  ├─ BITNET_CPP_DIR → BitNet installation root (Priority 3)
│  └─ LLAMA_CPP_DIR → Llama installation root (Priority 3)
│
└─ How do I see what paths are searched?
   └─ Run with VERBOSE=1 → Shows all search paths and detection results
```

---

## Appendix C: Future Enhancements

### Enhancement 1: Auto-Rebuild on Stale Detection (Interactive Mode)

**Proposal**: Prompt user to rebuild xtask when stale build detected in interactive terminal

**Implementation**:
```rust
if !is_ci() && is_tty() {
    eprintln!("⚠️  Stale build detected. Rebuild xtask now? (y/n)");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok();
    if input.trim() == "y" {
        rebuild_xtask_interactive()?;
    }
}
```

**Risk**: User interruption, non-deterministic test execution

**Timeline**: Post-MVP (v0.3+)

---

### Enhancement 2: Unified Multi-Backend Warning

**Proposal**: Single warning for dual-backend stale builds

**Example**:
```
⚠️  STALE BUILD: Multiple backends found at runtime but not at build time:
  • bitnet.cpp (found at /home/user/.cache/bitnet_cpp/build)
  • llama.cpp (found at /home/user/.cache/llama_cpp/build)
Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all
```

**Timeline**: Post-MVP (v0.3+)

---

### Enhancement 3: JSON Warning Format for CI Tooling

**Proposal**: Machine-readable warning output for CI/CD integration

**Example**:
```json
{
  "type": "stale_build_warning",
  "backend": "bitnet",
  "matched_path": "/home/user/.cache/bitnet_cpp/build",
  "rebuild_command": "cargo clean -p crossval && cargo build -p xtask --features crossval-all",
  "ci_mode": false
}
```

**Timeline**: Post-MVP (v0.3+)

---

**End of Specification**
