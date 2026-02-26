# BitNet-rs Preflight & Diagnostics Infrastructure Report

## Executive Summary

The dual-backend cross-validation system has **well-structured preflight and diagnostics infrastructure** that is:
- **Partially integrated**: Preflight checks exist and are called in `crossval-per-token`
- **Diagnostics flags documented but unimplemented**: `--dump-ids` and `--dump-cpp-ids` are documented in CLAUDE.md but not wired to the CLI
- **Backend detection working**: Auto-detection from model paths is functional
- **Test scaffolding complete**: Tests properly marked `#[ignore]` with clear skip reasons
- **Build-time detection solid**: `crossval/build.rs` correctly detects C++ libraries and emits environment variables

## File Locations & Key Line Numbers

### 1. Preflight Functions

#### Primary Implementation: `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/preflight.rs`

- **`preflight_backend_libs()` function** (lines 34-79)
  - Takes `backend: CppBackend` and `verbose: bool`
  - Checks compile-time env vars `CROSSVAL_HAS_BITNET` and `CROSSVAL_HAS_LLAMA`
  - Returns `Result<()>` with actionable error messages
  - **Currently USED**: Called in `crossval-per-token` at line 3002 of xtask/src/main.rs
  - **Current status**: ✅ Working properly

- **`print_backend_status()` function** (lines 85-121)
  - Prints human-readable backend availability
  - Marked `#[allow(dead_code)]` - reserved for future preflight command
  - **Currently UNUSED**: Not called anywhere

#### Backend Method: `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/backend.rs`

- **`CppBackend::required_libs()` method** (lines 90-95)
  - Returns `&[&'static str]` with library names to check
  - BitNet: `["libbitnet"]`
  - LLaMA: `["libllama", "libggml"]`
  - **Currently USED**: Called by `preflight_backend_libs()` for error messages (line 69)
  - **Current status**: ✅ Working properly

- **`CppBackend::setup_command()` method** (lines 100-105)
  - Returns command users should run to set up C++ reference
  - **Currently USED**: Called in error messages (line 67)
  - **Current status**: ✅ Working properly

- **`CppBackend::from_model_path()` method** (lines 50-61)
  - Auto-detects backend from model path heuristics
  - Checks for "bitnet" string → BitNet, "llama" → LLaMA
  - Default fallback: LLaMA (safer)
  - **Currently USED**: Called in `crossval-per-token` at line 2991
  - **Current status**: ✅ Working properly

### 2. Diagnostic Infrastructure

#### Build-Time Detection: `/home/steven/code/Rust/BitNet-rs/crossval/build.rs`

- **Library detection** (lines 86-124)
  - Scans multiple possible library directories
  - Searches for `libbitnet*`, `libllama*`, `libggml*` files
  - **Generated variables** (lines 127-128):
    - `CROSSVAL_HAS_BITNET=true|false`
    - `CROSSVAL_HAS_LLAMA=true|false`
  - **Current status**: ✅ Comprehensive multi-tier search paths

- **Diagnostic messages** (lines 134-150)
  - Emits cargo warnings about what was found
  - Guides users to set BITNET_CPP_DIR if nothing found
  - **Current status**: ✅ Good user feedback

#### CLI Flags in `crossval-per-token` Command: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs`

**Defined command struct** (lines 435-481):
```rust
#[command(name = "crossval-per-token")]
CrossvalPerToken {
    #[arg(long)]
    model: PathBuf,                          // ✅ Used
    
    #[arg(long)]
    tokenizer: PathBuf,                      // ✅ Used
    
    #[arg(long)]
    prompt: String,                          // ✅ Used
    
    #[arg(long, default_value_t = 4)]
    max_tokens: usize,                       // Prefixed with _ - Not used yet
    
    #[arg(long, default_value_t = 0.999)]
    cos_tol: f32,                           // ✅ Used
    
    #[arg(long, default_value = "text")]
    format: String,                         // ✅ Used
    
    #[arg(long, default_value = "auto")]
    prompt_template: PromptTemplateArg,     // ✅ Used
    
    #[arg(long)]
    system_prompt: Option<String>,          // Prefixed with _ - Not used yet
    
    #[arg(long, value_enum)]
    cpp_backend: Option<CppBackend>,        // ✅ Used
    
    #[arg(long)]
    verbose: bool,                          // ✅ Used (but limited)
}
```

**Missing flags documented in CLAUDE.md (lines 622, 663):**
- `--dump-ids` - "Dump Rust token IDs to stderr for debugging" ❌ **NOT DEFINED**
- `--dump-cpp-ids` - "Dump C++ token IDs to stderr for debugging" ❌ **NOT DEFINED**

#### Handler Function: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` line 2975

Function signature shows what's actually used:
```rust
fn crossval_per_token_cmd(
    model_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    _max_tokens: usize,              // NOT USED (prefixed with _)
    cos_tol: f32,
    format: &str,
    prompt_template: PromptTemplateArg,
    _system_prompt: Option<&str>,    // NOT USED (prefixed with _)
    cpp_backend: Option<CppBackend>,
    verbose: bool,
) -> Result<()>
```

#### Diagnostic Output in Handler: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs`

**What's currently printed:**
- Line 2993-2999: Backend selection (name + explicit/auto-detected)
- Line 3016-3018: Template factsheet (only in verbose mode)
- Line 3020-3028: Model, prompt, tolerance (always)
- Line 3095: C++ token count (always)
- Line 3119: Token parity result (always)

**What's missing:**
- Token ID dumps (no `--dump-ids` flag)
- Token ID dumps for C++ (no `--dump-cpp-ids` flag)
- Library search path details (not shown even in verbose)

### 3. Error Handling & Messages

#### Token Parity Errors: `/home/steven/code/Rust/BitNet-rs/crossval/src/token_parity.rs`

**`TokenParityError` struct** (lines 31-43):
- Contains both token sequences, first diff index, prompt, and backend
- Backend field enables context-aware error messages

**`format_token_mismatch_error()` function** (lines 158-258):
- **Header**: Includes backend name (line 167)
- **Body**: Shows first 64 tokens of each sequence (lines 175-188)
- **Diff position**: Highlighted with actual token values (lines 191-206)
- **Backend-specific troubleshooting** (lines 216-239):
  - BitNet: Suggests trying llama backend, mentions model compatibility
  - LLaMA: Suggests trying bitnet backend, mentions tokenizer compatibility
- **Common fixes** (lines 242-245):
  - Template override, BOS flag, GGUF metadata checking
- **Copy-paste example** (lines 248-255):
  - Includes actual prompt, model path, backend
- **Current status**: ✅ Excellent error messages

**Error exit code** (line 3117 of xtask):
- Exits with code 2 on token mismatch
- Matches documented specification

### 4. Test Infrastructure

#### Dual Backend Integration Tests: `/home/steven/code/Rust/BitNet-rs/crossval/tests/dual_backend_integration.rs`

**Test categories:**

1. **Backend Auto-Detection Tests** (lines 29-120) - ✅ Always Run
   - `test_backend_autodetect_bitnet()` 
   - `test_backend_autodetect_llama()`
   - `test_backend_from_name()`
   - No external dependencies needed

2. **Preflight Validation Tests** (lines 122-209) - ⚠️ Ignored (Requires Libs)
   - `test_preflight_bitnet_available()` - Requires `CROSSVAL_HAS_BITNET=true`
   - `test_preflight_llama_available()` - Requires `CROSSVAL_HAS_LLAMA=true`
   - `test_preflight_env_var_reporting()` - Checks env vars are valid
   - **Skip annotation**: `#[ignore = "Requires ... libraries"]`

3. **Lane A - BitNet Cross-Validation** (lines 213-311) - ⚠️ Ignored (Requires Model)
   - `test_lane_a_bitnet_crossval()` 
   - Checks: `CROSSVAL_HAS_BITNET=true`, model file exists, tokenizer exists
   - **Skip annotation**: `#[ignore = "Requires BitNet libs + GGUF model"]`
   - **Feature gated**: `#[cfg(all(feature = "ffi", have_cpp))]`
   - **Status**: Test scaffolding with TODO comments (lines 287-310)

4. **Lane B - LLaMA Cross-Validation** (lines 313-412) - ⚠️ Ignored (Requires Model)
   - `test_lane_b_llama_crossval()`
   - Similar structure to Lane A
   - **Status**: Test scaffolding with TODO comments

5. **Error Handling Tests** (lines 414-545) - ✅ Always Run
   - `test_backend_error_when_unavailable()`
   - `test_cpp_bindings_availability_reporting()`
   - `test_parity_error_includes_backend()`
   - `test_backend_specific_troubleshooting()`
   - `test_token_parity_success_both_backends()`
   - No external dependencies

6. **Environment Variable Tests** (lines 575-615) - ⚠️ Serial (Mutate Env)
   - Uses `#[serial(bitnet_env)]` for safe parallel execution
   - `test_backend_env_override()` - TODO: Not implemented
   - `test_debug_logging_env_var()` - TODO: Ignored with reason

7. **Documentation Tests** (lines 617-676) - ✅ Always Run
   - `test_backend_display_names()`
   - `test_build_diagnostics()`

#### Smoke Tests: `/home/steven/code/Rust/BitNet-rs/crossval/tests/smoke.rs`

- `smoke_env_preflight()` - Validates CROSSVAL_GGUF and LD_LIBRARY_PATH set
- `smoke_first_token_logits_parity()` - Minimal parity check (ignored, requires model)
- `smoke_vocab_lock_validation()` - Checks model lock file (ignored)

#### xtask Preflight Tests: `/home/steven/code/Rust/BitNet-rs/xtask/tests/preflight.rs`

**Issue #439 Acceptance Criteria Tests:**
- `ac5_preflight_detects_no_gpu_with_fake_none()` - Tests GPU fake modes
- `ac5_preflight_detects_gpu_with_fake_cuda()`
- `ac5_preflight_real_gpu_detection()`
- `ac5_preflight_invalid_fake_value_fallback()` - Tests error handling
- `ac5_preflight_reports_compile_status()` - Feature distinction
- `ac5_preflight_exit_code_success()` - Exit code validation
- **Status**: These test GPU preflight, not C++ backend preflight

### 5. CLI Integration Points

#### Preflight Call in `crossval-per-token`: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs`

**Location**: Line 3002
```rust
// Preflight validation - verify required libraries are available
crate::crossval::preflight_backend_libs(backend, verbose)?;
```

**Execution flow**:
1. Line 2991: Determine backend (explicit or auto-detected)
2. Lines 2993-2999: Print backend selection (if verbose)
3. Line 3002: **Call preflight check** ← Entry point for wiring
4. Lines 3004-3018: Template setup
5. Lines 3031-3039: Rust tokenization
6. Lines 3042-3046: Check if C++ FFI is available (second check!)
7. Lines 3048-3094: C++ tokenization
8. Lines 3101-3119: Token parity validation (FAIL-FAST)
9. Lines 3122-3124: Rust logits evaluation
10. Lines 3131-3166: C++ logits evaluation
11. Lines 3176-3239: Comparison and output

**Current issue**: Line 3042 calls `bitnet_sys::is_available()` which is a SECOND availability check after preflight. This should be unified or at minimum use same information.

## What's Implemented vs Unused

### ✅ Implemented & Working

| Component | Location | Status |
|-----------|----------|--------|
| Backend auto-detection | `backend.rs:50-61` | ✅ Used in crossval-per-token |
| Backend name/description | `backend.rs:72-77, 52-57` | ✅ Used in diagnostics |
| Required libs metadata | `backend.rs:90-95` | ✅ Used in error messages |
| Setup command guidance | `backend.rs:100-105` | ✅ Used in error messages |
| Preflight validation function | `preflight.rs:34-79` | ✅ Called at line 3002 |
| Token parity error formatting | `token_parity.rs:158-258` | ✅ Used in token mismatch |
| Backend-specific troubleshooting | `token_parity.rs:216-239` | ✅ Backend context in errors |
| Build-time lib detection | `crossval/build.rs:86-124` | ✅ Emits env vars |
| Test skip annotations | `dual_backend_integration.rs` | ✅ Clear markers |

### ⚠️ Partially Implemented

| Component | Location | Status |
|-----------|----------|--------|
| Verbose flag handling | `main.rs:2993-2999, 3016-3018` | ⚠️ Minimal output in verbose mode |
| Diagnostic output | `main.rs:2993-3119` | ⚠️ Basic info, no lib paths shown |
| `--dump-ids` flag | CLAUDE.md only | ⚠️ Documented but not wired |
| `--dump-cpp-ids` flag | CLAUDE.md only | ⚠️ Documented but not wired |

### ❌ Unused Infrastructure

| Component | Location | Status |
|-----------|----------|--------|
| `print_backend_status()` | `preflight.rs:85-121` | ❌ Dead code, unused |
| C++ availability check | `main.rs:3042` | ❌ Duplicate of preflight |

### 🔴 Missing Implementations

| Feature | Expected Location | Status |
|---------|-------------------|--------|
| Token ID dumps on --dump-ids | `crossval_per_token_cmd` | ❌ Not implemented |
| Token ID dumps on --dump-cpp-ids | `crossval_per_token_cmd` | ❌ Not implemented |
| Environment override `BITNET_CPP_BACKEND` | `dual_backend_integration.rs:586` | ❌ Test scaffolding only |
| Debug logging via `BITNET_CROSSVAL_VERBOSE` | `dual_backend_integration.rs:606` | ❌ Test scaffolding only |

## Gaps in Integration

### 1. **--dump-ids and --dump-cpp-ids Flags Not Wired**

**Problem**: CLAUDE.md documents these flags (lines 622, 663), but they're not:
- Defined in the `CrossvalPerToken` command struct
- Passed to `crossval_per_token_cmd()` function
- Implemented in the handler

**Required changes**:
1. Add `#[arg(long)] dump_ids: bool` to command struct (line ~480)
2. Add `#[arg(long)] dump_cpp_ids: bool` to command struct (line ~481)
3. Add parameters to `crossval_per_token_cmd()` function signature
4. Pass through match expression (line 960-971)
5. Implement token dump logic in handler:
   - If `dump_ids`: Print Rust token IDs after tokenization (line 3034)
   - If `dump_cpp_ids`: Print C++ token IDs after tokenization (line 3095)

### 2. **Duplicate C++ Availability Checks**

**Current flow**:
- Line 3002: `preflight_backend_libs()` checks compile-time `CROSSVAL_HAS_*`
- Line 3042: `bitnet_sys::is_available()` checks... something different?

**Problem**: Two different checks for the same thing
- Preflight is backend-aware (BitNet vs LLaMA)
- `bitnet_sys::is_available()` is generic

**Should consolidate**: Use preflight result for both

### 3. **Limited Verbose Output**

**Current verbose output**:
- Backend selection and method (lines 2993-2999)
- Template configuration (lines 3016-3018)
- Model/tokenizer paths (lines 3024-3025)

**Missing from verbose**:
- Library search paths that were checked
- Libraries found vs missing
- Environment variables used
- Backend setup command (if not available)
- Feature flags detected at compile time

### 4. **print_backend_status() Marked Dead Code**

**Location**: `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/preflight.rs:85`

**Current annotation**: `#[allow(dead_code)] // Reserved for future preflight command`

**Status**: Suggests a `xtask preflight` command that doesn't exist yet

**Should implement**: 
- A top-level `xtask preflight` command (or `xtask preflight --backend` subcommand)
- Would call `print_backend_status()` to show availability status
- Useful for CI/CD diagnostics before running expensive cross-validation

### 5. **Test Scaffolding for Environment Variables**

**Location**: `/home/steven/code/Rust/BitNet-rs/crossval/tests/dual_backend_integration.rs:586-615`

Tests for environment overrides that don't exist yet:
- `BITNET_CPP_BACKEND` - Force backend selection
- `BITNET_CROSSVAL_VERBOSE` - Enable debug logging

These are good TDD placeholders but not wired up.

## CI Configuration

### Current Cross-Validation Jobs

**File**: `/home/steven/code/Rust/BitNet-rs/.github/workflows/crossval-fast.yml`

- Runs on schedule (nightly) or manual trigger
- Tests only run if 'crossval' label on PR
- Uses cached C++ binaries
- **Preflight integration**: Not visible in workflow (likely runs as part of tests)

## Recommendations for Wiring

### Phase 1: Wire Documented Flags (Quick Win)

1. **Add `--dump-ids` and `--dump-cpp-ids` flags**
   - Files to modify:
     - `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` (lines 435-481, 2975-2986, 960-971)
   - Implementation: Print token IDs to stderr when flags set
   - Effort: ~30 minutes
   - Impact: Enables documented debugging workflow

### Phase 2: Enhance Verbose Output

1. **Expand verbose diagnostics**
   - Show library search paths checked
   - Show environment variables used
   - Show feature flags detected
   - Files: `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/preflight.rs` (enhance `preflight_backend_libs()`)
   - Effort: ~20 minutes

2. **Consolidate availability checks**
   - Remove redundant `bitnet_sys::is_available()` call
   - Use preflight result instead
   - Files: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` (lines 3042-3046)
   - Effort: ~15 minutes

### Phase 3: Implement Future Features

1. **Create `xtask preflight` command**
   - Use `print_backend_status()` function
   - Useful for CI diagnostics
   - Files:
     - `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` (add to `Cmd` enum)
     - `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/preflight.rs` (already has logic)
   - Effort: ~45 minutes

2. **Implement environment variable overrides**
   - `BITNET_CPP_BACKEND` - Force backend selection
   - `BITNET_CROSSVAL_VERBOSE` - Enable extra logging
   - Files: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs`
   - Effort: ~30 minutes

## Summary

The infrastructure is **solid but incomplete**:
- **Foundation**: Build-time detection, error handling, and test structure are excellent
- **Integration**: Preflight checks are called, but diagnostic flags aren't wired
- **Documentation**: Comprehensive in CLAUDE.md but implementation lags
- **Testing**: Proper ignore markers and scaffolding, ready for feature implementation

The system is ready for gradual implementation without needing major refactoring. The 
priority should be wiring the documented `--dump-ids` flags, as they're already in 
CLAUDE.md and likely expected by users.
