# Dual-Backend Cross-Validation - Implementation Roadmap

**Status:** Partially Implemented — See CLAUDE.md for current state
**Based On:**
- Comprehensive exploration (10 reports, 150KB documentation)
- ChatGPT detailed plan (Phases A-E)
- dual-backend-crossval-spec.md (v1.0)

**Last Updated:** 2026-02-28

---

## Executive Summary

This document provides a file-by-file implementation breakdown for dual-backend cross-validation support in bitnet-rs. All estimates are based on comprehensive codebase exploration and mapped integration points.

### Key Findings from Exploration

✅ **Current Flow Bug Identified** (xtask/src/main.rs:2933-2963)
- Rust logits evaluated BEFORE token parity check
- Wastes 20-30 seconds on token mismatches
- **Fix:** Move token parity to line 2933 (before logits evaluation)

✅ **FFI Infrastructure Ready** (90% complete)
- llama.cpp wrapper functional (wrapper.rs lines 145-186, 269-293)
- Per-position logits infrastructure exists (get_logits_ith, get_all_logits)
- BitNet.cpp C API available (7 functions confirmed in exploration report)

✅ **Template System Complete** (but not integrated into crossval-per-token)
- Template types defined: Raw, Instruct, Llama3Chat
- Auto-detection from GGUF metadata working
- **Gap:** crossval-per-token doesn't use templates (hardcoded raw tokenization)

✅ **Test Infrastructure Mature**
- EnvGuard pattern for environment isolation
- Serial test patterns (#[serial(bitnet_env)])
- 12/12 unit tests passing in token_parity.rs

❌ **Critical Gaps Identified** (as of Oct 2025 — see status notes below)
1. No --cpp-backend flag (backend selection missing)
2. No --prompt-template flag in crossval-per-token
3. No bitnet.cpp FFI shim (need to create bitnet_cpp_wrapper.c)
4. Library discovery not backend-aware (build.rs needs updates)
5. Per-position logits not exposed in public API

---

## Implementation Status (Feb 2026)

This document was written in Oct 2025 as a pre-implementation plan for dual-backend
cross-validation. As of Feb 2026, most of the planned work has shipped:

**Phase A (Shipped):** CLI flow order bug fixed; `--dump-ids`, `--dump-cpp-ids`, `--verbose` flags implemented
**Phase B (Shipped):** `--prompt-template` flag integrated; template applied before tokenization; `--system-prompt` supported
**Phase C (Shipped):** `CppBackend` enum implemented; `--cpp-backend` flag; auto-detection from model path; preflight validation
**Phase D (Shipped):** `bitnet_cpp_wrapper.c` created; `crossval/build.rs` updated; Rust bindings and backend dispatcher implemented
**Phase E (Shipped):** Integration tests added; documentation updated

The 19-27 hour estimate reflected Oct 2025 planning. Actual implementation delivered
all phases. Remaining gaps are noted in `docs/reference/implementation-targets.md`.

See `CLAUDE.md` -> Cross-Validation CLI Reference for current command usage.

---

## Implementation Phases

### Phase A: CLI & Flow Fixes (Priority 1 - Quick Wins)
**Estimated Effort:** 3-4 hours
**Dependencies:** None
**Risk:** Low

### Phase B: Template Integration (Priority 1)
**Estimated Effort:** 2-3 hours
**Dependencies:** Phase A
**Risk:** Low

### Phase C: Backend Selection Infrastructure (Priority 2)
**Estimated Effort:** 4-6 hours
**Dependencies:** None (can parallelize with A/B)
**Risk:** Low

### Phase D: BitNet FFI Shim (Priority 2)
**Estimated Effort:** 6-8 hours
**Dependencies:** Phase C
**Risk:** Medium (depends on bitnet.cpp API stability)

### Phase E: Testing & Documentation (Priority 3)
**Estimated Effort:** 4-6 hours
**Dependencies:** All previous phases
**Risk:** Low

**Total Estimated Effort:** 19-27 hours (2.5-3.5 developer-days)

---

## Detailed Task Breakdown

## Phase A: CLI & Flow Fixes

### Task A1: Fix crossval-per-token flow order bug
**File:** `xtask/src/main.rs`
**Lines:** 2933-2972
**Estimated LOC:** 40 lines changed
**Effort:** 30 minutes
**Risk:** Low

**Current Flow (WRONG):**
```rust
// Line 2933: Rust logits evaluated first (20-30 seconds)
let rust_logits = eval_rust_logits_all_positions(&engine, &model, &rust_tokens)?;

// Line 2963: Token parity checked AFTER expensive work
validate_token_parity(&rust_tokens, &cpp_tokens, &args.prompt)?;
```

**Fixed Flow:**
```rust
// Line 2933: Token parity FIRST (fail-fast)
validate_token_parity(&rust_tokens, &cpp_tokens, &args.prompt)?;

// Line 2945: Logits evaluation ONLY if tokens match
let rust_logits = eval_rust_logits_all_positions(&engine, &model, &rust_tokens)?;
```

**Acceptance:** AC-F1 (token parity pre-gate before logits)

**Status: RESOLVED** — Flow order fixed; token parity pre-gate is in place.

---

### Task A2: Add --dump-ids and --dump-cpp-ids flags
**File:** `xtask/src/main.rs`
**Lines:** 389-430 (CrossvalPerToken struct)
**Estimated LOC:** +10 lines
**Effort:** 20 minutes
**Risk:** Low

**Implementation:**
```rust
#[derive(Parser)]
struct CrossvalPerToken {
    // ... existing fields ...

    /// Dump Rust token IDs to stderr before comparison
    #[arg(long)]
    dump_ids: bool,

    /// Dump C++ backend token IDs to stderr before comparison
    #[arg(long)]
    dump_cpp_ids: bool,
}
```

**Usage in flow:**
```rust
// After tokenization
if args.dump_ids {
    eprintln!("Rust tokens: {:?}", rust_tokens);
}
if args.dump_cpp_ids {
    eprintln!("C++ tokens: {:?}", cpp_tokens);
}
```

**Acceptance:** AC7 (diagnostic flags)

**Status: RESOLVED** — `--dump-ids` and `--dump-cpp-ids` flags implemented.

---

### Task A3: Add --verbose flag
**File:** `xtask/src/main.rs`
**Lines:** 389-430
**Estimated LOC:** +5 lines
**Effort:** 15 minutes
**Risk:** Low

**Implementation:**
```rust
#[derive(Parser)]
struct CrossvalPerToken {
    // ... existing fields ...

    /// Verbose output: backend selection, library discovery, model info
    #[arg(long)]
    verbose: bool,
}
```

**Acceptance:** AC7 (diagnostic flags)

**Status: RESOLVED** — `--verbose` flag implemented.

---

## Phase B: Template Integration

### Task B1: Add --prompt-template flag to crossval-per-token
**File:** `xtask/src/main.rs`
**Lines:** 389-430
**Estimated LOC:** +15 lines
**Effort:** 30 minutes
**Risk:** Low

**Implementation:**
```rust
use bitnet_inference::prompt_template::TemplateType;

#[derive(Parser)]
struct CrossvalPerToken {
    // ... existing fields ...

    /// Prompt template type (auto-detects from GGUF if not specified)
    #[arg(long, value_enum, default_value = "auto")]
    prompt_template: TemplateTypeArg,

    /// System prompt (for chat templates)
    #[arg(long)]
    system_prompt: Option<String>,
}

#[derive(Clone, Copy, ValueEnum)]
enum TemplateTypeArg {
    Auto,
    Raw,
    Instruct,
    Llama3Chat,
}

impl From<TemplateTypeArg> for TemplateType {
    fn from(arg: TemplateTypeArg) -> Self {
        match arg {
            TemplateTypeArg::Auto => {
                // Auto-detection logic (from GGUF metadata or tokenizer)
                TemplateType::Raw  // Fallback
            }
            TemplateTypeArg::Raw => TemplateType::Raw,
            TemplateTypeArg::Instruct => TemplateType::Instruct,
            TemplateTypeArg::Llama3Chat => TemplateType::Llama3Chat,
        }
    }
}
```

**Acceptance:** FR2 (template support), ChatGPT Phase B

**Status: RESOLVED** — `--prompt-template` flag integrated with auto-detection.

---

### Task B2: Apply template to prompt before tokenization
**File:** `xtask/src/main.rs`
**Lines:** ~2900 (tokenization call site)
**Estimated LOC:** +20 lines
**Effort:** 45 minutes
**Risk:** Low

**Current Code:**
```rust
// Hardcoded: no template, no BOS control
let rust_tokens = tokenizer.encode(prompt, false, false)?;
```

**Fixed Code:**
```rust
// 1. Resolve template type
let template = match args.prompt_template {
    TemplateTypeArg::Auto => {
        // Auto-detect from GGUF metadata or tokenizer
        detect_template_from_model(&args.model, &tokenizer)?
    }
    arg => arg.into(),
};

// 2. Apply template to prompt
let formatted_prompt = template.apply(&args.prompt, args.system_prompt.as_deref());

// 3. Get BOS/special token policy from template
let add_bos = template.should_add_bos();
let parse_special = template.parse_special();

// 4. Tokenize with template-aware flags
let rust_tokens = tokenizer.encode(&formatted_prompt, add_bos, parse_special)?;
```

**References:**
- Template application: `crates/bitnet-inference/src/prompt_template.rs:68-150`
- Auto-detection: `crates/bitnet-inference/src/prompt_template.rs:200-320`

**Acceptance:** FR2 (template support), ChatGPT Phase B

**Status: RESOLVED** — Template applied before tokenization; `--system-prompt` supported.

---

## Phase C: Backend Selection Infrastructure

### Task C1: Add CppBackend enum
**File:** `xtask/src/main.rs` (or new `xtask/src/crossval/backend.rs`)
**Estimated LOC:** +80 lines (new file)
**Effort:** 1 hour
**Risk:** Low

**Implementation:**
```rust
/// C++ backend selection for cross-validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CppBackend {
    /// Use bitnet.cpp for tokenization and evaluation
    BitNet,
    /// Use llama.cpp for tokenization and evaluation
    Llama,
}

impl CppBackend {
    /// Auto-detect backend from model path
    pub fn from_model_path(path: &Path) -> Self {
        let path_str = path.to_string_lossy().to_lowercase();
        if path_str.contains("bitnet") || path_str.contains("microsoft/bitnet") {
            Self::BitNet
        } else {
            Self::Llama  // Conservative default
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::BitNet => "bitnet.cpp",
            Self::Llama => "llama.cpp",
        }
    }

    pub fn required_libs(&self) -> &[&'static str] {
        match self {
            Self::BitNet => &["libbitnet"],
            Self::Llama => &["libllama", "libggml"],
        }
    }
}
```

**Acceptance:** FR1 (backend selection), ChatGPT Phase A

**Status: RESOLVED** — `CppBackend` enum implemented with auto-detection heuristics.

---

### Task C2: Add --cpp-backend CLI flag
**File:** `xtask/src/main.rs`
**Lines:** 389-430
**Estimated LOC:** +10 lines
**Effort:** 20 minutes
**Risk:** Low

**Implementation:**
```rust
#[derive(Parser)]
struct CrossvalPerToken {
    // ... existing fields ...

    /// C++ backend selection (auto-detects from model path if not specified)
    #[arg(long, value_enum)]
    cpp_backend: Option<CppBackend>,
}
```

**Acceptance:** FR1 (backend selection)

**Status: RESOLVED** — `--cpp-backend` flag implemented; auto-detection from model path works.

---

### Task C3: Add backend preflight validation
**File:** `xtask/src/main.rs` (or new `xtask/src/crossval/preflight.rs`)
**Estimated LOC:** +60 lines
**Effort:** 1 hour
**Risk:** Low

**Implementation:**
```rust
/// Verify required libraries are available for selected backend
fn preflight_backend_libs(backend: CppBackend, verbose: bool) -> Result<()> {
    // 1. Check compile-time detection
    let has_libs = match backend {
        CppBackend::BitNet => {
            // Emit from build.rs: CROSSVAL_HAS_BITNET=true/false
            option_env!("CROSSVAL_HAS_BITNET")
                .map(|v| v == "true")
                .unwrap_or(false)
        }
        CppBackend::Llama => {
            option_env!("CROSSVAL_HAS_LLAMA")
                .map(|v| v == "true")
                .unwrap_or(false)
        }
    };

    if !has_libs {
        let setup_cmd = match backend {
            CppBackend::BitNet => {
                "eval \"$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)\""
            }
            CppBackend::Llama => {
                "eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\""
            }
        };

        bail!(
            "Backend '{}' selected but required libraries not found.\n\
             \n\
             Setup instructions:\n\
             1. Install C++ reference implementation:\n\
                {}\n\
             \n\
             2. Verify libraries are loaded:\n\
                cargo run -p xtask -- preflight --backend {}\n\
             \n\
             Required libraries: {:?}",
            backend.name(),
            setup_cmd,
            backend.name(),
            backend.required_libs()
        );
    }

    if verbose {
        println!("✓ Backend '{}' libraries found", backend.name());
    }

    Ok(())
}
```

**Acceptance:** FR2 (library preflight), ChatGPT Phase A

**Status: RESOLVED** — Preflight validation implemented; `cargo run -p xtask --features crossval-all -- preflight --backend bitnet` works.

---

### Task C4: Integrate backend selection into crossval flow
**File:** `xtask/src/main.rs`
**Lines:** ~2880 (crossval_per_token_cmd entry point)
**Estimated LOC:** +15 lines
**Effort:** 30 minutes
**Risk:** Low

**Implementation:**
```rust
fn crossval_per_token_cmd(args: CrossvalPerToken) -> Result<()> {
    // 1. Backend selection (auto-detect or explicit)
    let backend = args.cpp_backend
        .unwrap_or_else(|| CppBackend::from_model_path(&args.model));

    if args.verbose {
        println!("Selected backend: {} ({})",
                 backend.name(),
                 if args.cpp_backend.is_some() { "explicit" } else { "auto-detected" });
    }

    // 2. Library preflight
    preflight_backend_libs(backend, args.verbose)?;

    // 3. Rest of flow (tokenization, parity, logits)
    // ... (backend passed to tokenize_cpp and eval_cpp functions)
}
```

**Acceptance:** FR1 (backend selection), FR2 (preflight)

**Status: RESOLVED** — Backend selection integrated into crossval flow.

---

## Phase D: BitNet FFI Shim

### Task D1: Create bitnet_cpp_wrapper.c
**File:** `crossval/src/bitnet_cpp_wrapper.c` (new file)
**Estimated LOC:** +250 lines
**Effort:** 3-4 hours
**Risk:** Medium (depends on bitnet.cpp API)

**Implementation:** See spec Appendix A for full signatures

**Key Functions:**
```c
extern "C" int bitnet_tokenize(
    const char* model_path,
    const char* prompt,
    int add_bos,
    int32_t* out_tokens,
    int32_t* out_n_tokens,
    char* err,
    int32_t err_len
);

extern "C" int bitnet_eval_with_tokens(
    const char* model_path,
    const int32_t* tokens,
    int32_t n_tokens,
    int32_t n_ctx,
    float* logits,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
);
```

**Dependencies:**
- bitnet.cpp C++ API headers
- Build system integration (Task D2)

**Acceptance:** FR3 (bitnet.cpp tokenizer), FR4 (bitnet.cpp eval), ChatGPT Phase C

**Status: RESOLVED** — `bitnet_cpp_wrapper.c` created; Rust bindings implemented.

---

### Task D2: Update crossval/build.rs for backend-aware discovery
**File:** `crossval/build.rs`
**Lines:** 1-100 (entire file)
**Estimated LOC:** +100 lines
**Effort:** 2 hours
**Risk:** Low

**Implementation:** See spec Section "Library Discovery (Build System)" for full code

**Key Changes:**
1. Search for libbitnet* in addition to libllama*/libggml*
2. Priority-based search (BITNET_CROSSVAL_LIBDIR -> standard paths)
3. Emit build metadata: `CROSSVAL_HAS_BITNET`, `CROSSVAL_HAS_LLAMA`
4. Print diagnostic warnings about which libs were found

**References:**
- Exploration report: `docs/reference/BUILD_RS_LIBRARY_DISCOVERY_AND_LINKING.md`
- Current build.rs: `crossval/build.rs:1-100`

**Acceptance:** NFR2 (build system compatibility), ChatGPT Phase C

**Status: RESOLVED** — `crossval/build.rs` updated with backend-aware library discovery.

---

### Task D3: Add Rust bindings for bitnet_cpp_wrapper
**File:** `crossval/src/cpp_bindings.rs` (extend existing)
**Estimated LOC:** +80 lines
**Effort:** 1.5 hours
**Risk:** Low

**Implementation:**
```rust
#[cfg(feature = "ffi")]
extern "C" {
    fn bitnet_tokenize(
        model_path: *const c_char,
        prompt: *const c_char,
        add_bos: c_int,
        out_tokens: *mut i32,
        out_n_tokens: *mut i32,
        err: *mut c_char,
        err_len: i32,
    ) -> c_int;

    fn bitnet_eval_with_tokens(
        model_path: *const c_char,
        tokens: *const i32,
        n_tokens: i32,
        n_ctx: i32,
        logits: *mut f32,
        out_rows: *mut i32,
        out_cols: *mut i32,
        err: *mut c_char,
        err_len: i32,
    ) -> c_int;
}

/// Safe wrapper for bitnet_tokenize
pub fn tokenize_bitnet(
    model_path: &Path,
    prompt: &str,
    add_bos: bool,
) -> Result<Vec<i32>> {
    // CString conversion, error handling, etc.
    // Pattern matches existing llama.cpp wrapper style
}

/// Safe wrapper for bitnet_eval_with_tokens
pub fn eval_bitnet(
    model_path: &Path,
    tokens: &[i32],
    n_ctx: usize,
) -> Result<Vec<Vec<f32>>> {
    // Allocation, FFI call, error handling
}
```

**References:**
- FFI patterns: `crates/bitnet-sys/src/wrapper.rs:145-186`
- Error handling: `crates/bitnet-ffi/src/error.rs:1-270`

**Acceptance:** FR3, FR4 (bitnet.cpp integration)

**Status: RESOLVED** — Rust bindings and safe wrappers implemented.

---

### Task D4: Create backend dispatcher in xtask
**File:** `xtask/src/main.rs` (or new `xtask/src/crossval/dispatch.rs`)
**Estimated LOC:** +60 lines
**Effort:** 1 hour
**Risk:** Low

**Implementation:**
```rust
/// Tokenize using selected backend
fn tokenize_cpp(
    backend: CppBackend,
    model_path: &Path,
    prompt: &str,
    add_bos: bool,
) -> Result<Vec<i32>> {
    match backend {
        CppBackend::BitNet => {
            crossval::tokenize_bitnet(model_path, prompt, add_bos)
        }
        CppBackend::Llama => {
            // Use existing wrapper.rs Context::tokenize
            let ctx = bitnet_sys::Context::new(model_path)?;
            ctx.tokenize(prompt, add_bos)
        }
    }
}

/// Evaluate using selected backend
fn eval_cpp(
    backend: CppBackend,
    model_path: &Path,
    tokens: &[i32],
) -> Result<Vec<Vec<f32>>> {
    match backend {
        CppBackend::BitNet => {
            crossval::eval_bitnet(model_path, tokens, 2048)
        }
        CppBackend::Llama => {
            let mut ctx = bitnet_sys::Context::new(model_path)?;
            ctx.eval_all_logits(tokens)
        }
    }
}
```

**Acceptance:** FR1 (backend selection), FR3/FR4 (backend-specific calls)

**Status: RESOLVED** — Backend dispatcher implemented and routes correctly.

---

## Phase E: Testing & Documentation

### Task E1: Update token_parity.rs with backend context
**File:** `crossval/src/token_parity.rs`
**Lines:** 1-150
**Estimated LOC:** +30 lines
**Effort:** 45 minutes
**Risk:** Low

**Current Signature:**
```rust
pub fn validate_token_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[i32],
    prompt: &str,
) -> Result<()>
```

**Updated Signature:**
```rust
pub fn validate_token_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[i32],
    prompt: &str,
    backend: CppBackend,  // NEW
) -> Result<()>
```

**Error Message Enhancement:**
```rust
fn format_token_mismatch_error(error: &TokenParityError) -> String {
    // ... existing formatting ...

    // Backend-specific suggestions
    writeln!(output, "\n{}:", style("Backend-specific fixes").green().bold()).unwrap();
    match error.backend {
        CppBackend::BitNet => {
            writeln!(output, "  • Verify BitNet model uses BitNet-compatible tokenizer").unwrap();
            writeln!(output, "  • Try --cpp-backend llama if model is LLaMA-based").unwrap();
        }
        CppBackend::Llama => {
            writeln!(output, "  • Verify LLaMA model uses LLaMA-compatible tokenizer").unwrap();
            writeln!(output, "  • Try --cpp-backend bitnet if model is BitNet-based").unwrap();
        }
    }

    output
}
```

**Acceptance:** AC3, AC4 (backend-specific error messages)

**Status: RESOLVED** — Token parity errors include backend context.

---

### Task E2: Add integration tests
**File:** `crossval/tests/dual_backend_integration.rs` (new file)
**Estimated LOC:** +300 lines
**Effort:** 2-3 hours
**Risk:** Low

**Test Coverage:**
```rust
// AC1: Backend auto-detection
#[test]
fn test_backend_autodetect_bitnet() {
    let path = Path::new("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf");
    assert_eq!(CppBackend::from_model_path(path), CppBackend::BitNet);
}

// AC2: Library preflight
#[test]
#[ignore = "requires bitnet.cpp installation"]
fn test_bitnet_backend_preflight() {
    preflight_backend_libs(CppBackend::BitNet, false).expect("bitnet.cpp libs");
}

// AC8: End-to-end Lane A (BitNet vs bitnet.cpp)
#[test]
#[ignore = "requires bitnet.cpp + model"]
#[serial(bitnet_env)]
fn test_lane_a_bitnet_parity() {
    // Full pipeline test
}

// AC9: End-to-end Lane B (bitnet-rs vs llama.cpp)
#[test]
#[ignore = "requires llama.cpp + model"]
#[serial(bitnet_env)]
fn test_lane_b_llama_parity() {
    // Full pipeline test
}
```

**References:**
- Test patterns: `crossval/tests/per_position_logits.rs:1-295`
- EnvGuard: `tests/helpers/env_guard.rs`
- Serial tests: `#[serial(bitnet_env)]` pattern

**Acceptance:** AC8, AC9, AC12 (test tags)

**Status: RESOLVED** — Integration tests added; real-model tests gated with `#[ignore]` and justification strings per workspace conventions.

---

### Task E3: Update documentation
**Files:**
- `docs/howto/cpp-setup.md` (extend with bitnet.cpp setup)
- `docs/explanation/dual-backend-crossval.md` (architecture overview)
- `CLAUDE.md` (add new crossval-per-token flags)

**Estimated LOC:** +400 lines across 3 files
**Effort:** 2 hours
**Risk:** Low

**Content:**
1. **cpp-setup.md**: Add bitnet.cpp installation instructions (parallel to llama.cpp)
2. **dual-backend-crossval.md**: Architecture overview, backend selection, usage examples
3. **CLAUDE.md**: Update crossval-per-token command reference with new flags

**Acceptance:** Documentation completeness

**Status: RESOLVED** — Documentation updated; `CLAUDE.md` Cross-Validation CLI Reference section reflects all shipped flags.

---

## Implementation Order & Dependencies

### Recommended Execution Order (Historical — All Shipped)

```
Day 1 (Morning):
  ✓ Task A1: Fix flow order bug (30 min)
  ✓ Task A2: Add --dump-ids flags (20 min)
  ✓ Task A3: Add --verbose flag (15 min)
  ✓ Task B1: Add --prompt-template flag (30 min)
  ✓ Task B2: Apply template to prompt (45 min)
  Total: 2h 20m

Day 1 (Afternoon):
  ✓ Task C1: Add CppBackend enum (1 hour)
  ✓ Task C2: Add --cpp-backend flag (20 min)
  ✓ Task C3: Add preflight validation (1 hour)
  ✓ Task C4: Integrate backend selection (30 min)
  Total: 2h 50m

Day 2 (Morning):
  ✓ Task D2: Update build.rs (2 hours)
  Total: 2 hours

Day 2 (Afternoon):
  ✓ Task D1: Create bitnet_cpp_wrapper.c (3-4 hours)
  Total: 3-4 hours

Day 3 (Morning):
  ✓ Task D3: Add Rust bindings (1.5 hours)
  ✓ Task D4: Create backend dispatcher (1 hour)
  Total: 2.5 hours

Day 3 (Afternoon):
  ✓ Task E1: Update token_parity.rs (45 min)
  ✓ Task E2: Add integration tests (2-3 hours)
  Total: 2h 45m - 3h 45m

Day 4:
  ✓ Task E3: Update documentation (2 hours)
  ✓ Manual testing and polish (2-3 hours)
  Total: 4-5 hours
```

**Total: 19-27 hours (2.5-3.5 developer-days)**

---

## Risk Mitigation Strategies

### Risk 1: bitnet.cpp API Instability
**Mitigation:**
- Start with minimal FFI (tokenize + eval only)
- Version-pin bitnet.cpp in setup-cpp-auto
- Document API assumptions in wrapper.c comments
- Add API compatibility tests

### Risk 2: Library Discovery Conflicts
**Mitigation:**
- Priority-based search (explicit env var > auto-detect)
- Clear diagnostic messages
- Runtime backend selection decouples build from execution
- Test with single-backend builds

### Risk 3: Template Parity Edge Cases
**Mitigation:**
- Comprehensive test suite (BOS, EOS, special tokens)
- Diagnostic flags (--dump-ids) for manual inspection
- Document known differences per backend
- Add fuzzing for template + tokenization

---

## Validation Checklist

Before marking each phase complete:

**Phase A:**
- [x] Flow order fix verified (token parity before logits)
- [x] --dump-ids prints Rust tokens
- [x] --dump-cpp-ids prints C++ tokens
- [x] --verbose prints backend selection

**Phase B:**
- [x] --prompt-template accepts {raw,instruct,llama3-chat,auto}
- [x] --system-prompt works with chat templates
- [x] Template applied before tokenization
- [x] BOS/special token handling correct per template

**Phase C:**
- [x] CppBackend enum defined (BitNet, Llama)
- [x] --cpp-backend flag works
- [x] Auto-detection from model path works
- [x] Preflight fails gracefully with setup instructions

**Phase D:**
- [x] bitnet_cpp_wrapper.c compiles
- [x] build.rs finds libbitnet*
- [x] Rust bindings safe and tested
- [x] Backend dispatcher routes correctly

**Phase E:**
- [x] Token parity errors include backend context
- [x] Integration tests added (real-model tests gated with #[ignore])
- [x] Documentation complete and accurate
- [x] Manual smoke tests successful

---

## Success Criteria

### Functional Success
- ✅ Lane A (BitNet vs bitnet.cpp) passes end-to-end test
- ✅ Lane B (bitnet-rs vs llama.cpp) passes end-to-end test
- ✅ Auto-detection works for both model types
- ✅ Token parity pre-gate catches mismatches
- ✅ Error messages actionable and backend-specific

### Quality Success
- ✅ All acceptance criteria have tests with AC tags
- ✅ No breaking changes to existing commands
- ✅ Build passes with all library configurations
- ✅ Documentation includes both lanes

### Performance Success
- ✅ Library preflight <100ms
- ✅ Token parity <50ms
- ✅ No logits comparison regression

---

## Post-Implementation Tasks

**Nice-to-Have Enhancements (Backlog):**
- [ ] Extend bitnet_cpp_wrapper with all-positions logits API
- [ ] Add --jinja flag for Jinja template rendering
- [ ] Implement CppSession trait abstraction (unified interface)
- [ ] Add model factsheet command (print specials/EOG/template from GGUF)
- [ ] Add receipt schema v2 with backend metadata
- [ ] CI matrix testing (bitnet-only, llama-only, dual-backend)

---

## File Summary

**Files Created:**
1. `crossval/src/bitnet_cpp_wrapper.c` (~250 LOC)
2. `crossval/tests/dual_backend_integration.rs` (~300 LOC)
3. `docs/explanation/dual-backend-crossval.md` (~200 LOC)

**Files Modified:**
1. `xtask/src/main.rs` (+200 LOC)
2. `crossval/build.rs` (+100 LOC)
3. `crossval/src/cpp_bindings.rs` (+80 LOC)
4. `crossval/src/token_parity.rs` (+30 LOC)
5. `docs/howto/cpp-setup.md` (+100 LOC)
6. `CLAUDE.md` (+100 LOC)

**Total New Code:** ~1,360 LOC
**Total Modified Code:** ~610 LOC
**Total Documentation:** ~400 LOC

---

## Appendix: Exploration Reports Referenced

1. `CROSSVAL_EXPLORATION_REPORT.md` - Complete crossval architecture (1,177 lines)
2. `docs/explanation/xtask-crossval-per-token-implementation-analysis.md` - Flow bug identified
3. `docs/reference/BUILD_RS_LIBRARY_DISCOVERY_AND_LINKING.md` - Library discovery patterns
4. `TOKENIZER_INTEGRATION_REPORT.md` - Template system architecture
5. `CLI_ARGUMENT_PARSING_PATTERNS.md` - Clap patterns and flag examples
6. `docs/reports/BITNET_CPP_C_API_EXPLORATION_REPORT.md` - bitnet.cpp API surface
7. `LOGGING_AND_DIAGNOSTIC_PATTERNS.md` - Error handling and diagnostics
8. `docs/explanation/crossval-test-infrastructure.md` - Test patterns
9. `docs/explanation/backend-detection-and-device-selection-patterns.md` - Backend patterns
10. `PROMPT_TEMPLATE_SYSTEM_REPORT.md` - Template types and auto-detection
