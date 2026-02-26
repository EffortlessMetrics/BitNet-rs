# Dual-Backend Cross-Validation - Continuation Plan

**Thread Handoff Document**
**Date:** 2025-10-25
**Status:** M1 Complete, M2 In Progress (1/4 tasks done)

---

## Executive Summary

We are implementing dual-backend cross-validation support for BitNet-rs, enabling true end-to-end parity testing between Rust and C++ implementations (bitnet.cpp and llama.cpp). The work is organized into 3 milestones (M1, M2, M3) following a detailed implementation roadmap.

**Progress:** ~40% complete (9/19 tasks done)

---

## ✅ What's Been Completed (M1 + partial M2)

### Milestone M1: Close Day-1 Manual Tasks (COMPLETE ✅)

**Task A1: Fail-fast parity pre-gate ✅**
- **File:** `xtask/src/main.rs`
- **What:** Reordered execution flow to check token parity BEFORE expensive logits evaluation
- **Impact:** Saves 20-30 seconds on token mismatch (fails fast in ~50ms)
- **Status:** DONE - Flow now: tokenize → parity check → (only if match) evaluate logits

**Task A2: Diagnostic flags ✅** (completed earlier)
- **Files:** `xtask/src/main.rs`
- **What:** Added `--dump-ids` and `--dump-cpp-ids` flags
- **Status:** DONE - Both flags working, output to stderr

**Task A3: Complete verbose plumbing ✅**
- **File:** `xtask/src/main.rs`
- **What:** Wired `--verbose` to show backend selection, preflight results, template factsheet
- **Status:** DONE - Verbose diagnostics fully functional

**Task B1: Add prompt-template flag ✅** (completed earlier)
- **File:** `xtask/src/main.rs`
- **What:** Added `--prompt-template` and `--system-prompt` CLI flags
- **Status:** DONE - Flags parse correctly

**Task B2: Apply template before tokenization ✅**
- **File:** `xtask/src/main.rs`
- **What:** Integrated template formatting into tokenization flow with proper BOS/special token handling
- **Status:** DONE - Template application working for Raw, Instruct, Llama3Chat

**Task C1-C4: Backend selection infrastructure ✅** (completed earlier)
- **Files:** `xtask/src/crossval/backend.rs`, `xtask/src/crossval/preflight.rs`, `xtask/src/main.rs`
- **What:** CppBackend enum, auto-detection, preflight validation, CLI integration
- **Status:** DONE - Backend selection fully functional

### Milestone M2: Backend Plumbing (IN PROGRESS - 1/4 done)

**Task D2: Build.rs backend-aware discovery ✅**
- **File:** `crossval/build.rs`
- **What:** Multi-tier library search, discovers libbitnet* and libllama*, emits CROSSVAL_HAS_* env vars
- **Status:** DONE - Build script functional, env vars exported

---

## 🚧 What Remains (M2 continued + M3)

### Milestone M2: Backend Plumbing (3 tasks remaining)

**Task D1: Create bitnet_cpp_wrapper.c** (NEXT - HIGH PRIORITY)
- **File to create:** `crossval/src/bitnet_cpp_wrapper.c` (new file, ~250 LOC)
- **Objective:** Minimal C/C++ FFI shim exposing BitNet.cpp tokenization and evaluation
- **Required functions:**
  - `bitnet_tokenize()` - Tokenize using bitnet.cpp's tokenizer
  - `bitnet_eval_with_tokens()` - Evaluate tokens and return logits
- **Dependencies:** Needs bitnet.cpp headers available (optional at build time)
- **Effort:** 3-4 hours
- **Risk:** Medium (depends on bitnet.cpp API stability)

**Task D3: Rust FFI bindings** (depends on D1)
- **File to modify:** `crossval/src/cpp_bindings.rs` (~80 LOC additions)
- **Objective:** Safe Rust wrappers for bitnet_cpp_wrapper.c functions
- **Required functions:**
  - `tokenize_bitnet(model_path, prompt, add_bos) -> Result<Vec<i32>>`
  - `eval_bitnet(model_path, tokens, n_ctx) -> Result<Vec<Vec<f32>>>`
- **Dependencies:** Task D1 complete
- **Effort:** 1.5 hours
- **Risk:** Low

**Task D4: Backend dispatcher** (depends on D3)
- **File to modify:** `xtask/src/main.rs` or new `xtask/src/crossval/dispatch.rs` (~60 LOC)
- **Objective:** Route tokenize/eval calls based on CppBackend enum
- **Implementation:**
  ```rust
  fn tokenize_cpp(backend, model, prompt, add_bos) -> Result<Vec<i32>> {
      match backend {
          CppBackend::BitNet => crossval::tokenize_bitnet(...),
          CppBackend::Llama => /* existing llama.cpp wrapper */,
      }
  }
  ```
- **Dependencies:** Task D3 complete
- **Effort:** 1 hour
- **Risk:** Low

### Milestone M3: Tests & Docs (3 tasks)

**Task E1: Backend-aware token parity messages**
- **File to modify:** `crossval/src/token_parity.rs` (~30 LOC)
- **Objective:** Add backend context to error messages
- **Effort:** 45 minutes
- **Risk:** Low

**Task E2: Integration tests**
- **File to create:** `crossval/tests/dual_backend_integration.rs` (~300 LOC)
- **Objective:** End-to-end tests for both lanes (BitNet vs bitnet.cpp, LLaMA vs llama.cpp)
- **Tests:** Auto-detection, preflight, Lane A, Lane B (use `#[ignore]` for C++-dependent tests)
- **Effort:** 2-3 hours
- **Risk:** Low

**Task E3: Documentation**
- **Files to update:**
  - `docs/howto/cpp-setup.md` - Add bitnet.cpp installation
  - `docs/explanation/dual-backend-crossval.md` - NEW architecture doc
  - `CLAUDE.md` - Update CLI reference with new flags
- **Effort:** 2 hours
- **Risk:** Low

---

## 📋 Immediate Next Steps (Priority Order)

### Step 1: Complete M2 (Backend Plumbing)

**Agent Call 1: Task D1 - Create bitnet_cpp_wrapper.c**
```bash
# Launch impl-creator agent
# Context: Need C wrapper for bitnet.cpp tokenization and evaluation
# File: crossval/src/bitnet_cpp_wrapper.c (new)
# Dependencies: None (can proceed immediately)
# Acceptance: Compiles when bitnet.cpp headers available, guarded with #ifdef
```

**Agent Call 2: Task D3 - Rust FFI bindings**
```bash
# Launch impl-creator agent
# Context: Safe Rust wrappers for D1's C functions
# File: crossval/src/cpp_bindings.rs
# Dependencies: D1 complete
# Acceptance: Safe wrappers with proper error handling, no UB
```

**Agent Call 3: Task D4 - Backend dispatcher**
```bash
# Launch impl-creator agent
# Context: Route tokenize/eval based on CppBackend
# File: xtask/src/main.rs (or new dispatch.rs module)
# Dependencies: D3 complete
# Acceptance: Routing works for both BitNet and LLaMA backends
```

### Step 2: Complete M3 (Tests & Docs)

**Agent Call 4: Task E1 - Backend-aware parity errors**
**Agent Call 5: Task E2 - Integration tests**
**Agent Call 6: Task E3 - Documentation sweep**

---

## 🔑 Key Context for Next Thread

### Current Codebase State

**Working Features:**
- ✅ `--cpp-backend {bitnet|llama}` - Backend selection with auto-detection
- ✅ `--prompt-template {raw|instruct|llama3-chat|auto}` - Template support
- ✅ `--system-prompt <text>` - System prompt for chat templates
- ✅ `--dump-ids` / `--dump-cpp-ids` - Token debugging
- ✅ `--verbose` - Diagnostic output
- ✅ Backend preflight validation
- ✅ Template-aware tokenization (formatted prompt + BOS/special flags)
- ✅ Fail-fast token parity (saves 20-30s on mismatch)
- ✅ Build-time library discovery (CROSSVAL_HAS_BITNET/LLAMA env vars)

**Missing Features (blocking full functionality):**
- ❌ BitNet.cpp FFI shim (D1) - **CRITICAL BLOCKER**
- ❌ Rust bindings for BitNet (D3) - depends on D1
- ❌ Backend dispatcher (D4) - depends on D3
- ❌ Integration tests (E2)
- ❌ Documentation (E3)

**Current Flow:**
```
User runs crossval-per-token
  ↓
Backend selected (auto-detect or explicit)
  ↓
Preflight check (validates libs available)
  ↓
Template applied to prompt
  ↓
Rust tokenization (with template flags)
  ↓
C++ tokenization (uses llama.cpp wrapper - needs D1/D3/D4 for BitNet)
  ↓
Token parity check (FAIL-FAST)
  ↓
Rust logits evaluation
  ↓
C++ logits evaluation (uses llama.cpp wrapper - needs D1/D3/D4 for BitNet)
  ↓
Divergence detection
```

### File Locations Reference

**Modified Files:**
- `xtask/src/main.rs` - Main CLI and crossval-per-token command
- `xtask/src/crossval/backend.rs` - CppBackend enum and auto-detection
- `xtask/src/crossval/preflight.rs` - Library preflight validation
- `xtask/src/crossval/mod.rs` - Module exports
- `xtask/src/lib.rs` - Crate-level crossval export
- `crossval/build.rs` - Backend-aware library discovery
- `crossval/src/lib.rs` - Test for env vars

**Files to Create (D1-E3):**
- `crossval/src/bitnet_cpp_wrapper.c` - C FFI shim for bitnet.cpp
- `xtask/src/crossval/dispatch.rs` - Backend dispatcher (optional - can go in main.rs)
- `crossval/tests/dual_backend_integration.rs` - Integration tests
- `docs/explanation/dual-backend-crossval.md` - Architecture documentation

**Files to Modify (D3-E3):**
- `crossval/src/cpp_bindings.rs` - Add Rust FFI bindings
- `crossval/src/token_parity.rs` - Add backend context to errors
- `docs/howto/cpp-setup.md` - Add bitnet.cpp setup
- `CLAUDE.md` - Update CLI reference

### Environment Variables

**Build-time (exported by crossval/build.rs):**
- `CROSSVAL_HAS_BITNET=true|false` - BitNet.cpp libraries available
- `CROSSVAL_HAS_LLAMA=true|false` - LLaMA.cpp libraries available

**Runtime (used for setup/testing):**
- `BITNET_CROSSVAL_LIBDIR` - Explicit lib directory (priority 1)
- `BITNET_CPP_DIR` - BitNet.cpp installation root
- `BITNET_CPP_PATH` - Alternative to BITNET_CPP_DIR

### Known Issues / Blockers

**Critical:**
1. **BitNet.cpp API unknown** - We don't know if bitnet.cpp exposes:
   - `bitnet_tokenize()` function
   - `bitnet_eval()` with per-position logits
   - Model loading/context management
   - **Mitigation:** Create minimal shim with placeholders, update when API confirmed

**Medium:**
2. **Per-position logits limitation** - BitNet.cpp may only expose last-position logits
   - **Mitigation:** Loop with prefill + eval_last for each position (slower but works)

**Low:**
3. **Template auto-detection simple** - Currently uses safe fallback (Raw)
   - **Enhancement:** Add GGUF metadata detection in future iteration

---

## 🎯 Success Criteria for Completion

**M2 Complete When:**
- ✅ bitnet_cpp_wrapper.c compiles (with #ifdef guards for missing headers)
- ✅ Rust bindings compile and provide safe API
- ✅ Backend dispatcher routes correctly for both backends
- ✅ Can run crossval-per-token with `--cpp-backend bitnet` (even if it uses stub/mock)

**M3 Complete When:**
- ✅ Token parity errors include backend-specific guidance
- ✅ Integration tests compile and pass (ignored tests skip gracefully)
- ✅ Documentation updated with both lanes' usage examples
- ✅ All acceptance criteria from IMPLEMENTATION_ROADMAP.md met

**Overall Feature Complete When:**
- ✅ Can run Lane A (BitNet vs bitnet.cpp) end-to-end
- ✅ Can run Lane B (BitNet-rs vs llama.cpp) end-to-end
- ✅ Token parity pre-gate works for both lanes
- ✅ Backend-specific error messages guide users
- ✅ Documentation covers installation, usage, troubleshooting
- ✅ Tests validate both lanes (with graceful skip when C++ missing)

---

## 📖 Documentation Reference

**Exploration Reports (150KB+):**
- `CROSSVAL_EXPLORATION_REPORT.md` - Crossval architecture (1,177 lines)
- `docs/explanation/xtask-crossval-per-token-implementation-analysis.md` - Flow analysis
- `docs/reference/BUILD_RS_LIBRARY_DISCOVERY_AND_LINKING.md` - Library discovery patterns
- `TOKENIZER_INTEGRATION_REPORT.md` - Template system (962 lines)
- `CLI_ARGUMENT_PARSING_PATTERNS.md` - Clap patterns
- `docs/reports/BITNET_CPP_C_API_EXPLORATION_REPORT.md` - BitNet.cpp API surface
- `LOGGING_AND_DIAGNOSTIC_PATTERNS.md` - Error handling patterns
- `docs/explanation/crossval-test-infrastructure.md` - Test patterns
- `docs/explanation/backend-detection-and-device-selection-patterns.md` - Backend patterns
- `PROMPT_TEMPLATE_SYSTEM_REPORT.md` - Template types and auto-detection

**Specifications:**
- `docs/explanation/dual-backend-crossval-spec.md` (50KB) - Feature specification
- `docs/explanation/architecture/adr-016-dual-backend-crossval-architecture.md` (17KB) - ADR
- `docs/explanation/dual-backend-crossval-impact-analysis.md` (22KB) - Impact analysis
- `docs/explanation/IMPLEMENTATION_ROADMAP.md` (8.7KB) - Task breakdown (this document's source)

---

## 🚀 Agent Call Templates (Copy-Paste Ready)

### D1: Create bitnet_cpp_wrapper.c

```
Agent: impl-creator
Task: D1 - Create bitnet_cpp_wrapper.c FFI shim

Goal: Minimal C/C++ shim exposing bitnet.cpp tokenization and evaluation

Context:
- Create new file: crossval/src/bitnet_cpp_wrapper.c
- Must compile when bitnet.cpp headers available
- Guard with #ifdef to allow compilation without headers
- Follow llama.cpp wrapper pattern for consistency

Required Functions:
1. bitnet_tokenize() - Tokenize text using bitnet.cpp
   Signature: int bitnet_tokenize(const char* model_path, const char* prompt,
                                   int add_bos, int32_t* out_tokens,
                                   int32_t* out_n_tokens, char* err, int32_t err_len)

2. bitnet_eval_with_tokens() - Evaluate tokens and return logits
   Signature: int bitnet_eval_with_tokens(const char* model_path, const int32_t* tokens,
                                           int32_t n_tokens, int32_t n_ctx, float* logits,
                                           int32_t* out_rows, int32_t* out_cols,
                                           char* err, int32_t err_len)

Implementation Guidelines:
- Load model per-call (no session state yet)
- Return 0 on success, -1 on error (write error string to err buffer)
- Use extern "C" for C linkage
- Add header comment documenting bitnet.cpp API assumptions

Acceptance Criteria:
- File compiles when bitnet.cpp headers available
- Guarded with #ifdef so project builds without headers
- Error handling with string messages
- No memory leaks

Verification:
cargo build -p crossval --features ffi
```

### D3: Rust FFI bindings

```
Agent: impl-creator
Task: D3 - Rust FFI bindings for bitnet_cpp_wrapper

Goal: Safe Rust wrappers for bitnet_tokenize and bitnet_eval_with_tokens

Context:
- Modify: crossval/src/cpp_bindings.rs
- Pattern: Follow existing llama.cpp wrapper style
- Feature gate: #[cfg(feature = "ffi")]

Required Functions:
1. tokenize_bitnet(model_path: &Path, prompt: &str, add_bos: bool) -> Result<Vec<i32>>
2. eval_bitnet(model_path: &Path, tokens: &[i32], n_ctx: usize) -> Result<Vec<Vec<f32>>>

Implementation Guidelines:
- CString conversion with proper error handling
- Pre-allocate buffers conservatively
- Convert C errors to anyhow::Error
- No unsafe code outside extern "C" blocks
- Memory safety: no buffer overflows, null checks

Acceptance Criteria:
- Safe API (no UB)
- Proper error propagation
- Doc comments with examples
- Feature-gated compilation

Verification:
cargo check -p crossval --features ffi
cargo clippy -p crossval --features ffi -- -D warnings
```

### D4: Backend dispatcher

```
Agent: impl-creator
Task: D4 - Backend dispatcher in xtask

Goal: Route tokenize_cpp/eval_cpp based on CppBackend enum

Context:
- Modify: xtask/src/main.rs (or create xtask/src/crossval/dispatch.rs)
- Pattern: Simple match-based routing

Required Functions:
1. tokenize_cpp(backend: CppBackend, model: &Path, prompt: &str, add_bos: bool) -> Result<Vec<i32>>
2. eval_cpp(backend: CppBackend, model: &Path, tokens: &[i32]) -> Result<Vec<Vec<f32>>>

Implementation:
match backend {
    CppBackend::BitNet => crossval::tokenize_bitnet(...),
    CppBackend::Llama => /* existing llama.cpp wrapper */,
}

Acceptance Criteria:
- Routing works for both backends
- Existing llama.cpp path unchanged
- BitNet path calls new bindings
- Compiles with --features crossval,ffi

Verification:
cargo check -p xtask --features crossval-all
```

---

## 💡 Tips for Next Thread

1. **Start with D1** - It's the critical blocker; everything else depends on it
2. **Use exploration reports** - All patterns documented, don't reinvent
3. **Test incrementally** - Build after each task to catch issues early
4. **Feature gates matter** - Always use `#[cfg(feature = "ffi")]` for C++ code
5. **Mock if needed** - If bitnet.cpp API unclear, create stub that compiles
6. **Check CLAUDE.md** - Updated context about current features and usage

**Estimated Time Remaining:**
- M2 (3 tasks): 5-6 hours
- M3 (3 tasks): 4-5 hours
- **Total: 9-11 hours (1.5-2 developer days)**

---

## 📞 Quick Reference

**Compile checks:**
```bash
cargo check -p xtask --features crossval-all
cargo check -p crossval --features ffi
cargo clippy -p crossval --features ffi -- -D warnings
cargo test -p crossval --features ffi
```

**Run crossval-per-token:**
```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/test.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" \
  --cpp-backend bitnet \
  --prompt-template raw \
  --verbose \
  --max-tokens 4
```

**Useful env vars:**
```bash
export BITNET_CROSSVAL_LIBDIR=/path/to/libs  # Override lib search
export BITNET_CPP_DIR=/path/to/bitnet_cpp    # BitNet.cpp root
export RUST_LOG=debug                         # Verbose logging
```

---

**End of Continuation Plan**

This document contains everything needed to pick up where we left off and complete the dual-backend cross-validation feature. Good luck! 🚀
