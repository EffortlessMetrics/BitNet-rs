# CrossVal FFI Infrastructure Exploration Report

**Date**: 2025-10-25  
**Scope**: Comprehensive analysis of existing FFI patterns, build infrastructure, and patterns to follow  
**Goal**: Understand what exists, what needs to be created, and what gaps exist

---

## Executive Summary

The BitNet.rs codebase has a **mature, production-quality FFI infrastructure** with:
- Two-tier FFI architecture: `bitnet-sys` (low-level) + `bitnet-crossval` (high-level)
- Complete build.rs infrastructure with library discovery and linking
- Safe wrapper patterns around C FFI with proper error handling
- Feature-gated compilation to prevent unwanted dependencies
- Two-pass size negotiation pattern for buffer management (from llama.cpp)

**Key Finding**: The infrastructure is _complete for validation_ but has **known gaps** in actual C++ integration (currently using mock/stubs). The wrapper layer is production-ready; the C++ implementation side needs completion.

---

## Directory Structure & File Organization

### CrossVal Crate Layout

```
crossval/
├── Cargo.toml                  # Feature flags: ffi, crossval, crossval-all
├── build.rs                    # C++ discovery, linking, wrapper compilation
├── src/
│   ├── lib.rs                  # Main module, error types, CrossvalConfig
│   ├── cpp_bindings.rs         # High-level safe C++ wrappers (CppModel)
│   ├── bitnet_cpp_wrapper.c    # CURRENTLY A MOCK - needs real impl
│   ├── comparison.rs           # Cross-validation runner
│   ├── token_parity.rs         # Token sequence validation (fail-fast)
│   ├── logits_compare.rs       # Numerical comparison helpers
│   ├── validation.rs           # Comprehensive validation suite
│   ├── fixtures.rs             # Test fixtures & test data
│   ├── utils.rs                # Utilities: perf measurement, logging
│   └── score.rs                # Scoring/metrics
└── tests/
    ├── ffi_integration.rs      # FFI tests (requires ffi feature)
    ├── token_equivalence.rs    # Token parity tests
    ├── performance_validation.rs
    ├── iq2s_validation.rs      # IQ2_S FFI validation
    └── parity_bitnetcpp.rs     # Full integration tests
```

### BitNet-Sys Crate (Lower-Level FFI)

```
crates/bitnet-sys/
├── Cargo.toml
├── build.rs                    # Bindgen config, C++ shim compilation
├── src/
│   ├── lib.rs                  # Module organization, feature gates
│   ├── wrapper.rs              # Safe wrappers: Model, Context, Session
│   └── bindings/               # Auto-generated from bindgen (OUT_DIR)
├── include/
│   └── bitnet_c.h              # C FFI header (our custom wrapper)
├── csrc/
│   └── bitnet_c_shim.cc        # C++ shim (to be created if needed)
└── tests/
    ├── ffi_lifecycle.rs
    ├── lifecycle.rs
    └── disabled.rs
```

---

## Existing FFI Patterns: What We Can Reuse

### 1. Feature-Gated Compilation (Best Practice)

**File**: `crossval/Cargo.toml` (lines 10-18)

```toml
[features]
default = []
crossval = ["dep:bindgen", "dep:cc", "dep:bitnet-sys", "bitnet-sys/ffi", "bitnet-inference/ffi", "ffi"]
ffi = ["dep:cc", "bitnet-inference/ffi"]
iq2s-ffi = ["bitnet-models/iq2s-ffi", "bitnet-ggml-ffi/iq2s-ffi"]
cpp-probe = []
integration-tests = []
cpu = ["bitnet-inference/cpu", "bitnet-models/cpu"]
gpu = ["bitnet-inference/gpu", "bitnet-models/gpu"]
```

**Pattern to Follow**:
- `ffi`: Core FFI dependencies (only when needed)
- `crossval`: Full cross-validation (depends on `ffi`)
- Feature-gated modules in `lib.rs`:

```rust
#[cfg(feature = "crossval")]
pub mod cpp_bindings;

#[cfg(feature = "crossval")]
pub mod comparison;
```

### 2. C String Handling (Safe Pattern)

**File**: `crates/bitnet-sys/src/wrapper.rs` (lines 65-76, 146-159)

**Pattern 1: Path strings**
```rust
pub fn load(path: &str) -> Result<Self> {
    let c_path = CString::new(path)?;
    // ... use c_path.as_ptr() ...
}
```

**Pattern 2: Text strings**
```rust
pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
    let c_text = CString::new(text)?;
    let n_tokens = unsafe {
        llama_tokenize(
            model,
            c_text.as_ptr(),
            text.len() as i32,
            ptr::null_mut(),
            0,
            add_special,
            false,
        )
    };
    // ...
}
```

**Error Handling**:
- `CString::new()` returns `Result<CString, NulError>` - catches null bytes in strings
- Wraps in custom `CppError::InvalidCString` or `CppError::InvalidUtf8`

### 3. Two-Pass Size Negotiation (Critical Pattern)

**File**: `crates/bitnet-sys/src/wrapper.rs` (lines 145-186)

This is the **key pattern** for C APIs that require buffer allocation:

```rust
pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
    let c_text = CString::new(text)?;
    let model = unsafe { llama_get_model(self.ptr) };

    // PASS 1: Get count (NULL buffer, 0 capacity)
    let n_tokens = unsafe {
        llama_tokenize(
            model,
            c_text.as_ptr(),
            text.len() as i32,
            ptr::null_mut(),      // No buffer
            0,                      // No capacity
            add_special,
            false,
        )
    };

    if n_tokens < 0 {
        return Err(CppError::LlamaError("Tokenization failed".to_string()));
    }

    // PASS 2: Allocate and fill
    let mut tokens = vec![0i32; n_tokens as usize];
    let actual_n = unsafe {
        llama_tokenize(
            model,
            c_text.as_ptr(),
            text.len() as i32,
            tokens.as_mut_ptr(),   // Real buffer
            tokens.len() as i32,   // Real capacity
            add_special,
            false,
        )
    };

    if actual_n < 0 {
        return Err(CppError::LlamaError("Tokenization failed".to_string()));
    }

    tokens.truncate(actual_n as usize);
    Ok(tokens)
}
```

**Why This Matters**: 
- C APIs don't know how much data to return until after processing
- First call with NULL buffer returns required size
- Second call with allocated buffer fills it
- This avoids memory copies and pre-allocating large buffers

### 4. Null Pointer Checks (Safety)

**File**: `crates/bitnet-sys/src/wrapper.rs` (lines 64-78)

```rust
pub fn load(path: &str) -> Result<Self> {
    let c_path = CString::new(path)?;
    let params = unsafe { llama_model_default_params() };
    let ptr = unsafe { llama_load_model_from_file(c_path.as_ptr(), params) };

    if ptr.is_null() {
        return Err(CppError::ModelLoadError(format!("Failed to load model from: {}", path)));
    }

    Ok(Model { ptr })
}
```

**Pattern**: Always check for NULL return from C functions before using.

### 5. RAII Pattern for Resource Cleanup

**File**: `crates/bitnet-sys/src/wrapper.rs` (lines 102-117, 316-327)

```rust
impl Drop for Model {
    fn drop(&mut self) {
        // Use mem::replace to prevent double-free
        let ptr = std::mem::replace(&mut self.ptr, std::ptr::null_mut());
        if !ptr.is_null() {
            unsafe {
                llama_free_model(ptr);
            }
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let ptr = std::mem::replace(&mut self.ptr, std::ptr::null_mut());
        if !ptr.is_null() {
            unsafe {
                llama_free(ptr);
            }
        }
    }
}
```

**Key Points**:
- `mem::replace()` ensures pointer is null'd before Drop completes
- Prevents double-free if Drop is called twice
- Checked against null to avoid freeing null pointers

### 6. Wrapper Structs for Type Safety

**File**: `crates/bitnet-sys/src/wrapper.rs` (lines 59-117, 120-313, 330-394)

```rust
pub struct Model {
    ptr: *mut llama_model,
}

pub struct Context {
    ptr: *mut llama_context,
}

pub struct Session {
    pub model: Model,
    pub context: Context,
}

// Session provides convenient combined operations
impl Session {
    pub fn load_deterministic(model_path: &str) -> Result<Self> {
        let model = Model::load(model_path)?;
        let context = Context::new(&model, 2048, 512, 1)?;
        Ok(Session { model, context })
    }

    pub fn eval_and_get_logits(&mut self, tokens: &[i32], n_past: i32) -> Result<Vec<f32>> {
        self.context.eval(tokens, n_past)?;
        self.context.get_logits()
    }
}
```

**Advantages**:
- Type-safe wrappers prevent accidental mixing of different pointer types
- Encapsulation hides raw pointers from users
- Composable: `Session` combines `Model` + `Context`

### 7. Two-Tier Abstraction (Low-level + High-level)

**Low-level** (bitnet-sys):
```rust
// In crates/bitnet-sys/src/wrapper.rs
pub unsafe extern "C" {
    fn llama_backend_init();
    fn llama_load_model_from_file(path: *const c_char, params: llama_model_params) -> *mut llama_model;
    // ... raw bindings from bindgen ...
}

pub struct Model { ptr: *mut llama_model }
impl Model {
    pub fn load(path: &str) -> Result<Self> { /* ... */ }
    pub fn n_vocab(&self) -> i32 { unsafe { llama_n_vocab(self.ptr) } }
}
```

**High-level** (crossval):
```rust
// In crossval/src/cpp_bindings.rs
pub struct CppModel {
    handle: *mut c_void,  // Opaque handle to C++ model
}

impl CppModel {
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let path_str = model_path.as_ref().to_str()?;
        let c_path = CString::new(path_str)?;
        let handle = unsafe { bitnet_cpp_create_model(c_path.as_ptr()) };
        if handle.is_null() {
            return Err(CrossvalError::ModelLoadError(...));
        }
        Ok(CppModel { handle })
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<Vec<u32>> {
        // Business logic here, calls C FFI
    }
}
```

### 8. Error Type Unification

**File**: `crates/bitnet-sys/src/wrapper.rs` (lines 11-30)

```rust
#[derive(Debug, thiserror::Error)]
pub enum CppError {
    #[error("Null pointer returned from C++")]
    NullPointer,

    #[error("Invalid UTF-8 string: {0}")]
    InvalidUtf8(#[from] std::str::Utf8Error),

    #[error("Invalid C string: {0}")]
    InvalidCString(#[from] std::ffi::NulError),

    #[error("LLAMA error: {0}")]
    LlamaError(String),

    #[error("Model loading failed: {0}")]
    ModelLoadError(String),
}

pub type Result<T> = std::result::Result<T, CppError>;
```

**Pattern**: 
- Single unified error type for all FFI operations
- Uses `thiserror` for auto-implementing `std::error::Error`
- `#[from]` annotations enable transparent error conversion

### 9. Thread Safety Markers

**File**: `crates/bitnet-sys/src/wrapper.rs` (lines 115-117, 327-328)

```rust
// Safety: llama models are thread-safe for reading
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

// For cross-validation in tests
unsafe impl Send for Context {}
```

**Pattern**: Explicitly state thread-safety contracts (even if unsafe).

---

## Build System: build.rs Patterns

### Crossval build.rs Flow

**File**: `crossval/build.rs` (lines 26-132)

```
1. Feature check: if !ffi feature, skip all native compilation
   └─ Return early with just env vars

2. Library discovery:
   ├─ Priority 1: BITNET_CROSSVAL_LIBDIR (explicit)
   ├─ Priority 2: BITNET_CPP_DIR or BITNET_CPP_PATH
   ├─ Priority 3: $HOME/.cache/bitnet_cpp
   └─ Search paths: build/, build/lib/, build/3rdparty/llama.cpp/src, etc.

3. Link libraries:
   ├─ For each search path that exists:
   │  └─ println!("cargo:rustc-link-search=native={}", path)
   ├─ Auto-detect and link: libbitnet, libllama, libggml
   └─ Link C++ runtime: stdc++ (Linux), c++ (macOS)

4. Emit metadata:
   ├─ CROSSVAL_HAS_BITNET (true/false)
   ├─ CROSSVAL_HAS_LLAMA (true/false)
   └─ Diagnostic warnings
```

**Key Patterns**:
- `println!("cargo:rustc-link-search=native={}", path)` - Tell Rust where to find libraries
- `println!("cargo:rustc-link-lib=dylib=libname")` - Link library by name
- `println!("cargo:rustc-env=VAR=value")` - Export env var to Rust code
- `env!("VAR")` at compile-time to check what was found

### BitNet-Sys build.rs (More Advanced)

**File**: `crates/bitnet-sys/build.rs` (lines 73-195)

```
1. Feature check: if !ffi feature, return early
   
2. Locate C++ root:
   ├─ Try BITNET_CPP_DIR / BITNET_CPP_PATH
   └─ Panic if not found (explicit feature-gated failure)

3. Link C++ libraries:
   ├─ Search paths (multiple locations)
   ├─ Link dynamic libraries: llama, ggml, stdc++/c++
   └─ Add RPATH: println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path)

4. Compile C++ shim:
   ├─ Check csrc/bitnet_c_shim.cc exists
   ├─ Add local includes: -I (warnings visible)
   ├─ Add system includes: -isystem (warnings suppressed)
   └─ Use xtask_build_helper::compile_cpp_shim()

5. Generate bindings (bindgen):
   ├─ Find bitnet_c.h (local custom wrapper)
   ├─ Find llama.h (from BitNet C++ repo)
   ├─ Generate Rust bindings → OUT_DIR/bindings.rs
   └─ Clean up (allowlist, blocklist)
```

**Key Pattern: RPATH**
```rust
println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
```
This embeds library search paths into the binary, eliminating need for LD_LIBRARY_PATH at runtime.

---

## Mock vs Real Implementation Gap

### Current State: Mock Implementation

**File**: `crossval/src/bitnet_cpp_wrapper.c`

```c
// Mock implementations - just return dummy data
void* bitnet_cpp_create_model(const char* model_path) {
    // Checks file exists, returns opaque handle
    bitnet_model_t* model = (bitnet_model_t*)malloc(...);
    return model;
}

int bitnet_cpp_generate(...) {
    // Returns dummy tokens (100 + i, etc.)
    *tokens_count = (max_tokens < 10) ? max_tokens : 10;
    for (int i = 0; i < *tokens_count; i++) {
        tokens_out[i] = 100 + i;
    }
    return 0;
}

void bitnet_cpp_destroy_model(void* model) {
    // Just frees malloc'd memory
}
```

**Known Issues**:
- These are stubs only - no actual inference
- Actual C++ implementation would call into llama.cpp or BitNet.cpp
- `bitnet_c.h` shows the _intended_ API (more complete)

### What Needs to Be Done

1. **Implement csrc/bitnet_c_shim.cc**: Bridge Rust FFI → C++ implementation
   - Use `bitnet_c.h` (already defined)
   - Actually call into BitNet.cpp or llama.cpp
   - Handle tokenization, model loading, inference

2. **Update bitnet_cpp_wrapper.c** or replace entirely with:
   - Real model loading
   - Real tokenization
   - Real inference via C++ backend

3. **Verify bindgen integration**:
   - Make sure `OUT_DIR/bindings.rs` is generated correctly
   - Check that bitnet-sys can actually link against libraries

---

## Key FFI Header: bitnet_c.h

**File**: `crates/bitnet-sys/include/bitnet_c.h`

```c
// Opaque pointers (caller doesn't need to know structure)
typedef struct bitnet_model bitnet_model_t;
typedef struct bitnet_ctx   bitnet_ctx_t;

// Configuration struct (passed by value)
typedef struct {
  int32_t n_ctx;
  int32_t n_threads;
  int32_t seed;
  float   rope_freq;
} bitnet_params_t;

// Model lifecycle
bitnet_model_t* bitnet_model_new_from_file(const char* gguf_path);
void            bitnet_model_free(bitnet_model_t*);

// Context lifecycle
bitnet_ctx_t*   bitnet_context_new(bitnet_model_t*, const bitnet_params_t*);
void            bitnet_context_free(bitnet_ctx_t*);

// Core operations (two-pass pattern for tokenize, one-pass for eval)
int bitnet_tokenize(bitnet_model_t*, const char* text, int add_bos, int parse_special,
                    int32_t* out_ids, int out_cap);

int bitnet_eval(bitnet_ctx_t*, const int32_t* ids, int n_ids,
                float* logits_out, int logits_cap);

int bitnet_prefill(bitnet_ctx_t*, const int32_t* ids, int n_ids);

int bitnet_vocab_size(bitnet_ctx_t* ctx);

int bitnet_decode_greedy(bitnet_model_t* model, bitnet_ctx_t* ctx,
                         int eos_id, int eot_id, int max_steps,
                         int* out_token_ids, int out_cap);
```

**Pattern Analysis**:
- **Opaque pointers** for models/contexts (internals hidden from Rust)
- **Config struct** passed by value (stack-safe, simple)
- **Return codes** (int): negative = error, non-negative = success/size
- **Buffer pattern**: `out_*` pointer + capacity (`*_cap`)
- **Two-pass for variable-length**: `bitnet_tokenize()` with `out_cap=0` returns needed size

---

## Feature Architecture: Conditional Compilation

### How Feature Gates Work

**At cargo level**:
```toml
[features]
ffi = ["dep:cc", ...]           # Opt-in FFI support
crossval = ["ffi", ...]         # Depends on ffi feature
```

**At crate level** (`lib.rs`):
```rust
#[cfg(feature = "crossval")]
pub mod cpp_bindings;           // Only if crossval feature enabled

#[cfg(feature = "crossval")]
pub mod comparison;

#[cfg(not(feature = "crossval"))]
pub fn assert_first_logits_match(_model_path: &str, _prompt: &str) {
    panic!("crossval feature required");
}
```

**At module level** (`cpp_bindings.rs`):
```rust
#[cfg(all(feature = "ffi", have_cpp))]
mod imp {
    // Real implementation with FFI calls
    unsafe extern "C" {
        fn bitnet_cpp_create_model(...);
        // ...
    }
}

#[cfg(any(not(feature = "ffi"), not(have_cpp)))]
mod imp {
    // Stub implementation that returns errors
    pub struct CppModel;
    impl CppModel {
        pub fn load(...) -> Result<Self> {
            Err(CrossvalError::ModelLoadError("unavailable"))
        }
    }
}

// Always export one or the other
pub use imp::*;
```

**At build.rs level** (`build.rs`):
```rust
// Check if feature was enabled at compile time
if env::var("CARGO_FEATURE_FFI").is_err() {
    return;  // Skip all native compilation
}

// Emit have_cpp cfg if libraries found
println!("cargo:rustc-cfg=have_cpp");
```

**Result**: Can build without C++ deps by using `--features cpu` instead of `--features ffi,crossval`.

---

## Validation Infrastructure (What's Already Here)

### Token Parity Module

**File**: `crossval/src/token_parity.rs`

Provides **fail-fast** validation before expensive logits comparison:

```rust
pub fn validate_token_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[i32],
    prompt: &str,
) -> anyhow::Result<()> {
    // Convert i32 → u32
    let cpp_tokens_u32: Vec<u32> = cpp_tokens.iter().map(|&id| id as u32).collect();

    // Check exact match
    if rust_tokens != cpp_tokens_u32.as_slice() {
        let first_diff = find_first_diff(rust_tokens, &cpp_tokens_u32);
        eprintln!("{}", format_token_mismatch_error(&error));
        anyhow::bail!("Token sequence mismatch at index {}", first_diff);
    }

    Ok(())  // Silent success
}
```

**Why Useful**: Catches tokenizer mismatches (duplicate BOS, template differences) before wasting time on logits comparison.

### Comparison Runner

**File**: `crossval/src/comparison.rs`

High-level orchestration:

```rust
pub struct CrossValidator {
    config: CrossvalConfig,
}

impl CrossValidator {
    pub fn validate_fixture(&self, fixture: &TestFixture) -> Result<Vec<ComparisonResult>> {
        let cpp_model = CppModel::load(&fixture.model_path)?;
        
        for prompt in &fixture.test_prompts {
            let result = self.compare_single_prompt(&fixture.name, prompt, &cpp_model);
            results.push(result);
        }
        
        Ok(results)
    }

    fn compare_single_prompt(...) -> ComparisonResult {
        // Generate with Rust
        let (rust_perf, rust_tokens) = self.generate_rust(prompt)?;
        
        // Generate with C++
        let (cpp_perf, cpp_tokens) = cpp_model.generate(prompt, max_tokens)?;
        
        // Compare
        let tokens_match = compare_tokens(&rust_tokens, &cpp_tokens, &self.config)?;
        
        // Return result struct with both outputs
    }
}
```

---

## Summary: What Exists, What's Needed

### ✅ COMPLETE (Production-Ready)

1. **Feature-gated compilation**
   - `ffi` and `crossval` features are properly defined
   - Conditional compilation prevents unwanted dependencies
   - Stubs provided when features disabled

2. **Build.rs infrastructure**
   - Library discovery (multiple search paths)
   - Linking (auto-detect libbitnet, libllama, libggml)
   - C++ shim compilation support
   - Diagnostic output (CROSSVAL_HAS_BITNET, etc.)

3. **Safe wrapper patterns**
   - Two-pass size negotiation for tokenization
   - Null pointer checks throughout
   - RAII pattern for cleanup (Drop impl)
   - CString handling with proper error checks

4. **Two-tier abstraction**
   - bitnet-sys: low-level FFI (bindgen-generated + wrapper.rs)
   - crossval: high-level domain logic (CppModel, comparison, validation)

5. **Validation framework**
   - Token parity pre-gate (fail-fast)
   - Comparison runner (orchestrate Rust vs C++ tests)
   - Fixtures (test data)
   - Logits comparison with tolerance

### ⚠️ INCOMPLETE (Needs Work)

1. **C++ Shim Implementation** (`csrc/bitnet_c_shim.cc`)
   - Currently referenced in build.rs but no file exists (would be in crates/bitnet-sys/csrc/)
   - Needs to bridge `bitnet_c.h` API → actual llama.cpp / BitNet.cpp calls

2. **Mock Wrapper** (`crossval/src/bitnet_cpp_wrapper.c`)
   - Currently returns dummy tokens
   - Needs to call into actual C++ backend via shim
   - Or replace entirely with calls to generated bindings

3. **Bindgen Configuration** (`crates/bitnet-sys/build.rs`, lines 245+)
   - Incomplete in provided code
   - Needs to properly configure allowlist/blocklist
   - Must handle llama.h generation correctly

4. **Integration Testing**
   - Tests exist but blocked by mock implementation
   - `tests/parity_bitnetcpp.rs` requires real integration

### 🔧 PARTIAL (Framework Exists, Content Needed)

1. **bitnet_c.h** - Signature complete, but implementation unclear
   - Defines API contract
   - Actual C++ implementation needs to satisfy this

2. **Test fixtures** (`crossval/src/fixtures.rs`)
   - Framework in place
   - Test cases need to be populated

---

## Recommended Next Steps

### 1. Create C++ Shim
```
crates/bitnet-sys/csrc/bitnet_c_shim.cc
```
- Implement functions from `bitnet_c.h`
- Call into llama.cpp / BitNet.cpp internals
- Handle tokenization, model loading, inference
- Return proper error codes

### 2. Verify Bindgen Output
- Confirm `OUT_DIR/bindings.rs` is generated
- Check that llama API symbols are available
- Test linking in isolation first

### 3. Update Wrapper Tests
- `crates/bitnet-sys/tests/ffi_lifecycle.rs`
- Verify basic model loading and inference work

### 4. Integration Tests
- Update `crossval/tests/parity_bitnetcpp.rs`
- Add real models to fixtures
- Verify Rust vs C++ parity

---

## Code References (Absolute Paths)

### Core FFI Files
- `/home/steven/code/Rust/BitNet-rs/crossval/src/cpp_bindings.rs` - High-level wrappers
- `/home/steven/code/Rust/BitNet-rs/crossval/src/bitnet_cpp_wrapper.c` - Mock C wrapper
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/src/wrapper.rs` - Safe wrappers
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/include/bitnet_c.h` - C API header

### Build Files
- `/home/steven/code/Rust/BitNet-rs/crossval/build.rs` - Library discovery & linking
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/build.rs` - Advanced linking + bindgen

### Feature Configuration
- `/home/steven/code/Rust/BitNet-rs/crossval/Cargo.toml` - Features: ffi, crossval, etc.
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/Cargo.toml` - FFI feature definition

### Validation
- `/home/steven/code/Rust/BitNet-rs/crossval/src/token_parity.rs` - Token parity pre-gate
- `/home/steven/code/Rust/BitNet-rs/crossval/src/comparison.rs` - Cross-validation runner
- `/home/steven/code/Rust/BitNet-rs/crossval/src/validation.rs` - Comprehensive validation suite

---

## Key Takeaways

1. **The infrastructure is solid** - production-quality patterns throughout
2. **Feature gates prevent cascading dependencies** - can build without C++ when not needed
3. **Two-pass pattern is essential** - will be needed for any buffer-oriented C API
4. **Error handling is comprehensive** - CppError covers all known failure modes
5. **The gap is in C++ integration**, not in Rust FFI design
6. **RPATH embedding eliminates LD_LIBRARY_PATH** - binary is self-contained
7. **Safe wrappers make testing easier** - tests don't need unsafe code
8. **Two-tier architecture separates concerns** - bitnet-sys is low-level, crossval is business logic

