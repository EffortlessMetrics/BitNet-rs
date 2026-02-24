# CrossVal FFI Quick Reference

**TL;DR**: The infrastructure is solid and production-ready. The gap is in C++ implementation, not Rust FFI design. Copy existing patterns to implement new wrappers.

---

## File Structure at a Glance

```
├── crossval/
│   ├── build.rs                    ← C++ discovery & linking
│   └── src/
│       ├── cpp_bindings.rs         ← High-level wrappers (pattern: FOLLOW THIS)
│       ├── bitnet_cpp_wrapper.c    ← Mock C wrapper (REPLACE WITH REAL)
│       ├── comparison.rs           ← Cross-validation runner
│       └── token_parity.rs         ← Token validation pre-gate
│
└── crates/bitnet-sys/
    ├── build.rs                    ← Bindgen + C++ shim compilation
    ├── include/
    │   └── bitnet_c.h              ← C FFI API (the contract)
    ├── src/
    │   └── wrapper.rs              ← Safe wrappers (PATTERN LIBRARY)
    └── csrc/
        └── bitnet_c_shim.cc        ← (NEEDS TO BE CREATED)
```

---

## 9 Essential FFI Patterns to Reuse

### 1. Feature-Gated Module Declaration
```rust
#[cfg(feature = "crossval")]
pub mod cpp_bindings;

#[cfg(feature = "crossval")]
pub mod comparison;

// Stub when disabled
#[cfg(not(feature = "crossval"))]
pub fn some_function(...) {
    panic!("crossval feature required");
}
```

### 2. Safe CString Conversion
```rust
let c_path = CString::new(path)?;  // Returns Result, catches null bytes
let handle = unsafe { ffi_func(c_path.as_ptr()) };
```

### 3. Two-Pass Buffer Pattern (CRITICAL)
```rust
// PASS 1: Get required size
let count = unsafe { ffi_tokenize(model, text, std::ptr::null_mut(), 0) };
if count < 0 { return Err(...); }

// PASS 2: Allocate and fill
let mut tokens = vec![0i32; count as usize];
let actual = unsafe { ffi_tokenize(model, text, tokens.as_mut_ptr(), count) };
tokens.truncate(actual as usize);
Ok(tokens)
```

### 4. Null Pointer Check
```rust
if ptr.is_null() {
    return Err(CppError::NullPointer);
}
```

### 5. RAII Cleanup Pattern
```rust
impl Drop for Model {
    fn drop(&mut self) {
        let ptr = std::mem::replace(&mut self.ptr, std::ptr::null_mut());
        if !ptr.is_null() {
            unsafe { ffi_free_model(ptr); }
        }
    }
}
```

### 6. Wrapper Struct for Type Safety
```rust
pub struct Model {
    ptr: *mut ffi_model,  // Hidden implementation detail
}

impl Model {
    pub fn load(path: &str) -> Result<Self> {
        let c_path = CString::new(path)?;
        let ptr = unsafe { ffi_load(c_path.as_ptr()) };
        if ptr.is_null() { return Err(...); }
        Ok(Model { ptr })
    }
}
```

### 7. Error Type with Transparent Conversion
```rust
#[derive(Debug, thiserror::Error)]
pub enum CppError {
    #[error("Null pointer")]
    NullPointer,
    
    #[error("Invalid C string: {0}")]
    InvalidCString(#[from] std::ffi::NulError),  // Auto-convert
}

pub type Result<T> = std::result::Result<T, CppError>;
```

### 8. Thread Safety Markers
```rust
unsafe impl Send for Model {}
unsafe impl Sync for Model {}
```

### 9. Two-Tier Architecture
```
Low-level (bitnet-sys/wrapper.rs):
  ├─ Raw FFI bindings (bindgen-generated)
  ├─ Safe wrappers (Model, Context, Session)
  └─ Direct 1:1 mapping to C functions

High-level (crossval/cpp_bindings.rs):
  ├─ Business logic (CppModel, generate, compare)
  ├─ Error handling specific to domain
  └─ Hides low-level details from users
```

---

## Build System Checklist

### Cargo.toml Features
```toml
[features]
ffi = ["dep:cc", ...]           # Core FFI deps
crossval = ["ffi", ...]         # Full cross-validation
```

### build.rs Steps
- [x] Feature check: `if CARGO_FEATURE_FFI.is_err() { return; }`
- [x] Library discovery: multiple search paths with priority
- [x] Link libraries: `-lllama`, `-lggml`, `-lstdc++`
- [x] RPATH embedding: `-Wl,-rpath,{path}` (eliminates LD_LIBRARY_PATH)
- [x] Emit env vars: `println!("cargo:rustc-env=VAR=...")`
- [x] Compile C++ shim (if needed): cc::Build
- [x] Generate bindings: bindgen

### Environment Variable Priorities
1. BITNET_CROSSVAL_LIBDIR (explicit)
2. BITNET_CPP_DIR or BITNET_CPP_PATH
3. $HOME/.cache/bitnet_cpp
4. Search default paths

---

## Validation Infrastructure

### Token Parity Pre-Gate (Fail-Fast)
```rust
pub fn validate_token_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[i32],
    prompt: &str,
) -> anyhow::Result<()> {
    let cpp_u32: Vec<u32> = cpp_tokens.iter().map(|&id| id as u32).collect();
    if rust_tokens != cpp_u32.as_slice() {
        eprintln!("Token mismatch at index {}", find_first_diff(...));
        anyhow::bail!(...);
    }
    Ok(())  // Silent success
}
```

### Comparison Runner
```rust
pub fn compare_single_prompt(...) -> ComparisonResult {
    let rust_tokens = self.generate_rust(prompt)?;
    let cpp_tokens = cpp_model.generate(prompt, max_tokens)?;
    let match = compare_tokens(&rust_tokens, &cpp_tokens, &config)?;
    ComparisonResult { tokens_match: match, ... }
}
```

---

## What's Missing (Gaps to Fill)

### 1. C++ Shim: crates/bitnet-sys/csrc/bitnet_c_shim.cc
Create this file to bridge Rust FFI → actual C++ implementation:
```cpp
// Implement functions from bitnet_c.h
bitnet_model_t* bitnet_model_new_from_file(const char* gguf_path) {
    // Load model via llama.cpp or BitNet.cpp
    // Return opaque pointer or NULL on error
}

int bitnet_eval(bitnet_ctx_t* ctx, const int32_t* ids, int n_ids,
                float* logits_out, int logits_cap) {
    // Actually run inference
    // Return vocab size or negative on error
}
```

### 2. Real C Wrapper: crossval/src/bitnet_cpp_wrapper.c
Replace mock with real calls to the shim, OR remove and use bindgen directly.

### 3. Complete Bindgen Config: crates/bitnet-sys/build.rs
Configure allowlist/blocklist to generate clean bindings from llama.h and bitnet_c.h.

---

## Quick Start: Adding a New FFI Function

### Step 1: Define in bitnet_c.h
```c
// Header file declares the C API contract
float* bitnet_get_embeddings(bitnet_ctx_t* ctx, int token_id);
void   bitnet_free_embeddings(float* ptr);
```

### Step 2: Implement in C++ Shim
```cpp
// csrc/bitnet_c_shim.cc
float* bitnet_get_embeddings(bitnet_ctx_t* ctx, int token_id) {
    auto* cpp_ctx = static_cast<BitNetContext*>(ctx);
    float* embeddings = new float[...];
    // Fill embeddings...
    return embeddings;
}
```

### Step 3: Safe Wrapper in bitnet-sys
```rust
// crates/bitnet-sys/src/wrapper.rs
impl Context {
    pub fn get_embeddings(&self, token_id: i32) -> Result<Vec<f32>> {
        let ptr = unsafe { bitnet_get_embeddings(self.ptr, token_id) };
        if ptr.is_null() { return Err(CppError::NullPointer); }
        
        let embeddings = unsafe {
            std::slice::from_raw_parts(ptr, EMB_DIM).to_vec()
        };
        unsafe { bitnet_free_embeddings(ptr); }
        
        Ok(embeddings)
    }
}
```

### Step 4: High-Level Wrapper in crossval
```rust
// crossval/src/cpp_bindings.rs
impl CppModel {
    pub fn get_embeddings(&self, token_id: u32) -> Result<Vec<f32>> {
        // Call Session::get_embeddings via FFI
        // Add business logic here (validation, caching, etc.)
    }
}
```

---

## Environment Debugging

### Check What Was Detected
```rust
// In lib.rs tests
let has_bitnet = env!("CROSSVAL_HAS_BITNET");
let has_llama = env!("CROSSVAL_HAS_LLAMA");
println!("CROSSVAL_HAS_BITNET = {}", has_bitnet);
println!("CROSSVAL_HAS_LLAMA = {}", has_llama);
```

### Check Runtime Availability
```rust
use bitnet_sys::safe;
if safe::is_available() {
    println!("C++ implementation is available");
} else {
    println!("C++ implementation not linked");
}
```

---

## Common Pitfalls & Solutions

| Problem | Solution |
|---------|----------|
| `undefined reference to 'llama_...'` | Check `BITNET_CPP_DIR`, run build.rs in verbose mode |
| `println!("cargo:rustc-link-lib=...")` wrong format | Use `dylib=llama` not `llama` |
| Memory leak in wrapper | Implement Drop, use mem::replace to null ptr |
| CString contains null bytes | Use `?` operator on `CString::new()` result |
| Two-pass not working | First call must use `ptr::null_mut()` and `0` capacity |
| Feature gate mismatch | Use `#[cfg(any(feature = "gpu", feature = "cuda"))]` pattern |
| RPATH not set | Add `-Wl,-rpath,{path}` in build.rs println |

---

## File Checklist for Full Implementation

- [x] `crossval/Cargo.toml` - Features defined
- [x] `crossval/build.rs` - Library discovery
- [x] `crossval/src/lib.rs` - Module gating
- [x] `crossval/src/cpp_bindings.rs` - High-level wrappers
- [x] `crossval/src/token_parity.rs` - Validation pre-gate
- [x] `crates/bitnet-sys/Cargo.toml` - Feature gates
- [x] `crates/bitnet-sys/build.rs` - Binding generation
- [x] `crates/bitnet-sys/src/lib.rs` - Feature-gated exports
- [x] `crates/bitnet-sys/src/wrapper.rs` - Safe wrappers
- [x] `crates/bitnet-sys/include/bitnet_c.h` - C API contract
- [ ] `crates/bitnet-sys/csrc/bitnet_c_shim.cc` - **NEEDS CREATION**
- [ ] Update `crossval/src/bitnet_cpp_wrapper.c` - **NEEDS REAL IMPL**
- [ ] Complete bindgen config - **NEEDS COMPLETION**

---

## Key References

**For C string patterns**: See `crates/bitnet-sys/src/wrapper.rs` lines 145-186 (tokenize function)

**For two-pass pattern**: See `crates/bitnet-sys/src/wrapper.rs` lines 145-186

**For RAII cleanup**: See `crates/bitnet-sys/src/wrapper.rs` lines 102-117 (Drop impl)

**For feature gates**: See `crossval/src/cpp_bindings.rs` lines 27-167 (imp module pattern)

**For error handling**: See `crates/bitnet-sys/src/wrapper.rs` lines 11-30 (CppError enum)

**For build.rs**: See `crates/bitnet-sys/build.rs` lines 73-195 (library linking)

---

## When You Get Stuck

1. Check if feature is enabled: `cargo build --features ffi,crossval`
2. Run build.rs in verbose: `RUST_LOG=debug cargo build 2>&1 | head -100`
3. Verify libraries found: `ls -la $BITNET_CPP_DIR/build/lib/`
4. Check linker symbols: `nm -D /lib/x86_64-linux-gnu/libc.so | grep tokenize`
5. Review existing wrapper.rs - it has all the patterns you need

---

**Status**: Infrastructure is complete. Focus on C++ shim implementation.
