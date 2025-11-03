# BitNet.cpp FFI Integration Sockets - Technical Specification

**Component**: BitNet.cpp FFI Integration Layer
**Location**: `crossval/src/bitnet_cpp_wrapper.cc`, `crossval/src/cpp_bindings.rs`
**Dependencies**: BitNet.cpp runtime libraries, dlopen(3), symbol resolution
**Version**: 1.0.0
**Date**: 2025-10-25

## Executive Summary

This specification defines six missing FFI sockets required to complete BitNet.cpp integration with a dual-mode architecture supporting both static linking (compile-time headers) and dynamic loading (runtime dlopen). The design maintains stable Rust ABI while accommodating C++ API evolution, provides graceful degradation when libraries are unavailable, and enables persistent context management for 10-100× performance improvements over current per-call model loading.

### Current State (MVP)

- **Working**: Two-pass tokenization and evaluation via llama.cpp API
- **Limitation**: Per-call model reload (100-500ms overhead per call)
- **Mode**: Static linking only (STUB/AVAILABLE conditional compilation)
- **Status**: Production-ready for MVP with reference implementations commented

### Missing Sockets (This Spec)

1. **Context Initialization** - Persistent model/context handles (v0.2 critical)
2. **BitNet Tokenization** - Optional BitNet-native tokenizer (v0.2 optimization)
3. **BitNet Inference** - 1-bit optimized inference kernels (v0.2 high priority)
4. **Session API** - Session lifecycle management (v0.2 critical for 10-100× speedup)
5. **GPU Support** - GPU-accelerated inference (v0.3)
6. **Capability Detection** - Runtime feature discovery (v0.3)

### Key Design Principles

- **Stable Rust ABI**: No changes to public Rust FFI API when C++ evolves
- **Graceful Degradation**: Missing symbols don't cause linker errors
- **Two-Pass Pattern**: All buffer-returning functions use size query → fill
- **Error Transparency**: Actionable error messages guide user to fix root cause
- **Performance**: Session API eliminates per-call overhead

---

## 1. Socket Definitions

### Socket 1: Context Initialization (Persistent Model Loading)

**Purpose**: Eliminate per-call model reload overhead (currently 100-500ms).

**Priority**: v0.2 critical (performance bottleneck)

**Expected Performance Impact**: 10-100× speedup for multi-call workflows

#### C Function Signature

```c
// bitnet.h or bitnet_cpp.h (expected location)

/// Opaque context handle for BitNet.cpp
typedef struct bitnet_context_t bitnet_context_t;

/// Initialize persistent BitNet context
///
/// Two-pass pattern NOT applicable (single call returns handle).
///
/// Args:
///   out_ctx: [out] Opaque context handle (caller frees with bitnet_cpp_free_context)
///   model_path: Path to GGUF model file
///   n_ctx: Context size for inference (e.g., 512, 2048)
///   n_gpu_layers: Number of layers to offload to GPU (0=CPU-only)
///   err: Error message buffer (512 bytes recommended)
///   err_len: Capacity of error buffer
///
/// Returns: 0 on success, -1 on error
int bitnet_cpp_init_context(
    bitnet_context_t** out_ctx,
    const char* model_path,
    int32_t n_ctx,
    int32_t n_gpu_layers,
    char* err,
    int32_t err_len
);

/// Free BitNet context (releases model and context)
///
/// Args:
///   ctx: Context handle to free (NULL-safe)
///
/// Returns: 0 on success, -1 on error
int bitnet_cpp_free_context(
    bitnet_context_t* ctx
);
```

#### Rust FFI Declaration

```rust
// crossval/src/cpp_bindings.rs

#[cfg(feature = "ffi")]
mod imp {
    use std::ffi::{c_char, c_int, c_void};

    // Opaque context handle (matches C typedef)
    #[repr(C)]
    pub struct BitnetContext {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        /// Initialize persistent BitNet context
        fn bitnet_cpp_init_context(
            out_ctx: *mut *mut BitnetContext,
            model_path: *const c_char,
            n_ctx: i32,
            n_gpu_layers: i32,
            err: *mut c_char,
            err_len: i32,
        ) -> c_int;

        /// Free BitNet context
        fn bitnet_cpp_free_context(
            ctx: *mut BitnetContext
        ) -> c_int;
    }
}
```

#### Safe Rust Wrapper

```rust
// crossval/src/cpp_bindings.rs

#[cfg(feature = "ffi")]
pub struct BitnetSession {
    ctx: *mut BitnetContext,
    model_path: PathBuf,
    n_ctx: i32,
}

#[cfg(feature = "ffi")]
impl BitnetSession {
    /// Create persistent session (replaces per-call model load)
    pub fn create(
        model_path: &Path,
        n_ctx: i32,
        n_gpu_layers: i32,
    ) -> Result<Self> {
        // Early availability check
        if !matches!(option_env!("CROSSVAL_HAS_BITNET"), Some("true")) {
            return Err(CrossvalError::CppNotAvailable);
        }

        let model_path_c = CString::new(model_path.to_str().unwrap())?;
        let mut err_buf = vec![0u8; 512];
        let mut ctx_ptr: *mut BitnetContext = std::ptr::null_mut();

        let result = unsafe {
            bitnet_cpp_init_context(
                &mut ctx_ptr,
                model_path_c.as_ptr(),
                n_ctx,
                n_gpu_layers,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        if result != 0 {
            let error_msg = extract_error_message(&err_buf);
            return Err(CrossvalError::InferenceError(error_msg));
        }

        if ctx_ptr.is_null() {
            return Err(CrossvalError::InferenceError(
                "bitnet_cpp_init_context returned null context".into()
            ));
        }

        Ok(Self {
            ctx: ctx_ptr,
            model_path: model_path.to_path_buf(),
            n_ctx,
        })
    }

    /// Tokenize using persistent session (Socket 2 integration)
    pub fn tokenize(&self, prompt: &str) -> Result<Vec<i32>> {
        // Delegates to Socket 2 if available, otherwise llama.cpp fallback
        todo!("Integrate Socket 2 tokenization")
    }

    /// Evaluate tokens using persistent session (Socket 3 integration)
    pub fn evaluate(&self, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
        // Delegates to Socket 3 if available, otherwise llama.cpp fallback
        todo!("Integrate Socket 3 evaluation")
    }
}

#[cfg(feature = "ffi")]
impl Drop for BitnetSession {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                let _ = bitnet_cpp_free_context(self.ctx);
            }
            self.ctx = std::ptr::null_mut();
        }
    }
}
```

---

### Socket 2: BitNet-Specific Tokenization (Optional)

**Purpose**: Use BitNet-native tokenizer if optimized differently than llama.cpp.

**Priority**: v0.2 optional (fallback to llama.cpp if unavailable)

**Fallback Strategy**: If symbol not found via dlopen, use existing `crossval_bitnet_tokenize` (llama.cpp-based)

#### C Function Signature

```c
// bitnet.h (expected location, optional)

/// Tokenize text using BitNet-native tokenizer
///
/// Two-pass pattern:
/// 1. Call with out_tokens=NULL to query size -> fills out_len, returns 0
/// 2. Call with out_tokens buffer -> fills tokens up to out_capacity, returns 0
///
/// Args:
///   ctx: BitNet context handle from bitnet_cpp_init_context
///   prompt: Input text to tokenize
///   add_bos: Whether to add BOS token (1=yes, 0=no)
///   parse_special: Whether to parse special tokens (1=yes, 0=no)
///   out_tokens: Output buffer for token IDs (NULL for size query)
///   out_capacity: Capacity of out_tokens buffer (ignored if out_tokens=NULL)
///   out_len: [out] Number of tokens produced
///   err: Error message buffer
///   err_len: Capacity of error buffer
///
/// Returns: 0 on success, -1 on error
int bitnet_cpp_tokenize_with_context(
    const bitnet_context_t* ctx,
    const char* prompt,
    int add_bos,
    int parse_special,
    int32_t* out_tokens,
    int32_t out_capacity,
    int32_t* out_len,
    char* err,
    int32_t err_len
);
```

#### Rust FFI Declaration

```rust
#[cfg(feature = "ffi")]
unsafe extern "C" {
    /// BitNet-specific tokenization (optional, may not exist)
    fn bitnet_cpp_tokenize_with_context(
        ctx: *const BitnetContext,
        prompt: *const c_char,
        add_bos: c_int,
        parse_special: c_int,
        out_tokens: *mut i32,
        out_capacity: i32,
        out_len: *mut i32,
        err: *mut c_char,
        err_len: i32,
    ) -> c_int;
}
```

#### Fallback Strategy

```rust
impl BitnetSession {
    pub fn tokenize(&self, prompt: &str) -> Result<Vec<i32>> {
        // Try Socket 2 (BitNet-native) first
        if let Some(tokenize_fn) = dlopen_loader::try_resolve_symbol(
            "bitnet_cpp_tokenize_with_context"
        ) {
            return self.tokenize_via_bitnet_native(prompt, tokenize_fn);
        }

        // Fallback: Use existing llama.cpp-based tokenization
        // (current MVP implementation in crossval_bitnet_tokenize)
        warn!("bitnet_cpp_tokenize_with_context not found, falling back to llama.cpp tokenizer");
        self.tokenize_via_llama_fallback(prompt)
    }

    fn tokenize_via_bitnet_native(
        &self,
        prompt: &str,
        tokenize_fn: Symbol<BitnetTokenizeFn>,
    ) -> Result<Vec<i32>> {
        // Two-pass pattern with Socket 2
        todo!("Implement BitNet-native tokenization")
    }

    fn tokenize_via_llama_fallback(&self, prompt: &str) -> Result<Vec<i32>> {
        // Delegate to existing crossval_bitnet_tokenize (MVP implementation)
        tokenize_bitnet(
            &self.model_path,
            prompt,
            true,  // add_bos
            false, // parse_special
        )
    }
}
```

---

### Socket 3: BitNet-Specific Inference (1-bit Optimized)

**Purpose**: Use BitNet-optimized 1-bit quantization kernels instead of generic llama.cpp.

**Priority**: v0.2 high (enables BitNet-specific optimizations)

**Fallback Strategy**: If symbol not found, use existing `crossval_bitnet_eval_with_tokens` (llama.cpp-based)

#### C Function Signature

```c
// bitnet.h (expected location)

/// Evaluate tokens using BitNet-optimized 1-bit kernels
///
/// Two-pass pattern:
/// 1. Call with out_logits=NULL to query shape -> fills out_rows/out_cols, returns 0
/// 2. Call with out_logits buffer -> fills logits up to logits_capacity, returns 0
///
/// Args:
///   ctx: BitNet context handle
///   tokens: Input token IDs
///   n_tokens: Number of tokens
///   seq_id: Sequence ID for batch processing (0 for single sequence)
///   out_logits: Output buffer for logits (NULL for size query)
///   logits_capacity: Capacity of out_logits buffer in floats
///   out_rows: [out] Number of rows (positions) in logits
///   out_cols: [out] Number of columns (vocab size) in logits
///   err: Error message buffer
///   err_len: Capacity of error buffer
///
/// Returns: 0 on success, -1 on error
int bitnet_cpp_eval_with_context(
    const bitnet_context_t* ctx,
    const int32_t* tokens,
    int32_t n_tokens,
    int32_t seq_id,
    float* out_logits,
    int32_t logits_capacity,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
);
```

#### Rust FFI Declaration

```rust
#[cfg(feature = "ffi")]
unsafe extern "C" {
    /// BitNet-specific inference (optional, may not exist)
    fn bitnet_cpp_eval_with_context(
        ctx: *const BitnetContext,
        tokens: *const i32,
        n_tokens: i32,
        seq_id: i32,
        out_logits: *mut f32,
        logits_capacity: i32,
        out_rows: *mut i32,
        out_cols: *mut i32,
        err: *mut c_char,
        err_len: i32,
    ) -> c_int;
}
```

#### Fallback Strategy

```rust
impl BitnetSession {
    pub fn evaluate(&self, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
        // Try Socket 3 (BitNet-native) first
        if let Some(eval_fn) = dlopen_loader::try_resolve_symbol(
            "bitnet_cpp_eval_with_context"
        ) {
            return self.evaluate_via_bitnet_native(tokens, eval_fn);
        }

        // Fallback: Use existing llama.cpp-based evaluation
        warn!("bitnet_cpp_eval_with_context not found, falling back to llama.cpp eval");
        self.evaluate_via_llama_fallback(tokens)
    }

    fn evaluate_via_bitnet_native(
        &self,
        tokens: &[i32],
        eval_fn: Symbol<BitnetEvalFn>,
    ) -> Result<Vec<Vec<f32>>> {
        // Two-pass pattern with Socket 3
        todo!("Implement BitNet-native evaluation")
    }

    fn evaluate_via_llama_fallback(&self, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
        // Delegate to existing crossval_bitnet_eval_with_tokens (MVP)
        eval_bitnet(
            &self.model_path,
            tokens,
            self.n_ctx,
        )
    }
}
```

---

### Socket 4: Session API (High-Level Lifecycle Management)

**Purpose**: Higher-level session management wrapping Sockets 1-3.

**Priority**: v0.2 critical (replaces per-call model loading)

**Note**: This is an alternative to Socket 1 if BitNet.cpp provides a complete session API instead of low-level context handles.

#### C Function Signature

```c
// bitnet.h (alternative to Socket 1)

/// Opaque session handle
typedef struct bitnet_session_t bitnet_session_t;

/// Create BitNet session with integrated model/context/tokenizer
///
/// Args:
///   out_session: [out] Session handle (caller frees with bitnet_cpp_session_free)
///   model_path: Path to GGUF model file
///   tokenizer_path: Path to tokenizer file (NULL=auto-discover from model)
///   n_ctx: Context size
///   n_gpu_layers: GPU layers (0=CPU-only)
///   err: Error message buffer
///   err_len: Capacity of error buffer
///
/// Returns: 0 on success, -1 on error
int bitnet_cpp_session_create(
    bitnet_session_t** out_session,
    const char* model_path,
    const char* tokenizer_path,
    int32_t n_ctx,
    int32_t n_gpu_layers,
    char* err,
    int32_t err_len
);

/// Free BitNet session
int bitnet_cpp_session_free(
    bitnet_session_t* session
);

/// Tokenize using session (integrated tokenizer)
int bitnet_cpp_session_tokenize(
    const bitnet_session_t* session,
    const char* prompt,
    int add_bos,
    int parse_special,
    int32_t* out_tokens,
    int32_t out_capacity,
    int32_t* out_len,
    char* err,
    int32_t err_len
);

/// Evaluate tokens using session
int bitnet_cpp_session_eval(
    const bitnet_session_t* session,
    const int32_t* tokens,
    int32_t n_tokens,
    float* out_logits,
    int32_t logits_capacity,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
);
```

**Decision Point**: Use Socket 4 if BitNet.cpp provides session API, otherwise use Socket 1+2+3.

---

### Socket 5: GPU Support (v0.3)

**Purpose**: GPU-accelerated inference with layer offloading.

**Priority**: v0.3 (post-MVP performance optimization)

#### C Function Signature

```c
// bitnet.h (v0.3)

/// Evaluate tokens with GPU acceleration
///
/// Same signature as bitnet_cpp_eval_with_context but uses GPU kernels
/// for layers specified by n_gpu_layers in context initialization.
///
/// Returns: 0 on success, -1 on error
int bitnet_cpp_eval_gpu(
    const bitnet_context_t* ctx,
    const int32_t* tokens,
    int32_t n_tokens,
    float* out_logits,
    int32_t logits_capacity,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
);
```

---

### Socket 6: Capability Detection (v0.3)

**Purpose**: Runtime feature detection for optimal kernel selection.

**Priority**: v0.3 (enables runtime optimization)

#### C Function Signature

```c
// bitnet.h (v0.3)

/// BitNet.cpp runtime capabilities
typedef struct {
    int has_avx2;       // x86 AVX2 SIMD
    int has_avx512;     // x86 AVX-512 SIMD
    int has_neon;       // ARM NEON SIMD
    int has_cuda;       // NVIDIA CUDA GPU
    int has_metal;      // Apple Metal GPU
    int has_hip;        // AMD ROCm GPU
} bitnet_capabilities_t;

/// Get BitNet.cpp runtime capabilities
///
/// Args:
///   out_caps: [out] Capabilities structure
///
/// Returns: 0 on success, -1 on error
int bitnet_cpp_get_capabilities(
    bitnet_capabilities_t* out_caps
);
```

#### Rust FFI Declaration

```rust
#[repr(C)]
pub struct BitnetCapabilities {
    pub has_avx2: i32,
    pub has_avx512: i32,
    pub has_neon: i32,
    pub has_cuda: i32,
    pub has_metal: i32,
    pub has_hip: i32,
}

#[cfg(feature = "ffi")]
unsafe extern "C" {
    fn bitnet_cpp_get_capabilities(
        out_caps: *mut BitnetCapabilities
    ) -> c_int;
}
```

---

## 2. dlopen Loader Architecture

### Design Goals

- **Graceful Symbol Resolution**: Missing symbols don't cause linker errors
- **Fallback to llama.cpp**: If BitNet-specific symbols unavailable, use generic llama.cpp
- **Runtime Reconfiguration**: Library path changes don't require recompile
- **Diagnostic Transparency**: Clear error messages guide user to fix root cause

### Symbol Resolution Strategy

```rust
// crossval/src/dlopen_loader.rs (new module)

use libloading::{Library, Symbol};
use std::path::PathBuf;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref BITNET_LIB: Mutex<Option<Library>> = Mutex::new(None);
}

pub struct DlopenLoader {
    library_path: PathBuf,
}

impl DlopenLoader {
    /// Load BitNet.cpp library at runtime
    pub fn load(library_path: &Path) -> Result<Self> {
        let lib = unsafe { Library::new(library_path)? };

        // Validate minimum required symbols
        let required_symbols = [
            "bitnet_cpp_init_context",
            "bitnet_cpp_free_context",
        ];

        for sym in &required_symbols {
            if unsafe { lib.get::<Symbol<extern "C" fn()>>(sym.as_bytes()) }.is_err() {
                return Err(CrossvalError::InferenceError(
                    format!("Required symbol '{}' not found in {}", sym, library_path.display())
                ));
            }
        }

        // Store library handle
        *BITNET_LIB.lock().unwrap() = Some(lib);

        Ok(Self {
            library_path: library_path.to_path_buf(),
        })
    }

    /// Try to resolve optional symbol (returns None if not found)
    pub fn try_resolve_symbol<T>(symbol_name: &str) -> Option<Symbol<T>> {
        let lib_guard = BITNET_LIB.lock().unwrap();
        let lib = lib_guard.as_ref()?;

        unsafe {
            lib.get::<Symbol<T>>(symbol_name.as_bytes()).ok()
        }
    }

    /// Resolve required symbol (errors if not found)
    pub fn resolve_symbol<T>(symbol_name: &str) -> Result<Symbol<T>> {
        Self::try_resolve_symbol(symbol_name)
            .ok_or_else(|| CrossvalError::InferenceError(
                format!("Required symbol '{}' not found", symbol_name)
            ))
    }
}
```

### Library Discovery at Runtime

```rust
impl DlopenLoader {
    /// Discover BitNet.cpp library using environment variables
    pub fn discover() -> Result<Self> {
        // Priority 1: Explicit BITNET_CPP_LIBDIR
        if let Some(lib_dir) = env::var("BITNET_CPP_LIBDIR").ok() {
            return Self::try_load_from_dir(&PathBuf::from(lib_dir));
        }

        // Priority 2: BITNET_CPP_DIR/build/bin
        if let Some(cpp_dir) = env::var("BITNET_CPP_DIR").ok() {
            let build_bin = PathBuf::from(cpp_dir).join("build/bin");
            if let Ok(loader) = Self::try_load_from_dir(&build_bin) {
                return Ok(loader);
            }
        }

        // Priority 3: System library paths
        #[cfg(target_os = "linux")]
        let lib_name = "libbitnet.so";
        #[cfg(target_os = "macos")]
        let lib_name = "libbitnet.dylib";
        #[cfg(target_os = "windows")]
        let lib_name = "bitnet.dll";

        Self::load(&PathBuf::from(lib_name))
    }

    fn try_load_from_dir(dir: &Path) -> Result<Self> {
        // Platform-specific library extensions
        let extensions = if cfg!(target_os = "linux") {
            vec!["so"]
        } else if cfg!(target_os = "macos") {
            vec!["dylib"]
        } else {
            vec!["dll"]
        };

        for ext in extensions {
            let lib_path = dir.join(format!("libbitnet.{}", ext));
            if lib_path.exists() {
                return Self::load(&lib_path);
            }
        }

        Err(CrossvalError::InferenceError(
            format!("No BitNet library found in {}", dir.display())
        ))
    }
}
```

### Fallback Hierarchy

```
Symbol Resolution Fallback Chain:

1. Try BitNet-specific symbol (Socket 2, 3)
   └─ dlopen_loader::try_resolve_symbol("bitnet_cpp_*")

2. If NOT FOUND → Fallback to llama.cpp
   └─ Use existing crossval_bitnet_tokenize/eval (MVP implementation)

3. If llama.cpp ALSO unavailable → Return CppNotAvailable
   └─ Guided error: "Set BITNET_CPP_DIR to enable cross-validation"
```

Example usage:

```rust
impl BitnetSession {
    pub fn tokenize(&self, prompt: &str) -> Result<Vec<i32>> {
        // Level 1: Try BitNet-native (Socket 2)
        if let Some(sym) = dlopen_loader::try_resolve_symbol("bitnet_cpp_tokenize_with_context") {
            return self.tokenize_via_bitnet(sym, prompt);
        }

        // Level 2: Fallback to llama.cpp (MVP)
        if option_env!("CROSSVAL_HAS_LLAMA") == Some("true") {
            return tokenize_bitnet(&self.model_path, prompt, true, false);
        }

        // Level 3: No backend available
        Err(CrossvalError::CppNotAvailable)
    }
}
```

---

## 3. Backward Compatibility with STUB/AVAILABLE

### Current STUB/AVAILABLE Modes

The existing MVP uses mutually exclusive conditional compilation:

```cpp
#ifdef BITNET_STUB
    // Returns friendly errors when C++ unavailable
    snprintf(err, err_len, "STUB mode - BitNet.cpp not available");
    return -1;

#elif defined(BITNET_AVAILABLE)
    // Full implementation using llama.cpp API
    llama_model* model = llama_load_model_from_file(model_path, model_params);
    // ...
#endif
```

### Integration with dlopen Loader

The dlopen loader **extends** (not replaces) the STUB/AVAILABLE architecture:

```cpp
// crossval/src/bitnet_cpp_wrapper.cc (updated)

#ifdef BITNET_STUB
    // Mode 1: STUB - no C++ libraries at build time
    snprintf(err, err_len, "STUB mode - BitNet.cpp not available");
    return -1;

#elif defined(BITNET_AVAILABLE)
    // Mode 2: AVAILABLE - static linking to llama.cpp (MVP default)
    // Use existing implementations (commented reference code)
    llama_model* model = llama_load_model_from_file(model_path, model_params);
    // ...

#elif defined(BITNET_DLOPEN)
    // Mode 3: DLOPEN - runtime symbol resolution (v0.2)
    // Delegate to Rust-side dlopen loader
    return bitnet_dlopen_tokenize(model_path, prompt, ...);

#else
    #error "Must define BITNET_STUB, BITNET_AVAILABLE, or BITNET_DLOPEN"
#endif
```

### Rust-Side Coordination

```rust
// crossval/src/cpp_bindings.rs

#[cfg(feature = "ffi")]
mod imp {
    // Build-time mode detection
    const FFI_MODE: &str = env!("CROSSVAL_FFI_MODE");  // "stub", "available", "dlopen"

    pub fn tokenize_bitnet(
        model_path: &Path,
        prompt: &str,
        add_bos: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>> {
        match FFI_MODE {
            "stub" => {
                Err(CrossvalError::CppNotAvailable)
            }
            "available" => {
                // Static linking - call C wrapper directly
                tokenize_via_static_link(model_path, prompt, add_bos, parse_special)
            }
            "dlopen" => {
                // Dynamic loading - use dlopen loader
                tokenize_via_dlopen(model_path, prompt, add_bos, parse_special)
            }
            _ => unreachable!(),
        }
    }
}
```

---

## 4. Migration Path from Current TODOs

### Phase 1: Uncomment Reference Implementations (v0.1.1)

**Goal**: Enable existing llama.cpp-based tokenization/evaluation.

**Changes**:
- Uncomment lines 90-165 in `crossval_bitnet_tokenize` (tokenization reference)
- Uncomment lines 234-320 in `crossval_bitnet_eval_with_tokens` (evaluation reference)
- Test with `BITNET_AVAILABLE` mode against real models

**Validation**:
```bash
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
cargo test -p crossval --features ffi -- test_tokenize_bitnet_available
cargo test -p crossval --features ffi -- test_eval_bitnet_available
```

### Phase 2: Add Socket 1 (Context Initialization) (v0.2.0)

**Goal**: Persistent context to eliminate per-call model reload.

**Changes**:
1. Add `bitnet_cpp_init_context` and `bitnet_cpp_free_context` to C wrapper
2. Add Rust FFI declarations and safe `BitnetSession` wrapper
3. Refactor existing tokenize/eval to optionally use persistent context

**Performance Target**: 10-100× speedup for multi-call workflows

**Validation**:
```bash
# Benchmark: per-call vs persistent session
cargo bench --bench crossval_session_perf --features ffi
```

### Phase 3: Add Socket 2+3 (BitNet-Specific Tokenization/Inference) (v0.2.1)

**Goal**: Enable BitNet-native kernels if available, fallback to llama.cpp.

**Changes**:
1. Add `bitnet_cpp_tokenize_with_context` (optional symbol)
2. Add `bitnet_cpp_eval_with_context` (optional symbol)
3. Implement dlopen loader with fallback to llama.cpp

**Validation**:
```bash
# Test with BitNet-specific symbols
cargo test -p crossval --features ffi -- test_bitnet_native_tokenize

# Test fallback when symbols missing
BITNET_FORCE_LLAMA_FALLBACK=1 cargo test -p crossval --features ffi -- test_llama_fallback
```

### Phase 4: Add dlopen Loader (v0.2.2)

**Goal**: Runtime symbol resolution without recompile.

**Changes**:
1. Add `crossval/src/dlopen_loader.rs` module
2. Implement `DlopenLoader::discover()` and symbol resolution
3. Add `BITNET_DLOPEN` mode to build.rs

**Validation**:
```bash
# Test runtime library switching
export BITNET_CPP_LIBDIR=/path/to/bitnet.cpp/build/bin
cargo run -p crossval --features ffi -- --dlopen-test
```

### Phase 5: Add Socket 5+6 (GPU + Capabilities) (v0.3.0)

**Goal**: GPU acceleration and runtime feature detection.

**Changes**:
1. Add `bitnet_cpp_eval_gpu` for GPU-accelerated inference
2. Add `bitnet_cpp_get_capabilities` for feature detection
3. Implement kernel selection based on capabilities

**Validation**:
```bash
# GPU acceleration test
cargo test -p crossval --features ffi,gpu -- test_bitnet_gpu_eval

# Capability detection
cargo run -p crossval --features ffi -- --show-capabilities
```

---

## 5. Testing Strategy

### Unit Tests (Per Socket)

```rust
// crossval/tests/socket_tests.rs

#[cfg(feature = "ffi")]
mod socket1_context_init {
    use super::*;

    #[test]
    fn test_context_init_and_free() {
        let session = BitnetSession::create(
            Path::new("models/model.gguf"),
            512,  // n_ctx
            0,    // n_gpu_layers
        ).expect("context init failed");

        // Session auto-freed on drop
        drop(session);
    }

    #[test]
    fn test_context_init_invalid_model() {
        let result = BitnetSession::create(
            Path::new("nonexistent.gguf"),
            512,
            0,
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            CrossvalError::InferenceError(msg) => {
                assert!(msg.contains("Failed to load model"));
            }
            _ => panic!("Wrong error type"),
        }
    }
}

#[cfg(feature = "ffi")]
mod socket2_bitnet_tokenize {
    use super::*;

    #[test]
    fn test_bitnet_tokenize_with_fallback() {
        let session = BitnetSession::create(
            Path::new("models/model.gguf"),
            512,
            0,
        ).unwrap();

        let tokens = session.tokenize("Hello world").unwrap();

        assert!(!tokens.is_empty());
        assert_eq!(tokens[0], 1);  // BOS token
    }

    #[test]
    fn test_bitnet_tokenize_fallback_to_llama() {
        // Force llama.cpp fallback
        env::set_var("BITNET_FORCE_LLAMA_FALLBACK", "1");

        let session = BitnetSession::create(
            Path::new("models/model.gguf"),
            512,
            0,
        ).unwrap();

        let tokens = session.tokenize("Test").unwrap();
        assert!(!tokens.is_empty());

        env::remove_var("BITNET_FORCE_LLAMA_FALLBACK");
    }
}

#[cfg(feature = "ffi")]
mod socket3_bitnet_eval {
    use super::*;

    #[test]
    fn test_bitnet_eval_with_context() {
        let session = BitnetSession::create(
            Path::new("models/model.gguf"),
            512,
            0,
        ).unwrap();

        let tokens = vec![1, 4872, 338];  // "Hello world"
        let logits = session.evaluate(&tokens).unwrap();

        assert_eq!(logits.len(), tokens.len());
        assert!(!logits[0].is_empty());
    }
}
```

### Integration Tests (Cross-Socket)

```rust
// crossval/tests/session_integration.rs

#[cfg(feature = "ffi")]
mod session_integration {
    use super::*;

    #[test]
    fn test_full_session_lifecycle() {
        // Socket 1: Create session
        let session = BitnetSession::create(
            Path::new("models/model.gguf"),
            512,
            0,
        ).unwrap();

        // Socket 2: Tokenize
        let tokens = session.tokenize("What is 2+2?").unwrap();
        assert!(!tokens.is_empty());

        // Socket 3: Evaluate
        let logits = session.evaluate(&tokens).unwrap();
        assert_eq!(logits.len(), tokens.len());

        // Implicit: Socket 1 free on drop
    }

    #[test]
    fn test_multiple_inferences_with_session() {
        let session = BitnetSession::create(
            Path::new("models/model.gguf"),
            512,
            0,
        ).unwrap();

        // Multiple calls should NOT reload model
        for prompt in &["Test 1", "Test 2", "Test 3"] {
            let tokens = session.tokenize(prompt).unwrap();
            let logits = session.evaluate(&tokens).unwrap();
            assert_eq!(logits.len(), tokens.len());
        }
    }
}
```

### dlopen Loader Tests

```rust
// crossval/tests/dlopen_loader_tests.rs

#[cfg(feature = "ffi")]
mod dlopen_tests {
    use super::*;

    #[test]
    fn test_dlopen_discover() {
        let loader = DlopenLoader::discover().unwrap();
        assert!(loader.library_path.exists());
    }

    #[test]
    fn test_symbol_resolution() {
        let loader = DlopenLoader::discover().unwrap();

        // Required symbols should resolve
        let init_fn = loader.resolve_symbol::<InitContextFn>(
            "bitnet_cpp_init_context"
        ).unwrap();

        assert!(!init_fn.is_null());
    }

    #[test]
    fn test_optional_symbol_fallback() {
        let loader = DlopenLoader::discover().unwrap();

        // Optional symbol may not exist
        let opt_sym = loader.try_resolve_symbol::<TokenizeFn>(
            "bitnet_cpp_tokenize_with_context"
        );

        // Should return None without panic if symbol missing
        if opt_sym.is_none() {
            println!("Optional symbol not found, fallback will be used");
        }
    }
}
```

### Performance Benchmarks

```rust
// crossval/benches/session_perf.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_per_call_vs_session(c: &mut Criterion) {
    let model_path = Path::new("models/model.gguf");

    c.bench_function("per_call_tokenize", |b| {
        b.iter(|| {
            // Per-call: loads model every time (current MVP)
            tokenize_bitnet(
                model_path,
                black_box("Hello world"),
                true,
                false,
            ).unwrap()
        });
    });

    c.bench_function("session_tokenize", |b| {
        let session = BitnetSession::create(model_path, 512, 0).unwrap();

        b.iter(|| {
            // Session: reuses loaded model (Socket 1)
            session.tokenize(black_box("Hello world")).unwrap()
        });
    });
}

criterion_group!(benches, bench_per_call_vs_session);
criterion_main!(benches);
```

**Expected Results**:
- Per-call tokenize: ~100-500ms per call
- Session tokenize: ~1-10ms per call (10-100× speedup)

---

## 6. Error Handling & Diagnostics

### Error Taxonomy

| Error Class | Cause | User Action |
|-------------|-------|-------------|
| `CppNotAvailable` | C++ libraries not compiled | Set `BITNET_CPP_DIR`, rebuild with `--features ffi` |
| `LibraryNotFound` | dlopen can't find libbitnet.so | Set `LD_LIBRARY_PATH` or `BITNET_CPP_LIBDIR` |
| `SymbolNotFound` | Required symbol missing | Verify BitNet.cpp version, rebuild |
| `ModelLoadError` | Invalid GGUF file | Check model path, validate with `compat-check` |
| `InferenceError` | C++ inference failed | Check error message, enable `--verbose` |

### Actionable Error Messages

```rust
impl BitnetSession {
    pub fn create(
        model_path: &Path,
        n_ctx: i32,
        n_gpu_layers: i32,
    ) -> Result<Self> {
        // Error 1: C++ not available at build time
        if !matches!(option_env!("CROSSVAL_HAS_BITNET"), Some("true")) {
            return Err(CrossvalError::CppNotAvailable.with_context(
                "BitNet.cpp not available. To enable:\n\
                 1. Set BITNET_CPP_DIR=/path/to/bitnet.cpp\n\
                 2. Rebuild: cargo build -p crossval --features ffi\n\
                 3. Run: cargo test -p crossval --features ffi"
            ));
        }

        // Error 2: Library not found at runtime
        let loader = DlopenLoader::discover().map_err(|e| {
            CrossvalError::LibraryNotFound(format!(
                "libbitnet.so not found. To fix:\n\
                 1. Set LD_LIBRARY_PATH=$BITNET_CPP_DIR/build/bin\n\
                 2. Or set BITNET_CPP_LIBDIR=/path/to/libs\n\
                 Original error: {}", e
            ))
        })?;

        // Error 3: Symbol not found
        let init_fn = loader.resolve_symbol("bitnet_cpp_init_context").map_err(|e| {
            CrossvalError::SymbolNotFound(format!(
                "Symbol 'bitnet_cpp_init_context' not found in libbitnet.so.\n\
                 This may indicate version mismatch. To fix:\n\
                 1. Verify BitNet.cpp version: git -C $BITNET_CPP_DIR log -1 --oneline\n\
                 2. Rebuild BitNet.cpp: cd $BITNET_CPP_DIR && cmake --build build\n\
                 Original error: {}", e
            ))
        })?;

        // ... rest of implementation
    }
}
```

### Diagnostic Flags

```bash
# Show library resolution diagnostics
cargo run -p crossval --features ffi -- --dlopen-diagnostics

# Example output:
# Library Discovery:
#   BITNET_CPP_DIR: /home/user/.cache/bitnet_cpp
#   BITNET_CPP_LIBDIR: (not set)
#   LD_LIBRARY_PATH: /home/user/.cache/bitnet_cpp/build/bin
#
# Library Found: /home/user/.cache/bitnet_cpp/build/bin/libbitnet.so
#
# Symbol Resolution:
#   ✓ bitnet_cpp_init_context (required)
#   ✓ bitnet_cpp_free_context (required)
#   ✓ bitnet_cpp_tokenize_with_context (optional)
#   ✗ bitnet_cpp_eval_with_context (optional, fallback to llama.cpp)
#   ✗ bitnet_cpp_eval_gpu (optional, not available)
#   ✗ bitnet_cpp_get_capabilities (optional, not available)
```

---

## 7. Performance Specifications

### Throughput Targets

| Operation | Current MVP (per-call) | Target (session API) | Speedup |
|-----------|------------------------|----------------------|---------|
| Tokenization (100 tokens) | ~200ms | ~5ms | 40× |
| Evaluation (1 token) | ~300ms | ~10ms | 30× |
| Evaluation (4 tokens) | ~500ms | ~40ms | 12× |
| Full inference (32 tokens) | ~5s | ~300ms | 16× |

### Memory Overhead

- **Per-call mode**: Model loaded/unloaded per call (no persistent overhead)
- **Session mode**: ~600 MB persistent (model + context + vocab)
- **Trade-off**: Memory for speed (acceptable for multi-call workflows)

### Validation Commands

```bash
# Benchmark per-call vs session
cargo bench --bench crossval_session_perf --features ffi

# Profile memory usage
valgrind --tool=massif \
  cargo run -p crossval --features ffi -- \
  --benchmark-session

# Check session overhead
cargo run -p crossval --features ffi -- \
  --profile-session \
  --prompts "Test 1" "Test 2" "Test 3"
```

---

## 8. Success Criteria

### Functional Requirements

- [ ] Socket 1 (Context Init): `BitnetSession::create()` succeeds with valid model
- [ ] Socket 1 (Context Free): Session properly freed on drop (no memory leaks)
- [ ] Socket 2 (BitNet Tokenize): Tokenization works with BitNet-native or llama.cpp fallback
- [ ] Socket 3 (BitNet Eval): Evaluation works with BitNet-native or llama.cpp fallback
- [ ] dlopen Loader: Successfully discovers and loads libraries at runtime
- [ ] Symbol Resolution: Required symbols resolved, optional symbols gracefully fallback
- [ ] Error Handling: Actionable error messages guide user to fix root cause

### Performance Requirements

- [ ] Session API provides ≥10× speedup vs per-call (benchmark validation)
- [ ] Memory overhead ≤700 MB for 2B model with persistent session
- [ ] Symbol resolution overhead ≤5ms per session creation

### Test Coverage

- [ ] Unit tests: All 6 sockets individually tested
- [ ] Integration tests: Cross-socket workflows (tokenize → eval in session)
- [ ] Error path tests: Missing library, missing symbol, invalid model
- [ ] Fallback tests: BitNet-native → llama.cpp fallback chain
- [ ] Performance tests: Per-call vs session benchmarks

### Documentation

- [ ] Socket API reference: C signatures and Rust FFI declarations
- [ ] Migration guide: Per-call → session API refactoring
- [ ] Troubleshooting guide: Common dlopen errors and fixes
- [ ] Example usage: End-to-end session workflow

---

## 9. Risk Mitigation

### Risk: C++ API Changes

**Scenario**: BitNet.cpp changes function signatures between versions.

**Mitigation**:
1. Version detection via dlopen symbol lookup
2. Fallback to llama.cpp if BitNet-specific symbols unavailable
3. Stable Rust ABI (no public API changes when C++ evolves)

**Validation**:
```rust
// Check BitNet.cpp version before calling
if let Some(version_fn) = loader.try_resolve_symbol("bitnet_cpp_version") {
    let version = unsafe { version_fn() };
    if version < MIN_REQUIRED_VERSION {
        warn!("BitNet.cpp version {} < required {}, using fallback",
              version, MIN_REQUIRED_VERSION);
        return self.tokenize_via_llama_fallback(prompt);
    }
}
```

### Risk: Missing Symbols at Runtime

**Scenario**: User builds with old BitNet.cpp that lacks new symbols.

**Mitigation**:
1. Required symbols (Socket 1) validated at session creation
2. Optional symbols (Socket 2, 3) gracefully fallback to llama.cpp
3. Diagnostic flag shows which symbols available

**Validation**:
```bash
# Show symbol availability
cargo run -p crossval --features ffi -- --show-symbols

# Output:
# Required Symbols:
#   ✓ bitnet_cpp_init_context
#   ✓ bitnet_cpp_free_context
#
# Optional Symbols:
#   ✗ bitnet_cpp_tokenize_with_context (using llama.cpp fallback)
#   ✗ bitnet_cpp_eval_with_context (using llama.cpp fallback)
```

### Risk: Memory Leaks in Session Management

**Scenario**: Sessions not properly freed, leading to memory accumulation.

**Mitigation**:
1. RAII pattern via `Drop` trait (automatic cleanup)
2. Valgrind integration in CI
3. Memory leak tests in test suite

**Validation**:
```bash
# CI memory leak check
valgrind --leak-check=full --error-exitcode=1 \
  cargo test -p crossval --features ffi -- test_session_lifecycle
```

### Risk: Performance Regression

**Scenario**: Session API doesn't improve performance as expected.

**Mitigation**:
1. Performance benchmarks in CI (fail if <10× improvement)
2. Profiling tools integration
3. Fallback to per-call if session overhead too high

**Validation**:
```bash
# CI performance gate
cargo bench --bench crossval_session_perf --features ffi -- \
  --baseline per_call --fail-if-slower 10x
```

---

## 10. Related Documentation

- [`docs/specs/bitnet-session-api.md`](bitnet-session-api.md) - High-level session API design
- [`docs/specs/bitnet-available-wiring.md`](bitnet-available-wiring.md) - AVAILABLE mode implementation details
- [`docs/explanation/dual-backend-crossval.md`](../explanation/dual-backend-crossval.md) - Dual-backend architecture
- [`docs/howto/cpp-setup.md`](../howto/cpp-setup.md) - C++ reference setup guide
- [`CLAUDE.md`](../../CLAUDE.md) - Cross-validation quick reference

---

## 11. Glossary

- **Socket**: FFI function signature pair (C declaration + Rust binding)
- **Two-Pass Pattern**: Buffer negotiation via NULL pointer size query followed by actual fill
- **dlopen**: Dynamic library loading API (POSIX standard)
- **Symbol Resolution**: Mapping function name to runtime address
- **Fallback Chain**: Ordered list of alternatives when preferred option unavailable
- **Session**: Persistent context holding model, tokenizer, and inference state
- **STUB Mode**: Compile mode with no C++ dependencies (returns errors)
- **AVAILABLE Mode**: Compile mode with static linking to C++ libraries
- **DLOPEN Mode**: Compile mode with runtime dynamic loading

---

## Appendix A: Function Type Aliases

```rust
// Type aliases for dlopen symbol resolution

type InitContextFn = unsafe extern "C" fn(
    out_ctx: *mut *mut BitnetContext,
    model_path: *const c_char,
    n_ctx: i32,
    n_gpu_layers: i32,
    err: *mut c_char,
    err_len: i32,
) -> c_int;

type FreeContextFn = unsafe extern "C" fn(
    ctx: *mut BitnetContext
) -> c_int;

type TokenizeFn = unsafe extern "C" fn(
    ctx: *const BitnetContext,
    prompt: *const c_char,
    add_bos: c_int,
    parse_special: c_int,
    out_tokens: *mut i32,
    out_capacity: i32,
    out_len: *mut i32,
    err: *mut c_char,
    err_len: i32,
) -> c_int;

type EvalFn = unsafe extern "C" fn(
    ctx: *const BitnetContext,
    tokens: *const i32,
    n_tokens: i32,
    seq_id: i32,
    out_logits: *mut f32,
    logits_capacity: i32,
    out_rows: *mut i32,
    out_cols: *mut i32,
    err: *mut c_char,
    err_len: i32,
) -> c_int;

type CapabilitiesFn = unsafe extern "C" fn(
    out_caps: *mut BitnetCapabilities
) -> c_int;
```

---

## Appendix B: Build Configuration Examples

### Build with Static Linking (AVAILABLE Mode)

```bash
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/bin:$LD_LIBRARY_PATH"

cargo build -p crossval --features ffi
cargo test -p crossval --features ffi
```

### Build with dlopen Mode (v0.2+)

```bash
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
export CROSSVAL_FFI_MODE=dlopen

cargo build -p crossval --features ffi,dlopen
cargo test -p crossval --features ffi,dlopen
```

### Build with STUB Mode (No C++)

```bash
# No environment variables needed
cargo build -p crossval
cargo test -p crossval
```

---

## Appendix C: Example End-to-End Usage

```rust
use crossval::BitnetSession;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create persistent session (Socket 1)
    let session = BitnetSession::create(
        Path::new("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"),
        512,  // n_ctx
        0,    // n_gpu_layers (CPU-only)
    )?;

    // Multiple inferences WITHOUT reloading model
    for prompt in &["What is 2+2?", "What is the capital of France?"] {
        // Socket 2: Tokenize (BitNet-native or llama.cpp fallback)
        let tokens = session.tokenize(prompt)?;
        println!("Tokens: {:?}", tokens);

        // Socket 3: Evaluate (BitNet-native or llama.cpp fallback)
        let logits = session.evaluate(&tokens)?;
        println!("Logits shape: {} positions × {} vocab", logits.len(), logits[0].len());
    }

    // Session auto-freed on drop (Socket 1 cleanup)
    Ok(())
}
```

**Expected Performance**:
- Session creation: ~500ms (one-time cost)
- Per-inference: ~10-40ms (10-100× faster than per-call reload)
- Total for 2 prompts: ~600ms vs ~10s without session API

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-10-25 | Initial specification |

---

**Status**: Draft for Review
**Reviewers**: BitNet.rs maintainers, C++ integration team
**Next Steps**: Review → Implement Socket 1 → Integrate Socket 2+3 → Add dlopen → GPU support
