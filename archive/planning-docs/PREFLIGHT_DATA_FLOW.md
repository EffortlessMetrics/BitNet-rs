# Preflight & Diagnostics Data Flow

## Build-Time Library Detection

### 1. Compilation Phase (`crossval/build.rs`)

```
┌─────────────────────────────────────┐
│ Check BITNET_CPP_DIR or HOME        │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ Scan multiple lib directories:      │
│ - $BITNET_CROSSVAL_LIBDIR          │
│ - $BITNET_CPP_DIR/build            │
│ - $BITNET_CPP_DIR/build/lib        │
│ - $HOME/.cache/bitnet_cpp/build    │
└──────────────┬──────────────────────┘
               ↓
    ┌──────────┴──────────┐
    ↓                     ↓
┌──────────────┐  ┌──────────────┐
│ Find libbitnet*  │  │ Find libllama* │
│ & libggml*   │  │ or libggml*  │
└──────┬───────┘  └──────┬───────┘
       ↓                 ↓
       found_bitnet=true found_llama=true
       ↓                 ↓
┌──────────────────────────────────┐
│ Emit cargo:rustc-env variables:  │
│                                  │
│ CROSSVAL_HAS_BITNET={true|false} │
│ CROSSVAL_HAS_LLAMA={true|false}  │
│                                  │
│ Also emit:                       │
│ - cargo:rustc-link-search       │
│ - cargo:rustc-link-lib          │
│ - cargo:rustc-cfg=have_cpp      │
└──────────┬───────────────────────┘
           ↓
    (Variables stored in binary)
```

## Runtime Flow: `crossval-per-token` Command

### 2. CLI Parsing Phase

```
User Input:
  cargo run -p xtask -- crossval-per-token \
    --model <path> \
    --tokenizer <path> \
    --prompt "..." \
    --cpp-backend bitnet \
    --verbose

       ↓
clap parsing
       ↓
Creates CrossvalPerToken struct:
{
  model: PathBuf,
  tokenizer: PathBuf,
  prompt: String,
  cpp_backend: Option<CppBackend>,  // Some(BitNet)
  verbose: bool,                      // true
  // ... other fields
}
       ↓
Dispatches to match expression (line ~960)
       ↓
crossval_per_token_cmd(
  model_path,
  tokenizer_path,
  prompt,
  ...,
  cpp_backend = Some(BitNet),
  verbose = true
)
```

### 3. Backend Determination Phase (Line 2991)

```
┌───────────────────────────────────┐
│ cpp_backend: Option<CppBackend>   │
└──────────┬────────────┬───────────┘
           │            │
    Some(x) provided   None (not specified)
      │                 │
      ↓                 ↓
    Use explicit      Auto-detect from path:
    backend            CppBackend::from_model_path()
                         ↓
                  model_path.contains("bitnet")?
                    ↓              ↓
                   YES             NO
                    │              │
                BitNet        Check "llama"?
                              ↓              ↓
                             YES             NO
                              │              │
                            Llama       Default:Llama
       ↓
┌──────────────────────────────┐
│ backend: CppBackend selected │
└────────────┬─────────────────┘
             ↓
       (proceed to preflight)
```

### 4. Preflight Validation Phase (Line 3002)

```
┌─────────────────────────────────────────────┐
│ preflight_backend_libs(backend, verbose)    │
└────────────┬────────────────────────────────┘
             ↓
       Read env var:
       option_env!("CROSSVAL_HAS_BITNET")  ← from build.rs
                    or
       option_env!("CROSSVAL_HAS_LLAMA")
             ↓
       ┌─────┴─────┐
       ↓           ↓
    true      false (not found at build time)
    ✓             │
    │             ↓
    │        ┌─────────────────────────┐
    │        │ Generate error message: │
    │        │ - Backend name          │
    │        │ - Setup command         │
    │        │ - Required libraries    │
    │        │ - Next steps            │
    │        │ - Exit with error       │
    │        └─────────────────────────┘
    │
    └─────→ Continue (silently if not verbose)
             ↓
       (proceed to template setup)
```

### 5. Diagnostic Output Phase

```
If verbose=true:
┌────────────────────────────────────┐
│ Print backend selection:           │
│ "Selected backend: bitnet.cpp      │
│  (explicit|auto-detected)"         │
│                                    │
│ Print template info:               │
│ "Using template: auto              │
│  Tokenization: add_bos=true"       │
└────────────────────────────────────┘

Always:
┌────────────────────────────────────┐
│ Print model info:                  │
│ "Model: path/to/model.gguf"        │
│ "Prompt: \"...\" "                 │
│ "Cosine tolerance: 0.999"          │
└────────────────────────────────────┘
```

### 6. Tokenization Phase

```
Rust Tokenization:
┌─────────────────────────────────────┐
│ Load tokenizer(tokenizer_path)      │
│ Encode(formatted_prompt, ...)       │
│ → Vec<u32>: token_ids              │
│                                     │
│ Optional (if --dump-ids):          │
│ eprintln!("Rust token IDs: {:?}") │
└──────────────┬────────────────────┘
               ↓
         Count tokens

C++ Tokenization (backend-specific):
┌─────────────────────────────────────┐
│ If BitNet backend:                  │
│  bitnet_crossval::tokenize_bitnet()│
│  → Vec<u32>: cpp_tokens            │
│                                     │
│ If LLaMA backend:                  │
│  Session::load() → tokenize()      │
│  → Vec<u32>: cpp_tokens            │
│                                     │
│ Optional (if --dump-cpp-ids):      │
│ eprintln!("C++ token IDs: {:?}")  │
└──────────────┬────────────────────┘
               ↓
         Count tokens
```

### 7. Token Parity Pre-Gate (FAIL-FAST) - Line 3101-3119

```
┌─────────────────────────────────────────────┐
│ validate_token_parity(rust_tokens,          │
│                       cpp_tokens,           │
│                       prompt,               │
│                       backend)              │
└────────────┬────────────────────────────────┘
             ↓
       ┌─────┴─────┐
       ↓           ↓
   Match      Mismatch
    │              │
    ✓              ↓
    │         find_first_diff()
    │              ↓
    │         Create TokenParityError:
    │         {
    │           rust_tokens,
    │           cpp_tokens,
    │           first_diff_index,
    │           prompt,
    │           backend  ← BACKEND CONTEXT!
    │         }
    │              ↓
    │         format_token_mismatch_error()
    │              ↓
    │         ┌─────────────────────────────────┐
    │         │ Print colored error header:     │
    │         │ "❌ Token Sequence Mismatch     │
    │         │  with C++ Backend: BitNet"     │
    │         │                                │
    │         │ Print both token sequences     │
    │         │ (first 64 tokens each)        │
    │         │                                │
    │         │ Highlight first difference:   │
    │         │ "First diff at index: 1        │
    │         │  Mismatch: Rust=128000,        │
    │         │  C++=1229"                     │
    │         │                                │
    │         │ Backend-specific help:        │
    │         │ (different for BitNet/LLaMA) │
    │         │                                │
    │         │ Example fix command            │
    │         │ (copy-pasteable)              │
    │         └─────────────────────────────────┘
    │              ↓
    │         Exit with code 2 (usage error)
    │
    └─→ Silent success, continue
         ↓
    (expensive logits evaluation next)
```

### 8. Logits Evaluation & Comparison

```
Rust Logits Evaluation:
┌────────────────────────────────────┐
│ eval_logits_all_positions(         │
│   model_path,                      │
│   token_ids)                       │
│ → Vec<Vec<f32>>: rust_logits      │
└────────────────────────────────────┘

C++ Logits Evaluation (backend-specific):
┌────────────────────────────────────┐
│ If BitNet:                         │
│  eval_bitnet(model_path, tokens) │
│                                    │
│ If LLaMA:                          │
│  Session::eval() →                │
│  get_all_logits()                 │
│ → Vec<Vec<f32>>: cpp_logits       │
└────────────────────────────────────┘

Comparison:
┌────────────────────────────────────┐
│ compare_per_position_logits(       │
│   rust_logits,                     │
│   cpp_logits)                      │
│ → DivergenceReport {              │
│     first_divergence_token,        │
│     per_token_cosine_sim,          │
│     per_token_l2_dist,             │
│     max_absolute_diff              │
│   }                                │
└────────────────────────────────────┘
```

### 9. Output Phase

```
┌─────────────────┐
│ format="text"?  │
└────────┬────────┘
         ├─YES──→ Text output
         │        ├─→ Per-token cosine sim & L2 dist
         │        ├─→ Max absolute diff
         │        └─→ Divergence details (if any)
         │
         └─NO──→ format="json"
                 └─→ JSON with all metrics
                    and status:"ok"|"diverged"
```

## Error Handling Paths

### Missing Libraries (Preflight Failure)

```
Preflight → Libraries not found
         ↓
    Display error:
    ┌────────────────────────────────────┐
    │ Backend 'bitnet.cpp' selected but  │
    │ required libraries not found.      │
    │                                    │
    │ Setup instructions:                │
    │ 1. Install C++ reference:          │
    │    eval "$(cargo run -p xtask -- │
    │    setup-cpp-auto --bitnet ...)"  │
    │ 2. Verify libraries:               │
    │    cargo run -p xtask --          │
    │    preflight --backend bitnet     │
    │ 3. Rebuild xtask                  │
    │                                    │
    │ Required libraries: ["libbitnet"]  │
    └────────────────────────────────────┘
         ↓
    Exit with error code 1
```

### Token Sequence Mismatch (FAIL-FAST)

```
Token parity validation → Mismatch detected
         ↓
    ┌─────────────────────────────────────┐
    │ ❌ Token Sequence Mismatch           │
    │ with C++ Backend: BitNet            │
    │ Fix BOS/template before logits      │
    │                                     │
    │ Rust tokens (4):                    │
    │   [128000, 128000, 1229, 374]      │
    │                                     │
    │ C++ tokens (3):                     │
    │   [128000, 1229, 374]              │
    │                                     │
    │ First diff at index: 1              │
    │ Mismatch: Rust token=128000,        │
    │           C++ token=1229            │
    │                                     │
    │ Troubleshooting for BitNet backend: │
    │  • Verify BitNet-compatible model  │
    │  • Check tokenizer path            │
    │  • Try --cpp-backend llama         │
    │  • Verify --prompt-template        │
    │  • Check BOS token with --dump-ids │
    │                                     │
    │ Example command:                    │
    │  cargo run -p xtask --             │
    │  crossval-per-token ...            │
    │  --prompt-template raw \           │
    │  --cpp-backend bitnet \            │
    │  --dump-ids --dump-cpp-ids         │
    └─────────────────────────────────────┘
         ↓
    Exit with code 2 (usage error - wrong backend?)
```

### Logits Divergence (After Validation)

```
Comparison → Divergence detected
         ↓
    Display divergence report:
    ┌───────────────────────────────────┐
    │ Per-token comparison:              │
    │                                   │
    │ ✓ t=0 cosine=0.9999 l2=0.0042   │
    │ ✓ t=1 cosine=0.9997 l2=0.0051   │
    │ ✗ t=2 cosine=0.9955 l2=0.0084   │
    │   ↑ First divergence at token 2   │
    │                                   │
    │ Max absolute diff: 2.3e-3         │
    │                                   │
    │ Next steps:                       │
    │ # 1. Capture Rust trace           │
    │ BITNET_TRACE_DIR=/tmp/rs ...      │
    │ cargo run -p bitnet-cli ...       │
    │                                   │
    │ # 2. Capture C++ trace            │
    │ BITNET_TRACE_DIR_CPP=/tmp/cpp ... │
    │                                   │
    │ # 3. Compare traces               │
    │ cargo run -p xtask --             │
    │ trace-diff /tmp/rs /tmp/cpp       │
    └───────────────────────────────────┘
         ↓
    Exit with code 1 (divergence found)
```

## Feature Flag Interactions

```
Build:
  cargo build -p crossval           → No FFI (stub mode)
  cargo build -p crossval           
    --features ffi                  → FFI wrappers only
  cargo build -p crossval
    --features crossval             → Full C++ integration
                                       (requires libs at build time)

Runtime:
  cfg(feature = "ffi")              → #[cfg] blocks for FFI
  cfg(all(feature = "ffi", have_cpp)) → Tests needing real C++
  option_env!("CROSSVAL_HAS_*")    → Build-time detection
```

## Summary

1. **Build-time**: crossval/build.rs detects libraries, emits CROSSVAL_HAS_*
2. **CLI parse**: clap creates command struct with all flags
3. **Backend determine**: Use explicit or auto-detect from path
4. **Preflight**: Check compile-time detection, fail fast with actionable error
5. **Tokenization**: Rust and C++ paths, optional token dumps
6. **Token parity**: FAIL-FAST validation with backend-aware error messages
7. **Logits eval**: Both backends compute logits (expensive)
8. **Comparison**: Find first divergence, output results
9. **Cleanup**: Exit with appropriate code (0=ok, 1=divergence, 2=token mismatch)
