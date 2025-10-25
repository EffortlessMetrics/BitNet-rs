# BitNet.rs FFI and Tracing Infrastructure Exploration Report

**Exploration Date**: 2025-10-24  
**Thoroughness**: Medium  
**Status**: Complete

---

## Executive Summary

BitNet.rs has a well-structured FFI interface for C++ cross-validation and a comprehensive trace infrastructure for debugging layer-by-layer divergence. The system supports:

- **C++ FFI Bridge**: Optional FFI bindings to the Microsoft BitNet C++ reference implementation
- **Trace Infrastructure**: Per-layer, per-token activation capture for root-cause analysis
- **Logits Comparison**: Per-position divergence detection with cosine similarity and L2 metrics
- **Weight Mapping**: Sophisticated handling of tied weights (embedding ↔ LM head), transposition, and quantization

---

## 1. FFI Interface Architecture

### 1.1 FFI Crates Structure

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/`

#### Primary FFI Crates:

- **`bitnet-ffi`** - Provides C API bindings for language interop
  - Location: `crates/bitnet-ffi/src/`
  - Modules:
    - `c_api.rs` - Core C functions with exact signature compatibility
    - `inference.rs` - Inference manager with thread-safe operations
    - `model.rs` - Model management
    - `streaming.rs` - Streaming inference support
    - `config.rs` - Configuration structures
    - `memory.rs` - Memory management
    - `threading.rs` - Thread pool management
    - `error.rs` - Error handling
    - `llama_compat.rs` - LLaMA compatibility layer

- **`bitnet-ggml-ffi`** - GGML-specific FFI bindings
  - Minimal wrapper around GGML functions

- **`crossval` crate** (main cross-validation framework)
  - Location: `crossval/src/`
  - Modules:
    - `cpp_bindings.rs` - Safe Rust wrappers around C++ BitNet implementation
    - `comparison.rs` - Cross-validation comparison logic
    - `logits_compare.rs` - Per-position logits comparison with divergence detection
    - `fixtures.rs` - Test fixture management
    - `validation.rs` - Validation framework
    - `score.rs` - Scoring and metrics

### 1.2 C API Surface

**File**: `crates/bitnet-ffi/src/c_api.rs` (1178 lines)

#### Core Functions Exposed:

**Model Management:**
```c
c_int bitnet_model_load(const char* path);
c_int bitnet_model_load_with_config(const char* path, const BitNetCConfig* config);
c_int bitnet_model_free(c_int model_id);
c_int bitnet_model_is_loaded(c_int model_id);
c_int bitnet_model_get_info(c_int model_id, BitNetCModel* info);
```

**Inference:**
```c
c_int bitnet_inference(c_int model_id, const char* prompt, char* output, size_t max_len);
c_int bitnet_inference_with_config(c_int model_id, const char* prompt, 
                                   const BitNetCInferenceConfig* config, 
                                   char* output, size_t max_len);
```

**Batch & Streaming:**
```c
c_int bitnet_batch_inference(c_int model_id, const char** prompts, size_t num_prompts,
                            const BitNetCInferenceConfig* config,
                            char** outputs, const size_t* max_lens);

c_int bitnet_start_streaming(c_int model_id, const char* prompt,
                            const BitNetCInferenceConfig* config,
                            const BitNetCStreamConfig* stream_config);
c_int bitnet_stream_next_token(c_int stream_id, char* token, size_t max_len);
c_int bitnet_stop_streaming(c_int stream_id);
```

**Performance & Metrics:**
```c
c_int bitnet_get_performance_metrics(c_int model_id, BitNetCPerformanceMetrics* metrics);
c_int bitnet_reset_performance_metrics(c_int model_id);
```

**GPU Control:**
```c
c_int bitnet_set_gpu_enabled(c_int enable);
c_int bitnet_is_gpu_available(void);
c_int bitnet_switch_model_backend(c_int model_id, c_uint backend_preference);
```

**Error Handling:**
```c
const char* bitnet_get_last_error(void);
void bitnet_clear_last_error(void);
```

### 1.3 C++ FFI Bindings

**File**: `crossval/src/cpp_bindings.rs` (200 lines)

#### Low-Level C++ Interface:

```rust
#[cfg(all(feature = "ffi", have_cpp))]
mod imp {
    unsafe extern "C" {
        fn bitnet_cpp_create_model(model_path: *const c_char) -> *mut c_void;
        fn bitnet_cpp_destroy_model(model: *mut c_void);
        fn bitnet_cpp_generate(
            model: *mut c_void,
            prompt: *const c_char,
            max_tokens: c_int,
            tokens_out: *mut u32,
            tokens_count: *mut c_int,
        ) -> c_int;
    }
}
```

**High-Level Rust Wrapper:**
```rust
pub struct CppModel { handle: *mut c_void }

impl CppModel {
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self>
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<Vec<u32>>
    pub fn model_info(&self) -> Result<ModelInfo>
    pub fn is_ready(&self) -> bool
}
```

---

## 2. Token Passing Capabilities

### 2.1 Current Token Handling

**Key Finding**: The FFI layer **primarily accepts strings (prompts)**, not raw token IDs.

**Current Flow:**

1. **Rust → C++ Flow**:
   - `bitnet_inference()` takes `prompt: *const c_char`
   - C++ implementation tokenizes internally
   - Returns generated text as string

2. **Token-level Control** (available but limited):
   - `crossval/src/cpp_bindings.rs` shows `generate()` can return `Vec<u32>` tokens
   - But the FFI C functions don't expose direct token input/output yet

### 2.2 Missing Token-Level FFI

**Current Limitation**: No direct C FFI function for:
```c
// NOT YET EXPOSED:
c_int bitnet_tokenize(const char* text, uint32_t* tokens_out, c_int* count_out);
c_int bitnet_inference_tokens(c_int model_id, const uint32_t* tokens_in, size_t num_tokens,
                              uint32_t* tokens_out, size_t* num_out);
```

**Workaround in Tests**:
```rust
// File: crossval/tests/per_position_logits.rs
let tokens = cpp_session.tokenize(prompt)?;  // Via bitnet_sys wrapper
let cpp_logits_last = cpp_session.eval_and_get_logits(&tokens, 0)?;  // Takes tokens directly
```

**Implication**: If you need to pass token IDs directly to C++, use the `bitnet_sys::wrapper::Session` API (lower-level C++ bindings), not the higher-level FFI C API.

---

## 3. Trace Infrastructure

### 3.1 Tracing System Overview

**Location**: `crates/bitnet-trace/src/lib.rs` (281 lines)

#### How Tracing Works:

Controlled by environment variable: `BITNET_TRACE_DIR=/path/to/traces`

**Automatic Capture** (no code changes needed):
```bash
export BITNET_TRACE_DIR=/tmp/my_traces
cargo run -p bitnet-cli -- run --model model.gguf --prompt "test" --max-tokens 4
ls /tmp/my_traces/  # View captured traces
```

#### Trace Record Format

**Structure** (`TraceRecord`):
```rust
pub struct TraceRecord {
    pub name: String,           // Tensor identifier (e.g., "blk0/attn_norm")
    pub shape: Vec<usize>,      // Tensor shape
    pub dtype: String,          // Data type (e.g., "F32")
    pub blake3: String,         // Blake3 hash of raw F32 bytes
    pub rms: f64,               // Root mean square of tensor values
    pub num_elements: usize,    // Total element count
    pub seq: Option<usize>,     // Token position (0=prefill, 1+=decode)
    pub layer: Option<isize>,   // Layer index (-1=embeddings/logits, 0+=transformer)
    pub stage: Option<String>,  // Stage (e.g., "embeddings", "q_proj", "attn_out", "ffn_out", "logits")
}
```

**JSON Serialization**:
```json
{
  "name": "blk0/attn_norm",
  "shape": [1, 2560],
  "dtype": "F32",
  "blake3": "abc123def...",
  "rms": 0.9982,
  "num_elements": 2560,
  "seq": 0,
  "layer": 0,
  "stage": "attn_norm"
}
```

### 3.2 Trace Capture Points

**Locations in transformer.rs**:

The codebase instruments traces at strategic computation stages:

1. **Embeddings** (layer=-1, stage="embeddings"):
   ```rust
   // After embed_tokens, before first transformer block
   bitnet_trace::dump_trace("t0/embeddings", &hidden, Some(0), Some(-1), Some("embeddings"))?;
   ```

2. **Q/K/V Projections** (stage="q_proj", "k_proj", "v_proj"):
   ```rust
   bitnet_trace::dump_trace(&format!("blk{}/q_proj", layer_idx), &q, Some(seq), Some(layer), Some("q_proj"))?;
   ```

3. **Attention Output** (stage="attn_out"):
   ```rust
   bitnet_trace::dump_trace(&format!("blk{}/attn_out", layer_idx), &attn_output, ...)?;
   ```

4. **FFN Output** (stage="ffn_out"):
   ```rust
   bitnet_trace::dump_trace(&format!("blk{}/ffn_out", layer_idx), &output, ...)?;
   ```

5. **Layer Norm** (intermediate):
   ```rust
   bitnet_trace::dump_trace(&format!("blk{}/attn_norm", layer_idx), &normalized, ...)?;
   ```

6. **Logits** (layer=-1, stage="logits"):
   ```rust
   bitnet_trace::dump_trace("t0/logits", &logits, Some(0), Some(-1), Some("logits"))?;
   ```

### 3.3 Trace Comparison (trace_diff)

**File**: `xtask/src/trace_diff.rs` (172 lines)

#### Workflow:

1. **Capture Rust traces**:
   ```bash
   BITNET_TRACE_DIR=/tmp/rs RUST_LOG=warn BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
     cargo run -p bitnet-cli --features cpu,trace -- run \
     --model model.gguf --tokenizer tok.json --prompt "What is 2+2?" --max-tokens 4 --greedy
   ```

2. **Capture C++ traces** (via C++ instrumentation):
   ```bash
   # See docs/howto/cpp-setup.md for C++ reference setup
   ```

3. **Compare traces**:
   ```bash
   cargo run -p xtask -- trace-diff /tmp/rs /tmp/cpp
   ```

#### Comparison Mechanism:

- Invokes Python script: `scripts/trace_diff.py`
- Compares Blake3 hashes of corresponding trace files
- Reports **first divergence point** as `(seq, layer, stage)`
- Example output:
  ```
  First divergence: token 2, layer 3, stage "attn_out"
  Rust hash:  abc123...
  C++ hash:   def456...
  ```

**Error Handling**:
- Validates both trace directories exist
- Checks Python 3 availability
- Provides helpful diagnostics for missing traces

---

## 4. Logits Comparison Infrastructure

**File**: `crossval/src/logits_compare.rs` (160+ lines)

### 4.1 Per-Position Divergence Detection

```rust
pub struct LogitsDivergence {
    pub first_divergence_token: Option<usize>,  // Position where logits first diverge
    pub per_token_cosine_sim: Vec<f32>,         // Cosine similarity per token position
    pub per_token_l2_dist: Vec<f32>,            // L2 distance per token position
    pub max_absolute_diff: f32,                 // Maximum absolute diff across all positions
}
```

### 4.2 Comparison Metrics

**Cosine Similarity** (default threshold: `1e-4`):
```rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}
```

**L2 Distance**:
```rust
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}
```

**Max Absolute Difference**:
```rust
let max_diff_at_pos = rs_vec
    .iter()
    .zip(cpp_vec.iter())
    .map(|(r, c)| (r - c).abs())
    .fold(0.0f32, f32::max);
```

### 4.3 Test Usage

**File**: `crossval/tests/per_position_logits.rs`

```rust
#[test]
fn test_multi_token_generation_divergence() -> Result<()> {
    let model_path = get_test_model()?;
    wrapper::init_backend();
    let _guard = scopeguard::guard((), |_| wrapper::free_backend());
    
    let mut cpp_session = CppSession::load_deterministic(&model_path)?;
    let tokens = cpp_session.tokenize("The capital of France is")?;
    
    // Get logits for each position
    let mut rust_logits = Vec::new();
    let mut cpp_logits = Vec::new();
    
    for pos in 0..tokens.len() {
        rust_logits.push(eval_logits_once(&model_path, &tokens[..pos+1])?);
        cpp_logits.push(cpp_session.eval_and_get_logits(&tokens, pos as i32)?);
    }
    
    // Compare per-position
    let divergence = compare_per_position_logits(&rust_logits, &cpp_logits);
    
    if let Some(div_pos) = divergence.first_divergence_token {
        eprintln!("Divergence at position {}: cosine_sim = {}", 
                  div_pos, divergence.per_token_cosine_sim[div_pos]);
    }
}
```

---

## 5. Weight Mapping & LM Head / Embeddings

### 5.1 Architecture Overview

**File**: `crates/bitnet-models/src/transformer.rs` (1800+ lines)

#### TransformerModel Structure:

```rust
pub struct TransformerModel {
    pub embed_tokens: candle_nn::Embedding,     // Token embeddings [vocab_size, hidden_size]
    pub embed_transposed: bool,                 // True if stored as [hidden_size, vocab_size]
    pub embed_tied_weight: Option<Tensor>,      // Cached [hidden_size, vocab_size] for tied models
    
    pub lm_head: Option<Linear>,                // Dedicated LM head (if not tied)
    pub lm_head_weight: Option<Tensor>,         // Direct weight access
    pub lm_head_transposed: bool,               // True if stored as [hidden_size, vocab_size]
    
    pub config: BitNetConfig,
    pub blocks: Vec<TransformerBlock>,
    pub norm: LayerNorm,
}
```

### 5.2 Head-Tie Handling

**Tied Weights (LM Head = Embedding Matrix)**

When a model has **no dedicated LM head**, the embedding weight is reused:

1. **Load Time** (lines 1355-1375):
   ```rust
   let (embed_transposed, embed_tied_weight) = if lm_head.is_none() {
       // Pre-transpose embeddings [V,H] -> [H,V] once at load
       let embed_weight = embed_tokens.embeddings();  // Always [V,H] from Candle
       let transposed_weight = embed_weight.transpose(0, 1)?;  // [H,V]
       tracing::info!("Pre-transposing tied embeddings [V,H] -> [H,V] for logits");
       (embed_transposed, Some(transposed_weight))  // Cache it!
   } else {
       (embed_transposed, None)
   };
   ```

2. **Inference Time** (lines 1632-1640):
   ```rust
   let logits = if self.embed_transposed {
       // Embeddings are [hidden, vocab] - use directly
       let embeddings = self.embed_tokens.embeddings();
       hidden.matmul(embeddings)?
   } else if let Some(ref cached_weight) = self.embed_tied_weight {
       // Use pre-transposed cached weight [H, V] - avoids per-step transpose!
       hidden.matmul(cached_weight)?
   } else {
       // Fallback: transpose on-demand
       let embeddings = self.embed_tokens.embeddings();
       let w = embeddings.transpose(0, 1)?;  // [H, V]
       hidden.matmul(&w)?
   };
   ```

**Key Optimization**: Pre-transpose tied embeddings at load time, avoiding per-token overhead.

### 5.3 Transposition Handling

**GGUF Embedding Storage Variants**:

The loader handles multiple GGUF embedding tensor name conventions:
```rust
fn is_embedding_tensor(name: &str) -> bool {
    matches!(name,
        "embed_tokens.weight"
        | "tok_embeddings.weight"
        | "model.embed_tokens.weight"
    )
}
```

**Detection of Transpose**:
```rust
// Read transpose flag for embeddings (1-element tensor)
let embed_transposed = match vb.get((1,), "embed_tokens.transposed") {
    Ok(t) => {
        let val = t.to_vec0::<u8>()?;
        val != 0
    }
    Err(_) => {
        // No explicit flag - infer from shape
        embedding_is_transposed(dims)
    }
};
```

### 5.4 LM Head Variants

**Case 1: Dedicated LM Head**
```rust
// Use lm_head.forward(hidden) directly
if let Some(ref lm_head) = self.lm_head {
    let logits = lm_head.forward(hidden)?;
    logits.reshape(&[b, vocab_size])?
}
```

**Case 2: Tied Weights (No LM Head)**
```rust
// Use embedding matrix transposed
} else {
    if self.embed_transposed {
        let embeddings = self.embed_tokens.embeddings();
        hidden.matmul(embeddings)?
    } else if let Some(ref cached_weight) = self.embed_tied_weight {
        hidden.matmul(cached_weight)?
    }
}
```

### 5.5 Validation & Sanity Checks

**Debug Mode** (enabled with `BITNET_DEBUG_LOGITS`):
```rust
if std::env::var("BITNET_DEBUG_LOGITS").is_ok() {
    static SANITY_LOGGED: std::sync::Once = std::sync::Once::new();
    SANITY_LOGGED.call_once(|| {
        // Compare tied logits with float reference
        if let Ok(emb) = self.embed_tokens.embeddings().transpose(0, 1)
            && let Ok(ref_logits) = hidden.matmul(&emb)
        {
            tracing::info!("Quantized vs float logits sanity check passed");
        }
    });
}
```

---

## 6. Data Flow Diagrams

### 6.1 Inference Path (Text Prompt)

```
User API Call
    ↓
bitnet_inference(model_id, "What is 2+2?", output, max_len)
    ↓
[C API Wrapper] validate inputs
    ↓
InferenceManager::generate_with_config()
    ↓
[Tokenizer] Prompt → Token IDs
    ↓
[Model::embed] Tokens → Embeddings [B, T, H]
    ↓
[TransformerBlocks] Embeddings → Hidden [B, T, H]
    ├─ Per-layer:
    │  ├─ LayerNorm
    │  ├─ MultiHeadAttention (with QK256 if quantized)
    │  └─ FFN
    ├─ [TRACE] Capture per layer
    └─ Return final hidden [B, T, H]
    ↓
[Model::logits] Hidden → Logits [B, V]
    ├─ If lm_head exists: use Linear layer
    ├─ If tied weights: use embed_tied_weight (pre-transposed)
    └─ [TRACE] Capture logits
    ↓
[Sampler] Logits → Next Token ID
    ↓
[Tokenizer::decode] Token ID → String
    ↓
Copy to output buffer + null-terminate
```

### 6.2 Cross-Validation Path (Rust vs C++)

```
[Rust Forward Pass]
    ├─ Full trace capture: BITNET_TRACE_DIR=/tmp/rs
    ├─ Per-layer: embeddings, attn, ffn, logits
    └─ Blake3 hash each tensor → /tmp/rs/*.trace

[C++ Forward Pass] (via FFI or external)
    ├─ Full trace capture (instrumented C++ binary)
    └─ Per-layer: embeddings, attn, ffn, logits → /tmp/cpp/*.trace

[trace-diff Tool]
    ├─ Read /tmp/rs and /tmp/cpp trace files
    ├─ Compare Blake3 hashes
    └─ Report first divergence: (token_pos, layer, stage)
```

---

## 7. Summary: Strengths & Limitations

### 7.1 Strengths

| Aspect | Capability |
|--------|-----------|
| **C API Surface** | Comprehensive (30+ functions), complete error handling |
| **Token-Level Control** | Available via `bitnet_sys::wrapper::Session`, not exposed in high-level C API |
| **Trace Infrastructure** | Excellent per-layer, per-token capture with Blake3 hashing |
| **Divergence Detection** | Sophisticated per-position logits comparison (cosine similarity, L2, max diff) |
| **Weight Mapping** | Robust handling of tied weights, transposition, quantization variants |
| **Performance** | Pre-transpose optimization for tied weights eliminates per-token overhead |

### 7.2 Limitations

| Aspect | Issue |
|--------|-------|
| **Token FFI** | No C FFI functions for direct token-level inference (tokenize, infer_tokens) |
| **Streaming Logits** | C streaming API doesn't expose per-token logits (only text tokens) |
| **C++ Dependency** | FFI requires manual C++ reference setup (BITNET_CPP_DIR) |
| **Trace Overhead** | JSON serialization per tensor adds latency (mitigated by env var gating) |
| **Multi-GPU** | No explicit multi-GPU FFI support (single-device only) |

---

## 8. Quick Reference Commands

### Capturing Traces

```bash
# Rust traces (with determinism)
BITNET_TRACE_DIR=/tmp/rs BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
  cargo run -p bitnet-cli --features cpu,trace -- run \
  --model model.gguf --tokenizer tok.json --prompt "test" --max-tokens 4 --greedy

# C++ traces (requires C++ setup)
# See docs/howto/cpp-setup.md
```

### Comparing Traces

```bash
cargo run -p xtask -- trace-diff /tmp/rs /tmp/cpp
```

### Checking Weight Mappings

```bash
# Inspect a model's weight structure
cargo run -p bitnet-cli -- compat-check model.gguf --show-kv

# Debug tied weights and transposition
BITNET_DEBUG_LOGITS=1 cargo run -p bitnet-cli -- run \
  --model model.gguf --prompt "test" --max-tokens 1
```

### Per-Position Logits Divergence

```bash
# Requires C++ reference setup
BITNET_CPP_DIR=/path/to/bitnet.cpp \
  cargo test -p crossval --test per_position_logits -- --nocapture
```

---

## 9. Files Reference Map

```
Key FFI & Tracing Files:

crates/bitnet-ffi/
├─ src/c_api.rs              (1178 lines) - Core C API functions
├─ src/inference.rs          (495 lines)  - Inference manager
└─ src/*.rs                  - Supporting modules

crates/bitnet-trace/
└─ src/lib.rs                (281 lines)  - Trace capture API

crossval/
├─ src/cpp_bindings.rs       (200 lines)  - C++ FFI wrappers
├─ src/logits_compare.rs     (160 lines)  - Per-position logits
├─ src/comparison.rs         (150 lines)  - Cross-validation runner
└─ tests/per_position_logits.rs          - Divergence detection tests

xtask/
└─ src/trace_diff.rs         (172 lines)  - Trace comparison tool

crates/bitnet-models/
└─ src/transformer.rs        (1800 lines) - Weight mapping, logits
```

---

## 10. Next Steps for Integration

If you need to extend FFI or tracing:

1. **Add Token-Level FFI Functions**:
   - Implement `bitnet_tokenize()` and `bitnet_inference_tokens()` in `c_api.rs`
   - Wire through InferenceManager to support token-level control

2. **Enhanced Trace Metadata**:
   - Add `position_embedding` and `batch_id` fields to `TraceRecord`
   - Support trace filtering by layer/stage for large models

3. **Streaming Logits**:
   - Extend `BitNetCStreamConfig` to capture per-token logits
   - Add callback for logits output alongside text tokens

4. **Multi-Device Support**:
   - Add device_id parameter to FFI functions
   - Extend InferenceManager to manage per-device model instances

