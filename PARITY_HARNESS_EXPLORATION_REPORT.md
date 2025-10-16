# Parity Harness Implementation - Comprehensive Exploration Report

**Date**: 2025-10-16  
**Branch**: feat/crossval-parity-harness  
**Task**: PR #468 - Understand state and document APIs for implementation

## 1. Parity Harness Structure

### Location
- **Main Test**: `/home/steven/code/Rust/BitNet-rs/crossval/tests/parity_bitnetcpp.rs`
- **Status**: Infrastructure-complete with Rust-side implementation, awaiting C++ FFI integration

### Current State

The test file (`parity_bitnetcpp.rs`, lines 1-291) contains:

1. **Helper Function 1: `rust_side_tokenize_and_meta()`** (lines 140-176)
   - **Signature**: 
     ```rust
     fn rust_side_tokenize_and_meta(
         model_path: &std::path::Path,
         prompt: &str,
     ) -> Result<(Vec<u32>, bool, bool, u32, usize)>
     ```
   - **Returns**: `(token_ids, add_bos, add_special, eos_token_id, vocab_size)`
   - **Implementation**:
     - Loads tokenizer via `bitnet_tokenizers::auto::load_auto(model_path, None)`
     - Auto-detects template type from model path
     - Uses `template.should_add_bos()` to determine BOS policy
     - For LLaMA-3 chat: Encodes `<|eot_id|>` token directly (token-level)
     - Falls back to tokenizer's standard EOS token
     - Applies template via `template.apply(prompt, None)`
     - Encodes formatted prompt with tokenizer settings

2. **Helper Function 2: `rust_eval_last_logits()`** (lines 199-237)
   - **Signature**:
     ```rust
     async fn rust_eval_last_logits(
         model_path: &std::path::Path,
         ids: &[u32],
         expected_vocab_size: usize,
     ) -> Result<Vec<f32>>
     ```
   - **Returns**: `Vec<f32>` logits for vocabulary
   - **Implementation**:
     - Creates `ModelLoader::new(BNDevice::Cpu)` 
     - Loads model via `loader.load(model_path)?`
     - Wraps in `Arc<dyn bitnet_models::Model>`
     - Loads tokenizer via `bitnet_tokenizers::auto::load_auto()`
     - Creates `InferenceEngine::new(model_arc, tokenizer_arc, BNDevice::Cpu)?`
     - Calls `engine.eval_ids(ids).await?`
     - Extracts last position's logits (last vocab_size elements)
     - **Critical**: Returns flat vector, not 2D tensor

3. **Helper Function 3: `rust_decode_n_greedy()`** (lines 240-290)
   - **Signature**:
     ```rust
     async fn rust_decode_n_greedy(
         model_path: &std::path::Path,
         prompt_ids: &[u32],
         n_steps: usize,
         eos_id: u32,
     ) -> Result<Vec<u32>>
     ```
   - **Returns**: `Vec<u32>` decoded token sequence
   - **Implementation**:
     - Uses same model/tokenizer loading as `rust_eval_last_logits()`
     - Configures `GenerationConfig`:
       ```rust
       GenerationConfig {
           max_new_tokens: n_steps as u32,
           temperature: 0.0,        // Greedy
           top_k: Some(1),
           top_p: Some(1.0),
           repetition_penalty: 1.0,
           seed: Some(0),           // Deterministic
           stop_sequences: vec![],
       }
       ```
     - Calls `engine.generate_tokens(prompt_ids, &config).await?`
     - Truncates at EOS token or n_steps limit

4. **Auto-Detect Template** (lines 179-196)
   - **Path-based heuristics**:
     - Contains "llama" + "3" → LLaMA-3 chat
     - Contains "instruct" or "chat" → Instruct
     - Default → Instruct (v0.9.x change for UX)

5. **Main Test Function: `parity_bitnetcpp()`** (lines 34-136)
   - Entry point: `#[tokio::test]`
   - Skips gracefully if `CROSSVAL_GGUF` not set
   - Calls three helpers in sequence
   - Writes JSON receipt to `docs/baselines/{DATE}/parity-bitnetcpp.json`
   - **C++ Integration**: Awaits `build.rs` compilation

---

## 2. InferenceEngine API

### Location
- **File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/engine.rs`
- **Lines**: 716-2165

### Constructor Signature

```rust
pub fn new(
    model: Arc<dyn Model>,
    tokenizer: Arc<dyn Tokenizer>,
    device: Device,
) -> Result<Self>
```

**Parameters**:
- `model`: Model trait object (implementations: BitNetModel, MockModel)
- `tokenizer`: Tokenizer trait object (HF, GGUF, SentencePiece)
- `device`: `Device::Cpu`, `Device::Cuda(id)`, or `Device::Metal`

**Validates**:
- Model hyperparameters during init (line 775-777)
- Quantization sanity checks (line 779-780)

### Core Evaluation Methods

#### `eval_ids()` - Used in Parity
```rust
pub async fn eval_ids(&mut self, ids: &[u32]) -> Result<Vec<f32>>
```

**Lines**: 842-866

**Behavior**:
- Converts token IDs to Candle tensor (shape: `[1, ids.len()]`)
- Runs forward pass through model backend
- Returns flattened logits as `Vec<f32>`
- Mock fallback: Returns `vec![0.0; 100]` for testing

**Critical for Parity**: Used in `rust_eval_last_logits()` to get vocabulary logits

#### `prefill()` - Cache Warming
```rust
pub async fn prefill(&mut self, tokens: &[u32]) -> Result<()>
```

**Lines**: 1033-1084

**Behavior**:
- Runs forward pass to populate KV cache
- Validates token count against context length
- Validates tokens are within vocabulary range
- Returns early if token list is empty
- Used for latency measurement

#### `generate_tokens()` - Main Generation Method
```rust
pub async fn generate_tokens(
    &self,
    input_tokens: &[u32],
    config: &GenerationConfig,
) -> Result<Vec<u32>>
```

**Lines**: 1087-1228

**Behavior**:
- Full forward pass on first step
- Incremental forward passes thereafter (last token only)
- Samples next token using `SamplingStrategy`
- Checks for stop conditions (EOS token, stop sequences)
- Enforces context length limits (sliding window fallback)
- Returns vector of generated token IDs

### Logits Extraction Methods

#### `forward_pass()` - Internal
```rust
async fn forward_pass(&self, tokens: &[u32]) -> Result<Vec<f32>>
```

**Lines**: 1231-1267

**Operations**:
1. Records embedding kernel
2. Converts tokens to tensor via `tokens_to_tensor()`
3. Acquires cache lock
4. Records prefill/decode kernel based on length
5. Records I2S quantization kernel
6. Calls backend forward pass
7. Records logits projection kernel
8. Extracts logits via `tensor_to_logits()`

#### `tensor_to_logits()` - Internal
```rust
fn tensor_to_logits(&self, tensor: &ConcreteTensor) -> Result<Vec<f32>>
```

**Lines**: 1276-1312

**Implementation**:
- Calls `model.logits(tensor)` to get `[B,T,V]` tensor
- Validates 3D shape (batch=1)
- Uses Candle `narrow()` and `squeeze()` to extract last timestep
- Converts to F32 if needed
- Returns as `Vec<f32>`

### Performance Tracking

#### `get_performance_metrics()`
```rust
pub async fn get_performance_metrics(&self) -> Result<PerformanceMetrics>
```

**Fields** (PerformanceMetrics struct, lines 548-562):
- `total_latency_ms`: u64
- `tokens_generated`: usize
- `tokens_per_second`: f64
- `first_token_latency_ms`: Option<u64>
- `average_token_latency_ms`: Option<f64>
- `memory_usage_bytes`: Option<usize>
- `cache_hit_rate`: Option<f64>
- `backend_type`: String
- Component timings: model_load, tokenizer_encode/decode, forward_pass, sampling

---

## 3. Tokenizer and Template APIs

### Tokenizer Auto-Discovery

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-tokenizers/src/auto.rs` (lines 1-44)

#### `load_auto()` Function
```rust
pub fn load_auto(
    model_path: &Path,
    explicit: Option<&Path>,
) -> Result<Arc<dyn Tokenizer + Send + Sync>>
```

**Priority**:
1. **Explicit path**: Uses `explicit` if provided
2. **GGUF embedded**: Checks GGUF for tokenizer data
3. **Sibling files**: Looks for `tokenizer.json` or `tokenizer.model` in model directory
4. **Fails**: Returns error if no tokenizer found (no fallback)

**Returns**: `Arc<dyn Tokenizer + Send + Sync>` trait object

### Template Type and Methods

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/prompt_template.rs` (lines 1-507)

#### `TemplateType` Enum
```rust
pub enum TemplateType {
    Raw,          // No formatting
    Instruct,     // Q&A format with stop sequences
    Llama3Chat,   // LLaMA-3 with special tokens
}
```

#### Detection Method
```rust
pub fn detect(
    tokenizer_name: Option<&str>,
    chat_template_jinja: Option<&str>
) -> Self
```

**Lines**: 83-109

**Priority**:
1. GGUF `chat_template` metadata (Jinja detection)
2. Tokenizer family heuristics (llama3, instruct, mistral)
3. Fallback to Raw

#### `should_add_bos()` - Used in Parity
```rust
pub fn should_add_bos(&self) -> bool
```

**Lines**: 180-185

**Behavior**:
- **Raw**: `true` (add BOS)
- **Instruct**: `true` (add BOS)
- **Llama3Chat**: `false` (template includes `<|begin_of_text|>`)

#### `apply()` - Template Application
```rust
pub fn apply(
    &self,
    user_text: &str,
    system_prompt: Option<&str>
) -> String
```

**Lines**: 112-118

**Implementations**:

**Raw** (line 114):
```
{user_text}
```

**Instruct** (lines 121-135):
```
System: {system_prompt}

Q: {user_text}
A:
```

**Llama3Chat** (lines 148-167):
```
<|begin_of_text|>
[<|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>]
<|start_header_id|>user<|end_header_id|>
{user_text}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

```

#### LLaMA-3 EOT Token Handling
**Location**: Lines 159-164 in parity test

```rust
let eos_id = if matches!(template, TemplateType::Llama3Chat) {
    let eot_ids = tokenizer.encode("<|eot_id|>", false, true)?;
    eot_ids.get(0).copied().unwrap_or_else(|| {
        tokenizer.eos_token_id().unwrap_or(128009)
    })
} else {
    tokenizer.eos_token_id().unwrap_or(2)
};
```

**Pattern**: Encode with `add_bos=false, add_special=true` to get token ID

#### `default_stop_sequences()`
```rust
pub fn default_stop_sequences(&self) -> Vec<String>
```

**Returns**:
- **Raw**: `[]` (empty)
- **Instruct**: `["\n\nQ:", "\n\nHuman:"]`
- **Llama3Chat**: `["<|eot_id|>", "<|end_of_text|>"]`

---

## 4. bitnet-sys FFI Structure

### Location
- **Header**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/include/bitnet_c.h`
- **C Shim**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/csrc/bitnet_c_shim.cc`
- **Build Script**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/build.rs`

### Build Configuration

**File**: `build.rs` (lines 1-246)

**Feature Gate**: `if env::var("CARGO_FEATURE_FFI").is_err() { return; }`

**Compilation Steps** (when `--features ffi` enabled):
1. Locates BitNet C++ via `BITNET_CPP_DIR` or `BITNET_CPP_PATH` env vars
2. Validates build directory exists
3. **Links**: `llama`, `ggml`, platform libraries (stdc++, pthread, Accelerate, etc.)
4. **Generates Bindings**: Via `bindgen` from `llama.h`
   - Allowlist: `llama_*`, `ggml_*`, `ggml_bitnet_*`
   - Patches: Adds `unsafe extern "C"` for Rust 2024
   - Output: `$OUT_DIR/bindings.rs`

### C++ Shim Implementation

**File**: `bitnet_c_shim.cc` (lines 1-100+, structure visible)

**Internal Structs** (lines 42-51):
```cpp
struct bitnet_model {
    llama::model* model = nullptr;
    int vocab_size = 0;
};

struct bitnet_ctx {
    llama::context* context = nullptr;
    llama::model* model = nullptr;
    int n_threads = 1;
};
```

**C Interface Functions** (lines 55-87):
- `bitnet_model_t* bitnet_model_new_from_file(const char* gguf_path)`
- `void bitnet_model_free(bitnet_model_t* m)`
- `bitnet_ctx_t* bitnet_context_new(bitnet_model_t* m, const bitnet_params_t* p)`

**Forward Declarations** (lines 15-39):
```cpp
namespace llama {
    struct model;
    struct context;
    model* load_model_from_file(...);
    context* new_context(...);
    int tokenize(...);
    bool eval(...);
    const float* get_logits(...);
    int get_vocab_size(...);
    int sample_greedy(...);
}
```

**Integration Status**: ⏳ Awaiting build.rs configuration to compile `.cc` file

### Build.rs Integration Needed

**Pseudo-code** (from PR #468 summary):
```rust
#[cfg(feature = "ffi")]
fn compile_cpp_shim(cpp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    cc::Build::new()
        .cpp(true)
        .file("csrc/bitnet_c_shim.cc")
        .include(cpp_dir.join("include"))
        .include(cpp_dir.join("3rdparty/llama.cpp/include"))
        .compile("bitnet_c_shim");
    Ok(())
}
```

---

## 5. Model Loading Patterns

### bitnet-models ModelLoader

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/loader.rs` (lines 1-150)

#### Constructor
```rust
pub fn new(device: Device) -> Self
```

**Parameters**: `Device::Cpu`, `Device::Cuda(id)`, or `Device::Metal`

**Registers Loaders**:
- `GgufLoader`
- `SafeTensorsLoader`
- `HuggingFaceLoader`

#### Load Method (Used in Parity)
```rust
pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<Box<dyn Model>>
```

**Lines**: 70-72

**Implementation**:
```rust
self.load_with_config(path, &LoadConfig::default())
```

**LoadConfig Default**:
- `use_mmap: true`
- `validate_checksums: true`
- `progress_callback: None`

#### Detection Strategy
1. **Extension** (`.gguf`, `.safetensors`)
2. **Magic bytes** (file header parsing)
3. **Directory structure** (HuggingFace repo markers)

#### Error Handling
- Returns `Result<Box<dyn Model>>` trait object
- Fails fast on format detection failure
- Validates model compatibility before returning

### Model Trait

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/bitnet.rs` (lines 12-21)

```rust
pub trait Model: Send + Sync {
    fn config(&self) -> &BitNetConfig;
    fn forward(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
    ) -> Result<ConcreteTensor>;
    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor>;
    fn logits(&self, hidden: &ConcreteTensor) -> Result<ConcreteTensor>;
}
```

**Used in Parity**:
- `config()`: Get model hyperparameters
- `embed()`: Used by engine's `tokens_to_tensor()`
- `logits()`: Used by engine's `tensor_to_logits()`

---

## 6. Generation Configuration

### GenerationConfig Struct

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/config.rs` (lines 39-72)

**Fields**:
```rust
pub struct GenerationConfig {
    pub max_new_tokens: u32,
    pub temperature: f32,           // 0.0 = greedy
    pub top_k: u32,                 // 0 = disabled
    pub top_p: f32,                 // 1.0 = disabled
    pub repetition_penalty: f32,    // 1.0 = no penalty
    pub stop_sequences: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub seed: Option<u64>,
    pub skip_special_tokens: bool,
    pub eos_token_id: Option<u32>,
    pub logits_tap_steps: usize,
    pub logits_topk: usize,
    pub logits_cb: Option<LogitsCallback>,
    pub add_bos: bool,
}
```

**Default**:
- `max_new_tokens: 100`
- `temperature: 0.7`
- `top_k: 50`
- `top_p: 0.95`
- `repetition_penalty: 1.0`
- `seed: None`
- `add_bos: false`

**Parity Usage** (lines 264-272 in test):
```rust
GenerationConfig {
    max_new_tokens: n_steps as u32,
    temperature: 0.0,                    // Greedy
    top_k: Some(1),
    top_p: Some(1.0),
    repetition_penalty: 1.0,
    seed: Some(0),                       // Deterministic
    stop_sequences: vec![],
}
```

---

## 7. Implementation Patterns Already Established

### Pattern 1: Async Test with Real Engine
```rust
#[tokio::test]
async fn parity_bitnetcpp() -> Result<()> {
    // Load model and tokenizer
    // Call async engine methods
    // Validate outputs
    // Write receipts
    Ok(())
}
```

### Pattern 2: Arc Wrapping for Model/Tokenizer
```rust
let loader = ModelLoader::new(BNDevice::Cpu);
let model = loader.load(model_path)?;
let model_arc: Arc<dyn bitnet_models::Model> = model.into();

let tokenizer = auto::load_auto(model_path, None)?;
let tokenizer_arc: Arc<dyn bitnet_tokenizers::Tokenizer> = tokenizer;

let engine = InferenceEngine::new(model_arc, tokenizer_arc, BNDevice::Cpu)?;
```

### Pattern 3: Greedy Decoding Config
```rust
GenerationConfig {
    temperature: 0.0,
    top_k: Some(1),
    seed: Some(0),
    ..Default::default()
}
```

### Pattern 4: JSON Receipt Writing
```rust
let receipt = json!({
    "timestamp": ts,
    "commit": commit,
    "rust": { /* outputs */ },
    "parity": { /* metrics */ },
});
fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
```

---

## 8. Critical Implementation Details for PR #468

### EOT Token Handling (LLaMA-3 Specific)
```rust
// Pattern from parity test, lines 159-164
let eos_id = if matches!(template, TemplateType::Llama3Chat) {
    let eot_ids = tokenizer.encode("<|eot_id|>", false, true)?;
    eot_ids.get(0).copied().unwrap_or_else(|| {
        tokenizer.eos_token_id().unwrap_or(128009)
    })
};
```

**Key**: Encode with `add_special=true` to capture token ID correctly

### Logits Extraction (Must Match Vocab Size)
```rust
// From rust_eval_last_logits, lines 225-227
let last_logits = if logits.len() >= expected_vocab_size {
    logits[logits.len() - expected_vocab_size..].to_vec()
} else {
    anyhow::bail!("Logits size mismatch")
};
```

**Key**: Last `vocab_size` elements represent predictions for next token

### Template Detection (Path-Based Fallback)
```rust
// From auto_detect_template, lines 182-195
let path_str = model_path.to_string_lossy().to_lowercase();
if path_str.contains("llama") && path_str.contains("3") {
    TemplateType::Llama3Chat
} else if path_str.contains("instruct") || path_str.contains("chat") {
    TemplateType::Instruct
} else {
    TemplateType::Instruct  // Default (v0.9.x change)
}
```

---

## 9. Recommended Next Steps for Implementation

### Immediate (For PR #468 completion)
1. **Fix ModelLoader usage**: Current test imports from wrong location
   - Change: `use bitnet_inference::InferenceEngine;` pattern
   - Verify: ModelLoader in bitnet-models vs bitnet-inference
   
2. **Validate Async Pattern**:
   - Test runs `#[tokio::test]` correctly
   - All `.await` calls on async methods
   - Error handling proper

3. **Build Integration**:
   - Implement C++ shim compilation in `build.rs`
   - Validate llama.cpp header paths
   - Test linking against external libraries

### Follow-up (For C++ Parity)
1. **FFI Bindings**:
   - Generate Rust bindings from llama.h
   - Implement wrapper functions in bitnet_c_shim.cc
   
2. **C++ Comparison Logic**:
   - Call C++ functions via FFI
   - Compute cosine similarity (already implemented at line 18-31)
   - Update receipt with real metrics

3. **CI Integration**:
   - Label-based workflow trigger
   - Nightly baseline generation
   - Parity report publication

---

## Summary Tables

### Function Signatures Used

| Function | Signature | Location |
|----------|-----------|----------|
| `load_auto` | `(Path, Option<Path>) -> Arc<dyn Tokenizer>` | bitnet-tokenizers/auto.rs:6 |
| `InferenceEngine::new` | `(Arc<Model>, Arc<Tokenizer>, Device) -> Result<Engine>` | bitnet-inference/engine.rs:737 |
| `eval_ids` | `(&mut self, &[u32]) -> async Result<Vec<f32>>` | bitnet-inference/engine.rs:842 |
| `generate_tokens` | `(&self, &[u32], &GenerationConfig) -> async Result<Vec<u32>>` | bitnet-inference/engine.rs:1087 |
| `TemplateType::should_add_bos` | `(&self) -> bool` | bitnet-inference/prompt_template.rs:180 |
| `TemplateType::apply` | `(&self, &str, Option<&str>) -> String` | bitnet-inference/prompt_template.rs:112 |
| `ModelLoader::new` | `(Device) -> Self` | bitnet-models/loader.rs:58 |
| `ModelLoader::load` | `(&self, Path) -> Result<Box<dyn Model>>` | bitnet-models/loader.rs:70 |

### Template Behavior Matrix

| Template | BOS | Format | Stop Sequences |
|----------|-----|--------|-----------------|
| Raw | ✓ | Plain text | None |
| Instruct | ✓ | Q: ... A: | `\n\nQ:`, `\n\nHuman:` |
| Llama3Chat | ✗ | Special tokens | `<\|eot_id\|>`, `<\|end_of_text\|>` |

### Test Execution Flow

```
parity_bitnetcpp()
├── Load model via ModelLoader::new(Cpu).load()
├── Load tokenizer via auto::load_auto()
├── Detect template from path
├── rust_side_tokenize_and_meta()
│   ├── Apply template.apply()
│   └── Tokenize with tokenizer.encode()
├── rust_eval_last_logits()
│   ├── Create InferenceEngine
│   ├── Call engine.eval_ids().await
│   └── Extract logits[vocab_size:]
├── rust_decode_n_greedy()
│   ├── Generate with temp=0.0, seed=0
│   └── Truncate at EOS
└── Write JSON receipt to docs/baselines/
```

