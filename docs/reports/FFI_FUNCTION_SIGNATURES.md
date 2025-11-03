# C++ FFI Function Signatures - Complete Reference

## Source: crates/bitnet-sys/src/wrapper.rs

### Session API (Public, Lines 330-394)

```rust
/// Combined session for easy use in tests with deterministic settings
pub struct Session {
    pub model: Model,
    pub context: Context,
}

impl Session {
    /// Load a model and create a context with deterministic settings
    pub fn load_deterministic(model_path: &str) -> Result<Self>
    
    /// Tokenize text to token IDs
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>>
    
    /// Decode tokens back to text
    pub fn decode(&self, tokens: &[i32]) -> Result<String>
    
    /// Evaluate tokens and return logits for LAST position only
    pub fn eval_and_get_logits(&mut self, tokens: &[i32], n_past: i32) -> Result<Vec<f32>>
    
    /// Generate tokens greedily from a prompt
    pub fn generate_greedy(&mut self, prompt: &str, max_tokens: usize) -> Result<Vec<i32>>
}
```

### Context API (Lower-level, Lines 120-327)

```rust
pub struct Context {
    ptr: *mut llama_context,
}

impl Context {
    /// Create a new context from a model
    pub fn new(model: &Model, n_ctx: u32, n_batch: u32, n_threads: i32) -> Result<Self>
    
    /// Tokenize text into token IDs
    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>>
    
    /// Decode tokens back to text
    pub fn decode(&self, tokens: &[i32]) -> Result<String>
    
    /// Evaluate tokens (primary function for inference)
    pub fn eval(&mut self, tokens: &[i32], n_past: i32) -> Result<()>
    
    /// Get logits from the last evaluation (LAST position only)
    pub fn get_logits(&self) -> Result<Vec<f32>>
    
    /// Get logits for a specific token index (requires logits_all=true)
    pub fn get_logits_ith(&self, i: i32) -> Result<Vec<f32>>
    
    /// Get ALL logits for each token position (requires logits_all=true)
    pub fn get_all_logits(&self, n_tokens: usize) -> Result<Vec<Vec<f32>>>
    
    /// Sample a token from logits using greedy sampling
    pub fn sample_greedy(&self, logits: &[f32]) -> i32
}
```

### Model API (Lines 58-117)

```rust
pub struct Model {
    ptr: *mut llama_model,
}

impl Model {
    /// Load a model from a GGUF file
    pub fn load(path: &str) -> Result<Self>
    
    /// Get the number of tokens in the model's vocabulary
    pub fn n_vocab(&self) -> i32
    
    /// Get the model's context size
    pub fn n_ctx_train(&self) -> i32
    
    /// Get the model's embedding dimension
    pub fn n_embd(&self) -> i32
}
```

### Custom BitNet C Shim API (Lines 400-659)

```rust
/// Safe wrapper for bitnet_model_t from bitnet_c_shim.cc
pub struct BitnetModel {
    ptr: *mut crate::bindings::bitnet_model_t,
}

impl BitnetModel {
    /// Load a model from a GGUF file using the custom C shim
    pub fn from_file(path: &str) -> Result<Self>
}

/// Safe wrapper for bitnet_ctx_t from bitnet_c_shim.cc
pub struct BitnetContext {
    ptr: *mut crate::bindings::bitnet_ctx_t,
}

impl BitnetContext {
    /// Create a new context with specified parameters
    pub fn new(model: &BitnetModel, n_ctx: i32, n_threads: i32, seed: i32) -> Result<Self>
}

/// Tokenize text using the custom C shim
pub fn bitnet_tokenize_text(
    model: &BitnetModel,
    text: &str,
    add_bos: bool,
    parse_special: bool,
) -> Result<Vec<i32>>

/// Evaluate tokens and get last-position logits using the custom C shim
/// Returns: Vec<f32> (vocab_size elements, logits for last token only)
pub fn bitnet_eval_tokens(
    ctx: &BitnetContext,
    ids: &[i32],        // ← DIRECT TOKEN IDS (key function!)
    vocab_size: usize,
) -> Result<Vec<f32>>

/// Prefill the context with prompt tokens (primes KV cache and sets n_past)
pub fn bitnet_prefill(
    ctx: &BitnetContext,
    ids: &[i32],        // ← DIRECT TOKEN IDS
) -> Result<()>

/// Get vocabulary size from the C++ context
pub fn cpp_vocab_size(ctx: &BitnetContext) -> Result<usize>

/// Perform greedy decoding using the capacity-safe C shim
/// The context must be pre-filled with `bitnet_prefill()` before calling
pub fn cpp_decode_greedy(
    model: &BitnetModel,
    ctx: &BitnetContext,
    eos_id: i32,
    eot_id: Option<i32>,
    max_steps: usize,
    out: &mut [i32],    // Output buffer for generated token IDs
) -> Result<usize>      // Returns number of tokens generated
```

### Utility Functions (Lines 33-56)

```rust
/// Initialize the llama backend
pub fn init_backend()

/// Free the llama backend
pub fn free_backend()

/// Get the llama.cpp version string
pub fn get_version() -> String
```

---

## Usage Examples

### Example 1: Load Model and Evaluate Tokens (Session API)

```rust
use bitnet_sys::wrapper::{Session, init_backend, free_backend};

// Initialize backend
init_backend();

// Load model deterministically
let mut session = Session::load_deterministic("model.gguf")?;

// Method 1: String → tokenize → eval
let prompt = "Hello world";
let tokens = session.tokenize(prompt)?;  // Returns Vec<i32>
let logits = session.eval_and_get_logits(&tokens, 0)?;  // Returns Vec<f32>

// Method 2: Pre-tokenized eval (token IDs only)
let my_tokens = vec![1, 2, 3, 4];
let logits = session.eval_and_get_logits(&my_tokens, 0)?;

// Cleanup
drop(session);
free_backend();
```

### Example 2: Custom BitNet C Shim

```rust
use bitnet_sys::wrapper::{BitnetModel, BitnetContext, bitnet_eval_tokens, cpp_vocab_size, init_backend};

init_backend();

let model = BitnetModel::from_file("model.gguf")?;
let mut ctx = BitnetContext::new(&model, 2048, 1, 42)?;

// Direct token evaluation (no string tokenization needed!)
let tokens = vec![1, 2, 3, 4];
let vocab = cpp_vocab_size(&ctx)?;
let logits = bitnet_eval_tokens(&ctx, &tokens, vocab)?;  // Returns Vec<f32>
```

### Example 3: Per-Position Logits (Current Workaround)

```rust
// Current: Evaluate each position separately
let mut all_logits = Vec::new();
let mut ctx = Context::new(&model, 2048, 512, 1)?;

for i in 1..=tokens.len() {
    ctx.eval(&tokens[..i], 0)?;
    all_logits.push(ctx.get_logits()?);
}

// Proposed: One call
ctx.eval(&tokens, 0)?;
let all_logits = ctx.get_all_logits(tokens.len())?;  // ← Already exists!
```

---

## Return Type Breakdown

### Token IDs
- **Input format**: `&[i32]` (i32 for llama.cpp compatibility)
- **Output format**: `Vec<i32>`
- **Conversion to u32**: `tokens.iter().map(|t| *t as u32).collect()`

### Logits
- **Single position**: `Vec<f32>` where len() == vocab_size
- **Multiple positions**: `Vec<Vec<f32>>` where outer.len() == num_positions, inner.len() == vocab_size
- **Current limitation**: Most API functions return single position only

### Errors
- `Result<T>` type: `std::result::Result<T, CppError>`
- Error variants: `NullPointer`, `InvalidUtf8`, `LlamaError`, `ModelLoadError`

---

## Function Call Pattern

### Direct Token Evaluation Pattern

```rust
// Pattern 1: String → Tokens → Logits
let tokens = session.tokenize("prompt")?;
let logits = session.eval_and_get_logits(&tokens, 0)?;

// Pattern 2: Tokens → Logits (skip tokenization)
let tokens = vec![1, 2, 3];  // Pre-tokenized
let logits = session.eval_and_get_logits(&tokens, 0)?;

// Pattern 3: Multi-step generation
let mut tokens = session.tokenize("prompt")?;
for _ in 0..max_new_tokens {
    let logits = session.eval_and_get_logits(&tokens, 0)?;
    let next_token = context.sample_greedy(&logits);
    tokens.push(next_token);
}
```

---

## Key Integration Points

### For per-position logits support, expose this:

```rust
// In bitnet-sys/src/wrapper.rs
impl Session {
    pub fn eval_and_get_all_logits(&mut self, tokens: &[i32], n_past: i32) -> Result<Vec<Vec<f32>>> {
        self.context.eval(tokens, n_past)?;
        self.context.get_all_logits(tokens.len())  // Calls existing line 285
    }
}
```

### For crossval integration, use this:

```rust
// In crossval/src/cpp_bindings.rs
pub struct CppModel {
    session: bitnet_sys::wrapper::Session,  // Replace *mut c_void
}

impl CppModel {
    pub fn eval_tokens(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let ids: Vec<i32> = tokens.iter().map(|t| *t as i32).collect();
        self.session.eval_and_get_logits(&ids, 0)
    }
}
```

