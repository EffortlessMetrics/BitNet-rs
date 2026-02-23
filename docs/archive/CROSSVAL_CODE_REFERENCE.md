# Cross-Validation Code Reference

Quick lookup for code snippets, locations, and implementation details.

## Command Registration & Dispatch

### Command Enum Definition
**File**: `xtask/src/main.rs:405-430`
```rust
#[cfg(feature = "inference")]
#[command(name = "crossval-per-token")]
CrossvalPerToken {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 4)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.999)]
    cos_tol: f32,
    #[arg(long, default_value = "text")]
    format: String,
},
```

### Command Dispatch
**File**: `xtask/src/main.rs:895-898`
```rust
Commands::CrossvalPerToken { model, tokenizer, prompt, max_tokens, cos_tol, format } => {
    crossval_per_token_cmd(&model, &tokenizer, &prompt, max_tokens, cos_tol, &format)?;
},
```

---

## Tokenization

### Rust Tokenizer (Raw)
**File**: `xtask/src/main.rs:2920-2922`
```rust
let tokenizer = bitnet_tokenizers::loader::load_tokenizer(tokenizer_path)?;
let tokens = tokenizer.encode(prompt, false, false)?;  // add_bos=false, add_special=false
let token_ids: Vec<i32> = tokens.iter().map(|&id| id as u32).collect();
```

### C++ Tokenizer (llama.cpp)
**File**: `crates/bitnet-sys/src/wrapper.rs:144-186`
```rust
pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
    let c_text = CString::new(text)?;
    let model = unsafe { llama_get_model(self.ptr) };

    // Two-pass tokenization via llama_tokenize()
    let n_tokens = unsafe {
        llama_tokenize(
            model,
            c_text.as_ptr(),
            text.len() as i32,
            ptr::null_mut(),
            0,
            add_special,
            false, // parse_special
        )
    };

    if n_tokens < 0 {
        return Err(CppError::LlamaError("Tokenization failed".to_string()));
    }

    // Second call to get actual tokens
    let mut tokens = vec![0i32; n_tokens as usize];
    let actual_n = unsafe {
        llama_tokenize(
            model,
            c_text.as_ptr(),
            text.len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
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

### Session Tokenize Wrapper
**File**: `crates/bitnet-sys/src/wrapper.rs:351-352`
```rust
pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
    self.context.tokenize(text, true)  // add_special=true
}
```

---

## Logits Evaluation

### Rust Logits (All Positions)
**File**: `crates/bitnet-inference/src/parity.rs:157-223`
```rust
pub fn eval_logits_all_positions(model_path: &str, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
    // Load model tensors with Rust GGUF loader
    let (config, model) = match load_gguf_full(
        Path::new(model_path),
        Device::Cpu,
        bitnet_models::GGUFLoaderConfig::default(),
    ) {
        Ok(result) => {
            // Convert i2s_qk256 map to raw_tensors map with key remapping
            let mut raw_tensors_unmapped = std::collections::HashMap::new();
            for (key, qk256_tensor) in result.i2s_qk256.iter() {
                let raw_bytes_tensor = candle_core::Tensor::from_raw_buffer(
                    &qk256_tensor.qs,
                    candle_core::DType::U8,
                    &[qk256_tensor.rows, qk256_tensor.row_stride_bytes],
                    &candle_core::Device::Cpu,
                )?;

                let qk256_key = format!("{}.qk256_qs", key);
                raw_tensors_unmapped.insert(qk256_key, raw_bytes_tensor);
            }

            // Remap keys from GGUF format to model format
            let raw_tensors = bitnet_models::weight_mapper::remap_gguf_weights_with_options(
                &raw_tensors_unmapped,
                false, // non-strict
            )?;

            let model = BitNetModel::from_gguf(
                result.config.clone(),
                result.tensors,
                raw_tensors,
                Device::Cpu,
            )?;
            (result.config, model)
        }
        Err(e) => anyhow::bail!("Failed to load GGUF model: {}", e),
    };

    // Convert i32 tokens to u32
    let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();

    // Get embeddings for the tokens
    let embedded = model.embed(&tokens_u32)?;

    // Create KV cache
    let cache = KVCache::new(&config, 1, &candle_core::Device::Cpu)?;
    let mut any_cache: Box<dyn std::any::Any> = Box::new(cache);

    // Run forward pass through the model to get output for all positions
    let output = model.forward(&embedded, any_cache.as_mut())?;

    // Get logits tensor from the output [B, T, V]
    let logits_tensor = model.logits(&output)?;

    // Extract per-position logits into Vec<Vec<f32>>
    let seq_len = tokens.len();
    extract_all_position_logits(logits_tensor, seq_len)
}
```

### C++ Logits (All Positions)
**File**: `crates/bitnet-sys/src/wrapper.rs:285-293`
```rust
pub fn get_all_logits(&self, n_tokens: usize) -> Result<Vec<Vec<f32>>> {
    let mut all_logits = Vec::with_capacity(n_tokens);

    for i in 0..n_tokens {
        all_logits.push(self.get_logits_ith(i as i32)?);
    }

    Ok(all_logits)
}
```

### Get Logits for Specific Position
**File**: `crates/bitnet-sys/src/wrapper.rs:270-282`
```rust
pub fn get_logits_ith(&self, i: i32) -> Result<Vec<f32>> {
    let model = unsafe { llama_get_model(self.ptr) };
    let n_vocab = unsafe { llama_n_vocab(model) };

    let logits_ptr = unsafe { llama_get_logits_ith(self.ptr, i) };
    if logits_ptr.is_null() {
        return Err(CppError::NullPointer);
    }

    let logits = unsafe { slice::from_raw_parts(logits_ptr, n_vocab as usize) };

    Ok(logits.to_vec())
}
```

---

## FFI Session Management

### Session Load (Deterministic Settings)
**File**: `crates/bitnet-sys/src/wrapper.rs:344-348`
```rust
pub fn load_deterministic(model_path: &str) -> Result<Self> {
    let model = Model::load(model_path)?;
    let context = Context::new(&model, 2048, 512, 1)?;  // 2048 context, 512 batch, 1 thread
    Ok(Session { model, context })
}
```

### FFI Availability Check
**File**: `crates/bitnet-sys/src/lib.rs:71-77`
```rust
pub fn is_available() -> bool {
    panic::catch_unwind(|| {
        // Try calling a simple function to ensure the library is linked.
        wrapper::get_version();
    }).is_ok()
}
```

### FFI Initialization & Cleanup
**File**: `xtask/src/main.rs:2944-2952`
```rust
if !bitnet_sys::is_available() {
    anyhow::bail!(
        "C++ FFI not available. Compile with --features crossval or set BITNET_CPP_DIR"
    );
}

// Use C++ wrapper with proper initialization
bitnet_sys::wrapper::init_backend();
let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
```

### Session Creation
**File**: `xtask/src/main.rs:2954-2963`
```rust
let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;

// Tokenize with C++ tokenizer
let cpp_tokens = cpp_session.tokenize(prompt)?;

// Evaluate all positions
cpp_session.context.eval(&cpp_tokens, 0)?;

// Get all logits (requires logits_all=true in context)
let cpp_logits = cpp_session.context.get_all_logits(cpp_tokens.len())?;
```

---

## Comparison Logic

### Per-Position Logits Comparison
**File**: `crossval/src/logits_compare.rs:49-102`
```rust
pub fn compare_per_position_logits(
    rs_logits: &[Vec<f32>],
    cpp_logits: &[Vec<f32>],
) -> LogitsDivergence {
    let n_positions = rs_logits.len().min(cpp_logits.len());

    let mut per_token_cosine_sim = Vec::with_capacity(n_positions);
    let mut per_token_l2_dist = Vec::with_capacity(n_positions);
    let mut max_absolute_diff = 0.0f32;
    let mut first_divergence_token = None;

    for pos in 0..n_positions {
        let rs_vec = &rs_logits[pos];
        let cpp_vec = &cpp_logits[pos];

        // Skip if vector sizes don't match
        if rs_vec.len() != cpp_vec.len() {
            per_token_cosine_sim.push(0.0);
            per_token_l2_dist.push(f32::INFINITY);
            if first_divergence_token.is_none() {
                first_divergence_token = Some(pos);
            }
            continue;
        }

        // Calculate cosine similarity
        let cosine_sim = cosine_similarity(rs_vec, cpp_vec);
        per_token_cosine_sim.push(cosine_sim);

        // Calculate L2 distance
        let l2_dist = l2_distance(rs_vec, cpp_vec);
        per_token_l2_dist.push(l2_dist);

        // Track max absolute difference
        let max_diff_at_pos = rs_vec.iter().zip(cpp_vec.iter())
            .map(|(r, c)| (r - c).abs())
            .fold(0.0f32, f32::max);

        if max_diff_at_pos > max_absolute_diff {
            max_absolute_diff = max_diff_at_pos;
        }

        // Check if this is the first divergence
        if first_divergence_token.is_none() && (1.0 - cosine_sim) > COSINE_SIMILARITY_THRESHOLD {
            first_divergence_token = Some(pos);
        }
    }

    LogitsDivergence {
        first_divergence_token,
        per_token_cosine_sim,
        per_token_l2_dist,
        max_absolute_diff,
    }
}
```

### Cosine Similarity Calculation
**File**: `crossval/src/logits_compare.rs:107-123`
```rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Avoid division by zero
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}
```

### L2 Distance Calculation
**File**: `crossval/src/logits_compare.rs:125-139`
```rust
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }

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

---

## Command Output

### Text Output Format (Lines 2992-3005)
**File**: `xtask/src/main.rs:2992-3005`
```rust
for (t, (&cosine, &l2)) in divergence
    .per_token_cosine_sim
    .iter()
    .zip(divergence.per_token_l2_dist.iter())
    .enumerate()
{
    let ok = cosine >= cos_tol;
    let symbol = if ok { "✓" } else { "✗" };
    println!("{} t={} cosine={:.6} l2={:.2e}", symbol, t, cosine, l2);

    if !ok && divergence.first_divergence_token == Some(t) {
        println!("   ↑ First divergence detected at token {}", t);
    }
}
```

### JSON Output Format (Lines 2978-2988)
**File**: `xtask/src/main.rs:2978-2988`
```rust
let output = serde_json::json!({
    "first_divergence_token": divergence.first_divergence_token,
    "per_token_cosine_sim": divergence.per_token_cosine_sim,
    "per_token_l2_dist": divergence.per_token_l2_dist,
    "max_absolute_diff": divergence.max_absolute_diff,
    "threshold": cos_tol,
    "status": if divergence.first_divergence_token.is_none() { "ok" } else { "diverged" }
});
println!("{}", serde_json::to_string_pretty(&output)?);
```

---

## Memory Safety Patterns

### Model Drop Implementation
**File**: `crates/bitnet-sys/src/wrapper.rs:102-113`
```rust
impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: Use mem::replace to prevent double-free
        // Only free if pointer is non-null and we have ownership
        let ptr = std::mem::replace(&mut self.ptr, std::ptr::null_mut());
        if !ptr.is_null() {
            unsafe {
                llama_free_model(ptr);
            }
        }
    }
}
```

### Context Drop Implementation
**File**: `crates/bitnet-sys/src/wrapper.rs:316-327`
```rust
impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: Use mem::replace to prevent double-free
        // Only free if pointer is non-null and we have ownership
        let ptr = std::mem::replace(&mut self.ptr, std::ptr::null_mut());
        if !ptr.is_null() {
            unsafe {
                llama_free(ptr);
            }
        }
    }
}
```

---

## Key Constants & Thresholds

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `COSINE_SIMILARITY_THRESHOLD` | `1e-4` | `crossval/src/logits_compare.rs:25` | Hardcoded divergence detection threshold |
| `n_ctx` | `2048` | `crates/bitnet-sys/src/wrapper.rs:346` | Context window for C++ session |
| `n_batch` | `512` | `crates/bitnet-sys/src/wrapper.rs:346` | Batch size for C++ session |
| `n_threads` | `1` | `crates/bitnet-sys/src/wrapper.rs:346` | Threads for determinism |
| `max_tokens` (default) | `4` | `xtask/src/main.rs:420` | Default max tokens for generation |
| `cos_tol` (default) | `0.999` | `xtask/src/main.rs:424` | Default cosine similarity tolerance |
| `add_bos` (Rust) | `false` | `xtask/src/main.rs:2921` | Rust tokenizer: no BOS |
| `add_special` (Rust) | `false` | `xtask/src/main.rs:2921` | Rust tokenizer: no special tokens |
| `add_special` (C++) | `true` | `crates/bitnet-sys/src/wrapper.rs:352` | C++ tokenizer: includes special handling |

---

## Error Codes

| Code | Meaning | Source |
|------|---------|--------|
| 0 | Success, no divergence | `xtask/src/main.rs:3036` |
| 1 | Divergence detected | `xtask/src/main.rs:3033` |
| 2 | Pre-gate: Token mismatch | Proposed (from design doc) |
| 3 | Pre-gate: Logits validation failed | Proposed (from design doc) |

---

## Related Documentation

- **Implementation Analysis**: `docs/CROSSVAL_PER_TOKEN_IMPLEMENTATION.md`
- **Pre-Gate Design**: `docs/CROSSVAL_TOKEN_PARITY_PRE_GATE_DESIGN.md`
- **Project Status**: `CLAUDE.md`
- **Existing Cross-Validation Docs**: `docs/CROSSVAL.md`, `docs/CROSSVAL_TESTING.md`

