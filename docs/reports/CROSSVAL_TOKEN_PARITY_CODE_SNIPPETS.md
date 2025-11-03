# Cross-Validation Token Parity: Code Snippets & Analysis

**Date**: 2025-10-24  
**Focus**: Critical code sections showing gap and solutions

## 1. THE GAP: CLI Command Missing Token Validation

### File: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` (lines ~410-450)

**Current Implementation (Incomplete)**:

```rust
fn crossval_per_token_cmd(
    model_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    _max_tokens: usize,
    cos_tol: f32,
    format: &str,
) -> Result<()> {
    use bitnet_crossval::logits_compare::compare_per_position_logits;
    use bitnet_inference::parity::eval_logits_all_positions;
    
    println!("🔍 Per-token logits parity check");
    println!("Model: {}", model_path.display());
    println!("Prompt: \"{}\"", prompt);
    println!("Cosine tolerance: {}", cos_tol);
    println!();
    
    // Tokenize the prompt
    println!("📝 Tokenizing prompt...");
    let tokenizer = bitnet_tokenizers::loader::load_tokenizer(tokenizer_path)?;
    let tokens = tokenizer.encode(prompt, false, false)?;
    //                                                ^^^^^ ^^^^^ 
    //                                          add_bos=false, add_eos=false
    //                                          NO SPECIAL TOKENS
    let token_ids: Vec<i32> = tokens.iter().map(|&id| id as i32).collect();
    
    // Limit to prompt tokens only (no generation in this mode)
    let total_len = token_ids.len();
    println!("Tokens: {} (prompt)", total_len);
    println!();
    
    // Get Rust logits
    println!("🦀 Evaluating Rust logits for all positions...");
    let model_path_str =
        model_path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid model path"))?;
    let rust_logits = eval_logits_all_positions(model_path_str, &token_ids)?;
    println!(
        "✓ Rust: {} positions, vocab_size={}",
        rust_logits.len(),
        rust_logits.first().map(|v| v.len()).unwrap_or(0)
    );
    
    // Get C++ logits
    println!("🔧 Evaluating C++ logits for all positions...");
    
    // Check if C++ is available
    if !bitnet_sys::is_available() {
        anyhow::bail!(
            "C++ FFI not available. Compile with --features crossval or set BITNET_CPP_DIR"
        );
    }
    
    // Use C++ wrapper with proper initialization
    bitnet_sys::wrapper::init_backend();
    let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
    let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;
    
    // Tokenize with C++ tokenizer
    let cpp_tokens = cpp_session.tokenize(prompt)?;
    //                                      ^ add_special defaults to true!
    //                                      INCLUDES BOS/EOS - MISMATCH!
    
    // ❌ MISSING: Token validation here!
    // if token_ids != cpp_tokens {
    //     eprintln!("ERROR: Token sequence mismatch!");
    //     eprintln!("Rust: {:?}", token_ids);
    //     eprintln!("C++:  {:?}", cpp_tokens);
    //     return Err(...);
    // }
    
    // Evaluate all positions
    cpp_session.context.eval(&cpp_tokens, 0)?;
    
    // Get all logits (requires logits_all=true in context)
    let cpp_logits = cpp_session.context.get_all_logits(cpp_tokens.len())?;
    println!(
        "✓ C++: {} positions, vocab_size={}",
        cpp_logits.len(),
        cpp_logits.first().map(|v| v.len()).unwrap_or(0)
    );
    println!();
    
    // Compare per-position
    println!("📊 Comparing logits per position...");
    let divergence = compare_per_position_logits(&rust_logits, &cpp_logits);
    
    // ... output results ...
    Ok(())
}
```

**Issue**: After obtaining `token_ids` and `cpp_tokens`, the code proceeds directly to logits comparison without validating they match.

---

## 2. THE SOLUTION: Token Pre-Gate Validation

### Proposed Addition (Insert after line ~440)

```rust
// Validate token sequences match BEFORE logits comparison
println!("✅ Validating token sequences...");
if token_ids.len() != cpp_tokens.len() {
    eprintln!("❌ Token sequence length mismatch:");
    eprintln!("  Rust: {} tokens", token_ids.len());
    eprintln!("  C++:  {} tokens", cpp_tokens.len());
    std::process::exit(1);
}

// Check for token ID divergence
let mut first_divergence = None;
for (i, (rust_id, cpp_id)) in token_ids.iter().zip(cpp_tokens.iter()).enumerate() {
    if rust_id != cpp_id {
        if first_divergence.is_none() {
            first_divergence = Some(i);
        }
        if i < 5 || i == first_divergence.unwrap() {
            eprintln!("  Position {}: Rust={} (0x{:04x}), C++={} (0x{:04x})",
                      i, rust_id, rust_id, cpp_id, cpp_id);
        }
    }
}

if let Some(div_pos) = first_divergence {
    eprintln!("❌ Token IDs diverge at position {}", div_pos);
    eprintln!("   Tokens before divergence match: {}", div_pos);
    eprintln!();
    eprintln!("Full Rust tokens: {:?}", token_ids);
    eprintln!("Full C++ tokens:  {:?}", cpp_tokens);
    std::process::exit(1);
}

println!("✅ Token sequences match: {} tokens", token_ids.len());
println!();
```

---

## 3. THE ROOT CAUSE: Special Token Flag Mismatch

### CLI Command Tokenization (main.rs)

```rust
let tokens = tokenizer.encode(prompt, false, false)?;
//                                       ^^^^^ ^^^^^ 
//                                       NO SPECIAL TOKENS
```

**Expected tokens for prompt "2+2="**: `[882, 28754, ...]`
**With BOS prepended**: `[128000, 882, 28754, ...]`

### C++ Tokenization Wrapper (wrapper.rs:145)

```rust
pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
    // ... implementation ...
    // Returns Vec<i32> with BOS/EOS when add_special=true
}
```

### C++ Session Call (main.rs:440)

```rust
let cpp_tokens = cpp_session.tokenize(prompt)?;
//               This uses default add_special=true, see Context::tokenize:

// From wrapper.rs:157
pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
    self.context.tokenize(text, true)  // add_special=true!
}
```

**Result**:
- Rust: `[882, 28754, ...]` (no BOS)
- C++: `[128000, 882, 28754, ...]` (BOS added)

This explains the duplicate BOS finding in NEXT_STEPS.md.

---

## 4. TEST IMPLEMENTATION: How It SHOULD Work

### File: `/home/steven/code/Rust/BitNet-rs/crossval/tests/parity.rs` (lines 141-196)

```rust
#[test]
fn test_tokenization_parity() -> Result<()> {
    let model_path = match test_model_path() {
        Some(p) => p,
        None => return Ok(()),
    };

    // Initialize C++ backend
    wrapper::init_backend();
    let _guard = scopeguard::guard((), |_| wrapper::free_backend());

    // Prepare tokenizer
    let gguf_bytes = std::fs::read(&model_path)?;
    let reader = GgufReader::new(&gguf_bytes)?;
    let tokenizer = load_tokenizer_from_gguf_reader(&reader)?;

    // Test various prompts
    let test_prompts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "1 + 1 = ",
        "def fibonacci(n):",
        "🦀 Rust is awesome! 🚀",
    ];

    for prompt in &test_prompts {
        // Load fresh C++ session for each prompt to avoid residual state
        let mut cpp_session = CppSession::load_deterministic(&model_path)?;
        let cpp_tokens = cpp_session.tokenize(prompt)?;
        //                                       ^ add_special=true (default)

        // Tokenize with Rust
        let rust_tokens_u32 = tokenizer.encode(prompt, true, true)?;
        //                                              ^^^ ^^^
        //                                       SPECIAL TOKENS INCLUDED
        let rust_tokens: Vec<i32> = rust_tokens_u32
            .iter()
            .map(|&t| i32::try_from(t).expect("Token ID too large for i32"))
            .collect();
            
        println!("Prompt: {:?}", prompt);
        println!("C++ tokens: {:?}", cpp_tokens);
        println!("Rust tokens: {:?}", rust_tokens);

        // ✅ VALIDATION: Ensure tokenization parity
        assert_eq!(rust_tokens, cpp_tokens, "Tokenization mismatch for: {}", prompt);
        
        // ✅ Only proceed to logits comparison if tokens match
        // Get logits from both implementations
        let cpp_logits = cpp_session.eval_and_get_logits(&cpp_tokens, 0)?;
        let rust_logits = eval_logits_once(&model_path, &rust_tokens)?;
        compare_logits(&rust_logits, &cpp_logits, 0)?;

        // ✅ Compare next-token selection (argmax)
        let cpp_next = argmax(&cpp_logits);
        let rust_next = argmax(&rust_logits);
        assert_eq!(rust_next, cpp_next, "Next token mismatch for: {}", prompt);
    }

    Ok(())
}
```

**Key difference**: Tests use `encode(..., true, true)` to match C++ `add_special=true`.

---

## 5. LOGITS COMPARISON MODULE

### File: `/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs`

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
        let max_diff_at_pos =
            rs_vec.iter().zip(cpp_vec.iter()).map(|(r, c)| (r - c).abs()).fold(0.0f32, f32::max);

        if max_diff_at_pos > max_absolute_diff {
            max_absolute_diff = max_diff_at_pos;
        }

        // Check if this is the first divergence (cosine similarity too low)
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

## 6. RUST LOGITS EVALUATION

### File: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/parity.rs` (lines 157-223)

```rust
pub fn eval_logits_all_positions(model_path: &str, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
    // Load model tensors with Rust GGUF loader (fail-closed, no FFI routing)
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
        Err(e) => {
            anyhow::bail!("Failed to load GGUF model: {}", e);
        }
    };

    // Convert i32 tokens to u32
    let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();

    // Get embeddings for the tokens
    let embedded = model.embed(&tokens_u32)?;

    // Create KV cache for the model (batch size 1, CPU)
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

fn extract_all_position_logits(
    logits: bitnet_common::ConcreteTensor,
    seq_len: usize,
) -> Result<Vec<Vec<f32>>> {
    use bitnet_common::ConcreteTensor;

    match logits {
        ConcreteTensor::BitNet(tensor) => {
            let candle_tensor = tensor.as_candle();
            let dims = candle_tensor.dims();
            
            if dims.len() != 3 {
                anyhow::bail!("Expected rank-3 logits tensor [B,T,V], got rank {}", dims.len());
            }

            let batch_size = dims[0];
            let tensor_seq_len = dims[1];
            let vocab_size = dims[2];

            if batch_size != 1 {
                anyhow::bail!(
                    "Expected batch_size=1 for per-position extraction, got {}",
                    batch_size
                );
            }

            if tensor_seq_len < seq_len {
                anyhow::bail!(
                    "Tensor seq_len {} is less than requested {}",
                    tensor_seq_len,
                    seq_len
                );
            }

            // Extract each position
            let mut all_positions = Vec::with_capacity(seq_len);

            for t in 0..seq_len {
                // Extract logits for position t: [B,T,V] -> [B,1,V] -> [V]
                let position_logits = candle_tensor
                    .narrow(1, t, 1)? // [B,1,V]
                    .squeeze(0)? // [1,V]
                    .squeeze(0)?; // [V]

                let position_logits = if position_logits.dtype() != DType::F32 {
                    position_logits.to_dtype(DType::F32)?
                } else {
                    position_logits.clone()
                };

                let logits_vec = position_logits.to_vec1::<f32>()?;

                if logits_vec.len() != vocab_size {
                    anyhow::bail!(
                        "Position {} logits has length {}, expected vocab_size {}",
                        t,
                        logits_vec.len(),
                        vocab_size
                    );
                }

                all_positions.push(logits_vec);
            }

            Ok(all_positions)
        }
        ConcreteTensor::Mock(mock) => {
            // For mock tensors, return zeros for each position
            let vocab_size = mock.shape()[2];
            let all_positions = vec![vec![0.0f32; vocab_size]; seq_len];
            Ok(all_positions)
        }
    }
}
```

---

## 7. C++ TOKENIZATION WRAPPER

### File: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/src/wrapper.rs` (lines 144-186)

```rust
pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
    let c_text = CString::new(text)?;
    let model = unsafe { llama_get_model(self.ptr) };

    // First call to get the number of tokens
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

    // Second call to get the actual tokens
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

// Called from Session::tokenize()
pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
    self.context.tokenize(text, true)  // ← add_special=true by default!
}
```

---

## Summary of Changes Needed

| File | Line | Issue | Fix |
|------|------|-------|-----|
| `xtask/src/main.rs` | ~440 | No token validation | Add `if token_ids != cpp_tokens { exit() }` |
| `xtask/src/main.rs` | ~410 | `encode(prompt, false, false)` | Change to `encode(prompt, true, true)` |
| `bitnet-sys/src/wrapper.rs` | ~147 | `tokenize(text, true)` | Consider `tokenize(text, false)` for consistency |

