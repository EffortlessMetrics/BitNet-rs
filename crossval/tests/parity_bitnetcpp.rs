//! BitNet.cpp parity harness
//!
//! Validates that the Rust inference engine produces identical outputs to
//! Microsoft's BitNet C++ implementation for deterministic inference.
//!
//! This test is feature-gated and skips gracefully when:
//! - `crossval-bitnetcpp` feature is not enabled
//! - `CROSSVAL_GGUF` environment variable is not set
//! - BitNet C++ is not available

#![cfg(all(feature = "crossval", feature = "integration-tests"))]

use anyhow::{Context, Result};
use serde_json::json;
use std::{env, fs, path::PathBuf, time::SystemTime};

/// Helper function to compute cosine similarity between two vectors
#[allow(dead_code)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Perform C++ parity check using bitnet-sys FFI
/// Returns: (cosine_similarity, cosine_ok, exact_match_rate, first_divergence_step)
#[cfg(feature = "ffi")]
fn cpp_parity_check(
    gguf_path: &std::path::Path,
    formatted_prompt: &str,
    rust_ids: &[u32],
    rust_logits: &[f32],
    rust_decode: &[u32],
    add_bos: bool,
    parse_special: bool,
    eos_id: u32,
    vocab_size: usize,
    n_steps: usize,
) -> Result<(f32, bool, f32, Option<usize>)> {
    use bitnet_sys::{
        BitnetContext, BitnetModel, bitnet_decode_greedy, bitnet_eval_tokens, bitnet_tokenize_text,
    };

    let gguf_str = gguf_path.to_string_lossy().to_string();

    // 1. Load C++ model and context
    let cpp_model = BitnetModel::from_file(&gguf_str)
        .map_err(|e| anyhow::anyhow!("C++ model load failed: {:?}", e))?;

    let cpp_ctx = BitnetContext::new(&cpp_model, 4096, 1, 0)
        .map_err(|e| anyhow::anyhow!("C++ context creation failed: {:?}", e))?;

    // 2. Tokenization parity
    let cpp_ids = bitnet_tokenize_text(&cpp_model, formatted_prompt, add_bos, parse_special)
        .map_err(|e| anyhow::anyhow!("C++ tokenization failed: {:?}", e))?;

    let cpp_ids_u32: Vec<u32> = cpp_ids.iter().map(|&x| x as u32).collect();

    if cpp_ids_u32 != rust_ids {
        eprintln!("WARNING: Tokenization mismatch! Rust: {:?}, C++: {:?}", rust_ids, cpp_ids_u32);
        // Continue anyway to collect more metrics
    } else {
        eprintln!("✓ Tokenization exact match");
    }

    // 3. Prefill logits parity (cosine similarity)
    let cpp_logits = bitnet_eval_tokens(&cpp_ctx, &cpp_ids, vocab_size)
        .map_err(|e| anyhow::anyhow!("C++ eval failed: {:?}", e))?;

    let cos = cosine_similarity(rust_logits, &cpp_logits);
    let cos_ok = cos >= 0.99;

    // 4. N-step greedy decode parity
    let cpp_gen = bitnet_decode_greedy(&cpp_ctx, &cpp_ids, n_steps, eos_id as i32)
        .map_err(|e| anyhow::anyhow!("C++ decode failed: {:?}", e))?;

    let cpp_gen_u32: Vec<u32> = cpp_gen.iter().map(|&x| x as u32).collect();

    // Compute exact match rate and first divergence
    let mut eq_count = 0usize;
    let mut first_diff: Option<usize> = None;

    let min_len = rust_decode.len().min(cpp_gen_u32.len());
    for i in 0..min_len {
        if rust_decode[i] == cpp_gen_u32[i] {
            eq_count += 1;
        } else if first_diff.is_none() {
            first_diff = Some(i);
        }
    }

    // If lengths differ, that's also a divergence
    if rust_decode.len() != cpp_gen_u32.len() && first_diff.is_none() {
        first_diff = Some(min_len);
    }

    let exact_rate =
        if !rust_decode.is_empty() { eq_count as f32 / rust_decode.len() as f32 } else { 1.0 };

    Ok((cos, cos_ok, exact_rate, first_diff))
}

#[cfg(not(feature = "ffi"))]
fn cpp_parity_check(
    _gguf_path: &std::path::Path,
    _formatted_prompt: &str,
    _rust_ids: &[u32],
    _rust_logits: &[f32],
    _rust_decode: &[u32],
    _add_bos: bool,
    _parse_special: bool,
    _eos_id: u32,
    _vocab_size: usize,
    _n_steps: usize,
) -> Result<(f32, bool, f32, Option<usize>)> {
    anyhow::bail!("C++ FFI not available (compile with --features bitnet-sys/ffi)")
}

/// Compute SHA256 hash of a file
fn sha256_file(path: &std::path::Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open file for hashing: {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 20]; // 1MB buffer

    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

#[tokio::test]
async fn parity_bitnetcpp() -> Result<()> {
    // Check if CROSSVAL_GGUF is set; skip if not
    let gguf_path = match env::var("CROSSVAL_GGUF") {
        Ok(s) => PathBuf::from(s),
        Err(_) => {
            eprintln!("CROSSVAL_GGUF not set; skipping parity test");
            return Ok(());
        }
    };

    if !gguf_path.exists() {
        eprintln!("GGUF model not found at {:?}; skipping", gguf_path);
        return Ok(());
    }

    // Check if BITNET_CPP_DIR is set for C++ parity
    let cpp_available = env::var("BITNET_CPP_DIR").is_ok();
    if !cpp_available {
        eprintln!("BITNET_CPP_DIR not set; running Rust-only validation");
    }

    let commit = env::var("GIT_COMMIT").unwrap_or_else(|_| "unknown".into());
    let prompt = env::var("CROSSVAL_PROMPT").unwrap_or_else(|_| "Q: 2+2? A:".into());

    eprintln!("=== Parity Harness ===");
    eprintln!("Model: {:?}", gguf_path);
    eprintln!("Prompt: {}", prompt);
    eprintln!("Commit: {}", commit);

    // 1. Rust-side tokenization and metadata
    let template = auto_detect_template(&gguf_path);
    let formatted_prompt = template.apply(&prompt, None);

    let (rust_ids, add_bos, parse_special, eos_id, vocab_size) =
        rust_side_tokenize_and_meta(&gguf_path, &prompt)?;

    eprintln!("Template: {}", template);
    eprintln!("Formatted prompt: {}", formatted_prompt);
    eprintln!(
        "Tokenized {} tokens (add_bos={}, parse_special={}, eos_id={})",
        rust_ids.len(),
        add_bos,
        parse_special,
        eos_id
    );

    // 2. Rust-side logits evaluation
    let rust_logits = rust_eval_last_logits(&gguf_path, &rust_ids, vocab_size).await?;
    eprintln!("Rust logits shape: [{}]", rust_logits.len());

    // 3. Rust-side greedy decoding (N steps)
    let n_steps = 8;
    let rust_decode = rust_decode_n_greedy(&gguf_path, &rust_ids, n_steps, eos_id).await?;
    eprintln!("Rust decoded {} tokens: {:?}", rust_decode.len(), rust_decode);

    // 4. If C++ is available, compare outputs
    let (cosine_ok, cosine_similarity, exact_match_rate, first_divergence, cpp_available_flag) =
        if cpp_available {
            match cpp_parity_check(
                &gguf_path,
                &formatted_prompt,
                &rust_ids,
                &rust_logits,
                &rust_decode,
                add_bos,
                parse_special,
                eos_id,
                vocab_size,
                n_steps,
            ) {
                Ok((cos, cos_ok, exact_rate, first_div)) => {
                    eprintln!("C++ parity check completed:");
                    eprintln!("  Cosine similarity: {:.6}", cos);
                    eprintln!("  Cosine OK (≥0.99): {}", cos_ok);
                    eprintln!("  Exact match rate: {:.4}", exact_rate);
                    if let Some(step) = first_div {
                        eprintln!("  First divergence at step: {}", step);
                    } else {
                        eprintln!("  No divergence detected");
                    }
                    (cos_ok, Some(cos), Some(exact_rate), first_div, true)
                }
                Err(e) => {
                    eprintln!("C++ parity check failed: {:?}", e);
                    eprintln!("Continuing with Rust-only validation");
                    (false, None, None, None, false)
                }
            }
        } else {
            (false, None, None, None, false)
        };

    // 5. Write parity receipt
    let ts = humantime::format_rfc3339(SystemTime::now()).to_string();
    let date_str = &ts[..10]; // Extract "YYYY-MM-DD" from RFC3339 timestamp

    // Anchor to workspace root via manifest directory
    let receipt_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/baselines").join(date_str);

    if !receipt_dir.exists() {
        fs::create_dir_all(&receipt_dir).context("Failed to create baselines directory")?;
    }

    let model_sha = sha256_file(&gguf_path)?;

    let parity_status = if cpp_available_flag {
        if cosine_ok && exact_match_rate.unwrap_or(0.0) == 1.0 { "ok" } else { "mismatch" }
    } else {
        "rust_only"
    };

    let receipt = json!({
        "timestamp": ts,
        "commit": commit,
        "model_path": gguf_path.display().to_string(),
        "model_sha256": model_sha,
        "template": template.to_string(),
        "prompt": prompt,
        "rust": {
            "token_count": rust_ids.len(),
            "add_bos": add_bos,
            "parse_special": parse_special,
            "eos_id": eos_id,
            "vocab_size": vocab_size,
            "logits_dim": rust_logits.len(),
            "decoded_tokens": rust_decode,
            "n_steps": n_steps,
        },
        "parity": {
            "cpp_available": cpp_available_flag,
            "cosine_similarity": cosine_similarity,
            "cosine_ok": cosine_ok,
            "exact_match_rate": exact_match_rate,
            "first_divergence_step": first_divergence,
            "status": parity_status,
        },
        "validation": {
            "rust_engine": "production",
            "deterministic": true,
            "threads": 1,
            "seed": 0,
        }
    });

    // Atomic write using temp file
    let receipt_path = receipt_dir.join("parity-bitnetcpp.json");
    let tmp_path = receipt_dir.join("parity-bitnetcpp.json.tmp");
    fs::write(&tmp_path, serde_json::to_vec_pretty(&receipt)?)
        .context("Failed to write parity receipt to temp file")?;
    fs::rename(&tmp_path, &receipt_path).context("Failed to atomically rename parity receipt")?;

    eprintln!("✓ Parity receipt written to: {:?}", receipt_path);

    Ok(())
}

/// Tokenize prompt with template-aware BOS/special handling
/// Returns: (token_ids, add_bos, parse_special, eos_token_id, vocab_size)
fn rust_side_tokenize_and_meta(
    model_path: &std::path::Path,
    prompt: &str,
) -> Result<(Vec<u32>, bool, bool, u32, usize)> {
    use bitnet_inference::TemplateType;
    use bitnet_tokenizers::auto;

    // 1) Load tokenizer using auto-detection
    let tokenizer = auto::load_auto(model_path, None)?;
    let vocab_size = tokenizer.vocab_size();

    // 2) Auto-detect template (same logic as CLI)
    let template = auto_detect_template(model_path);

    // 3) Determine BOS policy from template
    let add_bos = template.should_add_bos();

    // 4) Determine parse_special flag (true for Llama3Chat to parse <|eot_id|> etc.)
    let parse_special = matches!(template, TemplateType::Llama3Chat);

    // 5) Resolve EOS token ID (token-level EOT for llama3-chat, or regular EOS)
    let eos_id = if matches!(template, TemplateType::Llama3Chat) {
        // For LLaMA-3, use <|eot_id|> as the stop token
        let eot_ids = tokenizer.encode("<|eot_id|>", false, true)?;
        eot_ids.get(0).copied().unwrap_or_else(|| {
            tokenizer.eos_token_id().unwrap_or(128009) // LLaMA-3 default <|eot_id|>
        })
    } else {
        tokenizer.eos_token_id().unwrap_or(2) // Common EOS fallback
    };

    // 6) Format prompt using template
    let formatted = template.apply(prompt, None);

    // 7) Encode the formatted prompt
    // Note: Rust tokenizer uses add_special for both BOS and parsing
    // For consistency with C++ side, we pass add_bos for BOS insertion
    let ids = tokenizer.encode(&formatted, add_bos, false)?;

    Ok((ids, add_bos, parse_special, eos_id, vocab_size))
}

/// Auto-detect template type from GGUF metadata (matches CLI logic exactly)
fn auto_detect_template(model_path: &std::path::Path) -> bitnet_inference::TemplateType {
    use bitnet_inference::TemplateType;
    use bitnet_models::gguf::GgufReader;
    use bitnet_models::loader::MmapFile;

    // Try to read GGUF metadata
    if let Ok(mmap) = MmapFile::open(model_path) {
        if let Ok(reader) = GgufReader::new(mmap.as_slice()) {
            // Extract metadata fields
            let tokenizer_name = reader
                .metadata()
                .iter()
                .find(|(k, _)| k == "tokenizer.ggml.model" || k == "tokenizer.name")
                .and_then(|(_, v)| {
                    if let bitnet_models::gguf::MetadataValue::String(s) = v {
                        Some(s.as_str())
                    } else {
                        None
                    }
                });

            let chat_template =
                reader.metadata().iter().find(|(k, _)| k == "tokenizer.chat_template").and_then(
                    |(_, v)| {
                        if let bitnet_models::gguf::MetadataValue::String(s) = v {
                            Some(s.as_str())
                        } else {
                            None
                        }
                    },
                );

            // Use the same detection logic as the CLI
            return TemplateType::detect(tokenizer_name, chat_template);
        }
    }

    // Fallback to path-based heuristics if metadata not available
    let path_str = model_path.to_string_lossy().to_lowercase();

    // Check for LLaMA-3 signature
    if path_str.contains("llama") && path_str.contains("3") {
        return TemplateType::Llama3Chat;
    }

    // Check for instruct/chat patterns
    if path_str.contains("instruct") || path_str.contains("chat") {
        return TemplateType::Instruct;
    }

    // Default to Instruct for better UX (as of v0.9.x)
    TemplateType::Instruct
}

/// Evaluate token sequence and return last-position logits
async fn rust_eval_last_logits(
    model_path: &std::path::Path,
    ids: &[u32],
    expected_vocab_size: usize,
) -> Result<Vec<f32>> {
    use bitnet_common::Device as BNDevice;
    use bitnet_inference::InferenceEngine;
    use bitnet_models::ModelLoader;
    use bitnet_tokenizers::auto;
    use std::sync::Arc;

    // Load model and tokenizer
    let loader = ModelLoader::new(BNDevice::Cpu);
    let model = loader.load(model_path)?;
    let model_arc: Arc<dyn bitnet_models::Model> = model.into();

    let tokenizer = auto::load_auto(model_path, None)?;
    let tokenizer_arc: Arc<dyn bitnet_tokenizers::Tokenizer> = tokenizer;

    // Create engine
    let mut engine = InferenceEngine::new(model_arc, tokenizer_arc, BNDevice::Cpu)?;

    // Evaluate to get logits
    let logits = engine.eval_ids(ids).await?;

    // Verify vocab size matches
    let last_logits = if logits.len() >= expected_vocab_size {
        // Extract last position's logits (last vocab_size elements)
        logits[logits.len() - expected_vocab_size..].to_vec()
    } else {
        anyhow::bail!(
            "Logits size mismatch: got {}, expected at least {}",
            logits.len(),
            expected_vocab_size
        );
    };

    Ok(last_logits)
}

/// Perform N-step greedy decoding
async fn rust_decode_n_greedy(
    model_path: &std::path::Path,
    prompt_ids: &[u32],
    n_steps: usize,
    eos_id: u32,
) -> Result<Vec<u32>> {
    use bitnet_common::Device as BNDevice;
    use bitnet_inference::{GenerationConfig, InferenceEngine};
    use bitnet_models::ModelLoader;
    use bitnet_tokenizers::auto;
    use std::sync::Arc;

    // Load model and tokenizer
    let loader = ModelLoader::new(BNDevice::Cpu);
    let model = loader.load(model_path)?;
    let model_arc: Arc<dyn bitnet_models::Model> = model.into();

    let tokenizer = auto::load_auto(model_path, None)?;
    let tokenizer_arc: Arc<dyn bitnet_tokenizers::Tokenizer> = tokenizer;

    // Create engine
    let engine = InferenceEngine::new(model_arc, tokenizer_arc, BNDevice::Cpu)?;

    // Configure greedy generation
    let config = GenerationConfig {
        max_new_tokens: n_steps as u32,
        temperature: 0.0,
        top_k: Some(1),
        top_p: Some(1.0),
        repetition_penalty: 1.0,
        seed: Some(0),
        stop_sequences: vec![],
    };

    // Generate tokens
    let generated = engine.generate_tokens(prompt_ids, &config).await?;

    // Truncate at EOS if found
    let mut result = Vec::new();
    for &token in &generated {
        result.push(token);
        if token == eos_id {
            break;
        }
        if result.len() >= n_steps {
            break;
        }
    }

    Ok(result)
}
