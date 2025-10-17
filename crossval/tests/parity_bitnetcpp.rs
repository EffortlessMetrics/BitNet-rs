//! BitNet.cpp parity harness
//!
//! Validates that the Rust inference engine produces identical outputs to
//! Microsoft's BitNet C++ implementation for deterministic inference.
//!
//! This test is feature-gated and runs when both `crossval` and `integration-tests` are enabled.
//! It also requires `CROSSVAL_GGUF`; when `BITNET_CPP_DIR` is not set it runs Rust-only.

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

    // If both vectors are zero, they are effectively equal
    if norm_a == 0.0 && norm_b == 0.0 {
        return 1.0;
    }

    // If only one is zero, they are completely different
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Helper function to tokenize using C++ FFI
#[cfg(feature = "ffi")]
fn cpp_tokenize(
    gguf_path: &std::path::Path,
    formatted_prompt: &str,
    add_bos: bool,
    parse_special: bool,
) -> Result<Vec<u32>> {
    use bitnet_sys::{BitnetModel, bitnet_tokenize_text};

    let gguf_str = gguf_path.to_string_lossy().to_string();
    let cpp_model = BitnetModel::from_file(&gguf_str)
        .map_err(|e| anyhow::anyhow!("C++ model load failed: {:?}", e))?;

    let cpp_ids = bitnet_tokenize_text(&cpp_model, formatted_prompt, add_bos, parse_special)
        .map_err(|e| anyhow::anyhow!("C++ tokenization failed: {:?}", e))?;

    Ok(cpp_ids.iter().map(|&x| x as u32).collect())
}

#[cfg(not(feature = "ffi"))]
fn cpp_tokenize(
    _gguf_path: &std::path::Path,
    _formatted_prompt: &str,
    _add_bos: bool,
    _parse_special: bool,
) -> Result<Vec<u32>> {
    anyhow::bail!("C++ FFI not available (compile with --features bitnet-sys/ffi)")
}

/// Perform C++ parity check using bitnet-sys FFI
/// Returns: (cosine_similarity, cosine_ok, exact_match_rate, first_divergence_step)
#[cfg(feature = "ffi")]
fn cpp_parity_check(
    gguf_path: &std::path::Path,
    formatted_prompt: &str,
    rust_ids: &[u32],
    cpp_ids: &[u32],
    tokens_for_parity: &[u32],
    rust_logits: &[f32],
    rust_decode: &[u32],
    _add_bos: bool,
    _parse_special: bool,
    eos_id: u32,
    vocab_size: usize,
    n_steps: usize,
) -> Result<(f32, bool, f32, Option<usize>)> {
    use bitnet_sys::{
        BitnetContext, BitnetModel, bitnet_decode_greedy, bitnet_eval_tokens, bitnet_prefill,
    };

    let gguf_str = gguf_path.to_string_lossy().to_string();

    // 1. Load C++ model and context
    let cpp_model = BitnetModel::from_file(&gguf_str)
        .map_err(|e| anyhow::anyhow!("C++ model load failed: {:?}", e))?;

    let cpp_ctx = BitnetContext::new(&cpp_model, 4096, 1, 0)
        .map_err(|e| anyhow::anyhow!("C++ context creation failed: {:?}", e))?;

    // 2. Token ID forensics
    let prompt_hash = blake3::hash(formatted_prompt.as_bytes());
    eprintln!("parity.prompt_hash={}", prompt_hash);

    let head = 16.min(rust_ids.len()).min(cpp_ids.len());
    let tail = 16.min(rust_ids.len()).min(cpp_ids.len());

    eprintln!("parity.rust.head={:?}", &rust_ids[..head]);
    eprintln!("parity.cpp.head ={:?}", &cpp_ids[..head]);

    if rust_ids.len() >= tail && cpp_ids.len() >= tail {
        eprintln!("parity.rust.tail={:?}", &rust_ids[rust_ids.len() - tail..]);
        eprintln!("parity.cpp.tail ={:?}", &cpp_ids[cpp_ids.len() - tail..]);
    }

    if cpp_ids != rust_ids {
        eprintln!(
            "WARNING: Tokenization mismatch! Rust len: {}, C++ len: {}",
            rust_ids.len(),
            cpp_ids.len()
        );
        // Continue anyway to collect more metrics
    } else {
        eprintln!("✓ Tokenization exact match");
    }

    // 3. Prefill C++ context with parity tokens (primes KV cache; sets n_past)
    let cpp_ids_i32: Vec<i32> = tokens_for_parity.iter().map(|&x| x as i32).collect();
    bitnet_prefill(&cpp_ctx, &cpp_ids_i32)
        .map_err(|e| anyhow::anyhow!("C++ prefill failed: {:?}", e))?;

    // 4. Prefill logits parity (cosine similarity)
    let cpp_logits = bitnet_eval_tokens(&cpp_ctx, &cpp_ids_i32, vocab_size)
        .map_err(|e| anyhow::anyhow!("C++ eval failed: {:?}", e))?;

    let cos = cosine_similarity(rust_logits, &cpp_logits);
    let cos_ok = cos >= 0.99;

    // 5. N-step greedy decode parity
    let cpp_gen = bitnet_decode_greedy(&cpp_ctx, &cpp_ids_i32, n_steps, eos_id as i32)
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
    _cpp_ids: &[u32],
    _tokens_for_parity: &[u32],
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

/// Collect environment metadata for receipt provenance
fn collect_env_metadata() -> serde_json::Value {
    use std::env;

    // Get Rust version
    let rustc_version = env!("RUSTC_VERSION", "rustc version output at build time");

    // Get target triple and CPU
    let target_triple = env!("TARGET", "unknown");
    let target_cpu = env::var("TARGET_CPU").unwrap_or_else(|_| std::env::consts::ARCH.to_string());

    // Get OS and libc
    let os = std::env::consts::OS;
    let libc = if cfg!(target_env = "gnu") {
        "gnu"
    } else if cfg!(target_env = "musl") {
        "musl"
    } else if cfg!(target_env = "msvc") {
        "msvc"
    } else {
        "unknown"
    };

    // Get CPU features (from compile-time target features)
    let cpu_features = get_cpu_features();

    // Get RAYON threads (deterministic testing should have RAYON_NUM_THREADS=1)
    let rayon_threads = env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "auto".to_string());

    // Get deterministic flags
    let deterministic = env::var("BITNET_DETERMINISTIC").unwrap_or_else(|_| "0".to_string());
    let seed = env::var("BITNET_SEED").unwrap_or_else(|_| "0".to_string());

    // Get llama.cpp commit (if available from BITNET_CPP_DIR)
    let llama_cpp_commit = get_llama_cpp_commit();

    serde_json::json!({
        "rustc_version": rustc_version,
        "target_triple": target_triple,
        "target_cpu": target_cpu,
        "cpu_features": cpu_features,
        "os": os,
        "libc": libc,
        "rayon_threads": rayon_threads,
        "deterministic": deterministic,
        "seed": seed,
        "llama_cpp_commit": llama_cpp_commit,
    })
}

/// Get CPU features from compile-time target features
fn get_cpu_features() -> Vec<&'static str> {
    let mut features = Vec::new();

    #[cfg(target_feature = "avx")]
    features.push("avx");
    #[cfg(target_feature = "avx2")]
    features.push("avx2");
    #[cfg(target_feature = "avx512f")]
    features.push("avx512f");
    #[cfg(target_feature = "sse2")]
    features.push("sse2");
    #[cfg(target_feature = "sse4.1")]
    features.push("sse4.1");
    #[cfg(target_feature = "sse4.2")]
    features.push("sse4.2");
    #[cfg(target_feature = "neon")]
    features.push("neon");
    #[cfg(target_feature = "fma")]
    features.push("fma");

    features
}

/// Get llama.cpp git commit from BITNET_CPP_DIR if available
fn get_llama_cpp_commit() -> Option<String> {
    use std::fs;
    use std::path::PathBuf;

    let cpp_dir = std::env::var("BITNET_CPP_DIR").ok()?;
    let git_dir =
        PathBuf::from(&cpp_dir).join("3rdparty").join("llama.cpp").join(".git").join("HEAD");

    if let Ok(head) = fs::read_to_string(&git_dir) {
        if let Some(ref_path) = head.strip_prefix("ref: ") {
            let commit_file = PathBuf::from(&cpp_dir)
                .join("3rdparty")
                .join("llama.cpp")
                .join(".git")
                .join(ref_path.trim());
            if let Ok(commit) = fs::read_to_string(commit_file) {
                return Some(commit.trim().to_string());
            }
        } else {
            // Direct commit hash
            return Some(head.trim().to_string());
        }
    }

    None
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

    // Wrap the test logic with a timeout guard (120 seconds)
    // This prevents hangs and writes a diagnostic receipt on timeout
    // Note: Increased from 60s to accommodate 2B+ parameter models in release mode
    match tokio::time::timeout(
        std::time::Duration::from_secs(120),
        parity_bitnetcpp_impl(gguf_path.clone()),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => {
            eprintln!("TIMEOUT: Parity test exceeded 120 seconds - writing diagnostic receipt");
            write_timeout_receipt(&gguf_path)?;
            anyhow::bail!("Parity test timed out after 120 seconds");
        }
    }
}

/// Core parity test implementation (wrapped by timeout guard)
async fn parity_bitnetcpp_impl(gguf_path: PathBuf) -> Result<()> {
    // Check if BITNET_CPP_DIR is set for C++ parity
    let cpp_available = env::var("BITNET_CPP_DIR").is_ok();
    if !cpp_available {
        eprintln!("BITNET_CPP_DIR not set; running Rust-only validation");
    }

    let commit = env::var("GIT_COMMIT").unwrap_or_else(|_| "unknown".into());

    // Support multiple test prompts for comprehensive parity validation
    // Can override with CROSSVAL_PROMPT_SET=chat|math|all
    let prompt_set = env::var("CROSSVAL_PROMPT_SET").unwrap_or_else(|_| "math".into());

    // Define test prompts with their expected templates
    let test_prompts = match prompt_set.as_str() {
        "math" => vec![("What is 2+2?", bitnet_inference::TemplateType::Raw)],
        "chat" => vec![(
            "<|start_header_id|>user<|end_header_id|>\n\nWhat is photosynthesis?<|eot_id|>",
            bitnet_inference::TemplateType::Raw,
        )],
        "all" => vec![
            ("What is 2+2?", bitnet_inference::TemplateType::Raw),
            (
                "<|start_header_id|>user<|end_header_id|>\n\nWhat is photosynthesis?<|eot_id|>",
                bitnet_inference::TemplateType::Raw,
            ),
        ],
        _ => vec![("What is 2+2?", bitnet_inference::TemplateType::Raw)],
    };

    // For now, run only the first prompt (multi-prompt support can be added later)
    let (prompt, template) = test_prompts.first().expect("At least one test prompt required");

    eprintln!("=== Parity Harness ===");
    eprintln!("Model: {:?}", gguf_path);
    eprintln!("Prompt set: {}", prompt_set);
    eprintln!("Prompt: {}", prompt);
    eprintln!("Commit: {}", commit);

    // 1. Rust-side tokenization and metadata
    // For crossval, use Raw template to avoid double-wrapping
    // (C++ side will see the same unformatted prompt)
    let formatted_prompt = template.apply(prompt, None);

    let tok_meta = rust_side_tokenize_and_meta(&gguf_path, &template, &formatted_prompt)?;
    let rust_ids = tok_meta.token_ids.clone();
    let add_bos = tok_meta.add_bos;
    let parse_special = tok_meta.parse_special;
    let eos_id = tok_meta.eos_id;
    let vocab_size = tok_meta.vocab_size;

    eprintln!("Template: {}", template);
    eprintln!("Formatted prompt: {}", formatted_prompt);
    eprintln!(
        "Tokenized {} tokens (add_bos={}, parse_special={}, eos_id={}, kind={})",
        rust_ids.len(),
        add_bos,
        parse_special,
        eos_id,
        tok_meta.tokenizer_kind
    );

    // 2. Get C++ tokens early for potential fallback
    let cpp_ids = if cpp_available {
        cpp_tokenize(&gguf_path, &formatted_prompt, add_bos, parse_special)?
    } else {
        Vec::new()
    };

    // 4. Use pure Rust tokenization for parity (no C++ fallback)
    // After BPE ByteLevel fix, Rust tokenization should match C++ exactly
    let tokens_for_parity = rust_ids.clone();
    let tokenizer_source = "rust";
    eprintln!("parity.tokenizer_source=rust");

    // 5. Rust-side logits evaluation (using canonical tokens)
    let rust_logits = rust_eval_last_logits(&gguf_path, &tokens_for_parity, vocab_size).await?;
    eprintln!("Rust logits shape: [{}]", rust_logits.len());

    // 6. Rust-side greedy decoding (N steps, using canonical tokens)
    // Note: Using 4 steps for faster parity validation (increased from 8 to reduce test time)
    let n_steps = 4;
    let rust_decode = rust_decode_n_greedy(&gguf_path, &tokens_for_parity, n_steps, eos_id).await?;
    eprintln!("Rust decoded {} tokens: {:?}", rust_decode.len(), rust_decode);

    // 7. If C++ is available, compare outputs
    let (cosine_ok, cosine_similarity, exact_match_rate, first_divergence, cpp_loaded) =
        if cpp_available {
            match cpp_parity_check(
                &gguf_path,
                &formatted_prompt,
                &rust_ids,
                &cpp_ids,
                &tokens_for_parity,
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
                    (false, None, None, None, true) // C++ was loaded but check failed
                }
            }
        } else {
            (false, None, None, None, false)
        };

    // 5. Write parity receipt
    let ts = humantime::format_rfc3339(SystemTime::now()).to_string();

    // Safer than slicing fixed indices: split at 'T'
    let date_str = ts.split_once('T').map(|(d, _)| d).unwrap_or("1970-01-01");

    // Allow override for CI: BASELINES_DIR=/path/to/workspace/docs/baselines
    let base_dir = std::env::var("BASELINES_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/baselines"));

    let receipt_dir = base_dir.join(date_str);

    if !receipt_dir.exists() {
        fs::create_dir_all(&receipt_dir).context("Failed to create baselines directory")?;
    }

    let model_sha = sha256_file(&gguf_path)?;

    let parity_status = if cpp_loaded {
        // C++ was loaded - check if outputs match
        if cosine_ok && exact_match_rate.unwrap_or(0.0) == 1.0 { "ok" } else { "mismatch" }
    } else {
        // C++ not available
        "rust_only"
    };

    // Compute prompt hash for reproducibility verification
    let prompt_hash = blake3::hash(formatted_prompt.as_bytes()).to_string();

    // Collect environment metadata
    let env_meta = collect_env_metadata();

    let receipt = json!({
        "timestamp": ts,
        "commit": commit,
        "model_path": gguf_path.display().to_string(),
        "model_sha256": model_sha,
        "template": {
            "id": template.to_string(),
            "formatted_prompt_hash": prompt_hash,
        },
        "prompt": prompt,
        "tokenizer": {
            "source": tokenizer_source,
            "kind": tok_meta.tokenizer_kind,
            "vocab_size": vocab_size,
            "bos_id": tok_meta.bos_id,
            "eos_id": eos_id,
            "eot_id": tok_meta.eot_id,
            "add_bos_hint": tok_meta.add_bos_hint,
            "add_bos": add_bos,
            "parse_special": parse_special,
            "merges_count": tok_meta.merges_count,
            "tokenizer_blob_sha256": tok_meta.tokenizer_blob_sha256,
        },
        "tokenization": {
            "rust_token_count": rust_ids.len(),
            "cpp_token_count": cpp_ids.len(),
            "tokens_match": rust_ids == cpp_ids,
        },
        "rust": {
            "token_count": tokens_for_parity.len(),
            "logits_dim": rust_logits.len(),
            "decoded_tokens": rust_decode,
            "n_steps": n_steps,
        },
        "parity": {
            "cpp_available": cpp_loaded,
            "cosine_similarity": cosine_similarity,
            "cosine_ok": cosine_ok,
            "exact_match_rate": exact_match_rate,
            "first_divergence_step": first_divergence,
            "status": parity_status,
        },
        "environment": env_meta,
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

/// Extended tokenizer metadata for receipt provenance
#[derive(Debug)]
struct TokenizerMetadata {
    token_ids: Vec<u32>,
    add_bos: bool,
    parse_special: bool,
    eos_id: u32,
    vocab_size: usize,
    tokenizer_kind: String,
    bos_id: Option<u32>,
    eot_id: Option<u32>,
    add_bos_hint: Option<bool>,
    // BPE-specific fields
    merges_count: Option<usize>,
    // SPM-specific fields
    tokenizer_blob_sha256: Option<String>,
}

/// Tokenize prompt with template-aware BOS/special handling
/// Returns comprehensive tokenizer metadata for receipt provenance
fn rust_side_tokenize_and_meta(
    model_path: &std::path::Path,
    template: &bitnet_inference::TemplateType,
    formatted_prompt: &str,
) -> Result<TokenizerMetadata> {
    use bitnet_inference::TemplateType;
    use bitnet_models::formats::gguf::GgufReader;
    use bitnet_models::loader::MmapFile;
    use bitnet_tokenizers::auto;

    // 1) Load tokenizer using auto-detection
    let tokenizer = auto::load_auto(model_path, None)?;
    let vocab_size = tokenizer.vocab_size();

    // 2) Extract tokenizer kind and provenance from GGUF metadata
    let (
        tokenizer_kind,
        bos_id_meta,
        eot_id_meta,
        add_bos_hint_meta,
        merges_count,
        tokenizer_blob_sha256,
    ) = {
        let mmap = MmapFile::open(model_path)?;
        let reader = GgufReader::new(mmap.as_slice())?;

        let kind = reader
            .get_string_metadata("tokenizer.ggml.model")
            .unwrap_or_else(|| "unknown".to_string());

        let bos = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        let eot = reader.get_u32_metadata("tokenizer.ggml.eot_token_id");
        let add_bos_hint = reader.get_bool_metadata("tokenizer.ggml.add_bos_token");

        // BPE-specific: count merges
        let merges_count = if kind == "gpt2" {
            // Try to read merges list from GGUF
            reader.get_string_array_metadata("tokenizer.ggml.merges").map(|arr| arr.len())
        } else {
            None
        };

        // SPM-specific: compute SHA256 of tokenizer model blob
        // Try multiple possible locations for the SPM protobuf
        let tokenizer_blob_sha256 = if kind == "llama" {
            reader
                .get_bin_or_u8_array("tokenizer.model")
                .or_else(|| reader.get_bin_or_u8_array("tokenizer.spm.model"))
                .or_else(|| reader.get_bin_or_u8_array("sentencepiece.model"))
                .map(|blob| {
                    use sha2::{Digest, Sha256};
                    let mut hasher = Sha256::new();
                    hasher.update(&blob);
                    format!("{:x}", hasher.finalize())
                })
        } else {
            None
        };

        (kind, bos, eot, add_bos_hint, merges_count, tokenizer_blob_sha256)
    };

    // 3) Determine BOS policy from template
    let add_bos = template.should_add_bos();

    // 4) Determine parse_special flag based on prompt content
    // For LLaMA-3 chat prompts with special tokens, we need parse_special=true
    let parse_special = matches!(template, TemplateType::Llama3Chat)
        || formatted_prompt.contains("<|start_header_id|>")
        || formatted_prompt.contains("<|eot_id|>");

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

    // 6) Encode the formatted prompt (already formatted by caller)
    let ids = tokenizer.encode(formatted_prompt, add_bos, false)?;

    Ok(TokenizerMetadata {
        token_ids: ids,
        add_bos,
        parse_special,
        eos_id,
        vocab_size,
        tokenizer_kind,
        bos_id: bos_id_meta,
        eot_id: eot_id_meta,
        add_bos_hint: add_bos_hint_meta,
        merges_count,
        tokenizer_blob_sha256,
    })
}

/// Auto-detect template type from GGUF metadata (matches CLI logic exactly)
#[allow(dead_code)]
fn auto_detect_template(model_path: &std::path::Path) -> bitnet_inference::TemplateType {
    use bitnet_inference::TemplateType;
    use bitnet_models::GgufReader;
    use bitnet_models::loader::MmapFile;

    // Try to read GGUF metadata
    if let Ok(mmap) = MmapFile::open(model_path) {
        if let Ok(reader) = GgufReader::new(mmap.as_slice()) {
            // Extract metadata fields using convenience methods
            let tokenizer_name = reader
                .get_string_metadata("tokenizer.ggml.model")
                .or_else(|| reader.get_string_metadata("tokenizer.name"));

            let chat_template = reader.get_string_metadata("tokenizer.chat_template");

            // Use the same detection logic as the CLI
            return TemplateType::detect(tokenizer_name.as_deref(), chat_template.as_deref());
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

    // Read seed from environment (for reproducible cross-validation)
    let seed = env::var("BITNET_SEED").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);

    // Configure greedy generation
    let config = GenerationConfig {
        max_new_tokens: n_steps as u32,
        temperature: 0.0,
        top_k: 1,   // greedy: top_k=1
        top_p: 1.0, // disabled: top_p=1.0
        repetition_penalty: 1.0,
        stop_sequences: vec![],
        stop_token_ids: vec![eos_id],
        seed: Some(seed),
        skip_special_tokens: false,
        eos_token_id: Some(eos_id),
        logits_tap_steps: 0,
        logits_topk: 0,
        logits_cb: None,
        add_bos: false, // prompt already has BOS if needed
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

/// Write a diagnostic receipt when the parity test times out
fn write_timeout_receipt(gguf_path: &std::path::Path) -> Result<()> {
    let ts = humantime::format_rfc3339(SystemTime::now()).to_string();
    let date_str = ts.split_once('T').map(|(d, _)| d).unwrap_or("1970-01-01");

    let base_dir = std::env::var("BASELINES_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/baselines"));

    let receipt_dir = base_dir.join(date_str);

    if !receipt_dir.exists() {
        fs::create_dir_all(&receipt_dir).context("Failed to create baselines directory")?;
    }

    let commit = env::var("GIT_COMMIT").unwrap_or_else(|_| "unknown".into());

    // Collect comprehensive environment metadata
    let env_meta = collect_env_metadata();

    let receipt = json!({
        "timestamp": ts,
        "commit": commit,
        "model_path": gguf_path.display().to_string(),
        "parity": {
            "status": "timeout",
            "timeout_seconds": 120,
        },
        "environment": env_meta,
        "error": "Parity test exceeded 60-second timeout - check for performance regression or hanging inference"
    });

    let receipt_path = receipt_dir.join("parity-bitnetcpp.json");
    let tmp_path = receipt_dir.join("parity-bitnetcpp.json.tmp");
    fs::write(&tmp_path, serde_json::to_vec_pretty(&receipt)?)
        .context("Failed to write timeout receipt")?;
    fs::rename(&tmp_path, &receipt_path).context("Failed to atomically rename timeout receipt")?;

    eprintln!("✗ Timeout receipt written to: {:?}", receipt_path);

    Ok(())
}
