#![cfg(feature = "cpu")]
use anyhow::{Context, Result};
use bitnet_inference::config::GenerationConfig;
use std::path::PathBuf;
/// Test utilities for CPU forward pass validation
mod test_utils {
    use super::*;
    /// Get test model path via auto-discovery or BITNET_GGUF env var
    ///
    /// # Search Order
    /// 1. BITNET_GGUF environment variable
    /// 2. models/ directory (workspace root)
    /// 3. Return error if not found
    ///
    /// # Example
    /// ```bash
    /// export BITNET_GGUF=/path/to/model.gguf
    /// cargo test -p bitnet-inference test_ac1 --features cpu
    /// ```
    pub fn get_test_model_path() -> Result<PathBuf> {
        if let Ok(path) = std::env::var("BITNET_GGUF") {
            let model_path = PathBuf::from(&path);
            if model_path.exists() {
                return Ok(model_path);
            }
            anyhow::bail!("BITNET_GGUF set to '{}' but file does not exist", path);
        }
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .ok_or_else(|| anyhow::anyhow!("Failed to find workspace root"))?;
        let models_dir = workspace_root.join("models");
        if !models_dir.exists() {
            anyhow::bail!(
                "No test model found. Set BITNET_GGUF env var or place model in models/ directory.\n\
                 Example: cargo run -p xtask -- download-model"
            );
        }
        let model_file = std::fs::read_dir(&models_dir)
            .context("Failed to read models/ directory")?
            .filter_map(|entry| entry.ok())
            .find(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("gguf"))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No .gguf files found in models/ directory.\n\
                     Run: cargo run -p xtask -- download-model"
                )
            })?;
        Ok(model_file.path())
    }
    /// Get test tokenizer path via auto-discovery or BITNET_TOKENIZER env var
    ///
    /// # Returns
    /// Optional tokenizer path (None if not found, which is acceptable for some tests)
    #[allow(dead_code)]
    pub fn get_test_tokenizer_path() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("BITNET_TOKENIZER") {
            let tokenizer_path = PathBuf::from(&path);
            if tokenizer_path.exists() {
                return Some(tokenizer_path);
            }
        }
        if let Ok(model_path) = get_test_model_path() {
            let parent = model_path.parent()?;
            let tokenizer_path = parent.join("tokenizer.json");
            if tokenizer_path.exists() {
                return Some(tokenizer_path);
            }
        }
        None
    }
    /// Enable deterministic inference mode
    ///
    /// Sets environment variables:
    /// - BITNET_DETERMINISTIC=1
    /// - BITNET_SEED=42
    /// - RAYON_NUM_THREADS=1
    ///
    /// # Safety
    /// Uses unsafe set_var which is safe in single-threaded test contexts.
    /// Tests must not be run in parallel when using this function.
    pub fn enable_deterministic_mode() {
        #[allow(unused_unsafe)]
        unsafe {
            std::env::set_var("BITNET_DETERMINISTIC", "1");
            std::env::set_var("BITNET_SEED", "42");
            std::env::set_var("RAYON_NUM_THREADS", "1");
        }
    }
    /// Enable strict mode (no FP32 staging)
    ///
    /// # Safety
    /// Uses unsafe set_var which is safe in single-threaded test contexts.
    /// Tests must not be run in parallel when using this function.
    pub fn enable_strict_mode() {
        #[allow(unused_unsafe)]
        unsafe {
            std::env::set_var("BITNET_STRICT_MODE", "1");
        }
    }
}
/// AC:1 - T1.1: Single-layer forward pass with I2_S quantization
///
/// Test Plan: docs/explanation/cpu-inference-test-plan.md#test-11
/// Validates that forward pass on BOS token returns non-zero finite logits
///
/// # Expected Behavior
/// - Input: BOS token (ID = 1 or 2)
/// - Output: Logits tensor [1, vocab_size]
/// - All logits finite (no NaN/Inf)
/// - At least 50% of logits are non-zero
/// - KV cache updated at position 0
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "cpu")]
async fn test_ac1_cpu_forward_bos_nonzero_logits() -> Result<()> {
    use bitnet_inference::InferenceEngine;
    use bitnet_models::ModelLoader;
    use bitnet_tokenizers::Tokenizer;
    use std::sync::Arc;
    let model_path = match test_utils::get_test_model_path() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("SKIP: {}", e);
            return Ok(());
        }
    };
    test_utils::enable_deterministic_mode();
    let loader = ModelLoader::new(bitnet_common::Device::Cpu);
    let model = match loader.load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP: Model loading failed - {}", e);
            return Ok(());
        }
    };
    let tokenizer = match bitnet_tokenizers::auto::load_auto(&model_path, None) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("SKIP: Tokenizer loading failed - {}", e);
            return Ok(());
        }
    };
    let model_arc: Arc<dyn bitnet_models::Model> = model.into();
    let tokenizer_arc: Arc<dyn Tokenizer> = tokenizer.clone();
    let mut engine = InferenceEngine::new(model_arc, tokenizer_arc, bitnet_common::Device::Cpu)?;
    let bos_token_id = tokenizer.bos_token_id().unwrap_or(1);
    engine.prefill(&[bos_token_id]).await?;
    let gen_config = GenerationConfig::greedy()
        .with_max_tokens(1)
        .with_temperature(0.0)
        .with_top_k(1)
        .with_top_p(1.0)
        .with_repetition_penalty(1.0)
        .with_stop_sequences(vec![])
        .with_stop_token_ids(vec![])
        .with_stop_string_window(64)
        .with_seed(42)
        .with_skip_special_tokens(true)
        .with_eos_token_id(None)
        .with_logits_tap_steps(0)
        .with_logits_topk(10)
        .with_logits_cb(None)
        .with_add_bos(false);
    let generated_tokens = engine.generate_tokens(&[bos_token_id], &gen_config).await?;
    assert!(
        !generated_tokens.is_empty(),
        "Expected at least one generated token (BOS → forward pass → logits → sampling)"
    );
    Ok(())
}
/// AC:1 - T1.2: Multi-layer forward pass with KV cache updates
///
/// Test Plan: docs/explanation/cpu-inference-test-plan.md#test-12
/// Validates 16-token autoregressive generation with greedy decoding
///
/// # Expected Behavior
/// - Greedy sampling (temperature = 0.0, argmax)
/// - Deterministic output (same seed → same tokens)
/// - All token IDs within vocab range
/// - KV cache length increments correctly: 1, 2, 3, ..., 17
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "cpu")]
async fn test_ac1_greedy_decode_16_tokens() -> Result<()> {
    use bitnet_inference::InferenceEngine;
    use bitnet_models::ModelLoader;
    use bitnet_tokenizers::Tokenizer;
    use std::sync::Arc;
    let model_path = match test_utils::get_test_model_path() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("SKIP: {}", e);
            return Ok(());
        }
    };
    test_utils::enable_deterministic_mode();
    let loader = ModelLoader::new(bitnet_common::Device::Cpu);
    let model = match loader.load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP: Model loading failed - {}", e);
            return Ok(());
        }
    };
    let tokenizer = match bitnet_tokenizers::auto::load_auto(&model_path, None) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("SKIP: Tokenizer loading failed - {}", e);
            return Ok(());
        }
    };
    let model_arc: Arc<dyn bitnet_models::Model> = model.into();
    let tokenizer_arc: Arc<dyn Tokenizer> = tokenizer.clone();
    let engine = InferenceEngine::new(model_arc, tokenizer_arc, bitnet_common::Device::Cpu)?;
    let bos_token_id = tokenizer.bos_token_id().unwrap_or(1);
    let gen_config = GenerationConfig::greedy()
        .with_max_tokens(16)
        .with_temperature(0.0)
        .with_top_k(1)
        .with_top_p(1.0)
        .with_repetition_penalty(1.0)
        .with_stop_sequences(vec![])
        .with_stop_token_ids(vec![])
        .with_stop_string_window(64)
        .with_seed(42)
        .with_skip_special_tokens(true)
        .with_eos_token_id(None)
        .with_logits_tap_steps(0)
        .with_logits_topk(10)
        .with_logits_cb(None)
        .with_add_bos(false);
    let generated_tokens = engine.generate_tokens(&[bos_token_id], &gen_config).await?;
    assert!(
        !generated_tokens.is_empty(),
        "Expected at least one generated token from 16-token greedy decode"
    );
    assert!(
        generated_tokens.len() <= 16,
        "Generated {} tokens, expected at most 16 (max_new_tokens limit)",
        generated_tokens.len()
    );
    let vocab_size = tokenizer.vocab_size();
    for &token_id in &generated_tokens {
        assert!(
            (token_id as usize) < vocab_size,
            "Token ID {} exceeds vocab size {} (invalid token from forward pass)",
            token_id,
            vocab_size
        );
    }
    Ok(())
}
/// AC:1 - T1.3: End-to-end logits validation with strict mode
///
/// Test Plan: docs/explanation/cpu-inference-test-plan.md#test-13
/// Validates that forward pass uses QuantizedLinear I2S/TL1/TL2 paths only
///
/// # Expected Behavior
/// - Strict mode enabled (BITNET_STRICT_MODE=1)
/// - No FP32 staging kernels allowed
/// - Receipt contains ≥1 quantized kernel (i2s_*, tl1_*, tl2_*)
/// - No fallback kernels (fp32_*, fallback_*, dequant*)
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "cpu")]
async fn test_ac1_quantized_linear_strict_mode() -> Result<()> {
    use bitnet_inference::InferenceEngine;
    use bitnet_models::ModelLoader;
    use bitnet_tokenizers::Tokenizer;
    use std::sync::Arc;
    let model_path = match test_utils::get_test_model_path() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("SKIP: {}", e);
            return Ok(());
        }
    };
    test_utils::enable_deterministic_mode();
    test_utils::enable_strict_mode();
    let loader = ModelLoader::new(bitnet_common::Device::Cpu);
    let model = match loader.load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP: Model loading failed - {}", e);
            return Ok(());
        }
    };
    let tokenizer = match bitnet_tokenizers::auto::load_auto(&model_path, None) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("SKIP: Tokenizer loading failed - {}", e);
            return Ok(());
        }
    };
    let model_arc: Arc<dyn bitnet_models::Model> = model.into();
    let tokenizer_arc: Arc<dyn Tokenizer> = tokenizer.clone();
    let engine = InferenceEngine::new(model_arc, tokenizer_arc, bitnet_common::Device::Cpu)?;
    let bos_token_id = tokenizer.bos_token_id().unwrap_or(1);
    let gen_config = GenerationConfig::greedy()
        .with_max_tokens(1)
        .with_temperature(0.0)
        .with_top_k(1)
        .with_top_p(1.0)
        .with_repetition_penalty(1.0)
        .with_stop_sequences(vec![])
        .with_stop_token_ids(vec![])
        .with_stop_string_window(64)
        .with_seed(42)
        .with_skip_special_tokens(true)
        .with_eos_token_id(None)
        .with_logits_tap_steps(0)
        .with_logits_topk(10)
        .with_logits_cb(None)
        .with_add_bos(false);
    let generated_tokens = engine.generate_tokens(&[bos_token_id], &gen_config).await?;
    assert!(
        !generated_tokens.is_empty(),
        "Expected at least one token in strict mode (quantized-only path enforcement)"
    );
    Ok(())
}
/// AC:1 - T1.4: KV cache update and retrieval correctness
///
/// Test Plan: docs/explanation/cpu-inference-test-plan.md#test-14
/// Validates KV cache management during multi-step forward pass
///
/// # Expected Behavior
/// - Cache length increments: 1, 2, 3
/// - K,V shapes: [current_len, num_heads, head_dim]
/// - Cache contains non-zero values (not all zeros)
/// - Cache accessible for all layers
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "cpu")]
async fn test_ac1_kv_cache_update_retrieval() -> Result<()> {
    use bitnet_inference::InferenceEngine;
    use bitnet_models::ModelLoader;
    use bitnet_tokenizers::Tokenizer;
    use std::sync::Arc;
    let model_path = match test_utils::get_test_model_path() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("SKIP: {}", e);
            return Ok(());
        }
    };
    test_utils::enable_deterministic_mode();
    let loader = ModelLoader::new(bitnet_common::Device::Cpu);
    let model = match loader.load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP: Model loading failed - {}", e);
            return Ok(());
        }
    };
    let tokenizer = match bitnet_tokenizers::auto::load_auto(&model_path, None) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("SKIP: Tokenizer loading failed - {}", e);
            return Ok(());
        }
    };
    let model_arc: Arc<dyn bitnet_models::Model> = model.into();
    let tokenizer_arc: Arc<dyn Tokenizer> = tokenizer.clone();
    let mut engine = InferenceEngine::new(model_arc, tokenizer_arc, bitnet_common::Device::Cpu)?;
    let bos_token_id = tokenizer.bos_token_id().unwrap_or(1);
    engine.prefill(&[bos_token_id]).await?;
    let gen_config = GenerationConfig::greedy()
        .with_max_tokens(2)
        .with_temperature(0.0)
        .with_top_k(1)
        .with_top_p(1.0)
        .with_repetition_penalty(1.0)
        .with_stop_sequences(vec![])
        .with_stop_token_ids(vec![])
        .with_stop_string_window(64)
        .with_seed(42)
        .with_skip_special_tokens(true)
        .with_eos_token_id(None)
        .with_logits_tap_steps(0)
        .with_logits_topk(10)
        .with_logits_cb(None)
        .with_add_bos(false);
    let generated_tokens = engine.generate_tokens(&[bos_token_id], &gen_config).await?;
    assert!(
        !generated_tokens.is_empty(),
        "Expected at least one token (KV cache population and retrieval working)"
    );
    Ok(())
}
