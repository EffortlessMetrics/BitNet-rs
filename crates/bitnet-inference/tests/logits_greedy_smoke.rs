//! Smoke test for logits extraction and greedy decoding
//!
//! This test isolates the Rust inference path without FFI to diagnose
//! the issue where greedy decoding produces [0, 67938, 67938, 67938].

use bitnet_common::Device as BNDevice;
use bitnet_inference::{GenerationConfig, InferenceEngine};
use bitnet_models::ModelLoader;
use bitnet_tokenizers::auto;
use std::env;

#[tokio::test]
async fn logits_and_greedy_smoke() {
    // Skip if no model available
    let model_path = match env::var("CROSSVAL_GGUF") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("CROSSVAL_GGUF not set, skipping test");
            return;
        }
    };

    let add_bos = true; // for GPT-2-like BitNet models
    let text = "What is 2+2?";

    // Load model/tokenizer
    eprintln!("Loading model from: {}", model_path);
    let loader = ModelLoader::new(BNDevice::Cpu);
    let model = loader.load::<&std::path::Path>(model_path.as_ref()).expect("failed to load model");
    let tokenizer =
        auto::load_auto(std::path::Path::new(&model_path), None).expect("failed to load tokenizer");

    // Encode
    let ids = tokenizer.encode(text, add_bos, /*parse_special*/ false).expect("failed to encode");
    eprintln!("Encoded {} tokens: {:?}", ids.len(), ids);
    assert!(!ids.is_empty(), "encoded tokens empty");

    // Create engine
    let vocab_size = tokenizer.vocab_size();
    eprintln!("Vocab size: {}", vocab_size);

    let mut engine = InferenceEngine::new(model.into(), tokenizer, BNDevice::Cpu)
        .expect("failed to create engine");

    // Evaluate and get logits
    eprintln!("Calling eval_ids...");
    let logits = engine.eval_ids(&ids).await.expect("eval_ids failed");
    eprintln!("Got logits, len={}", logits.len());

    // Check logits shape - should be exactly vocab_size for last position
    assert_eq!(
        logits.len(),
        vocab_size,
        "logits length {} != vocab_size {}",
        logits.len(),
        vocab_size
    );

    // Check that logits are not all zero (pathological case)
    let non_zero_count = logits.iter().filter(|&&x| x != 0.0).count();
    eprintln!("Non-zero logits: {}/{}", non_zero_count, vocab_size);

    // Find max logit value and argmax
    let (mut argmax, mut best) = (0usize, logits[0]);
    for (i, &x) in logits.iter().enumerate().skip(1) {
        if x > best || (x == best && i < argmax) {
            best = x;
            argmax = i;
        }
    }

    eprintln!("Argmax: id={} value={}", argmax, best);
    eprintln!("Top-5 logits:");
    let mut indexed: Vec<(usize, &f32)> = logits.iter().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
    for &(i, &val) in indexed.iter().take(5) {
        eprintln!("  id={} val={}", i, val);
    }

    // Sanity checks
    assert!(argmax < vocab_size, "argmax {} out of range", argmax);

    // The logits should have meaningful values (not all zeros)
    assert!(best > 0.0 || best < 0.0, "all logits appear to be zero - inference not working");

    // Greedy argmax should not be 0 for a GPT-2 model (0 is typically "!" token)
    // unless the model genuinely predicts it
    if argmax == 0 {
        eprintln!("Warning: argmax is 0, which is unusual");
    }

    // Short greedy decode
    eprintln!("\nTesting greedy generation...");
    let cfg = GenerationConfig {
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        stop_sequences: vec![],
        stop_token_ids: vec![],
        seed: Some(42),
        skip_special_tokens: false,
        eos_token_id: None,
        logits_tap_steps: 0,
        logits_topk: 0,
        logits_cb: None,
        add_bos: false,
    };

    let generated = engine.generate_tokens(&ids, &cfg).await.expect("generate_tokens failed");
    eprintln!("Generated tokens: {:?}", generated);

    assert!(!generated.is_empty(), "greedy produced no tokens");

    // The first generated token should match our argmax from eval_ids
    assert_eq!(
        generated[0] as usize, argmax,
        "first generated token {} != argmax {}",
        generated[0], argmax
    );

    eprintln!("\n✅ Logits smoke test passed");
}
