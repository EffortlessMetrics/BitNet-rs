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
    let model_path = match env::var("CROSSVAL_GGUF") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("CROSSVAL_GGUF not set, skipping test");
            return;
        }
    };
    let add_bos = true;
    let text = "What is 2+2?";
    eprintln!("Loading model from: {}", model_path);
    let loader = ModelLoader::new(BNDevice::Cpu);
    let model = loader.load::<&std::path::Path>(model_path.as_ref()).expect("failed to load model");
    let tokenizer =
        auto::load_auto(std::path::Path::new(&model_path), None).expect("failed to load tokenizer");
    let ids = tokenizer.encode(text, add_bos, false).expect("failed to encode");
    eprintln!("Encoded {} tokens: {:?}", ids.len(), ids);
    assert!(!ids.is_empty(), "encoded tokens empty");
    let vocab_size = tokenizer.vocab_size();
    eprintln!("Vocab size: {}", vocab_size);
    let mut engine = InferenceEngine::new(model.into(), tokenizer, BNDevice::Cpu)
        .expect("failed to create engine");
    eprintln!("Calling eval_ids...");
    let logits = engine.eval_ids(&ids).await.expect("eval_ids failed");
    eprintln!("Got logits, len={}", logits.len());
    assert_eq!(
        logits.len(),
        vocab_size,
        "logits length {} != vocab_size {}",
        logits.len(),
        vocab_size
    );
    let non_zero_count = logits.iter().filter(|&&x| x != 0.0).count();
    eprintln!("Non-zero logits: {}/{}", non_zero_count, vocab_size);
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
    assert!(argmax < vocab_size, "argmax {} out of range", argmax);
    assert!(best != 0.0, "all logits appear to be zero - inference not working");
    if argmax == 0 {
        eprintln!("Warning: argmax is 0, which is unusual");
    }
    eprintln!("\nTesting greedy generation...");
    let cfg = GenerationConfig::greedy()
        .with_max_tokens(4)
        .with_temperature(0.0)
        .with_top_k(1)
        .with_top_p(1.0)
        .with_repetition_penalty(1.0)
        .with_stop_sequences(vec![])
        .with_stop_token_ids(vec![])
        .with_stop_string_window(64)
        .with_seed(42)
        .with_skip_special_tokens(false)
        .with_eos_token_id(None)
        .with_logits_tap_steps(0)
        .with_logits_topk(0)
        .with_logits_cb(None)
        .with_add_bos(false);
    let generated = engine.generate_tokens(&ids, &cfg).await.expect("generate_tokens failed");
    eprintln!("Generated tokens: {:?}", generated);
    assert!(!generated.is_empty(), "greedy produced no tokens");
    assert_eq!(
        generated[0] as usize, argmax,
        "first generated token {} != argmax {}",
        generated[0], argmax
    );
    eprintln!("\n✅ Logits smoke test passed");
}
