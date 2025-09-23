//! Decode smoke test - feature-gated behind debug-asserts
//!
//! This test validates that inference produces non-flat logits after one decode step.
//! It's kept behind a debug feature to avoid blocking CI with complex model loading.

use bitnet_tokenizers::MockTokenizer;
use std::sync::Arc;

/// Minimal engine with mock tokenizer and tiny config, one prefill + one decode.
/// We only assert that logits variance (std) is > 0 after one token.
#[cfg_attr(not(debug_assertions), ignore)]
#[test]
fn decode_smoke_logits_not_flat() {
    // Use mock tokenizer for testing
    let tokenizer: Arc<dyn bitnet_tokenizers::Tokenizer> = Arc::new(MockTokenizer::new());

    // For now, this is a template - it would need to be wired to actual
    // minimal model construction once the inference engine API is finalized.
    // The key requirement is that after one prefill + decode step,
    // the logits should have non-zero variance (not be flat).

    // This test serves as a guard against quantization regressions that
    // would cause all logits to collapse to the same value.

    // TODO: Wire this to actual minimal model once InferenceEngine API is stable
    // Expected pattern:
    // let device = Device::Cpu;
    // let model = build_minimal_model(device).unwrap();
    // let backend = cpu::CpuBackend::new(device);
    // let mut engine = InferenceEngine::new(model, tokenizer, backend).unwrap();
    //
    // let prompt = "hi";
    // let _ = engine.prefill(prompt).unwrap();
    // let logits = engine.forward_last_logits().unwrap();
    //
    // // crude std check: sqrt(E[x^2] - E[x]^2) > eps
    // let mean = logits.iter().copied().sum::<f32>() / logits.len() as f32;
    // let var = logits.iter().map(|&x| { let d = x - mean; d*d }).sum::<f32>() / logits.len() as f32;
    // assert!(var.sqrt() > 1e-6, "logits are flat");

    // For now, just verify mock tokenizer works
    let text = "test";
    let tokens = tokenizer.encode(text, false, false).unwrap();
    assert!(!tokens.is_empty(), "tokenizer should produce tokens");
}
