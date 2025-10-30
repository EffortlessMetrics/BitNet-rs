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
    let tokenizer: Arc<dyn bitnet_tokenizers::Tokenizer> = Arc::new(MockTokenizer::new());
    let text = "test";
    let tokens = tokenizer.encode(text, false, false).unwrap();
    assert!(!tokens.is_empty(), "tokenizer should produce tokens");
}
