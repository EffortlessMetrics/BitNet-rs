//! Per-position logits comparison tests
//!
//! These tests compare logits at each token position between Rust and C++
//! implementations to identify the first divergence point in multi-token generation.

#![cfg(feature = "crossval")]
#![cfg(feature = "integration-tests")]

use anyhow::Result;
use bitnet_crossval::logits_compare::{compare_per_position_logits, COSINE_SIMILARITY_THRESHOLD};
use bitnet_inference::eval_logits_once;
use bitnet_sys::wrapper::{self, Session as CppSession};

/// Get the test model path from the environment, or skip the test
fn test_model_path() -> Option<String> {
    if !bitnet_sys::is_available() {
        eprintln!("Skipping test: C++ backend unavailable (set BITNET_CPP_DIR)");
        return None;
    }

    match std::env::var("CROSSVAL_GGUF") {
        Ok(path) => Some(path),
        Err(_) => {
            eprintln!("Skipping test: set CROSSVAL_GGUF to path of test model");
            None
        }
    }
}

#[test]
fn test_single_token_logits_parity() -> Result<()> {
    let model_path = match test_model_path() {
        Some(p) => p,
        None => return Ok(()),
    };

    // Initialize C++ backend
    wrapper::init_backend();
    let _guard = scopeguard::guard((), |_| wrapper::free_backend());

    // Create C++ session
    let mut cpp_session = CppSession::load_deterministic(&model_path)?;

    // Test prompt
    let prompt = "The capital of France is";
    let tokens = cpp_session.tokenize(prompt)?;

    println!("Testing single-token generation with {} tokens", tokens.len());
    println!("Prompt: '{}'", prompt);

    // Get logits from both implementations for the entire prompt
    let cpp_logits_last = cpp_session.eval_and_get_logits(&tokens, 0)?;
    let rust_logits_last = eval_logits_once(&model_path, &tokens)?;

    // Compare the last-token logits
    assert_eq!(
        rust_logits_last.len(),
        cpp_logits_last.len(),
        "Logits vector length mismatch"
    );

    // Calculate differences
    let max_diff = rust_logits_last
        .iter()
        .zip(cpp_logits_last.iter())
        .map(|(r, c)| (r - c).abs())
        .fold(0.0f32, f32::max);

    println!("Single-token generation:");
    println!("  Max absolute difference: {:.6e}", max_diff);

    // For single-token comparison, we just check the last position
    let divergence = compare_per_position_logits(&vec![rust_logits_last], &vec![cpp_logits_last]);

    println!("  First divergence token: {:?}", divergence.first_divergence_token);
    println!("  Per-token cosine sim: {:?}", divergence.per_token_cosine_sim);
    println!("  Max absolute diff: {:.6e}", divergence.max_absolute_diff);

    // Should have no divergence for a single token
    if let Some(div_pos) = divergence.first_divergence_token {
        println!(
            "WARNING: Divergence detected at position {} (cosine sim threshold: {:.6e})",
            div_pos, COSINE_SIMILARITY_THRESHOLD
        );
    }

    // Ensure cosine similarity is high
    assert!(
        divergence.per_token_cosine_sim[0] > 0.9999,
        "Cosine similarity too low: {}",
        divergence.per_token_cosine_sim[0]
    );

    Ok(())
}

#[test]
fn test_multi_token_generation_divergence() -> Result<()> {
    let model_path = match test_model_path() {
        Some(p) => p,
        None => return Ok(()),
    };

    // Initialize C++ backend
    wrapper::init_backend();
    let _guard = scopeguard::guard((), |_| wrapper::free_backend());

    // Create C++ session
    let mut cpp_session = CppSession::load_deterministic(&model_path)?;

    // Test prompt
    let prompt = "2+2=";
    let initial_tokens = cpp_session.tokenize(prompt)?;

    println!("Testing multi-token generation with {} initial tokens", initial_tokens.len());
    println!("Prompt: '{}'", prompt);

    // Generate 5 tokens step-by-step and collect logits at each position
    let max_new_tokens = 5;
    let mut rust_all_logits = Vec::new();
    let mut cpp_all_logits = Vec::new();
    let mut current_tokens = initial_tokens.clone();

    for step in 0..max_new_tokens {
        println!("\n=== Step {} ===", step);

        // Get logits from both implementations
        let cpp_logits = cpp_session.eval_and_get_logits(&current_tokens, 0)?;
        let rust_logits = eval_logits_once(&model_path, &current_tokens)?;

        // Store logits for comparison
        rust_all_logits.push(rust_logits.clone());
        cpp_all_logits.push(cpp_logits.clone());

        // Sample next token (greedy - argmax)
        let cpp_next_token = cpp_session.context.sample_greedy(&cpp_logits);
        let rust_next_token = argmax(&rust_logits);

        println!("C++ next token: {}", cpp_next_token);
        println!("Rust next token: {}", rust_next_token);

        // Check if tokens match
        if cpp_next_token != rust_next_token {
            println!("WARNING: Token mismatch at step {}", step);
        }

        // Add the next token to the sequence
        current_tokens.push(cpp_next_token);
    }

    // Now compare all logits position-by-position
    println!("\n=== Per-Position Logits Comparison ===");
    let divergence = compare_per_position_logits(&rust_all_logits, &cpp_all_logits);

    println!("First divergence token: {:?}", divergence.first_divergence_token);
    println!("Max absolute diff: {:.6e}", divergence.max_absolute_diff);

    for (i, (cosine_sim, l2_dist)) in divergence
        .per_token_cosine_sim
        .iter()
        .zip(divergence.per_token_l2_dist.iter())
        .enumerate()
    {
        println!("  Position {}: cosine_sim={:.6}, L2_dist={:.6e}", i, cosine_sim, l2_dist);
    }

    // Report if there's a divergence
    if let Some(div_pos) = divergence.first_divergence_token {
        println!(
            "\nDivergence detected at position {} (cosine sim threshold: {:.6e})",
            div_pos, COSINE_SIMILARITY_THRESHOLD
        );
        println!(
            "  Cosine similarity at divergence: {:.6}",
            divergence.per_token_cosine_sim[div_pos]
        );
        println!("  L2 distance at divergence: {:.6e}", divergence.per_token_l2_dist[div_pos]);
    } else {
        println!("\nNo divergence detected across {} positions", rust_all_logits.len());
    }

    // We don't assert no divergence here, as this test is for detection
    // Just ensure the comparison ran successfully
    assert_eq!(divergence.per_token_cosine_sim.len(), max_new_tokens);
    assert_eq!(divergence.per_token_l2_dist.len(), max_new_tokens);

    Ok(())
}

#[test]
fn test_prefill_decode_logits_comparison() -> Result<()> {
    let model_path = match test_model_path() {
        Some(p) => p,
        None => return Ok(()),
    };

    // Initialize C++ backend
    wrapper::init_backend();
    let _guard = scopeguard::guard((), |_| wrapper::free_backend());

    // Create C++ session
    let mut cpp_session = CppSession::load_deterministic(&model_path)?;

    // Test prompt with multiple tokens
    let prompt = "The capital of France is Paris, and the capital of Germany is";
    let tokens = cpp_session.tokenize(prompt)?;

    println!("Testing prefill vs decode with {} tokens", tokens.len());
    println!("Prompt: '{}'", prompt);

    // Phase 1: Prefill - evaluate all tokens at once
    println!("\n=== Prefill Phase ===");
    let cpp_prefill_logits = cpp_session.eval_and_get_logits(&tokens, 0)?;
    let rust_prefill_logits = eval_logits_once(&model_path, &tokens)?;

    let prefill_divergence = compare_per_position_logits(
        &vec![rust_prefill_logits.clone()],
        &vec![cpp_prefill_logits.clone()],
    );

    println!("Prefill logits:");
    println!("  Cosine similarity: {:.6}", prefill_divergence.per_token_cosine_sim[0]);
    println!("  L2 distance: {:.6e}", prefill_divergence.per_token_l2_dist[0]);
    println!("  Max absolute diff: {:.6e}", prefill_divergence.max_absolute_diff);

    // Phase 2: Decode - generate one new token
    println!("\n=== Decode Phase ===");

    // Get the next token using greedy sampling from C++
    let cpp_next_token = cpp_session.context.sample_greedy(&cpp_prefill_logits);
    println!("C++ next token: {}", cpp_next_token);

    // Evaluate with the new token appended
    let mut tokens_with_next = tokens.clone();
    tokens_with_next.push(cpp_next_token);

    let cpp_decode_logits = cpp_session.eval_and_get_logits(&tokens_with_next, 0)?;
    let rust_decode_logits = eval_logits_once(&model_path, &tokens_with_next)?;

    let decode_divergence =
        compare_per_position_logits(&vec![rust_decode_logits], &vec![cpp_decode_logits]);

    println!("Decode logits:");
    println!("  Cosine similarity: {:.6}", decode_divergence.per_token_cosine_sim[0]);
    println!("  L2 distance: {:.6e}", decode_divergence.per_token_l2_dist[0]);
    println!("  Max absolute diff: {:.6e}", decode_divergence.max_absolute_diff);

    // Both prefill and decode should have high similarity
    assert!(
        prefill_divergence.per_token_cosine_sim[0] > 0.9999,
        "Prefill cosine similarity too low"
    );
    assert!(
        decode_divergence.per_token_cosine_sim[0] > 0.9999,
        "Decode cosine similarity too low"
    );

    Ok(())
}

/// Helper to get argmax token from logits
fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

#[test]
fn test_logits_compare_module() {
    // Unit test for the logits_compare module (no FFI required)
    use bitnet_crossval::logits_compare::compare_per_position_logits;

    // Create identical logits for 3 positions
    let rs_logits = vec![
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.5, 0.6, 0.7, 0.8],
        vec![0.9, 1.0, 1.1, 1.2],
    ];
    let cpp_logits = rs_logits.clone();

    let divergence = compare_per_position_logits(&rs_logits, &cpp_logits);

    // No divergence expected
    assert!(divergence.first_divergence_token.is_none());
    assert_eq!(divergence.per_token_cosine_sim.len(), 3);
    assert_eq!(divergence.per_token_l2_dist.len(), 3);

    // All cosine similarities should be 1.0
    for (i, sim) in divergence.per_token_cosine_sim.iter().enumerate() {
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Position {} cosine similarity: {} (expected 1.0)",
            i,
            sim
        );
    }

    // All L2 distances should be 0.0
    for (i, dist) in divergence.per_token_l2_dist.iter().enumerate() {
        assert!(dist.abs() < 1e-6, "Position {} L2 distance: {} (expected 0.0)", i, dist);
    }

    println!("Logits compare module test passed!");
}
