//! Fixture Generation for Universal Tokenizer Discovery Tests
//!
//! Run this to generate all required test fixtures.
//! Usage: cargo test --package bitnet-tokenizers --test generate_test_fixtures -- --ignored

#![cfg(test)]

mod fixtures;

use fixtures::gguf_fixtures::generate_all_gguf_fixtures;
use fixtures::tokenizer_fixtures::generate_all_tokenizer_fixtures;

#[test]
#[ignore] // Only run when explicitly requested
fn generate_all_fixtures() {
    println!("Generating GGUF test fixtures...");
    generate_all_gguf_fixtures().expect("Failed to generate GGUF fixtures");
    println!("✓ GGUF fixtures generated");

    println!("Generating tokenizer test fixtures...");
    generate_all_tokenizer_fixtures().expect("Failed to generate tokenizer fixtures");
    println!("✓ Tokenizer fixtures generated");

    println!("\nAll test fixtures generated successfully!");
    println!("Location: crates/bitnet-tokenizers/tests/fixtures/");
}
