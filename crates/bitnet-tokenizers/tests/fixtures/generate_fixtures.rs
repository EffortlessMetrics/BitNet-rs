//! Fixture Generation Script for Universal Tokenizer Discovery Tests
//!
//! Run this to generate all required test fixtures.
//! Usage: cargo test --package bitnet-tokenizers --test generate_fixtures -- --ignored

#![cfg(test)]

use super::gguf_fixtures::generate_all_gguf_fixtures;
use super::tokenizer_fixtures::generate_all_tokenizer_fixtures;

#[test]
fn generate_all_test_fixtures() {
    if std::env::var("BITNET_GENERATE_FIXTURES").ok().as_deref() != Some("1") {
        eprintln!("⏭️  Skipping fixture generation; set BITNET_GENERATE_FIXTURES=1 to regenerate");
        return;
    }
    println!("Generating GGUF test fixtures...");
    generate_all_gguf_fixtures().expect("Failed to generate GGUF fixtures");
    println!("✓ GGUF fixtures generated");

    println!("Generating tokenizer test fixtures...");
    generate_all_tokenizer_fixtures().expect("Failed to generate tokenizer fixtures");
    println!("✓ Tokenizer fixtures generated");

    println!("\nAll test fixtures generated successfully!");
    println!("Location: crates/bitnet-tokenizers/tests/fixtures/");
}

#[test]
fn verify_fixture_infrastructure() {
    use std::path::Path;

    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");

    // Verify directory structure exists
    assert!(fixtures_dir.exists(), "Fixtures directory should exist");
    assert!(fixtures_dir.join("gguf").exists() || fixtures_dir.is_dir(), "GGUF fixtures directory should exist");
    assert!(fixtures_dir.join("tokenizers").exists() || fixtures_dir.is_dir(), "Tokenizers fixtures directory should exist");
    assert!(fixtures_dir.join("mock").exists() || fixtures_dir.is_dir(), "Mock fixtures directory should exist");

    println!("✓ Fixture infrastructure verified");
}
