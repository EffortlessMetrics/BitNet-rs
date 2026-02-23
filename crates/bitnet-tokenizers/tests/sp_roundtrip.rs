#![cfg(feature = "integration-tests")]
#![cfg(feature = "spm")]

#[test]
#[ignore = "Run with cargo test -- --ignored when SPM env var is set"]
fn sp_roundtrip() {
    use bitnet_tokenizers::Tokenizer;
    use bitnet_tokenizers::sp_tokenizer::SpTokenizer;
    use std::path::Path;

    let spm = std::env::var("SPM").unwrap_or_else(|_| {
        eprintln!("Set SPM=/path/to/tokenizer.model to run this test");
        String::new()
    });

    if spm.is_empty() {
        return;
    }

    let t = SpTokenizer::from_file(Path::new(&spm)).expect("load tokenizer");
    let ids = t.encode("The capital of France is", false, false).expect("encode");
    let txt = t.decode(&ids).expect("decode");

    println!("Input: 'The capital of France is'");
    println!("IDs: {:?}", ids);
    println!("Decoded: '{}'", txt);

    assert!(txt.to_lowercase().contains("france"), "Decoded text should contain 'france'");
    assert!(!ids.is_empty(), "Should produce some tokens");
}
