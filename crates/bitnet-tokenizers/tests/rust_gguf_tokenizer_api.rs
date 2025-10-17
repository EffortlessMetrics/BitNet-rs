//! Integration tests for RustGgufTokenizer API
//!
//! These tests verify that the RustGgufTokenizer wrapper correctly
//! integrates with the public Tokenizer trait and provides expected API.

// Type aliases to simplify complex type signatures
type RustTokenizerFactory<'a> =
    &'a dyn Fn(&bitnet_models::GgufReader) -> anyhow::Result<bitnet_tokenizers::RustTokenizer>;
type TokenizerFactory<'a> =
    &'a dyn Fn(
        &bitnet_models::GgufReader,
    ) -> bitnet_common::Result<std::sync::Arc<dyn bitnet_tokenizers::Tokenizer>>;
type BosEosEotTuple = (Option<u32>, Option<u32>, Option<u32>);

#[test]
fn test_api_exports() {
    // Verify that the public API exports the expected types
    // This is a compile-time test - if it compiles, the exports are correct

    // Should be able to import the tokenizer types
    use bitnet_tokenizers::GgufTokKind;

    // Verify enum variants are accessible
    let _ = GgufTokKind::Spm;
    let _ = GgufTokKind::Bpe;

    // Type checks pass
    let _: Option<RustTokenizerFactory> = None;
    let _: Option<TokenizerFactory> = None;
}

#[test]
fn test_tokenizer_builder_signature() {
    // Verify that TokenizerBuilder::from_gguf_reader has the expected signature
    // This is a compile-time test

    use bitnet_tokenizers::TokenizerBuilder;

    // Should accept &GgufReader and return Result<Arc<dyn Tokenizer>>
    let _: fn(
        &bitnet_models::GgufReader,
    ) -> bitnet_common::Result<std::sync::Arc<dyn bitnet_tokenizers::Tokenizer>> =
        TokenizerBuilder::from_gguf_reader;
}

#[test]
fn test_rust_gguf_tokenizer_methods() {
    // Verify that RustGgufTokenizer exposes expected helper methods
    // This is a compile-time test

    use bitnet_tokenizers::RustGgufTokenizer;

    // Create a type-level check for the methods
    fn check_methods<T>(_t: &T)
    where
        T: ?Sized,
    {
        // This would fail to compile if the methods don't exist with correct signatures
        let _: fn(&RustGgufTokenizer) -> BosEosEotTuple = RustGgufTokenizer::bos_eos_eot;
        let _: fn(&RustGgufTokenizer) -> Option<bool> = RustGgufTokenizer::add_bos_hint;
        let _: fn(&RustGgufTokenizer) -> bitnet_tokenizers::GgufTokKind = RustGgufTokenizer::kind;
    }

    // Type check succeeds - function signature is verified
    let _ = check_methods::<()>;
}

#[test]
fn test_tokenizer_trait_implementation() {
    // Verify that RustGgufTokenizer implements Tokenizer trait
    // This is a compile-time test

    use bitnet_tokenizers::{RustGgufTokenizer, Tokenizer};

    // Should be able to use as dyn Tokenizer
    fn accepts_tokenizer(_t: &dyn Tokenizer) {}

    // This would fail to compile if RustGgufTokenizer doesn't impl Tokenizer
    fn test_with_rust_gguf(tok: &RustGgufTokenizer) {
        accepts_tokenizer(tok);
    }

    // Type check succeeds - function signature is verified
    let _ = test_with_rust_gguf;
}

#[cfg(feature = "spm")]
#[test]
fn test_feature_gating() {
    // Verify that SPM feature is properly gated
    // This test only compiles when spm feature is enabled

    use bitnet_tokenizers::GgufTokKind;

    // SPM variant should be available with spm feature
    let kind = GgufTokKind::Spm;
    assert_eq!(format!("{:?}", kind), "Spm");
}

#[test]
fn test_documentation_examples_compile() {
    // Verify that documentation examples would compile
    // This is a compile-time test

    // From RustGgufTokenizer docs
    #[allow(dead_code)]
    fn example_rust_gguf_tokenizer(path: &std::path::Path) -> bitnet_common::Result<()> {
        use bitnet_models::{GgufReader, loader::MmapFile};
        use bitnet_tokenizers::RustGgufTokenizer;

        let mmap = MmapFile::open(path)?;
        let reader = GgufReader::new(mmap.as_slice())?;
        let _tokenizer = RustGgufTokenizer::from_gguf(&reader).map_err(|e| {
            bitnet_common::BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: e.to_string(),
            })
        })?;
        Ok(())
    }

    // From TokenizerBuilder::from_gguf_reader docs
    #[allow(dead_code)]
    fn example_tokenizer_builder(path: &std::path::Path) -> bitnet_common::Result<()> {
        use bitnet_models::{GgufReader, loader::MmapFile};
        use bitnet_tokenizers::TokenizerBuilder;

        let mmap = MmapFile::open(path)?;
        let reader = GgufReader::new(mmap.as_slice())?;
        let _tokenizer = TokenizerBuilder::from_gguf_reader(&reader)?;
        Ok(())
    }
}
