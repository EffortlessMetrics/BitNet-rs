use anyhow::Result;
use bitnet_tokenizers::Tokenizer;
use std::path::Path;
use std::sync::Arc;

/// Tokenizer resolution output normalized for the generation command.
pub(crate) struct LoadedTokenizer {
    pub(crate) tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    pub(crate) source: bitnet_tokenizers::auto::TokenizerSource,
    pub(crate) strict: bool,
}

/// Resolves the tokenizer authority, applying strict-mode and mock fallback
/// policy in one place.
pub(crate) fn load_generation_tokenizer(
    model_path: &Path,
    tokenizer_path: Option<&Path>,
    is_hf_directory: bool,
    effective_strict_tokenizer: bool,
    allow_mock: bool,
) -> Result<LoadedTokenizer> {
    match bitnet_tokenizers::auto::resolve_tokenizer(
        model_path,
        tokenizer_path,
        effective_strict_tokenizer,
    ) {
        Ok(resolution) => {
            match resolution.source {
                bitnet_tokenizers::auto::TokenizerSource::Explicit
                | bitnet_tokenizers::auto::TokenizerSource::Sibling => {
                    if let Some(path) = &resolution.path {
                        println!("Loading tokenizer from: {}", path.display());
                    }
                }
                bitnet_tokenizers::auto::TokenizerSource::GgufMetadata => {
                    println!("Successfully loaded tokenizer from GGUF metadata");
                }
                bitnet_tokenizers::auto::TokenizerSource::CompatibilityFallback => {}
            }
            Ok(LoadedTokenizer {
                tokenizer: resolution.tokenizer,
                source: resolution.source,
                strict: resolution.strict,
            })
        }
        Err(e) => {
            crate::answer_corpus_child_phase(
                "tokenizer_load_error",
                serde_json::json!({
                    "strict_tokenizer": effective_strict_tokenizer,
                    "allow_mock": allow_mock,
                    "error": e.to_string(),
                }),
            );
            if effective_strict_tokenizer {
                eprintln!("Strict tokenizer failed: {e}");
                std::process::exit(crate::EXIT_STRICT_TOKENIZER);
            }
            if !allow_mock {
                let model_dir = if is_hf_directory {
                    model_path
                } else {
                    model_path.parent().unwrap_or_else(|| Path::new("."))
                };
                anyhow::bail!(
                    "{e}\n\
                     \n\
                     No tokenizer found. Solutions:\n\
                     1. Download tokenizer:\n\
                        cargo run -p xtask -- tokenizer --into {}\n\
                     2. Provide explicit tokenizer path:\n\
                        --tokenizer /path/to/tokenizer.json\n\
                     3. Use mock tokenizer for testing only:\n\
                        --allow-mock",
                    model_dir.display()
                );
            }
            println!("Warning: Using mock tokenizer due to: {e}");
            Ok(LoadedTokenizer {
                tokenizer: Arc::new(bitnet_tokenizers::MockTokenizer::new()),
                source: bitnet_tokenizers::auto::TokenizerSource::CompatibilityFallback,
                strict: false,
            })
        }
    }
}
