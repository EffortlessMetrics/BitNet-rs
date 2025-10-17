//! Pure-Rust GGUF tokenizer loader
//!
//! This module provides support for loading tokenizers directly from GGUF model files.
//! It can extract and instantiate both SentencePiece (SPM) and Byte-Pair Encoding (BPE)
//! tokenizers from GGUF metadata without requiring external tokenizer files.
//!
//! The loader supports:
//! - **SentencePiece (SPM)**: Loaded from serialized protobuf blobs in GGUF metadata
//! - **BPE**: Constructed from vocab and merge arrays in GGUF metadata
//!
//! # Architecture
//!
//! The loader follows a fail-closed security model:
//! 1. Detect tokenizer kind from `tokenizer.ggml.model` metadata
//! 2. Extract special token IDs (bos, eos, eot) from metadata
//! 3. Load tokenizer-specific data (protobuf for SPM, vocab/merges for BPE)
//! 4. Return error if required data is missing or invalid
//!
//! # Example
//!
//! ```no_run
//! use bitnet_models::formats::gguf::GgufReader;
//! use bitnet_tokenizers::gguf_loader::RustTokenizer;
//!
//! # fn example(reader: &GgufReader) -> anyhow::Result<()> {
//! // Load tokenizer from GGUF metadata
//! let tokenizer = RustTokenizer::from_gguf(reader)?;
//!
//! // Encode text with BOS token
//! let tokens = tokenizer.encode("Hello world", true, false)?;
//!
//! // Get special token ID
//! let eos_id = tokenizer.id_for_special("<|eos|>");
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use bitnet_models::formats::gguf::GgufReader;

/// Tokenizer kind detected from GGUF metadata
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTokKind {
    /// SentencePiece tokenizer (LLaMA, Mistral, etc.)
    Spm,
    /// Byte-Pair Encoding tokenizer (GPT-2, GPT-J, etc.)
    Bpe,
}

/// Pure-Rust tokenizer loaded from GGUF metadata
///
/// This struct wraps either a SentencePiece or BPE tokenizer and provides
/// a unified interface for encoding text and querying special tokens.
///
/// The tokenizer is constructed from GGUF metadata arrays and special token
/// IDs, allowing models to be self-contained without external tokenizer files.
pub struct RustTokenizer {
    /// Tokenizer kind (SPM or BPE)
    kind: GgufTokKind,
    /// SentencePiece processor (when kind == Spm)
    #[cfg(feature = "spm")]
    spm: Option<sentencepiece::SentencePieceProcessor>,
    /// HuggingFace BPE tokenizer (when kind == Bpe)
    bpe: Option<tokenizers::Tokenizer>,
    /// Beginning-of-sequence token ID
    bos_id: Option<u32>,
    /// End-of-sequence token ID
    eos_id: Option<u32>,
    /// End-of-turn token ID (for LLaMA-3 chat)
    eot_id: Option<u32>,
    /// Hint for whether to add BOS by default
    add_bos_hint: Option<bool>,
}

impl RustTokenizer {
    /// Load tokenizer from GGUF metadata
    ///
    /// This method detects the tokenizer kind, extracts special token IDs,
    /// and loads the appropriate tokenizer implementation from GGUF metadata.
    ///
    /// # Metadata keys used
    ///
    /// ## Common keys
    /// - `tokenizer.ggml.model`: Tokenizer kind ("llama" → SPM, "gpt2"/"bpe" → BPE)
    /// - `tokenizer.ggml.bos_token_id`: Beginning-of-sequence token ID (u32)
    /// - `tokenizer.ggml.eos_token_id`: End-of-sequence token ID (u32)
    /// - `tokenizer.ggml.add_bos_token`: Whether to add BOS by default (bool)
    ///
    /// ## LLaMA-3 specific
    /// - `tokenizer.ggml.eot_token_id`: End-of-turn token ID (u32)
    ///
    /// ## SPM-specific keys
    /// - `tokenizer.model`: Serialized SentencePiece protobuf (binary array)
    /// - `tokenizer.spm.model`: Alternative location for SPM protobuf
    /// - `sentencepiece.model`: Legacy location for SPM protobuf
    ///
    /// ## BPE-specific keys
    /// - `tokenizer.ggml.tokens`: Vocabulary array (strings)
    /// - `tokenizer.ggml.bpe_merges`: Merge rules array (strings)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tokenizer kind cannot be detected from metadata
    /// - Required tokenizer data is missing (SPM protobuf or BPE vocab/merges)
    /// - Tokenizer construction fails (invalid data)
    /// - SPM feature is not enabled when loading SPM tokenizer
    ///
    /// # Security
    ///
    /// This method follows a fail-closed security model: if required data is
    /// missing or invalid, it returns an error rather than silently falling back
    /// to a default tokenizer.
    pub fn from_gguf(reader: &GgufReader) -> Result<Self> {
        // 1. Detect tokenizer kind from metadata
        let kind = Self::detect_kind(reader)?;

        // 2. Extract special token IDs from metadata
        let bos_id = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        let eos_id = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
        let eot_id = reader.get_u32_metadata("tokenizer.ggml.eot_token_id");
        let add_bos_hint = reader.get_bool_metadata("tokenizer.ggml.add_bos_token");

        tracing::debug!(
            "GGUF tokenizer metadata: kind={:?}, bos={:?}, eos={:?}, eot={:?}, add_bos={:?}",
            kind,
            bos_id,
            eos_id,
            eot_id,
            add_bos_hint
        );

        // 3. Load tokenizer based on detected kind
        match kind {
            GgufTokKind::Spm => {
                #[cfg(feature = "spm")]
                {
                    let spm = Self::load_spm(reader)?;
                    Ok(Self {
                        kind,
                        spm: Some(spm),
                        bpe: None,
                        bos_id,
                        eos_id,
                        eot_id,
                        add_bos_hint,
                    })
                }
                #[cfg(not(feature = "spm"))]
                {
                    anyhow::bail!(
                        "GGUF contains SentencePiece tokenizer, but `spm` feature is not enabled. \
                        Build with `--features spm` to load SentencePiece tokenizers."
                    );
                }
            }
            GgufTokKind::Bpe => {
                let bpe = Self::load_bpe(reader)?;
                Ok(Self {
                    kind,
                    #[cfg(feature = "spm")]
                    spm: None,
                    bpe: Some(bpe),
                    bos_id,
                    eos_id,
                    eot_id,
                    add_bos_hint,
                })
            }
        }
    }

    /// Detect tokenizer kind from GGUF metadata
    ///
    /// The tokenizer kind is detected from the `tokenizer.ggml.model` metadata key:
    /// - "llama" → SentencePiece (SPM)
    /// - "gpt2" or "bpe" → Byte-Pair Encoding (BPE)
    ///
    /// Returns an error if the metadata key is missing or contains an unsupported value.
    fn detect_kind(reader: &GgufReader) -> Result<GgufTokKind> {
        let model_type = reader
            .get_string_metadata("tokenizer.ggml.model")
            .context("Missing tokenizer.ggml.model metadata - cannot detect tokenizer kind")?;

        match model_type.to_lowercase().as_str() {
            "llama" => Ok(GgufTokKind::Spm),
            "gpt2" | "bpe" => Ok(GgufTokKind::Bpe),
            other => {
                anyhow::bail!(
                    "Unsupported tokenizer model type '{}'. Supported types: llama (SPM), gpt2/bpe (BPE)",
                    other
                )
            }
        }
    }

    /// Load SentencePiece tokenizer from GGUF metadata
    ///
    /// Searches for serialized SentencePiece protobuf in these metadata keys (in order):
    /// 1. `tokenizer.model`
    /// 2. `tokenizer.spm.model`
    /// 3. `sentencepiece.model`
    ///
    /// Returns an error if no protobuf is found or if deserialization fails.
    #[cfg(feature = "spm")]
    fn load_spm(reader: &GgufReader) -> Result<sentencepiece::SentencePieceProcessor> {
        // Try multiple locations for SPM protobuf (different GGUF exporters use different keys)
        let blob = reader
            .get_bin_or_u8_array("tokenizer.model")
            .or_else(|| reader.get_bin_or_u8_array("tokenizer.spm.model"))
            .or_else(|| reader.get_bin_or_u8_array("sentencepiece.model"))
            .context(
                "Missing SentencePiece protobuf in GGUF metadata. Expected one of: \
                tokenizer.model, tokenizer.spm.model, sentencepiece.model",
            )?;

        tracing::debug!("Loading SentencePiece from {} bytes of protobuf", blob.len());

        sentencepiece::SentencePieceProcessor::from_serialized_proto(&blob)
            .map_err(|e| anyhow::anyhow!("Failed to load SentencePiece from protobuf: {}", e))
            .context(
                "SentencePiece deserialization failed - protobuf may be corrupted or incompatible",
            )
    }

    /// Load BPE tokenizer from GGUF metadata
    ///
    /// Constructs a BPE tokenizer from vocabulary and merge arrays in GGUF metadata:
    /// - `tokenizer.ggml.tokens`: Array of token strings (indexed by token ID)
    /// - `tokenizer.ggml.bpe_merges`: Array of merge rules ("token1 token2")
    ///
    /// The vocabulary is converted to a HashMap for the BPE model, and a ByteLevel
    /// pre-tokenizer and decoder are added for compatibility with GPT-2 style models.
    ///
    /// Returns an error if vocab or merges are missing or invalid.
    fn load_bpe(reader: &GgufReader) -> Result<tokenizers::Tokenizer> {
        use ahash::AHashMap;
        use tokenizers::{
            decoders::byte_level::ByteLevel, models::bpe::BPE, pre_tokenizers::byte_level,
        };

        // Load vocabulary array from metadata
        let vocab_strings = reader
            .get_string_array_metadata("tokenizer.ggml.tokens")
            .context("Missing tokenizer.ggml.tokens metadata - required for BPE tokenizer")?;

        tracing::debug!("Loading BPE tokenizer with {} vocab tokens", vocab_strings.len());

        // Convert to AHashMap<String, u32> indexed by position
        let vocab: AHashMap<String, u32> =
            vocab_strings.into_iter().enumerate().map(|(i, tok)| (tok, i as u32)).collect();

        // Load merge rules array from metadata (try multiple possible keys)
        let merges_strings = reader
            .get_string_array_metadata("tokenizer.ggml.merges")
            .or_else(|| reader.get_string_array_metadata("tokenizer.ggml.bpe_merges"))
            .context("Missing tokenizer.ggml.merges or tokenizer.ggml.bpe_merges metadata - required for BPE tokenizer")?;

        tracing::debug!("Loading BPE tokenizer with {} merge rules", merges_strings.len());

        // Convert merge strings "token1 token2" to pairs
        let merges: Vec<(String, String)> = merges_strings
            .iter()
            .filter_map(|m| {
                let mut parts = m.split_whitespace();
                let a = parts.next()?.to_string();
                let b = parts.next()?.to_string();
                Some((a, b))
            })
            .collect();

        if merges.len() != merges_strings.len() {
            tracing::warn!(
                "Some BPE merge rules were invalid (got {} valid merges from {} strings)",
                merges.len(),
                merges_strings.len()
            );
        }

        // Build BPE model from vocab and merges
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build BPE tokenizer: {}", e))
            .context("BPE tokenizer construction failed - vocab or merges may be invalid")?;

        // Create tokenizer with ByteLevel pre-tokenizer and decoder (GPT-2 style)
        // NOTE: add_prefix_space=true ensures first token is treated like subsequent tokens
        // (llama.cpp GPT-2 behavior: pretend there is a space before the first token)
        let mut tokenizer = tokenizers::Tokenizer::new(bpe);
        tokenizer.with_pre_tokenizer(Some(byte_level::ByteLevel::default().add_prefix_space(true)));
        tokenizer.with_decoder(Some(ByteLevel::default()));

        Ok(tokenizer)
    }

    /// Encode text to token IDs
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to tokenize
    /// * `add_bos` - Whether to prepend BOS token (if available)
    /// * `parse_special` - Whether to parse special tokens in text (BPE only)
    ///
    /// # Returns
    ///
    /// Vector of token IDs
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails (e.g., invalid UTF-8 or tokenizer error)
    pub fn encode(&self, text: &str, add_bos: bool, parse_special: bool) -> Result<Vec<u32>> {
        match self.kind {
            GgufTokKind::Spm => {
                #[cfg(feature = "spm")]
                {
                    let spm = self.spm.as_ref().expect("SPM tokenizer should be present");

                    // Encode text using SentencePiece
                    let pieces = spm
                        .encode(text)
                        .map_err(|e| anyhow::anyhow!("SentencePiece encode error: {}", e))?;

                    let mut ids: Vec<u32> = pieces.into_iter().map(|p| p.id).collect();

                    // Prepend BOS if requested and not already present
                    if add_bos
                        && let Some(bos) = self.bos_id
                        && ids.first().copied() != Some(bos)
                    {
                        ids.insert(0, bos);
                    }

                    Ok(ids)
                }
                #[cfg(not(feature = "spm"))]
                {
                    anyhow::bail!(
                        "SPM tokenizer loaded but `spm` feature is not enabled. \
                        This should never happen - please report this bug."
                    );
                }
            }
            GgufTokKind::Bpe => {
                let bpe = self.bpe.as_ref().expect("BPE tokenizer should be present");

                // Encode text using BPE
                let encoding = bpe
                    .encode(text, parse_special)
                    .map_err(|e| anyhow::anyhow!("BPE encode error: {}", e))?;

                let mut ids = encoding.get_ids().to_vec();

                // Prepend BOS if requested and not already present
                if add_bos
                    && let Some(bos) = self.bos_id
                    && ids.first().copied() != Some(bos)
                {
                    ids.insert(0, bos);
                }

                Ok(ids)
            }
        }
    }

    /// Get token ID for a special token piece
    ///
    /// This method maps special token strings to their IDs. It recognizes
    /// common special token patterns used across different tokenizer formats.
    ///
    /// # Supported tokens
    ///
    /// - `<|eot_id|>`, `<|end_of_turn|>` → `eot_id` (LLaMA-3 chat)
    /// - `<|eos|>`, `</s>`, `<eos>` → `eos_id`
    /// - `<|bos|>`, `<s>`, `<bos>` → `bos_id`
    ///
    /// # Arguments
    ///
    /// * `piece` - Special token string to look up
    ///
    /// # Returns
    ///
    /// Token ID if the special token is recognized and configured, None otherwise
    pub fn id_for_special(&self, piece: &str) -> Option<u32> {
        match piece {
            // LLaMA-3 end-of-turn tokens
            "<|eot_id|>" | "<|end_of_turn|>" => self.eot_id,
            // Common EOS patterns
            "<|eos|>" | "</s>" | "<eos>" => self.eos_id,
            // Common BOS patterns
            "<|bos|>" | "<s>" | "<bos>" => self.bos_id,
            _ => None,
        }
    }

    /// Get tokenizer kind
    pub fn kind(&self) -> GgufTokKind {
        self.kind
    }

    /// Get BOS token ID
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    /// Get EOS token ID
    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    /// Get EOT (end-of-turn) token ID
    pub fn eot_id(&self) -> Option<u32> {
        self.eot_id
    }

    /// Get hint for whether to add BOS by default
    pub fn add_bos_hint(&self) -> Option<bool> {
        self.add_bos_hint
    }
}

// Implement the Tokenizer trait for RustTokenizer
impl crate::Tokenizer for RustTokenizer {
    fn encode(
        &self,
        text: &str,
        add_bos: bool,
        add_special: bool,
    ) -> bitnet_common::Result<Vec<u32>> {
        // Use the existing encode method with parse_special parameter
        self.encode(text, add_bos, add_special).map_err(|e| {
            bitnet_common::BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: e.to_string(),
            })
        })
    }

    fn decode(&self, tokens: &[u32]) -> bitnet_common::Result<String> {
        match self.kind {
            GgufTokKind::Spm => {
                #[cfg(feature = "spm")]
                {
                    let spm = self.spm.as_ref().expect("SPM tokenizer should be present");

                    // Decode using SentencePiece (decode_piece_ids handles special tokens)
                    spm.decode_piece_ids(tokens).map_err(|e| {
                        bitnet_common::BitNetError::Model(
                            bitnet_common::ModelError::LoadingFailed {
                                reason: format!("SentencePiece decode error: {}", e),
                            },
                        )
                    })
                }
                #[cfg(not(feature = "spm"))]
                {
                    Err(bitnet_common::BitNetError::Model(
                        bitnet_common::ModelError::LoadingFailed {
                            reason: "SPM tokenizer loaded but `spm` feature is not enabled"
                                .to_string(),
                        },
                    ))
                }
            }
            GgufTokKind::Bpe => {
                let bpe = self.bpe.as_ref().expect("BPE tokenizer should be present");

                // Decode using BPE tokenizer
                bpe.decode(tokens, true).map_err(|e| {
                    bitnet_common::BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                        reason: format!("BPE decode error: {}", e),
                    })
                })
            }
        }
    }

    fn vocab_size(&self) -> usize {
        match self.kind {
            GgufTokKind::Spm => {
                #[cfg(feature = "spm")]
                {
                    self.spm.as_ref().map_or(0, |spm| spm.len())
                }
                #[cfg(not(feature = "spm"))]
                {
                    0
                }
            }
            GgufTokKind::Bpe => self.bpe.as_ref().map_or(0, |bpe| bpe.get_vocab_size(true)),
        }
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        match self.kind {
            GgufTokKind::Spm => {
                #[cfg(feature = "spm")]
                {
                    self.spm.as_ref().and_then(|spm| spm.decode_piece_ids(&[token]).ok())
                }
                #[cfg(not(feature = "spm"))]
                {
                    None
                }
            }
            GgufTokKind::Bpe => {
                self.bpe.as_ref().and_then(|bpe| bpe.id_to_token(token).map(|s| s.to_string()))
            }
        }
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_id
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kind_detection() {
        // This would require a real GgufReader with metadata
        // For now, just verify the types compile
        let _ = GgufTokKind::Spm;
        let _ = GgufTokKind::Bpe;
    }

    #[test]
    fn test_special_token_lookup() {
        let tok = RustTokenizer {
            kind: GgufTokKind::Spm,
            #[cfg(feature = "spm")]
            spm: None,
            bpe: None,
            bos_id: Some(1),
            eos_id: Some(2),
            eot_id: Some(128009),
            add_bos_hint: Some(true),
        };

        // Test BOS patterns
        assert_eq!(tok.id_for_special("<|bos|>"), Some(1));
        assert_eq!(tok.id_for_special("<s>"), Some(1));
        assert_eq!(tok.id_for_special("<bos>"), Some(1));

        // Test EOS patterns
        assert_eq!(tok.id_for_special("<|eos|>"), Some(2));
        assert_eq!(tok.id_for_special("</s>"), Some(2));
        assert_eq!(tok.id_for_special("<eos>"), Some(2));

        // Test EOT patterns
        assert_eq!(tok.id_for_special("<|eot_id|>"), Some(128009));
        assert_eq!(tok.id_for_special("<|end_of_turn|>"), Some(128009));

        // Test unknown token
        assert_eq!(tok.id_for_special("<unknown>"), None);
    }

    #[test]
    fn test_getters() {
        let tok = RustTokenizer {
            kind: GgufTokKind::Bpe,
            #[cfg(feature = "spm")]
            spm: None,
            bpe: None,
            bos_id: Some(10),
            eos_id: Some(11),
            eot_id: Some(12),
            add_bos_hint: Some(false),
        };

        assert_eq!(tok.kind(), GgufTokKind::Bpe);
        assert_eq!(tok.bos_id(), Some(10));
        assert_eq!(tok.eos_id(), Some(11));
        assert_eq!(tok.eot_id(), Some(12));
        assert_eq!(tok.add_bos_hint(), Some(false));
    }
}
