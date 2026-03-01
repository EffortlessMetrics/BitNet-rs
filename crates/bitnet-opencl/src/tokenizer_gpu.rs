//! GPU-accelerated tokenization support.
//!
//! Provides a `GpuTokenizer` wrapper with batch encode/decode and a
//! GPU-ready interface.  The MVP implementation delegates to CPU-side
//! byte-level encoding; future versions will offload to `OpenCL` kernels.

use std::collections::HashMap;

/// Padding strategy for batch tokenization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad all sequences to the longest in the batch.
    Longest,
    /// Pad all sequences to `max_length`.
    MaxLength,
    /// No padding.
    None,
}

/// Truncation strategy for sequences exceeding `max_length`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Keep the first `max_length` tokens.
    TruncateLeft,
    /// Keep the last `max_length` tokens.
    TruncateRight,
    /// Do not truncate.
    None,
}

/// Configuration for the GPU tokenizer.
#[derive(Debug, Clone)]
pub struct GpuTokenizerConfig {
    /// Maximum sequence length (0 = unlimited).
    pub max_length: usize,
    /// Padding strategy.
    pub padding: PaddingStrategy,
    /// Truncation strategy.
    pub truncation: TruncationStrategy,
    /// Vocabulary size.
    pub vocab_size: usize,
}

impl Default for GpuTokenizerConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            padding: PaddingStrategy::Longest,
            truncation: TruncationStrategy::TruncateLeft,
            vocab_size: 50257,
        }
    }
}

/// GPU-ready tokenizer wrapper.
///
/// For the MVP this performs byte-level tokenization on the CPU while
/// exposing batch-oriented APIs suitable for GPU offloading.
pub struct GpuTokenizer {
    config: GpuTokenizerConfig,
    /// Optional special-token map (token string → id).
    special_tokens: HashMap<String, u32>,
}

impl GpuTokenizer {
    /// Create a new `GpuTokenizer` with the given config.
    pub fn new(config: GpuTokenizerConfig) -> Self {
        Self { config, special_tokens: HashMap::new() }
    }

    /// Create a tokenizer with default settings.
    pub fn with_defaults() -> Self {
        Self::new(GpuTokenizerConfig::default())
    }

    /// Register a special token.
    pub fn add_special_token(&mut self, token: &str, id: u32) {
        self.special_tokens.insert(token.to_owned(), id);
    }

    /// Return an immutable reference to the config.
    pub const fn config(&self) -> &GpuTokenizerConfig {
        &self.config
    }

    // -- encoding ---------------------------------------------------------

    /// Encode a single text to a token-id sequence (byte-level MVP).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = text.bytes().map(u32::from).collect();
        self.apply_truncation(&mut ids);
        ids
    }

    /// Encode a batch of texts in parallel (conceptually).
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    // -- decoding ---------------------------------------------------------

    /// Decode a single token-id sequence back to a string.
    pub fn decode(&self, ids: &[u32]) -> String {
        #[allow(clippy::cast_possible_truncation)] // id < 256 guaranteed by filter
        let bytes: Vec<u8> = ids.iter().filter(|&&id| id < 256).map(|&id| id as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode a batch of token-id sequences.
    pub fn decode_batch(&self, id_seqs: &[&[u32]]) -> Vec<String> {
        id_seqs.iter().map(|ids| self.decode(ids)).collect()
    }

    // -- padding & masks --------------------------------------------------

    /// Pad sequences to equal length, returning the padded sequences and
    /// original lengths.
    pub fn pad_sequences(
        &self,
        sequences: &[Vec<u32>],
        pad_id: u32,
    ) -> (Vec<Vec<u32>>, Vec<usize>) {
        if sequences.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let max_len = match self.config.padding {
            PaddingStrategy::MaxLength => self.config.max_length,
            PaddingStrategy::Longest => sequences.iter().map(Vec::len).max().unwrap_or(0),
            PaddingStrategy::None => {
                let lengths: Vec<usize> = sequences.iter().map(Vec::len).collect();
                return (sequences.to_vec(), lengths);
            }
        };

        let mut padded = Vec::with_capacity(sequences.len());
        let mut lengths = Vec::with_capacity(sequences.len());
        for seq in sequences {
            lengths.push(seq.len());
            let mut s = seq.clone();
            if s.len() < max_len {
                s.resize(max_len, pad_id);
            }
            padded.push(s);
        }
        (padded, lengths)
    }

    /// Create binary attention masks from original lengths.
    pub fn create_attention_mask(lengths: &[usize], max_len: usize) -> Vec<Vec<u8>> {
        lengths.iter().map(|&len| (0..max_len).map(|i| u8::from(i < len)).collect()).collect()
    }

    // -- helpers ----------------------------------------------------------

    fn apply_truncation(&self, ids: &mut Vec<u32>) {
        let max = self.config.max_length;
        if max == 0 || ids.len() <= max {
            return;
        }
        match self.config.truncation {
            TruncationStrategy::TruncateLeft => {
                ids.truncate(max);
            }
            TruncationStrategy::TruncateRight => {
                let start = ids.len() - max;
                *ids = ids[start..].to_vec();
            }
            TruncationStrategy::None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_empty_returns_empty() {
        let tok = GpuTokenizer::with_defaults();
        assert!(tok.encode("").is_empty());
    }

    #[test]
    fn encode_ascii_byte_level() {
        let tok = GpuTokenizer::with_defaults();
        assert_eq!(tok.encode("AB"), vec![65, 66]);
    }

    #[test]
    fn decode_roundtrip_ascii() {
        let tok = GpuTokenizer::with_defaults();
        let ids = tok.encode("hello");
        assert_eq!(tok.decode(&ids), "hello");
    }

    #[test]
    fn truncation_left() {
        let cfg = GpuTokenizerConfig {
            max_length: 3,
            truncation: TruncationStrategy::TruncateLeft,
            ..Default::default()
        };
        let tok = GpuTokenizer::new(cfg);
        assert_eq!(tok.encode("abcde").len(), 3);
        assert_eq!(tok.encode("abcde"), vec![97, 98, 99]);
    }

    #[test]
    fn truncation_right() {
        let cfg = GpuTokenizerConfig {
            max_length: 3,
            truncation: TruncationStrategy::TruncateRight,
            ..Default::default()
        };
        let tok = GpuTokenizer::new(cfg);
        assert_eq!(tok.encode("abcde"), vec![99, 100, 101]);
    }

    #[test]
    fn no_truncation_when_within_limit() {
        let cfg = GpuTokenizerConfig { max_length: 10, ..Default::default() };
        let tok = GpuTokenizer::new(cfg);
        assert_eq!(tok.encode("abc").len(), 3);
    }
}
