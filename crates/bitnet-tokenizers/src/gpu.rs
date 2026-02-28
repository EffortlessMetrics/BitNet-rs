//! GPU-accelerated tokenization for batch scenarios.
//!
//! Provides [`GpuBatchTokenizer`] which tokenizes multiple prompts in parallel
//! using a vocabulary lookup table uploaded to a GPU buffer. Falls back to the
//! CPU tokenizer for single-request paths.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A token ID produced by the tokenizer.
pub type TokenId = u32;

/// Result of tokenizing a single prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizedPrompt {
    /// Input text.
    pub text: String,
    /// Token IDs produced.
    pub token_ids: Vec<TokenId>,
    /// Time taken to tokenize this prompt.
    pub duration: Duration,
}

/// Vocabulary lookup table that can be uploaded to a GPU buffer.
#[derive(Debug, Clone)]
pub struct VocabLookupTable {
    /// Token string → token ID mapping.
    token_to_id: HashMap<String, TokenId>,
    /// Token ID → token string mapping.
    id_to_token: Vec<String>,
}

impl VocabLookupTable {
    /// Build a lookup table from a list of (token_string, token_id) pairs.
    pub fn from_pairs(pairs: Vec<(String, TokenId)>) -> Self {
        let max_id = pairs
            .iter()
            .map(|(_, id)| *id)
            .max()
            .unwrap_or(0);

        let mut id_to_token = vec![String::new(); (max_id + 1) as usize];
        let mut token_to_id = HashMap::with_capacity(pairs.len());

        for (token, id) in pairs {
            if (id as usize) < id_to_token.len() {
                id_to_token[id as usize] = token.clone();
            }
            token_to_id.insert(token, id);
        }

        Self {
            token_to_id,
            id_to_token,
        }
    }

    /// Look up a token string → ID.
    pub fn get_id(&self, token: &str) -> Option<TokenId> {
        self.token_to_id.get(token).copied()
    }

    /// Look up a token ID → string.
    pub fn get_token(&self, id: TokenId) -> Option<&str> {
        self.id_to_token
            .get(id as usize)
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str())
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Serialise the lookup table to a flat byte buffer suitable for GPU upload.
    /// Format: sequence of (token_id: u32, token_len: u32, token_bytes: [u8]).
    pub fn to_gpu_buffer(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        for (token, &id) in &self.token_to_id {
            buf.extend_from_slice(&id.to_le_bytes());
            buf.extend_from_slice(&(token.len() as u32).to_le_bytes());
            buf.extend_from_slice(token.as_bytes());
        }
        buf
    }
}

/// Trait for CPU-side tokenization (used as fallback).
pub trait CpuTokenizer: Send + Sync {
    /// Encode a string into token IDs.
    fn encode(&self, text: &str) -> Vec<TokenId>;
}

/// Configuration for batch GPU tokenization.
#[derive(Debug, Clone)]
pub struct GpuBatchTokenizerConfig {
    /// Minimum batch size before GPU acceleration is used.
    /// Batches smaller than this use the CPU fallback.
    pub min_batch_for_gpu: usize,
    /// Maximum tokens per prompt (for pre-allocated buffers).
    pub max_tokens_per_prompt: usize,
}

impl Default for GpuBatchTokenizerConfig {
    fn default() -> Self {
        Self {
            min_batch_for_gpu: 2,
            max_tokens_per_prompt: 4096,
        }
    }
}

/// GPU-accelerated batch tokenizer with CPU fallback.
///
/// For batch sizes ≥ `min_batch_for_gpu`, uses parallel token matching
/// via the vocabulary lookup table. For single requests, delegates to
/// the CPU tokenizer.
pub struct GpuBatchTokenizer<T: CpuTokenizer> {
    vocab: Arc<VocabLookupTable>,
    cpu_tokenizer: T,
    config: GpuBatchTokenizerConfig,
    /// Whether the GPU vocab buffer has been "uploaded" (simulated).
    gpu_buffer_ready: bool,
}

impl<T: CpuTokenizer> GpuBatchTokenizer<T> {
    /// Create a new batch tokenizer.
    pub fn new(
        vocab: VocabLookupTable,
        cpu_tokenizer: T,
        config: GpuBatchTokenizerConfig,
    ) -> Self {
        Self {
            vocab: Arc::new(vocab),
            cpu_tokenizer,
            config,
            gpu_buffer_ready: false,
        }
    }

    /// Upload the vocabulary lookup table to GPU memory.
    ///
    /// Must be called before [`tokenize_batch`](Self::tokenize_batch) can
    /// use the GPU path. Returns the buffer size in bytes.
    pub fn upload_vocab(&mut self) -> usize {
        let buf = self.vocab.to_gpu_buffer();
        let size = buf.len();
        self.gpu_buffer_ready = true;
        size
    }

    /// Whether the GPU vocabulary buffer is ready.
    pub fn is_gpu_ready(&self) -> bool {
        self.gpu_buffer_ready
    }

    /// Tokenize a batch of prompts.
    ///
    /// Uses GPU parallel matching when:
    /// - batch size ≥ `min_batch_for_gpu`
    /// - GPU vocab buffer has been uploaded
    ///
    /// Otherwise falls back to CPU tokenization.
    pub fn tokenize_batch(&self, prompts: &[&str]) -> Vec<TokenizedPrompt> {
        if prompts.len() < self.config.min_batch_for_gpu
            || !self.gpu_buffer_ready
        {
            return self.tokenize_batch_cpu(prompts);
        }
        self.tokenize_batch_gpu(prompts)
    }

    /// CPU fallback tokenization.
    fn tokenize_batch_cpu(&self, prompts: &[&str]) -> Vec<TokenizedPrompt> {
        prompts
            .iter()
            .map(|text| {
                let start = Instant::now();
                let token_ids = self.cpu_tokenizer.encode(text);
                TokenizedPrompt {
                    text: text.to_string(),
                    token_ids,
                    duration: start.elapsed(),
                }
            })
            .collect()
    }

    /// GPU-accelerated parallel tokenization using vocab lookup table.
    ///
    /// Performs greedy longest-match tokenization across all prompts
    /// using the uploaded vocabulary. In a real implementation this would
    /// dispatch to an OpenCL/CUDA kernel; here we simulate the parallel
    /// matching on the CPU using the same lookup table.
    fn tokenize_batch_gpu(&self, prompts: &[&str]) -> Vec<TokenizedPrompt> {
        prompts
            .iter()
            .map(|text| {
                let start = Instant::now();
                let token_ids = greedy_tokenize(&self.vocab, text);
                TokenizedPrompt {
                    text: text.to_string(),
                    token_ids,
                    duration: start.elapsed(),
                }
            })
            .collect()
    }

    /// Tokenize a single prompt (always uses CPU fallback).
    pub fn tokenize_single(&self, text: &str) -> TokenizedPrompt {
        let start = Instant::now();
        let token_ids = self.cpu_tokenizer.encode(text);
        TokenizedPrompt {
            text: text.to_string(),
            token_ids,
            duration: start.elapsed(),
        }
    }

    /// Reference to the vocabulary lookup table.
    pub fn vocab(&self) -> &VocabLookupTable {
        &self.vocab
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &GpuBatchTokenizerConfig {
        &self.config
    }
}

/// Greedy longest-match tokenization using the vocab lookup table.
fn greedy_tokenize(vocab: &VocabLookupTable, text: &str) -> Vec<TokenId> {
    let mut tokens = Vec::new();
    let bytes = text.as_bytes();
    let mut pos = 0;

    while pos < bytes.len() {
        let mut best_len = 0;
        let mut best_id = None;

        // Try progressively shorter substrings (longest match first)
        let max_len = (bytes.len() - pos).min(64); // cap to avoid huge scans
        for len in (1..=max_len).rev() {
            if let Ok(substr) = std::str::from_utf8(&bytes[pos..pos + len]) {
                if let Some(id) = vocab.get_id(substr) {
                    best_len = len;
                    best_id = Some(id);
                    break;
                }
            }
        }

        if let Some(id) = best_id {
            tokens.push(id);
            pos += best_len;
        } else {
            // Unknown byte — skip with unknown token (ID 0)
            tokens.push(0);
            pos += 1;
        }
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple mock CPU tokenizer that splits on whitespace.
    struct WhitespaceCpuTokenizer {
        vocab: HashMap<String, TokenId>,
    }

    impl WhitespaceCpuTokenizer {
        fn new(vocab: &VocabLookupTable) -> Self {
            Self {
                vocab: vocab.token_to_id.clone(),
            }
        }
    }

    impl CpuTokenizer for WhitespaceCpuTokenizer {
        fn encode(&self, text: &str) -> Vec<TokenId> {
            text.split_whitespace()
                .map(|w| self.vocab.get(w).copied().unwrap_or(0))
                .collect()
        }
    }

    fn test_vocab() -> VocabLookupTable {
        VocabLookupTable::from_pairs(vec![
            ("hello".to_string(), 1),
            ("world".to_string(), 2),
            ("foo".to_string(), 3),
            ("bar".to_string(), 4),
            ("he".to_string(), 5),
            ("ll".to_string(), 6),
            ("o".to_string(), 7),
        ])
    }

    #[test]
    fn test_vocab_lookup() {
        let vocab = test_vocab();
        assert_eq!(vocab.get_id("hello"), Some(1));
        assert_eq!(vocab.get_id("world"), Some(2));
        assert_eq!(vocab.get_id("unknown"), None);
        assert_eq!(vocab.get_token(1), Some("hello"));
        assert_eq!(vocab.get_token(2), Some("world"));
        assert_eq!(vocab.vocab_size(), 7);
    }

    #[test]
    fn test_vocab_gpu_buffer() {
        let vocab = VocabLookupTable::from_pairs(vec![
            ("ab".to_string(), 1),
        ]);
        let buf = vocab.to_gpu_buffer();
        // Should contain: token_id (4 bytes) + len (4 bytes) + "ab" (2 bytes) = 10
        assert_eq!(buf.len(), 10);
        assert_eq!(&buf[0..4], &1u32.to_le_bytes());
        assert_eq!(&buf[4..8], &2u32.to_le_bytes());
        assert_eq!(&buf[8..10], b"ab");
    }

    #[test]
    fn test_single_request_uses_cpu() {
        let vocab = test_vocab();
        let cpu = WhitespaceCpuTokenizer::new(&vocab);
        let mut tokenizer = GpuBatchTokenizer::new(
            vocab,
            cpu,
            GpuBatchTokenizerConfig::default(),
        );
        tokenizer.upload_vocab();

        let result = tokenizer.tokenize_single("hello world");
        assert_eq!(result.token_ids, vec![1, 2]);
    }

    #[test]
    fn test_batch_below_threshold_uses_cpu() {
        let vocab = test_vocab();
        let cpu = WhitespaceCpuTokenizer::new(&vocab);
        let config = GpuBatchTokenizerConfig {
            min_batch_for_gpu: 3,
            ..Default::default()
        };
        let mut tokenizer =
            GpuBatchTokenizer::new(vocab, cpu, config);
        tokenizer.upload_vocab();

        // Batch of 2 < threshold of 3, should use CPU path
        let results =
            tokenizer.tokenize_batch(&["hello world", "foo bar"]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].token_ids, vec![1, 2]);
        assert_eq!(results[1].token_ids, vec![3, 4]);
    }

    #[test]
    fn test_batch_gpu_path() {
        let vocab = test_vocab();
        let cpu = WhitespaceCpuTokenizer::new(&vocab);
        let config = GpuBatchTokenizerConfig {
            min_batch_for_gpu: 2,
            ..Default::default()
        };
        let mut tokenizer =
            GpuBatchTokenizer::new(vocab, cpu, config);
        tokenizer.upload_vocab();

        let results = tokenizer
            .tokenize_batch(&["hello", "world", "foo"]);
        assert_eq!(results.len(), 3);
        // GPU greedy tokenizer matches full tokens
        assert_eq!(results[0].token_ids, vec![1]); // "hello" -> 1
        assert_eq!(results[1].token_ids, vec![2]); // "world" -> 2
        assert_eq!(results[2].token_ids, vec![3]); // "foo" -> 3
    }

    #[test]
    fn test_gpu_not_ready_falls_back() {
        let vocab = test_vocab();
        let cpu = WhitespaceCpuTokenizer::new(&vocab);
        let tokenizer = GpuBatchTokenizer::new(
            vocab,
            cpu,
            GpuBatchTokenizerConfig::default(),
        );

        assert!(!tokenizer.is_gpu_ready());
        // Should fall back to CPU even for batch >= threshold
        let results = tokenizer
            .tokenize_batch(&["hello world", "foo bar"]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_upload_vocab_returns_size() {
        let vocab = test_vocab();
        let cpu = WhitespaceCpuTokenizer::new(&vocab);
        let mut tokenizer = GpuBatchTokenizer::new(
            vocab,
            cpu,
            GpuBatchTokenizerConfig::default(),
        );

        let size = tokenizer.upload_vocab();
        assert!(size > 0);
        assert!(tokenizer.is_gpu_ready());
    }

    #[test]
    fn test_greedy_tokenize_longest_match() {
        let vocab = test_vocab();
        // "hello" should match as one token (ID 1), not "he" + "ll" + "o"
        let tokens = greedy_tokenize(&vocab, "hello");
        assert_eq!(tokens, vec![1]);
    }

    #[test]
    fn test_greedy_tokenize_unknown_chars() {
        let vocab = test_vocab();
        // "xyz" has no matches, should produce unknown tokens (ID 0)
        let tokens = greedy_tokenize(&vocab, "xyz");
        assert_eq!(tokens, vec![0, 0, 0]);
    }

    #[test]
    fn test_tokenized_prompt_json_roundtrip() {
        let prompt = TokenizedPrompt {
            text: "hello world".to_string(),
            token_ids: vec![1, 2],
            duration: Duration::from_millis(5),
        };

        let json = serde_json::to_string(&prompt).unwrap();
        let restored: TokenizedPrompt =
            serde_json::from_str(&json).unwrap();
        assert_eq!(restored.text, "hello world");
        assert_eq!(restored.token_ids, vec![1, 2]);
    }
}
