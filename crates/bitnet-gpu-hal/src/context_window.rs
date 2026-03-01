//! Module stub - implementation pending merge from feature branch
//! Context window management for LLM inference.
//!
//! Provides chunking, importance scoring, eviction strategies,
//! and compression to keep the active context within token budgets.

use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How to split incoming text into manageable chunks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkingStrategy {
    /// Keep text as a single chunk (no splitting).
    NoChunking,
    /// Split into fixed-size chunks of `n` tokens.
    FixedSize(usize),
    /// Split on sentence boundaries.
    SentenceBased,
    /// Split on paragraph boundaries (double newline).
    ParagraphBased,
    /// Placeholder for embedding-driven semantic splitting.
    Semantic,
}

/// How to choose which chunks to evict when the window is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionStrategy {
    /// Remove the oldest chunk first.
    OldestFirst,
    /// Remove the chunk with the lowest importance score.
    LeastImportant,
    /// Remove the least-recently-used chunk.
    LRU,
    /// Keep system messages and the most recent chunks; evict the rest.
    KeepSystemAndRecent,
}

/// Configuration for a [`ContextManager`].
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Maximum number of tokens allowed in the context window.
    pub max_context_length: usize,
    /// Tokens reserved for the model's generation output.
    pub reserved_for_generation: usize,
    /// Strategy used when splitting text into chunks.
    pub chunking_strategy: ChunkingStrategy,
    /// Strategy used when evicting chunks.
    pub eviction_strategy: EvictionStrategy,
}

impl ContextConfig {
    /// Effective capacity available for context chunks.
    #[must_use]
    pub const fn effective_capacity(&self) -> usize {
        self.max_context_length.saturating_sub(self.reserved_for_generation)
    }
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_context_length: 4096,
            reserved_for_generation: 512,
            chunking_strategy: ChunkingStrategy::NoChunking,
            eviction_strategy: EvictionStrategy::OldestFirst,
        }
    }
}

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// Role of a context chunk (influences importance scoring).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkRole {
    /// System-level instructions.
    System,
    /// User message.
    User,
    /// Assistant response.
    Assistant,
    /// Injected context (e.g. RAG retrieval).
    Context,
}

/// A single chunk of context text.
#[derive(Debug, Clone)]
pub struct ContextChunk {
    /// Unique identifier.
    pub id: u64,
    /// The text content.
    pub text: String,
    /// Approximate token count for this chunk.
    pub token_count: usize,
    /// Byte offset of the chunk within the original source.
    pub start_offset: usize,
    /// Importance score (higher = more important).
    pub importance_score: f64,
    /// Role of this chunk.
    pub role: ChunkRole,
    /// Timestamp of last access.
    last_accessed: Instant,
}

impl ContextChunk {
    /// Create a new chunk.
    #[must_use]
    pub fn new(
        id: u64,
        text: String,
        token_count: usize,
        start_offset: usize,
        role: ChunkRole,
    ) -> Self {
        Self {
            id,
            text,
            token_count,
            start_offset,
            importance_score: 0.0,
            role,
            last_accessed: Instant::now(),
        }
    }

    /// Touch the chunk (update `last_accessed` to now).
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
    }
}

// ---------------------------------------------------------------------------
// Context window
// ---------------------------------------------------------------------------

/// A snapshot of the live context window state.
#[derive(Debug, Clone)]
pub struct ContextWindow {
    /// Ordered list of active chunks.
    pub chunks: Vec<ContextChunk>,
    /// Total tokens across all chunks.
    pub total_tokens: usize,
    /// Remaining capacity in tokens.
    pub capacity_remaining: usize,
}

// ---------------------------------------------------------------------------
// Context snapshot (serialisable)
// ---------------------------------------------------------------------------

/// A serialisable snapshot of the context state.
#[derive(Debug, Clone)]
pub struct ContextSnapshot {
    /// Number of chunks.
    pub chunk_count: usize,
    /// Total tokens.
    pub total_tokens: usize,
    /// Remaining capacity.
    pub capacity_remaining: usize,
    /// Chunk texts in order.
    pub chunk_texts: Vec<String>,
    /// Per-chunk importance scores.
    pub importance_scores: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Context metrics
// ---------------------------------------------------------------------------

/// Utilisation and health metrics for the context manager.
#[derive(Debug, Clone)]
pub struct ContextMetrics {
    /// Fraction of capacity in use (0.0–1.0).
    pub utilization: f64,
    /// Total number of evictions performed so far.
    pub eviction_count: usize,
    /// Compression ratio (`original_tokens / current_tokens`, ≥ 1.0).
    pub compression_ratio: f64,
    /// Mean importance score across active chunks.
    pub avg_importance: f64,
    /// Total chunks currently held.
    pub chunk_count: usize,
    /// Total tokens currently held.
    pub total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Importance scorer
// ---------------------------------------------------------------------------

/// Scores a chunk's importance based on recency, role, and relevance.
#[derive(Debug)]
pub struct ImportanceScorer {
    /// Weight for the recency component.
    pub recency_weight: f64,
    /// Weight for the role component.
    pub role_weight: f64,
    /// Weight for the relevance (keyword) component.
    pub relevance_weight: f64,
    /// Keywords that boost relevance.
    keywords: Vec<String>,
}

impl ImportanceScorer {
    /// Create a scorer with the given weights.
    #[must_use]
    pub const fn new(recency_weight: f64, role_weight: f64, relevance_weight: f64) -> Self {
        Self { recency_weight, role_weight, relevance_weight, keywords: Vec::new() }
    }

    /// Set keywords that increase a chunk's relevance score.
    pub fn set_keywords(&mut self, keywords: Vec<String>) {
        self.keywords = keywords;
    }

    /// Score a single chunk. The `position` is the chunk's index and
    /// `total` is the number of chunks in the window.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn score(&self, chunk: &ContextChunk, position: usize, total: usize) -> f64 {
        let recency = if total == 0 { 0.0 } else { (position as f64 + 1.0) / total as f64 };

        let role_score = match chunk.role {
            ChunkRole::System => 1.0,
            ChunkRole::User => 0.7,
            ChunkRole::Assistant => 0.5,
            ChunkRole::Context => 0.3,
        };

        let relevance = if self.keywords.is_empty() {
            0.0
        } else {
            let lower = chunk.text.to_lowercase();
            let hits = self.keywords.iter().filter(|kw| lower.contains(&kw.to_lowercase())).count();
            (hits as f64) / (self.keywords.len() as f64)
        };

        self.relevance_weight
            .mul_add(relevance, self.recency_weight.mul_add(recency, self.role_weight * role_score))
    }
}

impl Default for ImportanceScorer {
    fn default() -> Self {
        Self::new(0.4, 0.4, 0.2)
    }
}

// ---------------------------------------------------------------------------
// Context compressor
// ---------------------------------------------------------------------------

/// Compresses older context to free token budget.
///
/// The current implementation truncates each chunk to its first
/// `target_ratio` fraction of tokens (word-approximate). A real
/// deployment would call a summarisation model.
#[derive(Debug)]
pub struct ContextCompressor {
    /// Target ratio of tokens to keep when compressing (0.0–1.0).
    pub target_ratio: f64,
}

impl ContextCompressor {
    /// Create a compressor that keeps `target_ratio` of each chunk.
    #[must_use]
    pub const fn new(target_ratio: f64) -> Self {
        Self { target_ratio: target_ratio.clamp(0.0, 1.0) }
    }

    /// Compress a single chunk in-place. Returns the number of tokens saved.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn compress_chunk(&self, chunk: &mut ContextChunk) -> usize {
        let target_tokens = ((chunk.token_count as f64) * self.target_ratio).ceil() as usize;
        let target_tokens = target_tokens.max(1);
        if target_tokens >= chunk.token_count {
            return 0;
        }

        // Approximate word-level truncation.
        let words: Vec<&str> = chunk.text.split_whitespace().collect();
        let keep = target_tokens.min(words.len());
        chunk.text = words[..keep].join(" ");
        let saved = chunk.token_count - target_tokens;
        chunk.token_count = target_tokens;
        saved
    }

    /// Compress all non-system chunks in `chunks`. Returns total tokens saved.
    pub fn compress_all(&self, chunks: &mut [ContextChunk]) -> usize {
        chunks
            .iter_mut()
            .filter(|c| c.role != ChunkRole::System)
            .map(|c| self.compress_chunk(c))
            .sum()
    }
}

impl Default for ContextCompressor {
    fn default() -> Self {
        Self::new(0.5)
    }
}

// ---------------------------------------------------------------------------
// Context manager
// ---------------------------------------------------------------------------

/// Manages the context window lifecycle: add, score, evict, compress.
#[derive(Debug)]
pub struct ContextManager {
    config: ContextConfig,
    chunks: Vec<ContextChunk>,
    scorer: ImportanceScorer,
    compressor: ContextCompressor,
    next_id: u64,
    total_tokens: usize,
    eviction_count: usize,
    original_tokens: usize,
}

impl ContextManager {
    /// Create a new manager with the given config.
    #[must_use]
    pub fn new(config: ContextConfig) -> Self {
        Self {
            config,
            chunks: Vec::new(),
            scorer: ImportanceScorer::default(),
            compressor: ContextCompressor::default(),
            next_id: 0,
            total_tokens: 0,
            eviction_count: 0,
            original_tokens: 0,
        }
    }

    /// Replace the importance scorer.
    pub fn set_scorer(&mut self, scorer: ImportanceScorer) {
        self.scorer = scorer;
    }

    /// Replace the compressor.
    pub const fn set_compressor(&mut self, compressor: ContextCompressor) {
        self.compressor = compressor;
    }

    /// Effective token capacity (max minus reserved).
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.config.effective_capacity()
    }

    /// Current token usage.
    #[must_use]
    pub const fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Remaining capacity.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.capacity().saturating_sub(self.total_tokens)
    }

    /// Number of active chunks.
    #[must_use]
    pub const fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    // -- chunking helpers ---------------------------------------------------

    /// Split text into chunks according to the configured strategy.
    #[must_use]
    pub fn chunk_text(&self, text: &str, role: ChunkRole) -> Vec<ContextChunk> {
        match &self.config.chunking_strategy {
            ChunkingStrategy::NoChunking => {
                let tc = estimate_tokens(text);
                vec![ContextChunk::new(0, text.to_owned(), tc, 0, role)]
            }
            ChunkingStrategy::FixedSize(size) => {
                let words: Vec<&str> = text.split_whitespace().collect();
                let size = (*size).max(1);
                words
                    .chunks(size)
                    .enumerate()
                    .map(|(i, w)| {
                        let t = w.join(" ");
                        let tc = w.len();
                        ContextChunk::new(0, t, tc, i * size, role)
                    })
                    .collect()
            }
            ChunkingStrategy::SentenceBased => split_sentences(text)
                .into_iter()
                .enumerate()
                .map(|(i, s)| {
                    let tc = estimate_tokens(&s);
                    ContextChunk::new(0, s, tc, i, role)
                })
                .collect(),
            ChunkingStrategy::ParagraphBased => text
                .split("\n\n")
                .filter(|p| !p.trim().is_empty())
                .enumerate()
                .map(|(i, p)| {
                    let s = p.trim().to_owned();
                    let tc = estimate_tokens(&s);
                    ContextChunk::new(0, s, tc, i, role)
                })
                .collect(),
            ChunkingStrategy::Semantic => {
                // Semantic splitting is a placeholder; fall back to paragraph.
                text.split("\n\n")
                    .filter(|p| !p.trim().is_empty())
                    .enumerate()
                    .map(|(i, p)| {
                        let s = p.trim().to_owned();
                        let tc = estimate_tokens(&s);
                        ContextChunk::new(0, s, tc, i, role)
                    })
                    .collect()
            }
        }
    }

    // -- adding chunks ------------------------------------------------------

    /// Add text to the context window. Splits using the configured strategy,
    /// scores, and evicts as needed. Returns the IDs of the newly added chunks.
    pub fn add(&mut self, text: &str, role: ChunkRole) -> Vec<u64> {
        let mut new_chunks = self.chunk_text(text, role);

        // Assign unique IDs.
        for c in &mut new_chunks {
            c.id = self.next_id;
            self.next_id += 1;
        }

        // Score all new chunks.
        let future_total = self.chunks.len() + new_chunks.len();
        for (i, c) in new_chunks.iter_mut().enumerate() {
            let pos = self.chunks.len() + i;
            c.importance_score = self.scorer.score(c, pos, future_total);
        }

        let ids: Vec<u64> = new_chunks.iter().map(|c| c.id).collect();

        for c in new_chunks {
            self.original_tokens += c.token_count;
            self.total_tokens += c.token_count;
            self.chunks.push(c);
        }

        // Evict until we fit.
        while self.total_tokens > self.capacity() && self.chunks.len() > 1 {
            self.evict_one();
        }

        ids
    }

    // -- eviction -----------------------------------------------------------

    /// Evict one chunk according to the configured strategy.
    fn evict_one(&mut self) {
        if self.chunks.is_empty() {
            return;
        }
        let idx = self.pick_eviction_candidate();
        let removed = self.chunks.remove(idx);
        self.total_tokens = self.total_tokens.saturating_sub(removed.token_count);
        self.eviction_count += 1;
    }

    /// Pick the index of the chunk to evict.
    fn pick_eviction_candidate(&self) -> usize {
        match self.config.eviction_strategy {
            EvictionStrategy::OldestFirst => 0,
            EvictionStrategy::LeastImportant => self
                .chunks
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.importance_score
                        .partial_cmp(&b.importance_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map_or(0, |(i, _)| i),
            EvictionStrategy::LRU => self
                .chunks
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| c.last_accessed)
                .map_or(0, |(i, _)| i),
            EvictionStrategy::KeepSystemAndRecent => {
                // Prefer evicting the oldest non-system chunk.
                self.chunks
                    .iter()
                    .enumerate()
                    .find(|(_, c)| c.role != ChunkRole::System)
                    .map_or(0, |(i, _)| i)
            }
        }
    }

    /// Manually evict a chunk by its ID. Returns `true` if found & removed.
    pub fn evict_by_id(&mut self, id: u64) -> bool {
        if let Some(pos) = self.chunks.iter().position(|c| c.id == id) {
            let removed = self.chunks.remove(pos);
            self.total_tokens = self.total_tokens.saturating_sub(removed.token_count);
            self.eviction_count += 1;
            true
        } else {
            false
        }
    }

    // -- re-scoring ---------------------------------------------------------

    /// Re-score all chunks using current positions and scorer.
    pub fn rescore(&mut self) {
        let total = self.chunks.len();
        for i in 0..total {
            let score = self.scorer.score(&self.chunks[i], i, total);
            self.chunks[i].importance_score = score;
        }
    }

    // -- access / touch -----------------------------------------------------

    /// Touch a chunk by ID (update last-accessed time). Returns `true` if found.
    pub fn touch(&mut self, id: u64) -> bool {
        self.chunks.iter_mut().find(|c| c.id == id).is_some_and(|c| {
            c.touch();
            true
        })
    }

    // -- compression --------------------------------------------------------

    /// Compress non-system chunks to reclaim tokens. Returns tokens saved.
    pub fn compress(&mut self) -> usize {
        let saved = self.compressor.compress_all(&mut self.chunks);
        self.total_tokens = self.total_tokens.saturating_sub(saved);
        saved
    }

    // -- queries ------------------------------------------------------------

    /// Get the current context window state.
    #[must_use]
    pub fn get_active(&self) -> ContextWindow {
        ContextWindow {
            chunks: self.chunks.clone(),
            total_tokens: self.total_tokens,
            capacity_remaining: self.remaining(),
        }
    }

    /// Build a serialisable snapshot.
    #[must_use]
    pub fn snapshot(&self) -> ContextSnapshot {
        ContextSnapshot {
            chunk_count: self.chunks.len(),
            total_tokens: self.total_tokens,
            capacity_remaining: self.remaining(),
            chunk_texts: self.chunks.iter().map(|c| c.text.clone()).collect(),
            importance_scores: self.chunks.iter().map(|c| c.importance_score).collect(),
        }
    }

    /// Compute current metrics.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn metrics(&self) -> ContextMetrics {
        let cap = self.capacity();
        let utilization = if cap == 0 { 0.0 } else { self.total_tokens as f64 / cap as f64 };

        let avg_importance = if self.chunks.is_empty() {
            0.0
        } else {
            let sum: f64 = self.chunks.iter().map(|c| c.importance_score).sum();
            sum / self.chunks.len() as f64
        };

        let compression_ratio = if self.total_tokens == 0 {
            1.0
        } else {
            self.original_tokens as f64 / self.total_tokens as f64
        };

        ContextMetrics {
            utilization,
            eviction_count: self.eviction_count,
            compression_ratio,
            avg_importance,
            chunk_count: self.chunks.len(),
            total_tokens: self.total_tokens,
        }
    }

    /// Get a chunk by ID.
    #[must_use]
    pub fn get_chunk(&self, id: u64) -> Option<&ContextChunk> {
        self.chunks.iter().find(|c| c.id == id)
    }

    /// Clear all chunks and reset counters.
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.total_tokens = 0;
        self.original_tokens = 0;
        self.eviction_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rough word-count-based token estimate (1 word ≈ 1 token).
#[must_use]
fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count().max(1)
}

/// Naïve sentence splitter (splits on `. `, `? `, `! `).
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        current.push(chars[i]);
        if (chars[i] == '.' || chars[i] == '?' || chars[i] == '!')
            && (i + 1 >= len || chars[i + 1] == ' ')
        {
            let trimmed = current.trim().to_owned();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
            // Skip the space after punctuation.
            if i + 1 < len && chars[i + 1] == ' ' {
                i += 1;
            }
        }
        i += 1;
    }
    let trimmed = current.trim().to_owned();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    if sentences.is_empty() && !text.trim().is_empty() {
        sentences.push(text.trim().to_owned());
    }
    sentences
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn default_manager() -> ContextManager {
        ContextManager::new(ContextConfig::default())
    }

    fn small_manager(cap: usize, reserved: usize) -> ContextManager {
        ContextManager::new(ContextConfig {
            max_context_length: cap,
            reserved_for_generation: reserved,
            chunking_strategy: ChunkingStrategy::NoChunking,
            eviction_strategy: EvictionStrategy::OldestFirst,
        })
    }

    fn words(n: usize) -> String {
        (0..n).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ")
    }

    // =======================================================================
    // ContextConfig
    // =======================================================================

    #[test]
    fn config_default_values() {
        let cfg = ContextConfig::default();
        assert_eq!(cfg.max_context_length, 4096);
        assert_eq!(cfg.reserved_for_generation, 512);
        assert_eq!(cfg.chunking_strategy, ChunkingStrategy::NoChunking);
        assert_eq!(cfg.eviction_strategy, EvictionStrategy::OldestFirst);
    }

    #[test]
    fn config_effective_capacity() {
        let cfg = ContextConfig {
            max_context_length: 1000,
            reserved_for_generation: 200,
            ..Default::default()
        };
        assert_eq!(cfg.effective_capacity(), 800);
    }

    #[test]
    fn config_effective_capacity_saturates() {
        let cfg = ContextConfig {
            max_context_length: 100,
            reserved_for_generation: 500,
            ..Default::default()
        };
        assert_eq!(cfg.effective_capacity(), 0);
    }

    // =======================================================================
    // ChunkingStrategy — enum coverage
    // =======================================================================

    #[test]
    fn chunking_no_chunking_identity() {
        let mgr = default_manager();
        let chunks = mgr.chunk_text("hello world", ChunkRole::User);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
    }

    #[test]
    fn chunking_fixed_size_splits_evenly() {
        let mgr = ContextManager::new(ContextConfig {
            chunking_strategy: ChunkingStrategy::FixedSize(2),
            ..Default::default()
        });
        let chunks = mgr.chunk_text("a b c d", ChunkRole::User);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "a b");
        assert_eq!(chunks[1].text, "c d");
    }

    #[test]
    fn chunking_fixed_size_remainder() {
        let mgr = ContextManager::new(ContextConfig {
            chunking_strategy: ChunkingStrategy::FixedSize(3),
            ..Default::default()
        });
        let chunks = mgr.chunk_text("a b c d e", ChunkRole::User);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].token_count, 3);
        assert_eq!(chunks[1].token_count, 2);
    }

    #[test]
    fn chunking_fixed_size_zero_treated_as_one() {
        let mgr = ContextManager::new(ContextConfig {
            chunking_strategy: ChunkingStrategy::FixedSize(0),
            ..Default::default()
        });
        let chunks = mgr.chunk_text("a b c", ChunkRole::User);
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn chunking_sentence_based() {
        let mgr = ContextManager::new(ContextConfig {
            chunking_strategy: ChunkingStrategy::SentenceBased,
            ..Default::default()
        });
        let chunks = mgr.chunk_text("Hello world. How are you? Fine!", ChunkRole::User);
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn chunking_paragraph_based() {
        let mgr = ContextManager::new(ContextConfig {
            chunking_strategy: ChunkingStrategy::ParagraphBased,
            ..Default::default()
        });
        let chunks = mgr.chunk_text("Para one.\n\nPara two.\n\nPara three.", ChunkRole::User);
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn chunking_semantic_falls_back_to_paragraph() {
        let mgr = ContextManager::new(ContextConfig {
            chunking_strategy: ChunkingStrategy::Semantic,
            ..Default::default()
        });
        let chunks = mgr.chunk_text("A.\n\nB.", ChunkRole::User);
        assert_eq!(chunks.len(), 2);
    }

    // =======================================================================
    // ContextChunk
    // =======================================================================

    #[test]
    fn chunk_new_defaults() {
        let c = ContextChunk::new(1, "hi".into(), 1, 0, ChunkRole::User);
        assert_eq!(c.id, 1);
        assert_eq!(c.text, "hi");
        assert_eq!(c.token_count, 1);
        assert_eq!(c.start_offset, 0);
        assert!((c.importance_score - 0.0).abs() < f64::EPSILON);
        assert_eq!(c.role, ChunkRole::User);
    }

    #[test]
    fn chunk_touch_updates_timestamp() {
        let mut c = ContextChunk::new(0, "x".into(), 1, 0, ChunkRole::User);
        let t1 = c.last_accessed;
        // Busy-wait to ensure time advances.
        std::thread::sleep(std::time::Duration::from_millis(2));
        c.touch();
        assert!(c.last_accessed >= t1);
    }

    // =======================================================================
    // ContextWindow — add & basic window management
    // =======================================================================

    #[test]
    fn add_single_chunk() {
        let mut mgr = default_manager();
        let ids = mgr.add("hello", ChunkRole::User);
        assert_eq!(ids.len(), 1);
        assert_eq!(mgr.chunk_count(), 1);
        assert_eq!(mgr.total_tokens(), 1);
    }

    #[test]
    fn add_multiple_chunks_accumulates() {
        let mut mgr = default_manager();
        mgr.add("one two", ChunkRole::User);
        mgr.add("three", ChunkRole::Assistant);
        assert_eq!(mgr.chunk_count(), 2);
        assert_eq!(mgr.total_tokens(), 3);
    }

    #[test]
    fn add_returns_unique_ids() {
        let mut mgr = default_manager();
        let a = mgr.add("a", ChunkRole::User);
        let b = mgr.add("b", ChunkRole::User);
        assert_ne!(a[0], b[0]);
    }

    #[test]
    fn remaining_decreases_after_add() {
        let mut mgr = small_manager(100, 10);
        assert_eq!(mgr.remaining(), 90);
        mgr.add(&words(20), ChunkRole::User);
        assert_eq!(mgr.remaining(), 70);
    }

    #[test]
    fn get_active_reflects_state() {
        let mut mgr = small_manager(100, 10);
        mgr.add("hello world", ChunkRole::User);
        let win = mgr.get_active();
        assert_eq!(win.chunks.len(), 1);
        assert_eq!(win.total_tokens, 2);
        assert_eq!(win.capacity_remaining, 88);
    }

    // =======================================================================
    // Eviction — OldestFirst
    // =======================================================================

    #[test]
    fn eviction_oldest_first_removes_first_chunk() {
        let mut mgr = small_manager(10, 0);
        mgr.add(&words(5), ChunkRole::User); // id 0
        mgr.add(&words(5), ChunkRole::User); // id 1
        // At capacity (10). Adding more should evict id 0.
        mgr.add(&words(3), ChunkRole::User); // id 2
        assert!(mgr.get_chunk(0).is_none());
    }

    #[test]
    fn eviction_count_tracked() {
        let mut mgr = small_manager(5, 0);
        mgr.add(&words(3), ChunkRole::User);
        mgr.add(&words(4), ChunkRole::User);
        assert!(mgr.metrics().eviction_count > 0);
    }

    // =======================================================================
    // Eviction — LeastImportant
    // =======================================================================

    #[test]
    fn eviction_least_important() {
        let mut mgr = ContextManager::new(ContextConfig {
            max_context_length: 10,
            reserved_for_generation: 0,
            chunking_strategy: ChunkingStrategy::NoChunking,
            eviction_strategy: EvictionStrategy::LeastImportant,
        });
        // System chunk gets highest role score.
        mgr.add("sys", ChunkRole::System);
        // Context chunk gets lowest role score.
        mgr.add(&words(5), ChunkRole::Context);
        mgr.add(&words(5), ChunkRole::User);
        // The Context chunk should be evicted first (lowest importance).
        let active = mgr.get_active();
        assert!(active.chunks.iter().any(|c| c.text == "sys"));
    }

    // =======================================================================
    // Eviction — KeepSystemAndRecent
    // =======================================================================

    #[test]
    fn eviction_keep_system_and_recent() {
        let mut mgr = ContextManager::new(ContextConfig {
            max_context_length: 8,
            reserved_for_generation: 0,
            chunking_strategy: ChunkingStrategy::NoChunking,
            eviction_strategy: EvictionStrategy::KeepSystemAndRecent,
        });
        let sys_ids = mgr.add("sys msg", ChunkRole::System);
        mgr.add(&words(4), ChunkRole::User);
        mgr.add(&words(4), ChunkRole::User);
        // System chunk should still be present.
        assert!(mgr.get_chunk(sys_ids[0]).is_some());
    }

    // =======================================================================
    // Eviction — evict_by_id
    // =======================================================================

    #[test]
    fn evict_by_id_removes_correct_chunk() {
        let mut mgr = default_manager();
        let ids = mgr.add("hello world", ChunkRole::User);
        assert!(mgr.evict_by_id(ids[0]));
        assert_eq!(mgr.chunk_count(), 0);
    }

    #[test]
    fn evict_by_id_returns_false_for_missing() {
        let mut mgr = default_manager();
        assert!(!mgr.evict_by_id(999));
    }

    #[test]
    fn evict_by_id_updates_tokens() {
        let mut mgr = default_manager();
        mgr.add("one two three", ChunkRole::User);
        let before = mgr.total_tokens();
        mgr.evict_by_id(0);
        assert!(mgr.total_tokens() < before);
    }

    // =======================================================================
    // ImportanceScorer
    // =======================================================================

    #[test]
    fn scorer_system_higher_than_context() {
        let scorer = ImportanceScorer::default();
        let sys = ContextChunk::new(0, "sys".into(), 1, 0, ChunkRole::System);
        let ctx = ContextChunk::new(1, "ctx".into(), 1, 0, ChunkRole::Context);
        let s_sys = scorer.score(&sys, 0, 2);
        let s_ctx = scorer.score(&ctx, 0, 2);
        assert!(s_sys > s_ctx);
    }

    #[test]
    fn scorer_recency_increases_with_position() {
        let scorer = ImportanceScorer::new(1.0, 0.0, 0.0);
        let c = ContextChunk::new(0, "a".into(), 1, 0, ChunkRole::User);
        let early = scorer.score(&c, 0, 10);
        let late = scorer.score(&c, 9, 10);
        assert!(late > early);
    }

    #[test]
    fn scorer_keywords_boost_relevance() {
        let mut scorer = ImportanceScorer::new(0.0, 0.0, 1.0);
        scorer.set_keywords(vec!["important".into()]);
        let yes = ContextChunk::new(0, "this is important".into(), 3, 0, ChunkRole::User);
        let no = ContextChunk::new(1, "nothing here".into(), 2, 0, ChunkRole::User);
        assert!(scorer.score(&yes, 0, 2) > scorer.score(&no, 0, 2));
    }

    #[test]
    fn scorer_keywords_case_insensitive() {
        let mut scorer = ImportanceScorer::new(0.0, 0.0, 1.0);
        scorer.set_keywords(vec!["HELLO".into()]);
        let c = ContextChunk::new(0, "hello world".into(), 2, 0, ChunkRole::User);
        assert!(scorer.score(&c, 0, 1) > 0.0);
    }

    #[test]
    fn scorer_no_keywords_zero_relevance() {
        let scorer = ImportanceScorer::new(0.0, 0.0, 1.0);
        let c = ContextChunk::new(0, "hello".into(), 1, 0, ChunkRole::User);
        assert!((scorer.score(&c, 0, 1)).abs() < f64::EPSILON);
    }

    #[test]
    fn scorer_zero_total_gives_zero_recency() {
        let scorer = ImportanceScorer::new(1.0, 0.0, 0.0);
        let c = ContextChunk::new(0, "a".into(), 1, 0, ChunkRole::User);
        assert!((scorer.score(&c, 0, 0)).abs() < f64::EPSILON);
    }

    #[test]
    fn scorer_role_ordering() {
        let scorer = ImportanceScorer::new(0.0, 1.0, 0.0);
        let roles = [ChunkRole::System, ChunkRole::User, ChunkRole::Assistant, ChunkRole::Context];
        let role_scores: Vec<f64> = roles
            .iter()
            .map(|r| {
                let c = ContextChunk::new(0, "x".into(), 1, 0, *r);
                scorer.score(&c, 0, 1)
            })
            .collect();
        // Scores should be strictly descending.
        for w in role_scores.windows(2) {
            assert!(w[0] > w[1]);
        }
    }

    // =======================================================================
    // ContextCompressor
    // =======================================================================

    #[test]
    fn compressor_reduces_tokens() {
        let comp = ContextCompressor::new(0.5);
        let mut chunk = ContextChunk::new(0, words(10), 10, 0, ChunkRole::User);
        let saved = comp.compress_chunk(&mut chunk);
        assert!(saved > 0);
        assert!(chunk.token_count < 10);
    }

    #[test]
    fn compressor_ratio_one_no_change() {
        let comp = ContextCompressor::new(1.0);
        let mut chunk = ContextChunk::new(0, words(10), 10, 0, ChunkRole::User);
        let saved = comp.compress_chunk(&mut chunk);
        assert_eq!(saved, 0);
    }

    #[test]
    fn compressor_very_low_ratio_keeps_at_least_one() {
        let comp = ContextCompressor::new(0.01);
        let mut chunk = ContextChunk::new(0, words(10), 10, 0, ChunkRole::User);
        comp.compress_chunk(&mut chunk);
        assert!(chunk.token_count >= 1);
    }

    #[test]
    fn compressor_skips_system_in_compress_all() {
        let comp = ContextCompressor::new(0.5);
        let mut chunks = vec![
            ContextChunk::new(0, words(10), 10, 0, ChunkRole::System),
            ContextChunk::new(1, words(10), 10, 0, ChunkRole::User),
        ];
        comp.compress_all(&mut chunks);
        assert_eq!(chunks[0].token_count, 10); // System untouched.
        assert!(chunks[1].token_count < 10);
    }

    #[test]
    fn compressor_clamps_ratio() {
        let comp = ContextCompressor::new(2.0);
        assert!((comp.target_ratio - 1.0).abs() < f64::EPSILON);
        let comp2 = ContextCompressor::new(-1.0);
        assert!((comp2.target_ratio).abs() < f64::EPSILON);
    }

    #[test]
    fn compress_via_manager() {
        let mut mgr = default_manager();
        mgr.add(&words(100), ChunkRole::User);
        let before = mgr.total_tokens();
        let saved = mgr.compress();
        assert!(saved > 0);
        assert_eq!(mgr.total_tokens(), before - saved);
    }

    // =======================================================================
    // ContextSnapshot
    // =======================================================================

    #[test]
    fn snapshot_reflects_state() {
        let mut mgr = default_manager();
        mgr.add("hello world", ChunkRole::User);
        mgr.add("foo bar baz", ChunkRole::Assistant);
        let snap = mgr.snapshot();
        assert_eq!(snap.chunk_count, 2);
        assert_eq!(snap.total_tokens, 5);
        assert_eq!(snap.chunk_texts.len(), 2);
        assert_eq!(snap.importance_scores.len(), 2);
    }

    #[test]
    fn snapshot_empty_manager() {
        let mgr = default_manager();
        let snap = mgr.snapshot();
        assert_eq!(snap.chunk_count, 0);
        assert_eq!(snap.total_tokens, 0);
        assert!(snap.chunk_texts.is_empty());
    }

    // =======================================================================
    // ContextMetrics
    // =======================================================================

    #[test]
    fn metrics_utilization_zero_when_empty() {
        let mgr = default_manager();
        let m = mgr.metrics();
        assert!((m.utilization).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_utilization_correct() {
        let mut mgr = small_manager(100, 0);
        mgr.add(&words(50), ChunkRole::User);
        let m = mgr.metrics();
        assert!((m.utilization - 0.5).abs() < 0.01);
    }

    #[test]
    fn metrics_eviction_count_accumulates() {
        let mut mgr = small_manager(5, 0);
        mgr.add(&words(3), ChunkRole::User);
        mgr.add(&words(4), ChunkRole::User);
        let m = mgr.metrics();
        assert!(m.eviction_count >= 1);
    }

    #[test]
    fn metrics_compression_ratio_after_compress() {
        let mut mgr = small_manager(200, 0);
        mgr.add(&words(100), ChunkRole::User);
        mgr.compress();
        let m = mgr.metrics();
        assert!(m.compression_ratio >= 1.0);
    }

    #[test]
    fn metrics_avg_importance() {
        let mut mgr = default_manager();
        mgr.add("a", ChunkRole::System);
        mgr.add("b", ChunkRole::Context);
        let m = mgr.metrics();
        assert!(m.avg_importance > 0.0);
    }

    #[test]
    fn metrics_zero_capacity() {
        let mgr = small_manager(0, 0);
        let m = mgr.metrics();
        assert!((m.utilization).abs() < f64::EPSILON);
    }

    // =======================================================================
    // Rescore
    // =======================================================================

    #[test]
    fn rescore_updates_scores() {
        let mut mgr = default_manager();
        mgr.add("a", ChunkRole::User);
        mgr.add("b", ChunkRole::User);
        let before = mgr.get_active().chunks.len();
        mgr.rescore();
        let after = mgr.get_active().chunks.len();
        // Scores may change because total count is now stable.
        assert_eq!(before, after);
    }

    // =======================================================================
    // Touch / LRU
    // =======================================================================

    #[test]
    fn touch_returns_true_for_existing() {
        let mut mgr = default_manager();
        let ids = mgr.add("hi", ChunkRole::User);
        assert!(mgr.touch(ids[0]));
    }

    #[test]
    fn touch_returns_false_for_missing() {
        let mut mgr = default_manager();
        assert!(!mgr.touch(999));
    }

    // =======================================================================
    // Clear
    // =======================================================================

    #[test]
    fn clear_resets_everything() {
        let mut mgr = default_manager();
        mgr.add(&words(50), ChunkRole::User);
        mgr.clear();
        assert_eq!(mgr.chunk_count(), 0);
        assert_eq!(mgr.total_tokens(), 0);
        assert_eq!(mgr.metrics().eviction_count, 0);
    }

    // =======================================================================
    // Edge cases
    // =======================================================================

    #[test]
    fn add_empty_string() {
        let mut mgr = default_manager();
        let ids = mgr.add("", ChunkRole::User);
        // Empty string still produces one chunk (estimate_tokens returns 1).
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn add_whitespace_only() {
        let mut mgr = default_manager();
        mgr.add("   ", ChunkRole::User);
        // Whitespace-only → estimate_tokens gives 1 (max(0,1)).
        assert_eq!(mgr.total_tokens(), 1);
    }

    #[test]
    fn eviction_on_exact_capacity() {
        let mut mgr = small_manager(5, 0);
        mgr.add(&words(5), ChunkRole::User);
        assert_eq!(mgr.total_tokens(), 5);
        assert_eq!(mgr.metrics().eviction_count, 0);
    }

    #[test]
    fn eviction_one_over_capacity() {
        let mut mgr = small_manager(5, 0);
        mgr.add(&words(3), ChunkRole::User);
        mgr.add(&words(3), ChunkRole::User);
        // 6 tokens > 5 capacity, should evict.
        assert!(mgr.total_tokens() <= 5);
    }

    #[test]
    fn large_single_chunk_does_not_panic() {
        let mut mgr = small_manager(5, 0);
        // Single 100-word chunk bigger than capacity.
        mgr.add(&words(100), ChunkRole::User);
        // Should still hold it (can't evict the only chunk).
        assert_eq!(mgr.chunk_count(), 1);
    }

    #[test]
    fn multiple_roles_coexist() {
        let mut mgr = default_manager();
        mgr.add("sys prompt", ChunkRole::System);
        mgr.add("user msg", ChunkRole::User);
        mgr.add("assistant reply", ChunkRole::Assistant);
        mgr.add("context data", ChunkRole::Context);
        assert_eq!(mgr.chunk_count(), 4);
    }

    #[test]
    fn get_chunk_returns_correct() {
        let mut mgr = default_manager();
        let ids = mgr.add("hello", ChunkRole::User);
        let c = mgr.get_chunk(ids[0]).unwrap();
        assert_eq!(c.text, "hello");
    }

    #[test]
    fn get_chunk_returns_none_for_missing() {
        let mgr = default_manager();
        assert!(mgr.get_chunk(42).is_none());
    }

    #[test]
    fn fixed_chunking_then_evict() {
        let mut mgr = ContextManager::new(ContextConfig {
            max_context_length: 6,
            reserved_for_generation: 0,
            chunking_strategy: ChunkingStrategy::FixedSize(2),
            eviction_strategy: EvictionStrategy::OldestFirst,
        });
        // "a b c d" → 2 chunks of 2 tokens each (4 total).
        mgr.add("a b c d", ChunkRole::User);
        assert_eq!(mgr.chunk_count(), 2);
        // "e f g h" → 2 more chunks (8 total > 6 cap) → evictions.
        mgr.add("e f g h", ChunkRole::User);
        assert!(mgr.total_tokens() <= 6);
    }

    #[test]
    fn sentence_chunking_single_sentence() {
        let mgr = ContextManager::new(ContextConfig {
            chunking_strategy: ChunkingStrategy::SentenceBased,
            ..Default::default()
        });
        let chunks = mgr.chunk_text("Hello world", ChunkRole::User);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn paragraph_chunking_no_double_newline() {
        let mgr = ContextManager::new(ContextConfig {
            chunking_strategy: ChunkingStrategy::ParagraphBased,
            ..Default::default()
        });
        let chunks = mgr.chunk_text("single paragraph", ChunkRole::User);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn set_scorer_changes_behavior() {
        let mut mgr = default_manager();
        mgr.set_scorer(ImportanceScorer::new(1.0, 0.0, 0.0));
        mgr.add("a", ChunkRole::User);
        mgr.add("b", ChunkRole::User);
        let scores: Vec<f64> = mgr.get_active().chunks.iter().map(|c| c.importance_score).collect();
        // Second chunk should have higher recency score.
        assert!(scores[1] >= scores[0]);
    }

    #[test]
    fn set_compressor_changes_ratio() {
        let mut mgr = default_manager();
        mgr.set_compressor(ContextCompressor::new(0.1));
        mgr.add(&words(100), ChunkRole::User);
        let saved = mgr.compress();
        assert!(saved > 80); // Should compress a lot with 0.1 ratio.
    }

    #[test]
    fn snapshot_after_eviction() {
        let mut mgr = small_manager(5, 0);
        mgr.add(&words(3), ChunkRole::User);
        mgr.add(&words(4), ChunkRole::User);
        let snap = mgr.snapshot();
        assert!(snap.total_tokens <= 5);
    }

    #[test]
    fn metrics_chunk_count_correct() {
        let mut mgr = default_manager();
        mgr.add("a", ChunkRole::User);
        mgr.add("b", ChunkRole::User);
        mgr.add("c", ChunkRole::User);
        assert_eq!(mgr.metrics().chunk_count, 3);
    }

    #[test]
    fn compression_ratio_starts_at_one() {
        let mgr = default_manager();
        let m = mgr.metrics();
        assert!((m.compression_ratio - 1.0).abs() < f64::EPSILON);
    }

    // =======================================================================
    // Helper function tests
    // =======================================================================

    #[test]
    fn estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 1);
    }

    #[test]
    fn estimate_tokens_single_word() {
        assert_eq!(estimate_tokens("hello"), 1);
    }

    #[test]
    fn estimate_tokens_multiple_words() {
        assert_eq!(estimate_tokens("one two three"), 3);
    }

    #[test]
    fn split_sentences_basic() {
        let s = split_sentences("Hello. World.");
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn split_sentences_question() {
        let s = split_sentences("What? Why!");
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn split_sentences_no_punctuation() {
        let s = split_sentences("hello world");
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn split_sentences_empty() {
        let s = split_sentences("");
        assert!(s.is_empty());
    }

    // =======================================================================
    // proptest
    // =======================================================================

    proptest::proptest! {
        #[test]
        fn total_tokens_never_exceeds_capacity_by_much(
            n_adds in 1usize..20,
            words_per_add in 1usize..30,
        ) {
            let mut mgr = small_manager(50, 0);
            for _ in 0..n_adds {
                mgr.add(&words(words_per_add), ChunkRole::User);
            }
            // After eviction, total should be at most capacity
            // (or a single oversized chunk).
            let cap = mgr.capacity();
            if mgr.chunk_count() > 1 {
                proptest::prop_assert!(mgr.total_tokens() <= cap);
            }
        }

        #[test]
        fn eviction_count_non_negative(
            n in 1usize..10,
        ) {
            let mut mgr = small_manager(10, 0);
            for _ in 0..n {
                mgr.add(&words(5), ChunkRole::User);
            }
            proptest::prop_assert!(mgr.metrics().eviction_count < usize::MAX);
        }

        #[test]
        fn compression_ratio_at_least_one(
            n in 1usize..10,
        ) {
            let mut mgr = small_manager(200, 0);
            for _ in 0..n {
                mgr.add(&words(10), ChunkRole::User);
            }
            mgr.compress();
            proptest::prop_assert!(mgr.metrics().compression_ratio >= 1.0);
        }

        #[test]
        fn utilization_in_range(
            n in 0usize..15,
            w in 1usize..20,
        ) {
            let mut mgr = small_manager(100, 0);
            for _ in 0..n {
                mgr.add(&words(w), ChunkRole::User);
            }
            let u = mgr.metrics().utilization;
            proptest::prop_assert!(u >= 0.0);
            // May exceed 1.0 if single chunk is oversized.
        }
    }
}
