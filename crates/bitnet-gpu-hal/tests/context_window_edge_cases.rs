//! Edge-case tests for context_window module.
//!
//! Covers: ChunkingStrategy, EvictionStrategy, ChunkRole,
//! ContextConfig, ContextChunk, ContextWindow, ContextSnapshot,
//! ContextMetrics, ImportanceScorer, ContextCompressor, ContextManager.

use bitnet_gpu_hal::context_window::*;

// ── ChunkingStrategy ────────────────────────────────────────────

#[test]
fn chunking_strategy_all_variants() {
    let variants = vec![
        ChunkingStrategy::NoChunking,
        ChunkingStrategy::FixedSize(512),
        ChunkingStrategy::SentenceBased,
        ChunkingStrategy::ParagraphBased,
        ChunkingStrategy::Semantic,
    ];
    assert_eq!(variants.len(), 5);
}

#[test]
fn chunking_strategy_clone_eq() {
    let a = ChunkingStrategy::FixedSize(256);
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn chunking_strategy_debug() {
    let dbg = format!("{:?}", ChunkingStrategy::SentenceBased);
    assert!(dbg.contains("SentenceBased"));
}

// ── EvictionStrategy ────────────────────────────────────────────

#[test]
fn eviction_strategy_all_variants() {
    let variants = vec![
        EvictionStrategy::OldestFirst,
        EvictionStrategy::LeastImportant,
        EvictionStrategy::LRU,
        EvictionStrategy::KeepSystemAndRecent,
    ];
    assert_eq!(variants.len(), 4);
}

#[test]
fn eviction_strategy_clone_copy_eq() {
    let a = EvictionStrategy::LRU;
    let b = a;
    assert_eq!(a, b);
}

// ── ChunkRole ───────────────────────────────────────────────────

#[test]
fn chunk_role_all_variants() {
    let variants =
        vec![ChunkRole::System, ChunkRole::User, ChunkRole::Assistant, ChunkRole::Context];
    assert_eq!(variants.len(), 4);
}

#[test]
fn chunk_role_clone_copy_eq() {
    let a = ChunkRole::System;
    let b = a;
    assert_eq!(a, b);
}

// ── ContextConfig ───────────────────────────────────────────────

#[test]
fn context_config_default() {
    let c = ContextConfig::default();
    assert!(c.max_context_length > 0);
    assert!(c.reserved_for_generation > 0);
    assert!(c.effective_capacity() > 0);
    assert!(c.effective_capacity() < c.max_context_length);
}

#[test]
fn context_config_custom() {
    let c = ContextConfig {
        max_context_length: 4096,
        reserved_for_generation: 512,
        chunking_strategy: ChunkingStrategy::FixedSize(128),
        eviction_strategy: EvictionStrategy::LRU,
    };
    assert_eq!(c.effective_capacity(), 3584);
}

#[test]
fn context_config_clone() {
    let c = ContextConfig::default();
    let c2 = c.clone();
    assert_eq!(c2.max_context_length, c.max_context_length);
}

// ── ContextChunk ────────────────────────────────────────────────

#[test]
fn context_chunk_new() {
    let chunk = ContextChunk::new(1, "Hello world".to_string(), 2, 0, ChunkRole::User);
    assert_eq!(chunk.id, 1);
    assert_eq!(chunk.text, "Hello world");
    assert_eq!(chunk.token_count, 2);
    assert_eq!(chunk.start_offset, 0);
    assert_eq!(chunk.role, ChunkRole::User);
}

#[test]
fn context_chunk_touch() {
    let mut chunk = ContextChunk::new(1, "test".to_string(), 1, 0, ChunkRole::Context);
    chunk.touch();
    // touch updates last_accessed — no panic
}

#[test]
fn context_chunk_clone() {
    let chunk = ContextChunk::new(2, "data".to_string(), 1, 5, ChunkRole::Assistant);
    let chunk2 = chunk.clone();
    assert_eq!(chunk2.id, 2);
    assert_eq!(chunk2.text, "data");
}

#[test]
fn context_chunk_importance_default() {
    let chunk = ContextChunk::new(1, "x".to_string(), 1, 0, ChunkRole::User);
    // Default importance should be 0.0
    assert!(chunk.importance_score >= 0.0);
}

// ── ImportanceScorer ────────────────────────────────────────────

#[test]
fn importance_scorer_default() {
    let scorer = ImportanceScorer::default();
    let _ = format!("{:?}", scorer);
}

#[test]
fn importance_scorer_custom() {
    let scorer = ImportanceScorer::new(0.5, 0.3, 0.2);
    assert_eq!(scorer.recency_weight, 0.5);
    assert_eq!(scorer.role_weight, 0.3);
    assert_eq!(scorer.relevance_weight, 0.2);
}

#[test]
fn importance_scorer_score() {
    let scorer = ImportanceScorer::new(0.5, 0.3, 0.2);
    let chunk = ContextChunk::new(1, "hello".to_string(), 1, 0, ChunkRole::User);
    let score = scorer.score(&chunk, 0, 1);
    assert!(score >= 0.0);
}

#[test]
fn importance_scorer_system_role_higher() {
    let scorer = ImportanceScorer::new(0.0, 1.0, 0.0);
    let system = ContextChunk::new(1, "sys".to_string(), 1, 0, ChunkRole::System);
    let user = ContextChunk::new(2, "usr".to_string(), 1, 0, ChunkRole::User);
    let sys_score = scorer.score(&system, 0, 2);
    let usr_score = scorer.score(&user, 1, 2);
    // System role should score higher with role_weight=1.0
    assert!(sys_score >= usr_score);
}

#[test]
fn importance_scorer_with_keywords() {
    let mut scorer = ImportanceScorer::new(0.0, 0.0, 1.0);
    scorer.set_keywords(vec!["important".to_string()]);
    let relevant =
        ContextChunk::new(1, "this is important text".to_string(), 4, 0, ChunkRole::User);
    let irrelevant = ContextChunk::new(2, "nothing here".to_string(), 2, 0, ChunkRole::User);
    let r_score = scorer.score(&relevant, 0, 2);
    let i_score = scorer.score(&irrelevant, 1, 2);
    assert!(r_score >= i_score);
}

// ── ContextCompressor ───────────────────────────────────────────

#[test]
fn context_compressor_default() {
    let c = ContextCompressor::default();
    let _ = format!("{:?}", c);
}

#[test]
fn context_compressor_custom() {
    let c = ContextCompressor::new(0.5);
    assert_eq!(c.target_ratio, 0.5);
}

#[test]
fn context_compressor_compress_chunk() {
    let c = ContextCompressor::new(0.5);
    let mut chunk = ContextChunk::new(
        1,
        "This is a fairly long piece of text that should get compressed".to_string(),
        12,
        0,
        ChunkRole::User,
    );
    let saved = c.compress_chunk(&mut chunk);
    // saved is a count of tokens saved
    let _ = saved;
}

#[test]
fn context_compressor_compress_all() {
    let c = ContextCompressor::new(0.5);
    let mut chunks = vec![
        ContextChunk::new(1, "First chunk of text here".to_string(), 5, 0, ChunkRole::User),
        ContextChunk::new(
            2,
            "Second chunk with more content".to_string(),
            6,
            5,
            ChunkRole::Context,
        ),
    ];
    let total_saved = c.compress_all(&mut chunks);
    let _ = total_saved;
}

// ── ContextManager ──────────────────────────────────────────────

#[test]
fn context_manager_new() {
    let config = ContextConfig::default();
    let mgr = ContextManager::new(config.clone());
    assert_eq!(mgr.capacity(), config.effective_capacity());
    assert_eq!(mgr.total_tokens(), 0);
    assert_eq!(mgr.chunk_count(), 0);
}

#[test]
fn context_manager_add() {
    let config = ContextConfig {
        max_context_length: 1000,
        reserved_for_generation: 100,
        chunking_strategy: ChunkingStrategy::NoChunking,
        eviction_strategy: EvictionStrategy::OldestFirst,
    };
    let mut mgr = ContextManager::new(config);
    let ids = mgr.add("Hello world", ChunkRole::User);
    assert!(!ids.is_empty());
    assert!(mgr.total_tokens() > 0);
    assert!(mgr.chunk_count() > 0);
}

#[test]
fn context_manager_add_multiple() {
    let config = ContextConfig::default();
    let mut mgr = ContextManager::new(config);
    mgr.add("First message", ChunkRole::System);
    mgr.add("Second message", ChunkRole::User);
    assert!(mgr.chunk_count() >= 2);
}

#[test]
fn context_manager_evict_by_id() {
    let config = ContextConfig::default();
    let mut mgr = ContextManager::new(config);
    let ids = mgr.add("removable", ChunkRole::Context);
    let id = ids[0];
    assert!(mgr.evict_by_id(id));
    assert!(!mgr.evict_by_id(id)); // already removed
}

#[test]
fn context_manager_touch() {
    let config = ContextConfig::default();
    let mut mgr = ContextManager::new(config);
    let ids = mgr.add("touchable", ChunkRole::User);
    assert!(mgr.touch(ids[0]));
    assert!(!mgr.touch(99999)); // nonexistent
}

#[test]
fn context_manager_get_chunk() {
    let config = ContextConfig::default();
    let mut mgr = ContextManager::new(config);
    let ids = mgr.add("findable", ChunkRole::User);
    let chunk = mgr.get_chunk(ids[0]);
    assert!(chunk.is_some());
    assert!(chunk.unwrap().text.contains("findable"));
}

#[test]
fn context_manager_clear() {
    let config = ContextConfig::default();
    let mut mgr = ContextManager::new(config);
    mgr.add("a", ChunkRole::User);
    mgr.add("b", ChunkRole::Assistant);
    mgr.clear();
    assert_eq!(mgr.chunk_count(), 0);
    assert_eq!(mgr.total_tokens(), 0);
}

#[test]
fn context_manager_get_active() {
    let config = ContextConfig::default();
    let mut mgr = ContextManager::new(config);
    mgr.add("active chunk", ChunkRole::User);
    let window = mgr.get_active();
    assert!(!window.chunks.is_empty());
    assert!(window.total_tokens > 0);
}

#[test]
fn context_manager_snapshot() {
    let config = ContextConfig::default();
    let mut mgr = ContextManager::new(config);
    mgr.add("snap chunk", ChunkRole::User);
    let snap = mgr.snapshot();
    assert_eq!(snap.chunk_count, mgr.chunk_count());
    assert!(!snap.chunk_texts.is_empty());
}

#[test]
fn context_manager_metrics() {
    let config = ContextConfig::default();
    let mut mgr = ContextManager::new(config);
    mgr.add("metrics test", ChunkRole::User);
    let metrics = mgr.metrics();
    assert!(metrics.utilization >= 0.0);
    assert!(metrics.chunk_count > 0);
    assert!(metrics.total_tokens > 0);
}

#[test]
fn context_manager_remaining() {
    let config = ContextConfig {
        max_context_length: 1000,
        reserved_for_generation: 100,
        chunking_strategy: ChunkingStrategy::NoChunking,
        eviction_strategy: EvictionStrategy::OldestFirst,
    };
    let mut mgr = ContextManager::new(config);
    let before = mgr.remaining();
    mgr.add("some text", ChunkRole::User);
    let after = mgr.remaining();
    assert!(after < before);
}

#[test]
fn context_manager_rescore() {
    let config = ContextConfig::default();
    let mut mgr = ContextManager::new(config);
    mgr.set_scorer(ImportanceScorer::new(0.5, 0.3, 0.2));
    mgr.add("hello", ChunkRole::User);
    mgr.add("world", ChunkRole::Assistant);
    mgr.rescore(); // Should not panic
}

#[test]
fn context_manager_compress() {
    let config = ContextConfig::default();
    let mut mgr = ContextManager::new(config);
    mgr.set_compressor(ContextCompressor::new(0.5));
    mgr.add("This is a longer piece of text to compress", ChunkRole::User);
    let saved = mgr.compress();
    let _ = saved;
}

#[test]
fn context_manager_set_scorer() {
    let config = ContextConfig::default();
    let mut mgr = ContextManager::new(config);
    mgr.set_scorer(ImportanceScorer::new(1.0, 0.0, 0.0));
    // No panic
}

#[test]
fn context_manager_chunk_text() {
    let config = ContextConfig {
        max_context_length: 1000,
        reserved_for_generation: 100,
        chunking_strategy: ChunkingStrategy::FixedSize(10),
        eviction_strategy: EvictionStrategy::OldestFirst,
    };
    let mgr = ContextManager::new(config);
    let chunks = mgr.chunk_text("This is a test sentence for chunking.", ChunkRole::User);
    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert_eq!(chunk.role, ChunkRole::User);
    }
}

// ── ContextWindow ───────────────────────────────────────────────

#[test]
fn context_window_fields() {
    let config = ContextConfig::default();
    let mut mgr = ContextManager::new(config);
    mgr.add("window test", ChunkRole::User);
    let window = mgr.get_active();
    assert!(!window.chunks.is_empty());
    assert!(window.total_tokens > 0);
    assert!(window.capacity_remaining > 0);
}

// ── ContextSnapshot ─────────────────────────────────────────────

#[test]
fn context_snapshot_fields() {
    let config = ContextConfig::default();
    let mut mgr = ContextManager::new(config);
    mgr.add("snapshot test", ChunkRole::System);
    let snap = mgr.snapshot();
    assert_eq!(snap.chunk_count, 1);
    assert!(snap.total_tokens > 0);
    assert!(!snap.chunk_texts.is_empty());
    assert_eq!(snap.importance_scores.len(), snap.chunk_count);
}

// ── ContextMetrics ──────────────────────────────────────────────

#[test]
fn context_metrics_empty() {
    let config = ContextConfig::default();
    let mgr = ContextManager::new(config);
    let metrics = mgr.metrics();
    assert_eq!(metrics.chunk_count, 0);
    assert_eq!(metrics.total_tokens, 0);
    assert_eq!(metrics.eviction_count, 0);
}
