//! Vocabulary and output projection management for GPU-efficient token
//! embedding/unembedding operations.
//!
//! Provides CPU reference implementations for:
//!
//! - **Embedding lookup**: token ID → dense vector from a learned table
//! - **Output projection**: hidden state → logits via `hidden @ weight^T`
//! - **Tied embeddings**: shared weight matrix for embedding and projection
//! - **Nearest-token search**: cosine similarity over the vocabulary
//! - **Batch operations**: vectorised lookup and projection for sequences

use std::fmt;

// ── Error type ──────────────────────────────────────────────────────

/// Errors specific to vocabulary management.
#[derive(Debug, Clone, PartialEq)]
pub enum VocabError {
    /// Token ID exceeds the vocabulary size.
    TokenOutOfRange { token: u32, vocab_size: usize },
    /// No embedding table has been loaded.
    EmbeddingNotLoaded,
    /// No projection head has been loaded.
    ProjectionNotLoaded,
    /// Vector dimension does not match `hidden_dim`.
    DimensionMismatch,
}

impl fmt::Display for VocabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenOutOfRange { token, vocab_size } => {
                write!(f, "token {token} out of range (vocab_size={vocab_size})")
            }
            Self::EmbeddingNotLoaded => write!(f, "embedding table not loaded"),
            Self::ProjectionNotLoaded => {
                write!(f, "projection head not loaded")
            }
            Self::DimensionMismatch => write!(f, "dimension mismatch"),
        }
    }
}

impl std::error::Error for VocabError {}

// ── Configuration ───────────────────────────────────────────────────

/// Configuration for vocabulary and projection layers.
#[derive(Debug, Clone)]
pub struct VocabConfig {
    /// Number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Hidden (embedding) dimensionality.
    pub hidden_dim: usize,
    /// Optional padding index: tokens with this ID produce zero vectors.
    pub padding_idx: Option<u32>,
    /// Whether the projection head shares embedding weights.
    pub tie_embeddings: bool,
}

// ── Embedding table ─────────────────────────────────────────────────

/// Token embedding table: maps token IDs to dense vectors.
#[derive(Debug, Clone)]
pub struct EmbeddingTable {
    /// Weight matrix in row-major layout `[vocab_size, hidden_dim]`.
    pub weights: Vec<f32>,
    pub vocab_size: usize,
    pub hidden_dim: usize,
}

// ── Projection head ─────────────────────────────────────────────────

/// Output projection (lm_head) that maps hidden states to logits.
#[derive(Debug, Clone)]
pub struct ProjectionHead {
    /// Weight matrix `[vocab_size, hidden_dim]`.
    pub weights: Vec<f32>,
    /// Optional bias vector `[vocab_size]`.
    pub bias: Option<Vec<f32>>,
    pub vocab_size: usize,
    pub hidden_dim: usize,
}

// ── Stats ───────────────────────────────────────────────────────────

/// Cumulative statistics for vocabulary operations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VocabStats {
    pub lookups: u64,
    pub projections: u64,
    pub cache_hits: u64,
}

// ── Manager ─────────────────────────────────────────────────────────

/// Central manager for embedding lookup and output projection.
#[derive(Debug, Clone)]
pub struct VocabManager {
    pub config: VocabConfig,
    pub embedding: Option<EmbeddingTable>,
    pub projection: Option<ProjectionHead>,
    pub stats: VocabStats,
}

// ── Public API (free functions) ─────────────────────────────────────

/// Create a new `VocabManager` with no weights loaded.
pub fn create_vocab_manager(config: VocabConfig) -> VocabManager {
    VocabManager { config, embedding: None, projection: None, stats: VocabStats::default() }
}

/// Load an embedding table into the manager.
///
/// # Panics
/// Panics if `weights.len() != vocab_size * hidden_dim`.
pub fn cpu_load_embeddings(mgr: &mut VocabManager, weights: Vec<f32>) {
    let expected = mgr.config.vocab_size * mgr.config.hidden_dim;
    assert_eq!(
        weights.len(),
        expected,
        "embedding weight length {} != vocab_size({}) * hidden_dim({})",
        weights.len(),
        mgr.config.vocab_size,
        mgr.config.hidden_dim,
    );
    mgr.embedding = Some(EmbeddingTable {
        weights,
        vocab_size: mgr.config.vocab_size,
        hidden_dim: mgr.config.hidden_dim,
    });
}

/// Load a projection head into the manager.
///
/// # Panics
/// Panics if `weights.len() != vocab_size * hidden_dim` or if `bias`
/// length does not equal `vocab_size`.
pub fn cpu_load_projection(mgr: &mut VocabManager, weights: Vec<f32>, bias: Option<Vec<f32>>) {
    let expected = mgr.config.vocab_size * mgr.config.hidden_dim;
    assert_eq!(
        weights.len(),
        expected,
        "projection weight length {} != vocab_size({}) * hidden_dim({})",
        weights.len(),
        mgr.config.vocab_size,
        mgr.config.hidden_dim,
    );
    if let Some(ref b) = bias {
        assert_eq!(
            b.len(),
            mgr.config.vocab_size,
            "bias length {} != vocab_size({})",
            b.len(),
            mgr.config.vocab_size,
        );
    }
    mgr.projection = Some(ProjectionHead {
        weights,
        bias,
        vocab_size: mgr.config.vocab_size,
        hidden_dim: mgr.config.hidden_dim,
    });
}

/// Look up the embedding vector for a single token.
pub fn cpu_lookup_embedding(mgr: &VocabManager, token_id: u32) -> Result<Vec<f32>, VocabError> {
    let emb = mgr.embedding.as_ref().ok_or(VocabError::EmbeddingNotLoaded)?;
    let tid = token_id as usize;
    if tid >= emb.vocab_size {
        return Err(VocabError::TokenOutOfRange { token: token_id, vocab_size: emb.vocab_size });
    }
    // Padding index returns zeros.
    if mgr.config.padding_idx == Some(token_id) {
        return Ok(vec![0.0; emb.hidden_dim]);
    }
    let start = tid * emb.hidden_dim;
    let end = start + emb.hidden_dim;
    Ok(emb.weights[start..end].to_vec())
}

/// Batch embedding lookup. Updates `stats.lookups`.
pub fn cpu_batch_lookup(
    mgr: &mut VocabManager,
    token_ids: &[u32],
) -> Result<Vec<Vec<f32>>, VocabError> {
    let mut result = Vec::with_capacity(token_ids.len());
    for &tid in token_ids {
        result.push(cpu_lookup_embedding(mgr, tid)?);
    }
    mgr.stats.lookups += token_ids.len() as u64;
    Ok(result)
}

/// Project a single hidden state to vocabulary logits.
///
/// When `tie_embeddings` is set and no explicit projection is loaded,
/// the embedding table weights are used.
pub fn cpu_project_to_vocab(
    mgr: &mut VocabManager,
    hidden: &[f32],
) -> Result<Vec<f32>, VocabError> {
    let (weights, bias, vocab_size, hidden_dim) = resolve_projection(mgr)?;

    if hidden.len() != hidden_dim {
        return Err(VocabError::DimensionMismatch);
    }

    let mut logits = Vec::with_capacity(vocab_size);
    for v in 0..vocab_size {
        let row_start = v * hidden_dim;
        let mut dot: f32 = 0.0;
        for j in 0..hidden_dim {
            dot += weights[row_start + j] * hidden[j];
        }
        if let Some(b) = bias {
            dot += b[v];
        }
        logits.push(dot);
    }
    mgr.stats.projections += 1;
    Ok(logits)
}

/// Batch projection: project multiple hidden states to logits.
pub fn cpu_batch_project(
    mgr: &mut VocabManager,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, VocabError> {
    let mut results = Vec::with_capacity(hidden_states.len());
    for h in hidden_states {
        results.push(cpu_project_to_vocab(mgr, h)?);
    }
    Ok(results)
}

/// L2 norm of a token's embedding vector.
pub fn cpu_get_embedding_norm(mgr: &VocabManager, token_id: u32) -> Result<f32, VocabError> {
    let vec = cpu_lookup_embedding(mgr, token_id)?;
    Ok(vec.iter().map(|x| x * x).sum::<f32>().sqrt())
}

/// Find the token whose embedding is most similar (cosine) to `vector`.
///
/// Returns `(token_id, cosine_similarity)`.
pub fn cpu_find_nearest_token(
    mgr: &VocabManager,
    vector: &[f32],
) -> Result<(u32, f32), VocabError> {
    let emb = mgr.embedding.as_ref().ok_or(VocabError::EmbeddingNotLoaded)?;
    if vector.len() != emb.hidden_dim {
        return Err(VocabError::DimensionMismatch);
    }

    let query_norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if query_norm == 0.0 {
        return Ok((0, 0.0));
    }

    let mut best_id: u32 = 0;
    let mut best_sim: f32 = f32::NEG_INFINITY;

    for v in 0..emb.vocab_size {
        // Skip padding token.
        if mgr.config.padding_idx == Some(v as u32) {
            continue;
        }
        let start = v * emb.hidden_dim;
        let row = &emb.weights[start..start + emb.hidden_dim];
        let mut dot: f32 = 0.0;
        let mut row_norm_sq: f32 = 0.0;
        for (w, v_j) in row.iter().zip(vector.iter()) {
            dot += w * v_j;
            row_norm_sq += w * w;
        }
        let row_norm = row_norm_sq.sqrt();
        if row_norm == 0.0 {
            continue;
        }
        let sim = dot / (row_norm * query_norm);
        if sim > best_sim {
            best_sim = sim;
            best_id = v as u32;
        }
    }
    Ok((best_id, best_sim))
}

/// Total memory footprint (bytes) of loaded weights in the manager.
pub fn cpu_memory_footprint(mgr: &VocabManager) -> usize {
    let emb_bytes = mgr.embedding.as_ref().map_or(0, |e| e.weights.len() * size_of::<f32>());
    let proj_bytes = mgr.projection.as_ref().map_or(0, |p| {
        let w = p.weights.len() * size_of::<f32>();
        let b = p.bias.as_ref().map_or(0, |b| b.len() * size_of::<f32>());
        w + b
    });
    emb_bytes + proj_bytes
}

/// Return a snapshot of cumulative statistics.
pub fn cpu_get_stats(mgr: &VocabManager) -> VocabStats {
    mgr.stats.clone()
}

/// Human-readable summary.
pub fn format_vocab_info(mgr: &VocabManager) -> String {
    let emb_status = if mgr.embedding.is_some() { "loaded" } else { "not loaded" };
    let proj_status = if mgr.projection.is_some() {
        "loaded"
    } else if mgr.config.tie_embeddings && mgr.embedding.is_some() {
        "tied"
    } else {
        "not loaded"
    };
    format!(
        "VocabManager {{ vocab_size: {}, hidden_dim: {}, \
         embedding: {}, projection: {}, tie_embeddings: {}, \
         lookups: {}, projections: {} }}",
        mgr.config.vocab_size,
        mgr.config.hidden_dim,
        emb_status,
        proj_status,
        mgr.config.tie_embeddings,
        mgr.stats.lookups,
        mgr.stats.projections,
    )
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Resolved projection weights for a single projection call.
type ProjectionRef<'a> = (&'a [f32], &'a Option<Vec<f32>>, usize, usize);

/// Resolve projection weights, falling back to tied embeddings.
fn resolve_projection(mgr: &VocabManager) -> Result<ProjectionRef<'_>, VocabError> {
    if let Some(proj) = &mgr.projection {
        return Ok((&proj.weights, &proj.bias, proj.vocab_size, proj.hidden_dim));
    }
    if mgr.config.tie_embeddings
        && let Some(emb) = &mgr.embedding
    {
        return Ok((&emb.weights, &None, emb.vocab_size, emb.hidden_dim));
    }
    Err(VocabError::ProjectionNotLoaded)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────

    fn simple_config(vocab: usize, dim: usize) -> VocabConfig {
        VocabConfig { vocab_size: vocab, hidden_dim: dim, padding_idx: None, tie_embeddings: false }
    }

    fn identity_weights(vocab: usize, dim: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; vocab * dim];
        for v in 0..vocab {
            for d in 0..dim {
                w[v * dim + d] = if v == d { 1.0 } else { 0.0 };
            }
        }
        w
    }

    fn sequential_weights(vocab: usize, dim: usize) -> Vec<f32> {
        (0..vocab * dim).map(|i| i as f32).collect()
    }

    fn ones_weights(vocab: usize, dim: usize) -> Vec<f32> {
        vec![1.0f32; vocab * dim]
    }

    fn make_loaded_mgr(vocab: usize, dim: usize) -> VocabManager {
        let cfg = simple_config(vocab, dim);
        let mut mgr = create_vocab_manager(cfg);
        cpu_load_embeddings(&mut mgr, sequential_weights(vocab, dim));
        cpu_load_projection(&mut mgr, sequential_weights(vocab, dim), None);
        mgr
    }

    // ── loading ─────────────────────────────────────────────────

    #[test]
    fn test_create_vocab_manager() {
        let mgr = create_vocab_manager(simple_config(100, 64));
        assert!(mgr.embedding.is_none());
        assert!(mgr.projection.is_none());
        assert_eq!(mgr.stats, VocabStats::default());
    }

    #[test]
    fn test_load_embeddings() {
        let mut mgr = create_vocab_manager(simple_config(4, 3));
        cpu_load_embeddings(&mut mgr, sequential_weights(4, 3));
        assert!(mgr.embedding.is_some());
        let emb = mgr.embedding.as_ref().unwrap();
        assert_eq!(emb.vocab_size, 4);
        assert_eq!(emb.hidden_dim, 3);
    }

    #[test]
    #[should_panic(expected = "embedding weight length")]
    fn test_load_embeddings_wrong_size() {
        let mut mgr = create_vocab_manager(simple_config(4, 3));
        cpu_load_embeddings(&mut mgr, vec![0.0; 10]);
    }

    #[test]
    fn test_load_projection() {
        let mut mgr = create_vocab_manager(simple_config(4, 3));
        let bias = vec![0.1, 0.2, 0.3, 0.4];
        cpu_load_projection(&mut mgr, sequential_weights(4, 3), Some(bias));
        assert!(mgr.projection.is_some());
        assert!(mgr.projection.as_ref().unwrap().bias.is_some());
    }

    #[test]
    #[should_panic(expected = "projection weight length")]
    fn test_load_projection_wrong_size() {
        let mut mgr = create_vocab_manager(simple_config(4, 3));
        cpu_load_projection(&mut mgr, vec![0.0; 5], None);
    }

    #[test]
    #[should_panic(expected = "bias length")]
    fn test_load_projection_wrong_bias() {
        let mut mgr = create_vocab_manager(simple_config(4, 3));
        cpu_load_projection(&mut mgr, sequential_weights(4, 3), Some(vec![1.0, 2.0]));
    }

    // ── single lookup ───────────────────────────────────────────

    #[test]
    fn test_lookup_embedding_basic() {
        let mgr = make_loaded_mgr(4, 3);
        let vec = cpu_lookup_embedding(&mgr, 0).unwrap();
        assert_eq!(vec, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_lookup_embedding_last_token() {
        let mgr = make_loaded_mgr(4, 3);
        let vec = cpu_lookup_embedding(&mgr, 3).unwrap();
        assert_eq!(vec, vec![9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_lookup_out_of_range() {
        let mgr = make_loaded_mgr(4, 3);
        let err = cpu_lookup_embedding(&mgr, 4).unwrap_err();
        assert_eq!(err, VocabError::TokenOutOfRange { token: 4, vocab_size: 4 });
    }

    #[test]
    fn test_lookup_embedding_not_loaded() {
        let mgr = create_vocab_manager(simple_config(4, 3));
        assert_eq!(cpu_lookup_embedding(&mgr, 0).unwrap_err(), VocabError::EmbeddingNotLoaded,);
    }

    #[test]
    fn test_lookup_returns_correct_dim() {
        let mgr = make_loaded_mgr(8, 16);
        let vec = cpu_lookup_embedding(&mgr, 5).unwrap();
        assert_eq!(vec.len(), 16);
    }

    // ── batch lookup ────────────────────────────────────────────

    #[test]
    fn test_batch_lookup() {
        let mut mgr = make_loaded_mgr(4, 3);
        let vecs = cpu_batch_lookup(&mut mgr, &[0, 2]).unwrap();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0], vec![0.0, 1.0, 2.0]);
        assert_eq!(vecs[1], vec![6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_batch_lookup_updates_stats() {
        let mut mgr = make_loaded_mgr(4, 3);
        cpu_batch_lookup(&mut mgr, &[0, 1, 2]).unwrap();
        assert_eq!(mgr.stats.lookups, 3);
    }

    #[test]
    fn test_batch_lookup_empty() {
        let mut mgr = make_loaded_mgr(4, 3);
        let vecs = cpu_batch_lookup(&mut mgr, &[]).unwrap();
        assert!(vecs.is_empty());
    }

    #[test]
    fn test_batch_lookup_out_of_range() {
        let mut mgr = make_loaded_mgr(4, 3);
        let err = cpu_batch_lookup(&mut mgr, &[0, 5]).unwrap_err();
        assert_eq!(err, VocabError::TokenOutOfRange { token: 5, vocab_size: 4 });
    }

    // ── projection ──────────────────────────────────────────────

    #[test]
    fn test_project_to_vocab() {
        let mut mgr = make_loaded_mgr(3, 2);
        // weights: [[0,1],[2,3],[4,5]], hidden=[1,0]
        let logits = cpu_project_to_vocab(&mut mgr, &[1.0, 0.0]).unwrap();
        assert_eq!(logits, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_project_with_bias() {
        let mut mgr = create_vocab_manager(simple_config(3, 2));
        cpu_load_projection(
            &mut mgr,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec![10.0, 20.0, 30.0]),
        );
        let logits = cpu_project_to_vocab(&mut mgr, &[1.0, 0.0]).unwrap();
        assert_eq!(logits, vec![10.0, 22.0, 34.0]);
    }

    #[test]
    fn test_project_dim_mismatch() {
        let mut mgr = make_loaded_mgr(3, 2);
        let err = cpu_project_to_vocab(&mut mgr, &[1.0, 2.0, 3.0]).unwrap_err();
        assert_eq!(err, VocabError::DimensionMismatch);
    }

    #[test]
    fn test_project_not_loaded() {
        let cfg = simple_config(4, 3);
        let mut mgr = create_vocab_manager(cfg);
        let err = cpu_project_to_vocab(&mut mgr, &[1.0, 2.0, 3.0]).unwrap_err();
        assert_eq!(err, VocabError::ProjectionNotLoaded);
    }

    #[test]
    fn test_project_returns_vocab_size_logits() {
        let mut mgr = make_loaded_mgr(10, 4);
        let logits = cpu_project_to_vocab(&mut mgr, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(logits.len(), 10);
    }

    #[test]
    fn test_project_updates_stats() {
        let mut mgr = make_loaded_mgr(3, 2);
        cpu_project_to_vocab(&mut mgr, &[1.0, 0.0]).unwrap();
        cpu_project_to_vocab(&mut mgr, &[0.0, 1.0]).unwrap();
        assert_eq!(mgr.stats.projections, 2);
    }

    // ── batch projection ────────────────────────────────────────

    #[test]
    fn test_batch_project() {
        let mut mgr = make_loaded_mgr(3, 2);
        let hidden = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let results = cpu_batch_project(&mut mgr, &hidden).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], vec![0.0, 2.0, 4.0]);
        assert_eq!(results[1], vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_batch_project_empty() {
        let mut mgr = make_loaded_mgr(3, 2);
        let results = cpu_batch_project(&mut mgr, &[]).unwrap();
        assert!(results.is_empty());
    }

    // ── tied embeddings ─────────────────────────────────────────

    #[test]
    fn test_tied_embeddings_project() {
        let mut cfg = simple_config(3, 2);
        cfg.tie_embeddings = true;
        let mut mgr = create_vocab_manager(cfg);
        cpu_load_embeddings(&mut mgr, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        // No explicit projection loaded — should use embedding weights.
        let logits = cpu_project_to_vocab(&mut mgr, &[1.0, 0.0]).unwrap();
        assert_eq!(logits, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_tied_no_embedding_fails() {
        let mut cfg = simple_config(3, 2);
        cfg.tie_embeddings = true;
        let mut mgr = create_vocab_manager(cfg);
        // Neither embedding nor projection loaded.
        let err = cpu_project_to_vocab(&mut mgr, &[1.0, 0.0]).unwrap_err();
        assert_eq!(err, VocabError::ProjectionNotLoaded);
    }

    #[test]
    fn test_tied_explicit_proj_takes_priority() {
        let mut cfg = simple_config(3, 2);
        cfg.tie_embeddings = true;
        let mut mgr = create_vocab_manager(cfg);
        cpu_load_embeddings(&mut mgr, ones_weights(3, 2));
        // Load an explicit projection with different weights.
        cpu_load_projection(&mut mgr, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], None);
        let logits = cpu_project_to_vocab(&mut mgr, &[3.0, 7.0]).unwrap();
        // explicit: [3*1+7*0, 3*0+7*1, 3*1+7*1] = [3, 7, 10]
        assert_eq!(logits, vec![3.0, 7.0, 10.0]);
    }

    // ── padding ─────────────────────────────────────────────────

    #[test]
    fn test_padding_idx_returns_zeros() {
        let mut cfg = simple_config(4, 3);
        cfg.padding_idx = Some(2);
        let mut mgr = create_vocab_manager(cfg);
        cpu_load_embeddings(&mut mgr, sequential_weights(4, 3));
        let vec = cpu_lookup_embedding(&mgr, 2).unwrap();
        assert_eq!(vec, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_non_padding_unaffected() {
        let mut cfg = simple_config(4, 3);
        cfg.padding_idx = Some(2);
        let mut mgr = create_vocab_manager(cfg);
        cpu_load_embeddings(&mut mgr, sequential_weights(4, 3));
        let vec = cpu_lookup_embedding(&mgr, 1).unwrap();
        assert_eq!(vec, vec![3.0, 4.0, 5.0]);
    }

    // ── embedding norm ──────────────────────────────────────────

    #[test]
    fn test_embedding_norm() {
        let mut mgr = create_vocab_manager(simple_config(2, 3));
        cpu_load_embeddings(&mut mgr, vec![3.0, 4.0, 0.0, 0.0, 0.0, 1.0]);
        let norm = cpu_get_embedding_norm(&mgr, 0).unwrap();
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_norm_not_loaded() {
        let mgr = create_vocab_manager(simple_config(2, 3));
        assert_eq!(cpu_get_embedding_norm(&mgr, 0).unwrap_err(), VocabError::EmbeddingNotLoaded,);
    }

    // ── nearest token ───────────────────────────────────────────

    #[test]
    fn test_find_nearest_token_exact() {
        let mut mgr = create_vocab_manager(simple_config(3, 3));
        cpu_load_embeddings(&mut mgr, identity_weights(3, 3));
        let (id, sim) = cpu_find_nearest_token(&mgr, &[0.0, 1.0, 0.0]).unwrap();
        assert_eq!(id, 1);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_nearest_token_closest() {
        let mut mgr = create_vocab_manager(simple_config(3, 2));
        // token 0: [1,0], token 1: [0,1], token 2: [1,1]
        cpu_load_embeddings(&mut mgr, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let (id, _sim) = cpu_find_nearest_token(&mgr, &[0.9, 0.1]).unwrap();
        assert_eq!(id, 0);
    }

    #[test]
    fn test_find_nearest_embedding_not_loaded() {
        let mgr = create_vocab_manager(simple_config(3, 3));
        assert_eq!(
            cpu_find_nearest_token(&mgr, &[1.0, 0.0, 0.0]).unwrap_err(),
            VocabError::EmbeddingNotLoaded,
        );
    }

    #[test]
    fn test_find_nearest_dim_mismatch() {
        let mut mgr = create_vocab_manager(simple_config(3, 3));
        cpu_load_embeddings(&mut mgr, identity_weights(3, 3));
        assert_eq!(
            cpu_find_nearest_token(&mgr, &[1.0, 0.0]).unwrap_err(),
            VocabError::DimensionMismatch,
        );
    }

    #[test]
    fn test_find_nearest_skips_padding() {
        let mut cfg = simple_config(3, 3);
        cfg.padding_idx = Some(1);
        let mut mgr = create_vocab_manager(cfg);
        // token 0: [1,0,0], token 1: [0,1,0] (pad), token 2: [0,0,1]
        cpu_load_embeddings(&mut mgr, identity_weights(3, 3));
        let (id, _sim) = cpu_find_nearest_token(&mgr, &[0.0, 1.0, 0.0]).unwrap();
        // token 1 is padding → skipped; best remaining is 0 or 2 (both 0 sim).
        assert_ne!(id, 1);
    }

    #[test]
    fn test_find_nearest_zero_vector() {
        let mut mgr = create_vocab_manager(simple_config(3, 3));
        cpu_load_embeddings(&mut mgr, identity_weights(3, 3));
        let (id, sim) = cpu_find_nearest_token(&mgr, &[0.0, 0.0, 0.0]).unwrap();
        assert_eq!(id, 0);
        assert_eq!(sim, 0.0);
    }

    // ── memory footprint ────────────────────────────────────────

    #[test]
    fn test_memory_footprint_empty() {
        let mgr = create_vocab_manager(simple_config(4, 3));
        assert_eq!(cpu_memory_footprint(&mgr), 0);
    }

    #[test]
    fn test_memory_footprint_embedding_only() {
        let mut mgr = create_vocab_manager(simple_config(4, 3));
        cpu_load_embeddings(&mut mgr, sequential_weights(4, 3));
        assert_eq!(cpu_memory_footprint(&mgr), 4 * 3 * 4); // 48 bytes
    }

    #[test]
    fn test_memory_footprint_full() {
        let mut mgr = create_vocab_manager(simple_config(4, 3));
        cpu_load_embeddings(&mut mgr, sequential_weights(4, 3));
        cpu_load_projection(&mut mgr, sequential_weights(4, 3), Some(vec![0.0; 4]));
        // emb: 4*3*4 = 48, proj weights: 48, proj bias: 4*4 = 16
        assert_eq!(cpu_memory_footprint(&mgr), 48 + 48 + 16);
    }

    // ── stats ───────────────────────────────────────────────────

    #[test]
    fn test_get_stats_initial() {
        let mgr = create_vocab_manager(simple_config(4, 3));
        let s = cpu_get_stats(&mgr);
        assert_eq!(s.lookups, 0);
        assert_eq!(s.projections, 0);
        assert_eq!(s.cache_hits, 0);
    }

    #[test]
    fn test_stats_accumulate() {
        let mut mgr = make_loaded_mgr(4, 3);
        cpu_batch_lookup(&mut mgr, &[0, 1]).unwrap();
        cpu_project_to_vocab(&mut mgr, &[1.0, 0.0, 0.0]).unwrap();
        let s = cpu_get_stats(&mgr);
        assert_eq!(s.lookups, 2);
        assert_eq!(s.projections, 1);
    }

    // ── format_vocab_info ───────────────────────────────────────

    #[test]
    fn test_format_vocab_info_empty() {
        let mgr = create_vocab_manager(simple_config(100, 64));
        let info = format_vocab_info(&mgr);
        assert!(info.contains("vocab_size: 100"));
        assert!(info.contains("hidden_dim: 64"));
        assert!(info.contains("embedding: not loaded"));
    }

    #[test]
    fn test_format_vocab_info_loaded() {
        let mgr = make_loaded_mgr(4, 3);
        let info = format_vocab_info(&mgr);
        assert!(info.contains("embedding: loaded"));
        assert!(info.contains("projection: loaded"));
    }

    #[test]
    fn test_format_vocab_info_tied() {
        let mut cfg = simple_config(4, 3);
        cfg.tie_embeddings = true;
        let mut mgr = create_vocab_manager(cfg);
        cpu_load_embeddings(&mut mgr, sequential_weights(4, 3));
        let info = format_vocab_info(&mgr);
        assert!(info.contains("projection: tied"));
        assert!(info.contains("tie_embeddings: true"));
    }

    // ── edge cases ──────────────────────────────────────────────

    #[test]
    fn test_edge_vocab_size_1_dim_1() {
        let mut mgr = create_vocab_manager(simple_config(1, 1));
        cpu_load_embeddings(&mut mgr, vec![42.0]);
        cpu_load_projection(&mut mgr, vec![2.0], None);
        let emb = cpu_lookup_embedding(&mgr, 0).unwrap();
        assert_eq!(emb, vec![42.0]);
        let logits = cpu_project_to_vocab(&mut mgr, &[3.0]).unwrap();
        assert_eq!(logits, vec![6.0]);
    }

    #[test]
    fn test_edge_large_token_id_boundary() {
        let mgr = make_loaded_mgr(4, 3);
        // Just inside: token 3, just outside: token 4
        assert!(cpu_lookup_embedding(&mgr, 3).is_ok());
        assert!(cpu_lookup_embedding(&mgr, 4).is_err());
    }

    #[test]
    fn test_error_display() {
        let e = VocabError::TokenOutOfRange { token: 99, vocab_size: 50 };
        assert!(e.to_string().contains("99"));
        assert!(e.to_string().contains("50"));
        assert!(VocabError::EmbeddingNotLoaded.to_string().contains("embedding"));
        assert!(VocabError::ProjectionNotLoaded.to_string().contains("projection"));
        assert!(VocabError::DimensionMismatch.to_string().contains("dimension"));
    }
}
