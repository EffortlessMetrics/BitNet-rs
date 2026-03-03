//! Semantic / vector search engine for GPU-accelerated embedding lookups.
//!
//! Provides [`SemanticSearchEngine`] with pluggable index backends:
//! - [`FlatIndex`]: brute-force exact search (small datasets)
//! - [`HNSWIndex`]: approximate nearest-neighbor via Hierarchical Navigable Small World graphs
//! - [`IVFIndex`]: inverted-file cluster-based partitioning

use std::collections::HashMap;

// ── Configuration ────────────────────────────────────────────────────

/// Distance metric used for vector similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

/// Index algorithm selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    Flat,
    HNSW,
    IVF,
}

/// Search engine configuration.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub embedding_dim: usize,
    pub distance_metric: DistanceMetric,
    pub max_results: usize,
    pub index_type: IndexType,
    /// HNSW: beam width during search.
    pub ef_search: usize,
    /// HNSW: beam width during construction.
    pub ef_construction: usize,
    /// IVF: number of clusters.
    pub num_clusters: usize,
    /// IVF: number of clusters to probe during search.
    pub num_probes: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            distance_metric: DistanceMetric::Cosine,
            max_results: 10,
            index_type: IndexType::Flat,
            ef_search: 64,
            ef_construction: 128,
            num_clusters: 8,
            num_probes: 2,
        }
    }
}

/// Errors returned by the search engine.
#[derive(Debug, Clone, PartialEq)]
pub enum SearchError {
    DimensionMismatch { expected: usize, got: usize },
    EmptyVector,
    InvalidConfig(String),
    VectorNotFound(String),
    EmptyIndex,
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::EmptyVector => write!(f, "empty vector"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::VectorNotFound(id) => write!(f, "vector not found: {id}"),
            Self::EmptyIndex => write!(f, "index is empty"),
        }
    }
}

impl std::error::Error for SearchError {}

// ── Core types ───────────────────────────────────────────────────────

/// An embedding vector with associated metadata.
#[derive(Debug, Clone)]
pub struct EmbeddingVector {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

/// A single search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub distance: f32,
    pub metadata: HashMap<String, String>,
}

/// Aggregate statistics about an index.
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub num_vectors: usize,
    pub dimensions: usize,
    pub memory_bytes: usize,
    pub index_type: IndexType,
}

// ── Distance functions ───────────────────────────────────────────────

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    1.0 - (d / (na * nb))
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -dot(a, b)
}

fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

fn compute_distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::Euclidean => euclidean_distance(a, b),
        DistanceMetric::DotProduct => dot_product_distance(a, b),
        DistanceMetric::Manhattan => manhattan_distance(a, b),
    }
}

fn distance_to_score(metric: DistanceMetric, distance: f32) -> f32 {
    match metric {
        DistanceMetric::Cosine => 1.0 - distance,
        DistanceMetric::Euclidean => 1.0 / (1.0 + distance),
        DistanceMetric::DotProduct => -distance,
        DistanceMetric::Manhattan => 1.0 / (1.0 + distance),
    }
}

// ── FlatIndex ────────────────────────────────────────────────────────

/// Brute-force exact nearest-neighbor index.
#[derive(Debug, Clone)]
pub struct FlatIndex {
    vectors: Vec<EmbeddingVector>,
    dim: usize,
    metric: DistanceMetric,
}

impl FlatIndex {
    pub fn new(dim: usize, metric: DistanceMetric) -> Self {
        Self { vectors: Vec::new(), dim, metric }
    }

    pub fn insert(&mut self, vec: EmbeddingVector) -> Result<(), SearchError> {
        if vec.vector.is_empty() {
            return Err(SearchError::EmptyVector);
        }
        if vec.vector.len() != self.dim {
            return Err(SearchError::DimensionMismatch {
                expected: self.dim,
                got: vec.vector.len(),
            });
        }
        self.vectors.push(vec);
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, SearchError> {
        if query.len() != self.dim {
            return Err(SearchError::DimensionMismatch { expected: self.dim, got: query.len() });
        }
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }
        let mut scored: Vec<(f32, usize)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, ev)| (compute_distance(self.metric, query, &ev.vector), i))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored
            .into_iter()
            .take(k)
            .map(|(dist, i)| {
                let ev = &self.vectors[i];
                SearchResult {
                    id: ev.id.clone(),
                    score: distance_to_score(self.metric, dist),
                    distance: dist,
                    metadata: ev.metadata.clone(),
                }
            })
            .collect())
    }

    pub fn delete(&mut self, id: &str) -> bool {
        let before = self.vectors.len();
        self.vectors.retain(|v| v.id != id);
        self.vectors.len() < before
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn stats(&self) -> IndexStats {
        IndexStats {
            num_vectors: self.vectors.len(),
            dimensions: self.dim,
            memory_bytes: self.vectors.len() * self.dim * size_of::<f32>(),
            index_type: IndexType::Flat,
        }
    }
}

// ── HNSWNode / HNSWIndex ────────────────────────────────────────────

/// A node in the HNSW graph.
#[derive(Debug, Clone)]
pub struct HNSWNode {
    pub id: String,
    pub level: usize,
    /// Neighbor ids per level.
    pub neighbors: Vec<Vec<usize>>,
}

/// Hierarchical Navigable Small World approximate index.
#[derive(Debug, Clone)]
pub struct HNSWIndex {
    nodes: Vec<HNSWNode>,
    vectors: Vec<EmbeddingVector>,
    dim: usize,
    metric: DistanceMetric,
    max_level: usize,
    ef_construction: usize,
    ef_search: usize,
    max_neighbors: usize,
    entry_point: Option<usize>,
    level_mult: f64,
}

impl HNSWIndex {
    pub fn new(
        dim: usize,
        metric: DistanceMetric,
        ef_construction: usize,
        ef_search: usize,
    ) -> Self {
        Self {
            nodes: Vec::new(),
            vectors: Vec::new(),
            dim,
            metric,
            max_level: 0,
            ef_construction,
            ef_search,
            max_neighbors: 16,
            entry_point: None,
            level_mult: 1.0 / (16_f64).ln(),
        }
    }

    fn random_level(&self, id_hash: u64) -> usize {
        // Deterministic level from id hash for reproducibility.
        let uniform = ((id_hash.wrapping_mul(6364136223846793005).wrapping_add(1)) as f64)
            / (u64::MAX as f64);
        let level = (-uniform.ln() * self.level_mult) as usize;
        level.min(12)
    }

    pub fn insert(&mut self, vec: EmbeddingVector) -> Result<(), SearchError> {
        if vec.vector.is_empty() {
            return Err(SearchError::EmptyVector);
        }
        if vec.vector.len() != self.dim {
            return Err(SearchError::DimensionMismatch {
                expected: self.dim,
                got: vec.vector.len(),
            });
        }

        let idx = self.vectors.len();
        let id_hash = hash_id(&vec.id);
        let level = self.random_level(id_hash);

        let node = HNSWNode { id: vec.id.clone(), level, neighbors: vec![Vec::new(); level + 1] };

        self.vectors.push(vec);
        self.nodes.push(node);

        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            self.max_level = level;
            return Ok(());
        }

        let entry = self.entry_point.unwrap();

        // Greedy descent from top to insert level + 1.
        let mut current = entry;
        for lev in (level + 1..=self.max_level).rev() {
            current = self.greedy_closest(current, &self.vectors[idx].vector, lev);
        }

        // Insert into each level from min(level, max_level) down to 0.
        for lev in (0..=level.min(self.max_level)).rev() {
            let neighbors =
                self.search_level(current, &self.vectors[idx].vector, self.ef_construction, lev);
            let selected = Self::select_neighbors(&neighbors, self.max_neighbors);

            self.nodes[idx].neighbors[lev] = selected.clone();
            for &nb in &selected {
                if lev < self.nodes[nb].neighbors.len() {
                    self.nodes[nb].neighbors[lev].push(idx);
                    if self.nodes[nb].neighbors[lev].len() > self.max_neighbors * 2 {
                        self.prune_neighbors(nb, lev);
                    }
                }
            }
            if !neighbors.is_empty() {
                current = neighbors[0].0;
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(idx);
        }

        Ok(())
    }

    fn greedy_closest(&self, mut current: usize, query: &[f32], level: usize) -> usize {
        let mut best_dist = compute_distance(self.metric, query, &self.vectors[current].vector);
        loop {
            let mut changed = false;
            if level < self.nodes[current].neighbors.len() {
                for &nb in &self.nodes[current].neighbors[level] {
                    let d = compute_distance(self.metric, query, &self.vectors[nb].vector);
                    if d < best_dist {
                        best_dist = d;
                        current = nb;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    fn search_level(
        &self,
        entry: usize,
        query: &[f32],
        ef: usize,
        level: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited = vec![false; self.vectors.len()];
        visited[entry] = true;
        let entry_dist = compute_distance(self.metric, query, &self.vectors[entry].vector);

        let mut candidates: Vec<(usize, f32)> = vec![(entry, entry_dist)];
        let mut results: Vec<(usize, f32)> = vec![(entry, entry_dist)];

        while let Some(pos) = candidates
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let (ci, (c_idx, c_dist)) = (pos.0, *pos.1);
            let worst_result = results
                .iter()
                .map(|r| r.1)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(f32::MAX);

            if c_dist > worst_result && results.len() >= ef {
                break;
            }
            candidates.remove(ci);

            if level < self.nodes[c_idx].neighbors.len() {
                for &nb in &self.nodes[c_idx].neighbors[level] {
                    if visited[nb] {
                        continue;
                    }
                    visited[nb] = true;
                    let d = compute_distance(self.metric, query, &self.vectors[nb].vector);
                    let worst = results
                        .iter()
                        .map(|r| r.1)
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(f32::MAX);

                    if d < worst || results.len() < ef {
                        candidates.push((nb, d));
                        results.push((nb, d));
                        if results.len() > ef {
                            results.sort_by(|a, b| {
                                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            results.truncate(ef);
                        }
                    }
                }
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    fn select_neighbors(candidates: &[(usize, f32)], max_n: usize) -> Vec<usize> {
        candidates.iter().take(max_n).map(|&(idx, _)| idx).collect()
    }

    fn prune_neighbors(&mut self, node: usize, level: usize) {
        let query = &self.vectors[node].vector.clone();
        let neighbors = &self.nodes[node].neighbors[level];
        let mut scored: Vec<(usize, f32)> = neighbors
            .iter()
            .map(|&nb| (nb, compute_distance(self.metric, query, &self.vectors[nb].vector)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.max_neighbors);
        self.nodes[node].neighbors[level] = scored.into_iter().map(|(idx, _)| idx).collect();
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, SearchError> {
        if query.len() != self.dim {
            return Err(SearchError::DimensionMismatch { expected: self.dim, got: query.len() });
        }
        let entry = match self.entry_point {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };

        let mut current = entry;
        for lev in (1..=self.max_level).rev() {
            current = self.greedy_closest(current, query, lev);
        }

        let results = self.search_level(current, query, self.ef_search.max(k), 0);

        Ok(results
            .into_iter()
            .take(k)
            .map(|(idx, dist)| {
                let ev = &self.vectors[idx];
                SearchResult {
                    id: ev.id.clone(),
                    score: distance_to_score(self.metric, dist),
                    distance: dist,
                    metadata: ev.metadata.clone(),
                }
            })
            .collect())
    }

    pub fn delete(&mut self, id: &str) -> bool {
        if let Some(idx) = self.vectors.iter().position(|v| v.id == id) {
            // Remove references from neighbors.
            for n in &self.nodes[idx].neighbors.clone() {
                for &nb in n {
                    for level_neighbors in &mut self.nodes[nb].neighbors {
                        level_neighbors.retain(|&x| x != idx);
                    }
                }
            }
            self.nodes[idx].neighbors = vec![Vec::new(); self.nodes[idx].level + 1];
            // Mark as deleted by clearing vector (tombstone).
            self.vectors[idx].vector = vec![0.0; self.dim];
            self.vectors[idx].id = String::new();
            if self.entry_point == Some(idx) {
                self.entry_point = self.nodes.iter().enumerate().find_map(|(i, n)| {
                    if !self.vectors[i].id.is_empty() && n.level == self.max_level {
                        Some(i)
                    } else {
                        None
                    }
                });
            }
            true
        } else {
            false
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.iter().filter(|v| !v.id.is_empty()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stats(&self) -> IndexStats {
        let active = self.len();
        IndexStats {
            num_vectors: active,
            dimensions: self.dim,
            memory_bytes: self.vectors.len() * self.dim * size_of::<f32>()
                + self
                    .nodes
                    .iter()
                    .map(|n| n.neighbors.iter().map(|l| l.len() * 8).sum::<usize>())
                    .sum::<usize>(),
            index_type: IndexType::HNSW,
        }
    }
}

fn hash_id(id: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in id.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ── IVFIndex ─────────────────────────────────────────────────────────

/// Inverted-file index with k-means clustering.
#[derive(Debug, Clone)]
pub struct IVFIndex {
    centroids: Vec<Vec<f32>>,
    buckets: Vec<Vec<usize>>,
    vectors: Vec<EmbeddingVector>,
    dim: usize,
    metric: DistanceMetric,
    num_clusters: usize,
    num_probes: usize,
    trained: bool,
}

impl IVFIndex {
    pub fn new(dim: usize, metric: DistanceMetric, num_clusters: usize, num_probes: usize) -> Self {
        Self {
            centroids: Vec::new(),
            buckets: Vec::new(),
            vectors: Vec::new(),
            dim,
            metric,
            num_clusters,
            num_probes: num_probes.min(num_clusters),
            trained: false,
        }
    }

    pub fn insert(&mut self, vec: EmbeddingVector) -> Result<(), SearchError> {
        if vec.vector.is_empty() {
            return Err(SearchError::EmptyVector);
        }
        if vec.vector.len() != self.dim {
            return Err(SearchError::DimensionMismatch {
                expected: self.dim,
                got: vec.vector.len(),
            });
        }
        let idx = self.vectors.len();
        self.vectors.push(vec);

        if self.trained {
            let bucket = self.nearest_centroid(&self.vectors[idx].vector);
            self.buckets[bucket].push(idx);
        }
        Ok(())
    }

    /// Train the IVF centroids via simplified k-means.
    pub fn train(&mut self, max_iterations: usize) {
        if self.vectors.is_empty() {
            return;
        }
        let k = self.num_clusters.min(self.vectors.len());
        // Initialize centroids from evenly-spaced vectors.
        self.centroids = (0..k)
            .map(|i| {
                let idx = i * self.vectors.len() / k;
                self.vectors[idx].vector.clone()
            })
            .collect();
        self.buckets = vec![Vec::new(); k];

        for _ in 0..max_iterations {
            // Assign.
            let mut assignments = vec![Vec::new(); k];
            for (vi, v) in self.vectors.iter().enumerate() {
                let best = self.nearest_centroid(&v.vector);
                assignments[best].push(vi);
            }
            // Update centroids.
            let mut changed = false;
            for (ci, cluster) in assignments.iter().enumerate() {
                if cluster.is_empty() {
                    continue;
                }
                let mut new_centroid = vec![0.0f32; self.dim];
                for &vi in cluster {
                    for (j, val) in self.vectors[vi].vector.iter().enumerate() {
                        new_centroid[j] += val;
                    }
                }
                let len = cluster.len() as f32;
                for v in &mut new_centroid {
                    *v /= len;
                }
                if new_centroid != self.centroids[ci] {
                    changed = true;
                }
                self.centroids[ci] = new_centroid;
            }
            self.buckets = assignments;
            if !changed {
                break;
            }
        }
        self.trained = true;
    }

    fn nearest_centroid(&self, query: &[f32]) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, compute_distance(self.metric, query, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn nearest_centroids(&self, query: &[f32], n: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, compute_distance(self.metric, query, c)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(n).map(|(i, _)| i).collect()
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, SearchError> {
        if query.len() != self.dim {
            return Err(SearchError::DimensionMismatch { expected: self.dim, got: query.len() });
        }
        if !self.trained || self.vectors.is_empty() {
            return Ok(Vec::new());
        }
        let probes = self.nearest_centroids(query, self.num_probes);
        let mut candidates: Vec<(f32, usize)> = Vec::new();
        for bucket_id in probes {
            for &vi in &self.buckets[bucket_id] {
                if self.vectors[vi].id.is_empty() {
                    continue;
                }
                let d = compute_distance(self.metric, query, &self.vectors[vi].vector);
                candidates.push((d, vi));
            }
        }
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(candidates
            .into_iter()
            .take(k)
            .map(|(dist, vi)| {
                let ev = &self.vectors[vi];
                SearchResult {
                    id: ev.id.clone(),
                    score: distance_to_score(self.metric, dist),
                    distance: dist,
                    metadata: ev.metadata.clone(),
                }
            })
            .collect())
    }

    pub fn delete(&mut self, id: &str) -> bool {
        if let Some(idx) = self.vectors.iter().position(|v| v.id == id) {
            self.vectors[idx].id = String::new();
            self.vectors[idx].vector = vec![0.0; self.dim];
            for bucket in &mut self.buckets {
                bucket.retain(|&vi| vi != idx);
            }
            true
        } else {
            false
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.iter().filter(|v| !v.id.is_empty()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stats(&self) -> IndexStats {
        IndexStats {
            num_vectors: self.len(),
            dimensions: self.dim,
            memory_bytes: self.vectors.len() * self.dim * size_of::<f32>()
                + self.centroids.len() * self.dim * size_of::<f32>(),
            index_type: IndexType::IVF,
        }
    }
}

// ── SemanticSearchEngine ─────────────────────────────────────────────

enum IndexBackend {
    Flat(FlatIndex),
    Hnsw(HNSWIndex),
    Ivf(IVFIndex),
}

impl std::fmt::Debug for IndexBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Flat(_) => f.write_str("IndexBackend::Flat(..)"),
            Self::Hnsw(_) => f.write_str("IndexBackend::Hnsw(..)"),
            Self::Ivf(_) => f.write_str("IndexBackend::Ivf(..)"),
        }
    }
}

/// Unified semantic search engine.
#[derive(Debug)]
pub struct SemanticSearchEngine {
    config: SearchConfig,
    backend: IndexBackend,
}

impl SemanticSearchEngine {
    pub fn new(config: SearchConfig) -> Result<Self, SearchError> {
        if config.embedding_dim == 0 {
            return Err(SearchError::InvalidConfig("embedding_dim must be > 0".into()));
        }
        if config.max_results == 0 {
            return Err(SearchError::InvalidConfig("max_results must be > 0".into()));
        }
        if config.index_type == IndexType::IVF && config.num_clusters == 0 {
            return Err(SearchError::InvalidConfig("num_clusters must be > 0 for IVF".into()));
        }
        let backend = match config.index_type {
            IndexType::Flat => {
                IndexBackend::Flat(FlatIndex::new(config.embedding_dim, config.distance_metric))
            }
            IndexType::HNSW => IndexBackend::Hnsw(HNSWIndex::new(
                config.embedding_dim,
                config.distance_metric,
                config.ef_construction,
                config.ef_search,
            )),
            IndexType::IVF => IndexBackend::Ivf(IVFIndex::new(
                config.embedding_dim,
                config.distance_metric,
                config.num_clusters,
                config.num_probes,
            )),
        };
        Ok(Self { config, backend })
    }

    pub fn insert(&mut self, vec: EmbeddingVector) -> Result<(), SearchError> {
        match &mut self.backend {
            IndexBackend::Flat(idx) => idx.insert(vec),
            IndexBackend::Hnsw(idx) => idx.insert(vec),
            IndexBackend::Ivf(idx) => idx.insert(vec),
        }
    }

    pub fn batch_insert(&mut self, vecs: Vec<EmbeddingVector>) -> Result<usize, SearchError> {
        let mut count = 0;
        for v in vecs {
            self.insert(v)?;
            count += 1;
        }
        Ok(count)
    }

    pub fn search(
        &self,
        query: &[f32],
        k: Option<usize>,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let k = k.unwrap_or(self.config.max_results);
        match &self.backend {
            IndexBackend::Flat(idx) => idx.search(query, k),
            IndexBackend::Hnsw(idx) => idx.search(query, k),
            IndexBackend::Ivf(idx) => idx.search(query, k),
        }
    }

    pub fn search_with_filter<F>(
        &self,
        query: &[f32],
        k: Option<usize>,
        filter: F,
    ) -> Result<Vec<SearchResult>, SearchError>
    where
        F: Fn(&HashMap<String, String>) -> bool,
    {
        let all = self.search(query, Some(self.stats().num_vectors.max(1)))?;
        let k = k.unwrap_or(self.config.max_results);
        Ok(all.into_iter().filter(|r| filter(&r.metadata)).take(k).collect())
    }

    pub fn delete(&mut self, id: &str) -> bool {
        match &mut self.backend {
            IndexBackend::Flat(idx) => idx.delete(id),
            IndexBackend::Hnsw(idx) => idx.delete(id),
            IndexBackend::Ivf(idx) => idx.delete(id),
        }
    }

    pub fn rebuild_index(&mut self) {
        if let IndexBackend::Ivf(idx) = &mut self.backend {
            idx.train(20);
        }
    }

    pub fn len(&self) -> usize {
        match &self.backend {
            IndexBackend::Flat(idx) => idx.len(),
            IndexBackend::Hnsw(idx) => idx.len(),
            IndexBackend::Ivf(idx) => idx.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stats(&self) -> IndexStats {
        match &self.backend {
            IndexBackend::Flat(idx) => idx.stats(),
            IndexBackend::Hnsw(idx) => idx.stats(),
            IndexBackend::Ivf(idx) => idx.stats(),
        }
    }

    pub fn config(&self) -> &SearchConfig {
        &self.config
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ──────────────────────────────────────────────────────────

    fn make_vec(id: &str, v: Vec<f32>) -> EmbeddingVector {
        EmbeddingVector { id: id.to_string(), vector: v, metadata: HashMap::new() }
    }

    fn make_vec_meta(id: &str, v: Vec<f32>, meta: Vec<(&str, &str)>) -> EmbeddingVector {
        let metadata = meta.into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        EmbeddingVector { id: id.to_string(), vector: v, metadata }
    }

    fn unit_vec(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[axis] = 1.0;
        v
    }

    fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..dim)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn default_engine(index_type: IndexType) -> SemanticSearchEngine {
        let config =
            SearchConfig { embedding_dim: 4, index_type, max_results: 10, ..Default::default() };
        SemanticSearchEngine::new(config).unwrap()
    }

    // ── Distance metric correctness ──────────────────────────────────

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let d = cosine_distance(&v, &v);
        assert!(d.abs() < 1e-6, "identical vectors should have cosine distance ~0, got {d}");
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let d = cosine_distance(&a, &b);
        assert!(
            (d - 1.0).abs() < 1e-6,
            "orthogonal vectors should have cosine distance ~1, got {d}"
        );
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let d = cosine_distance(&a, &b);
        assert!((d - 2.0).abs() < 1e-6, "opposite vectors should have cosine distance ~2, got {d}");
    }

    #[test]
    fn cosine_zero_vector_returns_one() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        assert_eq!(cosine_distance(&a, &b), 1.0);
    }

    #[test]
    fn euclidean_same_point() {
        let v = vec![3.0, 4.0];
        assert!(euclidean_distance(&v, &v).abs() < 1e-6);
    }

    #[test]
    fn euclidean_known_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = euclidean_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-6);
    }

    #[test]
    fn dot_product_known() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn dot_product_distance_sign() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        // dot_product_distance negates dot, so similar = negative distance
        assert!(dot_product_distance(&a, &b) < 0.0);
    }

    #[test]
    fn manhattan_known() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 3.0];
        let d = manhattan_distance(&a, &b);
        assert!((d - 7.0).abs() < 1e-6);
    }

    #[test]
    fn manhattan_same_point() {
        let v = vec![1.0, 2.0];
        assert!(manhattan_distance(&v, &v).abs() < 1e-6);
    }

    // ── Distance metric properties ───────────────────────────────────

    #[test]
    fn euclidean_symmetry() {
        let a = random_vec(8, 1);
        let b = random_vec(8, 2);
        assert!((euclidean_distance(&a, &b) - euclidean_distance(&b, &a)).abs() < 1e-5);
    }

    #[test]
    fn manhattan_symmetry() {
        let a = random_vec(8, 3);
        let b = random_vec(8, 4);
        assert!((manhattan_distance(&a, &b) - manhattan_distance(&b, &a)).abs() < 1e-5);
    }

    #[test]
    fn cosine_symmetry() {
        let a = random_vec(8, 5);
        let b = random_vec(8, 6);
        assert!((cosine_distance(&a, &b) - cosine_distance(&b, &a)).abs() < 1e-5);
    }

    #[test]
    fn euclidean_triangle_inequality() {
        let a = random_vec(8, 10);
        let b = random_vec(8, 11);
        let c = random_vec(8, 12);
        let ab = euclidean_distance(&a, &b);
        let bc = euclidean_distance(&b, &c);
        let ac = euclidean_distance(&a, &c);
        assert!(ac <= ab + bc + 1e-5, "triangle inequality violated: {ac} > {ab} + {bc}");
    }

    #[test]
    fn manhattan_triangle_inequality() {
        let a = random_vec(8, 20);
        let b = random_vec(8, 21);
        let c = random_vec(8, 22);
        let ab = manhattan_distance(&a, &b);
        let bc = manhattan_distance(&b, &c);
        let ac = manhattan_distance(&a, &c);
        assert!(ac <= ab + bc + 1e-5);
    }

    #[test]
    fn euclidean_non_negative() {
        let a = random_vec(8, 30);
        let b = random_vec(8, 31);
        assert!(euclidean_distance(&a, &b) >= 0.0);
    }

    #[test]
    fn manhattan_non_negative() {
        let a = random_vec(8, 32);
        let b = random_vec(8, 33);
        assert!(manhattan_distance(&a, &b) >= 0.0);
    }

    #[test]
    fn cosine_bounded() {
        let a = random_vec(8, 40);
        let b = random_vec(8, 41);
        let d = cosine_distance(&a, &b);
        assert!((-1e-6..=2.0 + 1e-6).contains(&d), "cosine distance out of [0,2]: {d}");
    }

    // ── Score conversion ─────────────────────────────────────────────

    #[test]
    fn cosine_score_identical() {
        let d = 0.0; // identical
        let s = distance_to_score(DistanceMetric::Cosine, d);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_score_zero_distance() {
        let s = distance_to_score(DistanceMetric::Euclidean, 0.0);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_score_decreases_with_distance() {
        let s1 = distance_to_score(DistanceMetric::Euclidean, 1.0);
        let s2 = distance_to_score(DistanceMetric::Euclidean, 10.0);
        assert!(s1 > s2);
    }

    // ── FlatIndex ────────────────────────────────────────────────────

    #[test]
    fn flat_insert_and_search() {
        let mut idx = FlatIndex::new(3, DistanceMetric::Euclidean);
        idx.insert(make_vec("a", vec![1.0, 0.0, 0.0])).unwrap();
        idx.insert(make_vec("b", vec![0.0, 1.0, 0.0])).unwrap();
        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn flat_exact_results_sorted_by_distance() {
        let mut idx = FlatIndex::new(2, DistanceMetric::Euclidean);
        idx.insert(make_vec("far", vec![10.0, 10.0])).unwrap();
        idx.insert(make_vec("mid", vec![2.0, 0.0])).unwrap();
        idx.insert(make_vec("close", vec![0.1, 0.0])).unwrap();
        let results = idx.search(&[0.0, 0.0], 3).unwrap();
        assert_eq!(results[0].id, "close");
        assert_eq!(results[1].id, "mid");
        assert_eq!(results[2].id, "far");
    }

    #[test]
    fn flat_dimension_mismatch_insert() {
        let mut idx = FlatIndex::new(3, DistanceMetric::Euclidean);
        let err = idx.insert(make_vec("x", vec![1.0, 2.0])).unwrap_err();
        assert_eq!(err, SearchError::DimensionMismatch { expected: 3, got: 2 });
    }

    #[test]
    fn flat_dimension_mismatch_search() {
        let idx = FlatIndex::new(3, DistanceMetric::Euclidean);
        let err = idx.search(&[1.0], 1).unwrap_err();
        assert_eq!(err, SearchError::DimensionMismatch { expected: 3, got: 1 });
    }

    #[test]
    fn flat_empty_vector_rejected() {
        let mut idx = FlatIndex::new(3, DistanceMetric::Euclidean);
        assert_eq!(idx.insert(make_vec("x", vec![])).unwrap_err(), SearchError::EmptyVector);
    }

    #[test]
    fn flat_search_empty_index() {
        let idx = FlatIndex::new(3, DistanceMetric::Euclidean);
        let results = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn flat_delete() {
        let mut idx = FlatIndex::new(2, DistanceMetric::Euclidean);
        idx.insert(make_vec("a", vec![1.0, 0.0])).unwrap();
        idx.insert(make_vec("b", vec![0.0, 1.0])).unwrap();
        assert_eq!(idx.len(), 2);
        assert!(idx.delete("a"));
        assert_eq!(idx.len(), 1);
        assert!(!idx.delete("a")); // already deleted
    }

    #[test]
    fn flat_delete_nonexistent() {
        let mut idx = FlatIndex::new(2, DistanceMetric::Euclidean);
        assert!(!idx.delete("missing"));
    }

    #[test]
    fn flat_search_after_delete() {
        let mut idx = FlatIndex::new(2, DistanceMetric::Cosine);
        idx.insert(make_vec("a", vec![1.0, 0.0])).unwrap();
        idx.insert(make_vec("b", vec![0.0, 1.0])).unwrap();
        idx.delete("a");
        let results = idx.search(&[1.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn flat_cosine_search() {
        let mut idx = FlatIndex::new(3, DistanceMetric::Cosine);
        idx.insert(make_vec("same_dir", vec![2.0, 0.0, 0.0])).unwrap();
        idx.insert(make_vec("orthogonal", vec![0.0, 1.0, 0.0])).unwrap();
        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, "same_dir");
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn flat_dot_product_search() {
        let mut idx = FlatIndex::new(2, DistanceMetric::DotProduct);
        idx.insert(make_vec("big", vec![10.0, 10.0])).unwrap();
        idx.insert(make_vec("small", vec![0.1, 0.1])).unwrap();
        let results = idx.search(&[1.0, 1.0], 1).unwrap();
        assert_eq!(results[0].id, "big");
    }

    #[test]
    fn flat_manhattan_search() {
        let mut idx = FlatIndex::new(2, DistanceMetric::Manhattan);
        idx.insert(make_vec("close", vec![0.5, 0.5])).unwrap();
        idx.insert(make_vec("far", vec![10.0, 10.0])).unwrap();
        let results = idx.search(&[0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, "close");
    }

    #[test]
    fn flat_top_k_limited() {
        let mut idx = FlatIndex::new(2, DistanceMetric::Euclidean);
        for i in 0..20 {
            idx.insert(make_vec(&format!("v{i}"), vec![i as f32, 0.0])).unwrap();
        }
        let results = idx.search(&[0.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn flat_single_vector() {
        let mut idx = FlatIndex::new(2, DistanceMetric::Euclidean);
        idx.insert(make_vec("only", vec![1.0, 1.0])).unwrap();
        let results = idx.search(&[0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "only");
    }

    #[test]
    fn flat_stats() {
        let mut idx = FlatIndex::new(4, DistanceMetric::Euclidean);
        idx.insert(make_vec("a", vec![1.0; 4])).unwrap();
        idx.insert(make_vec("b", vec![2.0; 4])).unwrap();
        let s = idx.stats();
        assert_eq!(s.num_vectors, 2);
        assert_eq!(s.dimensions, 4);
        assert_eq!(s.index_type, IndexType::Flat);
        assert!(s.memory_bytes > 0);
    }

    #[test]
    fn flat_is_empty() {
        let idx = FlatIndex::new(2, DistanceMetric::Euclidean);
        assert!(idx.is_empty());
    }

    // ── HNSWIndex ────────────────────────────────────────────────────

    #[test]
    fn hnsw_insert_and_search() {
        let mut idx = HNSWIndex::new(3, DistanceMetric::Euclidean, 32, 32);
        idx.insert(make_vec("a", vec![1.0, 0.0, 0.0])).unwrap();
        idx.insert(make_vec("b", vec![0.0, 1.0, 0.0])).unwrap();
        idx.insert(make_vec("c", vec![0.0, 0.0, 1.0])).unwrap();
        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn hnsw_recall_small_dataset() {
        let dim = 8;
        let n = 50;
        let mut idx = HNSWIndex::new(dim, DistanceMetric::Euclidean, 64, 64);
        let mut flat = FlatIndex::new(dim, DistanceMetric::Euclidean);
        for i in 0..n {
            let v = random_vec(dim, i as u64 + 100);
            idx.insert(make_vec(&format!("v{i}"), v.clone())).unwrap();
            flat.insert(make_vec(&format!("v{i}"), v)).unwrap();
        }
        // Measure recall@10 for 10 random queries.
        let mut total_recall = 0.0;
        let queries = 10;
        let k = 10;
        for q in 0..queries {
            let query = random_vec(dim, q as u64 + 1000);
            let exact = flat.search(&query, k).unwrap();
            let approx = idx.search(&query, k).unwrap();
            let exact_ids: std::collections::HashSet<_> = exact.iter().map(|r| &r.id).collect();
            let hits = approx.iter().filter(|r| exact_ids.contains(&r.id)).count();
            total_recall += hits as f64 / k as f64;
        }
        let avg_recall = total_recall / queries as f64;
        assert!(avg_recall > 0.7, "HNSW recall@10 = {avg_recall:.2}, expected > 0.7");
    }

    #[test]
    fn hnsw_empty_search() {
        let idx = HNSWIndex::new(3, DistanceMetric::Euclidean, 32, 32);
        let results = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn hnsw_dimension_mismatch() {
        let mut idx = HNSWIndex::new(3, DistanceMetric::Euclidean, 32, 32);
        let err = idx.insert(make_vec("x", vec![1.0])).unwrap_err();
        assert_eq!(err, SearchError::DimensionMismatch { expected: 3, got: 1 });
    }

    #[test]
    fn hnsw_empty_vector() {
        let mut idx = HNSWIndex::new(3, DistanceMetric::Euclidean, 32, 32);
        assert_eq!(idx.insert(make_vec("x", vec![])).unwrap_err(), SearchError::EmptyVector);
    }

    #[test]
    fn hnsw_delete() {
        let mut idx = HNSWIndex::new(2, DistanceMetric::Euclidean, 32, 32);
        idx.insert(make_vec("a", vec![1.0, 0.0])).unwrap();
        idx.insert(make_vec("b", vec![0.0, 1.0])).unwrap();
        assert_eq!(idx.len(), 2);
        assert!(idx.delete("a"));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn hnsw_delete_nonexistent() {
        let mut idx = HNSWIndex::new(2, DistanceMetric::Euclidean, 32, 32);
        assert!(!idx.delete("missing"));
    }

    #[test]
    fn hnsw_cosine_search() {
        let mut idx = HNSWIndex::new(3, DistanceMetric::Cosine, 32, 32);
        idx.insert(make_vec("aligned", vec![3.0, 0.0, 0.0])).unwrap();
        idx.insert(make_vec("perp", vec![0.0, 5.0, 0.0])).unwrap();
        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, "aligned");
    }

    #[test]
    fn hnsw_stats() {
        let mut idx = HNSWIndex::new(4, DistanceMetric::Euclidean, 32, 32);
        idx.insert(make_vec("a", vec![1.0; 4])).unwrap();
        let s = idx.stats();
        assert_eq!(s.num_vectors, 1);
        assert_eq!(s.index_type, IndexType::HNSW);
    }

    #[test]
    fn hnsw_is_empty() {
        let idx = HNSWIndex::new(2, DistanceMetric::Euclidean, 32, 32);
        assert!(idx.is_empty());
    }

    #[test]
    fn hnsw_single_vector_search() {
        let mut idx = HNSWIndex::new(2, DistanceMetric::Euclidean, 32, 32);
        idx.insert(make_vec("only", vec![5.0, 5.0])).unwrap();
        let results = idx.search(&[0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "only");
    }

    #[test]
    fn hnsw_many_inserts() {
        let mut idx = HNSWIndex::new(4, DistanceMetric::Euclidean, 32, 32);
        for i in 0..100 {
            idx.insert(make_vec(&format!("v{i}"), random_vec(4, i))).unwrap();
        }
        assert_eq!(idx.len(), 100);
        let results = idx.search(&random_vec(4, 999), 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn hnsw_search_dimension_mismatch() {
        let idx = HNSWIndex::new(4, DistanceMetric::Euclidean, 32, 32);
        let err = idx.search(&[1.0], 1).unwrap_err();
        assert_eq!(err, SearchError::DimensionMismatch { expected: 4, got: 1 });
    }

    // ── IVFIndex ─────────────────────────────────────────────────────

    #[test]
    fn ivf_insert_train_search() {
        let mut idx = IVFIndex::new(3, DistanceMetric::Euclidean, 2, 2);
        idx.insert(make_vec("a", vec![1.0, 0.0, 0.0])).unwrap();
        idx.insert(make_vec("b", vec![0.0, 1.0, 0.0])).unwrap();
        idx.insert(make_vec("c", vec![0.0, 0.0, 1.0])).unwrap();
        idx.insert(make_vec("d", vec![1.0, 1.0, 0.0])).unwrap();
        idx.train(10);
        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn ivf_search_untrained() {
        let idx = IVFIndex::new(3, DistanceMetric::Euclidean, 2, 1);
        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn ivf_dimension_mismatch() {
        let mut idx = IVFIndex::new(3, DistanceMetric::Euclidean, 2, 1);
        let err = idx.insert(make_vec("x", vec![1.0])).unwrap_err();
        assert_eq!(err, SearchError::DimensionMismatch { expected: 3, got: 1 });
    }

    #[test]
    fn ivf_empty_vector() {
        let mut idx = IVFIndex::new(3, DistanceMetric::Euclidean, 2, 1);
        assert_eq!(idx.insert(make_vec("x", vec![])).unwrap_err(), SearchError::EmptyVector);
    }

    #[test]
    fn ivf_delete() {
        let mut idx = IVFIndex::new(2, DistanceMetric::Euclidean, 2, 2);
        idx.insert(make_vec("a", vec![1.0, 0.0])).unwrap();
        idx.insert(make_vec("b", vec![0.0, 1.0])).unwrap();
        idx.train(5);
        assert!(idx.delete("a"));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn ivf_delete_nonexistent() {
        let mut idx = IVFIndex::new(2, DistanceMetric::Euclidean, 2, 1);
        assert!(!idx.delete("missing"));
    }

    #[test]
    fn ivf_stats() {
        let mut idx = IVFIndex::new(4, DistanceMetric::Euclidean, 2, 1);
        idx.insert(make_vec("a", vec![1.0; 4])).unwrap();
        let s = idx.stats();
        assert_eq!(s.num_vectors, 1);
        assert_eq!(s.index_type, IndexType::IVF);
    }

    #[test]
    fn ivf_train_empty() {
        let mut idx = IVFIndex::new(3, DistanceMetric::Euclidean, 2, 1);
        idx.train(10); // should not panic
        assert!(!idx.trained);
    }

    #[test]
    fn ivf_cosine_search() {
        let mut idx = IVFIndex::new(3, DistanceMetric::Cosine, 2, 2);
        idx.insert(make_vec("aligned", vec![5.0, 0.0, 0.0])).unwrap();
        idx.insert(make_vec("perp", vec![0.0, 5.0, 0.0])).unwrap();
        idx.insert(make_vec("diag", vec![1.0, 1.0, 0.0])).unwrap();
        idx.train(10);
        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, "aligned");
    }

    #[test]
    fn ivf_is_empty() {
        let idx = IVFIndex::new(2, DistanceMetric::Euclidean, 2, 1);
        assert!(idx.is_empty());
    }

    #[test]
    fn ivf_search_after_delete() {
        let mut idx = IVFIndex::new(2, DistanceMetric::Euclidean, 2, 2);
        idx.insert(make_vec("a", vec![1.0, 0.0])).unwrap();
        idx.insert(make_vec("b", vec![0.0, 1.0])).unwrap();
        idx.insert(make_vec("c", vec![0.5, 0.5])).unwrap();
        idx.train(10);
        idx.delete("a");
        let results = idx.search(&[1.0, 0.0], 5).unwrap();
        assert!(results.iter().all(|r| r.id != "a"));
    }

    // ── SemanticSearchEngine ─────────────────────────────────────────

    #[test]
    fn engine_flat_insert_search() {
        let mut engine = default_engine(IndexType::Flat);
        engine.insert(make_vec("a", vec![1.0, 0.0, 0.0, 0.0])).unwrap();
        engine.insert(make_vec("b", vec![0.0, 1.0, 0.0, 0.0])).unwrap();
        let results = engine.search(&[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn engine_hnsw_insert_search() {
        let mut engine = default_engine(IndexType::HNSW);
        engine.insert(make_vec("a", vec![1.0, 0.0, 0.0, 0.0])).unwrap();
        engine.insert(make_vec("b", vec![0.0, 1.0, 0.0, 0.0])).unwrap();
        let results = engine.search(&[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn engine_ivf_insert_train_search() {
        let config = SearchConfig {
            embedding_dim: 4,
            index_type: IndexType::IVF,
            num_clusters: 2,
            num_probes: 2,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config).unwrap();
        engine.insert(make_vec("a", vec![1.0, 0.0, 0.0, 0.0])).unwrap();
        engine.insert(make_vec("b", vec![0.0, 1.0, 0.0, 0.0])).unwrap();
        engine.insert(make_vec("c", vec![0.0, 0.0, 1.0, 0.0])).unwrap();
        engine.rebuild_index();
        let results = engine.search(&[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn engine_batch_insert() {
        let mut engine = default_engine(IndexType::Flat);
        let vecs = vec![
            make_vec("a", vec![1.0, 0.0, 0.0, 0.0]),
            make_vec("b", vec![0.0, 1.0, 0.0, 0.0]),
            make_vec("c", vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let count = engine.batch_insert(vecs).unwrap();
        assert_eq!(count, 3);
        assert_eq!(engine.len(), 3);
    }

    #[test]
    fn engine_batch_insert_fails_on_bad_dim() {
        let mut engine = default_engine(IndexType::Flat);
        let vecs = vec![make_vec("a", vec![1.0, 0.0, 0.0, 0.0]), make_vec("bad", vec![1.0, 0.0])];
        let err = engine.batch_insert(vecs).unwrap_err();
        assert_eq!(err, SearchError::DimensionMismatch { expected: 4, got: 2 });
    }

    #[test]
    fn engine_delete() {
        let mut engine = default_engine(IndexType::Flat);
        engine.insert(make_vec("a", vec![1.0, 0.0, 0.0, 0.0])).unwrap();
        assert!(engine.delete("a"));
        assert!(engine.is_empty());
    }

    #[test]
    fn engine_search_with_k() {
        let mut engine = default_engine(IndexType::Flat);
        for i in 0..10 {
            engine.insert(make_vec(&format!("v{i}"), unit_vec(4, i % 4))).unwrap();
        }
        let results = engine.search(&[1.0, 0.0, 0.0, 0.0], Some(3)).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn engine_stats() {
        let mut engine = default_engine(IndexType::Flat);
        engine.insert(make_vec("a", vec![1.0; 4])).unwrap();
        let s = engine.stats();
        assert_eq!(s.num_vectors, 1);
        assert_eq!(s.dimensions, 4);
    }

    #[test]
    fn engine_config() {
        let engine = default_engine(IndexType::Flat);
        assert_eq!(engine.config().embedding_dim, 4);
        assert_eq!(engine.config().index_type, IndexType::Flat);
    }

    #[test]
    fn engine_is_empty() {
        let engine = default_engine(IndexType::Flat);
        assert!(engine.is_empty());
        assert_eq!(engine.len(), 0);
    }

    // ── Config validation ────────────────────────────────────────────

    #[test]
    fn config_zero_dim_rejected() {
        let config = SearchConfig { embedding_dim: 0, ..Default::default() };
        let err = SemanticSearchEngine::new(config).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig(_)));
    }

    #[test]
    fn config_zero_max_results_rejected() {
        let config = SearchConfig { max_results: 0, ..Default::default() };
        let err = SemanticSearchEngine::new(config).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig(_)));
    }

    #[test]
    fn config_ivf_zero_clusters_rejected() {
        let config =
            SearchConfig { index_type: IndexType::IVF, num_clusters: 0, ..Default::default() };
        let err = SemanticSearchEngine::new(config).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig(_)));
    }

    #[test]
    fn config_default_is_valid() {
        let engine = SemanticSearchEngine::new(SearchConfig::default());
        assert!(engine.is_ok());
    }

    // ── Metadata filtering ───────────────────────────────────────────

    #[test]
    fn filter_by_metadata_key() {
        let mut engine = default_engine(IndexType::Flat);
        engine
            .insert(make_vec_meta("doc1", vec![1.0, 0.0, 0.0, 0.0], vec![("type", "article")]))
            .unwrap();
        engine
            .insert(make_vec_meta("doc2", vec![0.9, 0.1, 0.0, 0.0], vec![("type", "book")]))
            .unwrap();
        engine
            .insert(make_vec_meta("doc3", vec![0.8, 0.2, 0.0, 0.0], vec![("type", "article")]))
            .unwrap();
        let results = engine
            .search_with_filter(&[1.0, 0.0, 0.0, 0.0], Some(10), |m| {
                m.get("type").is_some_and(|t| t == "article")
            })
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.metadata.get("type").unwrap() == "article"));
    }

    #[test]
    fn filter_returns_empty_when_nothing_matches() {
        let mut engine = default_engine(IndexType::Flat);
        engine.insert(make_vec_meta("a", vec![1.0; 4], vec![("lang", "en")])).unwrap();
        let results = engine
            .search_with_filter(&[1.0; 4], None, |m| m.get("lang").is_some_and(|l| l == "fr"))
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn filter_respects_k_limit() {
        let mut engine = default_engine(IndexType::Flat);
        for i in 0..10 {
            engine
                .insert(make_vec_meta(
                    &format!("v{i}"),
                    vec![i as f32, 0.0, 0.0, 0.0],
                    vec![("ok", "yes")],
                ))
                .unwrap();
        }
        let results = engine
            .search_with_filter(&[5.0, 0.0, 0.0, 0.0], Some(3), |m| {
                m.get("ok").is_some_and(|v| v == "yes")
            })
            .unwrap();
        assert_eq!(results.len(), 3);
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn search_for_deleted_vector() {
        let mut engine = default_engine(IndexType::Flat);
        engine.insert(make_vec("a", vec![1.0, 0.0, 0.0, 0.0])).unwrap();
        engine.insert(make_vec("b", vec![0.0, 1.0, 0.0, 0.0])).unwrap();
        engine.delete("a");
        let results = engine.search(&[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert!(results.iter().all(|r| r.id != "a"));
    }

    #[test]
    fn top_k_exceeds_index_size() {
        let mut engine = default_engine(IndexType::Flat);
        engine.insert(make_vec("only", vec![1.0; 4])).unwrap();
        let results = engine.search(&[1.0; 4], Some(100)).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn duplicate_ids_allowed() {
        let mut engine = default_engine(IndexType::Flat);
        engine.insert(make_vec("dup", vec![1.0, 0.0, 0.0, 0.0])).unwrap();
        engine.insert(make_vec("dup", vec![0.0, 1.0, 0.0, 0.0])).unwrap();
        assert_eq!(engine.len(), 2);
    }

    #[test]
    fn high_dimensional_vectors() {
        let dim = 256;
        let config =
            SearchConfig { embedding_dim: dim, index_type: IndexType::Flat, ..Default::default() };
        let mut engine = SemanticSearchEngine::new(config).unwrap();
        engine.insert(make_vec("a", random_vec(dim, 1))).unwrap();
        engine.insert(make_vec("b", random_vec(dim, 2))).unwrap();
        let results = engine.search(&random_vec(dim, 1), Some(1)).unwrap();
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn rebuild_index_noop_for_flat() {
        let mut engine = default_engine(IndexType::Flat);
        engine.insert(make_vec("a", vec![1.0; 4])).unwrap();
        engine.rebuild_index(); // should not panic
        assert_eq!(engine.len(), 1);
    }

    #[test]
    fn rebuild_index_noop_for_hnsw() {
        let mut engine = default_engine(IndexType::HNSW);
        engine.insert(make_vec("a", vec![1.0; 4])).unwrap();
        engine.rebuild_index(); // should not panic
        assert_eq!(engine.len(), 1);
    }

    #[test]
    fn search_result_distances_ordered() {
        let mut engine = default_engine(IndexType::Flat);
        for i in 0..10 {
            engine.insert(make_vec(&format!("v{i}"), vec![i as f32, 0.0, 0.0, 0.0])).unwrap();
        }
        let results = engine.search(&[5.0, 0.0, 0.0, 0.0], Some(10)).unwrap();
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance + 1e-6);
        }
    }

    #[test]
    fn search_result_scores_ordered() {
        let mut engine = default_engine(IndexType::Flat);
        for i in 0..10 {
            engine.insert(make_vec(&format!("v{i}"), vec![i as f32, 0.0, 0.0, 0.0])).unwrap();
        }
        let results = engine.search(&[5.0, 0.0, 0.0, 0.0], Some(10)).unwrap();
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score - 1e-6);
        }
    }

    #[test]
    fn hash_id_deterministic() {
        assert_eq!(hash_id("hello"), hash_id("hello"));
        assert_ne!(hash_id("hello"), hash_id("world"));
    }

    #[test]
    fn hash_id_empty_string() {
        let _ = hash_id(""); // should not panic
    }

    // ── compute_distance dispatch ────────────────────────────────────

    #[test]
    fn compute_distance_cosine() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!(compute_distance(DistanceMetric::Cosine, &a, &b).abs() < 1e-6);
    }

    #[test]
    fn compute_distance_euclidean() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((compute_distance(DistanceMetric::Euclidean, &a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn compute_distance_dot_product() {
        let a = vec![2.0, 3.0];
        let b = vec![4.0, 5.0];
        assert!((compute_distance(DistanceMetric::DotProduct, &a, &b) - (-23.0)).abs() < 1e-6);
    }

    #[test]
    fn compute_distance_manhattan() {
        let a = vec![1.0, 2.0];
        let b = vec![4.0, 6.0];
        assert!((compute_distance(DistanceMetric::Manhattan, &a, &b) - 7.0).abs() < 1e-6);
    }

    // ── Error Display ────────────────────────────────────────────────

    #[test]
    fn error_display_dimension_mismatch() {
        let e = SearchError::DimensionMismatch { expected: 3, got: 5 };
        assert_eq!(e.to_string(), "dimension mismatch: expected 3, got 5");
    }

    #[test]
    fn error_display_empty_vector() {
        assert_eq!(SearchError::EmptyVector.to_string(), "empty vector");
    }

    #[test]
    fn error_display_invalid_config() {
        let e = SearchError::InvalidConfig("bad".into());
        assert_eq!(e.to_string(), "invalid config: bad");
    }

    #[test]
    fn error_display_not_found() {
        let e = SearchError::VectorNotFound("v1".into());
        assert_eq!(e.to_string(), "vector not found: v1");
    }

    #[test]
    fn error_display_empty_index() {
        assert_eq!(SearchError::EmptyIndex.to_string(), "index is empty");
    }
}
