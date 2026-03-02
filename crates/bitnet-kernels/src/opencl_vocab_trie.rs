//! Vocabulary trie for efficient token lookup and prefix search (OpenCL backend).
//!
//! Provides a trie data structure built from a token vocabulary, supporting:
//!
//! - **Exact match lookup**: find a token ID for a given string
//! - **Prefix search**: find all tokens sharing a common prefix
//! - **Longest-match tokenization**: greedy left-to-right tokenization
//! - **Compact trie**: flat array layout suitable for GPU buffer transfer
//! - **LRU prefix cache**: frequently searched prefixes cached for speed
//!
//! # A770-Specific Optimizations
//!
//! - 64-byte aligned node arrays for cache line efficiency
//! - Compact encoding for GPU buffer transfer (flat `u32` arrays)
//! - Batch prefix search for parallel workgroup execution
//!
//! # OpenCL kernel source
//!
//! An embedded OpenCL kernel performs parallel prefix search across a batch
//! of query strings using the compact trie representation.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

// ── Constants ─────────────────────────────────────────────────────

/// Cache-line alignment for GPU-friendly node arrays (A770 = 64 bytes).
pub const NODE_ALIGNMENT: usize = 64;

/// Sentinel value indicating "no token" in compact representation.
pub const NO_TOKEN: u32 = u32::MAX;

/// Sentinel value indicating "no child" in compact representation.
pub const NO_CHILD: u32 = u32::MAX;

/// Default LRU cache capacity.
pub const DEFAULT_CACHE_CAPACITY: usize = 1024;

// ── TrieNode ──────────────────────────────────────────────────────

/// A single node in the vocabulary trie.
#[derive(Debug, Clone)]
pub struct TrieNode {
    /// Children keyed by character.
    pub children: BTreeMap<char, TrieNode>,
    /// Token ID if this node terminates a valid token.
    pub token_id: Option<u32>,
    /// Number of tokens reachable from this node (including self).
    pub prefix_count: u32,
}

impl TrieNode {
    /// Create an empty node.
    pub fn new() -> Self {
        Self { children: BTreeMap::new(), token_id: None, prefix_count: 0 }
    }

    /// Returns `true` if this node has no children.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Returns the depth of the deepest path from this node.
    pub fn max_depth(&self) -> usize {
        if self.children.is_empty() {
            0
        } else {
            1 + self.children.values().map(|c| c.max_depth()).max().unwrap_or(0)
        }
    }

    /// Count total nodes in this subtree (including self).
    pub fn node_count(&self) -> usize {
        1 + self.children.values().map(|c| c.node_count()).sum::<usize>()
    }
}

impl Default for TrieNode {
    fn default() -> Self {
        Self::new()
    }
}

// ── TrieSearchResult ──────────────────────────────────────────────

/// Result of a trie search operation.
#[derive(Debug, Clone, PartialEq)]
pub struct TrieSearchResult {
    /// Token ID of exact match, if any.
    pub exact_match: Option<u32>,
    /// Number of tokens that share the query as a prefix.
    pub prefix_match_count: u32,
    /// Completions: list of (suffix, token_id) pairs.
    pub completions: Vec<(String, u32)>,
}

impl TrieSearchResult {
    /// Create an empty result (nothing found).
    pub fn empty() -> Self {
        Self { exact_match: None, prefix_match_count: 0, completions: Vec::new() }
    }

    /// Returns `true` if neither exact nor prefix matches were found.
    pub fn is_empty(&self) -> bool {
        self.exact_match.is_none() && self.prefix_match_count == 0
    }
}

// ── VocabTrie ─────────────────────────────────────────────────────

/// Trie built from a token vocabulary for efficient lookup and prefix search.
#[derive(Debug, Clone)]
pub struct VocabTrie {
    root: TrieNode,
    /// Total number of tokens inserted.
    token_count: u32,
}

impl VocabTrie {
    /// Create an empty trie.
    pub fn new() -> Self {
        Self { root: TrieNode::new(), token_count: 0 }
    }

    /// Build a trie from an iterator of `(token_string, token_id)` pairs.
    pub fn from_vocabulary<I, S>(vocab: I) -> Self
    where
        I: IntoIterator<Item = (S, u32)>,
        S: AsRef<str>,
    {
        let mut trie = Self::new();
        for (token, id) in vocab {
            trie.insert(token.as_ref(), id);
        }
        trie
    }

    /// Insert a token string with the given ID.
    pub fn insert(&mut self, token: &str, token_id: u32) {
        let mut node = &mut self.root;
        node.prefix_count += 1;
        for ch in token.chars() {
            node = node.children.entry(ch).or_default();
            node.prefix_count += 1;
        }
        if node.token_id.is_none() {
            self.token_count += 1;
        }
        node.token_id = Some(token_id);
    }

    /// Look up a token string and return its ID if present.
    pub fn search(&self, token: &str) -> Option<u32> {
        let mut node = &self.root;
        for ch in token.chars() {
            match node.children.get(&ch) {
                Some(child) => node = child,
                None => return None,
            }
        }
        node.token_id
    }

    /// Perform a prefix search: navigate to the node matching `prefix`,
    /// then collect all completions below it.
    pub fn prefix_search(&self, prefix: &str) -> TrieSearchResult {
        let mut node = &self.root;
        for ch in prefix.chars() {
            match node.children.get(&ch) {
                Some(child) => node = child,
                None => return TrieSearchResult::empty(),
            }
        }
        let exact_match = node.token_id;
        let prefix_match_count = node.prefix_count;
        let mut completions = Vec::new();
        Self::collect_completions(node, &mut String::new(), &mut completions);
        TrieSearchResult { exact_match, prefix_match_count, completions }
    }

    /// Recursively collect `(suffix, token_id)` pairs below `node`.
    fn collect_completions(
        node: &TrieNode,
        current: &mut String,
        out: &mut Vec<(String, u32)>,
    ) {
        if let Some(id) = node.token_id {
            out.push((current.clone(), id));
        }
        for (&ch, child) in &node.children {
            current.push(ch);
            Self::collect_completions(child, current, out);
            current.pop();
        }
    }

    /// Return the total number of distinct tokens.
    pub fn token_count(&self) -> u32 {
        self.token_count
    }

    /// Return a reference to the root node.
    pub fn root(&self) -> &TrieNode {
        &self.root
    }

    /// Check whether the trie contains a given token string.
    pub fn contains(&self, token: &str) -> bool {
        self.search(token).is_some()
    }

    /// Return all tokens whose string starts with `prefix`.
    pub fn tokens_with_prefix(&self, prefix: &str) -> Vec<(String, u32)> {
        let result = self.prefix_search(prefix);
        result
            .completions
            .into_iter()
            .map(|(suffix, id)| (format!("{prefix}{suffix}"), id))
            .collect()
    }

    /// Compute statistics about the trie.
    pub fn stats(&self) -> TrieStats {
        let node_count = self.root.node_count();
        let max_depth = self.root.max_depth();
        let memory_bytes =
            node_count * std::mem::size_of::<TrieNode>() + (self.token_count as usize) * 4;
        TrieStats {
            node_count,
            max_depth,
            token_count: self.token_count,
            memory_bytes,
            search_timings: Vec::new(),
        }
    }
}

impl Default for VocabTrie {
    fn default() -> Self {
        Self::new()
    }
}

// ── CompactTrieNode ───────────────────────────────────────────────

/// A single node in the compact (GPU-friendly) trie representation.
///
/// Fixed-size struct suitable for packing into a flat `u32` buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct CompactTrieNode {
    /// Token ID stored at this node, or `NO_TOKEN`.
    pub token_id: u32,
    /// Number of children.
    pub num_children: u32,
    /// Index into the children array where this node's children start.
    pub children_offset: u32,
    /// Number of tokens reachable from this node.
    pub prefix_count: u32,
}

impl CompactTrieNode {
    /// Size of one node in `u32` units.
    pub const U32_SIZE: usize = 4;
}

/// A child entry in the compact trie: `(character_codepoint, node_index)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct CompactChildEntry {
    /// Unicode codepoint of the edge label.
    pub codepoint: u32,
    /// Index of the child node in the node array.
    pub node_index: u32,
}

impl CompactChildEntry {
    /// Size of one entry in `u32` units.
    pub const U32_SIZE: usize = 2;
}

// ── CompactTrie ───────────────────────────────────────────────────

/// Array-based trie for GPU-friendly memory layout.
///
/// All data is stored in flat arrays so it can be uploaded to OpenCL
/// buffers in a single transfer.
#[derive(Debug, Clone)]
pub struct CompactTrie {
    /// Flat array of nodes.
    pub nodes: Vec<CompactTrieNode>,
    /// Flat array of child entries.
    pub children: Vec<CompactChildEntry>,
}

impl CompactTrie {
    /// Convert a `VocabTrie` into a compact representation.
    pub fn from_vocab_trie(trie: &VocabTrie) -> Self {
        let node_count = trie.root().node_count();
        let mut nodes = Vec::with_capacity(node_count);
        let mut children_list: Vec<CompactChildEntry> = Vec::new();

        // BFS to assign contiguous indices.
        let mut queue: VecDeque<&TrieNode> = VecDeque::new();
        // Map from TrieNode pointer to compact index — we use BFS order.
        // First, count children to reserve space; then fill in offsets.

        // Phase 1: BFS to collect nodes in order.
        let mut bfs_nodes: Vec<&TrieNode> = Vec::with_capacity(node_count);
        queue.push_back(&trie.root);
        while let Some(node) = queue.pop_front() {
            bfs_nodes.push(node);
            for child in node.children.values() {
                queue.push_back(child);
            }
        }

        // Phase 2: assign child offsets and build arrays.
        // We need a mapping from TrieNode ptr → index.
        let mut ptr_to_idx: HashMap<*const TrieNode, u32> = HashMap::new();
        for (i, node) in bfs_nodes.iter().enumerate() {
            ptr_to_idx.insert(*node as *const TrieNode, i as u32);
        }

        for &node in &bfs_nodes {
            let children_offset = children_list.len() as u32;
            let num_children = node.children.len() as u32;
            for (&ch, child) in &node.children {
                let child_idx = ptr_to_idx[&(child as *const TrieNode)];
                children_list.push(CompactChildEntry {
                    codepoint: ch as u32,
                    node_index: child_idx,
                });
            }
            nodes.push(CompactTrieNode {
                token_id: node.token_id.unwrap_or(NO_TOKEN),
                num_children,
                children_offset,
                prefix_count: node.prefix_count,
            });
        }

        Self { nodes, children: children_list }
    }

    /// Search for an exact match in the compact trie (CPU reference).
    pub fn search(&self, token: &str) -> Option<u32> {
        let mut idx: u32 = 0; // root
        for ch in token.chars() {
            let cp = ch as u32;
            let node = &self.nodes[idx as usize];
            let start = node.children_offset as usize;
            let end = start + node.num_children as usize;
            let mut found = false;
            for entry in &self.children[start..end] {
                if entry.codepoint == cp {
                    idx = entry.node_index;
                    found = true;
                    break;
                }
            }
            if !found {
                return None;
            }
        }
        let tid = self.nodes[idx as usize].token_id;
        if tid == NO_TOKEN { None } else { Some(tid) }
    }

    /// Return the number of tokens that have `prefix` as a prefix.
    pub fn prefix_count(&self, prefix: &str) -> u32 {
        let mut idx: u32 = 0;
        for ch in prefix.chars() {
            let cp = ch as u32;
            let node = &self.nodes[idx as usize];
            let start = node.children_offset as usize;
            let end = start + node.num_children as usize;
            let mut found = false;
            for entry in &self.children[start..end] {
                if entry.codepoint == cp {
                    idx = entry.node_index;
                    found = true;
                    break;
                }
            }
            if !found {
                return 0;
            }
        }
        self.nodes[idx as usize].prefix_count
    }

    /// Pack the trie into a flat `u32` buffer for GPU upload.
    ///
    /// Layout: `[num_nodes, num_children, ...node_data..., ...child_data...]`
    pub fn to_gpu_buffer(&self) -> Vec<u32> {
        let header_size = 2;
        let nodes_size = self.nodes.len() * CompactTrieNode::U32_SIZE;
        let children_size = self.children.len() * CompactChildEntry::U32_SIZE;
        let mut buf = Vec::with_capacity(header_size + nodes_size + children_size);
        buf.push(self.nodes.len() as u32);
        buf.push(self.children.len() as u32);
        for n in &self.nodes {
            buf.push(n.token_id);
            buf.push(n.num_children);
            buf.push(n.children_offset);
            buf.push(n.prefix_count);
        }
        for c in &self.children {
            buf.push(c.codepoint);
            buf.push(c.node_index);
        }
        buf
    }

    /// Reconstruct a `CompactTrie` from a flat `u32` GPU buffer.
    pub fn from_gpu_buffer(buf: &[u32]) -> Option<Self> {
        if buf.len() < 2 {
            return None;
        }
        let num_nodes = buf[0] as usize;
        let num_children = buf[1] as usize;
        let expected =
            2 + num_nodes * CompactTrieNode::U32_SIZE + num_children * CompactChildEntry::U32_SIZE;
        if buf.len() < expected {
            return None;
        }
        let mut nodes = Vec::with_capacity(num_nodes);
        let mut offset = 2;
        for _ in 0..num_nodes {
            nodes.push(CompactTrieNode {
                token_id: buf[offset],
                num_children: buf[offset + 1],
                children_offset: buf[offset + 2],
                prefix_count: buf[offset + 3],
            });
            offset += CompactTrieNode::U32_SIZE;
        }
        let mut children = Vec::with_capacity(num_children);
        for _ in 0..num_children {
            children.push(CompactChildEntry {
                codepoint: buf[offset],
                node_index: buf[offset + 1],
            });
            offset += CompactChildEntry::U32_SIZE;
        }
        Some(Self { nodes, children })
    }

    /// Total node count.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Memory size of the GPU buffer in bytes.
    pub fn gpu_buffer_bytes(&self) -> usize {
        let u32s = 2
            + self.nodes.len() * CompactTrieNode::U32_SIZE
            + self.children.len() * CompactChildEntry::U32_SIZE;
        u32s * 4
    }

    /// Check that GPU buffer size is a multiple of `NODE_ALIGNMENT`.
    pub fn is_aligned(&self) -> bool {
        self.gpu_buffer_bytes().is_multiple_of(NODE_ALIGNMENT)
    }

    /// Return the padded GPU buffer with 64-byte alignment.
    pub fn to_aligned_gpu_buffer(&self) -> Vec<u32> {
        let mut buf = self.to_gpu_buffer();
        let alignment_u32 = NODE_ALIGNMENT / 4;
        let remainder = buf.len() % alignment_u32;
        if remainder != 0 {
            buf.resize(buf.len() + (alignment_u32 - remainder), 0);
        }
        buf
    }
}

// ── TrieTokenizer ─────────────────────────────────────────────────

/// Greedy longest-match tokenizer using the vocabulary trie.
#[derive(Debug)]
pub struct TrieTokenizer<'a> {
    trie: &'a VocabTrie,
    /// Fallback token ID for unknown single characters.
    unknown_token_id: u32,
}

impl<'a> TrieTokenizer<'a> {
    /// Create a tokenizer backed by the given trie.
    pub fn new(trie: &'a VocabTrie, unknown_token_id: u32) -> Self {
        Self { trie, unknown_token_id }
    }

    /// Tokenize `text` using greedy longest-match.
    ///
    /// Returns a list of token IDs. Characters with no trie match are
    /// assigned `unknown_token_id` and consumed one at a time.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let chars: Vec<char> = text.chars().collect();
        let mut tokens = Vec::new();
        let mut pos = 0;
        while pos < chars.len() {
            let mut node = &self.trie.root;
            let mut last_match: Option<(u32, usize)> = None; // (token_id, end_pos)
            for (offset, &ch) in chars[pos..].iter().enumerate() {
                match node.children.get(&ch) {
                    Some(child) => {
                        node = child;
                        if let Some(id) = node.token_id {
                            last_match = Some((id, pos + offset + 1));
                        }
                    }
                    None => break,
                }
            }
            match last_match {
                Some((id, end)) => {
                    tokens.push(id);
                    pos = end;
                }
                None => {
                    tokens.push(self.unknown_token_id);
                    pos += 1;
                }
            }
        }
        tokens
    }

    /// Tokenize and return `(token_id, token_string)` pairs.
    pub fn tokenize_with_strings(&self, text: &str) -> Vec<(u32, String)> {
        let chars: Vec<char> = text.chars().collect();
        let mut result = Vec::new();
        let mut pos = 0;
        while pos < chars.len() {
            let mut node = &self.trie.root;
            let mut last_match: Option<(u32, usize)> = None;
            for (offset, &ch) in chars[pos..].iter().enumerate() {
                match node.children.get(&ch) {
                    Some(child) => {
                        node = child;
                        if let Some(id) = node.token_id {
                            last_match = Some((id, pos + offset + 1));
                        }
                    }
                    None => break,
                }
            }
            match last_match {
                Some((id, end)) => {
                    let s: String = chars[pos..end].iter().collect();
                    result.push((id, s));
                    pos = end;
                }
                None => {
                    let s: String = chars[pos..pos + 1].iter().collect();
                    result.push((self.unknown_token_id, s));
                    pos += 1;
                }
            }
        }
        result
    }
}

// ── PrefixCache ───────────────────────────────────────────────────

/// LRU cache for frequently searched prefix results.
#[derive(Debug)]
pub struct PrefixCache {
    capacity: usize,
    /// Order of access (most recent at back).
    order: VecDeque<String>,
    /// Cached results.
    entries: HashMap<String, TrieSearchResult>,
    /// Statistics.
    hits: u64,
    misses: u64,
}

impl PrefixCache {
    /// Create a new cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            order: VecDeque::with_capacity(capacity),
            entries: HashMap::with_capacity(capacity),
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a prefix in the cache.
    pub fn get(&mut self, prefix: &str) -> Option<&TrieSearchResult> {
        if self.entries.contains_key(prefix) {
            self.hits += 1;
            // Move to back (most recently used).
            self.order.retain(|k| k != prefix);
            self.order.push_back(prefix.to_string());
            self.entries.get(prefix)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a result into the cache, evicting the LRU entry if full.
    pub fn insert(&mut self, prefix: String, result: TrieSearchResult) {
        if self.entries.contains_key(&prefix) {
            self.order.retain(|k| k != &prefix);
        } else if self.entries.len() >= self.capacity
            && let Some(evicted) = self.order.pop_front()
        {
            self.entries.remove(&evicted);
        }
        self.order.push_back(prefix.clone());
        self.entries.insert(prefix, result);
    }

    /// Current number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of cache hits.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Number of cache misses.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Hit rate as a fraction in `[0.0, 1.0]`.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    /// Clear all entries and reset statistics.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Capacity of the cache.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ── TrieStats ─────────────────────────────────────────────────────

/// Statistics about a vocabulary trie.
#[derive(Debug, Clone)]
pub struct TrieStats {
    /// Total number of nodes.
    pub node_count: usize,
    /// Maximum depth of the trie.
    pub max_depth: usize,
    /// Number of distinct tokens.
    pub token_count: u32,
    /// Approximate memory usage in bytes.
    pub memory_bytes: usize,
    /// Recorded search timings.
    pub search_timings: Vec<Duration>,
}

impl TrieStats {
    /// Record a search timing.
    pub fn record_search(&mut self, dur: Duration) {
        self.search_timings.push(dur);
    }

    /// Average search time, if any timings have been recorded.
    pub fn avg_search_time(&self) -> Option<Duration> {
        if self.search_timings.is_empty() {
            return None;
        }
        let total: Duration = self.search_timings.iter().sum();
        Some(total / self.search_timings.len() as u32)
    }

    /// Median search time.
    pub fn median_search_time(&self) -> Option<Duration> {
        if self.search_timings.is_empty() {
            return None;
        }
        let mut sorted = self.search_timings.clone();
        sorted.sort();
        Some(sorted[sorted.len() / 2])
    }
}

impl fmt::Display for TrieStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TrieStats {{ nodes: {}, depth: {}, tokens: {}, mem: {} B }}",
            self.node_count, self.max_depth, self.token_count, self.memory_bytes,
        )
    }
}

// ── Batch prefix search (CPU reference) ───────────────────────────

/// Perform prefix search for a batch of queries.
///
/// Returns one `TrieSearchResult` per query. This is the CPU reference
/// implementation of the parallel workgroup kernel.
pub fn batch_prefix_search(
    trie: &VocabTrie,
    queries: &[&str],
) -> Vec<TrieSearchResult> {
    queries.iter().map(|q| trie.prefix_search(q)).collect()
}

/// Perform batch prefix search using the compact trie (CPU reference).
///
/// Returns `(exact_match_token_id, prefix_count)` per query.
pub fn batch_compact_search(
    compact: &CompactTrie,
    queries: &[&str],
) -> Vec<(Option<u32>, u32)> {
    queries
        .iter()
        .map(|q| {
            let exact = compact.search(q);
            let count = compact.prefix_count(q);
            (exact, count)
        })
        .collect()
}

/// Timed prefix search that records duration into stats.
pub fn timed_prefix_search(
    trie: &VocabTrie,
    prefix: &str,
    stats: &mut TrieStats,
) -> TrieSearchResult {
    let start = Instant::now();
    let result = trie.prefix_search(prefix);
    stats.record_search(start.elapsed());
    result
}

// ── OpenCL kernel source ──────────────────────────────────────────

/// OpenCL kernel source for parallel prefix search on the compact trie.
///
/// The kernel processes one query per work-item. Each work-item walks
/// the compact trie from root to the end of its query string and writes
/// `(exact_token_id, prefix_count)` into the output buffer.
pub const VOCAB_TRIE_CL: &str = r#"
// Compact trie layout (u32 buffer):
//   [0] = num_nodes
//   [1] = num_children
//   nodes start at offset 2, each node = 4 u32s:
//     [token_id, num_children, children_offset, prefix_count]
//   children start after nodes, each entry = 2 u32s:
//     [codepoint, node_index]

#define NO_TOKEN   0xFFFFFFFFu
#define NO_CHILD   0xFFFFFFFFu
#define NODE_U32   4
#define CHILD_U32  2

// Retrieve node field from the flat buffer.
inline uint trie_node_field(__global const uint* trie, uint node_idx, uint field) {
    uint offset = 2 + node_idx * NODE_U32 + field;
    return trie[offset];
}

// Retrieve child entry from the flat buffer.
inline uint2 trie_child_entry(
    __global const uint* trie, uint num_nodes, uint entry_idx
) {
    uint base = 2 + num_nodes * NODE_U32 + entry_idx * CHILD_U32;
    return (uint2)(trie[base], trie[base + 1]);
}

// Search for a child with the given codepoint.
inline uint find_child(
    __global const uint* trie,
    uint num_nodes,
    uint node_idx,
    uint codepoint
) {
    uint n_children   = trie_node_field(trie, node_idx, 1);
    uint child_offset = trie_node_field(trie, node_idx, 2);
    for (uint i = 0; i < n_children; i++) {
        uint2 entry = trie_child_entry(trie, num_nodes, child_offset + i);
        if (entry.x == codepoint) return entry.y;
    }
    return NO_CHILD;
}

// ── parallel prefix search kernel ────────────────────────────────
//
// Args:
//   trie_buf        – flat compact trie (u32)
//   queries         – concatenated query codepoints (u32)
//   query_offsets   – start offset of each query in `queries`
//   query_lengths   – length of each query (in codepoints)
//   results         – output: [exact_token_id, prefix_count] per query
//   num_queries     – total number of queries
//
__kernel void prefix_search_batch(
    __global const uint* trie_buf,
    __global const uint* queries,
    __global const uint* query_offsets,
    __global const uint* query_lengths,
    __global uint*       results,
    const uint           num_queries
) {
    uint gid = get_global_id(0);
    if (gid >= num_queries) return;

    uint num_nodes = trie_buf[0];
    uint q_off = query_offsets[gid];
    uint q_len = query_lengths[gid];

    uint node_idx = 0;  // root
    bool valid = true;

    for (uint i = 0; i < q_len; i++) {
        uint cp = queries[q_off + i];
        uint child = find_child(trie_buf, num_nodes, node_idx, cp);
        if (child == NO_CHILD) {
            valid = false;
            break;
        }
        node_idx = child;
    }

    if (valid) {
        results[gid * 2]     = trie_node_field(trie_buf, node_idx, 0);  // token_id
        results[gid * 2 + 1] = trie_node_field(trie_buf, node_idx, 3);  // prefix_count
    } else {
        results[gid * 2]     = NO_TOKEN;
        results[gid * 2 + 1] = 0;
    }
}
"#;

/// Validate that the OpenCL kernel source is syntactically non-empty and
/// contains expected entry points. This is a compile-time smoke check.
pub fn validate_kernel_source() -> bool {
    VOCAB_TRIE_CL.contains("__kernel")
        && VOCAB_TRIE_CL.contains("prefix_search_batch")
        && VOCAB_TRIE_CL.contains("find_child")
        && VOCAB_TRIE_CL.contains("NO_TOKEN")
}

/// Prepare batch query data for GPU upload.
///
/// Returns `(codepoints, offsets, lengths)` suitable for passing to the
/// OpenCL kernel.
pub fn prepare_batch_queries(queries: &[&str]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut codepoints = Vec::new();
    let mut offsets = Vec::new();
    let mut lengths = Vec::new();
    for &q in queries {
        offsets.push(codepoints.len() as u32);
        let chars: Vec<u32> = q.chars().map(|c| c as u32).collect();
        lengths.push(chars.len() as u32);
        codepoints.extend(chars);
    }
    (codepoints, offsets, lengths)
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a small vocabulary trie.
    fn sample_vocab() -> Vec<(&'static str, u32)> {
        vec![
            ("hello", 0),
            ("help", 1),
            ("he", 2),
            ("world", 3),
            ("word", 4),
            ("wor", 5),
            ("a", 6),
            ("an", 7),
            ("and", 8),
            ("ant", 9),
        ]
    }

    fn build_sample_trie() -> VocabTrie {
        VocabTrie::from_vocabulary(sample_vocab())
    }

    // ── Construction tests ──────────────────────────────────────

    #[test]
    fn test_empty_trie() {
        let trie = VocabTrie::new();
        assert_eq!(trie.token_count(), 0);
        assert!(trie.search("anything").is_none());
    }

    #[test]
    fn test_single_insert() {
        let mut trie = VocabTrie::new();
        trie.insert("cat", 42);
        assert_eq!(trie.token_count(), 1);
        assert_eq!(trie.search("cat"), Some(42));
    }

    #[test]
    fn test_from_vocabulary() {
        let trie = build_sample_trie();
        assert_eq!(trie.token_count(), 10);
    }

    #[test]
    fn test_duplicate_insert_overwrites() {
        let mut trie = VocabTrie::new();
        trie.insert("tok", 1);
        trie.insert("tok", 2);
        assert_eq!(trie.search("tok"), Some(2));
        assert_eq!(trie.token_count(), 1);
    }

    #[test]
    fn test_insert_prefix_and_extension() {
        let mut trie = VocabTrie::new();
        trie.insert("a", 0);
        trie.insert("ab", 1);
        trie.insert("abc", 2);
        assert_eq!(trie.search("a"), Some(0));
        assert_eq!(trie.search("ab"), Some(1));
        assert_eq!(trie.search("abc"), Some(2));
        assert_eq!(trie.token_count(), 3);
    }

    // ── Exact match tests ───────────────────────────────────────

    #[test]
    fn test_search_hit() {
        let trie = build_sample_trie();
        assert_eq!(trie.search("hello"), Some(0));
        assert_eq!(trie.search("help"), Some(1));
        assert_eq!(trie.search("world"), Some(3));
        assert_eq!(trie.search("a"), Some(6));
    }

    #[test]
    fn test_search_miss() {
        let trie = build_sample_trie();
        assert!(trie.search("hel").is_none());
        assert!(trie.search("worlds").is_none());
        assert!(trie.search("xyz").is_none());
        assert!(trie.search("").is_none());
    }

    #[test]
    fn test_contains() {
        let trie = build_sample_trie();
        assert!(trie.contains("hello"));
        assert!(!trie.contains("hel"));
    }

    #[test]
    fn test_search_partial_key() {
        let trie = build_sample_trie();
        // "he" is a valid token, "hel" is not
        assert_eq!(trie.search("he"), Some(2));
        assert!(trie.search("hel").is_none());
    }

    #[test]
    fn test_search_longer_than_any_token() {
        let trie = build_sample_trie();
        assert!(trie.search("hello_world").is_none());
    }

    // ── Prefix search tests ─────────────────────────────────────

    #[test]
    fn test_prefix_search_basic() {
        let trie = build_sample_trie();
        let result = trie.prefix_search("he");
        assert_eq!(result.exact_match, Some(2));
        assert!(result.prefix_match_count >= 3); // he, hello, help
        assert!(!result.completions.is_empty());
    }

    #[test]
    fn test_prefix_search_no_match() {
        let trie = build_sample_trie();
        let result = trie.prefix_search("xyz");
        assert!(result.is_empty());
        assert_eq!(result.completions.len(), 0);
    }

    #[test]
    fn test_prefix_search_full_token() {
        let trie = build_sample_trie();
        let result = trie.prefix_search("hello");
        assert_eq!(result.exact_match, Some(0));
        assert_eq!(result.completions.len(), 1); // only "hello" itself
    }

    #[test]
    fn test_prefix_search_root() {
        let trie = build_sample_trie();
        let result = trie.prefix_search("");
        assert!(result.exact_match.is_none());
        // root prefix covers all tokens
        assert_eq!(result.completions.len(), 10);
    }

    #[test]
    fn test_prefix_completions_content() {
        let trie = build_sample_trie();
        let result = trie.prefix_search("an");
        assert_eq!(result.exact_match, Some(7)); // "an" → 7
        let ids: Vec<u32> = result.completions.iter().map(|(_, id)| *id).collect();
        assert!(ids.contains(&7)); // "an"
        assert!(ids.contains(&8)); // "and"
        assert!(ids.contains(&9)); // "ant"
    }

    #[test]
    fn test_tokens_with_prefix() {
        let trie = build_sample_trie();
        let tokens = trie.tokens_with_prefix("wor");
        let ids: Vec<u32> = tokens.iter().map(|(_, id)| *id).collect();
        assert!(ids.contains(&5)); // "wor"
        assert!(ids.contains(&4)); // "word"
        assert!(ids.contains(&3)); // "world"
    }

    // ── Longest-match tokenization tests ────────────────────────

    #[test]
    fn test_tokenize_simple() {
        let trie = build_sample_trie();
        let tok = TrieTokenizer::new(&trie, 999);
        let tokens = tok.tokenize("hello");
        assert_eq!(tokens, vec![0]); // "hello" → 0
    }

    #[test]
    fn test_tokenize_greedy_longest() {
        let trie = build_sample_trie();
        let tok = TrieTokenizer::new(&trie, 999);
        // "help" should match as a single token (1), not "he" + unknown
        let tokens = tok.tokenize("help");
        assert_eq!(tokens, vec![1]);
    }

    #[test]
    fn test_tokenize_multiple_tokens() {
        let trie = build_sample_trie();
        let tok = TrieTokenizer::new(&trie, 999);
        // "and" → 8, no separator, so next starts from space or similar
        let tokens = tok.tokenize("and");
        assert_eq!(tokens, vec![8]);
    }

    #[test]
    fn test_tokenize_unknown_chars() {
        let trie = build_sample_trie();
        let tok = TrieTokenizer::new(&trie, 999);
        let tokens = tok.tokenize("xyz");
        assert_eq!(tokens, vec![999, 999, 999]);
    }

    #[test]
    fn test_tokenize_mixed() {
        let trie = build_sample_trie();
        let tok = TrieTokenizer::new(&trie, 999);
        // "he" → 2, "l" → unknown, "a" → 6
        // Actually "hel" — greedy: "he" matches (2), then "l" is unknown
        let tokens = tok.tokenize("hela");
        assert_eq!(tokens, vec![2, 999, 6]); // "he" + "l"(unk) + "a"
    }

    #[test]
    fn test_tokenize_empty_input() {
        let trie = build_sample_trie();
        let tok = TrieTokenizer::new(&trie, 999);
        let tokens = tok.tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_with_strings() {
        let trie = build_sample_trie();
        let tok = TrieTokenizer::new(&trie, 999);
        let result = tok.tokenize_with_strings("help");
        assert_eq!(result, vec![(1, "help".to_string())]);
    }

    #[test]
    fn test_tokenize_with_strings_mixed() {
        let trie = build_sample_trie();
        let tok = TrieTokenizer::new(&trie, 999);
        let result = tok.tokenize_with_strings("hela");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], (2, "he".to_string()));
        assert_eq!(result[1], (999, "l".to_string()));
        assert_eq!(result[2], (6, "a".to_string()));
    }

    // ── Compact trie tests ──────────────────────────────────────

    #[test]
    fn test_compact_from_vocab_trie() {
        let trie = build_sample_trie();
        let compact = CompactTrie::from_vocab_trie(&trie);
        assert!(compact.node_count() > 0);
        assert!(!compact.children.is_empty());
    }

    #[test]
    fn test_compact_search_equivalence() {
        let trie = build_sample_trie();
        let compact = CompactTrie::from_vocab_trie(&trie);
        for (tok, id) in sample_vocab() {
            assert_eq!(
                compact.search(tok),
                Some(id),
                "compact search mismatch for '{tok}'"
            );
        }
    }

    #[test]
    fn test_compact_search_miss() {
        let trie = build_sample_trie();
        let compact = CompactTrie::from_vocab_trie(&trie);
        assert!(compact.search("xyz").is_none());
        assert!(compact.search("hel").is_none());
        assert!(compact.search("").is_none());
    }

    #[test]
    fn test_compact_prefix_count() {
        let trie = build_sample_trie();
        let compact = CompactTrie::from_vocab_trie(&trie);
        let count = compact.prefix_count("he");
        assert!(count >= 3); // he, hello, help
    }

    #[test]
    fn test_compact_prefix_count_no_match() {
        let trie = build_sample_trie();
        let compact = CompactTrie::from_vocab_trie(&trie);
        assert_eq!(compact.prefix_count("xyz"), 0);
    }

    #[test]
    fn test_compact_gpu_buffer_roundtrip() {
        let trie = build_sample_trie();
        let compact = CompactTrie::from_vocab_trie(&trie);
        let buf = compact.to_gpu_buffer();
        let restored = CompactTrie::from_gpu_buffer(&buf).unwrap();
        assert_eq!(compact.nodes, restored.nodes);
        assert_eq!(compact.children, restored.children);
    }

    #[test]
    fn test_compact_gpu_buffer_too_short() {
        assert!(CompactTrie::from_gpu_buffer(&[]).is_none());
        assert!(CompactTrie::from_gpu_buffer(&[1]).is_none());
    }

    #[test]
    fn test_compact_gpu_buffer_invalid_counts() {
        // header says 100 nodes but buffer is tiny
        assert!(CompactTrie::from_gpu_buffer(&[100, 100]).is_none());
    }

    #[test]
    fn test_compact_aligned_buffer() {
        let trie = build_sample_trie();
        let compact = CompactTrie::from_vocab_trie(&trie);
        let aligned = compact.to_aligned_gpu_buffer();
        // Must be multiple of NODE_ALIGNMENT / 4 u32s = 16
        assert_eq!(aligned.len() % (NODE_ALIGNMENT / 4), 0);
    }

    #[test]
    fn test_compact_search_after_roundtrip() {
        let trie = build_sample_trie();
        let compact = CompactTrie::from_vocab_trie(&trie);
        let buf = compact.to_gpu_buffer();
        let restored = CompactTrie::from_gpu_buffer(&buf).unwrap();
        for (tok, id) in sample_vocab() {
            assert_eq!(restored.search(tok), Some(id));
        }
    }

    // ── LRU cache tests ─────────────────────────────────────────

    #[test]
    fn test_cache_miss() {
        let mut cache = PrefixCache::new(4);
        assert!(cache.get("abc").is_none());
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_cache_hit() {
        let mut cache = PrefixCache::new(4);
        let result = TrieSearchResult {
            exact_match: Some(42),
            prefix_match_count: 1,
            completions: vec![],
        };
        cache.insert("abc".to_string(), result.clone());
        assert!(cache.get("abc").is_some());
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = PrefixCache::new(2);
        let r = TrieSearchResult::empty();
        cache.insert("a".to_string(), r.clone());
        cache.insert("b".to_string(), r.clone());
        assert_eq!(cache.len(), 2);
        cache.insert("c".to_string(), r.clone());
        // "a" should have been evicted
        assert_eq!(cache.len(), 2);
        assert!(cache.get("a").is_none());
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn test_cache_lru_order() {
        let mut cache = PrefixCache::new(2);
        let r = TrieSearchResult::empty();
        cache.insert("a".to_string(), r.clone());
        cache.insert("b".to_string(), r.clone());
        // Access "a" to make it recently used
        let _ = cache.get("a");
        // Insert "c" should evict "b" (LRU)
        cache.insert("c".to_string(), r.clone());
        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_none());
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = PrefixCache::new(4);
        let r = TrieSearchResult::empty();
        cache.insert("x".to_string(), r);
        let _ = cache.get("x"); // hit
        let _ = cache.get("y"); // miss
        assert!((cache.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = PrefixCache::new(4);
        let r = TrieSearchResult::empty();
        cache.insert("x".to_string(), r);
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn test_cache_capacity() {
        let cache = PrefixCache::new(16);
        assert_eq!(cache.capacity(), 16);
    }

    #[test]
    fn test_cache_zero_capacity_becomes_one() {
        let cache = PrefixCache::new(0);
        assert_eq!(cache.capacity(), 1);
    }

    #[test]
    fn test_cache_overwrite_existing() {
        let mut cache = PrefixCache::new(4);
        let r1 = TrieSearchResult {
            exact_match: Some(1),
            prefix_match_count: 1,
            completions: vec![],
        };
        let r2 = TrieSearchResult {
            exact_match: Some(2),
            prefix_match_count: 1,
            completions: vec![],
        };
        cache.insert("a".to_string(), r1);
        cache.insert("a".to_string(), r2);
        assert_eq!(cache.len(), 1);
        let cached = cache.get("a").unwrap();
        assert_eq!(cached.exact_match, Some(2));
    }

    // ── Unicode tests ───────────────────────────────────────────

    #[test]
    fn test_unicode_insert_search() {
        let mut trie = VocabTrie::new();
        trie.insert("café", 10);
        trie.insert("naïve", 11);
        trie.insert("日本語", 12);
        trie.insert("🦀", 13);
        assert_eq!(trie.search("café"), Some(10));
        assert_eq!(trie.search("naïve"), Some(11));
        assert_eq!(trie.search("日本語"), Some(12));
        assert_eq!(trie.search("🦀"), Some(13));
    }

    #[test]
    fn test_unicode_prefix_search() {
        let mut trie = VocabTrie::new();
        trie.insert("日本", 0);
        trie.insert("日本語", 1);
        trie.insert("日本人", 2);
        let result = trie.prefix_search("日本");
        assert_eq!(result.exact_match, Some(0));
        assert_eq!(result.completions.len(), 3);
    }

    #[test]
    fn test_unicode_compact_trie() {
        let mut trie = VocabTrie::new();
        trie.insert("café", 10);
        trie.insert("🦀", 13);
        let compact = CompactTrie::from_vocab_trie(&trie);
        assert_eq!(compact.search("café"), Some(10));
        assert_eq!(compact.search("🦀"), Some(13));
    }

    #[test]
    fn test_unicode_tokenization() {
        let mut trie = VocabTrie::new();
        trie.insert("café", 10);
        trie.insert("🦀", 13);
        let tok = TrieTokenizer::new(&trie, 999);
        let tokens = tok.tokenize("🦀café");
        assert_eq!(tokens, vec![13, 10]);
    }

    // ── Edge case tests ─────────────────────────────────────────

    #[test]
    fn test_single_char_tokens() {
        let mut trie = VocabTrie::new();
        for (i, ch) in ('a'..='z').enumerate() {
            trie.insert(&ch.to_string(), i as u32);
        }
        assert_eq!(trie.token_count(), 26);
        assert_eq!(trie.search("a"), Some(0));
        assert_eq!(trie.search("z"), Some(25));
    }

    #[test]
    fn test_deep_nesting() {
        let mut trie = VocabTrie::new();
        let deep: String = "a".repeat(100);
        trie.insert(&deep, 0);
        assert_eq!(trie.search(&deep), Some(0));
        assert!(trie.search(&"a".repeat(99)).is_none());
    }

    #[test]
    fn test_deep_nesting_compact() {
        let mut trie = VocabTrie::new();
        let deep: String = "a".repeat(100);
        trie.insert(&deep, 0);
        let compact = CompactTrie::from_vocab_trie(&trie);
        assert_eq!(compact.search(&deep), Some(0));
    }

    #[test]
    fn test_node_count_and_depth() {
        let trie = build_sample_trie();
        let stats = trie.stats();
        assert!(stats.node_count > 10);
        assert!(stats.max_depth >= 5); // "hello" and "world" are 5 chars
        assert_eq!(stats.token_count, 10);
    }

    #[test]
    fn test_empty_trie_stats() {
        let trie = VocabTrie::new();
        let stats = trie.stats();
        assert_eq!(stats.node_count, 1); // root
        assert_eq!(stats.max_depth, 0);
        assert_eq!(stats.token_count, 0);
    }

    #[test]
    fn test_trie_node_is_leaf() {
        let node = TrieNode::new();
        assert!(node.is_leaf());
    }

    #[test]
    fn test_trie_node_default() {
        let node = TrieNode::default();
        assert!(node.is_leaf());
        assert!(node.token_id.is_none());
        assert_eq!(node.prefix_count, 0);
    }

    // ── Batch search tests ──────────────────────────────────────

    #[test]
    fn test_batch_prefix_search() {
        let trie = build_sample_trie();
        let queries = vec!["he", "wor", "xyz"];
        let results = batch_prefix_search(&trie, &queries);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].exact_match, Some(2)); // "he"
        assert_eq!(results[1].exact_match, Some(5)); // "wor"
        assert!(results[2].is_empty()); // "xyz"
    }

    #[test]
    fn test_batch_compact_search() {
        let trie = build_sample_trie();
        let compact = CompactTrie::from_vocab_trie(&trie);
        let queries = vec!["hello", "world", "xyz"];
        let results = batch_compact_search(&compact, &queries);
        assert_eq!(results[0], (Some(0), 1)); // "hello" exact, prefix_count=1
        assert_eq!(results[1], (Some(3), 1)); // "world"
        assert_eq!(results[2], (None, 0)); // "xyz"
    }

    #[test]
    fn test_batch_empty_queries() {
        let trie = build_sample_trie();
        let results = batch_prefix_search(&trie, &[]);
        assert!(results.is_empty());
    }

    // ── Property-like tests ─────────────────────────────────────

    #[test]
    fn test_insert_then_find_always_succeeds() {
        let mut trie = VocabTrie::new();
        let tokens = vec![
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
            "theta", "iota", "kappa",
        ];
        for (i, t) in tokens.iter().enumerate() {
            trie.insert(t, i as u32);
        }
        for (i, t) in tokens.iter().enumerate() {
            assert_eq!(trie.search(t), Some(i as u32), "failed for '{t}'");
        }
    }

    #[test]
    fn test_compact_matches_original_for_all() {
        let vocab: Vec<(&str, u32)> = vec![
            ("the", 0),
            ("there", 1),
            ("their", 2),
            ("them", 3),
            ("then", 4),
            ("these", 5),
            ("this", 6),
            ("that", 7),
            ("those", 8),
        ];
        let trie = VocabTrie::from_vocabulary(vocab.clone());
        let compact = CompactTrie::from_vocab_trie(&trie);
        for (tok, id) in &vocab {
            assert_eq!(trie.search(tok), Some(*id));
            assert_eq!(compact.search(tok), Some(*id));
        }
        // Also check misses
        for miss in &["th", "the ", "x", "thei", ""] {
            assert_eq!(trie.search(miss), compact.search(miss), "mismatch for '{miss}'");
        }
    }

    #[test]
    fn test_all_inserted_tokens_appear_in_prefix_search() {
        let trie = build_sample_trie();
        let result = trie.prefix_search("");
        let ids: Vec<u32> = result.completions.iter().map(|(_, id)| *id).collect();
        for (_, expected_id) in sample_vocab() {
            assert!(ids.contains(&expected_id), "missing token id {expected_id}");
        }
    }

    // ── Timed search test ───────────────────────────────────────

    #[test]
    fn test_timed_prefix_search() {
        let trie = build_sample_trie();
        let mut stats = trie.stats();
        let result = timed_prefix_search(&trie, "he", &mut stats);
        assert_eq!(result.exact_match, Some(2));
        assert_eq!(stats.search_timings.len(), 1);
    }

    // ── Stats tests ─────────────────────────────────────────────

    #[test]
    fn test_stats_display() {
        let trie = build_sample_trie();
        let stats = trie.stats();
        let s = format!("{stats}");
        assert!(s.contains("TrieStats"));
        assert!(s.contains("nodes:"));
    }

    #[test]
    fn test_stats_avg_and_median() {
        let mut stats = TrieStats {
            node_count: 0,
            max_depth: 0,
            token_count: 0,
            memory_bytes: 0,
            search_timings: vec![],
        };
        assert!(stats.avg_search_time().is_none());
        assert!(stats.median_search_time().is_none());
        stats.record_search(Duration::from_micros(100));
        stats.record_search(Duration::from_micros(200));
        stats.record_search(Duration::from_micros(300));
        let avg = stats.avg_search_time().unwrap();
        assert_eq!(avg, Duration::from_micros(200));
        let median = stats.median_search_time().unwrap();
        assert_eq!(median, Duration::from_micros(200));
    }

    // ── OpenCL kernel source tests ──────────────────────────────

    #[test]
    fn test_kernel_source_not_empty() {
        assert!(!VOCAB_TRIE_CL.is_empty());
    }

    #[test]
    fn test_kernel_source_contains_entry_point() {
        assert!(VOCAB_TRIE_CL.contains("__kernel"));
        assert!(VOCAB_TRIE_CL.contains("prefix_search_batch"));
    }

    #[test]
    fn test_kernel_source_contains_helpers() {
        assert!(VOCAB_TRIE_CL.contains("find_child"));
        assert!(VOCAB_TRIE_CL.contains("trie_node_field"));
        assert!(VOCAB_TRIE_CL.contains("trie_child_entry"));
    }

    #[test]
    fn test_kernel_source_contains_sentinels() {
        assert!(VOCAB_TRIE_CL.contains("NO_TOKEN"));
        assert!(VOCAB_TRIE_CL.contains("NO_CHILD"));
    }

    #[test]
    fn test_validate_kernel_source() {
        assert!(validate_kernel_source());
    }

    // ── Prepare batch queries test ──────────────────────────────

    #[test]
    fn test_prepare_batch_queries() {
        let queries = vec!["he", "world"];
        let (codepoints, offsets, lengths) = prepare_batch_queries(&queries);
        assert_eq!(offsets, vec![0, 2]);
        assert_eq!(lengths, vec![2, 5]);
        assert_eq!(codepoints.len(), 7); // 2 + 5
        assert_eq!(codepoints[0], 'h' as u32);
        assert_eq!(codepoints[1], 'e' as u32);
        assert_eq!(codepoints[2], 'w' as u32);
    }

    #[test]
    fn test_prepare_batch_queries_empty() {
        let (codepoints, offsets, lengths) = prepare_batch_queries(&[]);
        assert!(codepoints.is_empty());
        assert!(offsets.is_empty());
        assert!(lengths.is_empty());
    }

    #[test]
    fn test_prepare_batch_queries_unicode() {
        let queries = vec!["🦀"];
        let (codepoints, offsets, lengths) = prepare_batch_queries(&queries);
        assert_eq!(offsets, vec![0]);
        assert_eq!(lengths, vec![1]);
        assert_eq!(codepoints[0], '🦀' as u32);
    }

    // ── TrieSearchResult tests ──────────────────────────────────

    #[test]
    fn test_search_result_empty() {
        let r = TrieSearchResult::empty();
        assert!(r.is_empty());
        assert!(r.exact_match.is_none());
        assert_eq!(r.prefix_match_count, 0);
    }

    #[test]
    fn test_search_result_not_empty_with_exact() {
        let r = TrieSearchResult {
            exact_match: Some(1),
            prefix_match_count: 1,
            completions: vec![],
        };
        assert!(!r.is_empty());
    }

    #[test]
    fn test_search_result_not_empty_with_prefix() {
        let r = TrieSearchResult {
            exact_match: None,
            prefix_match_count: 5,
            completions: vec![],
        };
        assert!(!r.is_empty());
    }

    // ── Compact trie node size ──────────────────────────────────

    #[test]
    fn test_compact_node_u32_size() {
        assert_eq!(CompactTrieNode::U32_SIZE, 4);
        assert_eq!(CompactChildEntry::U32_SIZE, 2);
    }

    #[test]
    fn test_compact_gpu_buffer_bytes() {
        let trie = build_sample_trie();
        let compact = CompactTrie::from_vocab_trie(&trie);
        let expected = (2
            + compact.nodes.len() * CompactTrieNode::U32_SIZE
            + compact.children.len() * CompactChildEntry::U32_SIZE)
            * 4;
        assert_eq!(compact.gpu_buffer_bytes(), expected);
    }

    // ── VocabTrie default ───────────────────────────────────────

    #[test]
    fn test_vocab_trie_default() {
        let trie = VocabTrie::default();
        assert_eq!(trie.token_count(), 0);
    }
}
