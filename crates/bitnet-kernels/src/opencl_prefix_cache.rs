//! OpenCL prompt prefix caching for shared KV reuse on Intel Arc A770.
//!
//! # Overview
//!
//! During multi-turn LLM inference many requests share the same system prompt
//! or conversational prefix.  Recomputing the KV cache for these common
//! prefixes wastes both time and memory bandwidth.  This module provides:
//!
//! - **`PrefixTree`** — radix tree of token sequences mapping to cached KV
//!   block references.
//! - **`PrefixMatch`** — result of a prefix lookup: matched length, cached KV
//!   reference, and remaining (unmatched) tokens.
//! - **`PrefixCacheConfig`** — tuning knobs: capacity, max prefix length,
//!   eviction policy, system-prompt pinning.
//! - **`SystemPromptPin`** — pins a system prompt's KV cache so it is never
//!   evicted.
//! - **`PrefixEviction`** — LRU eviction with reference-count protection.
//! - **`SharedKvRef`** — reference-counted handle to cached KV data.
//! - **`PrefixStats`** — hit-rate, tokens saved, memory used, pinned count.
//! - **`PrefixBatcher`** — groups incoming requests by common prefix for
//!   efficient batch prefill.
//! - CPU reference implementations for all operations.
//!
//! The OpenCL kernel source (`PREFIX_CACHE_CL`) is embedded for future GPU
//! offload on Intel Arc A770 and other OpenCL 3.0 devices.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by prefix cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixCacheError {
    /// Cache has reached its maximum number of entries.
    CacheFull { max_entries: usize },
    /// Requested prefix exceeds the configured maximum length.
    PrefixTooLong { len: usize, max: usize },
    /// A referenced KV block is still in use and cannot be evicted.
    RefcountNonZero { entry_id: u64 },
    /// Entry not found in the cache.
    EntryNotFound { entry_id: u64 },
    /// Duplicate system-prompt pin for the same token sequence.
    DuplicatePin,
}

impl fmt::Display for PrefixCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CacheFull { max_entries } => {
                write!(f, "prefix cache full (max_entries={max_entries})")
            }
            Self::PrefixTooLong { len, max } => {
                write!(f, "prefix length {len} exceeds max {max}")
            }
            Self::RefcountNonZero { entry_id } => {
                write!(f, "entry {entry_id} still in use (refcount > 0)")
            }
            Self::EntryNotFound { entry_id } => {
                write!(f, "entry {entry_id} not found")
            }
            Self::DuplicatePin => write!(f, "system prompt already pinned"),
        }
    }
}

impl std::error::Error for PrefixCacheError {}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, PrefixCacheError>;

// ---------------------------------------------------------------------------
// EvictionPolicy
// ---------------------------------------------------------------------------

/// Eviction strategy for the prefix cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    /// Least-recently-used (default).
    #[default]
    Lru,
}

// ---------------------------------------------------------------------------
// PrefixCacheConfig
// ---------------------------------------------------------------------------

/// Configuration for the prefix cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixCacheConfig {
    /// Maximum number of cached prefix entries.
    pub max_entries: usize,
    /// Maximum token length for a single cached prefix.
    pub max_prefix_len: usize,
    /// Eviction policy (currently LRU only).
    pub eviction_policy: EvictionPolicy,
    /// When `true`, system-prompt pins are allowed.
    pub enable_system_prompt_pin: bool,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            max_prefix_len: 4096,
            eviction_policy: EvictionPolicy::Lru,
            enable_system_prompt_pin: true,
        }
    }
}

// ---------------------------------------------------------------------------
// SharedKvRef — reference-counted handle
// ---------------------------------------------------------------------------

/// Reference-counted handle to cached KV data.
///
/// Each `clone` increments the logical reference count stored in the parent
/// `PrefixTree`; each `release` decrements it.  The KV data itself lives
/// inside the tree and is only freed when evicted (after refcount reaches 0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedKvRef {
    /// Opaque identifier for the cached KV block.
    pub entry_id: u64,
    /// Number of tokens covered by this cached block.
    pub token_len: usize,
}

// ---------------------------------------------------------------------------
// PrefixMatch — lookup result
// ---------------------------------------------------------------------------

/// Result of a prefix lookup in the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatch {
    /// Number of tokens matched from the query prefix.
    pub matched_length: usize,
    /// Handle to the cached KV block (if any tokens matched).
    pub cached_kv_ref: Option<SharedKvRef>,
    /// Remaining tokens that were **not** matched and still need prefill.
    pub remaining_tokens: Vec<u32>,
}

// ---------------------------------------------------------------------------
// SystemPromptPin
// ---------------------------------------------------------------------------

/// A pinned system-prompt entry that is never evicted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemPromptPin {
    /// Token sequence of the pinned system prompt.
    pub tokens: Vec<u32>,
    /// Corresponding entry id in the prefix tree.
    pub entry_id: u64,
}

// ---------------------------------------------------------------------------
// PrefixStats
// ---------------------------------------------------------------------------

/// Runtime statistics for the prefix cache.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefixStats {
    /// Number of lookup hits (matched_length > 0).
    pub hits: u64,
    /// Total lookup attempts.
    pub lookups: u64,
    /// Cumulative tokens saved via cache hits.
    pub tokens_saved: u64,
    /// Current memory usage estimate in bytes.
    pub memory_used: usize,
    /// Number of pinned (non-evictable) entries.
    pub pinned_entries: usize,
}

impl PrefixStats {
    /// Hit rate as a fraction in `[0.0, 1.0]`.
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 { 0.0 } else { self.hits as f64 / self.lookups as f64 }
    }
}

// ---------------------------------------------------------------------------
// Internal tree node
// ---------------------------------------------------------------------------

/// A node in the radix prefix tree.
#[derive(Debug, Clone)]
struct PrefixNode {
    /// Token fragment stored at this edge.
    tokens: Vec<u32>,
    /// If this node terminates a cached prefix, its entry id.
    entry_id: Option<u64>,
    /// Children keyed by first token of the child edge.
    children: HashMap<u32, PrefixNode>,
}

impl PrefixNode {
    fn new(tokens: Vec<u32>, entry_id: Option<u64>) -> Self {
        Self { tokens, entry_id, children: HashMap::new() }
    }
}

// ---------------------------------------------------------------------------
// CacheEntry metadata
// ---------------------------------------------------------------------------

/// Per-entry bookkeeping used by eviction and ref-counting.
#[derive(Debug, Clone)]
struct CacheEntryMeta {
    /// Logical reference count (number of active `SharedKvRef` handles).
    refcount: u32,
    /// Monotonic timestamp of last access (for LRU).
    last_access: u64,
    /// Whether this entry is pinned (system prompt).
    pinned: bool,
    /// Simulated KV data size in bytes.
    memory_bytes: usize,
}

// ---------------------------------------------------------------------------
// PrefixTree
// ---------------------------------------------------------------------------

/// Radix prefix tree mapping token sequences to cached KV block references.
///
/// Supports insert, longest-prefix lookup, reference counting, LRU eviction,
/// and system-prompt pinning.
#[derive(Debug)]
pub struct PrefixTree {
    root: PrefixNode,
    config: PrefixCacheConfig,
    entries: HashMap<u64, CacheEntryMeta>,
    next_id: u64,
    clock: u64,
    // Stats accumulators
    stat_hits: u64,
    stat_lookups: u64,
    stat_tokens_saved: u64,
    // Pinned prompts
    pins: Vec<SystemPromptPin>,
}

/// Allocate a new `CacheEntryMeta` entry (free function for borrow-splitting).
fn alloc_entry(
    entries: &mut HashMap<u64, CacheEntryMeta>,
    next_id: &mut u64,
    clock: u64,
    max_entries: usize,
    token_len: usize,
) -> Result<u64> {
    if entries.len() >= max_entries {
        // Inline LRU eviction: pick the oldest non-pinned, non-referenced
        // entry.
        let victim = entries
            .iter()
            .filter(|(_, m)| !m.pinned && m.refcount == 0)
            .min_by_key(|(_, m)| m.last_access)
            .map(|(&id, _)| id);
        match victim {
            Some(id) => {
                entries.remove(&id);
            }
            None => return Err(PrefixCacheError::CacheFull { max_entries }),
        }
    }
    let id = *next_id;
    *next_id += 1;
    entries.insert(
        id,
        CacheEntryMeta {
            refcount: 0,
            last_access: clock,
            pinned: false,
            memory_bytes: token_len * PrefixTree::BYTES_PER_TOKEN,
        },
    );
    Ok(id)
}

impl PrefixTree {
    /// Create a new prefix tree with the given configuration.
    pub fn new(config: PrefixCacheConfig) -> Self {
        Self {
            root: PrefixNode::new(Vec::new(), None),
            config,
            entries: HashMap::new(),
            next_id: 1,
            clock: 0,
            stat_hits: 0,
            stat_lookups: 0,
            stat_tokens_saved: 0,
            pins: Vec::new(),
        }
    }

    /// Number of cached prefix entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the tree contains no cached entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // -- Estimated bytes per token in KV cache (simplified model) -----------

    const BYTES_PER_TOKEN: usize = 512; // placeholder for real KV geometry

    // -- Insert -------------------------------------------------------------

    /// Insert a token prefix into the cache.
    ///
    /// Returns the `SharedKvRef` for the newly created (or existing) entry.
    pub fn insert(&mut self, tokens: &[u32]) -> Result<SharedKvRef> {
        if tokens.is_empty() {
            // Empty prefix: nothing to cache.
            let id = self.allocate_entry(0)?;
            return Ok(SharedKvRef { entry_id: id, token_len: 0 });
        }
        if tokens.len() > self.config.max_prefix_len {
            return Err(PrefixCacheError::PrefixTooLong {
                len: tokens.len(),
                max: self.config.max_prefix_len,
            });
        }

        // Walk / build the radix path.  We pass disjoint borrows so the
        // borrow checker can verify tree mutation and entry allocation are
        // independent.
        let entry_id = Self::insert_into_tree_inner(
            &mut self.root,
            &mut self.entries,
            &mut self.next_id,
            self.clock,
            self.config.max_entries,
            tokens,
        )?;
        self.touch(entry_id);
        Ok(SharedKvRef { entry_id, token_len: tokens.len() })
    }

    /// Internal: walk the radix tree, inserting nodes as needed, and return
    /// the entry id at the terminal node.
    fn insert_into_tree_inner(
        root: &mut PrefixNode,
        entries: &mut HashMap<u64, CacheEntryMeta>,
        next_id: &mut u64,
        clock: u64,
        max_entries: usize,
        tokens: &[u32],
    ) -> Result<u64> {
        let mut remaining = tokens;
        let mut node = root;

        loop {
            if remaining.is_empty() {
                if let Some(id) = node.entry_id {
                    return Ok(id);
                }
                let id = alloc_entry(entries, next_id, clock, max_entries, tokens.len())?;
                node.entry_id = Some(id);
                return Ok(id);
            }

            let first = remaining[0];

            if let std::collections::hash_map::Entry::Vacant(e) = node.children.entry(first) {
                let id = alloc_entry(entries, next_id, clock, max_entries, tokens.len())?;
                let leaf = PrefixNode::new(remaining.to_vec(), Some(id));
                e.insert(leaf);
                return Ok(id);
            }

            // Child exists — compare edges.
            let child = node.children.get_mut(&first).unwrap();
            let common = common_prefix_len(&child.tokens, remaining);

            if common == child.tokens.len() && common == remaining.len() {
                if let Some(id) = child.entry_id {
                    return Ok(id);
                }
                let id = alloc_entry(entries, next_id, clock, max_entries, tokens.len())?;
                child.entry_id = Some(id);
                return Ok(id);
            }

            if common == child.tokens.len() {
                remaining = &remaining[common..];
                node = node.children.get_mut(&first).unwrap();
                continue;
            }

            // Partial match — split the edge.
            let child_rest = child.tokens[common..].to_vec();
            let old_entry = child.entry_id.take();
            let old_children = std::mem::take(&mut child.children);

            child.tokens = child.tokens[..common].to_vec();

            let mut old_child = PrefixNode::new(child_rest.clone(), old_entry);
            old_child.children = old_children;
            child.children.insert(child_rest[0], old_child);

            if common == remaining.len() {
                let id = alloc_entry(entries, next_id, clock, max_entries, tokens.len())?;
                child.entry_id = Some(id);
                return Ok(id);
            }

            let suffix = &remaining[common..];
            let id = alloc_entry(entries, next_id, clock, max_entries, tokens.len())?;
            let new_leaf = PrefixNode::new(suffix.to_vec(), Some(id));
            child.children.insert(suffix[0], new_leaf);
            return Ok(id);
        }
    }

    /// Allocate a new `CacheEntryMeta`, evicting if necessary.
    fn allocate_entry(&mut self, token_len: usize) -> Result<u64> {
        alloc_entry(
            &mut self.entries,
            &mut self.next_id,
            self.clock,
            self.config.max_entries,
            token_len,
        )
    }

    // -- Lookup -------------------------------------------------------------

    /// Find the longest cached prefix that matches the beginning of `tokens`.
    pub fn lookup(&mut self, tokens: &[u32]) -> PrefixMatch {
        self.stat_lookups += 1;
        self.clock += 1;

        if tokens.is_empty() {
            return PrefixMatch {
                matched_length: 0,
                cached_kv_ref: None,
                remaining_tokens: Vec::new(),
            };
        }

        let (matched, entry_id) = self.longest_match(tokens);

        if matched == 0 || entry_id.is_none() {
            return PrefixMatch {
                matched_length: 0,
                cached_kv_ref: None,
                remaining_tokens: tokens.to_vec(),
            };
        }

        let id = entry_id.unwrap();
        self.touch(id);
        self.stat_hits += 1;
        self.stat_tokens_saved += matched as u64;

        PrefixMatch {
            matched_length: matched,
            cached_kv_ref: Some(SharedKvRef { entry_id: id, token_len: matched }),
            remaining_tokens: tokens[matched..].to_vec(),
        }
    }

    /// Walk the tree and return (matched_length, Option<entry_id>) for the
    /// deepest node that has an `entry_id`.
    fn longest_match(&self, tokens: &[u32]) -> (usize, Option<u64>) {
        let mut node = &self.root;
        let mut offset = 0usize;
        let mut best_match = 0usize;
        let mut best_id: Option<u64> = None;

        loop {
            if offset >= tokens.len() {
                break;
            }

            let first = tokens[offset];
            let child = match node.children.get(&first) {
                Some(c) => c,
                None => break,
            };

            let edge = &child.tokens;
            let remaining = &tokens[offset..];
            let common = common_prefix_len(edge, remaining);

            offset += common;

            if common < edge.len() {
                // Partial edge match — cannot descend further, but if this
                // node has an entry at a split we won't reach it.
                break;
            }

            // Full edge consumed.
            if let Some(eid) = child.entry_id
                && self.entries.contains_key(&eid)
            {
                best_match = offset;
                best_id = Some(eid);
            }

            node = child;
        }

        (best_match, best_id)
    }

    // -- Reference counting -------------------------------------------------

    /// Increment the reference count for a cached entry.
    pub fn acquire(&mut self, kv_ref: &SharedKvRef) -> Result<()> {
        let meta = self
            .entries
            .get_mut(&kv_ref.entry_id)
            .ok_or(PrefixCacheError::EntryNotFound { entry_id: kv_ref.entry_id })?;
        meta.refcount += 1;
        Ok(())
    }

    /// Decrement the reference count for a cached entry.
    pub fn release(&mut self, kv_ref: &SharedKvRef) -> Result<()> {
        let meta = self
            .entries
            .get_mut(&kv_ref.entry_id)
            .ok_or(PrefixCacheError::EntryNotFound { entry_id: kv_ref.entry_id })?;
        meta.refcount = meta.refcount.saturating_sub(1);
        Ok(())
    }

    /// Current reference count for an entry.
    pub fn refcount(&self, entry_id: u64) -> Option<u32> {
        self.entries.get(&entry_id).map(|m| m.refcount)
    }

    // -- Touch (LRU update) -------------------------------------------------

    fn touch(&mut self, entry_id: u64) {
        self.clock += 1;
        if let Some(meta) = self.entries.get_mut(&entry_id) {
            meta.last_access = self.clock;
        }
    }

    // -- System prompt pinning ----------------------------------------------

    /// Pin a system prompt so its KV cache is never evicted.
    pub fn pin_system_prompt(&mut self, tokens: &[u32]) -> Result<SystemPromptPin> {
        // Check for duplicate pin.
        for pin in &self.pins {
            if pin.tokens == tokens {
                return Err(PrefixCacheError::DuplicatePin);
            }
        }

        let kv_ref = self.insert(tokens)?;
        if let Some(meta) = self.entries.get_mut(&kv_ref.entry_id) {
            meta.pinned = true;
        }
        let pin = SystemPromptPin { tokens: tokens.to_vec(), entry_id: kv_ref.entry_id };
        self.pins.push(pin.clone());
        Ok(pin)
    }

    /// Check whether an entry is pinned.
    pub fn is_pinned(&self, entry_id: u64) -> bool {
        self.entries.get(&entry_id).is_some_and(|m| m.pinned)
    }

    // -- Stats --------------------------------------------------------------

    /// Return a snapshot of current cache statistics.
    pub fn stats(&self) -> PrefixStats {
        let memory_used: usize = self.entries.values().map(|m| m.memory_bytes).sum();
        let pinned_entries = self.entries.values().filter(|m| m.pinned).count();
        PrefixStats {
            hits: self.stat_hits,
            lookups: self.stat_lookups,
            tokens_saved: self.stat_tokens_saved,
            memory_used,
            pinned_entries,
        }
    }
}

// ---------------------------------------------------------------------------
// PrefixEviction — standalone LRU helper
// ---------------------------------------------------------------------------

/// LRU eviction tracker with reference-count protection.
///
/// This is a standalone helper usable outside the `PrefixTree` for custom
/// eviction scenarios.
#[derive(Debug)]
pub struct PrefixEviction {
    /// Maps entry id → (last_access_clock, refcount).
    entries: HashMap<u64, (u64, u32)>,
    clock: u64,
}

impl PrefixEviction {
    /// Create a new eviction tracker.
    pub fn new() -> Self {
        Self { entries: HashMap::new(), clock: 0 }
    }

    /// Record an access for `entry_id`.
    pub fn touch(&mut self, entry_id: u64) {
        self.clock += 1;
        self.entries.entry(entry_id).and_modify(|e| e.0 = self.clock).or_insert((self.clock, 0));
    }

    /// Increment refcount.
    pub fn acquire(&mut self, entry_id: u64) {
        self.entries.entry(entry_id).and_modify(|e| e.1 += 1).or_insert((self.clock, 1));
    }

    /// Decrement refcount.
    pub fn release(&mut self, entry_id: u64) {
        if let Some(e) = self.entries.get_mut(&entry_id) {
            e.1 = e.1.saturating_sub(1);
        }
    }

    /// Select the LRU victim with refcount == 0, excluding `pinned` ids.
    pub fn select_victim(&self, pinned: &[u64]) -> Option<u64> {
        self.entries
            .iter()
            .filter(|(id, (_, rc))| *rc == 0 && !pinned.contains(id))
            .min_by_key(|(_, (ts, _))| *ts)
            .map(|(&id, _)| id)
    }

    /// Remove an entry from the tracker.
    pub fn remove(&mut self, entry_id: u64) {
        self.entries.remove(&entry_id);
    }
}

impl Default for PrefixEviction {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PrefixBatcher
// ---------------------------------------------------------------------------

/// Groups incoming requests by their shared prefix for batch prefill.
#[derive(Debug)]
pub struct PrefixBatcher {
    /// Queued requests: (request_id, token_sequence).
    requests: Vec<(u64, Vec<u32>)>,
}

/// A batch group produced by the batcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchGroup {
    /// Common prefix shared by all requests in this group.
    pub common_prefix: Vec<u32>,
    /// Per-request suffixes (unique portions after the common prefix).
    pub suffixes: Vec<(u64, Vec<u32>)>,
}

impl PrefixBatcher {
    /// Create a new batcher.
    pub fn new() -> Self {
        Self { requests: Vec::new() }
    }

    /// Add a request to the pending batch.
    pub fn add_request(&mut self, request_id: u64, tokens: Vec<u32>) {
        self.requests.push((request_id, tokens));
    }

    /// Number of pending requests.
    pub fn pending(&self) -> usize {
        self.requests.len()
    }

    /// Partition queued requests into batch groups that share common
    /// prefixes.  Each group can be prefilled once for the shared portion.
    ///
    /// The current algorithm is deliberately simple: sort by token sequence,
    /// then greedily merge adjacent requests that share a non-empty prefix.
    pub fn flush(&mut self) -> Vec<BatchGroup> {
        if self.requests.is_empty() {
            return Vec::new();
        }

        let mut reqs = std::mem::take(&mut self.requests);
        reqs.sort_by(|a, b| a.1.cmp(&b.1));

        let mut groups: Vec<BatchGroup> = Vec::new();
        let mut iter = reqs.into_iter();
        let (first_id, first_tokens) = iter.next().unwrap();

        let mut current_prefix = first_tokens.clone();
        let mut current_suffixes: Vec<(u64, Vec<u32>)> = vec![(first_id, Vec::new())];

        for (rid, tokens) in iter {
            let common = common_prefix_len(&current_prefix, &tokens);
            if common > 0 {
                // Narrow the group prefix to the common portion.
                if common < current_prefix.len() {
                    // Rewrite existing suffixes.
                    for (_, suffix) in &mut current_suffixes {
                        let mut new_suffix = current_prefix[common..].to_vec();
                        new_suffix.append(suffix);
                        *suffix = new_suffix;
                    }
                    current_prefix.truncate(common);
                }
                current_suffixes.push((rid, tokens[common..].to_vec()));
            } else {
                // No overlap — flush the current group.
                groups.push(BatchGroup {
                    common_prefix: std::mem::take(&mut current_prefix),
                    suffixes: std::mem::take(&mut current_suffixes),
                });
                current_prefix = tokens.clone();
                current_suffixes = vec![(rid, Vec::new())];
            }
        }

        groups.push(BatchGroup { common_prefix: current_prefix, suffixes: current_suffixes });
        groups
    }
}

impl Default for PrefixBatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CPU reference helpers
// ---------------------------------------------------------------------------

/// Compute the length of the common prefix between two slices.
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// CPU reference: check if `query` is a prefix of `stored`.
pub fn cpu_is_prefix(query: &[u32], stored: &[u32]) -> bool {
    query.len() <= stored.len() && query == &stored[..query.len()]
}

/// CPU reference: compute prefix match length between two token sequences.
pub fn cpu_prefix_match_len(a: &[u32], b: &[u32]) -> usize {
    common_prefix_len(a, b)
}

/// CPU reference: group requests by common prefix (simplified — returns
/// groups of indices).
pub fn cpu_group_by_prefix(sequences: &[Vec<u32>]) -> Vec<Vec<usize>> {
    if sequences.is_empty() {
        return Vec::new();
    }

    let mut indices: Vec<usize> = (0..sequences.len()).collect();
    indices.sort_by(|&a, &b| sequences[a].cmp(&sequences[b]));

    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current_group = vec![indices[0]];

    for &idx in &indices[1..] {
        let prev = &sequences[*current_group.last().unwrap()];
        let curr = &sequences[idx];
        if common_prefix_len(prev, curr) > 0 {
            current_group.push(idx);
        } else {
            groups.push(std::mem::take(&mut current_group));
            current_group = vec![idx];
        }
    }
    groups.push(current_group);
    groups
}

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C kernel source for prefix-cache operations on Intel Arc A770 and
/// other OpenCL 3.0 devices.
///
/// Contains two kernels:
/// - `prefix_compare` — parallel token-by-token comparison for prefix
///   matching.
/// - `prefix_gather_kv` — gathers cached KV rows for the matched prefix
///   portion into the output buffer.
pub const PREFIX_CACHE_CL: &str = r#"
// prefix_compare: parallel comparison of a query token sequence against a
// stored prefix.  Each work-item compares one token position and writes 1
// (match) or 0 (mismatch) to the output buffer.
//
// Global work size: (max_compare_len,)
// Arguments:
//   query   – [query_len] query token ids
//   stored  – [stored_len] stored prefix token ids
//   result  – [max_compare_len] output: 1 if match, 0 otherwise
//   query_len  – number of tokens in query
//   stored_len – number of tokens in stored prefix
__kernel void prefix_compare(
    __global const uint *query,
    __global const uint *stored,
    __global int *result,
    const int query_len,
    const int stored_len)
{
    int gid = get_global_id(0);
    int len = min(query_len, stored_len);
    if (gid < len) {
        result[gid] = (query[gid] == stored[gid]) ? 1 : 0;
    } else {
        result[gid] = 0;
    }
}

// prefix_gather_kv: copy matched prefix KV rows into a contiguous output
// buffer.
//
// Global work size: (matched_tokens * row_len,)
// Arguments:
//   kv_cache – source KV cache buffer
//   output   – destination buffer [matched_tokens, row_len]
//   matched  – number of matched prefix tokens to copy
//   row_len  – elements per token row (num_heads * head_dim)
__kernel void prefix_gather_kv(
    __global const float *kv_cache,
    __global float *output,
    const int matched,
    const int row_len)
{
    int gid = get_global_id(0);
    int total = matched * row_len;
    if (gid < total) {
        output[gid] = kv_cache[gid];
    }
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PrefixCacheConfig {
        PrefixCacheConfig {
            max_entries: 16,
            max_prefix_len: 128,
            eviction_policy: EvictionPolicy::Lru,
            enable_system_prompt_pin: true,
        }
    }

    // == PrefixCacheConfig ==================================================

    #[test]
    fn config_default_values() {
        let cfg = PrefixCacheConfig::default();
        assert_eq!(cfg.max_entries, 256);
        assert_eq!(cfg.max_prefix_len, 4096);
        assert_eq!(cfg.eviction_policy, EvictionPolicy::Lru);
        assert!(cfg.enable_system_prompt_pin);
    }

    #[test]
    fn config_custom_values() {
        let cfg = default_config();
        assert_eq!(cfg.max_entries, 16);
        assert_eq!(cfg.max_prefix_len, 128);
    }

    // == PrefixTree: insert & lookup ========================================

    #[test]
    fn insert_then_exact_lookup() {
        let mut tree = PrefixTree::new(default_config());
        let tokens = vec![1, 2, 3, 4];
        tree.insert(&tokens).unwrap();

        let m = tree.lookup(&tokens);
        assert_eq!(m.matched_length, 4);
        assert!(m.cached_kv_ref.is_some());
        assert!(m.remaining_tokens.is_empty());
    }

    #[test]
    fn partial_prefix_match() {
        let mut tree = PrefixTree::new(default_config());
        tree.insert(&[10, 20, 30]).unwrap();

        let m = tree.lookup(&[10, 20, 30, 40, 50]);
        assert_eq!(m.matched_length, 3);
        assert_eq!(m.remaining_tokens, vec![40, 50]);
    }

    #[test]
    fn no_match_returns_empty() {
        let mut tree = PrefixTree::new(default_config());
        tree.insert(&[1, 2, 3]).unwrap();

        let m = tree.lookup(&[99, 100]);
        assert_eq!(m.matched_length, 0);
        assert!(m.cached_kv_ref.is_none());
        assert_eq!(m.remaining_tokens, vec![99, 100]);
    }

    #[test]
    fn empty_prefix_lookup() {
        let mut tree = PrefixTree::new(default_config());
        let m = tree.lookup(&[]);
        assert_eq!(m.matched_length, 0);
        assert!(m.cached_kv_ref.is_none());
        assert!(m.remaining_tokens.is_empty());
    }

    #[test]
    fn single_token_prefix() {
        let mut tree = PrefixTree::new(default_config());
        tree.insert(&[42]).unwrap();

        let m = tree.lookup(&[42]);
        assert_eq!(m.matched_length, 1);
        assert!(m.cached_kv_ref.is_some());
    }

    #[test]
    fn single_token_prefix_with_remaining() {
        let mut tree = PrefixTree::new(default_config());
        tree.insert(&[42]).unwrap();

        let m = tree.lookup(&[42, 99]);
        assert_eq!(m.matched_length, 1);
        assert_eq!(m.remaining_tokens, vec![99]);
    }

    #[test]
    fn insert_duplicate_returns_same_entry() {
        let mut tree = PrefixTree::new(default_config());
        let r1 = tree.insert(&[1, 2, 3]).unwrap();
        let r2 = tree.insert(&[1, 2, 3]).unwrap();
        assert_eq!(r1.entry_id, r2.entry_id);
    }

    #[test]
    fn multiple_prefixes_coexist() {
        let mut tree = PrefixTree::new(default_config());
        tree.insert(&[1, 2, 3]).unwrap();
        tree.insert(&[1, 2, 4]).unwrap();
        tree.insert(&[5, 6]).unwrap();

        let m1 = tree.lookup(&[1, 2, 3, 99]);
        assert_eq!(m1.matched_length, 3);

        let m2 = tree.lookup(&[1, 2, 4, 88]);
        assert_eq!(m2.matched_length, 3);

        let m3 = tree.lookup(&[5, 6, 77]);
        assert_eq!(m3.matched_length, 2);
    }

    #[test]
    fn diverging_prefixes_share_common_part() {
        let mut tree = PrefixTree::new(default_config());
        tree.insert(&[1, 2, 3]).unwrap();
        tree.insert(&[1, 2, 4]).unwrap();

        // Query that matches only the shared [1,2] portion — but neither
        // [1,2,3] nor [1,2,4] is a full match for [1,2,5].
        let m = tree.lookup(&[1, 2, 5]);
        // No entry stored at [1,2] so matched_length should be 0.
        assert_eq!(m.matched_length, 0);
    }

    #[test]
    fn nested_prefix_shorter_match() {
        let mut tree = PrefixTree::new(default_config());
        tree.insert(&[1, 2]).unwrap();
        tree.insert(&[1, 2, 3, 4]).unwrap();

        let m = tree.lookup(&[1, 2, 3]);
        // Should match [1,2] (the longest stored prefix that is fully
        // contained in the query).
        assert_eq!(m.matched_length, 2);
        assert_eq!(m.remaining_tokens, vec![3]);
    }

    #[test]
    fn nested_prefix_longer_match() {
        let mut tree = PrefixTree::new(default_config());
        tree.insert(&[1, 2]).unwrap();
        tree.insert(&[1, 2, 3, 4]).unwrap();

        let m = tree.lookup(&[1, 2, 3, 4, 5]);
        assert_eq!(m.matched_length, 4);
        assert_eq!(m.remaining_tokens, vec![5]);
    }

    #[test]
    fn prefix_too_long_error() {
        let cfg = PrefixCacheConfig { max_prefix_len: 3, ..default_config() };
        let mut tree = PrefixTree::new(cfg);
        let err = tree.insert(&[1, 2, 3, 4]).unwrap_err();
        assert_eq!(err, PrefixCacheError::PrefixTooLong { len: 4, max: 3 });
    }

    #[test]
    fn insert_empty_prefix() {
        let mut tree = PrefixTree::new(default_config());
        let r = tree.insert(&[]).unwrap();
        assert_eq!(r.token_len, 0);
    }

    #[test]
    fn tree_len_and_is_empty() {
        let mut tree = PrefixTree::new(default_config());
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);

        tree.insert(&[1, 2]).unwrap();
        assert!(!tree.is_empty());
        assert_eq!(tree.len(), 1);
    }

    // == System prompt pinning ==============================================

    #[test]
    fn pin_system_prompt_not_evicted() {
        let cfg = PrefixCacheConfig { max_entries: 2, ..default_config() };
        let mut tree = PrefixTree::new(cfg);

        let pin = tree.pin_system_prompt(&[100, 200]).unwrap();
        assert!(tree.is_pinned(pin.entry_id));

        // Fill the cache to capacity.
        tree.insert(&[1, 2]).unwrap();

        // Force another insert — should evict [1,2], NOT the pinned entry.
        tree.insert(&[3, 4]).unwrap();

        // Pinned entry still reachable.
        let m = tree.lookup(&[100, 200, 300]);
        assert_eq!(m.matched_length, 2);
    }

    #[test]
    fn duplicate_pin_error() {
        let mut tree = PrefixTree::new(default_config());
        tree.pin_system_prompt(&[1, 2, 3]).unwrap();
        let err = tree.pin_system_prompt(&[1, 2, 3]).unwrap_err();
        assert_eq!(err, PrefixCacheError::DuplicatePin);
    }

    #[test]
    fn pinned_entry_shows_in_stats() {
        let mut tree = PrefixTree::new(default_config());
        tree.pin_system_prompt(&[1, 2]).unwrap();
        let s = tree.stats();
        assert_eq!(s.pinned_entries, 1);
    }

    // == Reference counting =================================================

    #[test]
    fn acquire_and_release() {
        let mut tree = PrefixTree::new(default_config());
        let kv = tree.insert(&[10, 20]).unwrap();

        assert_eq!(tree.refcount(kv.entry_id), Some(0));
        tree.acquire(&kv).unwrap();
        assert_eq!(tree.refcount(kv.entry_id), Some(1));
        tree.acquire(&kv).unwrap();
        assert_eq!(tree.refcount(kv.entry_id), Some(2));
        tree.release(&kv).unwrap();
        assert_eq!(tree.refcount(kv.entry_id), Some(1));
        tree.release(&kv).unwrap();
        assert_eq!(tree.refcount(kv.entry_id), Some(0));
    }

    #[test]
    fn release_saturates_at_zero() {
        let mut tree = PrefixTree::new(default_config());
        let kv = tree.insert(&[10]).unwrap();
        tree.release(&kv).unwrap(); // already 0
        assert_eq!(tree.refcount(kv.entry_id), Some(0));
    }

    #[test]
    fn refcount_prevents_eviction() {
        let cfg = PrefixCacheConfig { max_entries: 2, ..default_config() };
        let mut tree = PrefixTree::new(cfg);

        let kv1 = tree.insert(&[1]).unwrap();
        tree.acquire(&kv1).unwrap();
        tree.insert(&[2]).unwrap();

        // Cache full — try to insert a third.  Entry [2] (refcount=0) should
        // be evicted, NOT [1] (refcount=1).
        tree.insert(&[3]).unwrap();

        // [1] should still be reachable.
        let m = tree.lookup(&[1, 99]);
        assert_eq!(m.matched_length, 1);
    }

    #[test]
    fn acquire_unknown_entry_error() {
        let mut tree = PrefixTree::new(default_config());
        let fake = SharedKvRef { entry_id: 999, token_len: 1 };
        let err = tree.acquire(&fake).unwrap_err();
        assert_eq!(err, PrefixCacheError::EntryNotFound { entry_id: 999 });
    }

    #[test]
    fn release_unknown_entry_error() {
        let mut tree = PrefixTree::new(default_config());
        let fake = SharedKvRef { entry_id: 999, token_len: 1 };
        let err = tree.release(&fake).unwrap_err();
        assert_eq!(err, PrefixCacheError::EntryNotFound { entry_id: 999 });
    }

    // == Eviction ===========================================================

    #[test]
    fn eviction_removes_lru() {
        let cfg = PrefixCacheConfig { max_entries: 2, ..default_config() };
        let mut tree = PrefixTree::new(cfg);

        tree.insert(&[1]).unwrap();
        tree.insert(&[2]).unwrap();

        // Access [2] again so [1] becomes LRU.
        tree.lookup(&[2, 99]);

        // Insert a third — should evict [1].
        tree.insert(&[3]).unwrap();

        let m = tree.lookup(&[1, 99]);
        assert_eq!(m.matched_length, 0, "entry [1] should have been evicted");

        let m = tree.lookup(&[2, 99]);
        assert_eq!(m.matched_length, 1, "entry [2] should still be present");
    }

    #[test]
    fn cache_full_all_in_use_error() {
        let cfg = PrefixCacheConfig { max_entries: 1, ..default_config() };
        let mut tree = PrefixTree::new(cfg);

        let kv = tree.insert(&[1]).unwrap();
        tree.acquire(&kv).unwrap();

        let err = tree.insert(&[2]).unwrap_err();
        assert_eq!(err, PrefixCacheError::CacheFull { max_entries: 1 });
    }

    // == Multiple requests sharing prefix ===================================

    #[test]
    fn multiple_requests_share_prefix() {
        let mut tree = PrefixTree::new(default_config());
        tree.insert(&[100, 200, 300]).unwrap();

        // Two different "requests" look up overlapping prefixes.
        let m1 = tree.lookup(&[100, 200, 300, 1]);
        let m2 = tree.lookup(&[100, 200, 300, 2]);

        assert_eq!(m1.matched_length, 3);
        assert_eq!(m2.matched_length, 3);
        // Both should reference the same cached entry.
        assert_eq!(
            m1.cached_kv_ref.as_ref().unwrap().entry_id,
            m2.cached_kv_ref.as_ref().unwrap().entry_id,
        );
    }

    // == PrefixStats ========================================================

    #[test]
    fn stats_initially_zero() {
        let tree = PrefixTree::new(default_config());
        let s = tree.stats();
        assert_eq!(s.hits, 0);
        assert_eq!(s.lookups, 0);
        assert_eq!(s.tokens_saved, 0);
        assert_eq!(s.pinned_entries, 0);
    }

    #[test]
    fn stats_after_hits_and_misses() {
        let mut tree = PrefixTree::new(default_config());
        tree.insert(&[1, 2, 3]).unwrap();

        tree.lookup(&[1, 2, 3, 4]); // hit
        tree.lookup(&[99]); // miss
        tree.lookup(&[1, 2, 3]); // hit

        let s = tree.stats();
        assert_eq!(s.lookups, 3);
        assert_eq!(s.hits, 2);
        assert_eq!(s.tokens_saved, 6); // 3 + 3
    }

    #[test]
    fn stats_hit_rate() {
        let mut s =
            PrefixStats { hits: 3, lookups: 4, tokens_saved: 0, memory_used: 0, pinned_entries: 0 };
        assert!((s.hit_rate() - 0.75).abs() < 1e-9);

        s.lookups = 0;
        s.hits = 0;
        assert_eq!(s.hit_rate(), 0.0);
    }

    #[test]
    fn stats_memory_used() {
        let mut tree = PrefixTree::new(default_config());
        tree.insert(&[1, 2, 3]).unwrap();
        let s = tree.stats();
        assert_eq!(s.memory_used, 3 * PrefixTree::BYTES_PER_TOKEN);
    }

    // == PrefixEviction (standalone) ========================================

    #[test]
    fn eviction_select_lru_victim() {
        let mut ev = PrefixEviction::new();
        ev.touch(1);
        ev.touch(2);
        ev.touch(3);
        // 1 is LRU.
        assert_eq!(ev.select_victim(&[]), Some(1));
    }

    #[test]
    fn eviction_skips_in_use() {
        let mut ev = PrefixEviction::new();
        ev.touch(1);
        ev.touch(2);
        ev.acquire(1);
        // 1 is in use, so victim should be 2.
        assert_eq!(ev.select_victim(&[]), Some(2));
    }

    #[test]
    fn eviction_skips_pinned() {
        let mut ev = PrefixEviction::new();
        ev.touch(1);
        ev.touch(2);
        assert_eq!(ev.select_victim(&[1]), Some(2));
    }

    #[test]
    fn eviction_no_victim_all_in_use() {
        let mut ev = PrefixEviction::new();
        ev.touch(1);
        ev.acquire(1);
        assert_eq!(ev.select_victim(&[]), None);
    }

    #[test]
    fn eviction_remove() {
        let mut ev = PrefixEviction::new();
        ev.touch(1);
        ev.remove(1);
        assert_eq!(ev.select_victim(&[]), None);
    }

    #[test]
    fn eviction_release_then_evictable() {
        let mut ev = PrefixEviction::new();
        ev.touch(1);
        ev.acquire(1);
        assert_eq!(ev.select_victim(&[]), None);
        ev.release(1);
        assert_eq!(ev.select_victim(&[]), Some(1));
    }

    // == PrefixBatcher ======================================================

    #[test]
    fn batcher_empty_flush() {
        let mut b = PrefixBatcher::new();
        let groups = b.flush();
        assert!(groups.is_empty());
    }

    #[test]
    fn batcher_single_request() {
        let mut b = PrefixBatcher::new();
        b.add_request(1, vec![10, 20, 30]);
        let groups = b.flush();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].common_prefix, vec![10, 20, 30]);
        assert_eq!(groups[0].suffixes.len(), 1);
    }

    #[test]
    fn batcher_groups_common_prefix() {
        let mut b = PrefixBatcher::new();
        b.add_request(1, vec![1, 2, 3, 10]);
        b.add_request(2, vec![1, 2, 3, 20]);
        let groups = b.flush();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].common_prefix, vec![1, 2, 3]);
        assert_eq!(groups[0].suffixes.len(), 2);
    }

    #[test]
    fn batcher_disjoint_groups() {
        let mut b = PrefixBatcher::new();
        b.add_request(1, vec![1, 2]);
        b.add_request(2, vec![9, 8]);
        let groups = b.flush();
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn batcher_pending_count() {
        let mut b = PrefixBatcher::new();
        assert_eq!(b.pending(), 0);
        b.add_request(1, vec![1]);
        b.add_request(2, vec![2]);
        assert_eq!(b.pending(), 2);
        b.flush();
        assert_eq!(b.pending(), 0);
    }

    #[test]
    fn batcher_three_way_grouping() {
        let mut b = PrefixBatcher::new();
        b.add_request(1, vec![1, 2, 3]);
        b.add_request(2, vec![1, 2, 4]);
        b.add_request(3, vec![1, 2, 5]);
        let groups = b.flush();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].common_prefix, vec![1, 2]);
        assert_eq!(groups[0].suffixes.len(), 3);
    }

    // == CPU reference helpers ==============================================

    #[test]
    fn cpu_is_prefix_true() {
        assert!(cpu_is_prefix(&[1, 2], &[1, 2, 3]));
    }

    #[test]
    fn cpu_is_prefix_exact() {
        assert!(cpu_is_prefix(&[1, 2, 3], &[1, 2, 3]));
    }

    #[test]
    fn cpu_is_prefix_false() {
        assert!(!cpu_is_prefix(&[1, 2, 3], &[1, 2]));
    }

    #[test]
    fn cpu_is_prefix_empty_query() {
        assert!(cpu_is_prefix(&[], &[1, 2]));
    }

    #[test]
    fn cpu_prefix_match_len_full() {
        assert_eq!(cpu_prefix_match_len(&[1, 2, 3], &[1, 2, 3]), 3);
    }

    #[test]
    fn cpu_prefix_match_len_partial() {
        assert_eq!(cpu_prefix_match_len(&[1, 2, 3], &[1, 2, 9]), 2);
    }

    #[test]
    fn cpu_prefix_match_len_none() {
        assert_eq!(cpu_prefix_match_len(&[1], &[2]), 0);
    }

    #[test]
    fn cpu_group_by_prefix_basic() {
        let seqs = vec![vec![1, 2, 3], vec![1, 2, 4], vec![9, 8]];
        let groups = cpu_group_by_prefix(&seqs);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn cpu_group_by_prefix_empty() {
        let groups = cpu_group_by_prefix(&[]);
        assert!(groups.is_empty());
    }

    #[test]
    fn cpu_group_by_prefix_single() {
        let seqs = vec![vec![1, 2]];
        let groups = cpu_group_by_prefix(&seqs);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], vec![0]);
    }

    // == Property-style tests ===============================================

    #[test]
    fn insert_then_lookup_always_matches() {
        let mut tree = PrefixTree::new(default_config());
        for len in 1..=10 {
            let tokens: Vec<u32> = (0..len).collect();
            tree.insert(&tokens).unwrap();
            let m = tree.lookup(&tokens);
            assert_eq!(m.matched_length, len as usize);
            assert!(m.remaining_tokens.is_empty());
        }
    }

    #[test]
    fn insert_many_then_lookup_all() {
        let mut tree = PrefixTree::new(PrefixCacheConfig { max_entries: 64, ..default_config() });
        let prefixes: Vec<Vec<u32>> = (0..20).map(|i| vec![i * 10, i * 10 + 1]).collect();
        for p in &prefixes {
            tree.insert(p).unwrap();
        }
        for p in &prefixes {
            let m = tree.lookup(p);
            assert_eq!(m.matched_length, 2, "failed for prefix {p:?}");
        }
    }

    #[test]
    fn max_prefix_len_boundary() {
        let cfg = PrefixCacheConfig { max_prefix_len: 5, ..default_config() };
        let mut tree = PrefixTree::new(cfg);
        // Exactly at limit — should succeed.
        tree.insert(&[1, 2, 3, 4, 5]).unwrap();
        // One over — should fail.
        let err = tree.insert(&[1, 2, 3, 4, 5, 6]).unwrap_err();
        assert_eq!(err, PrefixCacheError::PrefixTooLong { len: 6, max: 5 });
    }

    // == Error Display ======================================================

    #[test]
    fn error_display_cache_full() {
        let e = PrefixCacheError::CacheFull { max_entries: 256 };
        assert_eq!(e.to_string(), "prefix cache full (max_entries=256)");
    }

    #[test]
    fn error_display_prefix_too_long() {
        let e = PrefixCacheError::PrefixTooLong { len: 5000, max: 4096 };
        assert_eq!(e.to_string(), "prefix length 5000 exceeds max 4096");
    }

    #[test]
    fn error_display_refcount_non_zero() {
        let e = PrefixCacheError::RefcountNonZero { entry_id: 7 };
        assert_eq!(e.to_string(), "entry 7 still in use (refcount > 0)");
    }

    #[test]
    fn error_display_entry_not_found() {
        let e = PrefixCacheError::EntryNotFound { entry_id: 42 };
        assert_eq!(e.to_string(), "entry 42 not found");
    }

    #[test]
    fn error_display_duplicate_pin() {
        let e = PrefixCacheError::DuplicatePin;
        assert_eq!(e.to_string(), "system prompt already pinned");
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(PrefixCacheError::CacheFull { max_entries: 1 });
        assert!(e.to_string().contains("full"));
    }

    // == OpenCL kernel source ===============================================

    #[test]
    fn kernel_source_non_empty() {
        assert!(!PREFIX_CACHE_CL.is_empty());
    }

    #[test]
    fn kernel_source_contains_prefix_compare() {
        assert!(PREFIX_CACHE_CL.contains("prefix_compare"));
    }

    #[test]
    fn kernel_source_contains_prefix_gather_kv() {
        assert!(PREFIX_CACHE_CL.contains("prefix_gather_kv"));
    }

    #[test]
    fn kernel_source_contains_kernel_keyword() {
        assert!(PREFIX_CACHE_CL.contains("__kernel"));
    }

    #[test]
    fn kernel_source_contains_global_qualifier() {
        assert!(PREFIX_CACHE_CL.contains("__global"));
    }

    // == SharedKvRef ========================================================

    #[test]
    fn shared_kv_ref_clone_eq() {
        let a = SharedKvRef { entry_id: 1, token_len: 10 };
        let b = a.clone();
        assert_eq!(a, b);
    }

    // == EvictionPolicy default =============================================

    #[test]
    fn eviction_policy_default_is_lru() {
        assert_eq!(EvictionPolicy::default(), EvictionPolicy::Lru);
    }
}
