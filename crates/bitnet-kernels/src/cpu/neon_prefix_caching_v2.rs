//! NEON-optimized prefix caching v2 for Apple Silicon.
//! Efficient KV cache reuse for shared prompt prefixes,
//! with radix-tree prefix matching and cache eviction.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// FNV-1a offset basis (64-bit).
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
/// FNV-1a prime (64-bit).
const FNV_PRIME: u64 = 0x00000100000001B3;
/// Maximum children per radix tree node.
const RADIX_MAX_CHILDREN: usize = 256;

// ---------------------------------------------------------------------------
// Radix tree types
// ---------------------------------------------------------------------------

/// A node in the radix prefix tree.
#[derive(Clone, Debug)]
pub struct RadixNode {
    /// Token subsequence stored at this edge.
    pub tokens: Vec<u32>,
    /// Index into the external cache table (`None` for internal-only nodes).
    pub cache_index: Option<usize>,
    /// Children keyed by first token of the child edge.
    pub children: Vec<(u32, RadixNode)>,
}

impl RadixNode {
    /// Create an empty root node.
    #[must_use]
    pub fn new_root() -> Self {
        Self {
            tokens: Vec::new(),
            cache_index: None,
            children: Vec::new(),
        }
    }

    fn find_child(&self, key: u32) -> Option<usize> {
        self.children.iter().position(|(k, _)| *k == key)
    }

    fn find_child_mut(&mut self, key: u32) -> Option<&mut RadixNode> {
        self.children
            .iter_mut()
            .find(|(k, _)| *k == key)
            .map(|(_, node)| node)
    }
}

/// LRU metadata entry for a single cache slot.
#[derive(Clone, Copy, Debug)]
pub struct LruEntry {
    /// Logical timestamp of last access.
    pub last_used: u64,
    /// Length of the cached prefix (in tokens).
    pub prefix_len: u32,
    /// Whether this slot is occupied.
    pub occupied: bool,
}

impl Default for LruEntry {
    fn default() -> Self {
        Self {
            last_used: 0,
            prefix_len: 0,
            occupied: false,
        }
    }
}

// ===================================================================
// 1. prefix_hash_v2 — NEON-accelerated FNV-1a hash
// ===================================================================

/// Compute a 64-bit FNV-1a hash of a token sequence.
///
/// Uses NEON to accelerate the XOR-fold stage on aarch64; falls back
/// to pure scalar otherwise.
pub fn prefix_hash_v2(tokens: &[u32]) -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { prefix_hash_v2_neon(tokens) };
        }
    }
    prefix_hash_v2_scalar(tokens)
}

#[cfg(target_arch = "aarch64")]
unsafe fn prefix_hash_v2_neon(tokens: &[u32]) -> u64 {
    if tokens.is_empty() {
        return FNV_OFFSET;
    }
    let mut hash = FNV_OFFSET;

    // Load 4 tokens at a time via NEON, extract each u32, then feed
    // bytes in the same order as the scalar path for bit-exact parity.
    let chunks = tokens.len() / 4;
    let remainder = tokens.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        let v = vld1q_u32(tokens.as_ptr().add(base));
        let t0 = vgetq_lane_u32(v, 0);
        let t1 = vgetq_lane_u32(v, 1);
        let t2 = vgetq_lane_u32(v, 2);
        let t3 = vgetq_lane_u32(v, 3);
        for tok in [t0, t1, t2, t3] {
            for byte_idx in 0..4u32 {
                let byte = ((tok >> (byte_idx * 8)) & 0xFF) as u8;
                hash ^= byte as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
    }

    // Scalar tail.
    let tail_start = chunks * 4;
    for &tok in &tokens[tail_start..tail_start + remainder] {
        for byte_idx in 0..4u32 {
            let byte = ((tok >> (byte_idx * 8)) & 0xFF) as u8;
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    hash
}

/// Pure-scalar FNV-1a over the byte representation of each token.
pub fn prefix_hash_v2_scalar(tokens: &[u32]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &tok in tokens {
        for byte_idx in 0..4u32 {
            let byte = ((tok >> (byte_idx * 8)) & 0xFF) as u8;
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

// ===================================================================
// 2. prefix_match_v2 — SIMD prefix comparison
// ===================================================================

/// Return the length of the longest common prefix between `a` and `b`
/// (measured in tokens).
pub fn prefix_match_v2(a: &[u32], b: &[u32]) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { prefix_match_v2_neon(a, b) };
        }
    }
    prefix_match_v2_scalar(a, b)
}

#[cfg(target_arch = "aarch64")]
unsafe fn prefix_match_v2_neon(a: &[u32], b: &[u32]) -> usize {
    let min_len = a.len().min(b.len());
    let chunks = min_len / 4;
    let mut matched: usize = 0;

    for i in 0..chunks {
        let base = i * 4;
        let va = vld1q_u32(a.as_ptr().add(base));
        let vb = vld1q_u32(b.as_ptr().add(base));
        let cmp = vceqq_u32(va, vb);
        // All lanes equal ⇒ each lane is 0xFFFF_FFFF.
        let min_val = vminvq_u32(cmp);
        if min_val == 0xFFFF_FFFF {
            matched += 4;
        } else {
            // Find exact mismatch position inside the chunk.
            for j in 0..4usize {
                if a[base + j] != b[base + j] {
                    return matched + j;
                }
            }
            return matched + 4; // unreachable in practice
        }
    }

    // Scalar tail.
    let tail_start = chunks * 4;
    for i in tail_start..min_len {
        if a[i] != b[i] {
            return matched + (i - tail_start);
        }
    }
    matched + (min_len - tail_start)
}

/// Scalar longest-common-prefix length.
pub fn prefix_match_v2_scalar(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

// ===================================================================
// 3. cache_copy_v2 — NEON bulk copy
// ===================================================================

/// Copy `len` f32 values from `src[src_off..]` to `dst[dst_off..]`.
pub fn cache_copy_v2(
    src: &[f32],
    dst: &mut [f32],
    src_off: usize,
    dst_off: usize,
    len: usize,
) {
    assert!(
        src_off + len <= src.len(),
        "source overflow: src_off={src_off} len={len} src.len={}",
        src.len()
    );
    assert!(
        dst_off + len <= dst.len(),
        "dest overflow: dst_off={dst_off} len={len} dst.len={}",
        dst.len()
    );

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                cache_copy_v2_neon(src, dst, src_off, dst_off, len);
            }
            return;
        }
    }
    cache_copy_v2_scalar(src, dst, src_off, dst_off, len);
}

#[cfg(target_arch = "aarch64")]
unsafe fn cache_copy_v2_neon(
    src: &[f32],
    dst: &mut [f32],
    src_off: usize,
    dst_off: usize,
    len: usize,
) {
    let s = src.as_ptr().add(src_off);
    let d = dst.as_mut_ptr().add(dst_off);
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let base = i * 4;
        let v = vld1q_f32(s.add(base));
        vst1q_f32(d.add(base), v);
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        *d.add(tail + i) = *s.add(tail + i);
    }
}

/// Scalar cache copy.
pub fn cache_copy_v2_scalar(
    src: &[f32],
    dst: &mut [f32],
    src_off: usize,
    dst_off: usize,
    len: usize,
) {
    dst[dst_off..dst_off + len].copy_from_slice(&src[src_off..src_off + len]);
}

// ===================================================================
// 4. cache_evict_lru — LRU eviction with NEON metadata scan
// ===================================================================

/// Find the LRU (least-recently-used) occupied slot. Returns `None` if
/// all slots are empty.
pub fn cache_evict_lru(entries: &[LruEntry]) -> Option<usize> {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { cache_evict_lru_neon(entries) };
        }
    }
    cache_evict_lru_scalar(entries)
}

#[cfg(target_arch = "aarch64")]
unsafe fn cache_evict_lru_neon(entries: &[LruEntry]) -> Option<usize> {
    // Extract timestamps into a contiguous vec so we can SIMD-scan them.
    // Non-occupied slots get u64::MAX so they are never selected.
    let n = entries.len();
    if n == 0 {
        return None;
    }

    let mut timestamps: Vec<u64> = entries
        .iter()
        .map(|e| if e.occupied { e.last_used } else { u64::MAX })
        .collect();

    // NEON works on 32-bit lanes; we only need to find the argmin, so
    // if all timestamps fit in u32 we can fast-path. Otherwise fall back
    // to scalar argmin.
    let all_fit_u32 = timestamps.iter().all(|&t| t <= u32::MAX as u64);

    if all_fit_u32 && n >= 4 {
        let ts32: Vec<u32> = timestamps.iter().map(|&t| t as u32).collect();
        let chunks = n / 4;
        let mut global_min = u32::MAX;

        for i in 0..chunks {
            let v = vld1q_u32(ts32.as_ptr().add(i * 4));
            let lane_min = vminvq_u32(v);
            if lane_min < global_min {
                global_min = lane_min;
            }
        }
        // Also check the tail.
        for i in (chunks * 4)..n {
            if ts32[i] < global_min {
                global_min = ts32[i];
            }
        }
        if global_min == u32::MAX {
            return None;
        }
        // Linear scan for the index (first occurrence).
        return ts32.iter().position(|&t| t == global_min);
    }

    // Fall back to scalar for large timestamps or tiny arrays.
    cache_evict_lru_scalar(entries)
}

/// Scalar LRU eviction: linear scan for minimum timestamp.
pub fn cache_evict_lru_scalar(entries: &[LruEntry]) -> Option<usize> {
    let mut best: Option<(usize, u64)> = None;
    for (i, e) in entries.iter().enumerate() {
        if e.occupied {
            match best {
                None => best = Some((i, e.last_used)),
                Some((_, ts)) if e.last_used < ts => best = Some((i, e.last_used)),
                _ => {}
            }
        }
    }
    best.map(|(idx, _)| idx)
}

// ===================================================================
// 5. radix_prefix_lookup — radix-tree prefix search
// ===================================================================

/// Insert `tokens` into the radix tree, associating with `cache_index`.
pub fn radix_insert(root: &mut RadixNode, tokens: &[u32], cache_index: usize) {
    if tokens.is_empty() {
        root.cache_index = Some(cache_index);
        return;
    }
    radix_insert_inner(root, tokens, cache_index);
}

fn radix_insert_inner(node: &mut RadixNode, tokens: &[u32], cache_index: usize) {
    if tokens.is_empty() {
        node.cache_index = Some(cache_index);
        return;
    }

    let first = tokens[0];

    if let Some(child) = node.find_child_mut(first) {
        let common = prefix_match_v2(&child.tokens, tokens);
        if common == child.tokens.len() {
            // Entire child edge consumed — recurse into child.
            radix_insert_inner(child, &tokens[common..], cache_index);
        } else {
            // Split the child edge at `common`.
            let old_suffix = child.tokens[common..].to_vec();
            let new_suffix = tokens[common..].to_vec();

            let mut split = RadixNode {
                tokens: child.tokens[..common].to_vec(),
                cache_index: None,
                children: Vec::new(),
            };

            // Move old child under split.
            let mut old_child = std::mem::replace(
                child,
                RadixNode {
                    tokens: Vec::new(),
                    cache_index: None,
                    children: Vec::new(),
                },
            );
            old_child.tokens = old_suffix.clone();
            split.children.push((old_suffix[0], old_child));

            if new_suffix.is_empty() {
                split.cache_index = Some(cache_index);
            } else {
                let leaf = RadixNode {
                    tokens: new_suffix.clone(),
                    cache_index: Some(cache_index),
                    children: Vec::new(),
                };
                split.children.push((new_suffix[0], leaf));
            }
            *child = split;
        }
    } else {
        if node.children.len() >= RADIX_MAX_CHILDREN {
            // Silently refuse when full (production would evict).
            return;
        }
        let leaf = RadixNode {
            tokens: tokens.to_vec(),
            cache_index: Some(cache_index),
            children: Vec::new(),
        };
        node.children.push((first, leaf));
    }
}

/// Look up the longest matching prefix for `tokens` in the radix tree.
/// Returns `(matched_len, cache_index)` of the deepest node that has a
/// `cache_index`.
pub fn radix_prefix_lookup(root: &RadixNode, tokens: &[u32]) -> Option<(usize, usize)> {
    let mut node = root;
    let mut pos: usize = 0;
    let mut best: Option<(usize, usize)> = root.cache_index.map(|ci| (0, ci));

    loop {
        if pos >= tokens.len() {
            break;
        }
        let first = tokens[pos];
        match node.find_child(first) {
            None => break,
            Some(idx) => {
                let child = &node.children[idx].1;
                let remaining = &tokens[pos..];
                let common = prefix_match_v2(&child.tokens, remaining);
                if common == 0 {
                    break;
                }
                pos += common;
                if common == child.tokens.len() {
                    if let Some(ci) = child.cache_index {
                        best = Some((pos, ci));
                    }
                    node = child;
                } else {
                    // Partial match inside an edge — stop.
                    break;
                }
            }
        }
    }

    best
}

// ===================================================================
// 6. batch_prefix_match — batch prefix matching
// ===================================================================

/// Result of a single prefix match within a batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchMatchResult {
    /// Index of the request in the batch.
    pub request_idx: usize,
    /// Number of tokens matched from the cache.
    pub matched_len: usize,
    /// Cache slot index (if any).
    pub cache_index: Option<usize>,
}

/// Match a batch of token sequences against a radix tree.
pub fn batch_prefix_match(
    root: &RadixNode,
    requests: &[&[u32]],
) -> Vec<BatchMatchResult> {
    requests
        .iter()
        .enumerate()
        .map(|(i, tokens)| {
            let lookup = radix_prefix_lookup(root, tokens);
            BatchMatchResult {
                request_idx: i,
                matched_len: lookup.map_or(0, |(len, _)| len),
                cache_index: lookup.map(|(_, ci)| ci),
            }
        })
        .collect()
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // prefix_hash_v2
    // ---------------------------------------------------------------

    #[test]
    fn test_hash_empty() {
        assert_eq!(prefix_hash_v2(&[]), FNV_OFFSET);
    }

    #[test]
    fn test_hash_single_token() {
        let h = prefix_hash_v2(&[42]);
        assert_ne!(h, FNV_OFFSET);
    }

    #[test]
    fn test_hash_deterministic() {
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(prefix_hash_v2(&tokens), prefix_hash_v2(&tokens));
    }

    #[test]
    fn test_hash_different_inputs_differ() {
        let a = prefix_hash_v2(&[1, 2, 3]);
        let b = prefix_hash_v2(&[4, 5, 6]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash_order_matters() {
        let a = prefix_hash_v2(&[1, 2]);
        let b = prefix_hash_v2(&[2, 1]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash_scalar_matches_dispatch() {
        let tokens = vec![10, 20, 30, 40, 50];
        assert_eq!(
            prefix_hash_v2(&tokens),
            prefix_hash_v2_scalar(&tokens),
        );
    }

    #[test]
    fn test_hash_large_sequence() {
        let tokens: Vec<u32> = (0..1024).collect();
        let h = prefix_hash_v2(&tokens);
        assert_ne!(h, FNV_OFFSET);
    }

    #[test]
    fn test_hash_collision_resistance_small() {
        let mut hashes = std::collections::HashSet::new();
        for i in 0u32..256 {
            hashes.insert(prefix_hash_v2(&[i]));
        }
        // At least 250/256 unique hashes (FNV-1a is well-distributed).
        assert!(hashes.len() >= 250, "too many collisions: {}", hashes.len());
    }

    #[test]
    fn test_hash_prefix_extension_changes() {
        let h1 = prefix_hash_v2(&[1, 2, 3]);
        let h2 = prefix_hash_v2(&[1, 2, 3, 4]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_all_zeros() {
        let h = prefix_hash_v2(&[0, 0, 0, 0]);
        assert_ne!(h, FNV_OFFSET);
    }

    #[test]
    fn test_hash_max_tokens() {
        let tokens = vec![u32::MAX; 8];
        let h = prefix_hash_v2(&tokens);
        assert_ne!(h, 0);
    }

    // ---------------------------------------------------------------
    // prefix_match_v2
    // ---------------------------------------------------------------

    #[test]
    fn test_match_identical() {
        let a = vec![1, 2, 3, 4];
        assert_eq!(prefix_match_v2(&a, &a), 4);
    }

    #[test]
    fn test_match_empty_a() {
        assert_eq!(prefix_match_v2(&[], &[1, 2]), 0);
    }

    #[test]
    fn test_match_empty_b() {
        assert_eq!(prefix_match_v2(&[1, 2], &[]), 0);
    }

    #[test]
    fn test_match_both_empty() {
        assert_eq!(prefix_match_v2(&[], &[]), 0);
    }

    #[test]
    fn test_match_partial() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 9, 10];
        assert_eq!(prefix_match_v2(&a, &b), 3);
    }

    #[test]
    fn test_match_no_common() {
        assert_eq!(prefix_match_v2(&[1], &[2]), 0);
    }

    #[test]
    fn test_match_different_lengths() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(prefix_match_v2(&a, &b), 3);
    }

    #[test]
    fn test_match_scalar_parity() {
        let a: Vec<u32> = (0..17).collect();
        let b: Vec<u32> = (0..13).collect();
        assert_eq!(
            prefix_match_v2(&a, &b),
            prefix_match_v2_scalar(&a, &b),
        );
    }

    #[test]
    fn test_match_single_token_same() {
        assert_eq!(prefix_match_v2(&[42], &[42]), 1);
    }

    #[test]
    fn test_match_large_common() {
        let a: Vec<u32> = (0..512).collect();
        let mut b = a.clone();
        b[500] = 999_999;
        assert_eq!(prefix_match_v2(&a, &b), 500);
    }

    #[test]
    fn test_match_mismatch_at_chunk_boundary() {
        let a = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut b = a.clone();
        b[4] = 99;
        assert_eq!(prefix_match_v2(&a, &b), 4);
    }

    // ---------------------------------------------------------------
    // cache_copy_v2
    // ---------------------------------------------------------------

    #[test]
    fn test_copy_basic() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![0.0f32; 5];
        cache_copy_v2(&src, &mut dst, 0, 0, 5);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_copy_with_offset() {
        let src = vec![0.0f32, 0.0, 10.0, 20.0, 30.0];
        let mut dst = vec![0.0f32; 5];
        cache_copy_v2(&src, &mut dst, 2, 1, 3);
        assert_eq!(dst, vec![0.0, 10.0, 20.0, 30.0, 0.0]);
    }

    #[test]
    fn test_copy_zero_length() {
        let src = vec![1.0f32; 4];
        let mut dst = vec![0.0f32; 4];
        cache_copy_v2(&src, &mut dst, 0, 0, 0);
        assert_eq!(dst, vec![0.0; 4]);
    }

    #[test]
    fn test_copy_large() {
        let n = 1024;
        let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; n];
        cache_copy_v2(&src, &mut dst, 0, 0, n);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_copy_scalar_parity() {
        let src: Vec<f32> = (0..33).map(|i| i as f32 * 0.5).collect();
        let mut dst_neon = vec![0.0f32; 33];
        let mut dst_scalar = vec![0.0f32; 33];
        cache_copy_v2(&src, &mut dst_neon, 0, 0, 33);
        cache_copy_v2_scalar(&src, &mut dst_scalar, 0, 0, 33);
        assert_eq!(dst_neon, dst_scalar);
    }

    #[test]
    #[should_panic(expected = "source overflow")]
    fn test_copy_src_overflow() {
        let src = vec![1.0f32; 4];
        let mut dst = vec![0.0f32; 8];
        cache_copy_v2(&src, &mut dst, 2, 0, 4);
    }

    #[test]
    #[should_panic(expected = "dest overflow")]
    fn test_copy_dst_overflow() {
        let src = vec![1.0f32; 8];
        let mut dst = vec![0.0f32; 2];
        cache_copy_v2(&src, &mut dst, 0, 0, 4);
    }

    #[test]
    fn test_copy_non_aligned_len() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut dst = vec![0.0f32; 7];
        cache_copy_v2(&src, &mut dst, 0, 0, 7);
        assert_eq!(dst, src);
    }

    // ---------------------------------------------------------------
    // cache_evict_lru
    // ---------------------------------------------------------------

    #[test]
    fn test_evict_empty() {
        assert_eq!(cache_evict_lru(&[]), None);
    }

    #[test]
    fn test_evict_none_occupied() {
        let entries = vec![LruEntry::default(); 4];
        assert_eq!(cache_evict_lru(&entries), None);
    }

    #[test]
    fn test_evict_single_occupied() {
        let mut entries = vec![LruEntry::default(); 4];
        entries[2].occupied = true;
        entries[2].last_used = 10;
        assert_eq!(cache_evict_lru(&entries), Some(2));
    }

    #[test]
    fn test_evict_picks_oldest() {
        let entries = vec![
            LruEntry { last_used: 5, prefix_len: 3, occupied: true },
            LruEntry { last_used: 1, prefix_len: 2, occupied: true },
            LruEntry { last_used: 9, prefix_len: 4, occupied: true },
        ];
        assert_eq!(cache_evict_lru(&entries), Some(1));
    }

    #[test]
    fn test_evict_skips_unoccupied() {
        let entries = vec![
            LruEntry { last_used: 100, prefix_len: 1, occupied: true },
            LruEntry { last_used: 0, prefix_len: 0, occupied: false },
            LruEntry { last_used: 50, prefix_len: 2, occupied: true },
        ];
        assert_eq!(cache_evict_lru(&entries), Some(2));
    }

    #[test]
    fn test_evict_first_of_ties() {
        let entries = vec![
            LruEntry { last_used: 3, prefix_len: 1, occupied: true },
            LruEntry { last_used: 3, prefix_len: 1, occupied: true },
            LruEntry { last_used: 5, prefix_len: 1, occupied: true },
        ];
        assert_eq!(cache_evict_lru(&entries), Some(0));
    }

    #[test]
    fn test_evict_scalar_parity() {
        let entries = vec![
            LruEntry { last_used: 10, prefix_len: 2, occupied: true },
            LruEntry { last_used: 2, prefix_len: 4, occupied: true },
            LruEntry { last_used: 7, prefix_len: 1, occupied: true },
            LruEntry { last_used: 15, prefix_len: 3, occupied: true },
            LruEntry { last_used: 1, prefix_len: 5, occupied: true },
        ];
        assert_eq!(
            cache_evict_lru(&entries),
            cache_evict_lru_scalar(&entries),
        );
    }

    #[test]
    fn test_evict_large_timestamps() {
        let entries = vec![
            LruEntry {
                last_used: u64::MAX - 1,
                prefix_len: 1,
                occupied: true,
            },
            LruEntry {
                last_used: u64::MAX - 2,
                prefix_len: 1,
                occupied: true,
            },
        ];
        assert_eq!(cache_evict_lru(&entries), Some(1));
    }

    // ---------------------------------------------------------------
    // radix tree: insert + lookup
    // ---------------------------------------------------------------

    #[test]
    fn test_radix_empty_tree() {
        let root = RadixNode::new_root();
        assert_eq!(radix_prefix_lookup(&root, &[1, 2, 3]), None);
    }

    #[test]
    fn test_radix_insert_single() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[1, 2, 3], 0);
        let result = radix_prefix_lookup(&root, &[1, 2, 3]);
        assert_eq!(result, Some((3, 0)));
    }

    #[test]
    fn test_radix_insert_empty_tokens() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[], 42);
        assert_eq!(root.cache_index, Some(42));
    }

    #[test]
    fn test_radix_partial_lookup() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[1, 2, 3, 4, 5], 0);
        // Query with prefix of the stored sequence.
        let result = radix_prefix_lookup(&root, &[1, 2, 3]);
        // No node at len=3, only at len=5.
        assert_eq!(result, None);
    }

    #[test]
    fn test_radix_superset_query() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[1, 2, 3], 7);
        let result = radix_prefix_lookup(&root, &[1, 2, 3, 4, 5]);
        assert_eq!(result, Some((3, 7)));
    }

    #[test]
    fn test_radix_two_disjoint() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[1, 2, 3], 0);
        radix_insert(&mut root, &[4, 5, 6], 1);
        assert_eq!(radix_prefix_lookup(&root, &[1, 2, 3]), Some((3, 0)));
        assert_eq!(radix_prefix_lookup(&root, &[4, 5, 6]), Some((3, 1)));
        assert_eq!(radix_prefix_lookup(&root, &[7, 8]), None);
    }

    #[test]
    fn test_radix_shared_prefix_split() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[1, 2, 3, 4], 0);
        radix_insert(&mut root, &[1, 2, 5, 6], 1);
        assert_eq!(radix_prefix_lookup(&root, &[1, 2, 3, 4]), Some((4, 0)));
        assert_eq!(radix_prefix_lookup(&root, &[1, 2, 5, 6]), Some((4, 1)));
    }

    #[test]
    fn test_radix_nested_prefixes() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[1, 2], 10);
        radix_insert(&mut root, &[1, 2, 3, 4], 20);
        assert_eq!(
            radix_prefix_lookup(&root, &[1, 2, 3, 4, 5]),
            Some((4, 20))
        );
        assert_eq!(radix_prefix_lookup(&root, &[1, 2, 9]), Some((2, 10)));
    }

    #[test]
    fn test_radix_overwrite() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[1, 2, 3], 0);
        radix_insert(&mut root, &[1, 2, 3], 99);
        assert_eq!(
            radix_prefix_lookup(&root, &[1, 2, 3]),
            Some((3, 99))
        );
    }

    #[test]
    fn test_radix_single_token() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[42], 5);
        assert_eq!(radix_prefix_lookup(&root, &[42]), Some((1, 5)));
        assert_eq!(radix_prefix_lookup(&root, &[43]), None);
    }

    #[test]
    fn test_radix_many_children() {
        let mut root = RadixNode::new_root();
        for i in 0u32..64 {
            radix_insert(&mut root, &[i, i + 100], i as usize);
        }
        for i in 0u32..64 {
            assert_eq!(
                radix_prefix_lookup(&root, &[i, i + 100]),
                Some((2, i as usize))
            );
        }
    }

    #[test]
    fn test_radix_deep_tree() {
        let mut root = RadixNode::new_root();
        let seq: Vec<u32> = (0..100).collect();
        radix_insert(&mut root, &seq, 0);
        assert_eq!(radix_prefix_lookup(&root, &seq), Some((100, 0)));
    }

    #[test]
    fn test_radix_no_match_wrong_first_token() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[1, 2, 3], 0);
        assert_eq!(radix_prefix_lookup(&root, &[9, 2, 3]), None);
    }

    // ---------------------------------------------------------------
    // batch_prefix_match
    // ---------------------------------------------------------------

    #[test]
    fn test_batch_empty_requests() {
        let root = RadixNode::new_root();
        let results = batch_prefix_match(&root, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_single_request() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[1, 2, 3], 0);
        let reqs: Vec<&[u32]> = vec![&[1, 2, 3, 4]];
        let results = batch_prefix_match(&root, &reqs);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].matched_len, 3);
        assert_eq!(results[0].cache_index, Some(0));
    }

    #[test]
    fn test_batch_mixed_hits() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[1, 2, 3], 0);
        radix_insert(&mut root, &[4, 5], 1);
        let reqs: Vec<&[u32]> = vec![
            &[1, 2, 3, 9],
            &[7, 8, 9],
            &[4, 5, 6],
        ];
        let results = batch_prefix_match(&root, &reqs);
        assert_eq!(results[0].matched_len, 3);
        assert_eq!(results[0].cache_index, Some(0));
        assert_eq!(results[1].matched_len, 0);
        assert_eq!(results[1].cache_index, None);
        assert_eq!(results[2].matched_len, 2);
        assert_eq!(results[2].cache_index, Some(1));
    }

    #[test]
    fn test_batch_preserves_request_idx() {
        let root = RadixNode::new_root();
        let reqs: Vec<&[u32]> = vec![&[1], &[2], &[3]];
        let results = batch_prefix_match(&root, &reqs);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.request_idx, i);
        }
    }

    #[test]
    fn test_batch_all_miss() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[100, 200], 0);
        let reqs: Vec<&[u32]> = vec![&[1], &[2], &[3]];
        let results = batch_prefix_match(&root, &reqs);
        for r in &results {
            assert_eq!(r.matched_len, 0);
            assert_eq!(r.cache_index, None);
        }
    }

    #[test]
    fn test_batch_all_hit() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[1, 2], 0);
        let reqs: Vec<&[u32]> = vec![&[1, 2, 3], &[1, 2, 4], &[1, 2]];
        let results = batch_prefix_match(&root, &reqs);
        for r in &results {
            assert_eq!(r.matched_len, 2);
            assert_eq!(r.cache_index, Some(0));
        }
    }

    #[test]
    fn test_batch_empty_prefix_request() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[1, 2], 0);
        let empty: &[u32] = &[];
        let reqs: Vec<&[u32]> = vec![empty];
        let results = batch_prefix_match(&root, &reqs);
        assert_eq!(results[0].matched_len, 0);
    }

    // ---------------------------------------------------------------
    // NEON vs scalar parity (dispatch-level)
    // ---------------------------------------------------------------

    #[test]
    fn test_hash_neon_scalar_parity_many() {
        for len in 0..32 {
            let tokens: Vec<u32> = (0..len).collect();
            assert_eq!(
                prefix_hash_v2(&tokens),
                prefix_hash_v2_scalar(&tokens),
                "parity fail at len={len}"
            );
        }
    }

    #[test]
    fn test_match_neon_scalar_parity_sweep() {
        let a: Vec<u32> = (0..64).collect();
        for split in 0..=64 {
            let mut b = a[..split].to_vec();
            b.extend(std::iter::repeat(999u32).take(64 - split));
            assert_eq!(
                prefix_match_v2(&a, &b),
                prefix_match_v2_scalar(&a, &b),
                "parity fail at split={split}"
            );
        }
    }

    #[test]
    fn test_copy_neon_scalar_parity_sweep() {
        for len in 0..33 {
            let src: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let mut d1 = vec![0.0f32; len];
            let mut d2 = vec![0.0f32; len];
            cache_copy_v2(&src, &mut d1, 0, 0, len);
            cache_copy_v2_scalar(&src, &mut d2, 0, 0, len);
            assert_eq!(d1, d2, "parity fail at len={len}");
        }
    }

    #[test]
    fn test_evict_neon_scalar_parity_sweep() {
        for n in 0..16 {
            let entries: Vec<LruEntry> = (0..n)
                .map(|i| LruEntry {
                    last_used: (n as u64).wrapping_sub(i as u64),
                    prefix_len: i as u32,
                    occupied: true,
                })
                .collect();
            assert_eq!(
                cache_evict_lru(&entries),
                cache_evict_lru_scalar(&entries),
                "parity fail at n={n}"
            );
        }
    }

    // ---------------------------------------------------------------
    // Edge-case and stress tests
    // ---------------------------------------------------------------

    #[test]
    fn test_hash_consecutive_values() {
        let h1 = prefix_hash_v2(&[0]);
        let h2 = prefix_hash_v2(&[1]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_match_one_element_arrays() {
        assert_eq!(prefix_match_v2(&[5], &[5]), 1);
        assert_eq!(prefix_match_v2(&[5], &[6]), 0);
    }

    #[test]
    fn test_copy_single_element() {
        let src = vec![42.0f32];
        let mut dst = vec![0.0f32];
        cache_copy_v2(&src, &mut dst, 0, 0, 1);
        assert_eq!(dst[0], 42.0);
    }

    #[test]
    fn test_evict_all_same_timestamp() {
        let entries = vec![
            LruEntry { last_used: 7, prefix_len: 1, occupied: true },
            LruEntry { last_used: 7, prefix_len: 2, occupied: true },
            LruEntry { last_used: 7, prefix_len: 3, occupied: true },
        ];
        // Picks the first one.
        assert_eq!(cache_evict_lru(&entries), Some(0));
    }

    #[test]
    fn test_radix_prefix_is_subset() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[10, 20, 30], 0);
        // Query with the exact stored prefix should work.
        assert_eq!(
            radix_prefix_lookup(&root, &[10, 20, 30]),
            Some((3, 0))
        );
    }

    #[test]
    fn test_radix_insert_then_partial_overlap() {
        let mut root = RadixNode::new_root();
        radix_insert(&mut root, &[1, 2, 3, 4], 0);
        radix_insert(&mut root, &[1, 2, 5, 6], 1);
        radix_insert(&mut root, &[1, 2], 2);
        // Now [1,2] has cache_index=2, acts as intermediate node.
        assert_eq!(radix_prefix_lookup(&root, &[1, 2, 9]), Some((2, 2)));
        assert_eq!(
            radix_prefix_lookup(&root, &[1, 2, 3, 4]),
            Some((4, 0))
        );
    }

    #[test]
    fn test_batch_large() {
        let mut root = RadixNode::new_root();
        for i in 0u32..32 {
            radix_insert(&mut root, &[i, i + 1], i as usize);
        }
        let reqs: Vec<Vec<u32>> = (0u32..32).map(|i| vec![i, i + 1, i + 2]).collect();
        let req_refs: Vec<&[u32]> = reqs.iter().map(|v| v.as_slice()).collect();
        let results = batch_prefix_match(&root, &req_refs);
        assert_eq!(results.len(), 32);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.matched_len, 2, "request {i}");
            assert_eq!(r.cache_index, Some(i), "request {i}");
        }
    }

    #[test]
    fn test_full_cache_eviction_cycle() {
        let n = 8;
        let mut entries: Vec<LruEntry> = (0..n)
            .map(|i| LruEntry {
                last_used: i as u64 * 10,
                prefix_len: 4,
                occupied: true,
            })
            .collect();
        // Evict should return index 0 (timestamp 0).
        assert_eq!(cache_evict_lru(&entries), Some(0));
        // Mark slot 0 as unused, re-evict.
        entries[0].occupied = false;
        assert_eq!(cache_evict_lru(&entries), Some(1));
    }

    #[test]
    fn test_hash_stability_across_calls() {
        let tokens = vec![7, 14, 21, 28, 35, 42, 49, 56];
        let first = prefix_hash_v2(&tokens);
        for _ in 0..100 {
            assert_eq!(prefix_hash_v2(&tokens), first);
        }
    }
}
