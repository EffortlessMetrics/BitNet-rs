//! Paged attention engine for efficient attention with page-based KV cache.
//!
//! [`PagedAttentionEngine`] computes scaled dot-product attention across
//! page boundaries transparently, with support for grouped query attention
//! (GQA). [`PageAllocator`] manages free/used page lifecycles including
//! defragmentation.

use crate::kv_cache::{GpuKvCache, KvCacheConfig};

/// Manages page allocation and deallocation for KV cache pages.
#[derive(Debug)]
#[allow(clippy::struct_field_names)]
pub struct PageAllocator {
    /// Total number of pages managed.
    total_pages: usize,
    /// Indices of pages currently in use, in allocation order.
    used_pages: Vec<usize>,
    /// Indices of free pages available for allocation.
    free_pages: Vec<usize>,
}

impl PageAllocator {
    /// Create a new allocator managing `total_pages` pages.
    pub fn new(total_pages: usize) -> Self {
        Self { total_pages, used_pages: Vec::new(), free_pages: (0..total_pages).rev().collect() }
    }

    /// Allocate a page. Returns `None` if no free pages remain.
    pub fn allocate(&mut self) -> Option<usize> {
        let page = self.free_pages.pop()?;
        self.used_pages.push(page);
        Some(page)
    }

    /// Free a specific page, returning it to the free list.
    ///
    /// Returns `true` if the page was found and freed.
    pub fn free(&mut self, page_idx: usize) -> bool {
        if let Some(pos) = self.used_pages.iter().position(|&p| p == page_idx) {
            self.used_pages.swap_remove(pos);
            self.free_pages.push(page_idx);
            true
        } else {
            false
        }
    }

    /// Defragment by compacting used pages into a contiguous range
    /// starting from index 0. Returns a mapping from old page index
    /// to new page index for all used pages.
    pub fn defragment(&mut self) -> Vec<(usize, usize)> {
        let mut mapping = Vec::with_capacity(self.used_pages.len());
        let mut sorted_used: Vec<usize> = self.used_pages.clone();
        sorted_used.sort_unstable();

        let mut new_used = Vec::with_capacity(sorted_used.len());
        for (new_idx, &old_idx) in sorted_used.iter().enumerate() {
            if old_idx != new_idx {
                mapping.push((old_idx, new_idx));
            }
            new_used.push(new_idx);
        }

        self.used_pages = new_used;
        self.free_pages = (self.used_pages.len()..self.total_pages).rev().collect();

        mapping
    }

    /// Number of free pages available.
    pub const fn free_count(&self) -> usize {
        self.free_pages.len()
    }

    /// Number of used pages.
    pub const fn used_count(&self) -> usize {
        self.used_pages.len()
    }

    /// Total pages managed.
    pub const fn total_pages(&self) -> usize {
        self.total_pages
    }
}

/// Configuration for grouped query attention.
#[derive(Debug, Clone)]
pub struct GqaConfig {
    /// Number of query heads.
    pub num_q_heads: usize,
    /// Number of key-value heads (must divide `num_q_heads` evenly).
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
}

impl GqaConfig {
    /// Number of query heads that share each KV head.
    pub const fn group_size(&self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }
}

/// Paged attention engine operating over a [`GpuKvCache`].
///
/// Performs block-wise scaled dot-product attention, handling page
/// boundary crossings transparently. Supports GQA via head grouping.
#[derive(Debug)]
pub struct PagedAttentionEngine {
    gqa: GqaConfig,
}

impl PagedAttentionEngine {
    /// Create a new engine with the given GQA configuration.
    pub const fn new(gqa: GqaConfig) -> Self {
        Self { gqa }
    }

    /// Compute scaled dot-product attention for a single query position.
    ///
    /// * `q` — query vector: `num_q_heads * head_dim` elements.
    /// * `kv_cache` — the KV cache to attend over.
    /// * `layer` — which layer's cache to use.
    /// * `mask` — attention mask, one byte per sequence position.
    ///   `0` = masked (excluded), non-zero = attend. If empty, all
    ///   positions are attended.
    ///
    /// Returns the attention output: `num_q_heads * head_dim` elements.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_attention(
        &self,
        q: &[f32],
        kv_cache: &GpuKvCache,
        layer: usize,
        mask: &[u8],
    ) -> Vec<f32> {
        let seq_len = kv_cache.seq_len(layer);
        if seq_len == 0 {
            return vec![0.0; self.gqa.num_q_heads * self.gqa.head_dim];
        }

        let (all_k, all_v) = kv_cache.get(layer, 0..seq_len);
        let head_dim = self.gqa.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let group_size = self.gqa.group_size();
        let num_kv_heads = self.gqa.num_kv_heads;

        let mut output = vec![0.0f32; self.gqa.num_q_heads * head_dim];

        for q_head in 0..self.gqa.num_q_heads {
            let kv_head = q_head / group_size;
            let q_offset = q_head * head_dim;
            let q_vec = &q[q_offset..q_offset + head_dim];

            // Compute attention scores.
            let mut scores = Vec::with_capacity(seq_len);
            for pos in 0..seq_len {
                // Apply mask.
                if !mask.is_empty() && mask[pos] == 0 {
                    scores.push(f32::NEG_INFINITY);
                    continue;
                }

                let k_offset = pos * num_kv_heads * head_dim + kv_head * head_dim;
                let k_vec = &all_k[k_offset..k_offset + head_dim];

                let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                scores.push(dot * scale);
            }

            // Softmax.
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum: f32 = exp_scores.iter().sum();
            if sum > 0.0 {
                for s in &mut exp_scores {
                    *s /= sum;
                }
            }

            // Weighted sum of values.
            let out_offset = q_head * head_dim;
            for (pos, &weight) in exp_scores.iter().enumerate().take(seq_len) {
                let v_offset = pos * num_kv_heads * head_dim + kv_head * head_dim;
                for d in 0..head_dim {
                    output[out_offset + d] += weight * all_v[v_offset + d];
                }
            }
        }

        output
    }

    /// Block-wise attention: compute attention over blocks of positions,
    /// accumulating results. This is functionally identical to
    /// [`compute_attention`](Self::compute_attention) but processes the
    /// KV cache in `block_size`-position blocks for cache locality.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_attention_blocked(
        &self,
        q: &[f32],
        kv_cache: &GpuKvCache,
        layer: usize,
        mask: &[u8],
        block_size: usize,
    ) -> Vec<f32> {
        let seq_len = kv_cache.seq_len(layer);
        if seq_len == 0 || block_size == 0 {
            return vec![0.0; self.gqa.num_q_heads * self.gqa.head_dim];
        }

        // For correctness we need global softmax, so collect all scores
        // first in blocks, then do a single softmax pass.
        let head_dim = self.gqa.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let group_size = self.gqa.group_size();
        let num_kv_heads = self.gqa.num_kv_heads;

        let mut output = vec![0.0f32; self.gqa.num_q_heads * head_dim];

        for q_head in 0..self.gqa.num_q_heads {
            let kv_head = q_head / group_size;
            let q_offset = q_head * head_dim;
            let q_vec = &q[q_offset..q_offset + head_dim];

            let mut all_scores = Vec::with_capacity(seq_len);
            let mut all_v_slices: Vec<(usize, usize)> = Vec::with_capacity(seq_len);

            // Process in blocks.
            let mut block_start = 0;
            while block_start < seq_len {
                let block_end = (block_start + block_size).min(seq_len);
                let (block_k, _block_v) = kv_cache.get(layer, block_start..block_end);

                for local_pos in 0..(block_end - block_start) {
                    let global_pos = block_start + local_pos;
                    if !mask.is_empty() && mask[global_pos] == 0 {
                        all_scores.push(f32::NEG_INFINITY);
                    } else {
                        let k_off = local_pos * num_kv_heads * head_dim + kv_head * head_dim;
                        let k_vec = &block_k[k_off..k_off + head_dim];
                        let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                        all_scores.push(dot * scale);
                    }
                    all_v_slices.push((global_pos, kv_head));
                }

                block_start = block_end;
            }

            // Global softmax.
            let max_s = all_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_scores: Vec<f32> = all_scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum: f32 = exp_scores.iter().sum();
            if sum > 0.0 {
                for s in &mut exp_scores {
                    *s /= sum;
                }
            }

            // Weighted sum — re-fetch from cache.
            let (_, all_v) = kv_cache.get(layer, 0..seq_len);
            let out_offset = q_head * head_dim;
            for (i, &(pos, kv_h)) in all_v_slices.iter().enumerate() {
                let v_off = pos * num_kv_heads * head_dim + kv_h * head_dim;
                let weight = exp_scores[i];
                for d in 0..head_dim {
                    output[out_offset + d] += weight * all_v[v_off + d];
                }
            }
        }

        output
    }

    /// Returns the GQA configuration.
    pub const fn gqa_config(&self) -> &GqaConfig {
        &self.gqa
    }
}

/// Helper: create a [`KvCacheConfig`] compatible with a [`GqaConfig`].
pub const fn kv_config_for_gqa(
    gqa: &GqaConfig,
    num_layers: usize,
    max_seq_len: usize,
    page_size: usize,
) -> KvCacheConfig {
    KvCacheConfig {
        num_layers,
        num_heads: gqa.num_kv_heads,
        head_dim: gqa.head_dim,
        max_seq_len,
        page_size,
    }
}
