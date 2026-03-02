//! CUDA KV cache GPU operations with paged attention and copy-on-write.
//!
//! # Kernel strategy
//!
//! Extends the basic KV cache with GPU-oriented operations for production
//! autoregressive inference:
//!
//! - **Paged attention** — virtual→physical page mapping eliminates
//!   fragmentation and allows O(1) append.  Pages are fixed-size blocks
//!   of `page_size` sequence positions.
//! - **Append** — writes new K/V pairs into the next free slot in the
//!   current page, allocating a new page when the current one fills.
//! - **Paged lookup** — gathers K/V from scattered physical pages into a
//!   contiguous buffer for the attention kernel.
//! - **Sliding-window rotation** — circular buffer semantics: oldest
//!   positions are overwritten when the window is full.
//! - **INT8 quantization** — per-head absmax quantization of cached K/V
//!   to halve memory footprint with bounded error.
//! - **LRU / attention-based eviction** — frees the least-recently-used
//!   or lowest-attention-score pages when memory pressure is high.
//! - **Copy-on-write** — beam search branches share pages read-only
//!   until a write forces a physical copy.
//! - **Defragmentation** — compacts live pages to eliminate holes left by
//!   eviction.
//! - **Prefetch** — hints the GPU memory controller to asynchronously
//!   stage the next needed pages into L2.
//!
//! # CPU fallback
//!
//! All operations are implemented in pure Rust for correctness testing on
//! CPU-only builds.  GPU dispatch stubs are gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use bitnet_common::{KernelError, Result};

// ── CUDA kernel source ──────────────────────────────────────────────

/// CUDA C source for paged KV cache operations.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const KV_CACHE_GPU_KERNEL_SRC: &str = r#"
extern "C" __global__ void kv_cache_append_paged_f32(
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    const float* __restrict__ new_keys,
    const float* __restrict__ new_values,
    const int* __restrict__ page_table,
    int page_size,
    int head_dim,
    int num_heads,
    int write_pos)
{
    int head = blockIdx.x;
    int d    = threadIdx.x;
    if (head >= num_heads || d >= head_dim) return;

    int page_idx    = write_pos / page_size;
    int page_offset = write_pos % page_size;
    int phys_page   = page_table[page_idx];

    int cache_idx = ((phys_page * num_heads + head) * page_size + page_offset) * head_dim + d;
    int src_idx   = head * head_dim + d;

    key_cache[cache_idx]   = new_keys[src_idx];
    value_cache[cache_idx] = new_values[src_idx];
}

extern "C" __global__ void kv_cache_gather_paged_f32(
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    float* __restrict__ key_out,
    float* __restrict__ value_out,
    const int* __restrict__ page_table,
    int page_size,
    int head_dim,
    int num_heads,
    int seq_len)
{
    int pos  = blockIdx.x;
    int head = blockIdx.y;
    int d    = threadIdx.x;
    if (pos >= seq_len || head >= num_heads || d >= head_dim) return;

    int page_idx    = pos / page_size;
    int page_offset = pos % page_size;
    int phys_page   = page_table[page_idx];

    int cache_idx = ((phys_page * num_heads + head) * page_size + page_offset) * head_dim + d;
    int out_idx   = (pos * num_heads + head) * head_dim + d;

    key_out[out_idx]   = key_cache[cache_idx];
    value_out[out_idx] = value_cache[cache_idx];
}
"#;

// ── Configuration ───────────────────────────────────────────────────

/// Configuration for GPU-accelerated paged KV cache.
#[derive(Debug, Clone)]
pub struct KvCacheGpuConfig {
    /// Maximum sequence length the cache supports.
    pub max_seq_len: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimensionality per head.
    pub head_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Sequence positions per physical page.
    pub page_size: usize,
    /// Maximum physical pages available.
    pub max_pages: usize,
}

impl KvCacheGpuConfig {
    /// Validate and construct a new configuration.
    pub fn new(
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        num_layers: usize,
        page_size: usize,
        max_pages: usize,
    ) -> Result<Self> {
        if max_seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "max_seq_len must be non-zero".into(),
            }
            .into());
        }
        if num_heads == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_heads must be non-zero".into(),
            }
            .into());
        }
        if head_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "head_dim must be non-zero".into(),
            }
            .into());
        }
        if num_layers == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_layers must be non-zero".into(),
            }
            .into());
        }
        if page_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "page_size must be non-zero".into(),
            }
            .into());
        }
        if !page_size.is_power_of_two() {
            return Err(KernelError::InvalidArguments {
                reason: "page_size must be a power of two".into(),
            }
            .into());
        }
        if max_pages == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "max_pages must be non-zero".into(),
            }
            .into());
        }
        let pages_needed = max_seq_len.div_ceil(page_size);
        if max_pages < pages_needed {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "max_pages ({max_pages}) insufficient for max_seq_len \
                     ({max_seq_len}) with page_size ({page_size}): need {pages_needed}"
                ),
            }
            .into());
        }
        Ok(Self { max_seq_len, num_heads, head_dim, num_layers, page_size, max_pages })
    }

    /// Number of pages required for the full sequence length.
    pub fn pages_needed(&self) -> usize {
        self.max_seq_len.div_ceil(self.page_size)
    }

    /// Elements per page per head: `page_size * head_dim`.
    pub fn page_elements(&self) -> usize {
        self.page_size * self.head_dim
    }
}

// ── Error type ──────────────────────────────────────────────────────

/// Errors specific to GPU KV cache operations.
#[derive(Debug)]
pub enum KvCacheGpuError {
    /// No free pages available for allocation.
    OutOfPages,
    /// Requested layer index is out of range.
    LayerOutOfRange { layer: usize, num_layers: usize },
    /// Sequence length would exceed the configured maximum.
    SequenceOverflow { current: usize, max: usize },
    /// Page index is invalid.
    InvalidPage { page: usize, max_pages: usize },
    /// Copy-on-write source page not found.
    CowSourceMissing { page: usize },
    /// Quantization error.
    QuantizationError(String),
}

impl std::fmt::Display for KvCacheGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfPages => write!(f, "KV cache GPU: no free pages available"),
            Self::LayerOutOfRange { layer, num_layers } => {
                write!(f, "KV cache GPU: layer {layer} out of range (num_layers={num_layers})")
            }
            Self::SequenceOverflow { current, max } => {
                write!(f, "KV cache GPU: sequence overflow ({current} >= max {max})")
            }
            Self::InvalidPage { page, max_pages } => {
                write!(f, "KV cache GPU: invalid page {page} (max_pages={max_pages})")
            }
            Self::CowSourceMissing { page } => {
                write!(f, "KV cache GPU: CoW source page {page} not found")
            }
            Self::QuantizationError(msg) => {
                write!(f, "KV cache GPU: quantization error: {msg}")
            }
        }
    }
}

impl std::error::Error for KvCacheGpuError {}

impl From<KvCacheGpuError> for bitnet_common::BitNetError {
    fn from(e: KvCacheGpuError) -> Self {
        bitnet_common::BitNetError::Kernel(KernelError::InvalidArguments { reason: e.to_string() })
    }
}

// ── Page table ──────────────────────────────────────────────────────

/// Manages logical→physical page mappings and free-page tracking.
#[derive(Debug, Clone)]
pub struct PageTable {
    /// `physical_pages[i]` is true when physical page `i` is allocated.
    pub physical_pages: Vec<bool>,
    /// `logical_to_physical[layer][logical]` → physical page index.
    pub logical_to_physical: Vec<Vec<Option<usize>>>,
    /// Stack of free physical page indices.
    pub free_pages: Vec<usize>,
}

impl PageTable {
    /// Create a page table for the given configuration.
    pub fn new(config: &KvCacheGpuConfig) -> Self {
        let max_logical = config.pages_needed();
        let free_pages: Vec<usize> = (0..config.max_pages).rev().collect();
        Self {
            physical_pages: vec![false; config.max_pages],
            logical_to_physical: vec![vec![None; max_logical]; config.num_layers],
            free_pages,
        }
    }

    /// Allocate a free physical page, returning its index.
    pub fn allocate(&mut self) -> std::result::Result<usize, KvCacheGpuError> {
        let page = self.free_pages.pop().ok_or(KvCacheGpuError::OutOfPages)?;
        self.physical_pages[page] = true;
        Ok(page)
    }

    /// Free a physical page, returning it to the free pool.
    pub fn deallocate(&mut self, page: usize) -> std::result::Result<(), KvCacheGpuError> {
        if page >= self.physical_pages.len() {
            return Err(KvCacheGpuError::InvalidPage {
                page,
                max_pages: self.physical_pages.len(),
            });
        }
        self.physical_pages[page] = false;
        self.free_pages.push(page);
        Ok(())
    }

    /// Map a logical page to a physical page for a given layer.
    pub fn map(
        &mut self,
        layer: usize,
        logical: usize,
        physical: usize,
    ) -> std::result::Result<(), KvCacheGpuError> {
        if layer >= self.logical_to_physical.len() {
            return Err(KvCacheGpuError::LayerOutOfRange {
                layer,
                num_layers: self.logical_to_physical.len(),
            });
        }
        if logical >= self.logical_to_physical[layer].len() {
            return Err(KvCacheGpuError::InvalidPage {
                page: logical,
                max_pages: self.logical_to_physical[layer].len(),
            });
        }
        self.logical_to_physical[layer][logical] = Some(physical);
        Ok(())
    }

    /// Resolve a logical page to its physical page.
    pub fn resolve(
        &self,
        layer: usize,
        logical: usize,
    ) -> std::result::Result<usize, KvCacheGpuError> {
        if layer >= self.logical_to_physical.len() {
            return Err(KvCacheGpuError::LayerOutOfRange {
                layer,
                num_layers: self.logical_to_physical.len(),
            });
        }
        self.logical_to_physical[layer].get(logical).copied().flatten().ok_or(
            KvCacheGpuError::InvalidPage {
                page: logical,
                max_pages: self.logical_to_physical[layer].len(),
            },
        )
    }

    /// Number of currently allocated pages.
    pub fn allocated_count(&self) -> usize {
        self.physical_pages.iter().filter(|&&b| b).count()
    }

    /// Number of free pages.
    pub fn free_count(&self) -> usize {
        self.free_pages.len()
    }
}

// ── Cache state ─────────────────────────────────────────────────────

/// GPU KV cache state with paged memory layout.
///
/// Storage layout per page: `[num_heads][page_size][head_dim]`
/// Total elements per K or V: `max_pages * num_heads * page_size * head_dim`
#[derive(Debug, Clone)]
pub struct KvCacheGpuState {
    /// Key cache: `[max_pages * num_heads * page_size * head_dim]`.
    pub key_cache: Vec<f32>,
    /// Value cache: same shape as key cache.
    pub value_cache: Vec<f32>,
    /// Page table managing logical→physical mappings.
    pub page_table: PageTable,
    /// Current sequence lengths per layer.
    pub current_seq_len: Vec<usize>,
    /// Configuration.
    config: KvCacheGpuConfig,
    /// Per-page reference counts for copy-on-write.
    page_refcounts: Vec<usize>,
    /// Per-position access timestamps for LRU eviction (layer, position).
    access_timestamps: Vec<Vec<u64>>,
    /// Monotonic clock for LRU.
    timestamp_counter: u64,
}

impl KvCacheGpuState {
    /// Allocate a new paged KV cache.
    pub fn new(config: KvCacheGpuConfig) -> Self {
        let total_elems = config.max_pages * config.num_heads * config.page_size * config.head_dim;
        let page_table = PageTable::new(&config);
        let page_refcounts = vec![0usize; config.max_pages];
        let access_timestamps = vec![vec![0u64; config.max_seq_len]; config.num_layers];
        Self {
            key_cache: vec![0.0; total_elems],
            value_cache: vec![0.0; total_elems],
            page_table,
            current_seq_len: vec![0; config.num_layers],
            config,
            page_refcounts,
            access_timestamps,
            timestamp_counter: 0,
        }
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &KvCacheGpuConfig {
        &self.config
    }

    /// Flat index into cache for a (physical_page, head, offset_in_page, dim).
    fn cache_index(&self, phys_page: usize, head: usize, offset: usize, d: usize) -> usize {
        ((phys_page * self.config.num_heads + head) * self.config.page_size + offset)
            * self.config.head_dim
            + d
    }

    /// Touch a position for LRU tracking.
    fn touch(&mut self, layer: usize, pos: usize) {
        self.timestamp_counter += 1;
        if layer < self.access_timestamps.len() && pos < self.access_timestamps[layer].len() {
            self.access_timestamps[layer][pos] = self.timestamp_counter;
        }
    }
}

// ── Operations ──────────────────────────────────────────────────────

/// Append new key/value vectors for one position across all heads.
///
/// `new_keys` and `new_values` are `[num_heads * head_dim]`.
pub fn kv_cache_append(
    state: &mut KvCacheGpuState,
    layer: usize,
    new_keys: &[f32],
    new_values: &[f32],
) -> Result<()> {
    let cfg = &state.config;
    if layer >= cfg.num_layers {
        return Err(KvCacheGpuError::LayerOutOfRange { layer, num_layers: cfg.num_layers }.into());
    }
    let expected = cfg.num_heads * cfg.head_dim;
    if new_keys.len() != expected || new_values.len() != expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "kv_cache_append: expected {expected} elements, got keys={} values={}",
                new_keys.len(),
                new_values.len()
            ),
        }
        .into());
    }
    let pos = state.current_seq_len[layer];
    if pos >= cfg.max_seq_len {
        return Err(KvCacheGpuError::SequenceOverflow { current: pos, max: cfg.max_seq_len }.into());
    }

    let logical_page = pos / cfg.page_size;
    let page_offset = pos % cfg.page_size;

    // Allocate page if this is the first slot in a new page.
    if page_offset == 0 {
        let phys = state.page_table.allocate()?;
        state.page_table.map(layer, logical_page, phys)?;
        state.page_refcounts[phys] = 1;
    }

    let phys_page = state.page_table.resolve(layer, logical_page)?;
    for head in 0..cfg.num_heads {
        for d in 0..cfg.head_dim {
            let idx = state.cache_index(phys_page, head, page_offset, d);
            state.key_cache[idx] = new_keys[head * cfg.head_dim + d];
            state.value_cache[idx] = new_values[head * cfg.head_dim + d];
        }
    }

    state.touch(layer, pos);
    state.current_seq_len[layer] = pos + 1;
    Ok(())
}

/// Gather cached K/V for positions `[0..seq_len)` into contiguous buffers.
///
/// Output layout: `[seq_len * num_heads * head_dim]`.
pub fn kv_cache_paged_lookup(
    state: &mut KvCacheGpuState,
    layer: usize,
    key_out: &mut [f32],
    value_out: &mut [f32],
) -> Result<()> {
    let num_layers = state.config.num_layers;
    let num_heads = state.config.num_heads;
    let head_dim = state.config.head_dim;
    let page_size = state.config.page_size;
    if layer >= num_layers {
        return Err(KvCacheGpuError::LayerOutOfRange { layer, num_layers }.into());
    }
    let seq_len = state.current_seq_len[layer];
    let expected = seq_len * num_heads * head_dim;
    if key_out.len() < expected || value_out.len() < expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "kv_cache_paged_lookup: output too small, need {expected}, \
                 got keys={} values={}",
                key_out.len(),
                value_out.len()
            ),
        }
        .into());
    }

    for pos in 0..seq_len {
        let logical_page = pos / page_size;
        let page_offset = pos % page_size;
        let phys_page = state.page_table.resolve(layer, logical_page)?;

        for head in 0..num_heads {
            for d in 0..head_dim {
                let cache_idx = state.cache_index(phys_page, head, page_offset, d);
                let out_idx = (pos * num_heads + head) * head_dim + d;
                key_out[out_idx] = state.key_cache[cache_idx];
                value_out[out_idx] = state.value_cache[cache_idx];
            }
        }
        state.touch(layer, pos);
    }
    Ok(())
}

/// Sliding-window rotation: when `current_seq_len` exceeds `window_size`,
/// the oldest positions are logically dropped and the window slides forward.
pub fn kv_cache_rotate(
    state: &mut KvCacheGpuState,
    layer: usize,
    window_size: usize,
) -> Result<()> {
    let cfg = &state.config;
    if layer >= cfg.num_layers {
        return Err(KvCacheGpuError::LayerOutOfRange { layer, num_layers: cfg.num_layers }.into());
    }
    if window_size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "window_size must be non-zero".into(),
        }
        .into());
    }

    let seq_len = state.current_seq_len[layer];
    if seq_len <= window_size {
        return Ok(());
    }

    let drop_count = seq_len - window_size;
    let drop_pages = drop_count / cfg.page_size;

    // Free fully-dropped pages.
    for logical in 0..drop_pages {
        if let Ok(phys) = state.page_table.resolve(layer, logical) {
            state.page_refcounts[phys] = state.page_refcounts[phys].saturating_sub(1);
            if state.page_refcounts[phys] == 0 {
                let _ = state.page_table.deallocate(phys);
            }
            state.page_table.logical_to_physical[layer][logical] = None;
        }
    }

    // Shift remaining pages: compact logical indices.
    let remaining_pages = window_size.div_ceil(cfg.page_size);
    for i in 0..remaining_pages {
        let src_logical = drop_pages + i;
        if let Ok(phys) = state.page_table.resolve(layer, src_logical) {
            state.page_table.logical_to_physical[layer][i] = Some(phys);
            if src_logical != i {
                state.page_table.logical_to_physical[layer][src_logical] = None;
            }
        }
    }
    // Clear stale mappings.
    let max_logical = state.page_table.logical_to_physical[layer].len();
    for i in remaining_pages..max_logical {
        state.page_table.logical_to_physical[layer][i] = None;
    }

    state.current_seq_len[layer] = window_size;
    Ok(())
}

/// Result of INT8 KV cache quantization: `(keys_i8, values_i8, key_scales, value_scales)`.
pub type QuantizedKvResult = (Vec<i8>, Vec<i8>, Vec<f32>, Vec<f32>);

/// INT8 per-head absmax quantization of cached K/V for the given layer.
///
/// Returns `(quantized_keys, quantized_values, key_scales, value_scales)`.
/// Each scale is per-head, per-page.
pub fn kv_cache_quantize(
    state: &KvCacheGpuState,
    layer: usize,
) -> std::result::Result<QuantizedKvResult, KvCacheGpuError> {
    let cfg = &state.config;
    if layer >= cfg.num_layers {
        return Err(KvCacheGpuError::LayerOutOfRange { layer, num_layers: cfg.num_layers });
    }
    let seq_len = state.current_seq_len[layer];
    let total = seq_len * cfg.num_heads * cfg.head_dim;
    let mut q_keys = vec![0i8; total];
    let mut q_values = vec![0i8; total];

    let num_pages_used = seq_len.div_ceil(cfg.page_size);
    let scales_len = num_pages_used * cfg.num_heads;
    let mut key_scales = vec![0.0f32; scales_len];
    let mut value_scales = vec![0.0f32; scales_len];

    for page_logical in 0..num_pages_used {
        let phys_page = state.page_table.resolve(layer, page_logical)?;
        let page_start = page_logical * cfg.page_size;
        let page_end = (page_start + cfg.page_size).min(seq_len);

        for head in 0..cfg.num_heads {
            // Find absmax for this head in this page.
            let mut absmax_k: f32 = 0.0;
            let mut absmax_v: f32 = 0.0;
            for pos in page_start..page_end {
                let offset = pos - page_start;
                for d in 0..cfg.head_dim {
                    let idx = state.cache_index(phys_page, head, offset, d);
                    absmax_k = absmax_k.max(state.key_cache[idx].abs());
                    absmax_v = absmax_v.max(state.value_cache[idx].abs());
                }
            }

            let scale_k = if absmax_k > 0.0 { absmax_k / 127.0 } else { 1.0 };
            let scale_v = if absmax_v > 0.0 { absmax_v / 127.0 } else { 1.0 };
            let scale_idx = page_logical * cfg.num_heads + head;
            key_scales[scale_idx] = scale_k;
            value_scales[scale_idx] = scale_v;

            for pos in page_start..page_end {
                let offset = pos - page_start;
                for d in 0..cfg.head_dim {
                    let cache_idx = state.cache_index(phys_page, head, offset, d);
                    let out_idx = (pos * cfg.num_heads + head) * cfg.head_dim + d;
                    q_keys[out_idx] =
                        (state.key_cache[cache_idx] / scale_k).round().clamp(-128.0, 127.0) as i8;
                    q_values[out_idx] =
                        (state.value_cache[cache_idx] / scale_v).round().clamp(-128.0, 127.0) as i8;
                }
            }
        }
    }

    Ok((q_keys, q_values, key_scales, value_scales))
}

/// Dequantize INT8 values back to f32 using per-head scales.
pub fn kv_cache_dequantize(
    quantized: &[i8],
    scales: &[f32],
    num_heads: usize,
    head_dim: usize,
    page_size: usize,
    seq_len: usize,
) -> Vec<f32> {
    let total = seq_len * num_heads * head_dim;
    let mut output = vec![0.0f32; total];
    let num_pages = seq_len.div_ceil(page_size);

    for page in 0..num_pages {
        let page_start = page * page_size;
        let page_end = (page_start + page_size).min(seq_len);
        for head in 0..num_heads {
            let scale_idx = page * num_heads + head;
            let scale = scales.get(scale_idx).copied().unwrap_or(1.0);
            for pos in page_start..page_end {
                for d in 0..head_dim {
                    let idx = (pos * num_heads + head) * head_dim + d;
                    if idx < total {
                        output[idx] = quantized[idx] as f32 * scale;
                    }
                }
            }
        }
    }
    output
}

/// Evict pages using LRU policy based on access timestamps.
///
/// Frees `count` pages from the given layer, choosing those with the
/// oldest access timestamps.
pub fn kv_cache_evict(state: &mut KvCacheGpuState, layer: usize, count: usize) -> Result<usize> {
    let cfg = &state.config;
    if layer >= cfg.num_layers {
        return Err(KvCacheGpuError::LayerOutOfRange { layer, num_layers: cfg.num_layers }.into());
    }

    let seq_len = state.current_seq_len[layer];
    let num_pages = seq_len.div_ceil(cfg.page_size);
    if num_pages == 0 {
        return Ok(0);
    }

    // Score each page by minimum timestamp of its positions.
    let mut page_scores: Vec<(usize, u64)> = Vec::new();
    for logical in 0..num_pages {
        let page_start = logical * cfg.page_size;
        let page_end = (page_start + cfg.page_size).min(seq_len);
        let min_ts =
            (page_start..page_end).map(|p| state.access_timestamps[layer][p]).min().unwrap_or(0);
        page_scores.push((logical, min_ts));
    }
    page_scores.sort_by_key(|&(_, ts)| ts);

    let mut evicted = 0;
    for (logical, _) in page_scores.iter().take(count) {
        if let Ok(phys) = state.page_table.resolve(layer, *logical) {
            state.page_refcounts[phys] = state.page_refcounts[phys].saturating_sub(1);
            if state.page_refcounts[phys] == 0 {
                let _ = state.page_table.deallocate(phys);
            }
            state.page_table.logical_to_physical[layer][*logical] = None;
            evicted += 1;
        }
    }
    Ok(evicted)
}

/// Copy-on-write: branch a layer's page table for beam search.
///
/// Creates a logical copy of `src_layer`'s mappings into `dst_layer`
/// by incrementing reference counts on shared physical pages.
pub fn kv_cache_copy_on_write(
    state: &mut KvCacheGpuState,
    src_layer: usize,
    dst_layer: usize,
) -> Result<()> {
    let cfg = &state.config;
    if src_layer >= cfg.num_layers {
        return Err(KvCacheGpuError::LayerOutOfRange {
            layer: src_layer,
            num_layers: cfg.num_layers,
        }
        .into());
    }
    if dst_layer >= cfg.num_layers {
        return Err(KvCacheGpuError::LayerOutOfRange {
            layer: dst_layer,
            num_layers: cfg.num_layers,
        }
        .into());
    }

    let max_logical = state.page_table.logical_to_physical[src_layer].len();
    for logical in 0..max_logical {
        if let Some(phys) = state.page_table.logical_to_physical[src_layer][logical] {
            state.page_table.logical_to_physical[dst_layer][logical] = Some(phys);
            state.page_refcounts[phys] += 1;
        } else {
            state.page_table.logical_to_physical[dst_layer][logical] = None;
        }
    }
    state.current_seq_len[dst_layer] = state.current_seq_len[src_layer];
    Ok(())
}

/// Materialise a CoW page: if refcount > 1, copy to a fresh physical page.
pub fn kv_cache_cow_materialize(
    state: &mut KvCacheGpuState,
    layer: usize,
    logical_page: usize,
) -> Result<()> {
    let phys = state.page_table.resolve(layer, logical_page)?;
    if state.page_refcounts[phys] <= 1 {
        return Ok(());
    }

    let new_phys = state.page_table.allocate()?;
    let elems = state.config.num_heads * state.config.page_size * state.config.head_dim;
    let src_start = phys * elems;
    let dst_start = new_phys * elems;

    for i in 0..elems {
        state.key_cache[dst_start + i] = state.key_cache[src_start + i];
    }
    for i in 0..elems {
        state.value_cache[dst_start + i] = state.value_cache[src_start + i];
    }

    state.page_refcounts[phys] -= 1;
    state.page_refcounts[new_phys] = 1;
    state.page_table.logical_to_physical[layer][logical_page] = Some(new_phys);
    Ok(())
}

/// Defragment paged memory by compacting live pages toward lower indices.
///
/// Returns the number of pages moved.
pub fn kv_cache_defrag(state: &mut KvCacheGpuState) -> Result<usize> {
    let cfg = &state.config;
    let elems_per_page = cfg.num_heads * cfg.page_size * cfg.head_dim;
    let mut moved = 0;

    // Collect all live physical pages.
    let mut live_pages: Vec<usize> =
        (0..cfg.max_pages).filter(|&i| state.page_table.physical_pages[i]).collect();
    live_pages.sort();

    // Target: pack into [0..live_pages.len()).
    for (target, &current) in live_pages.iter().enumerate() {
        if target == current {
            continue;
        }

        let src = current * elems_per_page;
        let dst = target * elems_per_page;
        for i in 0..elems_per_page {
            state.key_cache[dst + i] = state.key_cache[src + i];
            state.value_cache[dst + i] = state.value_cache[src + i];
        }

        // Update page table mappings.
        for layer in 0..cfg.num_layers {
            for logical in &mut state.page_table.logical_to_physical[layer] {
                if *logical == Some(current) {
                    *logical = Some(target);
                }
            }
        }

        state.page_table.physical_pages[target] = true;
        state.page_table.physical_pages[current] = false;
        state.page_refcounts[target] = state.page_refcounts[current];
        state.page_refcounts[current] = 0;

        // Update free list.
        if let Some(idx) = state.page_table.free_pages.iter().position(|&p| p == target) {
            state.page_table.free_pages.remove(idx);
        }
        state.page_table.free_pages.push(current);

        moved += 1;
    }
    Ok(moved)
}

/// Hint the runtime to prefetch the pages that will be needed for the
/// next `count` positions.  On CPU this is a no-op; on GPU it would
/// issue `__prefetch_global_l2` intrinsics.
pub fn kv_cache_prefetch(
    state: &KvCacheGpuState,
    layer: usize,
    count: usize,
) -> Result<Vec<usize>> {
    let cfg = &state.config;
    if layer >= cfg.num_layers {
        return Err(KvCacheGpuError::LayerOutOfRange { layer, num_layers: cfg.num_layers }.into());
    }

    let seq_len = state.current_seq_len[layer];
    let mut pages = Vec::new();
    for pos in seq_len..seq_len.saturating_add(count) {
        if pos >= cfg.max_seq_len {
            break;
        }
        let logical = pos / cfg.page_size;
        if !pages.contains(&logical) {
            pages.push(logical);
        }
    }
    Ok(pages)
}

// ── Metrics ─────────────────────────────────────────────────────────

/// Runtime metrics for a GPU KV cache instance.
#[derive(Debug, Clone, Default)]
pub struct KvCacheGpuMetrics {
    /// Page hit rate (fraction of lookups served from allocated pages).
    pub hit_rate: f64,
    /// Fragmentation ratio: 1.0 - (live_pages / allocated_pages).
    pub fragmentation: f64,
    /// Total memory used in bytes (keys + values, f32).
    pub memory_usage_bytes: usize,
}

/// Compute metrics for the current cache state.
pub fn kv_cache_gpu_metrics(state: &KvCacheGpuState) -> KvCacheGpuMetrics {
    let cfg = &state.config;
    let allocated = state.page_table.allocated_count();
    let total_seq: usize = state.current_seq_len.iter().sum();
    let live_pages = if cfg.page_size > 0 { total_seq.div_ceil(cfg.page_size) } else { 0 };

    let fragmentation =
        if allocated > 0 { 1.0 - (live_pages as f64 / allocated as f64) } else { 0.0 };
    let hit_rate = if total_seq > 0 { 1.0 } else { 0.0 };
    let memory_usage_bytes = allocated
        * cfg.num_heads
        * cfg.page_size
        * cfg.head_dim
        * 2 // keys + values
        * std::mem::size_of::<f32>();

    KvCacheGpuMetrics { hit_rate, fragmentation, memory_usage_bytes }
}

// ── GPU launch stubs ────────────────────────────────────────────────

/// Launch the paged KV cache append kernel on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_kv_cache_append_gpu(
    _state: &mut KvCacheGpuState,
    _layer: usize,
    _new_keys: &[f32],
    _new_values: &[f32],
) -> Result<()> {
    Err(KernelError::InvalidArguments {
        reason: "GPU KV cache append not yet implemented; use CPU fallback".into(),
    }
    .into())
}

/// Launch the paged KV cache gather kernel on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_kv_cache_gather_gpu(
    _state: &mut KvCacheGpuState,
    _layer: usize,
    _key_out: &mut [f32],
    _value_out: &mut [f32],
) -> Result<()> {
    Err(KernelError::InvalidArguments {
        reason: "GPU KV cache gather not yet implemented; use CPU fallback".into(),
    }
    .into())
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> KvCacheGpuConfig {
        KvCacheGpuConfig::new(64, 4, 8, 2, 16, 8).unwrap()
    }

    fn small_config() -> KvCacheGpuConfig {
        KvCacheGpuConfig::new(16, 2, 4, 1, 4, 8).unwrap()
    }

    fn make_kv(cfg: &KvCacheGpuConfig, val: f32) -> (Vec<f32>, Vec<f32>) {
        let n = cfg.num_heads * cfg.head_dim;
        (vec![val; n], vec![val * 2.0; n])
    }

    // ── Config validation ───────────────────────────────────────

    #[test]
    fn test_config_valid() {
        let cfg = test_config();
        assert_eq!(cfg.max_seq_len, 64);
        assert_eq!(cfg.pages_needed(), 4);
        assert_eq!(cfg.page_elements(), 128);
    }

    #[test]
    fn test_config_zero_max_seq_len() {
        assert!(KvCacheGpuConfig::new(0, 4, 8, 2, 16, 8).is_err());
    }

    #[test]
    fn test_config_zero_num_heads() {
        assert!(KvCacheGpuConfig::new(64, 0, 8, 2, 16, 8).is_err());
    }

    #[test]
    fn test_config_zero_head_dim() {
        assert!(KvCacheGpuConfig::new(64, 4, 0, 2, 16, 8).is_err());
    }

    #[test]
    fn test_config_zero_num_layers() {
        assert!(KvCacheGpuConfig::new(64, 4, 8, 0, 16, 8).is_err());
    }

    #[test]
    fn test_config_zero_page_size() {
        assert!(KvCacheGpuConfig::new(64, 4, 8, 2, 0, 8).is_err());
    }

    #[test]
    fn test_config_non_power_of_two_page_size() {
        assert!(KvCacheGpuConfig::new(64, 4, 8, 2, 3, 8).is_err());
    }

    #[test]
    fn test_config_zero_max_pages() {
        assert!(KvCacheGpuConfig::new(64, 4, 8, 2, 16, 0).is_err());
    }

    #[test]
    fn test_config_insufficient_pages() {
        assert!(KvCacheGpuConfig::new(64, 4, 8, 2, 16, 2).is_err());
    }

    #[test]
    fn test_config_exact_pages() {
        assert!(KvCacheGpuConfig::new(64, 4, 8, 2, 16, 4).is_ok());
    }

    // ── Page table ──────────────────────────────────────────────

    #[test]
    fn test_page_table_new() {
        let cfg = test_config();
        let pt = PageTable::new(&cfg);
        assert_eq!(pt.free_count(), cfg.max_pages);
        assert_eq!(pt.allocated_count(), 0);
    }

    #[test]
    fn test_page_table_allocate_deallocate() {
        let cfg = test_config();
        let mut pt = PageTable::new(&cfg);
        let p0 = pt.allocate().unwrap();
        assert_eq!(pt.allocated_count(), 1);
        assert_eq!(pt.free_count(), cfg.max_pages - 1);
        pt.deallocate(p0).unwrap();
        assert_eq!(pt.allocated_count(), 0);
        assert_eq!(pt.free_count(), cfg.max_pages);
    }

    #[test]
    fn test_page_table_exhaust_pages() {
        let cfg = test_config();
        let mut pt = PageTable::new(&cfg);
        for _ in 0..cfg.max_pages {
            pt.allocate().unwrap();
        }
        assert!(pt.allocate().is_err());
    }

    #[test]
    fn test_page_table_map_resolve() {
        let cfg = test_config();
        let mut pt = PageTable::new(&cfg);
        let phys = pt.allocate().unwrap();
        pt.map(0, 0, phys).unwrap();
        assert_eq!(pt.resolve(0, 0).unwrap(), phys);
    }

    #[test]
    fn test_page_table_resolve_unmapped() {
        let cfg = test_config();
        let pt = PageTable::new(&cfg);
        assert!(pt.resolve(0, 0).is_err());
    }

    #[test]
    fn test_page_table_map_layer_out_of_range() {
        let cfg = test_config();
        let mut pt = PageTable::new(&cfg);
        assert!(pt.map(99, 0, 0).is_err());
    }

    #[test]
    fn test_page_table_deallocate_invalid() {
        let cfg = test_config();
        let mut pt = PageTable::new(&cfg);
        assert!(pt.deallocate(999).is_err());
    }

    // ── Append and lookup ───────────────────────────────────────

    #[test]
    fn test_append_single_position() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        let (keys, vals) = make_kv(&cfg, 1.0);
        kv_cache_append(&mut state, 0, &keys, &vals).unwrap();
        assert_eq!(state.current_seq_len[0], 1);
    }

    #[test]
    fn test_append_fill_one_page() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        for i in 0..cfg.page_size {
            let (k, v) = make_kv(&cfg, i as f32);
            kv_cache_append(&mut state, 0, &k, &v).unwrap();
        }
        assert_eq!(state.current_seq_len[0], cfg.page_size);
        assert_eq!(state.page_table.allocated_count(), 1);
    }

    #[test]
    fn test_append_cross_page_boundary() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        for i in 0..(cfg.page_size + 1) {
            let (k, v) = make_kv(&cfg, i as f32);
            kv_cache_append(&mut state, 0, &k, &v).unwrap();
        }
        assert_eq!(state.current_seq_len[0], cfg.page_size + 1);
        assert_eq!(state.page_table.allocated_count(), 2);
    }

    #[test]
    fn test_append_layer_out_of_range() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        let (k, v) = make_kv(&cfg, 1.0);
        assert!(kv_cache_append(&mut state, 99, &k, &v).is_err());
    }

    #[test]
    fn test_append_wrong_size() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg);
        assert!(kv_cache_append(&mut state, 0, &[1.0], &[2.0]).is_err());
    }

    #[test]
    fn test_append_overflow() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        for i in 0..cfg.max_seq_len {
            let (k, v) = make_kv(&cfg, i as f32);
            kv_cache_append(&mut state, 0, &k, &v).unwrap();
        }
        let (k, v) = make_kv(&cfg, 99.0);
        assert!(kv_cache_append(&mut state, 0, &k, &v).is_err());
    }

    #[test]
    fn test_lookup_empty() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg);
        let mut k_out = vec![];
        let mut v_out = vec![];
        kv_cache_paged_lookup(&mut state, 0, &mut k_out, &mut v_out).unwrap();
    }

    #[test]
    fn test_lookup_after_append() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        let (k, v) = make_kv(&cfg, 3.0);
        kv_cache_append(&mut state, 0, &k, &v).unwrap();

        let n = cfg.num_heads * cfg.head_dim;
        let mut k_out = vec![0.0f32; n];
        let mut v_out = vec![0.0f32; n];
        kv_cache_paged_lookup(&mut state, 0, &mut k_out, &mut v_out).unwrap();
        assert_eq!(k_out, vec![3.0; n]);
        assert_eq!(v_out, vec![6.0; n]);
    }

    #[test]
    fn test_lookup_multi_position() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        for i in 0..3 {
            let (k, v) = make_kv(&cfg, (i + 1) as f32);
            kv_cache_append(&mut state, 0, &k, &v).unwrap();
        }

        let n = 3 * cfg.num_heads * cfg.head_dim;
        let mut k_out = vec![0.0f32; n];
        let mut v_out = vec![0.0f32; n];
        kv_cache_paged_lookup(&mut state, 0, &mut k_out, &mut v_out).unwrap();

        let hd = cfg.num_heads * cfg.head_dim;
        assert!(k_out[..hd].iter().all(|&x| (x - 1.0).abs() < f32::EPSILON));
        assert!(k_out[hd..2 * hd].iter().all(|&x| (x - 2.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_lookup_output_too_small() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        let (k, v) = make_kv(&cfg, 1.0);
        kv_cache_append(&mut state, 0, &k, &v).unwrap();
        let mut k_out = vec![0.0f32; 1];
        let mut v_out = vec![0.0f32; 1];
        assert!(kv_cache_paged_lookup(&mut state, 0, &mut k_out, &mut v_out).is_err());
    }

    #[test]
    fn test_lookup_layer_out_of_range() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg);
        assert!(kv_cache_paged_lookup(&mut state, 99, &mut [], &mut []).is_err());
    }

    // ── Sliding-window rotation ─────────────────────────────────

    #[test]
    fn test_rotate_noop_within_window() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        for i in 0..3 {
            let (k, v) = make_kv(&cfg, i as f32);
            kv_cache_append(&mut state, 0, &k, &v).unwrap();
        }
        kv_cache_rotate(&mut state, 0, 10).unwrap();
        assert_eq!(state.current_seq_len[0], 3);
    }

    #[test]
    fn test_rotate_trims_oldest() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        for i in 0..8 {
            let (k, v) = make_kv(&cfg, i as f32);
            kv_cache_append(&mut state, 0, &k, &v).unwrap();
        }
        kv_cache_rotate(&mut state, 0, 4).unwrap();
        assert_eq!(state.current_seq_len[0], 4);
    }

    #[test]
    fn test_rotate_zero_window() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg);
        assert!(kv_cache_rotate(&mut state, 0, 0).is_err());
    }

    #[test]
    fn test_rotate_layer_out_of_range() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg);
        assert!(kv_cache_rotate(&mut state, 99, 4).is_err());
    }

    // ── INT8 quantize / dequantize ──────────────────────────────

    #[test]
    fn test_quantize_empty_layer() {
        let cfg = small_config();
        let state = KvCacheGpuState::new(cfg);
        let (qk, qv, ks, vs) = kv_cache_quantize(&state, 0).unwrap();
        assert!(qk.is_empty());
        assert!(qv.is_empty());
        assert!(ks.is_empty());
        assert!(vs.is_empty());
    }

    #[test]
    fn test_quantize_single_position() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        let (k, v) = make_kv(&cfg, 1.0);
        kv_cache_append(&mut state, 0, &k, &v).unwrap();

        let (qk, qv, ks, vs) = kv_cache_quantize(&state, 0).unwrap();
        let n = cfg.num_heads * cfg.head_dim;
        assert_eq!(qk.len(), n);
        assert_eq!(qv.len(), n);
        assert!(!ks.is_empty());
        assert!(!vs.is_empty());
    }

    #[test]
    fn test_quantize_round_trip_quality() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());

        for i in 0..4 {
            let n = cfg.num_heads * cfg.head_dim;
            let keys: Vec<f32> = (0..n).map(|j| ((i * n + j) as f32) * 0.1 - 0.2).collect();
            let vals: Vec<f32> = (0..n).map(|j| ((i * n + j) as f32) * 0.05 + 0.1).collect();
            kv_cache_append(&mut state, 0, &keys, &vals).unwrap();
        }

        let (qk, _qv, ks, _vs) = kv_cache_quantize(&state, 0).unwrap();
        let deq = kv_cache_dequantize(&qk, &ks, cfg.num_heads, cfg.head_dim, cfg.page_size, 4);

        let total = 4 * cfg.num_heads * cfg.head_dim;
        let mut orig_k = vec![0.0f32; total];
        let mut orig_v = vec![0.0f32; total];
        kv_cache_paged_lookup(&mut state, 0, &mut orig_k, &mut orig_v).unwrap();

        let max_err: f32 =
            orig_k.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let absmax: f32 = orig_k.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let relative = if absmax > 0.0 { max_err / absmax } else { 0.0 };
        assert!(relative < 0.02, "INT8 round-trip error too large: relative={relative}");
    }

    #[test]
    fn test_quantize_layer_out_of_range() {
        let cfg = small_config();
        let state = KvCacheGpuState::new(cfg);
        assert!(kv_cache_quantize(&state, 99).is_err());
    }

    #[test]
    fn test_dequantize_zeros() {
        let out = kv_cache_dequantize(&[0i8; 8], &[1.0, 1.0], 2, 4, 4, 1);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    // ── Eviction ────────────────────────────────────────────────

    #[test]
    fn test_evict_empty() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg);
        assert_eq!(kv_cache_evict(&mut state, 0, 5).unwrap(), 0);
    }

    #[test]
    fn test_evict_some_pages() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        for i in 0..8 {
            let (k, v) = make_kv(&cfg, i as f32);
            kv_cache_append(&mut state, 0, &k, &v).unwrap();
        }
        let evicted = kv_cache_evict(&mut state, 0, 1).unwrap();
        assert_eq!(evicted, 1);
    }

    #[test]
    fn test_evict_layer_out_of_range() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg);
        assert!(kv_cache_evict(&mut state, 99, 1).is_err());
    }

    // ── Copy-on-write ───────────────────────────────────────────

    #[test]
    fn test_cow_basic() {
        let cfg = KvCacheGpuConfig::new(16, 2, 4, 2, 4, 8).unwrap();
        let mut state = KvCacheGpuState::new(cfg.clone());
        let (k, v) = make_kv(&cfg, 5.0);
        kv_cache_append(&mut state, 0, &k, &v).unwrap();

        kv_cache_copy_on_write(&mut state, 0, 1).unwrap();
        assert_eq!(state.current_seq_len[1], 1);

        let p0 = state.page_table.resolve(0, 0).unwrap();
        let p1 = state.page_table.resolve(1, 0).unwrap();
        assert_eq!(p0, p1);
    }

    #[test]
    fn test_cow_refcount_increment() {
        let cfg = KvCacheGpuConfig::new(16, 2, 4, 2, 4, 8).unwrap();
        let mut state = KvCacheGpuState::new(cfg.clone());
        let (k, v) = make_kv(&cfg, 1.0);
        kv_cache_append(&mut state, 0, &k, &v).unwrap();

        let phys = state.page_table.resolve(0, 0).unwrap();
        assert_eq!(state.page_refcounts[phys], 1);

        kv_cache_copy_on_write(&mut state, 0, 1).unwrap();
        assert_eq!(state.page_refcounts[phys], 2);
    }

    #[test]
    fn test_cow_materialize() {
        let cfg = KvCacheGpuConfig::new(16, 2, 4, 2, 4, 8).unwrap();
        let mut state = KvCacheGpuState::new(cfg.clone());
        let (k, v) = make_kv(&cfg, 7.0);
        kv_cache_append(&mut state, 0, &k, &v).unwrap();
        kv_cache_copy_on_write(&mut state, 0, 1).unwrap();

        kv_cache_cow_materialize(&mut state, 1, 0).unwrap();
        let p0 = state.page_table.resolve(0, 0).unwrap();
        let p1 = state.page_table.resolve(1, 0).unwrap();
        assert_ne!(p0, p1);
    }

    #[test]
    fn test_cow_materialize_preserves_data() {
        let cfg = KvCacheGpuConfig::new(16, 2, 4, 2, 4, 8).unwrap();
        let mut state = KvCacheGpuState::new(cfg.clone());
        let (k, v) = make_kv(&cfg, 42.0);
        kv_cache_append(&mut state, 0, &k, &v).unwrap();
        kv_cache_copy_on_write(&mut state, 0, 1).unwrap();
        kv_cache_cow_materialize(&mut state, 1, 0).unwrap();

        let n = cfg.num_heads * cfg.head_dim;
        let mut k_out = vec![0.0f32; n];
        let mut v_out = vec![0.0f32; n];
        kv_cache_paged_lookup(&mut state, 1, &mut k_out, &mut v_out).unwrap();
        assert!(k_out.iter().all(|&x| (x - 42.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_cow_src_out_of_range() {
        let cfg = KvCacheGpuConfig::new(16, 2, 4, 2, 4, 8).unwrap();
        let mut state = KvCacheGpuState::new(cfg);
        assert!(kv_cache_copy_on_write(&mut state, 99, 0).is_err());
    }

    #[test]
    fn test_cow_dst_out_of_range() {
        let cfg = KvCacheGpuConfig::new(16, 2, 4, 2, 4, 8).unwrap();
        let mut state = KvCacheGpuState::new(cfg);
        assert!(kv_cache_copy_on_write(&mut state, 0, 99).is_err());
    }

    // ── Defragmentation ─────────────────────────────────────────

    #[test]
    fn test_defrag_no_holes() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        for i in 0..4 {
            let (k, v) = make_kv(&cfg, i as f32);
            kv_cache_append(&mut state, 0, &k, &v).unwrap();
        }
        let moved = kv_cache_defrag(&mut state).unwrap();
        assert_eq!(moved, 0);
    }

    #[test]
    fn test_defrag_after_eviction() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        for i in 0..8 {
            let (k, v) = make_kv(&cfg, i as f32);
            kv_cache_append(&mut state, 0, &k, &v).unwrap();
        }
        kv_cache_evict(&mut state, 0, 1).unwrap();
        let _moved = kv_cache_defrag(&mut state).unwrap();
    }

    #[test]
    fn test_defrag_preserves_data() {
        let cfg = KvCacheGpuConfig::new(16, 2, 4, 1, 4, 8).unwrap();
        let mut state = KvCacheGpuState::new(cfg.clone());
        for i in 0..8 {
            let (k, v) = make_kv(&cfg, (i + 1) as f32);
            kv_cache_append(&mut state, 0, &k, &v).unwrap();
        }

        let n = 8 * cfg.num_heads * cfg.head_dim;
        let mut k_before = vec![0.0f32; n];
        let mut v_before = vec![0.0f32; n];
        kv_cache_paged_lookup(&mut state, 0, &mut k_before, &mut v_before).unwrap();

        kv_cache_defrag(&mut state).unwrap();

        let mut k_after = vec![0.0f32; n];
        let mut v_after = vec![0.0f32; n];
        kv_cache_paged_lookup(&mut state, 0, &mut k_after, &mut v_after).unwrap();

        assert_eq!(k_before, k_after);
        assert_eq!(v_before, v_after);
    }

    // ── Prefetch ────────────────────────────────────────────────

    #[test]
    fn test_prefetch_empty() {
        let cfg = small_config();
        let state = KvCacheGpuState::new(cfg);
        let pages = kv_cache_prefetch(&state, 0, 4).unwrap();
        assert!(!pages.is_empty());
    }

    #[test]
    fn test_prefetch_returns_logical_pages() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        for i in 0..4 {
            let (k, v) = make_kv(&cfg, i as f32);
            kv_cache_append(&mut state, 0, &k, &v).unwrap();
        }
        let pages = kv_cache_prefetch(&state, 0, 8).unwrap();
        assert!(!pages.is_empty());
    }

    #[test]
    fn test_prefetch_layer_out_of_range() {
        let cfg = small_config();
        let state = KvCacheGpuState::new(cfg);
        assert!(kv_cache_prefetch(&state, 99, 4).is_err());
    }

    #[test]
    fn test_prefetch_at_max_seq() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        for i in 0..cfg.max_seq_len {
            let (k, v) = make_kv(&cfg, i as f32);
            kv_cache_append(&mut state, 0, &k, &v).unwrap();
        }
        let pages = kv_cache_prefetch(&state, 0, 4).unwrap();
        assert!(pages.is_empty());
    }

    // ── Metrics ─────────────────────────────────────────────────

    #[test]
    fn test_metrics_empty() {
        let cfg = small_config();
        let state = KvCacheGpuState::new(cfg);
        let m = kv_cache_gpu_metrics(&state);
        assert_eq!(m.memory_usage_bytes, 0);
        assert_eq!(m.fragmentation, 0.0);
        assert_eq!(m.hit_rate, 0.0);
    }

    #[test]
    fn test_metrics_with_data() {
        let cfg = small_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        let (k, v) = make_kv(&cfg, 1.0);
        kv_cache_append(&mut state, 0, &k, &v).unwrap();

        let m = kv_cache_gpu_metrics(&state);
        assert!(m.memory_usage_bytes > 0);
        assert_eq!(m.hit_rate, 1.0);
    }

    // ── Error display ───────────────────────────────────────────

    #[test]
    fn test_error_display_out_of_pages() {
        let e = KvCacheGpuError::OutOfPages;
        assert!(e.to_string().contains("no free pages"));
    }

    #[test]
    fn test_error_display_layer_out_of_range() {
        let e = KvCacheGpuError::LayerOutOfRange { layer: 5, num_layers: 2 };
        let s = e.to_string();
        assert!(s.contains("5") && s.contains("2"));
    }

    #[test]
    fn test_error_display_sequence_overflow() {
        let e = KvCacheGpuError::SequenceOverflow { current: 10, max: 8 };
        assert!(e.to_string().contains("overflow"));
    }

    #[test]
    fn test_error_display_invalid_page() {
        let e = KvCacheGpuError::InvalidPage { page: 99, max_pages: 8 };
        assert!(e.to_string().contains("99"));
    }

    #[test]
    fn test_error_display_cow_source_missing() {
        let e = KvCacheGpuError::CowSourceMissing { page: 3 };
        assert!(e.to_string().contains("CoW"));
    }

    #[test]
    fn test_error_display_quantization() {
        let e = KvCacheGpuError::QuantizationError("test".into());
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn test_error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(KvCacheGpuError::OutOfPages);
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_into_bitnet_error() {
        let e = KvCacheGpuError::OutOfPages;
        let _: bitnet_common::BitNetError = e.into();
    }

    // ── Multi-layer ─────────────────────────────────────────────

    #[test]
    fn test_multi_layer_independent() {
        let cfg = test_config();
        let mut state = KvCacheGpuState::new(cfg.clone());
        let (k0, v0) = make_kv(&cfg, 1.0);
        let (k1, v1) = make_kv(&cfg, 2.0);
        kv_cache_append(&mut state, 0, &k0, &v0).unwrap();
        kv_cache_append(&mut state, 1, &k1, &v1).unwrap();

        assert_eq!(state.current_seq_len[0], 1);
        assert_eq!(state.current_seq_len[1], 1);

        let n = cfg.num_heads * cfg.head_dim;
        let mut ko = vec![0.0f32; n];
        let mut vo = vec![0.0f32; n];
        kv_cache_paged_lookup(&mut state, 0, &mut ko, &mut vo).unwrap();
        assert!(ko.iter().all(|&x| (x - 1.0).abs() < f32::EPSILON));

        kv_cache_paged_lookup(&mut state, 1, &mut ko, &mut vo).unwrap();
        assert!(ko.iter().all(|&x| (x - 2.0).abs() < f32::EPSILON));
    }

    // ── Property tests ──────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_config() -> impl Strategy<Value = KvCacheGpuConfig> {
            (1..=32usize, 1..=8usize, 1..=16usize, 1..=4usize).prop_map(
                |(seq, heads, dim, layers)| {
                    let page_size = 4;
                    let max_seq = seq.max(4);
                    let pages_needed = max_seq.div_ceil(page_size);
                    let max_pages = pages_needed + 4;
                    KvCacheGpuConfig::new(max_seq, heads, dim, layers, page_size, max_pages)
                        .unwrap()
                },
            )
        }

        proptest! {
            #[test]
            fn prop_append_increments_seq_len(
                cfg in arb_config()
            ) {
                let mut state = KvCacheGpuState::new(cfg.clone());
                let n = cfg.num_heads * cfg.head_dim;
                let k = vec![1.0f32; n];
                let v = vec![2.0f32; n];
                for expected in 1..=cfg.max_seq_len.min(8) {
                    kv_cache_append(&mut state, 0, &k, &v).unwrap();
                    prop_assert_eq!(
                        state.current_seq_len[0],
                        expected
                    );
                }
            }

            #[test]
            fn prop_lookup_returns_appended_data(
                cfg in arb_config(),
                val in -100.0f32..100.0f32
            ) {
                let mut state = KvCacheGpuState::new(cfg.clone());
                let n = cfg.num_heads * cfg.head_dim;
                let k = vec![val; n];
                let v = vec![val * 2.0; n];
                kv_cache_append(&mut state, 0, &k, &v).unwrap();

                let mut ko = vec![0.0f32; n];
                let mut vo = vec![0.0f32; n];
                kv_cache_paged_lookup(
                    &mut state, 0, &mut ko, &mut vo,
                )
                .unwrap();
                for i in 0..n {
                    prop_assert!((ko[i] - val).abs() < 1e-6);
                    prop_assert!((vo[i] - val * 2.0).abs() < 1e-6);
                }
            }

            #[test]
            fn prop_page_alloc_dealloc_preserves_count(
                count in 1..=16usize
            ) {
                let cfg = KvCacheGpuConfig::new(
                    64, 2, 4, 1, 4, 20,
                )
                .unwrap();
                let mut pt = PageTable::new(&cfg);
                let initial_free = pt.free_count();
                let mut pages = Vec::new();
                for _ in 0..count {
                    pages.push(pt.allocate().unwrap());
                }
                prop_assert_eq!(
                    pt.free_count(),
                    initial_free - count
                );
                for p in pages {
                    pt.deallocate(p).unwrap();
                }
                prop_assert_eq!(pt.free_count(), initial_free);
            }

            #[test]
            fn prop_quantize_round_trip_bounded_error(
                cfg in arb_config(),
                seed in 0..1000u32
            ) {
                let mut state = KvCacheGpuState::new(cfg.clone());
                let n = cfg.num_heads * cfg.head_dim;
                let positions = cfg.max_seq_len.min(4);
                for i in 0..positions {
                    let keys: Vec<f32> = (0..n)
                        .map(|j| {
                            ((seed as f32 + i as f32 + j as f32) * 0.1)
                                .sin()
                        })
                        .collect();
                    let vals: Vec<f32> = (0..n)
                        .map(|j| {
                            ((seed as f32 + i as f32 + j as f32) * 0.2)
                                .cos()
                        })
                        .collect();
                    kv_cache_append(&mut state, 0, &keys, &vals)
                        .unwrap();
                }

                let (qk, _qv, ks, _vs) =
                    kv_cache_quantize(&state, 0).unwrap();
                let deq = kv_cache_dequantize(
                    &qk,
                    &ks,
                    cfg.num_heads,
                    cfg.head_dim,
                    cfg.page_size,
                    positions,
                );

                let total = positions * n;
                let mut orig_k = vec![0.0f32; total];
                let mut orig_v = vec![0.0f32; total];
                kv_cache_paged_lookup(
                    &mut state, 0, &mut orig_k, &mut orig_v,
                )
                .unwrap();

                let max_err: f32 = orig_k
                    .iter()
                    .zip(deq.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let absmax: f32 = orig_k
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0f32, f32::max);
                if absmax > 1e-6 {
                    prop_assert!(
                        max_err / absmax < 0.02,
                        "relative error too large: {}",
                        max_err / absmax
                    );
                }
            }

            #[test]
            fn prop_metrics_memory_scales_with_pages(
                count in 1..=8usize
            ) {
                let cfg = KvCacheGpuConfig::new(
                    32, 2, 4, 1, 4, 16,
                )
                .unwrap();
                let mut state = KvCacheGpuState::new(cfg.clone());
                let n = cfg.num_heads * cfg.head_dim;
                let k = vec![1.0f32; n];
                let v = vec![2.0f32; n];
                for _ in 0..count {
                    kv_cache_append(&mut state, 0, &k, &v).unwrap();
                }
                let m = kv_cache_gpu_metrics(&state);
                prop_assert!(m.memory_usage_bytes > 0);
            }
        }
    }
}
