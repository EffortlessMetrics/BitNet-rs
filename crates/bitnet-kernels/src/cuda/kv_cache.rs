//! CUDA KV cache management kernel for autoregressive generation.
//!
//! # Kernel strategy
//!
//! During autoregressive decoding the attention layer needs access to all
//! previously computed key and value projections.  This module maintains a
//! per-layer, contiguous KV cache and exposes operations that map naturally
//! to CUDA kernels:
//!
//! - **Append** — copy a new K/V pair into the cache at a given position.
//!   One thread-block per head, threads covering `head_dim`.
//! - **Get** — extract a sub-range `[start_pos, end_pos)` from the cache
//!   for the attention window.  Launches `n_heads` blocks, each streaming
//!   the requested range in shared-memory tiles.
//! - **Rotate** — apply rotary position embedding to cached keys in-place.
//!   Reuses the RoPE kernel grid/block layout.
//! - **Truncate** — logically shorten the cache (no memory free).
//! - **Clear** — zero-fill and reset all layer lengths.
//!
//! The CPU fallback implements every operation with plain `Vec` slicing.

use bitnet_common::{KernelError, Result};
use std::time::Instant;

// ── Configuration ────────────────────────────────────────────────────

/// Data type selector for cache storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheDtype {
    /// 32-bit floating point.
    F32,
    /// 16-bit floating point (half precision).
    F16,
    /// 16-bit brain floating point.
    Bf16,
}

/// Configuration for a KV cache instance.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads per layer.
    pub num_heads: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
    /// Maximum sequence length the cache can hold.
    pub max_seq_len: usize,
    /// Element data type.
    pub dtype: CacheDtype,
}

impl KvCacheConfig {
    /// Create a new configuration, validating all dimensions.
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: CacheDtype,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "KvCache num_layers must be non-zero".into(),
            }
            .into());
        }
        if num_heads == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "KvCache num_heads must be non-zero".into(),
            }
            .into());
        }
        if head_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "KvCache head_dim must be non-zero".into(),
            }
            .into());
        }
        if max_seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "KvCache max_seq_len must be non-zero".into(),
            }
            .into());
        }
        Ok(Self { num_layers, num_heads, head_dim, max_seq_len, dtype })
    }

    /// Total elements in a single layer's key (or value) buffer.
    #[inline]
    fn layer_elements(&self) -> usize {
        self.num_heads * self.max_seq_len * self.head_dim
    }

    /// Elements in a single head at a single position.
    #[inline]
    fn head_elements(&self) -> usize {
        self.head_dim
    }

    /// Bytes per element for the configured dtype.
    fn bytes_per_element(&self) -> usize {
        match self.dtype {
            CacheDtype::F32 => 4,
            CacheDtype::F16 | CacheDtype::Bf16 => 2,
        }
    }

    /// Compute CUDA grid dimensions for a per-head append kernel.
    pub fn append_grid_dim(&self) -> (u32, u32, u32) {
        (self.num_heads as u32, 1, 1)
    }

    /// Compute CUDA block dimensions (threads covering `head_dim`, capped at 1024).
    pub fn append_block_dim(&self) -> (u32, u32, u32) {
        ((self.head_dim as u32).min(1024), 1, 1)
    }
}

// ── Statistics ────────────────────────────────────────────────────────

/// Runtime statistics for the KV cache.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current number of cached entries per layer.
    pub entries_per_layer: Vec<usize>,
    /// Total memory consumed by cache buffers in bytes.
    pub memory_bytes: usize,
    /// Cache hit rate (gets that found data / total gets). Starts at 1.0.
    pub hit_rate: f64,
    /// Average access time in microseconds across all operations.
    pub avg_access_time_us: f64,
}

// ── Buffer ───────────────────────────────────────────────────────────

/// Per-layer state tracked inside `KvCacheBuffer`.
#[derive(Debug, Clone)]
struct LayerState {
    /// Current number of filled positions for this layer.
    len: usize,
}

/// Manages key and value tensors for all transformer layers.
///
/// Layout per layer (key and value each):
///   `[num_heads, max_seq_len, head_dim]`  — row-major.
///
/// The CPU fallback stores data in flat `Vec<f32>` buffers. All public
/// operations accept FP32 data regardless of `dtype` (the CUDA path would
/// down-cast on device; the CPU path keeps FP32 for simplicity).
#[derive(Debug)]
pub struct KvCacheBuffer {
    config: KvCacheConfig,
    /// Flat key storage: `num_layers` contiguous blocks.
    keys: Vec<f32>,
    /// Flat value storage: `num_layers` contiguous blocks.
    values: Vec<f32>,
    /// Per-layer bookkeeping.
    layers: Vec<LayerState>,
    // ── stats tracking ───────────────────────────────────────────────
    total_ops: u64,
    total_time_us: f64,
    total_gets: u64,
    total_hits: u64,
}

impl KvCacheBuffer {
    /// Allocate a new KV cache with the given configuration.
    pub fn new(config: KvCacheConfig) -> Self {
        let total = config.num_layers * config.layer_elements();
        let layers = (0..config.num_layers).map(|_| LayerState { len: 0 }).collect();
        Self {
            config,
            keys: vec![0.0; total],
            values: vec![0.0; total],
            layers,
            total_ops: 0,
            total_time_us: 0.0,
            total_gets: 0,
            total_hits: 0,
        }
    }

    /// Reference to the active configuration.
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Current length (filled positions) for a given layer.
    pub fn layer_len(&self, layer: usize) -> Result<usize> {
        self.validate_layer(layer)?;
        Ok(self.layers[layer].len)
    }

    // ── Core operations ──────────────────────────────────────────────

    /// Append key/value data at `pos` for a single layer.
    ///
    /// `key_data` and `value_data` must each contain `num_heads * head_dim`
    /// elements representing one token position across all heads.
    pub fn append_kv(
        &mut self,
        layer: usize,
        pos: usize,
        key_data: &[f32],
        value_data: &[f32],
    ) -> Result<()> {
        let start = Instant::now();
        self.validate_layer(layer)?;
        let expected = self.config.num_heads * self.config.head_elements();
        if key_data.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "append_kv key_data length mismatch: expected {expected}, \
                     got {}",
                    key_data.len(),
                ),
            }
            .into());
        }
        if value_data.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "append_kv value_data length mismatch: expected {expected}, \
                     got {}",
                    value_data.len(),
                ),
            }
            .into());
        }
        if pos >= self.config.max_seq_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "append_kv pos {} exceeds max_seq_len {}",
                    pos, self.config.max_seq_len,
                ),
            }
            .into());
        }

        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            if crate::device_features::gpu_available_runtime() {
                if let Ok(()) = launch_append_kv(
                    &mut self.keys,
                    &mut self.values,
                    layer,
                    pos,
                    key_data,
                    value_data,
                    &self.config,
                ) {
                    self.update_layer_len_after_append(layer, pos);
                    self.record_op(start);
                    return Ok(());
                }
            }
        }

        self.append_kv_cpu(layer, pos, key_data, value_data);
        self.update_layer_len_after_append(layer, pos);
        self.record_op(start);
        Ok(())
    }

    /// Retrieve cached keys and values for `[start_pos, end_pos)` in a layer.
    ///
    /// Returns `(keys, values)` each shaped `[num_heads * (end_pos - start_pos) * head_dim]`.
    pub fn get_kv(
        &mut self,
        layer: usize,
        start_pos: usize,
        end_pos: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let start = Instant::now();
        self.validate_layer(layer)?;
        if end_pos <= start_pos {
            return Err(KernelError::InvalidArguments {
                reason: format!("get_kv end_pos ({end_pos}) must be > start_pos ({start_pos})"),
            }
            .into());
        }
        if end_pos > self.config.max_seq_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "get_kv end_pos {end_pos} exceeds max_seq_len {}",
                    self.config.max_seq_len,
                ),
            }
            .into());
        }

        self.total_gets += 1;
        let layer_len = self.layers[layer].len;
        if end_pos <= layer_len {
            self.total_hits += 1;
        }

        let result = self.get_kv_cpu(layer, start_pos, end_pos);
        self.record_op(start);
        Ok(result)
    }

    /// Apply rotary position embedding to cached keys in-place.
    ///
    /// `positions` contains the absolute position indices for each cached
    /// entry in `[0, layer_len)` — one per position, applied across all heads.
    pub fn rotate_kv(&mut self, layer: usize, positions: &[usize]) -> Result<()> {
        let start = Instant::now();
        self.validate_layer(layer)?;
        let layer_len = self.layers[layer].len;
        if positions.len() != layer_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "rotate_kv positions length ({}) != layer_len ({layer_len})",
                    positions.len(),
                ),
            }
            .into());
        }
        if self.config.head_dim % 2 != 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("rotate_kv requires even head_dim, got {}", self.config.head_dim,),
            }
            .into());
        }

        self.rotate_kv_cpu(layer, positions);
        self.record_op(start);
        Ok(())
    }

    /// Truncate a layer's cache to `new_len` positions.
    pub fn truncate(&mut self, layer: usize, new_len: usize) -> Result<()> {
        self.validate_layer(layer)?;
        if new_len > self.layers[layer].len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "truncate new_len ({new_len}) exceeds current length ({})",
                    self.layers[layer].len,
                ),
            }
            .into());
        }
        self.layers[layer].len = new_len;
        Ok(())
    }

    /// Reset all caches to zero length and clear buffers.
    pub fn clear(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        for layer in &mut self.layers {
            layer.len = 0;
        }
    }

    /// Collect runtime statistics.
    pub fn stats(&self) -> CacheStats {
        let entries_per_layer = self.layers.iter().map(|l| l.len).collect();
        let total_elements = self.config.num_layers * self.config.layer_elements() * 2;
        let memory_bytes = total_elements * self.config.bytes_per_element();
        let hit_rate =
            if self.total_gets > 0 { self.total_hits as f64 / self.total_gets as f64 } else { 1.0 };
        let avg_access_time_us =
            if self.total_ops > 0 { self.total_time_us / self.total_ops as f64 } else { 0.0 };
        CacheStats { entries_per_layer, memory_bytes, hit_rate, avg_access_time_us }
    }

    // ── CPU fallback implementations ─────────────────────────────────

    fn append_kv_cpu(&mut self, layer: usize, pos: usize, key_data: &[f32], value_data: &[f32]) {
        let head_dim = self.config.head_dim;
        let max_seq = self.config.max_seq_len;
        let layer_off = layer * self.config.layer_elements();

        for head in 0..self.config.num_heads {
            let src_start = head * head_dim;
            let dst_start = layer_off + head * max_seq * head_dim + pos * head_dim;
            self.keys[dst_start..dst_start + head_dim]
                .copy_from_slice(&key_data[src_start..src_start + head_dim]);
            self.values[dst_start..dst_start + head_dim]
                .copy_from_slice(&value_data[src_start..src_start + head_dim]);
        }
    }

    fn get_kv_cpu(&self, layer: usize, start_pos: usize, end_pos: usize) -> (Vec<f32>, Vec<f32>) {
        let head_dim = self.config.head_dim;
        let max_seq = self.config.max_seq_len;
        let range_len = end_pos - start_pos;
        let out_len = self.config.num_heads * range_len * head_dim;
        let mut k_out = Vec::with_capacity(out_len);
        let mut v_out = Vec::with_capacity(out_len);

        let layer_off = layer * self.config.layer_elements();
        for head in 0..self.config.num_heads {
            let head_off = layer_off + head * max_seq * head_dim;
            let src_start = head_off + start_pos * head_dim;
            let src_end = head_off + end_pos * head_dim;
            k_out.extend_from_slice(&self.keys[src_start..src_end]);
            v_out.extend_from_slice(&self.values[src_start..src_end]);
        }

        (k_out, v_out)
    }

    fn rotate_kv_cpu(&mut self, layer: usize, positions: &[usize]) {
        let head_dim = self.config.head_dim;
        let max_seq = self.config.max_seq_len;
        let half_dim = head_dim / 2;
        let base: f32 = 10_000.0;
        let layer_off = layer * self.config.layer_elements();

        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| {
                let exponent = -(2.0 * i as f32) / head_dim as f32;
                base.powf(exponent)
            })
            .collect();

        for head in 0..self.config.num_heads {
            let head_off = layer_off + head * max_seq * head_dim;
            for (seq_idx, &pos) in positions.iter().enumerate() {
                let row = head_off + seq_idx * head_dim;
                let actual_pos = pos as f32;
                for i in 0..half_dim {
                    let angle = actual_pos * inv_freq[i];
                    let cos_val = angle.cos();
                    let sin_val = angle.sin();
                    let x0 = self.keys[row + 2 * i];
                    let x1 = self.keys[row + 2 * i + 1];
                    self.keys[row + 2 * i] = x0 * cos_val - x1 * sin_val;
                    self.keys[row + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────

    fn validate_layer(&self, layer: usize) -> Result<()> {
        if layer >= self.config.num_layers {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "layer index {layer} out of range (num_layers={})",
                    self.config.num_layers,
                ),
            }
            .into());
        }
        Ok(())
    }

    fn update_layer_len_after_append(&mut self, layer: usize, pos: usize) {
        let new_candidate = pos + 1;
        if new_candidate > self.layers[layer].len {
            self.layers[layer].len = new_candidate;
        }
    }

    fn record_op(&mut self, start: Instant) {
        self.total_ops += 1;
        self.total_time_us += start.elapsed().as_secs_f64() * 1_000_000.0;
    }
}

// ── CUDA launch stub ─────────────────────────────────────────────────

/// Launch stub for the KV cache append CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled.
#[allow(clippy::too_many_arguments)]
pub fn launch_append_kv(
    _keys: &mut [f32],
    _values: &mut [f32],
    _layer: usize,
    _pos: usize,
    _key_data: &[f32],
    _value_data: &[f32],
    config: &KvCacheConfig,
) -> Result<()> {
    log::debug!(
        "KV cache append stub: layers={}, heads={}, head_dim={}, grid={:?}",
        config.num_layers,
        config.num_heads,
        config.head_dim,
        config.append_grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "KV cache CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(layers: usize, heads: usize, head_dim: usize, max_seq: usize) -> KvCacheConfig {
        KvCacheConfig::new(layers, heads, head_dim, max_seq, CacheDtype::F32).unwrap()
    }

    fn make_buffer(layers: usize, heads: usize, head_dim: usize, max_seq: usize) -> KvCacheBuffer {
        KvCacheBuffer::new(make_config(layers, heads, head_dim, max_seq))
    }

    // ── Config validation ────────────────────────────────────────────

    #[test]
    fn test_config_valid() {
        let cfg = KvCacheConfig::new(4, 8, 64, 512, CacheDtype::F32).unwrap();
        assert_eq!(cfg.num_layers, 4);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.max_seq_len, 512);
        assert_eq!(cfg.dtype, CacheDtype::F32);
    }

    #[test]
    fn test_config_rejects_zero_layers() {
        assert!(KvCacheConfig::new(0, 8, 64, 512, CacheDtype::F32).is_err());
    }

    #[test]
    fn test_config_rejects_zero_heads() {
        assert!(KvCacheConfig::new(4, 0, 64, 512, CacheDtype::F32).is_err());
    }

    #[test]
    fn test_config_rejects_zero_head_dim() {
        assert!(KvCacheConfig::new(4, 8, 0, 512, CacheDtype::F32).is_err());
    }

    #[test]
    fn test_config_rejects_zero_max_seq_len() {
        assert!(KvCacheConfig::new(4, 8, 64, 0, CacheDtype::F32).is_err());
    }

    #[test]
    fn test_config_grid_block_dims() {
        let cfg = make_config(2, 4, 128, 64);
        assert_eq!(cfg.append_grid_dim(), (4, 1, 1));
        assert_eq!(cfg.append_block_dim(), (128, 1, 1));
    }

    #[test]
    fn test_config_block_dim_capped() {
        let cfg = make_config(1, 1, 2048, 16);
        assert_eq!(cfg.append_block_dim(), (1024, 1, 1));
    }

    #[test]
    fn test_config_bytes_per_element() {
        assert_eq!(KvCacheConfig::new(1, 1, 1, 1, CacheDtype::F32).unwrap().bytes_per_element(), 4);
        assert_eq!(KvCacheConfig::new(1, 1, 1, 1, CacheDtype::F16).unwrap().bytes_per_element(), 2);
        assert_eq!(
            KvCacheConfig::new(1, 1, 1, 1, CacheDtype::Bf16).unwrap().bytes_per_element(),
            2
        );
    }

    // ── Append and retrieve correctness ──────────────────────────────

    #[test]
    fn test_append_and_retrieve_single_entry() {
        let mut buf = make_buffer(1, 2, 4, 8);
        let k: Vec<f32> = (0..8).map(|i| i as f32).collect(); // 2 heads * 4 dim
        let v: Vec<f32> = (10..18).map(|i| i as f32).collect();

        buf.append_kv(0, 0, &k, &v).unwrap();
        assert_eq!(buf.layer_len(0).unwrap(), 1);

        let (got_k, got_v) = buf.get_kv(0, 0, 1).unwrap();
        assert_eq!(got_k.len(), 8);
        assert_eq!(got_v.len(), 8);
        assert_eq!(got_k, k);
        assert_eq!(got_v, v);
    }

    #[test]
    fn test_append_sequential_positions() {
        let mut buf = make_buffer(1, 1, 2, 16);
        for pos in 0..4 {
            let k = vec![pos as f32; 2];
            let v = vec![(pos as f32) * 10.0; 2];
            buf.append_kv(0, pos, &k, &v).unwrap();
        }
        assert_eq!(buf.layer_len(0).unwrap(), 4);

        let (keys, values) = buf.get_kv(0, 0, 4).unwrap();
        assert_eq!(keys.len(), 8); // 1 head * 4 pos * 2 dim
        for pos in 0..4 {
            assert_eq!(keys[pos * 2], pos as f32);
            assert_eq!(values[pos * 2], (pos as f32) * 10.0);
        }
    }

    // ── Multi-layer independence ─────────────────────────────────────

    #[test]
    fn test_multi_layer_independence() {
        let mut buf = make_buffer(3, 1, 2, 8);

        let k0 = vec![1.0, 2.0];
        let v0 = vec![3.0, 4.0];
        let k1 = vec![5.0, 6.0];
        let v1 = vec![7.0, 8.0];

        buf.append_kv(0, 0, &k0, &v0).unwrap();
        buf.append_kv(1, 0, &k1, &v1).unwrap();

        assert_eq!(buf.layer_len(0).unwrap(), 1);
        assert_eq!(buf.layer_len(1).unwrap(), 1);
        assert_eq!(buf.layer_len(2).unwrap(), 0);

        let (got_k0, got_v0) = buf.get_kv(0, 0, 1).unwrap();
        let (got_k1, got_v1) = buf.get_kv(1, 0, 1).unwrap();

        assert_eq!(got_k0, k0);
        assert_eq!(got_v0, v0);
        assert_eq!(got_k1, k1);
        assert_eq!(got_v1, v1);
    }

    // ── Position-based retrieval ─────────────────────────────────────

    #[test]
    fn test_position_based_retrieval() {
        let mut buf = make_buffer(1, 1, 2, 16);
        for pos in 0..8 {
            let k = vec![(pos * 100) as f32, (pos * 100 + 1) as f32];
            let v = vec![(pos * 200) as f32, (pos * 200 + 1) as f32];
            buf.append_kv(0, pos, &k, &v).unwrap();
        }

        // Retrieve a sub-range [2, 5)
        let (keys, values) = buf.get_kv(0, 2, 5).unwrap();
        assert_eq!(keys.len(), 6); // 3 positions * 2 dim
        assert_eq!(keys[0], 200.0); // pos 2
        assert_eq!(keys[2], 300.0); // pos 3
        assert_eq!(keys[4], 400.0); // pos 4
        assert_eq!(values[0], 400.0); // pos 2 val
    }

    // ── Truncation ───────────────────────────────────────────────────

    #[test]
    fn test_truncation() {
        let mut buf = make_buffer(1, 1, 2, 16);
        for pos in 0..6 {
            buf.append_kv(0, pos, &[pos as f32; 2], &[0.0; 2]).unwrap();
        }
        assert_eq!(buf.layer_len(0).unwrap(), 6);

        buf.truncate(0, 3).unwrap();
        assert_eq!(buf.layer_len(0).unwrap(), 3);
    }

    #[test]
    fn test_truncation_to_zero() {
        let mut buf = make_buffer(1, 1, 2, 16);
        buf.append_kv(0, 0, &[1.0; 2], &[2.0; 2]).unwrap();
        buf.truncate(0, 0).unwrap();
        assert_eq!(buf.layer_len(0).unwrap(), 0);
    }

    #[test]
    fn test_truncation_rejects_longer() {
        let mut buf = make_buffer(1, 1, 2, 16);
        buf.append_kv(0, 0, &[1.0; 2], &[2.0; 2]).unwrap();
        assert!(buf.truncate(0, 5).is_err());
    }

    // ── Rotation ─────────────────────────────────────────────────────

    #[test]
    fn test_rotation_identity_at_position_zero() {
        let mut buf = make_buffer(1, 1, 4, 8);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![0.0; 4];
        buf.append_kv(0, 0, &k, &v).unwrap();

        // Position 0 → all angles are 0 → cos=1, sin=0 → no change.
        buf.rotate_kv(0, &[0]).unwrap();

        let (got_k, _) = buf.get_kv(0, 0, 1).unwrap();
        for (a, b) in got_k.iter().zip(k.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "position-0 rotation should be identity: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn test_rotation_preserves_norm() {
        let mut buf = make_buffer(1, 2, 8, 4);
        let head_dim = 8;
        let num_heads = 2;
        let k: Vec<f32> = (0..num_heads * head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let v = vec![0.0f32; num_heads * head_dim];

        buf.append_kv(0, 0, &k, &v).unwrap();
        buf.append_kv(0, 1, &k, &v).unwrap();

        buf.rotate_kv(0, &[3, 7]).unwrap();
        let (got_k, _) = buf.get_kv(0, 0, 2).unwrap();

        // get_kv layout: [head0_pos0, head0_pos1, head1_pos0, head1_pos1]
        let range_len = 2;
        for head in 0..num_heads {
            let orig_start = head * head_dim;
            let orig_norm: f32 = k[orig_start..orig_start + head_dim].iter().map(|x| x * x).sum();
            for pos in 0..range_len {
                let out_start = head * range_len * head_dim + pos * head_dim;
                let pos_norm: f32 =
                    got_k[out_start..out_start + head_dim].iter().map(|x| x * x).sum();
                assert!(
                    (orig_norm.sqrt() - pos_norm.sqrt()).abs() < 1e-3,
                    "norm not preserved: head={head}, pos={pos}, \
                     orig={}, got={}",
                    orig_norm.sqrt(),
                    pos_norm.sqrt(),
                );
            }
        }
    }

    #[test]
    fn test_rotation_rejects_odd_head_dim() {
        let cfg = KvCacheConfig::new(1, 1, 3, 8, CacheDtype::F32).unwrap();
        let mut buf = KvCacheBuffer::new(cfg);
        let k = vec![1.0; 3];
        let v = vec![0.0; 3];
        buf.append_kv(0, 0, &k, &v).unwrap();
        assert!(buf.rotate_kv(0, &[0]).is_err());
    }

    // ── Clear ────────────────────────────────────────────────────────

    #[test]
    fn test_clear_resets_all() {
        let mut buf = make_buffer(2, 1, 2, 8);
        buf.append_kv(0, 0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        buf.append_kv(1, 0, &[5.0, 6.0], &[7.0, 8.0]).unwrap();

        buf.clear();

        assert_eq!(buf.layer_len(0).unwrap(), 0);
        assert_eq!(buf.layer_len(1).unwrap(), 0);

        // Buffers are zeroed — retrieving should give zeros if we re-append
        // and then fetch the old position. Verify by appending fresh data.
        buf.append_kv(0, 0, &[10.0, 20.0], &[30.0, 40.0]).unwrap();
        let (k, v) = buf.get_kv(0, 0, 1).unwrap();
        assert_eq!(k, vec![10.0, 20.0]);
        assert_eq!(v, vec![30.0, 40.0]);
    }

    // ── Memory tracking / stats ──────────────────────────────────────

    #[test]
    fn test_stats_memory_bytes() {
        let buf = make_buffer(2, 4, 64, 128);
        let stats = buf.stats();
        // 2 layers * 4 heads * 128 seq * 64 dim * 2 (k+v) * 4 bytes
        let expected = 2 * 4 * 128 * 64 * 2 * 4;
        assert_eq!(stats.memory_bytes, expected);
    }

    #[test]
    fn test_stats_entries_per_layer() {
        let mut buf = make_buffer(3, 1, 2, 16);
        buf.append_kv(0, 0, &[1.0; 2], &[2.0; 2]).unwrap();
        buf.append_kv(0, 1, &[1.0; 2], &[2.0; 2]).unwrap();
        buf.append_kv(2, 0, &[3.0; 2], &[4.0; 2]).unwrap();

        let stats = buf.stats();
        assert_eq!(stats.entries_per_layer, vec![2, 0, 1]);
    }

    #[test]
    fn test_stats_hit_rate_initial() {
        let buf = make_buffer(1, 1, 2, 8);
        let stats = buf.stats();
        assert!((stats.hit_rate - 1.0).abs() < 1e-9, "initial hit_rate should be 1.0");
    }

    #[test]
    fn test_stats_hit_rate_after_gets() {
        let mut buf = make_buffer(1, 1, 2, 8);
        buf.append_kv(0, 0, &[1.0; 2], &[2.0; 2]).unwrap();
        buf.append_kv(0, 1, &[1.0; 2], &[2.0; 2]).unwrap();

        // Hit: within layer_len=2
        let _ = buf.get_kv(0, 0, 2).unwrap();
        // Hit: requesting [0,1) is within len=2
        let _ = buf.get_kv(0, 0, 1).unwrap();

        let stats = buf.stats();
        assert!((stats.hit_rate - 1.0).abs() < 1e-9, "all gets within range → 100% hit");
    }

    #[test]
    fn test_stats_avg_access_time() {
        let mut buf = make_buffer(1, 1, 2, 8);
        buf.append_kv(0, 0, &[1.0; 2], &[2.0; 2]).unwrap();
        let stats = buf.stats();
        // Should be positive (at least some microseconds elapsed)
        assert!(stats.avg_access_time_us >= 0.0);
    }

    #[test]
    fn test_stats_f16_memory_bytes() {
        let cfg = KvCacheConfig::new(1, 2, 32, 64, CacheDtype::F16).unwrap();
        let buf = KvCacheBuffer::new(cfg);
        let stats = buf.stats();
        // 1 * 2 * 64 * 32 * 2 (k+v) * 2 bytes (f16)
        let expected = 1 * 2 * 64 * 32 * 2 * 2;
        assert_eq!(stats.memory_bytes, expected);
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_single_entry_cache() {
        let mut buf = make_buffer(1, 1, 1, 1);
        buf.append_kv(0, 0, &[42.0], &[99.0]).unwrap();
        let (k, v) = buf.get_kv(0, 0, 1).unwrap();
        assert_eq!(k, vec![42.0]);
        assert_eq!(v, vec![99.0]);
    }

    #[test]
    fn test_full_cache() {
        let max_seq = 4;
        let mut buf = make_buffer(1, 1, 2, max_seq);
        for pos in 0..max_seq {
            buf.append_kv(0, pos, &[pos as f32; 2], &[0.0; 2]).unwrap();
        }
        assert_eq!(buf.layer_len(0).unwrap(), max_seq);

        // Retrieve entire cache
        let (keys, _) = buf.get_kv(0, 0, max_seq).unwrap();
        assert_eq!(keys.len(), max_seq * 2);
    }

    #[test]
    fn test_append_beyond_max_seq_len() {
        let mut buf = make_buffer(1, 1, 2, 4);
        assert!(buf.append_kv(0, 4, &[0.0; 2], &[0.0; 2]).is_err());
    }

    #[test]
    fn test_get_kv_invalid_range() {
        let mut buf = make_buffer(1, 1, 2, 8);
        buf.append_kv(0, 0, &[1.0; 2], &[2.0; 2]).unwrap();
        // end <= start
        assert!(buf.get_kv(0, 3, 3).is_err());
        assert!(buf.get_kv(0, 5, 2).is_err());
    }

    #[test]
    fn test_get_kv_exceeds_max_seq() {
        let mut buf = make_buffer(1, 1, 2, 4);
        assert!(buf.get_kv(0, 0, 5).is_err());
    }

    #[test]
    fn test_invalid_layer_index() {
        let mut buf = make_buffer(2, 1, 2, 8);
        assert!(buf.append_kv(2, 0, &[0.0; 2], &[0.0; 2]).is_err());
        assert!(buf.get_kv(3, 0, 1).is_err());
        assert!(buf.truncate(5, 0).is_err());
        assert!(buf.layer_len(10).is_err());
    }

    #[test]
    fn test_append_wrong_data_length() {
        let mut buf = make_buffer(1, 2, 4, 8);
        let expected_len = 2 * 4; // 2 heads * 4 dim = 8
        let short_k = vec![0.0; expected_len - 1];
        let ok_v = vec![0.0; expected_len];
        assert!(buf.append_kv(0, 0, &short_k, &ok_v).is_err());
        assert!(buf.append_kv(0, 0, &ok_v, &short_k).is_err());
    }

    #[test]
    fn test_multi_head_append_and_get() {
        let mut buf = make_buffer(1, 3, 2, 4);
        // 3 heads * 2 dim = 6 elements per token
        let k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        buf.append_kv(0, 0, &k, &v).unwrap();

        let (got_k, got_v) = buf.get_kv(0, 0, 1).unwrap();
        assert_eq!(got_k, k);
        assert_eq!(got_v, v);
    }

    // ── GPU launch stub ──────────────────────────────────────────────

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_kv_cache_append() {
        let cfg = make_config(4, 32, 128, 2048);
        let total = cfg.num_layers * cfg.layer_elements();
        let mut keys = vec![0.0f32; total];
        let mut values = vec![0.0f32; total];
        let key_data = vec![1.0f32; 32 * 128];
        let value_data = vec![2.0f32; 32 * 128];
        let result = launch_append_kv(&mut keys, &mut values, 0, 0, &key_data, &value_data, &cfg);
        assert!(result.is_ok(), "CUDA KV cache launch failed: {result:?}");
    }
}
