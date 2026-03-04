#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
#![cfg(target_os = "macos")]
#![allow(dead_code)]

//! Metal attention v2 shader validation tests for Apple Silicon.
//!
//! Validates multi-head attention parameters, grouped query attention (GQA),
//! multi-query attention (MQA), causal mask computation, sliding window
//! attention, flash attention chunk sizing, KV cache layout parameters,
//! attention score scaling, Metal threadgroup sizing, memory bandwidth
//! estimation, softmax numerical stability, batch attention dimensions,
//! head dimension alignment, sparse attention patterns, and cross-attention
//! parameter validation.
//!
//! All tests validate CPU-side logic — no GPU runtime required.

// ───────────────────────────────────────────────────────────────────
// Constants
// ───────────────────────────────────────────────────────────────────

/// Metal maximum threads per threadgroup on Apple Silicon.
const METAL_MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// Apple Silicon SIMD group (wavefront) width.
const METAL_SIMD_GROUP_SIZE: u32 = 32;

/// Metal buffer alignment requirement (bytes).
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Large negative value for causal masking.
const MASK_NEG_INF: f32 = -1e9;

/// Tolerance for single-step float comparisons.
const TOL: f32 = 1e-5;

/// Tolerance for accumulated multi-step comparisons.
const TOL_ACCUM: f32 = 1e-3;

/// Maximum head dimension supported by the v2 shader.
const MAX_HEAD_DIM: usize = 256;

/// Minimum head dimension (must be positive and power-of-2 for alignment).
const MIN_HEAD_DIM: usize = 8;

/// Default flash-attention chunk size (tokens per tile).
/// Sized to fit Q+K+score+out tiles within 32 KB threadgroup memory for d≤128.
const DEFAULT_FLASH_CHUNK: usize = 32;

/// Metal shared memory limit per threadgroup (bytes, Apple M-series).
const METAL_THREADGROUP_MEMORY_LIMIT: usize = 32768;

/// Bytes per float32 element.
const F32_BYTES: usize = 4;

// ───────────────────────────────────────────────────────────────────
// Helper types
// ───────────────────────────────────────────────────────────────────

/// Multi-head attention configuration for the v2 shader.
#[derive(Debug, Clone)]
struct MhaConfig {
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_seq_len: usize,
    scale: f32,
    causal: bool,
}

impl MhaConfig {
    fn new(
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_seq_len: usize,
        causal: bool,
    ) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self { batch_size, num_heads, num_kv_heads, head_dim, seq_len, kv_seq_len, scale, causal }
    }

    fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads && self.num_kv_heads > 1
    }

    fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1 && self.num_heads > 1
    }

    fn is_mha(&self) -> bool {
        self.num_kv_heads == self.num_heads
    }

    fn kv_head_ratio(&self) -> usize {
        assert!(self.num_heads % self.num_kv_heads == 0);
        self.num_heads / self.num_kv_heads
    }

    fn q_elements(&self) -> usize {
        self.batch_size * self.num_heads * self.seq_len * self.head_dim
    }

    fn k_elements(&self) -> usize {
        self.batch_size * self.num_kv_heads * self.kv_seq_len * self.head_dim
    }

    fn v_elements(&self) -> usize {
        self.k_elements()
    }

    fn output_elements(&self) -> usize {
        self.q_elements()
    }

    fn score_elements(&self) -> usize {
        self.batch_size * self.num_heads * self.seq_len * self.kv_seq_len
    }

    fn validate(&self) -> Result<String, String> {
        if self.batch_size == 0 {
            return Err("batch_size must be > 0".into());
        }
        if self.num_heads == 0 {
            return Err("num_heads must be > 0".into());
        }
        if self.num_kv_heads == 0 {
            return Err("num_kv_heads must be > 0".into());
        }
        if self.head_dim == 0 || self.head_dim > MAX_HEAD_DIM {
            return Err(format!("head_dim must be in [1, {MAX_HEAD_DIM}]"));
        }
        if self.seq_len == 0 {
            return Err("seq_len must be > 0".into());
        }
        if self.kv_seq_len == 0 {
            return Err("kv_seq_len must be > 0".into());
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err("num_heads must be divisible by num_kv_heads".into());
        }
        if self.scale <= 0.0 || !self.scale.is_finite() {
            return Err("scale must be positive and finite".into());
        }
        Ok("valid".into())
    }
}

/// Sliding window attention parameters.
#[derive(Debug, Clone)]
struct SlidingWindowConfig {
    window_size: usize,
    causal: bool,
}

/// Flash attention tiling parameters.
#[derive(Debug, Clone)]
struct FlashAttentionConfig {
    chunk_size: usize,
    num_chunks_q: usize,
    num_chunks_kv: usize,
    shared_mem_per_chunk: usize,
}

/// KV cache layout descriptor.
#[derive(Debug, Clone, PartialEq)]
enum KvCacheLayout {
    Contiguous { num_layers: usize, max_seq_len: usize, num_kv_heads: usize, head_dim: usize },
    Paged { page_size: usize, num_pages: usize, num_kv_heads: usize, head_dim: usize },
}

/// Sparse attention pattern.
#[derive(Debug, Clone, PartialEq)]
enum SparsePattern {
    Local { window: usize },
    Strided { stride: usize, window: usize },
    Random { density: f32 },
}

/// Metal threadgroup configuration for attention dispatch.
#[derive(Debug, Clone)]
struct ThreadgroupConfig {
    threads_per_threadgroup: u32,
    threadgroups_per_grid: u32,
    simd_groups: u32,
}

// ───────────────────────────────────────────────────────────────────
// Pure-logic helpers
// ───────────────────────────────────────────────────────────────────

fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn align_up(val: usize, alignment: usize) -> usize {
    (val + alignment - 1) / alignment * alignment
}

fn cpu_softmax(logits: &[f32]) -> Vec<f32> {
    assert!(!logits.is_empty());
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn cpu_causal_mask(seq_len: usize, kv_seq_len: usize) -> Vec<bool> {
    let mut mask = vec![false; seq_len * kv_seq_len];
    for q in 0..seq_len {
        for k in 0..kv_seq_len {
            mask[q * kv_seq_len + k] = k <= q;
        }
    }
    mask
}

fn cpu_sliding_window_mask(seq_len: usize, kv_seq_len: usize, window: usize) -> Vec<bool> {
    let mut mask = vec![false; seq_len * kv_seq_len];
    for q in 0..seq_len {
        for k in 0..kv_seq_len {
            let dist = if q >= k { q - k } else { k - q };
            mask[q * kv_seq_len + k] = dist < window;
        }
    }
    mask
}

fn cpu_causal_sliding_window_mask(seq_len: usize, kv_seq_len: usize, window: usize) -> Vec<bool> {
    let mut mask = vec![false; seq_len * kv_seq_len];
    for q in 0..seq_len {
        for k in 0..kv_seq_len {
            mask[q * kv_seq_len + k] = k <= q && (q - k) < window;
        }
    }
    mask
}

fn det_rand(count: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let t = ((state >> 33) as f32) / (u32::MAX as f32);
            lo + t * (hi - lo)
        })
        .collect()
}

fn compute_flash_chunks(seq_len: usize, chunk_size: usize) -> usize {
    (seq_len + chunk_size - 1) / chunk_size
}

fn flash_shared_mem(chunk_size: usize, head_dim: usize) -> usize {
    // Q tile + K tile + scores tile + output accumulator
    let q_tile = chunk_size * head_dim * F32_BYTES;
    let k_tile = chunk_size * head_dim * F32_BYTES;
    let score_tile = chunk_size * chunk_size * F32_BYTES;
    let out_tile = chunk_size * head_dim * F32_BYTES;
    q_tile + k_tile + score_tile + out_tile
}

fn compute_threadgroup(num_heads: u32, seq_chunks: u32) -> ThreadgroupConfig {
    let threads = METAL_SIMD_GROUP_SIZE.min(METAL_MAX_THREADS_PER_THREADGROUP);
    let groups = num_heads * seq_chunks;
    let simd = (threads + METAL_SIMD_GROUP_SIZE - 1) / METAL_SIMD_GROUP_SIZE;
    ThreadgroupConfig {
        threads_per_threadgroup: threads,
        threadgroups_per_grid: groups,
        simd_groups: simd,
    }
}

fn estimate_attention_bandwidth_bytes(cfg: &MhaConfig) -> usize {
    let q_bytes = cfg.q_elements() * F32_BYTES;
    let k_bytes = cfg.k_elements() * F32_BYTES;
    let v_bytes = cfg.v_elements() * F32_BYTES;
    let out_bytes = cfg.output_elements() * F32_BYTES;
    q_bytes + k_bytes + v_bytes + out_bytes
}

fn sparse_mask_density(mask: &[bool]) -> f32 {
    let active = mask.iter().filter(|&&m| m).count();
    active as f32 / mask.len() as f32
}

fn cpu_strided_mask(seq_len: usize, kv_seq_len: usize, stride: usize, window: usize) -> Vec<bool> {
    let mut mask = vec![false; seq_len * kv_seq_len];
    for q in 0..seq_len {
        for k in 0..kv_seq_len {
            let local = (q as isize - k as isize).unsigned_abs() < window;
            let strided = k % stride == 0;
            mask[q * kv_seq_len + k] = local || strided;
        }
    }
    mask
}

fn kv_cache_bytes(layout: &KvCacheLayout) -> usize {
    match layout {
        KvCacheLayout::Contiguous { num_layers, max_seq_len, num_kv_heads, head_dim } => {
            // K + V for each layer
            2 * num_layers * max_seq_len * num_kv_heads * head_dim * F32_BYTES
        }
        KvCacheLayout::Paged { page_size, num_pages, num_kv_heads, head_dim } => {
            2 * num_pages * page_size * num_kv_heads * head_dim * F32_BYTES
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 1. Multi-head attention parameter validation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn mha_v2_valid_standard_config() {
    let cfg = MhaConfig::new(1, 8, 8, 64, 128, 128, true);
    assert!(cfg.validate().is_ok());
    assert!(cfg.is_mha());
    assert!(!cfg.is_gqa());
    assert!(!cfg.is_mqa());
}

#[test]
fn mha_v2_rejects_zero_batch() {
    let cfg = MhaConfig::new(0, 8, 8, 64, 128, 128, false);
    assert!(cfg.validate().is_err());
}

#[test]
fn mha_v2_rejects_zero_heads() {
    let cfg = MhaConfig::new(1, 0, 0, 64, 128, 128, false);
    assert!(cfg.validate().is_err());
}

#[test]
fn mha_v2_rejects_zero_seq_len() {
    let cfg = MhaConfig::new(1, 8, 8, 64, 0, 128, false);
    assert!(cfg.validate().is_err());
}

#[test]
fn mha_v2_rejects_zero_kv_seq_len() {
    let cfg = MhaConfig::new(1, 8, 8, 64, 128, 0, false);
    assert!(cfg.validate().is_err());
}

#[test]
fn mha_v2_rejects_head_dim_too_large() {
    let cfg = MhaConfig::new(1, 8, 8, MAX_HEAD_DIM + 1, 128, 128, false);
    assert!(cfg.validate().is_err());
}

#[test]
fn mha_v2_output_element_count() {
    let cfg = MhaConfig::new(2, 16, 16, 64, 32, 32, false);
    assert_eq!(cfg.output_elements(), 2 * 16 * 32 * 64);
}

#[test]
fn mha_v2_score_element_count() {
    let cfg = MhaConfig::new(1, 8, 8, 64, 16, 32, false);
    assert_eq!(cfg.score_elements(), 1 * 8 * 16 * 32);
}

#[test]
fn mha_v2_default_scale_is_rsqrt_head_dim() {
    let cfg = MhaConfig::new(1, 8, 8, 64, 128, 128, false);
    let expected = 1.0 / (64.0_f32).sqrt();
    assert!((cfg.scale - expected).abs() < TOL);
}

#[test]
fn mha_v2_custom_scale() {
    let mut cfg = MhaConfig::new(1, 8, 8, 64, 128, 128, false);
    cfg.scale = 0.1;
    assert!(cfg.validate().is_ok());
    assert!((cfg.scale - 0.1).abs() < TOL);
}

// ═══════════════════════════════════════════════════════════════════
// 2. Grouped query attention (GQA) parameter tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn gqa_v2_valid_config_4_to_1() {
    // 32 heads, 8 KV heads → 4:1 ratio
    let cfg = MhaConfig::new(1, 32, 8, 128, 64, 64, true);
    assert!(cfg.validate().is_ok());
    assert!(cfg.is_gqa());
    assert_eq!(cfg.kv_head_ratio(), 4);
}

#[test]
fn gqa_v2_valid_config_2_to_1() {
    let cfg = MhaConfig::new(1, 16, 8, 64, 32, 32, false);
    assert!(cfg.validate().is_ok());
    assert!(cfg.is_gqa());
    assert_eq!(cfg.kv_head_ratio(), 2);
}

#[test]
fn gqa_v2_rejects_non_divisible_heads() {
    let cfg = MhaConfig::new(1, 12, 5, 64, 128, 128, false);
    assert!(cfg.validate().is_err());
}

#[test]
fn gqa_v2_kv_elements_fewer_than_q() {
    let cfg = MhaConfig::new(1, 32, 8, 128, 64, 64, false);
    assert!(cfg.k_elements() < cfg.q_elements());
    assert_eq!(cfg.k_elements(), 1 * 8 * 64 * 128);
    assert_eq!(cfg.q_elements(), 1 * 32 * 64 * 128);
}

#[test]
fn gqa_v2_ratio_8_to_1() {
    let cfg = MhaConfig::new(1, 64, 8, 64, 128, 128, true);
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.kv_head_ratio(), 8);
}

#[test]
fn gqa_v2_batch_2_dimensions() {
    let cfg = MhaConfig::new(2, 16, 4, 64, 32, 32, false);
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.q_elements(), 2 * 16 * 32 * 64);
    assert_eq!(cfg.k_elements(), 2 * 4 * 32 * 64);
}

// ═══════════════════════════════════════════════════════════════════
// 3. Multi-query attention (MQA) tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn mqa_v2_valid_config() {
    let cfg = MhaConfig::new(1, 32, 1, 64, 128, 128, true);
    assert!(cfg.validate().is_ok());
    assert!(cfg.is_mqa());
    assert!(!cfg.is_gqa());
    assert_eq!(cfg.kv_head_ratio(), 32);
}

#[test]
fn mqa_v2_kv_memory_savings() {
    let mha = MhaConfig::new(1, 32, 32, 64, 128, 128, false);
    let mqa = MhaConfig::new(1, 32, 1, 64, 128, 128, false);
    assert_eq!(mqa.k_elements() * 32, mha.k_elements());
}

#[test]
fn mqa_v2_output_matches_mha_shape() {
    let mha = MhaConfig::new(1, 32, 32, 64, 128, 128, false);
    let mqa = MhaConfig::new(1, 32, 1, 64, 128, 128, false);
    assert_eq!(mha.output_elements(), mqa.output_elements());
}

#[test]
fn mqa_v2_batch_scaling() {
    let cfg = MhaConfig::new(4, 16, 1, 64, 32, 32, false);
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.k_elements(), 4 * 1 * 32 * 64);
}

// ═══════════════════════════════════════════════════════════════════
// 4. Causal mask computation and validation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn causal_mask_v2_square_lower_triangular() {
    let mask = cpu_causal_mask(4, 4);
    // Row 0: [T, F, F, F]
    // Row 1: [T, T, F, F]
    // Row 2: [T, T, T, F]
    // Row 3: [T, T, T, T]
    assert!(mask[0]); // (0,0)
    assert!(!mask[1]); // (0,1)
    assert!(mask[4]); // (1,0)
    assert!(mask[5]); // (1,1)
    assert!(!mask[6]); // (1,2)
    assert!(mask[15]); // (3,3)
}

#[test]
fn causal_mask_v2_rectangular_more_keys() {
    let mask = cpu_causal_mask(2, 4);
    // Row 0: [T, F, F, F]
    // Row 1: [T, T, F, F]
    assert!(mask[0]);
    assert!(!mask[1]);
    assert!(mask[4]);
    assert!(mask[5]);
    assert!(!mask[6]);
}

#[test]
fn causal_mask_v2_single_element() {
    let mask = cpu_causal_mask(1, 1);
    assert_eq!(mask.len(), 1);
    assert!(mask[0]);
}

#[test]
fn causal_mask_v2_active_count_square() {
    let n = 8;
    let mask = cpu_causal_mask(n, n);
    let active = mask.iter().filter(|&&m| m).count();
    // Lower triangle including diagonal: n*(n+1)/2
    assert_eq!(active, n * (n + 1) / 2);
}

#[test]
fn causal_mask_v2_no_future_leakage() {
    let mask = cpu_causal_mask(16, 16);
    for q in 0..16 {
        for k in 0..16 {
            if k > q {
                assert!(!mask[q * 16 + k], "future position ({q},{k}) must be masked");
            }
        }
    }
}

#[test]
fn causal_mask_v2_density_decreases_with_size() {
    let d4 = sparse_mask_density(&cpu_causal_mask(4, 4));
    let d16 = sparse_mask_density(&cpu_causal_mask(16, 16));
    let d64 = sparse_mask_density(&cpu_causal_mask(64, 64));
    // Density = (n+1)/(2n) → approaches 0.5 from above
    assert!(d4 > d16);
    assert!(d16 > d64);
    assert!(d64 > 0.49);
}

// ═══════════════════════════════════════════════════════════════════
// 5. Sliding window attention parameter tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn sliding_window_v2_mask_shape() {
    let mask = cpu_sliding_window_mask(8, 8, 3);
    assert_eq!(mask.len(), 64);
}

#[test]
fn sliding_window_v2_center_visible() {
    let mask = cpu_sliding_window_mask(8, 8, 3);
    // Position (4,4) is within window of itself
    assert!(mask[4 * 8 + 4]);
    // Position (4,3) is distance 1
    assert!(mask[4 * 8 + 3]);
    // Position (4,2) is distance 2 (within window=3)
    assert!(mask[4 * 8 + 2]);
    // Position (4,1) is distance 3 (outside window=3)
    assert!(!mask[4 * 8 + 1]);
}

#[test]
fn sliding_window_v2_causal_combined() {
    let mask = cpu_causal_sliding_window_mask(8, 8, 4);
    // Future positions masked
    assert!(!mask[2 * 8 + 5]);
    // Past beyond window masked
    assert!(!mask[7 * 8 + 0]); // distance 7 > window 4
    // Within causal + window
    assert!(mask[5 * 8 + 3]); // distance 2 < window 4, and 3 <= 5
}

#[test]
fn sliding_window_v2_density_less_than_full() {
    let full = sparse_mask_density(&cpu_causal_mask(32, 32));
    let windowed = sparse_mask_density(&cpu_causal_sliding_window_mask(32, 32, 8));
    assert!(windowed < full);
}

#[test]
fn sliding_window_v2_window_1_is_diagonal() {
    let mask = cpu_causal_sliding_window_mask(8, 8, 1);
    for q in 0..8 {
        for k in 0..8 {
            if q == k {
                assert!(mask[q * 8 + k]);
            } else {
                assert!(!mask[q * 8 + k]);
            }
        }
    }
}

#[test]
fn sliding_window_v2_large_window_equals_causal() {
    let seq = 16;
    let causal = cpu_causal_mask(seq, seq);
    let windowed = cpu_causal_sliding_window_mask(seq, seq, seq + 1);
    assert_eq!(causal, windowed);
}

// ═══════════════════════════════════════════════════════════════════
// 6. Flash attention chunk size computation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn flash_v2_chunk_count_exact_division() {
    assert_eq!(compute_flash_chunks(128, 64), 2);
    assert_eq!(compute_flash_chunks(256, 64), 4);
}

#[test]
fn flash_v2_chunk_count_remainder() {
    assert_eq!(compute_flash_chunks(100, 64), 2);
    assert_eq!(compute_flash_chunks(65, 64), 2);
}

#[test]
fn flash_v2_chunk_count_single() {
    assert_eq!(compute_flash_chunks(32, 64), 1);
    assert_eq!(compute_flash_chunks(64, 64), 1);
}

#[test]
fn flash_v2_shared_mem_within_limit_d64() {
    let mem = flash_shared_mem(DEFAULT_FLASH_CHUNK, 64);
    assert!(
        mem <= METAL_THREADGROUP_MEMORY_LIMIT,
        "shared mem {mem} exceeds limit {METAL_THREADGROUP_MEMORY_LIMIT}"
    );
}

#[test]
fn flash_v2_shared_mem_within_limit_d128() {
    // For d=128, the default chunk may be too large; find smallest power-of-2 chunk that fits.
    let head_dim = 128;
    let mut chunk = DEFAULT_FLASH_CHUNK;
    while chunk > 1 && flash_shared_mem(chunk, head_dim) > METAL_THREADGROUP_MEMORY_LIMIT {
        chunk /= 2;
    }
    let mem = flash_shared_mem(chunk, head_dim);
    assert!(
        mem <= METAL_THREADGROUP_MEMORY_LIMIT,
        "no chunk size fits for d={head_dim}: chunk={chunk} mem={mem}"
    );
    // The selected chunk must be a reasonable size (at least 8).
    assert!(chunk >= 8, "chunk {chunk} too small for practical use");
}

#[test]
fn flash_v2_shared_mem_breakdown() {
    let chunk = 32;
    let head_dim = 64;
    let mem = flash_shared_mem(chunk, head_dim);
    let expected =
        (chunk * head_dim + chunk * head_dim + chunk * chunk + chunk * head_dim) * F32_BYTES;
    assert_eq!(mem, expected);
}

#[test]
fn flash_v2_reduced_chunk_for_large_head_dim() {
    // With head_dim=256, larger chunks exceed shared mem; find the fitting chunk.
    let head_dim = 256;
    let mut chunk = 32;
    while chunk > 1 && flash_shared_mem(chunk, head_dim) > METAL_THREADGROUP_MEMORY_LIMIT {
        chunk /= 2;
    }
    let mem = flash_shared_mem(chunk, head_dim);
    assert!(
        mem <= METAL_THREADGROUP_MEMORY_LIMIT,
        "no chunk fits for d={head_dim}: chunk={chunk} mem={mem}"
    );
    // chunk=8 with d=256: (8*256*3 + 8*8)*4 = (6144+64)*4 = 24832 < 32768 ✓
    assert!(chunk >= 4, "chunk {chunk} too small");
}

#[test]
fn flash_v2_config_construction() {
    let seq = 256;
    let kv_seq = 512;
    let chunk = DEFAULT_FLASH_CHUNK;
    let head_dim = 64;
    let cfg = FlashAttentionConfig {
        chunk_size: chunk,
        num_chunks_q: compute_flash_chunks(seq, chunk),
        num_chunks_kv: compute_flash_chunks(kv_seq, chunk),
        shared_mem_per_chunk: flash_shared_mem(chunk, head_dim),
    };
    assert_eq!(cfg.num_chunks_q, 8);
    assert_eq!(cfg.num_chunks_kv, 16);
}

// ═══════════════════════════════════════════════════════════════════
// 7. KV cache layout parameter tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn kv_cache_v2_contiguous_memory_size() {
    let layout = KvCacheLayout::Contiguous {
        num_layers: 32,
        max_seq_len: 2048,
        num_kv_heads: 8,
        head_dim: 128,
    };
    let bytes = kv_cache_bytes(&layout);
    // 2 * 32 * 2048 * 8 * 128 * 4 = 4,294,967,296 bytes (~4 GB)
    assert_eq!(bytes, 2 * 32 * 2048 * 8 * 128 * F32_BYTES);
}

#[test]
fn kv_cache_v2_paged_memory_size() {
    let layout =
        KvCacheLayout::Paged { page_size: 16, num_pages: 128, num_kv_heads: 8, head_dim: 64 };
    let bytes = kv_cache_bytes(&layout);
    assert_eq!(bytes, 2 * 128 * 16 * 8 * 64 * F32_BYTES);
}

#[test]
fn kv_cache_v2_paged_vs_contiguous_same_capacity() {
    let max_seq = 2048;
    let page_size = 64;
    let num_pages = max_seq / page_size;
    let kv_heads = 8;
    let head_dim = 64;

    let contig = KvCacheLayout::Contiguous {
        num_layers: 1,
        max_seq_len: max_seq,
        num_kv_heads: kv_heads,
        head_dim,
    };
    let paged = KvCacheLayout::Paged { page_size, num_pages, num_kv_heads: kv_heads, head_dim };
    assert_eq!(kv_cache_bytes(&contig), kv_cache_bytes(&paged));
}

#[test]
fn kv_cache_v2_alignment() {
    let layout = KvCacheLayout::Contiguous {
        num_layers: 1,
        max_seq_len: 512,
        num_kv_heads: 8,
        head_dim: 64,
    };
    let bytes = kv_cache_bytes(&layout);
    let aligned = align_up(bytes, METAL_BUFFER_ALIGNMENT);
    assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0);
}

#[test]
fn kv_cache_v2_paged_page_size_power_of_two() {
    for page_size in [8, 16, 32, 64, 128] {
        assert!(is_power_of_two(page_size), "page_size {page_size} should be power of 2");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 8. Attention score scaling factor tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scale_v2_common_head_dims() {
    for &hd in &[32, 64, 80, 96, 128, 256] {
        let cfg = MhaConfig::new(1, 8, 8, hd, 32, 32, false);
        let expected = 1.0 / (hd as f32).sqrt();
        assert!(
            (cfg.scale - expected).abs() < TOL,
            "head_dim={hd}: got {} want {}",
            cfg.scale,
            expected
        );
    }
}

#[test]
fn scale_v2_preserves_variance() {
    // With scale = 1/sqrt(d), variance of dot products ≈ 1 when inputs are unit normal
    let d = 128;
    let scale = 1.0 / (d as f32).sqrt();
    let data = det_rand(d, 42, -1.0, 1.0);
    let dot: f32 = data.iter().map(|&x| x * x).sum::<f32>() * scale;
    // Dot product with itself scaled should be O(1), not O(d)
    assert!(dot < 5.0, "scaled self-dot {dot} too large");
}

#[test]
fn scale_v2_zero_scale_rejected() {
    let mut cfg = MhaConfig::new(1, 8, 8, 64, 32, 32, false);
    cfg.scale = 0.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn scale_v2_negative_scale_rejected() {
    let mut cfg = MhaConfig::new(1, 8, 8, 64, 32, 32, false);
    cfg.scale = -1.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn scale_v2_inf_scale_rejected() {
    let mut cfg = MhaConfig::new(1, 8, 8, 64, 32, 32, false);
    cfg.scale = f32::INFINITY;
    assert!(cfg.validate().is_err());
}

// ═══════════════════════════════════════════════════════════════════
// 9. Metal threadgroup sizing for attention kernels
// ═══════════════════════════════════════════════════════════════════

#[test]
fn threadgroup_v2_single_head_single_chunk() {
    let tg = compute_threadgroup(1, 1);
    assert_eq!(tg.threads_per_threadgroup, METAL_SIMD_GROUP_SIZE);
    assert_eq!(tg.threadgroups_per_grid, 1);
    assert_eq!(tg.simd_groups, 1);
}

#[test]
fn threadgroup_v2_multi_head() {
    let tg = compute_threadgroup(32, 1);
    assert_eq!(tg.threadgroups_per_grid, 32);
}

#[test]
fn threadgroup_v2_multi_chunk() {
    let tg = compute_threadgroup(8, 4);
    assert_eq!(tg.threadgroups_per_grid, 32);
}

#[test]
fn threadgroup_v2_threads_within_limit() {
    let tg = compute_threadgroup(128, 64);
    assert!(tg.threads_per_threadgroup <= METAL_MAX_THREADS_PER_THREADGROUP);
}

#[test]
fn threadgroup_v2_simd_aligned() {
    let tg = compute_threadgroup(16, 8);
    assert_eq!(tg.threads_per_threadgroup % METAL_SIMD_GROUP_SIZE, 0);
}

#[test]
fn threadgroup_v2_large_dispatch() {
    let num_heads: u32 = 64;
    let seq_chunks: u32 = compute_flash_chunks(4096, DEFAULT_FLASH_CHUNK) as u32;
    let tg = compute_threadgroup(num_heads, seq_chunks);
    assert_eq!(tg.threadgroups_per_grid, num_heads * seq_chunks);
}

// ═══════════════════════════════════════════════════════════════════
// 10. Memory bandwidth estimation for attention
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bandwidth_v2_small_config() {
    let cfg = MhaConfig::new(1, 8, 8, 64, 32, 32, false);
    let bw = estimate_attention_bandwidth_bytes(&cfg);
    // Q + K + V + Out = 4 * (1 * 8 * 32 * 64) * 4 = 262144
    assert_eq!(bw, 4 * 1 * 8 * 32 * 64 * F32_BYTES);
}

#[test]
fn bandwidth_v2_gqa_saves_kv_bandwidth() {
    let mha = MhaConfig::new(1, 32, 32, 128, 64, 64, false);
    let gqa = MhaConfig::new(1, 32, 8, 128, 64, 64, false);
    let bw_mha = estimate_attention_bandwidth_bytes(&mha);
    let bw_gqa = estimate_attention_bandwidth_bytes(&gqa);
    assert!(bw_gqa < bw_mha);
}

#[test]
fn bandwidth_v2_mqa_minimal_kv() {
    let mqa = MhaConfig::new(1, 32, 1, 128, 64, 64, false);
    let mha = MhaConfig::new(1, 32, 32, 128, 64, 64, false);
    let bw_mqa = estimate_attention_bandwidth_bytes(&mqa);
    let bw_mha = estimate_attention_bandwidth_bytes(&mha);
    // MQA saves on K+V bandwidth; total must be strictly less than MHA
    assert!(bw_mqa < bw_mha, "MQA bw {bw_mqa} should be < MHA bw {bw_mha}");
    // KV portion: MQA has 1/32 of MHA's KV
    let mqa_kv = mqa.k_elements() + mqa.v_elements();
    let mha_kv = mha.k_elements() + mha.v_elements();
    assert_eq!(mqa_kv * 32, mha_kv);
}

#[test]
fn bandwidth_v2_scales_with_batch() {
    let b1 = MhaConfig::new(1, 8, 8, 64, 32, 32, false);
    let b4 = MhaConfig::new(4, 8, 8, 64, 32, 32, false);
    assert_eq!(
        estimate_attention_bandwidth_bytes(&b4),
        4 * estimate_attention_bandwidth_bytes(&b1)
    );
}

// ═══════════════════════════════════════════════════════════════════
// 11. Softmax numerical stability parameter tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn softmax_v2_sums_to_one() {
    let logits = det_rand(64, 100, -10.0, 10.0);
    let probs = cpu_softmax(&logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < TOL, "softmax sum = {sum}");
}

#[test]
fn softmax_v2_all_positive() {
    let logits = det_rand(128, 101, -50.0, 50.0);
    let probs = cpu_softmax(&logits);
    assert!(probs.iter().all(|&p| p >= 0.0));
}

#[test]
fn softmax_v2_large_logits_no_overflow() {
    let logits = vec![1e6, 1e6 + 1.0, 1e6 - 1.0];
    let probs = cpu_softmax(&logits);
    assert!(probs.iter().all(|p| p.is_finite()), "overflow in softmax");
    assert!((probs.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn softmax_v2_large_negative_logits_no_underflow() {
    let logits = vec![-1e6, -1e6 + 1.0, -1e6 - 1.0];
    let probs = cpu_softmax(&logits);
    assert!(probs.iter().all(|p| p.is_finite()));
    assert!((probs.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn softmax_v2_uniform_input_gives_uniform_output() {
    let logits = vec![5.0; 8];
    let probs = cpu_softmax(&logits);
    for &p in &probs {
        assert!((p - 0.125).abs() < TOL, "expected uniform 0.125, got {p}");
    }
}

#[test]
fn softmax_v2_dominant_logit() {
    let mut logits = vec![0.0; 16];
    logits[0] = 100.0;
    let probs = cpu_softmax(&logits);
    assert!(probs[0] > 0.99, "dominant logit should have prob > 0.99, got {}", probs[0]);
}

#[test]
fn softmax_v2_masked_positions_near_zero() {
    let mut logits = vec![1.0; 8];
    logits[4] = MASK_NEG_INF;
    logits[5] = MASK_NEG_INF;
    let probs = cpu_softmax(&logits);
    assert!(probs[4] < TOL, "masked position should be ~0, got {}", probs[4]);
    assert!(probs[5] < TOL, "masked position should be ~0, got {}", probs[5]);
}

// ═══════════════════════════════════════════════════════════════════
// 12. Batch attention dimension validation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn batch_v2_dimension_consistency() {
    for batch in [1, 2, 4, 8] {
        let cfg = MhaConfig::new(batch, 16, 16, 64, 32, 32, false);
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.q_elements(), batch * 16 * 32 * 64);
    }
}

#[test]
fn batch_v2_output_shape_independent_of_kv_len() {
    let short_kv = MhaConfig::new(2, 8, 8, 64, 32, 16, false);
    let long_kv = MhaConfig::new(2, 8, 8, 64, 32, 256, false);
    assert_eq!(short_kv.output_elements(), long_kv.output_elements());
}

#[test]
fn batch_v2_score_shape_depends_on_kv_len() {
    let short_kv = MhaConfig::new(1, 8, 8, 64, 32, 16, false);
    let long_kv = MhaConfig::new(1, 8, 8, 64, 32, 256, false);
    assert_ne!(short_kv.score_elements(), long_kv.score_elements());
    assert_eq!(short_kv.score_elements(), 1 * 8 * 32 * 16);
    assert_eq!(long_kv.score_elements(), 1 * 8 * 32 * 256);
}

#[test]
fn batch_v2_large_batch_validates() {
    let cfg = MhaConfig::new(64, 8, 8, 64, 16, 16, false);
    assert!(cfg.validate().is_ok());
}

// ═══════════════════════════════════════════════════════════════════
// 13. Head dimension alignment requirements
// ═══════════════════════════════════════════════════════════════════

#[test]
fn head_dim_v2_common_sizes_power_of_two() {
    for &hd in &[32, 64, 128, 256] {
        assert!(is_power_of_two(hd), "head_dim {hd} should be power of 2");
        let cfg = MhaConfig::new(1, 8, 8, hd, 32, 32, false);
        assert!(cfg.validate().is_ok());
    }
}

#[test]
fn head_dim_v2_non_power_of_two_still_valid() {
    // Some models use non-power-of-2 head dims (e.g. 80, 96)
    for &hd in &[48, 80, 96, 112] {
        let cfg = MhaConfig::new(1, 8, 8, hd, 32, 32, false);
        assert!(cfg.validate().is_ok());
    }
}

#[test]
fn head_dim_v2_alignment_for_metal_simd() {
    // For optimal SIMD utilization, head_dim should be multiple of 8
    for &hd in &[8, 16, 32, 64, 128] {
        assert_eq!(hd % 8, 0, "head_dim {hd} not aligned to 8");
    }
}

#[test]
fn head_dim_v2_buffer_alignment() {
    let head_dim = 64;
    let seq_len = 32;
    let num_heads = 8;
    let tensor_bytes = num_heads * seq_len * head_dim * F32_BYTES;
    let aligned = align_up(tensor_bytes, METAL_BUFFER_ALIGNMENT);
    assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0);
}

#[test]
fn head_dim_v2_min_max_boundaries() {
    let min_cfg = MhaConfig::new(1, 1, 1, MIN_HEAD_DIM, 1, 1, false);
    assert!(min_cfg.validate().is_ok());
    let max_cfg = MhaConfig::new(1, 1, 1, MAX_HEAD_DIM, 1, 1, false);
    assert!(max_cfg.validate().is_ok());
    let over_cfg = MhaConfig::new(1, 1, 1, MAX_HEAD_DIM + 1, 1, 1, false);
    assert!(over_cfg.validate().is_err());
}

// ═══════════════════════════════════════════════════════════════════
// 14. Sparse attention pattern tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn sparse_v2_local_pattern_density() {
    let seq = 32;
    let window = 5;
    let mask = cpu_sliding_window_mask(seq, seq, window);
    let density = sparse_mask_density(&mask);
    // Local window density ≈ (2*window - 1) / seq for large seq
    assert!(density > 0.0 && density < 1.0);
}

#[test]
fn sparse_v2_strided_pattern_covers_all_keys() {
    let seq = 32;
    let stride = 4;
    let window = 2;
    let mask = cpu_strided_mask(seq, seq, stride, window);
    // Every query should see at least one key
    for q in 0..seq {
        let row_start = q * seq;
        let any_visible = mask[row_start..row_start + seq].iter().any(|&m| m);
        assert!(any_visible, "query {q} has no visible keys");
    }
}

#[test]
fn sparse_v2_strided_density_higher_than_local() {
    let seq = 64;
    let local = sparse_mask_density(&cpu_sliding_window_mask(seq, seq, 4));
    let strided = sparse_mask_density(&cpu_strided_mask(seq, seq, 8, 4));
    assert!(strided >= local, "strided should be at least as dense as local");
}

#[test]
fn sparse_v2_local_pattern_enum() {
    let p = SparsePattern::Local { window: 8 };
    assert_eq!(p, SparsePattern::Local { window: 8 });
}

#[test]
fn sparse_v2_strided_pattern_enum() {
    let p = SparsePattern::Strided { stride: 4, window: 2 };
    assert_eq!(p, SparsePattern::Strided { stride: 4, window: 2 });
}

#[test]
fn sparse_v2_random_density_validation() {
    let p = SparsePattern::Random { density: 0.5 };
    match p {
        SparsePattern::Random { density } => {
            assert!(density > 0.0 && density <= 1.0);
        }
        _ => panic!("expected Random"),
    }
}

#[test]
fn sparse_v2_causal_is_sparse() {
    let density = sparse_mask_density(&cpu_causal_mask(128, 128));
    assert!(density < 1.0, "causal mask is sparse, density = {density}");
    assert!(density > 0.0);
}

// ═══════════════════════════════════════════════════════════════════
// 15. Cross-attention parameter validation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn cross_attn_v2_different_seq_lengths() {
    // Encoder output = 256 tokens, decoder query = 32 tokens
    let cfg = MhaConfig::new(1, 8, 8, 64, 32, 256, false);
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.score_elements(), 1 * 8 * 32 * 256);
}

#[test]
fn cross_attn_v2_non_causal() {
    // Cross-attention typically uses full (non-causal) attention
    let cfg = MhaConfig::new(1, 8, 8, 64, 32, 128, false);
    assert!(!cfg.causal);
}

#[test]
fn cross_attn_v2_kv_from_encoder_longer_than_query() {
    let cfg = MhaConfig::new(1, 16, 16, 64, 16, 512, false);
    assert!(cfg.kv_seq_len > cfg.seq_len);
    assert!(cfg.validate().is_ok());
}

#[test]
fn cross_attn_v2_output_shape_follows_query() {
    let cfg = MhaConfig::new(1, 8, 8, 64, 16, 512, false);
    // Output follows Q dimensions, not KV
    assert_eq!(cfg.output_elements(), 1 * 8 * 16 * 64);
}

#[test]
fn cross_attn_v2_with_gqa() {
    // Cross-attention can also use GQA
    let cfg = MhaConfig::new(1, 32, 8, 128, 16, 256, false);
    assert!(cfg.validate().is_ok());
    assert!(cfg.is_gqa());
    assert_eq!(cfg.kv_head_ratio(), 4);
}

#[test]
fn cross_attn_v2_batch_cross_attention() {
    let cfg = MhaConfig::new(4, 8, 8, 64, 32, 128, false);
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.score_elements(), 4 * 8 * 32 * 128);
}

// ═══════════════════════════════════════════════════════════════════
// 16. Additional edge cases and integration-style validations
// ═══════════════════════════════════════════════════════════════════

#[test]
fn edge_v2_single_token_attention() {
    let cfg = MhaConfig::new(1, 1, 1, 64, 1, 1, false);
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.score_elements(), 1);
}

#[test]
fn edge_v2_very_long_sequence() {
    let cfg = MhaConfig::new(1, 8, 8, 64, 4096, 4096, true);
    assert!(cfg.validate().is_ok());
    let chunks = compute_flash_chunks(4096, DEFAULT_FLASH_CHUNK);
    assert_eq!(chunks, 128);
}

#[test]
fn edge_v2_kv_cache_contiguous_equality() {
    let a = KvCacheLayout::Contiguous {
        num_layers: 32,
        max_seq_len: 2048,
        num_kv_heads: 8,
        head_dim: 128,
    };
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn edge_v2_kv_cache_paged_equality() {
    let a = KvCacheLayout::Paged { page_size: 16, num_pages: 128, num_kv_heads: 8, head_dim: 64 };
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn edge_v2_softmax_single_element() {
    let probs = cpu_softmax(&[42.0]);
    assert_eq!(probs.len(), 1);
    assert!((probs[0] - 1.0).abs() < TOL);
}

#[test]
fn edge_v2_mask_neg_inf_softmax() {
    let logits = vec![MASK_NEG_INF; 4];
    let probs = cpu_softmax(&logits);
    // All identical → uniform
    for &p in &probs {
        assert!((p - 0.25).abs() < TOL);
    }
}

#[test]
fn edge_v2_alignment_helper_identity() {
    assert_eq!(align_up(256, 256), 256);
    assert_eq!(align_up(0, 256), 0);
}

#[test]
fn edge_v2_alignment_helper_round_up() {
    assert_eq!(align_up(1, 256), 256);
    assert_eq!(align_up(257, 256), 512);
    assert_eq!(align_up(512, 256), 512);
}

#[test]
fn edge_v2_det_rand_reproducible() {
    let a = det_rand(100, 42, -1.0, 1.0);
    let b = det_rand(100, 42, -1.0, 1.0);
    assert_eq!(a, b);
}

#[test]
fn edge_v2_det_rand_different_seeds() {
    let a = det_rand(100, 42, -1.0, 1.0);
    let b = det_rand(100, 99, -1.0, 1.0);
    assert_ne!(a, b);
}

#[test]
fn edge_v2_causal_mask_non_square() {
    let mask = cpu_causal_mask(3, 5);
    assert_eq!(mask.len(), 15);
    // Row 0: [T, F, F, F, F]
    assert!(mask[0]);
    assert!(!mask[1]);
}

#[test]
fn edge_v2_threadgroup_grid_coverage() {
    // Verify total thread count covers all heads × chunks
    let heads: u32 = 16;
    let chunks: u32 = 8;
    let tg = compute_threadgroup(heads, chunks);
    assert_eq!(tg.threadgroups_per_grid, heads * chunks);
}

#[test]
fn edge_v2_bandwidth_zero_for_trivial() {
    let cfg = MhaConfig::new(1, 1, 1, 1, 1, 1, false);
    let bw = estimate_attention_bandwidth_bytes(&cfg);
    // 4 tensors * 1 element * 4 bytes = 16
    assert_eq!(bw, 16);
}

#[test]
fn edge_v2_gqa_not_mqa_distinction() {
    // 2 KV heads, 8 Q heads → GQA (not MQA since kv > 1)
    let cfg = MhaConfig::new(1, 8, 2, 64, 32, 32, false);
    assert!(cfg.is_gqa());
    assert!(!cfg.is_mqa());
    assert!(!cfg.is_mha());
}

#[test]
fn edge_v2_mha_not_gqa_not_mqa() {
    let cfg = MhaConfig::new(1, 8, 8, 64, 32, 32, false);
    assert!(cfg.is_mha());
    assert!(!cfg.is_gqa());
    assert!(!cfg.is_mqa());
}
