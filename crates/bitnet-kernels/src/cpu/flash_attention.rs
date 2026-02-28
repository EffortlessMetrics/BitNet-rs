//! CPU Flash Attention — tiled, O(N) memory attention for CPU inference.
//!
//! Implements the FlashAttention algorithm (Dao et al., 2022) for CPU:
//!
//! - **Block-wise softmax** with the online softmax trick (running max + sum)
//!   avoids materializing the full N×N attention matrix.
//! - **O(N) memory** instead of O(N²) — only block-sized scratch buffers are
//!   allocated, making long-context inference feasible on CPU.
//! - **Causal masking** built into the inner loop (zero-cost when enabled).
//! - **Grouped Query Attention (GQA)** — K/V heads are shared across query
//!   head groups when `num_kv_heads < num_heads`.
//! - **SIMD-friendly layout** — inner loops are written to auto-vectorize
//!   with `#[inline]` and sequential memory access patterns.
//!
//! # Algorithm
//!
//! For each query block `Br` of size `BLOCK_Q`:
//!   1. Load Q tile `[BLOCK_Q, head_dim]`
//!   2. For each KV block `Bc` of size `BLOCK_KV`:
//!      a. Compute S = Q_tile @ K_tile^T / sqrt(head_dim)  — `[BLOCK_Q, BLOCK_KV]`
//!      b. Apply causal mask (if enabled)
//!      c. Online softmax update: track running max and denominator
//!      d. Accumulate O_tile += softmax(S) @ V_tile
//!   3. Rescale O_tile by final denominator
//!
//! # Usage
//!
//! ```rust,ignore
//! use bitnet_kernels::cpu::flash_attention::{FlashAttentionConfig, flash_attention};
//!
//! let config = FlashAttentionConfig {
//!     num_heads: 32,
//!     num_kv_heads: 8,   // GQA: 4 query heads per KV head
//!     head_dim: 64,
//!     causal: true,
//!     block_q: 32,
//!     block_kv: 32,
//! };
//!
//! let mut output = vec![0.0f32; batch * seq_q * num_heads * head_dim];
//! flash_attention(&config, &q, &k, &v, &mut output, batch, seq_q, seq_kv)?;
//! ```

use bitnet_common::{KernelError, Result};

/// Configuration for CPU flash attention.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of key/value heads (GQA when < num_heads).
    pub num_kv_heads: usize,
    /// Per-head embedding dimension.
    pub head_dim: usize,
    /// Enable causal (autoregressive) masking.
    pub causal: bool,
    /// Query block size for tiling.
    pub block_q: usize,
    /// Key/Value block size for tiling.
    pub block_kv: usize,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 64,
            causal: true,
            block_q: 32,
            block_kv: 32,
        }
    }
}

impl FlashAttentionConfig {
    /// Validate configuration parameters.
    fn validate(&self) -> Result<()> {
        if self.num_heads == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "num_heads must be > 0".into() }.into()
            );
        }
        if self.num_kv_heads == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_kv_heads must be > 0".into(),
            }
            .into());
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "num_heads ({}) must be divisible by num_kv_heads ({})",
                    self.num_heads, self.num_kv_heads
                ),
            }
            .into());
        }
        if self.head_dim == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "head_dim must be > 0".into() }.into()
            );
        }
        if self.block_q == 0 || self.block_kv == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "block sizes must be > 0".into() }.into()
            );
        }
        Ok(())
    }

    /// Number of query heads per KV head group.
    #[inline]
    fn num_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

/// Run flash attention on CPU.
///
/// # Tensor layouts (row-major)
///
/// - `q`: `[batch, seq_q,  num_heads,    head_dim]`
/// - `k`: `[batch, seq_kv, num_kv_heads, head_dim]`
/// - `v`: `[batch, seq_kv, num_kv_heads, head_dim]`
/// - `output`: `[batch, seq_q, num_heads, head_dim]` (pre-allocated)
///
/// All tensors are contiguous f32 slices in row-major order.
pub fn flash_attention(
    config: &FlashAttentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    batch_size: usize,
    seq_q: usize,
    seq_kv: usize,
) -> Result<()> {
    config.validate()?;

    let expected_q = batch_size * seq_q * config.num_heads * config.head_dim;
    let expected_kv = batch_size * seq_kv * config.num_kv_heads * config.head_dim;
    let expected_out = batch_size * seq_q * config.num_heads * config.head_dim;

    if q.len() != expected_q {
        return Err(KernelError::InvalidArguments {
            reason: format!("Q length mismatch: expected {expected_q}, got {}", q.len()),
        }
        .into());
    }
    if k.len() != expected_kv {
        return Err(KernelError::InvalidArguments {
            reason: format!("K length mismatch: expected {expected_kv}, got {}", k.len()),
        }
        .into());
    }
    if v.len() != expected_kv {
        return Err(KernelError::InvalidArguments {
            reason: format!("V length mismatch: expected {expected_kv}, got {}", v.len()),
        }
        .into());
    }
    if output.len() != expected_out {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "Output length mismatch: expected {expected_out}, got {}",
                output.len()
            ),
        }
        .into());
    }

    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let num_groups = config.num_groups();

    for b in 0..batch_size {
        for h in 0..config.num_heads {
            let kv_head = h / num_groups;
            let mut args = SingleHeadArgs {
                config,
                q,
                k,
                v,
                output: &mut *output,
                scale,
                batch: b,
                head: h,
                kv_head,
                seq_q,
                seq_kv,
            };
            flash_attention_single_head(&mut args);
        }
    }

    Ok(())
}

/// Bundled arguments for a single-head flash attention call.
struct SingleHeadArgs<'a> {
    config: &'a FlashAttentionConfig,
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    output: &'a mut [f32],
    scale: f32,
    batch: usize,
    head: usize,
    kv_head: usize,
    seq_q: usize,
    seq_kv: usize,
}

/// Flash attention for a single (batch, head) pair.
///
/// Uses the online softmax trick: for each query block, we stream through KV
/// blocks accumulating `O`, `row_max`, and `row_sum` incrementally, then
/// rescale once at the end.
#[inline(never)] // Prevent inlining for profiling visibility
fn flash_attention_single_head(args: &mut SingleHeadArgs<'_>) {
    let config = args.config;
    let q = args.q;
    let k = args.k;
    let v = args.v;
    let scale = args.scale;
    let batch = args.batch;
    let head = args.head;
    let kv_head = args.kv_head;
    let seq_q = args.seq_q;
    let seq_kv = args.seq_kv;

    let head_dim = config.head_dim;
    let block_q = config.block_q.min(seq_q);
    let block_kv = config.block_kv.min(seq_kv);

    // Strides for indexing into the flat arrays.
    // q layout: [batch, seq_q, num_heads, head_dim]
    let q_batch_stride = seq_q * config.num_heads * head_dim;
    let q_seq_stride = config.num_heads * head_dim;
    let q_head_stride = head_dim;

    // k/v layout: [batch, seq_kv, num_kv_heads, head_dim]
    let kv_batch_stride = seq_kv * config.num_kv_heads * head_dim;
    let kv_seq_stride = config.num_kv_heads * head_dim;
    let kv_head_stride = head_dim;

    // output layout: [batch, seq_q, num_heads, head_dim]
    let o_batch_stride = seq_q * config.num_heads * head_dim;
    let o_seq_stride = config.num_heads * head_dim;
    let o_head_stride = head_dim;

    // Scratch buffers — O(block_q * (block_kv + head_dim)) memory.
    let mut scores = vec![0.0f32; block_q * block_kv];
    let mut row_max = vec![f32::NEG_INFINITY; block_q];
    let mut row_sum = vec![0.0f32; block_q];
    let mut acc = vec![0.0f32; block_q * head_dim];

    // Process query rows in blocks of block_q.
    let mut qi = 0;
    while qi < seq_q {
        let q_end = (qi + block_q).min(seq_q);
        let q_block_len = q_end - qi;

        // Reset accumulators for this query block.
        row_max[..q_block_len].fill(f32::NEG_INFINITY);
        row_sum[..q_block_len].fill(0.0);
        acc[..q_block_len * head_dim].fill(0.0);

        // Stream through KV blocks.
        let kv_limit = if config.causal { seq_kv.min(q_end) } else { seq_kv };
        let mut kj = 0;
        while kj < kv_limit {
            let k_end = (kj + block_kv).min(kv_limit);
            let k_block_len = k_end - kj;

            // --- Compute S = Q_tile @ K_tile^T * scale ---
            compute_qk_scores(
                q,
                k,
                &mut scores,
                scale,
                batch,
                head,
                kv_head,
                qi,
                kj,
                q_block_len,
                k_block_len,
                head_dim,
                q_batch_stride,
                q_seq_stride,
                q_head_stride,
                kv_batch_stride,
                kv_seq_stride,
                kv_head_stride,
                block_kv,
            );

            // --- Apply causal mask ---
            if config.causal {
                apply_causal_mask(&mut scores, qi, kj, q_block_len, k_block_len, block_kv);
            }

            // --- Online softmax update + accumulate O ---
            online_softmax_accumulate(
                &scores,
                v,
                &mut acc,
                &mut row_max,
                &mut row_sum,
                batch,
                kv_head,
                kj,
                q_block_len,
                k_block_len,
                head_dim,
                kv_batch_stride,
                kv_seq_stride,
                kv_head_stride,
                block_kv,
            );

            kj = k_end;
        }

        // --- Final rescale: O[i, :] /= row_sum[i] ---
        for i in 0..q_block_len {
            let denom = row_sum[i];
            if denom > 0.0 {
                let inv_denom = 1.0 / denom;
                let acc_row = &mut acc[i * head_dim..(i + 1) * head_dim];
                for val in acc_row.iter_mut() {
                    *val *= inv_denom;
                }
            }
        }

        // --- Write output ---
        for i in 0..q_block_len {
            let o_offset = batch * o_batch_stride + (qi + i) * o_seq_stride + head * o_head_stride;
            let acc_row = &acc[i * head_dim..(i + 1) * head_dim];
            args.output[o_offset..o_offset + head_dim].copy_from_slice(acc_row);
        }

        qi = q_end;
    }
}

/// Compute Q_tile @ K_tile^T * scale into `scores`.
///
/// Written as a straightforward dot-product loop that the compiler can
/// auto-vectorize with AVX2/NEON.
#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_qk_scores(
    q: &[f32],
    k: &[f32],
    scores: &mut [f32],
    scale: f32,
    batch: usize,
    head: usize,
    kv_head: usize,
    qi: usize,
    kj: usize,
    q_block_len: usize,
    k_block_len: usize,
    head_dim: usize,
    q_batch_stride: usize,
    q_seq_stride: usize,
    q_head_stride: usize,
    kv_batch_stride: usize,
    kv_seq_stride: usize,
    kv_head_stride: usize,
    score_stride: usize,
) {
    for i in 0..q_block_len {
        let q_offset = batch * q_batch_stride + (qi + i) * q_seq_stride + head * q_head_stride;
        let q_row = &q[q_offset..q_offset + head_dim];

        for j in 0..k_block_len {
            let k_offset =
                batch * kv_batch_stride + (kj + j) * kv_seq_stride + kv_head * kv_head_stride;
            let k_row = &k[k_offset..k_offset + head_dim];

            // Dot product — auto-vectorizable.
            let dot = dot_product(q_row, k_row);
            scores[i * score_stride + j] = dot * scale;
        }
    }
}

/// Vectorization-friendly dot product.
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for (av, bv) in a.iter().zip(b.iter()) {
        sum += av * bv;
    }
    sum
}

/// Apply causal mask: set scores[i][j] = -inf where kj + j > qi + i.
#[inline]
fn apply_causal_mask(
    scores: &mut [f32],
    qi: usize,
    kj: usize,
    q_block_len: usize,
    k_block_len: usize,
    score_stride: usize,
) {
    for i in 0..q_block_len {
        for j in 0..k_block_len {
            if kj + j > qi + i {
                scores[i * score_stride + j] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Online softmax update and value accumulation.
///
/// For each query row `i`:
///   1. Find new block max `m_new = max(row_max[i], max(scores[i, :]))`
///   2. Rescale existing accumulator: `acc[i] *= exp(row_max[i] - m_new)`
///   3. Compute `exp(scores[i, j] - m_new)` and accumulate into `row_sum` and `acc`
///   4. Update `row_max[i] = m_new`
#[inline]
#[allow(clippy::too_many_arguments)]
fn online_softmax_accumulate(
    scores: &[f32],
    v: &[f32],
    acc: &mut [f32],
    row_max: &mut [f32],
    row_sum: &mut [f32],
    batch: usize,
    kv_head: usize,
    kj: usize,
    q_block_len: usize,
    k_block_len: usize,
    head_dim: usize,
    kv_batch_stride: usize,
    kv_seq_stride: usize,
    kv_head_stride: usize,
    block_kv: usize,
) {
    for i in 0..q_block_len {
        let score_row = &scores[i * block_kv..i * block_kv + k_block_len];

        // 1. Find block max.
        let mut block_max = f32::NEG_INFINITY;
        for &s in score_row {
            if s > block_max {
                block_max = s;
            }
        }

        // Skip fully-masked rows (all -inf).
        if block_max == f32::NEG_INFINITY {
            continue;
        }

        // 2. Compute new global max and rescale factor.
        let old_max = row_max[i];
        let new_max = if old_max > block_max { old_max } else { block_max };

        // Rescale existing accumulator and sum.
        if old_max != f32::NEG_INFINITY {
            let rescale = (old_max - new_max).exp();
            row_sum[i] *= rescale;
            let acc_row = &mut acc[i * head_dim..(i + 1) * head_dim];
            for val in acc_row.iter_mut() {
                *val *= rescale;
            }
        }

        // 3. Accumulate new softmax weights × V.
        let acc_row = &mut acc[i * head_dim..(i + 1) * head_dim];
        for (j, &s) in score_row.iter().enumerate().take(k_block_len) {
            if s == f32::NEG_INFINITY {
                continue;
            }
            let w = (s - new_max).exp();
            row_sum[i] += w;

            let v_offset =
                batch * kv_batch_stride + (kj + j) * kv_seq_stride + kv_head * kv_head_stride;
            let v_row = &v[v_offset..v_offset + head_dim];
            for (acc_val, &v_val) in acc_row.iter_mut().zip(v_row.iter()) {
                *acc_val += w * v_val;
            }
        }

        // 4. Update running max.
        row_max[i] = new_max;
    }
}

/// Naive (standard) attention for reference/testing: materializes the full N×N
/// score matrix. **O(N²) memory** — use only for correctness validation.
///
/// Tensor layouts identical to [`flash_attention`].
pub fn naive_attention(
    config: &FlashAttentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    batch_size: usize,
    seq_q: usize,
    seq_kv: usize,
) -> Result<()> {
    config.validate()?;

    let head_dim = config.head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let num_groups = config.num_groups();

    for b in 0..batch_size {
        for h in 0..config.num_heads {
            let kv_h = h / num_groups;

            // Compute full score matrix [seq_q, seq_kv].
            let mut scores = vec![0.0f32; seq_q * seq_kv];
            for i in 0..seq_q {
                let q_off = b * seq_q * config.num_heads * head_dim
                    + i * config.num_heads * head_dim
                    + h * head_dim;
                for j in 0..seq_kv {
                    let k_off = b * seq_kv * config.num_kv_heads * head_dim
                        + j * config.num_kv_heads * head_dim
                        + kv_h * head_dim;
                    let dot = dot_product(&q[q_off..q_off + head_dim], &k[k_off..k_off + head_dim]);
                    scores[i * seq_kv + j] = dot * scale;
                }
            }

            // Apply causal mask.
            if config.causal {
                for i in 0..seq_q {
                    for j in 0..seq_kv {
                        if j > i {
                            scores[i * seq_kv + j] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            // Row-wise softmax.
            for i in 0..seq_q {
                let row = &mut scores[i * seq_kv..(i + 1) * seq_kv];
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                if max_val == f32::NEG_INFINITY {
                    row.fill(0.0);
                    continue;
                }

                let mut sum = 0.0f32;
                for s in row.iter_mut() {
                    *s = (*s - max_val).exp();
                    sum += *s;
                }
                if sum > 0.0 {
                    let inv = 1.0 / sum;
                    for s in row.iter_mut() {
                        *s *= inv;
                    }
                }
            }

            // Weighted sum: O = Softmax(S) @ V.
            for i in 0..seq_q {
                let o_off = b * seq_q * config.num_heads * head_dim
                    + i * config.num_heads * head_dim
                    + h * head_dim;
                let o_row = &mut output[o_off..o_off + head_dim];
                o_row.fill(0.0);
                for j in 0..seq_kv {
                    let w = scores[i * seq_kv + j];
                    if w == 0.0 {
                        continue;
                    }
                    let v_off = b * seq_kv * config.num_kv_heads * head_dim
                        + j * config.num_kv_heads * head_dim
                        + kv_h * head_dim;
                    let v_row = &v[v_off..v_off + head_dim];
                    for (o_val, &v_val) in o_row.iter_mut().zip(v_row.iter()) {
                        *o_val += w * v_val;
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fill a buffer with deterministic pseudo-random values for testing.
    fn fill_deterministic(buf: &mut [f32], seed: u64) {
        let mut state = seed;
        for val in buf.iter_mut() {
            // Simple xorshift64 for determinism.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *val = ((state as i64 % 1000) as f32) / 1000.0;
        }
    }

    /// Compare two slices element-wise with an absolute tolerance.
    fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (av - bv).abs();
            assert!(
                diff <= tol,
                "{msg}: element [{i}] differs by {diff:.6} (flash={av:.6}, naive={bv:.6})"
            );
        }
    }

    #[test]
    fn test_config_validation() {
        let bad = FlashAttentionConfig { num_heads: 0, ..Default::default() };
        assert!(bad.validate().is_err());

        let bad = FlashAttentionConfig { num_kv_heads: 0, ..Default::default() };
        assert!(bad.validate().is_err());

        let bad = FlashAttentionConfig { num_heads: 7, num_kv_heads: 3, ..Default::default() };
        assert!(bad.validate().is_err());

        let bad = FlashAttentionConfig { head_dim: 0, ..Default::default() };
        assert!(bad.validate().is_err());

        let bad = FlashAttentionConfig { block_q: 0, ..Default::default() };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_input_length_validation() {
        let cfg = FlashAttentionConfig {
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            causal: false,
            block_q: 2,
            block_kv: 2,
        };
        let mut out = vec![0.0; 2 * 4];
        let q = vec![0.0; 2 * 4];
        let k = vec![0.0; 2 * 4];
        let v = vec![0.0; 2 * 4];
        // batch=1, seq_q=1, seq_kv=1 → expected q len = 1*1*2*4 = 8
        assert!(flash_attention(&cfg, &q, &k, &v, &mut out, 1, 1, 1).is_ok());
    }

    #[test]
    fn test_flash_vs_naive_no_causal() {
        let cfg = FlashAttentionConfig {
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            causal: false,
            block_q: 4,
            block_kv: 4,
        };
        let (batch, seq_q, seq_kv) = (1, 6, 6);
        let q_len = batch * seq_q * cfg.num_heads * cfg.head_dim;
        let kv_len = batch * seq_kv * cfg.num_kv_heads * cfg.head_dim;
        let o_len = batch * seq_q * cfg.num_heads * cfg.head_dim;

        let mut q = vec![0.0f32; q_len];
        let mut k = vec![0.0f32; kv_len];
        let mut v = vec![0.0f32; kv_len];
        fill_deterministic(&mut q, 42);
        fill_deterministic(&mut k, 123);
        fill_deterministic(&mut v, 456);

        let mut out_flash = vec![0.0f32; o_len];
        let mut out_naive = vec![0.0f32; o_len];

        flash_attention(&cfg, &q, &k, &v, &mut out_flash, batch, seq_q, seq_kv).unwrap();
        naive_attention(&cfg, &q, &k, &v, &mut out_naive, batch, seq_q, seq_kv).unwrap();

        assert_close(&out_flash, &out_naive, 1e-4, "no-causal flash vs naive");
    }

    #[test]
    fn test_flash_vs_naive_causal() {
        let cfg = FlashAttentionConfig {
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            causal: true,
            block_q: 4,
            block_kv: 4,
        };
        let (batch, seq_q, seq_kv) = (1, 6, 6);
        let q_len = batch * seq_q * cfg.num_heads * cfg.head_dim;
        let kv_len = batch * seq_kv * cfg.num_kv_heads * cfg.head_dim;
        let o_len = batch * seq_q * cfg.num_heads * cfg.head_dim;

        let mut q = vec![0.0f32; q_len];
        let mut k = vec![0.0f32; kv_len];
        let mut v = vec![0.0f32; kv_len];
        fill_deterministic(&mut q, 42);
        fill_deterministic(&mut k, 123);
        fill_deterministic(&mut v, 456);

        let mut out_flash = vec![0.0f32; o_len];
        let mut out_naive = vec![0.0f32; o_len];

        flash_attention(&cfg, &q, &k, &v, &mut out_flash, batch, seq_q, seq_kv).unwrap();
        naive_attention(&cfg, &q, &k, &v, &mut out_naive, batch, seq_q, seq_kv).unwrap();

        assert_close(&out_flash, &out_naive, 1e-4, "causal flash vs naive");
    }

    #[test]
    fn test_flash_vs_naive_gqa() {
        // 4 query heads, 2 KV heads → 2 groups
        let cfg = FlashAttentionConfig {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            causal: true,
            block_q: 3,
            block_kv: 3,
        };
        let (batch, seq_q, seq_kv) = (1, 5, 5);
        let q_len = batch * seq_q * cfg.num_heads * cfg.head_dim;
        let kv_len = batch * seq_kv * cfg.num_kv_heads * cfg.head_dim;
        let o_len = batch * seq_q * cfg.num_heads * cfg.head_dim;

        let mut q = vec![0.0f32; q_len];
        let mut k = vec![0.0f32; kv_len];
        let mut v = vec![0.0f32; kv_len];
        fill_deterministic(&mut q, 789);
        fill_deterministic(&mut k, 101);
        fill_deterministic(&mut v, 202);

        let mut out_flash = vec![0.0f32; o_len];
        let mut out_naive = vec![0.0f32; o_len];

        flash_attention(&cfg, &q, &k, &v, &mut out_flash, batch, seq_q, seq_kv).unwrap();
        naive_attention(&cfg, &q, &k, &v, &mut out_naive, batch, seq_q, seq_kv).unwrap();

        assert_close(&out_flash, &out_naive, 1e-4, "GQA flash vs naive");
    }

    #[test]
    fn test_flash_vs_naive_batched() {
        let cfg = FlashAttentionConfig {
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            causal: true,
            block_q: 4,
            block_kv: 4,
        };
        let (batch, seq_q, seq_kv) = (3, 4, 4);
        let q_len = batch * seq_q * cfg.num_heads * cfg.head_dim;
        let kv_len = batch * seq_kv * cfg.num_kv_heads * cfg.head_dim;
        let o_len = batch * seq_q * cfg.num_heads * cfg.head_dim;

        let mut q = vec![0.0f32; q_len];
        let mut k = vec![0.0f32; kv_len];
        let mut v = vec![0.0f32; kv_len];
        fill_deterministic(&mut q, 11);
        fill_deterministic(&mut k, 22);
        fill_deterministic(&mut v, 33);

        let mut out_flash = vec![0.0f32; o_len];
        let mut out_naive = vec![0.0f32; o_len];

        flash_attention(&cfg, &q, &k, &v, &mut out_flash, batch, seq_q, seq_kv).unwrap();
        naive_attention(&cfg, &q, &k, &v, &mut out_naive, batch, seq_q, seq_kv).unwrap();

        assert_close(&out_flash, &out_naive, 1e-4, "batched flash vs naive");
    }

    #[test]
    fn test_single_token_decode() {
        // Simulates autoregressive decode: seq_q=1, seq_kv=N
        let cfg = FlashAttentionConfig {
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            causal: false, // No causal needed when seq_q=1
            block_q: 1,
            block_kv: 4,
        };
        let (batch, seq_q, seq_kv) = (1, 1, 8);
        let q_len = batch * seq_q * cfg.num_heads * cfg.head_dim;
        let kv_len = batch * seq_kv * cfg.num_kv_heads * cfg.head_dim;
        let o_len = batch * seq_q * cfg.num_heads * cfg.head_dim;

        let mut q = vec![0.0f32; q_len];
        let mut k = vec![0.0f32; kv_len];
        let mut v = vec![0.0f32; kv_len];
        fill_deterministic(&mut q, 55);
        fill_deterministic(&mut k, 66);
        fill_deterministic(&mut v, 77);

        let mut out_flash = vec![0.0f32; o_len];
        let mut out_naive = vec![0.0f32; o_len];

        flash_attention(&cfg, &q, &k, &v, &mut out_flash, batch, seq_q, seq_kv).unwrap();
        naive_attention(&cfg, &q, &k, &v, &mut out_naive, batch, seq_q, seq_kv).unwrap();

        assert_close(&out_flash, &out_naive, 1e-4, "decode flash vs naive");
    }

    #[test]
    fn test_block_boundary_alignment() {
        // seq_len not divisible by block size to exercise remainder handling.
        let cfg = FlashAttentionConfig {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
            causal: true,
            block_q: 3,
            block_kv: 3,
        };
        let (batch, seq_q, seq_kv) = (1, 7, 7);
        let q_len = batch * seq_q * cfg.num_heads * cfg.head_dim;
        let kv_len = batch * seq_kv * cfg.num_kv_heads * cfg.head_dim;
        let o_len = batch * seq_q * cfg.num_heads * cfg.head_dim;

        let mut q = vec![0.0f32; q_len];
        let mut k = vec![0.0f32; kv_len];
        let mut v = vec![0.0f32; kv_len];
        fill_deterministic(&mut q, 99);
        fill_deterministic(&mut k, 88);
        fill_deterministic(&mut v, 77);

        let mut out_flash = vec![0.0f32; o_len];
        let mut out_naive = vec![0.0f32; o_len];

        flash_attention(&cfg, &q, &k, &v, &mut out_flash, batch, seq_q, seq_kv).unwrap();
        naive_attention(&cfg, &q, &k, &v, &mut out_naive, batch, seq_q, seq_kv).unwrap();

        assert_close(&out_flash, &out_naive, 1e-4, "block-boundary flash vs naive");
    }

    #[test]
    fn test_uniform_attention_single_head() {
        // All Q and K vectors identical → uniform attention weights.
        let cfg = FlashAttentionConfig {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
            causal: false,
            block_q: 2,
            block_kv: 2,
        };
        let (batch, seq) = (1, 4);
        let q_len = batch * seq * cfg.num_heads * cfg.head_dim;
        let kv_len = batch * seq * cfg.num_kv_heads * cfg.head_dim;

        // All Q and K entries = 1.0 → identical dot products → uniform softmax
        let q = vec![1.0f32; q_len];
        let k = vec![1.0f32; kv_len];
        // V = [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]  (identity)
        let mut v = vec![0.0f32; kv_len];
        for j in 0..seq {
            v[j * cfg.head_dim + j] = 1.0;
        }

        let mut out = vec![0.0f32; q_len];
        flash_attention(&cfg, &q, &k, &v, &mut out, batch, seq, seq).unwrap();

        // With uniform attention over identity V, each output row = [0.25, 0.25, 0.25, 0.25]
        for i in 0..seq {
            for d in 0..cfg.head_dim {
                let val = out[i * cfg.head_dim + d];
                assert!(
                    (val - 0.25).abs() < 1e-5,
                    "row {i}, dim {d}: expected ~0.25, got {val:.6}"
                );
            }
        }
    }

    #[test]
    fn test_large_head_dim() {
        let cfg = FlashAttentionConfig {
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 128,
            causal: true,
            block_q: 8,
            block_kv: 8,
        };
        let (batch, seq) = (1, 16);
        let q_len = batch * seq * cfg.num_heads * cfg.head_dim;
        let kv_len = batch * seq * cfg.num_kv_heads * cfg.head_dim;

        let mut q = vec![0.0f32; q_len];
        let mut k = vec![0.0f32; kv_len];
        let mut v = vec![0.0f32; kv_len];
        fill_deterministic(&mut q, 1);
        fill_deterministic(&mut k, 2);
        fill_deterministic(&mut v, 3);

        let mut out_flash = vec![0.0f32; q_len];
        let mut out_naive = vec![0.0f32; q_len];

        flash_attention(&cfg, &q, &k, &v, &mut out_flash, batch, seq, seq).unwrap();
        naive_attention(&cfg, &q, &k, &v, &mut out_naive, batch, seq, seq).unwrap();

        assert_close(&out_flash, &out_naive, 1e-3, "large head_dim flash vs naive");
    }

    #[test]
    fn test_cross_attention_seq_q_neq_seq_kv() {
        // Cross-attention scenario: different Q and KV sequence lengths.
        let cfg = FlashAttentionConfig {
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            causal: false,
            block_q: 4,
            block_kv: 4,
        };
        let (batch, seq_q, seq_kv) = (1, 3, 10);
        let q_len = batch * seq_q * cfg.num_heads * cfg.head_dim;
        let kv_len = batch * seq_kv * cfg.num_kv_heads * cfg.head_dim;
        let o_len = batch * seq_q * cfg.num_heads * cfg.head_dim;

        let mut q = vec![0.0f32; q_len];
        let mut k = vec![0.0f32; kv_len];
        let mut v = vec![0.0f32; kv_len];
        fill_deterministic(&mut q, 111);
        fill_deterministic(&mut k, 222);
        fill_deterministic(&mut v, 333);

        let mut out_flash = vec![0.0f32; o_len];
        let mut out_naive = vec![0.0f32; o_len];

        flash_attention(&cfg, &q, &k, &v, &mut out_flash, batch, seq_q, seq_kv).unwrap();
        naive_attention(&cfg, &q, &k, &v, &mut out_naive, batch, seq_q, seq_kv).unwrap();

        assert_close(&out_flash, &out_naive, 1e-4, "cross-attention flash vs naive");
    }

    #[test]
    fn test_gqa_extreme_ratio() {
        // 32 query heads, 1 KV head (MQA — multi-query attention)
        let cfg = FlashAttentionConfig {
            num_heads: 32,
            num_kv_heads: 1,
            head_dim: 8,
            causal: true,
            block_q: 4,
            block_kv: 4,
        };
        let (batch, seq) = (1, 4);
        let q_len = batch * seq * cfg.num_heads * cfg.head_dim;
        let kv_len = batch * seq * cfg.num_kv_heads * cfg.head_dim;

        let mut q = vec![0.0f32; q_len];
        let mut k = vec![0.0f32; kv_len];
        let mut v = vec![0.0f32; kv_len];
        fill_deterministic(&mut q, 50);
        fill_deterministic(&mut k, 60);
        fill_deterministic(&mut v, 70);

        let mut out_flash = vec![0.0f32; q_len];
        let mut out_naive = vec![0.0f32; q_len];

        flash_attention(&cfg, &q, &k, &v, &mut out_flash, batch, seq, seq).unwrap();
        naive_attention(&cfg, &q, &k, &v, &mut out_naive, batch, seq, seq).unwrap();

        assert_close(&out_flash, &out_naive, 1e-4, "MQA flash vs naive");
    }
}
