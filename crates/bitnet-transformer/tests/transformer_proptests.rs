//! Additional property-based tests for `bitnet-transformer`.
//!
//! Complements `property_tests.rs` with invariants across:
//!
//! - `RotaryEmbedding`: shape preservation, output finiteness, determinism,
//!   and position-sensitivity.
//! - `LayerKVCache`: clear-then-reappend, batch-dim preservation.
//! - `KVCache`: GQA divisibility rejection, per-layer `max_seq_len` consistency.
#![cfg(feature = "cpu")]

use bitnet_common::config::{BitNetConfig, ModelConfig};
use bitnet_transformer::{KVCache, LayerKVCache, RotaryEmbedding};
use candle_core::{DType, Device, Tensor};
use proptest::prelude::*;

// ── helpers ───────────────────────────────────────────────────────────────────

fn ones_4d(batch: usize, heads: usize, seq: usize, dim: usize) -> Tensor {
    Tensor::ones(&[batch, heads, seq, dim], DType::F32, &Device::Cpu).unwrap()
}

fn zeros_kv(batch: usize, heads: usize, seq: usize, dim: usize) -> Tensor {
    Tensor::zeros(&[batch, heads, seq, dim], DType::F32, &Device::Cpu).unwrap()
}

/// Make a `BitNetConfig` where `hidden == num_heads * head_dim` (always valid).
fn valid_config(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
) -> BitNetConfig {
    BitNetConfig {
        model: ModelConfig {
            hidden_size: num_heads * head_dim,
            num_heads,
            num_key_value_heads: num_kv_heads,
            num_layers: 2,
            max_position_embeddings: max_seq,
            vocab_size: 32,
            intermediate_size: num_heads * head_dim * 4,
            rms_norm_eps: Some(1e-5),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Strategy producing `(num_heads, num_kv_heads)` pairs where
/// `num_heads % num_kv_heads != 0` and `num_kv_heads < num_heads`.
fn non_divisible_kv_pairs() -> impl Strategy<Value = (usize, usize)> {
    prop_oneof![
        Just((4_usize, 3_usize)),
        Just((6_usize, 4_usize)),
        Just((6_usize, 5_usize)),
        Just((8_usize, 3_usize)),
        Just((8_usize, 5_usize)),
        Just((8_usize, 6_usize)),
        Just((9_usize, 4_usize)),
        Just((10_usize, 3_usize)),
    ]
}

// ── RotaryEmbedding: shape preservation ──────────────────────────────────────

proptest! {
    /// `RotaryEmbedding::apply` must return a tensor with the exact same shape
    /// as the input for any valid `[batch, heads, seq, head_dim]`.
    #[test]
    fn rope_shape_preserved_for_varying_dims(
        batch    in prop_oneof![Just(1_usize), Just(2)],
        heads    in prop_oneof![Just(1_usize), Just(2), Just(4)],
        seq      in 1_usize..=4,
        head_dim in prop_oneof![Just(4_usize), Just(8), Just(16)],
    ) {
        let max_seq = seq + 4;
        let rope = RotaryEmbedding::new(head_dim, max_seq, None, &Device::Cpu).unwrap();
        let x = ones_4d(batch, heads, seq, head_dim);
        let out = rope.apply(&x, 0).unwrap();
        prop_assert_eq!(
            out.dims(),
            x.dims(),
            "RoPE must not change tensor shape: got {:?}", out.dims()
        );
    }
}

// ── RotaryEmbedding: output finiteness ───────────────────────────────────────

proptest! {
    /// `RotaryEmbedding::apply` must produce only finite values for any valid
    /// combination of position, sequence length, and head dimension.
    #[test]
    fn rope_output_is_always_finite(
        position in 0_usize..=4,
        seq      in 1_usize..=3,
        head_dim in prop_oneof![Just(4_usize), Just(8), Just(16)],
        theta    in prop_oneof![
            Just(None::<f32>),
            Just(Some(10_000.0_f32)),
            Just(Some(500_000.0_f32)),
        ],
    ) {
        let max_seq = position + seq + 4;
        let rope = RotaryEmbedding::new(head_dim, max_seq, theta, &Device::Cpu).unwrap();
        let x = ones_4d(1, 2, seq, head_dim);
        let out = rope.apply(&x, position).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        prop_assert!(
            vals.iter().all(|v| v.is_finite()),
            "RoPE output must be finite: position={} seq={} head_dim={}", position, seq, head_dim
        );
    }
}

// ── RotaryEmbedding: determinism ─────────────────────────────────────────────

proptest! {
    /// Applying `RotaryEmbedding::apply` twice with identical inputs must return
    /// byte-identical results (no hidden state or randomness).
    #[test]
    fn rope_apply_is_deterministic(
        position in 0_usize..=5,
        head_dim in prop_oneof![Just(4_usize), Just(8)],
        heads    in prop_oneof![Just(1_usize), Just(2)],
    ) {
        let max_seq = position + 3;
        let rope = RotaryEmbedding::new(head_dim, max_seq, None, &Device::Cpu).unwrap();
        let x = ones_4d(1, heads, 1, head_dim);
        let a: Vec<f32> = rope
            .apply(&x, position).unwrap()
            .flatten_all().unwrap()
            .to_vec1().unwrap();
        let b: Vec<f32> = rope
            .apply(&x, position).unwrap()
            .flatten_all().unwrap()
            .to_vec1().unwrap();
        prop_assert_eq!(a, b, "RoPE must be deterministic for position={}", position);
    }
}

// ── RotaryEmbedding: position sensitivity ────────────────────────────────────

proptest! {
    /// Two different positions must produce at least one differing output element
    /// (encoding is position-sensitive).
    #[test]
    fn rope_different_positions_produce_different_outputs(
        pos_a    in 0_usize..=4,
        offset   in 1_usize..=4,
        head_dim in prop_oneof![Just(8_usize), Just(16)],
        heads    in prop_oneof![Just(2_usize), Just(4)],
    ) {
        let pos_b = pos_a + offset;
        let max_seq = pos_b + 4;
        let rope = RotaryEmbedding::new(head_dim, max_seq, None, &Device::Cpu).unwrap();
        let x = ones_4d(1, heads, 1, head_dim);
        let out_a: Vec<f32> = rope
            .apply(&x, pos_a).unwrap()
            .flatten_all().unwrap()
            .to_vec1().unwrap();
        let out_b: Vec<f32> = rope
            .apply(&x, pos_b).unwrap()
            .flatten_all().unwrap()
            .to_vec1().unwrap();
        let any_diff = out_a.iter().zip(out_b.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
        prop_assert!(
            any_diff,
            "positions {} and {} must produce different RoPE encodings", pos_a, pos_b
        );
    }
}

// ── LayerKVCache: clear then re-append ───────────────────────────────────────

proptest! {
    /// After filling a `LayerKVCache` to capacity and clearing it, subsequent
    /// appends must succeed and report only the new tokens in `seq_len`.
    #[test]
    fn layer_kv_cache_clear_and_reappend_succeeds(
        cap      in 2_usize..=6,
        n_refill in 1_usize..=4,
        heads    in prop_oneof![Just(1_usize), Just(2)],
        head_dim in prop_oneof![Just(4_usize), Just(8)],
    ) {
        let n_refill = n_refill.min(cap);
        let mut cache = LayerKVCache::new(1, heads, cap, head_dim, &Device::Cpu).unwrap();

        // Fill to capacity
        for _ in 0..cap {
            let k = zeros_kv(1, heads, 1, head_dim);
            let v = zeros_kv(1, heads, 1, head_dim);
            cache.append(&k, &v).unwrap();
        }
        prop_assert_eq!(cache.seq_len, cap, "pre-clear seq_len must equal cap");

        cache.clear();
        prop_assert_eq!(cache.seq_len, 0, "seq_len must be 0 after clear");

        // Re-append after clear
        for step in 1..=n_refill {
            let k = zeros_kv(1, heads, 1, head_dim);
            let v = zeros_kv(1, heads, 1, head_dim);
            let result = cache.append(&k, &v);
            prop_assert!(result.is_ok(), "append after clear must succeed at step {}", step);
        }
        prop_assert_eq!(
            cache.seq_len,
            n_refill,
            "seq_len after re-append must be {}, not carry over from cleared state", n_refill
        );
    }
}

// ── LayerKVCache: batch dimension preserved ──────────────────────────────────

proptest! {
    /// After N appends the batch dimension (dim 0) of the cached k and v tensors
    /// must equal the batch size used at construction.
    #[test]
    fn layer_kv_cache_batch_dim_preserved_after_appends(
        batch    in 1_usize..=3,
        n_steps  in 1_usize..=4,
        heads    in prop_oneof![Just(1_usize), Just(2)],
        head_dim in prop_oneof![Just(4_usize), Just(8)],
    ) {
        let max_seq = n_steps + 4;
        let mut cache = LayerKVCache::new(batch, heads, max_seq, head_dim, &Device::Cpu).unwrap();

        for _ in 0..n_steps {
            let k = zeros_kv(batch, heads, 1, head_dim);
            let v = zeros_kv(batch, heads, 1, head_dim);
            cache.append(&k, &v).unwrap();
        }

        prop_assert_eq!(
            cache.k.dims()[0],
            batch,
            "k batch dim must equal {} after {} appends", batch, n_steps
        );
        prop_assert_eq!(
            cache.v.dims()[0],
            batch,
            "v batch dim must equal {} after {} appends", batch, n_steps
        );
    }
}

// ── KVCache: GQA divisibility rejection ──────────────────────────────────────

proptest! {
    /// `KVCache::new` must return an error when `num_heads` is not divisible by
    /// `num_key_value_heads` (fundamental GQA constraint).
    #[test]
    fn kv_cache_rejects_non_divisible_kv_heads(
        (num_heads, num_kv_heads) in non_divisible_kv_pairs(),
        head_dim in prop_oneof![Just(4_usize), Just(8)],
    ) {
        // hidden_size is divisible by num_heads so the first guard passes;
        // the GQA divisibility check must then reject.
        let cfg = valid_config(num_heads, num_kv_heads, head_dim, 16);
        let result = KVCache::new(&cfg, 1, &Device::Cpu);
        prop_assert!(
            result.is_err(),
            "KVCache must reject num_heads={} / num_kv_heads={}", num_heads, num_kv_heads
        );
    }
}

// ── KVCache: per-layer max_seq_len matches config ────────────────────────────

proptest! {
    /// Every layer in a freshly constructed `KVCache` must have
    /// `max_seq_len == config.model.max_position_embeddings`.
    #[test]
    fn kv_cache_all_layers_max_seq_len_matches_config(
        n_layers in 1_usize..=5,
        max_seq  in prop_oneof![Just(16_usize), Just(32), Just(64)],
        heads    in prop_oneof![Just(2_usize), Just(4)],
        head_dim in prop_oneof![Just(4_usize), Just(8)],
    ) {
        let mut cfg = valid_config(heads, heads, head_dim, max_seq);
        cfg.model.num_layers = n_layers;
        let kv = KVCache::new(&cfg, 1, &Device::Cpu).unwrap();

        for (i, layer) in kv.layers.iter().enumerate() {
            prop_assert_eq!(
                layer.max_seq_len,
                max_seq,
                "layer {} max_seq_len must equal config.max_position_embeddings={}", i, max_seq
            );
        }
    }
}
