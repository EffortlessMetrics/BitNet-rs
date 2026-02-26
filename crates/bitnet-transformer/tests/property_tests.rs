//! Property-based tests for `bitnet-transformer`.
//!
//! Key invariants tested:
//! - `KVCache`: initial seq_len is 0; after N single-token appends, seq_len == N;
//!   after clear, seq_len returns to 0.
//! - `LayerKVCache`: capacity rejection holds for any valid cap/append sizes.
//! - Config validation: hidden % num_heads == 0 is required for model construction.
#![cfg(feature = "cpu")]

use bitnet_common::BitNetConfig;
use bitnet_transformer::{KVCache, LayerKVCache};
use candle_core::{DType, Device, Tensor};
use proptest::prelude::*;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build a `BitNetConfig` with `hidden = heads * head_dim` so it's always valid.
fn valid_config(heads: usize, head_dim: usize, max_seq: usize) -> BitNetConfig {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_layers = 1;
    cfg.model.num_heads = heads;
    cfg.model.num_key_value_heads = heads;
    cfg.model.hidden_size = heads * head_dim;
    cfg.model.max_position_embeddings = max_seq;
    cfg.model.vocab_size = 32;
    cfg
}

fn zeros_kv(batch: usize, heads: usize, seq: usize, dim: usize) -> Tensor {
    Tensor::zeros(&[batch, heads, seq, dim], DType::F32, &Device::Cpu).unwrap()
}

// ── KVCache sequence-length invariants ───────────────────────────────────────

proptest! {
    /// After N single-token appends, `KVCache` reports `max_seq_len` as N × 1.
    /// The KVCache seq_len after N appends equals N (using the first layer as proxy).
    #[test]
    fn kv_cache_seq_len_equals_append_count(
        n_steps in 1_usize..=8,
        heads in prop_oneof![Just(1_usize), Just(2), Just(4)],
        head_dim in prop_oneof![Just(4_usize), Just(8)],
    ) {
        let max_seq = n_steps + 4; // ensure we don't hit the cap
        let cfg = valid_config(heads, head_dim, max_seq);
        let mut kv = KVCache::new(&cfg, 1, &Device::Cpu).unwrap();

        for step in 1..=n_steps {
            let k = zeros_kv(1, heads, 1, head_dim);
            let v = zeros_kv(1, heads, 1, head_dim);
            kv.layer_mut(0).unwrap().append(&k, &v).unwrap();
            prop_assert_eq!(
                kv.layer_mut(0).unwrap().seq_len,
                step,
                "seq_len after step {} should be {}", step, step
            );
        }
    }

    /// After appending K tokens and then calling `clear()`, seq_len returns to 0.
    #[test]
    fn kv_cache_clear_resets_seq_len(
        n_steps in 1_usize..=6,
        heads in prop_oneof![Just(2_usize), Just(4)],
    ) {
        let head_dim = 4_usize;
        let max_seq = n_steps + 2;
        let cfg = valid_config(heads, head_dim, max_seq);
        let mut kv = KVCache::new(&cfg, 1, &Device::Cpu).unwrap();

        for _ in 0..n_steps {
            let k = zeros_kv(1, heads, 1, head_dim);
            let v = zeros_kv(1, heads, 1, head_dim);
            kv.layer_mut(0).unwrap().append(&k, &v).unwrap();
        }
        prop_assert!(kv.layer_mut(0).unwrap().seq_len > 0, "should have non-zero seq_len after appends");

        kv.clear();
        prop_assert_eq!(kv.layer_mut(0).unwrap().seq_len, 0, "seq_len must be 0 after clear");
    }
}

// ── LayerKVCache capacity enforcement ────────────────────────────────────────

proptest! {
    /// A `LayerKVCache` with capacity C rejects an append that would exceed C.
    #[test]
    fn layer_kv_cache_rejects_overflow(
        cap in 2_usize..=8,
        heads in prop_oneof![Just(1_usize), Just(2)],
        head_dim in prop_oneof![Just(4_usize), Just(8)],
    ) {
        // Fill to capacity
        let mut cache = LayerKVCache::new(1, heads, cap, head_dim, &Device::Cpu).unwrap();
        for _ in 0..cap {
            let k = zeros_kv(1, heads, 1, head_dim);
            let v = zeros_kv(1, heads, 1, head_dim);
            cache.append(&k, &v).unwrap();
        }
        prop_assert_eq!(cache.seq_len, cap, "should be at capacity");

        // One more append should fail
        let k = zeros_kv(1, heads, 1, head_dim);
        let v = zeros_kv(1, heads, 1, head_dim);
        prop_assert!(
            cache.append(&k, &v).is_err(),
            "append beyond capacity must fail"
        );
    }

    /// Initial seq_len of a freshly created `LayerKVCache` is always 0.
    #[test]
    fn layer_kv_cache_initial_seq_len_is_always_zero(
        batch in 1_usize..=2,
        heads in 1_usize..=4,
        cap in 1_usize..=16,
        head_dim in prop_oneof![Just(4_usize), Just(8)],
    ) {
        let cache = LayerKVCache::new(batch, heads, cap, head_dim, &Device::Cpu).unwrap();
        prop_assert_eq!(cache.seq_len, 0, "fresh LayerKVCache must have seq_len 0");
    }
}

// ── Shape invariant after N appends ──────────────────────────────────────────

proptest! {
    /// After N single-token appends, `LayerKVCache` key and value tensors
    /// must have shape `[batch, heads, N, head_dim]`.
    #[test]
    fn shape_invariant_after_n_appends(
        n_steps in 1_usize..=6,
        batch in 1_usize..=2,
        heads in prop_oneof![Just(1_usize), Just(2), Just(4)],
        head_dim in prop_oneof![Just(4_usize), Just(8)],
    ) {
        let max_seq = n_steps + 4;
        let mut cache = LayerKVCache::new(batch, heads, max_seq, head_dim, &Device::Cpu).unwrap();

        for _ in 0..n_steps {
            let k = zeros_kv(batch, heads, 1, head_dim);
            let v = zeros_kv(batch, heads, 1, head_dim);
            cache.append(&k, &v).unwrap();
        }

        let expected = [batch, heads, n_steps, head_dim];
        prop_assert_eq!(
            cache.k.dims(),
            expected.as_slice(),
            "k shape after {} appends must be {:?}", n_steps, expected
        );
        prop_assert_eq!(
            cache.v.dims(),
            expected.as_slice(),
            "v shape after {} appends must be {:?}", n_steps, expected
        );
    }
}

// ── Concurrent layer independence ─────────────────────────────────────────────

proptest! {
    /// Two separate `LayerKVCache` instances are fully independent:
    /// appending to one must not affect the other's `seq_len`.
    #[test]
    fn concurrent_layer_independence(
        n_a in 1_usize..=4,
        n_b in 1_usize..=4,
        heads in prop_oneof![Just(1_usize), Just(2)],
        head_dim in prop_oneof![Just(4_usize), Just(8)],
    ) {
        let max_seq = n_a.max(n_b) + 4;
        let mut cache_a = LayerKVCache::new(1, heads, max_seq, head_dim, &Device::Cpu).unwrap();
        let mut cache_b = LayerKVCache::new(1, heads, max_seq, head_dim, &Device::Cpu).unwrap();

        for _ in 0..n_a {
            let k = zeros_kv(1, heads, 1, head_dim);
            let v = zeros_kv(1, heads, 1, head_dim);
            cache_a.append(&k, &v).unwrap();
        }
        prop_assert_eq!(cache_b.seq_len, 0, "appending to cache_a must not affect cache_b");

        for _ in 0..n_b {
            let k = zeros_kv(1, heads, 1, head_dim);
            let v = zeros_kv(1, heads, 1, head_dim);
            cache_b.append(&k, &v).unwrap();
        }

        prop_assert_eq!(cache_a.seq_len, n_a, "cache_a seq_len must equal n_a");
        prop_assert_eq!(cache_b.seq_len, n_b, "cache_b seq_len must equal n_b");
    }
}

// ── KVCache layer count ────────────────────────────────────────────────────────

proptest! {
    /// `KVCache::new` creates exactly `config.model.num_layers` layers.
    #[test]
    fn kv_cache_layer_count(
        n_layers in 1_usize..=6,
        heads in prop_oneof![Just(2_usize), Just(4)],
        head_dim in prop_oneof![Just(4_usize), Just(8)],
    ) {
        let cfg = valid_config(heads, head_dim, 16);
        let mut cfg = cfg;
        cfg.model.num_layers = n_layers;
        let kv = KVCache::new(&cfg, 1, &Device::Cpu).unwrap();
        prop_assert_eq!(
            kv.layers.len(),
            n_layers,
            "KVCache must have exactly {} layers", n_layers
        );
    }
}

// ── Config validation: head divisibility ──────────────────────────────────────

proptest! {
    /// `KVCache::new` returns an error when `hidden_size` is not divisible by `num_heads`.
    #[test]
    fn config_validation_head_divisibility(
        heads in prop_oneof![Just(3_usize), Just(5), Just(7)],
    ) {
        // heads * 4 + 1 is never divisible by heads (since heads > 1)
        let hidden_size = heads * 4 + 1;
        let mut cfg = BitNetConfig::default();
        cfg.model.num_layers = 1;
        cfg.model.num_heads = heads;
        cfg.model.num_key_value_heads = heads;
        cfg.model.hidden_size = hidden_size;
        cfg.model.max_position_embeddings = 16;
        cfg.model.vocab_size = 32;

        let result = KVCache::new(&cfg, 1, &Device::Cpu);
        prop_assert!(
            result.is_err(),
            "KVCache must reject hidden_size={} not divisible by num_heads={}", hidden_size, heads
        );
    }
}

// ── seq_len monotonically increases ───────────────────────────────────────────

proptest! {
    /// After each multi-token append to `LayerKVCache`, `seq_len` increases
    /// by exactly the number of tokens appended.
    #[test]
    fn seq_len_monotonically_increases(
        n_steps in 1_usize..=5,
        tokens_per_step in prop_oneof![Just(1_usize), Just(2), Just(3)],
        heads in prop_oneof![Just(1_usize), Just(2)],
        head_dim in prop_oneof![Just(4_usize), Just(8)],
    ) {
        let max_seq = n_steps * tokens_per_step + 4;
        let mut cache = LayerKVCache::new(1, heads, max_seq, head_dim, &Device::Cpu).unwrap();

        let mut expected_seq_len = 0_usize;
        for step in 1..=n_steps {
            let k = zeros_kv(1, heads, tokens_per_step, head_dim);
            let v = zeros_kv(1, heads, tokens_per_step, head_dim);
            cache.append(&k, &v).unwrap();
            expected_seq_len += tokens_per_step;
            prop_assert_eq!(
                cache.seq_len,
                expected_seq_len,
                "seq_len after step {} must be {}", step, expected_seq_len
            );
        }
    }
}
