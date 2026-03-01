//! Edge-case tests for attention_compute module.
//!
//! Covers: AttentionConfig, AttentionType, QKVProjection,
//! OutputProjection, CausalMask, AttentionMask, KVCacheEntry,
//! KVCache, ScaledDotProductAttention, AttentionEngine.

use bitnet_gpu_hal::attention_compute::*;

// ── AttentionConfig ─────────────────────────────────────────────

#[test]
fn attention_config_new() {
    let cfg = AttentionConfig::new(8, 64, 2048, true);
    assert_eq!(cfg.num_heads, 8);
    assert_eq!(cfg.head_dim, 64);
    assert_eq!(cfg.max_seq_len, 2048);
    assert!(cfg.causal);
}

#[test]
fn attention_config_model_dim() {
    let cfg = AttentionConfig::new(8, 64, 2048, true);
    assert_eq!(cfg.model_dim(), 512);
}

#[test]
fn attention_config_scale_factor() {
    let cfg = AttentionConfig::new(4, 16, 1024, false);
    // scale_factor = 1/sqrt(head_dim) = 1/4
    assert!((cfg.scale_factor - 0.25).abs() < 1e-5);
}

#[test]
fn attention_config_clone() {
    let cfg = AttentionConfig::new(4, 32, 512, true);
    let cfg2 = cfg.clone();
    assert_eq!(cfg2.num_heads, 4);
    assert_eq!(cfg2.model_dim(), 128);
}

// ── AttentionType ───────────────────────────────────────────────

#[test]
fn attention_type_multi_head() {
    let at = AttentionType::MultiHead;
    assert_eq!(at.num_kv_heads(8), 8);
    // MultiHead: each head is its own group (group size = 1)
    assert_eq!(at.heads_per_group(8), Some(1));
}

#[test]
fn attention_type_multi_query() {
    let at = AttentionType::MultiQuery;
    assert_eq!(at.num_kv_heads(8), 1);
}

#[test]
fn attention_type_grouped_query() {
    let at = AttentionType::GroupedQuery(2);
    assert_eq!(at.num_kv_heads(8), 2);
    assert_eq!(at.heads_per_group(8), Some(4));
}

#[test]
fn attention_type_cross_attention() {
    let at = AttentionType::CrossAttention;
    let _ = format!("{}", at);
}

#[test]
fn attention_type_display() {
    assert_eq!(format!("{}", AttentionType::MultiHead), "MHA");
    assert_eq!(format!("{}", AttentionType::MultiQuery), "MQA");
    assert_eq!(format!("{}", AttentionType::GroupedQuery(4)), "GQA(4)");
    assert_eq!(format!("{}", AttentionType::CrossAttention), "CrossAttn");
}

#[test]
fn attention_type_clone_eq() {
    let a = AttentionType::GroupedQuery(2);
    let b = a.clone();
    assert_eq!(a, b);
}

// ── QKVProjection ───────────────────────────────────────────────

#[test]
fn qkv_projection_no_bias() {
    let proj = QKVProjection::new(64, 64, false);
    let input = vec![0.0; 64];
    let (q, k, v) = proj.forward(&input);
    assert_eq!(q.len(), 64);
    assert_eq!(k.len(), 64);
    assert_eq!(v.len(), 64);
}

#[test]
fn qkv_projection_with_bias() {
    let proj = QKVProjection::new(32, 32, true);
    let input = vec![1.0; 32];
    let (q, k, v) = proj.forward(&input);
    assert_eq!(q.len(), 32);
    assert_eq!(k.len(), 32);
    assert_eq!(v.len(), 32);
}

#[test]
fn qkv_projection_clone() {
    let proj = QKVProjection::new(16, 16, false);
    let proj2 = proj.clone();
    let input = vec![0.0; 16];
    let (q, _, _) = proj2.forward(&input);
    assert_eq!(q.len(), 16);
}

// ── OutputProjection ────────────────────────────────────────────

#[test]
fn output_projection_no_bias() {
    let proj = OutputProjection::new(64, 64, false);
    let input = vec![1.0; 64];
    let out = proj.forward(&input);
    assert_eq!(out.len(), 64);
}

#[test]
fn output_projection_with_bias() {
    let proj = OutputProjection::new(32, 32, true);
    let input = vec![1.0; 32];
    let out = proj.forward(&input);
    assert_eq!(out.len(), 32);
}

// ── CausalMask ──────────────────────────────────────────────────

#[test]
fn causal_mask_basic() {
    let mask = CausalMask::new(4);
    assert_eq!(mask.size(), 4);
    // Lower triangle (including diagonal) should be allowed
    assert!(mask.is_allowed(0, 0));
    assert!(mask.is_allowed(1, 0));
    assert!(mask.is_allowed(1, 1));
    // Upper triangle should be blocked
    assert!(!mask.is_allowed(0, 1));
}

#[test]
fn causal_mask_apply() {
    let mask = CausalMask::new(2);
    // 2x2 attention scores
    let mut scores = vec![1.0, 1.0, 1.0, 1.0];
    mask.apply(&mut scores, 2, f32::NEG_INFINITY);
    assert_eq!(scores[0], 1.0); // (0,0) allowed
    assert_eq!(scores[1], f32::NEG_INFINITY); // (0,1) blocked
    assert_eq!(scores[2], 1.0); // (1,0) allowed
    assert_eq!(scores[3], 1.0); // (1,1) allowed
}

#[test]
fn causal_mask_size_1() {
    let mask = CausalMask::new(1);
    assert!(mask.is_allowed(0, 0));
}

// ── AttentionMask ───────────────────────────────────────────────

#[test]
fn attention_mask_all_allowed() {
    let mask = AttentionMask::all_allowed(2, 3);
    assert_eq!(mask.dims(), (2, 3));
    assert!(mask.is_allowed(0, 0));
    assert!(mask.is_allowed(1, 2));
}

#[test]
fn attention_mask_padding() {
    let mask = AttentionMask::padding(2, 4, 2);
    // Seq len=2, total=4. Positions 0,1 allowed; 2,3 masked
    assert!(mask.is_allowed(0, 0));
    assert!(mask.is_allowed(0, 1));
    assert!(!mask.is_allowed(0, 2));
    assert!(!mask.is_allowed(0, 3));
}

#[test]
fn attention_mask_from_raw() {
    let raw = vec![true, false, true, false];
    let mask = AttentionMask::from_raw(raw, 2, 2);
    assert!(mask.is_allowed(0, 0));
    assert!(!mask.is_allowed(0, 1));
    assert!(mask.is_allowed(1, 0));
    assert!(!mask.is_allowed(1, 1));
}

#[test]
fn attention_mask_apply() {
    let mask = AttentionMask::padding(1, 3, 2);
    let mut scores = vec![1.0, 1.0, 1.0];
    mask.apply(&mut scores, f32::NEG_INFINITY);
    assert_eq!(scores[0], 1.0);
    assert_eq!(scores[1], 1.0);
    assert_eq!(scores[2], f32::NEG_INFINITY);
}

// ── KVCacheEntry ────────────────────────────────────────────────

#[test]
fn kv_cache_entry_new() {
    let entry = KVCacheEntry::new(10, 64);
    assert_eq!(entry.seq_len, 0);
    assert_eq!(entry.capacity, 10);
    assert_eq!(entry.remaining(), 10);
}

#[test]
fn kv_cache_entry_append() {
    let mut entry = KVCacheEntry::new(5, 2);
    let key = vec![1.0, 2.0];
    let val = vec![3.0, 4.0];
    assert!(entry.append(&key, &val));
    assert_eq!(entry.seq_len, 1);
    assert_eq!(entry.remaining(), 4);
}

#[test]
fn kv_cache_entry_get() {
    let mut entry = KVCacheEntry::new(5, 2);
    entry.append(&[1.0, 2.0], &[3.0, 4.0]);
    let k = entry.get_key(0).unwrap();
    assert_eq!(k, &[1.0, 2.0]);
    let v = entry.get_value(0).unwrap();
    assert_eq!(v, &[3.0, 4.0]);
}

#[test]
fn kv_cache_entry_get_out_of_bounds() {
    let entry = KVCacheEntry::new(5, 2);
    assert!(entry.get_key(0).is_none());
}

#[test]
fn kv_cache_entry_clear() {
    let mut entry = KVCacheEntry::new(5, 2);
    entry.append(&[1.0, 2.0], &[3.0, 4.0]);
    entry.clear();
    assert_eq!(entry.seq_len, 0);
    assert_eq!(entry.remaining(), 5);
}

#[test]
fn kv_cache_entry_fill_to_capacity() {
    let mut entry = KVCacheEntry::new(2, 1);
    assert!(entry.append(&[1.0], &[2.0]));
    assert!(entry.append(&[3.0], &[4.0]));
    assert!(!entry.append(&[5.0], &[6.0])); // full
    assert_eq!(entry.remaining(), 0);
}

// ── KVCache ─────────────────────────────────────────────────────

#[test]
fn kv_cache_new() {
    let cache = KVCache::new(4, 128, 64);
    assert_eq!(cache.num_layers(), 4);
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn kv_cache_layer_access() {
    let mut cache = KVCache::new(2, 10, 4);
    cache.layer_mut(0).append(&[1.0; 4], &[2.0; 4]);
    assert_eq!(cache.layer(0).seq_len, 1);
    assert_eq!(cache.layer(1).seq_len, 0);
}

#[test]
fn kv_cache_clear() {
    let mut cache = KVCache::new(2, 10, 4);
    cache.layer_mut(0).append(&[1.0; 4], &[2.0; 4]);
    cache.clear();
    assert_eq!(cache.layer(0).seq_len, 0);
}

// ── ScaledDotProductAttention ───────────────────────────────────

#[test]
fn sdpa_standard() {
    let sdpa = ScaledDotProductAttention::standard(64);
    // scale = 1/sqrt(64) = 0.125
    let _ = format!("{:?}", sdpa);
}

#[test]
fn sdpa_forward_trivial() {
    let sdpa = ScaledDotProductAttention::new(1.0);
    // 1 query, 1 key, 1 value, head_dim=2
    let q = vec![1.0, 0.0];
    let k = vec![1.0, 0.0];
    let v = vec![1.0, 2.0];
    let out = sdpa.forward(&q, &k, &v, 2, None);
    assert_eq!(out.len(), 2);
    // With only one key, output should be close to v
    assert!((out[0] - 1.0).abs() < 1e-3);
    assert!((out[1] - 2.0).abs() < 1e-3);
}

#[test]
fn sdpa_forward_multiple_keys() {
    let sdpa = ScaledDotProductAttention::new(0.5);
    let q = vec![1.0, 0.0]; // 1 query, head_dim=2
    let k = vec![1.0, 0.0, 0.0, 1.0]; // 2 keys
    let v = vec![1.0, 0.0, 0.0, 1.0]; // 2 values
    let out = sdpa.forward(&q, &k, &v, 2, None);
    assert_eq!(out.len(), 2);
}

// ── AttentionEngine ─────────────────────────────────────────────

#[test]
fn attention_engine_mha() {
    let cfg = AttentionConfig::new(1, 4, 16, false);
    let engine = AttentionEngine::new(cfg, AttentionType::MultiHead);
    let _ = format!("{:?}", engine);
}

#[test]
fn attention_engine_forward_with_cache() {
    let cfg = AttentionConfig::new(1, 4, 16, false);
    let engine = AttentionEngine::new(cfg, AttentionType::MultiHead);
    let mut cache = KVCacheEntry::new(16, 4);
    let input = vec![1.0; 4]; // model_dim = 1*4 = 4
    let out = engine.forward_with_cache(&input, &mut cache);
    assert_eq!(out.len(), 4);
    assert_eq!(cache.seq_len, 1);
}

#[test]
fn attention_engine_sequential() {
    let cfg = AttentionConfig::new(1, 4, 16, true);
    let engine = AttentionEngine::new(cfg, AttentionType::MultiHead);
    let mut cache = KVCacheEntry::new(16, 4);

    // Process 3 tokens sequentially
    let out1 = engine.forward_with_cache(&[1.0; 4], &mut cache);
    assert_eq!(cache.seq_len, 1);
    let out2 = engine.forward_with_cache(&[2.0; 4], &mut cache);
    assert_eq!(cache.seq_len, 2);
    let out3 = engine.forward_with_cache(&[3.0; 4], &mut cache);
    assert_eq!(cache.seq_len, 3);

    assert_eq!(out1.len(), 4);
    assert_eq!(out2.len(), 4);
    assert_eq!(out3.len(), 4);
}
