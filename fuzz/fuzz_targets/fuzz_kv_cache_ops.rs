#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_memory_usage,
    kv_cache_slice,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct KvCacheInput {
    num_layers: u8,
    num_heads: u8,
    head_dim: u8,
    max_seq_len: u8,
    ops: Vec<(u8, u8, Vec<u8>)>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: KvCacheInput| {
    let num_layers = (input.num_layers as usize % 4) + 1;
    let num_heads = (input.num_heads as usize % 4) + 1;
    let head_dim = (input.head_dim as usize % 32) + 1;
    let max_seq_len = (input.max_seq_len as usize % 64) + 1;
    let token_elems = num_heads * head_dim;

    let config =
        KvCacheConfig { num_layers, num_heads, head_dim, max_seq_len, dtype: KvDtype::F32 };

    let mut cache = match KvCache::new(config) {
        Ok(c) => c,
        Err(_) => return,
    };

    // Invariant: freshly created cache has zero sequence length.
    for layer in 0..num_layers {
        assert_eq!(cache.seq_len(layer).unwrap(), 0);
    }

    for (op, param, raw_data) in input.ops.iter().take(64) {
        let layer = *param as usize % num_layers;
        match op % 4 {
            // Append single token
            0 => {
                let keys = bytes_to_f32(raw_data, token_elems);
                let values =
                    bytes_to_f32(raw_data.get(token_elems * 4..).unwrap_or(&[]), token_elems);
                if keys.len() < token_elems || values.len() < token_elems {
                    continue;
                }
                // Filter non-finite values.
                let keys: Vec<f32> = keys[..token_elems]
                    .iter()
                    .map(|&v| if v.is_finite() { v } else { 0.0 })
                    .collect();
                let values: Vec<f32> = values[..token_elems]
                    .iter()
                    .map(|&v| if v.is_finite() { v } else { 0.0 })
                    .collect();

                let prev_len = cache.seq_len(layer).unwrap_or(0);
                if kv_cache_append(&mut cache, layer, &keys, &values).is_ok() {
                    let new_len = cache.seq_len(layer).unwrap();
                    assert!(
                        new_len > prev_len,
                        "seq_len should increase after append: {prev_len} -> {new_len}"
                    );
                }
            }
            // Slice
            1 => {
                let seq = cache.seq_len(layer).unwrap_or(0);
                if seq > 0 {
                    let start = *param as usize % seq;
                    let end = start + 1;
                    if let Ok((k_slice, v_slice)) = kv_cache_slice(&cache, layer, start, end) {
                        assert_eq!(k_slice.len(), token_elems);
                        assert_eq!(v_slice.len(), token_elems);
                    }
                }
            }
            // Clear
            2 => {
                kv_cache_clear(&mut cache);
                for l in 0..num_layers {
                    assert_eq!(cache.seq_len(l).unwrap(), 0, "seq_len should be 0 after clear");
                }
            }
            // Memory usage
            _ => {
                let usage = kv_cache_memory_usage(&cache);
                assert!(usage > 0, "memory usage should be positive");
            }
        }
    }

    // Invariant: layer independence — appending to one layer doesn't affect others.
    kv_cache_clear(&mut cache);
    let token_keys = vec![1.0f32; token_elems];
    let token_values = vec![2.0f32; token_elems];
    if num_layers >= 2 {
        let _ = kv_cache_append(&mut cache, 0, &token_keys, &token_values);
        assert_eq!(cache.seq_len(0).unwrap(), 1);
        assert_eq!(cache.seq_len(1).unwrap(), 0, "layers should be independent");
    }
});
