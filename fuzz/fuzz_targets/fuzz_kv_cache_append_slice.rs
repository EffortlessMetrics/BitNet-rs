#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_memory_usage,
    kv_cache_slice, paged_kv_cache_alloc,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    num_layers: u8,
    num_heads: u8,
    head_dim: u8,
    max_seq_len: u8,
    ops: Vec<CacheOp>,
}

#[derive(Arbitrary, Debug)]
enum CacheOp {
    Append { layer: u8, data: Vec<u8> },
    Slice { layer: u8, start: u8, end: u8 },
    Clear,
    MemUsage,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: Input| {
    let num_layers = (input.num_layers as usize % 4) + 1;
    let num_heads = (input.num_heads as usize % 4) + 1;
    let head_dim = (input.head_dim as usize % 8) + 1;
    let max_seq_len = (input.max_seq_len as usize % 16) + 1;
    let token_elems = num_heads * head_dim;

    let config =
        KvCacheConfig { num_layers, num_heads, head_dim, max_seq_len, dtype: KvDtype::F32 };

    let mut cache = match KvCache::new(config) {
        Ok(c) => c,
        Err(_) => return,
    };

    // Invariant: fresh cache has num_layers blocks and zero seq_len
    assert_eq!(cache.num_layers(), num_layers);
    for l in 0..num_layers {
        assert_eq!(cache.seq_len(l).unwrap(), 0);
    }

    for op in input.ops.into_iter().take(128) {
        match op {
            CacheOp::Append { layer, data } => {
                let l = layer as usize % num_layers;
                let vals = bytes_to_f32(&data, token_elems * 2);
                if vals.len() >= token_elems {
                    let k = &vals[..token_elems];
                    let v = if vals.len() >= token_elems * 2 {
                        &vals[token_elems..token_elems * 2]
                    } else {
                        k
                    };
                    let prev = cache.seq_len(l).unwrap();
                    if kv_cache_append(&mut cache, l, k, v).is_ok() {
                        assert_eq!(cache.seq_len(l).unwrap(), prev + 1);
                    }
                }
            }
            CacheOp::Slice { layer, start, end } => {
                let l = layer as usize % num_layers;
                let s = start as usize;
                let e = end as usize;
                if let Ok((keys, values)) = kv_cache_slice(&cache, l, s, e) {
                    let expected = (e - s) * token_elems;
                    assert_eq!(keys.len(), expected);
                    assert_eq!(values.len(), expected);
                }
            }
            CacheOp::Clear => {
                kv_cache_clear(&mut cache);
                for l in 0..num_layers {
                    assert_eq!(cache.seq_len(l).unwrap(), 0);
                }
            }
            CacheOp::MemUsage => {
                let usage = kv_cache_memory_usage(&cache);
                assert!(usage > 0);
            }
        }
    }

    // Also exercise paged allocation path
    let _ = paged_kv_cache_alloc((num_layers % 4) + 1, (max_seq_len % 8) + 1, num_heads, head_dim);
});
