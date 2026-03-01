#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct KvCacheInput {
    n_layers: u8,
    n_heads: u8,
    head_dim: u8,
    ops: Vec<CacheOp>,
}

#[derive(Arbitrary, Debug)]
enum CacheOp {
    Append { layer: u8, data: Vec<u8> },
    ReadLayer { layer: u8 },
    ReadAll,
    Reset,
    TrimTo { max_seq: u8 },
}

struct KvCache {
    n_layers: usize,
    n_heads: usize,
    head_dim: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    seq_lens: Vec<usize>,
}

impl KvCache {
    fn new(n_layers: usize, n_heads: usize, head_dim: usize) -> Self {
        Self {
            n_layers,
            n_heads,
            head_dim,
            keys: vec![Vec::new(); n_layers],
            values: vec![Vec::new(); n_layers],
            seq_lens: vec![0; n_layers],
        }
    }

    fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) -> bool {
        if layer >= self.n_layers {
            return false;
        }
        let step_size = self.n_heads * self.head_dim;
        if k.len() != step_size || v.len() != step_size {
            return false;
        }
        self.keys[layer].extend_from_slice(k);
        self.values[layer].extend_from_slice(v);
        self.seq_lens[layer] += 1;
        true
    }

    fn read_layer(&self, layer: usize) -> Option<(&[f32], &[f32], usize)> {
        if layer >= self.n_layers {
            return None;
        }
        Some((&self.keys[layer], &self.values[layer], self.seq_lens[layer]))
    }

    fn seq_len(&self, layer: usize) -> usize {
        if layer >= self.n_layers {
            return 0;
        }
        self.seq_lens[layer]
    }

    fn reset(&mut self) {
        for layer in 0..self.n_layers {
            self.keys[layer].clear();
            self.values[layer].clear();
            self.seq_lens[layer] = 0;
        }
    }

    fn trim_to(&mut self, max_seq: usize) {
        let step_size = self.n_heads * self.head_dim;
        for layer in 0..self.n_layers {
            if self.seq_lens[layer] > max_seq {
                let keep = max_seq * step_size;
                self.keys[layer].truncate(keep);
                self.values[layer].truncate(keep);
                self.seq_lens[layer] = max_seq;
            }
        }
    }
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
    let n_layers = (input.n_layers as usize % 4) + 1;
    let n_heads = (input.n_heads as usize % 4) + 1;
    let head_dim = (input.head_dim as usize % 16) + 1;
    let step_size = n_heads * head_dim;

    let mut cache = KvCache::new(n_layers, n_heads, head_dim);

    // Invariant 1: Fresh cache has zero seq_len for all layers
    for l in 0..n_layers {
        assert_eq!(cache.seq_len(l), 0, "fresh cache layer {l} should have seq_len=0");
    }

    for op in input.ops.into_iter().take(256) {
        match op {
            CacheOp::Append { layer, data } => {
                let layer_idx = layer as usize % n_layers;
                let kv_data = bytes_to_f32(&data, step_size * 2);
                if kv_data.len() >= step_size * 2 {
                    let k = &kv_data[..step_size];
                    let v = &kv_data[step_size..step_size * 2];
                    let prev_len = cache.seq_len(layer_idx);
                    let ok = cache.append(layer_idx, k, v);
                    if ok {
                        // Invariant 2: seq_len increments by 1 on successful append
                        assert_eq!(
                            cache.seq_len(layer_idx),
                            prev_len + 1,
                            "seq_len should increment by 1"
                        );

                        // Invariant 3: Key/value buffer size matches seq_len * step_size
                        let (keys, values, seq) = cache.read_layer(layer_idx).unwrap();
                        assert_eq!(keys.len(), seq * step_size, "key buffer size mismatch");
                        assert_eq!(values.len(), seq * step_size, "value buffer size mismatch");
                    }
                }
            }
            CacheOp::ReadLayer { layer } => {
                let layer_idx = layer as usize % n_layers;
                let result = cache.read_layer(layer_idx);
                // Invariant 4: Valid layer read always succeeds
                assert!(result.is_some(), "valid layer {layer_idx} read should succeed");
                let (keys, values, seq) = result.unwrap();
                assert_eq!(keys.len(), values.len(), "k/v lengths should match");
                assert_eq!(keys.len(), seq * step_size, "buffer size vs seq_len mismatch");
            }
            CacheOp::ReadAll => {
                for l in 0..n_layers {
                    let (keys, values, seq) = cache.read_layer(l).unwrap();
                    assert_eq!(keys.len(), seq * step_size);
                    assert_eq!(values.len(), seq * step_size);
                }
            }
            CacheOp::Reset => {
                cache.reset();
                // Invariant 5: After reset, all layers have seq_len=0
                for l in 0..n_layers {
                    assert_eq!(cache.seq_len(l), 0, "after reset, layer {l} should have seq_len=0");
                    let (keys, values, _) = cache.read_layer(l).unwrap();
                    assert!(keys.is_empty(), "keys should be empty after reset");
                    assert!(values.is_empty(), "values should be empty after reset");
                }
            }
            CacheOp::TrimTo { max_seq } => {
                let max = (max_seq as usize % 32) + 1;
                cache.trim_to(max);
                // Invariant 6: After trim, all seq_lens <= max
                for l in 0..n_layers {
                    assert!(
                        cache.seq_len(l) <= max,
                        "after trim to {max}, layer {l} has seq_len={}",
                        cache.seq_len(l)
                    );
                }
            }
        }
    }

    // Invariant 7: Out-of-bounds layer reads return None
    assert!(cache.read_layer(n_layers).is_none(), "OOB layer read should return None");
    assert_eq!(cache.seq_len(n_layers), 0, "OOB layer seq_len should be 0");

    // Invariant 8: Layers are independent — appending to one doesn't affect others
    cache.reset();
    let dummy_k = vec![1.0f32; step_size];
    let dummy_v = vec![2.0f32; step_size];
    cache.append(0, &dummy_k, &dummy_v);
    for l in 1..n_layers {
        assert_eq!(cache.seq_len(l), 0, "layer {l} should be unaffected by append to layer 0");
    }
});
