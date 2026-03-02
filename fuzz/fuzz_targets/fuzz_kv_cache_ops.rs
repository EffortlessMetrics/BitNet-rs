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
    Evict { layer: u8, count: u8 },
    Compact { layer: u8 },
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
    tombstones: Vec<Vec<bool>>,
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
            tombstones: vec![Vec::new(); n_layers],
        }
    }

    fn step_size(&self) -> usize {
        self.n_heads * self.head_dim
    }

    fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) -> bool {
        if layer >= self.n_layers {
            return false;
        }
        let step = self.step_size();
        if k.len() != step || v.len() != step {
            return false;
        }
        self.keys[layer].extend_from_slice(k);
        self.values[layer].extend_from_slice(v);
        self.tombstones[layer].push(false);
        self.seq_lens[layer] += 1;
        true
    }

    fn evict(&mut self, layer: usize, count: usize) {
        if layer >= self.n_layers {
            return;
        }
        let mut evicted = 0;
        for i in 0..self.tombstones[layer].len() {
            if evicted >= count {
                break;
            }
            if !self.tombstones[layer][i] {
                self.tombstones[layer][i] = true;
                evicted += 1;
            }
        }
    }

    fn compact(&mut self, layer: usize) {
        if layer >= self.n_layers {
            return;
        }
        let step = self.step_size();
        let mut new_keys = Vec::new();
        let mut new_values = Vec::new();
        let mut new_tombstones = Vec::new();
        for (i, &dead) in self.tombstones[layer].iter().enumerate() {
            if !dead {
                let start = i * step;
                let end = start + step;
                if end <= self.keys[layer].len() {
                    new_keys.extend_from_slice(&self.keys[layer][start..end]);
                    new_values.extend_from_slice(&self.values[layer][start..end]);
                    new_tombstones.push(false);
                }
            }
        }
        let live_count = new_tombstones.len();
        self.keys[layer] = new_keys;
        self.values[layer] = new_values;
        self.tombstones[layer] = new_tombstones;
        self.seq_lens[layer] = live_count;
    }

    fn live_count(&self, layer: usize) -> usize {
        if layer >= self.n_layers {
            return 0;
        }
        self.tombstones[layer].iter().filter(|&&t| !t).count()
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
            self.tombstones[layer].clear();
        }
    }

    fn trim_to(&mut self, max_seq: usize) {
        let step = self.step_size();
        for layer in 0..self.n_layers {
            if self.seq_lens[layer] > max_seq {
                let keep = max_seq * step;
                self.keys[layer].truncate(keep);
                self.values[layer].truncate(keep);
                self.tombstones[layer].truncate(max_seq);
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
                        assert_eq!(
                            cache.seq_len(layer_idx),
                            prev_len + 1,
                            "seq_len should increment by 1"
                        );
                        let (keys, values, seq) = cache.read_layer(layer_idx).unwrap();
                        assert_eq!(keys.len(), seq * step_size, "key buffer size mismatch");
                        assert_eq!(values.len(), seq * step_size, "value buffer size mismatch");
                    }
                }
            }
            CacheOp::Evict { layer, count } => {
                let layer_idx = layer as usize % n_layers;
                let evict_n = (count as usize % 8) + 1;
                let live_before = cache.live_count(layer_idx);
                cache.evict(layer_idx, evict_n);
                let live_after = cache.live_count(layer_idx);
                // Eviction can only decrease or maintain live count
                assert!(
                    live_after <= live_before,
                    "evict must not increase live count: {live_before} -> {live_after}"
                );
                let evicted = live_before - live_after;
                assert!(evicted <= evict_n, "evicted more than requested: {evicted} > {evict_n}");
            }
            CacheOp::Compact { layer } => {
                let layer_idx = layer as usize % n_layers;
                let live_before = cache.live_count(layer_idx);
                cache.compact(layer_idx);
                // After compact: no tombstones, seq_len == live count
                assert_eq!(
                    cache.seq_len(layer_idx),
                    live_before,
                    "compact should preserve live entries"
                );
                assert_eq!(
                    cache.live_count(layer_idx),
                    live_before,
                    "compact should not lose live entries"
                );
                let (keys, values, seq) = cache.read_layer(layer_idx).unwrap();
                assert_eq!(keys.len(), seq * step_size, "compact key size mismatch");
                assert_eq!(values.len(), seq * step_size, "compact value size mismatch");
            }
            CacheOp::ReadLayer { layer } => {
                let layer_idx = layer as usize % n_layers;
                let result = cache.read_layer(layer_idx);
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
                for l in 0..n_layers {
                    assert_eq!(cache.seq_len(l), 0, "after reset, layer {l} should have seq_len=0");
                    assert_eq!(cache.live_count(l), 0, "after reset, layer {l} should have 0 live");
                    let (keys, values, _) = cache.read_layer(l).unwrap();
                    assert!(keys.is_empty(), "keys should be empty after reset");
                    assert!(values.is_empty(), "values should be empty after reset");
                }
            }
            CacheOp::TrimTo { max_seq } => {
                let max = (max_seq as usize % 32) + 1;
                cache.trim_to(max);
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

    // Out-of-bounds layer reads return None
    assert!(cache.read_layer(n_layers).is_none(), "OOB layer read should return None");
    assert_eq!(cache.seq_len(n_layers), 0, "OOB layer seq_len should be 0");

    // Layers are independent
    cache.reset();
    let dummy_k = vec![1.0f32; step_size];
    let dummy_v = vec![2.0f32; step_size];
    cache.append(0, &dummy_k, &dummy_v);
    for l in 1..n_layers {
        assert_eq!(cache.seq_len(l), 0, "layer {l} should be unaffected by append to layer 0");
    }

    // Evict-then-compact preserves data integrity
    cache.reset();
    for _ in 0..4 {
        cache.append(0, &dummy_k, &dummy_v);
    }
    cache.evict(0, 2);
    assert_eq!(cache.live_count(0), 2);
    cache.compact(0);
    assert_eq!(cache.seq_len(0), 2, "compact after evict should leave 2 entries");
    assert_eq!(cache.live_count(0), 2);
});
