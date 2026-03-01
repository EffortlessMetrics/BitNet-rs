#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct EvictionInput {
    n_layers: u8,
    n_heads: u8,
    head_dim: u8,
    capacity: u8,
    ops: Vec<EvictionOp>,
}

#[derive(Arbitrary, Debug)]
enum EvictionOp {
    Append { layer: u8, data: Vec<u8> },
    EvictOldest { layer: u8, count: u8 },
    EvictRange { layer: u8, start: u8, end: u8 },
    ReadSeqLen { layer: u8 },
    ReadSlice { layer: u8, start: u8, end: u8 },
    Reset,
}

struct EvictableKvCache {
    n_layers: usize,
    n_heads: usize,
    head_dim: usize,
    capacity: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    seq_lens: Vec<usize>,
}

impl EvictableKvCache {
    fn new(n_layers: usize, n_heads: usize, head_dim: usize, capacity: usize) -> Self {
        Self {
            n_layers,
            n_heads,
            head_dim,
            capacity,
            keys: vec![Vec::new(); n_layers],
            values: vec![Vec::new(); n_layers],
            seq_lens: vec![0; n_layers],
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
        // Evict oldest if at capacity.
        if self.seq_lens[layer] >= self.capacity && self.capacity > 0 {
            self.keys[layer].drain(..step);
            self.values[layer].drain(..step);
            self.seq_lens[layer] -= 1;
        }
        self.keys[layer].extend_from_slice(k);
        self.values[layer].extend_from_slice(v);
        self.seq_lens[layer] += 1;
        true
    }

    fn evict_oldest(&mut self, layer: usize, count: usize) -> usize {
        if layer >= self.n_layers {
            return 0;
        }
        let actual = count.min(self.seq_lens[layer]);
        let step = self.step_size();
        let remove = actual * step;
        self.keys[layer].drain(..remove);
        self.values[layer].drain(..remove);
        self.seq_lens[layer] -= actual;
        actual
    }

    fn evict_range(&mut self, layer: usize, start: usize, end: usize) -> usize {
        if layer >= self.n_layers || start >= end || start >= self.seq_lens[layer] {
            return 0;
        }
        let clamped_end = end.min(self.seq_lens[layer]);
        let count = clamped_end - start;
        let step = self.step_size();
        let byte_start = start * step;
        let byte_end = clamped_end * step;
        self.keys[layer].drain(byte_start..byte_end);
        self.values[layer].drain(byte_start..byte_end);
        self.seq_lens[layer] -= count;
        count
    }

    fn seq_len(&self, layer: usize) -> usize {
        if layer >= self.n_layers { 0 } else { self.seq_lens[layer] }
    }

    fn read_slice(&self, layer: usize, start: usize, end: usize) -> Option<(&[f32], &[f32])> {
        if layer >= self.n_layers || start >= end || end > self.seq_lens[layer] {
            return None;
        }
        let step = self.step_size();
        let s = start * step;
        let e = end * step;
        Some((&self.keys[layer][s..e], &self.values[layer][s..e]))
    }

    fn reset(&mut self) {
        for l in 0..self.n_layers {
            self.keys[l].clear();
            self.values[l].clear();
            self.seq_lens[l] = 0;
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

fuzz_target!(|input: EvictionInput| {
    let n_layers = (input.n_layers as usize % 4) + 1;
    let n_heads = (input.n_heads as usize % 4) + 1;
    let head_dim = (input.head_dim as usize % 16) + 1;
    let capacity = (input.capacity as usize % 32) + 1;
    let step = n_heads * head_dim;

    let mut cache = EvictableKvCache::new(n_layers, n_heads, head_dim, capacity);

    for op in input.ops.into_iter().take(256) {
        match op {
            EvictionOp::Append { layer, data } => {
                let layer_idx = layer as usize % n_layers;
                let kv = bytes_to_f32(&data, step * 2);
                if kv.len() >= step * 2 {
                    let prev = cache.seq_len(layer_idx);
                    let ok = cache.append(layer_idx, &kv[..step], &kv[step..step * 2]);
                    if ok {
                        let new_len = cache.seq_len(layer_idx);
                        // Invariant 1: After append, seq_len is at most capacity.
                        assert!(
                            new_len <= capacity,
                            "seq_len {} exceeds capacity {}",
                            new_len,
                            capacity
                        );
                        // Invariant 2: seq_len increments or stays same (eviction case).
                        assert!(
                            new_len == prev + 1 || new_len == prev,
                            "unexpected seq_len transition {prev} -> {new_len}"
                        );
                    }
                }
            }
            EvictionOp::EvictOldest { layer, count } => {
                let layer_idx = layer as usize % n_layers;
                let prev = cache.seq_len(layer_idx);
                let evicted = cache.evict_oldest(layer_idx, count as usize);
                let new_len = cache.seq_len(layer_idx);

                // Invariant 3: Evicted count never exceeds previous length.
                assert!(evicted <= prev, "evicted {evicted} > prev len {prev}");

                // Invariant 4: New length is prev - evicted.
                assert_eq!(new_len, prev - evicted, "seq_len after eviction mismatch");

                // Invariant 5: Buffer sizes are consistent.
                assert_eq!(cache.keys[layer_idx].len(), new_len * step);
                assert_eq!(cache.values[layer_idx].len(), new_len * step);
            }
            EvictionOp::EvictRange { layer, start, end } => {
                let layer_idx = layer as usize % n_layers;
                let prev = cache.seq_len(layer_idx);
                let s = start as usize;
                let e = end as usize;
                let evicted = cache.evict_range(layer_idx, s, e);
                let new_len = cache.seq_len(layer_idx);

                // Invariant 6: Evicted count never exceeds previous length.
                assert!(evicted <= prev);
                assert_eq!(new_len, prev - evicted);
                assert_eq!(cache.keys[layer_idx].len(), new_len * step);
            }
            EvictionOp::ReadSeqLen { layer } => {
                let layer_idx = layer as usize % n_layers;
                let len = cache.seq_len(layer_idx);
                // Invariant 7: seq_len matches actual buffer size / step.
                assert_eq!(cache.keys[layer_idx].len(), len * step);
                assert_eq!(cache.values[layer_idx].len(), len * step);
            }
            EvictionOp::ReadSlice { layer, start, end } => {
                let layer_idx = layer as usize % n_layers;
                let s = start as usize;
                let e = end as usize;
                if let Some((k, v)) = cache.read_slice(layer_idx, s, e) {
                    let expected = (e - s) * step;
                    // Invariant 8: Slice sizes match expected.
                    assert_eq!(k.len(), expected);
                    assert_eq!(v.len(), expected);
                }
            }
            EvictionOp::Reset => {
                cache.reset();
                for l in 0..n_layers {
                    assert_eq!(cache.seq_len(l), 0);
                    assert!(cache.keys[l].is_empty());
                    assert!(cache.values[l].is_empty());
                }
            }
        }
    }
});
