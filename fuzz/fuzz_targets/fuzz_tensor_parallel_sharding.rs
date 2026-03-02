#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// Fuzz tensor sharding with random shapes and core counts, verifying that
/// shard/unshard round-trips preserve data and shape invariants hold.
#[derive(Arbitrary, Debug)]
struct ShardingInput {
    rows: u8,
    cols: u8,
    num_cores: u8,
    _shard_axis: u8,
    data_bytes: Vec<u8>,
    ops: Vec<ShardOp>,
}

#[derive(Arbitrary, Debug)]
enum ShardOp {
    ShardRows,
    ShardCols,
    Unshard,
    VerifyShapes,
    ReduceSum,
    ReduceMax,
    Scatter { core: u8, offset: u8, value_byte: u8 },
    Gather { core: u8 },
}

struct TensorSharder {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
    num_cores: usize,
    shards: Vec<Vec<f32>>,
    shard_axis: usize, // 0 = rows, 1 = cols
    is_sharded: bool,
}

impl TensorSharder {
    fn new(data: Vec<f32>, rows: usize, cols: usize, num_cores: usize) -> Self {
        Self { data, rows, cols, num_cores, shards: Vec::new(), shard_axis: 0, is_sharded: false }
    }

    fn shard_by_rows(&mut self) {
        self.shards.clear();
        self.shard_axis = 0;
        let rows_per_core = self.rows / self.num_cores;
        let remainder = self.rows % self.num_cores;

        let mut offset = 0;
        for core in 0..self.num_cores {
            let core_rows = rows_per_core + if core < remainder { 1 } else { 0 };
            let count = core_rows * self.cols;
            let end = (offset + count).min(self.data.len());
            self.shards.push(self.data[offset..end].to_vec());
            offset = end;
        }
        self.is_sharded = true;
    }

    fn shard_by_cols(&mut self) {
        self.shards.clear();
        self.shard_axis = 1;
        let cols_per_core = self.cols / self.num_cores;
        let remainder = self.cols % self.num_cores;

        for core in 0..self.num_cores {
            let core_cols = cols_per_core + if core < remainder { 1 } else { 0 };
            let mut shard = Vec::with_capacity(self.rows * core_cols);
            let col_start: usize =
                (0..core).map(|c| cols_per_core + if c < remainder { 1 } else { 0 }).sum();

            for row in 0..self.rows {
                let row_offset = row * self.cols;
                for c in 0..core_cols {
                    let idx = row_offset + col_start + c;
                    if idx < self.data.len() {
                        shard.push(self.data[idx]);
                    }
                }
            }
            self.shards.push(shard);
        }
        self.is_sharded = true;
    }

    fn unshard(&mut self) -> Vec<f32> {
        if !self.is_sharded || self.shards.is_empty() {
            return self.data.clone();
        }

        if self.shard_axis == 0 {
            // Concatenate row shards
            let mut result = Vec::with_capacity(self.rows * self.cols);
            for shard in &self.shards {
                result.extend_from_slice(shard);
            }
            result
        } else {
            // Interleave column shards
            let cols_per_core: Vec<usize> = {
                let base = self.cols / self.num_cores;
                let rem = self.cols % self.num_cores;
                (0..self.num_cores).map(|c| base + if c < rem { 1 } else { 0 }).collect()
            };

            let mut result = vec![0.0f32; self.rows * self.cols];
            for row in 0..self.rows {
                let mut col_offset = 0;
                for (core, shard) in self.shards.iter().enumerate() {
                    let cc = cols_per_core[core];
                    for c in 0..cc {
                        let src_idx = row * cc + c;
                        let dst_idx = row * self.cols + col_offset + c;
                        if src_idx < shard.len() && dst_idx < result.len() {
                            result[dst_idx] = shard[src_idx];
                        }
                    }
                    col_offset += cc;
                }
            }
            result
        }
    }

    fn total_shard_elements(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    fn reduce_sum(&self) -> Vec<f32> {
        if self.shards.is_empty() {
            return vec![];
        }
        let max_len = self.shards.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut result = vec![0.0f32; max_len];
        for shard in &self.shards {
            for (i, &v) in shard.iter().enumerate() {
                if i < result.len() {
                    result[i] += v;
                }
            }
        }
        result
    }
}

fuzz_target!(|input: ShardingInput| {
    let rows = (input.rows as usize % 32) + 1;
    let cols = (input.cols as usize % 32) + 1;
    let num_cores = (input.num_cores as usize % 8) + 1;
    let total = rows * cols;

    // Build data from bytes, pad with zeros.
    let aligned = (input.data_bytes.len() / 4) * 4;
    let mut data: Vec<f32> = input.data_bytes[..aligned]
        .chunks_exact(4)
        .take(total)
        .map(|b| {
            let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            if v.is_finite() { v } else { 0.0 }
        })
        .collect();
    data.resize(total, 0.0);

    let original = data.clone();
    let mut sharder = TensorSharder::new(data, rows, cols, num_cores);

    for op in input.ops.iter().take(64) {
        match op {
            ShardOp::ShardRows => {
                sharder.shard_by_rows();
                // Invariant 1: Number of shards == num_cores.
                assert_eq!(sharder.shards.len(), num_cores);
                // Invariant 2: Total elements across shards == original total.
                assert_eq!(
                    sharder.total_shard_elements(),
                    total,
                    "row shard element count mismatch"
                );
            }
            ShardOp::ShardCols => {
                sharder.shard_by_cols();
                assert_eq!(sharder.shards.len(), num_cores);
                assert_eq!(
                    sharder.total_shard_elements(),
                    total,
                    "col shard element count mismatch"
                );
            }
            ShardOp::Unshard => {
                if sharder.is_sharded {
                    let recovered = sharder.unshard();
                    // Invariant 3: Unshard recovers original data.
                    assert_eq!(recovered.len(), total, "unshard length mismatch");
                    for (i, (&r, &o)) in recovered.iter().zip(original.iter()).enumerate() {
                        assert!((r - o).abs() < 1e-6, "unshard mismatch at {i}: {r} vs {o}");
                    }
                }
            }
            ShardOp::VerifyShapes => {
                if sharder.is_sharded {
                    // Invariant 4: No shard is empty (unless tensor is smaller than cores).
                    if rows >= num_cores && cols >= num_cores {
                        for (i, shard) in sharder.shards.iter().enumerate() {
                            assert!(!shard.is_empty(), "shard {i} is unexpectedly empty");
                        }
                    }
                }
            }
            ShardOp::ReduceSum => {
                if sharder.is_sharded {
                    let reduced = sharder.reduce_sum();
                    // Must not panic; output length is bounded.
                    assert!(reduced.len() <= total);
                }
            }
            ShardOp::ReduceMax => {
                // No-op: just verify we don't panic.
                if sharder.is_sharded && !sharder.shards.is_empty() {
                    for shard in &sharder.shards {
                        if !shard.is_empty() {
                            let _max = shard.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        }
                    }
                }
            }
            ShardOp::Scatter { core, offset, value_byte } => {
                if sharder.is_sharded && !sharder.shards.is_empty() {
                    let c = *core as usize % sharder.shards.len();
                    let shard = &mut sharder.shards[c];
                    if !shard.is_empty() {
                        let idx = *offset as usize % shard.len();
                        shard[idx] = *value_byte as f32 / 255.0;
                    }
                }
            }
            ShardOp::Gather { core } => {
                if sharder.is_sharded && !sharder.shards.is_empty() {
                    let c = *core as usize % sharder.shards.len();
                    let _ = sharder.shards[c].clone();
                }
            }
        }
    }

    // Final invariant: round-trip on fresh shard.
    sharder.shard_by_rows();
    let rt = sharder.unshard();
    assert_eq!(rt.len(), total);
});
