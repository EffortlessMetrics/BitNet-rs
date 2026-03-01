#![no_main]

use arbitrary::Arbitrary;
use bitnet_inference::tensor_parallel::{
    AllReduceOp, PartitionStrategy, TensorPartition, all_reduce, compute_load_balance, gather,
    partition_tensor, scatter,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ScatterInput {
    ops: Vec<ScatterOp>,
}

#[derive(Arbitrary, Debug)]
enum ScatterOp {
    Scatter { data_len: u8, num_partitions: u8 },
    Partition { shape: Vec<u8>, num_devices: u8, strategy: u8 },
    AllReduce { num_partitions: u8, partition_len: u8, op: u8 },
}

fuzz_target!(|input: ScatterInput| {
    for op in input.ops.into_iter().take(256) {
        match op {
            ScatterOp::Scatter { data_len, num_partitions } => {
                let len = (data_len as usize).min(128);
                let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
                let np = (num_partitions as usize).min(32);
                if let Ok(parts) = scatter(&data, np) {
                    // Roundtrip: gather must reconstruct original
                    if let Ok(gathered) = gather(&parts) {
                        assert_eq!(gathered.len(), data.len());
                        for (a, b) in gathered.iter().zip(data.iter()) {
                            assert!((a - b).abs() < 1e-6);
                        }
                    }
                    let _ = compute_load_balance(&parts);
                }
            }
            ScatterOp::Partition { shape, num_devices, strategy } => {
                let shape: Vec<usize> =
                    shape.iter().take(4).map(|&d| (d as usize).min(64)).collect();
                let nd = (num_devices as usize).min(16);
                let strat = match strategy % 3 {
                    0 => PartitionStrategy::RowParallel,
                    1 => PartitionStrategy::ColumnParallel,
                    _ => PartitionStrategy::Hybrid,
                };
                if let Ok(parts) = partition_tensor(&shape, nd, strat) {
                    let balance = compute_load_balance(&parts);
                    assert!(balance >= 0.0 && balance <= 1.0);
                }
            }
            ScatterOp::AllReduce { num_partitions, partition_len, op } => {
                let np = ((num_partitions as usize) % 8) + 1;
                let len = ((partition_len as usize) % 32) + 1;
                let parts: Vec<TensorPartition> = (0..np)
                    .map(|i| TensorPartition {
                        device_id: i,
                        offset: 0,
                        size: len,
                        shape: vec![len],
                        data: (0..len).map(|j| (i * len + j) as f32).collect(),
                    })
                    .collect();

                let reduce_op = match op % 3 {
                    0 => AllReduceOp::Sum,
                    1 => AllReduceOp::Mean,
                    _ => AllReduceOp::Max,
                };
                if let Ok(result) = all_reduce(&parts, reduce_op) {
                    assert_eq!(result.len(), len);
                }
            }
        }
    }
});
