#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType, PoolingKernel};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct PoolingInput {
    ops: Vec<PoolOp>,
}

#[derive(Arbitrary, Debug)]
struct PoolOp {
    pool_type: u8,
    kernel_size: u8,
    stride: u8,
    padding: u8,
    data: Vec<f32>,
    adaptive_output_size: u8,
}

fuzz_target!(|input: PoolingInput| {
    for op in input.ops.into_iter().take(256) {
        let data: Vec<f32> =
            op.data.into_iter().take(128).map(|v| if v.is_finite() { v } else { 0.0 }).collect();

        if data.is_empty() {
            continue;
        }

        let pool_type = match op.pool_type % 4 {
            0 => PoolType::Max,
            1 => PoolType::Average,
            2 => PoolType::GlobalMax,
            _ => PoolType::GlobalAverage,
        };

        let config = PoolConfig {
            pool_type,
            kernel_size: (op.kernel_size as usize).clamp(1, 32),
            stride: (op.stride as usize).clamp(1, 16),
            padding: (op.padding as usize) % 8,
        };

        match PoolingKernel::apply(&data, &config) {
            Ok(out) => {
                for v in &out {
                    assert!(v.is_finite(), "pooling produced non-finite: {v}");
                }
            }
            Err(_) => {}
        }

        // Exercise adaptive_config
        let out_size = (op.adaptive_output_size as usize).clamp(1, data.len().max(1));
        if let Ok(adaptive_cfg) = PoolingKernel::adaptive_config(pool_type, data.len(), out_size) {
            if let Ok(out) = PoolingKernel::apply(&data, &adaptive_cfg) {
                for v in &out {
                    assert!(v.is_finite(), "adaptive pooling produced non-finite: {v}");
                }
            }
        }
    }
});
