#![no_main]

//! Fuzz CUDA kernel launch parameter validation: exercises `MatmulConfig`
//! construction and validation with arbitrary dimensions, tile sizes, and
//! thread counts to ensure no panics in parameter validation paths.

use arbitrary::Arbitrary;
use bitnet_kernels::cuda::{MatmulConfig, MatmulDtype};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct CudaLaunchInput {
    m: u16,
    n: u16,
    k: u16,
    batch_size: u8,
    alpha: f32,
    beta: f32,
    dtype_selector: u8,
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
    threads_per_block: u32,
    shared_mem_bytes: u32,
    transpose_a: bool,
    transpose_b: bool,
}

fuzz_target!(|input: CudaLaunchInput| {
    let m = (input.m as usize).max(1);
    let n = (input.n as usize).max(1);
    let k = (input.k as usize).max(1);

    // Try constructing via for_shape — may reject invalid dims.
    let config_result = MatmulConfig::for_shape(m, n, k);

    if let Ok(mut config) = config_result {
        config.batch_size = (input.batch_size as usize).max(1);
        config.transpose_a = input.transpose_a;
        config.transpose_b = input.transpose_b;
        config.alpha = if input.alpha.is_finite() { input.alpha } else { 1.0 };
        config.beta = if input.beta.is_finite() { input.beta } else { 0.0 };
        config.dtype =
            if input.dtype_selector % 2 == 0 { MatmulDtype::F32 } else { MatmulDtype::F16 };
        config.tile_m = input.tile_m;
        config.tile_n = input.tile_n;
        config.tile_k = input.tile_k;
        config.threads_per_block = input.threads_per_block;
        config.shared_mem_bytes = input.shared_mem_bytes;

        // Debug formatting must not panic.
        let _ = format!("{:?}", config);
    }

    // Also exercise default construction.
    let default_config = MatmulConfig::default();
    let _ = format!("{:?}", default_config);
});
