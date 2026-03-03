#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cuda::matmul::MatmulConfig;
use bitnet_kernels::cuda::softmax::SoftmaxConfig;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct CudaConfigInput {
    m: u16,
    n: u16,
    k: u16,
    tile_size: u8,
    n_cols: u16,
    n_rows: u16,
    temperature_byte: u8,
}

fuzz_target!(|input: CudaConfigInput| {
    // Fuzz MatmulConfig::for_shape with arbitrary dimensions.
    let m = input.m as usize;
    let n = input.n as usize;
    let k = input.k as usize;

    let _ = MatmulConfig::for_shape(m, n, k);

    // Fuzz with tiled variant and arbitrary tile size.
    let tile = (input.tile_size as u32 % 128) + 1;
    let _ = MatmulConfig::for_shape_tiled(m, n, k, tile);

    // Zero dimensions must not panic.
    let _ = MatmulConfig::for_shape(0, n, k);
    let _ = MatmulConfig::for_shape(m, 0, k);
    let _ = MatmulConfig::for_shape(m, n, 0);

    // Fuzz SoftmaxConfig::for_shape with arbitrary rows/cols.
    let cols = input.n_cols as usize;
    let rows = input.n_rows as usize;
    let _ = SoftmaxConfig::for_shape(cols, rows);

    // Chain with_temperature on successful config.
    if let Ok(cfg) = SoftmaxConfig::for_shape(cols.max(1), rows.max(1)) {
        let temp = 0.01 + (input.temperature_byte as f32 / 255.0) * 9.99;
        let _ = cfg.clone().with_temperature(temp);
        // Extreme temperatures must not panic.
        let _ = cfg.clone().with_temperature(0.0);
        let _ = cfg.clone().with_temperature(-1.0);
        let _ = cfg.clone().with_temperature(f32::MAX);
        let _ = cfg.clone().with_temperature(f32::NAN);
    }
});
