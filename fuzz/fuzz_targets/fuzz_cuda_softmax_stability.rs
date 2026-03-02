#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cuda::{SoftmaxConfig, softmax_cpu};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct CudaSoftmaxInput {
    n_cols: u8,
    n_rows: u8,
    data: Vec<u8>,
    temperature_byte: u8,
    causal_mask: bool,
    inject_extreme: bool,
    extreme_positions: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: CudaSoftmaxInput| {
    let n_cols = (input.n_cols as usize % 64) + 1;
    let n_rows = (input.n_rows as usize % 16) + 1;
    let total = n_cols * n_rows;

    let mut data = bytes_to_f32(&input.data, total);
    if data.len() < total {
        return;
    }

    // Inject extreme finite values at fuzz-selected positions.
    if input.inject_extreme {
        for (i, &pos) in input.extreme_positions.iter().take(8).enumerate() {
            let idx = pos as usize % total;
            match i % 4 {
                0 => data[idx] = f32::MAX / 2.0,
                1 => data[idx] = f32::MIN / 2.0,
                2 => data[idx] = f32::MIN_POSITIVE,
                3 => data[idx] = -1e30,
                _ => {}
            }
        }
    }

    // Skip non-finite inputs.
    if data[..total].iter().any(|x| !x.is_finite()) {
        return;
    }

    // Map temperature to a reasonable positive range (0.01 .. 10.0).
    let temperature = 0.01 + (input.temperature_byte as f32 / 255.0) * 9.99;

    let config = match SoftmaxConfig::for_shape(n_cols, n_rows) {
        Ok(c) => c,
        Err(_) => return,
    };
    let config = match config.with_temperature(temperature) {
        Ok(c) => c,
        Err(_) => return,
    };

    let mut output = vec![0.0f32; total];
    if softmax_cpu(&data[..total], &mut output, &config).is_ok() {
        // Invariant 1: No NaN in output.
        for (i, &val) in output.iter().enumerate() {
            assert!(!val.is_nan(), "softmax output NaN at index {i}");
        }

        // Invariant 2: All values in [0, 1].
        for (i, &val) in output.iter().enumerate() {
            assert!(val >= 0.0, "softmax output negative at index {i}: {val}");
            assert!(val <= 1.0 + 1e-6, "softmax output >1 at index {i}: {val}");
        }

        // Invariant 3: Each row sums to ~1.0.
        for row in 0..n_rows {
            let start = row * n_cols;
            let row_sum: f32 = output[start..start + n_cols].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-3 || row_sum == 0.0,
                "softmax row {row} sum {row_sum} not ≈1.0 (n_cols={n_cols})"
            );
        }
    }

    // Also test with causal mask — must not panic.
    let causal_config = SoftmaxConfig { causal_mask: input.causal_mask, ..config };
    let mut causal_output = vec![0.0f32; total];
    let _ = softmax_cpu(&data[..total], &mut causal_output, &causal_config);
});
