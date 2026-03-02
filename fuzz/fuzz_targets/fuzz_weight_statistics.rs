#![no_main]

//! Fuzz weight statistics computation with extreme values: exercises
//! `TensorStats::from_f32` and anomaly detection with NaN, Inf, subnormals,
//! zeros, and extreme magnitudes to verify numerical robustness.

use arbitrary::Arbitrary;
use bitnet_models::weight_stats::{TensorStats, detect_anomalies};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct WeightStatsInput {
    name: String,
    shape_dims: Vec<u16>,
    raw_data: Vec<u8>,
    inject_nan_positions: Vec<u16>,
    inject_inf_positions: Vec<u16>,
    inject_neg_inf_positions: Vec<u16>,
}

fn bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fuzz_target!(|input: WeightStatsInput| {
    let mut data = bytes_to_f32_vec(&input.raw_data);
    if data.is_empty() {
        return;
    }

    // Inject special values at fuzzed positions.
    for &pos in input.inject_nan_positions.iter().take(8) {
        let idx = pos as usize % data.len();
        data[idx] = f32::NAN;
    }
    for &pos in input.inject_inf_positions.iter().take(8) {
        let idx = pos as usize % data.len();
        data[idx] = f32::INFINITY;
    }
    for &pos in input.inject_neg_inf_positions.iter().take(8) {
        let idx = pos as usize % data.len();
        data[idx] = f32::NEG_INFINITY;
    }

    let shape: Vec<usize> = if input.shape_dims.is_empty() {
        vec![data.len()]
    } else {
        input.shape_dims.iter().take(4).map(|&d| (d as usize).max(1)).collect()
    };

    let name = if input.name.is_empty() { "fuzz_tensor" } else { &input.name };

    let stats = TensorStats::from_f32(name, &shape, &data);

    // Derived metrics must not panic.
    let _ = stats.std_dev();
    let _ = stats.has_anomalies();
    let _ = stats.sparsity();
    let _ = stats.is_sparse();

    // Anomaly detection must not panic on any input.
    let anomalies = detect_anomalies(&stats);
    let _ = format!("{:?}", anomalies);

    // element_count should equal data length regardless of shape.
    assert_eq!(stats.element_count, data.len());

    // nan_count, inf_count, and zero_count must be bounded.
    assert!(stats.nan_count <= data.len());
    assert!(stats.inf_count <= data.len());
    assert!(stats.zero_count <= data.len());

    // Debug formatting must not panic.
    let _ = format!("{:?}", stats);
});
