#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::shaped_reduction::{ReductionOp, ShapedReductionConfig, reduce_f32};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ShapedReductionInput {
    data: Vec<u8>,
    shape_bytes: Vec<u8>,
    op: u8,
    axis_byte: u8,
    keepdim: bool,
    use_global: bool,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: ShapedReductionInput| {
    let op = match input.op % 5 {
        0 => ReductionOp::Sum,
        1 => ReductionOp::Max,
        2 => ReductionOp::Min,
        3 => ReductionOp::Mean,
        4 => ReductionOp::L2Norm,
        _ => unreachable!(),
    };

    // Build shape from fuzz bytes (1-4 dimensions, each 1-16)
    let shape: Vec<usize> = input
        .shape_bytes
        .iter()
        .take(4)
        .filter(|&&d| d > 0)
        .map(|&d| (d as usize % 16) + 1)
        .collect();
    if shape.is_empty() {
        return;
    }
    let numel: usize = shape.iter().product();
    if numel == 0 || numel > 4096 {
        return;
    }

    let mut values = bytes_to_f32(&input.data, numel);
    // Filter non-finite for deterministic testing
    for v in values.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    if values.len() < numel {
        values.resize(numel, 0.0);
    }

    let config = if input.use_global {
        ShapedReductionConfig::new(op, None, input.keepdim)
    } else {
        let axis = input.axis_byte as usize % shape.len();
        ShapedReductionConfig::new(op, Some(axis), input.keepdim)
    };

    if let Ok(result) = reduce_f32(&values, &shape, &config) {
        // Result should not be empty for non-empty input
        assert!(!result.is_empty(), "reduction produced empty output");

        // Global reduction produces exactly 1 element (possibly wrapped in keepdim shape)
        if config.axis.is_none() && !config.keepdim {
            assert_eq!(result.len(), 1, "global reduction must produce 1 element");
        }

        // For Sum of all-zeros, result should be zero
        if matches!(op, ReductionOp::Sum) && values.iter().all(|&v| v == 0.0) {
            for r in &result {
                assert!(r.abs() < 1e-6, "sum of zeros should be zero, got {r}");
            }
        }
    }
});
