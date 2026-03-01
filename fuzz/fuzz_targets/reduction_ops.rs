#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::reduction::{ReductionOp, reduce_f32, reduce_rows_f32};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ReductionInput {
    /// Raw data bytes (interpreted as f32).
    data: Vec<u8>,
    /// Number of rows for matrix reduction.
    rows: u8,
    /// Which operation to test (mod 5 maps to the enum variants).
    op_idx: u8,
}

fn op_from_idx(idx: u8) -> ReductionOp {
    match idx % 5 {
        0 => ReductionOp::Sum,
        1 => ReductionOp::Max,
        2 => ReductionOp::Min,
        3 => ReductionOp::Mean,
        _ => ReductionOp::L2Norm,
    }
}

fuzz_target!(|input: ReductionInput| {
    let aligned_len = (input.data.len() / 4) * 4;
    if aligned_len == 0 {
        return;
    }

    let data: Vec<f32> = input.data[..aligned_len]
        .chunks_exact(4)
        .take(256)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Filter to finite values to get meaningful results.
    let finite: Vec<f32> = data.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return;
    }

    let op = op_from_idx(input.op_idx);

    // Invariant 1: Flat reduction must not panic.
    let result = reduce_f32(&finite, op);

    // Invariant 2: Result must be finite for finite inputs.
    assert!(
        result.is_finite(),
        "reduce_f32({op:?}) produced non-finite {result} from {} finite elements",
        finite.len()
    );

    // Invariant 3: Operation-specific bounds.
    match op {
        ReductionOp::Sum => {
            // Sum of positives must be non-negative if all inputs non-negative.
            if finite.iter().all(|&x| x >= 0.0) {
                assert!(result >= 0.0, "Sum of non-negatives is negative: {result}");
            }
        }
        ReductionOp::Max => {
            // Max must be >= every element.
            for &v in &finite {
                assert!(result >= v, "Max {result} < element {v}");
            }
        }
        ReductionOp::Min => {
            // Min must be <= every element.
            for &v in &finite {
                assert!(result <= v, "Min {result} > element {v}");
            }
        }
        ReductionOp::Mean => {
            // Mean must be between min and max.
            let lo = finite.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                result >= lo - 1e-5 && result <= hi + 1e-5,
                "Mean {result} outside [{lo}, {hi}]"
            );
        }
        ReductionOp::L2Norm => {
            // L2 norm must be non-negative.
            assert!(result >= 0.0, "L2 norm is negative: {result}");
        }
    }

    // Invariant 4: Row-wise reduction must not panic and output length == rows.
    let rows = ((input.rows as usize) % 16) + 1;
    let cols = finite.len() / rows;
    if cols > 0 {
        let matrix = &finite[..rows * cols];
        if let Ok(row_results) = reduce_rows_f32(matrix, rows, cols, op) {
            assert_eq!(row_results.len(), rows, "row reduction output length mismatch");
            for (i, &v) in row_results.iter().enumerate() {
                assert!(v.is_finite(), "row reduction [{i}] non-finite: {v} for op {op:?}");
            }
        }
    }

    // Invariant 5: Empty slice returns identity.
    let empty_result = reduce_f32(&[], op);
    match op {
        ReductionOp::Sum | ReductionOp::Mean | ReductionOp::L2Norm => {
            assert_eq!(empty_result, 0.0, "empty {op:?} should be 0.0");
        }
        ReductionOp::Max => {
            assert_eq!(empty_result, f32::NEG_INFINITY, "empty Max should be -inf");
        }
        ReductionOp::Min => {
            assert_eq!(empty_result, f32::INFINITY, "empty Min should be +inf");
        }
    }
});
