#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ReductionInput {
    data: Vec<u8>,
    shape: Vec<u8>,
    reduce_axis: u8,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn reduce_sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

fn reduce_max(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn reduce_min(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::INFINITY, f32::min)
}

fn reduce_mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f32>() / data.len() as f32
}

fn reduce_along_axis(
    data: &[f32],
    rows: usize,
    cols: usize,
    axis: usize,
    op: fn(&[f32]) -> f32,
) -> Vec<f32> {
    if axis == 0 {
        // Reduce across rows, output has `cols` elements
        (0..cols)
            .map(|c| {
                let col_vals: Vec<f32> = (0..rows).map(|r| data[r * cols + c]).collect();
                op(&col_vals)
            })
            .collect()
    } else {
        // Reduce across cols, output has `rows` elements
        (0..rows).map(|r| op(&data[r * cols..(r + 1) * cols])).collect()
    }
}

fuzz_target!(|input: ReductionInput| {
    let values = bytes_to_f32(&input.data, 256);
    if values.is_empty() {
        return;
    }

    // Skip non-finite inputs
    if values.iter().any(|x| !x.is_finite()) {
        return;
    }

    // --- Global reductions ---
    let sum = reduce_sum(&values);
    let max = reduce_max(&values);
    let min = reduce_min(&values);
    let mean = reduce_mean(&values);

    // Invariant 1: All global reductions are finite
    assert!(sum.is_finite(), "sum is not finite: {sum}");
    assert!(max.is_finite(), "max is not finite: {max}");
    assert!(min.is_finite(), "min is not finite: {min}");
    assert!(mean.is_finite(), "mean is not finite: {mean}");

    // Invariant 2: min <= mean <= max (for finite values)
    assert!(min <= max, "min ({min}) should be <= max ({max})");

    // mean is between min and max
    assert!(
        mean >= min - 1e-5 && mean <= max + 1e-5,
        "mean ({mean}) should be between min ({min}) and max ({max})"
    );

    // Invariant 3: For all-positive values, max <= sum
    let all_positive = values.iter().all(|&x| x >= 0.0);
    if all_positive {
        assert!(max <= sum + 1e-5, "for positive values, max ({max}) should be <= sum ({sum})");
        // Also: min * len <= sum
        let expected_min_sum = min * values.len() as f32;
        assert!(
            sum >= expected_min_sum - 1e-3,
            "sum ({sum}) should be >= min*len ({expected_min_sum})"
        );
    }

    // Invariant 4: mean * len ≈ sum
    let expected_sum = mean * values.len() as f32;
    assert!(
        (expected_sum - sum).abs() < 1e-1 + sum.abs() * 1e-5,
        "mean*len ({expected_sum}) should ≈ sum ({sum})"
    );

    // Invariant 5: max and min are actual values in the array
    assert!(values.iter().any(|&x| (x - max).abs() < 1e-10), "max ({max}) not found in values");
    assert!(values.iter().any(|&x| (x - min).abs() < 1e-10), "min ({min}) not found in values");

    // --- Axis reductions (2D) ---
    let ndims = input.shape.iter().take(2).collect::<Vec<_>>();
    if ndims.len() >= 2 {
        let rows = (*ndims[0] as usize % 16) + 1;
        let cols = (*ndims[1] as usize % 16) + 1;
        let total = rows * cols;

        if values.len() >= total {
            let mat = &values[..total];
            let axis = (input.reduce_axis as usize) % 2;

            let sum_reduced = reduce_along_axis(mat, rows, cols, axis, reduce_sum);
            let max_reduced = reduce_along_axis(mat, rows, cols, axis, reduce_max);
            let min_reduced = reduce_along_axis(mat, rows, cols, axis, reduce_min);

            // Invariant 6: Axis reduction output has correct shape
            let expected_len = if axis == 0 { cols } else { rows };
            assert_eq!(
                sum_reduced.len(),
                expected_len,
                "axis-{axis} sum reduction shape: expected {expected_len}, got {}",
                sum_reduced.len()
            );
            assert_eq!(max_reduced.len(), expected_len);
            assert_eq!(min_reduced.len(), expected_len);

            // Invariant 7: Axis-reduced values are finite
            for (i, (&s, (&mx, &mn))) in
                sum_reduced.iter().zip(max_reduced.iter().zip(min_reduced.iter())).enumerate()
            {
                assert!(s.is_finite(), "axis sum non-finite at {i}");
                assert!(mx.is_finite(), "axis max non-finite at {i}");
                assert!(mn.is_finite(), "axis min non-finite at {i}");
                assert!(mn <= mx, "axis min ({mn}) > max ({mx}) at index {i}");
            }

            // Invariant 8: Sum of axis sums ≈ global sum of the submatrix
            let mat_sum: f32 = mat.iter().sum();
            let axis_sum_total: f32 = sum_reduced.iter().sum();
            assert!(
                (mat_sum - axis_sum_total).abs() < 1e-2 + mat_sum.abs() * 1e-4,
                "axis sum total ({axis_sum_total}) should ≈ matrix sum ({mat_sum})"
            );
        }
    }

    // Invariant 9: Single-element reduction is identity
    let single = &values[..1];
    assert_eq!(reduce_sum(single), values[0]);
    assert_eq!(reduce_max(single), values[0]);
    assert_eq!(reduce_min(single), values[0]);
    assert_eq!(reduce_mean(single), values[0]);
});
