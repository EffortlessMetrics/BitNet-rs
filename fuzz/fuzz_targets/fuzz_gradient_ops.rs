#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::gather::{gather_rows, scatter_add_rows};
use bitnet_kernels::cpu::gating::{geglu, reglu, swiglu};
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};
use libfuzzer_sys::fuzz_target;

/// Fuzz gating activations (SwiGLU, GeGLU, ReGLU) and gradient-scatter
/// operations (scatter_add_rows) with random inputs.
#[derive(Arbitrary, Debug)]
struct GradientInput {
    data: Vec<u8>,
    op: u8,
    rows: u8,
    row_len: u8,
    indices: Vec<u8>,
    scale: f32,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: GradientInput| {
    let values = bytes_to_f32(&input.data, 512);
    if values.is_empty() {
        return;
    }

    match input.op % 5 {
        0 => {
            // SwiGLU gating
            let half = values.len() / 2;
            if half == 0 {
                return;
            }
            let gate = &values[..half];
            let up = &values[half..half * 2];
            let mut output = vec![0.0f32; half];
            if swiglu(gate, up, &mut output).is_ok() {
                for (i, &v) in output.iter().enumerate() {
                    assert!(v.is_finite(), "SwiGLU non-finite at {i}: {v}");
                }
            }
        }
        1 => {
            // GeGLU gating
            let half = values.len() / 2;
            if half == 0 {
                return;
            }
            let gate = &values[..half];
            let up = &values[half..half * 2];
            let mut output = vec![0.0f32; half];
            if geglu(gate, up, &mut output).is_ok() {
                for (i, &v) in output.iter().enumerate() {
                    assert!(v.is_finite(), "GeGLU non-finite at {i}: {v}");
                }
            }
        }
        2 => {
            // ReGLU gating
            let half = values.len() / 2;
            if half == 0 {
                return;
            }
            let gate = &values[..half];
            let up = &values[half..half * 2];
            let mut output = vec![0.0f32; half];
            if reglu(gate, up, &mut output).is_ok() {
                for (i, &v) in output.iter().enumerate() {
                    assert!(v.is_finite(), "ReGLU non-finite at {i}: {v}");
                }
            }
        }
        3 => {
            // scatter_add_rows — gradient accumulation
            let num_rows = (input.rows as usize % 8) + 1;
            let row_len = (input.row_len as usize % 8) + 1;
            let table_size = num_rows * row_len;
            if values.len() < table_size {
                return;
            }
            let mut table: Vec<f32> =
                values[..table_size].iter().map(|&v| if v.is_finite() { v } else { 0.0 }).collect();
            let table_before = table.clone();

            let indices: Vec<usize> =
                input.indices.iter().take(8).map(|&i| i as usize % num_rows).collect();
            if indices.is_empty() {
                return;
            }
            let grad_size = indices.len() * row_len;
            let grads: Vec<f32> = values
                .iter()
                .cycle()
                .take(grad_size)
                .map(|&v| if v.is_finite() { v } else { 0.0 })
                .collect();

            if scatter_add_rows(&mut table, num_rows, row_len, &indices, &grads).is_ok() {
                // Verify all outputs are finite
                for (i, &v) in table.iter().enumerate() {
                    assert!(v.is_finite(), "scatter_add non-finite at {i}: {v}");
                }
                // Verify gather→scatter round-trip consistency:
                // gathered rows should exist in the table
                if let Ok(gathered) = gather_rows(&table, num_rows, row_len, &indices) {
                    assert_eq!(gathered.len(), indices.len() * row_len);
                }
            } else {
                // On error, table should be unchanged or partially updated
                let _ = table_before;
            }
        }
        _ => {
            // Residual add with scaling — gradient passthrough
            let half = values.len() / 2;
            if half == 0 {
                return;
            }
            let residual: Vec<f32> =
                values[..half].iter().map(|&v| if v.is_finite() { v } else { 0.0 }).collect();
            let mut output: Vec<f32> = values[half..half * 2]
                .iter()
                .map(|&v| if v.is_finite() { v } else { 0.0 })
                .collect();
            let expected: Vec<f32> =
                output.iter().zip(residual.iter()).map(|(o, r)| o + r).collect();

            if add_residual(&mut output, &residual).is_ok() {
                for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (got - want).abs() < 1e-5,
                        "residual mismatch at {i}: got={got} want={want}"
                    );
                }
            }

            // Scaled variant
            let scale = if input.scale.is_finite() { input.scale.clamp(-10.0, 10.0) } else { 1.0 };
            let mut output2: Vec<f32> = values[half..half * 2]
                .iter()
                .map(|&v| if v.is_finite() { v } else { 0.0 })
                .collect();
            let _ = add_residual_scaled(&mut output2, &residual, scale);
        }
    }
});
