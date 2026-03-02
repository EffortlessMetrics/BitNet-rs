#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled, add_residual_with_dropout};
use libfuzzer_sys::fuzz_target;

/// Fuzz all residual connection operations with arbitrary inputs.
#[derive(Arbitrary, Debug)]
struct ResidualInput {
    /// Base output buffer (raw bytes → f32).
    raw_output: Vec<u8>,
    /// Residual buffer (raw bytes → f32).
    raw_residual: Vec<u8>,
    /// Dropout mask bits (one bool per byte).
    mask_bytes: Vec<u8>,
    /// Scale factor for scaled residual.
    scale: f32,
    /// Which operation to fuzz (mod 3).
    op: u8,
}

fn bytes_to_f32(raw: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (raw.len() / 4) * 4;
    raw[..aligned]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .take(max_elems)
        .collect()
}

fuzz_target!(|input: ResidualInput| {
    let output_vals = bytes_to_f32(&input.raw_output, 512);
    let residual_vals = bytes_to_f32(&input.raw_residual, 512);

    match input.op % 3 {
        0 => {
            // add_residual: output += residual
            let mut output = output_vals.clone();
            let len = output.len().min(residual_vals.len());
            if len == 0 {
                let _ = add_residual(&mut output, &residual_vals);
                return;
            }
            let mut out = output_vals[..len].to_vec();
            let res = &residual_vals[..len];
            let _ = add_residual(&mut out, res);

            // Also test mismatched lengths (should error).
            if output_vals.len() != residual_vals.len() {
                let mut out2 = output_vals.clone();
                assert!(add_residual(&mut out2, &residual_vals).is_err());
            }
        }
        1 => {
            // add_residual_scaled: output += scale * residual
            let len = output_vals.len().min(residual_vals.len());
            if len == 0 {
                return;
            }
            let mut out = output_vals[..len].to_vec();
            let res = &residual_vals[..len];
            let _ = add_residual_scaled(&mut out, res, input.scale);
        }
        _ => {
            // add_residual_with_dropout: conditional residual add
            let len = output_vals.len().min(residual_vals.len());
            if len == 0 {
                return;
            }
            let mask: Vec<bool> = input.mask_bytes.iter().take(len).map(|&b| b & 1 != 0).collect();
            if mask.len() != len {
                return;
            }
            let mut out = output_vals[..len].to_vec();
            let res = &residual_vals[..len];
            let _ = add_residual_with_dropout(&mut out, res, &mask);
        }
    }
});
