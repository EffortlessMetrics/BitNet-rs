#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::concat::ConcatKernel;
use libfuzzer_sys::fuzz_target;

/// Fuzz concat, split, and stack operations with arbitrary shapes.
#[derive(Arbitrary, Debug)]
struct ConcatInput {
    /// Operation selector (mod 3): 0=concat, 1=split, 2=stack.
    op: u8,
    /// Axis for the operation.
    axis: u8,
    /// Number of inputs/splits (clamped).
    count: u8,
    /// Dimension sizes for a base shape.
    dim0: u8,
    dim1: u8,
    dim2: u8,
    /// Raw data bytes.
    raw_data: Vec<u8>,
}

fn bytes_to_f32(raw: &[u8], count: usize) -> Vec<f32> {
    let aligned = (raw.len() / 4) * 4;
    let mut out: Vec<f32> = raw[..aligned]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    out.resize(count, 0.0);
    out.truncate(count);
    out
}

fuzz_target!(|input: ConcatInput| {
    let d0 = (input.dim0 as usize % 8) + 1;
    let d1 = (input.dim1 as usize % 8) + 1;
    let count = (input.count as usize % 4) + 1;
    let axis = input.axis as usize;

    match input.op % 3 {
        0 => {
            // Concat along axis for 2-D shapes.
            if axis >= 2 {
                return;
            }
            let numel = d0 * d1;
            let mut data_vecs: Vec<Vec<f32>> = Vec::new();
            let mut shapes_owned: Vec<Vec<usize>> = Vec::new();

            for _ in 0..count {
                let data = bytes_to_f32(&input.raw_data, numel);
                data_vecs.push(data);
                shapes_owned.push(vec![d0, d1]);
            }

            let input_refs: Vec<&[f32]> = data_vecs.iter().map(|v| v.as_slice()).collect();
            let shape_refs: Vec<&[usize]> = shapes_owned.iter().map(|v| v.as_slice()).collect();

            let _ = ConcatKernel::concat(&input_refs, &shape_refs, axis);
        }
        1 => {
            // Split a tensor.
            let numel = d0 * d1;
            let data = bytes_to_f32(&input.raw_data, numel);
            let shape = vec![d0, d1];

            if axis >= 2 {
                return;
            }

            let dim_at_axis = if axis == 0 { d0 } else { d1 };
            if count > dim_at_axis || count == 0 {
                return;
            }

            let _ = ConcatKernel::split(&data, &shape, axis, count);
        }
        _ => {
            // Stack tensors along a new axis.
            let numel = d0 * d1;
            let mut data_vecs = Vec::new();

            for _ in 0..count {
                data_vecs.push(bytes_to_f32(&input.raw_data, numel));
            }

            let input_refs: Vec<&[f32]> = data_vecs.iter().map(|v| v.as_slice()).collect();
            let shape = vec![d0, d1];

            // Stack axis can be 0..=ndim.
            if axis > 2 {
                return;
            }
            let _ = ConcatKernel::stack(&input_refs, &shape, axis);
        }
    }
});
