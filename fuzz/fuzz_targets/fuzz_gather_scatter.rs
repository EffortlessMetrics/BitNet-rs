#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::gather::{gather_rows, index_select_dim, scatter_add_rows};
use libfuzzer_sys::fuzz_target;

/// Fuzz gather/scatter operations with arbitrary table sizes and indices.
#[derive(Arbitrary, Debug)]
struct GatherScatterInput {
    /// Operation selector (mod 3): 0=gather, 1=scatter_add, 2=index_select.
    op: u8,
    /// Number of rows in table.
    num_rows: u8,
    /// Row length.
    row_len: u8,
    /// Indices for lookup/scatter.
    indices: Vec<u8>,
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

fuzz_target!(|input: GatherScatterInput| {
    let num_rows = (input.num_rows as usize % 16) + 1;
    let row_len = (input.row_len as usize % 16) + 1;
    let table_len = num_rows * row_len;

    // Cap indices count.
    let idx_count = input.indices.len().min(32);
    let indices: Vec<usize> =
        input.indices[..idx_count].iter().map(|&i| i as usize % num_rows).collect();

    match input.op % 3 {
        0 => {
            // gather_rows: embedding lookup
            let table = bytes_to_f32(&input.raw_data, table_len);
            match gather_rows(&table, num_rows, row_len, &indices) {
                Ok(result) => {
                    assert_eq!(result.len(), indices.len() * row_len);
                }
                Err(_) => {}
            }
        }
        1 => {
            // scatter_add_rows: accumulate gradients
            let mut table = bytes_to_f32(&input.raw_data, table_len);
            let values_len = indices.len() * row_len;
            let values = bytes_to_f32(
                if input.raw_data.len() > table_len * 4 {
                    &input.raw_data[table_len * 4..]
                } else {
                    &input.raw_data
                },
                values_len,
            );
            let _ = scatter_add_rows(&mut table, num_rows, row_len, &indices, &values);
        }
        _ => {
            // index_select_dim: select along a dimension
            let outer = (input.num_rows as usize % 4) + 1;
            let dim_size = (input.row_len as usize % 8) + 1;
            let inner = 1_usize.max((input.indices.len() % 4) + 1);
            let data_len = outer * dim_size * inner;
            let data = bytes_to_f32(&input.raw_data, data_len);
            let sel_indices: Vec<usize> =
                input.indices.iter().take(8).map(|&i| i as usize % dim_size).collect();
            let _ = index_select_dim(&data, outer, dim_size, inner, &sel_indices);
        }
    }
});
