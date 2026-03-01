/// I2_S quantized matrix-vector multiply (the critical BitNet inference kernel).
///
/// Reads packed 2-bit weights, dequantizes on-the-fly, and accumulates the dot product.
/// One workgroup computes one output row.
///
/// Bindings:
/// - `group(0) binding(0)` — packed weight matrix `array<u32>` (row-major, 16 values per u32)
/// - `group(0) binding(1)` — per-row scale `array<f32>`
/// - `group(0) binding(2)` — input vector `array<f32>`
/// - `group(0) binding(3)` — output vector `array<f32>`
/// - `group(0) binding(4)` — `Params { n_cols: u32, n_packed_per_row: u32 }`
pub const I2S_MATVEC_SRC: &str = r#"
// I2_S matrix-vector multiply
// One workgroup per output row; threads cooperatively reduce the dot product.

struct Params {
    n_cols: u32,           // number of columns (input vector length)
    n_packed_per_row: u32, // ceil(n_cols / 16)
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> input_vec: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_vec: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;
const LUT: array<f32, 4> = array<f32, 4>(0.0, 1.0, -1.0, 0.0);

var<workgroup> shared_sums: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn i2s_matvec(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let row = wg_id.x;
    let tid = lid.x;
    let row_offset = row * params.n_packed_per_row;

    // Each thread accumulates partial dot product over a strided subset of packed words.
    var acc = 0.0;
    var pack_idx = tid;
    loop {
        if pack_idx >= params.n_packed_per_row {
            break;
        }
        let word = weights[row_offset + pack_idx];
        let col_base = pack_idx * 16u;
        for (var i = 0u; i < 16u; i = i + 1u) {
            let col = col_base + i;
            if col >= params.n_cols {
                break;
            }
            let bits = (word >> (i * 2u)) & 3u;
            let w = LUT[bits];
            acc = acc + w * input_vec[col];
        }
        pack_idx = pack_idx + WG_SIZE;
    }

    // Workgroup reduction
    shared_sums[tid] = acc;
    workgroupBarrier();

    var stride = WG_SIZE / 2u;
    loop {
        if stride == 0u {
            break;
        }
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if tid == 0u {
        output_vec[row] = shared_sums[0] * scales[row];
    }
}
"#;

/// Tiled I2_S matrix-vector multiply with shared-memory input caching.
///
/// Loads input-vector tiles into workgroup shared memory to reduce global reads.
/// One workgroup per output row, processes input in tiles of 4096 elements.
///
/// Bindings: same as `I2S_MATVEC_SRC`.
pub const I2S_MATVEC_TILED_SRC: &str = r#"
// Tiled I2_S matrix-vector multiply with shared-memory input caching.
// Tiles the input vector through workgroup shared memory for reduced bandwidth.

struct Params {
    n_cols: u32,
    n_packed_per_row: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> input_vec: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_vec: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;
const TILE_ELEMS: u32 = 4096u;  // elements per tile
const TILE_PACKS: u32 = 256u;   // 4096 / 16 packed words per tile
const LUT: array<f32, 4> = array<f32, 4>(0.0, 1.0, -1.0, 0.0);

var<workgroup> shared_input: array<f32, 4096>;
var<workgroup> shared_sums: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn i2s_matvec_tiled(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let row = wg_id.x;
    let tid = lid.x;
    let row_offset = row * params.n_packed_per_row;

    var acc = 0.0;
    var tile_start = 0u;

    // Process input in tiles
    loop {
        if tile_start >= params.n_cols {
            break;
        }
        // Cooperatively load tile of input vector into shared memory
        var load_idx = tid;
        loop {
            if load_idx >= TILE_ELEMS {
                break;
            }
            let global_idx = tile_start + load_idx;
            if global_idx < params.n_cols {
                shared_input[load_idx] = input_vec[global_idx];
            } else {
                shared_input[load_idx] = 0.0;
            }
            load_idx = load_idx + WG_SIZE;
        }
        workgroupBarrier();

        // Compute partial dot product for this tile
        let tile_pack_start = tile_start / 16u;
        let tile_pack_end = min(tile_pack_start + TILE_PACKS, params.n_packed_per_row);
        var pack_idx = tile_pack_start + tid;
        loop {
            if pack_idx >= tile_pack_end {
                break;
            }
            let word = weights[row_offset + pack_idx];
            let col_base = pack_idx * 16u;
            for (var i = 0u; i < 16u; i = i + 1u) {
                let col = col_base + i;
                if col >= params.n_cols {
                    break;
                }
                let local_idx = col - tile_start;
                let bits = (word >> (i * 2u)) & 3u;
                let w = LUT[bits];
                acc = acc + w * shared_input[local_idx];
            }
            pack_idx = pack_idx + WG_SIZE;
        }
        workgroupBarrier();

        tile_start = tile_start + TILE_ELEMS;
    }

    // Workgroup reduction
    shared_sums[tid] = acc;
    workgroupBarrier();

    var stride = WG_SIZE / 2u;
    loop {
        if stride == 0u {
            break;
        }
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if tid == 0u {
        output_vec[row] = shared_sums[0] * scales[row];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn validate_wgsl(source: &str) -> Result<(), String> {
        let module = naga::front::wgsl::parse_str(source).map_err(|e| format!("{e}"))?;
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module).map_err(|e| format!("{e}"))?;
        Ok(())
    }

    #[test]
    fn test_i2s_matvec_validates() {
        validate_wgsl(I2S_MATVEC_SRC).expect("I2S_MATVEC_SRC should be valid WGSL");
    }

    #[test]
    fn test_i2s_matvec_tiled_validates() {
        validate_wgsl(I2S_MATVEC_TILED_SRC).expect("I2S_MATVEC_TILED_SRC should be valid WGSL");
    }

    #[test]
    fn test_i2s_matvec_has_correct_entry_point() {
        let module = naga::front::wgsl::parse_str(I2S_MATVEC_SRC).unwrap();
        let entry = module.entry_points.iter().find(|ep| ep.name == "i2s_matvec");
        assert!(entry.is_some(), "should have i2s_matvec entry point");
        assert_eq!(entry.unwrap().stage, naga::ShaderStage::Compute);
    }

    #[test]
    fn test_i2s_matvec_tiled_has_correct_entry_point() {
        let module = naga::front::wgsl::parse_str(I2S_MATVEC_TILED_SRC).unwrap();
        let entry = module.entry_points.iter().find(|ep| ep.name == "i2s_matvec_tiled");
        assert!(entry.is_some(), "should have i2s_matvec_tiled entry point");
    }
}
