/// Flat I2_S dequantization: unpack 2-bit signed values to f32 and multiply by scale.
///
/// Bindings:
/// - `group(0) binding(0)` — packed `array<u32>` (16 values per u32)
/// - `group(0) binding(1)` — per-element scale `array<f32>` (one per 16-element group)
/// - `group(0) binding(2)` — output `array<f32>`
/// - `group(0) binding(3)` — `Params { n: u32 }` total number of output elements
///
/// 2-bit mapping: `0b00 → 0.0, 0b01 → 1.0, 0b10 → -1.0, 0b11 → 0.0`
pub const DEQUANT_I2S_SRC: &str = r#"
// I2_S flat dequantization kernel
// Each u32 packs 16 two-bit signed values.
// Mapping: 0b00 → 0.0, 0b01 → 1.0, 0b10 → −1.0, 0b11 → 0.0

struct Params {
    n: u32,  // total output element count
}

@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const LUT: array<f32, 4> = array<f32, 4>(0.0, 1.0, -1.0, 0.0);

@compute @workgroup_size(256, 1, 1)
fn dequant_i2s(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pack_idx = gid.x;
    let base_out = pack_idx * 16u;
    if base_out >= params.n {
        return;
    }
    let word = packed[pack_idx];
    let scale = scales[pack_idx];
    for (var i = 0u; i < 16u; i = i + 1u) {
        let out_idx = base_out + i;
        if out_idx >= params.n {
            return;
        }
        let bits = (word >> (i * 2u)) & 3u;
        output[out_idx] = LUT[bits] * scale;
    }
}
"#;

/// Block-wise I2_S dequantization for BitNet32-F16 format (block size 32).
///
/// Bindings:
/// - `group(0) binding(0)` — packed `array<u32>` (16 values per u32, 2 u32s per block)
/// - `group(0) binding(1)` — per-block scale `array<f32>`
/// - `group(0) binding(2)` — output `array<f32>`
/// - `group(0) binding(3)` — `Params { n_blocks: u32 }`
pub const DEQUANT_I2S_BLOCK_SRC: &str = r#"
// I2_S block dequantization — BitNet32-F16 format (32-element blocks)
// Each block: 2 packed u32 words + 1 f32 scale

struct Params {
    n_blocks: u32,
}

@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 32u;
const WORDS_PER_BLOCK: u32 = 2u;  // 32 values / 16 values-per-u32
const LUT: array<f32, 4> = array<f32, 4>(0.0, 1.0, -1.0, 0.0);

@compute @workgroup_size(256, 1, 1)
fn dequant_i2s_block(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_idx = gid.x;
    if block_idx >= params.n_blocks {
        return;
    }
    let scale = scales[block_idx];
    let pack_base = block_idx * WORDS_PER_BLOCK;
    let out_base = block_idx * BLOCK_SIZE;

    for (var w = 0u; w < WORDS_PER_BLOCK; w = w + 1u) {
        let word = packed[pack_base + w];
        for (var i = 0u; i < 16u; i = i + 1u) {
            let bits = (word >> (i * 2u)) & 3u;
            output[out_base + w * 16u + i] = LUT[bits] * scale;
        }
    }
}
"#;

/// QK256 dequantization: 256-element blocks with per-block scales.
///
/// Bindings:
/// - `group(0) binding(0)` — packed `array<u32>` (16 u32s per block = 256 values)
/// - `group(0) binding(1)` — per-block scale `array<f32>`
/// - `group(0) binding(2)` — output `array<f32>`
/// - `group(0) binding(3)` — `Params { n_blocks: u32 }`
pub const DEQUANT_QK256_SRC: &str = r#"
// QK256 dequantization — 256-element blocks
// Each block: 16 packed u32 words + 1 f32 scale

struct Params {
    n_blocks: u32,
}

@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const QK: u32 = 256u;
const WORDS_PER_BLOCK: u32 = 16u;  // 256 / 16
const LUT: array<f32, 4> = array<f32, 4>(0.0, 1.0, -1.0, 0.0);

@compute @workgroup_size(256, 1, 1)
fn dequant_qk256(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_idx = gid.x;
    if block_idx >= params.n_blocks {
        return;
    }
    let scale = scales[block_idx];
    let pack_base = block_idx * WORDS_PER_BLOCK;
    let out_base = block_idx * QK;

    for (var w = 0u; w < WORDS_PER_BLOCK; w = w + 1u) {
        let word = packed[pack_base + w];
        for (var i = 0u; i < 16u; i = i + 1u) {
            let bits = (word >> (i * 2u)) & 3u;
            output[out_base + w * 16u + i] = LUT[bits] * scale;
        }
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
    fn test_dequant_i2s_validates() {
        validate_wgsl(DEQUANT_I2S_SRC).expect("DEQUANT_I2S_SRC should be valid WGSL");
    }

    #[test]
    fn test_dequant_i2s_block_validates() {
        validate_wgsl(DEQUANT_I2S_BLOCK_SRC).expect("DEQUANT_I2S_BLOCK_SRC should be valid WGSL");
    }

    #[test]
    fn test_dequant_qk256_validates() {
        validate_wgsl(DEQUANT_QK256_SRC).expect("DEQUANT_QK256_SRC should be valid WGSL");
    }

    #[test]
    fn test_dequant_i2s_has_correct_entry_point() {
        let module = naga::front::wgsl::parse_str(DEQUANT_I2S_SRC).unwrap();
        let entry = module.entry_points.iter().find(|ep| ep.name == "dequant_i2s");
        assert!(entry.is_some(), "should have dequant_i2s entry point");
        assert_eq!(entry.unwrap().stage, naga::ShaderStage::Compute);
    }

    #[test]
    fn test_dequant_qk256_has_correct_entry_point() {
        let module = naga::front::wgsl::parse_str(DEQUANT_QK256_SRC).unwrap();
        let entry = module.entry_points.iter().find(|ep| ep.name == "dequant_qk256");
        assert!(entry.is_some(), "should have dequant_qk256 entry point");
    }
}
