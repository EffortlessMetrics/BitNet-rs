/// Absmax quantization: f32 → 2-bit signed using absmax scaling.
///
/// Finds the absolute maximum of each block, computes `scale = absmax`,
/// then quantizes: `round(x / scale)` clamped to {-1, 0, 1} and packed as 2-bit.
///
/// Bindings:
/// - `group(0) binding(0)` — input `array<f32>`
/// - `group(0) binding(1)` — output packed `array<u32>` (16 values per u32)
/// - `group(0) binding(2)` — output scales `array<f32>` (one per 16-element group)
/// - `group(0) binding(3)` — `Params { n: u32 }` total element count
///
/// Quantized mapping: `-1 → 0b10, 0 → 0b00, 1 → 0b01`
pub const QUANT_ABSMAX_SRC: &str = r#"
// Absmax quantization: f32 → 2-bit signed
// Each thread processes one group of 16 elements.
// Finds absmax, then packs quantized values into a u32.

struct Params {
    n: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn quant_absmax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let group_idx = gid.x;
    let base = group_idx * 16u;
    if base >= params.n {
        return;
    }

    // Find absmax over the group
    var amax = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let idx = base + i;
        if idx < params.n {
            amax = max(amax, abs(input[idx]));
        }
    }

    scales[group_idx] = amax;

    // Quantize and pack
    var word = 0u;
    let inv_scale = select(1.0 / amax, 0.0, amax == 0.0);
    for (var i = 0u; i < 16u; i = i + 1u) {
        let idx = base + i;
        if idx < params.n {
            let val = input[idx] * inv_scale;
            // Round to nearest integer and clamp to {-1, 0, 1}
            let rounded = i32(round(val));
            let clamped = clamp(rounded, -1, 1);
            // Encode: -1 → 0b10, 0 → 0b00, 1 → 0b01
            var bits = 0u;
            if clamped == 1 {
                bits = 1u;
            } else if clamped == -1 {
                bits = 2u;
            }
            word = word | (bits << (i * 2u));
        }
    }
    packed[group_idx] = word;
}
"#;

/// Per-block scale computation for pre-pass quantization.
///
/// Computes the absmax scale for each block of `block_size` elements.
/// Useful as a separate pass when scale computation is decoupled from packing.
///
/// Bindings:
/// - `group(0) binding(0)` — input `array<f32>`
/// - `group(0) binding(1)` — output scales `array<f32>` (one per block)
/// - `group(0) binding(2)` — `Params { n: u32, block_size: u32 }`
pub const QUANT_SCALE_SRC: &str = r#"
// Per-block scale computation (absmax)
// Each thread computes the scale for one block.

struct Params {
    n: u32,
    block_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> scales: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn quant_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_idx = gid.x;
    let base = block_idx * params.block_size;
    if base >= params.n {
        return;
    }

    var amax = 0.0;
    let end = min(base + params.block_size, params.n);
    var idx = base;
    loop {
        if idx >= end {
            break;
        }
        amax = max(amax, abs(input[idx]));
        idx = idx + 1u;
    }

    scales[block_idx] = amax;
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
    fn test_quant_absmax_validates() {
        validate_wgsl(QUANT_ABSMAX_SRC).expect("QUANT_ABSMAX_SRC should be valid WGSL");
    }

    #[test]
    fn test_quant_scale_validates() {
        validate_wgsl(QUANT_SCALE_SRC).expect("QUANT_SCALE_SRC should be valid WGSL");
    }

    #[test]
    fn test_quant_absmax_has_correct_entry_point() {
        let module = naga::front::wgsl::parse_str(QUANT_ABSMAX_SRC).unwrap();
        let entry = module.entry_points.iter().find(|ep| ep.name == "quant_absmax");
        assert!(entry.is_some(), "should have quant_absmax entry point");
        assert_eq!(entry.unwrap().stage, naga::ShaderStage::Compute);
    }
}
