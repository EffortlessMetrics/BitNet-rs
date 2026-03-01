#![cfg(all(target_os = "macos", feature = "cpu"))]

//! Pure-logic tests validating Metal Shading Language patterns for neural network
//! inference on Apple Silicon. No GPU required — these test the calculations and
//! layouts used when setting up Metal shaders.

// ---------------------------------------------------------------------------
// MSL source helpers
// ---------------------------------------------------------------------------

/// Minimal check that an MSL kernel source contains required keywords.
fn msl_source_has_required_keywords(source: &str) -> Vec<&'static str> {
    let required = &["kernel", "device", "threadgroup"];
    required.iter().copied().filter(|kw| !source.contains(kw)).collect()
}

// ---------------------------------------------------------------------------
// Threadgroup / workgroup helpers
// ---------------------------------------------------------------------------

const METAL_MAX_THREADGROUP_MEMORY_BYTES: usize = 32 * 1024; // 32 KB
const METAL_MAX_THREADS_PER_THREADGROUP: u32 = 1024;
const METAL_SIMD_GROUP_SIZE: u32 = 32;

/// Compute shared-memory bytes needed for a float reduction inside one threadgroup.
fn threadgroup_reduction_memory(threads: u32, element_bytes: usize) -> usize {
    threads as usize * element_bytes
}

/// Round `n` up to the next multiple of `align`.
fn align_up(n: usize, align: usize) -> usize {
    assert!(align > 0);
    (n + align - 1) / align * align
}

// ---------------------------------------------------------------------------
// Buffer binding layout
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BufferBinding {
    index: u32,
    label: &'static str,
}

/// Standard inference pipeline bindings.
const INFERENCE_BINDINGS: &[BufferBinding] = &[
    BufferBinding { index: 0, label: "weights" },
    BufferBinding { index: 1, label: "input" },
    BufferBinding { index: 2, label: "output" },
    BufferBinding { index: 3, label: "config" },
];

// ---------------------------------------------------------------------------
// Texture format compatibility
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TextureFormat {
    name: &'static str,
    bytes_per_pixel: u32,
    channels: u32,
}

const NEURAL_TEXTURE_FORMATS: &[TextureFormat] = &[
    TextureFormat { name: "r16float", bytes_per_pixel: 2, channels: 1 },
    TextureFormat { name: "rg16float", bytes_per_pixel: 4, channels: 2 },
    TextureFormat { name: "rgba16float", bytes_per_pixel: 8, channels: 4 },
    TextureFormat { name: "r32float", bytes_per_pixel: 4, channels: 1 },
];

// ---------------------------------------------------------------------------
// Argument buffer encoding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ArgumentBufferEntry {
    index: u32,
    size_bytes: usize,
    #[allow(dead_code)]
    label: &'static str,
}

fn compute_argument_buffer_size(entries: &[ArgumentBufferEntry], alignment: usize) -> usize {
    entries.iter().map(|e| align_up(e.size_bytes, alignment)).sum()
}

// ---------------------------------------------------------------------------
// Dispatch validation
// ---------------------------------------------------------------------------

/// Apple Silicon dispatch dimension limits (per axis).
const METAL_MAX_DISPATCH_DIM: u32 = 65535;

fn validate_dispatch(grid: [u32; 3], threadgroup_size: [u32; 3]) -> Result<(), &'static str> {
    let total_threads: u32 = threadgroup_size
        .iter()
        .copied()
        .try_fold(1u32, |acc, v| acc.checked_mul(v))
        .ok_or("threadgroup size overflows u32")?;

    if total_threads > METAL_MAX_THREADS_PER_THREADGROUP {
        return Err("exceeds maxTotalThreadsPerThreadgroup");
    }

    for &dim in &grid {
        if dim == 0 {
            return Err("grid dimension is zero");
        }
        if dim > METAL_MAX_DISPATCH_DIM {
            return Err("grid dimension exceeds device limit");
        }
    }

    for &dim in &threadgroup_size {
        if dim == 0 {
            return Err("threadgroup dimension is zero");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Memory barrier patterns
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BarrierScope {
    Threadgroup,
    Device,
    SIMDGroup,
}

#[derive(Debug, Clone)]
struct BarrierPoint {
    scope: BarrierScope,
    after_write: bool,
    before_read: bool,
}

fn validate_barrier_pattern(barriers: &[BarrierPoint]) -> Result<(), &'static str> {
    for b in barriers {
        if b.before_read && !b.after_write {
            return Err("barrier before read without prior write is suspicious");
        }
    }
    // Ensure at least one threadgroup-scope barrier for shared-memory reductions.
    let has_tg = barriers.iter().any(|b| b.scope == BarrierScope::Threadgroup);
    if !has_tg {
        return Err("reduction pattern requires at least one threadgroup barrier");
    }
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_msl_source_compilation_syntax() {
        let source = r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void matmul(
                device const float* weights [[buffer(0)]],
                device const float* input   [[buffer(1)]],
                device float* output        [[buffer(2)]],
                threadgroup float* shared   [[threadgroup(0)]],
                uint tid [[thread_position_in_grid]]
            ) {
                // neural network matmul kernel
            }
        "#;

        let missing = msl_source_has_required_keywords(source);
        assert!(missing.is_empty(), "MSL source missing keywords: {missing:?}");

        // Negative: source without required keywords should report them.
        let bad_source = "void foo() {}";
        let missing_bad = msl_source_has_required_keywords(bad_source);
        assert_eq!(missing_bad.len(), 3);
    }

    #[test]
    fn test_msl_numeric_precision_constants() {
        // IEEE 754 half-precision (f16) range used in MSL.
        let f16_max: f32 = 65504.0;
        let f16_min_positive: f32 = 6.103_515_6e-5; // smallest normal f16
        let f16_epsilon: f32 = 9.765_625e-4; // f16 machine epsilon

        assert!(f16_max.is_finite());
        assert!(f16_min_positive > 0.0);
        assert!(f16_epsilon > 0.0 && f16_epsilon < 1.0);

        // inf / nan handling mirrors Metal semantics.
        let inf = f32::INFINITY;
        let nan = f32::NAN;
        assert!(inf.is_infinite());
        assert!(nan.is_nan());
        assert!(nan != nan); // NaN != NaN in IEEE 754

        // Softmax numerical stability: subtracting max prevents overflow.
        let logits = [1.0_f32, 2.0, 3.0, f16_max];
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for &l in &logits {
            let shifted = l - max_val;
            assert!(shifted <= 0.0, "shifted logit must be <= 0 for stability");
        }
    }

    #[test]
    fn test_msl_threadgroup_memory_calculations() {
        // f32 reduction: 256 threads × 4 bytes = 1024 bytes (fits in 32 KB).
        let mem_256 = threadgroup_reduction_memory(256, 4);
        assert_eq!(mem_256, 1024);
        assert!(mem_256 <= METAL_MAX_THREADGROUP_MEMORY_BYTES);

        // f16 reduction: 1024 threads × 2 bytes = 2048 bytes.
        let mem_1024_f16 = threadgroup_reduction_memory(1024, 2);
        assert_eq!(mem_1024_f16, 2048);
        assert!(mem_1024_f16 <= METAL_MAX_THREADGROUP_MEMORY_BYTES);

        // Maximum practical: 1024 threads × 32 bytes (float8) = 32 KB exactly.
        let mem_max = threadgroup_reduction_memory(1024, 32);
        assert_eq!(mem_max, METAL_MAX_THREADGROUP_MEMORY_BYTES);

        // Exceeding the limit: 1024 threads × 64 bytes.
        let mem_over = threadgroup_reduction_memory(1024, 64);
        assert!(mem_over > METAL_MAX_THREADGROUP_MEMORY_BYTES);
    }

    #[test]
    fn test_msl_buffer_binding_layout() {
        // Indices must be unique and sequential starting from 0.
        let mut seen = std::collections::HashSet::new();
        for (expected_idx, binding) in INFERENCE_BINDINGS.iter().enumerate() {
            assert_eq!(binding.index as usize, expected_idx);
            assert!(seen.insert(binding.index), "duplicate binding index {}", binding.index);
            assert!(!binding.label.is_empty());
        }

        assert_eq!(INFERENCE_BINDINGS.len(), 4);
        assert_eq!(INFERENCE_BINDINGS[0].label, "weights");
        assert_eq!(INFERENCE_BINDINGS[3].label, "config");
    }

    #[test]
    fn test_msl_workgroup_dimension_limits() {
        // 1-D: 1024 threads is the maximum.
        assert!(validate_dispatch([1024, 1, 1], [1024, 1, 1]).is_ok());

        // 2-D: 32 × 32 = 1024 threads (valid).
        assert!(validate_dispatch([64, 64, 1], [32, 32, 1]).is_ok());

        // 3-D: 8 × 8 × 16 = 1024 threads (valid).
        assert!(validate_dispatch([16, 16, 4], [8, 8, 16]).is_ok());

        // Exceeds limit: 32 × 32 × 2 = 2048 threads.
        assert!(validate_dispatch([64, 64, 1], [32, 32, 2]).is_err());

        // Exceeds limit: single axis > 1024.
        assert!(validate_dispatch([1, 1, 1], [2048, 1, 1]).is_err());
    }

    #[test]
    fn test_msl_simd_group_size() {
        // Apple GPU SIMD width is always 32.
        assert_eq!(METAL_SIMD_GROUP_SIZE, 32);

        // Number of SIMD groups in a threadgroup.
        let threadgroup_size = 256u32;
        let simd_groups = threadgroup_size / METAL_SIMD_GROUP_SIZE;
        assert_eq!(simd_groups, 8);

        // For a parallel reduction the number of SIMD groups should be a power of 2.
        assert!(simd_groups.is_power_of_two());

        // Shared memory for SIMD-group reduction (one element per SIMD group).
        let reduction_mem = simd_groups as usize * std::mem::size_of::<f32>();
        assert_eq!(reduction_mem, 32); // 8 × 4 bytes
        assert!(reduction_mem <= METAL_MAX_THREADGROUP_MEMORY_BYTES);
    }

    #[test]
    fn test_msl_texture_format_compatibility() {
        assert_eq!(NEURAL_TEXTURE_FORMATS.len(), 4);

        // r16float: single-channel half precision.
        let r16 = &NEURAL_TEXTURE_FORMATS[0];
        assert_eq!(r16.name, "r16float");
        assert_eq!(r16.bytes_per_pixel, 2);
        assert_eq!(r16.channels, 1);

        // rgba16float: 4-channel half precision (activations).
        let rgba16 = &NEURAL_TEXTURE_FORMATS[2];
        assert_eq!(rgba16.name, "rgba16float");
        assert_eq!(rgba16.bytes_per_pixel, 8);
        assert_eq!(rgba16.channels, 4);

        // r32float: single-channel full precision (accumulation).
        let r32 = &NEURAL_TEXTURE_FORMATS[3];
        assert_eq!(r32.name, "r32float");
        assert_eq!(r32.bytes_per_pixel, 4);
        assert_eq!(r32.channels, 1);

        // bytes_per_pixel == channels * per-channel size.
        for fmt in NEURAL_TEXTURE_FORMATS {
            let per_ch = if fmt.name.contains("16") { 2u32 } else { 4u32 };
            assert_eq!(fmt.bytes_per_pixel, fmt.channels * per_ch, "mismatch for {}", fmt.name);
        }
    }

    #[test]
    fn test_msl_argument_buffer_encoding() {
        let entries = vec![
            ArgumentBufferEntry { index: 0, size_bytes: 64, label: "weight_matrix" },
            ArgumentBufferEntry { index: 1, size_bytes: 32, label: "bias_vector" },
            ArgumentBufferEntry { index: 2, size_bytes: 16, label: "scale_factors" },
            ArgumentBufferEntry { index: 3, size_bytes: 8, label: "config_params" },
        ];

        // 8-byte alignment (Metal minimum for buffer pointers).
        let total_8 = compute_argument_buffer_size(&entries, 8);
        // 64 + 32 + 16 + 8 = 120 (all already aligned to 8).
        assert_eq!(total_8, 120);

        // 16-byte alignment.
        let total_16 = compute_argument_buffer_size(&entries, 16);
        // 64 + 32 + 16 + 16 = 128
        assert_eq!(total_16, 128);

        // Indices must be unique.
        let mut indices: Vec<u32> = entries.iter().map(|e| e.index).collect();
        indices.sort();
        indices.dedup();
        assert_eq!(indices.len(), entries.len());
    }

    #[test]
    fn test_msl_dispatch_validation() {
        // Valid dispatches.
        assert!(validate_dispatch([128, 128, 1], [256, 1, 1]).is_ok());
        assert!(validate_dispatch([1, 1, 1], [1, 1, 1]).is_ok());
        assert!(validate_dispatch([65535, 65535, 65535], [1, 1, 1]).is_ok());

        // Grid dimension zero is invalid.
        assert_eq!(validate_dispatch([0, 128, 1], [256, 1, 1]), Err("grid dimension is zero"));

        // Grid dimension exceeds device limit.
        assert_eq!(
            validate_dispatch([65536, 1, 1], [1, 1, 1]),
            Err("grid dimension exceeds device limit")
        );

        // Threadgroup dimension zero is invalid.
        assert_eq!(validate_dispatch([1, 1, 1], [0, 1, 1]), Err("threadgroup dimension is zero"));

        // Threadgroup total exceeds 1024.
        assert_eq!(
            validate_dispatch([1, 1, 1], [1025, 1, 1]),
            Err("exceeds maxTotalThreadsPerThreadgroup")
        );
    }

    #[test]
    fn test_msl_memory_barrier_patterns() {
        // Valid write-then-barrier-then-read pattern.
        let valid_pattern = vec![BarrierPoint {
            scope: BarrierScope::Threadgroup,
            after_write: true,
            before_read: true,
        }];
        assert!(validate_barrier_pattern(&valid_pattern).is_ok());

        // SIMD-group barrier for warp-level reduction + threadgroup barrier.
        let simd_pattern = vec![
            BarrierPoint { scope: BarrierScope::SIMDGroup, after_write: true, before_read: true },
            BarrierPoint { scope: BarrierScope::Threadgroup, after_write: true, before_read: true },
        ];
        assert!(validate_barrier_pattern(&simd_pattern).is_ok());

        // Missing threadgroup barrier is an error for reductions.
        let no_tg = vec![BarrierPoint {
            scope: BarrierScope::Device,
            after_write: true,
            before_read: true,
        }];
        assert_eq!(
            validate_barrier_pattern(&no_tg),
            Err("reduction pattern requires at least one threadgroup barrier")
        );

        // Read without prior write is suspicious.
        let suspicious = vec![BarrierPoint {
            scope: BarrierScope::Threadgroup,
            after_write: false,
            before_read: true,
        }];
        assert_eq!(
            validate_barrier_pattern(&suspicious),
            Err("barrier before read without prior write is suspicious")
        );
    }
}
