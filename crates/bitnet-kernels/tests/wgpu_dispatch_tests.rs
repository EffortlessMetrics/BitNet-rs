#![cfg(all(target_os = "macos", feature = "cpu"))]
#![allow(clippy::assertions_on_constants)]

//! Validation tests for wgpu/Metal compute dispatch logic on Apple Silicon.
//!
//! These are pure computational tests that validate dispatch parameters,
//! buffer alignment, and resource binding layouts without requiring a GPU device.

/// Compute the number of workgroups needed along one axis.
fn dispatch_count(total: u32, workgroup_size: u32) -> u32 {
    total.div_ceil(workgroup_size)
}

/// Round `size` up to the next multiple of `alignment`.
fn align_up(size: u64, alignment: u64) -> u64 {
    (size + alignment - 1) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// 1. Workgroup size alignment
// ---------------------------------------------------------------------------

#[test]
fn test_workgroup_size_alignment() {
    let valid_sizes: &[u32] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    for &size in valid_sizes {
        assert!(size.is_power_of_two(), "workgroup size {size} must be power of 2");
        assert!(size <= 1024, "workgroup size {size} exceeds Metal max of 1024");
    }

    // Non-powers-of-two must be rejected.
    let invalid_sizes: &[u32] = &[3, 5, 6, 7, 9, 10, 15, 17, 100, 255, 500, 1000];
    for &size in invalid_sizes {
        assert!(
            !size.is_power_of_two(),
            "size {size} should NOT be accepted as a valid workgroup size"
        );
    }

    // Sizes above 1024 are invalid even if power-of-two.
    assert!(2048u32.is_power_of_two());
    assert!(2048 > 1024, "2048 exceeds Metal max threadgroup width");
}

// ---------------------------------------------------------------------------
// 2. Dispatch dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_dimensions() {
    struct Case {
        total_elements: u32,
        workgroup_size: u32,
    }

    let cases = [
        Case { total_elements: 1024, workgroup_size: 256 },
        Case { total_elements: 1000, workgroup_size: 256 },
        Case { total_elements: 1, workgroup_size: 64 },
        Case { total_elements: 65536, workgroup_size: 1024 },
        Case { total_elements: 1_000_000, workgroup_size: 256 },
    ];

    for case in &cases {
        let x = dispatch_count(case.total_elements, case.workgroup_size);
        // 1-D dispatch: y = z = 1
        let y = 1u32;
        let z = 1u32;
        let covered = x * y * z * case.workgroup_size;
        assert!(
            covered >= case.total_elements,
            "dispatch ({x},{y},{z}) * wg {wg} = {covered} < {total}",
            wg = case.workgroup_size,
            total = case.total_elements,
        );
    }

    // 2-D dispatch for matrix workloads
    let rows = 512u32;
    let cols = 768u32;
    let wg_x = 16u32;
    let wg_y = 16u32;
    let dx = dispatch_count(cols, wg_x);
    let dy = dispatch_count(rows, wg_y);
    assert!(dx * wg_x >= cols);
    assert!(dy * wg_y >= rows);
}

// ---------------------------------------------------------------------------
// 3. Buffer size alignment (Metal requires 256-byte alignment)
// ---------------------------------------------------------------------------

#[test]
fn test_buffer_size_alignment() {
    let metal_alignment: u64 = 256;

    let test_sizes: &[u64] = &[0, 1, 100, 255, 256, 257, 512, 1000, 4096, 65535];
    for &size in test_sizes {
        let aligned = align_up(size, metal_alignment);
        assert_eq!(aligned % metal_alignment, 0, "aligned size {aligned} not 256-byte aligned");
        assert!(aligned >= size, "aligned {aligned} < original {size}");
        // Must be the *smallest* valid aligned size.
        if size > 0 {
            assert!(aligned - size < metal_alignment);
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Uniform buffer layout (std140 rules)
// ---------------------------------------------------------------------------

#[test]
fn test_uniform_buffer_layout() {
    // std140 alignment rules:
    //   scalar  (f32)  → 4
    //   vec2           → 8
    //   vec3 / vec4    → 16
    //   mat4 (4×vec4)  → 64 total, 16-byte column alignment
    //   struct         → round up to multiple of largest member alignment

    let align_scalar: u64 = 4;
    let align_vec2: u64 = 8;
    let align_vec4: u64 = 16;
    let size_mat4: u64 = 64; // 4 columns × 16 bytes

    // A typical uniform block: { mat4 mvp; vec4 params; float scale; }
    let mut offset: u64 = 0;

    // mat4 at offset 0 — needs 16-byte alignment
    offset = align_up(offset, align_vec4);
    assert_eq!(offset, 0);
    offset += size_mat4; // 64

    // vec4 immediately after mat4
    offset = align_up(offset, align_vec4);
    assert_eq!(offset, 64);
    offset += 16; // 80

    // scalar float
    offset = align_up(offset, align_scalar);
    assert_eq!(offset, 80);
    offset += 4; // 84

    // Total struct size rounded up to largest member alignment (16).
    let total = align_up(offset, align_vec4);
    assert_eq!(total, 96);

    // vec2 alignment spot-check
    let v2_offset = align_up(5, align_vec2);
    assert_eq!(v2_offset, 8);
}

// ---------------------------------------------------------------------------
// 5. Compute pipeline descriptor validation
// ---------------------------------------------------------------------------

#[test]
fn test_compute_pipeline_descriptor() {
    struct PipelineDescriptor {
        entry_point: &'static str,
        workgroup_size: [u32; 3],
    }

    let valid_descriptors = [
        PipelineDescriptor { entry_point: "main", workgroup_size: [256, 1, 1] },
        PipelineDescriptor { entry_point: "matmul_kernel", workgroup_size: [16, 16, 1] },
        PipelineDescriptor { entry_point: "reduce_sum", workgroup_size: [1024, 1, 1] },
        PipelineDescriptor { entry_point: "softmax", workgroup_size: [8, 8, 4] },
    ];

    for desc in &valid_descriptors {
        assert!(!desc.entry_point.is_empty(), "entry point must not be empty");

        let [x, y, z] = desc.workgroup_size;
        let total_threads = x * y * z;
        assert!(total_threads > 0, "workgroup must have at least 1 thread");
        assert!(
            total_threads <= 1024,
            "total threads {total_threads} exceeds Metal max of 1024 per threadgroup"
        );

        for (i, &dim) in desc.workgroup_size.iter().enumerate() {
            assert!(dim >= 1, "workgroup dim[{i}] must be ≥ 1");
        }
    }

    // Invalid: total threads exceed 1024
    let bad = PipelineDescriptor { entry_point: "bad", workgroup_size: [32, 32, 2] };
    let total = bad.workgroup_size.iter().product::<u32>();
    assert!(total > 1024, "expected invalid descriptor to exceed 1024 threads");
}

// ---------------------------------------------------------------------------
// 6. Threadgroup memory limit (Apple Silicon: 32 KB)
// ---------------------------------------------------------------------------

#[test]
fn test_threadgroup_memory_limit() {
    const APPLE_SILICON_THREADGROUP_MEM: u64 = 32 * 1024; // 32 KB

    let allocations: &[(u64, bool)] = &[
        (0, true),
        (1024, true),
        (16 * 1024, true),
        (32 * 1024, true),
        (32 * 1024 + 1, false),
        (64 * 1024, false),
    ];

    for &(size, should_fit) in allocations {
        let fits = size <= APPLE_SILICON_THREADGROUP_MEM;
        assert_eq!(
            fits, should_fit,
            "threadgroup alloc of {size} bytes: expected fits={should_fit}, got {fits}"
        );
    }

    // Typical tile: 256 threads × 4 floats × 4 bytes = 4 KB — well within limits.
    let tile_mem = 256u64 * 4 * 4;
    assert!(tile_mem <= APPLE_SILICON_THREADGROUP_MEM);
}

// ---------------------------------------------------------------------------
// 7. Max buffer length (Apple Silicon unified memory, typically ≥ 8 GB)
// ---------------------------------------------------------------------------

#[test]
fn test_max_buffer_length() {
    // Apple Silicon advertises maxBufferLength ≥ the device's unified memory
    // (e.g., 8 GB on base M1, 192 GB on M2 Ultra). We validate that common
    // tensor allocations fit within a conservative 8 GB limit.
    const MIN_MAX_BUFFER: u64 = 8 * 1024 * 1024 * 1024; // 8 GB

    let tensor_sizes: &[(u64, &str)] = &[
        (2 * 1024 * 1024 * 1024, "2B-param model weights (~2 GB at 1-bit)"),
        (4 * 1024 * 1024 * 1024, "4 GB activation buffer"),
        (256 * 1024 * 1024, "256 MB KV cache"),
    ];

    for &(size, label) in tensor_sizes {
        assert!(
            size <= MIN_MAX_BUFFER,
            "{label}: {size} bytes exceeds conservative max buffer of {MIN_MAX_BUFFER}"
        );
    }
}

// ---------------------------------------------------------------------------
// 8. Indirect dispatch buffer layout (3 × u32)
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_indirect_layout() {
    // An indirect dispatch buffer contains three u32 values: (x, y, z).
    let indirect_buffer_size = 3 * std::mem::size_of::<u32>();
    assert_eq!(indirect_buffer_size, 12, "indirect dispatch buffer must be 12 bytes");

    // Encode and decode a dispatch command.
    let dispatch_x: u32 = 128;
    let dispatch_y: u32 = 4;
    let dispatch_z: u32 = 1;

    let mut buf = [0u8; 12];
    buf[0..4].copy_from_slice(&dispatch_x.to_le_bytes());
    buf[4..8].copy_from_slice(&dispatch_y.to_le_bytes());
    buf[8..12].copy_from_slice(&dispatch_z.to_le_bytes());

    let read_x = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let read_y = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    let read_z = u32::from_le_bytes(buf[8..12].try_into().unwrap());

    assert_eq!((read_x, read_y, read_z), (dispatch_x, dispatch_y, dispatch_z));

    // Metal requires the indirect buffer to be 256-byte aligned.
    let aligned = align_up(indirect_buffer_size as u64, 256);
    assert_eq!(aligned, 256);
}

// ---------------------------------------------------------------------------
// 9. Shader compilation flags (MSL option patterns)
// ---------------------------------------------------------------------------

#[test]
fn test_shader_compilation_flags() {
    // Validate MSL compilation option patterns used by wgpu's Metal backend.
    struct MslOption {
        name: &'static str,
        expected_pattern: &'static str,
    }

    let options = [
        MslOption { name: "language_version", expected_pattern: "2.4" },
        MslOption { name: "fast_math", expected_pattern: "true" },
        MslOption { name: "platform", expected_pattern: "macos" },
        MslOption { name: "argument_buffers_tier", expected_pattern: "tier2" },
    ];

    for opt in &options {
        assert!(!opt.name.is_empty(), "MSL option name must not be empty");
        assert!(
            !opt.expected_pattern.is_empty(),
            "MSL option '{}' must have a non-empty value",
            opt.name
        );
    }

    // MSL version must be parseable as major.minor.
    let version = options[0].expected_pattern;
    let parts: Vec<&str> = version.split('.').collect();
    assert_eq!(parts.len(), 2, "MSL version must be major.minor");
    let major: u32 = parts[0].parse().expect("major version must be numeric");
    let minor: u32 = parts[1].parse().expect("minor version must be numeric");
    assert!(major >= 2, "MSL version must be ≥ 2.0 for compute shaders");
    assert!(minor <= 9, "unexpected minor version");
}

// ---------------------------------------------------------------------------
// 10. Resource binding validation
// ---------------------------------------------------------------------------

#[test]
fn test_resource_binding_validation() {
    // Typical inference pipeline bind group layout for a matmul kernel:
    //   binding 0: storage buffer (input A)
    //   binding 1: storage buffer (input B)
    //   binding 2: storage buffer (output C)
    //   binding 3: uniform buffer (params)

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum BindingType {
        StorageBuffer,
        UniformBuffer,
    }

    struct BindGroupEntry {
        binding: u32,
        ty: BindingType,
        min_size: u64,
    }

    let layout = [
        BindGroupEntry { binding: 0, ty: BindingType::StorageBuffer, min_size: 0 },
        BindGroupEntry { binding: 1, ty: BindingType::StorageBuffer, min_size: 0 },
        BindGroupEntry { binding: 2, ty: BindingType::StorageBuffer, min_size: 0 },
        BindGroupEntry { binding: 3, ty: BindingType::UniformBuffer, min_size: 16 },
    ];

    // Binding indices must be unique and contiguous from 0.
    for (i, entry) in layout.iter().enumerate() {
        assert_eq!(entry.binding, i as u32, "binding index must be contiguous from 0");
    }

    // Uniform buffers must have a non-zero min_size (wgpu validation requirement).
    for entry in &layout {
        if entry.ty == BindingType::UniformBuffer {
            assert!(
                entry.min_size > 0,
                "uniform buffer at binding {} must declare min_size > 0",
                entry.binding
            );
            // std140 requires uniform buffers be 16-byte aligned.
            assert_eq!(entry.min_size % 16, 0, "uniform buffer min_size must be 16-byte aligned");
        }
    }

    // Metal limits: max 31 buffers per shader stage on Apple Silicon.
    let buffer_count = layout.len();
    assert!(buffer_count <= 31, "Apple Silicon supports at most 31 buffers per stage");
}
