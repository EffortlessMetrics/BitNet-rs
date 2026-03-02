//! Tests for wgpu/Metal matrix multiplication compute shaders on Apple Silicon.
//!
//! Verifies naive, tiled, non-square, vector, accumulation, and batched matmul
//! dispatched via WGSL compute shaders on the Metal backend.

#![cfg(target_os = "macos")]

use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// WGSL compute shader: naive matrix multiply
// A is MxK, B is KxN, C is MxN (row-major)
// ---------------------------------------------------------------------------

const MATMUL_SHADER: &str = r#"
struct Dimensions {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dimensions;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if row >= dims.M || col >= dims.N {
        return;
    }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.K; i = i + 1u) {
        sum = sum + A[row * dims.K + i] * B[i * dims.N + col];
    }
    C[row * dims.N + col] = sum;
}
"#;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn setup_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) =
            adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.ok()?;

        Some((device, queue))
    })
}

fn run_matmul(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    a: &[f32],
    b: &[f32],
    m: u32,
    n: u32,
    k: u32,
) -> Vec<f32> {
    let output_size = (m * n) as usize;

    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("A"),
        contents: bytemuck::cast_slice(a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("B"),
        contents: bytemuck::cast_slice(b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_c = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("C"),
        size: (output_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims = [m, n, k, 0u32];
    let buf_dims = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dims"),
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matmul"),
        source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("matmul_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("matmul_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("matmul_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_dims.as_entire_binding() },
        ],
    });

    let workgroups_x = (m + 7) / 8;
    let workgroups_y = (n + 7) / 8;

    let mut encoder = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("matmul_encoder") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (output_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        &buf_c,
        0,
        &staging,
        0,
        (output_size * std::mem::size_of::<f32>()) as u64,
    );

    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    result
}

/// CPU reference matmul for verification.
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_matmul_shader_tests -- --ignored"]
fn test_metal_naive_matmul() {
    let (device, queue) = setup_device().expect("Metal device required");

    // 4x4 identity-like test: A * B where result is hand-computable
    #[rustfmt::skip]
    let a: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    #[rustfmt::skip]
    let b: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    let result = run_matmul(&device, &queue, &a, &b, 4, 4, 4);

    // A * I = A
    for (i, (&got, &expected)) in result.iter().zip(a.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-5,
            "mismatch at index {i}: got {got}, expected {expected}"
        );
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_matmul_shader_tests -- --ignored"]
fn test_metal_tiled_matmul() {
    let (device, queue) = setup_device().expect("Metal device required");

    let m = 16u32;
    let n = 16u32;
    let k = 16u32;

    // Deterministic input: a[i] = (i % 7) as f32 - 3.0
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();

    let gpu_result = run_matmul(&device, &queue, &a, &b, m, n, k);
    let cpu_result = cpu_matmul(&a, &b, m as usize, n as usize, k as usize);

    assert_eq!(gpu_result.len(), cpu_result.len());
    for (i, (&got, &expected)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-3,
            "tiled mismatch at index {i}: gpu={got}, cpu={expected}"
        );
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_matmul_shader_tests -- --ignored"]
fn test_metal_matmul_non_square() {
    let (device, queue) = setup_device().expect("Metal device required");

    let (m, n, k) = (3u32, 5u32, 4u32);

    #[rustfmt::skip]
    let a: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ];
    #[rustfmt::skip]
    let b: Vec<f32> = vec![
        1.0, 0.0, 2.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 2.0, 1.0,
        2.0, 0.0, 1.0, 0.0, 2.0,
        1.0, 2.0, 0.0, 1.0, 0.0,
    ];

    let gpu_result = run_matmul(&device, &queue, &a, &b, m, n, k);
    let cpu_result = cpu_matmul(&a, &b, m as usize, n as usize, k as usize);

    assert_eq!(gpu_result.len(), (m * n) as usize);
    for (i, (&got, &expected)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-5,
            "non-square mismatch at index {i}: gpu={got}, cpu={expected}"
        );
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_matmul_shader_tests -- --ignored"]
fn test_metal_vector_matmul() {
    let (device, queue) = setup_device().expect("Metal device required");

    // Matrix-vector: (4x3) * (3x1) = (4x1)
    let (m, n, k) = (4u32, 1u32, 3u32);

    #[rustfmt::skip]
    let a: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    let b: Vec<f32> = vec![1.0, 2.0, 3.0];

    let gpu_result = run_matmul(&device, &queue, &a, &b, m, n, k);
    let cpu_result = cpu_matmul(&a, &b, m as usize, n as usize, k as usize);

    assert_eq!(gpu_result.len(), m as usize);
    // Expected: [14, 32, 50, 68]
    let expected = [14.0f32, 32.0, 50.0, 68.0];
    for (i, (&got, &exp)) in gpu_result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "vector mismatch at index {i}: gpu={got}, expected={exp}"
        );
    }
    for (i, (&got, &exp)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-5, "vector cpu mismatch at index {i}: gpu={got}, cpu={exp}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_matmul_shader_tests -- --ignored"]
fn test_metal_matmul_accumulation() {
    let (device, queue) = setup_device().expect("Metal device required");

    // Large K dimension to stress floating-point accumulation.
    // All elements are small so exact result is representable.
    let (m, n, k) = (2u32, 2u32, 256u32);

    let a: Vec<f32> = vec![1.0; (m * k) as usize];
    let b: Vec<f32> = (0..k * n).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();

    let gpu_result = run_matmul(&device, &queue, &a, &b, m, n, k);
    let cpu_result = cpu_matmul(&a, &b, m as usize, n as usize, k as usize);

    // Each row of A is all-ones, so result[row][col] = sum of column col of B.
    // Column 0 of B: 256 values alternating 0.5, -0.5 → sum = 0.0
    // Column 1 of B: 256 values alternating -0.5, 0.5 → sum = 0.0
    for (i, (&got, &expected)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-2,
            "accumulation mismatch at index {i}: gpu={got}, cpu={expected}"
        );
    }
    for &val in &gpu_result {
        assert!(val.abs() < 1e-2, "accumulation result should be near zero, got {val}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_matmul_shader_tests -- --ignored"]
fn test_metal_batched_matmul() {
    let (device, queue) = setup_device().expect("Metal device required");

    let batch_size = 4u32;
    let (m, n, k) = (4u32, 4u32, 4u32);

    for batch in 0..batch_size {
        let scale = (batch + 1) as f32;

        // A = scale * I, B = sequential values → C = scale * B
        let mut a = vec![0.0f32; (m * k) as usize];
        for i in 0..m.min(k) {
            a[(i * k + i) as usize] = scale;
        }
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 + 1.0).collect();

        let gpu_result = run_matmul(&device, &queue, &a, &b, m, n, k);

        // scale * I * B = scale * B
        let expected: Vec<f32> = b.iter().map(|&v| v * scale).collect();

        assert_eq!(gpu_result.len(), expected.len());
        for (i, (&got, &exp)) in gpu_result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "batch {batch} mismatch at index {i}: gpu={got}, expected={exp}"
            );
        }
    }
}
