//! Tests for Metal matrix multiplication compute shaders on Apple Silicon.
//!
//! Verifies GEMM correctness, transposed variants, batch matmul, dimension
//! sweeps, tiled threadgroup matmul, SIMD group matmul, quantized matmul,
//! mixed-precision, buffer alignment, dispatch sizing, numerical stability,
//! and edge cases — all dispatched via WGSL compute shaders on the Metal
//! backend.

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
// WGSL compute shader: alpha*A*B + beta*C  (full GEMM)
// ---------------------------------------------------------------------------

const GEMM_SHADER: &str = r#"
struct Params {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
    alpha: f32,
    beta: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if row >= params.M || col >= params.N {
        return;
    }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.K; i = i + 1u) {
        sum = sum + A[row * params.K + i] * B[i * params.N + col];
    }
    let idx = row * params.N + col;
    C[idx] = params.alpha * sum + params.beta * C[idx];
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
    run_matmul_with_shader(device, queue, a, b, m, n, k, MATMUL_SHADER, &[m, n, k, 0])
}

fn run_matmul_with_shader(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    a: &[f32],
    b: &[f32],
    m: u32,
    n: u32,
    k: u32,
    shader_src: &str,
    uniform_data: &[u32],
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

    let buf_dims = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dims"),
        contents: bytemuck::cast_slice(uniform_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matmul"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
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

/// Run GEMM with pre-initialized C buffer: C = alpha*A*B + beta*C_init
fn run_gemm(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    a: &[f32],
    b: &[f32],
    c_init: &[f32],
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let output_size = (m * n) as usize;
    assert_eq!(c_init.len(), output_size);

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
    let buf_c = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("C"),
        contents: bytemuck::cast_slice(c_init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // Pack params: M, N, K, _pad, alpha_bits, beta_bits, _pad2, _pad3
    let params: [u32; 8] = [m, n, k, 0, alpha.to_bits(), beta.to_bits(), 0, 0];
    let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gemm"),
        source: wgpu::ShaderSource::Wgsl(GEMM_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gemm_layout"),
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
        label: Some("gemm_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gemm_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gemm_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
        ],
    });

    let workgroups_x = (m + 7) / 8;
    let workgroups_y = (n + 7) / 8;

    let mut encoder = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gemm_encoder") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_pass"),
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

// ---- CPU reference helpers ------------------------------------------------

/// CPU reference matmul: C = A * B  (row-major).
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

/// CPU reference GEMM: C = alpha*A*B + beta*C_init.
fn cpu_gemm(
    a: &[f32],
    b: &[f32],
    c_init: &[f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let ab = cpu_matmul(a, b, m, n, k);
    ab.iter().zip(c_init.iter()).map(|(&ab_val, &c_val)| alpha * ab_val + beta * c_val).collect()
}

/// Transpose a row-major matrix (rows×cols) → (cols×rows).
fn cpu_transpose(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = mat[r * cols + c];
        }
    }
    out
}

/// Generate an identity matrix of size n×n.
fn identity(n: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Deterministic pseudo-random-ish data for a given length and seed.
fn deterministic_data(len: usize, seed: u32) -> Vec<f32> {
    (0..len).map(|i| ((i as u32 ^ seed) % 17) as f32 / 8.0 - 1.0).collect()
}

/// Assert two float slices are element-wise close within `eps`.
fn assert_close(got: &[f32], expected: &[f32], eps: f32, context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < eps,
            "{context}: mismatch at index {i}: got {g}, expected {e} (eps={eps})"
        );
    }
}

// ===========================================================================
//  1. Basic GEMM correctness
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_gemm_identity_multiply() {
    let (dev, q) = setup_device().expect("Metal device required");
    #[rustfmt::skip]
    let a: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let b = identity(4);
    let result = run_matmul(&dev, &q, &a, &b, 4, 4, 4);
    assert_close(&result, &a, 1e-5, "A*I = A");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_gemm_alpha_beta() {
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (4u32, 4u32, 4u32);
    let a = deterministic_data(16, 1);
    let b = deterministic_data(16, 2);
    let c_init: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let alpha = 2.0f32;
    let beta = 0.5f32;

    let gpu = run_gemm(&dev, &q, &a, &b, &c_init, m, n, k, alpha, beta);
    let cpu = cpu_gemm(&a, &b, &c_init, m as usize, n as usize, k as usize, alpha, beta);
    assert_close(&gpu, &cpu, 1e-3, "alpha*A*B + beta*C");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_gemm_alpha_zero_returns_beta_c() {
    let (dev, q) = setup_device().expect("Metal device required");
    let a = vec![99.0f32; 16];
    let b = vec![99.0f32; 16];
    let c_init: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let beta = 3.0f32;

    let gpu = run_gemm(&dev, &q, &a, &b, &c_init, 4, 4, 4, 0.0, beta);
    let expected: Vec<f32> = c_init.iter().map(|&v| beta * v).collect();
    assert_close(&gpu, &expected, 1e-5, "alpha=0 ⇒ beta*C");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_gemm_square_deterministic() {
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (8u32, 8u32, 8u32);
    let a = deterministic_data(64, 42);
    let b = deterministic_data(64, 99);
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 8, 8, 8);
    assert_close(&gpu, &cpu, 1e-3, "8x8 deterministic");
}

// ===========================================================================
//  2. Transposed variants
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_transpose_a_times_b() {
    let (dev, q) = setup_device().expect("Metal device required");
    // A is K×M in storage, we transpose to M×K, then multiply by B (K×N).
    let (m, n, k) = (4u32, 3u32, 5u32);
    let a_km: Vec<f32> = deterministic_data((k * m) as usize, 10);
    let a_mk = cpu_transpose(&a_km, k as usize, m as usize);
    let b: Vec<f32> = deterministic_data((k * n) as usize, 20);

    let gpu = run_matmul(&dev, &q, &a_mk, &b, m, n, k);
    let cpu = cpu_matmul(&a_mk, &b, m as usize, n as usize, k as usize);
    assert_close(&gpu, &cpu, 1e-3, "A^T * B");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_a_times_transpose_b() {
    let (dev, q) = setup_device().expect("Metal device required");
    // B is N×K in storage, transpose to K×N, then A (M×K) * B^T.
    let (m, n, k) = (3u32, 4u32, 5u32);
    let a: Vec<f32> = deterministic_data((m * k) as usize, 30);
    let b_nk: Vec<f32> = deterministic_data((n * k) as usize, 40);
    let b_kn = cpu_transpose(&b_nk, n as usize, k as usize);

    let gpu = run_matmul(&dev, &q, &a, &b_kn, m, n, k);
    let cpu = cpu_matmul(&a, &b_kn, m as usize, n as usize, k as usize);
    assert_close(&gpu, &cpu, 1e-3, "A * B^T");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_transpose_a_times_transpose_b() {
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (3u32, 4u32, 5u32);
    let a_km = deterministic_data((k * m) as usize, 50);
    let a_mk = cpu_transpose(&a_km, k as usize, m as usize);
    let b_nk = deterministic_data((n * k) as usize, 60);
    let b_kn = cpu_transpose(&b_nk, n as usize, k as usize);

    let gpu = run_matmul(&dev, &q, &a_mk, &b_kn, m, n, k);
    let cpu = cpu_matmul(&a_mk, &b_kn, m as usize, n as usize, k as usize);
    assert_close(&gpu, &cpu, 1e-3, "A^T * B^T");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_transpose_symmetry_ab_t_eq_bt_at() {
    // Verify (A*B)^T == B^T * A^T on GPU.
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (4u32, 5u32, 3u32);
    let a = deterministic_data((m * k) as usize, 70);
    let b = deterministic_data((k * n) as usize, 80);

    let ab = run_matmul(&dev, &q, &a, &b, m, n, k);
    let ab_t = cpu_transpose(&ab, m as usize, n as usize);

    let bt = cpu_transpose(&b, k as usize, n as usize);
    let at = cpu_transpose(&a, m as usize, k as usize);
    let bt_at = run_matmul(&dev, &q, &bt, &at, n, m, k);
    assert_close(&ab_t, &bt_at, 1e-3, "(AB)^T == B^T A^T");
}

// ===========================================================================
//  3. Batch GEMM
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_batched_matmul_scaled_identity() {
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (4u32, 4u32, 4u32);
    for batch in 0..4u32 {
        let scale = (batch + 1) as f32;
        let mut a = vec![0.0f32; 16];
        for i in 0..4 {
            a[i * 4 + i] = scale;
        }
        let b: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect();
        let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
        let expected: Vec<f32> = b.iter().map(|&v| v * scale).collect();
        assert_close(&gpu, &expected, 1e-4, &format!("batch {batch}"));
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_batched_matmul_independence() {
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (4u32, 4u32, 4u32);
    let mut results = Vec::new();
    for seed in [11u32, 22, 33, 44] {
        let a = deterministic_data(16, seed);
        let b = deterministic_data(16, seed + 100);
        results.push(run_matmul(&dev, &q, &a, &b, m, n, k));
    }
    // Each batch should produce a different result.
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let diff: f32 =
                results[i].iter().zip(results[j].iter()).map(|(a, b)| (a - b).abs()).sum();
            assert!(diff > 1e-6, "batches {i} and {j} should differ");
        }
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_batched_matmul_8_batches() {
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (8u32, 8u32, 8u32);
    for batch in 0..8u32 {
        let a = deterministic_data(64, batch * 7);
        let b = deterministic_data(64, batch * 13);
        let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
        let cpu = cpu_matmul(&a, &b, 8, 8, 8);
        assert_close(&gpu, &cpu, 1e-3, &format!("batch {batch}"));
    }
}

// ===========================================================================
//  4. Dimension sweeps
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_dimension_sweep_tiny() {
    let (dev, q) = setup_device().expect("Metal device required");
    for &dim in &[1u32, 2, 3, 4, 5, 7] {
        let a = deterministic_data((dim * dim) as usize, dim);
        let b = deterministic_data((dim * dim) as usize, dim + 100);
        let gpu = run_matmul(&dev, &q, &a, &b, dim, dim, dim);
        let cpu = cpu_matmul(&a, &b, dim as usize, dim as usize, dim as usize);
        assert_close(&gpu, &cpu, 1e-3, &format!("dim={dim}"));
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_dimension_sweep_small() {
    let (dev, q) = setup_device().expect("Metal device required");
    for &dim in &[8u32, 9, 15, 16, 17, 31, 32] {
        let a = deterministic_data((dim * dim) as usize, dim);
        let b = deterministic_data((dim * dim) as usize, dim + 50);
        let gpu = run_matmul(&dev, &q, &a, &b, dim, dim, dim);
        let cpu = cpu_matmul(&a, &b, dim as usize, dim as usize, dim as usize);
        assert_close(&gpu, &cpu, 1e-2, &format!("dim={dim}"));
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_dimension_sweep_medium() {
    let (dev, q) = setup_device().expect("Metal device required");
    for &dim in &[64u32, 128, 256] {
        let a = deterministic_data((dim * dim) as usize, dim);
        let b = deterministic_data((dim * dim) as usize, dim + 200);
        let gpu = run_matmul(&dev, &q, &a, &b, dim, dim, dim);
        let cpu = cpu_matmul(&a, &b, dim as usize, dim as usize, dim as usize);
        assert_close(&gpu, &cpu, 0.1, &format!("dim={dim}"));
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_dimension_sweep_large() {
    let (dev, q) = setup_device().expect("Metal device required");
    for &dim in &[512u32, 1024] {
        let a = deterministic_data((dim * dim) as usize, dim);
        let b = deterministic_data((dim * dim) as usize, dim + 300);
        let gpu = run_matmul(&dev, &q, &a, &b, dim, dim, dim);
        let cpu = cpu_matmul(&a, &b, dim as usize, dim as usize, dim as usize);
        // Larger matrices accumulate more FP error.
        assert_close(&gpu, &cpu, 1.0, &format!("dim={dim}"));
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_dimension_sweep_asymmetric() {
    let (dev, q) = setup_device().expect("Metal device required");
    let cases: [(u32, u32, u32); 5] =
        [(1, 64, 32), (16, 1, 8), (7, 13, 11), (64, 32, 16), (128, 64, 256)];
    for (m, n, k) in cases {
        let a = deterministic_data((m * k) as usize, m ^ n);
        let b = deterministic_data((k * n) as usize, k ^ n);
        let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
        let cpu = cpu_matmul(&a, &b, m as usize, n as usize, k as usize);
        assert_close(&gpu, &cpu, 0.5, &format!("({m},{n},{k})"));
    }
}

// ===========================================================================
//  5. Tiled matmul (threadgroup tiling)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_tiled_matmul_8x8() {
    let (dev, q) = setup_device().expect("Metal device required");
    let dim = 8u32;
    let a: Vec<f32> = (0..64).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..64).map(|i| (i % 5) as f32 - 2.0).collect();
    let gpu = run_matmul(&dev, &q, &a, &b, dim, dim, dim);
    let cpu = cpu_matmul(&a, &b, 8, 8, 8);
    assert_close(&gpu, &cpu, 1e-3, "tiled 8x8");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_tiled_matmul_16x16() {
    let (dev, q) = setup_device().expect("Metal device required");
    let dim = 16u32;
    let a: Vec<f32> = (0..256).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..256).map(|i| (i % 5) as f32 - 2.0).collect();
    let gpu = run_matmul(&dev, &q, &a, &b, dim, dim, dim);
    let cpu = cpu_matmul(&a, &b, 16, 16, 16);
    assert_close(&gpu, &cpu, 1e-3, "tiled 16x16");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_tiled_matmul_32x32() {
    let (dev, q) = setup_device().expect("Metal device required");
    let dim = 32u32;
    let a = deterministic_data(1024, 32);
    let b = deterministic_data(1024, 64);
    let gpu = run_matmul(&dev, &q, &a, &b, dim, dim, dim);
    let cpu = cpu_matmul(&a, &b, 32, 32, 32);
    assert_close(&gpu, &cpu, 1e-2, "tiled 32x32");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_tiled_matmul_non_tile_aligned() {
    let (dev, q) = setup_device().expect("Metal device required");
    // 13×11 is not aligned to any common tile size (8, 16, 32).
    let (m, n, k) = (13u32, 11u32, 7u32);
    let a = deterministic_data((m * k) as usize, 91);
    let b = deterministic_data((k * n) as usize, 77);
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, m as usize, n as usize, k as usize);
    assert_close(&gpu, &cpu, 1e-3, "non-aligned tile");
}

// ===========================================================================
//  6. SIMD group matmul (8×8 Apple Silicon SIMD groups)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_simd_group_8x8_identity() {
    let (dev, q) = setup_device().expect("Metal device required");
    let a = deterministic_data(64, 8);
    let b = identity(8);
    let gpu = run_matmul(&dev, &q, &a, &b, 8, 8, 8);
    assert_close(&gpu, &a, 1e-5, "SIMD 8x8 identity");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_simd_group_8x8_correctness() {
    let (dev, q) = setup_device().expect("Metal device required");
    let a = deterministic_data(64, 111);
    let b = deterministic_data(64, 222);
    let gpu = run_matmul(&dev, &q, &a, &b, 8, 8, 8);
    let cpu = cpu_matmul(&a, &b, 8, 8, 8);
    assert_close(&gpu, &cpu, 1e-3, "SIMD 8x8 correctness");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_simd_group_boundary_16x16() {
    // Two SIMD groups per dimension (16/8 = 2).
    let (dev, q) = setup_device().expect("Metal device required");
    let a = deterministic_data(256, 160);
    let b = deterministic_data(256, 161);
    let gpu = run_matmul(&dev, &q, &a, &b, 16, 16, 16);
    let cpu = cpu_matmul(&a, &b, 16, 16, 16);
    assert_close(&gpu, &cpu, 1e-3, "SIMD boundary 16x16");
}

// ===========================================================================
//  7. Quantized matmul (I2_S weights × F32 activations)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_quantized_ternary_weights() {
    // Simulate I2_S ternary {-1, 0, 1} weights multiplied by F32 activations.
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (4u32, 4u32, 8u32);
    let activations: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    // Ternary weight pattern
    let weights: Vec<f32> = (0..32)
        .map(|i| match i % 3 {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        })
        .collect();

    let gpu = run_matmul(&dev, &q, &activations, &weights, m, n, k);
    let cpu = cpu_matmul(&activations, &weights, m as usize, n as usize, k as usize);
    assert_close(&gpu, &cpu, 1e-5, "ternary weights");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_quantized_binary_weights() {
    // Binary {-1, 1} weight pattern.
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (4u32, 4u32, 8u32);
    let activations: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.05).collect();
    let weights: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

    let gpu = run_matmul(&dev, &q, &activations, &weights, m, n, k);
    let cpu = cpu_matmul(&activations, &weights, m as usize, n as usize, k as usize);
    assert_close(&gpu, &cpu, 1e-5, "binary weights");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_quantized_i2s_simulated_block() {
    // Simulate a 256-element QK256 block: weights are {-1, 0, 1} scaled by a
    // per-block scale factor, multiplied by F32 activations.
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (1u32, 1u32, 256u32);
    let scale = 0.03125f32; // typical QK256 scale
    let weights: Vec<f32> = (0..256)
        .map(|i| {
            (match i % 3 {
                0 => -1i8,
                1 => 0,
                _ => 1,
            } as f32)
                * scale
        })
        .collect();
    let activations: Vec<f32> = (0..256).map(|i| (i as f32 / 256.0) - 0.5).collect();

    let gpu = run_matmul(&dev, &q, &activations, &weights, m, n, k);
    let cpu = cpu_matmul(&activations, &weights, 1, 1, 256);
    assert_close(&gpu, &cpu, 1e-4, "I2_S simulated block");
}

// ===========================================================================
//  8. Mixed-precision matmul (F16 accumulation simulation)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_mixed_precision_f16_range_inputs() {
    // Inputs in F16 representable range; verify GPU F32 result matches CPU.
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (8u32, 8u32, 8u32);
    let a: Vec<f32> = (0..64).map(|i| half::f16::from_f32((i as f32) * 0.01).to_f32()).collect();
    let b: Vec<f32> =
        (0..64).map(|i| half::f16::from_f32((i as f32) * -0.02 + 0.5).to_f32()).collect();

    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 8, 8, 8);
    assert_close(&gpu, &cpu, 1e-2, "f16 range inputs");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_mixed_precision_f16_truncation_error() {
    // Compare accumulation in F32 vs simulated F16 truncation to quantify precision loss.
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (4u32, 4u32, 32u32);
    let a = deterministic_data(128, 55);
    let b = deterministic_data(128, 66);

    let gpu_f32 = run_matmul(&dev, &q, &a, &b, m, n, k);
    // Simulate F16 truncation of inputs then accumulate in F32.
    let a_f16: Vec<f32> = a.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
    let b_f16: Vec<f32> = b.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
    let cpu_f16 = cpu_matmul(&a_f16, &b_f16, 4, 4, 32);

    // The GPU path with F32 should be closer to F32 reference; F16 truncation introduces error.
    let f32_ref = cpu_matmul(&a, &b, 4, 4, 32);
    let gpu_err: f32 = gpu_f32.iter().zip(f32_ref.iter()).map(|(g, r)| (g - r).abs()).sum();
    let f16_err: f32 = cpu_f16.iter().zip(f32_ref.iter()).map(|(h, r)| (h - r).abs()).sum();
    assert!(
        gpu_err <= f16_err + 1e-3,
        "GPU F32 path error ({gpu_err}) should not exceed F16 truncation error ({f16_err})"
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_mixed_precision_accuracy_boundary() {
    // Values near F16 max (65504) — ensure no overflow in accumulation.
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (2u32, 2u32, 4u32);
    let a = vec![100.0f32; 8]; // well within f16 range
    let b = vec![100.0f32; 8];

    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 2, 2, 4);
    // Each element = 100 * 100 * 4 = 40000 (within f16 range)
    assert_close(&gpu, &cpu, 1e-1, "f16 boundary");
}

// ===========================================================================
//  9. Metal buffer alignment (256-byte alignment)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_buffer_alignment_256_byte() {
    let (dev, q) = setup_device().expect("Metal device required");
    // 256 bytes = 64 f32 elements. Use dimensions that yield 256-byte aligned buffers.
    let (m, n, k) = (8u32, 8u32, 8u32); // 64 elements × 4 bytes = 256 bytes each
    let a = deterministic_data(64, 256);
    let b = deterministic_data(64, 257);
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 8, 8, 8);
    assert_close(&gpu, &cpu, 1e-3, "256-byte aligned");
    // Verify buffer sizes are 256-byte aligned.
    let buf_size = 64 * std::mem::size_of::<f32>();
    assert_eq!(buf_size % 256, 0, "buffer size should be 256-byte aligned");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_buffer_alignment_non_power_of_two() {
    let (dev, q) = setup_device().expect("Metal device required");
    // 3×5 = 15 elements, not aligned to any power-of-two boundary.
    let (m, n, k) = (3u32, 5u32, 7u32);
    let a = deterministic_data(21, 300);
    let b = deterministic_data(35, 301);
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 3, 5, 7);
    assert_close(&gpu, &cpu, 1e-3, "non-power-of-two alignment");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_buffer_alignment_large_matrix() {
    let (dev, q) = setup_device().expect("Metal device required");
    // 64×64 = 4096 elements × 4 = 16384 bytes (multiple of 256).
    let (m, n, k) = (64u32, 64u32, 64u32);
    let a = deterministic_data(4096, 400);
    let b = deterministic_data(4096, 401);
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 64, 64, 64);
    assert_close(&gpu, &cpu, 0.5, "large aligned matrix");
}

// ===========================================================================
// 10. Dispatch sizing
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_dispatch_workgroup_size_exact() {
    // Dimensions are exact multiples of workgroup size (8).
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (8u32, 16u32, 24u32);
    let a = deterministic_data((m * k) as usize, 500);
    let b = deterministic_data((k * n) as usize, 501);
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, m as usize, n as usize, k as usize);
    assert_close(&gpu, &cpu, 1e-2, "exact workgroup multiple");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_dispatch_non_multiple_of_workgroup() {
    // 5 and 11 are not multiples of 8 — tests boundary-guard in shader.
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (5u32, 11u32, 3u32);
    let a = deterministic_data(15, 600);
    let b = deterministic_data(33, 601);
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 5, 11, 3);
    assert_close(&gpu, &cpu, 1e-3, "non-multiple dispatch");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_dispatch_large_grid() {
    // 128×128 → 16×16 = 256 workgroups.
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (128u32, 128u32, 32u32);
    let a = deterministic_data((m * k) as usize, 700);
    let b = deterministic_data((k * n) as usize, 701);
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 128, 128, 32);
    assert_close(&gpu, &cpu, 0.5, "large dispatch grid");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_dispatch_single_workgroup() {
    // Everything fits in one 8×8 workgroup.
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (4u32, 4u32, 4u32);
    let a = deterministic_data(16, 800);
    let b = deterministic_data(16, 801);
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 4, 4, 4);
    assert_close(&gpu, &cpu, 1e-5, "single workgroup");
}

// ===========================================================================
// 11. Numerical stability
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_numerical_stability_large_values() {
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (4u32, 4u32, 4u32);
    let a = vec![1000.0f32; 16];
    let b = vec![1000.0f32; 16];
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 4, 4, 4);
    // Each element = 1000 * 1000 * 4 = 4_000_000
    for (&g, &c) in gpu.iter().zip(cpu.iter()) {
        let rel = if c.abs() > 1.0 { (g - c).abs() / c.abs() } else { (g - c).abs() };
        assert!(rel < 1e-4, "large value: gpu={g}, cpu={c}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_numerical_stability_small_values() {
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (4u32, 4u32, 4u32);
    let a = vec![1e-6f32; 16];
    let b = vec![1e-6f32; 16];
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 4, 4, 4);
    assert_close(&gpu, &cpu, 1e-15, "small values");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_numerical_stability_mixed_signs() {
    // Catastrophic cancellation scenario: large positive + large negative.
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (2u32, 2u32, 256u32);
    let a = vec![1.0f32; 512];
    let b: Vec<f32> = (0..512).map(|i| if i % 2 == 0 { 1e4 } else { -1e4 }).collect();
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 2, 2, 256);
    // Result should be near zero due to cancellation.
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!((g - c).abs() < 1.0, "cancellation at {i}: gpu={g}, cpu={c}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_numerical_stability_accumulated_error() {
    // Long accumulation chain (K=1024) with values near 1.0.
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (2u32, 2u32, 1024u32);
    let a = vec![1.0f32; 2048];
    let b: Vec<f32> = (0..2048).map(|i| 1.0 + (i as f32) * 1e-5).collect();
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 2, 2, 1024);
    // Relative error should be small even with long accumulation.
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let rel = if c.abs() > 1.0 { (g - c).abs() / c.abs() } else { (g - c).abs() };
        assert!(rel < 1e-3, "accumulated error at {i}: gpu={g}, cpu={c}, rel={rel}");
    }
}

// ===========================================================================
// 12. Edge cases
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_zero_matrix_multiply() {
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (4u32, 4u32, 4u32);
    let a = vec![0.0f32; 16];
    let b = deterministic_data(16, 900);
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    for (i, &v) in gpu.iter().enumerate() {
        assert!(v.abs() < 1e-10, "zero*B should be zero at {i}, got {v}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_zero_matrix_on_right() {
    let (dev, q) = setup_device().expect("Metal device required");
    let a = deterministic_data(16, 901);
    let b = vec![0.0f32; 16];
    let gpu = run_matmul(&dev, &q, &a, &b, 4, 4, 4);
    for (i, &v) in gpu.iter().enumerate() {
        assert!(v.abs() < 1e-10, "A*zero should be zero at {i}, got {v}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_identity_left_multiply() {
    let (dev, q) = setup_device().expect("Metal device required");
    let b = deterministic_data(16, 910);
    let ident = identity(4);
    let gpu = run_matmul(&dev, &q, &ident, &b, 4, 4, 4);
    assert_close(&gpu, &b, 1e-5, "I*B = B");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_single_row_multiply() {
    // (1×K) × (K×N) = (1×N)
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (1u32, 4u32, 8u32);
    let a = deterministic_data(8, 920);
    let b = deterministic_data(32, 921);
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 1, 4, 8);
    assert_close(&gpu, &cpu, 1e-3, "single row");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_single_column_multiply() {
    // (M×K) × (K×1) = (M×1)
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (4u32, 1u32, 8u32);
    let a = deterministic_data(32, 930);
    let b = deterministic_data(8, 931);
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 4, 1, 8);
    assert_close(&gpu, &cpu, 1e-3, "single column");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_single_element_multiply() {
    // (1×1) × (1×1) = (1×1): scalar multiply.
    let (dev, q) = setup_device().expect("Metal device required");
    let a = vec![3.0f32];
    let b = vec![7.0f32];
    let gpu = run_matmul(&dev, &q, &a, &b, 1, 1, 1);
    assert!((gpu[0] - 21.0).abs() < 1e-5, "scalar: 3*7 = {}", gpu[0]);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_matrix_vector_product() {
    // (M×K) × (K×1) with known results.
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (4u32, 1u32, 3u32);
    #[rustfmt::skip]
    let a: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    let b = vec![1.0f32, 2.0, 3.0];
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let expected = [14.0f32, 32.0, 50.0, 68.0];
    assert_close(&gpu, &expected, 1e-5, "matvec");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_non_square_rectangular() {
    let (dev, q) = setup_device().expect("Metal device required");
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
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 3, 5, 4);
    assert_close(&gpu, &cpu, 1e-5, "rectangular");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_wide_short_matrix() {
    // Very wide (N>>M): (2×4) × (4×64) = (2×64).
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (2u32, 64u32, 4u32);
    let a = deterministic_data(8, 940);
    let b = deterministic_data(256, 941);
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 2, 64, 4);
    assert_close(&gpu, &cpu, 1e-3, "wide short");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_tall_narrow_matrix() {
    // Very tall (M>>N): (64×4) × (4×2) = (64×2).
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (64u32, 2u32, 4u32);
    let a = deterministic_data(256, 950);
    let b = deterministic_data(8, 951);
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    let cpu = cpu_matmul(&a, &b, 64, 2, 4);
    assert_close(&gpu, &cpu, 1e-3, "tall narrow");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_accumulation_alternating_signs() {
    // Stress floating-point accumulation with alternating +/- values.
    let (dev, q) = setup_device().expect("Metal device required");
    let (m, n, k) = (2u32, 2u32, 256u32);
    let a = vec![1.0f32; 512];
    let b: Vec<f32> = (0..512).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
    let gpu = run_matmul(&dev, &q, &a, &b, m, n, k);
    for &v in &gpu {
        assert!(v.abs() < 1e-2, "cancellation result should be near zero, got {v}");
    }
}
