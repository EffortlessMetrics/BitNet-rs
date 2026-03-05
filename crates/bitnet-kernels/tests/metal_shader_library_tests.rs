#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types)]
#![cfg(target_os = "macos")]

//! Metal shader library tests for Apple Silicon.
//!
//! Validates WGSL compute shader patterns needed for neural network inference
//! on Metal via wgpu. Each test compiles and dispatches a real compute shader
//! on the GPU, then reads back results for correctness verification.

use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Request a wgpu device + queue backed by the Metal backend.
/// Returns `None` when no compatible adapter is available.
fn create_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor { label: Some("bitnet-test"), ..Default::default() },
        None,
    ))
    .ok()?;
    Some((device, queue))
}

/// Compile `shader` (WGSL), bind `input_data` slices as storage buffers
/// (bindings 0..N-1), allocate an output storage buffer at binding N,
/// dispatch `workgroups`, and read back the output as `Vec<f32>`.
fn run_compute_shader(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shader: &str,
    input_data: &[&[f32]],
    output_size: usize,
    workgroups: (u32, u32, u32),
) -> Vec<f32> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test-shader"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });

    // --- bind group layout entries ---
    let mut layout_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
    for (i, _) in input_data.iter().enumerate() {
        layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    // output buffer
    layout_entries.push(wgpu::BindGroupLayoutEntry {
        binding: input_data.len() as u32,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("test-bgl"),
        entries: &layout_entries,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("test-pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("test-pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // --- create GPU buffers ---
    let mut bind_group_entries: Vec<wgpu::BindGroupEntry> = Vec::new();
    let mut _input_bufs: Vec<wgpu::Buffer> = Vec::new();
    for (i, data) in input_data.iter().enumerate() {
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("input-{i}")),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        _input_bufs.push(buf);
    }
    for (i, buf) in _input_bufs.iter().enumerate() {
        bind_group_entries
            .push(wgpu::BindGroupEntry { binding: i as u32, resource: buf.as_entire_binding() });
    }

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (output_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    bind_group_entries.push(wgpu::BindGroupEntry {
        binding: input_data.len() as u32,
        resource: output_buf.as_entire_binding(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test-bg"),
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });

    // --- dispatch ---
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    // --- readback ---
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (output_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        &output_buf,
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

    let view = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();
    result
}

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

const ELEMENTWISE_ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&a) {
        result[idx] = a[idx] + b[idx];
    }
}
"#;

const ELEMENTWISE_MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&a) {
        result[idx] = a[idx] * b[idx];
    }
}
"#;

const SOFTMAX_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(1)
fn main() {
    let n = arrayLength(&input);

    // Numerically stable softmax: subtract max first
    var max_val = input[0];
    for (var i = 1u; i < n; i++) {
        max_val = max(max_val, input[i]);
    }

    var sum_exp = 0.0;
    for (var i = 0u; i < n; i++) {
        sum_exp += exp(input[i] - max_val);
    }

    for (var i = 0u; i < n; i++) {
        result[i] = exp(input[i] - max_val) / sum_exp;
    }
}
"#;

const RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&input) {
        result[idx] = max(0.0, input[idx]);
    }
}
"#;

const SILU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&input) {
        let x = input[idx];
        let sigmoid_x = 1.0 / (1.0 + exp(-x));
        result[idx] = x * sigmoid_x;
    }
}
"#;

const LAYERNORM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

const EPS: f32 = 1e-5;

@compute @workgroup_size(1)
fn main() {
    let n = arrayLength(&input);
    let nf = f32(n);

    // Mean
    var sum = 0.0;
    for (var i = 0u; i < n; i++) {
        sum += input[i];
    }
    let mean = sum / nf;

    // Variance
    var var_acc = 0.0;
    for (var i = 0u; i < n; i++) {
        let diff = input[i] - mean;
        var_acc += diff * diff;
    }
    let variance = var_acc / nf;

    // Normalize
    let inv_std = 1.0 / sqrt(variance + EPS);
    for (var i = 0u; i < n; i++) {
        result[i] = (input[i] - mean) * inv_std;
    }
}
"#;

const REDUCE_SUM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let local_idx = lid.x;
    if gid.x < arrayLength(&input) {
        shared[local_idx] = input[gid.x];
    } else {
        shared[local_idx] = 0.0;
    }
    workgroupBarrier();

    // Tree reduction
    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if local_idx < stride {
            shared[local_idx] += shared[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if local_idx == 0u {
        result[0] = shared[0];
    }
}
"#;

const MATRIX_TRANSPOSE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

const ROWS: u32 = 4u;
const COLS: u32 = 4u;
const TILE: u32 = 4u;

var<workgroup> tile: array<f32, 16>;  // TILE * TILE

@compute @workgroup_size(4, 4)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = lid.y;
    let col = lid.x;

    // Load into shared memory
    if row < ROWS && col < COLS {
        tile[row * TILE + col] = input[row * COLS + col];
    }
    workgroupBarrier();

    // Write transposed
    if col < ROWS && row < COLS {
        result[row * ROWS + col] = tile[col * TILE + row];
    }
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_shader_library_tests -- --ignored"]
fn test_elementwise_add_shader() {
    let (device, queue) = create_device().expect("no Metal device available");
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let result =
        run_compute_shader(&device, &queue, ELEMENTWISE_ADD_SHADER, &[&a, &b], 4, (1, 1, 1));
    let expected: Vec<f32> = vec![11.0, 22.0, 33.0, 44.0];
    for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "elementwise add mismatch at [{i}]: got {got}, want {want}"
        );
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_shader_library_tests -- --ignored"]
fn test_elementwise_mul_shader() {
    let (device, queue) = create_device().expect("no Metal device available");
    let a: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0];
    let b: Vec<f32> = vec![0.5, 0.1, 2.5, -1.0];
    let result =
        run_compute_shader(&device, &queue, ELEMENTWISE_MUL_SHADER, &[&a, &b], 4, (1, 1, 1));
    let expected: Vec<f32> = vec![1.0, 0.3, 10.0, -5.0];
    for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "elementwise mul mismatch at [{i}]: got {got}, want {want}"
        );
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_shader_library_tests -- --ignored"]
fn test_softmax_shader() {
    let (device, queue) = create_device().expect("no Metal device available");
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let result = run_compute_shader(&device, &queue, SOFTMAX_SHADER, &[&input], 4, (1, 1, 1));

    // CPU reference (numerically stable)
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let expected: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

    // Probabilities must sum to 1
    let total: f32 = result.iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "softmax output should sum to 1.0, got {total}");
    for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-5, "softmax mismatch at [{i}]: got {got}, want {want}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_shader_library_tests -- --ignored"]
fn test_relu_shader() {
    let (device, queue) = create_device().expect("no Metal device available");
    let input: Vec<f32> = vec![-3.0, -1.0, 0.0, 0.5, 2.0, -0.1, 7.0, -100.0];
    let result =
        run_compute_shader(&device, &queue, RELU_SHADER, &[&input], input.len(), (1, 1, 1));
    let expected: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();
    for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-6, "relu mismatch at [{i}]: got {got}, want {want}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_shader_library_tests -- --ignored"]
fn test_silu_shader() {
    let (device, queue) = create_device().expect("no Metal device available");
    let input: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5];
    let result =
        run_compute_shader(&device, &queue, SILU_SHADER, &[&input], input.len(), (1, 1, 1));
    // SiLU: x * sigmoid(x)
    let expected: Vec<f32> = input.iter().map(|&x| x * (1.0 / (1.0 + (-x).exp()))).collect();
    for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-5, "silu mismatch at [{i}]: got {got}, want {want}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_shader_library_tests -- --ignored"]
fn test_layernorm_shader() {
    let (device, queue) = create_device().expect("no Metal device available");
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result =
        run_compute_shader(&device, &queue, LAYERNORM_SHADER, &[&input], input.len(), (1, 1, 1));

    // CPU reference
    let n = input.len() as f32;
    let mean: f32 = input.iter().sum::<f32>() / n;
    let variance: f32 = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let inv_std = 1.0 / (variance + 1e-5_f32).sqrt();
    let expected: Vec<f32> = input.iter().map(|&x| (x - mean) * inv_std).collect();

    // Normalized output should have near-zero mean
    let result_mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(result_mean.abs() < 1e-4, "layernorm output mean should be ~0, got {result_mean}");
    for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-4, "layernorm mismatch at [{i}]: got {got}, want {want}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_shader_library_tests -- --ignored"]
fn test_reduce_sum_shader() {
    let (device, queue) = create_device().expect("no Metal device available");
    // Use exactly 256 elements to match the workgroup size
    let input: Vec<f32> = (1..=256).map(|x| x as f32).collect();
    let expected_sum: f32 = input.iter().sum();

    let result = run_compute_shader(&device, &queue, REDUCE_SUM_SHADER, &[&input], 1, (1, 1, 1));
    assert!(
        (result[0] - expected_sum).abs() < 1e-1,
        "reduce sum: got {}, want {expected_sum}",
        result[0]
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_shader_library_tests -- --ignored"]
fn test_matrix_transpose_shader() {
    let (device, queue) = create_device().expect("no Metal device available");
    // 4×4 matrix: row-major
    #[rustfmt::skip]
    let input: Vec<f32> = vec![
         1.0,  2.0,  3.0,  4.0,
         5.0,  6.0,  7.0,  8.0,
         9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let result =
        run_compute_shader(&device, &queue, MATRIX_TRANSPOSE_SHADER, &[&input], 16, (1, 1, 1));
    #[rustfmt::skip]
    let expected: Vec<f32> = vec![
         1.0,  5.0,  9.0, 13.0,
         2.0,  6.0, 10.0, 14.0,
         3.0,  7.0, 11.0, 15.0,
         4.0,  8.0, 12.0, 16.0,
    ];
    for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-6, "transpose mismatch at [{i}]: got {got}, want {want}");
    }
}
