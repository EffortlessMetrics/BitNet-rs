#![allow(clippy::approx_constant)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::identity_op)]
#![allow(clippy::manual_abs_diff)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_contains)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::no_effect)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_vec)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::manual_saturating_arithmetic)]

//! Metal GPU LayerNorm / RMSNorm shader tests for Apple Silicon.
//!
//! 30+ tests validating Metal GPU LayerNorm operations including
//! mean/variance normalization, gamma/beta affine transforms, RMSNorm
//! variant, buffer alignment, dispatch sizing, and numerical stability.
//!
//! Each test computes expected values analytically via a CPU reference
//! implementation and compares against the wgpu/Metal compute shader output.
//!
//! All tests are `#[ignore]` because CI runs on Linux.

#![cfg(target_os = "macos")]

use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Metal requires 256-byte buffer alignment for optimal performance.
const METAL_BUFFER_ALIGNMENT: u64 = 256;

/// Metal max threads per threadgroup on Apple Silicon.
const MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// SIMD group (warp) width on Apple Silicon GPUs.
const SIMD_GROUP_SIZE: u32 = 32;

/// Default epsilon for normalization to avoid division by zero.
const NORM_EPSILON: f32 = 1e-5;

// ---------------------------------------------------------------------------
// Helper: Metal context
// ---------------------------------------------------------------------------

struct MetalContext {
    #[allow(dead_code)]
    instance: wgpu::Instance,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn create_metal_context() -> MetalContext {
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
            .await
            .expect("No Metal adapter found — is this running on Apple Silicon?");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("Failed to create wgpu device on Metal adapter");

        MetalContext { instance, adapter, device, queue }
    })
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// CPU reference: layer normalization.
/// output[i] = ((input[i] - mean) / sqrt(var + eps)) * gamma[i] + beta[i]
fn layer_norm_cpu(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len() as f32;
    let mean = input.iter().sum::<f32>() / n;
    let var = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    input
        .iter()
        .zip(gamma.iter())
        .zip(beta.iter())
        .map(|((&x, &g), &b)| (x - mean) * inv_std * g + b)
        .collect()
}

/// CPU reference: RMS normalization (no mean subtraction).
/// output[i] = (input[i] / sqrt(mean(input^2) + eps)) * weight[i]
fn rms_norm_cpu(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / n as f32;
    let rms = (mean_sq + eps).sqrt();
    input.iter().zip(weight.iter()).map(|(&x, &w)| (x / rms) * w).collect()
}

// ---------------------------------------------------------------------------
// WGSL shaders
// ---------------------------------------------------------------------------

const LAYER_NORM_SHADER: &str = r#"
struct Params {
    n: u32,
    eps: f32,
}
@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read>       gamma:  array<f32>;
@group(0) @binding(2) var<storage, read>       beta:   array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform>             params: Params;

var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> shared_sq:  array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let row  = wid.x;
    let tid  = lid.x;
    let n    = params.n;
    let base = row * n;

    var partial_sum: f32 = 0.0;
    var partial_sq: f32 = 0.0;
    var i = tid;
    while i < n {
        let v = input[base + i];
        partial_sum += v;
        partial_sq  += v * v;
        i += 256u;
    }
    shared_sum[tid] = partial_sum;
    shared_sq[tid]  = partial_sq;
    workgroupBarrier();

    var stride: u32 = 128u;
    while stride > 0u {
        if tid < stride {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sq[tid]  += shared_sq[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    let mean    = shared_sum[0] / f32(n);
    let var_val = shared_sq[0] / f32(n) - mean * mean;
    let inv_std = 1.0 / sqrt(var_val + params.eps);
    workgroupBarrier();

    i = tid;
    while i < n {
        output[base + i] =
            (input[base + i] - mean) * inv_std * gamma[i] + beta[i];
        i += 256u;
    }
}
"#;

const RMS_NORM_SHADER: &str = r#"
struct Params {
    n: u32,
    eps: f32,
}
@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let row   = wid.x;
    let tid   = lid.x;
    let n     = params.n;
    let base  = row * n;

    var partial: f32 = 0.0;
    var i = tid;
    while i < n {
        let v = input[base + i];
        partial += v * v;
        i += 256u;
    }
    shared_sum[tid] = partial;
    workgroupBarrier();

    var stride: u32 = 128u;
    while stride > 0u {
        if tid < stride {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    let rms = sqrt(shared_sum[0] / f32(n) + params.eps);
    workgroupBarrier();

    i = tid;
    while i < n {
        output[base + i] = (input[base + i] / rms) * weight[i];
        i += 256u;
    }
}
"#;

// ---------------------------------------------------------------------------
// GPU dispatch helpers
// ---------------------------------------------------------------------------

/// Runs layer norm shader with [input, gamma, beta, output, params].
fn run_layer_norm_gpu(
    ctx: &MetalContext,
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    n: u32,
    eps: f32,
    num_rows: u32,
) -> Vec<f32> {
    let total = (num_rows * n) as usize;

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("layer_norm"),
        source: wgpu::ShaderSource::Wgsl(LAYER_NORM_SHADER.into()),
    });

    let buf_input = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_gamma = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gamma"),
        contents: bytemuck::cast_slice(gamma),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_beta = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("beta"),
        contents: bytemuck::cast_slice(beta),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_size = (total * std::mem::size_of::<f32>()) as u64;
    let buf_output = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_data: [u32; 2] = [n, eps.to_bits()];
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ln_layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4),
        ],
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ln_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_gamma.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_beta.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_output.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: buf_params.as_entire_binding() },
        ],
    });

    let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ln_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ln_pipeline"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buf_staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ln_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(num_rows, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buf_output, 0, &buf_staging, 0, output_size);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    read_back_f32(ctx, &buf_staging, total)
}

/// Runs RMS norm shader with [input, weight, output, params].
fn run_rms_norm_gpu(
    ctx: &MetalContext,
    input: &[f32],
    weight: &[f32],
    n: u32,
    eps: f32,
    num_rows: u32,
) -> Vec<f32> {
    let total = (num_rows * n) as usize;
    assert_eq!(input.len(), total);
    assert_eq!(weight.len(), n as usize);

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rms_norm"),
        source: wgpu::ShaderSource::Wgsl(RMS_NORM_SHADER.into()),
    });

    let buf_input = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_weight = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weight"),
        contents: bytemuck::cast_slice(weight),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_size = (total * std::mem::size_of::<f32>()) as u64;
    let buf_output = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_data: [u32; 2] = [n, eps.to_bits()];
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rms_layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rms_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_weight.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_output.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
        ],
    });

    let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rms_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rms_pipeline"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buf_staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rms_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(num_rows, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buf_output, 0, &buf_staging, 0, output_size);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    read_back_f32(ctx, &buf_staging, total)
}

// ---------------------------------------------------------------------------
// Bind group layout helpers
// ---------------------------------------------------------------------------

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ---------------------------------------------------------------------------
// Read-back helper
// ---------------------------------------------------------------------------

fn read_back_f32(ctx: &MetalContext, staging: &wgpu::Buffer, count: usize) -> Vec<f32> {
    pollster::block_on(async {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data)[..count].to_vec();
        drop(data);
        staging.unmap();
        out
    })
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

fn assert_close(gpu: &[f32], cpu: &[f32], atol: f32, label: &str) {
    assert_eq!(
        gpu.len(),
        cpu.len(),
        "{label}: length mismatch (gpu={}, cpu={})",
        gpu.len(),
        cpu.len()
    );
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let diff = (g - c).abs();
        assert!(diff < atol, "{label}[{i}]: gpu={g}, cpu={c}, diff={diff} >= atol={atol}");
    }
}

fn assert_no_nan_inf(data: &[f32], label: &str) {
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "{label}[{i}]: non-finite value {v}");
    }
}

/// Round up to Metal 256-byte alignment.
fn align_to_256(byte_size: u64) -> u64 {
    (byte_size + METAL_BUFFER_ALIGNMENT - 1) & !(METAL_BUFFER_ALIGNMENT - 1)
}

// ===========================================================================
// 1. Basic LayerNorm correctness
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_basic() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 - 2.56).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "layernorm_basic");
}

// ===========================================================================
// 2. Output mean ≈ 0 (with gamma=1, beta=0)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_mean_zero() {
    let ctx = create_metal_context();
    let n: u32 = 512;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.73).sin() * 5.0).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let mean: f32 = gpu.iter().sum::<f32>() / gpu.len() as f32;
    assert!(mean.abs() < 1e-4, "output mean should be ~0, got {mean}");
}

// ===========================================================================
// 3. Output variance ≈ 1 (with gamma=1, beta=0)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_unit_variance() {
    let ctx = create_metal_context();
    let n: u32 = 512;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.1).cos() * 3.0).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let mean: f32 = gpu.iter().sum::<f32>() / gpu.len() as f32;
    let var: f32 = gpu.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / gpu.len() as f32;
    assert!((var - 1.0).abs() < 1e-3, "output variance should be ~1, got {var}");
}

// ===========================================================================
// 4. Gamma scaling
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_gamma_scale() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let gamma_val = 2.5f32;
    let gamma: Vec<f32> = vec![gamma_val; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu_scaled = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let gamma_one: Vec<f32> = vec![1.0; n as usize];
    let gpu_unscaled = run_layer_norm_gpu(&ctx, &input, &gamma_one, &beta, n, NORM_EPSILON, 1);

    for (i, (&s, &u)) in gpu_scaled.iter().zip(gpu_unscaled.iter()).enumerate() {
        let expected = u * gamma_val;
        let diff = (s - expected).abs();
        assert!(diff < 1e-4, "gamma_scale[{i}]: got {s}, expected {expected}, diff={diff}");
    }
}

// ===========================================================================
// 5. Beta shift
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_beta_shift() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta_val = 3.7f32;
    let beta: Vec<f32> = vec![beta_val; n as usize];

    let gpu_shifted = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let beta_zero: Vec<f32> = vec![0.0; n as usize];
    let gpu_unshifted = run_layer_norm_gpu(&ctx, &input, &gamma, &beta_zero, n, NORM_EPSILON, 1);

    for (i, (&s, &u)) in gpu_shifted.iter().zip(gpu_unshifted.iter()).enumerate() {
        let expected = u + beta_val;
        let diff = (s - expected).abs();
        assert!(diff < 1e-4, "beta_shift[{i}]: got {s}, expected {expected}, diff={diff}");
    }
}

// ===========================================================================
// 6. Identity weights (gamma=1, beta=0) matches bare normalization
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_identity_weights() {
    let ctx = create_metal_context();
    let n: u32 = 512;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin()).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "identity_weights");
}

// ===========================================================================
// 7. Epsilon prevents division by zero (constant input)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_eps_effect() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    // Constant input → variance = 0; epsilon must prevent NaN.
    let input: Vec<f32> = vec![42.0; n as usize];
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    assert_no_nan_inf(&gpu, "eps_effect");
    // All identical input → (x - mean) = 0 → output should be beta = 0
    for (i, &v) in gpu.iter().enumerate() {
        assert!(v.abs() < 1e-3, "eps_effect[{i}]: expected ~0, got {v}");
    }
}

// ===========================================================================
// 8. Degenerate d=1
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_single_element() {
    let ctx = create_metal_context();
    let n: u32 = 1;
    let input = vec![7.5f32];
    let gamma = vec![2.0f32];
    let beta = vec![1.0f32];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    // Single element: mean=7.5, var=0, (x-mean)*inv_std=0, output = 0*gamma+beta = 1.0
    assert!((gpu[0] - 1.0).abs() < 1e-3, "single_element: expected ~1.0, got {}", gpu[0]);
}

// ===========================================================================
// 9. Large hidden dimension (d=4096, typical transformer)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_large_hidden_dim() {
    let ctx = create_metal_context();
    let n: u32 = 4096;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0017).sin()).collect();
    let gamma: Vec<f32> = (0..n).map(|i| 0.9 + (i as f32) * 0.00005).collect();
    let beta: Vec<f32> = (0..n).map(|i| -0.05 + (i as f32) * 0.00002).collect();

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-3, "large_hidden_dim_4096");
}

// ===========================================================================
// 10. Batch consistency (same input in two rows → same output)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_batch_consistency() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let num_rows: u32 = 4;
    let single_row: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.5).cos()).collect();
    let input: Vec<f32> =
        single_row.iter().cycle().take((num_rows * n) as usize).copied().collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, num_rows);
    let first_row = &gpu[..n as usize];
    for row in 1..num_rows as usize {
        let start = row * n as usize;
        let end = start + n as usize;
        assert_close(&gpu[start..end], first_row, 1e-5, &format!("batch_row_{row}"));
    }
}

// ===========================================================================
// 11. Gradient stability (no NaN/Inf for extreme values)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_gradient_stability() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    // Mix of very large, very small, and negative values.
    let input: Vec<f32> = (0..n)
        .map(|i| match i % 4 {
            0 => 1e6,
            1 => -1e6,
            2 => 1e-7,
            _ => -1e-7,
        })
        .collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    assert_no_nan_inf(&gpu, "gradient_stability");
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-2, "gradient_stability_parity");
}

// ===========================================================================
// 12. RMSNorm variant (no mean subtraction)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_rmsnorm_variant() {
    let ctx = create_metal_context();
    let n: u32 = 512;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.73).sin()).collect();
    let weight: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.001).collect();

    let gpu = run_rms_norm_gpu(&ctx, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "rmsnorm_variant");
}

// ===========================================================================
// 13. Metal buffer alignment (256 bytes)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_buffer_alignment_256() {
    // Verify that our alignment helper produces valid 256-byte aligned sizes.
    for &n in &[1u32, 7, 63, 64, 127, 128, 255, 256, 1000, 4096] {
        let raw = (n as u64) * std::mem::size_of::<f32>() as u64;
        let aligned = align_to_256(raw);
        assert_eq!(
            aligned % METAL_BUFFER_ALIGNMENT,
            0,
            "alignment failed for n={n}: raw={raw}, aligned={aligned}"
        );
        assert!(aligned >= raw, "aligned ({aligned}) < raw ({raw}) for n={n}");
    }

    // Also run a GPU kernel with a non-power-of-2 dimension to verify no buffer issues.
    let ctx = create_metal_context();
    let n: u32 = 300;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "alignment_non_pow2");
}

// ===========================================================================
// 14. Workgroup dispatch dimensions
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_workgroup_dispatch() {
    // Verify dispatch calculation: one workgroup per row.
    for &num_rows in &[1u32, 2, 4, 8, 16, 32] {
        let workgroups_x = num_rows;
        assert!(workgroups_x <= 65535, "exceeds Metal max dispatch x-dimension");
    }
    // Verify workgroup size constraints.
    let workgroup_size: u32 = 256;
    assert!(
        workgroup_size <= MAX_THREADS_PER_THREADGROUP,
        "workgroup size {workgroup_size} exceeds Metal max {MAX_THREADS_PER_THREADGROUP}"
    );
    assert_eq!(
        workgroup_size % SIMD_GROUP_SIZE,
        0,
        "workgroup size must be a multiple of SIMD group size ({SIMD_GROUP_SIZE})"
    );

    // Run multi-row dispatch to validate.
    let ctx = create_metal_context();
    let n: u32 = 128;
    let num_rows: u32 = 16;
    let input: Vec<f32> = (0..num_rows * n).map(|i| ((i as f32) * 0.11).sin()).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, num_rows);
    for row in 0..num_rows as usize {
        let s = row * n as usize;
        let e = s + n as usize;
        let cpu = layer_norm_cpu(&input[s..e], &gamma, &beta, NORM_EPSILON);
        assert_close(&gpu[s..e], &cpu, 1e-4, &format!("dispatch_row_{row}"));
    }
}

// ===========================================================================
// 15. Negative gamma (reflective scaling)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_negative_gamma() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
    let gamma: Vec<f32> = vec![-1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "negative_gamma");
}

// ===========================================================================
// 16. Large beta offset
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_large_beta() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![1000.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-3, "large_beta");
}

// ===========================================================================
// 17. All zeros input
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_all_zeros() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = vec![0.0; n as usize];
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    assert_no_nan_inf(&gpu, "all_zeros");
    // All zero input → zero mean, zero var → (0-0)/sqrt(eps) * 1 + 0 = 0
    for (i, &v) in gpu.iter().enumerate() {
        assert!(v.abs() < 1e-3, "all_zeros[{i}]: expected ~0, got {v}");
    }
}

// ===========================================================================
// 18. Non-uniform gamma/beta (per-channel)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_per_channel_affine() {
    let ctx = create_metal_context();
    let n: u32 = 512;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.3).sin()).collect();
    let gamma: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32) * 0.0004).collect();
    let beta: Vec<f32> = (0..n).map(|i| -0.1 + (i as f32) * 0.0002).collect();

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "per_channel_affine");
}

// ===========================================================================
// 19. Small hidden dim (d=32)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_small_hidden_dim() {
    let ctx = create_metal_context();
    let n: u32 = 32;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) - 16.0).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "small_hidden_dim");
}

// ===========================================================================
// 20. Medium hidden dim (d=768, BERT-base)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_medium_hidden_dim() {
    let ctx = create_metal_context();
    let n: u32 = 768;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.003).cos()).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "medium_hidden_dim_768");
}

// ===========================================================================
// 21. d=2048 hidden dim
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_hidden_dim_2048() {
    let ctx = create_metal_context();
    let n: u32 = 2048;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0031).sin()).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-3, "hidden_dim_2048");
}

// ===========================================================================
// 22. RMSNorm multi-row
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_rmsnorm_multi_row() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let num_rows: u32 = 4;
    let input: Vec<f32> = (0..num_rows * n).map(|i| ((i as f32) * 0.37).cos()).collect();
    let weight: Vec<f32> = vec![1.0; n as usize];

    let gpu = run_rms_norm_gpu(&ctx, &input, &weight, n, NORM_EPSILON, num_rows);
    for row in 0..num_rows as usize {
        let s = row * n as usize;
        let e = s + n as usize;
        let cpu = rms_norm_cpu(&input[s..e], &weight, NORM_EPSILON);
        assert_close(&gpu[s..e], &cpu, 1e-4, &format!("rmsnorm_row_{row}"));
    }
}

// ===========================================================================
// 23. RMSNorm with learned weights
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_rmsnorm_learned_weights() {
    let ctx = create_metal_context();
    let n: u32 = 512;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11).sin()).collect();
    let weight: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.001).collect();

    let gpu = run_rms_norm_gpu(&ctx, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "rmsnorm_learned_weights");
}

// ===========================================================================
// 24. Very small epsilon
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_tiny_eps() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 - 2.56).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];
    let tiny_eps: f32 = 1e-12;

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, tiny_eps, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, tiny_eps);
    assert_close(&gpu, &cpu, 1e-4, "tiny_eps");
}

// ===========================================================================
// 25. Large epsilon
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_large_eps() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 - 2.56).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];
    let large_eps: f32 = 1.0;

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, large_eps, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, large_eps);
    assert_close(&gpu, &cpu, 1e-4, "large_eps");
}

// ===========================================================================
// 26. Alternating positive/negative input
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_alternating_signs() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "alternating_signs");
}

// ===========================================================================
// 27. Monotonically increasing input
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_monotonic_input() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "monotonic_input");
    // Monotonic input: normalized output should still be monotonic.
    for i in 1..gpu.len() {
        assert!(gpu[i] >= gpu[i - 1], "monotonicity broken at index {i}");
    }
}

// ===========================================================================
// 28. Symmetry: negate input → negate output (with gamma=1, beta=0)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_symmetry_negation() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 12.8).collect();
    let neg_input: Vec<f32> = input.iter().map(|x| -x).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu_pos = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let gpu_neg = run_layer_norm_gpu(&ctx, &neg_input, &gamma, &beta, n, NORM_EPSILON, 1);
    for (i, (&p, &ne)) in gpu_pos.iter().zip(gpu_neg.iter()).enumerate() {
        let diff = (p + ne).abs();
        assert!(diff < 1e-4, "symmetry[{i}]: pos={p}, neg={ne}, sum={diff}");
    }
}

// ===========================================================================
// 29. Scaled input (multiply by constant → same normalized output)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_scale_invariance() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 3.84).collect();
    let scaled: Vec<f32> = input.iter().map(|x| x * 100.0).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu_orig = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let gpu_scaled = run_layer_norm_gpu(&ctx, &scaled, &gamma, &beta, n, NORM_EPSILON, 1);
    // LayerNorm is scale-invariant: LN(c*x) = LN(x) for constant c.
    assert_close(&gpu_orig, &gpu_scaled, 1e-3, "scale_invariance");
}

// ===========================================================================
// 30. Translation invariance: shift input by constant → same normalized output
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_translation_invariance() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 3.84).collect();
    let shifted: Vec<f32> = input.iter().map(|x| x + 500.0).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu_orig = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let gpu_shifted = run_layer_norm_gpu(&ctx, &shifted, &gamma, &beta, n, NORM_EPSILON, 1);
    // LayerNorm is translation-invariant: LN(x + c) = LN(x).
    assert_close(&gpu_orig, &gpu_shifted, 1e-3, "translation_invariance");
}

// ===========================================================================
// 31. RMSNorm zero input → zero output
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_rmsnorm_zero_input() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = vec![0.0; n as usize];
    let weight: Vec<f32> = vec![1.0; n as usize];

    let gpu = run_rms_norm_gpu(&ctx, &input, &weight, n, NORM_EPSILON, 1);
    assert_no_nan_inf(&gpu, "rmsnorm_zero");
    for (i, &v) in gpu.iter().enumerate() {
        assert!(v.abs() < 1e-3, "rmsnorm_zero[{i}]: expected ~0, got {v}");
    }
}

// ===========================================================================
// 32. RMSNorm large hidden dim (d=4096)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_rmsnorm_large_dim() {
    let ctx = create_metal_context();
    let n: u32 = 4096;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0023).sin()).collect();
    let weight: Vec<f32> = vec![1.0; n as usize];

    let gpu = run_rms_norm_gpu(&ctx, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-3, "rmsnorm_large_dim");
}

// ===========================================================================
// 33. RMSNorm extreme values
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_rmsnorm_extreme_values() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n)
        .map(|i| match i % 3 {
            0 => 1e5,
            1 => -1e5,
            _ => 1e-6,
        })
        .collect();
    let weight: Vec<f32> = vec![1.0; n as usize];

    let gpu = run_rms_norm_gpu(&ctx, &input, &weight, n, NORM_EPSILON, 1);
    assert_no_nan_inf(&gpu, "rmsnorm_extreme");
}

// ===========================================================================
// 34. Multi-row with different distributions per row
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_multi_row_varied() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let num_rows: u32 = 4;
    let mut input = Vec::with_capacity((num_rows * n) as usize);
    // Row 0: uniform 1.0
    input.extend(std::iter::repeat_n(1.0f32, n as usize));
    // Row 1: linear ramp
    input.extend((0..n).map(|i| i as f32));
    // Row 2: sinusoidal
    input.extend((0..n).map(|i| ((i as f32) * 0.1).sin()));
    // Row 3: large values
    input.extend((0..n).map(|i| (i as f32) * 100.0));

    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, num_rows);
    for row in 0..num_rows as usize {
        let s = row * n as usize;
        let e = s + n as usize;
        let cpu = layer_norm_cpu(&input[s..e], &gamma, &beta, NORM_EPSILON);
        assert_close(&gpu[s..e], &cpu, 1e-3, &format!("multi_row_varied_{row}"));
    }
}

// ===========================================================================
// 35. SIMD-width aligned dimension (d=32, exact SIMD group)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_simd_aligned_dim() {
    let ctx = create_metal_context();
    let n: u32 = SIMD_GROUP_SIZE; // 32 — exactly one SIMD group
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 8.0).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "simd_aligned_dim");
}

// ===========================================================================
// 36. Non-power-of-2 dimension (d=300)
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_non_power_of_2() {
    let ctx = create_metal_context();
    let n: u32 = 300;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).sin()).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "non_pow2_dim");
}

// ===========================================================================
// 37. Output idempotence: LN(LN(x)) has mean~0, var~1
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_idempotence() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 12.8).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let first = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let second = run_layer_norm_gpu(&ctx, &first, &gamma, &beta, n, NORM_EPSILON, 1);
    // After double normalization output should still have mean~0, var~1.
    let mean: f32 = second.iter().sum::<f32>() / second.len() as f32;
    let var: f32 =
        second.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / second.len() as f32;
    assert!(mean.abs() < 1e-3, "idempotence mean: {mean}");
    assert!((var - 1.0).abs() < 1e-2, "idempotence var: {var}");
}

// ===========================================================================
// 38. Combined gamma+beta with multi-row
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_layernorm_affine_multi_row() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let num_rows: u32 = 8;
    let input: Vec<f32> = (0..num_rows * n).map(|i| ((i as f32) * 0.23).sin()).collect();
    let gamma: Vec<f32> = (0..n).map(|i| 0.9 + (i as f32) * 0.0008).collect();
    let beta: Vec<f32> = (0..n).map(|i| -0.5 + (i as f32) * 0.004).collect();

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, num_rows);
    for row in 0..num_rows as usize {
        let s = row * n as usize;
        let e = s + n as usize;
        let cpu = layer_norm_cpu(&input[s..e], &gamma, &beta, NORM_EPSILON);
        assert_close(&gpu[s..e], &cpu, 1e-4, &format!("affine_multi_row_{row}"));
    }
}
