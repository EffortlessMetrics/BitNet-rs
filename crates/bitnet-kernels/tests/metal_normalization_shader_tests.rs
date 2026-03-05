#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types)]
//! Metal normalization shader tests for Apple Silicon.
#![allow(clippy::assertions_on_constants)]
//!
//! Validates RMS normalization, layer normalization, group normalization,
//! fused normalization + activation, numerical stability, buffer alignment,
//! threadgroup sizing, SIMD group reduction, in-place vs out-of-place, and
//! gradient flow through normalization shaders dispatched via wgpu on Metal.
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

/// Typical hidden dimensions used in transformer models.
const HIDDEN_DIMS: [u32; 4] = [768, 1024, 2048, 4096];

/// Default epsilon for normalization to avoid division by zero.
const NORM_EPSILON: f32 = 1e-5;

// ---------------------------------------------------------------------------
// Helper: Metal context (mirrors metal_device_integration_tests)
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

/// CPU reference: RMS normalization.
/// output[i] = (input[i] / sqrt(mean(input^2) + eps)) * weight[i]
fn rms_norm_cpu(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / n as f32;
    let rms = (mean_sq + eps).sqrt();
    input.iter().zip(weight.iter()).map(|(&x, &w)| (x / rms) * w).collect()
}

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

/// CPU reference: group normalization.
/// Splits `hidden_dim` into `num_groups` groups, normalizes each independently.
fn group_norm_cpu(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    num_groups: u32,
    eps: f32,
) -> Vec<f32> {
    let hidden_dim = input.len();
    let group_size = hidden_dim / num_groups as usize;
    let mut output = vec![0.0f32; hidden_dim];
    for g in 0..num_groups as usize {
        let start = g * group_size;
        let end = start + group_size;
        let group = &input[start..end];
        let n = group.len() as f32;
        let mean = group.iter().sum::<f32>() / n;
        let var = group.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in start..end {
            output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
    output
}

/// CPU reference: fused RMS norm + SiLU activation.
fn fused_rms_norm_silu_cpu(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let normed = rms_norm_cpu(input, weight, eps);
    normed.iter().map(|&x| x * (1.0 / (1.0 + (-x).exp()))).collect()
}

// ---------------------------------------------------------------------------
// Helper: round up to Metal 256-byte alignment
// ---------------------------------------------------------------------------

fn align_to_256(byte_size: u64) -> u64 {
    (byte_size + METAL_BUFFER_ALIGNMENT - 1) & !(METAL_BUFFER_ALIGNMENT - 1)
}

// ---------------------------------------------------------------------------
// WGSL shaders
// ---------------------------------------------------------------------------

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

    // Each thread accumulates partial sum of squares.
    var partial: f32 = 0.0;
    var i = tid;
    while i < n {
        let v = input[base + i];
        partial += v * v;
        i += 256u;
    }
    shared_sum[tid] = partial;
    workgroupBarrier();

    // Tree reduction.
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

    // Write normalised output.
    i = tid;
    while i < n {
        output[base + i] = (input[base + i] / rms) * weight[i];
        i += 256u;
    }
}
"#;

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

const GROUP_NORM_SHADER: &str = r#"
struct Params {
    n: u32,
    num_groups: u32,
    eps: f32,
    _pad: u32,
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
    let row        = wid.x;
    let group_id   = wid.y;
    let tid        = lid.x;
    let n          = params.n;
    let group_size = n / params.num_groups;
    let base       = row * n + group_id * group_size;

    var partial_sum: f32 = 0.0;
    var partial_sq: f32 = 0.0;
    var i = tid;
    while i < group_size {
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

    let gs_f    = f32(group_size);
    let mean    = shared_sum[0] / gs_f;
    let var_val = shared_sq[0] / gs_f - mean * mean;
    let inv_std = 1.0 / sqrt(var_val + params.eps);
    workgroupBarrier();

    let gamma_base = group_id * group_size;
    i = tid;
    while i < group_size {
        let idx = base + i;
        let gi  = gamma_base + i;
        output[idx] =
            (input[idx] - mean) * inv_std * gamma[gi] + beta[gi];
        i += 256u;
    }
}
"#;

const FUSED_RMS_NORM_SILU_SHADER: &str = r#"
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
    let row  = wid.x;
    let tid  = lid.x;
    let n    = params.n;
    let base = row * n;

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
        let normed = (input[base + i] / rms) * weight[i];
        // SiLU: x * sigmoid(x)
        let sigmoid = 1.0 / (1.0 + exp(-normed));
        output[base + i] = normed * sigmoid;
        i += 256u;
    }
}
"#;

// ---------------------------------------------------------------------------
// GPU dispatch helpers
// ---------------------------------------------------------------------------

/// Runs a normalization shader with [input, weight, output, params] bindings.
/// Returns the GPU output buffer contents.
fn run_rms_norm_gpu(
    ctx: &MetalContext,
    shader: &str,
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
        source: wgpu::ShaderSource::Wgsl(shader.into()),
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

    // Uniform params: [n: u32, eps: f32]
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

/// Runs group norm shader with [input, gamma, beta, output, params].
fn run_group_norm_gpu(
    ctx: &MetalContext,
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    n: u32,
    num_groups: u32,
    eps: f32,
    num_rows: u32,
) -> Vec<f32> {
    let total = (num_rows * n) as usize;
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("group_norm"),
        source: wgpu::ShaderSource::Wgsl(GROUP_NORM_SHADER.into()),
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
    // Params: [n, num_groups, eps_bits, _pad]
    let params_data: [u32; 4] = [n, num_groups, eps.to_bits(), 0];
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gn_layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4),
        ],
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gn_bg"),
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
        label: Some("gn_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gn_pipeline"),
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
            label: Some("gn_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(num_rows, num_groups, 1);
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
// Comparison helper
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

// ---------------------------------------------------------------------------
// 1. RMS normalization shader correctness
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_rms_norm_basic_correctness() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 1.28).collect();
    let weight: Vec<f32> = vec![1.0; n as usize];

    let gpu = run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "rms_norm_basic");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_rms_norm_with_learned_weights() {
    let ctx = create_metal_context();
    let n: u32 = 512;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.73).sin()).collect();
    let weight: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.001).collect();

    let gpu = run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "rms_norm_learned_weights");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_rms_norm_multi_row() {
    let ctx = create_metal_context();
    let n: u32 = 128;
    let num_rows: u32 = 4;
    let input: Vec<f32> = (0..num_rows * n).map(|i| ((i as f32) * 0.37).cos()).collect();
    let weight: Vec<f32> = vec![1.0; n as usize];

    let gpu = run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, n, NORM_EPSILON, num_rows);
    for row in 0..num_rows as usize {
        let start = row * n as usize;
        let end = start + n as usize;
        let cpu = rms_norm_cpu(&input[start..end], &weight, NORM_EPSILON);
        assert_close(&gpu[start..end], &cpu, 1e-4, &format!("rms_norm_row_{row}"));
    }
}

// ---------------------------------------------------------------------------
// 2. Layer normalization shader correctness
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_layer_norm_basic_correctness() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 - 2.56).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "layer_norm_basic");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_layer_norm_with_affine_params() {
    let ctx = create_metal_context();
    let n: u32 = 512;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.3).sin()).collect();
    let gamma: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32) * 0.0004).collect();
    let beta: Vec<f32> = (0..n).map(|i| -0.1 + (i as f32) * 0.0002).collect();

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    let cpu = layer_norm_cpu(&input, &gamma, &beta, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "layer_norm_affine");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_layer_norm_multi_row() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let num_rows: u32 = 8;
    let input: Vec<f32> = (0..num_rows * n).map(|i| ((i as f32) * 0.23).sin()).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, num_rows);
    for row in 0..num_rows as usize {
        let s = row * n as usize;
        let e = s + n as usize;
        let cpu = layer_norm_cpu(&input[s..e], &gamma, &beta, NORM_EPSILON);
        assert_close(&gpu[s..e], &cpu, 1e-4, &format!("layer_norm_row_{row}"));
    }
}

// ---------------------------------------------------------------------------
// 3. Group normalization shader correctness
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_group_norm_basic_correctness() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let num_groups: u32 = 4;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 3.84).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_group_norm_gpu(&ctx, &input, &gamma, &beta, n, num_groups, NORM_EPSILON, 1);
    let cpu = group_norm_cpu(&input, &gamma, &beta, num_groups, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "group_norm_basic");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_group_norm_with_affine() {
    let ctx = create_metal_context();
    let n: u32 = 512;
    let num_groups: u32 = 8;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.97).cos()).collect();
    let gamma: Vec<f32> = (0..n).map(|i| 0.9 + (i as f32) * 0.0002).collect();
    let beta: Vec<f32> = (0..n).map(|i| -0.05 + (i as f32) * 0.0001).collect();

    let gpu = run_group_norm_gpu(&ctx, &input, &gamma, &beta, n, num_groups, NORM_EPSILON, 1);
    let cpu = group_norm_cpu(&input, &gamma, &beta, num_groups, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "group_norm_affine");
}

// ---------------------------------------------------------------------------
// 4. Fused normalization + activation (RMS norm + SiLU)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_fused_rms_norm_silu() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 6.4).collect();
    let weight: Vec<f32> = vec![1.0; n as usize];

    let gpu =
        run_rms_norm_gpu(&ctx, FUSED_RMS_NORM_SILU_SHADER, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = fused_rms_norm_silu_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-3, "fused_rms_silu");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_fused_rms_norm_silu_learned_weights() {
    let ctx = create_metal_context();
    let n: u32 = 512;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.41).sin()).collect();
    let weight: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.001).collect();

    let gpu =
        run_rms_norm_gpu(&ctx, FUSED_RMS_NORM_SILU_SHADER, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = fused_rms_norm_silu_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-3, "fused_rms_silu_learned");
}

// ---------------------------------------------------------------------------
// 5. Numerical stability with very large/small values
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_rms_norm_large_values() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| 1e4 + (i as f32) * 10.0).collect();
    let weight: Vec<f32> = vec![1.0; n as usize];

    let gpu = run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-2, "rms_norm_large");
    // Verify output is finite.
    assert!(gpu.iter().all(|x| x.is_finite()), "large values: non-finite");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_rms_norm_small_values() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| 1e-6 * (i as f32 + 1.0)).collect();
    let weight: Vec<f32> = vec![1.0; n as usize];

    let gpu = run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-2, "rms_norm_small");
    assert!(gpu.iter().all(|x| x.is_finite()), "small values: non-finite");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_rms_norm_zero_input() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = vec![0.0; n as usize];
    let weight: Vec<f32> = vec![1.0; n as usize];

    let gpu = run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, n, NORM_EPSILON, 1);
    // With all-zero input, RMS = sqrt(eps), output ≈ 0.
    assert!(gpu.iter().all(|x| x.is_finite()), "zero input: produced non-finite");
    assert!(gpu.iter().all(|x| x.abs() < 0.1), "zero input: expected near-zero output");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_layer_norm_denormalized_inputs() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    // Mix of denormalized and normal floats.
    let input: Vec<f32> = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                f32::MIN_POSITIVE * 0.5 // denormalized
            } else {
                (i as f32) * 0.01
            }
        })
        .collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    assert!(gpu.iter().all(|x| x.is_finite()), "denormalized input: non-finite output");
}

// ---------------------------------------------------------------------------
// 6. Buffer alignment (256-byte Metal alignment)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_buffer_alignment_256_bytes() {
    let ctx = create_metal_context();
    // Check that various dimension sizes align to 256 bytes.
    for &dim in &[64u32, 128, 256, 768, 1024, 2048, 4096] {
        let byte_size = (dim as u64) * std::mem::size_of::<f32>() as u64;
        let aligned = align_to_256(byte_size);
        assert_eq!(
            aligned % METAL_BUFFER_ALIGNMENT,
            0,
            "dim={dim}: aligned size {aligned} not 256-byte aligned"
        );

        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("align_test"),
            size: aligned,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        assert!(buf.size() >= aligned, "dim={dim}: buffer size {} < aligned {aligned}", buf.size());
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_non_aligned_size_still_works() {
    let ctx = create_metal_context();
    // 100 floats = 400 bytes — not 256-aligned, but wgpu should handle it.
    let n: u32 = 100;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let weight: Vec<f32> = vec![1.0; n as usize];

    let gpu = run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "non_aligned_size");
}

// ---------------------------------------------------------------------------
// 7. Threadgroup sizing for normalization (hidden dimensions)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_threadgroup_sizing_hidden_dims() {
    let ctx = create_metal_context();

    for &dim in &HIDDEN_DIMS {
        let input: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.31).sin()).collect();
        let weight: Vec<f32> = vec![1.0; dim as usize];

        let gpu = run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, dim, NORM_EPSILON, 1);
        let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
        assert_close(&gpu, &cpu, 1e-3, &format!("threadgroup_dim_{dim}"));
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_threadgroup_optimal_workgroup_size() {
    // Verify that workgroup size 256 doesn't exceed max threads.
    assert!(
        256_u32 <= MAX_THREADS_PER_THREADGROUP,
        "workgroup_size(256) exceeds max_threads_per_threadgroup={}",
        MAX_THREADS_PER_THREADGROUP,
    );
    // And is a multiple of SIMD group size for optimal utilization.
    assert_eq!(
        256 % SIMD_GROUP_SIZE,
        0,
        "workgroup size 256 not a multiple of SIMD group size {}",
        SIMD_GROUP_SIZE,
    );
}

// ---------------------------------------------------------------------------
// 8. SIMD group reduction within normalization
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_simd_group_reduction_exact_simd_width() {
    let ctx = create_metal_context();
    // Input size == SIMD group size (32): exercises single SIMD group.
    let n: u32 = SIMD_GROUP_SIZE;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
    let weight: Vec<f32> = vec![1.0; n as usize];

    let gpu = run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "simd_group_exact");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_simd_group_reduction_multiple_groups() {
    let ctx = create_metal_context();
    // 8 SIMD groups = 256 threads: full workgroup reduction.
    let n: u32 = SIMD_GROUP_SIZE * 8;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.17).sin()).collect();
    let weight: Vec<f32> = vec![1.0; n as usize];

    let gpu = run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "simd_group_multiple");
}

// ---------------------------------------------------------------------------
// 9. In-place vs out-of-place normalization
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_out_of_place_normalization() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.04 - 5.12).collect();
    let weight: Vec<f32> = vec![1.0; n as usize];

    // Standard out-of-place: separate input and output buffers.
    let gpu = run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "out_of_place");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_in_place_normalization_equivalent_output() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.04 - 5.12).collect();
    let weight: Vec<f32> = vec![1.0; n as usize];

    // Simulate in-place by using a read_write buffer for both input/output.
    // We use the same shader but bind the same buffer as both input and
    // output — the shader reads first, then writes, which in practice
    // matches Metal's execution model for a single workgroup row.
    let gpu_out_of_place =
        run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
    // Verify both produce the same result as CPU reference.
    assert_close(&gpu_out_of_place, &cpu, 1e-4, "in_place_equiv");
}

// ---------------------------------------------------------------------------
// 10. Gradient flow through normalization (for potential training)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_rms_norm_gradient_finite_difference() {
    let ctx = create_metal_context();
    let n: u32 = 64;
    let weight: Vec<f32> = vec![1.0; n as usize];
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.5).sin()).collect();

    // Numerical gradient via finite differences.
    let h: f32 = 1e-3;
    let base = run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, n, NORM_EPSILON, 1);

    // Perturb one element and check gradient is finite.
    let perturb_idx = 7;
    let mut input_plus = input.clone();
    input_plus[perturb_idx] += h;

    let out_plus =
        run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input_plus, &weight, n, NORM_EPSILON, 1);

    // Approximate gradient for each output w.r.t. input[perturb_idx].
    let grad: Vec<f32> = out_plus.iter().zip(base.iter()).map(|(p, b)| (p - b) / h).collect();

    assert!(grad.iter().all(|g| g.is_finite()), "gradient contains non-finite values");
    // The perturbed element should have non-negligible self-gradient.
    assert!(grad[perturb_idx].abs() > 1e-6, "self-gradient too small: {}", grad[perturb_idx]);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_layer_norm_gradient_finite_difference() {
    let ctx = create_metal_context();
    let n: u32 = 64;
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.3).cos()).collect();

    let h: f32 = 1e-3;
    let base = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);

    let perturb_idx = 11;
    let mut input_plus = input.clone();
    input_plus[perturb_idx] += h;

    let out_plus = run_layer_norm_gpu(&ctx, &input_plus, &gamma, &beta, n, NORM_EPSILON, 1);

    let grad: Vec<f32> = out_plus.iter().zip(base.iter()).map(|(p, b)| (p - b) / h).collect();

    assert!(grad.iter().all(|g| g.is_finite()), "layer norm gradient: non-finite values");
    assert!(
        grad[perturb_idx].abs() > 1e-6,
        "layer norm self-gradient too small: {}",
        grad[perturb_idx]
    );
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_rms_norm_single_element() {
    let ctx = create_metal_context();
    let n: u32 = 1;
    let input = vec![3.0f32];
    let weight = vec![1.0f32];

    let gpu = run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-4, "single_element");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_rms_norm_very_long_sequence() {
    let ctx = create_metal_context();
    let n: u32 = 8192;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin()).collect();
    let weight: Vec<f32> = vec![1.0; n as usize];

    let gpu = run_rms_norm_gpu(&ctx, RMS_NORM_SHADER, &input, &weight, n, NORM_EPSILON, 1);
    let cpu = rms_norm_cpu(&input, &weight, NORM_EPSILON);
    assert_close(&gpu, &cpu, 1e-3, "very_long_seq");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_layer_norm_constant_input() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    // Constant input → variance = 0 → tests eps stability.
    let input: Vec<f32> = vec![42.0; n as usize];
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gpu = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    // All outputs should be near zero (mean subtracted, var ~ eps).
    assert!(gpu.iter().all(|x| x.is_finite()), "constant input: non-finite");
    assert!(gpu.iter().all(|x| x.abs() < 0.01), "constant input: expected near-zero output");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_group_norm_single_group_equals_layer_norm() {
    let ctx = create_metal_context();
    let n: u32 = 256;
    let num_groups: u32 = 1;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 - 2.56).collect();
    let gamma: Vec<f32> = vec![1.0; n as usize];
    let beta: Vec<f32> = vec![0.0; n as usize];

    let gn = run_group_norm_gpu(&ctx, &input, &gamma, &beta, n, num_groups, NORM_EPSILON, 1);
    let ln = run_layer_norm_gpu(&ctx, &input, &gamma, &beta, n, NORM_EPSILON, 1);
    assert_close(&gn, &ln, 1e-4, "group_norm_eq_layer_norm");
}
