#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types)]
#![allow(clippy::manual_div_ceil, clippy::unnecessary_cast, clippy::needless_range_loop)]
//! Metal RoPE (Rotary Position Embedding) shader tests for Apple Silicon.
//!
//! Validates rotation angles, frequency computation, sin/cos cache,
//! batch consistency, NTK/YaRN scaling, buffer alignment, numerical
//! stability, and interleaved vs contiguous pair layouts dispatched
//! via WGSL compute shaders on the Metal backend.
//!
//! All tests are `#[ignore]` because CI runs on Linux.

#![cfg(target_os = "macos")]

use std::f32::consts::PI;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Metal requires 256-byte buffer alignment for optimal performance.
const METAL_BUFFER_ALIGNMENT: u64 = 256;

/// Default RoPE base frequency (GPT-NeoX / LLaMA).
const DEFAULT_BASE: f32 = 10_000.0;

/// Tolerance for floating-point comparisons on GPU.
const GPU_TOLERANCE: f32 = 1e-4;

/// Stricter tolerance for analytical checks.
const STRICT_TOLERANCE: f32 = 1e-5;

// ---------------------------------------------------------------------------
// WGSL compute shader: RoPE rotation (contiguous pair layout)
// ---------------------------------------------------------------------------

const ROPE_SHADER: &str = r#"
struct Params {
    dim: u32,
    half_dim: u32,
    seq_len: u32,
    base_freq: f32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> positions: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.seq_len * params.half_dim {
        return;
    }
    let seq_idx = idx / params.half_dim;
    let pair_idx = idx % params.half_dim;
    let pos = f32(positions[seq_idx]);

    let exponent = -f32(2u * pair_idx) / f32(params.dim);
    let theta = pow(params.base_freq, exponent);
    let angle = pos * theta;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    let base = seq_idx * params.dim + pair_idx * 2u;
    let x0 = data[base];
    let x1 = data[base + 1u];

    data[base]      = x0 * cos_val - x1 * sin_val;
    data[base + 1u] = x0 * sin_val + x1 * cos_val;
}
"#;

// ---------------------------------------------------------------------------
// WGSL compute shader: frequency table builder
// ---------------------------------------------------------------------------

const FREQ_TABLE_SHADER: &str = r#"
struct Params {
    dim: u32,
    half_dim: u32,
    max_seq: u32,
    base_freq: f32,
}

@group(0) @binding(0) var<storage, read_write> cos_table: array<f32>;
@group(0) @binding(1) var<storage, read_write> sin_table: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.max_seq * params.half_dim {
        return;
    }
    let pos = idx / params.half_dim;
    let i = idx % params.half_dim;

    let exponent = -f32(2u * i) / f32(params.dim);
    let theta = pow(params.base_freq, exponent);
    let angle = f32(pos) * theta;

    cos_table[idx] = cos(angle);
    sin_table[idx] = sin(angle);
}
"#;

// ---------------------------------------------------------------------------
// WGSL compute shader: interleaved RoPE layout
// ---------------------------------------------------------------------------

const ROPE_INTERLEAVED_SHADER: &str = r#"
struct Params {
    dim: u32,
    half_dim: u32,
    seq_len: u32,
    base_freq: f32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> positions: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.seq_len * params.half_dim {
        return;
    }
    let seq_idx = idx / params.half_dim;
    let pair_idx = idx % params.half_dim;
    let pos = f32(positions[seq_idx]);

    let exponent = -f32(2u * pair_idx) / f32(params.dim);
    let theta = pow(params.base_freq, exponent);
    let angle = pos * theta;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    // Interleaved: even indices are first half, odd indices are second half
    let x0_idx = seq_idx * params.dim + pair_idx;
    let x1_idx = seq_idx * params.dim + pair_idx + params.half_dim;
    let x0 = data[x0_idx];
    let x1 = data[x1_idx];

    data[x0_idx] = x0 * cos_val - x1 * sin_val;
    data[x1_idx] = x0 * sin_val + x1 * cos_val;
}
"#;

// ---------------------------------------------------------------------------
// WGSL compute shader: NTK-aware scaled RoPE
// ---------------------------------------------------------------------------

const ROPE_NTK_SHADER: &str = r#"
struct Params {
    dim: u32,
    half_dim: u32,
    seq_len: u32,
    base_freq: f32,
    scale_factor: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> positions: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.seq_len * params.half_dim {
        return;
    }
    let seq_idx = idx / params.half_dim;
    let pair_idx = idx % params.half_dim;
    let pos = f32(positions[seq_idx]);

    // NTK scaling: base' = base * scale_factor^(dim/(dim-2))
    let ntk_base = params.base_freq * pow(params.scale_factor, f32(params.dim) / f32(params.dim - 2u));
    let exponent = -f32(2u * pair_idx) / f32(params.dim);
    let theta = pow(ntk_base, exponent);
    let angle = pos * theta;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    let base = seq_idx * params.dim + pair_idx * 2u;
    let x0 = data[base];
    let x1 = data[base + 1u];

    data[base]      = x0 * cos_val - x1 * sin_val;
    data[base + 1u] = x0 * sin_val + x1 * cos_val;
}
"#;

// ---------------------------------------------------------------------------
// WGSL compute shader: YaRN interpolated RoPE
// ---------------------------------------------------------------------------

const ROPE_YARN_SHADER: &str = r#"
struct Params {
    dim: u32,
    half_dim: u32,
    seq_len: u32,
    base_freq: f32,
    scale_factor: f32,
    attn_factor: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> positions: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.seq_len * params.half_dim {
        return;
    }
    let seq_idx = idx / params.half_dim;
    let pair_idx = idx % params.half_dim;
    let pos = f32(positions[seq_idx]);

    // YaRN: linear interpolation of frequency + attention scaling
    let exponent = -f32(2u * pair_idx) / f32(params.dim);
    let base_theta = pow(params.base_freq, exponent);
    let scaled_pos = pos / params.scale_factor;
    let angle = scaled_pos * base_theta;
    let cos_val = cos(angle) * params.attn_factor;
    let sin_val = sin(angle) * params.attn_factor;

    let base = seq_idx * params.dim + pair_idx * 2u;
    let x0 = data[base];
    let x1 = data[base + 1u];

    data[base]      = x0 * cos_val - x1 * sin_val;
    data[base + 1u] = x0 * sin_val + x1 * cos_val;
}
"#;

// ---------------------------------------------------------------------------
// Test struct
// ---------------------------------------------------------------------------

/// Configuration for a Metal RoPE test case.
#[derive(Debug, Clone)]
struct MetalRopeTestCase {
    /// Total embedding dimension (must be even).
    input_dim: u32,
    /// Head dimension for per-head rotation (must be even, ≤ input_dim).
    head_dim: u32,
    /// Sequence length (number of positions).
    seq_len: u32,
    /// Base frequency for RoPE (default 10000).
    base_freq: f32,
    /// Human-readable description of expected behavior.
    expected: &'static str,
}

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
// CPU reference helpers (pure Rust, no SIMD)
// ---------------------------------------------------------------------------

/// Compute a single RoPE frequency: θ_i = base^(−2i/d).
fn rope_freq(dim: usize, pair_idx: usize, base: f32) -> f32 {
    let exponent = -(2.0 * pair_idx as f32) / dim as f32;
    base.powf(exponent)
}

/// Compute the rotation angle for a given position and dimension pair.
fn rope_angle(dim: usize, pair_idx: usize, pos: usize, base: f32) -> f32 {
    pos as f32 * rope_freq(dim, pair_idx, base)
}

/// Build reference cos/sin tables on CPU.
fn build_reference_tables(dim: usize, max_seq: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half_dim = dim / 2;
    let mut cos_table = Vec::with_capacity(max_seq * half_dim);
    let mut sin_table = Vec::with_capacity(max_seq * half_dim);
    for pos in 0..max_seq {
        for i in 0..half_dim {
            let angle = rope_angle(dim, i, pos, base);
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
    }
    (cos_table, sin_table)
}

/// Apply RoPE to data in-place on CPU (contiguous pair layout).
fn apply_rope_cpu(data: &mut [f32], dim: usize, positions: &[u32], base: f32) {
    let half_dim = dim / 2;
    for (seq_idx, &pos) in positions.iter().enumerate() {
        for i in 0..half_dim {
            let angle = rope_angle(dim, i, pos as usize, base);
            let cos_val = angle.cos();
            let sin_val = angle.sin();
            let base_idx = seq_idx * dim + i * 2;
            let x0 = data[base_idx];
            let x1 = data[base_idx + 1];
            data[base_idx] = x0 * cos_val - x1 * sin_val;
            data[base_idx + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

/// Apply RoPE with interleaved layout on CPU.
fn apply_rope_interleaved_cpu(data: &mut [f32], dim: usize, positions: &[u32], base: f32) {
    let half_dim = dim / 2;
    for (seq_idx, &pos) in positions.iter().enumerate() {
        for i in 0..half_dim {
            let angle = rope_angle(dim, i, pos as usize, base);
            let cos_val = angle.cos();
            let sin_val = angle.sin();
            let x0_idx = seq_idx * dim + i;
            let x1_idx = seq_idx * dim + i + half_dim;
            let x0 = data[x0_idx];
            let x1 = data[x1_idx];
            data[x0_idx] = x0 * cos_val - x1 * sin_val;
            data[x1_idx] = x0 * sin_val + x1 * cos_val;
        }
    }
}

/// Apply NTK-scaled RoPE on CPU.
fn apply_rope_ntk_cpu(
    data: &mut [f32],
    dim: usize,
    positions: &[u32],
    base: f32,
    scale_factor: f32,
) {
    let ntk_base = base * scale_factor.powf(dim as f32 / (dim as f32 - 2.0));
    let half_dim = dim / 2;
    for (seq_idx, &pos) in positions.iter().enumerate() {
        for i in 0..half_dim {
            let angle = rope_angle(dim, i, pos as usize, ntk_base);
            let cos_val = angle.cos();
            let sin_val = angle.sin();
            let base_idx = seq_idx * dim + i * 2;
            let x0 = data[base_idx];
            let x1 = data[base_idx + 1];
            data[base_idx] = x0 * cos_val - x1 * sin_val;
            data[base_idx + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

/// Apply YaRN-scaled RoPE on CPU.
fn apply_rope_yarn_cpu(
    data: &mut [f32],
    dim: usize,
    positions: &[u32],
    base: f32,
    scale_factor: f32,
    attn_factor: f32,
) {
    let half_dim = dim / 2;
    for (seq_idx, &pos) in positions.iter().enumerate() {
        for i in 0..half_dim {
            let scaled_pos = pos as f32 / scale_factor;
            let angle = scaled_pos * rope_freq(dim, i, base);
            let cos_val = angle.cos() * attn_factor;
            let sin_val = angle.sin() * attn_factor;
            let base_idx = seq_idx * dim + i * 2;
            let x0 = data[base_idx];
            let x1 = data[base_idx + 1];
            data[base_idx] = x0 * cos_val - x1 * sin_val;
            data[base_idx + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

/// Pad buffer size up to Metal 256-byte alignment.
fn align_to_metal(size: u64) -> u64 {
    (size + METAL_BUFFER_ALIGNMENT - 1) & !(METAL_BUFFER_ALIGNMENT - 1)
}

// ---------------------------------------------------------------------------
// GPU execution helpers
// ---------------------------------------------------------------------------

/// Run the RoPE shader on Metal and return the result buffer.
fn run_rope_gpu(
    ctx: &MetalContext,
    shader_src: &str,
    data: &[f32],
    positions: &[u32],
    dim: u32,
    seq_len: u32,
    base_freq: f32,
) -> Vec<f32> {
    let half_dim = dim / 2;

    let data_bytes = bytemuck::cast_slice::<f32, u8>(data);
    let buf_data = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rope_data"),
        contents: data_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        dim: u32,
        half_dim: u32,
        seq_len: u32,
        base_freq: f32,
    }
    let params = Params { dim, half_dim, seq_len, base_freq };
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rope_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let pos_bytes = bytemuck::cast_slice::<u32, u8>(positions);
    let buf_pos = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rope_positions"),
        contents: pos_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rope_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rope_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rope_pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rope_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rope_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_data.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_pos.as_entire_binding() },
        ],
    });

    let total_threads = seq_len * half_dim;
    let workgroups = (total_threads + 63) / 64;

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    let readback_size = align_to_metal((data.len() * 4) as u64);
    let buf_readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rope_readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_data, 0, &buf_readback, 0, (data.len() * 4) as u64);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = buf_readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&mapped[..data.len() * 4]).to_vec();
    drop(mapped);
    buf_readback.unmap();
    result
}

/// Run the NTK RoPE shader with extended params.
fn run_rope_ntk_gpu(
    ctx: &MetalContext,
    data: &[f32],
    positions: &[u32],
    dim: u32,
    seq_len: u32,
    base_freq: f32,
    scale_factor: f32,
) -> Vec<f32> {
    let half_dim = dim / 2;

    let data_bytes = bytemuck::cast_slice::<f32, u8>(data);
    let buf_data = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ntk_data"),
        contents: data_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct NtkParams {
        dim: u32,
        half_dim: u32,
        seq_len: u32,
        base_freq: f32,
        scale_factor: f32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }
    let params =
        NtkParams { dim, half_dim, seq_len, base_freq, scale_factor, _pad0: 0, _pad1: 0, _pad2: 0 };
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ntk_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let pos_bytes = bytemuck::cast_slice::<u32, u8>(positions);
    let buf_pos = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ntk_positions"),
        contents: pos_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ntk_shader"),
        source: wgpu::ShaderSource::Wgsl(ROPE_NTK_SHADER.into()),
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ntk_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ntk_pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ntk_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ntk_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_data.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_pos.as_entire_binding() },
        ],
    });

    let total_threads = seq_len * half_dim;
    let workgroups = (total_threads + 63) / 64;

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    let readback_size = align_to_metal((data.len() * 4) as u64);
    let buf_readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ntk_readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_data, 0, &buf_readback, 0, (data.len() * 4) as u64);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = buf_readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&mapped[..data.len() * 4]).to_vec();
    drop(mapped);
    buf_readback.unmap();
    result
}

/// Run the YaRN RoPE shader with extended params.
fn run_rope_yarn_gpu(
    ctx: &MetalContext,
    data: &[f32],
    positions: &[u32],
    dim: u32,
    seq_len: u32,
    base_freq: f32,
    scale_factor: f32,
    attn_factor: f32,
) -> Vec<f32> {
    let half_dim = dim / 2;

    let data_bytes = bytemuck::cast_slice::<f32, u8>(data);
    let buf_data = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("yarn_data"),
        contents: data_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct YarnParams {
        dim: u32,
        half_dim: u32,
        seq_len: u32,
        base_freq: f32,
        scale_factor: f32,
        attn_factor: f32,
        _pad0: u32,
        _pad1: u32,
    }
    let params = YarnParams {
        dim,
        half_dim,
        seq_len,
        base_freq,
        scale_factor,
        attn_factor,
        _pad0: 0,
        _pad1: 0,
    };
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("yarn_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let pos_bytes = bytemuck::cast_slice::<u32, u8>(positions);
    let buf_pos = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("yarn_positions"),
        contents: pos_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("yarn_shader"),
        source: wgpu::ShaderSource::Wgsl(ROPE_YARN_SHADER.into()),
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("yarn_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("yarn_pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("yarn_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("yarn_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_data.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_pos.as_entire_binding() },
        ],
    });

    let total_threads = seq_len * half_dim;
    let workgroups = (total_threads + 63) / 64;

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    let readback_size = align_to_metal((data.len() * 4) as u64);
    let buf_readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("yarn_readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_data, 0, &buf_readback, 0, (data.len() * 4) as u64);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = buf_readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&mapped[..data.len() * 4]).to_vec();
    drop(mapped);
    buf_readback.unmap();
    result
}

/// Run the frequency table shader and return (cos_table, sin_table).
fn run_freq_table_gpu(
    ctx: &MetalContext,
    dim: u32,
    max_seq: u32,
    base_freq: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = dim / 2;
    let table_len = (max_seq * half_dim) as usize;

    let zeros = vec![0.0f32; table_len];
    let zeros_bytes = bytemuck::cast_slice::<f32, u8>(&zeros);

    let buf_cos = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cos_table"),
        contents: zeros_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let buf_sin = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sin_table"),
        contents: zeros_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct FreqParams {
        dim: u32,
        half_dim: u32,
        max_seq: u32,
        base_freq: f32,
    }
    let params = FreqParams { dim, half_dim, max_seq, base_freq };
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("freq_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("freq_shader"),
        source: wgpu::ShaderSource::Wgsl(FREQ_TABLE_SHADER.into()),
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("freq_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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

    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("freq_pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("freq_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("freq_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_cos.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_sin.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_params.as_entire_binding() },
        ],
    });

    let total_threads = max_seq * half_dim;
    let workgroups = (total_threads + 63) / 64;

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    let byte_len = (table_len * 4) as u64;
    let readback_size = align_to_metal(byte_len);
    let buf_cos_rb = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cos_readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buf_sin_rb = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sin_readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_cos, 0, &buf_cos_rb, 0, byte_len);
    encoder.copy_buffer_to_buffer(&buf_sin, 0, &buf_sin_rb, 0, byte_len);
    ctx.queue.submit(Some(encoder.finish()));

    // Read cos
    let cos_slice = buf_cos_rb.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    cos_slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let cos_mapped = cos_slice.get_mapped_range();
    let cos_result: Vec<f32> = bytemuck::cast_slice(&cos_mapped[..table_len * 4]).to_vec();
    drop(cos_mapped);
    buf_cos_rb.unmap();

    // Read sin
    let sin_slice = buf_sin_rb.slice(..);
    let (tx2, rx2) = std::sync::mpsc::channel();
    sin_slice.map_async(wgpu::MapMode::Read, move |r| tx2.send(r).unwrap());
    ctx.device.poll(wgpu::Maintain::Wait);
    rx2.recv().unwrap().unwrap();
    let sin_mapped = sin_slice.get_mapped_range();
    let sin_result: Vec<f32> = bytemuck::cast_slice(&sin_mapped[..table_len * 4]).to_vec();
    drop(sin_mapped);
    buf_sin_rb.unmap();

    (cos_result, sin_result)
}

/// Assert two float slices are approximately equal.
fn assert_approx_eq(actual: &[f32], expected: &[f32], tol: f32, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{context}: index {i} — GPU {a} vs CPU {e} (diff {})",
            (a - e).abs()
        );
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_basic_rotation() {
    let _tc = MetalRopeTestCase {
        input_dim: 8,
        head_dim: 8,
        seq_len: 4,
        base_freq: DEFAULT_BASE,
        expected: "basic rotation matches CPU reference for small dim",
    };

    let ctx = create_metal_context();
    let dim = 8u32;
    let seq_len = 4u32;
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.1).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "basic_rotation");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_zero_position() {
    let _tc = MetalRopeTestCase {
        input_dim: 16,
        head_dim: 16,
        seq_len: 1,
        base_freq: DEFAULT_BASE,
        expected: "position 0 yields identity rotation (cos=1, sin=0)",
    };

    let ctx = create_metal_context();
    let dim = 16u32;
    let positions = vec![0u32];
    let data: Vec<f32> = (0..dim).map(|i| (i as f32) + 1.0).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, 1, DEFAULT_BASE);

    // At position 0, angle = 0 for all pairs → cos=1, sin=0 → identity.
    assert_approx_eq(&gpu_result, &data, STRICT_TOLERANCE, "zero_position");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_identity_at_head_dim_pairs() {
    let _tc = MetalRopeTestCase {
        input_dim: 8,
        head_dim: 8,
        seq_len: 1,
        base_freq: DEFAULT_BASE,
        expected: "each pair (x0,x1) rotates by angle = pos * theta_i",
    };

    let ctx = create_metal_context();
    let dim = 8u32;
    let pos = 3u32;
    let positions = vec![pos];
    // Unit vector in first pair
    let mut data = vec![0.0f32; dim as usize];
    data[0] = 1.0;
    data[1] = 0.0;

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, 1, DEFAULT_BASE);

    let angle = rope_angle(dim as usize, 0, pos as usize, DEFAULT_BASE);
    assert!((gpu_result[0] - angle.cos()).abs() < GPU_TOLERANCE, "cos mismatch");
    assert!((gpu_result[1] - angle.sin()).abs() < GPU_TOLERANCE, "sin mismatch");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_frequency_computation() {
    let _tc = MetalRopeTestCase {
        input_dim: 64,
        head_dim: 64,
        seq_len: 1,
        base_freq: DEFAULT_BASE,
        expected: "freq = 1/(base^(2i/d)) = base^(-2i/d)",
    };

    let dim = 64usize;
    let half_dim = dim / 2;
    for i in 0..half_dim {
        let freq = rope_freq(dim, i, DEFAULT_BASE);
        let expected = 1.0 / DEFAULT_BASE.powf(2.0 * i as f32 / dim as f32);
        assert!(
            (freq - expected).abs() < STRICT_TOLERANCE,
            "freq mismatch at pair {i}: {freq} vs {expected}"
        );
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_cache_computation() {
    let _tc = MetalRopeTestCase {
        input_dim: 16,
        head_dim: 16,
        seq_len: 32,
        base_freq: DEFAULT_BASE,
        expected: "GPU sin/cos cache matches analytical CPU reference",
    };

    let ctx = create_metal_context();
    let dim = 16u32;
    let max_seq = 32u32;

    let (gpu_cos, gpu_sin) = run_freq_table_gpu(&ctx, dim, max_seq, DEFAULT_BASE);
    let (cpu_cos, cpu_sin) = build_reference_tables(dim as usize, max_seq as usize, DEFAULT_BASE);

    assert_approx_eq(&gpu_cos, &cpu_cos, GPU_TOLERANCE, "cos_cache");
    assert_approx_eq(&gpu_sin, &cpu_sin, GPU_TOLERANCE, "sin_cache");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_batch_consistency() {
    let _tc = MetalRopeTestCase {
        input_dim: 16,
        head_dim: 16,
        seq_len: 4,
        base_freq: DEFAULT_BASE,
        expected: "same position produces same rotation across batch elements",
    };

    let ctx = create_metal_context();
    let dim = 16u32;
    let seq_len = 4u32;
    // All at the same position
    let positions = vec![5u32; seq_len as usize];
    let data: Vec<f32> = (0..seq_len).flat_map(|_| (0..dim).map(|j| (j as f32) * 0.5)).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);

    // All seq elements should produce identical results
    let first = &gpu_result[0..dim as usize];
    for s in 1..seq_len as usize {
        let segment = &gpu_result[s * dim as usize..(s + 1) * dim as usize];
        assert_approx_eq(segment, first, STRICT_TOLERANCE, "batch_consistency");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_sequence_ordering() {
    let _tc = MetalRopeTestCase {
        input_dim: 8,
        head_dim: 8,
        seq_len: 8,
        base_freq: DEFAULT_BASE,
        expected: "later positions rotate more (larger angle magnitude)",
    };

    let ctx = create_metal_context();
    let dim = 8u32;
    let seq_len = 8u32;
    let positions: Vec<u32> = (0..seq_len).collect();
    // Same input at every position: unit vector in first pair
    let data: Vec<f32> = (0..seq_len)
        .flat_map(|_| {
            let mut v = vec![0.0f32; dim as usize];
            v[0] = 1.0;
            v
        })
        .collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);

    // For pair 0 (highest frequency), angle increases with position.
    // The cos of first element should decrease from 1.0 as pos increases.
    let mut prev_cos = 2.0f32; // sentinel above any real cos
    for s in 0..seq_len as usize {
        let cos_val = gpu_result[s * dim as usize];
        if s > 0 {
            // cos(angle) should differ from previous position
            assert!(
                (cos_val - prev_cos).abs() > STRICT_TOLERANCE,
                "position {s} did not change rotation from position {}",
                s - 1
            );
        }
        prev_cos = cos_val;
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_numerical_stability() {
    let _tc = MetalRopeTestCase {
        input_dim: 64,
        head_dim: 64,
        seq_len: 4,
        base_freq: DEFAULT_BASE,
        expected: "no NaN/Inf for large positions",
    };

    let ctx = create_metal_context();
    let dim = 64u32;
    let positions = vec![0u32, 1_000u32, 100_000u32, 1_000_000u32];
    let data: Vec<f32> = (0..4 * dim).map(|i| ((i as f32) * 0.01).sin()).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, 4, DEFAULT_BASE);

    for (i, &v) in gpu_result.iter().enumerate() {
        assert!(v.is_finite(), "NaN/Inf at index {i} for large positions");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_head_dim_2() {
    let _tc = MetalRopeTestCase {
        input_dim: 2,
        head_dim: 2,
        seq_len: 4,
        base_freq: DEFAULT_BASE,
        expected: "minimal head_dim=2 (single rotation pair)",
    };

    let ctx = create_metal_context();
    let dim = 2u32;
    let seq_len = 4u32;
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.5];

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "head_dim_2");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_head_dim_128() {
    let _tc = MetalRopeTestCase {
        input_dim: 128,
        head_dim: 128,
        seq_len: 8,
        base_freq: DEFAULT_BASE,
        expected: "standard transformer head_dim=128",
    };

    let ctx = create_metal_context();
    let dim = 128u32;
    let seq_len = 8u32;
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = (0..seq_len * dim).map(|i| ((i as f32) * 0.037).sin()).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "head_dim_128");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_ntk_scaling() {
    let _tc = MetalRopeTestCase {
        input_dim: 32,
        head_dim: 32,
        seq_len: 4,
        base_freq: DEFAULT_BASE,
        expected: "NTK-aware scaling modifies base frequency correctly",
    };

    let ctx = create_metal_context();
    let dim = 32u32;
    let seq_len = 4u32;
    let scale_factor = 2.0f32;
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.1).collect();

    let gpu_result =
        run_rope_ntk_gpu(&ctx, &data, &positions, dim, seq_len, DEFAULT_BASE, scale_factor);
    let mut cpu_result = data.clone();
    apply_rope_ntk_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE, scale_factor);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "ntk_scaling");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_yarn_scaling() {
    let _tc = MetalRopeTestCase {
        input_dim: 32,
        head_dim: 32,
        seq_len: 4,
        base_freq: DEFAULT_BASE,
        expected: "YaRN interpolation applies position scaling and attention factor",
    };

    let ctx = create_metal_context();
    let dim = 32u32;
    let seq_len = 4u32;
    let scale_factor = 4.0f32;
    let attn_factor = 0.1f32 * (1.0 + (scale_factor as f32).ln() / (2.0f32 * PI).ln());
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.1).collect();

    let gpu_result = run_rope_yarn_gpu(
        &ctx,
        &data,
        &positions,
        dim,
        seq_len,
        DEFAULT_BASE,
        scale_factor,
        attn_factor,
    );
    let mut cpu_result = data.clone();
    apply_rope_yarn_cpu(
        &mut cpu_result,
        dim as usize,
        &positions,
        DEFAULT_BASE,
        scale_factor,
        attn_factor,
    );

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "yarn_scaling");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_interleaved_layout() {
    let _tc = MetalRopeTestCase {
        input_dim: 16,
        head_dim: 16,
        seq_len: 4,
        base_freq: DEFAULT_BASE,
        expected: "interleaved layout produces same rotation as contiguous pairs",
    };

    let ctx = create_metal_context();
    let dim = 16u32;
    let seq_len = 4u32;
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.1).collect();

    let gpu_result =
        run_rope_gpu(&ctx, ROPE_INTERLEAVED_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_interleaved_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "interleaved_layout");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_buffer_alignment() {
    let _tc = MetalRopeTestCase {
        input_dim: 64,
        head_dim: 64,
        seq_len: 1,
        base_freq: DEFAULT_BASE,
        expected: "256-byte Metal buffer alignment respected",
    };

    // Verify alignment helper
    assert_eq!(align_to_metal(1), 256);
    assert_eq!(align_to_metal(256), 256);
    assert_eq!(align_to_metal(257), 512);
    assert_eq!(align_to_metal(512), 512);

    let ctx = create_metal_context();
    let dim = 64u32;
    let positions = vec![7u32];
    let data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();

    // Verify readback buffer is properly aligned
    let byte_len = (data.len() * 4) as u64;
    let aligned = align_to_metal(byte_len);
    assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0, "readback not aligned");

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, 1, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "buffer_alignment");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_zero_input() {
    let _tc = MetalRopeTestCase {
        input_dim: 8,
        head_dim: 8,
        seq_len: 2,
        base_freq: DEFAULT_BASE,
        expected: "zero input remains zero after rotation",
    };

    let ctx = create_metal_context();
    let dim = 8u32;
    let seq_len = 2u32;
    let positions = vec![0u32, 42u32];
    let data = vec![0.0f32; (seq_len * dim) as usize];

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);

    for (i, &v) in gpu_result.iter().enumerate() {
        assert!(v.abs() < STRICT_TOLERANCE, "zero input produced nonzero at index {i}: {v}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_large_seq_len() {
    let _tc = MetalRopeTestCase {
        input_dim: 16,
        head_dim: 16,
        seq_len: 1024,
        base_freq: DEFAULT_BASE,
        expected: "handles large sequence lengths correctly",
    };

    let ctx = create_metal_context();
    let dim = 16u32;
    let seq_len = 1024u32;
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = (0..seq_len * dim).map(|i| ((i as f32) * 0.001).sin()).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "large_seq_len");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_non_power_of_2_dim() {
    let _tc = MetalRopeTestCase {
        input_dim: 6,
        head_dim: 6,
        seq_len: 3,
        base_freq: DEFAULT_BASE,
        expected: "non-power-of-2 dimension still rotates correctly",
    };

    let ctx = create_metal_context();
    let dim = 6u32;
    let seq_len = 3u32;
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.2).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "non_power_of_2");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_custom_base_freq() {
    let _tc = MetalRopeTestCase {
        input_dim: 16,
        head_dim: 16,
        seq_len: 4,
        base_freq: 1_000_000.0,
        expected: "custom base frequency (1M) produces different rotation than default",
    };

    let ctx = create_metal_context();
    let dim = 16u32;
    let seq_len = 4u32;
    let custom_base = 1_000_000.0f32;
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.1).collect();

    let gpu_default =
        run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);
    let gpu_custom = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, custom_base);

    // With a larger base, frequencies are lower → rotations are smaller.
    // Results should differ (except at position 0).
    let mut found_diff = false;
    for i in 0..gpu_default.len() {
        if (gpu_default[i] - gpu_custom[i]).abs() > STRICT_TOLERANCE {
            found_diff = true;
            break;
        }
    }
    assert!(found_diff, "custom base 1M should produce different results than default 10K");

    // Verify custom base result matches CPU reference
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, custom_base);
    assert_approx_eq(&gpu_custom, &cpu_result, GPU_TOLERANCE, "custom_base_freq");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_rotation_preserves_norm() {
    let _tc = MetalRopeTestCase {
        input_dim: 32,
        head_dim: 32,
        seq_len: 8,
        base_freq: DEFAULT_BASE,
        expected: "rotation preserves L2 norm of each pair",
    };

    let ctx = create_metal_context();
    let dim = 32u32;
    let seq_len = 8u32;
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.1 + 0.5).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);

    let half_dim = (dim / 2) as usize;
    for s in 0..seq_len as usize {
        for p in 0..half_dim {
            let idx = s * dim as usize + p * 2;
            let orig_norm = (data[idx] * data[idx] + data[idx + 1] * data[idx + 1]).sqrt();
            let new_norm = (gpu_result[idx] * gpu_result[idx]
                + gpu_result[idx + 1] * gpu_result[idx + 1])
                .sqrt();
            assert!(
                (orig_norm - new_norm).abs() < GPU_TOLERANCE,
                "norm not preserved at seq={s} pair={p}: {orig_norm} vs {new_norm}"
            );
        }
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_inverse_rotation() {
    let _tc = MetalRopeTestCase {
        input_dim: 16,
        head_dim: 16,
        seq_len: 1,
        base_freq: DEFAULT_BASE,
        expected: "applying rotation then negative rotation recovers original",
    };

    let ctx = create_metal_context();
    let dim = 16u32;
    let pos = 5u32;
    let positions = vec![pos];
    let data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.3 + 1.0).collect();

    // Forward rotation
    let rotated = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, 1, DEFAULT_BASE);

    // Manual inverse: apply rotation with negated sin
    let half_dim = (dim / 2) as usize;
    let mut recovered = rotated.clone();
    for i in 0..half_dim {
        let angle = rope_angle(dim as usize, i, pos as usize, DEFAULT_BASE);
        let cos_val = angle.cos();
        let sin_val = angle.sin();
        let idx = i * 2;
        let x0 = rotated[idx];
        let x1 = rotated[idx + 1];
        // Inverse: cos(−θ) = cos(θ), sin(−θ) = −sin(θ)
        recovered[idx] = x0 * cos_val + x1 * sin_val;
        recovered[idx + 1] = -x0 * sin_val + x1 * cos_val;
    }

    assert_approx_eq(&recovered, &data, GPU_TOLERANCE, "inverse_rotation");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_different_positions_per_seq() {
    let _tc = MetalRopeTestCase {
        input_dim: 16,
        head_dim: 16,
        seq_len: 4,
        base_freq: DEFAULT_BASE,
        expected: "non-sequential positions are handled correctly",
    };

    let ctx = create_metal_context();
    let dim = 16u32;
    let seq_len = 4u32;
    let positions = vec![0u32, 10, 100, 1000];
    let data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.05).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "different_positions");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_repeated_positions() {
    let _tc = MetalRopeTestCase {
        input_dim: 8,
        head_dim: 8,
        seq_len: 3,
        base_freq: DEFAULT_BASE,
        expected: "repeated positions with different data produce different outputs",
    };

    let ctx = create_metal_context();
    let dim = 8u32;
    let positions = vec![7u32, 7, 7];
    // Different data at each position
    let data = vec![
        1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -1.0, 2.0, // seq 0
        2.0, 1.0, -1.0, 0.0, 3.0, -0.5, 0.0, 0.0, // seq 1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // seq 2 (zeros)
    ];

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, 3, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "repeated_positions");
    // Seq 2 should remain zero
    for i in 16..24 {
        assert!(gpu_result[i].abs() < STRICT_TOLERANCE, "zeros should stay zero");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_head_dim_4() {
    let _tc = MetalRopeTestCase {
        input_dim: 4,
        head_dim: 4,
        seq_len: 8,
        base_freq: DEFAULT_BASE,
        expected: "head_dim=4 (2 rotation pairs)",
    };

    let ctx = create_metal_context();
    let dim = 4u32;
    let seq_len = 8u32;
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.25).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "head_dim_4");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_head_dim_64() {
    let _tc = MetalRopeTestCase {
        input_dim: 64,
        head_dim: 64,
        seq_len: 16,
        base_freq: DEFAULT_BASE,
        expected: "head_dim=64 (common in smaller transformers)",
    };

    let ctx = create_metal_context();
    let dim = 64u32;
    let seq_len = 16u32;
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = (0..seq_len * dim).map(|i| ((i as f32) * 0.01).cos()).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "head_dim_64");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_head_dim_256() {
    let _tc = MetalRopeTestCase {
        input_dim: 256,
        head_dim: 256,
        seq_len: 4,
        base_freq: DEFAULT_BASE,
        expected: "head_dim=256 (large model head dimension)",
    };

    let ctx = create_metal_context();
    let dim = 256u32;
    let seq_len = 4u32;
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = (0..seq_len * dim).map(|i| ((i as f32) * 0.003).sin()).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "head_dim_256");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_negative_values() {
    let _tc = MetalRopeTestCase {
        input_dim: 8,
        head_dim: 8,
        seq_len: 2,
        base_freq: DEFAULT_BASE,
        expected: "negative input values rotate correctly",
    };

    let ctx = create_metal_context();
    let dim = 8u32;
    let positions = vec![3u32, 7u32];
    let data: Vec<f32> = (0..2 * dim as usize).map(|i| -((i as f32) * 0.3 + 0.1)).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, 2, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "negative_values");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_large_values() {
    let _tc = MetalRopeTestCase {
        input_dim: 16,
        head_dim: 16,
        seq_len: 2,
        base_freq: DEFAULT_BASE,
        expected: "large magnitude inputs stay finite",
    };

    let ctx = create_metal_context();
    let dim = 16u32;
    let positions = vec![1u32, 100u32];
    let data: Vec<f32> =
        (0..2 * dim as usize).map(|i| if i % 2 == 0 { 1e6 } else { -1e6 }).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, 2, DEFAULT_BASE);

    for (i, &v) in gpu_result.iter().enumerate() {
        assert!(v.is_finite(), "NaN/Inf at index {i} for large values");
    }

    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);
    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "large_values");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_subnormal_values() {
    let _tc = MetalRopeTestCase {
        input_dim: 8,
        head_dim: 8,
        seq_len: 1,
        base_freq: DEFAULT_BASE,
        expected: "subnormal float inputs don't produce NaN",
    };

    let ctx = create_metal_context();
    let dim = 8u32;
    let positions = vec![5u32];
    let subnormal = f32::MIN_POSITIVE / 2.0;
    let data = vec![subnormal; dim as usize];

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, 1, DEFAULT_BASE);

    for (i, &v) in gpu_result.iter().enumerate() {
        assert!(!v.is_nan(), "NaN at index {i} for subnormal input");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_single_element_seq() {
    let _tc = MetalRopeTestCase {
        input_dim: 32,
        head_dim: 32,
        seq_len: 1,
        base_freq: DEFAULT_BASE,
        expected: "single-element sequence works",
    };

    let ctx = create_metal_context();
    let dim = 32u32;
    let positions = vec![42u32];
    let data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, 1, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "single_element");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_cache_large_dim() {
    let _tc = MetalRopeTestCase {
        input_dim: 128,
        head_dim: 128,
        seq_len: 64,
        base_freq: DEFAULT_BASE,
        expected: "frequency cache correct for large dim × seq product",
    };

    let ctx = create_metal_context();
    let dim = 128u32;
    let max_seq = 64u32;

    let (gpu_cos, gpu_sin) = run_freq_table_gpu(&ctx, dim, max_seq, DEFAULT_BASE);
    let (cpu_cos, cpu_sin) = build_reference_tables(dim as usize, max_seq as usize, DEFAULT_BASE);

    assert_approx_eq(&gpu_cos, &cpu_cos, GPU_TOLERANCE, "cache_large_cos");
    assert_approx_eq(&gpu_sin, &cpu_sin, GPU_TOLERANCE, "cache_large_sin");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_ntk_vs_standard() {
    let _tc = MetalRopeTestCase {
        input_dim: 32,
        head_dim: 32,
        seq_len: 4,
        base_freq: DEFAULT_BASE,
        expected: "NTK scaling produces different rotations than standard",
    };

    let ctx = create_metal_context();
    let dim = 32u32;
    let seq_len = 4u32;
    let positions: Vec<u32> = (1..=seq_len).collect(); // skip 0 to see differences
    let data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.1).collect();

    let standard = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);
    let ntk = run_rope_ntk_gpu(&ctx, &data, &positions, dim, seq_len, DEFAULT_BASE, 2.0);

    let mut found_diff = false;
    for i in 0..standard.len() {
        if (standard[i] - ntk[i]).abs() > STRICT_TOLERANCE {
            found_diff = true;
            break;
        }
    }
    assert!(found_diff, "NTK scaling should differ from standard RoPE");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_yarn_attn_factor() {
    let _tc = MetalRopeTestCase {
        input_dim: 16,
        head_dim: 16,
        seq_len: 2,
        base_freq: DEFAULT_BASE,
        expected: "YaRN attention factor scales the rotation output",
    };

    let ctx = create_metal_context();
    let dim = 16u32;
    let positions = vec![1u32, 5u32];
    let data: Vec<f32> = (0..2 * dim).map(|i| (i as f32) * 0.1).collect();

    let result_attn1 = run_rope_yarn_gpu(&ctx, &data, &positions, dim, 2, DEFAULT_BASE, 2.0, 1.0);
    let result_attn_half =
        run_rope_yarn_gpu(&ctx, &data, &positions, dim, 2, DEFAULT_BASE, 2.0, 0.5);

    // With attn_factor=0.5, the cos/sin values are halved, so the
    // resulting rotated values should differ approximately by that factor
    // (for non-zero inputs).
    let mut found_ratio = false;
    for i in 0..result_attn1.len() {
        if result_attn1[i].abs() > 0.01 && result_attn_half[i].abs() > 0.01 {
            // Not exact due to cos/sin mixing, but the outputs should differ
            if (result_attn1[i] - result_attn_half[i]).abs() > STRICT_TOLERANCE {
                found_ratio = true;
                break;
            }
        }
    }
    assert!(found_ratio, "different attn_factor should produce different results");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_frequency_decreasing() {
    let _tc = MetalRopeTestCase {
        input_dim: 64,
        head_dim: 64,
        seq_len: 1,
        base_freq: DEFAULT_BASE,
        expected: "frequencies decrease with increasing pair index",
    };

    let dim = 64usize;
    let half_dim = dim / 2;
    let mut prev_freq = f32::MAX;
    for i in 0..half_dim {
        let freq = rope_freq(dim, i, DEFAULT_BASE);
        assert!(
            freq < prev_freq,
            "frequency should decrease: pair {i} has freq {freq} >= prev {prev_freq}"
        );
        assert!(freq > 0.0, "frequency must be positive");
        prev_freq = freq;
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_orthogonality() {
    let _tc = MetalRopeTestCase {
        input_dim: 8,
        head_dim: 8,
        seq_len: 1,
        base_freq: DEFAULT_BASE,
        expected: "rotation of orthogonal vectors preserves orthogonality",
    };

    let ctx = create_metal_context();
    let dim = 8u32;
    let positions = vec![3u32];

    // Two orthogonal vectors (in the first pair)
    let mut v1 = vec![0.0f32; dim as usize];
    v1[0] = 1.0;
    let mut v2 = vec![0.0f32; dim as usize];
    v2[1] = 1.0;

    let r1 = run_rope_gpu(&ctx, ROPE_SHADER, &v1, &positions, dim, 1, DEFAULT_BASE);
    let r2 = run_rope_gpu(&ctx, ROPE_SHADER, &v2, &positions, dim, 1, DEFAULT_BASE);

    // Dot product of rotated vectors should remain ~0
    let dot: f32 = r1.iter().zip(r2.iter()).map(|(a, b)| a * b).sum();
    assert!(dot.abs() < GPU_TOLERANCE, "orthogonality not preserved: dot product = {dot}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_test_case_struct_fields() {
    let tc = MetalRopeTestCase {
        input_dim: 128,
        head_dim: 64,
        seq_len: 512,
        base_freq: 10_000.0,
        expected: "struct carries all fields for parameterised test cases",
    };

    assert_eq!(tc.input_dim, 128);
    assert_eq!(tc.head_dim, 64);
    assert_eq!(tc.seq_len, 512);
    assert!((tc.base_freq - 10_000.0).abs() < f32::EPSILON);
    assert!(!tc.expected.is_empty());
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_small_base_freq() {
    let _tc = MetalRopeTestCase {
        input_dim: 8,
        head_dim: 8,
        seq_len: 4,
        base_freq: 100.0,
        expected: "small base freq produces faster rotation",
    };

    let ctx = create_metal_context();
    let dim = 8u32;
    let seq_len = 4u32;
    let small_base = 100.0f32;
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.1).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, small_base);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, small_base);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "small_base_freq");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_dim_10_non_power_of_2() {
    let _tc = MetalRopeTestCase {
        input_dim: 10,
        head_dim: 10,
        seq_len: 5,
        base_freq: DEFAULT_BASE,
        expected: "dim=10 (odd half_dim=5) handled correctly",
    };

    let ctx = create_metal_context();
    let dim = 10u32;
    let seq_len = 5u32;
    let positions: Vec<u32> = (0..seq_len).collect();
    let data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.15).collect();

    let gpu_result = run_rope_gpu(&ctx, ROPE_SHADER, &data, &positions, dim, seq_len, DEFAULT_BASE);
    let mut cpu_result = data.clone();
    apply_rope_cpu(&mut cpu_result, dim as usize, &positions, DEFAULT_BASE);

    assert_approx_eq(&gpu_result, &cpu_result, GPU_TOLERANCE, "dim_10");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_cache_position_zero_identity() {
    let _tc = MetalRopeTestCase {
        input_dim: 32,
        head_dim: 32,
        seq_len: 16,
        base_freq: DEFAULT_BASE,
        expected: "cache at position 0: cos=1, sin=0 for all dims",
    };

    let ctx = create_metal_context();
    let dim = 32u32;
    let max_seq = 16u32;
    let half_dim = dim / 2;

    let (gpu_cos, gpu_sin) = run_freq_table_gpu(&ctx, dim, max_seq, DEFAULT_BASE);

    // Position 0 entries should be cos=1, sin=0
    for i in 0..half_dim as usize {
        assert!((gpu_cos[i] - 1.0).abs() < GPU_TOLERANCE, "cos[0][{i}] = {} != 1.0", gpu_cos[i]);
        assert!(gpu_sin[i].abs() < GPU_TOLERANCE, "sin[0][{i}] = {} != 0.0", gpu_sin[i]);
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_rope_double_rotation() {
    let _tc = MetalRopeTestCase {
        input_dim: 16,
        head_dim: 16,
        seq_len: 1,
        base_freq: DEFAULT_BASE,
        expected: "rotating at pos P twice equals rotating at pos 2P once",
    };

    let ctx = create_metal_context();
    let dim = 16u32;
    let pos = 3u32;
    let data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.2 + 0.1).collect();

    // Rotate twice at position P
    let once = run_rope_gpu(&ctx, ROPE_SHADER, &data, &[pos], dim, 1, DEFAULT_BASE);
    let twice = run_rope_gpu(&ctx, ROPE_SHADER, &once, &[pos], dim, 1, DEFAULT_BASE);

    // Single rotation at position 2P
    let double_pos = run_rope_gpu(&ctx, ROPE_SHADER, &data, &[2 * pos], dim, 1, DEFAULT_BASE);

    assert_approx_eq(&twice, &double_pos, GPU_TOLERANCE, "double_rotation");
}
