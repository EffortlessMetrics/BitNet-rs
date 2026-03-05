#![cfg(all(target_os = "macos", target_arch = "aarch64"))]
#![allow(clippy::manual_div_ceil, clippy::manual_is_multiple_of)]
//! Metal embedding shader tests for Apple Silicon.
//!
//! Validates token embedding lookup, positional embedding computation
//! (sinusoidal & RoPE), combined embeddings, batch lookup, out-of-range
//! handling, large vocabulary support, dimension validation, buffer
//! alignment, f16→f32 accumulation, and embedding gradient computation.
//!
//! All tests are `#[ignore]` because CI runs on Linux.

use half::f16;
use wgpu::util::DeviceExt;

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
// WGSL shaders
// ---------------------------------------------------------------------------

/// Token embedding lookup: table[token_id] → output row.
const EMBEDDING_LOOKUP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       table: array<f32>;
@group(0) @binding(1) var<storage, read>       token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    embed_dim: u32,
    vocab_size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let num_tokens = arrayLength(&token_ids);
    if tid >= num_tokens {
        return;
    }
    let token_id = token_ids[tid];
    let dim = params.embed_dim;
    // Clamp out-of-range token IDs to zero vector.
    let safe = select(0u, 1u, token_id < params.vocab_size);
    for (var d = 0u; d < dim; d = d + 1u) {
        output[tid * dim + d] = f32(safe) * table[token_id * dim + d];
    }
}
"#;

/// Sinusoidal positional embedding.
const SINUSOIDAL_POS_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

struct Params {
    seq_len: u32,
    embed_dim: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.embed_dim;
    if idx >= total {
        return;
    }
    let pos = idx / params.embed_dim;
    let d   = idx % params.embed_dim;
    let half_dim = params.embed_dim / 2u;
    let freq = 1.0 / pow(10000.0, f32(d % half_dim) / f32(half_dim));
    let angle = f32(pos) * freq;
    if d < half_dim {
        output[idx] = sin(angle);
    } else {
        output[idx] = cos(angle);
    }
}
"#;

/// RoPE positional embedding: applies rotary factors in-place.
const ROPE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> embeddings: array<f32>;

struct Params {
    seq_len: u32,
    embed_dim: u32,
    base: f32,
    _pad: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let half_dim = params.embed_dim / 2u;
    let total_pairs = params.seq_len * half_dim;
    if idx >= total_pairs {
        return;
    }
    let pos = idx / half_dim;
    let d   = idx % half_dim;
    let freq = 1.0 / pow(params.base, f32(d) / f32(half_dim));
    let angle = f32(pos) * freq;
    let cs = cos(angle);
    let sn = sin(angle);

    let base = pos * params.embed_dim;
    let x0 = embeddings[base + d];
    let x1 = embeddings[base + d + half_dim];
    embeddings[base + d]            = x0 * cs - x1 * sn;
    embeddings[base + d + half_dim] = x0 * sn + x1 * cs;
}
"#;

/// Combined token + positional embedding (element-wise add).
const COMBINE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       tok_emb: array<f32>;
@group(0) @binding(1) var<storage, read>       pos_emb: array<f32>;
@group(0) @binding(2) var<storage, read_write> output:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&tok_emb) {
        output[idx] = tok_emb[idx] + pos_emb[idx];
    }
}
"#;

/// f16 embedding table → f32 output (manual unpack via bitcast).
const F16_LOOKUP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       table_u32: array<u32>;
@group(0) @binding(1) var<storage, read>       token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    embed_dim: u32,
    vocab_size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

fn unpack_f16_lo(packed: u32) -> f32 {
    return unpack2x16float(packed).x;
}
fn unpack_f16_hi(packed: u32) -> f32 {
    return unpack2x16float(packed).y;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let num_tokens = arrayLength(&token_ids);
    if tid >= num_tokens {
        return;
    }
    let token_id = token_ids[tid];
    let dim = params.embed_dim;
    let pairs = dim / 2u;
    let safe = select(0.0, 1.0, token_id < params.vocab_size);
    for (var p = 0u; p < pairs; p = p + 1u) {
        let packed = table_u32[token_id * pairs + p];
        output[tid * dim + p * 2u]      = safe * unpack_f16_lo(packed);
        output[tid * dim + p * 2u + 1u] = safe * unpack_f16_hi(packed);
    }
}
"#;

/// Embedding gradient: dL/dE scatter-add (training backprop).
const EMBEDDING_GRAD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       grad_output: array<f32>;
@group(0) @binding(1) var<storage, read>       token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> grad_table: array<atomic<u32>>;

struct Params {
    embed_dim: u32,
    num_tokens: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

fn f32_to_sortable_u32(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    let mask = select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
    return bits ^ mask;
}
fn sortable_u32_to_f32(v: u32) -> f32 {
    let mask = select(0x80000000u, 0xFFFFFFFFu, (v & 0x80000000u) == 0u);
    return bitcast<f32>(v ^ mask);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.num_tokens * params.embed_dim;
    if idx >= total {
        return;
    }
    let tok = idx / params.embed_dim;
    let d   = idx % params.embed_dim;
    let token_id = token_ids[tok];
    let grad = grad_output[idx];

    // Atomic float add emulated via CAS loop on sortable-uint encoding.
    let addr = token_id * params.embed_dim + d;
    var old_bits = atomicLoad(&grad_table[addr]);
    loop {
        let old_val = sortable_u32_to_f32(old_bits);
        let new_val = old_val + grad;
        let new_bits = f32_to_sortable_u32(new_val);
        let prev = atomicCompareExchangeWeak(&grad_table[addr], old_bits, new_bits);
        if prev.exchanged {
            break;
        }
        old_bits = prev.old_value;
    }
}
"#;

// ---------------------------------------------------------------------------
// GPU dispatch helpers
// ---------------------------------------------------------------------------

/// Create a simple 1-read + 1-rw bind group layout with optional extras.
fn run_embedding_lookup(
    ctx: &MetalContext,
    table: &[f32],
    token_ids: &[u32],
    embed_dim: u32,
    vocab_size: u32,
) -> Vec<f32> {
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("embedding_lookup"),
        source: wgpu::ShaderSource::Wgsl(EMBEDDING_LOOKUP_SHADER.into()),
    });

    let num_tokens = token_ids.len();
    let out_len = num_tokens * embed_dim as usize;

    let buf_table = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("table"),
        contents: bytemuck::cast_slice(table),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_ids = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("token_ids"),
        contents: bytemuck::cast_slice(token_ids),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (out_len * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = [embed_dim, vocab_size];
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_table.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_ids.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
        ],
    });
    let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    gpu_dispatch_read(ctx, &pipeline, &bg, &buf_out, num_tokens as u32, out_len)
}

/// Inline GPU dispatch + readback for a single compute pass.
fn gpu_dispatch_read(
    ctx: &MetalContext,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    result_buf: &wgpu::Buffer,
    num_invocations: u32,
    out_len: usize,
) -> Vec<f32> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (out_len * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        let wg = (num_invocations + 63) / 64;
        pass.dispatch_workgroups(wg, 1, 1);
    }
    encoder.copy_buffer_to_buffer(result_buf, 0, &staging, 0, (out_len * 4) as u64);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    pollster::block_on(async {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec()
    })
}

// ---------------------------------------------------------------------------
// Bind-group-layout helpers
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
// CPU reference implementations
// ---------------------------------------------------------------------------

fn cpu_embedding_lookup(
    table: &[f32],
    token_ids: &[u32],
    embed_dim: usize,
    vocab_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; token_ids.len() * embed_dim];
    for (i, &tid) in token_ids.iter().enumerate() {
        if (tid as usize) < vocab_size {
            let src = &table[tid as usize * embed_dim..(tid as usize + 1) * embed_dim];
            out[i * embed_dim..(i + 1) * embed_dim].copy_from_slice(src);
        }
    }
    out
}

fn cpu_sinusoidal_pos(seq_len: usize, embed_dim: usize) -> Vec<f32> {
    let half = embed_dim / 2;
    let mut out = vec![0.0f32; seq_len * embed_dim];
    for pos in 0..seq_len {
        for d in 0..embed_dim {
            let freq = 1.0 / (10000.0f32).powf((d % half) as f32 / half as f32);
            let angle = pos as f32 * freq;
            out[pos * embed_dim + d] = if d < half { angle.sin() } else { angle.cos() };
        }
    }
    out
}

fn cpu_rope(embeddings: &mut [f32], seq_len: usize, embed_dim: usize, base: f32) {
    let half = embed_dim / 2;
    for pos in 0..seq_len {
        for d in 0..half {
            let freq = 1.0 / base.powf(d as f32 / half as f32);
            let angle = pos as f32 * freq;
            let cs = angle.cos();
            let sn = angle.sin();
            let off = pos * embed_dim;
            let x0 = embeddings[off + d];
            let x1 = embeddings[off + d + half];
            embeddings[off + d] = x0 * cs - x1 * sn;
            embeddings[off + d + half] = x0 * sn + x1 * cs;
        }
    }
}

fn cpu_f16_lookup(
    table_f16: &[f16],
    token_ids: &[u32],
    embed_dim: usize,
    vocab_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; token_ids.len() * embed_dim];
    for (i, &tid) in token_ids.iter().enumerate() {
        if (tid as usize) < vocab_size {
            for d in 0..embed_dim {
                out[i * embed_dim + d] = table_f16[tid as usize * embed_dim + d].to_f32();
            }
        }
    }
    out
}

fn cpu_embedding_grad(
    grad_output: &[f32],
    token_ids: &[u32],
    embed_dim: usize,
    vocab_size: usize,
) -> Vec<f32> {
    let mut grad_table = vec![0.0f32; vocab_size * embed_dim];
    for (i, &tid) in token_ids.iter().enumerate() {
        for d in 0..embed_dim {
            grad_table[tid as usize * embed_dim + d] += grad_output[i * embed_dim + d];
        }
    }
    grad_table
}

// ---------------------------------------------------------------------------
// Pipeline builder helpers (reduce boilerplate)
// ---------------------------------------------------------------------------

struct PipelineBundle {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

fn build_pipeline(
    ctx: &MetalContext,
    shader_src: &str,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> PipelineBundle {
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries });
    let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    PipelineBundle { pipeline, bgl }
}

/// Deterministic f32 embedding table seeded from index.
fn make_table(vocab_size: usize, embed_dim: usize) -> Vec<f32> {
    (0..vocab_size * embed_dim).map(|i| ((i as f32) * 0.01).sin()).collect()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
    assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "{msg}: mismatch at [{i}]: gpu={x} cpu={y} diff={}",
            (x - y).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// 1. Token embedding lookup shader correctness
#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_token_embedding_lookup_correctness() {
    let ctx = create_metal_context();
    let vocab: usize = 256;
    let dim: usize = 64;
    let table = make_table(vocab, dim);
    let ids: Vec<u32> = vec![0, 1, 42, 255];

    let pb = build_pipeline(
        &ctx,
        EMBEDDING_LOOKUP_SHADER,
        "lookup",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    );

    let buf_table = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&table),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_ids = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&ids),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_len = ids.len() * dim;
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (out_len * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = [dim as u32, vocab as u32];
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pb.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_table.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_ids.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
        ],
    });

    let gpu = gpu_dispatch_read(&ctx, &pb.pipeline, &bg, &buf_out, ids.len() as u32, out_len);
    let cpu = cpu_embedding_lookup(&table, &ids, dim, vocab);
    assert_close(&gpu, &cpu, 1e-6, "token_embedding_lookup");
}

// 2. Sinusoidal positional embedding
#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_sinusoidal_positional_embedding() {
    let ctx = create_metal_context();
    let seq_len: u32 = 16;
    let embed_dim: u32 = 64;
    let total = (seq_len * embed_dim) as usize;

    let pb = build_pipeline(
        &ctx,
        SINUSOIDAL_POS_SHADER,
        "sinusoidal",
        &[storage_entry(0, false), uniform_entry(1)],
    );

    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (total * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = [seq_len, embed_dim];
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pb.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_params.as_entire_binding() },
        ],
    });

    let gpu = gpu_dispatch_read(&ctx, &pb.pipeline, &bg, &buf_out, total as u32, total);
    let cpu = cpu_sinusoidal_pos(seq_len as usize, embed_dim as usize);
    // GPU transcendentals may differ slightly from CPU libm.
    assert_close(&gpu, &cpu, 1e-4, "sinusoidal_pos");
}

// 2b. RoPE positional embedding
#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_rope_positional_embedding() {
    let ctx = create_metal_context();
    let seq_len: u32 = 8;
    let embed_dim: u32 = 64;
    let base: f32 = 10000.0;
    let total = (seq_len * embed_dim) as usize;

    // Initialise with deterministic data.
    let mut cpu_data: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).cos()).collect();
    let gpu_data = cpu_data.clone();

    let pb =
        build_pipeline(&ctx, ROPE_SHADER, "rope", &[storage_entry(0, false), uniform_entry(1)]);

    let buf_emb = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&gpu_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    // Params: seq_len, embed_dim, base, _pad
    let params_raw: [u32; 4] = [seq_len, embed_dim, base.to_bits(), 0];
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&params_raw),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pb.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_emb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_params.as_entire_binding() },
        ],
    });

    let half = embed_dim / 2;
    let invocations = seq_len * half;
    let gpu = gpu_dispatch_read(&ctx, &pb.pipeline, &bg, &buf_emb, invocations, total);
    cpu_rope(&mut cpu_data, seq_len as usize, embed_dim as usize, base);
    assert_close(&gpu, &cpu_data, 1e-4, "rope");
}

// 3. Combined token + positional embedding
#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_combined_token_positional_embedding() {
    let ctx = create_metal_context();
    let n: usize = 512;
    let tok: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let pos: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();

    let pb = build_pipeline(
        &ctx,
        COMBINE_SHADER,
        "combine",
        &[storage_entry(0, true), storage_entry(1, true), storage_entry(2, false)],
    );

    let buf_tok = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&tok),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_pos = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&pos),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pb.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_tok.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_pos.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
        ],
    });

    let gpu = gpu_dispatch_read(&ctx, &pb.pipeline, &bg, &buf_out, n as u32, n);
    let cpu: Vec<f32> = tok.iter().zip(&pos).map(|(a, b)| a + b).collect();
    assert_close(&gpu, &cpu, 1e-6, "combined_embedding");
}

// 4. Batch embedding lookup (multiple sequences)
#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_batch_embedding_lookup() {
    let ctx = create_metal_context();
    let vocab: usize = 512;
    let dim: usize = 128;
    let table = make_table(vocab, dim);
    // Two sequences concatenated: [seq0_tok0, seq0_tok1, seq1_tok0, …]
    let ids: Vec<u32> = vec![10, 20, 30, 100, 200, 300, 400, 511];

    let pb = build_pipeline(
        &ctx,
        EMBEDDING_LOOKUP_SHADER,
        "batch_lookup",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    );

    let out_len = ids.len() * dim;
    let buf_table = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&table),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_ids = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&ids),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (out_len * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = [dim as u32, vocab as u32];
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pb.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_table.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_ids.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
        ],
    });

    let gpu = gpu_dispatch_read(&ctx, &pb.pipeline, &bg, &buf_out, ids.len() as u32, out_len);
    let cpu = cpu_embedding_lookup(&table, &ids, dim, vocab);
    assert_close(&gpu, &cpu, 1e-6, "batch_lookup");
}

// 5. Out-of-range token ID handling
#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_out_of_range_token_ids() {
    let ctx = create_metal_context();
    let vocab: usize = 100;
    let dim: usize = 32;
    let table = make_table(vocab, dim);
    // Include valid (0, 50) and invalid (100, 9999, u32::MAX) IDs.
    let ids: Vec<u32> = vec![0, 50, 100, 9999, u32::MAX];

    let pb = build_pipeline(
        &ctx,
        EMBEDDING_LOOKUP_SHADER,
        "oor",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    );

    let out_len = ids.len() * dim;
    let buf_table = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&table),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_ids = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&ids),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (out_len * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = [dim as u32, vocab as u32];
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pb.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_table.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_ids.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
        ],
    });

    let gpu = gpu_dispatch_read(&ctx, &pb.pipeline, &bg, &buf_out, ids.len() as u32, out_len);
    let cpu = cpu_embedding_lookup(&table, &ids, dim, vocab);
    assert_close(&gpu, &cpu, 1e-6, "out_of_range");

    // Verify invalid IDs produce zero vectors.
    for idx in 2..ids.len() {
        let row = &gpu[idx * dim..(idx + 1) * dim];
        assert!(
            row.iter().all(|&v| v == 0.0),
            "OOR token id {} should yield zero vector",
            ids[idx]
        );
    }
}

// 6. Large vocabulary support (32000)
#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_large_vocab_32k() {
    let ctx = create_metal_context();
    let vocab: usize = 32_000;
    let dim: usize = 64;
    let table = make_table(vocab, dim);
    let ids: Vec<u32> = vec![0, 1000, 15_999, 31_999];

    let pb = build_pipeline(
        &ctx,
        EMBEDDING_LOOKUP_SHADER,
        "vocab32k",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    );

    let out_len = ids.len() * dim;
    let buf_table = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&table),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_ids = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&ids),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (out_len * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = [dim as u32, vocab as u32];
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pb.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_table.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_ids.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
        ],
    });

    let gpu = gpu_dispatch_read(&ctx, &pb.pipeline, &bg, &buf_out, ids.len() as u32, out_len);
    let cpu = cpu_embedding_lookup(&table, &ids, dim, vocab);
    assert_close(&gpu, &cpu, 1e-6, "vocab32k");
}

// 6b. Large vocabulary support (128000)
#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_large_vocab_128k() {
    let ctx = create_metal_context();
    let vocab: usize = 128_000;
    let dim: usize = 32;
    let table = make_table(vocab, dim);
    let ids: Vec<u32> = vec![0, 64_000, 127_999];

    let pb = build_pipeline(
        &ctx,
        EMBEDDING_LOOKUP_SHADER,
        "vocab128k",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    );

    let out_len = ids.len() * dim;
    let buf_table = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&table),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_ids = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&ids),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (out_len * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = [dim as u32, vocab as u32];
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pb.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_table.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_ids.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
        ],
    });

    let gpu = gpu_dispatch_read(&ctx, &pb.pipeline, &bg, &buf_out, ids.len() as u32, out_len);
    let cpu = cpu_embedding_lookup(&table, &ids, dim, vocab);
    assert_close(&gpu, &cpu, 1e-6, "vocab128k");
}

// 7. Embedding dimension validation (768, 2048, 4096)
#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_embedding_dim_768() {
    let ctx = create_metal_context();
    let vocab: usize = 256;
    let dim: usize = 768;
    let table = make_table(vocab, dim);
    let ids: Vec<u32> = vec![0, 127, 255];

    let gpu = run_embedding_lookup(&ctx, &table, &ids, dim as u32, vocab as u32);
    let cpu = cpu_embedding_lookup(&table, &ids, dim, vocab);
    assert_close(&gpu, &cpu, 1e-6, "dim768");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_embedding_dim_2048() {
    let ctx = create_metal_context();
    let vocab: usize = 256;
    let dim: usize = 2048;
    let table = make_table(vocab, dim);
    let ids: Vec<u32> = vec![0, 128, 255];

    let gpu = run_embedding_lookup(&ctx, &table, &ids, dim as u32, vocab as u32);
    let cpu = cpu_embedding_lookup(&table, &ids, dim, vocab);
    assert_close(&gpu, &cpu, 1e-6, "dim2048");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_embedding_dim_4096() {
    let ctx = create_metal_context();
    let vocab: usize = 256;
    let dim: usize = 4096;
    let table = make_table(vocab, dim);
    let ids: Vec<u32> = vec![0, 64, 255];

    let gpu = run_embedding_lookup(&ctx, &table, &ids, dim as u32, vocab as u32);
    let cpu = cpu_embedding_lookup(&table, &ids, dim, vocab);
    assert_close(&gpu, &cpu, 1e-6, "dim4096");
}

// 8. Buffer alignment and memory layout
#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_buffer_alignment_256_bytes() {
    let ctx = create_metal_context();

    // Metal requires 256-byte alignment for storage buffers.
    for size_elems in [1usize, 63, 64, 65, 128, 256] {
        let byte_size = (size_elems * 4) as u64;
        // Round up to 256-byte alignment.
        let aligned = (byte_size + 255) & !255;
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("alignment_test"),
            size: aligned,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        assert!(
            buf.size() >= aligned,
            "Buffer for {size_elems} elems: expected >= {aligned}, got {}",
            buf.size()
        );
        assert!(buf.size() % 4 == 0, "Buffer size must be 4-byte aligned for f32");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_embedding_memory_layout_row_major() {
    let ctx = create_metal_context();
    let vocab: usize = 4;
    let dim: usize = 8;
    // Table laid out row-major: row 0 = [0..7], row 1 = [8..15], etc.
    let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
    let ids: Vec<u32> = vec![2]; // Should get row [16..23].

    let gpu = run_embedding_lookup(&ctx, &table, &ids, dim as u32, vocab as u32);
    let expected: Vec<f32> = (16..24).map(|i| i as f32).collect();
    assert_close(&gpu, &expected, 1e-6, "row_major_layout");
}

// 9. f16 embedding table with f32 accumulation
#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_f16_embedding_table_f32_output() {
    let ctx = create_metal_context();
    let vocab: usize = 256;
    let dim: usize = 64; // Must be even for u32-packed f16 pairs.
    let table_f16: Vec<f16> =
        (0..vocab * dim).map(|i| f16::from_f32((i as f32 * 0.01).sin())).collect();
    let ids: Vec<u32> = vec![0, 1, 42, 255];

    // Pack f16 pairs into u32 (little-endian: lo = even index, hi = odd).
    let table_u32: Vec<u32> = table_f16
        .chunks_exact(2)
        .map(|pair| (pair[0].to_bits() as u32) | ((pair[1].to_bits() as u32) << 16))
        .collect();

    let pb = build_pipeline(
        &ctx,
        F16_LOOKUP_SHADER,
        "f16_lookup",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    );

    let out_len = ids.len() * dim;
    let buf_table = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&table_u32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_ids = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&ids),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (out_len * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = [dim as u32, vocab as u32];
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pb.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_table.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_ids.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
        ],
    });

    let gpu = gpu_dispatch_read(&ctx, &pb.pipeline, &bg, &buf_out, ids.len() as u32, out_len);
    let cpu = cpu_f16_lookup(&table_f16, &ids, dim, vocab);
    // f16 round-trip loses precision — allow ~1e-3 tolerance.
    assert_close(&gpu, &cpu, 1e-3, "f16_lookup");
}

// 10. Embedding gradient (scatter-add for training backprop)
#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_embedding_gradient_scatter_add() {
    let ctx = create_metal_context();
    let vocab: usize = 16;
    let dim: usize = 8;
    let ids: Vec<u32> = vec![0, 3, 3, 7]; // Duplicate id=3 tests accumulation.
    let num_tokens = ids.len();
    let grad_output: Vec<f32> = (0..num_tokens * dim).map(|i| (i as f32) * 0.1).collect();

    let pb = build_pipeline(
        &ctx,
        EMBEDDING_GRAD_SHADER,
        "grad",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    );

    let buf_grad = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&grad_output),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_ids = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&ids),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // Grad table stored as atomic<u32> — init to sortable encoding of 0.0.
    let zero_bits = 0x80000000u32; // sortable encoding of 0.0
    let init_table: Vec<u32> = vec![zero_bits; vocab * dim];
    let buf_grad_table = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&init_table),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let params = [dim as u32, num_tokens as u32];
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pb.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_grad.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_ids.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_grad_table.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
        ],
    });

    let total_invocations = (num_tokens * dim) as u32;
    let raw_u32 = {
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (vocab * dim * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pb.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let wg = (total_invocations + 63) / 64;
            pass.dispatch_workgroups(wg, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buf_grad_table, 0, &staging, 0, (vocab * dim * 4) as u64);
        ctx.queue.submit(std::iter::once(encoder.finish()));

        pollster::block_on(async {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
            ctx.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec()
        })
    };

    // Decode sortable-uint back to f32.
    let gpu_grad: Vec<f32> = raw_u32
        .iter()
        .map(|&v| {
            let mask = if v & 0x80000000 == 0 { 0xFFFFFFFFu32 } else { 0x80000000u32 };
            f32::from_bits(v ^ mask)
        })
        .collect();

    let cpu_grad = cpu_embedding_grad(&grad_output, &ids, dim, vocab);
    assert_close(&gpu_grad, &cpu_grad, 1e-4, "embedding_grad");
}

// ---------------------------------------------------------------------------
// Additional edge cases
// ---------------------------------------------------------------------------

/// Workgroup size 1024 (max Metal threads per threadgroup) boundary.
#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_max_threads_per_threadgroup_boundary() {
    let ctx = create_metal_context();
    let vocab: usize = 128;
    let dim: usize = 32;
    let table = make_table(vocab, dim);
    // 1024 tokens → exactly fills max threadgroup at workgroup_size=64.
    let ids: Vec<u32> = (0..1024u32).map(|i| i % vocab as u32).collect();

    let gpu = run_embedding_lookup(&ctx, &table, &ids, dim as u32, vocab as u32);
    let cpu = cpu_embedding_lookup(&table, &ids, dim, vocab);
    assert_close(&gpu, &cpu, 1e-6, "max_threadgroup");
}

/// Single-token embedding (edge case).
#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_single_token_embedding() {
    let ctx = create_metal_context();
    let vocab: usize = 10;
    let dim: usize = 16;
    let table = make_table(vocab, dim);
    let ids: Vec<u32> = vec![5];

    let gpu = run_embedding_lookup(&ctx, &table, &ids, dim as u32, vocab as u32);
    let cpu = cpu_embedding_lookup(&table, &ids, dim, vocab);
    assert_close(&gpu, &cpu, 1e-6, "single_token");
}
