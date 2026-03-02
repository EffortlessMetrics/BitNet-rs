#![cfg(target_os = "macos")]

//! Tests for wgpu/Metal attention compute shaders on Apple Silicon.
//!
//! Each test compiles a WGSL compute shader, dispatches it on a Metal GPU,
//! reads back the results, and compares against a CPU reference implementation.

#[cfg(test)]
mod tests {
    use wgpu::{
        BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, DeviceDescriptor, InstanceDescriptor, MapMode,
        RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
    };

    // ── Helpers ─────────────────────────────────────────────────────

    /// Create a wgpu device + queue backed by the Metal backend.
    ///
    /// Returns `None` when no Metal adapter is available (e.g. Linux CI).
    fn setup_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, queue) = pollster::block_on(
            adapter.request_device(&DeviceDescriptor { ..Default::default() }, None),
        )
        .ok()?;

        Some((device, queue))
    }

    /// CPU reference: full scaled dot-product attention.
    ///
    /// Q, K, V are row-major `[seq_len, head_dim]`.
    /// Returns `softmax(Q·Kᵀ / √head_dim) · V` as `[seq_len, head_dim]`.
    fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], seq_len: usize, head_dim: usize) -> Vec<f32> {
        assert_eq!(q.len(), seq_len * head_dim);
        assert_eq!(k.len(), seq_len * head_dim);
        assert_eq!(v.len(), seq_len * head_dim);

        let scale = 1.0 / (head_dim as f32).sqrt();

        // scores = Q · Kᵀ / √d  →  [seq_len, seq_len]
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }

        // softmax per row
        for i in 0..seq_len {
            let row = &mut scores[i * seq_len..(i + 1) * seq_len];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }

        // output = scores · V  →  [seq_len, head_dim]
        let mut out = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq_len {
                    acc += scores[i * seq_len + j] * v[j * head_dim + d];
                }
                out[i * head_dim + d] = acc;
            }
        }
        out
    }

    /// Run a compute shader that reads one or more input buffers and writes one output buffer.
    /// Returns the output buffer contents as `Vec<f32>`.
    fn run_shader(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        wgsl: &str,
        inputs: &[&[f32]],
        output_len: usize,
    ) -> Vec<f32> {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("test_shader"),
            source: ShaderSource::Wgsl(wgsl.into()),
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("test_pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        // Create input GPU buffers and upload data.
        let input_bufs: Vec<wgpu::Buffer> = inputs
            .iter()
            .enumerate()
            .map(|(i, data)| {
                let buf = device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("input_{i}")),
                    size: (data.len() * 4) as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
                buf
            })
            .collect();

        // Output GPU buffer.
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("output"),
            size: (output_len * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Build bind-group entries.
        let mut entries: Vec<wgpu::BindGroupEntry> = input_bufs
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: out_buf.as_entire_binding(),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test_bg"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        let mut encoder =
            device.create_command_encoder(&CommandEncoderDescriptor { label: Some("test_enc") });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("test_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((output_len as u32) + 63) / 64, 1, 1);
        }

        // Staging read-back.
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("staging"),
            size: (output_len * 4) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, (output_len * 4) as u64);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        bytemuck::cast_slice(&data).to_vec()
    }

    // ── 1. Attention scores: QK^T / sqrt(d) ────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_attention_shader_tests -- --ignored"]
    fn test_metal_attention_scores() {
        let (device, queue) = setup_device().expect("Metal adapter required");

        const SEQ: usize = 4;
        const DIM: usize = 4;
        let scale = 1.0 / (DIM as f32).sqrt();

        #[rustfmt::skip]
        let q: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let k = q.clone();

        // WGSL: each invocation computes one element of the [SEQ, SEQ] score matrix.
        let wgsl = format!(
            r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;

const SEQ: u32 = {SEQ}u;
const DIM: u32 = {DIM}u;
const SCALE: f32 = {scale};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx >= SEQ * SEQ {{
        return;
    }}
    let row = idx / SEQ;
    let col = idx % SEQ;
    var dot: f32 = 0.0;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {{
        dot = dot + q[row * DIM + d] * k[col * DIM + d];
    }}
    scores[idx] = dot * SCALE;
}}
"#
        );

        let gpu_scores = run_shader(&device, &queue, &wgsl, &[&q, &k], SEQ * SEQ);

        // CPU reference
        let mut expected = vec![0.0f32; SEQ * SEQ];
        for i in 0..SEQ {
            for j in 0..SEQ {
                let mut dot = 0.0f32;
                for d in 0..DIM {
                    dot += q[i * DIM + d] * k[j * DIM + d];
                }
                expected[i * SEQ + j] = dot * scale;
            }
        }

        for (i, (g, e)) in gpu_scores.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-5, "score mismatch at {i}: gpu={g}, cpu={e}");
        }
    }

    // ── 2. Causal mask application ──────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_attention_shader_tests -- --ignored"]
    fn test_metal_causal_mask_application() {
        let (device, queue) = setup_device().expect("Metal adapter required");

        const SEQ: usize = 4;
        // Flat scores: all ones.
        let scores: Vec<f32> = vec![1.0; SEQ * SEQ];

        let wgsl = format!(
            r#"
@group(0) @binding(0) var<storage, read> scores_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> scores_out: array<f32>;

const SEQ: u32 = {SEQ}u;
const NEG_INF: f32 = -1.0e30;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx >= SEQ * SEQ {{
        return;
    }}
    let row = idx / SEQ;
    let col = idx % SEQ;
    if col > row {{
        scores_out[idx] = NEG_INF;
    }} else {{
        scores_out[idx] = scores_in[idx];
    }}
}}
"#
        );

        let masked = run_shader(&device, &queue, &wgsl, &[&scores], SEQ * SEQ);

        for i in 0..SEQ {
            for j in 0..SEQ {
                let val = masked[i * SEQ + j];
                if j > i {
                    assert!(val < -1.0e20, "position [{i},{j}] should be masked (got {val})");
                } else {
                    assert!(
                        (val - 1.0).abs() < 1e-5,
                        "position [{i},{j}] should be 1.0 (got {val})"
                    );
                }
            }
        }
    }

    // ── 3. Numerically stable softmax ───────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_attention_shader_tests -- --ignored"]
    fn test_metal_attention_softmax() {
        let (device, queue) = setup_device().expect("Metal adapter required");

        const SEQ: usize = 4;
        #[rustfmt::skip]
        let scores: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            4.0, 3.0, 2.0, 1.0,
            0.0, 0.0, 0.0, 0.0,
            -1.0, 0.0, 1.0, 2.0,
        ];

        // Single-workgroup softmax: one invocation per row.
        let wgsl = format!(
            r#"
@group(0) @binding(0) var<storage, read> scores_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;

const SEQ: u32 = {SEQ}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    if row >= SEQ {{
        return;
    }}
    // Find max for numerical stability.
    var max_val: f32 = scores_in[row * SEQ];
    for (var j: u32 = 1u; j < SEQ; j = j + 1u) {{
        max_val = max(max_val, scores_in[row * SEQ + j]);
    }}
    // exp and sum.
    var sum_exp: f32 = 0.0;
    for (var j: u32 = 0u; j < SEQ; j = j + 1u) {{
        let e = exp(scores_in[row * SEQ + j] - max_val);
        probs[row * SEQ + j] = e;
        sum_exp = sum_exp + e;
    }}
    // Normalize.
    for (var j: u32 = 0u; j < SEQ; j = j + 1u) {{
        probs[row * SEQ + j] = probs[row * SEQ + j] / sum_exp;
    }}
}}
"#
        );

        let gpu_probs = run_shader(&device, &queue, &wgsl, &[&scores], SEQ * SEQ);

        // CPU reference softmax.
        let mut expected = scores.clone();
        for i in 0..SEQ {
            let row = &mut expected[i * SEQ..(i + 1) * SEQ];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }

        for (i, (g, e)) in gpu_probs.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-5, "softmax mismatch at {i}: gpu={g}, cpu={e}");
        }

        // Each row must sum to ~1.0.
        for i in 0..SEQ {
            let row_sum: f32 = gpu_probs[i * SEQ..(i + 1) * SEQ].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-4, "row {i} sum = {row_sum}, expected ~1.0");
        }
    }

    // ── 4. Full attention output ────────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_attention_shader_tests -- --ignored"]
    fn test_metal_attention_output() {
        let (device, queue) = setup_device().expect("Metal adapter required");

        const SEQ: usize = 4;
        const DIM: usize = 4;
        let scale = 1.0 / (DIM as f32).sqrt();

        // Simple identity-like Q/K so we can reason about the output.
        #[rustfmt::skip]
        let q: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let k = q.clone();
        #[rustfmt::skip]
        let v: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];

        // Full attention in a single shader:
        //   scores = Q·Kᵀ / √d  →  softmax  →  output = probs · V
        let wgsl = format!(
            r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const SEQ: u32 = {SEQ}u;
const DIM: u32 = {DIM}u;
const SCALE: f32 = {scale};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx >= SEQ * DIM {{
        return;
    }}
    let row = idx / DIM;
    let d_out = idx % DIM;

    // Compute attention scores for this row.
    var scores: array<f32, {SEQ}>;
    var max_val: f32 = -1.0e30;
    for (var j: u32 = 0u; j < SEQ; j = j + 1u) {{
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < DIM; d = d + 1u) {{
            dot = dot + q[row * DIM + d] * k[j * DIM + d];
        }}
        scores[j] = dot * SCALE;
        max_val = max(max_val, scores[j]);
    }}

    // Softmax.
    var sum_exp: f32 = 0.0;
    for (var j: u32 = 0u; j < SEQ; j = j + 1u) {{
        scores[j] = exp(scores[j] - max_val);
        sum_exp = sum_exp + scores[j];
    }}
    for (var j: u32 = 0u; j < SEQ; j = j + 1u) {{
        scores[j] = scores[j] / sum_exp;
    }}

    // Weighted sum over V.
    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j < SEQ; j = j + 1u) {{
        acc = acc + scores[j] * v[j * DIM + d_out];
    }}
    output[idx] = acc;
}}
"#
        );

        let gpu_out = run_shader(&device, &queue, &wgsl, &[&q, &k, &v], SEQ * DIM);
        let cpu_out = cpu_attention(&q, &k, &v, SEQ, DIM);

        for (i, (g, e)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!((g - e).abs() < 1e-4, "attention output mismatch at {i}: gpu={g}, cpu={e}");
        }
    }

    // ── 5. Multi-head split ─────────────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_attention_shader_tests -- --ignored"]
    fn test_metal_multi_head_split() {
        let (device, queue) = setup_device().expect("Metal adapter required");

        const SEQ: usize = 2;
        const NUM_HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        const MODEL_DIM: usize = NUM_HEADS * HEAD_DIM; // 8

        // Input: [SEQ, MODEL_DIM] row-major.
        #[rustfmt::skip]
        let input: Vec<f32> = vec![
            // seq 0: head0=[1,2,3,4] head1=[5,6,7,8]
            1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,
            // seq 1: head0=[9,10,11,12] head1=[13,14,15,16]
            9.0, 10.0, 11.0, 12.0,  13.0, 14.0, 15.0, 16.0,
        ];

        // Output: [NUM_HEADS, SEQ, HEAD_DIM] — heads are the outermost dimension.
        let wgsl = format!(
            r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const SEQ: u32 = {SEQ}u;
const NUM_HEADS: u32 = {NUM_HEADS}u;
const HEAD_DIM: u32 = {HEAD_DIM}u;
const MODEL_DIM: u32 = {MODEL_DIM}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = NUM_HEADS * SEQ * HEAD_DIM;
    if idx >= total {{
        return;
    }}
    let h = idx / (SEQ * HEAD_DIM);
    let rem = idx % (SEQ * HEAD_DIM);
    let s = rem / HEAD_DIM;
    let d = rem % HEAD_DIM;

    // Source index in [SEQ, MODEL_DIM] layout.
    let src = s * MODEL_DIM + h * HEAD_DIM + d;
    output[idx] = input[src];
}}
"#
        );

        let total = NUM_HEADS * SEQ * HEAD_DIM;
        let gpu_out = run_shader(&device, &queue, &wgsl, &[&input], total);

        // CPU reference: output[h][s][d] = input[s][h * HEAD_DIM + d]
        let mut expected = vec![0.0f32; total];
        for h in 0..NUM_HEADS {
            for s in 0..SEQ {
                for d in 0..HEAD_DIM {
                    let dst = h * (SEQ * HEAD_DIM) + s * HEAD_DIM + d;
                    let src = s * MODEL_DIM + h * HEAD_DIM + d;
                    expected[dst] = input[src];
                }
            }
        }

        assert_eq!(gpu_out.len(), expected.len());
        for (i, (g, e)) in gpu_out.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-6, "split mismatch at {i}: gpu={g}, cpu={e}");
        }
    }

    // ── 6. Multi-head concat ────────────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_attention_shader_tests -- --ignored"]
    fn test_metal_multi_head_concat() {
        let (device, queue) = setup_device().expect("Metal adapter required");

        const SEQ: usize = 2;
        const NUM_HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        const MODEL_DIM: usize = NUM_HEADS * HEAD_DIM; // 8

        // Input: [NUM_HEADS, SEQ, HEAD_DIM] (split layout).
        #[rustfmt::skip]
        let input: Vec<f32> = vec![
            // head 0: seq0=[1,2,3,4] seq1=[9,10,11,12]
            1.0, 2.0, 3.0, 4.0,  9.0, 10.0, 11.0, 12.0,
            // head 1: seq0=[5,6,7,8] seq1=[13,14,15,16]
            5.0, 6.0, 7.0, 8.0,  13.0, 14.0, 15.0, 16.0,
        ];

        // Output: [SEQ, MODEL_DIM] — concatenated heads back to model dim.
        let wgsl = format!(
            r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const SEQ: u32 = {SEQ}u;
const NUM_HEADS: u32 = {NUM_HEADS}u;
const HEAD_DIM: u32 = {HEAD_DIM}u;
const MODEL_DIM: u32 = {MODEL_DIM}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = SEQ * MODEL_DIM;
    if idx >= total {{
        return;
    }}
    let s = idx / MODEL_DIM;
    let col = idx % MODEL_DIM;
    let h = col / HEAD_DIM;
    let d = col % HEAD_DIM;

    // Source index in [NUM_HEADS, SEQ, HEAD_DIM] layout.
    let src = h * (SEQ * HEAD_DIM) + s * HEAD_DIM + d;
    output[idx] = input[src];
}}
"#
        );

        let total = SEQ * MODEL_DIM;
        let gpu_out = run_shader(&device, &queue, &wgsl, &[&input], total);

        // CPU reference: output[s][h * HEAD_DIM + d] = input[h][s][d]
        let mut expected = vec![0.0f32; total];
        for s in 0..SEQ {
            for h in 0..NUM_HEADS {
                for d in 0..HEAD_DIM {
                    let dst = s * MODEL_DIM + h * HEAD_DIM + d;
                    let src = h * (SEQ * HEAD_DIM) + s * HEAD_DIM + d;
                    expected[dst] = input[src];
                }
            }
        }

        assert_eq!(gpu_out.len(), expected.len());
        for (i, (g, e)) in gpu_out.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-6, "concat mismatch at {i}: gpu={g}, cpu={e}");
        }

        // Verify round-trip: split then concat should give back the original flat layout.
        #[rustfmt::skip]
        let original_flat: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        assert_eq!(gpu_out, original_flat);
    }
}
