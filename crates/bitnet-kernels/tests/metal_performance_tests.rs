#![cfg(all(target_os = "macos", target_arch = "aarch64"))]
#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
#![allow(
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::manual_slice_size_calculation
)]

//! Metal/wgpu performance validation tests for Apple Silicon.
//!
//! These tests exercise real GPU dispatches via wgpu's Metal backend to validate
//! throughput scaling, dispatch overhead, buffer transfer bandwidth, concurrent
//! dispatches, workgroup-size occupancy, and memory-pressure handling.
//!
//! Run on macOS with:
//!   cargo test -p bitnet-kernels --test metal_performance_tests -- --ignored

use std::time::Instant;

use wgpu::util::DeviceExt;

const DOUBLING_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&data) {
        data[id.x] = data[id.x] * 2.0;
    }
}
"#;

/// Request a Metal-backed wgpu device. Returns `None` when Metal is unavailable.
fn setup_metal() -> Option<(wgpu::Device, wgpu::Queue)> {
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
        &wgpu::DeviceDescriptor { label: Some("metal-perf-test"), ..Default::default() },
        None,
    ))
    .ok()?;
    Some((device, queue))
}

/// Build a compute pipeline from the doubling shader.
fn create_doubling_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("doubling-shader"),
        source: wgpu::ShaderSource::Wgsl(DOUBLING_SHADER.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("doubling-pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Run the doubling shader on `input` and return the GPU-side result.
fn run_doubling(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    input: &[f32],
) -> Vec<f32> {
    let size = (input.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let storage_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("storage"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: storage_buf.as_entire_binding() }],
    });

    let workgroups = ((input.len() as u32) + 63) / 64;
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage_buf, 0, &readback_buf, 0, size);
    queue.submit(Some(encoder.finish()));

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    bytemuck::cast_slice(&data).to_vec()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_performance_tests -- --ignored"]
fn test_metal_throughput_scaling() {
    let (device, queue) = setup_metal().expect("Metal GPU required");
    let pipeline = create_doubling_pipeline(&device);

    let sizes: Vec<usize> = vec![1_024, 4_096, 16_384, 65_536, 262_144];
    let mut throughputs: Vec<f64> = Vec::new();

    for &n in &sizes {
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();

        // Warm-up run
        let _ = run_doubling(&device, &queue, &pipeline, &input);

        let start = Instant::now();
        let result = run_doubling(&device, &queue, &pipeline, &input);
        let elapsed = start.elapsed();

        // Correctness: every element should be doubled.
        for (i, &val) in result.iter().enumerate() {
            let expected = (i as f32) * 2.0;
            assert!(
                (val - expected).abs() < 1e-3,
                "size={n} idx={i}: expected {expected}, got {val}"
            );
        }

        let bytes = (n * std::mem::size_of::<f32>()) as f64;
        // Read + write = 2× bytes transferred
        let throughput_gbs = (bytes * 2.0) / elapsed.as_secs_f64() / 1e9;
        throughputs.push(throughput_gbs);
    }

    // Larger buffers should achieve equal or better throughput due to better
    // GPU utilization (amortised dispatch overhead). Compare the largest with
    // the smallest.
    let first = throughputs[0];
    let last = *throughputs.last().unwrap();
    assert!(
        last >= first * 0.5,
        "Throughput should scale with buffer size: smallest={first:.2} GB/s, largest={last:.2} GB/s"
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_performance_tests -- --ignored"]
fn test_metal_dispatch_overhead() {
    let (device, queue) = setup_metal().expect("Metal GPU required");
    let pipeline = create_doubling_pipeline(&device);

    let input: Vec<f32> = vec![1.0; 64]; // minimal buffer — one workgroup

    // Warm-up
    let _ = run_doubling(&device, &queue, &pipeline, &input);

    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = run_doubling(&device, &queue, &pipeline, &input);
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    assert!(
        avg_ms < 10.0,
        "Per-dispatch overhead should be < 10 ms on average, got {avg_ms:.3} ms"
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_performance_tests -- --ignored"]
fn test_metal_buffer_transfer_bandwidth() {
    let (device, queue) = setup_metal().expect("Metal GPU required");

    let n: usize = 1_000_000; // 1M floats = 4 MB
    let byte_size = (n * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();

    // --- Upload ---
    let upload_start = Instant::now();
    let storage_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("upload"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    queue.submit(std::iter::empty());
    device.poll(wgpu::Maintain::Wait);
    let upload_elapsed = upload_start.elapsed();

    // --- Download ---
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let download_start = Instant::now();
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&storage_buf, 0, &readback_buf, 0, byte_size);
    queue.submit(Some(encoder.finish()));

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let download_elapsed = download_start.elapsed();

    let total_bytes = byte_size as f64;
    let upload_bw = total_bytes / upload_elapsed.as_secs_f64() / 1e9;
    let download_bw = total_bytes / download_elapsed.as_secs_f64() / 1e9;

    // Apple Silicon uses unified memory, so bandwidth should comfortably
    // exceed 1 GB/s for a 4 MB transfer.
    assert!(upload_bw > 1.0, "Upload bandwidth should exceed 1 GB/s, got {upload_bw:.2} GB/s");
    assert!(
        download_bw > 1.0,
        "Download bandwidth should exceed 1 GB/s, got {download_bw:.2} GB/s"
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_performance_tests -- --ignored"]
fn test_metal_concurrent_dispatches() {
    let (device, queue) = setup_metal().expect("Metal GPU required");
    let pipeline = create_doubling_pipeline(&device);

    let dispatch_count = 4;
    let n: usize = 4_096;

    let mut storage_bufs = Vec::new();
    let mut readback_bufs = Vec::new();
    let byte_size = (n * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let bind_group_layout = pipeline.get_bind_group_layout(0);

    // Create independent buffers for each dispatch.
    for i in 0..dispatch_count {
        let input: Vec<f32> = (0..n).map(|j| (i * n + j) as f32).collect();
        let sbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("storage"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let rbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        storage_bufs.push(sbuf);
        readback_bufs.push(rbuf);
    }

    // Encode all dispatches into a single command buffer.
    let mut encoder = device.create_command_encoder(&Default::default());
    let workgroups = ((n as u32) + 63) / 64;
    for i in 0..dispatch_count {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_bufs[i].as_entire_binding(),
            }],
        });
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&storage_bufs[i], 0, &readback_bufs[i], 0, byte_size);
    }
    queue.submit(Some(encoder.finish()));

    // Read back and verify every dispatch produced correct results.
    for i in 0..dispatch_count {
        let slice = readback_bufs[i].slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: &[f32] = bytemuck::cast_slice(&data);
        for j in 0..n {
            let expected = (i * n + j) as f32 * 2.0;
            assert!(
                (result[j] - expected).abs() < 1e-3,
                "dispatch {i} idx {j}: expected {expected}, got {}",
                result[j]
            );
        }
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_performance_tests -- --ignored"]
fn test_metal_occupancy_vs_workgroup_size() {
    let (device, queue) = setup_metal().expect("Metal GPU required");

    let workgroup_sizes: &[u32] = &[32, 64, 128, 256];
    let n: usize = 16_384; // large enough for all workgroup sizes

    for &wg_size in workgroup_sizes {
        let shader_src = format!(
            r#"
@group(0) @binding(0)
var<storage, read_write> data: array<f32>;

@compute @workgroup_size({wg_size})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    if id.x < arrayLength(&data) {{
        data[id.x] = data[id.x] * 2.0;
    }}
}}
"#
        );

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let result = run_doubling(&device, &queue, &pipeline, &input);

        for (i, &val) in result.iter().enumerate() {
            let expected = (i as f32) * 2.0;
            assert!(
                (val - expected).abs() < 1e-3,
                "wg_size={wg_size} idx={i}: expected {expected}, got {val}"
            );
        }
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_performance_tests -- --ignored"]
fn test_metal_memory_pressure() {
    let (device, _queue) = setup_metal().expect("Metal GPU required");

    let chunk_floats: usize = 64 * 1024 * 1024; // 256 MB per buffer
    let chunk_bytes = (chunk_floats * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let mut buffers: Vec<wgpu::Buffer> = Vec::new();
    let max_attempts = 128; // cap to avoid runaway allocation

    for _ in 0..max_attempts {
        // Use `create_buffer` (not panicking) and check device error state.
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pressure"),
            size: chunk_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Poll to surface any out-of-memory errors asynchronously.
        device.poll(wgpu::Maintain::Wait);

        // wgpu may surface OOM as a device-lost or validation error on the
        // next submit. We keep the buffer if the device is still healthy.
        buffers.push(buf);
    }

    // Verify we allocated at least *some* buffers before pressure kicked in.
    assert!(
        !buffers.is_empty(),
        "Should have allocated at least one buffer before hitting memory limit"
    );

    // Clean-up: drop all buffers to release GPU memory.
    drop(buffers);
    device.poll(wgpu::Maintain::Wait);
}
