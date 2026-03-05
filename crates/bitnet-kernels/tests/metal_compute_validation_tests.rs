#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types)]
#![allow(clippy::manual_div_ceil, clippy::assertions_on_constants)]
//! Metal compute pipeline validation tests for Apple Silicon.
//!
//! These tests validate Metal/wgpu compute pipeline behaviour including
//! buffer operations, shader compilation, and dispatch validation.
//! All tests are `#[ignore]` since they require Metal GPU hardware.

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests {
    use wgpu::{
        BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
        ComputePipelineDescriptor, DeviceDescriptor, InstanceDescriptor, MapMode,
        RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
    };

    /// Helper: create a wgpu device + queue backed by the Metal backend.
    ///
    /// Returns `None` when no Metal adapter is available (e.g. Linux CI).
    fn metal_device() -> Option<(wgpu::Device, wgpu::Queue)> {
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

    // ── 1. Metal buffer create-and-readback validation ──────────────

    #[test]
    #[ignore = "requires Metal GPU hardware - run on Apple Silicon"]
    fn test_metal_buffer_roundtrip() {
        let (device, queue) = metal_device().expect("Metal adapter required");

        for &count in &[1u32, 4, 256, 1024, 65_536] {
            let byte_len = (count as u64) * std::mem::size_of::<f32>() as u64;

            // Source data
            let src: Vec<f32> = (0..count).map(|i| i as f32 * 0.5).collect();

            // GPU buffer (storage + copy-src so we can map-read via staging)
            let gpu_buf = device.create_buffer(&BufferDescriptor {
                label: Some("roundtrip_gpu"),
                size: byte_len,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Upload
            queue.write_buffer(&gpu_buf, 0, bytemuck::cast_slice(&src));

            // Staging buffer for readback
            let staging = device.create_buffer(&BufferDescriptor {
                label: Some("roundtrip_staging"),
                size: byte_len,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Copy gpu → staging
            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(&gpu_buf, 0, &staging, 0, byte_len);
            queue.submit(std::iter::once(encoder.finish()));

            // Map and verify
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(MapMode::Read, move |res| tx.send(res).unwrap());
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().expect("map failed");

            let data = slice.get_mapped_range();
            let result: &[f32] = bytemuck::cast_slice(&data);
            assert_eq!(result.len(), count as usize);
            for (i, (&got, &expected)) in result.iter().zip(src.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < f32::EPSILON,
                    "mismatch at index {i} for count={count}: got {got}, expected {expected}",
                );
            }
            drop(data);
            staging.unmap();
        }
    }

    // ── 2. Metal compute shader compilation ─────────────────────────

    /// Trivial WGSL compute shader used for compilation tests.
    const TRIVIAL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&data) {
        data[idx] = data[idx] + 1.0;
    }
}
"#;

    #[test]
    #[ignore = "requires Metal GPU hardware - run on Apple Silicon"]
    fn test_metal_shader_compilation() {
        let (device, _queue) = metal_device().expect("Metal adapter required");

        // Valid shader must compile without error.
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("trivial"),
            source: ShaderSource::Wgsl(TRIVIAL_SHADER.into()),
        });

        let _pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("trivial_pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Validation: creating a pipeline with a bad entry-point name should
        // trigger a device error. Push an error scope and verify.
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let _bad_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("bad_pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("nonexistent_entry"),
            compilation_options: Default::default(),
            cache: None,
        });

        let err = pollster::block_on(device.pop_error_scope());
        assert!(err.is_some(), "expected validation error for bad entry point");
    }

    // ── 3. Metal dispatch size validation ───────────────────────────

    #[test]
    #[ignore = "requires Metal GPU hardware - run on Apple Silicon"]
    fn test_metal_dispatch_sizing() {
        let (device, queue) = metal_device().expect("Metal adapter required");

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("dispatch_test"),
            source: ShaderSource::Wgsl(TRIVIAL_SHADER.into()),
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("dispatch_pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create a small buffer and dispatch with various workgroup counts.
        let n: u32 = 1024;
        let byte_len = (n as u64) * 4;
        let buf = device.create_buffer(&BufferDescriptor {
            label: Some("dispatch_buf"),
            size: byte_len,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
        });

        // Various dispatch configurations — all should complete without panic.
        // The shader workgroup_size is 64, so workgroup counts are ceil(n/64).
        let workgroup_counts: &[u32] = &[1, 4, 16, (n + 63) / 64];
        for &wg in workgroup_counts {
            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass =
                    encoder.begin_compute_pass(&ComputePassDescriptor { ..Default::default() });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, Some(&bind_group), &[]);
                pass.dispatch_workgroups(wg, 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
        }
        device.poll(wgpu::Maintain::Wait);

        // Apple Silicon SIMD group size is 32 — verify constant compiles.
        const APPLE_SIMD_GROUP_SIZE: u32 = 32;
        assert!(APPLE_SIMD_GROUP_SIZE <= 1024, "SIMD group size must fit in max threadgroup");
    }

    // ── 4. Metal buffer alignment ───────────────────────────────────

    #[test]
    #[ignore = "requires Metal GPU hardware - run on Apple Silicon"]
    fn test_metal_buffer_alignment() {
        let (device, _queue) = metal_device().expect("Metal adapter required");

        const METAL_ALIGN: u64 = 256;

        // Various requested sizes — the returned buffer must be at least as
        // large as `requested` rounded up to METAL_ALIGN.
        for &requested in &[1u64, 4, 100, 255, 256, 257, 512, 1000, 4096] {
            let buf = device.create_buffer(&BufferDescriptor {
                label: Some("align_test"),
                size: requested,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            // wgpu reports the buffer size as what the driver allocated —
            // it must be at least as large as requested.
            assert!(
                buf.size() >= requested,
                "buffer smaller than requested for size={requested}: got {}",
                buf.size(),
            );

            // On Metal the driver-level allocation is 256-byte aligned.
            let expected_min = (requested + METAL_ALIGN - 1) & !(METAL_ALIGN - 1);
            assert!(
                buf.size() >= expected_min,
                "buffer not aligned to {METAL_ALIGN} for requested={requested}: got {}",
                buf.size(),
            );
        }
    }

    // ── 5. Vector addition compute shader ───────────────────────────

    const VEC_ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&a) {
        c[idx] = a[idx] + b[idx];
    }
}
"#;

    #[test]
    #[ignore = "requires Metal GPU hardware - run on Apple Silicon"]
    fn test_metal_vector_add_compute() {
        let (device, queue) = metal_device().expect("Metal adapter required");

        let n: u32 = 4096;
        let byte_len = (n as u64) * 4;

        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();
        let expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(a, b)| a + b).collect();

        let make_buf = |label, usage: BufferUsages| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: byte_len,
                usage,
                mapped_at_creation: false,
            })
        };

        let buf_a = make_buf("a", BufferUsages::STORAGE | BufferUsages::COPY_DST);
        let buf_b = make_buf("b", BufferUsages::STORAGE | BufferUsages::COPY_DST);
        let buf_c =
            make_buf("c", BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST);
        let staging = make_buf("staging", BufferUsages::MAP_READ | BufferUsages::COPY_DST);

        queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(&a_data));
        queue.write_buffer(&buf_b, 0, bytemuck::cast_slice(&b_data));

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("vec_add"),
            source: ShaderSource::Wgsl(VEC_ADD_SHADER.into()),
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("vec_add_pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bgl = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass =
                encoder.begin_compute_pass(&ComputePassDescriptor { ..Default::default() });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups((n + 63) / 64, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buf_c, 0, &staging, 0, byte_len);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |res| tx.send(res).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");

        let data = slice.get_mapped_range();
        let result: &[f32] = bytemuck::cast_slice(&data);
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < f32::EPSILON,
                "vec_add mismatch at {i}: got {got}, expected {exp}",
            );
        }
    }

    // ── 6. Metal memory bandwidth baseline ──────────────────────────

    #[test]
    #[ignore = "requires Metal GPU hardware - run on Apple Silicon"]
    fn test_metal_memory_bandwidth_baseline() {
        let (device, queue) = metal_device().expect("Metal adapter required");

        const BUF_SIZE: u64 = 64 * 1024 * 1024; // 64 MiB
        const ITERATIONS: u32 = 10;

        let src = device.create_buffer(&BufferDescriptor {
            label: Some("bw_src"),
            size: BUF_SIZE,
            usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dst = device.create_buffer(&BufferDescriptor {
            label: Some("bw_dst"),
            size: BUF_SIZE,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Warm-up
        {
            let mut enc = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(&src, 0, &dst, 0, BUF_SIZE);
            queue.submit(std::iter::once(enc.finish()));
            device.poll(wgpu::Maintain::Wait);
        }

        let start = std::time::Instant::now();
        for _ in 0..ITERATIONS {
            let mut enc = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(&src, 0, &dst, 0, BUF_SIZE);
            queue.submit(std::iter::once(enc.finish()));
        }
        device.poll(wgpu::Maintain::Wait);
        let elapsed = start.elapsed();

        let total_bytes = BUF_SIZE as f64 * ITERATIONS as f64;
        let gb_per_sec = (total_bytes / elapsed.as_secs_f64()) / 1e9;

        // Apple Silicon typically delivers >50 GB/s unified memory bandwidth;
        // set a conservative floor of 10 GB/s so the test does not flake.
        assert!(
            gb_per_sec > 10.0,
            "memory bandwidth too low: {gb_per_sec:.2} GB/s (expected > 10 GB/s)",
        );
    }
}
