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

//! Metal compute pipeline integration tests for Apple Silicon.
//!
//! Validates wgpu/Metal compute pipeline behavior including adapter selection,
//! device limits, shader compilation, buffer round-trips, compute dispatch
//! correctness, buffer alignment, and workgroup size handling.
//!
//! All tests require a Metal-capable GPU and are `#[ignore]` for CI.

#![cfg(target_os = "macos")]

use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Shared WGSL shader: doubles every element in a storage buffer.
// ---------------------------------------------------------------------------

const DOUBLING_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    data[id.x] = data[id.x] * 2.0;
}
"#;

/// Parameterised variant: workgroup size is baked into the source at call site.
fn doubling_shader_with_workgroup_size(size: u32) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size({size})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    data[id.x] = data[id.x] * 2.0;
}}
"#
    )
}

// ---------------------------------------------------------------------------
// Helper: create Metal device + queue (returns None if unavailable)
// ---------------------------------------------------------------------------

fn create_metal_device() -> Option<(wgpu::Device, wgpu::Queue)> {
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

// ---------------------------------------------------------------------------
// Internal helpers for dispatch-heavy tests
// ---------------------------------------------------------------------------

/// Run the doubling shader on `data` using the given device/queue and return
/// the GPU-side result.
fn run_doubling_dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[f32],
    shader_src: &str,
) -> Vec<f32> {
    let byte_len = (data.len() * std::mem::size_of::<f32>()) as u64;

    let storage_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("storage"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("doubling"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: storage_buf.as_entire_binding() }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("encoder") });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = ((data.len() as u32) + 63) / 64;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&storage_buf, 0, &staging_buf, 0, byte_len);
    queue.submit(std::iter::once(encoder.finish()));

    pollster::block_on(async {
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec()
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const IGNORE_REASON: &str = "requires Metal GPU - run on \
    macOS with: cargo test --test metal_compute_pipeline_tests \
    -- --ignored";

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_compute_pipeline_tests -- --ignored"]
fn test_metal_adapter_selection() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));

    let adapter = adapter.expect("wgpu should find a Metal adapter on macOS");
    let info = adapter.get_info();
    assert_eq!(info.backend, wgpu::Backend::Metal, "Adapter backend must be Metal");
    assert!(!info.name.is_empty(), "Adapter name should be non-empty (e.g. 'Apple M1')");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_compute_pipeline_tests -- --ignored"]
fn test_metal_device_limits() {
    let (device, _queue) =
        create_metal_device().expect("Metal device should be available on macOS");
    let limits = device.limits();

    assert!(
        limits.max_compute_workgroup_size_x >= 256,
        "max_compute_workgroup_size_x should be >= 256, got {}",
        limits.max_compute_workgroup_size_x
    );
    assert!(
        limits.max_compute_workgroups_per_dimension >= 65535,
        "max_compute_workgroups_per_dimension should be >= 65535, got {}",
        limits.max_compute_workgroups_per_dimension
    );
    assert!(
        limits.max_buffer_size >= 256 * 1024 * 1024,
        "max_buffer_size should be >= 256 MiB, got {}",
        limits.max_buffer_size
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_compute_pipeline_tests -- --ignored"]
fn test_metal_shader_compilation() {
    let (device, _queue) =
        create_metal_device().expect("Metal device should be available on macOS");

    // Push an error scope so we can detect compilation failures.
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let _module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("doubling_shader"),
        source: wgpu::ShaderSource::Wgsl(DOUBLING_SHADER.into()),
    });

    let error = pollster::block_on(device.pop_error_scope());
    assert!(error.is_none(), "Shader compilation should succeed, got error: {error:?}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_compute_pipeline_tests -- --ignored"]
fn test_metal_buffer_roundtrip() {
    let (device, queue) = create_metal_device().expect("Metal device should be available on macOS");

    let original: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
    let byte_len = (original.len() * std::mem::size_of::<f32>()) as u64;

    // Upload via an init buffer, copy to a staging buffer, read back.
    let gpu_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_buf"),
        contents: bytemuck::cast_slice(&original),
        usage: wgpu::BufferUsages::COPY_SRC,
    });

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("copy_encoder") });
    encoder.copy_buffer_to_buffer(&gpu_buf, 0, &staging, 0, byte_len);
    queue.submit(std::iter::once(encoder.finish()));

    let readback = pollster::block_on(async {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec()
    });

    assert_eq!(readback.len(), original.len());
    for (i, (&got, &expected)) in readback.iter().zip(original.iter()).enumerate() {
        assert!(
            (got - expected).abs() < f32::EPSILON,
            "Mismatch at index {i}: got {got}, expected {expected}"
        );
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_compute_pipeline_tests -- --ignored"]
fn test_metal_compute_dispatch() {
    let (device, queue) = create_metal_device().expect("Metal device should be available on macOS");

    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let result = run_doubling_dispatch(&device, &queue, &input, DOUBLING_SHADER);

    let expected = [2.0_f32, 4.0, 6.0, 8.0];
    assert_eq!(result.len(), expected.len());
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "Mismatch at index {i}: got {got}, expected {exp}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_compute_pipeline_tests -- --ignored"]
fn test_metal_large_buffer_alignment() {
    let (device, _queue) =
        create_metal_device().expect("Metal device should be available on macOS");

    for size in [256_u64, 512, 1024, 4096] {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aligned_buf"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        assert!(
            buf.size() >= size,
            "Buffer size {actual} should be >= requested {size}",
            actual = buf.size()
        );
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_compute_pipeline_tests -- --ignored"]
fn test_metal_multiple_dispatches() {
    let (device, queue) = create_metal_device().expect("Metal device should be available on macOS");

    // Start with [1.0; 4], double three times → expect [8.0; 4].
    let mut data = vec![1.0_f32; 4];
    for _ in 0..3 {
        data = run_doubling_dispatch(&device, &queue, &data, DOUBLING_SHADER);
    }

    for (i, &val) in data.iter().enumerate() {
        assert!((val - 8.0).abs() < 1e-6, "After 3 doublings index {i}: got {val}, expected 8.0");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with: cargo test --test metal_compute_pipeline_tests -- --ignored"]
fn test_metal_workgroup_size_limits() {
    let (device, queue) = create_metal_device().expect("Metal device should be available on macOS");

    let input: Vec<f32> = (0..256).map(|i| i as f32).collect();

    for wg_size in [1_u32, 32, 64, 128, 256] {
        let shader = doubling_shader_with_workgroup_size(wg_size);
        let result = run_doubling_dispatch(&device, &queue, &input, &shader);

        assert_eq!(
            result.len(),
            input.len(),
            "Result length mismatch for workgroup_size={wg_size}"
        );
        for (i, (&got, &orig)) in result.iter().zip(input.iter()).enumerate() {
            let expected = orig * 2.0;
            assert!(
                (got - expected).abs() < 1e-6,
                "workgroup_size={wg_size} index {i}: \
                 got {got}, expected {expected}"
            );
        }
    }
}
