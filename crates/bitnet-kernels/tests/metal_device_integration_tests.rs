#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Integration tests for Metal GPU device creation and compute on Apple Silicon.
//!
//! These tests verify that wgpu can enumerate Metal adapters, create devices,
//! and dispatch simple compute shaders on macOS aarch64 (Apple Silicon).
//! All tests are `#[ignore]` because CI runs on Linux.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Helper: create wgpu Instance → Adapter → (Device, Queue) targeting Metal
// ---------------------------------------------------------------------------

struct MetalContext {
    #[allow(dead_code)]
    instance: wgpu::Instance,
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
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires macOS Metal GPU - run on Apple Silicon"]
fn test_metal_adapter_enumeration() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..Default::default()
    });

    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::METAL);
    assert!(!adapters.is_empty(), "Expected at least one Metal adapter on Apple Silicon");

    let info = adapters[0].get_info();
    assert_eq!(info.backend, wgpu::Backend::Metal, "First adapter should be Metal backend");
    assert!(!info.name.is_empty(), "Adapter name should be non-empty (e.g. 'Apple M1')");
}

#[test]
#[ignore = "requires macOS Metal GPU - run on Apple Silicon"]
fn test_metal_device_and_queue_creation() {
    let ctx = create_metal_context();

    let info = ctx.adapter.get_info();
    assert_eq!(info.backend, wgpu::Backend::Metal);
    // Device + queue creation succeeded (verified by reaching this point).
    assert!(
        info.device_type == wgpu::DeviceType::IntegratedGpu
            || info.device_type == wgpu::DeviceType::DiscreteGpu,
        "Expected a GPU device type, got {:?}",
        info.device_type
    );
}

#[test]
#[ignore = "requires macOS Metal GPU - run on Apple Silicon"]
fn test_metal_device_limits() {
    let ctx = create_metal_context();
    let limits = ctx.device.limits();

    // Apple Silicon Metal supports generous buffer sizes and workgroup dimensions.
    assert!(
        limits.max_buffer_size >= 256 * 1024 * 1024,
        "max_buffer_size should be >= 256 MiB, got {}",
        limits.max_buffer_size
    );
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
        limits.max_storage_buffer_binding_size >= 128 * 1024 * 1024,
        "max_storage_buffer_binding_size should be >= 128 MiB, got {}",
        limits.max_storage_buffer_binding_size
    );
    assert!(
        limits.max_bind_groups >= 4,
        "max_bind_groups should be >= 4, got {}",
        limits.max_bind_groups
    );
}

#[test]
#[ignore = "requires macOS Metal GPU - run on Apple Silicon"]
fn test_metal_compute_shader_add_buffers() {
    let ctx = create_metal_context();

    // WGSL shader: element-wise addition of two f32 buffers → output buffer.
    let shader_src = r#"
        @group(0) @binding(0) var<storage, read>       a: array<f32>;
        @group(0) @binding(1) var<storage, read>       b: array<f32>;
        @group(0) @binding(2) var<storage, read_write>  result: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let idx = gid.x;
            if idx < arrayLength(&a) {
                result[idx] = a[idx] + b[idx];
            }
        }
    "#;

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("add_buffers"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let n: usize = 1024;
    let input_a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let input_b: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();

    let buf_a = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("buf_a"),
        contents: bytemuck::cast_slice(&input_a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_b = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("buf_b"),
        contents: bytemuck::cast_slice(&input_b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_result = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("buf_result"),
        size: (n * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("buf_staging"),
        size: (n * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("add_layout"),
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
        ],
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("add_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_result.as_entire_binding() },
        ],
    });

    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("add_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("add_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("add_encoder") });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("add_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (n as u32).div_ceil(64);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&buf_result, 0, &buf_staging, 0, buf_staging.size());
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let result = pollster::block_on(async {
        let slice = buf_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, f32>(&data).to_vec()
    });

    assert_eq!(result.len(), n);
    for i in 0..n {
        let expected = input_a[i] + input_b[i];
        assert!(
            (result[i] - expected).abs() < 1e-6,
            "Mismatch at index {i}: got {}, expected {expected}",
            result[i]
        );
    }
}

#[test]
#[ignore = "requires macOS Metal GPU - run on Apple Silicon"]
fn test_metal_buffer_alignment_256() {
    let ctx = create_metal_context();

    // Metal requires 256-byte alignment for certain operations.
    let aligned_size: u64 = 256;
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("aligned_256"),
        size: aligned_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Buffer creation should succeed; verify the size is at least what we requested.
    assert!(buf.size() >= aligned_size);
}

#[test]
#[ignore = "requires macOS Metal GPU - run on Apple Silicon"]
fn test_metal_buffer_alignment_4096() {
    let ctx = create_metal_context();

    // 4096-byte (page-aligned) buffer — common for large tensor storage.
    let aligned_size: u64 = 4096;
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("aligned_4096"),
        size: aligned_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    assert!(buf.size() >= aligned_size);
}

#[test]
#[ignore = "requires macOS Metal GPU - run on Apple Silicon"]
fn test_metal_oversized_buffer_error() {
    let ctx = create_metal_context();
    let limits = ctx.device.limits();

    // Request a buffer larger than the device maximum — should fail gracefully.
    let oversized = limits.max_buffer_size + 1;

    // wgpu validates against device limits. Pushing an error scope lets us
    // capture the validation error without panicking.
    ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oversized"),
        size: oversized,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let error = pollster::block_on(ctx.device.pop_error_scope());
    assert!(
        error.is_some(),
        "Expected a validation error for buffer size {oversized} exceeding max {}",
        limits.max_buffer_size
    );
}
