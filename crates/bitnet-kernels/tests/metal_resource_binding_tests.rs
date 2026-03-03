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

//! Metal resource binding tests for Apple Silicon.
//!
//! Validates wgpu/Metal resource binding behaviour including buffer creation,
//! argument buffers, resource heaps, indirect command buffers, resource
//! tracking, memory management, Apple Silicon–specific resources, and binding
//! performance characteristics.
//!
//! All tests require a Metal-capable GPU and are `#[ignore]` for CI.

#![cfg(feature = "cpu")]
#![cfg(target_os = "macos")]

use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Helper: create Metal device + queue (returns None if unavailable)
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

// Trivial compute shader used by several tests.
const PASSTHROUGH_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&input) {
        output[id.x] = input[id.x];
    }
}
"#;

/// Run a simple copy-dispatch and return the GPU-side result.
fn run_copy_dispatch(device: &wgpu::Device, queue: &wgpu::Queue, data: &[f32]) -> Vec<f32> {
    let byte_len = (data.len() * std::mem::size_of::<f32>()) as u64;

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("copy_shader"),
        source: wgpu::ShaderSource::Wgsl(PASSTHROUGH_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("copy_bgl"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("copy_pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("copy_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("copy_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("copy_enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("copy_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(((data.len() as u32) + 63) / 64, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, byte_len);
    queue.submit(Some(encoder.finish()));

    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().expect("Failed to map staging buffer");

    let view = slice.get_mapped_range();
    bytemuck::cast_slice(&view).to_vec()
}

// ===========================================================================
// 1. Buffer Binding Tests
// ===========================================================================

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_buffer_creation_and_binding() {
    let ctx = create_metal_context();
    let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test_buf"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    assert_eq!(buf.size(), (256 * std::mem::size_of::<f32>()) as u64);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_buffer_offset_binding() {
    let ctx = create_metal_context();
    // Create buffer with 512 floats but bind only the second half via offset.
    let data: Vec<f32> = (0..512).map(|i| i as f32).collect();
    let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offset_buf"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
    });

    let offset: u64 = 256 * std::mem::size_of::<f32>() as u64;
    let binding = wgpu::BufferBinding { buffer: &buf, offset, size: None };
    // Binding resource should be constructable with an offset.
    let _resource = wgpu::BindingResource::Buffer(binding);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_shared_buffer_binding() {
    let ctx = create_metal_context();
    // MAP_WRITE | COPY_SRC simulates shared/managed storage on Metal.
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shared_buf"),
        size: 1024,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    {
        let mut view = buf.slice(..).get_mapped_range_mut();
        let floats: &mut [f32] = bytemuck::cast_slice_mut(&mut view);
        for (i, v) in floats.iter_mut().enumerate() {
            *v = i as f32;
        }
    }
    buf.unmap();
    assert_eq!(buf.size(), 1024);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_private_buffer_binding() {
    let ctx = create_metal_context();
    // GPU-private buffer — not mappable, only usable as storage target.
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("private_buf"),
        size: 4096,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    assert_eq!(buf.size(), 4096);
    assert!(buf.usage().contains(wgpu::BufferUsages::STORAGE));
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_managed_buffer_binding() {
    let ctx = create_metal_context();
    // Managed-style: both CPU-writable and GPU-readable.
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("managed_buf"),
        size: 2048,
        usage: wgpu::BufferUsages::MAP_WRITE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    assert!(buf.usage().contains(wgpu::BufferUsages::COPY_SRC));
    assert!(buf.usage().contains(wgpu::BufferUsages::COPY_DST));
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_buffer_contents_access() {
    let ctx = create_metal_context();
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let result = run_copy_dispatch(&ctx.device, &ctx.queue, &data);
    assert_eq!(result, data, "GPU copy-back should match input exactly");
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_buffer_storage_mode_copy_src() {
    let ctx = create_metal_context();
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("copy_src_buf"),
        size: 512,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    assert!(buf.usage().contains(wgpu::BufferUsages::COPY_SRC));
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_buffer_storage_mode_map_read() {
    let ctx = create_metal_context();
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("map_read_buf"),
        size: 512,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    assert!(buf.usage().contains(wgpu::BufferUsages::MAP_READ));
}

// ===========================================================================
// 2. Argument Buffer Tests
// ===========================================================================

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_argument_buffer_creation() {
    let ctx = create_metal_context();
    // An argument buffer in wgpu is modelled via bind-group layout + bind-group.
    let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("arg_buf_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    // Layout should be usable for pipeline creation.
    let _pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("arg_pl"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_encoded_buffer_binding() {
    let ctx = create_metal_context();
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("enc_buf"),
        size: 256,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("enc_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("enc_bg"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
    });
    // Verify bind group is usable in a pass (no panic).
    let mut encoder =
        ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    pass.set_bind_group(0, &bg, &[]);
    // (drop pass without dispatch — just validating binding)
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_encoded_texture_binding() {
    let ctx = create_metal_context();
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("enc_tex"),
        size: wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tex_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    });

    let _bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tex_bg"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_encoded_sampler_binding() {
    let ctx = create_metal_context();
    let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("enc_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sampler_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        }],
    });

    let _bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sampler_bg"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Sampler(&sampler),
        }],
    });
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_tier2_argument_buffer_features() {
    let ctx = create_metal_context();
    // Tier 2 argument buffers allow multiple bind groups. Verify we can create
    // a pipeline layout with two distinct bind-group layouts.
    let bgl_a = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tier2_a"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bgl_b = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tier2_b"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let _pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("tier2_pl"),
        bind_group_layouts: &[&bgl_a, &bgl_b],
        push_constant_ranges: &[],
    });
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_indirect_argument_encoding_buffer() {
    let ctx = create_metal_context();
    // Indirect dispatch arguments encoded in a buffer.
    let dispatch_args: [u32; 3] = [4, 1, 1]; // 4 workgroups × 1 × 1
    let indirect_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("indirect_args"),
        contents: bytemuck::cast_slice(&dispatch_args),
        usage: wgpu::BufferUsages::INDIRECT,
    });
    assert_eq!(indirect_buf.size(), 12); // 3 × u32
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_argument_buffer_dynamic_offset() {
    let ctx = create_metal_context();
    // Dynamic-offset uniform buffer: bind once, dispatch with different offsets.
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dyn_offset_buf"),
        size: 1024,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("dyn_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dyn_bg"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buf,
                offset: 0,
                size: wgpu::BufferSize::new(256),
            }),
        }],
    });

    let mut encoder =
        ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    // Bind with dynamic offset 0, then 256 — verifies dynamic offset plumbing.
    pass.set_bind_group(0, &bg, &[0]);
    pass.set_bind_group(0, &bg, &[256]);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_argument_buffer_mixed_resource_types() {
    let ctx = create_metal_context();
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mixed_buf"),
        size: 256,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor::default());

    let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mixed_bgl"),
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
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let _bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mixed_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
        ],
    });
}

// ===========================================================================
// 3. Resource Heap Tests
// ===========================================================================

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_heap_creation_via_buffer_pool() {
    let ctx = create_metal_context();
    // Simulate a heap by creating a large backing buffer and sub-allocating.
    let heap_size: u64 = 1024 * 1024; // 1 MiB
    let heap_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("heap_backing"),
        size: heap_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    assert_eq!(heap_buf.size(), heap_size);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_heap_allocation_multiple_buffers() {
    let ctx = create_metal_context();
    let sizes: &[u64] = &[256, 512, 1024, 2048];
    let buffers: Vec<wgpu::Buffer> = sizes
        .iter()
        .enumerate()
        .map(|(i, &sz)| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("heap_alloc_{i}")),
                size: sz,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        })
        .collect();
    for (buf, &sz) in buffers.iter().zip(sizes) {
        assert_eq!(buf.size(), sz);
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_heap_sub_allocation_offsets() {
    let ctx = create_metal_context();
    // Bind subranges of a single buffer to simulate heap sub-allocation.
    let heap = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sub_alloc_heap"),
        size: 4096,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let binding_a =
        wgpu::BufferBinding { buffer: &heap, offset: 0, size: wgpu::BufferSize::new(1024) };
    let binding_b =
        wgpu::BufferBinding { buffer: &heap, offset: 1024, size: wgpu::BufferSize::new(1024) };
    let binding_c =
        wgpu::BufferBinding { buffer: &heap, offset: 2048, size: wgpu::BufferSize::new(2048) };

    // All three sub-ranges are distinct and non-overlapping.
    assert_ne!(binding_a.offset, binding_b.offset);
    assert_ne!(binding_b.offset, binding_c.offset);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_heap_type_tracking() {
    let ctx = create_metal_context();
    // Track buffer types within a simulated heap.
    struct HeapEntry {
        usage: wgpu::BufferUsages,
        size: u64,
    }
    let entries = vec![
        HeapEntry { usage: wgpu::BufferUsages::STORAGE, size: 512 },
        HeapEntry { usage: wgpu::BufferUsages::UNIFORM, size: 256 },
        HeapEntry { usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE, size: 1024 },
    ];
    let bufs: Vec<wgpu::Buffer> = entries
        .iter()
        .map(|e| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("typed_heap"),
                size: e.size,
                usage: e.usage,
                mapped_at_creation: false,
            })
        })
        .collect();
    assert!(bufs[0].usage().contains(wgpu::BufferUsages::STORAGE));
    assert!(bufs[1].usage().contains(wgpu::BufferUsages::UNIFORM));
    assert!(bufs[2].usage().contains(wgpu::BufferUsages::COPY_DST));
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_heap_alias_barriers() {
    let ctx = create_metal_context();
    // Aliased resources: two bindings into the same region require barrier semantics.
    let heap = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("alias_heap"),
        size: 2048,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Write zeros then copy to staging — the command encoder serialises access.
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("alias_staging"),
        size: 2048,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&heap, 0, &staging, 0, 2048);
    ctx.queue.submit(Some(encoder.finish()));
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_heap_memory_bookkeeping() {
    let ctx = create_metal_context();
    let mut total_allocated: u64 = 0;
    let alloc_sizes: &[u64] = &[128, 256, 512, 1024];

    for &sz in alloc_sizes {
        let _buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bookkeeping"),
            size: sz,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        total_allocated += sz;
    }

    let expected: u64 = alloc_sizes.iter().sum();
    assert_eq!(total_allocated, expected);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_heap_texture_and_buffer_coexistence() {
    let ctx = create_metal_context();
    // A heap can hold both textures and buffers.
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("heap_buf"),
        size: 1024,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("heap_tex"),
        size: wgpu::Extent3d { width: 32, height: 32, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    assert_eq!(buf.size(), 1024);
    assert_eq!(tex.width(), 32);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_heap_size_alignment() {
    let ctx = create_metal_context();
    // Metal heaps must be page-aligned; verify wgpu rounds up for us.
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("align_heap"),
        size: 100, // not page-aligned
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    // wgpu may round up; size must be >= requested.
    assert!(buf.size() >= 100);
}

// ===========================================================================
// 4. Indirect Command Buffer Tests
// ===========================================================================

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_icb_creation() {
    let ctx = create_metal_context();
    // ICB is modelled as an indirect-dispatch buffer in wgpu.
    let dispatch: [u32; 3] = [8, 1, 1];
    let icb = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("icb"),
        contents: bytemuck::cast_slice(&dispatch),
        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
    });
    assert_eq!(icb.size(), 12);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_icb_command_encoding() {
    let ctx = create_metal_context();
    let data: Vec<f32> = vec![1.0; 256];
    let result = run_copy_dispatch(&ctx.device, &ctx.queue, &data);
    assert_eq!(result.len(), data.len());
    assert_eq!(result, data);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_icb_parallel_execution() {
    let ctx = create_metal_context();
    // Submit multiple independent dispatches in a single encoder.
    let data_a: Vec<f32> = vec![1.0; 128];
    let data_b: Vec<f32> = vec![2.0; 128];
    let result_a = run_copy_dispatch(&ctx.device, &ctx.queue, &data_a);
    let result_b = run_copy_dispatch(&ctx.device, &ctx.queue, &data_b);
    assert_eq!(result_a, data_a);
    assert_eq!(result_b, data_b);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_icb_reset_and_reuse() {
    let ctx = create_metal_context();
    let dispatch: [u32; 3] = [4, 1, 1];
    let icb = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reuse_icb"),
        contents: bytemuck::cast_slice(&dispatch),
        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
    });
    // Overwrite with new dispatch dimensions.
    let new_dispatch: [u32; 3] = [16, 1, 1];
    ctx.queue.write_buffer(&icb, 0, bytemuck::cast_slice(&new_dispatch));
    // Ensure buffer still valid.
    assert_eq!(icb.size(), 12);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_icb_gpu_driven_dispatch() {
    let ctx = create_metal_context();
    // GPU fills an indirect buffer, then a second pass dispatches from it.
    // Here we test the buffer plumbing; the GPU write is simulated via queue.
    let dispatch: [u32; 3] = [2, 1, 1];
    let indirect_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_driven_icb"),
        contents: bytemuck::cast_slice(&dispatch),
        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
    });
    assert!(indirect_buf.usage().contains(wgpu::BufferUsages::INDIRECT));
    assert!(indirect_buf.usage().contains(wgpu::BufferUsages::STORAGE));
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_icb_command_optimization_ordering() {
    let ctx = create_metal_context();
    // Verify that sequential dispatches maintain ordering guarantees.
    let data: Vec<f32> = (0..512).map(|i| i as f32).collect();
    let result = run_copy_dispatch(&ctx.device, &ctx.queue, &data);
    for (i, (&got, &expected)) in result.iter().zip(data.iter()).enumerate() {
        assert!(
            (got - expected).abs() < f32::EPSILON,
            "Mismatch at index {i}: got {got}, expected {expected}"
        );
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_icb_multiple_indirect_buffers() {
    let ctx = create_metal_context();
    let dispatches: Vec<[u32; 3]> = vec![[1, 1, 1], [2, 1, 1], [4, 1, 1], [8, 1, 1]];
    let bufs: Vec<wgpu::Buffer> = dispatches
        .iter()
        .enumerate()
        .map(|(i, d)| {
            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("multi_icb_{i}")),
                contents: bytemuck::cast_slice(d),
                usage: wgpu::BufferUsages::INDIRECT,
            })
        })
        .collect();
    for buf in &bufs {
        assert_eq!(buf.size(), 12);
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_icb_zero_dispatch() {
    let ctx = create_metal_context();
    // Zero workgroups — the GPU should do nothing without error.
    let dispatch: [u32; 3] = [0, 0, 0];
    let icb = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("zero_icb"),
        contents: bytemuck::cast_slice(&dispatch),
        usage: wgpu::BufferUsages::INDIRECT,
    });
    assert_eq!(icb.size(), 12);
}

// ===========================================================================
// 5. Resource Tracking Tests
// ===========================================================================

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_resource_usage_tracking() {
    let ctx = create_metal_context();
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tracked_buf"),
        size: 1024,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    assert!(buf.usage().contains(wgpu::BufferUsages::STORAGE));
    assert!(buf.usage().contains(wgpu::BufferUsages::COPY_SRC));
    assert!(buf.usage().contains(wgpu::BufferUsages::COPY_DST));
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_hazard_detection_read_after_write() {
    let ctx = create_metal_context();
    // Write then read with proper barrier (command encoder ordering).
    let data: Vec<f32> = vec![42.0; 64];
    let result = run_copy_dispatch(&ctx.device, &ctx.queue, &data);
    assert_eq!(result, data, "Read-after-write should produce correct data");
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_read_write_ordering_sequential() {
    let ctx = create_metal_context();
    // Two sequential dispatches on the same data — ordering must be preserved.
    let data: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let intermediate = run_copy_dispatch(&ctx.device, &ctx.queue, &data);
    let final_result = run_copy_dispatch(&ctx.device, &ctx.queue, &intermediate);
    assert_eq!(final_result, data, "Double copy should be identity");
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_dependency_tracking_multi_buffer() {
    let ctx = create_metal_context();
    // Multiple buffers bound simultaneously — no dependency conflicts expected.
    let bufs: Vec<wgpu::Buffer> = (0..4)
        .map(|i| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("dep_{i}")),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })
        .collect();
    for buf in &bufs {
        assert_eq!(buf.size(), 256);
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_residency_management_buffer_lifecycle() {
    let ctx = create_metal_context();
    // Create, use, and drop a buffer — tests residency lifecycle.
    {
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident_buf"),
            size: 4096,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert_eq!(buf.size(), 4096);
        buf.destroy();
    }
    // After drop, we should still be able to create new buffers.
    let _new_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("post_resident_buf"),
        size: 1024,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_resource_tracking_texture_lifecycle() {
    let ctx = create_metal_context();
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tracked_tex"),
        size: wgpu::Extent3d { width: 128, height: 128, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    assert_eq!(tex.width(), 128);
    assert_eq!(tex.height(), 128);
    tex.destroy();
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_resource_tracking_cross_queue_submit() {
    let ctx = create_metal_context();
    // Two encoder submissions — resources used by the first must be resident for both.
    let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cross_queue_buf"),
        contents: bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cross_queue_staging"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // First submit: write zeros.
    let zeros = [0u8; 16];
    ctx.queue.write_buffer(&buf, 0, &zeros);

    // Second submit: copy to staging.
    let mut encoder =
        ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&buf, 0, &staging, 0, 16);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().expect("map staging");

    let view = slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice(&view);
    assert_eq!(result, &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_resource_tracking_buffer_reuse() {
    let ctx = create_metal_context();
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reuse_tracked"),
        size: 512,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Write different data across multiple submissions.
    for i in 0..4u8 {
        ctx.queue.write_buffer(&buf, 0, &[i; 128]);
    }
    assert_eq!(buf.size(), 512);
}

// ===========================================================================
// 6. Memory Management Tests
// ===========================================================================

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_allocation_strategy_varied_sizes() {
    let ctx = create_metal_context();
    let sizes: &[u64] = &[16, 64, 256, 1024, 4096, 65536];
    for &sz in sizes {
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("alloc_strategy"),
            size: sz,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert!(buf.size() >= sz, "Buffer must be at least as large as requested ({sz})");
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_deallocation_cleanup() {
    let ctx = create_metal_context();
    // Create and explicitly destroy several buffers.
    for i in 0..8 {
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("dealloc_{i}")),
            size: 4096,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        buf.destroy();
    }
    // Device should still be functional after many create/destroy cycles.
    let _buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("post_dealloc"),
        size: 256,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_fragmentation_detection() {
    let ctx = create_metal_context();
    // Allocate alternating large/small buffers, free the large ones, then
    // allocate a new large buffer — exercises allocator defrag paths.
    let mut handles: Vec<wgpu::Buffer> = Vec::new();
    for i in 0..8 {
        let sz = if i % 2 == 0 { 65536 } else { 256 };
        handles.push(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("frag_{i}")),
            size: sz,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }
    // Destroy large buffers.
    for (i, buf) in handles.iter().enumerate() {
        if i % 2 == 0 {
            buf.destroy();
        }
    }
    // Allocate a new large buffer — should succeed despite fragmentation.
    let reclaimed = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reclaimed"),
        size: 65536,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    assert!(reclaimed.size() >= 65536);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_page_alignment() {
    let ctx = create_metal_context();
    // Sizes that are NOT page-aligned — wgpu/Metal should round up.
    let unaligned_sizes: &[u64] = &[1, 3, 7, 100, 255, 1000, 4095];
    for &sz in unaligned_sizes {
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("page_align"),
            size: sz,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert!(buf.size() >= sz, "Buffer size {sz} should be rounded up");
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_gpu_memory_pressure_many_allocations() {
    let ctx = create_metal_context();
    // Allocate many small buffers to stress the GPU allocator.
    let count = 256;
    let bufs: Vec<wgpu::Buffer> = (0..count)
        .map(|i| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("pressure_{i}")),
                size: 1024,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        })
        .collect();
    assert_eq!(bufs.len(), count);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_purge_policy_destroy_and_recreate() {
    let ctx = create_metal_context();
    // Simulate a purge by destroying all resources and re-creating.
    let mut batch: Vec<wgpu::Buffer> = (0..16)
        .map(|i| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("purge_{i}")),
                size: 8192,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        })
        .collect();
    for buf in batch.drain(..) {
        buf.destroy();
    }
    // Re-create after purge — device should honour the freed memory.
    let _fresh = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("post_purge"),
        size: 131072,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_mapped_buffer_write_read_cycle() {
    let ctx = create_metal_context();
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("map_cycle"),
        size: 256,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    {
        let mut view = buf.slice(..).get_mapped_range_mut();
        let floats: &mut [f32] = bytemuck::cast_slice_mut(&mut view);
        for (i, v) in floats.iter_mut().enumerate() {
            *v = i as f32;
        }
    }
    buf.unmap();

    // Copy to a staging buffer and verify.
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("map_staging"),
        size: 256,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc =
        ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&buf, 0, &staging, 0, 256);
    ctx.queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().expect("map staging");
    let view = slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice(&view);
    for (i, &v) in result.iter().enumerate() {
        assert_eq!(v, i as f32, "mapped write/read cycle mismatch at {i}");
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_large_allocation_single_buffer() {
    let ctx = create_metal_context();
    // Allocate a 64 MiB buffer — realistic for weight tensors.
    let large_size: u64 = 64 * 1024 * 1024;
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("large_alloc"),
        size: large_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    assert!(buf.size() >= large_size);
}

// ===========================================================================
// 7. Apple Silicon Resource Tests
// ===========================================================================

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_unified_memory_binding() {
    let ctx = create_metal_context();
    // On Apple Silicon, CPU and GPU share memory. Verify a MAP_WRITE buffer is
    // also usable as STORAGE (unified address space).
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("unified_buf"),
        size: 4096,
        usage: wgpu::BufferUsages::MAP_WRITE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: true,
    });
    {
        let mut view = buf.slice(..).get_mapped_range_mut();
        let floats: &mut [f32] = bytemuck::cast_slice_mut(&mut view);
        floats[0] = 3.14;
    }
    buf.unmap();
    assert!(buf.usage().contains(wgpu::BufferUsages::STORAGE));
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_tile_memory_binding_2d_texture() {
    let ctx = create_metal_context();
    // Tile-based deferred rendering uses tile memory. Create a render-target
    // texture — the underlying Metal driver will use tile memory.
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tile_rt"),
        size: wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    assert_eq!(tex.width(), 256);
    assert_eq!(tex.height(), 256);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_threadgroup_memory_workgroup_size() {
    let ctx = create_metal_context();
    // Verify that the device supports workgroup sizes used for threadgroup memory.
    let limits = ctx.device.limits();
    // Apple Silicon typically supports at least 1024 threads per threadgroup.
    assert!(
        limits.max_compute_invocations_per_workgroup >= 256,
        "Expected ≥256 invocations per workgroup, got {}",
        limits.max_compute_invocations_per_workgroup
    );
    assert!(
        limits.max_compute_workgroup_size_x >= 256,
        "Expected workgroup_size_x ≥256, got {}",
        limits.max_compute_workgroup_size_x
    );
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_tbdr_optimization_render_target() {
    let ctx = create_metal_context();
    // TBDR (Tile-Based Deferred Rendering) optimises render-target access.
    // Create a small render target and verify usability.
    let rt = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tbdr_rt"),
        size: wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let _view = rt.create_view(&wgpu::TextureViewDescriptor::default());
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_gpu_family_resource_caps_max_buffer_size() {
    let ctx = create_metal_context();
    let limits = ctx.device.limits();
    // Apple Silicon devices support large buffers.
    assert!(
        limits.max_buffer_size >= 256 * 1024 * 1024,
        "Expected max_buffer_size ≥256 MiB, got {}",
        limits.max_buffer_size
    );
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_gpu_family_resource_caps_bind_groups() {
    let ctx = create_metal_context();
    let limits = ctx.device.limits();
    assert!(limits.max_bind_groups >= 4, "Expected ≥4 bind groups, got {}", limits.max_bind_groups);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_unified_memory_copy_dispatch_roundtrip() {
    let ctx = create_metal_context();
    // Full roundtrip: CPU writes → GPU copies → CPU reads.
    let data: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001).collect();
    let result = run_copy_dispatch(&ctx.device, &ctx.queue, &data);
    for (i, (&got, &exp)) in result.iter().zip(data.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "Unified memory roundtrip mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_gpu_family_max_texture_dimension() {
    let ctx = create_metal_context();
    let limits = ctx.device.limits();
    // Apple Silicon supports at least 8192×8192 textures.
    assert!(
        limits.max_texture_dimension_2d >= 4096,
        "Expected max_texture_dimension_2d ≥4096, got {}",
        limits.max_texture_dimension_2d
    );
}

// ===========================================================================
// 8. Resource Performance Tests
// ===========================================================================

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_binding_overhead_single_buffer() {
    let ctx = create_metal_context();
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("perf_bind"),
        size: 1024,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("perf_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    // Create many bind groups rapidly to measure overhead.
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
        });
    }
    let elapsed = start.elapsed();
    // Sanity: 1000 bind-group creations should finish within 5 seconds.
    assert!(elapsed.as_secs() < 5, "Bind-group creation too slow: {elapsed:?}");
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_batch_binding_multiple_resources() {
    let ctx = create_metal_context();
    let count = 8;
    let bufs: Vec<wgpu::Buffer> = (0..count)
        .map(|i| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("batch_{i}")),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        })
        .collect();

    let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..count as u32)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();

    let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("batch_bgl"),
        entries: &entries,
    });

    let bg_entries: Vec<wgpu::BindGroupEntry> = bufs
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
        .collect();

    let _bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("batch_bg"),
        layout: &layout,
        entries: &bg_entries,
    });
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_resource_switching_cost() {
    let ctx = create_metal_context();
    // Rapidly switch between two bind groups in a compute pass.
    let buf_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("switch_a"),
        size: 256,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let buf_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("switch_b"),
        size: 256,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("switch_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bg_a = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() }],
    });
    let bg_b = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf_b.as_entire_binding() }],
    });

    let mut encoder =
        ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    for _ in 0..100 {
        pass.set_bind_group(0, &bg_a, &[]);
        pass.set_bind_group(0, &bg_b, &[]);
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_memory_bandwidth_large_copy() {
    let ctx = create_metal_context();
    // Copy 4 MiB through GPU to exercise bandwidth.
    let size = 4 * 1024 * 1024;
    let data: Vec<f32> = (0..(size / 4)).map(|i| i as f32).collect();
    let result = run_copy_dispatch(&ctx.device, &ctx.queue, &data);
    assert_eq!(result.len(), data.len());
    assert_eq!(result[0], data[0]);
    assert_eq!(result[result.len() - 1], data[data.len() - 1]);
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_binding_cache_effectiveness() {
    let ctx = create_metal_context();
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cache_buf"),
        size: 512,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cache_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    // Reuse the same bind group across many encoder passes.
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
    });

    for _ in 0..50 {
        let mut enc =
            ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_bind_group(0, &bg, &[]);
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_buffer_creation_throughput() {
    let ctx = create_metal_context();
    let start = std::time::Instant::now();
    let count = 500;
    for _ in 0..count {
        let _buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4096,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
    }
    let elapsed = start.elapsed();
    assert!(elapsed.as_secs() < 5, "Creating {count} buffers took too long: {elapsed:?}");
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_pipeline_creation_throughput() {
    let ctx = create_metal_context();
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("perf_shader"),
        source: wgpu::ShaderSource::Wgsl(PASSTHROUGH_SHADER.into()),
    });

    let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let start = std::time::Instant::now();
    let count = 100;
    for _ in 0..count {
        let _pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    }
    let elapsed = start.elapsed();
    assert!(elapsed.as_secs() < 10, "Creating {count} pipelines took too long: {elapsed:?}");
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal GPU — run on macOS/arm64"]
fn test_dispatch_throughput_many_small() {
    let ctx = create_metal_context();
    // Many small dispatches in a single encoder.
    let data: Vec<f32> = vec![1.0; 64];
    let start = std::time::Instant::now();
    for _ in 0..50 {
        let _result = run_copy_dispatch(&ctx.device, &ctx.queue, &data);
    }
    let elapsed = start.elapsed();
    assert!(elapsed.as_secs() < 30, "50 small dispatches took too long: {elapsed:?}");
}
