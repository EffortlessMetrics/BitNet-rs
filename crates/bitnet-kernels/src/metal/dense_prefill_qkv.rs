//! Runtime dispatch boundary for the Apple Metal dense prefill Q/K/V fixture.
//!
//! This module intentionally exposes only the validated phase fixture. It does
//! not route resident generation or claim full Metal inference.

use std::fmt;

use crate::metal::smoke::DenseMetalPrefillQkvFixture;
#[cfg(all(feature = "metal-runtime", target_os = "macos", target_arch = "aarch64"))]
use crate::metal::smoke::{
    DENSE_METAL_PREFILL_QKV_KERNEL_ID, SMOKE_WORKGROUP_SIZE, dense_prefill_qkv_shape_words,
};

#[derive(Debug, Clone, PartialEq)]
pub struct DensePrefillQkvMetalOutput {
    pub adapter_name: String,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DensePrefillQkvMetalError {
    UnsupportedTarget { target: &'static str },
    AdapterUnavailable { phase: &'static str },
    WrongBackend { backend: String },
    DeviceCreation { message: String },
    OutputMap { message: String },
    OutputReadback { message: String },
    OutputShape { expected: usize, actual: usize },
}

impl fmt::Display for DensePrefillQkvMetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedTarget { target } => {
                write!(f, "dense prefill Q/K/V Metal dispatch is unsupported on {target}")
            }
            Self::AdapterUnavailable { phase } => {
                write!(f, "no Metal adapter found for {phase}")
            }
            Self::WrongBackend { backend } => {
                write!(f, "dense prefill Q/K/V requires Metal backend, found {backend}")
            }
            Self::DeviceCreation { message } => {
                write!(f, "failed to create Metal device: {message}")
            }
            Self::OutputMap { message } => {
                write!(f, "failed to map dense prefill Q/K/V Metal output: {message}")
            }
            Self::OutputReadback { message } => {
                write!(f, "failed to read dense prefill Q/K/V Metal output: {message}")
            }
            Self::OutputShape { expected, actual } => write!(
                f,
                "dense prefill Q/K/V Metal output length mismatch: expected {expected}, actual {actual}"
            ),
        }
    }
}

impl std::error::Error for DensePrefillQkvMetalError {}

pub fn dense_prefill_qkv_runtime_api_available() -> bool {
    cfg!(all(feature = "metal-runtime", target_os = "macos", target_arch = "aarch64"))
}

pub fn run_dense_prefill_qkv_projection_blocking(
    fixture: &DenseMetalPrefillQkvFixture,
) -> Result<DensePrefillQkvMetalOutput, DensePrefillQkvMetalError> {
    #[cfg(all(feature = "metal-runtime", target_os = "macos", target_arch = "aarch64"))]
    {
        pollster::block_on(run_dense_prefill_qkv_projection(fixture))
    }

    #[cfg(not(all(feature = "metal-runtime", target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = fixture;
        Err(DensePrefillQkvMetalError::UnsupportedTarget { target: std::env::consts::ARCH })
    }
}

#[cfg(all(feature = "metal-runtime", target_os = "macos", target_arch = "aarch64"))]
pub async fn run_dense_prefill_qkv_projection(
    fixture: &DenseMetalPrefillQkvFixture,
) -> Result<DensePrefillQkvMetalOutput, DensePrefillQkvMetalError> {
    use wgpu::util::DeviceExt;

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
        .ok_or(DensePrefillQkvMetalError::AdapterUnavailable {
            phase: DENSE_METAL_PREFILL_QKV_KERNEL_ID,
        })?;

    let adapter_info = adapter.get_info();
    if adapter_info.backend != wgpu::Backend::Metal {
        return Err(DensePrefillQkvMetalError::WrongBackend {
            backend: format!("{:?}", adapter_info.backend),
        });
    }

    let (device, queue) =
        adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.map_err(
            |error| DensePrefillQkvMetalError::DeviceCreation { message: error.to_string() },
        )?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(DENSE_METAL_PREFILL_QKV_KERNEL_ID),
        source: wgpu::ShaderSource::Wgsl(DENSE_PREFILL_QKV_SHADER.into()),
    });

    let activations_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tiny_metal_dense_prefill_qkv_activations"),
        contents: bytemuck::cast_slice(&fixture.activations),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let q_weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tiny_metal_dense_prefill_qkv_q_weights"),
        contents: bytemuck::cast_slice(&fixture.q_weights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let k_weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tiny_metal_dense_prefill_qkv_k_weights"),
        contents: bytemuck::cast_slice(&fixture.k_weights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let v_weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tiny_metal_dense_prefill_qkv_v_weights"),
        contents: bytemuck::cast_slice(&fixture.v_weights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut bias =
        Vec::with_capacity(fixture.q_bias.len() + fixture.k_bias.len() + fixture.v_bias.len());
    bias.extend_from_slice(&fixture.q_bias);
    bias.extend_from_slice(&fixture.k_bias);
    bias.extend_from_slice(&fixture.v_bias);
    let bias_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tiny_metal_dense_prefill_qkv_bias"),
        contents: bytemuck::cast_slice(&bias),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let shape_words = dense_prefill_qkv_shape_words(fixture);
    let shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tiny_metal_dense_prefill_qkv_shape"),
        contents: bytemuck::cast_slice(&shape_words),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_len = fixture.expected_q.len() + fixture.expected_k.len() + fixture.expected_v.len();
    let byte_len = (output_len * std::mem::size_of::<f32>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tiny_metal_dense_prefill_qkv_output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tiny_metal_dense_prefill_qkv_staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tiny_metal_dense_prefill_qkv_layout"),
        entries: &[
            storage_buffer_entry(0, true),
            storage_buffer_entry(1, true),
            storage_buffer_entry(2, true),
            storage_buffer_entry(3, true),
            storage_buffer_entry(4, true),
            storage_buffer_entry(5, false),
            storage_buffer_entry(6, true),
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tiny_metal_dense_prefill_qkv_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: activations_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q_weights_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k_weights_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v_weights_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bias_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: shape_buffer.as_entire_binding() },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("tiny_metal_dense_prefill_qkv_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("tiny_metal_dense_prefill_qkv_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tiny_metal_dense_prefill_qkv_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tiny_metal_dense_prefill_qkv_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((output_len as u32).div_ceil(SMOKE_WORKGROUP_SIZE), 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, byte_len);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|error| DensePrefillQkvMetalError::OutputReadback { message: error.to_string() })?
        .map_err(|error| DensePrefillQkvMetalError::OutputMap { message: error.to_string() })?;

    let data = slice.get_mapped_range();
    let output = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    split_qkv_output(fixture, adapter_info.name, output)
}

#[cfg(any(test, all(feature = "metal-runtime", target_os = "macos", target_arch = "aarch64")))]
fn split_qkv_output(
    fixture: &DenseMetalPrefillQkvFixture,
    adapter_name: String,
    output: Vec<f32>,
) -> Result<DensePrefillQkvMetalOutput, DensePrefillQkvMetalError> {
    let expected_len =
        fixture.expected_q.len() + fixture.expected_k.len() + fixture.expected_v.len();
    if output.len() != expected_len {
        return Err(DensePrefillQkvMetalError::OutputShape {
            expected: expected_len,
            actual: output.len(),
        });
    }

    let q_end = fixture.expected_q.len();
    let k_end = q_end + fixture.expected_k.len();
    Ok(DensePrefillQkvMetalOutput {
        adapter_name,
        q: output[..q_end].to_vec(),
        k: output[q_end..k_end].to_vec(),
        v: output[k_end..].to_vec(),
    })
}

#[cfg(all(feature = "metal-runtime", target_os = "macos", target_arch = "aarch64"))]
fn storage_buffer_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

#[cfg(all(feature = "metal-runtime", target_os = "macos", target_arch = "aarch64"))]
const DENSE_PREFILL_QKV_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> activations: array<f32>;
@group(0) @binding(1) var<storage, read> q_weights: array<f32>;
@group(0) @binding(2) var<storage, read> k_weights: array<f32>;
@group(0) @binding(3) var<storage, read> v_weights: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<storage, read> shape: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_index = global_id.x;
    let batch_size = shape[0];
    let hidden_size = shape[1];
    let q_dim = shape[2];
    let kv_dim = shape[3];
    let q_count = batch_size * q_dim;
    let kv_count = batch_size * kv_dim;
    let total = q_count + kv_count + kv_count;

    if output_index >= total {
        return;
    }

    if output_index < q_count {
        let row = output_index / q_dim;
        let col = output_index % q_dim;
        var acc = bias[col];
        var k_index = 0u;
        loop {
            if k_index >= hidden_size {
                break;
            }
            acc = acc + activations[row * hidden_size + k_index] *
                q_weights[col * hidden_size + k_index];
            k_index = k_index + 1u;
        }
        output[output_index] = acc;
        return;
    }

    if output_index < q_count + kv_count {
        let local_index = output_index - q_count;
        let row = local_index / kv_dim;
        let col = local_index % kv_dim;
        var acc = bias[q_dim + col];
        var k_index = 0u;
        loop {
            if k_index >= hidden_size {
                break;
            }
            acc = acc + activations[row * hidden_size + k_index] *
                k_weights[col * hidden_size + k_index];
            k_index = k_index + 1u;
        }
        output[output_index] = acc;
        return;
    }

    let local_index = output_index - q_count - kv_count;
    let row = local_index / kv_dim;
    let col = local_index % kv_dim;
    var acc = bias[q_dim + kv_dim + col];
    var k_index = 0u;
    loop {
        if k_index >= hidden_size {
            break;
        }
        acc = acc + activations[row * hidden_size + k_index] *
            v_weights[col * hidden_size + k_index];
        k_index = k_index + 1u;
    }
    output[output_index] = acc;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::smoke::dense_metal_prefill_qkv_fixture;

    #[test]
    fn unsupported_target_is_explicit_off_apple_silicon() {
        if dense_prefill_qkv_runtime_api_available() {
            return;
        }

        let fixture = dense_metal_prefill_qkv_fixture();
        let error = run_dense_prefill_qkv_projection_blocking(&fixture)
            .expect_err("non-Apple-Silicon targets must fail clearly");
        assert!(matches!(error, DensePrefillQkvMetalError::UnsupportedTarget { .. }));
    }

    #[test]
    fn split_qkv_output_preserves_projection_boundaries() {
        let fixture = dense_metal_prefill_qkv_fixture();
        let mut output = Vec::new();
        output.extend_from_slice(&fixture.expected_q);
        output.extend_from_slice(&fixture.expected_k);
        output.extend_from_slice(&fixture.expected_v);

        let split = split_qkv_output(&fixture, "test-adapter".to_string(), output)
            .expect("valid concatenated qkv output");
        assert_eq!(split.adapter_name, "test-adapter");
        assert_eq!(split.q, fixture.expected_q);
        assert_eq!(split.k, fixture.expected_k);
        assert_eq!(split.v, fixture.expected_v);
    }
}
