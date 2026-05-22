//! Runtime dispatch boundary for the Apple Metal dense prefill attention-score fixture.
//!
//! This module intentionally exposes only Q*K^T score computation for one
//! deterministic phase fixture. It does not apply masks, softmax, value mixing,
//! output projection, decoding, or resident generation.

use std::fmt;

use crate::metal::smoke::DenseMetalPrefillAttentionScoresFixture;
#[cfg(all(feature = "metal-runtime", target_os = "macos", target_arch = "aarch64"))]
use crate::metal::smoke::{
    DENSE_METAL_PREFILL_ATTENTION_SCORES_KERNEL_ID, SMOKE_WORKGROUP_SIZE,
    dense_prefill_attention_scores_shape_words,
};

#[derive(Debug, Clone, PartialEq)]
pub struct DensePrefillAttentionScoresMetalOutput {
    pub adapter_name: String,
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DensePrefillAttentionScoresMetalError {
    UnsupportedTarget { target: &'static str },
    AdapterUnavailable { phase: &'static str },
    WrongBackend { backend: String },
    DeviceCreation { message: String },
    OutputMap { message: String },
    OutputReadback { message: String },
    OutputShape { expected: usize, actual: usize },
}

impl fmt::Display for DensePrefillAttentionScoresMetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedTarget { target } => {
                write!(f, "dense prefill attention-score Metal dispatch is unsupported on {target}")
            }
            Self::AdapterUnavailable { phase } => {
                write!(f, "no Metal adapter found for {phase}")
            }
            Self::WrongBackend { backend } => {
                write!(f, "dense prefill attention scores require Metal backend, found {backend}")
            }
            Self::DeviceCreation { message } => {
                write!(f, "failed to create Metal device: {message}")
            }
            Self::OutputMap { message } => {
                write!(f, "failed to map dense prefill attention-score Metal output: {message}")
            }
            Self::OutputReadback { message } => {
                write!(f, "failed to read dense prefill attention-score Metal output: {message}")
            }
            Self::OutputShape { expected, actual } => write!(
                f,
                "dense prefill attention-score Metal output length mismatch: expected {expected}, actual {actual}"
            ),
        }
    }
}

impl std::error::Error for DensePrefillAttentionScoresMetalError {}

pub fn dense_prefill_attention_scores_runtime_api_available() -> bool {
    cfg!(all(feature = "metal-runtime", target_os = "macos", target_arch = "aarch64"))
}

pub fn run_dense_prefill_attention_scores_blocking(
    fixture: &DenseMetalPrefillAttentionScoresFixture,
) -> Result<DensePrefillAttentionScoresMetalOutput, DensePrefillAttentionScoresMetalError> {
    #[cfg(all(feature = "metal-runtime", target_os = "macos", target_arch = "aarch64"))]
    {
        pollster::block_on(run_dense_prefill_attention_scores(fixture))
    }

    #[cfg(not(all(feature = "metal-runtime", target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = fixture;
        Err(DensePrefillAttentionScoresMetalError::UnsupportedTarget {
            target: std::env::consts::ARCH,
        })
    }
}

#[cfg(all(feature = "metal-runtime", target_os = "macos", target_arch = "aarch64"))]
pub async fn run_dense_prefill_attention_scores(
    fixture: &DenseMetalPrefillAttentionScoresFixture,
) -> Result<DensePrefillAttentionScoresMetalOutput, DensePrefillAttentionScoresMetalError> {
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
        .ok_or(DensePrefillAttentionScoresMetalError::AdapterUnavailable {
            phase: DENSE_METAL_PREFILL_ATTENTION_SCORES_KERNEL_ID,
        })?;

    let adapter_info = adapter.get_info();
    if adapter_info.backend != wgpu::Backend::Metal {
        return Err(DensePrefillAttentionScoresMetalError::WrongBackend {
            backend: format!("{:?}", adapter_info.backend),
        });
    }

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .map_err(|error| DensePrefillAttentionScoresMetalError::DeviceCreation {
            message: error.to_string(),
        })?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(DENSE_METAL_PREFILL_ATTENTION_SCORES_KERNEL_ID),
        source: wgpu::ShaderSource::Wgsl(DENSE_PREFILL_ATTENTION_SCORES_SHADER.into()),
    });

    let q_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tiny_metal_dense_prefill_attention_scores_q"),
        contents: bytemuck::cast_slice(&fixture.q),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let k_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tiny_metal_dense_prefill_attention_scores_k"),
        contents: bytemuck::cast_slice(&fixture.k),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let shape_words = dense_prefill_attention_scores_shape_words(fixture);
    let shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tiny_metal_dense_prefill_attention_scores_shape"),
        contents: bytemuck::cast_slice(&shape_words),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_len = fixture.expected_scores.len();
    let byte_len = (output_len * std::mem::size_of::<f32>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tiny_metal_dense_prefill_attention_scores_output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tiny_metal_dense_prefill_attention_scores_staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tiny_metal_dense_prefill_attention_scores_layout"),
        entries: &[
            storage_buffer_entry(0, true),
            storage_buffer_entry(1, true),
            storage_buffer_entry(2, false),
            storage_buffer_entry(3, true),
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tiny_metal_dense_prefill_attention_scores_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: q_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: k_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: shape_buffer.as_entire_binding() },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("tiny_metal_dense_prefill_attention_scores_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("tiny_metal_dense_prefill_attention_scores_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tiny_metal_dense_prefill_attention_scores_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tiny_metal_dense_prefill_attention_scores_pass"),
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
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|error| DensePrefillAttentionScoresMetalError::OutputReadback {
            message: error.to_string(),
        })?
        .map_err(|error| DensePrefillAttentionScoresMetalError::OutputMap {
            message: error.to_string(),
        })?;

    let data = slice.get_mapped_range();
    let scores = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    validate_attention_scores_output(fixture, adapter_info.name, scores)
}

#[cfg(any(test, all(feature = "metal-runtime", target_os = "macos", target_arch = "aarch64")))]
fn validate_attention_scores_output(
    fixture: &DenseMetalPrefillAttentionScoresFixture,
    adapter_name: String,
    scores: Vec<f32>,
) -> Result<DensePrefillAttentionScoresMetalOutput, DensePrefillAttentionScoresMetalError> {
    let expected = fixture.expected_scores.len();
    if scores.len() != expected {
        return Err(DensePrefillAttentionScoresMetalError::OutputShape {
            expected,
            actual: scores.len(),
        });
    }

    Ok(DensePrefillAttentionScoresMetalOutput { adapter_name, scores })
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
const DENSE_PREFILL_ATTENTION_SCORES_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> shape: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_index = global_id.x;
    let prefill_tokens = shape[0];
    let attention_heads = shape[1];
    let kv_heads = shape[2];
    let head_dim = shape[3];
    let score_count = attention_heads * prefill_tokens * prefill_tokens;

    if output_index >= score_count {
        return;
    }

    let tokens_square = prefill_tokens * prefill_tokens;
    let head = output_index / tokens_square;
    let token_pair = output_index % tokens_square;
    let query_token = token_pair / prefill_tokens;
    let key_token = token_pair % prefill_tokens;
    let heads_per_kv = attention_heads / kv_heads;
    let kv_head = head / heads_per_kv;

    var acc = 0.0;
    var dim = 0u;
    loop {
        if dim >= head_dim {
            break;
        }
        let q_index = ((query_token * attention_heads + head) * head_dim) + dim;
        let k_index = ((key_token * kv_heads + kv_head) * head_dim) + dim;
        acc = acc + q[q_index] * k[k_index];
        dim = dim + 1u;
    }

    output[output_index] = acc / sqrt(f32(head_dim));
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::smoke::dense_metal_prefill_attention_scores_fixture;

    #[test]
    fn unsupported_target_is_explicit_without_runtime_api() {
        if dense_prefill_attention_scores_runtime_api_available() {
            return;
        }

        let fixture = dense_metal_prefill_attention_scores_fixture();
        let error = run_dense_prefill_attention_scores_blocking(&fixture)
            .expect_err("non-runtime targets must fail clearly");
        assert!(matches!(error, DensePrefillAttentionScoresMetalError::UnsupportedTarget { .. }));
    }

    #[test]
    fn output_validation_preserves_attention_scores()
    -> Result<(), DensePrefillAttentionScoresMetalError> {
        let fixture = dense_metal_prefill_attention_scores_fixture();
        let output = validate_attention_scores_output(
            &fixture,
            "test-adapter".to_string(),
            fixture.expected_scores.clone(),
        )?;

        assert_eq!(output.adapter_name, "test-adapter");
        assert_eq!(output.scores, fixture.expected_scores);
        Ok(())
    }
}
