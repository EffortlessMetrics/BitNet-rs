#![cfg(feature = "metal")]

use bitnet_kernels::metal::smoke::{
    ARTIFACT_KIND, MACHINE_ID, PARITY_ARTIFACT_KIND, REFERENCE_BACKEND, REQUESTED_BACKEND,
    RUNTIME_API, SELECTED_BACKEND, SMOKE_WORKGROUP_SIZE, SmokeComparison,
    TINY_METAL_ADD_PARITY_KERNEL_ID, TINY_METAL_ADD_SMOKE_KERNEL_ID, TinyMetalAddParityReceipt,
    TinyMetalAddSmokeReceipt, compare_tiny_add_outputs, expected_tiny_add,
    is_apple_m4_adapter_name, metal_parity_artifact_path, metal_smoke_artifact_path,
    tiny_add_inputs,
};

#[test]
fn tiny_add_expected_output_matches_cpu_reference() {
    let (lhs, rhs) = tiny_add_inputs();
    let expected = expected_tiny_add(&lhs, &rhs).expect("valid smoke inputs");

    assert_eq!(expected.len(), lhs.len());
    for (index, value) in expected.iter().enumerate() {
        assert_eq!(*value, lhs[index] + rhs[index]);
    }
}

#[test]
fn receipt_contract_records_only_tiny_m4_metal_smoke() {
    let receipt = TinyMetalAddSmokeReceipt::passed(
        metal_smoke_artifact_path("2026-05-06"),
        64,
        SmokeComparison { max_abs_error: 0.0, mean_abs_error: 0.0 },
    );

    assert_eq!(receipt.machine_id, MACHINE_ID);
    assert_eq!(receipt.artifact_kind, ARTIFACT_KIND);
    assert_eq!(receipt.requested_backend, REQUESTED_BACKEND);
    assert_eq!(receipt.selected_backend, SELECTED_BACKEND);
    assert_eq!(receipt.runtime_api, RUNTIME_API);
    assert_eq!(receipt.kernel_id, TINY_METAL_ADD_SMOKE_KERNEL_ID);
    assert!(!receipt.fallback_used);
    assert_eq!(receipt.result, "pass");
    assert_eq!(receipt.artifact_path, "ci/hardware/apple-m4-mac-mini/2026-05-06/metal-smoke.json");
}

#[test]
fn parity_receipt_contract_records_cpu_neon_reference_and_metal_target() {
    let receipt = TinyMetalAddParityReceipt::passed(
        metal_parity_artifact_path("2026-05-06"),
        64,
        SmokeComparison { max_abs_error: 0.0, mean_abs_error: 0.0 },
    );

    assert_eq!(receipt.machine_id, MACHINE_ID);
    assert_eq!(receipt.artifact_kind, PARITY_ARTIFACT_KIND);
    assert_eq!(receipt.requested_backend, REQUESTED_BACKEND);
    assert_eq!(receipt.selected_backend, SELECTED_BACKEND);
    assert_eq!(receipt.runtime_api, RUNTIME_API);
    assert_eq!(receipt.reference_backend, REFERENCE_BACKEND);
    assert_eq!(receipt.target_backend, SELECTED_BACKEND);
    assert_eq!(receipt.kernel_id, TINY_METAL_ADD_PARITY_KERNEL_ID);
    assert!(!receipt.fallback_used);
    assert_eq!(receipt.result, "pass");
    assert_eq!(receipt.artifact_path, "ci/hardware/apple-m4-mac-mini/2026-05-06/metal-parity.json");
    assert_eq!(receipt.max_abs_error, 0.0);
    assert_eq!(receipt.mean_abs_error, 0.0);
}

#[test]
fn comparison_fails_instead_of_falling_back_to_cpu() {
    let expected = [1.0_f32, 2.0, 3.0];
    let actual = [1.0_f32, 20.0, 3.0];

    let error = compare_tiny_add_outputs(&expected, &actual, 1e-6)
        .expect_err("mismatch should fail the smoke contract");

    assert!(error.to_string().contains("output mismatch"), "unexpected error: {error}");
}

#[test]
fn apple_m4_adapter_name_detection_is_specific() {
    assert!(is_apple_m4_adapter_name("Apple M4"));
    assert!(is_apple_m4_adapter_name("Apple M4 Pro"));
    assert!(!is_apple_m4_adapter_name("Apple M3"));
    assert!(!is_apple_m4_adapter_name("AMD Radeon Pro"));
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod live_metal {
    use super::*;
    use serde_json::json;
    use std::error::Error;
    use std::io;
    use std::path::Path;
    use wgpu::util::DeviceExt;

    const RUN_ENV: &str = "BITNET_RUN_M4_METAL_SMOKE";
    const RECEIPT_ENV: &str = "BITNET_M4_METAL_SMOKE_RECEIPT";
    const ARTIFACT_PATH_ENV: &str = "BITNET_M4_METAL_SMOKE_ARTIFACT_PATH";
    const RUN_PARITY_ENV: &str = "BITNET_RUN_M4_METAL_PARITY";
    const PARITY_RECEIPT_ENV: &str = "BITNET_M4_METAL_PARITY_RECEIPT";
    const PARITY_ARTIFACT_PATH_ENV: &str = "BITNET_M4_METAL_PARITY_ARTIFACT_PATH";

    struct MetalSmokeOutput {
        adapter_name: String,
        output: Vec<f32>,
    }

    #[test]
    fn tiny_m4_metal_add_smoke_runs_when_enabled() -> Result<(), Box<dyn Error>> {
        if std::env::var(RUN_ENV).as_deref() != Ok("1") {
            eprintln!("skipping live M4 Metal smoke; set {RUN_ENV}=1 to run it");
            return Ok(());
        }

        let (lhs, rhs) = tiny_add_inputs();
        let expected = expected_tiny_add(&lhs, &rhs)?;
        let smoke_output = run_tiny_add_smoke(&lhs, &rhs)?;

        if !is_apple_m4_adapter_name(&smoke_output.adapter_name) {
            return Err(io_error(format!(
                "M4-005 proof requires an Apple M4-family Metal adapter; found '{}'",
                smoke_output.adapter_name
            )));
        }

        let comparison = compare_tiny_add_outputs(&expected, &smoke_output.output, 1e-6)?;
        let artifact_path = std::env::var(ARTIFACT_PATH_ENV)
            .or_else(|_| std::env::var(RECEIPT_ENV))
            .unwrap_or_else(|_| {
                "ci/hardware/apple-m4-mac-mini/<date>/metal-smoke.json".to_string()
            });
        let receipt =
            TinyMetalAddSmokeReceipt::passed(artifact_path.clone(), expected.len(), comparison);

        let receipt_json = json!({
            "machine_id": receipt.machine_id,
            "artifact_kind": receipt.artifact_kind,
            "requested_backend": receipt.requested_backend,
            "selected_backend": receipt.selected_backend,
            "runtime_api": receipt.runtime_api,
            "resolved_device": {
                "chip": smoke_output.adapter_name,
                "unified_memory": true
            },
            "kernel_id": receipt.kernel_id,
            "fallback_used": receipt.fallback_used,
            "result": receipt.result,
            "artifact_path": receipt.artifact_path,
            "element_count": receipt.element_count,
            "max_abs_error": receipt.max_abs_error,
            "mean_abs_error": receipt.mean_abs_error
        });

        if let Ok(path) = std::env::var(RECEIPT_ENV) {
            if let Some(parent) = Path::new(&path).parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, serde_json::to_string_pretty(&receipt_json)?)?;
        }

        println!("{}", serde_json::to_string_pretty(&receipt_json)?);
        Ok(())
    }

    #[test]
    fn tiny_m4_metal_add_matches_cpu_neon_reference_when_enabled() -> Result<(), Box<dyn Error>> {
        if std::env::var(RUN_PARITY_ENV).as_deref() != Ok("1") {
            eprintln!("skipping live M4 CPU/Metal parity; set {RUN_PARITY_ENV}=1 to run it");
            return Ok(());
        }

        let (lhs, rhs) = tiny_add_inputs();
        let expected = expected_tiny_add(&lhs, &rhs)?;
        let metal_output = run_tiny_add_smoke(&lhs, &rhs)?;

        if !is_apple_m4_adapter_name(&metal_output.adapter_name) {
            return Err(io_error(format!(
                "M4-006 parity requires an Apple M4-family Metal adapter; found '{}'",
                metal_output.adapter_name
            )));
        }

        let comparison = compare_tiny_add_outputs(&expected, &metal_output.output, 1e-6)?;
        let artifact_path = std::env::var(PARITY_ARTIFACT_PATH_ENV)
            .or_else(|_| std::env::var(PARITY_RECEIPT_ENV))
            .unwrap_or_else(|_| {
                "ci/hardware/apple-m4-mac-mini/<date>/metal-parity.json".to_string()
            });
        let receipt =
            TinyMetalAddParityReceipt::passed(artifact_path.clone(), expected.len(), comparison);

        let receipt_json = json!({
            "machine_id": receipt.machine_id,
            "artifact_kind": receipt.artifact_kind,
            "requested_backend": receipt.requested_backend,
            "selected_backend": receipt.selected_backend,
            "runtime_api": receipt.runtime_api,
            "resolved_device": {
                "chip": metal_output.adapter_name,
                "unified_memory": true
            },
            "parity": {
                "reference_backend": receipt.reference_backend,
                "target_backend": receipt.target_backend,
                "kernel_id": receipt.kernel_id,
                "max_abs_error": receipt.max_abs_error,
                "mean_abs_error": receipt.mean_abs_error,
                "token_agreement_for_greedy": null
            },
            "fallback_used": receipt.fallback_used,
            "result": receipt.result,
            "artifact_path": receipt.artifact_path,
            "element_count": receipt.element_count
        });

        if let Ok(path) = std::env::var(PARITY_RECEIPT_ENV) {
            if let Some(parent) = Path::new(&path).parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, serde_json::to_string_pretty(&receipt_json)?)?;
        }

        println!("{}", serde_json::to_string_pretty(&receipt_json)?);
        Ok(())
    }

    fn run_tiny_add_smoke(lhs: &[f32], rhs: &[f32]) -> Result<MetalSmokeOutput, Box<dyn Error>> {
        pollster::block_on(async move {
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
                .ok_or_else(|| io_error("no Metal adapter found for M4-005 smoke"))?;

            let adapter_info = adapter.get_info();
            if adapter_info.backend != wgpu::Backend::Metal {
                return Err(io_error(format!(
                    "M4-005 smoke requires Metal backend, found {:?}",
                    adapter_info.backend
                )));
            }

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .map_err(|error| io_error(format!("failed to create Metal device: {error}")))?;

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(TINY_METAL_ADD_SMOKE_KERNEL_ID),
                source: wgpu::ShaderSource::Wgsl(TINY_ADD_SHADER.into()),
            });

            let lhs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tiny_metal_add_lhs"),
                contents: bytemuck::cast_slice(lhs),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let rhs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tiny_metal_add_rhs"),
                contents: bytemuck::cast_slice(rhs),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let byte_len = std::mem::size_of_val(lhs) as u64;
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tiny_metal_add_output"),
                size: byte_len,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tiny_metal_add_staging"),
                size: byte_len,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("tiny_metal_add_layout"),
                    entries: &[
                        storage_buffer_entry(0, true),
                        storage_buffer_entry(1, true),
                        storage_buffer_entry(2, false),
                    ],
                });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tiny_metal_add_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: lhs_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: rhs_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tiny_metal_add_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tiny_metal_add_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tiny_metal_add_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("tiny_metal_add_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((lhs.len() as u32).div_ceil(SMOKE_WORKGROUP_SIZE), 1, 1);
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
                .map_err(|error| io_error(format!("failed to receive Metal map result: {error}")))?
                .map_err(|error| io_error(format!("failed to map Metal smoke output: {error}")))?;

            let data = slice.get_mapped_range();
            let output = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
            drop(data);
            staging_buffer.unmap();

            Ok(MetalSmokeOutput { adapter_name: adapter_info.name, output })
        })
    }

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

    fn io_error(message: impl Into<String>) -> Box<dyn Error> {
        Box::new(io::Error::other(message.into()))
    }

    const TINY_ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index < arrayLength(&lhs) {
        output[index] = lhs[index] + rhs[index];
    }
}
"#;
}
