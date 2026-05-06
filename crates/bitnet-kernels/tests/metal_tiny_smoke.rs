#![cfg(feature = "metal")]

use bitnet_device_probe::{AppleBackendReceipt, AppleResolvedDevice};
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
    use std::time::{Duration, Instant};
    use wgpu::util::DeviceExt;

    const RUN_ENV: &str = "BITNET_RUN_M4_METAL_SMOKE";
    const RECEIPT_ENV: &str = "BITNET_M4_METAL_SMOKE_RECEIPT";
    const ARTIFACT_PATH_ENV: &str = "BITNET_M4_METAL_SMOKE_ARTIFACT_PATH";
    const RUN_PARITY_ENV: &str = "BITNET_RUN_M4_METAL_PARITY";
    const PARITY_RECEIPT_ENV: &str = "BITNET_M4_METAL_PARITY_RECEIPT";
    const PARITY_ARTIFACT_PATH_ENV: &str = "BITNET_M4_METAL_PARITY_ARTIFACT_PATH";
    const RUN_BENCHMARK_ENV: &str = "BITNET_RUN_M4_METAL_BENCHMARK";
    const BENCHMARK_RECEIPT_ENV: &str = "BITNET_M4_METAL_BENCHMARK_RECEIPT";
    const BENCHMARK_ARTIFACT_PATH_ENV: &str = "BITNET_M4_METAL_BENCHMARK_ARTIFACT_PATH";
    const BENCHMARK_ITERATIONS_ENV: &str = "BITNET_M4_METAL_BENCHMARK_ITERATIONS";
    const TINY_KERNEL_SMOKE_PROFILE: &str = "tiny_kernel_smoke";

    struct MetalSmokeOutput {
        adapter_name: String,
        output: Vec<f32>,
    }

    struct MetalBenchmarkOutput {
        adapter_name: String,
        output: Vec<f32>,
        timing: BenchmarkTiming,
    }

    struct BenchmarkTiming {
        compile: Duration,
        first_dispatch: Duration,
        steady_state: Duration,
        cpu_reference: Duration,
        iterations: u32,
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

        let mut receipt_json = apple_backend_receipt_json(
            receipt.machine_id,
            receipt.artifact_kind,
            receipt.requested_backend,
            Some(receipt.selected_backend),
            receipt.runtime_api,
            smoke_output.adapter_name,
            receipt.fallback_used,
            receipt.artifact_path.clone(),
            Some(receipt.kernel_id),
            None,
            receipt.result,
        )?;
        extend_smoke_metrics(
            &mut receipt_json,
            receipt.element_count,
            receipt.max_abs_error,
            receipt.mean_abs_error,
        );

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

        let mut receipt_json = apple_backend_receipt_json(
            receipt.machine_id,
            receipt.artifact_kind,
            receipt.requested_backend,
            Some(receipt.selected_backend),
            receipt.runtime_api,
            metal_output.adapter_name,
            receipt.fallback_used,
            receipt.artifact_path.clone(),
            Some(receipt.kernel_id),
            None,
            receipt.result,
        )?;
        extend_parity_metrics(
            &mut receipt_json,
            receipt.element_count,
            receipt.reference_backend,
            receipt.target_backend,
            receipt.kernel_id,
            receipt.max_abs_error,
            receipt.mean_abs_error,
        );

        if let Ok(path) = std::env::var(PARITY_RECEIPT_ENV) {
            if let Some(parent) = Path::new(&path).parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, serde_json::to_string_pretty(&receipt_json)?)?;
        }

        println!("{}", serde_json::to_string_pretty(&receipt_json)?);
        Ok(())
    }

    #[test]
    fn tiny_m4_metal_add_benchmark_records_cpu_reference_when_enabled() -> Result<(), Box<dyn Error>>
    {
        if std::env::var(RUN_BENCHMARK_ENV).as_deref() != Ok("1") {
            eprintln!("skipping live M4 Metal benchmark; set {RUN_BENCHMARK_ENV}=1 to run it");
            return Ok(());
        }

        let iterations = benchmark_iterations()?;
        let (lhs, rhs) = tiny_add_inputs();

        let cpu_reference_start = Instant::now();
        let expected = expected_tiny_add(&lhs, &rhs)?;
        let cpu_reference = cpu_reference_start.elapsed();

        let benchmark_output = run_tiny_add_benchmark(&lhs, &rhs, iterations, cpu_reference)?;
        if !is_apple_m4_adapter_name(&benchmark_output.adapter_name) {
            return Err(io_error(format!(
                "M4-009 benchmark requires an Apple M4-family Metal adapter; found '{}'",
                benchmark_output.adapter_name
            )));
        }

        let comparison = compare_tiny_add_outputs(&expected, &benchmark_output.output, 1e-6)?;
        let artifact_path = std::env::var(BENCHMARK_ARTIFACT_PATH_ENV)
            .or_else(|_| std::env::var(BENCHMARK_RECEIPT_ENV))
            .unwrap_or_else(|_| {
                "ci/hardware/apple-m4-mac-mini/<date>/metal-benchmark.json".to_string()
            });

        let mut receipt_json = apple_backend_receipt_json(
            MACHINE_ID,
            "benchmark",
            REQUESTED_BACKEND,
            Some(SELECTED_BACKEND),
            RUNTIME_API,
            benchmark_output.adapter_name,
            false,
            artifact_path.clone(),
            Some(TINY_METAL_ADD_SMOKE_KERNEL_ID),
            None,
            "pass",
        )?;
        extend_benchmark_metrics(
            &mut receipt_json,
            expected.len(),
            &benchmark_output.timing,
            comparison.max_abs_error,
            comparison.mean_abs_error,
        );

        if let Ok(path) = std::env::var(BENCHMARK_RECEIPT_ENV) {
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

    fn run_tiny_add_benchmark(
        lhs: &[f32],
        rhs: &[f32],
        iterations: u32,
        cpu_reference: Duration,
    ) -> Result<MetalBenchmarkOutput, Box<dyn Error>> {
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
                .ok_or_else(|| io_error("no Metal adapter found for M4-009 benchmark"))?;

            let adapter_info = adapter.get_info();
            if adapter_info.backend != wgpu::Backend::Metal {
                return Err(io_error(format!(
                    "M4-009 benchmark requires Metal backend, found {:?}",
                    adapter_info.backend
                )));
            }

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .map_err(|error| io_error(format!("failed to create Metal device: {error}")))?;

            let compile_start = Instant::now();
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(TINY_METAL_ADD_SMOKE_KERNEL_ID),
                source: wgpu::ShaderSource::Wgsl(TINY_ADD_SHADER.into()),
            });

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("tiny_metal_add_benchmark_layout"),
                    entries: &[
                        storage_buffer_entry(0, true),
                        storage_buffer_entry(1, true),
                        storage_buffer_entry(2, false),
                    ],
                });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tiny_metal_add_benchmark_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tiny_metal_add_benchmark_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let compile = compile_start.elapsed();

            let lhs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tiny_metal_add_benchmark_lhs"),
                contents: bytemuck::cast_slice(lhs),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let rhs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tiny_metal_add_benchmark_rhs"),
                contents: bytemuck::cast_slice(rhs),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let byte_len = std::mem::size_of_val(lhs) as u64;
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tiny_metal_add_benchmark_output"),
                size: byte_len,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tiny_metal_add_benchmark_staging"),
                size: byte_len,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tiny_metal_add_benchmark_bind_group"),
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

            let first_dispatch_start = Instant::now();
            let mut output = dispatch_tiny_add(
                &device,
                &queue,
                &pipeline,
                &bind_group,
                &output_buffer,
                &staging_buffer,
                byte_len,
                lhs.len(),
            )?;
            let first_dispatch = first_dispatch_start.elapsed();

            let steady_start = Instant::now();
            for _ in 0..iterations {
                output = dispatch_tiny_add(
                    &device,
                    &queue,
                    &pipeline,
                    &bind_group,
                    &output_buffer,
                    &staging_buffer,
                    byte_len,
                    lhs.len(),
                )?;
            }
            let steady_state = steady_start.elapsed() / iterations;

            Ok(MetalBenchmarkOutput {
                adapter_name: adapter_info.name,
                output,
                timing: BenchmarkTiming {
                    compile,
                    first_dispatch,
                    steady_state,
                    cpu_reference,
                    iterations,
                },
            })
        })
    }

    fn dispatch_tiny_add(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        output_buffer: &wgpu::Buffer,
        staging_buffer: &wgpu::Buffer,
        byte_len: u64,
        element_count: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tiny_metal_add_benchmark_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tiny_metal_add_benchmark_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups((element_count as u32).div_ceil(SMOKE_WORKGROUP_SIZE), 1, 1);
        }

        encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, byte_len);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|error| io_error(format!("failed to receive Metal map result: {error}")))?
            .map_err(|error| io_error(format!("failed to map Metal benchmark output: {error}")))?;

        let data = slice.get_mapped_range();
        let output = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(output)
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

    #[allow(clippy::too_many_arguments)]
    fn apple_backend_receipt_json(
        machine_id: &str,
        artifact_kind: &str,
        requested_backend: &str,
        selected_backend: Option<&str>,
        runtime_api: &str,
        chip: String,
        fallback_used: bool,
        artifact_path: String,
        kernel_id: Option<&str>,
        graph_id: Option<&str>,
        result: &str,
    ) -> Result<serde_json::Value, Box<dyn Error>> {
        let mut receipt = AppleBackendReceipt::new(
            machine_id,
            artifact_kind,
            requested_backend,
            selected_backend,
            runtime_api,
            AppleResolvedDevice::new(chip).with_unified_memory(true),
            fallback_used,
            artifact_path,
        )
        .with_result(result);

        if let Some(kernel_id) = kernel_id {
            receipt = receipt.with_kernel_id(kernel_id);
        }
        if let Some(graph_id) = graph_id {
            receipt = receipt.with_graph_id(graph_id);
        }

        receipt.validate()?;
        Ok(serde_json::to_value(receipt)?)
    }

    fn extend_smoke_metrics(
        receipt_json: &mut serde_json::Value,
        element_count: usize,
        max_abs_error: f32,
        mean_abs_error: f32,
    ) {
        let object = receipt_json.as_object_mut().expect("Apple receipt JSON is an object");
        object.insert("element_count".to_string(), json!(element_count));
        object.insert("max_abs_error".to_string(), json!(max_abs_error));
        object.insert("mean_abs_error".to_string(), json!(mean_abs_error));
    }

    fn extend_parity_metrics(
        receipt_json: &mut serde_json::Value,
        element_count: usize,
        reference_backend: &str,
        target_backend: &str,
        kernel_id: &str,
        max_abs_error: f32,
        mean_abs_error: f32,
    ) {
        let object = receipt_json.as_object_mut().expect("Apple receipt JSON is an object");
        object.insert("element_count".to_string(), json!(element_count));
        object.insert(
            "parity".to_string(),
            json!({
                "reference_backend": reference_backend,
                "target_backend": target_backend,
                "kernel_id": kernel_id,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "token_agreement_for_greedy": null
            }),
        );
    }

    fn extend_benchmark_metrics(
        receipt_json: &mut serde_json::Value,
        element_count: usize,
        timing: &BenchmarkTiming,
        max_abs_error: f32,
        mean_abs_error: f32,
    ) {
        let object = receipt_json.as_object_mut().expect("Apple receipt JSON is an object");
        object.insert("element_count".to_string(), json!(element_count));
        object.insert(
            "benchmark".to_string(),
            json!({
                "profile": TINY_KERNEL_SMOKE_PROFILE,
                "reference_backend": REFERENCE_BACKEND,
                "target_backend": SELECTED_BACKEND,
                "kernel_id": TINY_METAL_ADD_SMOKE_KERNEL_ID,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error
            }),
        );
        object.insert(
            "timing".to_string(),
            json!({
                "compile_ms": duration_ms(timing.compile),
                "first_dispatch_ms": duration_ms(timing.first_dispatch),
                "steady_state_ms": duration_ms(timing.steady_state),
                "cpu_reference_ms": duration_ms(timing.cpu_reference),
                "iterations": timing.iterations
            }),
        );
        object.insert(
            "machine".to_string(),
            json!({
                "chip": object
                    .get("resolved_device")
                    .and_then(|value| value.get("chip"))
                    .cloned()
                    .unwrap_or_else(|| json!("unknown")),
                "memory_gb": null,
                "power_mode": "unknown",
                "thermal_state": "unknown"
            }),
        );
    }

    fn duration_ms(duration: Duration) -> f64 {
        duration.as_secs_f64() * 1_000.0
    }

    fn benchmark_iterations() -> Result<u32, Box<dyn Error>> {
        match std::env::var(BENCHMARK_ITERATIONS_ENV) {
            Ok(value) => {
                let iterations = value.parse::<u32>().map_err(|error| {
                    io_error(format!(
                        "{BENCHMARK_ITERATIONS_ENV} must be a positive integer: {error}"
                    ))
                })?;
                if iterations == 0 {
                    return Err(io_error(format!("{BENCHMARK_ITERATIONS_ENV} must be positive")));
                }
                Ok(iterations)
            }
            Err(_) => Ok(10),
        }
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
