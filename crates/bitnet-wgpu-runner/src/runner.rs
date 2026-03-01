use crate::error::{RunnerError, div_ceil};

/// A compiled compute kernel ready for dispatch.
pub struct CompiledKernel {
    /// The compute pipeline created from the shader.
    pub pipeline: wgpu::ComputePipeline,
    /// The bind group layout for binding buffers.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// The shader entry point name.
    pub entry_point: String,
}

/// Orchestrates wgpu shader compilation, buffer setup, and dispatch execution.
pub struct KernelRunner {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl KernelRunner {
    /// Create a new `KernelRunner` with Vulkan backend preference.
    pub async fn new() -> Result<Self, RunnerError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(RunnerError::AdapterNotFound)?;

        tracing::info!(
            adapter_name = %adapter.get_info().name,
            backend = ?adapter.get_info().backend,
            "wgpu adapter selected"
        );

        let (device, queue) =
            adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await?;

        Ok(Self { device, queue })
    }

    /// Compile a WGSL shader source into a ready-to-dispatch kernel.
    pub fn compile_shader(
        &self,
        wgsl_source: &str,
        entry_point: &str,
    ) -> Result<CompiledKernel, RunnerError> {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bitnet_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bitnet_bind_group_layout"),
                entries: &[
                    // binding 0: input A (read-only)
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
                    // binding 1: input B (read-only)
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
                    // binding 2: output C (read-write)
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
                    // binding 3: uniforms (dimensions)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bitnet_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bitnet_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(CompiledKernel { pipeline, bind_group_layout, entry_point: entry_point.to_string() })
    }

    /// Create a GPU buffer initialized with the given `f32` data.
    pub fn create_buffer_f32(&self, data: &[f32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bitnet_input_buffer"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    }

    /// Create a GPU buffer for uniform data.
    pub fn create_uniform_buffer(&self, data: &[u8]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bitnet_uniform_buffer"),
            contents: data,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
        })
    }

    /// Create a zero-initialized output buffer of the given byte size.
    pub fn create_output_buffer(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bitnet_output_buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Encode and submit a compute dispatch.
    pub fn dispatch(
        &self,
        kernel: &CompiledKernel,
        bind_group: &wgpu::BindGroup,
        workgroups: [u32; 3],
    ) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bitnet_dispatch_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bitnet_compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Map a GPU buffer back to CPU and read its contents as `f32` values.
    pub async fn read_buffer_f32(
        &self,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f32>, RunnerError> {
        let size = (count * std::mem::size_of::<f32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bitnet_staging_buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bitnet_readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap()?;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Access the underlying wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Access the underlying wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

/// Compute workgroup dimensions for a 2D grid (rows × cols).
pub fn matmul_workgroups(m: u32, n: u32, group_size: u32) -> [u32; 3] {
    [div_ceil(n, group_size), div_ceil(m, group_size), 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_workgroups_exact() {
        assert_eq!(matmul_workgroups(8, 8, 8), [1, 1, 1]);
    }

    #[test]
    fn matmul_workgroups_remainder() {
        assert_eq!(matmul_workgroups(9, 17, 8), [3, 2, 1]);
    }

    #[test]
    fn matmul_workgroups_single() {
        assert_eq!(matmul_workgroups(1, 1, 8), [1, 1, 1]);
    }

    #[test]
    fn matmul_workgroups_large() {
        let wg = matmul_workgroups(1024, 2048, 8);
        assert_eq!(wg, [256, 128, 1]);
    }
}
