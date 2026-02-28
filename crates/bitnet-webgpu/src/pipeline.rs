//! Compute pipeline construction from WGSL sources.

use crate::error::Result;
use tracing::debug;

/// A compiled compute pipeline ready for dispatch.
pub struct ComputePipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl ComputePipeline {
    /// Compile a WGSL shader and build the pipeline.
    pub fn new(
        device: &wgpu::Device,
        wgsl_source: &str,
        label: &str,
        entry_point: &str,
    ) -> Result<Self> {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        let _bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}-bgl")),
            entries: &[],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}-layout")),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        debug!(label, entry_point, "compiled compute pipeline");

        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    /// Create a bind group for this pipeline.
    pub fn bind_group(
        &self,
        device: &wgpu::Device,
        entries: &[wgpu::BindGroupEntry<'_>],
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute-bind-group"),
            layout: &self.bind_group_layout,
            entries,
        })
    }
}
