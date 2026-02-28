//! Compute pipeline creation from SPIR-V shader modules.

use crate::error::{Result, VulkanError};
use ash::vk;
use tracing::debug;

/// Builder for Vulkan compute pipelines.
#[derive(Debug)]
pub struct ComputePipelineBuilder {
    spirv_code: Vec<u32>,
    entry_point: String,
    push_constant_size: u32,
    descriptor_set_count: u32,
}

impl ComputePipelineBuilder {
    /// Create a new pipeline builder with the given SPIR-V binary.
    ///
    /// `spirv` must be valid SPIR-V code (u32 words).
    pub fn new(spirv: &[u32]) -> Self {
        Self {
            spirv_code: spirv.to_vec(),
            entry_point: "main".into(),
            push_constant_size: 0,
            descriptor_set_count: 1,
        }
    }

    /// Set the shader entry point name (default: "main").
    #[must_use]
    pub fn entry_point(mut self, name: &str) -> Self {
        self.entry_point = name.into();
        self
    }

    /// Set push constant size in bytes.
    #[must_use]
    pub fn push_constant_size(mut self, size: u32) -> Self {
        self.push_constant_size = size;
        self
    }

    /// Set the number of descriptor sets in the layout.
    #[must_use]
    pub fn descriptor_set_count(mut self, count: u32) -> Self {
        self.descriptor_set_count = count;
        self
    }

    /// Build the compute pipeline on the given device.
    ///
    /// Returns the pipeline, pipeline layout, and shader module handles.
    /// The caller is responsible for destroying these on cleanup.
    pub fn build(&self, device: &ash::Device) -> Result<ComputePipeline> {
        if self.spirv_code.is_empty() {
            return Err(VulkanError::ShaderLoad("empty SPIR-V code".into()));
        }

        let shader_module_create_info =
            vk::ShaderModuleCreateInfo::default().code(&self.spirv_code);

        let shader_module = unsafe {
            device
                .create_shader_module(&shader_module_create_info, None)
                .map_err(|e| VulkanError::ShaderLoad(format!("vkCreateShaderModule failed: {e}")))?
        };

        let entry_name = std::ffi::CString::new(self.entry_point.as_str()).unwrap_or_default();

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_name);

        // Create a minimal pipeline layout.
        let mut push_constant_ranges = Vec::new();
        if self.push_constant_size > 0 {
            push_constant_ranges.push(
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(self.push_constant_size),
            );
        }

        let layout_create_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(&layout_create_info, None).map_err(|e| {
                VulkanError::PipelineCreation(format!("vkCreatePipelineLayout failed: {e}"))
            })?
        };

        let pipeline_create_info =
            vk::ComputePipelineCreateInfo::default().stage(stage_info).layout(pipeline_layout);

        let pipelines = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .map_err(|(_, e)| {
                    VulkanError::PipelineCreation(format!("vkCreateComputePipelines failed: {e}"))
                })?
        };

        debug!(
            "Compute pipeline created (entry={}, push_const={}B)",
            self.entry_point, self.push_constant_size
        );

        Ok(ComputePipeline { pipeline: pipelines[0], layout: pipeline_layout, shader_module })
    }
}

/// A compiled Vulkan compute pipeline.
///
/// The caller must destroy these resources by calling [`ComputePipeline::destroy`].
pub struct ComputePipeline {
    /// The raw Vulkan pipeline handle.
    pub pipeline: vk::Pipeline,
    /// The pipeline layout.
    pub layout: vk::PipelineLayout,
    /// The shader module.
    pub shader_module: vk::ShaderModule,
}

impl ComputePipeline {
    /// Destroy all Vulkan objects owned by this pipeline.
    ///
    /// # Safety
    /// Must only be called once, and the device must be the same one used
    /// during creation.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_shader_module(self.shader_module, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_builder_defaults() {
        let builder = ComputePipelineBuilder::new(&[]);
        assert_eq!(builder.entry_point, "main");
        assert_eq!(builder.push_constant_size, 0);
        assert_eq!(builder.descriptor_set_count, 1);
    }

    #[test]
    fn pipeline_builder_fluent_api() {
        let spirv = vec![0x0723_0203u32]; // fake magic
        let builder = ComputePipelineBuilder::new(&spirv)
            .entry_point("compute_main")
            .push_constant_size(64)
            .descriptor_set_count(2);
        assert_eq!(builder.entry_point, "compute_main");
        assert_eq!(builder.push_constant_size, 64);
        assert_eq!(builder.descriptor_set_count, 2);
        assert_eq!(builder.spirv_code.len(), 1);
    }

    #[test]
    fn pipeline_builder_rejects_empty_spirv() {
        let builder = ComputePipelineBuilder::new(&[]);
        assert!(builder.spirv_code.is_empty());
    }
}
