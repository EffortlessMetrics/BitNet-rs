//! Command buffer recording and submission for compute dispatches.

use crate::error::{Result, VulkanError};
use ash::vk;
use tracing::debug;

/// A command pool paired with its queue family index.
pub struct CommandPool {
    /// Raw Vulkan command pool.
    pub pool: vk::CommandPool,
    /// Queue family index this pool targets.
    pub queue_family: u32,
}

impl CommandPool {
    /// Create a new command pool for the given queue family.
    pub fn new(device: &ash::Device, queue_family: u32) -> Result<Self> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let pool = unsafe {
            device.create_command_pool(&create_info, None).map_err(|e| {
                VulkanError::CommandBuffer(format!("vkCreateCommandPool failed: {e}"))
            })?
        };

        debug!("Command pool created for queue family {queue_family}");
        Ok(Self { pool, queue_family })
    }

    /// Allocate a single primary command buffer from this pool.
    pub fn allocate_command_buffer(&self, device: &ash::Device) -> Result<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buffers = unsafe {
            device.allocate_command_buffers(&alloc_info).map_err(|e| {
                VulkanError::CommandBuffer(format!("vkAllocateCommandBuffers failed: {e}"))
            })?
        };

        Ok(buffers[0])
    }

    /// Destroy the command pool.
    ///
    /// # Safety
    /// Must only be called once; all command buffers allocated from this pool
    /// become invalid.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// Record and submit a compute dispatch.
///
/// Records `begin → bind pipeline → dispatch → end` and submits to the given
/// queue, waiting on a fence for completion.
pub fn record_and_submit_compute(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    pipeline: vk::Pipeline,
    _pipeline_layout: vk::PipelineLayout,
    group_count: [u32; 3],
    queue: vk::Queue,
) -> Result<()> {
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device
            .begin_command_buffer(cmd, &begin_info)
            .map_err(|e| VulkanError::CommandBuffer(format!("vkBeginCommandBuffer failed: {e}")))?;

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_dispatch(cmd, group_count[0], group_count[1], group_count[2]);

        device
            .end_command_buffer(cmd)
            .map_err(|e| VulkanError::CommandBuffer(format!("vkEndCommandBuffer failed: {e}")))?;
    }

    // Create a fence for synchronization.
    let fence_info = vk::FenceCreateInfo::default();
    let fence = unsafe {
        device
            .create_fence(&fence_info, None)
            .map_err(|e| VulkanError::QueueSubmit(format!("create fence: {e}")))?
    };

    let cmd_buffers = [cmd];
    let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);

    unsafe {
        device
            .queue_submit(queue, &[submit_info], fence)
            .map_err(|e| VulkanError::QueueSubmit(format!("vkQueueSubmit: {e}")))?;

        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| VulkanError::QueueSubmit(format!("wait fence: {e}")))?;

        device.destroy_fence(fence, None);
    }

    debug!(
        "Compute dispatch submitted: groups=[{},{},{}]",
        group_count[0], group_count[1], group_count[2]
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_pool_stores_queue_family() {
        let pool = CommandPool { pool: vk::CommandPool::null(), queue_family: 2 };
        assert_eq!(pool.queue_family, 2);
    }
}
