//! GPU buffer management with staging and device-local memory.

use crate::error::{Result, VulkanError};
use ash::vk;
use tracing::debug;

/// Usage hint for buffer allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage {
    /// Host-visible staging buffer for CPU→GPU transfers.
    Staging,
    /// Device-local buffer for GPU compute (not host-visible).
    DeviceLocal,
    /// Device-local storage buffer for shader read/write.
    Storage,
}

/// Describes a buffer allocation request.
#[derive(Debug, Clone)]
pub struct BufferDescriptor {
    /// Size in bytes.
    pub size: vk::DeviceSize,
    /// Intended usage pattern.
    pub usage: BufferUsage,
    /// Human-readable label for debugging.
    pub label: String,
}

/// A GPU buffer with its backing memory.
///
/// The caller must call [`GpuBuffer::destroy`] before dropping.
pub struct GpuBuffer {
    /// Raw Vulkan buffer handle.
    pub buffer: vk::Buffer,
    /// Backing device memory.
    pub memory: vk::DeviceMemory,
    /// Allocated size in bytes.
    pub size: vk::DeviceSize,
    /// Usage that was requested.
    pub usage: BufferUsage,
}

impl GpuBuffer {
    /// Destroy the buffer and free its memory.
    ///
    /// # Safety
    /// Must only be called once. The device must be the same one used to
    /// create the buffer.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Find a memory type index matching the requested properties.
pub fn find_memory_type(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    required_flags: vk::MemoryPropertyFlags,
) -> Result<u32> {
    for i in 0..memory_properties.memory_type_count {
        let type_ok = (type_filter & (1 << i)) != 0;
        let flags_ok = memory_properties.memory_types[i as usize].property_flags
            & required_flags
            == required_flags;
        if type_ok && flags_ok {
            return Ok(i);
        }
    }
    Err(VulkanError::NoSuitableMemoryType(format!(
        "no memory type with filter={type_filter:#x} flags={required_flags:?}"
    )))
}

/// Allocate a GPU buffer with the given descriptor.
pub fn allocate_buffer(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    desc: &BufferDescriptor,
) -> Result<GpuBuffer> {
    let (vk_usage, mem_flags) = match desc.usage {
        BufferUsage::Staging => (
            vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        ),
        BufferUsage::DeviceLocal => (
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ),
        BufferUsage::Storage => (
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ),
    };

    let buffer_info = vk::BufferCreateInfo::default()
        .size(desc.size)
        .usage(vk_usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        device.create_buffer(&buffer_info, None).map_err(|e| {
            VulkanError::BufferAllocation(format!("vkCreateBuffer failed: {e}"))
        })?
    };

    let mem_requirements =
        unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type_index = find_memory_type(
        memory_properties,
        mem_requirements.memory_type_bits,
        mem_flags,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);

    let memory = unsafe {
        device.allocate_memory(&alloc_info, None).map_err(|e| {
            VulkanError::BufferAllocation(format!(
                "vkAllocateMemory failed: {e}"
            ))
        })?
    };

    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| {
                VulkanError::BufferAllocation(format!(
                    "vkBindBufferMemory failed: {e}"
                ))
            })?;
    }

    debug!(
        "Allocated {:?} buffer '{}' ({} bytes)",
        desc.usage, desc.label, desc.size
    );

    Ok(GpuBuffer {
        buffer,
        memory,
        size: desc.size,
        usage: desc.usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_descriptor_construction() {
        let desc = BufferDescriptor {
            size: 4096,
            usage: BufferUsage::Staging,
            label: "weights".into(),
        };
        assert_eq!(desc.size, 4096);
        assert_eq!(desc.usage, BufferUsage::Staging);
    }

    #[test]
    fn find_memory_type_no_match() {
        let props = vk::PhysicalDeviceMemoryProperties::default();
        let result = find_memory_type(
            &props,
            0xFFFF_FFFF,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        assert!(result.is_err());
    }
}
