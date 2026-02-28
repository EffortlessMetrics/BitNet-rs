//! Device properties query for Level-Zero devices.

use crate::ffi::{ZeComputeProperties, ZeDeviceProperties, ZeDeviceType, ZeMemoryProperties};

/// Comprehensive device capability snapshot.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Core device properties.
    pub properties: ZeDeviceProperties,
    /// Compute properties (work-group sizes, subgroups).
    pub compute: ZeComputeProperties,
    /// Memory properties per memory domain.
    pub memory: Vec<ZeMemoryProperties>,
}

impl DeviceCapabilities {
    /// Total number of execution units.
    pub fn total_eus(&self) -> u32 {
        self.properties.num_slices
            * self.properties.num_subslices_per_slice
            * self.properties.num_eu_per_subslice
    }

    /// Total number of hardware threads.
    pub fn total_threads(&self) -> u32 {
        self.total_eus() * self.properties.num_threads_per_eu
    }

    /// Whether this is a GPU device.
    pub fn is_gpu(&self) -> bool {
        self.properties.device_type == ZeDeviceType::Gpu
    }

    /// The device name.
    pub fn name(&self) -> &str {
        &self.properties.name
    }
}

/// Builder for querying device properties.
#[derive(Debug)]
pub struct DeviceQuery {
    filter_type: Option<ZeDeviceType>,
    min_memory_bytes: Option<u64>,
    min_eus: Option<u32>,
}

impl DeviceQuery {
    /// Create a new device query builder.
    pub fn new() -> Self {
        Self {
            filter_type: None,
            min_memory_bytes: None,
            min_eus: None,
        }
    }

    /// Filter by device type.
    pub fn device_type(mut self, dt: ZeDeviceType) -> Self {
        self.filter_type = Some(dt);
        self
    }

    /// Require at least `bytes` of device memory.
    pub fn min_memory(mut self, bytes: u64) -> Self {
        self.min_memory_bytes = Some(bytes);
        self
    }

    /// Require at least `count` execution units.
    pub fn min_eus(mut self, count: u32) -> Self {
        self.min_eus = Some(count);
        self
    }

    /// Check whether a device capability set matches this query.
    pub fn matches(&self, caps: &DeviceCapabilities) -> bool {
        if let Some(dt) = self.filter_type {
            if caps.properties.device_type != dt {
                return false;
            }
        }
        if let Some(min_mem) = self.min_memory_bytes {
            let total: u64 = caps.memory.iter().map(|m| m.total_size).sum();
            if total < min_mem {
                return false;
            }
        }
        if let Some(min_eu) = self.min_eus {
            if caps.total_eus() < min_eu {
                return false;
            }
        }
        true
    }
}

impl Default for DeviceQuery {
    fn default() -> Self {
        Self::new()
    }
}
