//! Multi-device work scheduler for heterogeneous GPU inference.
//!
//! Schedules layers and compute across multiple backends (CUDA, OpenCL, CPU)
//! using configurable strategies: single-device, layer-parallel, or pipeline.

use bitnet_common::Device;

/// Schedules work across multiple GPU backends (CUDA + OpenCL + CPU).
#[derive(Debug, Clone)]
pub struct MultiDeviceScheduler {
    devices: Vec<DeviceSlot>,
    strategy: SchedulingStrategy,
}

/// Strategy for distributing work across devices.
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulingStrategy {
    /// All layers on single best device.
    SingleDevice,
    /// Split layers across devices (e.g., layers 0-15 on CUDA, 16-31 on OpenCL).
    LayerParallel { split_points: Vec<usize> },
    /// Pipeline parallel (overlapping compute).
    Pipeline,
}

/// A slot representing one available compute device.
#[derive(Debug, Clone)]
pub struct DeviceSlot {
    /// The underlying device.
    pub device: Device,
    /// Available memory in bytes.
    pub memory_available: usize,
    /// Relative compute capability score (higher is faster).
    pub compute_capability: f64,
}

impl MultiDeviceScheduler {
    /// Create a new scheduler with explicit devices and strategy.
    pub fn new(devices: Vec<DeviceSlot>, strategy: SchedulingStrategy) -> Self {
        Self { devices, strategy }
    }

    /// Auto-configure scheduling based on available devices.
    ///
    /// - Single device → `SingleDevice` strategy
    /// - Multiple devices → `LayerParallel` with round-robin
    /// - Multiple devices → `LayerParallel` with an even split
    pub fn auto_configure(available_devices: &[Device]) -> Self {
        let slots: Vec<DeviceSlot> = available_devices
            .iter()
            .map(|d| DeviceSlot {
                device: *d,
                memory_available: default_memory_for_device(d),
                compute_capability: default_capability_for_device(d),
            })
            .collect();

        let strategy = if slots.len() <= 1 {
            SchedulingStrategy::SingleDevice
        } else {
            SchedulingStrategy::LayerParallel { split_points: vec![] }
        };

        Self { devices: slots, strategy }
    }

    /// Return which device should execute the given layer.
    pub fn assign_layer(&self, layer_idx: usize) -> Device {
        match &self.strategy {
            SchedulingStrategy::SingleDevice | SchedulingStrategy::Pipeline => self.best_device(),
            SchedulingStrategy::LayerParallel { split_points } => {
                if split_points.is_empty() {
                    let dev_count = self.devices.len().max(1);
                    self.devices[layer_idx % dev_count].device
                } else {
                    // Even round-robin across all devices
                    let dev_count = self.devices.len().max(1);
                    // Find which segment the layer belongs to
                    let segment = split_points
                        .iter()
                        .position(|&sp| layer_idx < sp)
                        .unwrap_or(split_points.len());
                    let idx = segment.min(self.devices.len() - 1);
                    self.devices[idx].device
                }
            }
        }
    }

    /// Plan memory allocation across devices for a model of given total size.
    ///
    /// Returns `(device, bytes)` pairs proportional to each device's available
    /// memory.
    pub fn memory_plan(&self, model_size: usize) -> Vec<(Device, usize)> {
        if self.devices.is_empty() {
            return vec![(Device::Cpu, model_size)];
        }

        match &self.strategy {
            SchedulingStrategy::SingleDevice | SchedulingStrategy::Pipeline => {
                vec![(self.best_device(), model_size)]
            }
            SchedulingStrategy::LayerParallel { .. } => {
                let total_mem: usize = self.devices.iter().map(|d| d.memory_available).sum();
                if total_mem == 0 {
                    return vec![(self.devices[0].device, model_size)];
                }
                self.devices
                    .iter()
                    .map(|slot| {
                        let share = (slot.memory_available as f64 / total_mem as f64
                            * model_size as f64) as usize;
                        (slot.device, share)
                    })
                    .collect()
            }
        }
    }

    /// Return the registered devices.
    pub fn devices(&self) -> &[DeviceSlot] {
        &self.devices
    }

    /// Return the current scheduling strategy.
    pub fn strategy(&self) -> &SchedulingStrategy {
        &self.strategy
    }

    /// Return the device with the highest compute capability.
    fn best_device(&self) -> Device {
        self.devices
            .iter()
            .max_by(|a, b| {
                a.compute_capability
                    .partial_cmp(&b.compute_capability)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.device)
            .unwrap_or(Device::Cpu)
    }
}

/// Default memory estimate for a device type (bytes).
fn default_memory_for_device(device: &Device) -> usize {
    match device {
        Device::Cpu => 16 * 1024 * 1024 * 1024,
        Device::Cuda(_) => 8 * 1024 * 1024 * 1024,
        Device::OpenCL(_) | Device::Hip(_) | Device::Npu => 8 * 1024 * 1024 * 1024,
        Device::Metal => 8 * 1024 * 1024 * 1024,
    }
}

/// Default relative compute capability score for a device type.
fn default_capability_for_device(device: &Device) -> f64 {
    match device {
        Device::Cpu => 1.0,
        Device::Cuda(_) => 10.0,
        Device::OpenCL(_) | Device::Hip(_) | Device::Npu => 6.0,
        Device::Metal => 7.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_device_auto_configure() {
        let scheduler = MultiDeviceScheduler::auto_configure(&[Device::Cpu]);
        assert_eq!(*scheduler.strategy(), SchedulingStrategy::SingleDevice);
        assert_eq!(scheduler.devices().len(), 1);
    }

    #[test]
    fn test_multi_device_auto_configure() {
        let devices = [Device::Cuda(0), Device::OpenCL(0)];
        let scheduler = MultiDeviceScheduler::auto_configure(&devices);
        assert!(matches!(scheduler.strategy(), SchedulingStrategy::LayerParallel { .. }));
        assert_eq!(scheduler.devices().len(), 2);
    }

    #[test]
    fn test_empty_device_auto_configure() {
        let scheduler = MultiDeviceScheduler::auto_configure(&[]);
        assert_eq!(*scheduler.strategy(), SchedulingStrategy::SingleDevice);
    }

    #[test]
    fn test_assign_layer_single_device() {
        let scheduler = MultiDeviceScheduler::auto_configure(&[Device::Cuda(0)]);
        assert_eq!(scheduler.assign_layer(0), Device::Cuda(0));
        assert_eq!(scheduler.assign_layer(15), Device::Cuda(0));
    }

    #[test]
    fn test_assign_layer_round_robin() {
        let devices = [Device::Cuda(0), Device::OpenCL(0)];
        let scheduler = MultiDeviceScheduler::auto_configure(&devices);
        assert_eq!(scheduler.assign_layer(0), Device::Cuda(0));
        assert_eq!(scheduler.assign_layer(1), Device::OpenCL(0));
        assert_eq!(scheduler.assign_layer(2), Device::Cuda(0));
    }

    #[test]
    fn test_assign_layer_with_split_points() {
        let slots = vec![
            DeviceSlot {
                device: Device::Cuda(0),
                memory_available: 8 * 1024 * 1024 * 1024,
                compute_capability: 10.0,
            },
            DeviceSlot {
                device: Device::OpenCL(0),
                memory_available: 8 * 1024 * 1024 * 1024,
                compute_capability: 6.0,
            },
        ];
        let strategy = SchedulingStrategy::LayerParallel { split_points: vec![16] };
        let scheduler = MultiDeviceScheduler::new(slots, strategy);
        assert_eq!(scheduler.assign_layer(0), Device::Cuda(0));
        assert_eq!(scheduler.assign_layer(15), Device::Cuda(0));
        assert_eq!(scheduler.assign_layer(16), Device::OpenCL(0));
        assert_eq!(scheduler.assign_layer(31), Device::OpenCL(0));
    }

    #[test]
    fn test_memory_plan_single_device() {
        let scheduler = MultiDeviceScheduler::auto_configure(&[Device::Cuda(0)]);
        let plan = scheduler.memory_plan(1_000_000);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].0, Device::Cuda(0));
        assert_eq!(plan[0].1, 1_000_000);
    }

    #[test]
    fn test_memory_plan_multi_device() {
        let slots = vec![
            DeviceSlot {
                device: Device::Cuda(0),
                memory_available: 8 * 1024 * 1024 * 1024,
                compute_capability: 10.0,
            },
            DeviceSlot {
                device: Device::OpenCL(0),
                memory_available: 8 * 1024 * 1024 * 1024,
                compute_capability: 6.0,
            },
        ];
        let strategy = SchedulingStrategy::LayerParallel { split_points: vec![16] };
        let scheduler = MultiDeviceScheduler::new(slots, strategy);
        let plan = scheduler.memory_plan(1_000_000);
        assert_eq!(plan.len(), 2);
        let total: usize = plan.iter().map(|(_, b)| b).sum();
        assert!(total > 0);
    }

    #[test]
    fn test_memory_plan_empty_devices() {
        let scheduler = MultiDeviceScheduler::auto_configure(&[]);
        let plan = scheduler.memory_plan(1_000_000);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].0, Device::Cpu);
    }

    #[test]
    fn test_pipeline_strategy_uses_best_device() {
        let slots = vec![
            DeviceSlot {
                device: Device::Cpu,
                memory_available: 16 * 1024 * 1024 * 1024,
                compute_capability: 1.0,
            },
            DeviceSlot {
                device: Device::Cuda(0),
                memory_available: 8 * 1024 * 1024 * 1024,
                compute_capability: 10.0,
            },
        ];
        let scheduler = MultiDeviceScheduler::new(slots, SchedulingStrategy::Pipeline);
        assert_eq!(scheduler.assign_layer(0), Device::Cuda(0));
        assert_eq!(scheduler.assign_layer(31), Device::Cuda(0));
    }

    #[test]
    fn test_best_device_selection() {
        let slots = vec![
            DeviceSlot {
                device: Device::Cpu,
                memory_available: 16 * 1024 * 1024 * 1024,
                compute_capability: 1.0,
            },
            DeviceSlot {
                device: Device::Cuda(0),
                memory_available: 8 * 1024 * 1024 * 1024,
                compute_capability: 10.0,
            },
            DeviceSlot {
                device: Device::OpenCL(0),
                memory_available: 8 * 1024 * 1024 * 1024,
                compute_capability: 6.0,
            },
        ];
        // SingleDevice strategy uses the best device for all layers
        let scheduler = MultiDeviceScheduler::new(slots, SchedulingStrategy::SingleDevice);
        assert_eq!(scheduler.assign_layer(0), Device::Cuda(0));
        assert_eq!(scheduler.assign_layer(5), Device::Cuda(0));
    }

    #[test]
    fn test_device_slot_properties() {
        let slot = DeviceSlot {
            device: Device::OpenCL(1),
            memory_available: 4 * 1024 * 1024 * 1024,
            compute_capability: 5.5,
        };
        assert_eq!(slot.device, Device::OpenCL(1));
        assert_eq!(slot.compute_capability, 5.5);
    }
}
