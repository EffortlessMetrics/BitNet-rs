//! Layer-level partitioning of transformer models across N devices.
//!
//! [`LayerPartitioner`] distributes layers respecting per-device memory
//! limits and produces a [`PartitionPlan`] with estimated costs.

use crate::model_sharding::{
    CrossDeviceTransfer, DeviceDescriptor, ModelArchitecture, ShardAssignment,
};
use std::fmt;

// ── Partition plan ──────────────────────────────────────────────────────

/// A complete partition plan with cost estimates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionPlan {
    pub entries: Vec<PartitionEntry>,
    /// Total estimated compute cost (arbitrary units, proportional to
    /// layers).
    pub total_compute_cost: u64,
    /// Total estimated cross-device transfer cost in microseconds.
    pub total_transfer_cost_us: u64,
}

impl PartitionPlan {
    /// Diagnostic summary of the partition plan.
    #[allow(clippy::cast_precision_loss)]
    pub fn print_summary(&self) {
        log::info!(
            "Partition plan: {} devices, compute={}, transfer={}µs",
            self.entries.len(),
            self.total_compute_cost,
            self.total_transfer_cost_us,
        );
        for entry in &self.entries {
            log::info!(
                "  Device {} ({}): layers [{}, {}) — mem {:.1} MB, \
                 compute {}",
                entry.device.id,
                entry.device.label,
                entry.assignment.start_layer,
                entry.assignment.end_layer,
                entry.estimated_memory_bytes as f64 / 1_048_576.0,
                entry.compute_cost,
            );
        }
    }
}

/// One device's slice of the partition plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionEntry {
    pub device: DeviceDescriptor,
    pub assignment: ShardAssignment,
    pub estimated_memory_bytes: u64,
    /// Compute cost proportional to the number of layers assigned.
    pub compute_cost: u64,
}

// ── Layer partitioner ───────────────────────────────────────────────────

/// Splits transformer layers across N devices.
///
/// The default mode distributes layers evenly
/// (`ceil(num_layers / num_devices)` per device). When
/// [`Self::memory_constrained`] is enabled the partitioner instead fills
/// each device up to its memory limit before spilling to the next.
pub struct LayerPartitioner {
    devices: Vec<DeviceDescriptor>,
    memory_constrained: bool,
    /// Activation size for transfer cost estimation (bytes).
    activation_bytes: u64,
    /// Compute cost per layer (arbitrary units).
    cost_per_layer: u64,
}

impl LayerPartitioner {
    #[must_use]
    pub const fn new(devices: Vec<DeviceDescriptor>) -> Self {
        Self { devices, memory_constrained: false, activation_bytes: 0, cost_per_layer: 1 }
    }

    /// Enable memory-constrained placement.
    #[must_use]
    pub const fn memory_constrained(mut self, enabled: bool) -> Self {
        self.memory_constrained = enabled;
        self
    }

    /// Set activation size for transfer cost estimation.
    #[must_use]
    pub const fn with_activation_bytes(mut self, bytes: u64) -> Self {
        self.activation_bytes = bytes;
        self
    }

    /// Set per-layer compute cost unit.
    #[must_use]
    pub const fn with_cost_per_layer(mut self, cost: u64) -> Self {
        self.cost_per_layer = cost;
        self
    }

    /// Build a partition plan for the model.
    ///
    /// # Errors
    ///
    /// Returns `Err` if there are no devices or the model cannot fit.
    pub fn partition(&self, model: &ModelArchitecture) -> Result<PartitionPlan, PartitionError> {
        if self.devices.is_empty() {
            return Err(PartitionError::NoDevices);
        }

        let entries = if self.memory_constrained {
            self.partition_memory_constrained(model)?
        } else {
            self.partition_even(model)
        };

        let total_compute_cost: u64 = entries.iter().map(|e| e.compute_cost).sum();
        let total_transfer_cost_us = self.compute_transfer_cost(&entries);

        Ok(PartitionPlan { entries, total_compute_cost, total_transfer_cost_us })
    }

    // ── Even partition ──────────────────────────────────────────────

    fn partition_even(&self, model: &ModelArchitecture) -> Vec<PartitionEntry> {
        let n = self.devices.len();
        let total = model.num_layers;
        let base = total / n;
        let remainder = total % n;

        let mut entries = Vec::with_capacity(n);
        let mut cursor = 0usize;

        for (i, device) in self.devices.iter().enumerate() {
            let count = if i < remainder { base + 1 } else { base };
            let overhead = if i == 0 { model.fixed_overhead_bytes } else { 0 };
            let mem = model.memory_per_layer_bytes * count as u64 + overhead;

            entries.push(PartitionEntry {
                device: device.clone(),
                assignment: ShardAssignment {
                    device_id: device.id,
                    start_layer: cursor,
                    end_layer: cursor + count,
                },
                estimated_memory_bytes: mem,
                compute_cost: count as u64 * self.cost_per_layer,
            });
            cursor += count;
        }
        entries
    }

    // ── Memory-constrained partition ────────────────────────────────

    fn partition_memory_constrained(
        &self,
        model: &ModelArchitecture,
    ) -> Result<Vec<PartitionEntry>, PartitionError> {
        let mut entries = Vec::new();
        let mut remaining_layers = model.num_layers;
        let mut cursor = 0usize;

        for (i, device) in self.devices.iter().enumerate() {
            if remaining_layers == 0 {
                break;
            }
            let overhead = if i == 0 { model.fixed_overhead_bytes } else { 0 };
            let usable = device.memory_bytes.saturating_sub(overhead);
            #[allow(clippy::cast_possible_truncation)]
            let max_layers = if model.memory_per_layer_bytes > 0 {
                (usable / model.memory_per_layer_bytes) as usize
            } else {
                remaining_layers
            };
            let count = max_layers.min(remaining_layers);
            let mem = model.memory_per_layer_bytes * count as u64 + overhead;

            entries.push(PartitionEntry {
                device: device.clone(),
                assignment: ShardAssignment {
                    device_id: device.id,
                    start_layer: cursor,
                    end_layer: cursor + count,
                },
                estimated_memory_bytes: mem,
                compute_cost: count as u64 * self.cost_per_layer,
            });
            cursor += count;
            remaining_layers -= count;
        }

        if remaining_layers > 0 {
            return Err(PartitionError::CannotFit { unplaced_layers: remaining_layers });
        }

        Ok(entries)
    }

    // ── Transfer cost ───────────────────────────────────────────────

    fn compute_transfer_cost(&self, entries: &[PartitionEntry]) -> u64 {
        entries
            .windows(2)
            .map(|pair| {
                let xfer = CrossDeviceTransfer {
                    source_device_id: pair[0].device.id,
                    target_device_id: pair[1].device.id,
                    tensor_bytes: self.activation_bytes,
                    same_backend: pair[0].device.backend == pair[1].device.backend,
                };
                xfer.estimated_cost_us()
            })
            .sum()
    }
}

// ── Errors ──────────────────────────────────────────────────────────────

/// Errors produced during layer partitioning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionError {
    /// No devices available.
    NoDevices,
    /// Not enough total memory to place all layers.
    CannotFit { unplaced_layers: usize },
}

impl fmt::Display for PartitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevices => {
                write!(f, "no devices available for partitioning")
            }
            Self::CannotFit { unplaced_layers } => {
                write!(f, "cannot fit model: {unplaced_layers} layers unplaced",)
            }
        }
    }
}

impl std::error::Error for PartitionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_sharding::{DeviceBackend, DeviceDescriptor};

    fn cuda_device(id: usize, mem_gb: u64) -> DeviceDescriptor {
        DeviceDescriptor {
            id,
            label: format!("CUDA:{id}"),
            memory_bytes: mem_gb * 1_073_741_824,
            backend: DeviceBackend::Cuda,
        }
    }

    fn sample_model(num_layers: usize) -> ModelArchitecture {
        ModelArchitecture {
            num_layers,
            memory_per_layer_bytes: 100 * 1_048_576,
            hidden_dim: 4096,
            num_attention_heads: 32,
            fixed_overhead_bytes: 50 * 1_048_576,
        }
    }

    #[test]
    fn even_partition_single_device() {
        let p = LayerPartitioner::new(vec![cuda_device(0, 8)]);
        let plan = p.partition(&sample_model(10)).unwrap();
        assert_eq!(plan.entries.len(), 1);
        assert_eq!(plan.entries[0].assignment.num_layers(), 10);
    }

    #[test]
    fn even_partition_two_devices() {
        let p = LayerPartitioner::new(vec![cuda_device(0, 8), cuda_device(1, 8)]);
        let plan = p.partition(&sample_model(10)).unwrap();
        assert_eq!(plan.entries[0].assignment.num_layers(), 5);
        assert_eq!(plan.entries[1].assignment.num_layers(), 5);
    }

    #[test]
    fn even_partition_uneven_layers() {
        let p =
            LayerPartitioner::new(vec![cuda_device(0, 8), cuda_device(1, 8), cuda_device(2, 8)]);
        let plan = p.partition(&sample_model(7)).unwrap();
        assert_eq!(plan.entries[0].assignment.num_layers(), 3);
        assert_eq!(plan.entries[1].assignment.num_layers(), 2);
        assert_eq!(plan.entries[2].assignment.num_layers(), 2);
    }

    #[test]
    fn memory_constrained_fills_up_device() {
        // Device 0: 350 MB → can hold 3 layers (300 MB) + 50 MB overhead.
        // Device 1: 8 GB → holds the remaining 7 layers.
        let devices = vec![
            DeviceDescriptor {
                id: 0,
                label: "small".into(),
                memory_bytes: 350 * 1_048_576,
                backend: DeviceBackend::Cuda,
            },
            cuda_device(1, 8),
        ];
        let p = LayerPartitioner::new(devices).memory_constrained(true);
        let plan = p.partition(&sample_model(10)).unwrap();

        assert_eq!(plan.entries[0].assignment.num_layers(), 3);
        assert_eq!(plan.entries[1].assignment.num_layers(), 7);
    }

    #[test]
    fn memory_constrained_cannot_fit() {
        // Only 150 MB → 1 layer + 50 MB overhead max = 1 layer.
        let devices = vec![DeviceDescriptor {
            id: 0,
            label: "tiny".into(),
            memory_bytes: 150 * 1_048_576,
            backend: DeviceBackend::Cuda,
        }];
        let p = LayerPartitioner::new(devices).memory_constrained(true);
        let result = p.partition(&sample_model(10));
        assert!(matches!(result, Err(PartitionError::CannotFit { .. })));
    }

    #[test]
    fn no_devices_returns_error() {
        let p = LayerPartitioner::new(vec![]);
        assert_eq!(p.partition(&sample_model(5)), Err(PartitionError::NoDevices),);
    }

    #[test]
    fn zero_layers_empty_model() {
        let p = LayerPartitioner::new(vec![cuda_device(0, 8)]);
        let plan = p.partition(&sample_model(0)).unwrap();
        assert_eq!(plan.entries[0].assignment.num_layers(), 0);
        assert_eq!(plan.total_compute_cost, 0);
    }

    #[test]
    fn transfer_cost_is_zero_for_single_device() {
        let p = LayerPartitioner::new(vec![cuda_device(0, 8)]).with_activation_bytes(1_048_576);
        let plan = p.partition(&sample_model(10)).unwrap();
        assert_eq!(plan.total_transfer_cost_us, 0);
    }

    #[test]
    fn transfer_cost_scales_with_devices() {
        let p =
            LayerPartitioner::new(vec![cuda_device(0, 8), cuda_device(1, 8), cuda_device(2, 8)])
                .with_activation_bytes(12_000);
        let plan = p.partition(&sample_model(12)).unwrap();
        // 2 transfers × 12_000 / 12 = 2 × 1000 = 2000 µs.
        assert_eq!(plan.total_transfer_cost_us, 2_000);
    }

    #[test]
    fn compute_cost_uses_custom_unit() {
        let p = LayerPartitioner::new(vec![cuda_device(0, 8)]).with_cost_per_layer(100);
        let plan = p.partition(&sample_model(5)).unwrap();
        assert_eq!(plan.total_compute_cost, 500);
    }

    #[test]
    fn partition_error_display() {
        let e = PartitionError::NoDevices;
        assert_eq!(e.to_string(), "no devices available for partitioning",);

        let e = PartitionError::CannotFit { unplaced_layers: 3 };
        assert!(e.to_string().contains("3 layers unplaced"));
    }
}
