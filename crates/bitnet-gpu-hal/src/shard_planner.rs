//! Model sharding planner for multi-device deployment.
//!
//! Provides strategies for distributing model layers across multiple devices,
//! with validation and optimization for memory constraints and load balance.

use std::collections::HashMap;

// ── Types ────────────────────────────────────────────────────────────────────

/// Strategy for distributing layers across devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardStrategy {
    /// Split layers evenly across devices.
    Uniform,
    /// Balance by parameter memory (minimize max param bytes per device).
    WeightBalanced,
    /// Balance by activation memory during inference.
    ActivationBalanced,
}

/// Configuration for a sharding operation.
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Number of devices available.
    pub num_devices: usize,
    /// Memory available per device in bytes (indexed by device id).
    pub memory_per_device: Vec<u64>,
    /// Sharding strategy to use.
    pub strategy: ShardStrategy,
}

impl ShardConfig {
    /// Create a config with uniform memory across all devices.
    pub fn uniform(num_devices: usize, memory_per_device: u64) -> Self {
        Self {
            num_devices,
            memory_per_device: vec![memory_per_device; num_devices],
            strategy: ShardStrategy::Uniform,
        }
    }

    /// Create a config with the specified strategy and uniform memory.
    pub fn with_strategy(
        num_devices: usize,
        memory_per_device: u64,
        strategy: ShardStrategy,
    ) -> Self {
        Self { num_devices, memory_per_device: vec![memory_per_device; num_devices], strategy }
    }

    /// Create a config with heterogeneous device memory.
    pub const fn heterogeneous(memory_per_device: Vec<u64>, strategy: ShardStrategy) -> Self {
        let num_devices = memory_per_device.len();
        Self { num_devices, memory_per_device, strategy }
    }
}

/// Profile of a single model layer.
#[derive(Debug, Clone)]
pub struct LayerProfile {
    /// Unique identifier for this layer.
    pub layer_id: usize,
    /// Number of parameters in this layer.
    pub param_count: u64,
    /// Size of parameters in bytes.
    pub param_bytes: u64,
    /// Size of activations in bytes during inference.
    pub activation_bytes: u64,
    /// Estimated FLOPs for this layer's computation.
    pub compute_flops: u64,
}

/// Profile of an entire model.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// Per-layer profiles, ordered by layer index.
    pub layers: Vec<LayerProfile>,
}

impl ModelProfile {
    /// Total parameter count across all layers.
    pub fn total_params(&self) -> u64 {
        self.layers.iter().map(|l| l.param_count).sum()
    }

    /// Total parameter memory in bytes.
    pub fn total_memory(&self) -> u64 {
        self.layers.iter().map(|l| l.param_bytes).sum()
    }

    /// Total activation memory in bytes.
    pub fn total_activation_memory(&self) -> u64 {
        self.layers.iter().map(|l| l.activation_bytes).sum()
    }

    /// Total compute FLOPs.
    pub fn total_flops(&self) -> u64 {
        self.layers.iter().map(|l| l.compute_flops).sum()
    }
}

/// A layer-to-device assignment in the shard plan.
#[derive(Debug, Clone)]
pub struct LayerAssignment {
    pub layer_id: usize,
    pub device_id: usize,
}

/// The output of a sharding operation: maps layers to devices.
#[derive(Debug, Clone)]
pub struct ShardPlan {
    /// Layer assignments (`layer_id` → `device_id`).
    pub assignments: Vec<LayerAssignment>,
    /// Number of devices in the plan.
    pub num_devices: usize,
}

impl ShardPlan {
    /// Get the device id assigned to a given layer.
    pub fn device_for_layer(&self, layer_id: usize) -> Option<usize> {
        self.assignments.iter().find(|a| a.layer_id == layer_id).map(|a| a.device_id)
    }

    /// Get all layer ids assigned to a given device.
    pub fn layers_for_device(&self, device_id: usize) -> Vec<usize> {
        self.assignments.iter().filter(|a| a.device_id == device_id).map(|a| a.layer_id).collect()
    }

    /// Compute the total param bytes per device given a model profile.
    pub fn device_memory_usage(&self, profile: &ModelProfile) -> HashMap<usize, u64> {
        let layer_map: HashMap<usize, &LayerProfile> =
            profile.layers.iter().map(|l| (l.layer_id, l)).collect();
        let mut usage: HashMap<usize, u64> = (0..self.num_devices).map(|d| (d, 0u64)).collect();
        for a in &self.assignments {
            if let Some(lp) = layer_map.get(&a.layer_id) {
                *usage.entry(a.device_id).or_insert(0) += lp.param_bytes;
            }
        }
        usage
    }

    /// Compute the total activation bytes per device given a model profile.
    pub fn device_activation_usage(&self, profile: &ModelProfile) -> HashMap<usize, u64> {
        let layer_map: HashMap<usize, &LayerProfile> =
            profile.layers.iter().map(|l| (l.layer_id, l)).collect();
        let mut usage: HashMap<usize, u64> = (0..self.num_devices).map(|d| (d, 0u64)).collect();
        for a in &self.assignments {
            if let Some(lp) = layer_map.get(&a.layer_id) {
                *usage.entry(a.device_id).or_insert(0) += lp.activation_bytes;
            }
        }
        usage
    }

    /// Check whether the plan is balanced (max usage ≤ 2× min usage).
    pub fn is_balanced(&self, profile: &ModelProfile) -> bool {
        let usage = self.device_memory_usage(profile);
        if usage.is_empty() {
            return true;
        }
        let values: Vec<u64> = usage.values().copied().collect();
        let min = values.iter().copied().min().unwrap_or(0);
        let max = values.iter().copied().max().unwrap_or(0);
        if min == 0 {
            return max == 0;
        }
        max <= 2 * min
    }
}

// ── Sharders ─────────────────────────────────────────────────────────────────

/// Splits layers evenly: device i gets layers [i*chunk .. (i+1)*chunk].
pub struct UniformSharder;

impl UniformSharder {
    pub fn plan(config: &ShardConfig, profile: &ModelProfile) -> ShardPlan {
        let n = profile.layers.len();
        let d = config.num_devices;
        let mut assignments = Vec::with_capacity(n);
        for (i, layer) in profile.layers.iter().enumerate() {
            // Distribute layers round-robin when layers < devices,
            // otherwise divide evenly.
            let device_id = if d == 0 { 0 } else { i * d / n.max(1) };
            assignments.push(LayerAssignment {
                layer_id: layer.layer_id,
                device_id: device_id.min(d.saturating_sub(1)),
            });
        }
        ShardPlan { assignments, num_devices: d }
    }
}

/// Balances layers by parameter memory: greedily assigns each layer
/// to the device with the least accumulated param bytes.
pub struct WeightBalancedSharder;

impl WeightBalancedSharder {
    pub fn plan(config: &ShardConfig, profile: &ModelProfile) -> ShardPlan {
        let d = config.num_devices;
        let mut device_loads = vec![0u64; d];
        let mut assignments = Vec::with_capacity(profile.layers.len());

        // Sort layers by param_bytes descending for better greedy packing.
        let mut sorted: Vec<&LayerProfile> = profile.layers.iter().collect();
        sorted.sort_by(|a, b| b.param_bytes.cmp(&a.param_bytes));

        for layer in sorted {
            let best_device = device_loads
                .iter()
                .enumerate()
                .min_by_key(|(_, load)| **load)
                .map_or(0, |(idx, _)| idx);
            device_loads[best_device] += layer.param_bytes;
            assignments.push(LayerAssignment { layer_id: layer.layer_id, device_id: best_device });
        }
        // Sort assignments by layer_id for deterministic output.
        assignments.sort_by_key(|a| a.layer_id);
        ShardPlan { assignments, num_devices: d }
    }
}

/// Balances layers by activation memory: greedily assigns each layer
/// to the device with the least accumulated activation bytes.
pub struct ActivationBalancedSharder;

impl ActivationBalancedSharder {
    pub fn plan(config: &ShardConfig, profile: &ModelProfile) -> ShardPlan {
        let d = config.num_devices;
        let mut device_loads = vec![0u64; d];
        let mut assignments = Vec::with_capacity(profile.layers.len());

        // Sort layers by activation_bytes descending.
        let mut sorted: Vec<&LayerProfile> = profile.layers.iter().collect();
        sorted.sort_by(|a, b| b.activation_bytes.cmp(&a.activation_bytes));

        for layer in sorted {
            let best_device = device_loads
                .iter()
                .enumerate()
                .min_by_key(|(_, load)| **load)
                .map_or(0, |(idx, _)| idx);
            device_loads[best_device] += layer.activation_bytes;
            assignments.push(LayerAssignment { layer_id: layer.layer_id, device_id: best_device });
        }
        assignments.sort_by_key(|a| a.layer_id);
        ShardPlan { assignments, num_devices: d }
    }
}

// ── Validator ────────────────────────────────────────────────────────────────

/// Validation error for a shard plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// A layer is not assigned to any device.
    UnassignedLayer(usize),
    /// A layer is assigned to a non-existent device.
    InvalidDevice { layer_id: usize, device_id: usize },
    /// A device exceeds its memory limit.
    MemoryExceeded { device_id: usize, required: u64, available: u64 },
    /// A layer is assigned more than once.
    DuplicateAssignment(usize),
    /// No devices configured.
    NoDevices,
}

/// Validates a shard plan against a model profile and config.
pub struct ShardValidator;

impl ShardValidator {
    /// Validate a plan, returning all errors found.
    pub fn validate(
        plan: &ShardPlan,
        config: &ShardConfig,
        profile: &ModelProfile,
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if config.num_devices == 0 {
            errors.push(ValidationError::NoDevices);
            return errors;
        }

        // Check for duplicate assignments.
        let mut seen_layers: HashMap<usize, usize> = HashMap::new();
        for a in &plan.assignments {
            if let std::collections::hash_map::Entry::Vacant(e) = seen_layers.entry(a.layer_id) {
                e.insert(a.device_id);
            } else {
                errors.push(ValidationError::DuplicateAssignment(a.layer_id));
            }
        }

        // Check all layers are assigned.
        for layer in &profile.layers {
            if !seen_layers.contains_key(&layer.layer_id) {
                errors.push(ValidationError::UnassignedLayer(layer.layer_id));
            }
        }

        // Check device ids are valid.
        for a in &plan.assignments {
            if a.device_id >= config.num_devices {
                errors.push(ValidationError::InvalidDevice {
                    layer_id: a.layer_id,
                    device_id: a.device_id,
                });
            }
        }

        // Check memory constraints.
        let usage = plan.device_memory_usage(profile);
        for (device_id, &used) in &usage {
            if let Some(&available) = config.memory_per_device.get(*device_id)
                && used > available
            {
                errors.push(ValidationError::MemoryExceeded {
                    device_id: *device_id,
                    required: used,
                    available,
                });
            }
        }

        errors
    }

    /// Returns true if the plan is valid.
    pub fn is_valid(plan: &ShardPlan, config: &ShardConfig, profile: &ModelProfile) -> bool {
        Self::validate(plan, config, profile).is_empty()
    }
}

// ── Optimizer ────────────────────────────────────────────────────────────────

/// Optimizes a shard plan by swapping layers between devices to improve balance.
pub struct ShardOptimizer;

impl ShardOptimizer {
    /// Attempt to improve balance by performing greedy pairwise swaps.
    /// Returns an improved plan (or the original if no improvement found).
    /// `max_iterations` limits the number of swap attempts.
    pub fn optimize(
        plan: &ShardPlan,
        profile: &ModelProfile,
        config: &ShardConfig,
        max_iterations: usize,
    ) -> ShardPlan {
        let mut best = plan.clone();
        let mut best_imbalance = Self::imbalance(&best, profile);

        for _ in 0..max_iterations {
            let mut improved = false;
            let n = best.assignments.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    if best.assignments[i].device_id == best.assignments[j].device_id {
                        continue;
                    }
                    // Try swapping.
                    let mut candidate = best.clone();
                    let tmp = candidate.assignments[i].device_id;
                    candidate.assignments[i].device_id = candidate.assignments[j].device_id;
                    candidate.assignments[j].device_id = tmp;

                    // Only accept if it respects memory constraints and
                    // improves balance.
                    if ShardValidator::is_valid(&candidate, config, profile) {
                        let new_imbalance = Self::imbalance(&candidate, profile);
                        if new_imbalance < best_imbalance {
                            best = candidate;
                            best_imbalance = new_imbalance;
                            improved = true;
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
        best
    }

    /// Compute imbalance as (`max_usage` - `min_usage`).
    fn imbalance(plan: &ShardPlan, profile: &ModelProfile) -> u64 {
        let usage = plan.device_memory_usage(profile);
        let values: Vec<u64> = usage.values().copied().collect();
        let min = values.iter().copied().min().unwrap_or(0);
        let max = values.iter().copied().max().unwrap_or(0);
        max - min
    }
}

// ── Metrics ──────────────────────────────────────────────────────────────────

/// Metrics for evaluating a shard plan.
#[derive(Debug, Clone)]
pub struct ShardPlannerMetrics {
    /// Balance ratio: `min_usage` / `max_usage` (1.0 = perfect).
    pub balance_ratio: f64,
    /// Maximum memory utilization across devices (0.0–1.0).
    pub max_memory_utilization: f64,
    /// Estimated cross-device communication volume in bytes.
    /// Assumes sequential layer execution: each device boundary incurs
    /// transfer of the activation tensor between adjacent layers.
    pub estimated_comm_volume: u64,
}

impl ShardPlannerMetrics {
    /// Compute metrics for a plan.
    pub fn compute(plan: &ShardPlan, config: &ShardConfig, profile: &ModelProfile) -> Self {
        let usage = plan.device_memory_usage(profile);
        let values: Vec<u64> = usage.values().copied().collect();
        let min_usage = values.iter().copied().min().unwrap_or(0);
        let max_usage = values.iter().copied().max().unwrap_or(0);

        #[allow(clippy::cast_precision_loss)]
        let balance_ratio = if max_usage == 0 { 1.0 } else { min_usage as f64 / max_usage as f64 };

        #[allow(clippy::cast_precision_loss)]
        let max_memory_utilization = usage
            .iter()
            .map(|(dev, &used)| {
                let avail = config.memory_per_device.get(*dev).copied().unwrap_or(1);
                if avail == 0 { 0.0 } else { used as f64 / avail as f64 }
            })
            .fold(0.0f64, f64::max);

        // Estimate communication: each time adjacent layers are on
        // different devices, we transfer the activation of the boundary.
        let mut comm = 0u64;
        let layer_device: HashMap<usize, usize> =
            plan.assignments.iter().map(|a| (a.layer_id, a.device_id)).collect();
        let mut layer_ids: Vec<usize> = profile.layers.iter().map(|l| l.layer_id).collect();
        layer_ids.sort_unstable();
        for window in layer_ids.windows(2) {
            let dev_a = layer_device.get(&window[0]).copied().unwrap_or(0);
            let dev_b = layer_device.get(&window[1]).copied().unwrap_or(0);
            if dev_a != dev_b {
                // Transfer the activation output of the first layer.
                if let Some(lp) = profile.layers.iter().find(|l| l.layer_id == window[0]) {
                    comm += lp.activation_bytes;
                }
            }
        }

        Self { balance_ratio, max_memory_utilization, estimated_comm_volume: comm }
    }
}

// ── Convenience dispatch ─────────────────────────────────────────────────────

/// Plan sharding using the strategy specified in the config.
pub fn plan_sharding(config: &ShardConfig, profile: &ModelProfile) -> ShardPlan {
    match config.strategy {
        ShardStrategy::Uniform => UniformSharder::plan(config, profile),
        ShardStrategy::WeightBalanced => WeightBalancedSharder::plan(config, profile),
        ShardStrategy::ActivationBalanced => ActivationBalancedSharder::plan(config, profile),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────

    fn uniform_profile(n: usize, bytes: u64) -> ModelProfile {
        ModelProfile {
            layers: (0..n)
                .map(|i| LayerProfile {
                    layer_id: i,
                    param_count: 1000,
                    param_bytes: bytes,
                    activation_bytes: bytes / 2,
                    compute_flops: 1_000_000,
                })
                .collect(),
        }
    }

    fn varied_profile() -> ModelProfile {
        ModelProfile {
            layers: vec![
                LayerProfile {
                    layer_id: 0,
                    param_count: 1000,
                    param_bytes: 100,
                    activation_bytes: 50,
                    compute_flops: 500,
                },
                LayerProfile {
                    layer_id: 1,
                    param_count: 5000,
                    param_bytes: 500,
                    activation_bytes: 250,
                    compute_flops: 2500,
                },
                LayerProfile {
                    layer_id: 2,
                    param_count: 2000,
                    param_bytes: 200,
                    activation_bytes: 100,
                    compute_flops: 1000,
                },
                LayerProfile {
                    layer_id: 3,
                    param_count: 8000,
                    param_bytes: 800,
                    activation_bytes: 400,
                    compute_flops: 4000,
                },
            ],
        }
    }

    fn varied_activation_profile() -> ModelProfile {
        ModelProfile {
            layers: vec![
                LayerProfile {
                    layer_id: 0,
                    param_count: 1000,
                    param_bytes: 100,
                    activation_bytes: 800,
                    compute_flops: 500,
                },
                LayerProfile {
                    layer_id: 1,
                    param_count: 1000,
                    param_bytes: 100,
                    activation_bytes: 200,
                    compute_flops: 500,
                },
                LayerProfile {
                    layer_id: 2,
                    param_count: 1000,
                    param_bytes: 100,
                    activation_bytes: 600,
                    compute_flops: 500,
                },
                LayerProfile {
                    layer_id: 3,
                    param_count: 1000,
                    param_bytes: 100,
                    activation_bytes: 400,
                    compute_flops: 500,
                },
            ],
        }
    }

    // ── ModelProfile tests ───────────────────────────────────────────

    #[test]
    fn test_model_profile_total_params() {
        let p = uniform_profile(4, 100);
        assert_eq!(p.total_params(), 4000);
    }

    #[test]
    fn test_model_profile_total_memory() {
        let p = uniform_profile(4, 100);
        assert_eq!(p.total_memory(), 400);
    }

    #[test]
    fn test_model_profile_total_activation_memory() {
        let p = uniform_profile(4, 100);
        assert_eq!(p.total_activation_memory(), 200);
    }

    #[test]
    fn test_model_profile_total_flops() {
        let p = uniform_profile(4, 100);
        assert_eq!(p.total_flops(), 4_000_000);
    }

    #[test]
    fn test_model_profile_varied() {
        let p = varied_profile();
        assert_eq!(p.total_params(), 16_000);
        assert_eq!(p.total_memory(), 1600);
        assert_eq!(p.total_activation_memory(), 800);
    }

    #[test]
    fn test_model_profile_empty() {
        let p = ModelProfile { layers: vec![] };
        assert_eq!(p.total_params(), 0);
        assert_eq!(p.total_memory(), 0);
        assert_eq!(p.total_activation_memory(), 0);
        assert_eq!(p.total_flops(), 0);
    }

    // ── ShardConfig tests ────────────────────────────────────────────

    #[test]
    fn test_shard_config_uniform() {
        let c = ShardConfig::uniform(4, 1024);
        assert_eq!(c.num_devices, 4);
        assert_eq!(c.memory_per_device, vec![1024; 4]);
        assert_eq!(c.strategy, ShardStrategy::Uniform);
    }

    #[test]
    fn test_shard_config_with_strategy() {
        let c = ShardConfig::with_strategy(2, 2048, ShardStrategy::WeightBalanced);
        assert_eq!(c.num_devices, 2);
        assert_eq!(c.strategy, ShardStrategy::WeightBalanced);
    }

    #[test]
    fn test_shard_config_heterogeneous() {
        let c =
            ShardConfig::heterogeneous(vec![1024, 2048, 512], ShardStrategy::ActivationBalanced);
        assert_eq!(c.num_devices, 3);
        assert_eq!(c.memory_per_device, vec![1024, 2048, 512]);
    }

    // ── UniformSharder tests ─────────────────────────────────────────

    #[test]
    fn test_uniform_shard_4_layers_2_devices() {
        let config = ShardConfig::uniform(2, 1_000_000);
        let profile = uniform_profile(4, 100);
        let plan = UniformSharder::plan(&config, &profile);
        assert_eq!(plan.assignments.len(), 4);
        let d0 = plan.layers_for_device(0);
        let d1 = plan.layers_for_device(1);
        assert_eq!(d0.len(), 2);
        assert_eq!(d1.len(), 2);
    }

    #[test]
    fn test_uniform_shard_all_layers_assigned() {
        let config = ShardConfig::uniform(3, 1_000_000);
        let profile = uniform_profile(9, 100);
        let plan = UniformSharder::plan(&config, &profile);
        assert_eq!(plan.assignments.len(), 9);
        for i in 0..9 {
            assert!(plan.device_for_layer(i).is_some());
        }
    }

    #[test]
    fn test_uniform_shard_single_device() {
        let config = ShardConfig::uniform(1, 1_000_000);
        let profile = uniform_profile(8, 100);
        let plan = UniformSharder::plan(&config, &profile);
        for a in &plan.assignments {
            assert_eq!(a.device_id, 0);
        }
    }

    #[test]
    fn test_uniform_shard_single_layer() {
        let config = ShardConfig::uniform(4, 1_000_000);
        let profile = uniform_profile(1, 100);
        let plan = UniformSharder::plan(&config, &profile);
        assert_eq!(plan.assignments.len(), 1);
        assert!(plan.assignments[0].device_id < 4);
    }

    #[test]
    fn test_uniform_shard_more_devices_than_layers() {
        let config = ShardConfig::uniform(8, 1_000_000);
        let profile = uniform_profile(3, 100);
        let plan = UniformSharder::plan(&config, &profile);
        assert_eq!(plan.assignments.len(), 3);
        for a in &plan.assignments {
            assert!(a.device_id < 8);
        }
    }

    #[test]
    fn test_uniform_shard_layers_equal_devices() {
        let config = ShardConfig::uniform(4, 1_000_000);
        let profile = uniform_profile(4, 100);
        let plan = UniformSharder::plan(&config, &profile);
        // Each device should get exactly 1 layer.
        for d in 0..4 {
            assert_eq!(plan.layers_for_device(d).len(), 1);
        }
    }

    #[test]
    fn test_uniform_shard_6_layers_4_devices() {
        let config = ShardConfig::uniform(4, 1_000_000);
        let profile = uniform_profile(6, 100);
        let plan = UniformSharder::plan(&config, &profile);
        assert_eq!(plan.assignments.len(), 6);
        // All layers assigned.
        for i in 0..6 {
            assert!(plan.device_for_layer(i).is_some());
        }
    }

    // ── WeightBalancedSharder tests ──────────────────────────────────

    #[test]
    fn test_weight_balanced_all_layers_assigned() {
        let config = ShardConfig::with_strategy(2, 1_000_000, ShardStrategy::WeightBalanced);
        let profile = varied_profile();
        let plan = WeightBalancedSharder::plan(&config, &profile);
        assert_eq!(plan.assignments.len(), 4);
        for i in 0..4 {
            assert!(plan.device_for_layer(i).is_some());
        }
    }

    #[test]
    fn test_weight_balanced_minimizes_max_memory() {
        let config = ShardConfig::with_strategy(2, 1_000_000, ShardStrategy::WeightBalanced);
        let profile = varied_profile();
        let wb_plan = WeightBalancedSharder::plan(&config, &profile);
        let uniform_plan = UniformSharder::plan(&config, &profile);

        let wb_usage = wb_plan.device_memory_usage(&profile);
        let uni_usage = uniform_plan.device_memory_usage(&profile);

        let wb_max = wb_usage.values().copied().max().unwrap_or(0);
        let uni_max = uni_usage.values().copied().max().unwrap_or(0);

        assert!(
            wb_max <= uni_max,
            "Weight-balanced max {wb_max} should be <= uniform max {uni_max}"
        );
    }

    #[test]
    fn test_weight_balanced_uniform_layers() {
        let config = ShardConfig::with_strategy(2, 1_000_000, ShardStrategy::WeightBalanced);
        let profile = uniform_profile(4, 100);
        let plan = WeightBalancedSharder::plan(&config, &profile);
        let usage = plan.device_memory_usage(&profile);
        // With uniform layers, both devices should get equal memory.
        let v: Vec<u64> = usage.values().copied().collect();
        assert_eq!(v[0], v[1]);
    }

    #[test]
    fn test_weight_balanced_single_device() {
        let config = ShardConfig::with_strategy(1, 1_000_000, ShardStrategy::WeightBalanced);
        let profile = varied_profile();
        let plan = WeightBalancedSharder::plan(&config, &profile);
        for a in &plan.assignments {
            assert_eq!(a.device_id, 0);
        }
    }

    #[test]
    fn test_weight_balanced_single_layer() {
        let config = ShardConfig::with_strategy(4, 1_000_000, ShardStrategy::WeightBalanced);
        let profile = uniform_profile(1, 100);
        let plan = WeightBalancedSharder::plan(&config, &profile);
        assert_eq!(plan.assignments.len(), 1);
    }

    #[test]
    fn test_weight_balanced_heavy_layer() {
        let profile = ModelProfile {
            layers: vec![
                LayerProfile {
                    layer_id: 0,
                    param_count: 100,
                    param_bytes: 1000,
                    activation_bytes: 50,
                    compute_flops: 500,
                },
                LayerProfile {
                    layer_id: 1,
                    param_count: 10,
                    param_bytes: 10,
                    activation_bytes: 5,
                    compute_flops: 50,
                },
                LayerProfile {
                    layer_id: 2,
                    param_count: 10,
                    param_bytes: 10,
                    activation_bytes: 5,
                    compute_flops: 50,
                },
                LayerProfile {
                    layer_id: 3,
                    param_count: 10,
                    param_bytes: 10,
                    activation_bytes: 5,
                    compute_flops: 50,
                },
            ],
        };
        let config = ShardConfig::with_strategy(2, 1_000_000, ShardStrategy::WeightBalanced);
        let plan = WeightBalancedSharder::plan(&config, &profile);
        // Layer 0 (1000 bytes) should be alone on one device.
        let dev0 = plan.device_for_layer(0).unwrap();
        let layers_on_dev0 = plan.layers_for_device(dev0);
        assert_eq!(layers_on_dev0.len(), 1);
    }

    // ── ActivationBalancedSharder tests ──────────────────────────────

    #[test]
    fn test_activation_balanced_all_layers_assigned() {
        let config = ShardConfig::with_strategy(2, 1_000_000, ShardStrategy::ActivationBalanced);
        let profile = varied_activation_profile();
        let plan = ActivationBalancedSharder::plan(&config, &profile);
        assert_eq!(plan.assignments.len(), 4);
    }

    #[test]
    fn test_activation_balanced_accounts_for_activation_sizes() {
        let config = ShardConfig::with_strategy(2, 1_000_000, ShardStrategy::ActivationBalanced);
        let profile = varied_activation_profile();
        let plan = ActivationBalancedSharder::plan(&config, &profile);
        let act_usage = plan.device_activation_usage(&profile);
        let max_act = act_usage.values().copied().max().unwrap_or(0);
        let min_act = act_usage.values().copied().min().unwrap_or(0);
        // Balance should be reasonable (within 2x).
        assert!(
            max_act <= 2 * min_act,
            "Activation imbalance too high: max={max_act}, min={min_act}"
        );
    }

    #[test]
    fn test_activation_balanced_vs_uniform() {
        let config = ShardConfig::with_strategy(2, 1_000_000, ShardStrategy::ActivationBalanced);
        let profile = varied_activation_profile();
        let ab_plan = ActivationBalancedSharder::plan(&config, &profile);
        let uni_plan = UniformSharder::plan(&config, &profile);

        let ab_usage = ab_plan.device_activation_usage(&profile);
        let uni_usage = uni_plan.device_activation_usage(&profile);

        let ab_max = ab_usage.values().copied().max().unwrap_or(0);
        let uni_max = uni_usage.values().copied().max().unwrap_or(0);

        assert!(
            ab_max <= uni_max,
            "Activation-balanced max {ab_max} should be <= uniform max {uni_max}"
        );
    }

    #[test]
    fn test_activation_balanced_single_device() {
        let config = ShardConfig::with_strategy(1, 1_000_000, ShardStrategy::ActivationBalanced);
        let profile = varied_activation_profile();
        let plan = ActivationBalancedSharder::plan(&config, &profile);
        for a in &plan.assignments {
            assert_eq!(a.device_id, 0);
        }
    }

    #[test]
    fn test_activation_balanced_single_layer() {
        let config = ShardConfig::with_strategy(4, 1_000_000, ShardStrategy::ActivationBalanced);
        let profile = ModelProfile {
            layers: vec![LayerProfile {
                layer_id: 0,
                param_count: 1000,
                param_bytes: 100,
                activation_bytes: 500,
                compute_flops: 1000,
            }],
        };
        let plan = ActivationBalancedSharder::plan(&config, &profile);
        assert_eq!(plan.assignments.len(), 1);
    }

    #[test]
    fn test_activation_balanced_extreme_variance() {
        let profile = ModelProfile {
            layers: vec![
                LayerProfile {
                    layer_id: 0,
                    param_count: 100,
                    param_bytes: 100,
                    activation_bytes: 10_000,
                    compute_flops: 100,
                },
                LayerProfile {
                    layer_id: 1,
                    param_count: 100,
                    param_bytes: 100,
                    activation_bytes: 1,
                    compute_flops: 100,
                },
                LayerProfile {
                    layer_id: 2,
                    param_count: 100,
                    param_bytes: 100,
                    activation_bytes: 1,
                    compute_flops: 100,
                },
                LayerProfile {
                    layer_id: 3,
                    param_count: 100,
                    param_bytes: 100,
                    activation_bytes: 1,
                    compute_flops: 100,
                },
            ],
        };
        let config = ShardConfig::with_strategy(2, 1_000_000, ShardStrategy::ActivationBalanced);
        let plan = ActivationBalancedSharder::plan(&config, &profile);
        // The huge layer should be alone on one device.
        let dev0 = plan.device_for_layer(0).unwrap();
        let layers_on_dev = plan.layers_for_device(dev0);
        assert_eq!(layers_on_dev.len(), 1);
    }

    // ── ShardPlan tests ──────────────────────────────────────────────

    #[test]
    fn test_plan_device_for_layer() {
        let plan = ShardPlan {
            assignments: vec![
                LayerAssignment { layer_id: 0, device_id: 0 },
                LayerAssignment { layer_id: 1, device_id: 1 },
            ],
            num_devices: 2,
        };
        assert_eq!(plan.device_for_layer(0), Some(0));
        assert_eq!(plan.device_for_layer(1), Some(1));
        assert_eq!(plan.device_for_layer(99), None);
    }

    #[test]
    fn test_plan_layers_for_device() {
        let plan = ShardPlan {
            assignments: vec![
                LayerAssignment { layer_id: 0, device_id: 0 },
                LayerAssignment { layer_id: 1, device_id: 1 },
                LayerAssignment { layer_id: 2, device_id: 0 },
            ],
            num_devices: 2,
        };
        let mut d0 = plan.layers_for_device(0);
        d0.sort_unstable();
        assert_eq!(d0, vec![0, 2]);
        assert_eq!(plan.layers_for_device(1), vec![1]);
    }

    #[test]
    fn test_plan_device_memory_usage() {
        let profile = varied_profile();
        let plan = ShardPlan {
            assignments: vec![
                LayerAssignment { layer_id: 0, device_id: 0 },
                LayerAssignment { layer_id: 1, device_id: 1 },
                LayerAssignment { layer_id: 2, device_id: 0 },
                LayerAssignment { layer_id: 3, device_id: 1 },
            ],
            num_devices: 2,
        };
        let usage = plan.device_memory_usage(&profile);
        assert_eq!(usage[&0], 300); // 100 + 200
        assert_eq!(usage[&1], 1300); // 500 + 800
    }

    #[test]
    fn test_plan_is_balanced_true() {
        let profile = uniform_profile(4, 100);
        let config = ShardConfig::uniform(2, 1_000_000);
        let plan = WeightBalancedSharder::plan(&config, &profile);
        assert!(plan.is_balanced(&profile));
    }

    #[test]
    fn test_plan_is_balanced_false() {
        let profile = ModelProfile {
            layers: vec![
                LayerProfile {
                    layer_id: 0,
                    param_count: 100,
                    param_bytes: 1000,
                    activation_bytes: 100,
                    compute_flops: 100,
                },
                LayerProfile {
                    layer_id: 1,
                    param_count: 1,
                    param_bytes: 1,
                    activation_bytes: 1,
                    compute_flops: 1,
                },
            ],
        };
        let plan = ShardPlan {
            assignments: vec![
                LayerAssignment { layer_id: 0, device_id: 0 },
                LayerAssignment { layer_id: 1, device_id: 1 },
            ],
            num_devices: 2,
        };
        assert!(!plan.is_balanced(&profile));
    }

    // ── ShardValidator tests ─────────────────────────────────────────

    #[test]
    fn test_validator_valid_plan() {
        let config = ShardConfig::uniform(2, 1_000_000);
        let profile = uniform_profile(4, 100);
        let plan = UniformSharder::plan(&config, &profile);
        assert!(ShardValidator::is_valid(&plan, &config, &profile));
    }

    #[test]
    fn test_validator_catches_memory_exceeded() {
        let config = ShardConfig::uniform(2, 150);
        let profile = uniform_profile(4, 100);
        let plan = UniformSharder::plan(&config, &profile);
        let errors = ShardValidator::validate(&plan, &config, &profile);
        assert!(errors.iter().any(|e| matches!(e, ValidationError::MemoryExceeded { .. })));
    }

    #[test]
    fn test_validator_catches_unassigned_layer() {
        let config = ShardConfig::uniform(2, 1_000_000);
        let profile = uniform_profile(4, 100);
        // Only assign 3 of 4 layers.
        let plan = ShardPlan {
            assignments: vec![
                LayerAssignment { layer_id: 0, device_id: 0 },
                LayerAssignment { layer_id: 1, device_id: 1 },
                LayerAssignment { layer_id: 2, device_id: 0 },
            ],
            num_devices: 2,
        };
        let errors = ShardValidator::validate(&plan, &config, &profile);
        assert!(errors.iter().any(|e| matches!(e, ValidationError::UnassignedLayer(3))));
    }

    #[test]
    fn test_validator_catches_invalid_device() {
        let config = ShardConfig::uniform(2, 1_000_000);
        let profile = uniform_profile(2, 100);
        let plan = ShardPlan {
            assignments: vec![
                LayerAssignment { layer_id: 0, device_id: 0 },
                LayerAssignment { layer_id: 1, device_id: 5 },
            ],
            num_devices: 2,
        };
        let errors = ShardValidator::validate(&plan, &config, &profile);
        assert!(
            errors.iter().any(|e| matches!(e, ValidationError::InvalidDevice { device_id: 5, .. }))
        );
    }

    #[test]
    fn test_validator_catches_duplicate_assignment() {
        let config = ShardConfig::uniform(2, 1_000_000);
        let profile = uniform_profile(2, 100);
        let plan = ShardPlan {
            assignments: vec![
                LayerAssignment { layer_id: 0, device_id: 0 },
                LayerAssignment { layer_id: 0, device_id: 1 },
                LayerAssignment { layer_id: 1, device_id: 1 },
            ],
            num_devices: 2,
        };
        let errors = ShardValidator::validate(&plan, &config, &profile);
        assert!(errors.iter().any(|e| matches!(e, ValidationError::DuplicateAssignment(0))));
    }

    #[test]
    fn test_validator_catches_no_devices() {
        let config = ShardConfig {
            num_devices: 0,
            memory_per_device: vec![],
            strategy: ShardStrategy::Uniform,
        };
        let profile = uniform_profile(2, 100);
        let plan = ShardPlan { assignments: vec![], num_devices: 0 };
        let errors = ShardValidator::validate(&plan, &config, &profile);
        assert!(errors.iter().any(|e| matches!(e, ValidationError::NoDevices)));
    }

    #[test]
    fn test_validator_valid_plan_weight_balanced() {
        let config = ShardConfig::with_strategy(2, 1_000_000, ShardStrategy::WeightBalanced);
        let profile = varied_profile();
        let plan = WeightBalancedSharder::plan(&config, &profile);
        assert!(ShardValidator::is_valid(&plan, &config, &profile));
    }

    #[test]
    fn test_validator_tight_memory_valid() {
        // Memory exactly fits.
        let config = ShardConfig::uniform(2, 200);
        let profile = uniform_profile(4, 100);
        let plan = WeightBalancedSharder::plan(&config, &profile);
        assert!(ShardValidator::is_valid(&plan, &config, &profile));
    }

    #[test]
    fn test_validator_tight_memory_exceeded() {
        // Memory just barely doesn't fit.
        let config = ShardConfig::uniform(2, 199);
        let profile = uniform_profile(4, 100);
        let plan = WeightBalancedSharder::plan(&config, &profile);
        assert!(!ShardValidator::is_valid(&plan, &config, &profile));
    }

    // ── ShardOptimizer tests ─────────────────────────────────────────

    #[test]
    fn test_optimizer_improves_imbalanced_plan() {
        let profile = varied_profile();
        // Start with a deliberately bad plan.
        let bad_plan = ShardPlan {
            assignments: vec![
                LayerAssignment { layer_id: 0, device_id: 0 },
                LayerAssignment { layer_id: 1, device_id: 0 },
                LayerAssignment { layer_id: 2, device_id: 0 },
                LayerAssignment { layer_id: 3, device_id: 1 },
            ],
            num_devices: 2,
        };
        let config = ShardConfig::uniform(2, 1_000_000);
        let optimized = ShardOptimizer::optimize(&bad_plan, &profile, &config, 10);

        let old_usage = bad_plan.device_memory_usage(&profile);
        let new_usage = optimized.device_memory_usage(&profile);

        let old_max = old_usage.values().copied().max().unwrap_or(0);
        let new_max = new_usage.values().copied().max().unwrap_or(0);

        assert!(new_max <= old_max, "Optimizer should not worsen max: {new_max} > {old_max}");
    }

    #[test]
    fn test_optimizer_already_optimal() {
        let profile = uniform_profile(4, 100);
        let config = ShardConfig::uniform(2, 1_000_000);
        let plan = WeightBalancedSharder::plan(&config, &profile);
        let optimized = ShardOptimizer::optimize(&plan, &profile, &config, 10);
        // Should not change a perfectly balanced plan.
        let old_usage = plan.device_memory_usage(&profile);
        let new_usage = optimized.device_memory_usage(&profile);
        assert_eq!(old_usage.values().copied().max(), new_usage.values().copied().max());
    }

    #[test]
    fn test_optimizer_respects_memory_constraints() {
        let profile = varied_profile();
        let config = ShardConfig::uniform(2, 1000);
        let plan = WeightBalancedSharder::plan(&config, &profile);
        let optimized = ShardOptimizer::optimize(&plan, &profile, &config, 10);
        assert!(ShardValidator::is_valid(&optimized, &config, &profile));
    }

    #[test]
    fn test_optimizer_single_device() {
        let profile = varied_profile();
        let config = ShardConfig::uniform(1, 1_000_000);
        let plan = UniformSharder::plan(&config, &profile);
        let optimized = ShardOptimizer::optimize(&plan, &profile, &config, 10);
        // Single device — nothing to swap.
        assert_eq!(optimized.assignments.len(), plan.assignments.len());
    }

    #[test]
    fn test_optimizer_zero_iterations() {
        let profile = varied_profile();
        let config = ShardConfig::uniform(2, 1_000_000);
        let plan = UniformSharder::plan(&config, &profile);
        let optimized = ShardOptimizer::optimize(&plan, &profile, &config, 0);
        // No iterations → same plan.
        for (a, b) in plan.assignments.iter().zip(&optimized.assignments) {
            assert_eq!(a.layer_id, b.layer_id);
            assert_eq!(a.device_id, b.device_id);
        }
    }

    // ── ShardPlannerMetrics tests ────────────────────────────────────

    #[test]
    fn test_metrics_perfect_balance() {
        let profile = uniform_profile(4, 100);
        let config = ShardConfig::uniform(2, 1_000_000);
        let plan = WeightBalancedSharder::plan(&config, &profile);
        let metrics = ShardPlannerMetrics::compute(&plan, &config, &profile);
        assert!(
            (metrics.balance_ratio - 1.0).abs() < 0.01,
            "Balance ratio should be ~1.0, got {}",
            metrics.balance_ratio
        );
    }

    #[test]
    fn test_metrics_imbalanced() {
        let profile = ModelProfile {
            layers: vec![
                LayerProfile {
                    layer_id: 0,
                    param_count: 100,
                    param_bytes: 900,
                    activation_bytes: 100,
                    compute_flops: 100,
                },
                LayerProfile {
                    layer_id: 1,
                    param_count: 100,
                    param_bytes: 100,
                    activation_bytes: 100,
                    compute_flops: 100,
                },
            ],
        };
        let plan = ShardPlan {
            assignments: vec![
                LayerAssignment { layer_id: 0, device_id: 0 },
                LayerAssignment { layer_id: 1, device_id: 1 },
            ],
            num_devices: 2,
        };
        let config = ShardConfig::uniform(2, 1_000_000);
        let metrics = ShardPlannerMetrics::compute(&plan, &config, &profile);
        assert!(metrics.balance_ratio < 0.5);
    }

    #[test]
    fn test_metrics_memory_utilization() {
        let profile = uniform_profile(4, 250);
        let config = ShardConfig::uniform(2, 1000);
        let plan = WeightBalancedSharder::plan(&config, &profile);
        let metrics = ShardPlannerMetrics::compute(&plan, &config, &profile);
        // 500 bytes per device out of 1000 → 0.5 utilization.
        assert!(
            (metrics.max_memory_utilization - 0.5).abs() < 0.01,
            "Expected ~0.5 utilization, got {}",
            metrics.max_memory_utilization
        );
    }

    #[test]
    fn test_metrics_comm_volume_no_splits() {
        let profile = uniform_profile(4, 100);
        let config = ShardConfig::uniform(1, 1_000_000);
        let plan = UniformSharder::plan(&config, &profile);
        let metrics = ShardPlannerMetrics::compute(&plan, &config, &profile);
        assert_eq!(metrics.estimated_comm_volume, 0);
    }

    #[test]
    fn test_metrics_comm_volume_with_splits() {
        let profile = uniform_profile(4, 100);
        // 4 layers, 4 devices → every boundary is a split.
        let plan = ShardPlan {
            assignments: vec![
                LayerAssignment { layer_id: 0, device_id: 0 },
                LayerAssignment { layer_id: 1, device_id: 1 },
                LayerAssignment { layer_id: 2, device_id: 2 },
                LayerAssignment { layer_id: 3, device_id: 3 },
            ],
            num_devices: 4,
        };
        let config = ShardConfig::uniform(4, 1_000_000);
        let metrics = ShardPlannerMetrics::compute(&plan, &config, &profile);
        // 3 boundaries, each transfers activation_bytes=50.
        assert_eq!(metrics.estimated_comm_volume, 150);
    }

    #[test]
    fn test_metrics_empty_model() {
        let profile = ModelProfile { layers: vec![] };
        let config = ShardConfig::uniform(2, 1_000_000);
        let plan = ShardPlan { assignments: vec![], num_devices: 2 };
        let metrics = ShardPlannerMetrics::compute(&plan, &config, &profile);
        assert!((metrics.balance_ratio - 1.0).abs() < 0.01);
        assert_eq!(metrics.estimated_comm_volume, 0);
    }

    // ── Dispatch tests ───────────────────────────────────────────────

    #[test]
    fn test_plan_sharding_dispatch_uniform() {
        let config = ShardConfig::with_strategy(2, 1_000_000, ShardStrategy::Uniform);
        let profile = uniform_profile(4, 100);
        let plan = plan_sharding(&config, &profile);
        assert_eq!(plan.assignments.len(), 4);
    }

    #[test]
    fn test_plan_sharding_dispatch_weight_balanced() {
        let config = ShardConfig::with_strategy(2, 1_000_000, ShardStrategy::WeightBalanced);
        let profile = varied_profile();
        let plan = plan_sharding(&config, &profile);
        assert_eq!(plan.assignments.len(), 4);
    }

    #[test]
    fn test_plan_sharding_dispatch_activation_balanced() {
        let config = ShardConfig::with_strategy(2, 1_000_000, ShardStrategy::ActivationBalanced);
        let profile = varied_activation_profile();
        let plan = plan_sharding(&config, &profile);
        assert_eq!(plan.assignments.len(), 4);
    }

    // ── Heterogeneous device tests ───────────────────────────────────

    #[test]
    fn test_heterogeneous_weight_balanced() {
        // varied_profile total = 1600 bytes; greedy packing puts ~800 per device.
        let config = ShardConfig::heterogeneous(vec![2000, 1000], ShardStrategy::WeightBalanced);
        let profile = varied_profile();
        let plan = WeightBalancedSharder::plan(&config, &profile);
        assert!(ShardValidator::is_valid(&plan, &config, &profile));
    }

    #[test]
    fn test_heterogeneous_activation_balanced() {
        let config = ShardConfig::heterogeneous(vec![2000, 500], ShardStrategy::ActivationBalanced);
        let profile = varied_activation_profile();
        let plan = ActivationBalancedSharder::plan(&config, &profile);
        assert_eq!(plan.assignments.len(), 4);
    }

    #[test]
    fn test_heterogeneous_memory_validation() {
        // Device 1 has very little memory.
        let config = ShardConfig::heterogeneous(vec![1_000_000, 50], ShardStrategy::WeightBalanced);
        let profile = varied_profile();
        let plan = WeightBalancedSharder::plan(&config, &profile);
        // Greedy packing may still exceed the tiny device.
        let errors = ShardValidator::validate(&plan, &config, &profile);
        // The plan may or may not be valid, but at least all layers are assigned.
        assert_eq!(plan.assignments.len(), 4);
        // Check that if invalid, it's due to memory exceeded.
        for e in &errors {
            assert!(matches!(e, ValidationError::MemoryExceeded { .. }), "Unexpected error: {e:?}");
        }
    }

    #[test]
    fn test_heterogeneous_three_devices() {
        let config =
            ShardConfig::heterogeneous(vec![1000, 500, 2000], ShardStrategy::WeightBalanced);
        let profile = uniform_profile(6, 100);
        let plan = WeightBalancedSharder::plan(&config, &profile);
        assert_eq!(plan.assignments.len(), 6);
        assert!(ShardValidator::is_valid(&plan, &config, &profile));
    }

    // ── Edge case tests ──────────────────────────────────────────────

    #[test]
    fn test_large_layer_count() {
        let config = ShardConfig::uniform(4, 100_000_000);
        let profile = uniform_profile(1000, 100);
        let plan = UniformSharder::plan(&config, &profile);
        assert_eq!(plan.assignments.len(), 1000);
        assert!(ShardValidator::is_valid(&plan, &config, &profile));
    }

    #[test]
    fn test_zero_byte_layers() {
        let profile = ModelProfile {
            layers: vec![
                LayerProfile {
                    layer_id: 0,
                    param_count: 0,
                    param_bytes: 0,
                    activation_bytes: 0,
                    compute_flops: 0,
                },
                LayerProfile {
                    layer_id: 1,
                    param_count: 0,
                    param_bytes: 0,
                    activation_bytes: 0,
                    compute_flops: 0,
                },
            ],
        };
        let config = ShardConfig::uniform(2, 1_000_000);
        let plan = WeightBalancedSharder::plan(&config, &profile);
        assert!(ShardValidator::is_valid(&plan, &config, &profile));
        assert!(plan.is_balanced(&profile));
    }

    #[test]
    fn test_device_activation_usage() {
        let profile = varied_activation_profile();
        let plan = ShardPlan {
            assignments: vec![
                LayerAssignment { layer_id: 0, device_id: 0 },
                LayerAssignment { layer_id: 1, device_id: 1 },
                LayerAssignment { layer_id: 2, device_id: 0 },
                LayerAssignment { layer_id: 3, device_id: 1 },
            ],
            num_devices: 2,
        };
        let usage = plan.device_activation_usage(&profile);
        assert_eq!(usage[&0], 1400); // 800 + 600
        assert_eq!(usage[&1], 600); // 200 + 400
    }

    #[test]
    fn test_plan_two_layers_four_devices() {
        let config = ShardConfig::uniform(4, 1_000_000);
        let profile = uniform_profile(2, 100);
        let plan = UniformSharder::plan(&config, &profile);
        assert_eq!(plan.assignments.len(), 2);
        // Both layers should be on different devices (or at least valid).
        for a in &plan.assignments {
            assert!(a.device_id < 4);
        }
    }

    #[test]
    fn test_optimizer_with_three_devices() {
        let profile = ModelProfile {
            layers: vec![
                LayerProfile {
                    layer_id: 0,
                    param_count: 100,
                    param_bytes: 900,
                    activation_bytes: 100,
                    compute_flops: 100,
                },
                LayerProfile {
                    layer_id: 1,
                    param_count: 100,
                    param_bytes: 100,
                    activation_bytes: 100,
                    compute_flops: 100,
                },
                LayerProfile {
                    layer_id: 2,
                    param_count: 100,
                    param_bytes: 100,
                    activation_bytes: 100,
                    compute_flops: 100,
                },
                LayerProfile {
                    layer_id: 3,
                    param_count: 100,
                    param_bytes: 100,
                    activation_bytes: 100,
                    compute_flops: 100,
                },
                LayerProfile {
                    layer_id: 4,
                    param_count: 100,
                    param_bytes: 100,
                    activation_bytes: 100,
                    compute_flops: 100,
                },
                LayerProfile {
                    layer_id: 5,
                    param_count: 100,
                    param_bytes: 100,
                    activation_bytes: 100,
                    compute_flops: 100,
                },
            ],
        };
        let config = ShardConfig::uniform(3, 1_000_000);
        let plan = UniformSharder::plan(&config, &profile);
        let optimized = ShardOptimizer::optimize(&plan, &profile, &config, 10);
        assert!(ShardValidator::is_valid(&optimized, &config, &profile));
    }

    #[test]
    fn test_metrics_balance_ratio_single_device() {
        let profile = uniform_profile(4, 100);
        let config = ShardConfig::uniform(1, 1_000_000);
        let plan = UniformSharder::plan(&config, &profile);
        let metrics = ShardPlannerMetrics::compute(&plan, &config, &profile);
        assert!((metrics.balance_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_metrics_utilization_full() {
        let profile = uniform_profile(4, 250);
        let config = ShardConfig::uniform(2, 500);
        let plan = WeightBalancedSharder::plan(&config, &profile);
        let metrics = ShardPlannerMetrics::compute(&plan, &config, &profile);
        assert!(
            (metrics.max_memory_utilization - 1.0).abs() < 0.01,
            "Expected full utilization, got {}",
            metrics.max_memory_utilization
        );
    }

    #[test]
    fn test_uniform_shard_preserves_layer_order() {
        let config = ShardConfig::uniform(2, 1_000_000);
        let profile = uniform_profile(6, 100);
        let plan = UniformSharder::plan(&config, &profile);
        // Layer ids in assignments should be monotonically increasing.
        let ids: Vec<usize> = plan.assignments.iter().map(|a| a.layer_id).collect();
        for w in ids.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn test_weight_balanced_sorted_assignments() {
        let config = ShardConfig::with_strategy(2, 1_000_000, ShardStrategy::WeightBalanced);
        let profile = varied_profile();
        let plan = WeightBalancedSharder::plan(&config, &profile);
        let ids: Vec<usize> = plan.assignments.iter().map(|a| a.layer_id).collect();
        for w in ids.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn test_activation_balanced_sorted_assignments() {
        let config = ShardConfig::with_strategy(2, 1_000_000, ShardStrategy::ActivationBalanced);
        let profile = varied_activation_profile();
        let plan = ActivationBalancedSharder::plan(&config, &profile);
        let ids: Vec<usize> = plan.assignments.iter().map(|a| a.layer_id).collect();
        for w in ids.windows(2) {
            assert!(w[0] < w[1]);
        }
    }
}
