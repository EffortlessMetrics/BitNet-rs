//! Data parallel execution engine for multi-GPU batch processing.
//!
//! Distributes batch inference across multiple GPUs with configurable
//! split strategies and synchronization methods.

// ── Types ─────────────────────────────────────────────────────────────────

/// Information about a single compute device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Zero-based device index.
    pub index: usize,
    /// Human-readable device name.
    pub name: String,
    /// Total device memory in GiB.
    pub memory_gb: f64,
    /// Number of compute units (SMs, CUs, etc.).
    pub compute_units: u32,
}

/// Strategy for splitting a batch across devices.
#[derive(Debug, Clone)]
pub enum SplitStrategy {
    /// Equal share to each device (remainder goes to the first devices).
    EvenSplit,
    /// Proportional to each device's memory capacity.
    ProportionalToMemory,
    /// Proportional to each device's compute-unit count.
    ProportionalToCompute,
    /// Explicit per-device weights (must sum to > 0).
    Custom(Vec<f64>),
}

/// Method used to synchronise results across devices.
#[derive(Debug, Clone)]
pub enum SyncMethod {
    /// Flat all-reduce (cost ∝ 2·(N−1)·T / N).
    AllReduce,
    /// Central parameter-server on the given device.
    ParameterServer { server_device: usize },
    /// Ring all-reduce (cost ∝ 2·(N−1)·T / N, bandwidth-optimal).
    RingAllReduce,
    /// Butterfly / recursive-halving all-reduce (cost ∝ log₂(N)·T).
    ButterflyAllReduce,
}

/// Configuration for the data-parallel engine.
#[derive(Debug, Clone)]
pub struct DpConfig {
    /// Number of devices to use.
    pub num_devices: usize,
    /// How to split batches across devices.
    pub batch_split_strategy: SplitStrategy,
    /// How to synchronise after each forward pass.
    pub sync_method: SyncMethod,
    /// Whether to accumulate gradients across micro-batches.
    pub gradient_accumulation: bool,
}

/// One slice of a batch assigned to a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchSplit {
    /// Which device owns this slice.
    pub device_index: usize,
    /// First element (inclusive).
    pub start_index: usize,
    /// One-past-the-last element.
    pub end_index: usize,
    /// Number of elements.
    pub size: usize,
}

/// Result of splitting a full batch.
#[derive(Debug, Clone)]
pub struct DpResult {
    /// Per-device splits.
    pub splits: Vec<BatchSplit>,
    /// Total batch size.
    pub total_batch_size: usize,
    /// Estimated synchronisation overhead in ms.
    pub sync_overhead_estimate_ms: f64,
}

/// Cumulative statistics for the engine.
#[derive(Debug, Clone)]
pub struct DpStats {
    /// Batches processed so far.
    pub total_batches: u64,
    /// Running average of split imbalance (max/min ratio − 1).
    pub avg_split_imbalance: f64,
    /// Total synchronisation time in ms.
    pub total_sync_time_ms: f64,
    /// Per-device utilisation in [0, 1].
    pub device_utilization: Vec<f64>,
}

// ── Engine ─────────────────────────────────────────────────────────────────

/// Data-parallel engine that distributes batch work across devices.
pub struct DataParallelEngine {
    devices: Vec<DeviceInfo>,
    config: DpConfig,
    stats: DpStats,
    /// Per-device measured times for adaptive rebalancing.
    device_times: Vec<Vec<f64>>,
    /// Adaptive weights computed by `rebalance()`.
    adaptive_weights: Option<Vec<f64>>,
}

impl DataParallelEngine {
    /// Create a new engine from a list of devices and configuration.
    ///
    /// # Panics
    /// Panics if `devices` is empty or if `config.num_devices` does not
    /// match `devices.len()`.
    pub fn new(devices: Vec<DeviceInfo>, config: DpConfig) -> Self {
        assert!(!devices.is_empty(), "at least one device is required");
        assert_eq!(
            devices.len(),
            config.num_devices,
            "device count must match config.num_devices"
        );
        let n = devices.len();
        Self {
            devices,
            config,
            stats: DpStats {
                total_batches: 0,
                avg_split_imbalance: 0.0,
                total_sync_time_ms: 0.0,
                device_utilization: vec![0.0; n],
            },
            device_times: vec![Vec::new(); n],
            adaptive_weights: None,
        }
    }

    /// Split `batch_size` items across the configured devices.
    pub fn split_batch(&mut self, batch_size: usize) -> DpResult {
        let n = self.devices.len();

        let sizes = if let Some(ref weights) = self.adaptive_weights {
            weighted_split(batch_size, weights)
        } else {
            match &self.config.batch_split_strategy {
                SplitStrategy::EvenSplit => even_split(batch_size, n),
                SplitStrategy::ProportionalToMemory => {
                    let weights: Vec<f64> =
                        self.devices.iter().map(|d| d.memory_gb).collect();
                    weighted_split(batch_size, &weights)
                }
                SplitStrategy::ProportionalToCompute => {
                    let weights: Vec<f64> =
                        self.devices.iter().map(|d| f64::from(d.compute_units))
                            .collect();
                    weighted_split(batch_size, &weights)
                }
                SplitStrategy::Custom(w) => weighted_split(batch_size, w),
            }
        };

        let splits = build_splits(&sizes);

        let sync_est = self.estimate_sync_cost(batch_size, 1024);
        self.stats.total_batches += 1;
        self.update_imbalance(&sizes);

        DpResult { splits, total_batch_size: batch_size, sync_overhead_estimate_ms: sync_est }
    }

    /// Estimate synchronisation cost in ms for the configured method.
    ///
    /// `tensor_size` is the number of elements in the tensor being reduced.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_sync_cost(&self, _batch_size: usize, tensor_size: usize) -> f64 {
        let n = self.devices.len() as f64;
        let t = tensor_size as f64;
        // Base latency per element (arbitrary unit for estimation).
        let base_latency_per_elem = 0.001; // ms per element
        match &self.config.sync_method {
            SyncMethod::AllReduce => {
                // 2·(N-1)/N · T
                2.0 * (n - 1.0) / n * t * base_latency_per_elem
            }
            SyncMethod::ParameterServer { .. } => {
                // 2·T (all send + all receive, through one server)
                2.0 * t * base_latency_per_elem
            }
            SyncMethod::RingAllReduce => {
                // 2·(N-1)/N · T (bandwidth-optimal)
                2.0 * (n - 1.0) / n * t * base_latency_per_elem
            }
            SyncMethod::ButterflyAllReduce => {
                // log2(N) · T
                n.log2() * t * base_latency_per_elem
            }
        }
    }

    /// Heuristic for the largest batch that fits in a single device's memory.
    ///
    /// Uses a rough estimate of 2 MB per sample.
    pub fn optimal_batch_size(&self, per_device_memory_gb: f64) -> usize {
        const MB_PER_SAMPLE: f64 = 2.0;
        let total_mb = per_device_memory_gb * 1024.0;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let max_samples = (total_mb / MB_PER_SAMPLE) as usize;
        // Round down to nearest multiple of device count for even splits.
        let n = self.devices.len();
        (max_samples / n) * n
    }

    /// Record the measured forward-pass time for a device (in ms).
    pub fn report_device_time(&mut self, device_index: usize, time_ms: f64) {
        if let Some(times) = self.device_times.get_mut(device_index) {
            times.push(time_ms);
        }
        // Update utilization.
        self.update_utilization();
    }

    /// Rebalance split weights based on measured device times.
    ///
    /// Devices that run faster get a larger share.
    pub fn rebalance(&mut self) {
        let means: Vec<f64> = self
            .device_times
            .iter()
            .map(|times| {
                if times.is_empty() {
                    1.0
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    let mean = times.iter().sum::<f64>() / times.len() as f64;
                    mean
                }
            })
            .collect();

        // Inverse-time weighting: faster device → higher weight.
        let inv: Vec<f64> = means.iter().map(|&m| 1.0 / m.max(f64::EPSILON)).collect();
        let sum: f64 = inv.iter().sum();
        let weights: Vec<f64> = inv.iter().map(|&w| w / sum).collect();
        self.adaptive_weights = Some(weights);
    }

    /// Return a snapshot of the current statistics.
    pub const fn stats(&self) -> &DpStats {
        &self.stats
    }

    // ── internal helpers ──────────────────────────────────────────────

    fn update_imbalance(&mut self, sizes: &[usize]) {
        let min = sizes.iter().copied().filter(|&s| s > 0).min().unwrap_or(1);
        let max = sizes.iter().copied().max().unwrap_or(1);
        #[allow(clippy::cast_precision_loss)]
        let imbalance = if min == 0 {
            0.0
        } else {
            max as f64 / min as f64 - 1.0
        };
        #[allow(clippy::cast_precision_loss)]
        let n = self.stats.total_batches as f64;
        // Running average.
        self.stats.avg_split_imbalance =
            self.stats.avg_split_imbalance.mul_add((n - 1.0) / n, imbalance / n);
    }

    fn update_utilization(&mut self) {
        let max_avg = self
            .device_times
            .iter()
            .map(|t| {
                if t.is_empty() {
                    0.0
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    let avg = t.iter().sum::<f64>() / t.len() as f64;
                    avg
                }
            })
            .fold(0.0_f64, f64::max);

        if max_avg <= 0.0 {
            return;
        }

        for (i, times) in self.device_times.iter().enumerate() {
            if times.is_empty() {
                self.stats.device_utilization[i] = 0.0;
            } else {
                #[allow(clippy::cast_precision_loss)]
                let avg = times.iter().sum::<f64>() / times.len() as f64;
                self.stats.device_utilization[i] = avg / max_avg;
            }
        }
    }
}

// ── Free helpers ──────────────────────────────────────────────────────────

/// Split `total` into `n` roughly-equal parts (remainder to first devices).
fn even_split(total: usize, n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    let base = total / n;
    let remainder = total % n;
    (0..n)
        .map(|i| if i < remainder { base + 1 } else { base })
        .collect()
}

/// Split `total` proportionally to `weights`.
fn weighted_split(total: usize, weights: &[f64]) -> Vec<usize> {
    let n = weights.len();
    if n == 0 {
        return vec![];
    }
    let sum: f64 = weights.iter().sum();
    if sum <= 0.0 {
        return even_split(total, n);
    }

    let mut sizes = Vec::with_capacity(n);
    let mut assigned = 0usize;
    for (i, &w) in weights.iter().enumerate() {
        if i == n - 1 {
            sizes.push(total - assigned);
        } else {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            #[allow(clippy::cast_precision_loss)]
            let t = total as f64;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let s = ((w / sum) * t).round() as usize;
            let s = s.min(total - assigned);
            sizes.push(s);
            assigned += s;
        }
    }
    sizes
}

/// Convert a list of per-device sizes into `BatchSplit` objects.
fn build_splits(sizes: &[usize]) -> Vec<BatchSplit> {
    let mut offset = 0;
    sizes
        .iter()
        .enumerate()
        .map(|(i, &size)| {
            let split = BatchSplit {
                device_index: i,
                start_index: offset,
                end_index: offset + size,
                size,
            };
            offset += size;
            split
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    fn make_devices(n: usize) -> Vec<DeviceInfo> {
        (0..n)
            .map(|i| DeviceInfo {
                index: i,
                name: format!("GPU-{i}"),
                memory_gb: 16.0,
                compute_units: 80,
            })
            .collect()
    }

    fn make_heterogeneous_devices() -> Vec<DeviceInfo> {
        vec![
            DeviceInfo {
                index: 0,
                name: "Big-GPU".into(),
                memory_gb: 80.0,
                compute_units: 132,
            },
            DeviceInfo {
                index: 1,
                name: "Small-GPU".into(),
                memory_gb: 16.0,
                compute_units: 40,
            },
        ]
    }

    fn default_config(n: usize) -> DpConfig {
        DpConfig {
            num_devices: n,
            batch_split_strategy: SplitStrategy::EvenSplit,
            sync_method: SyncMethod::AllReduce,
            gradient_accumulation: false,
        }
    }

    fn engine(n: usize) -> DataParallelEngine {
        DataParallelEngine::new(make_devices(n), default_config(n))
    }

    // ── EvenSplit ─────────────────────────────────────────────────────

    #[test]
    fn even_split_32_across_4() {
        let mut e = engine(4);
        let r = e.split_batch(32);
        assert_eq!(r.total_batch_size, 32);
        for s in &r.splits {
            assert_eq!(s.size, 8);
        }
    }

    #[test]
    fn even_split_33_across_4() {
        let mut e = engine(4);
        let r = e.split_batch(33);
        let sizes: Vec<usize> = r.splits.iter().map(|s| s.size).collect();
        assert_eq!(sizes, vec![9, 8, 8, 8]);
    }

    #[test]
    fn even_split_1_across_4() {
        let mut e = engine(4);
        let r = e.split_batch(1);
        let sizes: Vec<usize> = r.splits.iter().map(|s| s.size).collect();
        assert_eq!(sizes, vec![1, 0, 0, 0]);
    }

    #[test]
    fn even_split_0_across_4() {
        let mut e = engine(4);
        let r = e.split_batch(0);
        let sizes: Vec<usize> = r.splits.iter().map(|s| s.size).collect();
        assert_eq!(sizes, vec![0, 0, 0, 0]);
    }

    #[test]
    fn even_split_100_across_3() {
        let mut e = engine(3);
        let r = e.split_batch(100);
        let sizes: Vec<usize> = r.splits.iter().map(|s| s.size).collect();
        assert_eq!(sizes, vec![34, 33, 33]);
    }

    #[test]
    fn even_split_7_across_2() {
        let mut e = engine(2);
        let r = e.split_batch(7);
        let sizes: Vec<usize> = r.splits.iter().map(|s| s.size).collect();
        assert_eq!(sizes, vec![4, 3]);
    }

    // ── Single device ─────────────────────────────────────────────────

    #[test]
    fn single_device_no_split() {
        let mut e = engine(1);
        let r = e.split_batch(64);
        assert_eq!(r.splits.len(), 1);
        assert_eq!(r.splits[0].size, 64);
        assert_eq!(r.splits[0].start_index, 0);
        assert_eq!(r.splits[0].end_index, 64);
    }

    // ── ProportionalToMemory ──────────────────────────────────────────

    #[test]
    fn proportional_to_memory_bigger_gets_more() {
        let devices = make_heterogeneous_devices();
        let config = DpConfig {
            num_devices: 2,
            batch_split_strategy: SplitStrategy::ProportionalToMemory,
            sync_method: SyncMethod::AllReduce,
            gradient_accumulation: false,
        };
        let mut e = DataParallelEngine::new(devices, config);
        let r = e.split_batch(96);
        assert!(r.splits[0].size > r.splits[1].size);
        assert_eq!(
            r.splits.iter().map(|s| s.size).sum::<usize>(),
            96
        );
    }

    #[test]
    fn proportional_to_memory_ratio() {
        let devices = make_heterogeneous_devices(); // 80 + 16 = 96 total
        let config = DpConfig {
            num_devices: 2,
            batch_split_strategy: SplitStrategy::ProportionalToMemory,
            sync_method: SyncMethod::AllReduce,
            gradient_accumulation: false,
        };
        let mut e = DataParallelEngine::new(devices, config);
        let r = e.split_batch(96);
        // 80/96 ≈ 0.833 → 80 samples, 16/96 ≈ 0.167 → 16 samples
        assert_eq!(r.splits[0].size, 80);
        assert_eq!(r.splits[1].size, 16);
    }

    // ── ProportionalToCompute ─────────────────────────────────────────

    #[test]
    fn proportional_to_compute_faster_gets_more() {
        let devices = make_heterogeneous_devices();
        let config = DpConfig {
            num_devices: 2,
            batch_split_strategy: SplitStrategy::ProportionalToCompute,
            sync_method: SyncMethod::AllReduce,
            gradient_accumulation: false,
        };
        let mut e = DataParallelEngine::new(devices, config);
        let r = e.split_batch(172);
        // 132/(132+40) = 0.767 → ~132, 40/(172) ≈ 0.233 → ~40
        assert!(r.splits[0].size > r.splits[1].size);
        assert_eq!(
            r.splits.iter().map(|s| s.size).sum::<usize>(),
            172
        );
    }

    #[test]
    fn proportional_to_compute_exact() {
        let devices = vec![
            DeviceInfo {
                index: 0,
                name: "A".into(),
                memory_gb: 16.0,
                compute_units: 75,
            },
            DeviceInfo {
                index: 1,
                name: "B".into(),
                memory_gb: 16.0,
                compute_units: 25,
            },
        ];
        let config = DpConfig {
            num_devices: 2,
            batch_split_strategy: SplitStrategy::ProportionalToCompute,
            sync_method: SyncMethod::AllReduce,
            gradient_accumulation: false,
        };
        let mut e = DataParallelEngine::new(devices, config);
        let r = e.split_batch(100);
        assert_eq!(r.splits[0].size, 75);
        assert_eq!(r.splits[1].size, 25);
    }

    // ── Custom weights ────────────────────────────────────────────────

    #[test]
    fn custom_weights_50_50() {
        let config = DpConfig {
            num_devices: 2,
            batch_split_strategy: SplitStrategy::Custom(vec![1.0, 1.0]),
            sync_method: SyncMethod::AllReduce,
            gradient_accumulation: false,
        };
        let mut e = DataParallelEngine::new(make_devices(2), config);
        let r = e.split_batch(100);
        assert_eq!(r.splits[0].size, 50);
        assert_eq!(r.splits[1].size, 50);
    }

    #[test]
    fn custom_weights_70_30() {
        let config = DpConfig {
            num_devices: 2,
            batch_split_strategy: SplitStrategy::Custom(vec![7.0, 3.0]),
            sync_method: SyncMethod::AllReduce,
            gradient_accumulation: false,
        };
        let mut e = DataParallelEngine::new(make_devices(2), config);
        let r = e.split_batch(100);
        assert_eq!(r.splits[0].size, 70);
        assert_eq!(r.splits[1].size, 30);
    }

    #[test]
    fn custom_weights_three_devices() {
        let config = DpConfig {
            num_devices: 3,
            batch_split_strategy: SplitStrategy::Custom(vec![2.0, 1.0, 1.0]),
            sync_method: SyncMethod::AllReduce,
            gradient_accumulation: false,
        };
        let mut e = DataParallelEngine::new(make_devices(3), config);
        let r = e.split_batch(100);
        assert_eq!(r.splits[0].size, 50);
        let total: usize = r.splits.iter().map(|s| s.size).sum();
        assert_eq!(total, 100);
    }

    // ── Index contiguity and coverage ─────────────────────────────────

    #[test]
    fn splits_are_contiguous_even() {
        let mut e = engine(4);
        let r = e.split_batch(32);
        for i in 1..r.splits.len() {
            assert_eq!(r.splits[i].start_index, r.splits[i - 1].end_index);
        }
    }

    #[test]
    fn splits_are_contiguous_uneven() {
        let mut e = engine(3);
        let r = e.split_batch(100);
        for i in 1..r.splits.len() {
            assert_eq!(r.splits[i].start_index, r.splits[i - 1].end_index);
        }
    }

    #[test]
    fn splits_cover_full_batch() {
        let mut e = engine(4);
        let r = e.split_batch(37);
        assert_eq!(r.splits.first().unwrap().start_index, 0);
        assert_eq!(r.splits.last().unwrap().end_index, 37);
    }

    #[test]
    fn total_sizes_equal_batch_even() {
        let mut e = engine(4);
        let r = e.split_batch(32);
        let total: usize = r.splits.iter().map(|s| s.size).sum();
        assert_eq!(total, 32);
    }

    #[test]
    fn total_sizes_equal_batch_uneven() {
        let mut e = engine(4);
        let r = e.split_batch(33);
        let total: usize = r.splits.iter().map(|s| s.size).sum();
        assert_eq!(total, 33);
    }

    #[test]
    fn total_sizes_equal_batch_proportional() {
        let devices = make_heterogeneous_devices();
        let config = DpConfig {
            num_devices: 2,
            batch_split_strategy: SplitStrategy::ProportionalToMemory,
            sync_method: SyncMethod::AllReduce,
            gradient_accumulation: false,
        };
        let mut e = DataParallelEngine::new(devices, config);
        let r = e.split_batch(100);
        let total: usize = r.splits.iter().map(|s| s.size).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn split_size_matches_start_end() {
        let mut e = engine(4);
        let r = e.split_batch(50);
        for s in &r.splits {
            assert_eq!(s.end_index - s.start_index, s.size);
        }
    }

    #[test]
    fn no_overlapping_splits() {
        let mut e = engine(4);
        let r = e.split_batch(50);
        for i in 0..r.splits.len() {
            for j in (i + 1)..r.splits.len() {
                assert!(
                    r.splits[i].end_index <= r.splits[j].start_index
                        || r.splits[j].end_index <= r.splits[i].start_index
                );
            }
        }
    }

    // ── Empty batch ───────────────────────────────────────────────────

    #[test]
    fn empty_batch_all_zero() {
        let mut e = engine(4);
        let r = e.split_batch(0);
        assert_eq!(r.total_batch_size, 0);
        for s in &r.splits {
            assert_eq!(s.size, 0);
        }
    }

    // ── Sync cost estimation ──────────────────────────────────────────

    #[test]
    fn allreduce_sync_cost_positive() {
        let e = engine(4);
        let cost = e.estimate_sync_cost(32, 1024);
        assert!(cost > 0.0);
    }

    #[test]
    fn allreduce_sync_cost_scales_with_tensor() {
        let e = engine(4);
        let small = e.estimate_sync_cost(32, 512);
        let large = e.estimate_sync_cost(32, 2048);
        assert!(large > small);
    }

    #[test]
    fn ring_allreduce_cost() {
        let config = DpConfig {
            num_devices: 4,
            batch_split_strategy: SplitStrategy::EvenSplit,
            sync_method: SyncMethod::RingAllReduce,
            gradient_accumulation: false,
        };
        let e = DataParallelEngine::new(make_devices(4), config);
        let cost = e.estimate_sync_cost(32, 1024);
        assert!(cost > 0.0);
    }

    #[test]
    fn ring_cost_proportional_to_tensor_size() {
        let config = DpConfig {
            num_devices: 4,
            batch_split_strategy: SplitStrategy::EvenSplit,
            sync_method: SyncMethod::RingAllReduce,
            gradient_accumulation: false,
        };
        let e = DataParallelEngine::new(make_devices(4), config);
        let c1 = e.estimate_sync_cost(32, 1000);
        let c2 = e.estimate_sync_cost(32, 2000);
        let ratio = c2 / c1;
        assert!((ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn butterfly_cost() {
        let config = DpConfig {
            num_devices: 8,
            batch_split_strategy: SplitStrategy::EvenSplit,
            sync_method: SyncMethod::ButterflyAllReduce,
            gradient_accumulation: false,
        };
        let e = DataParallelEngine::new(make_devices(8), config);
        let cost = e.estimate_sync_cost(32, 1024);
        assert!(cost > 0.0);
    }

    #[test]
    fn butterfly_cost_log_scaling() {
        let config4 = DpConfig {
            num_devices: 4,
            batch_split_strategy: SplitStrategy::EvenSplit,
            sync_method: SyncMethod::ButterflyAllReduce,
            gradient_accumulation: false,
        };
        let e4 = DataParallelEngine::new(make_devices(4), config4);
        let c4 = e4.estimate_sync_cost(32, 1024);

        let config8 = DpConfig {
            num_devices: 8,
            batch_split_strategy: SplitStrategy::EvenSplit,
            sync_method: SyncMethod::ButterflyAllReduce,
            gradient_accumulation: false,
        };
        let e8 = DataParallelEngine::new(make_devices(8), config8);
        let c8 = e8.estimate_sync_cost(32, 1024);

        // log2(8)/log2(4) = 3/2 = 1.5
        let ratio = c8 / c4;
        assert!((ratio - 1.5).abs() < 0.01);
    }

    #[test]
    fn parameter_server_cost() {
        let config = DpConfig {
            num_devices: 4,
            batch_split_strategy: SplitStrategy::EvenSplit,
            sync_method: SyncMethod::ParameterServer { server_device: 0 },
            gradient_accumulation: false,
        };
        let e = DataParallelEngine::new(make_devices(4), config);
        let cost = e.estimate_sync_cost(32, 1024);
        assert!(cost > 0.0);
    }

    #[test]
    fn parameter_server_cost_independent_of_num_devices() {
        let config2 = DpConfig {
            num_devices: 2,
            batch_split_strategy: SplitStrategy::EvenSplit,
            sync_method: SyncMethod::ParameterServer { server_device: 0 },
            gradient_accumulation: false,
        };
        let e2 = DataParallelEngine::new(make_devices(2), config2);
        let c2 = e2.estimate_sync_cost(32, 1024);

        let config8 = DpConfig {
            num_devices: 8,
            batch_split_strategy: SplitStrategy::EvenSplit,
            sync_method: SyncMethod::ParameterServer { server_device: 0 },
            gradient_accumulation: false,
        };
        let e8 = DataParallelEngine::new(make_devices(8), config8);
        let c8 = e8.estimate_sync_cost(32, 1024);

        assert!((c2 - c8).abs() < f64::EPSILON);
    }

    #[test]
    fn single_device_sync_cost_zero() {
        let e = engine(1);
        let cost = e.estimate_sync_cost(32, 1024);
        assert!((cost - 0.0).abs() < f64::EPSILON);
    }

    // ── Rebalance ─────────────────────────────────────────────────────

    #[test]
    fn rebalance_shifts_to_faster_device() {
        let mut e = engine(2);
        // Device 0 is twice as fast.
        e.report_device_time(0, 10.0);
        e.report_device_time(1, 20.0);
        e.rebalance();
        let r = e.split_batch(100);
        assert!(r.splits[0].size > r.splits[1].size);
    }

    #[test]
    fn rebalance_equal_times_equal_split() {
        let mut e = engine(2);
        e.report_device_time(0, 10.0);
        e.report_device_time(1, 10.0);
        e.rebalance();
        let r = e.split_batch(100);
        assert_eq!(r.splits[0].size, 50);
        assert_eq!(r.splits[1].size, 50);
    }

    #[test]
    fn rebalance_multiple_reports() {
        let mut e = engine(2);
        for _ in 0..5 {
            e.report_device_time(0, 10.0);
            e.report_device_time(1, 30.0);
        }
        e.rebalance();
        let r = e.split_batch(100);
        // Device 0 should get ~75%.
        assert!(r.splits[0].size >= 70);
    }

    #[test]
    fn rebalance_overrides_config_strategy() {
        let config = DpConfig {
            num_devices: 2,
            batch_split_strategy: SplitStrategy::EvenSplit,
            sync_method: SyncMethod::AllReduce,
            gradient_accumulation: false,
        };
        let mut e = DataParallelEngine::new(make_devices(2), config);
        e.report_device_time(0, 5.0);
        e.report_device_time(1, 15.0);
        e.rebalance();
        let r = e.split_batch(100);
        assert!(r.splits[0].size > 50, "rebalance should override EvenSplit");
    }

    // ── Device utilization ────────────────────────────────────────────

    #[test]
    fn utilization_initially_zero() {
        let e = engine(4);
        for &u in &e.stats().device_utilization {
            assert!((u - 0.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn utilization_after_equal_times() {
        let mut e = engine(2);
        e.report_device_time(0, 10.0);
        e.report_device_time(1, 10.0);
        for &u in &e.stats().device_utilization {
            assert!((u - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn utilization_slower_device_is_max() {
        let mut e = engine(2);
        e.report_device_time(0, 10.0);
        e.report_device_time(1, 20.0);
        assert!((e.stats().device_utilization[1] - 1.0).abs() < 0.01);
        assert!(e.stats().device_utilization[0] < 1.0);
    }

    #[test]
    fn utilization_ratio() {
        let mut e = engine(2);
        e.report_device_time(0, 5.0);
        e.report_device_time(1, 10.0);
        let ratio =
            e.stats().device_utilization[0] / e.stats().device_utilization[1];
        assert!((ratio - 0.5).abs() < 0.01);
    }

    // ── Optimal batch size ────────────────────────────────────────────

    #[test]
    fn optimal_batch_size_positive() {
        let e = engine(4);
        let obs = e.optimal_batch_size(16.0);
        assert!(obs > 0);
    }

    #[test]
    fn optimal_batch_size_more_memory_bigger() {
        let e = engine(4);
        let small = e.optimal_batch_size(8.0);
        let large = e.optimal_batch_size(32.0);
        assert!(large > small);
    }

    #[test]
    fn optimal_batch_size_divisible_by_devices() {
        let e = engine(4);
        let obs = e.optimal_batch_size(16.0);
        assert_eq!(obs % 4, 0);
    }

    #[test]
    fn optimal_batch_size_zero_memory() {
        let e = engine(4);
        let obs = e.optimal_batch_size(0.0);
        assert_eq!(obs, 0);
    }

    // ── Stats ─────────────────────────────────────────────────────────

    #[test]
    fn stats_total_batches_increments() {
        let mut e = engine(4);
        assert_eq!(e.stats().total_batches, 0);
        e.split_batch(32);
        assert_eq!(e.stats().total_batches, 1);
        e.split_batch(64);
        assert_eq!(e.stats().total_batches, 2);
    }

    #[test]
    fn stats_imbalance_zero_for_even() {
        let mut e = engine(4);
        e.split_batch(32);
        assert!((e.stats().avg_split_imbalance - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_imbalance_nonzero_for_uneven() {
        let mut e = engine(4);
        e.split_batch(33);
        assert!(e.stats().avg_split_imbalance > 0.0);
    }

    #[test]
    fn stats_device_count_matches() {
        let e = engine(4);
        assert_eq!(e.stats().device_utilization.len(), 4);
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn large_batch_across_many_devices() {
        let mut e = engine(8);
        let r = e.split_batch(10000);
        let total: usize = r.splits.iter().map(|s| s.size).sum();
        assert_eq!(total, 10000);
    }

    #[test]
    fn batch_smaller_than_devices() {
        let mut e = engine(8);
        let r = e.split_batch(3);
        let total: usize = r.splits.iter().map(|s| s.size).sum();
        assert_eq!(total, 3);
        let nonzero: usize =
            r.splits.iter().filter(|s| s.size > 0).count();
        assert_eq!(nonzero, 3);
    }

    #[test]
    fn gradient_accumulation_flag_stored() {
        let config = DpConfig {
            num_devices: 2,
            batch_split_strategy: SplitStrategy::EvenSplit,
            sync_method: SyncMethod::AllReduce,
            gradient_accumulation: true,
        };
        let e = DataParallelEngine::new(make_devices(2), config);
        assert!(e.config.gradient_accumulation);
    }

    #[test]
    fn device_info_preserved() {
        let devices = make_heterogeneous_devices();
        let config = DpConfig {
            num_devices: 2,
            batch_split_strategy: SplitStrategy::EvenSplit,
            sync_method: SyncMethod::AllReduce,
            gradient_accumulation: false,
        };
        let e = DataParallelEngine::new(devices, config);
        assert_eq!(e.devices[0].name, "Big-GPU");
        assert_eq!(e.devices[1].name, "Small-GPU");
    }

    #[test]
    fn multiple_splits_sequential() {
        let mut e = engine(4);
        for batch in [16, 32, 64, 128] {
            let r = e.split_batch(batch);
            let total: usize = r.splits.iter().map(|s| s.size).sum();
            assert_eq!(total, batch);
        }
        assert_eq!(e.stats().total_batches, 4);
    }

    // ── helper unit tests ─────────────────────────────────────────────

    #[test]
    fn even_split_fn_basic() {
        assert_eq!(even_split(10, 3), vec![4, 3, 3]);
        assert_eq!(even_split(9, 3), vec![3, 3, 3]);
        assert_eq!(even_split(0, 3), vec![0, 0, 0]);
    }

    #[test]
    fn weighted_split_fn_basic() {
        let sizes = weighted_split(100, &[1.0, 1.0]);
        assert_eq!(sizes, vec![50, 50]);
    }

    #[test]
    fn weighted_split_fn_unequal() {
        let sizes = weighted_split(100, &[3.0, 1.0]);
        assert_eq!(sizes[0], 75);
        assert_eq!(sizes[1], 25);
    }

    #[test]
    fn weighted_split_fn_zero_weights_fallback() {
        let sizes = weighted_split(100, &[0.0, 0.0]);
        assert_eq!(sizes, vec![50, 50]);
    }

    #[test]
    fn build_splits_fn() {
        let splits = build_splits(&[10, 20, 30]);
        assert_eq!(splits[0].start_index, 0);
        assert_eq!(splits[0].end_index, 10);
        assert_eq!(splits[1].start_index, 10);
        assert_eq!(splits[1].end_index, 30);
        assert_eq!(splits[2].start_index, 30);
        assert_eq!(splits[2].end_index, 60);
    }
}
