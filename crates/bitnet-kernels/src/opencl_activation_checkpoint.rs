//! Activation checkpointing for memory-efficient inference.
//!
//! Trades compute for memory by selectively recomputing activations
//! instead of storing them all. Useful for large models on
//! memory-constrained devices (e.g. Intel A770 with 16 GB VRAM).

use std::collections::HashMap;
use std::fmt;

// ── Types ───────────────────────────────────────────────────────────

/// Strategy for deciding which layers to checkpoint (recompute).
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointStrategy {
    /// Keep all activations — no checkpointing.
    None,
    /// Checkpoint every N-th layer.
    EveryN { n: usize },
    /// Checkpoint only the listed layers.
    SelectiveLayers(Vec<usize>),
    /// Keep as many activations as fit within a memory budget.
    MemoryBudget { max_mb: f64 },
    /// Checkpoint every √N layers (good general-purpose trade-off).
    Sqrt,
}

/// Configuration that drives plan creation.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    pub strategy: CheckpointStrategy,
    pub num_layers: usize,
    pub activation_size_per_layer_mb: f64,
}

/// Per-layer bookkeeping.
#[derive(Debug, Clone)]
pub struct CheckpointedLayer {
    pub layer_id: usize,
    pub is_checkpointed: bool,
    pub activation: Option<Vec<f32>>,
    pub recompute_count: u64,
}

/// The result of planning: which layers are stored vs recomputed.
#[derive(Debug, Clone)]
pub struct CheckpointPlan {
    pub checkpointed_layers: Vec<usize>,
    pub stored_layers: Vec<usize>,
    pub peak_memory_mb: f64,
    pub recompute_cost_ratio: f64,
}

/// Runtime activation store.
pub struct ActivationStore {
    pub activations: HashMap<usize, Vec<f32>>,
    pub config: CheckpointConfig,
    pub plan: CheckpointPlan,
    pub stats: CheckpointStats,
}

/// Cumulative statistics.
#[derive(Debug, Clone, Default)]
pub struct CheckpointStats {
    pub total_stored: u64,
    pub total_recomputed: u64,
    pub memory_saved_mb: f64,
    pub recompute_overhead_pct: f64,
}

/// Errors specific to checkpoint operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointError {
    LayerNotFound(usize),
    ActivationNotStored(usize),
    MemoryBudgetExceeded,
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayerNotFound(id) => write!(f, "layer {id} not found"),
            Self::ActivationNotStored(id) => {
                write!(f, "activation for layer {id} not stored")
            }
            Self::MemoryBudgetExceeded => write!(f, "memory budget exceeded"),
        }
    }
}

impl std::error::Error for CheckpointError {}

// ── Plan helpers ────────────────────────────────────────────────────

/// Checkpoint every N-th layer (0-indexed). Returns *checkpointed* layer ids.
pub fn cpu_every_n_plan(num_layers: usize, n: usize) -> Vec<usize> {
    if n == 0 {
        return (0..num_layers).collect();
    }
    (0..num_layers).filter(|&l| l % n != 0).collect()
}

/// Checkpoint every √N layers. Returns *checkpointed* layer ids.
pub fn cpu_sqrt_plan(num_layers: usize) -> Vec<usize> {
    if num_layers == 0 {
        return Vec::new();
    }
    let step = (num_layers as f64).sqrt().ceil() as usize;
    let step = step.max(1);
    // Stored layers are 0, step, 2*step, …  Everything else is checkpointed.
    (0..num_layers).filter(|l| l % step != 0).collect()
}

/// Checkpoint layers that exceed a memory budget. Returns *checkpointed* ids.
pub fn cpu_memory_budget_plan(num_layers: usize, per_layer_mb: f64, budget_mb: f64) -> Vec<usize> {
    let max_stored =
        if per_layer_mb <= 0.0 { num_layers } else { (budget_mb / per_layer_mb).floor() as usize };
    let max_stored = max_stored.min(num_layers);
    if max_stored >= num_layers {
        return Vec::new(); // everything fits
    }
    // Keep first `max_stored` layers stored, checkpoint the rest.
    // Spread stored layers evenly.
    let step = if max_stored == 0 { 0 } else { num_layers / max_stored };
    let stored: Vec<usize> =
        if max_stored == 0 { Vec::new() } else { (0..max_stored).map(|i| i * step).collect() };
    (0..num_layers).filter(|l| !stored.contains(l)).collect()
}

/// Build a full `CheckpointPlan` from a config.
pub fn cpu_compute_plan(config: &CheckpointConfig) -> CheckpointPlan {
    let checkpointed = match &config.strategy {
        CheckpointStrategy::None => Vec::new(),
        CheckpointStrategy::EveryN { n } => cpu_every_n_plan(config.num_layers, *n),
        CheckpointStrategy::SelectiveLayers(layers) => layers.clone(),
        CheckpointStrategy::MemoryBudget { max_mb } => {
            cpu_memory_budget_plan(config.num_layers, config.activation_size_per_layer_mb, *max_mb)
        }
        CheckpointStrategy::Sqrt => cpu_sqrt_plan(config.num_layers),
    };

    let stored: Vec<usize> = (0..config.num_layers).filter(|l| !checkpointed.contains(l)).collect();

    let peak_memory_mb = stored.len() as f64 * config.activation_size_per_layer_mb;
    let recompute_cost_ratio = if config.num_layers == 0 {
        0.0
    } else {
        checkpointed.len() as f64 / config.num_layers as f64
    };

    CheckpointPlan {
        checkpointed_layers: checkpointed,
        stored_layers: stored,
        peak_memory_mb,
        recompute_cost_ratio,
    }
}

// ── Store operations ────────────────────────────────────────────────

/// Create a new `ActivationStore` from a config.
pub fn create_activation_store(config: CheckpointConfig) -> ActivationStore {
    let plan = cpu_compute_plan(&config);
    let no_checkpoint_mb = config.num_layers as f64 * config.activation_size_per_layer_mb;
    let saved = no_checkpoint_mb - plan.peak_memory_mb;
    ActivationStore {
        activations: HashMap::new(),
        config,
        plan,
        stats: CheckpointStats {
            total_stored: 0,
            total_recomputed: 0,
            memory_saved_mb: saved.max(0.0),
            recompute_overhead_pct: 0.0,
        },
    }
}

/// Whether a layer's activation should be stored (i.e. it is *not*
/// checkpointed).
pub fn cpu_should_store(store: &ActivationStore, layer_id: usize) -> bool {
    store.plan.stored_layers.contains(&layer_id)
}

/// Persist an activation for the given layer.
pub fn cpu_store_activation(
    store: &mut ActivationStore,
    layer_id: usize,
    activation: Vec<f32>,
) -> Result<(), CheckpointError> {
    if layer_id >= store.config.num_layers {
        return Err(CheckpointError::LayerNotFound(layer_id));
    }
    store.activations.insert(layer_id, activation);
    store.stats.total_stored += 1;
    Ok(())
}

/// Retrieve a previously-stored activation.
pub fn cpu_retrieve_activation(
    store: &ActivationStore,
    layer_id: usize,
) -> Result<&[f32], CheckpointError> {
    if layer_id >= store.config.num_layers {
        return Err(CheckpointError::LayerNotFound(layer_id));
    }
    store
        .activations
        .get(&layer_id)
        .map(|v| v.as_slice())
        .ok_or(CheckpointError::ActivationNotStored(layer_id))
}

/// Simulate recomputing an activation from an input (identity + tag).
pub fn cpu_recompute_activation(
    store: &mut ActivationStore,
    layer_id: usize,
    input: &[f32],
) -> Vec<f32> {
    store.stats.total_recomputed += 1;
    // Simulated recompute: scale by (layer_id + 1) so each layer is distinct.
    let scale = (layer_id + 1) as f32;
    input.iter().map(|&x| x * scale).collect()
}

// ── Estimation helpers ──────────────────────────────────────────────

/// Estimate peak memory in MB for a given plan.
pub fn cpu_estimate_peak_memory(plan: &CheckpointPlan, per_layer_mb: f64) -> f64 {
    plan.stored_layers.len() as f64 * per_layer_mb
}

/// Estimate the fraction of extra compute due to recomputation.
pub fn cpu_estimate_recompute_overhead(plan: &CheckpointPlan, num_layers: usize) -> f64 {
    if num_layers == 0 {
        return 0.0;
    }
    plan.checkpointed_layers.len() as f64 / num_layers as f64
}

/// Snapshot the current stats from a store.
pub fn cpu_get_stats(store: &ActivationStore) -> CheckpointStats {
    let total = store.stats.total_stored + store.stats.total_recomputed;
    let overhead =
        if total == 0 { 0.0 } else { store.stats.total_recomputed as f64 / total as f64 * 100.0 };
    CheckpointStats {
        total_stored: store.stats.total_stored,
        total_recomputed: store.stats.total_recomputed,
        memory_saved_mb: store.stats.memory_saved_mb,
        recompute_overhead_pct: overhead,
    }
}

/// Human-readable summary of a plan.
pub fn format_checkpoint_plan(plan: &CheckpointPlan) -> String {
    format!(
        "CheckpointPlan {{ stored: {}, checkpointed: {}, \
         peak_memory: {:.2} MB, recompute_ratio: {:.2}% }}",
        plan.stored_layers.len(),
        plan.checkpointed_layers.len(),
        plan.peak_memory_mb,
        plan.recompute_cost_ratio * 100.0,
    )
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── every-N plan ────────────────────────────────────────────────

    #[test]
    fn every_n_plan_correct_layers() {
        let cp = cpu_every_n_plan(10, 3);
        // Stored: 0,3,6,9 → checkpointed: 1,2,4,5,7,8
        assert_eq!(cp, vec![1, 2, 4, 5, 7, 8]);
    }

    #[test]
    fn every_n_plan_n_equals_1_stores_all() {
        let cp = cpu_every_n_plan(5, 1);
        // Every layer is stored (l%1==0 always), so nothing checkpointed.
        assert!(cp.is_empty());
    }

    #[test]
    fn every_n_plan_n_equals_2() {
        let cp = cpu_every_n_plan(6, 2);
        assert_eq!(cp, vec![1, 3, 5]);
    }

    #[test]
    fn every_n_plan_large_n() {
        let cp = cpu_every_n_plan(4, 10);
        // Only layer 0 satisfies 0%10==0 → checkpointed: 1,2,3
        assert_eq!(cp, vec![1, 2, 3]);
    }

    #[test]
    fn every_n_plan_zero_n_checkpoints_all() {
        let cp = cpu_every_n_plan(3, 0);
        assert_eq!(cp, vec![0, 1, 2]);
    }

    // ── sqrt plan ───────────────────────────────────────────────────

    #[test]
    fn sqrt_plan_approximately_sqrt_n_stored() {
        let num = 100;
        let cp = cpu_sqrt_plan(num);
        let stored = num - cp.len();
        let sqrt = (num as f64).sqrt().ceil() as usize;
        // stored ≈ num / sqrt
        assert!(stored <= sqrt + 1, "stored={stored}, sqrt={sqrt}");
    }

    #[test]
    fn sqrt_plan_32_layers() {
        let cp = cpu_sqrt_plan(32);
        let stored = 32 - cp.len();
        assert!(stored >= 4 && stored <= 8);
    }

    #[test]
    fn sqrt_plan_single_layer() {
        let cp = cpu_sqrt_plan(1);
        // Layer 0: 0%1==0 → stored. Nothing checkpointed.
        assert!(cp.is_empty());
    }

    #[test]
    fn sqrt_plan_zero_layers() {
        assert!(cpu_sqrt_plan(0).is_empty());
    }

    // ── memory budget plan ──────────────────────────────────────────

    #[test]
    fn memory_budget_stays_under_budget() {
        let cp = cpu_memory_budget_plan(32, 100.0, 800.0);
        let stored = 32 - cp.len();
        assert!((stored as f64 * 100.0) <= 800.0);
    }

    #[test]
    fn memory_budget_unlimited() {
        let cp = cpu_memory_budget_plan(10, 1.0, 100.0);
        assert!(cp.is_empty());
    }

    #[test]
    fn memory_budget_tight() {
        let cp = cpu_memory_budget_plan(10, 10.0, 10.0);
        let stored = 10 - cp.len();
        assert!(stored <= 1);
    }

    #[test]
    fn memory_budget_zero_budget() {
        let cp = cpu_memory_budget_plan(5, 10.0, 0.0);
        assert_eq!(cp.len(), 5);
    }

    // ── store / retrieve ────────────────────────────────────────────

    #[test]
    fn store_activation_persisted() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 4,
            activation_size_per_layer_mb: 1.0,
        };
        let mut store = create_activation_store(cfg);
        let data = vec![1.0, 2.0, 3.0];
        cpu_store_activation(&mut store, 0, data.clone()).unwrap();
        assert_eq!(cpu_retrieve_activation(&store, 0).unwrap(), data.as_slice());
    }

    #[test]
    fn retrieve_correct_data() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 4,
            activation_size_per_layer_mb: 1.0,
        };
        let mut store = create_activation_store(cfg);
        cpu_store_activation(&mut store, 1, vec![10.0, 20.0]).unwrap();
        cpu_store_activation(&mut store, 2, vec![30.0, 40.0]).unwrap();
        assert_eq!(cpu_retrieve_activation(&store, 1).unwrap(), &[10.0, 20.0]);
        assert_eq!(cpu_retrieve_activation(&store, 2).unwrap(), &[30.0, 40.0]);
    }

    #[test]
    fn retrieve_not_stored_error() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 4,
            activation_size_per_layer_mb: 1.0,
        };
        let store = create_activation_store(cfg);
        assert_eq!(
            cpu_retrieve_activation(&store, 0).unwrap_err(),
            CheckpointError::ActivationNotStored(0),
        );
    }

    #[test]
    fn store_out_of_range_error() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 2,
            activation_size_per_layer_mb: 1.0,
        };
        let mut store = create_activation_store(cfg);
        assert_eq!(
            cpu_store_activation(&mut store, 5, vec![1.0]).unwrap_err(),
            CheckpointError::LayerNotFound(5),
        );
    }

    #[test]
    fn retrieve_out_of_range_error() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 2,
            activation_size_per_layer_mb: 1.0,
        };
        let store = create_activation_store(cfg);
        assert_eq!(
            cpu_retrieve_activation(&store, 10).unwrap_err(),
            CheckpointError::LayerNotFound(10),
        );
    }

    // ── should_store ────────────────────────────────────────────────

    #[test]
    fn should_store_respects_plan() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::EveryN { n: 2 },
            num_layers: 6,
            activation_size_per_layer_mb: 1.0,
        };
        let store = create_activation_store(cfg);
        // Stored: 0,2,4
        assert!(cpu_should_store(&store, 0));
        assert!(!cpu_should_store(&store, 1));
        assert!(cpu_should_store(&store, 2));
    }

    #[test]
    fn should_store_none_strategy_stores_all() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 4,
            activation_size_per_layer_mb: 1.0,
        };
        let store = create_activation_store(cfg);
        for i in 0..4 {
            assert!(cpu_should_store(&store, i));
        }
    }

    // ── recompute ───────────────────────────────────────────────────

    #[test]
    fn recompute_produces_activation() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 4,
            activation_size_per_layer_mb: 1.0,
        };
        let mut store = create_activation_store(cfg);
        let input = vec![1.0, 2.0];
        let out = cpu_recompute_activation(&mut store, 2, &input);
        // scale = layer_id + 1 = 3
        assert_eq!(out, vec![3.0, 6.0]);
    }

    #[test]
    fn recompute_increments_stats() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 4,
            activation_size_per_layer_mb: 1.0,
        };
        let mut store = create_activation_store(cfg);
        cpu_recompute_activation(&mut store, 0, &[1.0]);
        cpu_recompute_activation(&mut store, 1, &[1.0]);
        assert_eq!(store.stats.total_recomputed, 2);
    }

    // ── peak memory / overhead estimation ───────────────────────────

    #[test]
    fn peak_memory_less_than_no_checkpoint() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::EveryN { n: 2 },
            num_layers: 10,
            activation_size_per_layer_mb: 50.0,
        };
        let plan = cpu_compute_plan(&cfg);
        let peak = cpu_estimate_peak_memory(&plan, 50.0);
        let full = 10.0 * 50.0;
        assert!(peak < full);
    }

    #[test]
    fn peak_memory_no_checkpoint() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 8,
            activation_size_per_layer_mb: 10.0,
        };
        let plan = cpu_compute_plan(&cfg);
        let peak = cpu_estimate_peak_memory(&plan, 10.0);
        assert!((peak - 80.0).abs() < 1e-6);
    }

    #[test]
    fn recompute_overhead_proportional() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::EveryN { n: 2 },
            num_layers: 10,
            activation_size_per_layer_mb: 1.0,
        };
        let plan = cpu_compute_plan(&cfg);
        let overhead = cpu_estimate_recompute_overhead(&plan, 10);
        // 5 checkpointed out of 10 → 0.5
        assert!((overhead - 0.5).abs() < 1e-6);
    }

    #[test]
    fn recompute_overhead_none_strategy() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 10,
            activation_size_per_layer_mb: 1.0,
        };
        let plan = cpu_compute_plan(&cfg);
        assert!(cpu_estimate_recompute_overhead(&plan, 10).abs() < 1e-9);
    }

    // ── edge cases ──────────────────────────────────────────────────

    #[test]
    fn edge_single_layer() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::EveryN { n: 2 },
            num_layers: 1,
            activation_size_per_layer_mb: 5.0,
        };
        let plan = cpu_compute_plan(&cfg);
        // Layer 0: 0%2==0 → stored. Nothing checkpointed.
        assert!(plan.checkpointed_layers.is_empty());
        assert_eq!(plan.stored_layers, vec![0]);
    }

    #[test]
    fn edge_all_layers_checkpointed() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::SelectiveLayers((0..5).collect()),
            num_layers: 5,
            activation_size_per_layer_mb: 10.0,
        };
        let plan = cpu_compute_plan(&cfg);
        assert_eq!(plan.checkpointed_layers.len(), 5);
        assert!(plan.stored_layers.is_empty());
        assert!(plan.peak_memory_mb.abs() < 1e-9);
    }

    #[test]
    fn edge_no_layers_checkpointed() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 5,
            activation_size_per_layer_mb: 10.0,
        };
        let plan = cpu_compute_plan(&cfg);
        assert!(plan.checkpointed_layers.is_empty());
        assert_eq!(plan.stored_layers.len(), 5);
    }

    // ── property-based ──────────────────────────────────────────────

    #[test]
    fn property_stored_plus_checkpointed_equals_total() {
        for n in [1, 5, 10, 32, 64, 100] {
            let cfg = CheckpointConfig {
                strategy: CheckpointStrategy::Sqrt,
                num_layers: n,
                activation_size_per_layer_mb: 1.0,
            };
            let plan = cpu_compute_plan(&cfg);
            assert_eq!(
                plan.stored_layers.len() + plan.checkpointed_layers.len(),
                n,
                "failed for n={n}"
            );
        }
    }

    #[test]
    fn property_stored_plus_checkpointed_every_n() {
        for n in [2, 3, 5, 7] {
            let cfg = CheckpointConfig {
                strategy: CheckpointStrategy::EveryN { n },
                num_layers: 20,
                activation_size_per_layer_mb: 1.0,
            };
            let plan = cpu_compute_plan(&cfg);
            assert_eq!(
                plan.stored_layers.len() + plan.checkpointed_layers.len(),
                20,
                "failed for n={n}"
            );
        }
    }

    #[test]
    fn property_no_duplicates_in_plan() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::Sqrt,
            num_layers: 50,
            activation_size_per_layer_mb: 2.0,
        };
        let plan = cpu_compute_plan(&cfg);
        let mut all: Vec<usize> =
            plan.stored_layers.iter().chain(plan.checkpointed_layers.iter()).copied().collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 50);
    }

    #[test]
    fn property_memory_savings_positive_when_checkpointing() {
        for strat in [CheckpointStrategy::EveryN { n: 2 }, CheckpointStrategy::Sqrt] {
            let cfg = CheckpointConfig {
                strategy: strat.clone(),
                num_layers: 20,
                activation_size_per_layer_mb: 10.0,
            };
            let plan = cpu_compute_plan(&cfg);
            let full = 20.0 * 10.0;
            assert!(plan.peak_memory_mb < full, "strategy {strat:?} did not save memory");
        }
    }

    #[test]
    fn property_recompute_cost_ratio_in_range() {
        for n in [4, 16, 64] {
            let cfg = CheckpointConfig {
                strategy: CheckpointStrategy::EveryN { n: 3 },
                num_layers: n,
                activation_size_per_layer_mb: 1.0,
            };
            let plan = cpu_compute_plan(&cfg);
            assert!(
                (0.0..=1.0).contains(&plan.recompute_cost_ratio),
                "ratio out of range for n={n}"
            );
        }
    }

    // ── A770 scenario ───────────────────────────────────────────────

    #[test]
    fn a770_16gb_budget_32_layers_2048_hidden() {
        // 2048 hidden * 4 bytes ≈ 8 KB per token; assume seq_len=2048
        // → ~16 MB per layer activation.
        let per_layer_mb = 16.0;
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::MemoryBudget { max_mb: 16_000.0 },
            num_layers: 32,
            activation_size_per_layer_mb: per_layer_mb,
        };
        let plan = cpu_compute_plan(&cfg);
        assert!(plan.peak_memory_mb <= 16_000.0);
        assert_eq!(plan.stored_layers.len() + plan.checkpointed_layers.len(), 32);
    }

    #[test]
    fn a770_sqrt_32_layers() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::Sqrt,
            num_layers: 32,
            activation_size_per_layer_mb: 16.0,
        };
        let plan = cpu_compute_plan(&cfg);
        let stored = plan.stored_layers.len();
        assert!(stored >= 4 && stored <= 8);
        assert!(plan.peak_memory_mb < 32.0 * 16.0);
    }

    // ── stats ───────────────────────────────────────────────────────

    #[test]
    fn stats_after_mixed_ops() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::EveryN { n: 2 },
            num_layers: 4,
            activation_size_per_layer_mb: 1.0,
        };
        let mut store = create_activation_store(cfg);
        cpu_store_activation(&mut store, 0, vec![1.0]).unwrap();
        cpu_recompute_activation(&mut store, 1, &[1.0]);
        let stats = cpu_get_stats(&store);
        assert_eq!(stats.total_stored, 1);
        assert_eq!(stats.total_recomputed, 1);
        assert!((stats.recompute_overhead_pct - 50.0).abs() < 1e-6);
    }

    #[test]
    fn stats_initial_zero() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 4,
            activation_size_per_layer_mb: 1.0,
        };
        let store = create_activation_store(cfg);
        let stats = cpu_get_stats(&store);
        assert_eq!(stats.total_stored, 0);
        assert_eq!(stats.total_recomputed, 0);
    }

    // ── format ──────────────────────────────────────────────────────

    #[test]
    fn format_plan_contains_fields() {
        let plan = CheckpointPlan {
            checkpointed_layers: vec![1, 3],
            stored_layers: vec![0, 2, 4],
            peak_memory_mb: 30.0,
            recompute_cost_ratio: 0.4,
        };
        let s = format_checkpoint_plan(&plan);
        assert!(s.contains("stored: 3"));
        assert!(s.contains("checkpointed: 2"));
        assert!(s.contains("30.00 MB"));
        assert!(s.contains("40.00%"));
    }

    // ── compute_plan integration ────────────────────────────────────

    #[test]
    fn compute_plan_none_stores_everything() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 8,
            activation_size_per_layer_mb: 5.0,
        };
        let plan = cpu_compute_plan(&cfg);
        assert_eq!(plan.stored_layers.len(), 8);
        assert!(plan.checkpointed_layers.is_empty());
    }

    #[test]
    fn compute_plan_selective_layers() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::SelectiveLayers(vec![2, 5]),
            num_layers: 8,
            activation_size_per_layer_mb: 10.0,
        };
        let plan = cpu_compute_plan(&cfg);
        assert_eq!(plan.checkpointed_layers, vec![2, 5]);
        assert_eq!(plan.stored_layers.len(), 6);
    }

    #[test]
    fn compute_plan_peak_memory_matches_stored() {
        let per = 7.5;
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::EveryN { n: 3 },
            num_layers: 12,
            activation_size_per_layer_mb: per,
        };
        let plan = cpu_compute_plan(&cfg);
        let expected = plan.stored_layers.len() as f64 * per;
        assert!((plan.peak_memory_mb - expected).abs() < 1e-9);
    }

    // ── create_activation_store ─────────────────────────────────────

    #[test]
    fn create_store_memory_saved_positive() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::EveryN { n: 2 },
            num_layers: 10,
            activation_size_per_layer_mb: 50.0,
        };
        let store = create_activation_store(cfg);
        assert!(store.stats.memory_saved_mb > 0.0);
    }

    #[test]
    fn create_store_memory_saved_zero_no_checkpoint() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 10,
            activation_size_per_layer_mb: 50.0,
        };
        let store = create_activation_store(cfg);
        assert!(store.stats.memory_saved_mb.abs() < 1e-9);
    }

    #[test]
    fn store_overwrite_activation() {
        let cfg = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            num_layers: 4,
            activation_size_per_layer_mb: 1.0,
        };
        let mut store = create_activation_store(cfg);
        cpu_store_activation(&mut store, 0, vec![1.0]).unwrap();
        cpu_store_activation(&mut store, 0, vec![2.0]).unwrap();
        assert_eq!(cpu_retrieve_activation(&store, 0).unwrap(), &[2.0]);
    }

    // ── CheckpointedLayer ───────────────────────────────────────────

    #[test]
    fn checkpointed_layer_default_fields() {
        let cl = CheckpointedLayer {
            layer_id: 3,
            is_checkpointed: true,
            activation: None,
            recompute_count: 0,
        };
        assert!(cl.is_checkpointed);
        assert!(cl.activation.is_none());
    }

    // ── error display ───────────────────────────────────────────────

    #[test]
    fn error_display_layer_not_found() {
        let e = CheckpointError::LayerNotFound(42);
        assert_eq!(format!("{e}"), "layer 42 not found");
    }

    #[test]
    fn error_display_not_stored() {
        let e = CheckpointError::ActivationNotStored(7);
        assert!(format!("{e}").contains("not stored"));
    }

    #[test]
    fn error_display_budget_exceeded() {
        let e = CheckpointError::MemoryBudgetExceeded;
        assert!(format!("{e}").contains("budget"));
    }
}
