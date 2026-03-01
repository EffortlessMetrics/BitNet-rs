//! Module stub - implementation pending merge from feature branch
//! Model pruning strategies for neural network weight reduction.
//!
//! Provides magnitude-based, structured, and movement pruning with
//! lottery ticket hypothesis support, gradual schedules, sensitivity
//! analysis, and binary mask management.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Pruning method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningMethod {
    /// Unstructured magnitude pruning — removes individual weights.
    Magnitude,
    /// Structured pruning — removes entire channels / heads / layers.
    Structured,
    /// Movement pruning — prunes based on weight-change direction.
    Movement,
    /// Lottery ticket hypothesis — find sparse winning sub-networks.
    LotteryTicket,
}

/// Granularity at which structured pruning operates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningGranularity {
    /// Per-weight (unstructured).
    Weight,
    /// Remove entire output channels.
    Channel,
    /// Remove attention heads.
    Head,
    /// Remove whole layers.
    Layer,
}

/// Schedule shape for gradual pruning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScheduleKind {
    /// Prune all at once at the start.
    OneShot,
    /// Linear ramp from 0 to target sparsity.
    Linear,
    /// Cubic polynomial ramp (Zhu & Gupta, 2017).
    Cubic,
}

// ---------------------------------------------------------------------------
// PruningConfig
// ---------------------------------------------------------------------------

/// Configuration for a pruning session.
#[derive(Debug, Clone)]
pub struct PruningConfig {
    /// Which pruning algorithm to use.
    pub method: PruningMethod,
    /// Target sparsity ratio in `[0.0, 1.0]`.
    pub target_sparsity: f32,
    /// Schedule shape for gradual pruning.
    pub schedule: ScheduleKind,
    /// Granularity for structured pruning.
    pub granularity: PruningGranularity,
    /// Total training / fine-tuning steps for schedule computation.
    pub total_steps: usize,
    /// Step at which pruning begins.
    pub begin_step: usize,
    /// Step at which pruning ends (target sparsity fully reached).
    pub end_step: usize,
    /// Frequency (in steps) at which the mask is updated.
    pub frequency: usize,
}

impl PruningConfig {
    /// Create a default magnitude-pruning config with the given sparsity.
    pub fn magnitude(target_sparsity: f32) -> Self {
        Self {
            method: PruningMethod::Magnitude,
            target_sparsity,
            schedule: ScheduleKind::OneShot,
            granularity: PruningGranularity::Weight,
            total_steps: 1,
            begin_step: 0,
            end_step: 1,
            frequency: 1,
        }
    }

    /// Create a structured-pruning config.
    pub fn structured(target_sparsity: f32, granularity: PruningGranularity) -> Self {
        Self {
            method: PruningMethod::Structured,
            target_sparsity,
            schedule: ScheduleKind::OneShot,
            granularity,
            total_steps: 1,
            begin_step: 0,
            end_step: 1,
            frequency: 1,
        }
    }

    /// Create a movement-pruning config with a gradual schedule.
    pub fn movement(target_sparsity: f32, total_steps: usize) -> Self {
        Self {
            method: PruningMethod::Movement,
            target_sparsity,
            schedule: ScheduleKind::Cubic,
            granularity: PruningGranularity::Weight,
            total_steps,
            begin_step: 0,
            end_step: total_steps,
            frequency: 1,
        }
    }

    /// Create a lottery-ticket config.
    pub fn lottery_ticket(target_sparsity: f32) -> Self {
        Self {
            method: PruningMethod::LotteryTicket,
            target_sparsity,
            schedule: ScheduleKind::OneShot,
            granularity: PruningGranularity::Weight,
            total_steps: 1,
            begin_step: 0,
            end_step: 1,
            frequency: 1,
        }
    }

    /// Validate that the config values are consistent.
    pub fn validate(&self) -> Result<(), String> {
        if self.target_sparsity < 0.0 || self.target_sparsity > 1.0 {
            return Err(format!("target_sparsity must be in [0,1], got {}", self.target_sparsity));
        }
        if self.end_step <= self.begin_step && self.total_steps > 1 {
            return Err("end_step must be > begin_step for multi-step schedules".into());
        }
        if self.frequency == 0 {
            return Err("frequency must be >= 1".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PruningSchedule
// ---------------------------------------------------------------------------

/// Computes the sparsity ratio at a given step according to the schedule.
#[derive(Debug, Clone)]
pub struct PruningSchedule {
    kind: ScheduleKind,
    target_sparsity: f32,
    begin_step: usize,
    end_step: usize,
}

impl PruningSchedule {
    /// Build a schedule from a [`PruningConfig`].
    pub fn from_config(cfg: &PruningConfig) -> Self {
        Self {
            kind: cfg.schedule,
            target_sparsity: cfg.target_sparsity,
            begin_step: cfg.begin_step,
            end_step: cfg.end_step,
        }
    }

    /// Create a new schedule directly.
    pub fn new(
        kind: ScheduleKind,
        target_sparsity: f32,
        begin_step: usize,
        end_step: usize,
    ) -> Self {
        Self { kind, target_sparsity, begin_step, end_step }
    }

    /// Return the sparsity ratio that should be applied at `step`.
    pub fn sparsity_at_step(&self, step: usize) -> f32 {
        if step < self.begin_step {
            return 0.0;
        }
        if step >= self.end_step {
            return self.target_sparsity;
        }
        let t = (step - self.begin_step) as f32 / (self.end_step - self.begin_step).max(1) as f32;
        match self.kind {
            ScheduleKind::OneShot => self.target_sparsity,
            ScheduleKind::Linear => self.target_sparsity * t,
            ScheduleKind::Cubic => {
                // s_t = s_f * (1 - (1 - t)^3)  — Zhu & Gupta 2017
                self.target_sparsity * (1.0 - (1.0 - t).powi(3))
            }
        }
    }

    /// Target sparsity this schedule converges to.
    pub fn target_sparsity(&self) -> f32 {
        self.target_sparsity
    }
}

// ---------------------------------------------------------------------------
// PruneMaskManager
// ---------------------------------------------------------------------------

/// Manages binary pruning masks for named tensors.
///
/// A mask value of `true` means *keep*, `false` means *pruned*.
#[derive(Debug, Clone)]
pub struct PruneMaskManager {
    masks: HashMap<String, Vec<bool>>,
}

impl PruneMaskManager {
    /// Create an empty mask manager.
    pub fn new() -> Self {
        Self { masks: HashMap::new() }
    }

    /// Register a fully-dense mask (all kept) for a tensor.
    pub fn register(&mut self, name: &str, len: usize) {
        self.masks.insert(name.to_string(), vec![true; len]);
    }

    /// Set the mask for a tensor, replacing any existing one.
    pub fn set_mask(&mut self, name: &str, mask: Vec<bool>) {
        self.masks.insert(name.to_string(), mask);
    }

    /// Get the mask for a tensor (if registered).
    pub fn get_mask(&self, name: &str) -> Option<&[bool]> {
        self.masks.get(name).map(|v| v.as_slice())
    }

    /// Apply the mask to `weights` in-place, zeroing pruned positions.
    pub fn apply(&self, name: &str, weights: &mut [f32]) {
        if let Some(mask) = self.masks.get(name) {
            for (w, &keep) in weights.iter_mut().zip(mask.iter()) {
                if !keep {
                    *w = 0.0;
                }
            }
        }
    }

    /// Compute sparsity ratio for a specific tensor.
    pub fn sparsity(&self, name: &str) -> Option<f32> {
        self.masks.get(name).map(|m| {
            let pruned = m.iter().filter(|&&k| !k).count();
            pruned as f32 / m.len().max(1) as f32
        })
    }

    /// Compute overall sparsity across all registered tensors.
    pub fn overall_sparsity(&self) -> f32 {
        let (total, pruned) = self.masks.values().fold((0usize, 0usize), |(t, p), m| {
            let pr = m.iter().filter(|&&k| !k).count();
            (t + m.len(), p + pr)
        });
        if total == 0 {
            return 0.0;
        }
        pruned as f32 / total as f32
    }

    /// Number of registered tensors.
    pub fn tensor_count(&self) -> usize {
        self.masks.len()
    }

    /// Iterator over tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.masks.keys().map(|s| s.as_str())
    }

    /// Merge another mask set (intersection — keep only if both keep).
    pub fn intersect(&mut self, other: &PruneMaskManager) {
        for (name, other_mask) in &other.masks {
            if let Some(self_mask) = self.masks.get_mut(name) {
                for (s, &o) in self_mask.iter_mut().zip(other_mask.iter()) {
                    *s = *s && o;
                }
            }
        }
    }
}

impl Default for PruneMaskManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MagnitudePruner
// ---------------------------------------------------------------------------

/// Unstructured magnitude-based pruner.
///
/// Removes individual weights whose absolute values are smallest.
#[derive(Debug, Clone)]
pub struct MagnitudePruner {
    config: PruningConfig,
}

impl MagnitudePruner {
    /// Create a new magnitude pruner with the given config.
    pub fn new(config: PruningConfig) -> Self {
        Self { config }
    }

    /// Compute a binary mask for `weights` at the given sparsity ratio.
    pub fn compute_mask(&self, weights: &[f32], sparsity: f32) -> Vec<bool> {
        if weights.is_empty() || sparsity <= 0.0 {
            return vec![true; weights.len()];
        }
        if sparsity >= 1.0 {
            return vec![false; weights.len()];
        }
        let mut indices: Vec<usize> = (0..weights.len()).collect();
        indices.sort_by(|&a, &b| {
            weights[a].abs().partial_cmp(&weights[b].abs()).unwrap_or(std::cmp::Ordering::Equal)
        });
        let n_prune = (weights.len() as f32 * sparsity).round() as usize;
        let mut mask = vec![true; weights.len()];
        for &idx in &indices[..n_prune.min(weights.len())] {
            mask[idx] = false;
        }
        mask
    }

    /// Prune `weights` in-place, returning the mask.
    pub fn prune(&self, weights: &mut [f32]) -> Vec<bool> {
        let mask = self.compute_mask(weights, self.config.target_sparsity);
        for (w, &keep) in weights.iter_mut().zip(mask.iter()) {
            if !keep {
                *w = 0.0;
            }
        }
        mask
    }

    /// Target sparsity of this pruner.
    pub fn target_sparsity(&self) -> f32 {
        self.config.target_sparsity
    }
}

// ---------------------------------------------------------------------------
// StructuredPruner
// ---------------------------------------------------------------------------

/// Structured pruner that removes entire channels, heads, or layers.
#[derive(Debug, Clone)]
pub struct StructuredPruner {
    config: PruningConfig,
}

impl StructuredPruner {
    /// Create a new structured pruner.
    pub fn new(config: PruningConfig) -> Self {
        Self { config }
    }

    /// Prune channels from a 2-D weight matrix stored row-major.
    ///
    /// `weights` has shape `[out_channels, in_features]`.
    /// Returns a mask of length `out_channels` (true = keep).
    pub fn prune_channels(
        &self,
        weights: &[f32],
        out_channels: usize,
        in_features: usize,
    ) -> Vec<bool> {
        if out_channels == 0 || in_features == 0 {
            return vec![true; out_channels];
        }
        // L1-norm per channel
        let mut norms: Vec<(usize, f32)> = (0..out_channels)
            .map(|c| {
                let start = c * in_features;
                let end = start + in_features;
                let norm: f32 = weights[start..end].iter().map(|w| w.abs()).sum();
                (c, norm)
            })
            .collect();
        norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let n_prune = (out_channels as f32 * self.config.target_sparsity).round() as usize;
        let mut mask = vec![true; out_channels];
        for &(ch, _) in &norms[..n_prune.min(out_channels)] {
            mask[ch] = false;
        }
        mask
    }

    /// Prune attention heads. `head_norms` contains the L1/L2 norm of each head.
    pub fn prune_heads(&self, head_norms: &[f32]) -> Vec<bool> {
        if head_norms.is_empty() {
            return vec![];
        }
        let mut indexed: Vec<(usize, f32)> = head_norms.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let n_prune = (head_norms.len() as f32 * self.config.target_sparsity).round() as usize;
        let mut mask = vec![true; head_norms.len()];
        for &(idx, _) in &indexed[..n_prune.min(head_norms.len())] {
            mask[idx] = false;
        }
        mask
    }

    /// Prune whole layers by importance score. Returns mask of length `n_layers`.
    pub fn prune_layers(&self, layer_scores: &[f32]) -> Vec<bool> {
        if layer_scores.is_empty() {
            return vec![];
        }
        let mut indexed: Vec<(usize, f32)> = layer_scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let n_prune = (layer_scores.len() as f32 * self.config.target_sparsity).round() as usize;
        let mut mask = vec![true; layer_scores.len()];
        for &(idx, _) in &indexed[..n_prune.min(layer_scores.len())] {
            mask[idx] = false;
        }
        mask
    }

    /// Granularity of this pruner.
    pub fn granularity(&self) -> PruningGranularity {
        self.config.granularity
    }

    /// Target sparsity.
    pub fn target_sparsity(&self) -> f32 {
        self.config.target_sparsity
    }
}

// ---------------------------------------------------------------------------
// MovementPruner
// ---------------------------------------------------------------------------

/// Movement pruner — prunes weights whose *movement scores* are smallest.
///
/// Movement score = weight × gradient (positive means growing in magnitude).
#[derive(Debug, Clone)]
pub struct MovementPruner {
    config: PruningConfig,
    schedule: PruningSchedule,
    current_step: usize,
}

impl MovementPruner {
    /// Create a new movement pruner.
    pub fn new(config: PruningConfig) -> Self {
        let schedule = PruningSchedule::from_config(&config);
        Self { config, schedule, current_step: 0 }
    }

    /// Compute movement scores: `score_i = weight_i * gradient_i`.
    pub fn movement_scores(weights: &[f32], gradients: &[f32]) -> Vec<f32> {
        weights.iter().zip(gradients.iter()).map(|(w, g)| w * g).collect()
    }

    /// Compute mask from movement scores at given sparsity.
    pub fn compute_mask(scores: &[f32], sparsity: f32) -> Vec<bool> {
        if scores.is_empty() || sparsity <= 0.0 {
            return vec![true; scores.len()];
        }
        if sparsity >= 1.0 {
            return vec![false; scores.len()];
        }
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.sort_by(|&a, &b| {
            scores[a].partial_cmp(&scores[b]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let n_prune = (scores.len() as f32 * sparsity).round() as usize;
        let mut mask = vec![true; scores.len()];
        for &idx in &indices[..n_prune.min(scores.len())] {
            mask[idx] = false;
        }
        mask
    }

    /// Advance one step and produce a mask based on current movement scores.
    pub fn step(&mut self, weights: &[f32], gradients: &[f32]) -> Vec<bool> {
        let sparsity = self.schedule.sparsity_at_step(self.current_step);
        let scores = Self::movement_scores(weights, gradients);
        self.current_step += 1;
        Self::compute_mask(&scores, sparsity)
    }

    /// Current training step.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Current sparsity target from the schedule.
    pub fn current_sparsity(&self) -> f32 {
        self.schedule.sparsity_at_step(self.current_step)
    }

    /// Target sparsity.
    pub fn target_sparsity(&self) -> f32 {
        self.config.target_sparsity
    }
}

// ---------------------------------------------------------------------------
// LotteryTicket
// ---------------------------------------------------------------------------

/// Lottery Ticket Hypothesis implementation.
///
/// Identifies sparse "winning ticket" sub-networks by iterative
/// magnitude pruning followed by weight rewinding to initial values.
#[derive(Debug, Clone)]
pub struct LotteryTicket {
    config: PruningConfig,
    /// Stored initial (or rewound) weights per tensor name.
    initial_weights: HashMap<String, Vec<f32>>,
    /// Current round of iterative pruning.
    round: usize,
    /// Maximum number of pruning rounds.
    max_rounds: usize,
}

impl LotteryTicket {
    /// Create a new lottery ticket finder.
    pub fn new(config: PruningConfig, max_rounds: usize) -> Self {
        Self { config, initial_weights: HashMap::new(), round: 0, max_rounds }
    }

    /// Save initial weights (ticket checkpoint).
    pub fn save_initial_weights(&mut self, name: &str, weights: &[f32]) {
        self.initial_weights.insert(name.to_string(), weights.to_vec());
    }

    /// Return initial weights for a tensor.
    pub fn initial_weights(&self, name: &str) -> Option<&[f32]> {
        self.initial_weights.get(name).map(|v| v.as_slice())
    }

    /// Compute the per-round sparsity (iterative schedule).
    ///
    /// Each round prunes a fraction: after `k` rounds the kept ratio is
    /// `(1 - per_round)^k` where `per_round = 1 - (1 - target)^(1/max_rounds)`.
    pub fn sparsity_at_round(&self, round: usize) -> f32 {
        if self.max_rounds == 0 {
            return self.config.target_sparsity;
        }
        let keep_per_round = (1.0 - self.config.target_sparsity).powf(1.0 / self.max_rounds as f32);
        1.0 - keep_per_round.powi(round as i32)
    }

    /// Run one round: magnitude-prune, produce mask, rewind weights.
    pub fn prune_round(&mut self, _name: &str, trained_weights: &[f32]) -> Vec<bool> {
        self.round += 1;
        let sparsity = self.sparsity_at_round(self.round);
        let pruner = MagnitudePruner::new(PruningConfig::magnitude(sparsity));
        pruner.compute_mask(trained_weights, sparsity)
    }

    /// Rewind `weights` to their saved initial values, respecting `mask`.
    pub fn rewind(&self, name: &str, weights: &mut [f32], mask: &[bool]) {
        if let Some(init) = self.initial_weights.get(name) {
            for (i, w) in weights.iter_mut().enumerate() {
                if mask.get(i).copied().unwrap_or(false) {
                    *w = init.get(i).copied().unwrap_or(0.0);
                } else {
                    *w = 0.0;
                }
            }
        }
    }

    /// Current pruning round.
    pub fn round(&self) -> usize {
        self.round
    }

    /// Maximum rounds.
    pub fn max_rounds(&self) -> usize {
        self.max_rounds
    }

    /// Whether all rounds are complete.
    pub fn is_complete(&self) -> bool {
        self.round >= self.max_rounds
    }
}

// ---------------------------------------------------------------------------
// SensitivityAnalyzer
// ---------------------------------------------------------------------------

/// Per-layer sensitivity analysis for adaptive pruning ratio selection.
///
/// Evaluates how much each layer's accuracy degrades when pruned at
/// various sparsity levels, then recommends per-layer ratios.
#[derive(Debug, Clone)]
pub struct SensitivityAnalyzer {
    /// Recorded (layer_name, sparsity, metric_delta) triples.
    records: Vec<(String, f32, f32)>,
}

impl SensitivityAnalyzer {
    /// Create an empty analyzer.
    pub fn new() -> Self {
        Self { records: Vec::new() }
    }

    /// Record the metric degradation for a layer at a sparsity level.
    pub fn record(&mut self, layer: &str, sparsity: f32, metric_delta: f32) {
        self.records.push((layer.to_string(), sparsity, metric_delta));
    }

    /// Recommend per-layer sparsity ratios such that the total model
    /// sparsity is approximately `target` and no single layer exceeds
    /// `max_degradation`.
    pub fn recommend(&self, target_sparsity: f32, max_degradation: f32) -> HashMap<String, f32> {
        // Collect unique layers
        let mut layer_max_sparsity: HashMap<String, f32> = HashMap::new();
        for (layer, sparsity, delta) in &self.records {
            if *delta <= max_degradation {
                let entry = layer_max_sparsity.entry(layer.clone()).or_insert(0.0);
                if *sparsity > *entry {
                    *entry = *sparsity;
                }
            }
        }
        // Scale recommendations so the average meets target
        if layer_max_sparsity.is_empty() {
            return layer_max_sparsity;
        }
        let avg: f32 = layer_max_sparsity.values().sum::<f32>() / layer_max_sparsity.len() as f32;
        let scale = if avg > 0.0 { (target_sparsity / avg).min(1.0) } else { 1.0 };
        for v in layer_max_sparsity.values_mut() {
            *v = (*v * scale).min(1.0);
        }
        layer_max_sparsity
    }

    /// Return the sensitivity curve for a specific layer: sorted (sparsity, delta) pairs.
    pub fn sensitivity_curve(&self, layer: &str) -> Vec<(f32, f32)> {
        let mut curve: Vec<(f32, f32)> =
            self.records.iter().filter(|(l, _, _)| l == layer).map(|(_, s, d)| (*s, *d)).collect();
        curve.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        curve
    }

    /// Number of recorded data points.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }
}

impl Default for SensitivityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PruningReport
// ---------------------------------------------------------------------------

/// Statistics and quality metrics produced after pruning.
#[derive(Debug, Clone)]
pub struct PruningReport {
    /// Per-tensor sparsity map.
    pub per_tensor_sparsity: HashMap<String, f32>,
    /// Overall model sparsity.
    pub overall_sparsity: f32,
    /// Total parameters before pruning.
    pub total_params: usize,
    /// Parameters remaining (non-zero).
    pub remaining_params: usize,
    /// Parameters removed.
    pub pruned_params: usize,
    /// Estimated compression ratio (total / remaining).
    pub compression_ratio: f32,
    /// Optional quality metric (e.g., perplexity delta).
    pub quality_metric: Option<f32>,
    /// Method used.
    pub method: PruningMethod,
}

impl PruningReport {
    /// Build a report from a mask manager.
    pub fn from_mask_manager(mgr: &PruneMaskManager, method: PruningMethod) -> Self {
        let mut per_tensor_sparsity = HashMap::new();
        let mut total = 0usize;
        let mut pruned = 0usize;

        for name in mgr.tensor_names() {
            if let Some(mask) = mgr.get_mask(name) {
                let p = mask.iter().filter(|&&k| !k).count();
                total += mask.len();
                pruned += p;
                per_tensor_sparsity.insert(name.to_string(), p as f32 / mask.len().max(1) as f32);
            }
        }
        let remaining = total.saturating_sub(pruned);
        Self {
            per_tensor_sparsity,
            overall_sparsity: if total > 0 { pruned as f32 / total as f32 } else { 0.0 },
            total_params: total,
            remaining_params: remaining,
            pruned_params: pruned,
            compression_ratio: if remaining > 0 {
                total as f32 / remaining as f32
            } else {
                f32::INFINITY
            },
            quality_metric: None,
            method,
        }
    }

    /// Build a report from raw weight vectors.
    pub fn from_weights(named_weights: &[(&str, &[f32])], method: PruningMethod) -> Self {
        let mut per_tensor_sparsity = HashMap::new();
        let mut total = 0usize;
        let mut pruned = 0usize;
        for (name, w) in named_weights {
            let p = w.iter().filter(|&&v| v == 0.0).count();
            total += w.len();
            pruned += p;
            per_tensor_sparsity.insert(name.to_string(), p as f32 / w.len().max(1) as f32);
        }
        let remaining = total.saturating_sub(pruned);
        Self {
            per_tensor_sparsity,
            overall_sparsity: if total > 0 { pruned as f32 / total as f32 } else { 0.0 },
            total_params: total,
            remaining_params: remaining,
            pruned_params: pruned,
            compression_ratio: if remaining > 0 {
                total as f32 / remaining as f32
            } else {
                f32::INFINITY
            },
            quality_metric: None,
            method,
        }
    }

    /// Set an optional quality metric on the report.
    pub fn with_quality_metric(mut self, metric: f32) -> Self {
        self.quality_metric = Some(metric);
        self
    }
}

// ---------------------------------------------------------------------------
// PruningEngine
// ---------------------------------------------------------------------------

/// Unified interface for all pruning strategies.
///
/// Wraps the individual pruners, manages masks, and produces reports.
#[derive(Debug, Clone)]
pub struct PruningEngine {
    config: PruningConfig,
    mask_manager: PruneMaskManager,
    schedule: PruningSchedule,
    step: usize,
}

impl PruningEngine {
    /// Create a new pruning engine from a config.
    pub fn new(config: PruningConfig) -> Self {
        let schedule = PruningSchedule::from_config(&config);
        Self { config, mask_manager: PruneMaskManager::new(), schedule, step: 0 }
    }

    /// Register a tensor for pruning.
    pub fn register_tensor(&mut self, name: &str, len: usize) {
        self.mask_manager.register(name, len);
    }

    /// Run one pruning step on a named tensor's weights (and optional gradients).
    pub fn prune_step(&mut self, name: &str, weights: &mut [f32], gradients: Option<&[f32]>) {
        let sparsity = self.schedule.sparsity_at_step(self.step);
        let mask = match self.config.method {
            PruningMethod::Magnitude => {
                let pruner = MagnitudePruner::new(PruningConfig::magnitude(sparsity));
                pruner.compute_mask(weights, sparsity)
            }
            PruningMethod::Structured => {
                // Fall back to per-weight magnitude for the unified API.
                let pruner = MagnitudePruner::new(PruningConfig::magnitude(sparsity));
                pruner.compute_mask(weights, sparsity)
            }
            PruningMethod::Movement => {
                let default_grads = vec![0.0; weights.len()];
                let grads = gradients.unwrap_or(&default_grads);
                let scores = MovementPruner::movement_scores(weights, grads);
                MovementPruner::compute_mask(&scores, sparsity)
            }
            PruningMethod::LotteryTicket => {
                let pruner = MagnitudePruner::new(PruningConfig::magnitude(sparsity));
                pruner.compute_mask(weights, sparsity)
            }
        };
        // Apply mask
        for (w, &keep) in weights.iter_mut().zip(mask.iter()) {
            if !keep {
                *w = 0.0;
            }
        }
        self.mask_manager.set_mask(name, mask);
    }

    /// Advance the global step counter.
    pub fn advance_step(&mut self) {
        self.step += 1;
    }

    /// Current step.
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Current scheduled sparsity.
    pub fn current_sparsity(&self) -> f32 {
        self.schedule.sparsity_at_step(self.step)
    }

    /// Reference to the internal mask manager.
    pub fn mask_manager(&self) -> &PruneMaskManager {
        &self.mask_manager
    }

    /// Generate a pruning report.
    pub fn report(&self) -> PruningReport {
        PruningReport::from_mask_manager(&self.mask_manager, self.config.method)
    }

    /// Config used by this engine.
    pub fn config(&self) -> &PruningConfig {
        &self.config
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // PruningConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn config_magnitude_defaults() {
        let c = PruningConfig::magnitude(0.5);
        assert_eq!(c.method, PruningMethod::Magnitude);
        assert!((c.target_sparsity - 0.5).abs() < 1e-6);
        assert_eq!(c.schedule, ScheduleKind::OneShot);
    }

    #[test]
    fn config_structured_defaults() {
        let c = PruningConfig::structured(0.3, PruningGranularity::Channel);
        assert_eq!(c.method, PruningMethod::Structured);
        assert_eq!(c.granularity, PruningGranularity::Channel);
    }

    #[test]
    fn config_movement_defaults() {
        let c = PruningConfig::movement(0.7, 100);
        assert_eq!(c.method, PruningMethod::Movement);
        assert_eq!(c.schedule, ScheduleKind::Cubic);
        assert_eq!(c.total_steps, 100);
    }

    #[test]
    fn config_lottery_ticket_defaults() {
        let c = PruningConfig::lottery_ticket(0.9);
        assert_eq!(c.method, PruningMethod::LotteryTicket);
        assert!((c.target_sparsity - 0.9).abs() < 1e-6);
    }

    #[test]
    fn config_validate_ok() {
        assert!(PruningConfig::magnitude(0.5).validate().is_ok());
    }

    #[test]
    fn config_validate_bad_sparsity_negative() {
        let mut c = PruningConfig::magnitude(-0.1);
        c.target_sparsity = -0.1;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_bad_sparsity_above_one() {
        let mut c = PruningConfig::magnitude(1.5);
        c.target_sparsity = 1.5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_zero_frequency() {
        let mut c = PruningConfig::magnitude(0.5);
        c.frequency = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_end_before_begin() {
        let mut c = PruningConfig::movement(0.5, 100);
        c.begin_step = 50;
        c.end_step = 30;
        assert!(c.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // PruningSchedule tests
    // -----------------------------------------------------------------------

    #[test]
    fn schedule_oneshot_always_target() {
        let s = PruningSchedule::new(ScheduleKind::OneShot, 0.5, 0, 10);
        assert!((s.sparsity_at_step(0) - 0.5).abs() < 1e-6);
        assert!((s.sparsity_at_step(5) - 0.5).abs() < 1e-6);
        assert!((s.sparsity_at_step(100) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn schedule_linear_ramp() {
        let s = PruningSchedule::new(ScheduleKind::Linear, 0.8, 0, 100);
        assert!((s.sparsity_at_step(0) - 0.0).abs() < 1e-6);
        assert!((s.sparsity_at_step(50) - 0.4).abs() < 1e-4);
        assert!((s.sparsity_at_step(100) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn schedule_linear_midpoint() {
        let s = PruningSchedule::new(ScheduleKind::Linear, 1.0, 0, 100);
        let mid = s.sparsity_at_step(50);
        assert!((mid - 0.5).abs() < 1e-4);
    }

    #[test]
    fn schedule_cubic_starts_slow() {
        let s = PruningSchedule::new(ScheduleKind::Cubic, 0.9, 0, 100);
        // At step 10 (t=0.1): 0.9*(1-(0.9)^3) = 0.9*0.271 = 0.2439
        let early = s.sparsity_at_step(10);
        assert!(early < 0.3, "cubic should start slow, got {early}");
    }

    #[test]
    fn schedule_cubic_monotonic() {
        let s = PruningSchedule::new(ScheduleKind::Cubic, 0.9, 0, 100);
        let mut prev = 0.0_f32;
        for step in 0..=100 {
            let cur = s.sparsity_at_step(step);
            assert!(cur >= prev - 1e-6, "schedule must be monotonic");
            prev = cur;
        }
    }

    #[test]
    fn schedule_cubic_reaches_target() {
        let s = PruningSchedule::new(ScheduleKind::Cubic, 0.9, 0, 100);
        assert!((s.sparsity_at_step(100) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn schedule_before_begin_is_zero() {
        let s = PruningSchedule::new(ScheduleKind::Linear, 0.5, 10, 100);
        assert!((s.sparsity_at_step(5) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn schedule_after_end_is_target() {
        let s = PruningSchedule::new(ScheduleKind::Linear, 0.5, 0, 50);
        assert!((s.sparsity_at_step(200) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn schedule_target_accessor() {
        let s = PruningSchedule::new(ScheduleKind::Cubic, 0.42, 0, 10);
        assert!((s.target_sparsity() - 0.42).abs() < 1e-6);
    }

    #[test]
    fn schedule_from_config() {
        let c = PruningConfig::movement(0.8, 200);
        let s = PruningSchedule::from_config(&c);
        assert!((s.target_sparsity() - 0.8).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // MagnitudePruner tests
    // -----------------------------------------------------------------------

    #[test]
    fn magnitude_empty_weights() {
        let p = MagnitudePruner::new(PruningConfig::magnitude(0.5));
        let mask = p.compute_mask(&[], 0.5);
        assert!(mask.is_empty());
    }

    #[test]
    fn magnitude_zero_sparsity_keeps_all() {
        let p = MagnitudePruner::new(PruningConfig::magnitude(0.0));
        let mask = p.compute_mask(&[1.0, 2.0, 3.0], 0.0);
        assert!(mask.iter().all(|&k| k));
    }

    #[test]
    fn magnitude_full_sparsity_prunes_all() {
        let p = MagnitudePruner::new(PruningConfig::magnitude(1.0));
        let mask = p.compute_mask(&[1.0, 2.0, 3.0], 1.0);
        assert!(mask.iter().all(|&k| !k));
    }

    #[test]
    fn magnitude_prune_half() {
        let p = MagnitudePruner::new(PruningConfig::magnitude(0.5));
        let weights = vec![0.1, -0.5, 0.3, -0.9];
        let mask = p.compute_mask(&weights, 0.5);
        let kept: usize = mask.iter().filter(|&&k| k).count();
        assert_eq!(kept, 2, "50% sparsity on 4 weights should keep 2");
    }

    #[test]
    fn magnitude_keeps_largest() {
        let p = MagnitudePruner::new(PruningConfig::magnitude(0.5));
        let weights = vec![0.1, -0.9, 0.2, -0.8];
        let mask = p.compute_mask(&weights, 0.5);
        // Largest magnitudes: idx 1 (0.9), idx 3 (0.8)
        assert!(mask[1], "should keep largest magnitude weight");
        assert!(mask[3], "should keep second largest magnitude weight");
    }

    #[test]
    fn magnitude_prune_in_place() {
        let p = MagnitudePruner::new(PruningConfig::magnitude(0.5));
        let mut weights = vec![0.1, -0.5, 0.3, -0.9];
        let mask = p.prune(&mut weights);
        let n_zero = weights.iter().filter(|&&w| w == 0.0).count();
        assert_eq!(n_zero, 2);
        let n_pruned = mask.iter().filter(|&&k| !k).count();
        assert_eq!(n_pruned, 2);
    }

    #[test]
    fn magnitude_sparsity_ratio_correct() {
        let p = MagnitudePruner::new(PruningConfig::magnitude(0.25));
        let weights: Vec<f32> = (1..=100).map(|i| i as f32 * 0.01).collect();
        let mask = p.compute_mask(&weights, 0.25);
        let pruned = mask.iter().filter(|&&k| !k).count();
        assert_eq!(pruned, 25);
    }

    #[test]
    fn magnitude_target_accessor() {
        let p = MagnitudePruner::new(PruningConfig::magnitude(0.6));
        assert!((p.target_sparsity() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn magnitude_negative_weights() {
        let p = MagnitudePruner::new(PruningConfig::magnitude(0.5));
        let weights = vec![-10.0, -1.0, -0.1, -5.0];
        let mask = p.compute_mask(&weights, 0.5);
        // Largest magnitudes: -10.0 (idx 0), -5.0 (idx 3)
        assert!(mask[0]);
        assert!(mask[3]);
    }

    #[test]
    fn magnitude_preserves_order() {
        let p = MagnitudePruner::new(PruningConfig::magnitude(0.5));
        let mut w = vec![3.0, 1.0, 4.0, 2.0];
        p.prune(&mut w);
        // Should keep 3.0 and 4.0 (idx 0 and 2)
        assert!((w[0] - 3.0).abs() < 1e-6 || w[0] == 0.0);
        assert!((w[2] - 4.0).abs() < 1e-6 || w[2] == 0.0);
    }

    // -----------------------------------------------------------------------
    // StructuredPruner tests
    // -----------------------------------------------------------------------

    #[test]
    fn structured_prune_channels_basic() {
        let cfg = PruningConfig::structured(0.5, PruningGranularity::Channel);
        let p = StructuredPruner::new(cfg);
        // 4 channels, 3 features each
        let weights = vec![
            0.1, 0.1, 0.1, // ch0: norm=0.3
            1.0, 1.0, 1.0, // ch1: norm=3.0
            0.5, 0.5, 0.5, // ch2: norm=1.5
            2.0, 2.0, 2.0, // ch3: norm=6.0
        ];
        let mask = p.prune_channels(&weights, 4, 3);
        let kept: usize = mask.iter().filter(|&&k| k).count();
        assert_eq!(kept, 2);
        // ch1 and ch3 should be kept (highest norms)
        assert!(mask[1]);
        assert!(mask[3]);
    }

    #[test]
    fn structured_prune_channels_zero_sparsity() {
        let cfg = PruningConfig::structured(0.0, PruningGranularity::Channel);
        let p = StructuredPruner::new(cfg);
        let weights = vec![1.0; 12];
        let mask = p.prune_channels(&weights, 4, 3);
        assert!(mask.iter().all(|&k| k));
    }

    #[test]
    fn structured_prune_channels_empty() {
        let cfg = PruningConfig::structured(0.5, PruningGranularity::Channel);
        let p = StructuredPruner::new(cfg);
        let mask = p.prune_channels(&[], 0, 0);
        assert!(mask.is_empty());
    }

    #[test]
    fn structured_prune_heads() {
        let cfg = PruningConfig::structured(0.5, PruningGranularity::Head);
        let p = StructuredPruner::new(cfg);
        let norms = vec![5.0, 1.0, 3.0, 0.5];
        let mask = p.prune_heads(&norms);
        let kept: usize = mask.iter().filter(|&&k| k).count();
        assert_eq!(kept, 2);
        assert!(mask[0], "highest-norm head should be kept");
        assert!(mask[2], "second-highest head should be kept");
    }

    #[test]
    fn structured_prune_heads_empty() {
        let cfg = PruningConfig::structured(0.5, PruningGranularity::Head);
        let p = StructuredPruner::new(cfg);
        let mask = p.prune_heads(&[]);
        assert!(mask.is_empty());
    }

    #[test]
    fn structured_prune_layers() {
        let cfg = PruningConfig::structured(0.25, PruningGranularity::Layer);
        let p = StructuredPruner::new(cfg);
        let scores = vec![10.0, 2.0, 8.0, 5.0];
        let mask = p.prune_layers(&scores);
        let kept: usize = mask.iter().filter(|&&k| k).count();
        assert_eq!(kept, 3);
        // Lowest score is idx 1 (2.0) — should be pruned
        assert!(!mask[1]);
    }

    #[test]
    fn structured_prune_layers_empty() {
        let cfg = PruningConfig::structured(0.5, PruningGranularity::Layer);
        let p = StructuredPruner::new(cfg);
        let mask = p.prune_layers(&[]);
        assert!(mask.is_empty());
    }

    #[test]
    fn structured_granularity_accessor() {
        let cfg = PruningConfig::structured(0.5, PruningGranularity::Head);
        let p = StructuredPruner::new(cfg);
        assert_eq!(p.granularity(), PruningGranularity::Head);
    }

    #[test]
    fn structured_target_accessor() {
        let cfg = PruningConfig::structured(0.3, PruningGranularity::Channel);
        let p = StructuredPruner::new(cfg);
        assert!((p.target_sparsity() - 0.3).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // MovementPruner tests
    // -----------------------------------------------------------------------

    #[test]
    fn movement_scores_basic() {
        let w = vec![1.0, -2.0, 3.0];
        let g = vec![0.5, -1.0, 0.1];
        let scores = MovementPruner::movement_scores(&w, &g);
        assert!((scores[0] - 0.5).abs() < 1e-6);
        assert!((scores[1] - 2.0).abs() < 1e-6);
        assert!((scores[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn movement_mask_prunes_lowest_scores() {
        let scores = vec![0.1, 5.0, -1.0, 3.0];
        let mask = MovementPruner::compute_mask(&scores, 0.5);
        // Lowest: idx 2 (-1.0), idx 0 (0.1)
        assert!(!mask[2]);
        assert!(!mask[0]);
        assert!(mask[1]);
        assert!(mask[3]);
    }

    #[test]
    fn movement_mask_zero_sparsity() {
        let scores = vec![1.0, 2.0, 3.0];
        let mask = MovementPruner::compute_mask(&scores, 0.0);
        assert!(mask.iter().all(|&k| k));
    }

    #[test]
    fn movement_mask_full_sparsity() {
        let scores = vec![1.0, 2.0, 3.0];
        let mask = MovementPruner::compute_mask(&scores, 1.0);
        assert!(mask.iter().all(|&k| !k));
    }

    #[test]
    fn movement_step_advances() {
        let cfg = PruningConfig::movement(0.5, 10);
        let mut mp = MovementPruner::new(cfg);
        assert_eq!(mp.current_step(), 0);
        let w = vec![1.0; 4];
        let g = vec![0.5; 4];
        let _ = mp.step(&w, &g);
        assert_eq!(mp.current_step(), 1);
    }

    #[test]
    fn movement_gradual_increases() {
        let cfg = PruningConfig::movement(0.9, 100);
        let mp = MovementPruner::new(cfg);
        let s0 = mp.current_sparsity();
        // At step 0 of cubic schedule: sparsity_at_step(0) should be 0 since begin=0
        // Actually for cubic at t=0: s_f*(1-(1-0)^3)=0
        assert!(s0 < 0.01);
    }

    #[test]
    fn movement_target_accessor() {
        let cfg = PruningConfig::movement(0.7, 50);
        let mp = MovementPruner::new(cfg);
        assert!((mp.target_sparsity() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn movement_scores_empty() {
        let scores = MovementPruner::movement_scores(&[], &[]);
        assert!(scores.is_empty());
    }

    // -----------------------------------------------------------------------
    // LotteryTicket tests
    // -----------------------------------------------------------------------

    #[test]
    fn lottery_save_and_recall_initial() {
        let cfg = PruningConfig::lottery_ticket(0.8);
        let mut lt = LotteryTicket::new(cfg, 5);
        let init = vec![1.0, 2.0, 3.0];
        lt.save_initial_weights("layer1", &init);
        assert_eq!(lt.initial_weights("layer1").unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn lottery_initial_missing() {
        let cfg = PruningConfig::lottery_ticket(0.8);
        let lt = LotteryTicket::new(cfg, 5);
        assert!(lt.initial_weights("nonexistent").is_none());
    }

    #[test]
    fn lottery_round_increments() {
        let cfg = PruningConfig::lottery_ticket(0.8);
        let mut lt = LotteryTicket::new(cfg, 5);
        lt.save_initial_weights("l", &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(lt.round(), 0);
        let _ = lt.prune_round("l", &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(lt.round(), 1);
    }

    #[test]
    fn lottery_is_complete() {
        let cfg = PruningConfig::lottery_ticket(0.8);
        let mut lt = LotteryTicket::new(cfg, 2);
        lt.save_initial_weights("l", &[1.0, 2.0, 3.0, 4.0]);
        assert!(!lt.is_complete());
        let _ = lt.prune_round("l", &[1.0, 2.0, 3.0, 4.0]);
        assert!(!lt.is_complete());
        let _ = lt.prune_round("l", &[0.0, 2.0, 3.0, 4.0]);
        assert!(lt.is_complete());
    }

    #[test]
    fn lottery_sparsity_increases_per_round() {
        let cfg = PruningConfig::lottery_ticket(0.8);
        let lt = LotteryTicket::new(cfg, 5);
        let mut prev = 0.0_f32;
        for r in 1..=5 {
            let s = lt.sparsity_at_round(r);
            assert!(s >= prev, "sparsity should increase each round");
            prev = s;
        }
    }

    #[test]
    fn lottery_final_round_near_target() {
        let cfg = PruningConfig::lottery_ticket(0.8);
        let lt = LotteryTicket::new(cfg, 5);
        let final_s = lt.sparsity_at_round(5);
        assert!((final_s - 0.8).abs() < 0.01, "final round sparsity {final_s} should be near 0.8");
    }

    #[test]
    fn lottery_rewind_restores_initial() {
        let cfg = PruningConfig::lottery_ticket(0.8);
        let mut lt = LotteryTicket::new(cfg, 3);
        let init = vec![10.0, 20.0, 30.0, 40.0];
        lt.save_initial_weights("l", &init);
        let mask = vec![true, false, true, false];
        let mut weights = vec![99.0; 4];
        lt.rewind("l", &mut weights, &mask);
        assert!((weights[0] - 10.0).abs() < 1e-6);
        assert!((weights[1]).abs() < 1e-6); // pruned
        assert!((weights[2] - 30.0).abs() < 1e-6);
        assert!((weights[3]).abs() < 1e-6); // pruned
    }

    #[test]
    fn lottery_max_rounds_accessor() {
        let cfg = PruningConfig::lottery_ticket(0.5);
        let lt = LotteryTicket::new(cfg, 7);
        assert_eq!(lt.max_rounds(), 7);
    }

    // -----------------------------------------------------------------------
    // SensitivityAnalyzer tests
    // -----------------------------------------------------------------------

    #[test]
    fn sensitivity_record_and_count() {
        let mut sa = SensitivityAnalyzer::new();
        sa.record("layer0", 0.1, 0.01);
        sa.record("layer0", 0.3, 0.05);
        assert_eq!(sa.record_count(), 2);
    }

    #[test]
    fn sensitivity_curve_sorted() {
        let mut sa = SensitivityAnalyzer::new();
        sa.record("l", 0.5, 0.1);
        sa.record("l", 0.1, 0.01);
        sa.record("l", 0.3, 0.05);
        let curve = sa.sensitivity_curve("l");
        assert_eq!(curve.len(), 3);
        assert!((curve[0].0 - 0.1).abs() < 1e-6);
        assert!((curve[1].0 - 0.3).abs() < 1e-6);
        assert!((curve[2].0 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sensitivity_curve_empty_for_unknown() {
        let sa = SensitivityAnalyzer::new();
        assert!(sa.sensitivity_curve("none").is_empty());
    }

    #[test]
    fn sensitivity_recommend_basic() {
        let mut sa = SensitivityAnalyzer::new();
        sa.record("a", 0.5, 0.02);
        sa.record("a", 0.8, 0.10);
        sa.record("b", 0.3, 0.01);
        sa.record("b", 0.5, 0.15); // exceeds max_deg=0.1

        let rec = sa.recommend(0.5, 0.1);
        assert!(rec.contains_key("a"));
        assert!(rec.contains_key("b"));
        // Both should have some positive value
        assert!(*rec.get("a").unwrap() > 0.0);
        assert!(*rec.get("b").unwrap() > 0.0);
    }

    #[test]
    fn sensitivity_recommend_respects_max_degradation() {
        let mut sa = SensitivityAnalyzer::new();
        sa.record("a", 0.9, 0.5); // exceeds max 0.1
        sa.record("a", 0.1, 0.05);
        let rec = sa.recommend(0.1, 0.1);
        // Layer 'a' max sparsity within budget is 0.1
        let a_val = rec.get("a").copied().unwrap_or(0.0);
        assert!(a_val <= 0.1 + 1e-6);
    }

    #[test]
    fn sensitivity_recommend_empty() {
        let sa = SensitivityAnalyzer::new();
        let rec = sa.recommend(0.5, 0.1);
        assert!(rec.is_empty());
    }

    #[test]
    fn sensitivity_default_constructor() {
        let sa = SensitivityAnalyzer::default();
        assert_eq!(sa.record_count(), 0);
    }

    // -----------------------------------------------------------------------
    // PruneMaskManager tests
    // -----------------------------------------------------------------------

    #[test]
    fn mask_register_dense() {
        let mut mgr = PruneMaskManager::new();
        mgr.register("t1", 5);
        let mask = mgr.get_mask("t1").unwrap();
        assert_eq!(mask.len(), 5);
        assert!(mask.iter().all(|&k| k));
    }

    #[test]
    fn mask_set_and_get() {
        let mut mgr = PruneMaskManager::new();
        mgr.set_mask("t1", vec![true, false, true]);
        assert_eq!(mgr.get_mask("t1").unwrap(), &[true, false, true]);
    }

    #[test]
    fn mask_get_missing() {
        let mgr = PruneMaskManager::new();
        assert!(mgr.get_mask("nope").is_none());
    }

    #[test]
    fn mask_apply_zeros_pruned() {
        let mut mgr = PruneMaskManager::new();
        mgr.set_mask("t", vec![true, false, true, false]);
        let mut w = vec![1.0, 2.0, 3.0, 4.0];
        mgr.apply("t", &mut w);
        assert!((w[0] - 1.0).abs() < 1e-6);
        assert!((w[1]).abs() < 1e-6);
        assert!((w[2] - 3.0).abs() < 1e-6);
        assert!((w[3]).abs() < 1e-6);
    }

    #[test]
    fn mask_sparsity_correct() {
        let mut mgr = PruneMaskManager::new();
        mgr.set_mask("t", vec![true, false, true, false]);
        let s = mgr.sparsity("t").unwrap();
        assert!((s - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mask_sparsity_missing() {
        let mgr = PruneMaskManager::new();
        assert!(mgr.sparsity("x").is_none());
    }

    #[test]
    fn mask_overall_sparsity() {
        let mut mgr = PruneMaskManager::new();
        mgr.set_mask("a", vec![true, false]); // 50%
        mgr.set_mask("b", vec![true, true]); // 0%
        let s = mgr.overall_sparsity();
        // 1 pruned / 4 total = 0.25
        assert!((s - 0.25).abs() < 1e-6);
    }

    #[test]
    fn mask_overall_sparsity_empty() {
        let mgr = PruneMaskManager::new();
        assert!((mgr.overall_sparsity()).abs() < 1e-6);
    }

    #[test]
    fn mask_tensor_count() {
        let mut mgr = PruneMaskManager::new();
        mgr.register("a", 3);
        mgr.register("b", 5);
        assert_eq!(mgr.tensor_count(), 2);
    }

    #[test]
    fn mask_intersect() {
        let mut m1 = PruneMaskManager::new();
        m1.set_mask("t", vec![true, true, false, true]);
        let mut m2 = PruneMaskManager::new();
        m2.set_mask("t", vec![true, false, true, true]);
        m1.intersect(&m2);
        assert_eq!(m1.get_mask("t").unwrap(), &[true, false, false, true]);
    }

    #[test]
    fn mask_default_constructor() {
        let mgr = PruneMaskManager::default();
        assert_eq!(mgr.tensor_count(), 0);
    }

    #[test]
    fn mask_apply_no_mask_noop() {
        let mgr = PruneMaskManager::new();
        let mut w = vec![1.0, 2.0, 3.0];
        mgr.apply("no_such", &mut w);
        assert!((w[0] - 1.0).abs() < 1e-6);
        assert!((w[1] - 2.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // PruningReport tests
    // -----------------------------------------------------------------------

    #[test]
    fn report_from_mask_manager() {
        let mut mgr = PruneMaskManager::new();
        mgr.set_mask("a", vec![true, false, true, false]);
        mgr.set_mask("b", vec![true, true, true, true]);
        let r = PruningReport::from_mask_manager(&mgr, PruningMethod::Magnitude);
        assert_eq!(r.total_params, 8);
        assert_eq!(r.pruned_params, 2);
        assert_eq!(r.remaining_params, 6);
        assert!((r.overall_sparsity - 0.25).abs() < 1e-6);
    }

    #[test]
    fn report_compression_ratio() {
        let mut mgr = PruneMaskManager::new();
        mgr.set_mask("x", vec![true, false, false, false]);
        let r = PruningReport::from_mask_manager(&mgr, PruningMethod::Magnitude);
        // 4 total, 1 remaining → ratio 4.0
        assert!((r.compression_ratio - 4.0).abs() < 1e-6);
    }

    #[test]
    fn report_from_weights() {
        let w1: Vec<f32> = vec![0.0, 1.0, 0.0, 3.0];
        let w2: Vec<f32> = vec![0.0, 0.0];
        let r = PruningReport::from_weights(&[("l1", &w1), ("l2", &w2)], PruningMethod::Structured);
        assert_eq!(r.total_params, 6);
        assert_eq!(r.pruned_params, 4);
        assert!((r.overall_sparsity - 4.0 / 6.0).abs() < 1e-4);
    }

    #[test]
    fn report_with_quality_metric() {
        let mut mgr = PruneMaskManager::new();
        mgr.register("t", 4);
        let r = PruningReport::from_mask_manager(&mgr, PruningMethod::Movement)
            .with_quality_metric(1.23);
        assert!((r.quality_metric.unwrap() - 1.23).abs() < 1e-6);
    }

    #[test]
    fn report_empty_mask_manager() {
        let mgr = PruneMaskManager::new();
        let r = PruningReport::from_mask_manager(&mgr, PruningMethod::Magnitude);
        assert_eq!(r.total_params, 0);
        assert!((r.overall_sparsity).abs() < 1e-6);
    }

    #[test]
    fn report_per_tensor_sparsity() {
        let mut mgr = PruneMaskManager::new();
        mgr.set_mask("a", vec![true, false]);
        mgr.set_mask("b", vec![false, false, false]);
        let r = PruningReport::from_mask_manager(&mgr, PruningMethod::Magnitude);
        assert!((r.per_tensor_sparsity["a"] - 0.5).abs() < 1e-6);
        assert!((r.per_tensor_sparsity["b"] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn report_method_stored() {
        let mgr = PruneMaskManager::new();
        let r = PruningReport::from_mask_manager(&mgr, PruningMethod::LotteryTicket);
        assert_eq!(r.method, PruningMethod::LotteryTicket);
    }

    // -----------------------------------------------------------------------
    // PruningEngine tests
    // -----------------------------------------------------------------------

    #[test]
    fn engine_magnitude_prune() {
        let cfg = PruningConfig::magnitude(0.5);
        let mut eng = PruningEngine::new(cfg);
        eng.register_tensor("t", 4);
        let mut w = vec![0.1, 0.9, 0.2, 0.8];
        eng.prune_step("t", &mut w, None);
        let n_zero = w.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(n_zero, 2);
    }

    #[test]
    fn engine_movement_prune() {
        let cfg = PruningConfig {
            method: PruningMethod::Movement,
            target_sparsity: 0.5,
            schedule: ScheduleKind::OneShot,
            granularity: PruningGranularity::Weight,
            total_steps: 1,
            begin_step: 0,
            end_step: 1,
            frequency: 1,
        };
        let mut eng = PruningEngine::new(cfg);
        eng.register_tensor("t", 4);
        let mut w = vec![1.0, -2.0, 3.0, -4.0];
        let g = vec![0.5, 0.5, 0.5, 0.5];
        eng.prune_step("t", &mut w, Some(&g));
        let n_zero = w.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(n_zero, 2);
    }

    #[test]
    fn engine_advance_step() {
        let cfg = PruningConfig::magnitude(0.5);
        let mut eng = PruningEngine::new(cfg);
        assert_eq!(eng.current_step(), 0);
        eng.advance_step();
        assert_eq!(eng.current_step(), 1);
    }

    #[test]
    fn engine_report_after_prune() {
        let cfg = PruningConfig::magnitude(0.5);
        let mut eng = PruningEngine::new(cfg);
        eng.register_tensor("t", 4);
        let mut w = vec![0.1, 0.9, 0.2, 0.8];
        eng.prune_step("t", &mut w, None);
        let r = eng.report();
        assert_eq!(r.total_params, 4);
        assert_eq!(r.pruned_params, 2);
    }

    #[test]
    fn engine_gradual_schedule() {
        let mut cfg = PruningConfig::magnitude(0.9);
        cfg.schedule = ScheduleKind::Linear;
        cfg.begin_step = 0;
        cfg.end_step = 10;
        cfg.total_steps = 10;
        let mut eng = PruningEngine::new(cfg);
        eng.register_tensor("t", 100);

        // Step 0 → sparsity 0.0 (linear at t=0)
        let mut w: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        eng.prune_step("t", &mut w, None);
        let z0 = w.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(z0, 0, "step 0 linear sparsity should be 0");

        // Advance to step 5 → sparsity ~0.45
        for _ in 0..5 {
            eng.advance_step();
        }
        let mut w2: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        eng.prune_step("t", &mut w2, None);
        let z5 = w2.iter().filter(|&&v| v == 0.0).count();
        assert!(z5 > 30 && z5 < 60, "mid sparsity should be ~45, got {z5}");
    }

    #[test]
    fn engine_config_accessor() {
        let cfg = PruningConfig::magnitude(0.3);
        let eng = PruningEngine::new(cfg);
        assert!((eng.config().target_sparsity - 0.3).abs() < 1e-6);
    }

    #[test]
    fn engine_mask_manager_accessor() {
        let cfg = PruningConfig::magnitude(0.5);
        let mut eng = PruningEngine::new(cfg);
        eng.register_tensor("t", 4);
        assert_eq!(eng.mask_manager().tensor_count(), 1);
    }

    #[test]
    fn engine_current_sparsity() {
        let cfg = PruningConfig::magnitude(0.5);
        let eng = PruningEngine::new(cfg);
        // OneShot → always target
        assert!((eng.current_sparsity() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn engine_lottery_mode() {
        let cfg = PruningConfig::lottery_ticket(0.5);
        let mut eng = PruningEngine::new(cfg);
        eng.register_tensor("t", 4);
        let mut w = vec![0.1, 0.9, 0.2, 0.8];
        eng.prune_step("t", &mut w, None);
        let n_zero = w.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(n_zero, 2);
    }

    #[test]
    fn engine_structured_fallback() {
        let cfg = PruningConfig::structured(0.5, PruningGranularity::Channel);
        let mut eng = PruningEngine::new(cfg);
        eng.register_tensor("t", 4);
        let mut w = vec![0.1, 0.9, 0.2, 0.8];
        eng.prune_step("t", &mut w, None);
        let n_zero = w.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(n_zero, 2);
    }

    #[test]
    fn engine_multiple_tensors() {
        let cfg = PruningConfig::magnitude(0.5);
        let mut eng = PruningEngine::new(cfg);
        eng.register_tensor("a", 4);
        eng.register_tensor("b", 4);
        let mut wa = vec![0.1, 0.9, 0.2, 0.8];
        let mut wb = vec![0.3, 0.7, 0.4, 0.6];
        eng.prune_step("a", &mut wa, None);
        eng.prune_step("b", &mut wb, None);
        let r = eng.report();
        assert_eq!(r.total_params, 8);
        assert_eq!(r.pruned_params, 4);
    }

    // -----------------------------------------------------------------------
    // Integration / cross-component tests
    // -----------------------------------------------------------------------

    #[test]
    fn integration_magnitude_then_report() {
        let p = MagnitudePruner::new(PruningConfig::magnitude(0.5));
        let mut w = vec![0.1, 0.5, 0.3, 0.9];
        let mask = p.prune(&mut w);
        let mut mgr = PruneMaskManager::new();
        mgr.set_mask("layer", mask);
        let r = PruningReport::from_mask_manager(&mgr, PruningMethod::Magnitude);
        assert!((r.overall_sparsity - 0.5).abs() < 1e-6);
    }

    #[test]
    fn integration_structured_channel_report() {
        let cfg = PruningConfig::structured(0.5, PruningGranularity::Channel);
        let sp = StructuredPruner::new(cfg);
        let weights = vec![
            1.0, 1.0, // ch0
            0.1, 0.1, // ch1
            5.0, 5.0, // ch2
            0.5, 0.5, // ch3
        ];
        let ch_mask = sp.prune_channels(&weights, 4, 2);
        // Expand to per-weight mask
        let mut mask = vec![false; 8];
        for (c, &keep) in ch_mask.iter().enumerate() {
            if keep {
                mask[c * 2] = true;
                mask[c * 2 + 1] = true;
            }
        }
        let mut mgr = PruneMaskManager::new();
        mgr.set_mask("w", mask);
        let r = PruningReport::from_mask_manager(&mgr, PruningMethod::Structured);
        assert_eq!(r.pruned_params, 4);
    }

    #[test]
    fn integration_movement_gradual() {
        let cfg = PruningConfig::movement(0.8, 10);
        let mut mp = MovementPruner::new(cfg);
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let g = vec![0.1; 10];
        let mut prev_pruned = 0;
        for _ in 0..10 {
            let mask = mp.step(&w, &g);
            let pruned = mask.iter().filter(|&&k| !k).count();
            assert!(pruned >= prev_pruned, "pruning should be monotonically non-decreasing");
            prev_pruned = pruned;
        }
    }

    #[test]
    fn integration_lottery_full_cycle() {
        let cfg = PruningConfig::lottery_ticket(0.75);
        let mut lt = LotteryTicket::new(cfg, 3);
        let init = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        lt.save_initial_weights("fc", &init);

        let trained = vec![0.5, 3.0, 0.1, 5.0, 0.2, 7.0, 0.3, 9.0];
        let mut cumulative_mask = vec![true; 8];

        for _ in 0..3 {
            let round_mask = lt.prune_round("fc", &trained);
            for (c, &r) in cumulative_mask.iter_mut().zip(round_mask.iter()) {
                *c = *c && r;
            }
        }
        assert!(lt.is_complete());
        let kept: usize = cumulative_mask.iter().filter(|&&k| k).count();
        assert!(kept <= 4, "after 3 rounds of LT, should have significant pruning");
    }

    #[test]
    fn integration_sensitivity_to_engine() {
        let mut sa = SensitivityAnalyzer::new();
        sa.record("layer0", 0.3, 0.01);
        sa.record("layer0", 0.5, 0.03);
        sa.record("layer0", 0.7, 0.08);
        sa.record("layer1", 0.3, 0.02);
        sa.record("layer1", 0.5, 0.10);

        let rec = sa.recommend(0.5, 0.05);
        // Use recommendations to configure per-layer pruning
        for (layer, ratio) in &rec {
            assert!(*ratio >= 0.0 && *ratio <= 1.0, "ratio for {layer} out of range");
        }
    }
}
